"""
Weather Arbitrage Paper Trader for Polymarket

Scans Polymarket daily temperature markets, compares to NOAA forecasts,
and paper trades undervalued temperature buckets for profit.

Strategy:
  - Fetch active temperature markets from Polymarket (Gamma API)
  - Fetch NOAA forecasts for matching locations
  - If NOAA forecast puts a temperature bucket at >60% probability
    but market prices that bucket at <$0.15, BUY it
  - Exit when price rises above $0.45 or market resolves
  - Track all trades for analysis

Based on viral weather trading bots:
  - @automatedAItradingbot: $65K profit trading NYC/London/Seoul weather
  - @0xf2e346ab: $1K -> $24K since Apr 2025 on London weather
"""

import asyncio
import json
import re
import time
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from functools import partial

import httpx

# Force unbuffered output
print = partial(print, flush=True)


# === NOAA LOCATION DATABASE ===
# Grid points for NOAA API + Weather Underground station codes
LOCATIONS = {
    "new york": {
        "lat": 40.7731, "lon": -73.8712,  # LaGuardia Airport
        "noaa_office": "OKX", "grid_x": 37, "grid_y": 38,
        "wunderground": "KLGA",
        "aliases": ["nyc", "new york city", "new york"],
    },
    "chicago": {
        "lat": 41.9742, "lon": -87.9073,  # O'Hare
        "noaa_office": "LOT", "grid_x": 65, "grid_y": 76,
        "wunderground": "KORD",
        "aliases": ["chicago"],
    },
    "seattle": {
        "lat": 47.4502, "lon": -122.3088,  # SeaTac
        "noaa_office": "SEW", "grid_x": 124, "grid_y": 67,
        "wunderground": "KSEA",
        "aliases": ["seattle"],
    },
    "atlanta": {
        "lat": 33.6407, "lon": -84.4277,  # Hartsfield
        "noaa_office": "FFC", "grid_x": 50, "grid_y": 86,
        "wunderground": "KATL",
        "aliases": ["atlanta"],
    },
    "dallas": {
        "lat": 32.8998, "lon": -97.0403,  # DFW
        "noaa_office": "FWD", "grid_x": 80, "grid_y": 103,
        "wunderground": "KDFW",
        "aliases": ["dallas"],
    },
    "toronto": {
        "lat": 43.6777, "lon": -79.6248,  # Pearson
        "noaa_office": None,  # Canada - no NOAA, use alternate
        "wunderground": "CYYZ",
        "aliases": ["toronto"],
    },
    "buenos aires": {
        "lat": -34.5588, "lon": -58.4156,
        "noaa_office": None,  # Argentina - no NOAA
        "wunderground": "SAEZ",
        "aliases": ["buenos aires"],
    },
    "london": {
        "lat": 51.4700, "lon": -0.4543,  # Heathrow
        "noaa_office": None,  # UK - no NOAA
        "wunderground": "EGLL",
        "aliases": ["london"],
    },
    "seoul": {
        "lat": 37.4602, "lon": 126.4407,  # Incheon
        "noaa_office": None,  # Korea - no NOAA
        "wunderground": "RKSI",
        "aliases": ["seoul"],
    },
    "miami": {
        "lat": 25.7959, "lon": -80.2870,  # MIA Airport
        "noaa_office": "MFL", "grid_x": 109, "grid_y": 50,
        "wunderground": "KMIA",
        "aliases": ["miami"],
    },
    "ankara": {
        "lat": 40.1281, "lon": 32.9951,  # Esenboga
        "noaa_office": None,
        "wunderground": "LTAC",
        "aliases": ["ankara"],
    },
    "wellington": {
        "lat": -41.3276, "lon": 174.8050,  # Wellington Airport
        "noaa_office": None,
        "wunderground": "NZWN",
        "aliases": ["wellington"],
    },
}


@dataclass
class WeatherTrade:
    """A paper weather trade."""
    trade_id: str
    market_title: str
    location: str
    temp_bucket: str          # e.g. "28-33F" or "27F_or_below"
    entry_price: float
    entry_time: str
    size_usd: float = 5.0
    noaa_forecast_high: float = 0.0   # NOAA predicted high temp
    noaa_probability: float = 0.0     # Our calculated probability
    market_probability: float = 0.0   # Market price at entry
    edge: float = 0.0                 # noaa_prob - market_prob
    confidence: float = 0.0          # V1.1: ensemble agreement (% within 2F of median)
    source: str = "triangular"       # V1.1: "ensemble" or "triangular"
    condition_id: str = ""
    status: str = "open"              # open, closed
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: float = 0.0
    exit_reason: str = ""             # resolved_win, resolved_loss, take_profit, timeout


class WeatherArbitrageTrader:
    """Paper trades weather markets using NOAA forecast arbitrage."""

    OUTPUT_FILE = Path(__file__).parent / "weather_paper_results.json"
    NOAA_BASE = "https://api.weather.gov"
    GAMMA_BASE = "https://gamma-api.polymarket.com"

    # === TRADING CONFIG ===
    ENTRY_THRESHOLD = 0.15      # Only buy buckets priced below $0.15
    EXIT_THRESHOLD = 0.45       # Take profit when price > $0.45
    MAX_POSITION = 5.00         # Max $5 per trade
    MIN_POSITION = 2.00         # Min $2 per trade
    MAX_TRADES_PER_RUN = 5      # Max new trades per scan cycle
    MIN_EDGE = 0.10             # NOAA prob must exceed market price by 10%+ (was 30%, too strict for narrow buckets)
    MIN_NOAA_PROBABILITY = 0.10 # NOAA must give >10% chance for this bucket (was 50%, impossible for 1-degree buckets)
    MIN_CONFIDENCE = 0.70       # V1.1: Multi-source confidence threshold (% of ensemble within 2F of median)
    SCAN_INTERVAL = 120         # Scan every 2 minutes
    MAX_OPEN_TRADES = 10        # Max concurrent open positions

    # === NOAA FORECAST SETTINGS ===
    # Temperature probability is estimated from forecast range
    # NOAA gives point forecast; we model uncertainty as +/- spread
    FORECAST_SPREAD_F = 8       # +/- 8F uncertainty around NOAA point forecast (was 5, too tight)
    FORECAST_SPREAD_C = 5       # +/- 5C for Celsius markets (was 3, too tight)

    def __init__(self):
        self.trades: Dict[str, WeatherTrade] = {}
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.scan_count = 0
        self.start_time = datetime.now(timezone.utc).isoformat()

        # Per-location stats
        self.location_stats: Dict[str, dict] = {}

        # Cache NOAA forecasts (refresh every 30 min)
        self._noaa_cache: Dict[str, dict] = {}
        self._noaa_cache_time: float = 0
        self.NOAA_CACHE_TTL = 1800  # 30 minutes

        # V1.1: Ensemble forecast cache {location: {date: [member_temps]}}
        self._ensemble_cache: Dict[str, Dict[str, List[float]]] = {}

        # Market cache
        self._market_cache: List[dict] = []

        self._load()
        self._cleanup_stuck_trades()

    def _load(self):
        """Load saved state."""
        if self.OUTPUT_FILE.exists():
            try:
                data = json.load(open(self.OUTPUT_FILE))
                for tid, t in data.get("trades", {}).items():
                    self.trades[tid] = WeatherTrade(**t)
                self.total_pnl = data.get("total_pnl", 0)
                self.wins = data.get("wins", 0)
                self.losses = data.get("losses", 0)
                self.scan_count = data.get("scan_count", 0)
                self.start_time = data.get("start_time", self.start_time)
                self.location_stats = data.get("location_stats", {})
                print(f"Loaded {len(self.trades)} weather trades from previous session")
            except Exception as e:
                print(f"Error loading state: {e}")

    def _save(self):
        """Save state to file."""
        data = {
            "start_time": self.start_time,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "trades": {tid: asdict(t) for tid, t in self.trades.items()},
            "total_pnl": self.total_pnl,
            "wins": self.wins,
            "losses": self.losses,
            "scan_count": self.scan_count,
            "location_stats": self.location_stats,
        }
        with open(self.OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def _cleanup_stuck_trades(self):
        """Close trades that are older than 48 hours and still open."""
        now = datetime.now(timezone.utc)
        fixed = 0
        for tid, trade in self.trades.items():
            if trade.status != "open":
                continue
            try:
                entry_dt = datetime.fromisoformat(trade.entry_time)
                age_hours = (now - entry_dt).total_seconds() / 3600
                if age_hours > 48:
                    trade.status = "closed"
                    trade.exit_time = now.isoformat()
                    trade.pnl = -trade.size_usd
                    trade.exit_reason = "timeout_cleanup"
                    self.total_pnl += trade.pnl
                    self.losses += 1
                    self._update_location_stats(trade.location, False, trade.pnl)
                    fixed += 1
            except Exception:
                pass
        if fixed:
            print(f"[CLEANUP] Closed {fixed} stuck trades (>48h old)")
            self._save()

    # === NOAA FORECAST FETCHING ===

    async def _fetch_noaa_forecast(self, client: httpx.AsyncClient, location: str) -> Optional[dict]:
        """Fetch NOAA forecast for a US location."""
        loc_data = LOCATIONS.get(location)
        if not loc_data or not loc_data.get("noaa_office"):
            return None  # Non-US location, skip NOAA

        office = loc_data["noaa_office"]
        gx, gy = loc_data["grid_x"], loc_data["grid_y"]

        try:
            r = await client.get(
                f"{self.NOAA_BASE}/gridpoints/{office}/{gx},{gy}/forecast",
                headers={"User-Agent": "(star-polymarket-weather, weather@bot.com)"},
                timeout=15
            )
            if r.status_code != 200:
                print(f"[NOAA] {location}: HTTP {r.status_code}")
                return None

            data = r.json()
            periods = data.get("properties", {}).get("periods", [])
            if not periods:
                return None

            # Build daily forecasts from periods
            forecasts = {}
            for p in periods:
                start = p.get("startTime", "")
                temp = p.get("temperature")
                unit = p.get("temperatureUnit", "F")
                name = p.get("name", "")
                is_daytime = p.get("isDaytime", True)

                if not start or temp is None:
                    continue

                # Parse date
                date_str = start[:10]  # YYYY-MM-DD

                if date_str not in forecasts:
                    forecasts[date_str] = {"high": None, "low": None, "unit": unit}

                if is_daytime:
                    forecasts[date_str]["high"] = temp
                else:
                    forecasts[date_str]["low"] = temp

            return forecasts

        except Exception as e:
            print(f"[NOAA] Error fetching {location}: {e}")
            return None

    async def _fetch_wunderground_forecast(self, client: httpx.AsyncClient, location: str) -> Optional[dict]:
        """Fetch Weather Underground forecast for non-US locations.
        Falls back to Open-Meteo API which is free and global."""
        loc_data = LOCATIONS.get(location)
        if not loc_data:
            return None

        lat, lon = loc_data["lat"], loc_data["lon"]

        try:
            # Use Open-Meteo free API for global forecasts
            r = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min",
                    "temperature_unit": "fahrenheit",
                    "forecast_days": 7,
                },
                timeout=15
            )
            if r.status_code != 200:
                return None

            data = r.json()
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            highs = daily.get("temperature_2m_max", [])
            lows = daily.get("temperature_2m_min", [])

            forecasts = {}
            for i, date in enumerate(dates):
                forecasts[date] = {
                    "high": round(highs[i]) if i < len(highs) and highs[i] is not None else None,
                    "low": round(lows[i]) if i < len(lows) and lows[i] is not None else None,
                    "unit": "F",
                }

            return forecasts

        except Exception as e:
            print(f"[METEO] Error fetching {location}: {e}")
            return None

    async def _fetch_ensemble_forecast(self, client: httpx.AsyncClient, location: str) -> Optional[Dict[str, List[float]]]:
        """V1.1: Fetch ensemble forecast (51 ECMWF members) for real probability distributions.

        Returns {date_str: [temp1, temp2, ..., temp51]} in Fahrenheit.
        This replaces the crude triangular distribution with actual model spread.
        """
        loc_data = LOCATIONS.get(location)
        if not loc_data:
            return None

        lat, lon = loc_data["lat"], loc_data["lon"]
        try:
            r = await client.get(
                "https://ensemble-api.open-meteo.com/v1/ensemble",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max",
                    "models": "ecmwf_ifs025",
                    "temperature_unit": "fahrenheit",
                    "forecast_days": 7,
                },
                timeout=15
            )
            if r.status_code != 200:
                return None

            data = r.json()
            daily = data.get("daily", {})
            dates = daily.get("time", [])

            # Collect all ensemble member values per date
            result = {}
            for i, date_str in enumerate(dates):
                members = []
                for key, vals in daily.items():
                    if key.startswith("temperature_2m_max_member") and i < len(vals):
                        if vals[i] is not None:
                            members.append(vals[i])
                if members:
                    result[date_str] = members

            return result if result else None

        except Exception as e:
            print(f"[ENSEMBLE] Error fetching {location}: {e}")
            return None

    def _ensemble_bucket_probability(self, members: List[float], bucket_low: float,
                                       bucket_high: float) -> Tuple[float, float]:
        """V1.1: Calculate bucket probability + confidence from ensemble members.

        Returns (probability, confidence) where:
          - probability = fraction of members falling in the bucket
          - confidence = fraction of members within 2°F of median (source agreement)
        """
        if not members:
            return 0.0, 0.0

        n = len(members)
        median_temp = statistics.median(members)

        # Probability: count members in bucket
        in_bucket = sum(1 for t in members
                        if (bucket_low <= t < bucket_high) or
                           (bucket_low == -999 and t < bucket_high) or
                           (bucket_high == 999 and t >= bucket_low))
        probability = in_bucket / n

        # Confidence: % of members within 2°F of median
        # High confidence = tight agreement, low spread
        close_to_median = sum(1 for t in members if abs(t - median_temp) <= 2.0)
        confidence = close_to_median / n

        return probability, confidence

    async def refresh_forecasts(self, client: httpx.AsyncClient):
        """Refresh all location forecasts (deterministic + ensemble)."""
        now = time.time()
        if now - self._noaa_cache_time < self.NOAA_CACHE_TTL:
            return  # Cache still fresh

        print("[WEATHER] Refreshing forecasts...")
        for location in LOCATIONS:
            loc_data = LOCATIONS[location]

            # Deterministic forecast (NOAA for US, Open-Meteo for intl)
            if loc_data.get("noaa_office"):
                forecast = await self._fetch_noaa_forecast(client, location)
            else:
                forecast = await self._fetch_wunderground_forecast(client, location)

            if forecast:
                self._noaa_cache[location] = forecast

            # V1.1: Ensemble forecast (51 ECMWF members for all locations)
            ensemble = await self._fetch_ensemble_forecast(client, location)
            if ensemble:
                self._ensemble_cache[location] = ensemble

            # Show today's forecast with ensemble spread
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_fc = (forecast or {}).get(today, {})
            today_ens = (ensemble or {}).get(today, [])
            if today_fc.get("high") is not None:
                ens_info = ""
                if today_ens:
                    lo = min(today_ens)
                    hi = max(today_ens)
                    med = statistics.median(today_ens)
                    ens_info = f" | Ensemble: {med:.0f}F [{lo:.0f}-{hi:.0f}] ({len(today_ens)} members)"
                print(f"  {location}: High {today_fc['high']}F{ens_info}")

            await asyncio.sleep(0.3)  # Rate limit

        self._noaa_cache_time = now
        n_ens = sum(1 for v in self._ensemble_cache.values() if v)
        print(f"[WEATHER] Cached forecasts for {len(self._noaa_cache)} locations ({n_ens} with ensemble)")

    # === MARKET SCANNING ===

    def _parse_market_question(self, question: str, event_title: str) -> Optional[dict]:
        """Parse a temperature market question into structured data.

        Examples:
          "Will the highest temperature in New York City be 27F or below on December 30?"
          "Will the highest temperature in New York City be between 28 and 33F on December 30?"
          "Will the highest temperature in New York City be 46F or above on December 30?"
        """
        q = question.lower()

        # Detect location
        matched_location = None
        for loc_name, loc_data in LOCATIONS.items():
            for alias in loc_data.get("aliases", [loc_name]):
                if alias.lower() in q:
                    matched_location = loc_name
                    break
            if matched_location:
                break

        if not matched_location:
            return None

        # Parse temperature bucket
        # Polymarket formats:
        #   "be 52°F or higher" / "be 27°F or below" / "be between 68-69°F"
        #   "be 9°C" (exact) / "be -7°C" (negative) / "be 5°C or higher"
        #   Old: "be 27F or below" / "between 28 and 33F"
        temp_bucket = None
        bucket_low = None
        bucket_high = None
        is_celsius = False

        # Normalize: remove degree symbol, handle °F/°C
        qn = q.replace('°', ' ').replace('º', ' ')

        # --- FAHRENHEIT PATTERNS ---
        # "XX F or below/lower"
        m = re.search(r'(-?\d+)\s*(?:degrees?\s*)?f(?:ahrenheit)?\s+or\s+(?:below|lower)', qn)
        if m:
            bucket_high = int(m.group(1))
            bucket_low = -999
            temp_bucket = f"{bucket_high}F_or_below"

        # "between XX-YY F" or "between XX and YY F"
        if not m:
            m = re.search(r'between\s+(-?\d+)\s*[-–]\s*(-?\d+)\s*(?:degrees?\s*)?f', qn)
            if not m:
                m = re.search(r'between\s+(-?\d+)\s+and\s+(-?\d+)\s*(?:degrees?\s*)?f', qn)
            if m:
                bucket_low = int(m.group(1))
                bucket_high = int(m.group(2))
                temp_bucket = f"{bucket_low}-{bucket_high}F"

        # "XX F or above/higher"
        if not m:
            m = re.search(r'(-?\d+)\s*(?:degrees?\s*)?f(?:ahrenheit)?\s+or\s+(?:above|higher)', qn)
            if m:
                bucket_low = int(m.group(1))
                bucket_high = 999
                temp_bucket = f"{bucket_low}F_or_above"

        # --- CELSIUS PATTERNS ---
        # "XX C or below/lower"
        if not m:
            m = re.search(r'(-?\d+)\s*(?:degrees?\s*)?c(?:elsius)?\s+or\s+(?:below|lower)', qn)
            if m:
                bucket_high = int(m.group(1))
                bucket_low = -999
                temp_bucket = f"{bucket_high}C_or_below"
                is_celsius = True

        # "between XX-YY C" or "between XX and YY C"
        if not m:
            m = re.search(r'between\s+(-?\d+)\s*[-–]\s*(-?\d+)\s*(?:degrees?\s*)?c', qn)
            if not m:
                m = re.search(r'between\s+(-?\d+)\s+and\s+(-?\d+)\s*(?:degrees?\s*)?c', qn)
            if m:
                bucket_low = int(m.group(1))
                bucket_high = int(m.group(2))
                temp_bucket = f"{bucket_low}-{bucket_high}C"
                is_celsius = True

        # "XX C or above/higher"
        if not m:
            m = re.search(r'(-?\d+)\s*(?:degrees?\s*)?c(?:elsius)?\s+or\s+(?:above|higher)', qn)
            if m:
                bucket_low = int(m.group(1))
                bucket_high = 999
                temp_bucket = f"{bucket_low}C_or_above"
                is_celsius = True

        # Exact single Celsius: "be 9 C" or "be -7 C" (1-degree bucket)
        if not m:
            m = re.search(r'be\s+(-?\d+)\s*(?:degrees?\s*)?c\b', qn)
            if m:
                val = int(m.group(1))
                bucket_low = val
                bucket_high = val + 1  # 1-degree bucket: 9°C means [9, 10)
                temp_bucket = f"{val}C"
                is_celsius = True

        # Exact single Fahrenheit: "be 52 F" (1-degree bucket)
        if not m:
            m = re.search(r'be\s+(-?\d+)\s*(?:degrees?\s*)?f\b', qn)
            if m:
                val = int(m.group(1))
                bucket_low = val
                bucket_high = val + 1
                temp_bucket = f"{val}F"

        if not temp_bucket:
            return None

        # Parse date from event title or question
        # "on December 30" / "on January 1" / "February 10"
        date_match = re.search(
            r'(?:on\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d+)',
            q
        )
        target_date = None
        if date_match:
            month_str = date_match.group(0).split()[0] if 'on' not in date_match.group(0) else date_match.group(0).split()[1]
            day = int(date_match.group(1))
            month_map = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12
            }
            month_num = month_map.get(month_str.lower(), 0)
            if month_num:
                year = datetime.now(timezone.utc).year
                try:
                    target_date = f"{year}-{month_num:02d}-{day:02d}"
                except:
                    pass

        # is_celsius already set during pattern matching above
        if not is_celsius:
            is_celsius = "c" in temp_bucket.lower() and "f" not in temp_bucket.lower()

        return {
            "location": matched_location,
            "temp_bucket": temp_bucket,
            "bucket_low": bucket_low,
            "bucket_high": bucket_high,
            "target_date": target_date,
            "is_celsius": is_celsius,
        }

    def _calculate_bucket_probability(self, forecast_high: float, bucket_low: float,
                                       bucket_high: float, spread: float = 5.0) -> float:
        """Calculate probability that actual temp falls in bucket.

        Uses a simple triangular distribution centered on NOAA forecast.
        Spread defines the +/- range (e.g., +/-5F = 10F total range).
        """
        # Triangular distribution: peak at forecast, extends +/- spread
        lo = forecast_high - spread
        hi = forecast_high + spread
        peak = forecast_high

        # Clamp bucket bounds
        a = max(lo, bucket_low) if bucket_low > -999 else lo
        b = min(hi, bucket_high) if bucket_high < 999 else hi

        if a >= b:
            # Bucket is entirely outside forecast range
            if bucket_high < 999 and bucket_high < lo:
                return 0.0  # Bucket too cold
            if bucket_low > -999 and bucket_low > hi:
                return 0.0  # Bucket too hot
            return 0.0

        # CDF of triangular distribution
        def tri_cdf(x):
            if x <= lo:
                return 0.0
            elif x <= peak:
                return (x - lo) ** 2 / ((hi - lo) * (peak - lo))
            elif x <= hi:
                return 1.0 - (hi - x) ** 2 / ((hi - lo) * (hi - peak))
            else:
                return 1.0

        prob = tri_cdf(b) - tri_cdf(a)
        return max(0.0, min(1.0, prob))

    async def fetch_weather_markets(self, client: httpx.AsyncClient) -> List[dict]:
        """Fetch active temperature markets from Polymarket."""
        try:
            # tag_slug=weather contains all temperature markets
            # (tag_slug=temperature returns empty — wrong tag)
            r = await client.get(
                f"{self.GAMMA_BASE}/events",
                params={
                    "tag_slug": "weather",
                    "active": "true",
                    "closed": "false",
                    "limit": 100,
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15
            )
            if r.status_code != 200:
                print(f"[GAMMA] HTTP {r.status_code}")
                return self._market_cache

            events = r.json()
            markets = []
            for event in events:
                event_title = event.get("title", "")
                # Only include "Highest temperature" events (skip general weather)
                if "temperature" not in event_title.lower() and "highest" not in event_title.lower():
                    continue
                for m in event.get("markets", []):
                    if m.get("closed"):
                        continue
                    m["_event_title"] = event_title
                    markets.append(m)

            if markets:
                self._market_cache = markets
                return markets
            else:
                return self._market_cache

        except Exception as e:
            print(f"[GAMMA] Error: {e}")
            return self._market_cache

    def _get_market_price(self, market: dict) -> Optional[float]:
        """Get YES price from market."""
        prices = market.get("outcomePrices", [])
        outcomes = market.get("outcomes", [])

        if isinstance(prices, str):
            prices = json.loads(prices)
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)

        for i, outcome in enumerate(outcomes):
            if i < len(prices) and str(outcome).lower() in ("yes", "above"):
                return float(prices[i])

        # Default: first price
        if prices:
            return float(prices[0])
        return None

    # === TRADING LOGIC ===

    async def run_cycle(self):
        """Run one scan + trade cycle."""
        self.scan_count += 1

        async with httpx.AsyncClient(timeout=15) as client:
            # Refresh NOAA/Open-Meteo forecasts if stale
            await self.refresh_forecasts(client)

            # Fetch active weather markets
            markets = await self.fetch_weather_markets(client)

            if not markets:
                print("[SCAN] No active temperature markets found")
                return

            print(f"[SCAN #{self.scan_count}] Found {len(markets)} active temperature markets")

            # Check existing open trades for resolution/take-profit
            await self._check_open_trades(markets)

            # Count open trades
            open_count = sum(1 for t in self.trades.values() if t.status == "open")
            if open_count >= self.MAX_OPEN_TRADES:
                print(f"[RISK] Max open trades ({self.MAX_OPEN_TRADES}) reached")
                return

            # Scan for new opportunities
            new_trades = 0
            for market in markets:
                if new_trades >= self.MAX_TRADES_PER_RUN:
                    break

                question = market.get("question", "")
                event_title = market.get("_event_title", "")
                condition_id = market.get("conditionId", "")

                # Skip if we already have ANY trade on this market (open OR closed)
                # Bug fix: checking only "open" caused rebuy loops on same market
                if any(t.condition_id == condition_id
                       for t in self.trades.values()):
                    continue

                # Parse market question
                parsed = self._parse_market_question(question, event_title)
                if not parsed:
                    continue

                location = parsed["location"]
                bucket_low = parsed["bucket_low"]
                bucket_high = parsed["bucket_high"]
                target_date = parsed["target_date"]
                is_celsius = parsed["is_celsius"]

                # Skip markets for dates that have already passed
                # Bug fix: bot was buying already-resolved past-day markets
                if target_date:
                    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    if target_date < today_str:
                        continue

                # Get market price (YES side = probability this bucket hits)
                market_price = self._get_market_price(market)
                if market_price is None:
                    continue

                # Entry filter: only buy cheap buckets, but not TOO cheap
                # Markets at $0.01 or below are likely resolved/dead
                if market_price >= self.ENTRY_THRESHOLD:
                    continue
                if market_price <= 0.01:
                    continue

                # Get forecast for this location + date
                forecast = self._noaa_cache.get(location, {})
                if not forecast:
                    continue

                date_forecast = forecast.get(target_date, {})
                forecast_high = date_forecast.get("high")
                if forecast_high is None:
                    continue

                # Convert Celsius bucket bounds to Fahrenheit for comparison
                # (forecasts are stored in Fahrenheit)
                calc_low = bucket_low
                calc_high = bucket_high
                if is_celsius:
                    calc_low = bucket_low * 9 / 5 + 32 if bucket_low > -999 else -999
                    calc_high = bucket_high * 9 / 5 + 32 if bucket_high < 999 else 999

                # V1.1: Use ensemble probability if available (much more accurate)
                ensemble_members = self._ensemble_cache.get(location, {}).get(target_date, [])
                confidence = 0.0

                if ensemble_members:
                    noaa_prob, confidence = self._ensemble_bucket_probability(
                        ensemble_members, calc_low, calc_high
                    )
                else:
                    # Fallback: triangular distribution from point forecast
                    spread = self.FORECAST_SPREAD_C * 9 / 5 if is_celsius else self.FORECAST_SPREAD_F
                    noaa_prob = self._calculate_bucket_probability(
                        forecast_high, calc_low, calc_high, spread
                    )
                    confidence = 0.5  # No ensemble = low confidence

                # Edge = NOAA probability - market price
                edge = noaa_prob - market_price

                # Log interesting cheap markets (first 3 scans for debug)
                src = "ENS" if ensemble_members else "TRI"
                if self.scan_count <= 3 and market_price < 0.10:
                    print(f"  [EVAL] {location} {parsed['temp_bucket']} | Mkt: ${market_price:.3f} {src}: {noaa_prob:.1%} Edge: {edge:.1%} Conf: {confidence:.0%} | Fcst: {forecast_high}")

                # Trade filters
                if noaa_prob < self.MIN_NOAA_PROBABILITY:
                    continue
                if edge < self.MIN_EDGE:
                    continue
                # V1.1: Confidence filter — only trade when ensemble agrees
                if confidence < self.MIN_CONFIDENCE:
                    continue

                # === EXECUTE PAPER TRADE ===
                # Simulate fill with +$0.01 spread offset
                entry_price = round(market_price + 0.01, 4)
                position_size = min(self.MAX_POSITION, max(self.MIN_POSITION,
                    self.MAX_POSITION * (edge / 0.50)))  # Scale by edge strength

                trade_id = f"wx_{condition_id[:8]}_{int(time.time())}"
                trade = WeatherTrade(
                    trade_id=trade_id,
                    market_title=question[:80],
                    location=location,
                    temp_bucket=parsed["temp_bucket"],
                    entry_price=entry_price,
                    entry_time=datetime.now(timezone.utc).isoformat(),
                    size_usd=round(position_size, 2),
                    noaa_forecast_high=forecast_high,
                    noaa_probability=round(noaa_prob, 4),
                    market_probability=round(market_price, 4),
                    edge=round(edge, 4),
                    confidence=round(confidence, 4),
                    source="ensemble" if ensemble_members else "triangular",
                    condition_id=condition_id,
                )
                self.trades[trade_id] = trade
                new_trades += 1

                shares = position_size / entry_price
                potential_profit = shares * (1.0 - entry_price)
                print(f"[NEW TRADE] {location.upper()} | {parsed['temp_bucket']} [{src}]")
                print(f"  Forecast: {forecast_high}F -> P(bucket)={noaa_prob:.0%} vs Market: ${market_price:.2f}")
                print(f"  EDGE: {edge:.0%} | Confidence: {confidence:.0%} | Entry: ${entry_price:.4f} | Size: ${position_size:.2f}")
                print(f"  Potential: ${potential_profit:.2f} if correct | {question[:60]}...")

        self._save()

    async def _check_open_trades(self, markets: List[dict]):
        """Check open trades for resolution or take-profit exit."""
        market_map = {m.get("conditionId", ""): m for m in markets}

        for tid, trade in list(self.trades.items()):
            if trade.status != "open":
                continue

            market = market_map.get(trade.condition_id)
            if not market:
                # Market may have resolved and disappeared
                # Check via age - if older than 48 hours, mark as timeout
                try:
                    entry_dt = datetime.fromisoformat(trade.entry_time)
                    age_hours = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
                    if age_hours > 48:
                        trade.status = "closed"
                        trade.exit_time = datetime.now(timezone.utc).isoformat()
                        trade.pnl = -trade.size_usd  # Assume loss
                        trade.exit_reason = "timeout"
                        self.total_pnl += trade.pnl
                        self.losses += 1
                        self._update_location_stats(trade.location, False, trade.pnl)
                        print(f"[TIMEOUT] {trade.location} {trade.temp_bucket} | PnL: ${trade.pnl:+.2f}")
                except:
                    pass
                continue

            # Check current price
            current_price = self._get_market_price(market)
            if current_price is None:
                continue

            is_closed = market.get("closed", False)

            # Resolution check
            # Bug fix: removed current_price <= 0.05 — cheap price doesn't mean resolved,
            # it just means unlikely. Only resolve on is_closed or price >= 0.95 (confirmed win).
            if is_closed or current_price >= 0.95:
                trade.exit_price = current_price
                trade.exit_time = datetime.now(timezone.utc).isoformat()
                trade.status = "closed"

                if current_price >= 0.95:
                    # WIN - bucket was correct
                    shares = trade.size_usd / trade.entry_price
                    trade.pnl = shares * 1.0 - trade.size_usd
                    trade.exit_reason = "resolved_win"
                    self.wins += 1
                elif is_closed and current_price <= 0.10:
                    # LOSS - market closed AND price near zero = wrong bucket
                    trade.pnl = -trade.size_usd
                    trade.exit_reason = "resolved_loss"
                    self.losses += 1
                else:
                    # Market closed at intermediate price
                    shares = trade.size_usd / trade.entry_price
                    trade.pnl = shares * current_price - trade.size_usd
                    trade.exit_reason = "resolved_partial"
                    if trade.pnl > 0:
                        self.wins += 1
                    else:
                        self.losses += 1

                self.total_pnl += trade.pnl
                self._update_location_stats(trade.location, trade.pnl > 0, trade.pnl)
                result = "WIN" if trade.pnl > 0 else "LOSS"
                print(f"[{result}] {trade.location} {trade.temp_bucket} | PnL: ${trade.pnl:+.2f} | {trade.exit_reason}")
                continue

            # Take-profit check
            if current_price >= self.EXIT_THRESHOLD:
                trade.exit_price = current_price
                trade.exit_time = datetime.now(timezone.utc).isoformat()
                trade.status = "closed"

                shares = trade.size_usd / trade.entry_price
                trade.pnl = shares * current_price - trade.size_usd
                trade.exit_reason = "take_profit"
                self.total_pnl += trade.pnl
                if trade.pnl > 0:
                    self.wins += 1
                else:
                    self.losses += 1
                self._update_location_stats(trade.location, trade.pnl > 0, trade.pnl)
                print(f"[TP] {trade.location} {trade.temp_bucket} | Sold ${current_price:.2f} | PnL: ${trade.pnl:+.2f}")

    def _update_location_stats(self, location: str, won: bool, pnl: float):
        """Update per-location performance stats."""
        if location not in self.location_stats:
            self.location_stats[location] = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}
        stats = self.location_stats[location]
        stats["trades"] += 1
        stats["wins" if won else "losses"] += 1
        stats["pnl"] += pnl

    def print_update(self):
        """Print status update."""
        now = datetime.now(timezone.utc)
        open_trades = [t for t in self.trades.values() if t.status == "open"]
        closed_trades = [t for t in self.trades.values() if t.status == "closed"]

        print()
        print("=" * 70)
        print(f"WEATHER ARBITRAGE PAPER TRADER - {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 70)

        total = self.wins + self.losses
        wr = (self.wins / max(1, total)) * 100
        print(f"  Scans: {self.scan_count} | Trades: {total} ({self.wins}W/{self.losses}L)")
        print(f"  Win Rate: {wr:.1f}% | Total PnL: ${self.total_pnl:+.2f}")
        print(f"  Open: {len(open_trades)} | Closed: {len(closed_trades)}")

        # Config
        print(f"\n  Config: Entry <${self.ENTRY_THRESHOLD} | Exit >${self.EXIT_THRESHOLD} | Edge >{self.MIN_EDGE:.0%}")
        print(f"  Positions: ${self.MIN_POSITION}-${self.MAX_POSITION} | Max open: {self.MAX_OPEN_TRADES}")

        # Forecast cache
        print(f"\n  Forecast cache: {len(self._noaa_cache)} locations")
        for loc in sorted(self._noaa_cache.keys()):
            fc = self._noaa_cache[loc]
            today = now.strftime("%Y-%m-%d")
            today_fc = fc.get(today, {})
            if today_fc.get("high") is not None:
                print(f"    {loc}: High {today_fc['high']}F / Low {today_fc.get('low', '?')}F")

        # Per-location stats
        if self.location_stats:
            print(f"\n  LOCATION PERFORMANCE:")
            for loc, stats in sorted(self.location_stats.items()):
                loc_total = stats["trades"]
                loc_wr = stats["wins"] / max(1, loc_total) * 100
                print(f"    {loc}: {loc_total}T {loc_wr:.0f}%WR ${stats['pnl']:+.2f}")

        # Open trades
        if open_trades:
            print(f"\n  OPEN TRADES ({len(open_trades)}):")
            for t in open_trades[:10]:
                print(f"    {t.location} {t.temp_bucket} @ ${t.entry_price:.4f} | NOAA:{t.noaa_probability:.0%} Edge:{t.edge:.0%} | ${t.size_usd:.2f}")

        # Recent closed
        if closed_trades:
            print(f"\n  RECENT CLOSED:")
            recent = sorted(closed_trades, key=lambda x: x.exit_time or "", reverse=True)[:5]
            for t in recent:
                result = "WIN" if t.pnl > 0 else "LOSS"
                print(f"    [{result}] {t.location} {t.temp_bucket} ${t.pnl:+.2f} ({t.exit_reason})")

        print("=" * 70)
        print()

    async def run(self):
        """Main loop."""
        print("=" * 70)
        print("WEATHER ARBITRAGE PAPER TRADER v1.1 — Multi-Source Ensemble")
        print("=" * 70)
        print("Strategy: NOAA + Open-Meteo + ECMWF Ensemble (51 members)")
        print(f"  Entry: Buy when market < ${self.ENTRY_THRESHOLD} and forecast says >{self.MIN_NOAA_PROBABILITY:.0%}")
        print(f"  Exit:  Sell when price > ${self.EXIT_THRESHOLD} or market resolves")
        print(f"  Edge:  Min {self.MIN_EDGE:.0%} | Confidence: Min {self.MIN_CONFIDENCE:.0%}")
        print(f"  Sizing: ${self.MIN_POSITION}-${self.MAX_POSITION} per trade")
        print(f"  Scan:  Every {self.SCAN_INTERVAL}s ({self.SCAN_INTERVAL/60:.0f} min)")
        print()
        print(f"  Locations ({len(LOCATIONS)}): {', '.join(sorted(LOCATIONS.keys()))}")
        print(f"  US (NOAA): {', '.join(k for k, v in LOCATIONS.items() if v.get('noaa_office'))}")
        print(f"  Global (Open-Meteo): {', '.join(k for k, v in LOCATIONS.items() if not v.get('noaa_office'))}")
        print()
        print(f"  Existing trades: {len(self.trades)} | PnL: ${self.total_pnl:+.2f}")
        print("=" * 70)
        print()

        last_update = 0
        cycle = 0

        while True:
            try:
                cycle += 1
                now = time.time()

                await self.run_cycle()

                # 10-minute update
                if now - last_update >= 600:
                    self.print_update()
                    last_update = now

                await asyncio.sleep(self.SCAN_INTERVAL)

            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(60)

        self.print_update()
        print(f"Results saved to: {self.OUTPUT_FILE}")


if __name__ == "__main__":
    from pid_lock import acquire_pid_lock, release_pid_lock
    acquire_pid_lock("weather_paper")
    try:
        trader = WeatherArbitrageTrader()
        asyncio.run(trader.run())
    finally:
        release_pid_lock("weather_paper")
