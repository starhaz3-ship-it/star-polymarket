"""
Sniper 5M LIVE Trader V3.2

Front-runs Chainlink oracle delay on 5-minute BTC+ETH Up/Down markets.
Enters at 62-61 seconds before market close when direction is clear from CLOB orderbook.

V3.2 — ML-driven profit maximization (22-trade deep analysis):
  Live data: DOWN 93.8% WR (15W/1L) vs UP 33.3% WR (2W/4L)
  Entry >= $0.75: 100% WR live (6W/0L). Every loss came from entries < $0.75.
  total_depth >= 500: 25% WR (1W/3L, -$2.34/trade). Below 500: 93.3% WR.
  Hours 9,10,12,23 UTC: all 5 losses. Without them: 17W/0L.
  Changes:
  - MIN_ENTRY_UP: $0.72 → $0.80 (UP only profitable at high confidence)
  - MIN_CONFIDENCE: $0.55 → $0.75 (raise floor for ALL trades)
  - SKIP_HOURS: {9} → {9, 10, 12, 23} (all live loss hours)
  - MAX_DEPTH filter: 500 shares (high-depth = informed market makers)
  - Theoretical: 14W/0L, 100% WR, +$4.93/day (6x improvement)

Usage:
  python -u run_sniper_5m_live.py --live
"""

import sys
import json
import time
import os
import math
import asyncio
import functools
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

import httpx
from dotenv import load_dotenv

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from pid_lock import acquire_pid_lock, release_pid_lock
from adaptive_tuner import ParameterTuner, SNIPER_TUNER_CONFIG

# ============================================================================
# CONFIG
# ============================================================================
SNIPER_WINDOW = 63          # seconds before close to start looking (front-run whales)
MIN_CONFIDENCE = 0.75       # V3.2: raised from $0.55 — every live loss was entry < $0.75. At $0.75+: 100% WR (6W/0L)
MAX_ENTRY_PRICE = 0.80      # V3.1: raised from $0.78 — paper shows 79.3% WR at $0.80, 42% more trades
BASE_TRADE_SIZE = 3.00      # $3/trade per Star directive
SCAN_INTERVAL = 5           # seconds between scans
MAX_CONCURRENT = 2          # 1 directional + 1 oracle

# Oracle front-run: enter AFTER close when Binance direction is KNOWN
ORACLE_MAX_ENTRY = 0.95     # max entry ($0.05+/sh guaranteed profit)
ORACLE_WINDOW = 60          # seconds after close to look
ORACLE_MIN_DELAY = 2        # min seconds after close (candle finalizes fast)
ORACLE_GTC_TIMEOUT = 15     # seconds to wait for GTC fill
ORACLE_BOOK_LOG = Path(__file__).parent / "oracle_book_data.json"
DAILY_LOSS_LIMIT = 30.0     # stop trading if daily losses exceed this
LOG_INTERVAL = 30           # print status every N seconds
RESOLVE_DELAY = 15          # seconds after market close before resolving
MIN_SHARES = 3              # minimum shares for fill confirmation
FOK_SLIPPAGE = 0.03         # cents above best ask to sweep multiple price levels

RESULTS_FILE = Path(__file__).parent / "sniper_5m_live_results.json"
ML_STATE_FILE = Path(__file__).parent / "sniper_ml_depth_state.json"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
GAMMA_API_URL = "https://gamma-api.polymarket.com/events"

# ML minimum trades before filter activates (collect data first)
ML_MIN_TRADES = 8
ML_MIN_WR_THRESHOLD = 0.50  # skip if estimated WR below this (DOWN trades)
ML_UP_WR_THRESHOLD = 0.80   # V2.1: UP needs 80%+ ML WR (DOWN=11W/0L but UP=2W/3L, losses at 72-76%)
MIN_ENTRY_UP = 0.80          # V3.2: raised from $0.72 — UP only profitable at $0.80+ (live: UP 33% WR overall, 100% at $0.80+)
UP_SIZE_MULT = 0.50          # V2.1: Half size on UP trades until UP direction proves itself
SKIP_HOURS_UTC = {9, 10, 12, 23}  # V3.2: All 5 live losses in these hours. Without them: 17W/0L, +$15.85
ETH_SIZE_MULT = 0.34        # V2.3: ETH minimum size (~$1/trade) until proven profitable
MAX_DEPTH_SHARES = 500      # V3.2: depth >= 500 = 25% WR (1W/3L, -$2.34/trade). Below 500: 93.3% WR (14W/1L)
ASSETS = {"bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH", "eth": "ETH"}
BINANCE_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}

# V3.0: Circuit Breaker + Kelly Sizing + SynthData
CIRCUIT_BREAKER_LOSSES = 3      # consecutive losses → full halt
CIRCUIT_BREAKER_HALFSIZE = 2    # consecutive losses → half-size mode
CIRCUIT_BREAKER_COOLDOWN = 1800  # 30 min cooldown after halt
CIRCUIT_BREAKER_RECOVERY = 2    # trades at half-size before resuming full
WARMUP_SECONDS = 30             # seconds of stable data required before trading
API_BACKOFF_BASE = 2.0          # exponential backoff base (seconds)
API_BACKOFF_MAX = 30.0          # max backoff delay
KELLY_CAP = 0.05                # 5% max Kelly fraction
KELLY_MIN_TRADES = 10           # min trades before Kelly activates
BANKROLL_ESTIMATE = 100.0       # conservative bankroll estimate
SYNTH_API_KEY = os.environ.get("SYNTHDATA_API_KEY", "")
SYNTH_EDGE_THRESHOLD = 0.05    # 5pp minimum edge for SynthData boost
SYNTH_SIZE_BOOST = 1.5         # 1.5x when Synth agrees with signal
SYNTH_SIZE_REDUCE = 0.5        # 0.5x when Synth strongly disagrees


# ============================================================================
# ML DEPTH FILTER — learns which orderbook shapes predict wins
# ============================================================================
class DepthMLFilter:
    """Thompson sampling on orderbook depth features to auto-tune entry quality."""

    def __init__(self):
        self.state = {"features": [], "buckets": {}}
        self._load()

    def _load(self):
        if ML_STATE_FILE.exists():
            try:
                self.state = json.loads(ML_STATE_FILE.read_text())
            except Exception:
                pass

    def _save(self):
        try:
            ML_STATE_FILE.write_text(json.dumps(self.state, indent=2))
        except Exception:
            pass

    @staticmethod
    def compute_features(levels: list, entry_price: float, confidence: float,
                         signal_source: str, up_ask: float, down_ask: float) -> dict:
        """Extract ML features from orderbook depth levels."""
        if not levels:
            return {
                "best_ask_ratio": 0.0, "best_is_largest": False,
                "depth_levels": 0, "total_depth": 0,
                "confidence": confidence, "signal_source": signal_source,
                "opposing_price": 0.0, "spread_sum": 0.0,
            }

        top4 = levels[:4]
        prices = [p for p, s in top4]
        sizes = [s for p, s in top4]
        total_top4 = sum(sizes)
        best_ask_size = sizes[0] if sizes else 0

        # Key features
        best_ask_ratio = best_ask_size / total_top4 if total_top4 > 0 else 0.0
        largest_idx = sizes.index(max(sizes)) if sizes else 0
        best_is_largest = (largest_idx == 0)
        opposing_price = min(up_ask, down_ask) if up_ask and down_ask else 0.0
        spread_sum = (up_ask or 0) + (down_ask or 0)

        return {
            "best_ask_ratio": round(best_ask_ratio, 3),
            "best_is_largest": best_is_largest,
            "depth_levels": len(top4),
            "total_depth": int(total_top4),
            "confidence": round(confidence, 3),
            "signal_source": signal_source,
            "opposing_price": round(opposing_price, 3),
            "spread_sum": round(spread_sum, 3),
        }

    def record_outcome(self, features: dict, won: bool):
        """Record a trade outcome with its features for learning."""
        entry = {**features, "won": won, "ts": datetime.now(timezone.utc).isoformat()}
        self.state["features"].append(entry)
        self.state["features"] = self.state["features"][-200:]  # keep last 200

        # Update Thompson buckets
        for bucket_name, bucket_val in self._get_buckets(features):
            if bucket_name not in self.state["buckets"]:
                self.state["buckets"][bucket_name] = {"alpha": 1, "beta": 1}
            b = self.state["buckets"][bucket_name]
            if won:
                b["alpha"] += 1
            else:
                b["beta"] += 1

        self._save()

    def _get_buckets(self, features: dict) -> list:
        """Map features to Thompson sampling buckets."""
        buckets = []

        # Bucket 1: best_is_largest (True/False)
        bil = features.get("best_is_largest", True)
        buckets.append((f"best_largest_{bil}", bil))

        # Bucket 2: best_ask_ratio ranges
        bar = features.get("best_ask_ratio", 0.5)
        if bar >= 0.50:
            buckets.append(("bar_high", bar))
        elif bar >= 0.25:
            buckets.append(("bar_mid", bar))
        else:
            buckets.append(("bar_low", bar))

        # Bucket 3: signal source
        src = features.get("signal_source", "clob")
        buckets.append((f"signal_{src}", src))

        # Bucket 4: confidence ranges
        conf = features.get("confidence", 0.70)
        if conf >= 0.74:
            buckets.append(("conf_high", conf))
        else:
            buckets.append(("conf_low", conf))

        # Bucket 5: opposing side strength
        opp = features.get("opposing_price", 0.30)
        if opp >= 0.30:
            buckets.append(("opp_strong", opp))
        else:
            buckets.append(("opp_weak", opp))

        return buckets

    def should_trade(self, features: dict, side: str = "DOWN") -> tuple:
        """Returns (should_trade: bool, reason: str, expected_wr: float).

        Collects data for ML_MIN_TRADES, then starts soft-filtering.
        V2.1: UP trades need ML_UP_WR_THRESHOLD (80%), DOWN uses ML_MIN_WR_THRESHOLD (50%).
        """
        total_data = len(self.state["features"])

        # Always trade while collecting data
        if total_data < ML_MIN_TRADES:
            return True, f"collecting_data({total_data}/{ML_MIN_TRADES})", 0.0

        # Calculate expected WR from matching buckets via Thompson sampling
        import random
        bucket_samples = []
        for bucket_name, _ in self._get_buckets(features):
            b = self.state["buckets"].get(bucket_name)
            if b and (b["alpha"] + b["beta"]) > 2:
                # Thompson sample from Beta distribution
                sample = random.betavariate(b["alpha"], b["beta"])
                bucket_samples.append(sample)

        if not bucket_samples:
            return True, "no_bucket_data", 0.0

        # Average across matching buckets (ensemble)
        expected_wr = sum(bucket_samples) / len(bucket_samples)

        # V2.1: Direction-aware threshold — UP needs higher confidence
        threshold = ML_UP_WR_THRESHOLD if side == "UP" else ML_MIN_WR_THRESHOLD

        if expected_wr < threshold:
            bad_buckets = []
            for bucket_name, _ in self._get_buckets(features):
                b = self.state["buckets"].get(bucket_name, {"alpha": 1, "beta": 1})
                mean = b["alpha"] / (b["alpha"] + b["beta"])
                if mean < 0.50:
                    bad_buckets.append(f"{bucket_name}={mean:.0%}")
            dir_tag = f"UP>={ML_UP_WR_THRESHOLD:.0%}" if side == "UP" else ""
            reason = f"ml_skip(wr={expected_wr:.0%} {dir_tag} " + " ".join(bad_buckets) + ")"
            return False, reason, expected_wr

        return True, f"ml_pass(wr={expected_wr:.0%})", expected_wr

    def get_report(self) -> str:
        """Return a short status line for logging."""
        total = len(self.state["features"])
        if total == 0:
            return "[ML] No data yet"
        wins = sum(1 for f in self.state["features"] if f.get("won"))
        lines = [f"[ML] {wins}W/{total-wins}L ({wins/total*100:.0f}%WR) | {len(self.state['buckets'])} buckets"]
        for bname, bdata in sorted(self.state["buckets"].items()):
            a, b = bdata["alpha"], bdata["beta"]
            n = a + b - 2  # subtract priors
            if n > 0:
                mean = a / (a + b)
                lines.append(f"  {bname}: {mean:.0%} ({n} trades)")
        return "\n".join(lines)


def get_trade_size(wins: int, losses: int, entry_price: float = 0.60,
                    cumulative_pnl: float = 0.0) -> float:
    """V3.0: Kelly criterion sizing — auto-scales based on empirical edge.
    f* = (p_win * (b+1) - 1) / b, capped at 5% of bankroll.
    Falls back to fixed $3 if <10 trades (not enough data for Kelly).
    Returns 0 if Kelly says no edge (negative f*) — caller should skip trade.
    """
    total = wins + losses
    if total < KELLY_MIN_TRADES:
        return BASE_TRADE_SIZE  # not enough data for Kelly

    bankroll = BANKROLL_ESTIMATE + cumulative_pnl
    if bankroll <= 10:
        return 1.0  # minimum when near-bankrupt

    p_win = wins / total  # empirical win rate
    p_mkt = max(min(entry_price, 0.95), 0.05)  # clamp to avoid div issues
    b = (1 - p_mkt) / p_mkt  # payout ratio (win $b for every $1 risked)

    f_star = (p_win * (b + 1) - 1) / b
    f_eff = min(max(f_star, 0), KELLY_CAP)

    if f_eff <= 0:
        return 0  # Kelly says no edge — don't trade

    size = round(bankroll * f_eff, 2)
    return max(min(size, 10.0), 1.0)  # $1 floor, $10 ceiling


# ============================================================================
# V3.0: SYNTHDATA API — external probability oracle for consensus
# ============================================================================
_synth_cache: dict = {"data": None, "ts": 0}


async def query_synthdata(asset: str = "BTC") -> Optional[dict]:
    """Query SynthData for hourly probability forecast. Cached 60s."""
    if not SYNTH_API_KEY:
        return None
    now = time.time()
    if _synth_cache["data"] and now - _synth_cache["ts"] < 60:
        cached = _synth_cache["data"]
        if cached.get("_asset") == asset:
            return cached
    try:
        url = "https://api.synthdata.co/insights/polymarket/up-down/hourly"
        headers = {"Authorization": f"Apikey {SYNTH_API_KEY}"}
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(url, headers=headers, params={"asset": asset})
            if r.status_code != 200:
                return None
            data = r.json()
        synth_up = float(data.get("synth_probability_up", 0.5))
        poly_up = float(data.get("polymarket_probability_up", 0.5))
        result = {
            "_asset": asset,
            "synth_up": synth_up,
            "synth_down": 1 - synth_up,
            "poly_up": poly_up,
            "edge_up": synth_up - poly_up,
            "edge_down": -(synth_up - poly_up),
        }
        _synth_cache["data"] = result
        _synth_cache["ts"] = now
        return result
    except Exception:
        return None


# ============================================================================
# SNIPER 5M LIVE TRADER
# ============================================================================
class Sniper5MLive:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.active: Dict[str, dict] = {}
        self.resolved: list = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0, "skipped": 0}
        self.session_pnl = 0.0
        self.attempted_cids: set = set()
        self.oracle_attempted_cids: set = set()
        self.running = True
        self._last_log_time = 0.0
        self.client = None
        self.ml_depth = DepthMLFilter()
        self.tuner = ParameterTuner(SNIPER_TUNER_CONFIG,
                                     str(Path(__file__).parent / "sniper_tuner_state.json"))
        self._load()

        # V3.0: Circuit breaker — graduated response
        self._circuit_halt_until = 0
        self._consecutive_losses = 0
        self._recovery_trades = 0  # V3.1: trades at half-size after halt
        for trade in reversed(self.resolved):
            if trade.get("pnl", 0) >= 0:
                break
            self._consecutive_losses += 1

        # V3.1: Warmup guard — no trading until data is stable
        self._warmup_start = time.time()
        self._warmup_ready = False

        # V3.1: API error backoff tracking
        self._api_consecutive_errors = 0

        if not self.paper:
            self._init_clob()

    # ========================================================================
    # CLOB CLIENT
    # ========================================================================

    def _init_clob(self):
        try:
            from py_clob_client.client import ClobClient
            from crypto_utils import decrypt_key

            password = os.getenv("POLYMARKET_PASSWORD", "")
            pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
            if pk.startswith("ENC:"):
                pk = decrypt_key(pk[4:], os.getenv("POLYMARKET_KEY_SALT", ""), password)
            proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "")

            self.client = ClobClient(
                host="https://clob.polymarket.com", key=pk, chain_id=137,
                signature_type=1, funder=proxy if proxy else None,
            )
            creds = self.client.derive_api_key()
            self.client.set_api_creds(creds)
            print(f"[CLOB] Client initialized - LIVE MODE")
            # V3.1: Startup cleanup — cancel orphaned orders from previous session
            self._cleanup_stale_orders()
        except Exception as e:
            print(f"[CLOB] Init failed: {e}")
            traceback.print_exc()
            print(f"[CLOB] Falling back to PAPER mode")
            self.paper = True

    def _cleanup_stale_orders(self):
        """Cancel any orphaned limit orders from previous session."""
        if not self.client:
            return
        try:
            resp = self.client.cancel_all()
            if resp:
                print(f"[CLEANUP] Cancelled stale orders from previous session")
        except Exception as e:
            print(f"[CLEANUP] No stale orders or error: {e}")

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _load(self):
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.active = data.get("active", {})
                self.resolved = data.get("resolved", [])
                self.stats = {**self.stats, **data.get("stats", {})}
                for cid in self.active:
                    self.attempted_cids.add(cid)
                for trade in self.resolved:
                    cid = trade.get("condition_id", "")
                    if cid:
                        if trade.get("strategy") == "oracle":
                            self.oracle_attempted_cids.add(cid)
                        else:
                            self.attempted_cids.add(cid)
                w, l = self.stats["wins"], self.stats["losses"]
                total = w + l
                wr = w / total * 100 if total > 0 else 0
                print(f"[LOAD] {len(self.resolved)} resolved, {len(self.active)} active | "
                      f"{w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f}")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _log_oracle_book(self, question: str, direction: str, secs_past: float,
                          best_ask: Optional[float], avail_shares: float, levels: list):
        """Log oracle orderbook state for ML sweet-spot analysis."""
        try:
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "question": question[:60],
                "direction": direction,
                "secs_past_close": round(secs_past, 1),
                "best_ask": round(best_ask, 4) if best_ask else None,
                "total_shares": round(avail_shares, 1),
                "levels": [(round(p, 4), round(s, 1)) for p, s in levels[:10]],
                "has_liquidity_095": best_ask is not None and best_ask <= ORACLE_MAX_ENTRY,
            }
            existing = []
            if ORACLE_BOOK_LOG.exists():
                try:
                    existing = json.loads(ORACLE_BOOK_LOG.read_text())
                except Exception:
                    existing = []
            existing.append(entry)
            existing = existing[-500:]  # keep last 500 samples
            ORACLE_BOOK_LOG.write_text(json.dumps(existing, indent=1))
        except Exception:
            pass

    def _save(self):
        data = {
            "active": self.active,
            "resolved": self.resolved[-500:],
            "stats": self.stats,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        tmp = RESULTS_FILE.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(RESULTS_FILE)
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    # ========================================================================
    # MARKET DISCOVERY
    # ========================================================================

    async def discover_5m_markets(self) -> list:
        markets = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    GAMMA_API_URL,
                    params={"tag_slug": "5M", "active": "true",
                            "closed": "false", "limit": 200},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if r.status_code != 200:
                    print(f"[API] Gamma error: {r.status_code}")
                    return markets

                now = datetime.now(timezone.utc)
                for event in r.json():
                    title = event.get("title", "").lower()
                    # V2.3: Match BTC and ETH markets
                    asset = None
                    for keyword, a in ASSETS.items():
                        if keyword in title:
                            asset = a
                            break
                    if not asset:
                        continue
                    for m in event.get("markets", []):
                        m["_asset"] = asset
                        if m.get("closed", True):
                            continue
                        end_date = m.get("endDate", "")
                        if not end_date:
                            continue
                        try:
                            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                            # Include markets up to ORACLE_WINDOW+30s past close
                            if end_dt < now - timedelta(seconds=ORACLE_WINDOW + 30):
                                continue
                            time_left_sec = (end_dt - now).total_seconds()
                            m["_time_left_sec"] = time_left_sec
                            m["_end_dt"] = end_dt.isoformat()
                            m["_end_dt_parsed"] = end_dt
                        except Exception:
                            continue
                        if not m.get("question"):
                            m["question"] = event.get("title", "")
                        markets.append(m)
        except Exception as e:
            print(f"[API] Discovery error: {e}")
        return markets

    # ========================================================================
    # ORDERBOOK READING
    # ========================================================================

    def get_token_ids(self, market: dict) -> tuple:
        outcomes = market.get("outcomes", [])
        token_ids = market.get("clobTokenIds", [])
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        up_tid, down_tid = None, None
        for i, o in enumerate(outcomes):
            if i < len(token_ids):
                if str(o).lower() == "up":
                    up_tid = token_ids[i]
                elif str(o).lower() == "down":
                    down_tid = token_ids[i]
        return up_tid, down_tid

    async def get_best_ask(self, token_id: str) -> Optional[float]:
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(CLOB_BOOK_URL, params={"token_id": token_id})
                if r.status_code != 200:
                    return None
                book = r.json()
                asks = book.get("asks", [])
                if not asks:
                    return None
                # CLOB asks are sorted descending — find the true best ask
                return min(float(a["price"]) for a in asks)
        except Exception:
            return None

    async def get_orderbook_depth(self, token_id: str, max_price: float) -> tuple:
        """Get total available shares up to max_price. Returns (best_ask, total_shares, levels)."""
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(CLOB_BOOK_URL, params={"token_id": token_id})
                if r.status_code != 200:
                    return None, 0, []
                book = r.json()
                asks = book.get("asks", [])
                if not asks:
                    return None, 0, []
                # Sort asks ascending by price
                sorted_asks = sorted(asks, key=lambda a: float(a["price"]))
                best_ask = float(sorted_asks[0]["price"])
                total_shares = 0
                levels = []
                for a in sorted_asks:
                    price = float(a["price"])
                    size = float(a["size"])
                    if price <= max_price:
                        total_shares += size
                        levels.append((price, size))
                return best_ask, total_shares, levels
        except Exception:
            return None, 0, []

    async def get_orderbook_prices(self, market: dict) -> tuple:
        up_tid, down_tid = self.get_token_ids(market)
        if not up_tid or not down_tid:
            return None, None
        up_ask, down_ask = await asyncio.gather(
            self.get_best_ask(up_tid), self.get_best_ask(down_tid)
        )
        return up_ask, down_ask

    # ========================================================================
    # BINANCE
    # ========================================================================

    async def resolve_via_binance(self, end_dt: datetime, symbol: str = "BTCUSDT") -> Optional[str]:
        try:
            start_dt = end_dt - timedelta(minutes=5)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(BINANCE_REST_URL, params={
                    "symbol": symbol, "interval": "5m",
                    "startTime": start_ms, "endTime": end_ms, "limit": 5,
                })
                r.raise_for_status()
                klines = r.json()

                if not klines:
                    r2 = await client.get(BINANCE_REST_URL, params={
                        "symbol": symbol, "interval": "1m",
                        "startTime": start_ms, "endTime": end_ms, "limit": 10,
                    })
                    r2.raise_for_status()
                    klines_1m = r2.json()
                    if not klines_1m:
                        return None
                    open_price = float(klines_1m[0][1])
                    close_price = float(klines_1m[-1][4])
                else:
                    best_candle = None
                    best_diff = float("inf")
                    for k in klines:
                        diff = abs(int(k[0]) - start_ms)
                        if diff < best_diff:
                            best_diff = diff
                            best_candle = k
                    if best_candle is None:
                        return None
                    open_price = float(best_candle[1])
                    close_price = float(best_candle[4])

                if close_price > open_price:
                    return "UP"
                elif close_price < open_price:
                    return "DOWN"
                else:
                    return "DOWN"
        except Exception as e:
            print(f"[BINANCE] Resolution error: {e}")
            return None

    async def _audit_signal_timing(self, end_dt: datetime, symbol: str = "BTCUSDT") -> dict:
        """Post-trade audit: what would Binance signal have been at 55/57/60/63/65s?"""
        audit = {}
        try:
            start_dt = end_dt - timedelta(minutes=5)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(BINANCE_REST_URL, params={
                    "symbol": symbol, "interval": "1m",
                    "startTime": start_ms, "endTime": end_ms, "limit": 10,
                })
                if r.status_code != 200:
                    return audit
                klines = r.json()
                if not klines:
                    return audit

                open_price = float(klines[0][1])

                # Build second-by-second price from 1-min candle closes
                # Each candle close represents the price at that minute boundary
                candle_closes = []
                for k in klines:
                    t_ms = int(k[0])
                    c = float(k[4])
                    # Candle at t_ms covers [t_ms, t_ms+60000)
                    # Close price is at t_ms + 60000
                    close_time = t_ms + 60000
                    secs_before_end = (end_ms - close_time) / 1000
                    candle_closes.append((secs_before_end, c))

                # Check signal at key time points
                for window in [55, 57, 60, 63, 65]:
                    # Find the candle close nearest to (but not after) this time point
                    best_price = None
                    for secs_left, price in candle_closes:
                        if secs_left >= window - 30:  # within 30s of target
                            best_price = price
                    if best_price is None and candle_closes:
                        best_price = candle_closes[-1][1]  # use latest

                    if best_price is not None:
                        pct = (best_price - open_price) / open_price * 100
                        if pct > 0.03:
                            sig = "UP"
                        elif pct < -0.03:
                            sig = "DOWN"
                        else:
                            sig = "FLAT"
                        audit[f"signal_{window}s"] = sig
                        audit[f"pct_{window}s"] = round(pct, 4)

                # Also record the actual close
                final_close = float(klines[-1][4])
                final_pct = (final_close - open_price) / open_price * 100
                audit["actual_direction"] = "UP" if final_close >= open_price else "DOWN"
                audit["actual_pct"] = round(final_pct, 4)
                audit["open_price"] = open_price
                audit["close_price"] = final_close
                audit["candle_closes"] = [(round(s, 1), round(p, 2)) for s, p in candle_closes]

        except Exception as e:
            audit["error"] = str(e)
        return audit

    async def _check_binance_direction(self, end_dt: datetime, symbol: str = "BTCUSDT") -> tuple:
        """Returns (direction_or_None, open_5m, current_price)."""
        try:
            start_dt = end_dt - timedelta(minutes=5)
            start_ms = int(start_dt.timestamp() * 1000)
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(BINANCE_REST_URL, params={
                    "symbol": symbol, "interval": "1m",
                    "startTime": start_ms, "endTime": now_ms, "limit": 10,
                })
                if r.status_code != 200:
                    return None, None, None
                klines = r.json()
                if not klines:
                    return None, None, None
                open_price = float(klines[0][1])
                close_price = float(klines[-1][4])
                pct = (close_price - open_price) / open_price * 100
                # V2.0 tuner: ML-tuned Binance threshold (was hardcoded 0.04)
                binance_thresh = self.tuner.get_active_value("binance_threshold")
                if pct > binance_thresh:
                    return "UP", open_price, close_price
                elif pct < -binance_thresh:
                    return "DOWN", open_price, close_price
                return None, open_price, close_price  # FLAT but still return prices
        except Exception:
            return None, None, None

    # ========================================================================
    # LIVE ORDER EXECUTION
    # ========================================================================

    async def place_live_order(self, market: dict, side: str, entry_price: float,
                                shares: float, trade_size: float) -> Optional[dict]:
        """Place a FOK buy order on the CLOB. Returns fill info or None."""
        if self.paper or not self.client:
            return {"filled": True, "shares": shares, "price": entry_price, "cost": trade_size}

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            up_tid, down_tid = self.get_token_ids(market)
            token_id = up_tid if side == "UP" else down_tid
            if not token_id:
                print(f"[LIVE] No token ID for {side}")
                return None

            # Add slippage to sweep multiple price levels (FOK needs all shares filled)
            fok_price = min(round(entry_price + FOK_SLIPPAGE, 2), 0.99)
            order_args = OrderArgs(
                price=fok_price,
                size=shares,
                side=BUY,
                token_id=token_id,
            )
            print(f"[LIVE] FOK {side} {shares}sh @ limit ${fok_price:.2f} (ask=${entry_price:.2f} +{FOK_SLIPPAGE:.0%} slip)")
            signed = self.client.create_order(order_args)
            resp = self.client.post_order(signed, OrderType.FOK)

            if not resp.get("success"):
                print(f"[LIVE] FOK failed: {resp.get('errorMsg', '?')}")
                return None

            order_id = resp.get("orderID", "?")
            print(f"[LIVE] FOK order {order_id[:20]}...")

            # Confirm fill
            if order_id and order_id != "?":
                await asyncio.sleep(1)
                try:
                    order_info = self.client.get_order(order_id)
                    # get_order may return string or dict
                    if isinstance(order_info, str):
                        order_info = json.loads(order_info)
                    size_matched = float(order_info.get("size_matched", 0) or 0)
                    if size_matched >= MIN_SHARES:
                        actual_price = entry_price
                        assoc = order_info.get("associate_trades", [])
                        if assoc and isinstance(assoc[0], dict) and assoc[0].get("price"):
                            actual_price = float(assoc[0]["price"])
                        actual_cost = round(actual_price * size_matched, 2)
                        self._api_consecutive_errors = 0  # V3.1: reset backoff on success
                        print(f"[LIVE] FILLED: {size_matched:.1f}sh @ ${actual_price:.2f} = ${actual_cost:.2f}")
                        return {
                            "filled": True,
                            "shares": size_matched,
                            "price": actual_price,
                            "cost": actual_cost,
                            "order_id": order_id,
                        }
                    else:
                        status = order_info.get("status", "?")
                        print(f"[LIVE] NOT FILLED: status={status} matched={size_matched}")
                        return None
                except json.JSONDecodeError:
                    # Can't parse — assume filled since FOK succeeded
                    print(f"[LIVE] Fill check parse error — assuming filled (FOK success)")
                    return {
                        "filled": True,
                        "shares": shares,
                        "price": entry_price,
                        "cost": trade_size,
                        "order_id": order_id,
                    }
                except Exception as e:
                    print(f"[LIVE] Fill check error: {e} — assuming filled (FOK success)")
                    return {
                        "filled": True,
                        "shares": shares,
                        "price": entry_price,
                        "cost": trade_size,
                        "order_id": order_id,
                    }
            return None
        except Exception as e:
            # V3.1: Exponential backoff on API errors
            self._api_consecutive_errors += 1
            backoff = min(API_BACKOFF_BASE ** self._api_consecutive_errors, API_BACKOFF_MAX)
            print(f"[LIVE] Order error #{self._api_consecutive_errors}: {e} — backoff {backoff:.0f}s")
            await asyncio.sleep(backoff)
            return None

    # ========================================================================
    # ORACLE ORDER EXECUTION
    # ========================================================================

    async def place_oracle_order(self, market: dict, side: str, entry_price: float,
                                  shares: float, trade_size: float) -> Optional[dict]:
        """GTC buy for oracle strategy — direction is KNOWN, need fill at <= $0.95."""
        if self.paper or not self.client:
            return {"filled": True, "shares": shares, "price": entry_price, "cost": trade_size}

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            up_tid, down_tid = self.get_token_ids(market)
            token_id = up_tid if side == "UP" else down_tid
            if not token_id:
                return None

            gtc_price = min(round(entry_price + 0.02, 2), ORACLE_MAX_ENTRY)
            order_args = OrderArgs(
                price=gtc_price,
                size=shares,
                side=BUY,
                token_id=token_id,
            )
            print(f"[ORACLE] GTC {side} {shares}sh @ ${gtc_price:.2f}")
            signed = self.client.create_order(order_args)
            resp = self.client.post_order(signed, OrderType.GTC)

            if not resp.get("success"):
                print(f"[ORACLE] Order failed: {resp.get('errorMsg', '?')}")
                return None

            order_id = resp.get("orderID", "?")

            # Poll for fill
            for _ in range(ORACLE_GTC_TIMEOUT // 3):
                await asyncio.sleep(3)
                try:
                    order_info = self.client.get_order(order_id)
                    if isinstance(order_info, str):
                        order_info = json.loads(order_info)
                    size_matched = float(order_info.get("size_matched", 0) or 0)
                    if size_matched >= MIN_SHARES:
                        actual_price = entry_price
                        assoc = order_info.get("associate_trades", [])
                        if assoc and isinstance(assoc[0], dict) and assoc[0].get("price"):
                            actual_price = float(assoc[0]["price"])
                        actual_cost = round(actual_price * size_matched, 2)
                        print(f"[ORACLE] FILLED: {size_matched:.1f}sh @ ${actual_price:.2f}")
                        return {
                            "filled": True,
                            "shares": size_matched,
                            "price": actual_price,
                            "cost": actual_cost,
                            "order_id": order_id,
                        }
                except Exception as e:
                    print(f"[ORACLE] Fill check: {e}")

            # Not filled — cancel
            try:
                self.client.cancel(order_id)
                print(f"[ORACLE] Cancelled unfilled order")
            except Exception:
                pass
            return None
        except Exception as e:
            print(f"[ORACLE] Order error: {e}")
            traceback.print_exc()
            return None

    # ========================================================================
    # SIGNAL + ENTRY
    # ========================================================================

    async def check_sniper_entry(self, markets: list):
        if len(self.active) >= MAX_CONCURRENT:
            return
        if self.session_pnl <= -DAILY_LOSS_LIMIT:
            return
        # V3.0: Circuit breaker — full halt after 3 losses
        if time.time() < self._circuit_halt_until:
            return
        # V3.1: Warmup guard — wait for stable data after startup
        if not self._warmup_ready:
            elapsed = time.time() - self._warmup_start
            if elapsed < WARMUP_SECONDS:
                return
            self._warmup_ready = True
            print(f"[WARMUP] Ready after {elapsed:.0f}s")

        for market in markets:
            if len(self.active) >= MAX_CONCURRENT:
                break

            cid = market.get("conditionId", "")
            if not cid or cid in self.attempted_cids:
                continue

            time_left_sec = market.get("_time_left_sec", 9999)

            # Only act in sniper window
            if time_left_sec > SNIPER_WINDOW or time_left_sec <= 0:
                continue

            question = market.get("question", "?")
            end_dt = market.get("_end_dt_parsed")

            # V2.2: Skip toxic hours (ALL 3 losses came from UTC 9+12)
            # V2.0 tuner: also skip hours with negative EV (ML-dynamic)
            if end_dt:
                dynamic_skip_hours = set(self.tuner.get_negative_ev_hours())
                all_skip_hours = SKIP_HOURS_UTC | dynamic_skip_hours
                if end_dt.hour in all_skip_hours:
                    source = "hardcoded" if end_dt.hour in SKIP_HOURS_UTC else "ML-tuned"
                    self.stats["skipped"] += 1
                    self.attempted_cids.add(cid)
                    print(f"[SKIP] Toxic hour {end_dt.hour} UTC ({source}) | {question[:50]}")
                    continue

            # STEP 1: Get CLOB prices for both sides
            up_tid, down_tid = self.get_token_ids(market)
            up_ask = (await self.get_best_ask(up_tid)) if up_tid else None
            down_ask = (await self.get_best_ask(down_tid)) if down_tid else None
            up_display = up_ask if up_ask is not None else 0.0
            down_display = down_ask if down_ask is not None else 0.0

            # V2.3: Asset-aware Binance symbol
            asset = market.get("_asset", "BTC")
            binance_sym = BINANCE_SYMBOLS.get(asset, "BTCUSDT")

            # STEP 2: Determine side — Binance first, CLOB fallback
            side = None
            signal_source = "binance"
            binance_open = None
            binance_current = None
            if end_dt:
                side, binance_open, binance_current = await self._check_binance_direction(end_dt, symbol=binance_sym)

            if not side:
                # V3.1: CLOB CONFIDENCE: $0.65+ with dominance → signal
                # Previous bug: clob_cap (~0.78) threw away 226 high-confidence signals at $0.79-$0.85
                # Now: use MAX_ENTRY_PRICE as cap (checked again at entry), let more signals through
                clob_cap = MAX_ENTRY_PRICE
                if up_ask and 0.65 <= up_ask <= clob_cap and (down_ask is None or up_ask > down_ask):
                    side = "UP"
                    signal_source = "clob"
                elif down_ask and 0.65 <= down_ask <= clob_cap and (up_ask is None or down_ask > up_ask):
                    side = "DOWN"
                    signal_source = "clob"

            if not side:
                # Shadow-track: if CLOB has ANY dominant side >= 0.65, record for ML learning
                dominant = max(up_ask or 0, down_ask or 0)
                shadow_side = "UP" if (up_ask or 0) > (down_ask or 0) else "DOWN"
                if dominant >= 0.65 and end_dt:
                    _bmove = ((binance_current - binance_open) / binance_open * 100
                              if binance_open and binance_current else 0)
                    self.tuner.record_shadow(
                        market_id=cid, end_time_iso=end_dt.isoformat(),
                        side=shadow_side, entry_price=dominant,
                        clob_dominant=dominant,
                        extra={"binance_move": round(_bmove, 4),
                               "time_left_sec": round(time_left_sec, 1)})
                self.stats["skipped"] += 1
                self.attempted_cids.add(cid)
                print(f"[SKIP] No signal | Binance flat + CLOB unclear "
                      f"UP=${up_display:.2f} DN=${down_display:.2f} | {time_left_sec:.0f}s | {question[:50]}")
                continue

            # STEP 2b: Binance alignment check — price must be on our side of 5M open
            # Analysis of 30 trades: 26W all aligned, 4L all misaligned (100% discriminator)
            if binance_open and binance_current:
                align_gap = binance_current - binance_open
                if side == "UP" and binance_current < binance_open:
                    align_bp = abs(align_gap) / binance_open * 10000
                    self.stats["skipped"] += 1
                    self.attempted_cids.add(cid)
                    print(f"[SKIP] Binance NOT aligned | Bet UP but {asset} ${binance_current:.0f} "
                          f"< 5M open ${binance_open:.0f} ({align_bp:.1f}bp wrong side) "
                          f"| {time_left_sec:.0f}s | {question[:40]}")
                    continue
                elif side == "DOWN" and binance_current > binance_open:
                    align_bp = abs(align_gap) / binance_open * 10000
                    self.stats["skipped"] += 1
                    self.attempted_cids.add(cid)
                    print(f"[SKIP] Binance NOT aligned | Bet DOWN but {asset} ${binance_current:.0f} "
                          f"> 5M open ${binance_open:.0f} ({align_bp:.1f}bp wrong side) "
                          f"| {time_left_sec:.0f}s | {question[:40]}")
                    continue

            # STEP 3: Get depth for our side
            sweep_limit = min(MAX_ENTRY_PRICE + FOK_SLIPPAGE, 0.99)
            target_tid = up_tid if side == "UP" else down_tid
            if target_tid:
                best_ask, avail_shares, levels = await self.get_orderbook_depth(target_tid, sweep_limit)
            else:
                best_ask, avail_shares, levels = None, 0, []

            entry_price = best_ask if best_ask is not None else (0.75 if time_left_sec > 30 else 0.85)
            confidence = entry_price

            depth_str = " | ".join(f"${p:.2f}x{int(s)}" for p, s in levels[:4])
            src_tag = "BINANCE" if signal_source == "binance" else "CLOB"
            total_depth = int(avail_shares)
            print(f"[{src_tag}] {asset} {side} | CLOB: UP=${up_display:.2f} DN=${down_display:.2f} "
                  f"| entry=${entry_price:.2f} | depth={total_depth}sh [{depth_str}] | {time_left_sec:.0f}s left")

            # V3.2: High-depth filter — depth >= 500 = 25% WR (informed traders positioning against you)
            if total_depth >= MAX_DEPTH_SHARES:
                self.stats["skipped"] += 1
                self.attempted_cids.add(cid)
                print(f"[SKIP] High depth ({total_depth}sh >= {MAX_DEPTH_SHARES}) "
                      f"| {side} @ ${entry_price:.2f} | 25% WR zone | {question[:50]}")
                continue

            # Cap entry price — ML-tuned (default $0.78)
            ml_max_entry = self.tuner.get_active_value("max_entry")
            if entry_price > ml_max_entry:
                # Shadow-track expensive skips for ML learning
                if end_dt:
                    _bmove2 = ((binance_current - binance_open) / binance_open * 100
                               if binance_open and binance_current else 0)
                    self.tuner.record_shadow(
                        market_id=cid, end_time_iso=end_dt.isoformat(),
                        side=side, entry_price=entry_price,
                        clob_dominant=max(up_display, down_display),
                        extra={"binance_move": round(_bmove2, 4),
                               "time_left_sec": round(time_left_sec, 1)})
                self.stats["skipped"] += 1
                self.attempted_cids.add(cid)
                print(f"[SKIP] Too expensive | {side} @ ${entry_price:.2f} > ${ml_max_entry:.2f} "
                      f"| {time_left_sec:.0f}s left | {question[:50]}")
                continue

            # V2.0 tuner: ML-tuned min_confidence (fallback to hardcoded MIN_CONFIDENCE)
            ml_min_conf = self.tuner.get_active_value("min_confidence")
            effective_min_conf = max(MIN_CONFIDENCE, ml_min_conf)
            if entry_price < effective_min_conf:
                self.stats["skipped"] += 1
                self.attempted_cids.add(cid)
                print(f"[SKIP] Too cheap (low liquidity) | {side} @ ${entry_price:.2f} "
                      f"< ${effective_min_conf:.2f} | {time_left_sec:.0f}s left | {question[:50]}")
                continue

            # V2.1: UP trades need higher minimum entry ($0.72+)
            # Data: UP below $0.72 = losses, UP at $0.72+ with high ML = wins
            if side == "UP" and entry_price < MIN_ENTRY_UP:
                self.stats["skipped"] += 1
                self.attempted_cids.add(cid)
                print(f"[SKIP] UP too cheap | ${entry_price:.2f} < ${MIN_ENTRY_UP} min for UP "
                      f"| {time_left_sec:.0f}s left | {question[:50]}")
                continue

            # V3.0: Kelly sizing — auto-scales based on empirical edge
            trade_size = get_trade_size(self.stats["wins"], self.stats["losses"],
                                        entry_price, self.stats["pnl"])
            if trade_size <= 0:
                self.stats["skipped"] += 1
                self.attempted_cids.add(cid)
                print(f"[KELLY-SKIP] No edge at {self.stats['wins']}W/{self.stats['losses']}L | {question[:50]}")
                continue

            # V3.0: SynthData consensus — boost/reduce based on external oracle
            synth = await query_synthdata(asset)
            synth_tag = ""
            if synth:
                s_edge = synth["edge_up"] if side == "UP" else synth["edge_down"]
                if s_edge >= SYNTH_EDGE_THRESHOLD:
                    trade_size = round(trade_size * SYNTH_SIZE_BOOST, 2)
                    synth_tag = f" SYNTH+{s_edge:.0%}"
                    print(f"  [SYNTH] Agrees: edge={s_edge:+.1%} → 1.5x size: ${trade_size:.2f}")
                elif s_edge <= -SYNTH_EDGE_THRESHOLD:
                    trade_size = round(trade_size * SYNTH_SIZE_REDUCE, 2)
                    synth_tag = f" SYNTH-{s_edge:.0%}"
                    print(f"  [SYNTH] Disagrees: edge={s_edge:+.1%} → 0.5x size: ${trade_size:.2f}")

            # V2.0 tuner: dynamic size multiplier based on conditional features
            size_mult = self.tuner.get_optimal_size_mult(
                hour=end_dt.hour if end_dt else None,
                direction=side,
                entry_price=entry_price,
            )
            if size_mult != 1.0:
                trade_size = round(trade_size * size_mult, 2)
                print(f"  [ML-SIZE] {size_mult:.2f}x -> ${trade_size:.2f}")

            # V3.1: Graduated circuit breaker — half-size when approaching halt
            if (self._consecutive_losses >= CIRCUIT_BREAKER_HALFSIZE or
                    self._recovery_trades < CIRCUIT_BREAKER_RECOVERY):
                trade_size = round(trade_size * 0.5, 2)
                reason = (f"recovery {self._recovery_trades}/{CIRCUIT_BREAKER_RECOVERY}"
                          if self._consecutive_losses < CIRCUIT_BREAKER_HALFSIZE
                          else f"{self._consecutive_losses} consecutive losses")
                print(f"  [HALFSIZE] ${trade_size:.2f} ({reason})")

            # V2.3: ETH minimum size until proven profitable
            if asset == "ETH":
                trade_size = round(trade_size * ETH_SIZE_MULT, 2)
                if trade_size < 1.0:
                    trade_size = 1.0
                print(f"  [ETH] Min size: ${trade_size:.2f} (unproven asset)")

            # V2.1: DOWN-bias sizing — UP gets half size (40% WR vs DOWN's 100%)
            if side == "UP":
                trade_size = round(trade_size * UP_SIZE_MULT, 2)
                print(f"  [UP-BIAS] Reduced size: ${trade_size:.2f} (50% for UP trades)")

            desired_shares = math.floor(trade_size / entry_price)
            if desired_shares < 1:
                desired_shares = 1

            # Cap shares at available depth (core fix for thin markets)
            avail_int = int(avail_shares)
            if avail_int < MIN_SHARES:
                self.stats["skipped"] += 1
                self.attempted_cids.add(cid)
                print(f"[SKIP] Too thin | only {avail_int}sh available (need {MIN_SHARES}) | {question[:50]}")
                continue

            shares = min(desired_shares, avail_int)
            actual_trade_size = round(shares * entry_price, 2)

            if shares < desired_shares:
                print(f"[DEPTH] Reduced order: {desired_shares} -> {shares}sh "
                      f"(${actual_trade_size:.2f} of ${trade_size:.2f})")

            # ML depth filter — compute features and check
            ml_features = DepthMLFilter.compute_features(
                levels, entry_price, confidence, signal_source,
                up_display, down_display)
            ml_ok, ml_reason, ml_wr = self.ml_depth.should_trade(ml_features, side=side)
            if not ml_ok:
                self.stats["skipped"] += 1
                self.attempted_cids.add(cid)
                print(f"[SKIP] {ml_reason} | {side} @ ${entry_price:.2f} "
                      f"| bar={ml_features['best_ask_ratio']:.0%} largest@best={ml_features['best_is_largest']} "
                      f"| {question[:40]}")
                continue

            # Mark as attempted
            self.attempted_cids.add(cid)

            # Place order (live or paper)
            fill = await self.place_live_order(market, side, entry_price, shares, actual_trade_size)
            if fill is None:
                print(f"[SKIP] Order not filled | {asset} {side} @ ${entry_price:.2f} | {question[:50]}")
                continue

            # Use actual fill data
            shares = fill["shares"]
            entry_price = fill["price"]
            cost = fill["cost"]

            now = datetime.now(timezone.utc)
            trade = {
                "condition_id": cid,
                "question": question,
                "side": side,
                "entry_price": round(entry_price, 4),
                "confidence": round(confidence, 4),
                "up_ask": round(up_ask, 4) if up_ask else 0.0,
                "down_ask": round(down_ask, 4) if down_ask else 0.0,
                "shares": round(shares, 4),
                "cost": cost,
                "trade_size_tier": trade_size,
                "time_remaining_sec": round(time_left_sec, 1),
                "entry_time": now.isoformat(),
                "end_dt": market.get("_end_dt", ""),
                "status": "open",
                "pnl": 0.0,
                "result": None,
                "order_id": fill.get("order_id", ""),
                "live": not self.paper,
                "asset": asset,
                "strategy": f"directional_{signal_source}",
                "_ml_features": ml_features,
                "_ml_reason": ml_reason,
                "_binance_open": round(binance_open, 2) if binance_open else None,
                "_binance_current": round(binance_current, 2) if binance_current else None,
                "_align_bp": round(((binance_current - binance_open) / binance_open * 10000) * (1 if side == "UP" else -1), 1) if binance_open and binance_current else None,
            }

            self.active[cid] = trade
            # Shadow-track actual trade for ML tuner
            if end_dt:
                _bmove3 = ((binance_current - binance_open) / binance_open * 100
                           if binance_open and binance_current else 0)
                self.tuner.record_shadow(
                    market_id=cid, end_time_iso=end_dt.isoformat(),
                    side=side, entry_price=entry_price,
                    clob_dominant=max(up_display, down_display),
                    extra={"binance_move": round(_bmove3, 4),
                           "time_left_sec": round(time_left_sec, 1)})
            self._save()

            mode = "LIVE" if not self.paper else "PAPER"
            align_tag = ""
            if binance_open and binance_current:
                a_bp = ((binance_current - binance_open) / binance_open * 10000) * (1 if side == "UP" else -1)
                align_tag = f" | align={a_bp:+.1f}bp"
            print(f"\n[{mode}] {asset} {side} @ ${entry_price:.2f} (conf={confidence:.2f}) | "
                  f"${cost:.2f} ({shares:.1f}sh) | "
                  f"{time_left_sec:.0f}s left | tier=${trade_size:.0f} | "
                  f"UP=${up_ask:.2f} DN=${down_ask:.2f}{align_tag} | "
                  f"{question[:50]}")

    async def check_oracle_entry(self, markets: list):
        """Oracle front-run: enter AFTER market closes when Binance direction is KNOWN."""
        oracle_active = sum(1 for t in self.active.values() if t.get("strategy") == "oracle")
        if oracle_active >= 1:
            return
        if self.session_pnl <= -DAILY_LOSS_LIMIT:
            return
        # V3.0: Circuit breaker
        if time.time() < self._circuit_halt_until:
            return

        for market in markets:
            oracle_active = sum(1 for t in self.active.values() if t.get("strategy") == "oracle")
            if oracle_active >= 1:
                break

            cid = market.get("conditionId", "")
            if not cid or cid in self.oracle_attempted_cids:
                continue

            time_left_sec = market.get("_time_left_sec", 9999)

            # Oracle window: ORACLE_MIN_DELAY to ORACLE_WINDOW seconds AFTER close
            if time_left_sec > -ORACLE_MIN_DELAY or time_left_sec < -ORACLE_WINDOW:
                continue

            question = market.get("question", "?")
            end_dt = market.get("_end_dt_parsed")
            if not end_dt:
                continue

            # V2.3: Asset-aware Binance symbol for oracle
            asset = market.get("_asset", "BTC")
            binance_sym = BINANCE_SYMBOLS.get(asset, "BTCUSDT")

            # Get COMPLETED Binance 5-min candle — direction is KNOWN
            direction = await self.resolve_via_binance(end_dt, symbol=binance_sym)
            if not direction:
                # Don't mark attempted — Binance may finalize on next poll
                secs_past = abs(time_left_sec)
                print(f"[ORACLE] No Binance data | {secs_past:.0f}s past close | {question[:50]}")
                continue

            # Get CLOB depth for the winning side
            up_tid, down_tid = self.get_token_ids(market)
            token_id = up_tid if direction == "UP" else down_tid
            if not token_id:
                continue

            # Also check the FULL book (not just ≤0.95) for ML data collection
            best_ask, avail_shares, levels = await self.get_orderbook_depth(token_id, 0.99)
            secs_past = abs(time_left_sec)

            # Log oracle book state for ML analysis (every check, hit or miss)
            self._log_oracle_book(question, direction, secs_past, best_ask, avail_shares, levels)

            # Filter to asks within our price limit
            levels_ok = [(p, s) for p, s in levels if p <= ORACLE_MAX_ENTRY]
            avail_ok = sum(s for _, s in levels_ok)
            best_ok = min((p for p, _ in levels_ok), default=None)

            if best_ok is None or avail_ok < MIN_SHARES:
                # DON'T mark as attempted — keep re-checking every poll cycle
                # Liquidity may appear at 10-20s when sellers dump winning tokens
                ask_str = f"${best_ask:.2f}" if best_ask else "none"
                full_depth = f" (full book: {ask_str}, {int(avail_shares)}sh)" if best_ask else ""
                print(f"[ORACLE] {direction} no liquidity <= ${ORACLE_MAX_ENTRY}{full_depth} | "
                      f"{secs_past:.0f}s past | {question[:50]}")
                continue

            # Use the filtered asks
            best_ask = best_ok
            avail_shares = avail_ok
            levels = levels_ok

            trade_size = get_trade_size(self.stats["wins"], self.stats["losses"])
            desired_shares = math.floor(trade_size / best_ask)
            shares = min(max(desired_shares, 1), int(avail_shares))
            actual_cost = round(shares * best_ask, 2)
            profit_per_share = round(1.00 - best_ask, 2)
            expected_profit = round(profit_per_share * shares, 2)

            self.oracle_attempted_cids.add(cid)

            depth_str = " | ".join(f"${p:.2f}x{int(s)}" for p, s in levels[:4])
            secs_past = abs(time_left_sec)
            print(f"[ORACLE] {asset} {direction} KNOWN | ask=${best_ask:.2f} | "
                  f"depth={int(avail_shares)}sh [{depth_str}] | "
                  f"profit=${profit_per_share:.2f}/sh (${expected_profit:.2f} total) | "
                  f"{secs_past:.0f}s past close")

            fill = await self.place_oracle_order(market, direction, best_ask, shares, actual_cost)
            if fill is None:
                print(f"[ORACLE] No fill | {asset} {direction} @ ${best_ask:.2f} | {question[:50]}")
                continue

            shares = fill["shares"]
            entry_price = fill["price"]
            cost = fill["cost"]

            now = datetime.now(timezone.utc)
            trade = {
                "condition_id": cid,
                "question": question,
                "side": direction,
                "entry_price": round(entry_price, 4),
                "shares": round(shares, 4),
                "cost": cost,
                "trade_size_tier": trade_size,
                "time_remaining_sec": round(time_left_sec, 1),
                "entry_time": now.isoformat(),
                "end_dt": market.get("_end_dt", ""),
                "status": "open",
                "pnl": 0.0,
                "result": None,
                "order_id": fill.get("order_id", ""),
                "asset": asset,
                "live": not self.paper,
                "strategy": "oracle",
            }

            self.active[f"oracle_{cid}"] = trade
            self._save()

            mode = "LIVE" if not self.paper else "PAPER"
            print(f"\n[{mode}|ORACLE] {asset} {direction} @ ${entry_price:.2f} | "
                  f"${cost:.2f} ({shares:.1f}sh) | "
                  f"DIRECTION KNOWN | min profit=${profit_per_share:.2f}/sh | "
                  f"{question[:50]}")

    # ========================================================================
    # RESOLUTION
    # ========================================================================

    async def resolve_trades(self):
        now = datetime.now(timezone.utc)
        to_remove = []

        for cid, trade in list(self.active.items()):
            if trade.get("status") != "open":
                continue

            end_dt_str = trade.get("end_dt", "")
            if not end_dt_str:
                continue

            try:
                end_dt = datetime.fromisoformat(end_dt_str)
            except Exception:
                continue

            seconds_past_close = (now - end_dt).total_seconds()
            if seconds_past_close < RESOLVE_DELAY:
                continue

            trade_asset = trade.get("asset", "BTC")
            resolve_sym = BINANCE_SYMBOLS.get(trade_asset, "BTCUSDT")
            outcome = await self.resolve_via_binance(end_dt, symbol=resolve_sym)
            if outcome is None:
                if seconds_past_close > 120:
                    trade["status"] = "closed"
                    trade["result"] = "unknown"
                    trade["pnl"] = round(-trade["cost"], 2)
                    trade["resolve_time"] = now.isoformat()
                    self.stats["losses"] += 1
                    self.stats["pnl"] += trade["pnl"]
                    self.session_pnl += trade["pnl"]
                    self.resolved.append(trade)
                    to_remove.append(cid)
                    print(f"[LOSS] {trade_asset}:{trade['side']} ${trade['pnl']:+.2f} | "
                          f"No Binance data | {trade['question'][:45]}")
                continue

            won = (trade["side"] == outcome)
            if won:
                pnl = round(trade["shares"] * 1.0 - trade["cost"], 2)
                trade["result"] = "WIN"
            else:
                pnl = round(-trade["cost"], 2)
                trade["result"] = "LOSS"

            trade["status"] = "closed"
            trade["pnl"] = pnl
            trade["market_outcome"] = outcome
            trade["resolve_time"] = now.isoformat()

            # Signal timing audit — what would signal have been at different entry times?
            try:
                audit = await self._audit_signal_timing(end_dt, symbol=resolve_sym)
                trade["signal_audit"] = audit
                # Print audit summary
                audit_parts = []
                for w_sec in [55, 57, 60, 63, 65]:
                    sig = audit.get(f"signal_{w_sec}s", "?")
                    pct = audit.get(f"pct_{w_sec}s", 0)
                    marker = "*" if sig == outcome else "X"
                    audit_parts.append(f"{w_sec}s={sig}({pct:+.3f}%){marker}")
                actual = audit.get("actual_direction", "?")
                actual_pct = audit.get("actual_pct", 0)
                print(f"[AUDIT] {' | '.join(audit_parts)}")
                print(f"[AUDIT] Actual={actual}({actual_pct:+.3f}%) | "
                      f"Entry was at {trade.get('time_remaining_sec',0):.0f}s | "
                      f"{'CORRECT' if trade['side'] == actual else 'WRONG'} signal")
            except Exception as e:
                print(f"[AUDIT] Error: {e}")

            self.stats["wins" if won else "losses"] += 1
            self.stats["pnl"] += pnl
            self.session_pnl += pnl

            # V3.1: Graduated circuit breaker tracking
            if won:
                if self._consecutive_losses >= CIRCUIT_BREAKER_HALFSIZE:
                    self._recovery_trades = 1  # start recovery at half-size
                    print(f"[RECOVERY] Win after {self._consecutive_losses} losses → "
                          f"half-size for {CIRCUIT_BREAKER_RECOVERY} trades")
                else:
                    self._recovery_trades = CIRCUIT_BREAKER_RECOVERY  # already recovered
                self._consecutive_losses = 0
            else:
                self._consecutive_losses += 1
                self._recovery_trades = 0
                if self._consecutive_losses >= CIRCUIT_BREAKER_LOSSES:
                    self._circuit_halt_until = time.time() + CIRCUIT_BREAKER_COOLDOWN
                    print(f"[CIRCUIT BREAKER] {self._consecutive_losses} consecutive losses → "
                          f"HALTED for {CIRCUIT_BREAKER_COOLDOWN // 60} min")
                elif self._consecutive_losses >= CIRCUIT_BREAKER_HALFSIZE:
                    print(f"[HALFSIZE] {self._consecutive_losses} consecutive losses → half-size mode")

            # ML depth learning — record outcome for auto-tuning
            ml_feats = trade.get("_ml_features")
            if ml_feats:
                self.ml_depth.record_outcome(ml_feats, won)

            # V2.0 tuner: feed resolved trade features for conditional sizing model
            resolve_hour = now.hour
            self.tuner.state.setdefault("resolved_log", []).append({
                "ts": now.isoformat(),
                "pnl": round(pnl, 4),
                "hour": resolve_hour,
                "won": won,
                "entry_price": trade.get("entry_price", 0),
                "side": trade.get("side", ""),
                "momentum": 0,  # sniper doesn't use momentum
                "rsi": 0,       # sniper doesn't use RSI
            })
            # Bound resolved_log
            if len(self.tuner.state["resolved_log"]) > 500:
                self.tuner.state["resolved_log"] = self.tuner.state["resolved_log"][-400:]
            # Update hour_stats
            h_str = str(resolve_hour)
            hs = self.tuner.state.setdefault("hour_stats", {}).setdefault(
                h_str, {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0})
            hs["trades"] += 1
            hs["pnl"] = round(hs["pnl"] + pnl, 4)
            if won:
                hs["wins"] += 1
            else:
                hs["losses"] += 1
            if self.tuner.state.get("first_resolve_ts") is None:
                self.tuner.state["first_resolve_ts"] = now.isoformat()
            self.tuner._save()

            self.resolved.append(trade)
            to_remove.append(cid)

            w, l = self.stats["wins"], self.stats["losses"]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            tag = "WIN" if won else "LOSS"
            tier = trade.get("trade_size_tier", BASE_TRADE_SIZE)
            strat_label = trade.get("strategy", "directional").upper()

            ml_tag = ""
            if ml_feats:
                ml_tag = (f" | bar={ml_feats['best_ask_ratio']:.0%}"
                          f" top={'Y' if ml_feats['best_is_largest'] else 'N'}")

            print(f"[{tag}|{strat_label}] {trade_asset}:{trade['side']} ${pnl:+.2f} | "
                  f"entry=${trade['entry_price']:.2f} tier=${tier:.0f} | "
                  f"market={outcome} | "
                  f"{w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f} | "
                  f"session=${self.session_pnl:+.2f}{ml_tag} | "
                  f"{trade['question'][:45]}")

        for cid in to_remove:
            del self.active[cid]
        if to_remove:
            self._save()

        # ML TUNER: resolve shadow entries (learn from skipped markets too)
        await self._resolve_shadow_entries()

    async def _resolve_shadow_entries(self):
        """Resolve shadow-tracked markets via Binance to learn optimal parameters."""
        now = datetime.now(timezone.utc)
        shadows = self.tuner.state.get("shadow", [])
        if not shadows:
            return

        resolutions = {}
        for entry in shadows:
            end_str = entry.get("end_time", "")
            if not end_str:
                continue
            try:
                end_dt = datetime.fromisoformat(end_str)
            except Exception:
                continue
            if (now - end_dt).total_seconds() < RESOLVE_DELAY + 5:
                continue  # not yet resolvable
            if (now - end_dt).total_seconds() > 600:
                # Too old, mark as stale (will be dropped)
                resolutions[entry["market_id"]] = None
                continue

            mid = entry["market_id"]
            if mid not in resolutions:
                outcome = await self.resolve_via_binance(end_dt)
                if outcome:
                    resolutions[mid] = outcome

        # Filter out None (stale entries just get removed from shadow list)
        valid = {k: v for k, v in resolutions.items() if v is not None}
        if valid:
            n = self.tuner.resolve_shadows(valid)
            if n > 0:
                total = self.tuner.state.get("total_resolved", 0)
                if total % 10 == 0:
                    print(self.tuner.get_report())

        # Clean stale shadows
        stale = {k for k, v in resolutions.items() if v is None}
        if stale:
            self.tuner.state["shadow"] = [
                s for s in self.tuner.state["shadow"]
                if s["market_id"] not in stale
            ]
            self.tuner._save()

    # ========================================================================
    # LOGGING
    # ========================================================================

    def print_status(self, markets: list):
        now = time.time()
        if now - self._last_log_time < LOG_INTERVAL:
            return
        self._last_log_time = now

        w, l = self.stats["wins"], self.stats["losses"]
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        tier = get_trade_size(w, l)

        now_dt = datetime.now(timezone.utc)
        now_mst = now_dt + timedelta(hours=-7)

        closest = None
        closest_time = 9999
        for m in markets:
            tl = m.get("_time_left_sec", 9999)
            if 0 < tl < closest_time:
                closest_time = tl
                closest = m

        market_str = "No active 5M markets"
        sniper_str = "NO"
        if closest:
            q = closest.get("question", "?")[:50]
            tl = closest.get("_time_left_sec", 0)
            sniper_str = "YES" if tl <= SNIPER_WINDOW else "NO"
            market_str = f"{q} | {tl:.0f}s left"

        mode = "LIVE" if not self.paper else "PAPER"
        print(f"\n--- {now_mst.strftime('%H:%M:%S MST')} | "
              f"SNIPER 5M {mode} | "
              f"Active: {len(self.active)} | Resolved: {len(self.resolved)} | "
              f"{w}W/{l}L {wr:.0f}%WR | "
              f"PnL: ${self.stats['pnl']:+.2f} | "
              f"Session: ${self.session_pnl:+.2f} | "
              f"Tier: ${tier:.0f} ---")
        print(f"    Market: {market_str}")
        print(f"    Sniper window: {sniper_str} (< {SNIPER_WINDOW}s)")

        if self.session_pnl <= -DAILY_LOSS_LIMIT:
            print(f"    *** DAILY LOSS LIMIT HIT: ${self.session_pnl:+.2f} ***")

        # Print ML report every 5th status
        if total > 0 and hasattr(self, '_status_count'):
            self._status_count += 1
        else:
            self._status_count = 1
        if self._status_count % 5 == 0:
            print(self.ml_depth.get_report())

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        mode = "LIVE" if not self.paper else "PAPER"
        tier = get_trade_size(self.stats["wins"], self.stats["losses"])
        print("=" * 70)
        print(f"  SNIPER 5M {mode} TRADER V2.3 — BTC + ETH")
        print("=" * 70)
        print(f"  V2.3: Added ETH markets | ETH size={ETH_SIZE_MULT:.0%} (~$1/trade) until proven")
        print(f"  V3.1: Skip UTC hours {SKIP_HOURS_UTC} | Binance threshold 0.04% | CLOB floor $0.65")
        print(f"  V2.1 DOWN-bias (16T: DOWN 11W/0L 100%, UP 2W/3L 40%):")
        print(f"    - ML WR threshold: UP >= {ML_UP_WR_THRESHOLD:.0%} | DOWN >= {ML_MIN_WR_THRESHOLD:.0%}")
        print(f"    - MIN entry: UP >= ${MIN_ENTRY_UP:.2f} | DOWN >= ${MIN_CONFIDENCE:.2f}")
        print(f"    - Size: UP = {UP_SIZE_MULT:.0%} (${tier*UP_SIZE_MULT:.2f}) | DOWN = full (${tier:.2f})")
        print(f"  Strategy 1a — DIRECTIONAL/BINANCE")
        print(f"    - Enter at {SNIPER_WINDOW}s before close | Binance signal > 0.07%")
        print(f"    - Entry: ${MIN_CONFIDENCE:.2f}-${MAX_ENTRY_PRICE:.2f} | FOK orders")
        print(f"  Strategy 1b — DIRECTIONAL/CLOB")
        print(f"    - Binance flat fallback | CLOB dominant side $0.70-$0.78")
        print(f"  Scan: every {SCAN_INTERVAL}s | Daily loss limit: ${DAILY_LOSS_LIMIT:.2f}")
        print(f"  Results: {RESULTS_FILE.name}")
        print("=" * 70)

        if self.resolved:
            w, l = self.stats["wins"], self.stats["losses"]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            print(f"[RESUME] {w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
                  f"{len(self.resolved)} resolved, {len(self.active)} active")

        cycle = 0
        while self.running:
            try:
                cycle += 1

                if self.session_pnl <= -DAILY_LOSS_LIMIT:
                    self.print_status([])
                    await asyncio.sleep(60)
                    continue

                markets = await self.discover_5m_markets()
                await self.resolve_trades()
                await self.check_sniper_entry(markets)
                # V1.1: Oracle DISABLED — phantom fills caused -$1.36 live loss
                # await self.check_oracle_entry(markets)
                self.print_status(markets)

                if cycle % 12 == 0:
                    self._save()

                # Fast-poll during sniper window + oracle window
                # Book reprices $0.60→$0.90 in seconds — need 2s polls to catch $0.75
                fast = any(
                    -ORACLE_WINDOW <= m.get("_time_left_sec", 9999) <= SNIPER_WINDOW
                    for m in markets
                )
                await asyncio.sleep(2 if fast else SCAN_INTERVAL)

            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Saving state...")
                self._save()
                break
            except Exception as e:
                print(f"[ERROR] Cycle {cycle}: {e}")
                traceback.print_exc()
                await asyncio.sleep(10)

        self._save()
        w, l = self.stats["wins"], self.stats["losses"]
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        print(f"\n[FINAL] {w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
              f"Session: ${self.session_pnl:+.2f}")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    paper = "--live" not in sys.argv
    if paper:
        print("[MODE] PAPER — pass --live for real trading")
    else:
        print("[MODE] *** LIVE TRADING — REAL MONEY ***")

    lock = acquire_pid_lock("sniper_5m_live")
    try:
        trader = Sniper5MLive(paper=paper)
        asyncio.run(trader.run())
    finally:
        release_pid_lock("sniper_5m_live")
