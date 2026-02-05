"""
TA Paper Trading Runner with 10-minute Updates + ML Auto-Tuner

Standalone script that runs TA-based paper trading for BTC 15m markets.
Includes ML-based automatic parameter tuning based on trade performance.
"""

import asyncio
import json
import time
import statistics
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from functools import partial
from collections import defaultdict

import httpx

# Force unbuffered output
print = partial(print, flush=True)

# Import from arbitrage package
import sys
sys.path.insert(0, str(Path(__file__).parent))

from arbitrage.ta_signals import TASignalGenerator, Candle, SignalStrength
from arbitrage.bregman_optimizer import BregmanOptimizer, enhance_ta_signal_with_bregman
from arbitrage.ml_engine_v3 import get_ml_engine, MLFeatures, AdvancedMLEngine


@dataclass
class MLTunerState:
    """State for ML auto-tuner."""
    # Current optimized parameters
    up_max_price: float = 0.45
    up_min_confidence: float = 0.60
    down_max_price: float = 0.40
    down_min_confidence: float = 0.72
    min_edge: float = 0.12
    down_momentum_threshold: float = -0.002

    # Performance tracking
    tune_count: int = 0
    last_tune_time: str = ""

    # Historical parameter performance
    param_history: List[Dict] = field(default_factory=list)


class MLAutoTuner:
    """
    Machine Learning Auto-Tuner for paper trading parameters.

    Analyzes completed trades to find optimal thresholds and automatically
    adjusts parameters to maximize profit while maintaining acceptable win rate.
    """

    STATE_FILE = Path(__file__).parent / "ml_tuner_state.json"
    BACKUP_DATA = Path(__file__).parent / "ta_paper_results_v1_backup.json"
    MIN_TRADES_FOR_TUNING = 10  # Need at least 10 trades before tuning
    TARGET_WIN_RATE = 0.55  # Target 55% win rate minimum

    def __init__(self):
        self.state = MLTunerState()
        self._load_state()

    def _load_state(self):
        """Load tuner state from file."""
        if self.STATE_FILE.exists():
            try:
                data = json.load(open(self.STATE_FILE))
                self.state = MLTunerState(**{k: v for k, v in data.items()
                                            if k in MLTunerState.__dataclass_fields__})
                print(f"[ML] Loaded tuner state (tune #{self.state.tune_count})")
            except Exception as e:
                print(f"[ML] Error loading state: {e}")

    def _save_state(self):
        """Save tuner state."""
        data = asdict(self.state)
        with open(self.STATE_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_historical_trades(self) -> List[Dict]:
        """Load historical trades for analysis."""
        trades = []

        # Load from backup (V1 data with 153 trades)
        if self.BACKUP_DATA.exists():
            try:
                data = json.load(open(self.BACKUP_DATA))
                for t in data.get("trades", {}).values():
                    if t.get("status") == "closed":
                        trades.append(t)
            except Exception as e:
                print(f"[ML] Error loading backup: {e}")

        return trades

    def analyze_trades(self, trades: List[Dict]) -> Dict:
        """
        Analyze trades to find winning patterns.
        Returns analysis dict with win rates by various dimensions.
        """
        if not trades:
            return {}

        analysis = {
            "total": len(trades),
            "by_side": defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0}),
            "by_price_bucket": defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0}),
            "by_edge_bucket": defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0}),
            "by_confidence_bucket": defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0}),
            "by_kl_bucket": defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0}),
        }

        for t in trades:
            side = t.get("side", "")
            pnl = t.get("pnl", 0)
            entry_price = t.get("entry_price", 0.5)
            edge = t.get("edge_at_entry", 0)
            kl = t.get("kl_divergence", 0)

            won = pnl > 0

            # By side
            if won:
                analysis["by_side"][side]["wins"] += 1
            else:
                analysis["by_side"][side]["losses"] += 1
            analysis["by_side"][side]["pnl"] += pnl

            # By price bucket (0.20-0.30, 0.30-0.40, 0.40-0.50, 0.50+)
            if entry_price < 0.30:
                bucket = "0.20-0.30"
            elif entry_price < 0.40:
                bucket = "0.30-0.40"
            elif entry_price < 0.50:
                bucket = "0.40-0.50"
            else:
                bucket = "0.50+"

            if won:
                analysis["by_price_bucket"][bucket]["wins"] += 1
            else:
                analysis["by_price_bucket"][bucket]["losses"] += 1
            analysis["by_price_bucket"][bucket]["pnl"] += pnl

            # By edge bucket
            if edge < 0.15:
                edge_bucket = "10-15%"
            elif edge < 0.20:
                edge_bucket = "15-20%"
            elif edge < 0.25:
                edge_bucket = "20-25%"
            else:
                edge_bucket = "25%+"

            if won:
                analysis["by_edge_bucket"][edge_bucket]["wins"] += 1
            else:
                analysis["by_edge_bucket"][edge_bucket]["losses"] += 1
            analysis["by_edge_bucket"][edge_bucket]["pnl"] += pnl

            # By KL divergence bucket
            if kl < 0.10:
                kl_bucket = "KL<0.10"
            elif kl < 0.15:
                kl_bucket = "KL_0.10-0.15"
            elif kl < 0.20:
                kl_bucket = "KL_0.15-0.20"
            else:
                kl_bucket = "KL>0.20"

            if won:
                analysis["by_kl_bucket"][kl_bucket]["wins"] += 1
            else:
                analysis["by_kl_bucket"][kl_bucket]["losses"] += 1
            analysis["by_kl_bucket"][kl_bucket]["pnl"] += pnl

        return analysis

    def calculate_optimal_params(self, analysis: Dict) -> Dict:
        """
        Calculate optimal parameters based on trade analysis.
        Returns dict of recommended parameter changes.
        """
        recommendations = {}

        # Analyze UP vs DOWN performance
        up_data = analysis["by_side"].get("UP", {"wins": 0, "losses": 0, "pnl": 0})
        down_data = analysis["by_side"].get("DOWN", {"wins": 0, "losses": 0, "pnl": 0})

        up_total = up_data["wins"] + up_data["losses"]
        down_total = down_data["wins"] + down_data["losses"]

        up_wr = up_data["wins"] / max(1, up_total)
        down_wr = down_data["wins"] / max(1, down_total)

        recommendations["up_win_rate"] = up_wr
        recommendations["down_win_rate"] = down_wr
        recommendations["up_pnl"] = up_data["pnl"]
        recommendations["down_pnl"] = down_data["pnl"]

        # If DOWN is underperforming, tighten requirements
        if down_wr < 0.50 and down_total >= 5:
            # Increase confidence requirement
            new_conf = min(0.80, self.state.down_min_confidence + 0.03)
            recommendations["down_min_confidence"] = new_conf
            # Lower max price
            new_price = max(0.30, self.state.down_max_price - 0.02)
            recommendations["down_max_price"] = new_price
        elif down_wr > 0.60 and down_total >= 5:
            # DOWN is doing well, can slightly relax
            new_conf = max(0.65, self.state.down_min_confidence - 0.02)
            recommendations["down_min_confidence"] = new_conf

        # If UP is underperforming, tighten requirements
        if up_wr < 0.50 and up_total >= 5:
            new_conf = min(0.75, self.state.up_min_confidence + 0.03)
            recommendations["up_min_confidence"] = new_conf
            new_price = max(0.35, self.state.up_max_price - 0.02)
            recommendations["up_max_price"] = new_price
        elif up_wr > 0.60 and up_total >= 5:
            # UP is doing well, can slightly relax
            new_conf = max(0.55, self.state.up_min_confidence - 0.02)
            recommendations["up_min_confidence"] = new_conf
            new_price = min(0.50, self.state.up_max_price + 0.02)
            recommendations["up_max_price"] = new_price

        # Analyze price buckets to find optimal entry prices
        best_price_bucket = None
        best_price_wr = 0
        for bucket, data in analysis["by_price_bucket"].items():
            total = data["wins"] + data["losses"]
            if total >= 3:
                wr = data["wins"] / total
                if wr > best_price_wr:
                    best_price_wr = wr
                    best_price_bucket = bucket

        if best_price_bucket:
            recommendations["best_price_bucket"] = best_price_bucket
            recommendations["best_price_wr"] = best_price_wr

        # Analyze edge buckets
        best_edge_bucket = None
        best_edge_wr = 0
        for bucket, data in analysis["by_edge_bucket"].items():
            total = data["wins"] + data["losses"]
            if total >= 3:
                wr = data["wins"] / total
                if wr > best_edge_wr:
                    best_edge_wr = wr
                    best_edge_bucket = bucket

        if best_edge_bucket:
            recommendations["best_edge_bucket"] = best_edge_bucket
            recommendations["best_edge_wr"] = best_edge_wr

            # Adjust min_edge based on what's working
            if best_edge_bucket == "25%+" and best_edge_wr > 0.55:
                recommendations["min_edge"] = 0.20  # Raise threshold
            elif best_edge_bucket == "15-20%" and best_edge_wr > 0.55:
                recommendations["min_edge"] = 0.12  # Keep moderate

        return recommendations

    def apply_recommendations(self, recommendations: Dict) -> List[str]:
        """Apply recommended parameter changes. Returns list of changes made."""
        changes = []

        # Apply DOWN confidence change
        if "down_min_confidence" in recommendations:
            old = self.state.down_min_confidence
            new = recommendations["down_min_confidence"]
            if abs(new - old) >= 0.01:
                self.state.down_min_confidence = new
                changes.append(f"DOWN conf: {old:.0%} -> {new:.0%}")

        # Apply DOWN price change
        if "down_max_price" in recommendations:
            old = self.state.down_max_price
            new = recommendations["down_max_price"]
            if abs(new - old) >= 0.01:
                self.state.down_max_price = new
                changes.append(f"DOWN max price: ${old:.2f} -> ${new:.2f}")

        # Apply UP confidence change
        if "up_min_confidence" in recommendations:
            old = self.state.up_min_confidence
            new = recommendations["up_min_confidence"]
            if abs(new - old) >= 0.01:
                self.state.up_min_confidence = new
                changes.append(f"UP conf: {old:.0%} -> {new:.0%}")

        # Apply UP price change
        if "up_max_price" in recommendations:
            old = self.state.up_max_price
            new = recommendations["up_max_price"]
            if abs(new - old) >= 0.01:
                self.state.up_max_price = new
                changes.append(f"UP max price: ${old:.2f} -> ${new:.2f}")

        # Apply edge change
        if "min_edge" in recommendations:
            old = self.state.min_edge
            new = recommendations["min_edge"]
            if abs(new - old) >= 0.01:
                self.state.min_edge = new
                changes.append(f"Min edge: {old:.0%} -> {new:.0%}")

        if changes:
            self.state.tune_count += 1
            self.state.last_tune_time = datetime.now(timezone.utc).isoformat()
            self.state.param_history.append({
                "time": self.state.last_tune_time,
                "changes": changes,
                "recommendations": {k: v for k, v in recommendations.items()
                                   if isinstance(v, (int, float))}
            })
            # Keep only last 20 tune events
            self.state.param_history = self.state.param_history[-20:]
            self._save_state()

        return changes

    def run_tuning_cycle(self, current_trades: List[Dict] = None) -> Tuple[List[str], Dict]:
        """
        Run a full tuning cycle.
        Returns (changes_made, analysis_summary).
        """
        # Load historical + current trades
        trades = self._load_historical_trades()
        if current_trades:
            trades.extend(current_trades)

        if len(trades) < self.MIN_TRADES_FOR_TUNING:
            return [], {"error": f"Need {self.MIN_TRADES_FOR_TUNING} trades, have {len(trades)}"}

        # Analyze
        analysis = self.analyze_trades(trades)

        # Calculate recommendations
        recommendations = self.calculate_optimal_params(analysis)

        # Apply changes
        changes = self.apply_recommendations(recommendations)

        # Build summary
        summary = {
            "total_trades_analyzed": len(trades),
            "up_wr": recommendations.get("up_win_rate", 0),
            "down_wr": recommendations.get("down_win_rate", 0),
            "up_pnl": recommendations.get("up_pnl", 0),
            "down_pnl": recommendations.get("down_pnl", 0),
            "best_price_bucket": recommendations.get("best_price_bucket"),
            "best_price_wr": recommendations.get("best_price_wr", 0),
            "tune_count": self.state.tune_count,
        }

        return changes, summary

    def get_current_params(self) -> Dict:
        """Get current tuned parameters."""
        return {
            "up_max_price": self.state.up_max_price,
            "up_min_confidence": self.state.up_min_confidence,
            "down_max_price": self.state.down_max_price,
            "down_min_confidence": self.state.down_min_confidence,
            "min_edge": self.state.min_edge,
            "down_momentum_threshold": self.state.down_momentum_threshold,
        }


@dataclass
class TAPaperTrade:
    """A paper trade record."""
    trade_id: str
    market_title: str
    side: str
    entry_price: float
    entry_time: str
    size_usd: float = 100.0
    signal_strength: str = ""
    edge_at_entry: float = 0.0
    # Bregman optimization metrics
    kl_divergence: float = 0.0
    kelly_fraction: float = 0.0
    guaranteed_profit: float = 0.0
    # Market numeric ID for Gamma API resolution
    market_numeric_id: Optional[int] = None
    # Exit
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: float = 0.0
    status: str = "open"


class TAPaperTrader:
    """Paper trades based on TA signals."""

    OUTPUT_FILE = Path(__file__).parent / "ta_paper_results.json"
    POSITION_SIZE = 100.0

    # === WINNING FORMULA V2 ===
    # Lessons learned from live trading losses:
    # 1. DOWN bets at 44-48¢ were losing - market was going UP
    # 2. Cheap UP entries (28-34¢) were profitable value plays
    # 3. Multiple orders per market = disaster (DCA gone wrong)
    # 4. Need momentum confirmation before betting direction

    # Asymmetric requirements - UP is safer than DOWN in bull market
    UP_MAX_PRICE = 0.45           # Only buy UP at value prices (<45¢)
    UP_MIN_CONFIDENCE = 0.60      # Lower bar for UP (trend follower)
    DOWN_MAX_PRICE = 0.40         # DOWN needs even cheaper price (<40¢)
    DOWN_MIN_CONFIDENCE = 0.72    # Higher bar for DOWN (counter-trend)
    DOWN_MIN_MOMENTUM_DROP = -0.002  # Price must be FALLING to bet DOWN

    # Risk management
    MAX_DAILY_LOSS = 300.0        # Stop after losing $300 in a day
    SKIP_HOURS_UTC = {0, 6, 7, 8, 14, 15, 19, 20, 21, 23}  # Whale-validated skip hours

    def __init__(self, bankroll: float = 1000.0):
        self.generator = TASignalGenerator()
        self.bregman = BregmanOptimizer(bankroll=bankroll)
        self.bankroll = bankroll
        self.trades: Dict[str, TAPaperTrade] = {}
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.signals_count = 0
        self.bregman_signals = 0
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.traded_markets_this_cycle: set = set()  # Prevent duplicate orders
        self.daily_pnl = 0.0  # Track daily P&L
        self.last_reset_day = datetime.now(timezone.utc).date()

        # ML Auto-Tuner
        self.ml_tuner = MLAutoTuner()
        self.last_tune_time = 0  # Unix timestamp of last tune
        self.TUNE_INTERVAL = 1800  # Tune every 30 minutes

        # ML V3 Engine (LightGBM + XGBoost ensemble)
        self.ml_engine = get_ml_engine()
        self.use_ml_v3 = True  # Enable advanced ML predictions

        self._load()
        self._apply_tuned_params()  # Apply ML-tuned parameters on startup

    def _apply_tuned_params(self):
        """Apply ML-tuned parameters to trading thresholds."""
        params = self.ml_tuner.get_current_params()
        self.UP_MAX_PRICE = params["up_max_price"]
        self.UP_MIN_CONFIDENCE = params["up_min_confidence"]
        self.DOWN_MAX_PRICE = params["down_max_price"]
        self.DOWN_MIN_CONFIDENCE = params["down_min_confidence"]
        self.MIN_EDGE = params["min_edge"]
        self.DOWN_MIN_MOMENTUM_DROP = params["down_momentum_threshold"]
        print(f"[ML] Applied tuned params: UP<${self.UP_MAX_PRICE}@{self.UP_MIN_CONFIDENCE:.0%} | DOWN<${self.DOWN_MAX_PRICE}@{self.DOWN_MIN_CONFIDENCE:.0%} | Edge>{self.MIN_EDGE:.0%}")

    def _load(self):
        if self.OUTPUT_FILE.exists():
            try:
                data = json.load(open(self.OUTPUT_FILE))
                for tid, t in data.get("trades", {}).items():
                    self.trades[tid] = TAPaperTrade(**t)
                self.total_pnl = data.get("total_pnl", 0)
                self.wins = data.get("wins", 0)
                self.losses = data.get("losses", 0)
                self.signals_count = data.get("signals_count", 0)
                self.start_time = data.get("start_time", self.start_time)
                print(f"Loaded {len(self.trades)} trades from previous session")
            except Exception as e:
                print(f"Error loading state: {e}")

    def _save(self):
        data = {
            "start_time": self.start_time,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "trades": {tid: asdict(t) for tid, t in self.trades.items()},
            "total_pnl": self.total_pnl,
            "wins": self.wins,
            "losses": self.losses,
            "signals_count": self.signals_count,
        }
        with open(self.OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    async def fetch_data(self):
        """Fetch BTC candles and price with rate limiting."""
        async with httpx.AsyncClient(timeout=15) as client:
            # Candles from Binance
            r = await client.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "1m", "limit": 240}
            )
            klines = r.json()

            # Current price
            pr = await client.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": "BTCUSDT"}
            )
            btc_price = float(pr.json()["price"])

            # Rate limit: delay before Polymarket call
            await asyncio.sleep(1.5)

            # BTC 15m markets - use EVENTS endpoint with tag_slug=15M
            # This returns all crypto 15m up/down markets, then filter for BTC
            markets = []
            try:
                mr = await client.get(
                    "https://gamma-api.polymarket.com/events",
                    params={"tag_slug": "15M", "active": "true", "closed": "false", "limit": 50},
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                if mr.status_code == 200:
                    events = mr.json()
                    for event in events:
                        title = event.get("title", "").lower()
                        # Filter for Bitcoin markets only
                        if "bitcoin" not in title and "btc" not in title:
                            continue
                        for m in event.get("markets", []):
                            if not m.get("closed", True):
                                # Copy event title to market question if missing
                                if not m.get("question"):
                                    m["question"] = event.get("title", "")
                                markets.append(m)
                    # Cache for rate limit fallback
                    if markets:
                        self._market_cache = markets
                else:
                    print(f"[API] Status {mr.status_code}, using cache")
                    markets = getattr(self, '_market_cache', [])
            except Exception as e:
                print(f"[API] Error: {e}, using cache")
                markets = getattr(self, '_market_cache', [])

        candles = [
            Candle(k[0]/1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
            for k in klines
        ]

        btc_markets = markets  # Already filtered by series slug

        if btc_markets:
            print(f"[Markets] Found {len(btc_markets)} BTC 15m Up/Down market(s)")
        else:
            print("[Markets] No active BTC 15m markets found")

        return candles, btc_price, btc_markets

    def get_market_prices(self, market: Dict) -> tuple:
        """Extract UP/DOWN prices."""
        outcomes = market.get("outcomes", [])
        prices = market.get("outcomePrices", [])

        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(prices, str):
            prices = json.loads(prices)

        up_price, down_price = None, None
        for i, o in enumerate(outcomes):
            if i < len(prices):
                p = float(prices[i])
                if str(o).lower() == "up":
                    up_price = p
                elif str(o).lower() == "down":
                    down_price = p

        return up_price, down_price

    def get_time_remaining(self, market: Dict) -> float:
        """Get minutes remaining."""
        end = market.get("endDate")
        if not end:
            return 15.0
        try:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return max(0, (end_dt - now).total_seconds() / 60)
        except:
            return 15.0

    # Conviction thresholds - prevent flip-flopping
    MIN_MODEL_CONFIDENCE = 0.65  # Model must be at least 65% confident in direction
    MIN_EDGE = 0.12              # At least 12% edge vs market price (was 10%)
    MAX_ENTRY_PRICE = 0.55       # Don't buy at prices above 55¢ (no edge zone)
    CANDLE_LOOKBACK = 15         # Only use last 15 minutes of price action

    def _get_price_momentum(self, candles: List, lookback: int = 5) -> float:
        """Calculate recent price momentum (% change over lookback candles)."""
        if len(candles) < lookback:
            return 0.0
        old_price = candles[-lookback].close
        new_price = candles[-1].close
        return (new_price - old_price) / old_price

    def _reset_daily_if_needed(self):
        """Reset daily P&L tracker at midnight UTC."""
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset_day:
            print(f"[DAILY RESET] Previous day PnL: ${self.daily_pnl:+.2f}")
            self.daily_pnl = 0.0
            self.last_reset_day = today

    async def run_cycle(self):
        """Run one trading cycle."""
        self._reset_daily_if_needed()
        self.traded_markets_this_cycle.clear()  # Fresh cycle

        # Check daily loss limit
        if self.daily_pnl <= -self.MAX_DAILY_LOSS:
            print(f"[RISK] Daily loss limit hit: ${self.daily_pnl:.2f} - resting")
            return None

        # Check skip hours
        current_hour = datetime.now(timezone.utc).hour
        if current_hour in self.SKIP_HOURS_UTC:
            print(f"[SKIP] Hour {current_hour} UTC is in skip list - resting")
            return None

        candles, btc_price, markets = await self.fetch_data()

        if not candles:
            return None

        # IMPORTANT: Only use the most recent 15 minutes of price action
        recent_candles = candles[-self.CANDLE_LOOKBACK:]

        # Calculate momentum for directional filter
        momentum = self._get_price_momentum(candles, lookback=5)

        # Generate main signal from recent price action only
        main_signal = self.generator.generate_signal(
            market_id="btc_main",
            candles=recent_candles,
            current_price=btc_price,
            market_yes_price=0.5,
            market_no_price=0.5,
            time_remaining_min=10.0
        )

        self.signals_count += 1

        # Sort markets by time remaining - only trade the NEAREST expiring one
        eligible_markets = []
        for market in markets:
            up_price, down_price = self.get_market_prices(market)
            time_left = self.get_time_remaining(market)
            if up_price is None or down_price is None:
                continue
            if time_left > 15 or time_left < 2:
                continue
            eligible_markets.append((time_left, market, up_price, down_price))

        # Sort by time remaining (ascending) - nearest expiring first
        eligible_markets.sort(key=lambda x: x[0])

        if eligible_markets:
            print(f"[Filter] {len(eligible_markets)} markets in window, trading nearest ({eligible_markets[0][0]:.1f}min left)")

        # Only trade the nearest expiring market for high turnover
        for time_left, market, up_price, down_price in eligible_markets[:1]:
            market_id = market.get("conditionId", "")
            market_numeric_id = market.get("id")
            question = market.get("question", "")

            # Generate signal from recent 15-min price action only
            signal = self.generator.generate_signal(
                market_id=market_id,
                candles=recent_candles,
                current_price=btc_price,
                market_yes_price=up_price,
                market_no_price=down_price,
                time_remaining_min=time_left
            )

            trade_key = f"{market_id}_{signal.side}" if signal.side else None

            # === DUPLICATE PROTECTION ===
            # One trade per market per cycle - NO exceptions
            if market_id in self.traded_markets_this_cycle:
                continue

            # === WINNING FORMULA V2 FILTERS ===
            skip_reason = None

            # 1. ASYMMETRIC CONFIDENCE: UP is easier, DOWN is harder
            if signal.side == "UP":
                if signal.model_up < self.UP_MIN_CONFIDENCE:
                    skip_reason = f"UP_conf_{signal.model_up:.0%}<{self.UP_MIN_CONFIDENCE:.0%}"
                elif up_price > self.UP_MAX_PRICE:
                    skip_reason = f"UP_price_{up_price:.2f}>{self.UP_MAX_PRICE}"
                # UP needs price NOT falling
                elif momentum < -0.001:
                    skip_reason = f"UP_momentum_negative_{momentum:.3%}"
            elif signal.side == "DOWN":
                if signal.model_down < self.DOWN_MIN_CONFIDENCE:
                    skip_reason = f"DOWN_conf_{signal.model_down:.0%}<{self.DOWN_MIN_CONFIDENCE:.0%}"
                elif down_price > self.DOWN_MAX_PRICE:
                    skip_reason = f"DOWN_price_{down_price:.2f}>{self.DOWN_MAX_PRICE}"
                # DOWN needs price FALLING (momentum confirmation)
                elif momentum > self.DOWN_MIN_MOMENTUM_DROP:
                    skip_reason = f"DOWN_momentum_not_falling_{momentum:.3%}"

            if skip_reason:
                print(f"[V2 Filter] {question[:30]}... | {skip_reason}")
                continue

            # 2. Minimum edge requirement (increased from 10% to 12%)
            best_edge = signal.edge_up if signal.side == "UP" else signal.edge_down
            if best_edge is not None and best_edge < self.MIN_EDGE:
                skip_reason = f"edge_{best_edge:.1%}<{self.MIN_EDGE:.0%}"
                print(f"[Edge] {question[:30]}... | {skip_reason}")
                continue

            # Debug: show why no trade
            if signal.action != "ENTER":
                edge = max(signal.edge_up or 0, signal.edge_down or 0)
                print(f"[Skip] {question[:35]}... | UP:{up_price:.2f} DOWN:{down_price:.2f} | Edge:{edge:.1%} | {signal.reason}")

            # Enter trade if signal - use Bregman optimization for sizing
            if signal.action == "ENTER" and signal.side and trade_key:
                if trade_key not in self.trades:
                    entry_price = up_price if signal.side == "UP" else down_price
                    edge = signal.edge_up if signal.side == "UP" else signal.edge_down

                    # Bregman divergence optimization
                    bregman_signal = self.bregman.calculate_optimal_trade(
                        model_prob=signal.model_up,
                        market_yes_price=up_price,
                        market_no_price=down_price
                    )
                    self.bregman_signals += 1

                    # === ML V3 PREDICTION ===
                    ml_approved = True
                    ml_str = ""
                    if self.use_ml_v3:
                        ml_features = self.ml_engine.extract_features(
                            candles=recent_candles,
                            current_price=btc_price,
                            up_price=up_price,
                            down_price=down_price,
                            signal_data={
                                'rsi': signal.rsi or 50,
                                'price_vs_vwap': (btc_price - (signal.vwap or btc_price)) / btc_price if signal.vwap else 0,
                                'kl_divergence': bregman_signal.kl_divergence,
                                'heiken_streak': signal.heiken_count,
                                'trend_strength': momentum
                            }
                        )
                        ml_prediction = self.ml_engine.predict(ml_features)

                        # Store features for training later
                        self._pending_ml_features = ml_features

                        # Check if ML agrees with TA signal
                        if ml_prediction.recommended_side == "SKIP":
                            ml_approved = False
                            ml_str = f"[ML:SKIP conf={ml_prediction.confidence:.0%}]"
                        elif ml_prediction.recommended_side != signal.side:
                            # ML disagrees - reduce confidence but don't block
                            if ml_prediction.confidence > 0.65:
                                ml_approved = False
                                ml_str = f"[ML:{ml_prediction.recommended_side} conf={ml_prediction.confidence:.0%} VETO]"
                            else:
                                ml_str = f"[ML:{ml_prediction.recommended_side} conf={ml_prediction.confidence:.0%} weak]"
                        else:
                            ml_str = f"[ML:AGREE conf={ml_prediction.confidence:.0%}]"

                    if not ml_approved:
                        print(f"[ML V3] {question[:30]}... | {ml_str} - skipping")
                        continue

                    # Use Kelly fraction for position sizing (capped)
                    optimal_size = min(self.POSITION_SIZE, self.bankroll * bregman_signal.kelly_fraction)
                    optimal_size = max(10.0, optimal_size)  # Minimum $10

                    trade = TAPaperTrade(
                        trade_id=trade_key,
                        market_title=question[:80],
                        side=signal.side,
                        entry_price=entry_price,
                        entry_time=datetime.now(timezone.utc).isoformat(),
                        size_usd=optimal_size,
                        signal_strength=signal.strength.value,
                        edge_at_entry=edge if edge else 0,
                        kl_divergence=bregman_signal.kl_divergence,
                        kelly_fraction=bregman_signal.kelly_fraction,
                        guaranteed_profit=bregman_signal.guaranteed_profit,
                        market_numeric_id=market_numeric_id,
                    )
                    self.trades[trade_key] = trade
                    self.traded_markets_this_cycle.add(market_id)  # PREVENT DUPLICATES
                    momentum_str = f"Mom:{momentum:+.2%}" if momentum else ""
                    print(f"[NEW] {signal.side} @ ${entry_price:.4f} | Edge: {edge:.1%} | {momentum_str} | {ml_str}")
                    print(f"      Size: ${optimal_size:.2f} | KL: {bregman_signal.kl_divergence:.4f} | {question[:45]}...")

            # Check for resolved markets
            if time_left < 0.5:
                for tid, trade in list(self.trades.items()):
                    if trade.status == "open" and market_id in tid:
                        price = up_price if trade.side == "UP" else down_price

                        # Determine outcome
                        if price >= 0.95:
                            exit_val = trade.size_usd / trade.entry_price
                        elif price <= 0.05:
                            exit_val = 0
                        else:
                            exit_val = (trade.size_usd / trade.entry_price) * price

                        trade.exit_price = price
                        trade.exit_time = datetime.now(timezone.utc).isoformat()
                        trade.pnl = exit_val - trade.size_usd
                        trade.status = "closed"

                        self.total_pnl += trade.pnl
                        self.daily_pnl += trade.pnl  # Track daily P&L
                        if trade.pnl > 0:
                            self.wins += 1
                        else:
                            self.losses += 1

                        result = "WIN" if trade.pnl > 0 else "LOSS"
                        print(f"[{result}] {trade.side} PnL: ${trade.pnl:+.2f} | Day: ${self.daily_pnl:+.2f} | {trade.market_title[:35]}...")

                        # === ML V3 ONLINE LEARNING ===
                        if self.use_ml_v3 and hasattr(self, '_pending_ml_features'):
                            try:
                                self.ml_engine.add_training_sample(
                                    features=self._pending_ml_features,
                                    outcome=result,
                                    pnl=trade.pnl,
                                    side_taken=trade.side
                                )
                            except Exception as e:
                                print(f"[ML V3] Training update error: {e}")

        # === ACTIVE MARKET RESOLUTION ===
        # Check ALL markets (not just eligible ones) for near-expiry resolution.
        # This catches markets with time_left < 2 that were filtered out above.
        for mkt in markets:
            mkt_time = self.get_time_remaining(mkt)
            if mkt_time < 0.5:
                mkt_id = mkt.get("conditionId", "")
                up_p, down_p = self.get_market_prices(mkt)
                if up_p is None or down_p is None:
                    continue
                for tid, trade in list(self.trades.items()):
                    if trade.status == "open" and mkt_id in tid:
                        price = up_p if trade.side == "UP" else down_p
                        if price >= 0.95:
                            exit_val = trade.size_usd / trade.entry_price
                        elif price <= 0.05:
                            exit_val = 0
                        else:
                            exit_val = (trade.size_usd / trade.entry_price) * price
                        trade.exit_price = price
                        trade.exit_time = datetime.now(timezone.utc).isoformat()
                        trade.pnl = exit_val - trade.size_usd
                        trade.status = "closed"
                        self.total_pnl += trade.pnl
                        self.daily_pnl += trade.pnl
                        if trade.pnl > 0:
                            self.wins += 1
                        else:
                            self.losses += 1
                        result = "WIN" if trade.pnl > 0 else "LOSS"
                        print(f"[{result}] {trade.side} PnL: ${trade.pnl:+.2f} | Day: ${self.daily_pnl:+.2f} | {trade.market_title[:35]}...")

        # === EXPIRY RESOLUTION ===
        # Markets disappear from active=true&closed=false after they close.
        # Check all open trades older than 16 minutes and resolve them via Gamma API.
        now = datetime.now(timezone.utc)
        for tid, open_trade in list(self.trades.items()):
            if open_trade.status != "open":
                continue
            try:
                entry_dt = datetime.fromisoformat(open_trade.entry_time)
            except Exception:
                continue
            age_minutes = (now - entry_dt).total_seconds() / 60
            if age_minutes > 16:
                if not open_trade.market_numeric_id:
                    # Old trades without numeric ID — can't resolve via API, mark as stale
                    open_trade.status = "closed"
                    open_trade.exit_price = 0.5
                    open_trade.exit_time = now.isoformat()
                    open_trade.pnl = 0.0  # Unknown outcome
                    print(f"[STALE] {open_trade.side} | No numeric ID, marking closed | {open_trade.market_title[:40]}...")
                    continue
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        r = await client.get(
                            f"https://gamma-api.polymarket.com/markets/{open_trade.market_numeric_id}",
                            headers={"User-Agent": "Mozilla/5.0"}
                        )
                        if r.status_code == 200:
                            rm = r.json()
                            up_p, down_p = self.get_market_prices(rm)
                            if up_p is not None and down_p is not None:
                                price = up_p if open_trade.side == "UP" else down_p
                                if price >= 0.95:
                                    exit_val = open_trade.size_usd / open_trade.entry_price
                                elif price <= 0.05:
                                    exit_val = 0
                                else:
                                    exit_val = (open_trade.size_usd / open_trade.entry_price) * price
                                open_trade.exit_price = price
                                open_trade.exit_time = now.isoformat()
                                open_trade.pnl = exit_val - open_trade.size_usd
                                open_trade.status = "closed"
                                self.total_pnl += open_trade.pnl
                                self.daily_pnl += open_trade.pnl
                                if open_trade.pnl > 0:
                                    self.wins += 1
                                else:
                                    self.losses += 1
                                result = "WIN" if open_trade.pnl > 0 else "LOSS"
                                print(f"[{result}] (expired) {open_trade.side} PnL: ${open_trade.pnl:+.2f} | Day: ${self.daily_pnl:+.2f} | {open_trade.market_title[:30]}...")
                    await asyncio.sleep(0.5)  # Rate limit between resolved market lookups
                except Exception as e:
                    print(f"[Expiry] Error resolving {tid[:20]}: {e}")

        self._save()
        return main_signal

    def print_update(self, signal):
        """Print 10-minute update."""
        now = datetime.now(timezone.utc)
        open_trades = [t for t in self.trades.values() if t.status == "open"]
        closed_trades = [t for t in self.trades.values() if t.status == "closed"]

        print()
        print("=" * 70)
        print(f"TA PAPER TRADING UPDATE - {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 70)

        # Signal info
        if signal:
            print(f"\nCURRENT TA SIGNAL:")
            print(f"  BTC: ${signal.current_price:,.2f}")
            print(f"  VWAP: ${signal.vwap:,.2f}" if signal.vwap else "  VWAP: N/A")
            print(f"  RSI: {signal.rsi:.1f}" if signal.rsi else "  RSI: N/A")
            print(f"  Heiken: {signal.heiken_color} x{signal.heiken_count}")
            print(f"  Regime: {signal.regime.value}")
            print(f"  Model: {signal.model_up:.1%} UP / {signal.model_down:.1%} DOWN")

        # Stats
        print(f"\nTRADING STATS:")
        print(f"  Running since: {self.start_time[:19]}")
        print(f"  TA Signals: {self.signals_count} | Bregman Signals: {self.bregman_signals}")
        print(f"  Total trades: {len(closed_trades)}")
        print(f"  Win/Loss: {self.wins}/{self.losses}")
        win_rate = (self.wins / max(1, self.wins + self.losses)) * 100
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Total PnL: ${self.total_pnl:+.2f}")
        print(f"  Today PnL: ${self.daily_pnl:+.2f}")

        # ML V3 Engine stats
        ml_status = self.ml_engine.get_status()
        print(f"\nML V3 ENGINE (LightGBM+XGBoost):")
        print(f"  Trained: {ml_status['is_trained']} | Samples: {ml_status['training_samples']}")
        print(f"  Predictions: {ml_status['predictions_made']} | Accuracy: {ml_status['accuracy']:.1%}")
        print(f"  Adaptive threshold: {ml_status['adaptive_threshold']:.0%}")
        if ml_status['recent_win_rate'] > 0:
            print(f"  Recent WR: {ml_status['recent_win_rate']:.1%}")

        # ML Tuner stats
        print(f"\nML AUTO-TUNER:")
        print(f"  Tune cycles: #{self.ml_tuner.state.tune_count}")
        print(f"  UP:   <${self.UP_MAX_PRICE:.2f} @ {self.UP_MIN_CONFIDENCE:.0%}")
        print(f"  DOWN: <${self.DOWN_MAX_PRICE:.2f} @ {self.DOWN_MIN_CONFIDENCE:.0%}")
        print(f"  Edge: >{self.MIN_EDGE:.0%}")

        # Bregman stats
        if closed_trades:
            avg_kl = sum(t.kl_divergence for t in closed_trades) / len(closed_trades)
            avg_kelly = sum(t.kelly_fraction for t in closed_trades) / len(closed_trades)
            print(f"\nBREGMAN OPTIMIZATION:")
            print(f"  Avg KL Divergence: {avg_kl:.4f}")
            print(f"  Avg Kelly Fraction: {avg_kelly:.1%}")

        # Open trades
        print(f"\nOPEN TRADES ({len(open_trades)}):")
        for t in open_trades[:5]:
            print(f"  {t.side} @ ${t.entry_price:.4f} | KL: {t.kl_divergence:.4f} | {t.market_title[:35]}...")

        # Recent closed
        print(f"\nRECENT CLOSED:")
        recent = sorted(closed_trades, key=lambda x: x.exit_time or "", reverse=True)[:5]
        for t in recent:
            result = "WIN" if t.pnl > 0 else "LOSS"
            print(f"  [{result}] {t.side} ${t.pnl:+.2f} | {t.market_title[:35]}...")

        print("=" * 70)
        print()

    async def _run_ml_tuning(self):
        """Run ML auto-tuning cycle."""
        print("\n" + "=" * 50)
        print("[ML AUTO-TUNER] Running optimization cycle...")
        print("=" * 50)

        # Get current session trades
        current_trades = [asdict(t) for t in self.trades.values() if t.status == "closed"]

        # Run tuning
        changes, summary = self.ml_tuner.run_tuning_cycle(current_trades)

        # Print analysis
        print(f"Trades analyzed: {summary.get('total_trades_analyzed', 0)}")
        print(f"UP:   {summary.get('up_wr', 0):.1%} WR | PnL: ${summary.get('up_pnl', 0):+.2f}")
        print(f"DOWN: {summary.get('down_wr', 0):.1%} WR | PnL: ${summary.get('down_pnl', 0):+.2f}")

        if summary.get("best_price_bucket"):
            print(f"Best price bucket: {summary['best_price_bucket']} ({summary.get('best_price_wr', 0):.1%} WR)")

        if changes:
            print("\n[ML CHANGES APPLIED]:")
            for change in changes:
                print(f"  -> {change}")
            self._apply_tuned_params()
        else:
            print("\n[ML] No parameter changes needed")

        print(f"Tune cycle #{self.ml_tuner.state.tune_count}")
        print("=" * 50 + "\n")

    async def run(self):
        """Main loop with 10-minute updates."""
        print("=" * 70)
        print("TA + BREGMAN PAPER TRADER V3 - LIGHTGBM + XGBOOST ENSEMBLE")
        print("=" * 70)
        print("Strategy: Momentum-confirmed + ML ensemble prediction")
        print("ML V3 Engine: LightGBM + XGBoost (online learning enabled)")
        ml_status = self.ml_engine.get_status()
        print(f"  Training samples: {ml_status['training_samples']} | Trained: {ml_status['is_trained']}")
        print("ML Auto-Tuner: ENABLED (tunes every 30 min based on performance)")
        print(f"Tune cycle: #{self.ml_tuner.state.tune_count}")
        print()
        print("CURRENT ML-TUNED PARAMETERS:")
        print(f"  UP:   Max price ${self.UP_MAX_PRICE:.2f} | Min conf {self.UP_MIN_CONFIDENCE:.0%} | Need rising momentum")
        print(f"  DOWN: Max price ${self.DOWN_MAX_PRICE:.2f} | Min conf {self.DOWN_MIN_CONFIDENCE:.0%} | Need falling momentum")
        print(f"  Min edge: {self.MIN_EDGE:.0%}")
        print()
        print(f"  Skip hours: {sorted(self.SKIP_HOURS_UTC)} UTC")
        print(f"  Daily loss limit: ${self.MAX_DAILY_LOSS}")
        print(f"Bankroll: ${self.bankroll} | Max Position: ${self.POSITION_SIZE}")
        print("Scan: 2min | Update: 10min | ML Tune: 30min")
        print("=" * 70)
        print()

        last_update = 0
        cycle = 0

        while True:
            try:
                cycle += 1
                now = time.time()

                signal = await self.run_cycle()

                if signal:
                    print(f"[Scan {cycle}] BTC ${signal.current_price:,.2f} | {signal.regime.value} | Model: {signal.model_up:.0%} UP")

                # 10-minute update
                if now - last_update >= 600:
                    self.print_update(signal)
                    last_update = now

                # ML Auto-Tune every 30 minutes
                if now - self.last_tune_time >= self.TUNE_INTERVAL:
                    await self._run_ml_tuning()
                    self.last_tune_time = now

                await asyncio.sleep(120)  # 2 minutes between scans to avoid rate limiting

            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

        self.print_update(None)
        print(f"Results saved to: {self.OUTPUT_FILE}")


if __name__ == "__main__":
    trader = TAPaperTrader()
    asyncio.run(trader.run())
