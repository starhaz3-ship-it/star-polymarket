"""
TA Paper Trading Runner with 10-minute Updates + ML Auto-Tuner

Standalone script that runs TA-based paper trading for BTC 15m markets.
Includes ML-based automatic parameter tuning based on trade performance.
"""

import asyncio
import json
import math
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
from arbitrage.nyu_volatility import NYUVolatilityModel, calculate_nyu_volatility
from arbitrage.meta_labeler import get_meta_labeler, MetaLabeler


@dataclass
class MLTunerState:
    """State for ML auto-tuner."""
    # Current optimized parameters
    up_max_price: float = 0.45
    up_min_confidence: float = 0.60
    down_max_price: float = 0.40
    down_min_confidence: float = 0.72
    min_edge: float = 0.30
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
            new_conf = min(0.80, self.state.down_min_confidence + 0.03)
            recommendations["down_min_confidence"] = new_conf
            new_price = max(0.30, self.state.down_max_price - 0.02)
            recommendations["down_max_price"] = new_price
        elif down_wr > 0.60 and down_total >= 5:
            # DOWN is doing well - relax confidence AND expand price range
            # Data shows $0.50-0.55 has 73% WR, best PnL bucket
            new_conf = max(0.55, self.state.down_min_confidence - 0.02)
            recommendations["down_min_confidence"] = new_conf
            # Allow up to $0.55 when DOWN is strong (73% WR in 0.50-0.55 bucket)
            new_price = min(0.55, self.state.down_max_price + 0.01)
            recommendations["down_max_price"] = new_price

        # If UP is underperforming, tighten requirements (slower ratchet)
        # Note: dynamic pricing in V2 filters overrides base max price for high-confidence signals
        if up_wr < 0.50 and up_total >= 5:
            # Only tighten if not already at floor - avoid constant no-op changes
            if self.state.up_min_confidence < 0.70:
                new_conf = min(0.70, self.state.up_min_confidence + 0.02)
                recommendations["up_min_confidence"] = new_conf
            if self.state.up_max_price > 0.38:
                new_price = max(0.38, self.state.up_max_price - 0.02)
                recommendations["up_max_price"] = new_price
        elif up_wr > 0.55 and up_total >= 5:
            # UP improving - relax more aggressively to capture opportunity
            new_conf = max(0.55, self.state.up_min_confidence - 0.03)
            recommendations["up_min_confidence"] = new_conf
            new_price = min(0.50, self.state.up_max_price + 0.03)
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
    size_usd: float = 10.0
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
    POSITION_SIZE = 10.0

    # === WINNING FORMULA V2 ===
    # Lessons learned from live trading losses:
    # 1. DOWN bets at 44-48¢ were losing - market was going UP
    # 2. Cheap UP entries (28-34¢) were profitable value plays
    # 3. Multiple orders per market = disaster (DCA gone wrong)
    # 4. Need momentum confirmation before betting direction

    # === BACKTEST-VALIDATED FILTERS (2026-02-06) ===
    # DOWN at cheap prices with falling momentum = best edge
    UP_MAX_PRICE = 0.40           # Strict: only cheap UP entries (<40¢)
    UP_MIN_CONFIDENCE = 0.65      # Higher bar for UP (underperforms)
    DOWN_MAX_PRICE = 0.40         # Critical: entry < $0.40 for edge
    DOWN_MIN_CONFIDENCE = 0.55    # DOWN is more reliable, can be lower
    DOWN_MIN_MOMENTUM_DROP = -0.002  # CRITICAL: Price must be FALLING

    # Risk management
    MAX_DAILY_LOSS = 30.0         # Stop after losing $30 in a day
    # V3.2: Data-driven skip hours from 158 trade analysis
    # REMOVED from skip (profitable): 6(56%WR +$108), 8(67%WR +$218), 10(67%WR +$74), 14(60%WR +$414)
    # ADDED to skip (losing): 1(40%WR -$21), 3(45%WR -$128), 16(25%WR -$236), 17(33%WR -$133), 22(33%WR -$23)
    # Best hours: 2(78%WR), 4(56%WR), 13(62%WR), 18(57%WR), 21(100%WR)
    SKIP_HOURS_UTC = {0, 1, 8, 22, 23}  # Opened US/EU overlap (UTC 15-17,19,20) + UTC 3 (proven profitable)

    def __init__(self, bankroll: float = 93.27):
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
        self.TUNE_INTERVAL = 7200  # Tune every 2 hours (was 30 min - too aggressive)

        # ML V3 Engine (LightGBM + XGBoost ensemble)
        self.ml_engine = get_ml_engine()
        self.use_ml_v3 = True  # Enable advanced ML predictions

        # NYU Two-Parameter Volatility Model
        # Key insight: Only price + time to expiry matter for volatility
        # Extreme prices (away from 0.50) have LOWER volatility = better edge
        self.nyu_model = NYUVolatilityModel()
        self.use_nyu_model = True  # Enable NYU volatility-based edge scoring

        # Meta-Labeler (Lopez de Prado framework)
        # Predicts P(primary signal is correct) instead of direction
        # Replaces binary ML veto with calibrated confidence + position sizing
        self.meta_labeler = get_meta_labeler()
        self.use_meta_labeler = True

        # Multi-Outcome Kelly (correlated asset allocation)
        # BTC/ETH/SOL are ~85% correlated - 3 same-direction bets ≈ 1 big bet
        # Allocates by quality (best P(correct) first) with correlation discount
        self.CRYPTO_CORRELATION = 0.85  # Same-direction correlation
        self.MAX_CYCLE_BUDGET = self.POSITION_SIZE * 2.5  # Max $25 total per cycle

        # Arbitrage Scanner (distinct-baguette strategy)
        # Buy both UP + DOWN when combined price < threshold for guaranteed profit
        self.ARB_ENABLED = True
        self.ARB_THRESHOLD = 0.97      # Max combined price (need sum < 0.97 for ~3% profit)
        self.ARB_FEE_RATE = 0.02       # Simulated Polymarket fee (2%)
        self.ARB_POSITION_SIZE = 10.0  # Total for both sides combined ($5 each side)
        self.ARB_MIN_TIME = 2.0        # Min minutes remaining
        self.ARB_MAX_TIME = 14.0       # Max minutes remaining
        self.arb_trades = {}           # {trade_id: arb_trade_dict}
        self.arb_stats = {"wins": 0, "losses": 0, "total_pnl": 0.0}

        # === HOURLY ML POSITION SIZING ===
        # Bayesian: start neutral, reduce bad hours, boost good hours after proof
        self.hourly_stats: Dict[int, dict] = {
            h: {"wins": 0, "losses": 0, "pnl": 0.0} for h in range(24)
        }

        # === SKIP HOUR SHADOW TRACKING ===
        # Paper-trade during paused hours WITHOUT affecting real stats.
        # Accumulates data to evaluate whether skip hours should be reopened.
        self.skip_hour_shadows: Dict[str, dict] = {}  # {trade_key: shadow_trade_dict}
        self.skip_hour_stats: Dict[int, dict] = {
            h: {"wins": 0, "losses": 0, "pnl": 0.0} for h in range(24)
        }

        self._load()
        self._apply_tuned_params()  # Apply ML-tuned parameters on startup

    def _get_hour_multiplier(self, hour: int) -> float:
        """
        Bayesian hourly position multiplier. Learns win rates per hour and adjusts.
        Phase 1 (< 10 total trades): only REDUCE for bad hours (0.5x-1.0x)
        Phase 2 (>= 10 total trades): allow BOOST for proven hours (0.5x-1.5x)
        Uses Beta prior: 5W/5L per hour (50% WR) for smoothing.
        """
        stats = self.hourly_stats.get(hour, {"wins": 0, "losses": 0, "pnl": 0.0})
        prior_wins, prior_losses = 5, 5
        post_wins = stats["wins"] + prior_wins
        post_losses = stats["losses"] + prior_losses
        post_wr = post_wins / (post_wins + post_losses)

        total_all_hours = sum(s["wins"] + s["losses"] for s in self.hourly_stats.values())
        phase2 = total_all_hours >= 10

        if post_wr < 0.50:
            mult = max(0.5, 0.5 + (post_wr / 0.50) * 0.5)
        elif phase2 and post_wr > 0.50:
            mult = min(1.5, 1.0 + ((post_wr - 0.50) / 0.20) * 0.5)
        else:
            mult = 1.0
        return round(mult, 2)

    def _apply_tuned_params(self):
        """Apply ML-tuned parameters to trading thresholds."""
        params = self.ml_tuner.get_current_params()
        self.UP_MAX_PRICE = params["up_max_price"]
        self.UP_MIN_CONFIDENCE = params["up_min_confidence"]
        self.DOWN_MAX_PRICE = params["down_max_price"]
        self.DOWN_MIN_CONFIDENCE = params["down_min_confidence"]
        self.MIN_EDGE = max(params["min_edge"], 0.30)  # V3.4: floor at 0.30
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
                # Load arb state
                self.arb_trades = data.get("arb_trades", {})
                self.arb_stats = data.get("arb_stats", {"wins": 0, "losses": 0, "total_pnl": 0.0})
                # Load hourly stats
                saved_hourly = data.get("hourly_stats", {})
                for h_str, stats in saved_hourly.items():
                    h = int(h_str)
                    if h in self.hourly_stats:
                        self.hourly_stats[h] = stats
                # Load skip-hour shadow tracking
                self.skip_hour_shadows = data.get("skip_hour_shadows", {})
                saved_skip_stats = data.get("skip_hour_stats", {})
                for h_str, stats in saved_skip_stats.items():
                    h = int(h_str)
                    if h in self.skip_hour_stats:
                        self.skip_hour_stats[h] = stats
                arb_open = sum(1 for a in self.arb_trades.values() if a.get("status") == "open")
                print(f"Loaded {len(self.trades)} trades + {len(self.arb_trades)} arb trades ({arb_open} open) from previous session")
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
            "arb_trades": self.arb_trades,
            "arb_stats": self.arb_stats,
            "hourly_stats": self.hourly_stats,
            "skip_hour_shadows": self.skip_hour_shadows,
            "skip_hour_stats": self.skip_hour_stats,
        }
        with open(self.OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    # Asset config: Binance symbols and Polymarket title keywords
    ASSETS = {
        "BTC": {"symbol": "BTCUSDT", "keywords": ["bitcoin", "btc"]},
        "ETH": {"symbol": "ETHUSDT", "keywords": ["ethereum", "eth"]},
        "SOL": {"symbol": "SOLUSDT", "keywords": ["solana", "sol"]},
    }

    async def _fetch_asset_data(self, client, symbol: str):
        """Fetch candles, price, orderbook and trades for one asset."""
        try:
            r = await client.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": symbol, "interval": "1m", "limit": 240}
            )
            klines = r.json()

            pr = await client.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": symbol}
            )
            price = float(pr.json()["price"])

            ob = await client.get(
                "https://api.binance.com/api/v3/depth",
                params={"symbol": symbol, "limit": 20}
            )
            ob_data = ob.json()
            bids = [(float(p), float(q)) for p, q in ob_data.get("bids", [])]
            asks = [(float(p), float(q)) for p, q in ob_data.get("asks", [])]

            tr = await client.get(
                "https://api.binance.com/api/v3/trades",
                params={"symbol": symbol, "limit": 500}
            )
            raw_trades = tr.json()
            trades = [{
                "t": t["time"] / 1000,
                "price": float(t["price"]),
                "qty": float(t["qty"]),
                "is_buy": not t["isBuyerMaker"]
            } for t in raw_trades]

            candles = [
                Candle(k[0]/1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
                for k in klines
            ]
            return candles, price, bids, asks, trades
        except Exception as e:
            print(f"[API] Error fetching {symbol}: {e}")
            return [], 0.0, [], [], []

    async def fetch_data(self):
        """Fetch candles, prices, orderbook and markets for BTC + ETH + SOL."""
        async with httpx.AsyncClient(timeout=15) as client:
            # Fetch all asset data in parallel
            import asyncio as aio
            asset_tasks = {
                asset: self._fetch_asset_data(client, cfg["symbol"])
                for asset, cfg in self.ASSETS.items()
            }
            asset_results = {}
            for asset, task in asset_tasks.items():
                asset_results[asset] = await task
                await aio.sleep(0.3)  # Rate limit between Binance calls

            # Cache per-asset data for signal generation
            self._asset_data = {}
            for asset, (candles, price, bids, asks, trades) in asset_results.items():
                self._asset_data[asset] = {
                    "candles": candles, "price": price,
                    "bids": bids, "asks": asks, "trades": trades
                }

            # Primary BTC data (for backward compat)
            btc_data = self._asset_data.get("BTC", {})
            btc_price = btc_data.get("price", 0.0)
            bids = btc_data.get("bids", [])
            asks = btc_data.get("asks", [])
            trades = btc_data.get("trades", [])
            self._bids = bids
            self._asks = asks
            self._trades = trades

            # Rate limit before Polymarket
            await aio.sleep(1.5)

            # Fetch ALL 15m markets (BTC + ETH + SOL)
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
                        # Match any supported asset
                        matched_asset = None
                        for asset, cfg in self.ASSETS.items():
                            if any(kw in title for kw in cfg["keywords"]):
                                matched_asset = asset
                                break
                        if not matched_asset:
                            continue
                        for m in event.get("markets", []):
                            if not m.get("closed", True):
                                if not m.get("question"):
                                    m["question"] = event.get("title", "")
                                m["_asset"] = matched_asset  # Tag with asset name
                                markets.append(m)
                    if markets:
                        self._market_cache = markets
                else:
                    print(f"[API] Status {mr.status_code}, using cache")
                    markets = getattr(self, '_market_cache', [])
            except Exception as e:
                print(f"[API] Error: {e}, using cache")
                markets = getattr(self, '_market_cache', [])

        # Use BTC candles as primary (backward compat)
        candles = btc_data.get("candles", [])

        # Count by asset
        asset_counts = {}
        for m in markets:
            a = m.get("_asset", "?")
            asset_counts[a] = asset_counts.get(a, 0) + 1
        counts_str = " | ".join(f"{a}:{c}" for a, c in sorted(asset_counts.items()))
        if markets:
            print(f"[Markets] Found {len(markets)} 15m markets ({counts_str})")
        else:
            print("[Markets] No active 15m markets found")

        return candles, btc_price, markets, bids, asks, trades

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
    MIN_EDGE = 0.30              # V3.4: edge<0.30 = 36% WR (96 paper trades)
    MIN_KL_DIVERGENCE = 0.15     # V3.4: KL<0.15 = 36% WR vs 67% above (96 paper trades)
    MIN_TIME_REMAINING = 5.0     # V3.4: 2-5min = 47% WR; 5-12min = 83% WR (96 paper trades)
    MAX_ENTRY_PRICE = 0.45       # V3.6: $0.45-0.55 = 50% WR coin flip, cut it
    MIN_ENTRY_PRICE = 0.15       # V3.6: <$0.15 = 16.7% WR, -$40 loss — hard floor
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

    def _scan_arbitrage(self, eligible_markets):
        """Scan for both-sides arbitrage: UP + DOWN < threshold for guaranteed profit."""
        if not self.ARB_ENABLED:
            return

        now = datetime.now(timezone.utc)

        for time_left, market, up_price, down_price in eligible_markets:
            market_id = market.get("conditionId", "")
            question = market.get("question", "")
            market_numeric_id = market.get("id")
            asset = market.get("_asset", "BTC")

            # Skip if we already have an arb trade on this market
            arb_key = f"arb_{market_id}"
            if arb_key in self.arb_trades:
                continue

            # Skip if we have a directional trade on this market
            if any(market_id in tid for tid in self.trades):
                continue

            # Skip if already traded this cycle
            if market_id in self.traded_markets_this_cycle:
                continue

            # Time filter
            if time_left < self.ARB_MIN_TIME or time_left > self.ARB_MAX_TIME:
                continue

            # Core arb logic: check if UP + DOWN < threshold
            spread = up_price + down_price

            if spread < self.ARB_THRESHOLD:
                guaranteed_profit_pct = (1.0 - spread - self.ARB_FEE_RATE) / spread
                half_size = self.ARB_POSITION_SIZE / 2

                # Calculate shares for each side
                up_shares = half_size / up_price
                down_shares = half_size / down_price

                arb_trade = {
                    "trade_id": arb_key,
                    "market_title": question[:80],
                    "asset": asset,
                    "up_entry": up_price,
                    "down_entry": down_price,
                    "combined_cost": spread,
                    "guaranteed_profit_pct": guaranteed_profit_pct,
                    "size_usd": self.ARB_POSITION_SIZE,
                    "up_size": half_size,
                    "down_size": half_size,
                    "up_shares": up_shares,
                    "down_shares": down_shares,
                    "entry_time": now.isoformat(),
                    "status": "open",
                    "market_numeric_id": market_numeric_id,
                    "condition_id": market_id,
                    "pnl": 0.0,
                }
                self.arb_trades[arb_key] = arb_trade
                self.traded_markets_this_cycle.add(market_id)

                print(f"[ARB] {asset} UP:${up_price:.2f} + DOWN:${down_price:.2f} = ${spread:.3f} | "
                      f"Profit: {guaranteed_profit_pct:.1%} | Size: ${self.ARB_POSITION_SIZE} | "
                      f"T-{time_left:.1f}min | {question[:40]}...")

    def _resolve_arb_trade(self, arb, up_p, down_p):
        """Resolve an arb trade given final UP/DOWN prices."""
        if arb["status"] != "open":
            return

        now = datetime.now(timezone.utc)

        if up_p >= 0.95:
            # UP won - UP shares pay out at $1.00 each
            payout = arb["up_shares"] * 1.0
        elif down_p >= 0.95:
            # DOWN won - DOWN shares pay out at $1.00 each
            payout = arb["down_shares"] * 1.0
        else:
            # Not yet resolved (price in between) - mark to market
            payout = arb["up_shares"] * up_p + arb["down_shares"] * down_p

        fee = payout * self.ARB_FEE_RATE
        net_pnl = payout - fee - arb["size_usd"]

        arb["pnl"] = net_pnl
        arb["exit_time"] = now.isoformat()
        arb["exit_up_price"] = up_p
        arb["exit_down_price"] = down_p
        arb["status"] = "closed"

        self.arb_stats["total_pnl"] += net_pnl
        if net_pnl > 0:
            self.arb_stats["wins"] += 1
        else:
            self.arb_stats["losses"] += 1

        result = "WIN" if net_pnl > 0 else "LOSS"
        winner = "UP" if up_p >= 0.95 else "DOWN"
        print(f"[ARB {result}] {winner} won | PnL: ${net_pnl:+.2f} | "
              f"Paid: ${arb['combined_cost']:.3f} | Payout: ${payout:.2f} | "
              f"{arb['market_title'][:40]}...")

    async def run_cycle(self):
        """Run one trading cycle."""
        self._reset_daily_if_needed()
        self.traded_markets_this_cycle.clear()  # Fresh cycle

        # Check daily loss limit
        if self.daily_pnl <= -self.MAX_DAILY_LOSS:
            print(f"[RISK] Daily loss limit hit: ${self.daily_pnl:.2f} - resting")
            return None

        # Check skip hours — shadow-trade instead of resting
        current_hour = datetime.now(timezone.utc).hour
        is_skip_hour = current_hour in self.SKIP_HOURS_UTC
        if is_skip_hour:
            print(f"[SKIP] Hour {current_hour} UTC is in skip list - shadow tracking only")

        candles, btc_price, markets, bids, asks, trades = await self.fetch_data()

        if not candles:
            return None

        # IMPORTANT: Only use the most recent 15 minutes of price action
        recent_candles = candles[-self.CANDLE_LOOKBACK:]

        # Calculate momentum for directional filter (BTC primary)
        momentum = self._get_price_momentum(candles, lookback=5)

        # Generate main signal from BTC price action + orderbook flow
        main_signal = self.generator.generate_signal(
            market_id="btc_main",
            candles=recent_candles,
            current_price=btc_price,
            market_yes_price=0.5,
            market_no_price=0.5,
            time_remaining_min=10.0,
            bids=bids,
            asks=asks,
            trades=trades
        )

        self.signals_count += 1

        # Sort markets by time remaining - trade nearest per asset
        eligible_markets = []
        for market in markets:
            up_price, down_price = self.get_market_prices(market)
            time_left = self.get_time_remaining(market)
            if up_price is None or down_price is None:
                continue
            if time_left > 15 or time_left < self.MIN_TIME_REMAINING:
                continue
            eligible_markets.append((time_left, market, up_price, down_price))

        # Sort by time remaining (ascending) - nearest expiring first
        eligible_markets.sort(key=lambda x: x[0])

        if eligible_markets:
            print(f"[Filter] {len(eligible_markets)} markets in window, trading nearest ({eligible_markets[0][0]:.1f}min left)")

        # V3.2: Trade the nearest expiring market PER ASSET (BTC + ETH + SOL)
        # Pick nearest market for each asset to maximize turnover across all assets
        nearest_per_asset = {}
        for time_left, market, up_price, down_price in eligible_markets:
            asset = market.get("_asset", "BTC")
            if asset not in nearest_per_asset:
                nearest_per_asset[asset] = (time_left, market, up_price, down_price)

        # === ARBITRAGE SCAN ===
        # Check ALL eligible markets for both-sides arb (UP + DOWN < $0.97)
        self._scan_arbitrage(eligible_markets)

        # === DIRECTIONAL TRADING (Phase 1: Collect candidates) ===
        # Gather all trades that pass filters, then allocate via multi-outcome Kelly
        trade_candidates = []
        for asset, (time_left, market, up_price, down_price) in nearest_per_asset.items():
            market_id = market.get("conditionId", "")
            market_numeric_id = market.get("id")
            question = market.get("question", "")

            # Use asset-specific candle data for signal generation
            asset_data = getattr(self, '_asset_data', {}).get(asset, {})
            asset_candles = asset_data.get("candles", candles)
            asset_price = asset_data.get("price", btc_price)
            asset_bids = asset_data.get("bids", bids)
            asset_asks = asset_data.get("asks", asks)
            asset_trades = asset_data.get("trades", trades)
            asset_recent = asset_candles[-self.CANDLE_LOOKBACK:] if asset_candles else recent_candles

            # Recalculate momentum for this specific asset
            momentum = self._get_price_momentum(asset_candles, lookback=5) if asset_candles else momentum

            # Generate signal from asset-specific price action
            signal = self.generator.generate_signal(
                market_id=market_id,
                candles=asset_recent,
                current_price=asset_price,
                market_yes_price=up_price,
                market_no_price=down_price,
                time_remaining_min=time_left,
                bids=asset_bids,
                asks=asset_asks,
                trades=asset_trades
            )

            trade_key = f"{market_id}_{signal.side}" if signal.side else None

            # === DUPLICATE PROTECTION ===
            # One trade per market per cycle - NO exceptions
            if market_id in self.traded_markets_this_cycle:
                continue

            # === WINNING FORMULA V3.2 FILTERS ===
            # Data-driven from 158 trades analyzed:
            #   DOWN <$0.35 = 100% WR (+$537)     = BEST
            #   DOWN $0.35-0.40 = 60% WR (+$328)  = GOOD
            #   DOWN $0.40-0.45 = 14% WR (-$321)  = DEATH ZONE - SKIP!
            #   DOWN $0.45-0.50 = 57% WR (+$609)  = DECENT
            #   DOWN $0.50-0.55 = 73% WR (+$793)  = GREAT
            skip_reason = None

            # 1. ASYMMETRIC CONFIDENCE with PRICE + CONFIDENCE + TREND AWARENESS
            if signal.side == "UP":
                # TREND FILTER: Don't take ultra-cheap UP (<$0.15) during downtrends
                if up_price < 0.15 and hasattr(signal, 'regime') and signal.regime.value == 'trend_down':
                    skip_reason = f"UP_cheap_contrarian_{up_price:.2f}_in_downtrend"
                else:
                    up_conf_req = self.UP_MIN_CONFIDENCE
                    if up_price < 0.25:
                        up_conf_req = max(0.45, up_conf_req - 0.15)
                    elif up_price < 0.35:
                        up_conf_req = max(0.50, up_conf_req - 0.10)

                    # Dynamic UP max price: high model confidence unlocks higher prices
                    # Base: UP_MAX_PRICE. At 80%+ conf → $0.48, 85%+ → $0.55
                    effective_up_max = self.UP_MAX_PRICE
                    if signal.model_up >= 0.85:
                        effective_up_max = max(effective_up_max, 0.55)
                    elif signal.model_up >= 0.80:
                        effective_up_max = max(effective_up_max, 0.48)
                    elif signal.model_up >= 0.70:
                        effective_up_max = max(effective_up_max, 0.42)

                    if signal.model_up < up_conf_req:
                        skip_reason = f"UP_conf_{signal.model_up:.0%}<{up_conf_req:.0%}"
                    elif up_price > effective_up_max:
                        skip_reason = f"UP_price_{up_price:.2f}>{effective_up_max:.2f}"
                    elif momentum < -0.001:
                        skip_reason = f"UP_momentum_negative_{momentum:.3%}"
            elif signal.side == "DOWN":
                # DEATH ZONE: $0.40-0.45 = 14% WR, -$321 PnL. NEVER trade here.
                if 0.40 <= down_price < 0.45:
                    skip_reason = f"DOWN_DEATH_ZONE_{down_price:.2f}_(14%WR)"
                # V3.4: Block ALL DOWN < $0.15 — 0% WR in 96 paper trades (3 trades, 3 total losses)
                elif down_price < 0.15:
                    skip_reason = f"DOWN_ultra_cheap_{down_price:.2f}_(0%WR)"
                else:
                    down_conf_req = self.DOWN_MIN_CONFIDENCE
                    # Break-even aware confidence: at price P, need P prob to break even
                    # Use 2.5x break-even for cheap entries, scaling down for expensive
                    if down_price < 0.10:
                        down_conf_req = max(down_price * 2.5, 0.08)  # 10:1+ payoff
                    elif down_price < 0.20:
                        down_conf_req = max(down_price * 2.0, 0.15)  # 5:1+ payoff
                    elif down_price < 0.30:
                        down_conf_req = max(0.40, down_conf_req - 0.15)  # 3.3:1 payoff
                    elif down_price < 0.40:
                        down_conf_req = max(0.45, down_conf_req - 0.10)  # 2.5:1 payoff

                    if signal.model_down < down_conf_req:
                        skip_reason = f"DOWN_conf_{signal.model_down:.0%}<{down_conf_req:.0%}"
                    elif down_price > self.DOWN_MAX_PRICE:
                        skip_reason = f"DOWN_price_{down_price:.2f}>{self.DOWN_MAX_PRICE}"
                    # Momentum confirmation only for non-cheap entries
                    elif down_price >= 0.30 and momentum > self.DOWN_MIN_MOMENTUM_DROP:
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

            # 3. NYU TWO-PARAMETER VOLATILITY MODEL
            # Key insight from NYU research: Only price + time matter
            # Extreme prices (away from 0.50) have LOWER volatility = better edge
            if self.use_nyu_model:
                entry_price_check = up_price if signal.side == "UP" else down_price
                nyu_result = self.nyu_model.calculate_volatility(entry_price_check, time_left)

                # Adaptive NYU threshold: high vol = more opportunity, relax gate (V3.6)
                nyu_threshold = {"low": 0.15, "medium": 0.10, "high": 0.05}.get(nyu_result.volatility_regime, 0.15)
                if nyu_result.edge_score < nyu_threshold:
                    skip_reason = f"NYU_edge_{nyu_result.edge_score:.2f}<{nyu_threshold} (vol={nyu_result.volatility_regime})"
                    print(f"[NYU] {question[:30]}... | {skip_reason}")
                    continue

                # Bonus: If NYU says TRADE, boost confidence in this signal
                if nyu_result.recommended_action == "TRADE":
                    print(f"[NYU] TRADE signal: edge={nyu_result.edge_score:.2f}, vol={nyu_result.instantaneous_volatility:.3f}")

            # Debug: show why no trade
            if signal.action != "ENTER":
                edge = max(signal.edge_up or 0, signal.edge_down or 0)
                print(f"[Skip] {question[:35]}... | UP:{up_price:.2f} DOWN:{down_price:.2f} | Edge:{edge:.1%} | {signal.reason}")

            # Collect candidate if signal passes all filters
            if signal.action == "ENTER" and signal.side and trade_key:
                if trade_key not in self.trades:
                    entry_price = up_price if signal.side == "UP" else down_price
                    edge = signal.edge_up if signal.side == "UP" else signal.edge_down

                    # Bregman divergence optimization + Frank-Wolfe profit guarantee
                    bregman_signal = self.bregman.calculate_optimal_trade(
                        model_prob=signal.model_up,
                        market_yes_price=up_price,
                        market_no_price=down_price,
                        time_remaining_min=time_left,
                    )
                    self.bregman_signals += 1

                    # FW profit guarantee: skip if execution costs eat too much edge
                    if not bregman_signal.fw_executable:
                        print(f"[FW] {question[:30]}... | profit_ratio={bregman_signal.profit_ratio:.0%}<{self.bregman.ALPHA_THRESHOLD:.0%} gap={bregman_signal.fw_gap:.4f}")
                        continue

                    # === KL DIVERGENCE FILTER (V3.4) ===
                    # KL < 0.15 = 36% WR (model agrees with market = no edge)
                    if bregman_signal.kl_divergence < self.MIN_KL_DIVERGENCE:
                        print(f"[KL] {question[:30]}... | KL={bregman_signal.kl_divergence:.3f}<{self.MIN_KL_DIVERGENCE} (no divergence edge)")
                        continue

                    # === ML V3 PREDICTION (kept for model_agreement meta-feature) ===
                    ml_prediction = None
                    ml_str = ""
                    if self.use_ml_v3:
                        # Build order flow dict for ML features
                        order_flow_dict = None
                        if signal.order_flow or signal.squeeze:
                            order_flow_dict = {
                                'obi': signal.order_flow.obi if signal.order_flow else 0.0,
                                'cvd_5m': signal.order_flow.cvd_5m if signal.order_flow else 0.0,
                                'squeeze_on': signal.squeeze.squeeze_on if signal.squeeze else False,
                                'squeeze_momentum': signal.squeeze.momentum if signal.squeeze else 0.0,
                                'ema_cross': signal.ema_cross or 'NEUTRAL'
                            }

                        # V3.2: Pass MACD + volume data for new ML features
                        macd_hist = 0.0
                        macd_sig = 0.0
                        vol_ratio = 1.0
                        if signal.macd:
                            macd_hist = signal.macd.histogram_delta if hasattr(signal.macd, 'histogram_delta') else 0.0
                            macd_sig = signal.macd.signal if hasattr(signal.macd, 'signal') else 0.0
                        if hasattr(signal, 'volume_recent') and hasattr(signal, 'volume_avg'):
                            vol_ratio = signal.volume_recent / signal.volume_avg if signal.volume_avg and signal.volume_avg > 0 else 1.0

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
                                'trend_strength': momentum,
                                'macd_histogram': macd_hist,
                                'macd_signal_line': macd_sig,
                                'volume_ratio': vol_ratio,
                            },
                            order_flow=order_flow_dict
                        )
                        ml_prediction = self.ml_engine.predict(ml_features)

                        # Store features for primary ML training later
                        self._pending_ml_features = ml_features

                        ml_str = f"[ML:{ml_prediction.recommended_side} conf={ml_prediction.confidence:.0%}]"

                    # === META-LABELING: Should we take this trade? ===
                    # Instead of binary ML veto, use calibrated P(correct) for filtering + sizing
                    nyu_result_for_meta = None
                    if self.use_nyu_model:
                        nyu_result_for_meta = self.nyu_model.calculate_volatility(entry_price, time_left)

                    meta_features = self.meta_labeler.extract_meta_features(
                        signal=signal,
                        bregman_signal=bregman_signal,
                        ml_prediction=ml_prediction,
                        nyu_result=nyu_result_for_meta,
                        candles=asset_recent,
                        entry_price=entry_price,
                        side=signal.side,
                        momentum=momentum,
                    )
                    meta_pred = self.meta_labeler.predict(meta_features)

                    if self.use_meta_labeler and meta_pred.recommended_action == "SKIP":
                        print(f"[META] P(correct)={meta_pred.p_correct:.0%} {meta_pred.confidence_tier} SKIP | {ml_str} | {question[:30]}...")
                        continue

                    # Individual meta-label size (before multi-Kelly adjustment)
                    individual_size = self.meta_labeler.calculate_position_size(
                        p_correct=meta_pred.p_correct,
                        base_size=self.POSITION_SIZE,
                        kelly_fraction=bregman_signal.kelly_fraction,
                    )
                    # === HOURLY ML MULTIPLIER ===
                    hour_mult = self._get_hour_multiplier(datetime.now(timezone.utc).hour)
                    if hour_mult != 1.0:
                        individual_size *= hour_mult
                    individual_size = max(5.0, individual_size)

                    # Collect candidate for multi-Kelly allocation
                    trade_candidates.append({
                        'asset': asset, 'trade_key': trade_key, 'signal': signal,
                        'entry_price': entry_price, 'edge': edge,
                        'individual_size': individual_size,
                        'bregman_signal': bregman_signal,
                        'meta_pred': meta_pred, 'meta_features': meta_features,
                        'ml_features': ml_features if ml_prediction else None,
                        'ml_str': ml_str, 'momentum': momentum,
                        'question': question, 'market_id': market_id,
                        'market_numeric_id': market_numeric_id,
                        'time_left': time_left, 'up_price': up_price,
                        'down_price': down_price,
                    })

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
                        won = trade.pnl > 0
                        if won:
                            self.wins += 1
                        else:
                            self.losses += 1

                        # === UPDATE HOURLY STATS ===
                        try:
                            entry_hour = datetime.fromisoformat(trade.entry_time).hour
                            if entry_hour not in self.hourly_stats:
                                self.hourly_stats[entry_hour] = {"wins": 0, "losses": 0, "pnl": 0.0}
                            self.hourly_stats[entry_hour]["wins" if won else "losses"] += 1
                            self.hourly_stats[entry_hour]["pnl"] += trade.pnl
                        except Exception:
                            pass

                        hour_mult = self._get_hour_multiplier(datetime.now(timezone.utc).hour)
                        result = "WIN" if won else "LOSS"
                        print(f"[{result}] {trade.side} PnL: ${trade.pnl:+.2f} | Day: ${self.daily_pnl:+.2f} | HrMult: {hour_mult}x | {trade.market_title[:35]}...")

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

                        # === META-LABEL ONLINE LEARNING ===
                        if self.use_meta_labeler and hasattr(self, '_pending_meta_features'):
                            try:
                                self.meta_labeler.add_sample(
                                    meta_features=self._pending_meta_features,
                                    won=(trade.pnl > 0),
                                    pnl=trade.pnl,
                                    side=getattr(self, '_pending_meta_side', trade.side),
                                )
                            except Exception as e:
                                print(f"[META] Training update error: {e}")

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
                        won2 = trade.pnl > 0
                        if won2:
                            self.wins += 1
                        else:
                            self.losses += 1
                        try:
                            eh = datetime.fromisoformat(trade.entry_time).hour
                            if eh not in self.hourly_stats:
                                self.hourly_stats[eh] = {"wins": 0, "losses": 0, "pnl": 0.0}
                            self.hourly_stats[eh]["wins" if won2 else "losses"] += 1
                            self.hourly_stats[eh]["pnl"] += trade.pnl
                        except Exception:
                            pass
                        result = "WIN" if won2 else "LOSS"
                        print(f"[{result}] {trade.side} PnL: ${trade.pnl:+.2f} | Day: ${self.daily_pnl:+.2f} | {trade.market_title[:35]}...")

                # Also resolve arb trades on this market
                for arb_key, arb in list(self.arb_trades.items()):
                    if arb["status"] == "open" and arb.get("condition_id") == mkt_id:
                        self._resolve_arb_trade(arb, up_p, down_p)

        # === ARB EXPIRY RESOLUTION ===
        # Check open arb trades older than 16 minutes via Gamma API
        now = datetime.now(timezone.utc)
        for arb_key, arb in list(self.arb_trades.items()):
            if arb["status"] != "open":
                continue
            try:
                entry_dt = datetime.fromisoformat(arb["entry_time"])
            except Exception:
                continue
            age_minutes = (now - entry_dt).total_seconds() / 60
            if age_minutes > 16:
                if not arb.get("market_numeric_id"):
                    arb["status"] = "closed"
                    arb["pnl"] = 0.0
                    arb["exit_time"] = now.isoformat()
                    print(f"[ARB STALE] No numeric ID | {arb['market_title'][:40]}...")
                    continue
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        r = await client.get(
                            f"https://gamma-api.polymarket.com/markets/{arb['market_numeric_id']}",
                            headers={"User-Agent": "Mozilla/5.0"}
                        )
                        if r.status_code == 200:
                            rm = r.json()
                            up_p, down_p = self.get_market_prices(rm)
                            if up_p is not None and down_p is not None:
                                self._resolve_arb_trade(arb, up_p, down_p)
                    await asyncio.sleep(0.5)
                except Exception as e:
                    print(f"[ARB Expiry] Error resolving {arb_key[:20]}: {e}")

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
                                won3 = open_trade.pnl > 0
                                if won3:
                                    self.wins += 1
                                else:
                                    self.losses += 1
                                try:
                                    eh3 = datetime.fromisoformat(open_trade.entry_time).hour
                                    if eh3 not in self.hourly_stats:
                                        self.hourly_stats[eh3] = {"wins": 0, "losses": 0, "pnl": 0.0}
                                    self.hourly_stats[eh3]["wins" if won3 else "losses"] += 1
                                    self.hourly_stats[eh3]["pnl"] += open_trade.pnl
                                except Exception:
                                    pass
                                result = "WIN" if won3 else "LOSS"
                                print(f"[{result}] (expired) {open_trade.side} PnL: ${open_trade.pnl:+.2f} | Day: ${self.daily_pnl:+.2f} | {open_trade.market_title[:30]}...")
                    await asyncio.sleep(0.5)  # Rate limit between resolved market lookups
                except Exception as e:
                    print(f"[Expiry] Error resolving {tid[:20]}: {e}")

        # === SKIP HOUR SHADOW RESOLUTION ===
        # Resolve open shadow trades from skip hours (runs every cycle)
        for skey, shadow in list(self.skip_hour_shadows.items()):
            if shadow.get("status") != "open":
                continue
            try:
                entry_dt = datetime.fromisoformat(shadow["entry_time"])
            except Exception:
                continue
            age_minutes = (now - entry_dt).total_seconds() / 60
            if age_minutes < 16:
                # Try resolving via current markets
                for mkt in markets:
                    mkt_id = mkt.get("conditionId", "")
                    if mkt_id != shadow.get("market_id"):
                        continue
                    mkt_time = self.get_time_remaining(mkt)
                    if mkt_time < 0.5:
                        up_p, down_p = self.get_market_prices(mkt)
                        if up_p is None or down_p is None:
                            continue
                        price = up_p if shadow["side"] == "UP" else down_p
                        if price >= 0.95:
                            exit_val = shadow["size_usd"] / shadow["entry_price"]
                        elif price <= 0.05:
                            exit_val = 0
                        else:
                            exit_val = (shadow["size_usd"] / shadow["entry_price"]) * price
                        shadow["exit_price"] = price
                        shadow["exit_time"] = now.isoformat()
                        shadow["pnl"] = exit_val - shadow["size_usd"]
                        shadow["status"] = "closed"
                        swon = shadow["pnl"] > 0
                        sh = shadow.get("entry_hour", 0)
                        if sh not in self.skip_hour_stats:
                            self.skip_hour_stats[sh] = {"wins": 0, "losses": 0, "pnl": 0.0}
                        self.skip_hour_stats[sh]["wins" if swon else "losses"] += 1
                        self.skip_hour_stats[sh]["pnl"] += shadow["pnl"]
                        sr = "WIN" if swon else "LOSS"
                        print(f"[SHADOW {sr}] UTC {sh} | {shadow['side']} ${shadow['pnl']:+.2f} | {shadow.get('title', '')[:40]}")
            else:
                # Resolve via Gamma API for stale shadows
                nid = shadow.get("market_numeric_id")
                if not nid:
                    shadow["status"] = "closed"
                    shadow["pnl"] = 0.0
                    shadow["exit_time"] = now.isoformat()
                    continue
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        r = await client.get(
                            f"https://gamma-api.polymarket.com/markets/{nid}",
                            headers={"User-Agent": "Mozilla/5.0"}
                        )
                        if r.status_code == 200:
                            rm = r.json()
                            up_p, down_p = self.get_market_prices(rm)
                            if up_p is not None and down_p is not None:
                                price = up_p if shadow["side"] == "UP" else down_p
                                if price >= 0.95:
                                    exit_val = shadow["size_usd"] / shadow["entry_price"]
                                elif price <= 0.05:
                                    exit_val = 0
                                else:
                                    exit_val = (shadow["size_usd"] / shadow["entry_price"]) * price
                                shadow["exit_price"] = price
                                shadow["exit_time"] = now.isoformat()
                                shadow["pnl"] = exit_val - shadow["size_usd"]
                                shadow["status"] = "closed"
                                swon = shadow["pnl"] > 0
                                sh = shadow.get("entry_hour", 0)
                                if sh not in self.skip_hour_stats:
                                    self.skip_hour_stats[sh] = {"wins": 0, "losses": 0, "pnl": 0.0}
                                self.skip_hour_stats[sh]["wins" if swon else "losses"] += 1
                                self.skip_hour_stats[sh]["pnl"] += shadow["pnl"]
                                sr = "WIN" if swon else "LOSS"
                                print(f"[SHADOW {sr}] UTC {sh} | {shadow['side']} ${shadow['pnl']:+.2f} | {shadow.get('title', '')[:40]}")
                    await asyncio.sleep(0.5)
                except Exception:
                    pass

        # === DIRECTIONAL TRADING (Phase 2: Multi-Outcome Kelly allocation) ===
        # Sort candidates by quality (highest P(correct) first = best signal gets most budget)
        # Then allocate with correlation discounts so correlated trades don't blow up exposure
        if trade_candidates and is_skip_hour:
            # SKIP HOUR: Record as shadow trades, don't touch real stats
            for cand in trade_candidates:
                skey = f"shadow_{cand['trade_key']}"
                if skey in self.skip_hour_shadows:
                    continue
                entry_price = cand['entry_price']
                self.skip_hour_shadows[skey] = {
                    "trade_key": skey,
                    "market_id": cand['market_id'],
                    "market_numeric_id": cand['market_numeric_id'],
                    "title": cand['question'][:80],
                    "asset": cand['asset'],
                    "side": cand['signal'].side,
                    "entry_price": entry_price,
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "entry_hour": current_hour,
                    "size_usd": cand['individual_size'],
                    "edge": cand['edge'] if cand['edge'] else 0,
                    "p_correct": cand['meta_pred'].p_correct,
                    "status": "open",
                    "exit_price": None,
                    "pnl": 0.0,
                }
                print(f"[SHADOW NEW] UTC {current_hour} | {cand['signal'].side} @ ${entry_price:.4f} | Edge: {cand['edge']:.1%} | P(correct): {cand['meta_pred'].p_correct:.0%} | {cand['question'][:40]}...")
            self._save()
            return main_signal

        if trade_candidates:
            trade_candidates.sort(key=lambda c: c['meta_pred'].p_correct, reverse=True)
            cycle_exposure = 0.0
            cycle_same_dir = {}  # {side: count}

            n_candidates = len(trade_candidates)
            sides = [c['signal'].side for c in trade_candidates]
            n_down = sum(1 for s in sides if s == "DOWN")
            n_up = sum(1 for s in sides if s == "UP")
            is_mixed = n_down > 0 and n_up > 0

            for cand in trade_candidates:
                side = cand['signal'].side
                n_same = cycle_same_dir.get(side, 0)

                # Correlation discount: each additional same-direction trade gets penalized
                # 1st trade: full size, 2nd: 54%, 3rd: 37% (for ρ=0.85)
                if n_same > 0:
                    discount = 1.0 / (1.0 + n_same * self.CRYPTO_CORRELATION)
                else:
                    discount = 1.0

                # Mixed directions are partially hedged - less discount
                if is_mixed:
                    discount = max(discount, 0.6)

                kelly_size = cand['individual_size'] * discount

                # Cap at remaining cycle budget
                remaining = self.MAX_CYCLE_BUDGET - cycle_exposure
                optimal_size = min(kelly_size, remaining)
                optimal_size = max(5.0, optimal_size)  # Floor $5

                if remaining < 5.0:
                    print(f"[KELLY] Cycle budget exhausted (${cycle_exposure:.0f}/${self.MAX_CYCLE_BUDGET:.0f}) | Skipping {cand['asset']} {side}")
                    continue

                # === EXECUTE TRADE ===
                meta_str = f"[META:P={cand['meta_pred'].p_correct:.0%} {cand['meta_pred'].confidence_tier} sz={cand['meta_pred'].position_size_mult:.1f}x]"
                kelly_str = f"[KELLY:{discount:.0%}of${cand['individual_size']:.0f}]" if discount < 1.0 else ""

                # Store meta-features for training on resolution
                self._pending_meta_features = cand['meta_features']
                self._pending_meta_side = side
                if cand['ml_features'] is not None:
                    self._pending_ml_features = cand['ml_features']

                trade = TAPaperTrade(
                    trade_id=cand['trade_key'],
                    market_title=cand['question'][:80],
                    side=side,
                    entry_price=cand['entry_price'],
                    entry_time=datetime.now(timezone.utc).isoformat(),
                    size_usd=optimal_size,
                    signal_strength=cand['signal'].strength.value,
                    edge_at_entry=cand['edge'] if cand['edge'] else 0,
                    kl_divergence=cand['bregman_signal'].kl_divergence,
                    kelly_fraction=cand['bregman_signal'].kelly_fraction,
                    guaranteed_profit=cand['bregman_signal'].guaranteed_profit,
                    market_numeric_id=cand['market_numeric_id'],
                )
                self.trades[cand['trade_key']] = trade
                self.traded_markets_this_cycle.add(cand['market_id'])

                momentum_str = f"Mom:{cand['momentum']:+.2%}" if cand['momentum'] else ""
                nyu_info = ""
                if self.use_nyu_model:
                    nyu_r = self.nyu_model.calculate_volatility(cand['entry_price'], cand['time_left'])
                    nyu_info = f"NYU:{nyu_r.edge_score:.2f}"

                print(f"[NEW] {side} @ ${cand['entry_price']:.4f} | Edge: {cand['edge']:.1%} | {momentum_str} | {nyu_info} | {meta_str} {kelly_str}")
                print(f"      Size: ${optimal_size:.2f} | KL: {cand['bregman_signal'].kl_divergence:.4f} | {cand['ml_str']} | T-{cand['time_left']:.1f}min | {cand['question'][:40]}...")

                cycle_exposure += optimal_size
                cycle_same_dir[side] = n_same + 1

            if len(trade_candidates) > 1:
                print(f"[KELLY] {len(trade_candidates)} candidates | Allocated ${cycle_exposure:.2f}/${self.MAX_CYCLE_BUDGET:.0f} budget | Sides: {dict(cycle_same_dir)}")

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

        # Meta-Labeler stats
        meta_status = self.meta_labeler.get_status()
        print(f"\nMETA-LABELER (Lopez de Prado):")
        print(f"  Trained: {meta_status['is_trained']} | Samples: {meta_status['training_samples']}")
        print(f"  Predictions: {meta_status['predictions_made']} | Avg P(correct): {meta_status['avg_p_correct']:.0%}")
        print(f"  Trade rate: {meta_status['trade_rate']:.0%} | Sizing: {meta_status['position_size_range']}")

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

        # Arbitrage Scanner stats
        arb_closed = [a for a in self.arb_trades.values() if a.get("status") == "closed"]
        arb_open = [a for a in self.arb_trades.values() if a.get("status") == "open"]
        print(f"\nARBITRAGE SCANNER:")
        print(f"  Arb trades: {len(self.arb_trades)} | Open: {len(arb_open)} | Closed: {len(arb_closed)}")
        print(f"  Arb W/L: {self.arb_stats['wins']}/{self.arb_stats['losses']}")
        print(f"  Arb PnL: ${self.arb_stats['total_pnl']:+.2f}")
        if arb_closed:
            avg_spread = sum(a["combined_cost"] for a in arb_closed) / len(arb_closed)
            print(f"  Avg spread captured: ${avg_spread:.3f}")

        # Hourly ML sizing
        total_hourly = sum(s["wins"] + s["losses"] for s in self.hourly_stats.values())
        phase = "BOOST+REDUCE" if total_hourly >= 10 else "REDUCE-ONLY"
        print(f"\nHOURLY ML SIZING ({phase}, {total_hourly} total trades):")
        active_h = []
        for h in range(24):
            s = self.hourly_stats[h]
            th = s["wins"] + s["losses"]
            if th > 0:
                wr_h = s["wins"] / th * 100
                mult = self._get_hour_multiplier(h)
                mst = (h - 7) % 24
                tag = ""
                if mult < 1.0:
                    tag = " REDUCED"
                elif mult > 1.0:
                    tag = " BOOSTED"
                active_h.append(f"    UTC {h:>2} (MST {mst:>2}h): {th}T {wr_h:.0f}% WR ${s['pnl']:+.1f} -> {mult}x{tag}")
        if active_h:
            for line in active_h:
                print(line)
        else:
            print("    No hourly data yet - all hours at 1.0x (collecting...)")
        cur_h = now.hour
        print(f"  Current: UTC {cur_h} -> {self._get_hour_multiplier(cur_h)}x multiplier")

        # Skip hour shadow tracking
        shadow_closed = [s for s in self.skip_hour_shadows.values() if s.get("status") == "closed"]
        shadow_open = [s for s in self.skip_hour_shadows.values() if s.get("status") == "open"]
        shadow_wins = sum(1 for s in shadow_closed if s.get("pnl", 0) > 0)
        shadow_losses = len(shadow_closed) - shadow_wins
        shadow_pnl = sum(s.get("pnl", 0) for s in shadow_closed)
        print(f"\nSKIP HOUR SHADOWS (NOT counted in real stats):")
        print(f"  Total: {len(shadow_closed)} closed, {len(shadow_open)} open")
        print(f"  W/L: {shadow_wins}/{shadow_losses} | WR: {shadow_wins / max(1, len(shadow_closed)) * 100:.0f}%")
        print(f"  Shadow PnL: ${shadow_pnl:+.2f}")
        skip_active = []
        for h in sorted(self.SKIP_HOURS_UTC):
            ss = self.skip_hour_stats.get(h, {"wins": 0, "losses": 0, "pnl": 0.0})
            st = ss["wins"] + ss["losses"]
            if st > 0:
                swr = ss["wins"] / st * 100
                mst = (h - 7) % 24
                verdict = "REOPEN?" if swr >= 55 and st >= 10 else "KEEP SKIP" if st >= 10 else "collecting..."
                skip_active.append(f"    UTC {h:>2} (MST {mst:>2}h): {st}T {swr:.0f}% WR ${ss['pnl']:+.1f} [{verdict}]")
        if skip_active:
            print("  Per-hour breakdown:")
            for line in skip_active:
                print(line)
        else:
            print("  No skip-hour shadow data yet — collecting...")

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
        print("TA + BREGMAN PAPER TRADER V3.4 - META-LABELING UPGRADE")
        print("=" * 70)
        print("Strategy: Momentum + ML ensemble + Meta-Label filter/sizing")
        print("ML V3 Engine: LightGBM + XGBoost (online learning enabled)")
        ml_status = self.ml_engine.get_status()
        print(f"  Training samples: {ml_status['training_samples']} | Trained: {ml_status['is_trained']}")
        meta_status = self.meta_labeler.get_status()
        print(f"Meta-Labeler: {'TRAINED' if meta_status['is_trained'] else 'HEURISTIC'} | {meta_status['training_samples']} samples")
        print(f"  P(correct) threshold: {self.meta_labeler.TRADE_THRESHOLD:.0%} | Position sizing: 0.5x-2.0x")
        print("ML Auto-Tuner: ENABLED (tunes every 2h based on performance)")
        print(f"Tune cycle: #{self.ml_tuner.state.tune_count}")
        print()
        print("CURRENT ML-TUNED PARAMETERS:")
        print(f"  UP:   Max price ${self.UP_MAX_PRICE:.2f} | Min conf {self.UP_MIN_CONFIDENCE:.0%} | Need rising momentum")
        print(f"  DOWN: Max price ${self.DOWN_MAX_PRICE:.2f} | Min conf {self.DOWN_MIN_CONFIDENCE:.0%} | Need falling momentum")
        print(f"  Min edge: {self.MIN_EDGE:.0%} | Min KL: {self.MIN_KL_DIVERGENCE} | Entry window: {self.MIN_TIME_REMAINING}-15 min")
        print(f"  V3.4: Death zone $0.40-0.45 | DOWN<$0.15=BLOCK | KL filter | {self.MIN_TIME_REMAINING}min entry floor")
        print()
        print(f"  Skip hours: {sorted(self.SKIP_HOURS_UTC)} UTC")
        print(f"  Daily loss limit: ${self.MAX_DAILY_LOSS}")
        print(f"Bankroll: ${self.bankroll} | Max Position: ${self.POSITION_SIZE}")
        print()
        arb_status = "ENABLED" if self.ARB_ENABLED else "DISABLED"
        print(f"ARBITRAGE SCANNER: {arb_status}")
        print(f"  Threshold: ${self.ARB_THRESHOLD} | Fee: {self.ARB_FEE_RATE:.0%} | Size: ${self.ARB_POSITION_SIZE}")
        print(f"  Arb history: {self.arb_stats['wins']}W/{self.arb_stats['losses']}L | PnL: ${self.arb_stats['total_pnl']:+.2f}")
        print()
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
                    # Show all asset prices
                    asset_data = getattr(self, '_asset_data', {})
                    prices_str = " | ".join(f"{a} ${d.get('price', 0):,.2f}" for a, d in sorted(asset_data.items()) if d.get('price', 0) > 0)
                    print(f"[Scan {cycle}] {prices_str} | {signal.regime.value} | Model: {signal.model_up:.0%} UP")

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
