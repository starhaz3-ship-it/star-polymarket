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
from datetime import datetime, timezone, timedelta
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

# Hydra strategy quarantine system (V3.14)
try:
    from hydra_strategies import scan_strategies, get_strategy_consensus, StrategySignal
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

# BTC ML Predictor for 5m and 15m (V3.19)
try:
    from btc_ml_predictor import BTCPredictor, MLPrediction as BTC_MLPrediction
    BTC_ML_AVAILABLE = True
except ImportError:
    BTC_ML_AVAILABLE = False
    print("[HYDRA] hydra_strategies.py not found - strategy quarantine disabled")


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

    # === Systematic Trading Features (V3.5) ===
    # Feature 1: BTC Overnight Seasonality (22-23 UTC = positive bias)
    overnight_btc_boost: float = 0.05  # Confidence boost for BTC UP during 22-23 UTC
    overnight_enabled: bool = True

    # Feature 2: Inverse Volatility Sizing (baseline_atr / current_atr)
    invvol_enabled: bool = True
    invvol_clamp_low: float = 0.5   # Min size multiplier
    invvol_clamp_high: float = 2.0  # Max size multiplier

    # Feature 3: Multi-Timeframe Confirmation
    mtf_enabled: bool = True
    mtf_short_window: int = 30      # Short TF: last 30 candles (30 min)
    mtf_long_window: int = 120      # Long TF: last 120 candles (2 hours)
    mtf_penalty: float = 0.5        # Size multiplier when TFs disagree

    # Feature 4: Volume Spike Filter
    volspike_enabled: bool = True
    volspike_threshold: float = 1.5  # Volume must be > 1.5x average for MR boost
    volspike_boost: float = 1.2      # Size boost when volume spike confirms signal
    volspike_low_penalty: float = 0.7  # Size penalty when volume is very low (<0.5x avg)

    # Feature 5: Volatility Regime Routing
    volregime_enabled: bool = True
    volregime_high_mr_boost: float = 0.05   # Confidence boost for MR in high vol
    volregime_low_trend_boost: float = 0.03  # Confidence boost for trend in low vol
    volregime_mismatch_penalty: float = 0.5  # Size mult when strategy mismatches regime

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
            # Systematic Trading Features (V3.5)
            "overnight_btc_boost": self.state.overnight_btc_boost,
            "overnight_enabled": self.state.overnight_enabled,
            "invvol_enabled": self.state.invvol_enabled,
            "invvol_clamp_low": self.state.invvol_clamp_low,
            "invvol_clamp_high": self.state.invvol_clamp_high,
            "mtf_enabled": self.state.mtf_enabled,
            "mtf_penalty": self.state.mtf_penalty,
            "volspike_enabled": self.state.volspike_enabled,
            "volspike_threshold": self.state.volspike_threshold,
            "volspike_boost": self.state.volspike_boost,
            "volregime_enabled": self.state.volregime_enabled,
            "volregime_high_mr_boost": self.state.volregime_high_mr_boost,
            "volregime_low_trend_boost": self.state.volregime_low_trend_boost,
            "volregime_mismatch_penalty": self.state.volregime_mismatch_penalty,
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

    # === V3.17 CSV-PROVEN FILTERS (2026-02-13) ===
    # CSV: 376 trades, 40.2% WR, -$214.20. V3.16 whale relaxations HURT us.
    # ONLY profitable single-side bucket: $0.50-0.60 (55% WR, +$10)
    # $0.40-0.50 = -$183 (-30% ROI). <$0.20 = -$101 (-73% ROI). UP = 32% WR.
    # ETH = 33% WR (-$172). Both-sides = 90% WR (+$53, only consistent winner).
    UP_MAX_PRICE = 0.38           # V3.17: TIGHTENED — CSV UP@$0.40-0.50 = coin flip
    UP_MIN_CONFIDENCE = 0.72      # V3.17: TIGHTENED — CSV UP = 32% WR overall
    DOWN_MAX_PRICE = 0.38         # V3.17: TIGHTENED — CSV $0.40-0.50 = -30% ROI death zone
    DOWN_MIN_CONFIDENCE = 0.62    # V3.17: TIGHTENED — CSV DOWN = 39% WR overall
    DOWN_MIN_MOMENTUM_DROP = -0.003  # V3.17: TIGHTENED — need STRONG falling price

    # Risk management
    MAX_DAILY_LOSS = 30.0         # Stop after losing $30 in a day
    # V3.2: Data-driven skip hours from 158 trade analysis
    # REMOVED from skip (profitable): 6(56%WR +$108), 8(67%WR +$218), 10(67%WR +$74), 14(60%WR +$414)
    # ADDED to skip (losing): 1(40%WR -$21), 3(45%WR -$128), 16(25%WR -$236), 17(33%WR -$133), 22(33%WR -$23)
    # Best hours: 2(78%WR), 4(56%WR), 13(62%WR), 18(57%WR), 21(100%WR)
    # V3.14: CSV 326 trades — hours 5,6,8,9,10,12,14,16 = 14.2% WR, -$499
    SKIP_HOURS_UTC = {5, 6, 8, 9, 10, 12, 14, 16}

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

        # === SYSTEMATIC TRADING FEATURES V3.5 ===
        # Feature tracking for ML auto-revoke
        self.systematic_stats = {
            "overnight": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            "invvol": {"total_scale": 0.0, "trades_scaled": 0},
            "mtf_agree": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            "mtf_disagree": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            "volspike": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            "volspike_normal": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            "volregime_match": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            "volregime_mismatch": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
        }
        # Per-trade feature tags (stored on trade for resolution tracking)
        self._trade_features: Dict[str, dict] = {}  # {trade_key: {overnight, mtf_agree, volspike, ...}}

        # Feature config (ML-tunable)
        self.OVERNIGHT_HOURS = {22, 23}  # UTC hours with BTC UP seasonality
        self.OVERNIGHT_BTC_BOOST = 0.05
        self.OVERNIGHT_ENABLED = True
        self.INVVOL_ENABLED = True
        self.INVVOL_CLAMP = (0.5, 2.0)
        self.MTF_ENABLED = True
        self.MTF_PENALTY = 0.5
        self.VOLSPIKE_ENABLED = True
        self.VOLSPIKE_THRESHOLD = 1.5
        self.VOLSPIKE_BOOST = 1.2
        self.VOLSPIKE_LOW_PENALTY = 0.7
        self.VOLREGIME_ENABLED = True
        self.VOLREGIME_HIGH_MR_BOOST = 0.05
        self.VOLREGIME_LOW_TREND_BOOST = 0.03
        self.VOLREGIME_MISMATCH_PENALTY = 0.5

        # === HYDRA STRATEGY QUARANTINE (V3.14) ===
        # Shadow-tracks 10 Hydra strategies adapted from Hyperliquid
        # Each strategy makes directional predictions that are tracked but NOT traded
        # After 20+ predictions, evaluate WR to decide promotion to live scoring
        self.HYDRA_STRATEGIES = [
            "TRENDLINE_BREAK", "MFI_DIVERGENCE", "CONNORS_RSI", "MULTI_SMA_TREND",
            "ALIGNMENT", "DC03_KALMAN_ADX", "CONSEC_CANDLE_REVERSAL",
            "SHORT_TERM_REVERSAL", "FVG_RETEST",
            "BB_BOUNCE", "EXTREME_MOMENTUM", "RUBBER_BAND",
        ]
        # V3.14d: FILTERED stats (what we act on — with RSI veto, trend-beats-reversion, etc.)
        self.hydra_quarantine: Dict[str, dict] = {
            name: {"predictions": 0, "correct": 0, "wrong": 0, "pnl": 0.0, "status": "QUARANTINE"}
            for name in self.HYDRA_STRATEGIES
        }
        # V3.14d: RAW stats (A/B comparison — ALL signals including filtered ones)
        self.hydra_quarantine_raw: Dict[str, dict] = {
            name: {"predictions": 0, "correct": 0, "wrong": 0, "pnl": 0.0}
            for name in self.HYDRA_STRATEGIES
        }
        # V3.14d: Per-filter tracking (which filters save the most money)
        self.hydra_filter_stats: Dict[str, dict] = {}  # {filter_name: {blocked: N, would_win: N, would_lose: N, saved_pnl: $}}
        self.hydra_pending: Dict[str, dict] = {}  # {pred_key: {strategy, asset, direction, market_id, entry_time, ...}}
        self.HYDRA_MIN_TRADES = 20  # Min predictions before evaluation
        self.HYDRA_PROMOTE_WR = 0.55  # 55% WR to promote

        # === 5M SHADOW PAPER TRADING (V3.18 + V3.19 ML) ===
        # Momentum + ML direction/sizing for 5-minute market trading
        self.shadow_5m_trades: Dict[str, dict] = {}  # {trade_key: trade_dict}
        self.shadow_5m_stats = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": []}
        self.shadow_5m_SIZE = 10.0  # Default paper position size
        # 5m entry rules: momentum-first, ML-enhanced
        self.SHADOW_5M_MIN_MOMENTUM = 0.0003  # 0.03% price move (relaxed for 5m)
        self.SHADOW_5M_MAX_ENTRY = 0.60       # 5m prices hover near 0.50 — allow wider range
        self.SHADOW_5M_MIN_ENTRY = 0.10       # Ultra-cheap = noise
        self.SHADOW_5M_TIME_WINDOW = (0.5, 4.8)  # Minutes before expiry
        self.SHADOW_5M_MAX_CONCURRENT = 5     # Max open 5m positions

        # === 15M MOMENTUM SHADOW PAPER TRADING (V3.20) ===
        # Same momentum continuation strategy as 5m, applied to 15-minute markets
        # Purpose: A/B test whether momentum edge holds on longer timeframe
        self.shadow_15m_mom_trades: Dict[str, dict] = {}  # {trade_key: trade_dict}
        self.shadow_15m_mom_stats = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": []}
        self.shadow_15m_mom_SIZE = 10.0  # Paper position size (same as 5m for fair comparison)
        self.SHADOW_15M_MOM_MIN_MOMENTUM = 0.0005  # 0.05% price move (slightly higher than 5m's 0.03%)
        self.SHADOW_15M_MOM_MAX_ENTRY = 0.60
        self.SHADOW_15M_MOM_MIN_ENTRY = 0.10
        self.SHADOW_15M_MOM_TIME_WINDOW = (2.0, 12.0)  # Minutes before expiry
        self.SHADOW_15M_MOM_MAX_CONCURRENT = 5

        # === FLASH CRASH DETECTION (V3.20 - from discountry/polymarket-trading-bot) ===
        # Track Polymarket odds over time. When a side drops >threshold in lookback window,
        # buy the crash (mean reversion). Inspired by MuseumOfBees whale ($13.6K from 5m BTC).
        self._market_price_history: Dict[str, Dict[str, list]] = {}  # {condition_id: {"up": [(ts, price), ...], "down": [...]}}
        self.FLASH_CRASH_DROP_THRESHOLD = 0.08  # 8-cent drop triggers flash crash (ref repo uses 0.30 — too aggressive)
        self.FLASH_CRASH_LOOKBACK_SEC = 120     # Look back 2 minutes (4 cycles at 30s)
        self.FLASH_CRASH_MAX_HISTORY = 20       # Keep last 20 price points per market
        self.FLASH_CRASH_SIZE = 10.0            # Paper position size for flash crash trades

        # === CROSS-MARKET SPILLOVER ARBITRAGE (V3.21) ===
        # When BTC moves and SOL/ETH Polymarket odds haven't adjusted, buy the lagging side.
        # Uses covariance matrix between crypto assets to predict expected moves.
        # Math: β = Σᵢⱼ/σᵢ² → expected_Δj = β * Δi → residual = actual_Δj - expected_Δj
        # Trade when |residual| > k * σⱼ (the follower is mispriced)
        self.spillover_trades: Dict[str, dict] = {}
        self.spillover_stats = {"wins": 0, "losses": 0, "pnl": 0.0, "trades": []}
        self.SPILLOVER_SIZE = 10.0              # Paper position size
        self.SPILLOVER_MIN_LEADER_MOVE = 0.0015 # Leader must move >0.15% in 10 min
        self.SPILLOVER_K_THRESHOLD = 1.5        # Z-score threshold for residual
        self.SPILLOVER_BETA_WINDOW = 60         # Rolling window: 60 candle returns
        self.SPILLOVER_MAX_CONCURRENT = 3       # Max open spillover positions
        self.SPILLOVER_TIME_WINDOW = (1.5, 12.0)  # Minutes before expiry

        # === BTC ML PREDICTORS (V3.19) ===
        # LightGBM models trained on Binance candle data
        # 3 models per timeframe: direction, volatility, momentum quality
        self.ml_5m: Optional['BTCPredictor'] = None
        self.ml_15m: Optional['BTCPredictor'] = None
        self._ml_initialized = False

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

    # === SYSTEMATIC TRADING FEATURE HELPERS (V3.5) ===

    def _calc_atr(self, candles, period: int = 14) -> float:
        """Calculate Average True Range from candles."""
        if len(candles) < period + 1:
            return 0.0
        trs = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        if len(trs) < period:
            return sum(trs) / len(trs) if trs else 0.0
        return sum(trs[-period:]) / period

    def _check_overnight_seasonality(self, hour: int, asset: str, side: str) -> float:
        """Feature 1: BTC overnight seasonality boost.
        Returns confidence boost (0 if not applicable)."""
        if not self.OVERNIGHT_ENABLED:
            return 0.0
        if hour in self.OVERNIGHT_HOURS and asset == "BTC" and side == "UP":
            return self.OVERNIGHT_BTC_BOOST
        return 0.0

    def _calc_inverse_vol_scale(self, candles) -> float:
        """Feature 2: Inverse volatility position sizing.
        Returns size multiplier based on baseline_atr / current_atr."""
        if not self.INVVOL_ENABLED or len(candles) < 60:
            return 1.0
        baseline_atr = self._calc_atr(candles[-120:], 60) if len(candles) >= 120 else self._calc_atr(candles, len(candles) - 1)
        current_atr = self._calc_atr(candles[-20:], 14)
        if current_atr <= 0 or baseline_atr <= 0:
            return 1.0
        ratio = baseline_atr / current_atr
        return max(self.INVVOL_CLAMP[0], min(self.INVVOL_CLAMP[1], ratio))

    def _check_mtf_confirmation(self, candles, signal_side: str) -> bool:
        """Feature 3: Multi-timeframe confirmation.
        Returns True if short and long TF agree on direction."""
        if not self.MTF_ENABLED or len(candles) < 60:
            return True  # Not enough data, assume agree
        short_candles = candles[-30:]
        long_candles = candles[-120:] if len(candles) >= 120 else candles

        # Short-term trend: slope of last 30 closes
        short_closes = [c.close for c in short_candles]
        short_slope = (short_closes[-1] - short_closes[0]) / short_closes[0] if short_closes[0] else 0

        # Long-term trend: slope of full lookback closes
        long_closes = [c.close for c in long_candles]
        long_slope = (long_closes[-1] - long_closes[0]) / long_closes[0] if long_closes[0] else 0

        # Direction agreement
        if signal_side == "UP":
            # UP signal needs at least one TF trending up (not both down)
            return not (short_slope < -0.001 and long_slope < -0.001)
        else:
            # DOWN signal needs at least one TF trending down (not both up)
            return not (short_slope > 0.001 and long_slope > 0.001)

    def _check_volume_spike(self, candles) -> str:
        """Feature 4: Volume spike detection.
        Returns 'spike' if volume > threshold*avg, 'low' if <0.5x avg, 'normal' otherwise."""
        if not self.VOLSPIKE_ENABLED or len(candles) < 30:
            return "normal"
        recent_vols = [c.volume for c in candles[-5:]]
        avg_vols = [c.volume for c in candles[-60:]] if len(candles) >= 60 else [c.volume for c in candles]
        recent_avg = sum(recent_vols) / len(recent_vols) if recent_vols else 0
        overall_avg = sum(avg_vols) / len(avg_vols) if avg_vols else 0
        if overall_avg <= 0:
            return "normal"
        ratio = recent_avg / overall_avg
        if ratio >= self.VOLSPIKE_THRESHOLD:
            return "spike"
        elif ratio < 0.5:
            return "low"
        return "normal"

    def _get_vol_regime(self, candles) -> str:
        """Feature 5: Volatility regime detection for routing.
        Returns 'high', 'low', or 'normal'."""
        if not self.VOLREGIME_ENABLED or len(candles) < 60:
            return "normal"
        current_atr = self._calc_atr(candles[-20:], 14)
        baseline_atr = self._calc_atr(candles[-120:], 60) if len(candles) >= 120 else self._calc_atr(candles, len(candles) - 1)
        if baseline_atr <= 0:
            return "normal"
        ratio = current_atr / baseline_atr
        if ratio > 1.3:
            return "high"
        elif ratio < 0.8:
            return "low"
        return "normal"

    def _get_volregime_boost(self, vol_regime: str, regime_value: str) -> float:
        """Feature 5: Confidence boost based on vol regime + strategy type match.
        Mean-reversion = range_bound regime, Trend = trend_up/trend_down regime.
        Returns confidence boost (can be negative for mismatch)."""
        if not self.VOLREGIME_ENABLED:
            return 0.0
        is_mr = regime_value == "range_bound"
        is_trend = regime_value in ("trend_up", "trend_down")
        if vol_regime == "high" and is_mr:
            return self.VOLREGIME_HIGH_MR_BOOST  # High vol + MR = good
        elif vol_regime == "low" and is_trend:
            return self.VOLREGIME_LOW_TREND_BOOST  # Low vol + trend = good
        elif vol_regime == "high" and is_trend:
            return -0.02  # High vol + trend = risky (whipsaws)
        elif vol_regime == "low" and is_mr:
            return -0.02  # Low vol + MR = no movement to revert
        return 0.0

    def _update_systematic_stats(self, trade_key: str, won: bool, pnl: float):
        """Update systematic feature stats when a trade resolves."""
        feats = self._trade_features.get(trade_key, {})
        if not feats:
            return
        if feats.get("overnight"):
            bucket = self.systematic_stats["overnight"]
            bucket["trades"] += 1
            bucket["wins" if won else "losses"] += 1
            bucket["pnl"] += pnl
        if feats.get("mtf_agree") is not None:
            key = "mtf_agree" if feats["mtf_agree"] else "mtf_disagree"
            bucket = self.systematic_stats[key]
            bucket["trades"] += 1
            bucket["wins" if won else "losses"] += 1
            bucket["pnl"] += pnl
        if feats.get("volspike"):
            key = "volspike" if feats["volspike"] == "spike" else "volspike_normal"
            bucket = self.systematic_stats[key]
            bucket["trades"] += 1
            bucket["wins" if won else "losses"] += 1
            bucket["pnl"] += pnl
        if feats.get("volregime_match") is not None:
            key = "volregime_match" if feats["volregime_match"] else "volregime_mismatch"
            bucket = self.systematic_stats[key]
            bucket["trades"] += 1
            bucket["wins" if won else "losses"] += 1
            bucket["pnl"] += pnl

    def _check_feature_revoke(self):
        """ML Auto-Revoke: disable features that are net-negative after threshold trades."""
        # Overnight: revoke after 10 trades if WR < 45% or PnL < 0
        ov = self.systematic_stats["overnight"]
        if ov["trades"] >= 10 and self.OVERNIGHT_ENABLED:
            wr = ov["wins"] / max(1, ov["trades"])
            if wr < 0.45 or ov["pnl"] < 0:
                self.OVERNIGHT_ENABLED = False
                print(f"[ML-REVOKE] Overnight disabled (WR={wr:.0%}, PnL=${ov['pnl']:+.2f})")
        # MTF: revoke if disagree trades actually outperform agree trades (filter is hurting)
        ma = self.systematic_stats["mtf_agree"]
        md = self.systematic_stats["mtf_disagree"]
        if ma["trades"] >= 8 and md["trades"] >= 5 and self.MTF_ENABLED:
            agree_wr = ma["wins"] / max(1, ma["trades"])
            disagree_wr = md["wins"] / max(1, md["trades"])
            if disagree_wr > agree_wr + 0.10:  # Disagree beats agree by 10%+
                self.MTF_ENABLED = False
                print(f"[ML-REVOKE] MTF disabled (agree WR={agree_wr:.0%} < disagree WR={disagree_wr:.0%})")
        # VolSpike: revoke if spike trades underperform normal
        vs = self.systematic_stats["volspike"]
        vn = self.systematic_stats["volspike_normal"]
        if vs["trades"] >= 8 and self.VOLSPIKE_ENABLED:
            if vs["pnl"] < -3.0:
                self.VOLSPIKE_ENABLED = False
                print(f"[ML-REVOKE] VolSpike disabled (PnL=${vs['pnl']:+.2f})")
        # VolRegime: revoke if matched regime trades don't beat mismatched
        vrm = self.systematic_stats["volregime_match"]
        vrmm = self.systematic_stats["volregime_mismatch"]
        if vrm["trades"] >= 8 and vrmm["trades"] >= 5 and self.VOLREGIME_ENABLED:
            if vrmm["pnl"] > vrm["pnl"] + 2.0:
                self.VOLREGIME_ENABLED = False
                print(f"[ML-REVOKE] VolRegime disabled (match PnL=${vrm['pnl']:+.2f} < mismatch PnL=${vrmm['pnl']:+.2f})")

    def _apply_tuned_params(self):
        """Apply ML-tuned parameters to trading thresholds."""
        params = self.ml_tuner.get_current_params()
        self.UP_MAX_PRICE = params["up_max_price"]
        self.UP_MIN_CONFIDENCE = params["up_min_confidence"]
        self.DOWN_MAX_PRICE = params["down_max_price"]
        self.DOWN_MIN_CONFIDENCE = params["down_min_confidence"]
        self.MIN_EDGE = max(params["min_edge"], 0.35)  # V3.17: floor at 0.35 (CSV: 0.30 not enough edge)
        self.DOWN_MIN_MOMENTUM_DROP = params["down_momentum_threshold"]
        # Systematic Trading Features V3.5
        self.OVERNIGHT_BTC_BOOST = params.get("overnight_btc_boost", 0.05)
        self.OVERNIGHT_ENABLED = params.get("overnight_enabled", True)
        self.INVVOL_ENABLED = params.get("invvol_enabled", True)
        self.INVVOL_CLAMP = (params.get("invvol_clamp_low", 0.5), params.get("invvol_clamp_high", 2.0))
        self.MTF_ENABLED = params.get("mtf_enabled", True)
        self.MTF_PENALTY = params.get("mtf_penalty", 0.5)
        self.VOLSPIKE_ENABLED = params.get("volspike_enabled", True)
        self.VOLSPIKE_THRESHOLD = params.get("volspike_threshold", 1.5)
        self.VOLSPIKE_BOOST = params.get("volspike_boost", 1.2)
        self.VOLREGIME_ENABLED = params.get("volregime_enabled", True)
        self.VOLREGIME_HIGH_MR_BOOST = params.get("volregime_high_mr_boost", 0.05)
        self.VOLREGIME_LOW_TREND_BOOST = params.get("volregime_low_trend_boost", 0.03)
        self.VOLREGIME_MISMATCH_PENALTY = params.get("volregime_mismatch_penalty", 0.5)
        print(f"[ML] Applied tuned params: UP<${self.UP_MAX_PRICE}@{self.UP_MIN_CONFIDENCE:.0%} | DOWN<${self.DOWN_MAX_PRICE}@{self.DOWN_MIN_CONFIDENCE:.0%} | Edge>{self.MIN_EDGE:.0%}")
        print(f"[V3.5] Overnight:{self.OVERNIGHT_ENABLED} InvVol:{self.INVVOL_ENABLED} MTF:{self.MTF_ENABLED} VolSpike:{self.VOLSPIKE_ENABLED} VolRegime:{self.VOLREGIME_ENABLED}")

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
                # Load systematic feature stats (V3.5)
                saved_sys = data.get("systematic_stats", {})
                for key in self.systematic_stats:
                    if key in saved_sys:
                        self.systematic_stats[key] = saved_sys[key]
                self._trade_features = data.get("trade_features", {})
                # Load hydra quarantine state (V3.14d with A/B tracking)
                saved_hydra = data.get("hydra_quarantine", {})
                for strat in self.hydra_quarantine:
                    if strat in saved_hydra:
                        self.hydra_quarantine[strat] = saved_hydra[strat]
                saved_hydra_raw = data.get("hydra_quarantine_raw", {})
                for strat in self.hydra_quarantine_raw:
                    if strat in saved_hydra_raw:
                        self.hydra_quarantine_raw[strat] = saved_hydra_raw[strat]
                self.hydra_filter_stats = data.get("hydra_filter_stats", {})
                self.hydra_pending = data.get("hydra_pending", {})
                # Load 5m shadow state (V3.18)
                self.shadow_5m_trades = data.get("shadow_5m_trades", {})
                saved_5m_stats = data.get("shadow_5m_stats", None)
                if saved_5m_stats:
                    self.shadow_5m_stats = saved_5m_stats
                # Load 15m momentum shadow state (V3.20)
                self.shadow_15m_mom_trades = data.get("shadow_15m_mom_trades", {})
                saved_15m_stats = data.get("shadow_15m_mom_stats", None)
                if saved_15m_stats:
                    self.shadow_15m_mom_stats = saved_15m_stats
                # Load spillover state (V3.21)
                self.spillover_trades = data.get("spillover_trades", {})
                saved_spill = data.get("spillover_stats", None)
                if saved_spill:
                    self.spillover_stats = saved_spill
                arb_open = sum(1 for a in self.arb_trades.values() if a.get("status") == "open")
                hydra_total = sum(q["predictions"] for q in self.hydra_quarantine.values())
                s5 = self.shadow_5m_stats
                s15 = self.shadow_15m_mom_stats
                sp = self.spillover_stats
                print(f"Loaded {len(self.trades)} trades + {len(self.arb_trades)} arb ({arb_open} open) + {hydra_total} hydra + "
                      f"5m:{s5['wins']}W/{s5['losses']}L ${s5['pnl']:+.2f} + "
                      f"15m-mom:{s15['wins']}W/{s15['losses']}L ${s15['pnl']:+.2f} + "
                      f"spill:{sp['wins']}W/{sp['losses']}L ${sp['pnl']:+.2f}")
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
            "systematic_stats": self.systematic_stats,
            "trade_features": self._trade_features,
            "hydra_quarantine": self.hydra_quarantine,
            "hydra_quarantine_raw": self.hydra_quarantine_raw,
            "hydra_filter_stats": self.hydra_filter_stats,
            "hydra_pending": self.hydra_pending,
            "shadow_5m_trades": self.shadow_5m_trades,
            "shadow_5m_stats": self.shadow_5m_stats,
            "shadow_15m_mom_trades": self.shadow_15m_mom_trades,
            "shadow_15m_mom_stats": self.shadow_15m_mom_stats,
            "spillover_trades": self.spillover_trades,
            "spillover_stats": self.spillover_stats,
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

            # Fetch ALL 15m + 5m markets (BTC + ETH + SOL)
            # V3.16: Added 5-min BTC markets based on whale analysis
            # Canine-Commandment ($173K/18d) + Bidou28old ($88.5K/1d) both trade 5m heavily
            markets = []
            for tag_slug in ["15M", "5M"]:
                try:
                    # V3.18: 5M needs larger limit (hundreds of events, default returns expired ones)
                    fetch_limit = 200 if tag_slug == "5M" else 50
                    mr = await client.get(
                        "https://gamma-api.polymarket.com/events",
                        params={"tag_slug": tag_slug, "active": "true", "closed": "false", "limit": fetch_limit},
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
                                    m["_timeframe"] = "5m" if tag_slug == "5M" else "15m"  # V3.16
                                    # V3.18: Skip expired 5m markets (API returns stale ones)
                                    if tag_slug == "5M":
                                        end = m.get("endDate", "")
                                        try:
                                            from datetime import timezone as _tz
                                            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                                            if (end_dt - datetime.now(_tz.utc)).total_seconds() < 0:
                                                continue  # Expired
                                        except:
                                            pass
                                    markets.append(m)
                    else:
                        print(f"[API] {tag_slug} status {mr.status_code}")
                    await aio.sleep(0.5)  # Rate limit between tag fetches
                except Exception as e:
                    print(f"[API] {tag_slug} error: {e}")
            if markets:
                self._market_cache = markets
            else:
                markets = getattr(self, '_market_cache', [])

        # Use BTC candles as primary (backward compat)
        candles = btc_data.get("candles", [])

        # Count by asset and timeframe
        asset_counts = {}
        tf_counts = {"5m": 0, "15m": 0}
        for m in markets:
            a = m.get("_asset", "?")
            asset_counts[a] = asset_counts.get(a, 0) + 1
            tf = m.get("_timeframe", "15m")
            tf_counts[tf] = tf_counts.get(tf, 0) + 1
        counts_str = " | ".join(f"{a}:{c}" for a, c in sorted(asset_counts.items()))
        tf_str = " | ".join(f"{tf}:{c}" for tf, c in sorted(tf_counts.items()) if c > 0)
        if markets:
            print(f"[Markets] {len(markets)} total ({counts_str}) [{tf_str}]")
        else:
            print("[Markets] No active markets found")

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
        tf = market.get("_timeframe", "15m")
        default = 5.0 if tf == "5m" else 15.0
        if not end:
            return default
        try:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return max(0, (end_dt - now).total_seconds() / 60)
        except:
            return default

    # Conviction thresholds - prevent flip-flopping
    MIN_MODEL_CONFIDENCE = 0.68  # V3.17: TIGHTENED — CSV 40% WR means model is wrong too often
    MIN_EDGE = 0.35              # V3.17: TIGHTENED from 0.25 — need bigger edge to overcome taker disadvantage
    MIN_KL_DIVERGENCE = 0.18     # V3.17: TIGHTENED from 0.12 — small divergences = coin flips
    MIN_TIME_REMAINING = 5.0     # V3.4: 2-5min = 47% WR; 5-12min = 83% WR (96 paper trades)
    MAX_ENTRY_PRICE = 0.42       # V3.17: TIGHTENED — CSV $0.40-0.50 = -30% ROI
    MIN_ENTRY_PRICE = 0.25       # V3.17: REVERTED — CSV <$0.20 = -73% ROI death
    CANDLE_LOOKBACK = 120        # V3.6b: Was 15 — killed MACD(35), TTM(25), EMA Cross(20), RSI slope. Now all 9 indicators active.

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

    def _record_market_prices(self, markets):
        """Record current Polymarket odds for flash crash detection."""
        now = time.time()
        for market in markets:
            cid = market.get("conditionId", "")
            if not cid:
                continue
            up_p, down_p = self.get_market_prices(market)
            if up_p is None or down_p is None:
                continue
            if cid not in self._market_price_history:
                self._market_price_history[cid] = {"up": [], "down": []}
            hist = self._market_price_history[cid]
            hist["up"].append((now, up_p))
            hist["down"].append((now, down_p))
            # Trim to max history
            if len(hist["up"]) > self.FLASH_CRASH_MAX_HISTORY:
                hist["up"] = hist["up"][-self.FLASH_CRASH_MAX_HISTORY:]
            if len(hist["down"]) > self.FLASH_CRASH_MAX_HISTORY:
                hist["down"] = hist["down"][-self.FLASH_CRASH_MAX_HISTORY:]

    def _detect_flash_crash(self, condition_id: str) -> Optional[tuple]:
        """Detect flash crash on a market. Returns (side, old_price, new_price, drop) or None."""
        if condition_id not in self._market_price_history:
            return None
        hist = self._market_price_history[condition_id]
        now = time.time()
        cutoff = now - self.FLASH_CRASH_LOOKBACK_SEC

        for side in ["up", "down"]:
            points = hist.get(side, [])
            if len(points) < 2:
                continue
            current_price = points[-1][1]
            # Find oldest price within lookback window
            old_price = None
            for ts, price in points:
                if ts >= cutoff:
                    old_price = price
                    break
            if old_price is None:
                continue
            drop = old_price - current_price
            if drop >= self.FLASH_CRASH_DROP_THRESHOLD:
                return (side, old_price, current_price, drop)
        return None

    async def _trade_5m_shadow(self, markets, candles, btc_price):
        """V3.18: Dedicated 5-minute market shadow paper trading."""
        try:
            await self._trade_5m_shadow_inner(markets, candles, btc_price)
        except Exception as e:
            print(f"[5M ERROR] {type(e).__name__}: {e}")

    async def _trade_5m_shadow_inner(self, markets, candles, btc_price):
        now = datetime.now(timezone.utc)
        n5 = sum(1 for m in markets if m.get("_timeframe") == "5m")
        # Find nearest 5m market
        nearest_5m = None
        for m in markets:
            if m.get("_timeframe") != "5m":
                continue
            tl = self.get_time_remaining(m)
            if nearest_5m is None or tl < nearest_5m[0]:
                nearest_5m = (tl, m.get("question", "")[:50])
        nm_str = f"nearest={nearest_5m[0]:.1f}m" if nearest_5m else "none"
        print(f"[5M DEBUG] {n5} 5m mkts | {nm_str} | candles={len(candles)} | btc=${btc_price:,.0f}")

        # Count open 5m positions
        open_5m = sum(1 for t in self.shadow_5m_trades.values() if t["status"] == "open")

        # --- RESOLVE expired 5m trades first ---
        for tid, trade in list(self.shadow_5m_trades.items()):
            if trade["status"] != "open":
                continue
            # Find this trade's market in current market list
            for mkt in markets:
                mkt_id = mkt.get("conditionId", "")
                if mkt_id != trade.get("condition_id"):
                    continue
                time_left = self.get_time_remaining(mkt)
                if time_left > 0.5:
                    continue
                up_p, down_p = self.get_market_prices(mkt)
                if up_p is None or down_p is None:
                    continue
                # Resolve
                price = up_p if trade["side"] == "UP" else down_p
                if price >= 0.95:
                    exit_val = trade["size_usd"] / trade["entry_price"]
                elif price <= 0.05:
                    exit_val = 0
                else:
                    exit_val = (trade["size_usd"] / trade["entry_price"]) * price
                trade["exit_price"] = price
                trade["exit_time"] = now.isoformat()
                trade["pnl"] = exit_val - trade["size_usd"]
                trade["status"] = "closed"

                won = trade["pnl"] > 0
                self.shadow_5m_stats["wins" if won else "losses"] += 1
                self.shadow_5m_stats["pnl"] += trade["pnl"]
                self.shadow_5m_stats["trades"].append({
                    "side": trade["side"], "entry": trade["entry_price"],
                    "exit": price, "pnl": trade["pnl"], "strategy": trade.get("strategy", "momentum"),
                    "asset": trade.get("asset", "BTC"), "time": trade["entry_time"],
                })
                open_5m -= 1

                w = self.shadow_5m_stats["wins"]
                l = self.shadow_5m_stats["losses"]
                wr = w / (w + l) * 100 if (w + l) > 0 else 0
                tag = "WIN" if won else "LOSS"
                ml_str = f" | ML={trade.get('ml_direction','?')}({trade.get('ml_score',0):.2f})" if trade.get('ml_direction') else ""
                print(f"[5M {tag}] {trade['side']} ${trade['pnl']:+.2f} | entry=${trade['entry_price']:.2f} exit=${price:.2f} | "
                      f"5m stats: {w}W/{l}L {wr:.0f}%WR ${self.shadow_5m_stats['pnl']:+.2f}{ml_str} | {trade['title'][:40]}")
                # V3.19: Feed outcome to ML for online learning
                if self._ml_initialized and self.ml_5m and trade.get('ml_features') is not None:
                    try:
                        import numpy as _np
                        self.ml_5m.add_outcome(
                            _np.array(trade['ml_features']),
                            trade['side'], won)
                    except Exception:
                        pass
                break

        # Also resolve via age (>6 min old = definitely expired)
        for tid, trade in list(self.shadow_5m_trades.items()):
            if trade["status"] != "open":
                continue
            try:
                entry_dt = datetime.fromisoformat(trade["entry_time"])
                age_min = (now - entry_dt).total_seconds() / 60
            except:
                age_min = 999
            if age_min > 6:
                # Market expired, resolve via Gamma API
                nid = trade.get("market_numeric_id")
                if nid:
                    try:
                        async with httpx.AsyncClient(timeout=8) as cl:
                            r = await cl.get(f"https://gamma-api.polymarket.com/markets/{nid}",
                                           headers={"User-Agent": "Mozilla/5.0"})
                            if r.status_code == 200:
                                rm = r.json()
                                up_p, down_p = self.get_market_prices(rm)
                                if up_p is not None:
                                    price = up_p if trade["side"] == "UP" else down_p
                                    if price >= 0.95:
                                        exit_val = trade["size_usd"] / trade["entry_price"]
                                    elif price <= 0.05:
                                        exit_val = 0
                                    else:
                                        exit_val = (trade["size_usd"] / trade["entry_price"]) * price
                                    trade["exit_price"] = price
                                    trade["exit_time"] = now.isoformat()
                                    trade["pnl"] = exit_val - trade["size_usd"]
                                    trade["status"] = "closed"
                                    won = trade["pnl"] > 0
                                    self.shadow_5m_stats["wins" if won else "losses"] += 1
                                    self.shadow_5m_stats["pnl"] += trade["pnl"]
                                    self.shadow_5m_stats["trades"].append({
                                        "side": trade["side"], "entry": trade["entry_price"],
                                        "exit": price, "pnl": trade["pnl"],
                                        "strategy": trade.get("strategy", "momentum"),
                                        "asset": trade.get("asset", "BTC"), "time": trade["entry_time"],
                                    })
                                    open_5m -= 1
                                    w = self.shadow_5m_stats["wins"]
                                    l = self.shadow_5m_stats["losses"]
                                    wr = w / (w + l) * 100 if (w + l) > 0 else 0
                                    tag = "WIN" if won else "LOSS"
                                    print(f"[5M-API {tag}] {trade['side']} ${trade['pnl']:+.2f} | 5m: {w}W/{l}L {wr:.0f}%WR ${self.shadow_5m_stats['pnl']:+.2f}")
                                    # V3.19: ML feedback
                                    if self._ml_initialized and self.ml_5m and trade.get('ml_features') is not None:
                                        try:
                                            import numpy as _np
                                            self.ml_5m.add_outcome(
                                                _np.array(trade['ml_features']),
                                                trade['side'], won)
                                        except Exception:
                                            pass
                    except Exception as e:
                        print(f"[5M] API resolve error: {e}")
                else:
                    # No numeric ID, mark as loss
                    trade["status"] = "closed"
                    trade["pnl"] = -trade["size_usd"]
                    self.shadow_5m_stats["losses"] += 1
                    self.shadow_5m_stats["pnl"] += trade["pnl"]
                    open_5m -= 1

        # --- ENTRY: Find new 5m trades ---
        if open_5m >= self.SHADOW_5M_MAX_CONCURRENT:
            return

        # Calculate short-term momentum from candles (Candle objects with .close)
        if len(candles) < 3:
            return

        # Momentum: price change over last 2-3 candles (recent 1-min bars)
        try:
            recent_closes = [c.close if hasattr(c, 'close') else float(c.get("c", c.get("close", 0))) for c in candles[-3:]]
            if recent_closes[-1] == 0 or recent_closes[0] == 0:
                return
            momentum_10m = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            momentum_5m = (recent_closes[-1] - recent_closes[-2]) / recent_closes[-2]
        except (IndexError, ValueError, ZeroDivisionError, AttributeError):
            return

        # Scan 5m markets in window
        found_5m = 0
        for market in markets:
            tf = market.get("_timeframe", "15m")
            if tf != "5m":
                continue
            found_5m += 1

            time_left = self.get_time_remaining(market)
            up_price, down_price = self.get_market_prices(market)

            if time_left < self.SHADOW_5M_TIME_WINDOW[0] or time_left > self.SHADOW_5M_TIME_WINDOW[1]:
                continue

            if up_price is None or down_price is None:
                continue

            asset = market.get("_asset", "BTC")
            cid = market.get("conditionId", "")
            question = market.get("question", "")
            nid = market.get("id")

            print(f"[5M SCAN] {asset} | UP=${up_price:.2f} DOWN=${down_price:.2f} | "
                  f"time={time_left:.1f}m | mom10m={momentum_10m:+.4%} mom5m={momentum_5m:+.4%} | "
                  f"{question[:50]}")

            # Skip if already trading this market
            trade_key_up = f"5m_{cid}_UP"
            trade_key_down = f"5m_{cid}_DOWN"
            if trade_key_up in self.shadow_5m_trades or trade_key_down in self.shadow_5m_trades:
                continue

            # === V3.19: ML PREDICTION ===
            # Get ML direction, volatility regime, and momentum quality
            ml_pred = None
            ml_dir = None
            ml_score = 0.0
            ml_size = self.shadow_5m_SIZE  # Default
            if self._ml_initialized and self.ml_5m and self.ml_5m.is_trained:
                try:
                    ml_pred = self.ml_5m.predict(self.ml_5m.candles_raw)
                    ml_dir = ml_pred.direction  # "UP", "DOWN", or "SKIP"
                    ml_score = ml_pred.trade_score

                    # ML-based position sizing
                    if ml_pred.kelly_size_usd > 0:
                        ml_size = ml_pred.kelly_size_usd
                    elif ml_pred.vol_regime == "low":
                        ml_size = self.shadow_5m_SIZE * 0.5  # Half size in low vol
                    elif ml_pred.momentum_quality > 0.6:
                        ml_size = min(self.shadow_5m_SIZE * 1.5, 15.0)  # Boost for quality momentum
                except Exception:
                    pass

            # === STRATEGY 1: MOMENTUM + TA + ML DIRECTIONAL ===
            # Combines price momentum with TA signals and ML prediction
            # Key edges: MACD expanding (87.5% WR), below VWAP (76.5% WR), BTC UP (90.9% WR)
            side = None
            entry_price = None
            strategy = "momentum"

            # Extract TA features from main signal (generated earlier in run_cycle)
            macd_expanding = False
            below_vwap = False
            heiken_count = 0
            if hasattr(self, '_last_signal') and self._last_signal:
                sig = self._last_signal
                if sig.macd and hasattr(sig.macd, 'expanding'):
                    macd_expanding = sig.macd.expanding
                elif sig.macd and hasattr(sig.macd, 'histogram_delta'):
                    macd_expanding = (sig.macd.histogram_delta or 0) > 0
                if sig.vwap and btc_price:
                    below_vwap = btc_price < sig.vwap
                heiken_count = sig.heiken_count or 0

            # Score the opportunity (higher = better)
            ta_score = 0
            if macd_expanding:
                ta_score += 2  # Strongest signal (87.5% WR)
            if below_vwap:
                ta_score += 1  # 76.5% WR
            if heiken_count == 3:
                ta_score -= 2  # Reversal danger zone

            # ML momentum quality gate: if quality model says momentum is weak, skip
            if ml_pred and ml_pred.momentum_quality < 0.25:
                # Very low quality momentum — high reversal risk
                print(f"[5M ML-SKIP] {asset} | quality={ml_pred.momentum_quality:.2f} | "
                      f"vol={ml_pred.vol_regime} | Momentum too weak")
                continue

            if momentum_10m > self.SHADOW_5M_MIN_MOMENTUM:
                # Upward momentum — buy UP
                if self.SHADOW_5M_MIN_ENTRY <= up_price <= self.SHADOW_5M_MAX_ENTRY:
                    side = "UP"
                    entry_price = round(up_price + 0.02, 2)
                    if momentum_5m > 0 and ta_score >= 2:
                        strategy = "ultra"  # Full alignment: momentum + MACD + VWAP
                    elif momentum_5m > 0:
                        strategy = "momentum_strong"
                    # V3.19: ML override — if ML says opposite with high confidence, flip
                    if ml_dir == "DOWN" and ml_score > 0.55:
                        side = "DOWN"
                        entry_price = round(down_price + 0.02, 2) if self.SHADOW_5M_MIN_ENTRY <= down_price <= self.SHADOW_5M_MAX_ENTRY else None
                        strategy = "ml_override"
                    elif ml_dir == "UP" and ml_score > 0.50:
                        strategy = "ml_confirmed" if strategy == "momentum" else strategy
            elif momentum_10m < -self.SHADOW_5M_MIN_MOMENTUM:
                # Downward momentum — buy DOWN
                if self.SHADOW_5M_MIN_ENTRY <= down_price <= self.SHADOW_5M_MAX_ENTRY:
                    side = "DOWN"
                    entry_price = round(down_price + 0.02, 2)
                    if momentum_5m < 0 and ta_score >= 2:
                        strategy = "ultra"
                    elif momentum_5m < 0:
                        strategy = "momentum_strong"
                    # V3.19: ML override
                    if ml_dir == "UP" and ml_score > 0.55:
                        side = "UP"
                        entry_price = round(up_price + 0.02, 2) if self.SHADOW_5M_MIN_ENTRY <= up_price <= self.SHADOW_5M_MAX_ENTRY else None
                        strategy = "ml_override"
                    elif ml_dir == "DOWN" and ml_score > 0.50:
                        strategy = "ml_confirmed" if strategy == "momentum" else strategy
            elif ml_dir in ("UP", "DOWN") and ml_score > 0.55:
                # V3.19: NO momentum but ML is confident — ML-only entry
                if ml_dir == "UP" and self.SHADOW_5M_MIN_ENTRY <= up_price <= self.SHADOW_5M_MAX_ENTRY:
                    side = "UP"
                    entry_price = round(up_price + 0.02, 2)
                    strategy = "ml_only"
                elif ml_dir == "DOWN" and self.SHADOW_5M_MIN_ENTRY <= down_price <= self.SHADOW_5M_MAX_ENTRY:
                    side = "DOWN"
                    entry_price = round(down_price + 0.02, 2)
                    strategy = "ml_only"

            # === STRATEGY 2: BOTH-SIDES (5m arb) ===
            # If UP + DOWN < 0.96, buy both for guaranteed ~4% profit
            combined = up_price + down_price
            if combined < 0.96 and not side:
                up_entry = round(up_price + 0.01, 2)
                down_entry = round(down_price + 0.01, 2)
                if up_entry + down_entry < 0.98:
                    for s, ep in [("UP", up_entry), ("DOWN", down_entry)]:
                        tk = f"5m_{cid}_{s}"
                        self.shadow_5m_trades[tk] = {
                            "side": s, "entry_price": ep, "size_usd": self.shadow_5m_SIZE / 2,
                            "entry_time": now.isoformat(), "condition_id": cid,
                            "market_numeric_id": nid, "title": question,
                            "asset": asset, "strategy": "both_sides_5m",
                            "status": "open", "pnl": 0.0,
                        }
                    print(f"[5M ARB] {asset} BOTH SIDES | UP=${up_entry:.2f} DOWN=${down_entry:.2f} | "
                          f"combined=${combined:.2f} | {question[:50]}")
                    open_5m += 2
                    continue

            # === STRATEGY 3: EXTREME PRICE (high-conviction directional) ===
            if not side:
                if up_price < 0.20 and momentum_10m > 0.001:
                    side = "UP"
                    entry_price = round(up_price + 0.02, 2)
                    strategy = "extreme_cheap"
                elif down_price < 0.20 and momentum_10m < -0.001:
                    side = "DOWN"
                    entry_price = round(down_price + 0.02, 2)
                    strategy = "extreme_cheap"

            # === STRATEGY 4: FLASH CRASH (mean reversion on probability drops) ===
            # Inspired by discountry/polymarket-trading-bot + MuseumOfBees whale
            if not side:
                crash = self._detect_flash_crash(cid)
                if crash:
                    crash_side, old_p, new_p, drop = crash
                    # Buy the crashed side (mean reversion)
                    if crash_side == "up" and self.SHADOW_5M_MIN_ENTRY <= up_price <= self.SHADOW_5M_MAX_ENTRY:
                        side = "UP"
                        entry_price = round(up_price + 0.02, 2)
                        strategy = "flash_crash"
                    elif crash_side == "down" and self.SHADOW_5M_MIN_ENTRY <= down_price <= self.SHADOW_5M_MAX_ENTRY:
                        side = "DOWN"
                        entry_price = round(down_price + 0.02, 2)
                        strategy = "flash_crash"
                    if side:
                        print(f"[5M FLASH CRASH] {asset} {crash_side.upper()} dropped {drop:.2f} "
                              f"({old_p:.2f}->{new_p:.2f}) in {self.FLASH_CRASH_LOOKBACK_SEC}s | "
                              f"Buying {side} @ ${entry_price:.2f}")

            if side and entry_price and entry_price <= self.SHADOW_5M_MAX_ENTRY:
                # V3.19: ML-adjusted position size
                trade_size = ml_size if ml_pred else self.shadow_5m_SIZE

                trade_key = f"5m_{cid}_{side}"
                self.shadow_5m_trades[trade_key] = {
                    "side": side, "entry_price": entry_price, "size_usd": trade_size,
                    "entry_time": now.isoformat(), "condition_id": cid,
                    "market_numeric_id": nid, "title": question,
                    "asset": asset, "strategy": strategy,
                    "momentum_10m": momentum_10m, "momentum_5m": momentum_5m,
                    "ta_score": ta_score, "macd_expanding": macd_expanding, "below_vwap": below_vwap,
                    "ml_direction": ml_dir, "ml_score": ml_score,
                    "ml_vol_regime": ml_pred.vol_regime if ml_pred else "n/a",
                    "ml_quality": ml_pred.momentum_quality if ml_pred else 0,
                    "ml_features": ml_pred.features.tolist() if ml_pred and ml_pred.features is not None else None,
                    "status": "open", "pnl": 0.0,
                }
                open_5m += 1
                ta_str = f"MACD={'Y' if macd_expanding else 'N'} VWAP={'below' if below_vwap else 'above'} HA={heiken_count}"
                ml_str = f"ML={ml_dir}({ml_score:.2f})" if ml_pred else "ML=off"
                print(f"[5M ENTRY] {asset} {side} ${entry_price:.2f} ${trade_size:.0f} | strat={strategy} ta={ta_score} | "
                      f"mom_10m={momentum_10m:+.3%} mom_5m={momentum_5m:+.3%} | "
                      f"{ta_str} | {ml_str} | time={time_left:.1f}m | {question[:50]}")

                if open_5m >= self.SHADOW_5M_MAX_CONCURRENT:
                    break

        if found_5m == 0:
            print(f"[5M] No 5m markets found in market list ({len(markets)} total)")

        # Save 5m state
        self._save()

    async def _trade_15m_momentum_shadow(self, markets, candles, btc_price):
        """V3.20: 15-minute momentum continuation shadow paper trading.

        Same momentum strategy as 5m shadow, applied to 15m markets.
        Purpose: A/B test whether the momentum continuation edge that works
        on 5m (64% WR, +$220) also holds on 15m timeframe.

        Key differences from 5m:
        - Filters for _timeframe == "15m" markets
        - Slightly higher momentum threshold (0.05% vs 0.03%)
        - Wider time window (2-12 min vs 0.5-4.8 min)
        - No ML integration (pure momentum for clean comparison)
        - Resolve via age >18 min (vs >6 min for 5m)
        """
        try:
            await self._trade_15m_momentum_shadow_inner(markets, candles, btc_price)
        except Exception as e:
            print(f"[15M-MOM ERROR] {type(e).__name__}: {e}")

    async def _trade_15m_momentum_shadow_inner(self, markets, candles, btc_price):
        now = datetime.now(timezone.utc)
        n15 = sum(1 for m in markets if m.get("_timeframe") == "15m")

        # Count open 15m momentum positions
        open_15m = sum(1 for t in self.shadow_15m_mom_trades.values() if t["status"] == "open")

        # --- RESOLVE expired 15m momentum trades ---
        for tid, trade in list(self.shadow_15m_mom_trades.items()):
            if trade["status"] != "open":
                continue
            for mkt in markets:
                mkt_id = mkt.get("conditionId", "")
                if mkt_id != trade.get("condition_id"):
                    continue
                time_left = self.get_time_remaining(mkt)
                if time_left > 0.5:
                    continue
                up_p, down_p = self.get_market_prices(mkt)
                if up_p is None or down_p is None:
                    continue
                price = up_p if trade["side"] == "UP" else down_p
                if price >= 0.95:
                    exit_val = trade["size_usd"] / trade["entry_price"]
                elif price <= 0.05:
                    exit_val = 0
                else:
                    exit_val = (trade["size_usd"] / trade["entry_price"]) * price
                trade["exit_price"] = price
                trade["exit_time"] = now.isoformat()
                trade["pnl"] = exit_val - trade["size_usd"]
                trade["status"] = "closed"

                won = trade["pnl"] > 0
                self.shadow_15m_mom_stats["wins" if won else "losses"] += 1
                self.shadow_15m_mom_stats["pnl"] += trade["pnl"]
                self.shadow_15m_mom_stats["trades"].append({
                    "side": trade["side"], "entry": trade["entry_price"],
                    "exit": price, "pnl": trade["pnl"], "strategy": trade.get("strategy", "momentum"),
                    "asset": trade.get("asset", "BTC"), "time": trade["entry_time"],
                })
                open_15m -= 1

                w = self.shadow_15m_mom_stats["wins"]
                l = self.shadow_15m_mom_stats["losses"]
                wr = w / (w + l) * 100 if (w + l) > 0 else 0
                tag = "WIN" if won else "LOSS"
                print(f"[15M-MOM {tag}] {trade['side']} ${trade['pnl']:+.2f} | entry=${trade['entry_price']:.2f} exit=${price:.2f} | "
                      f"15m-mom: {w}W/{l}L {wr:.0f}%WR ${self.shadow_15m_mom_stats['pnl']:+.2f} | {trade['title'][:40]}")
                break

        # Resolve via age (>18 min old = definitely expired for 15m markets)
        for tid, trade in list(self.shadow_15m_mom_trades.items()):
            if trade["status"] != "open":
                continue
            try:
                entry_dt = datetime.fromisoformat(trade["entry_time"])
                age_min = (now - entry_dt).total_seconds() / 60
            except:
                age_min = 999
            if age_min > 18:
                nid = trade.get("market_numeric_id")
                if nid:
                    try:
                        async with httpx.AsyncClient(timeout=8) as cl:
                            r = await cl.get(f"https://gamma-api.polymarket.com/markets/{nid}",
                                           headers={"User-Agent": "Mozilla/5.0"})
                            if r.status_code == 200:
                                rm = r.json()
                                up_p, down_p = self.get_market_prices(rm)
                                if up_p is not None:
                                    price = up_p if trade["side"] == "UP" else down_p
                                    if price >= 0.95:
                                        exit_val = trade["size_usd"] / trade["entry_price"]
                                    elif price <= 0.05:
                                        exit_val = 0
                                    else:
                                        exit_val = (trade["size_usd"] / trade["entry_price"]) * price
                                    trade["exit_price"] = price
                                    trade["exit_time"] = now.isoformat()
                                    trade["pnl"] = exit_val - trade["size_usd"]
                                    trade["status"] = "closed"
                                    won = trade["pnl"] > 0
                                    self.shadow_15m_mom_stats["wins" if won else "losses"] += 1
                                    self.shadow_15m_mom_stats["pnl"] += trade["pnl"]
                                    self.shadow_15m_mom_stats["trades"].append({
                                        "side": trade["side"], "entry": trade["entry_price"],
                                        "exit": price, "pnl": trade["pnl"],
                                        "strategy": trade.get("strategy", "momentum"),
                                        "asset": trade.get("asset", "BTC"), "time": trade["entry_time"],
                                    })
                                    open_15m -= 1
                                    w = self.shadow_15m_mom_stats["wins"]
                                    l = self.shadow_15m_mom_stats["losses"]
                                    wr = w / (w + l) * 100 if (w + l) > 0 else 0
                                    tag = "WIN" if won else "LOSS"
                                    print(f"[15M-MOM-API {tag}] {trade['side']} ${trade['pnl']:+.2f} | "
                                          f"15m-mom: {w}W/{l}L {wr:.0f}%WR ${self.shadow_15m_mom_stats['pnl']:+.2f}")
                    except Exception as e:
                        print(f"[15M-MOM] API resolve error: {e}")
                else:
                    trade["status"] = "closed"
                    trade["pnl"] = -trade["size_usd"]
                    self.shadow_15m_mom_stats["losses"] += 1
                    self.shadow_15m_mom_stats["pnl"] += trade["pnl"]
                    open_15m -= 1

        # --- ENTRY: Find new 15m momentum trades ---
        if open_15m >= self.SHADOW_15M_MOM_MAX_CONCURRENT:
            return

        # Calculate short-term momentum from candles
        if len(candles) < 3:
            return

        try:
            recent_closes = [c.close if hasattr(c, 'close') else float(c.get("c", c.get("close", 0))) for c in candles[-3:]]
            if recent_closes[-1] == 0 or recent_closes[0] == 0:
                return
            momentum_10m = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            momentum_5m = (recent_closes[-1] - recent_closes[-2]) / recent_closes[-2]
        except (IndexError, ValueError, ZeroDivisionError, AttributeError):
            return

        # Scan 15m markets
        for market in markets:
            tf = market.get("_timeframe", "15m")
            if tf != "15m":
                continue

            time_left = self.get_time_remaining(market)
            up_price, down_price = self.get_market_prices(market)

            if time_left < self.SHADOW_15M_MOM_TIME_WINDOW[0] or time_left > self.SHADOW_15M_MOM_TIME_WINDOW[1]:
                continue

            if up_price is None or down_price is None:
                continue

            asset = market.get("_asset", "BTC")
            cid = market.get("conditionId", "")
            question = market.get("question", "")
            nid = market.get("id")

            # Skip if already trading this market
            trade_key_up = f"15m_mom_{cid}_UP"
            trade_key_down = f"15m_mom_{cid}_DOWN"
            if trade_key_up in self.shadow_15m_mom_trades or trade_key_down in self.shadow_15m_mom_trades:
                continue

            # === PURE MOMENTUM STRATEGY (same logic as 5m, no ML) ===
            side = None
            entry_price = None
            strategy = "momentum"

            if momentum_10m > self.SHADOW_15M_MOM_MIN_MOMENTUM:
                # Upward momentum -- buy UP
                if self.SHADOW_15M_MOM_MIN_ENTRY <= up_price <= self.SHADOW_15M_MOM_MAX_ENTRY:
                    side = "UP"
                    entry_price = round(up_price + 0.02, 2)  # Spread simulation
                    if momentum_5m > 0:
                        strategy = "momentum_strong"
            elif momentum_10m < -self.SHADOW_15M_MOM_MIN_MOMENTUM:
                # Downward momentum -- buy DOWN
                if self.SHADOW_15M_MOM_MIN_ENTRY <= down_price <= self.SHADOW_15M_MOM_MAX_ENTRY:
                    side = "DOWN"
                    entry_price = round(down_price + 0.02, 2)
                    if momentum_5m < 0:
                        strategy = "momentum_strong"

            # === FLASH CRASH (mean reversion on 15m probability drops) ===
            if not side:
                crash = self._detect_flash_crash(cid)
                if crash:
                    crash_side, old_p, new_p, drop = crash
                    if crash_side == "up" and self.SHADOW_15M_MOM_MIN_ENTRY <= up_price <= self.SHADOW_15M_MOM_MAX_ENTRY:
                        side = "UP"
                        entry_price = round(up_price + 0.02, 2)
                        strategy = "flash_crash"
                    elif crash_side == "down" and self.SHADOW_15M_MOM_MIN_ENTRY <= down_price <= self.SHADOW_15M_MOM_MAX_ENTRY:
                        side = "DOWN"
                        entry_price = round(down_price + 0.02, 2)
                        strategy = "flash_crash"
                    if side:
                        print(f"[15M-MOM FLASH CRASH] {asset} {crash_side.upper()} dropped {drop:.2f} "
                              f"({old_p:.2f}->{new_p:.2f}) | Buying {side} @ ${entry_price:.2f}")

            # === BOTH-SIDES ARB (same as 5m) ===
            combined = up_price + down_price
            if combined < 0.96 and not side:
                up_entry = round(up_price + 0.01, 2)
                down_entry = round(down_price + 0.01, 2)
                if up_entry + down_entry < 0.98:
                    for s, ep in [("UP", up_entry), ("DOWN", down_entry)]:
                        tk = f"15m_mom_{cid}_{s}"
                        self.shadow_15m_mom_trades[tk] = {
                            "side": s, "entry_price": ep, "size_usd": self.shadow_15m_mom_SIZE / 2,
                            "entry_time": now.isoformat(), "condition_id": cid,
                            "market_numeric_id": nid, "title": question,
                            "asset": asset, "strategy": "both_sides_15m",
                            "status": "open", "pnl": 0.0,
                        }
                    print(f"[15M-MOM ARB] {asset} BOTH SIDES | UP=${up_entry:.2f} DOWN=${down_entry:.2f} | "
                          f"combined=${combined:.2f} | {question[:50]}")
                    open_15m += 2
                    continue

            if side and entry_price and entry_price <= self.SHADOW_15M_MOM_MAX_ENTRY:
                trade_key = f"15m_mom_{cid}_{side}"
                self.shadow_15m_mom_trades[trade_key] = {
                    "side": side, "entry_price": entry_price, "size_usd": self.shadow_15m_mom_SIZE,
                    "entry_time": now.isoformat(), "condition_id": cid,
                    "market_numeric_id": nid, "title": question,
                    "asset": asset, "strategy": strategy,
                    "momentum_10m": momentum_10m, "momentum_5m": momentum_5m,
                    "status": "open", "pnl": 0.0,
                }
                open_15m += 1
                print(f"[15M-MOM ENTRY] {asset} {side} ${entry_price:.2f} $10 | strat={strategy} | "
                      f"mom_10m={momentum_10m:+.3%} mom_5m={momentum_5m:+.3%} | "
                      f"time={time_left:.1f}m | {question[:50]}")

                if open_15m >= self.SHADOW_15M_MOM_MAX_CONCURRENT:
                    break

        # Save state
        self._save()

    def _compute_asset_betas(self):
        """Compute rolling betas between all asset pairs from Binance candles.

        β_ij = Cov(Rᵢ, Rⱼ) / Var(Rᵢ)  — how much j moves per unit move in i.
        Also computes σⱼ for z-score normalization of residuals.
        """
        asset_data = getattr(self, '_asset_data', {})
        if not asset_data:
            return {}, {}

        # Compute 1-min returns for each asset
        asset_returns = {}
        for asset_name, data in asset_data.items():
            candles = data.get("candles", [])
            if len(candles) < self.SPILLOVER_BETA_WINDOW:
                continue
            closes = []
            for c in candles[-self.SPILLOVER_BETA_WINDOW:]:
                p = c.close if hasattr(c, 'close') else float(c.get("c", c.get("close", 0)))
                closes.append(p)
            returns = []
            for i in range(1, len(closes)):
                if closes[i - 1] != 0:
                    returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
            if len(returns) >= 20:
                asset_returns[asset_name] = returns

        # Compute betas for all pairs
        betas = {}
        assets = list(asset_returns.keys())
        for i_name in assets:
            for j_name in assets:
                if i_name == j_name:
                    continue
                ri = asset_returns[i_name]
                rj = asset_returns[j_name]
                n = min(len(ri), len(rj))
                ri_n, rj_n = ri[-n:], rj[-n:]

                mean_i = sum(ri_n) / n
                mean_j = sum(rj_n) / n
                cov_ij = sum((ri_n[k] - mean_i) * (rj_n[k] - mean_j) for k in range(n)) / n
                var_i = sum((ri_n[k] - mean_i) ** 2 for k in range(n)) / n
                sigma_j = (sum((rj_n[k] - mean_j) ** 2 for k in range(n)) / n) ** 0.5

                if var_i > 1e-12:
                    beta = cov_ij / var_i
                    betas[(i_name, j_name)] = {"beta": beta, "sigma_j": sigma_j, "n": n}

        return betas, asset_returns

    async def _trade_spillover_arb(self, markets):
        """V3.21: Cross-market spillover arbitrage.

        When BTC moves significantly and SOL/ETH Polymarket odds haven't adjusted,
        buy the lagging side. Uses β = Σᵢⱼ/σᵢ² to predict expected moves.
        """
        try:
            await self._trade_spillover_arb_inner(markets)
        except Exception as e:
            print(f"[SPILLOVER ERROR] {type(e).__name__}: {e}")

    async def _trade_spillover_arb_inner(self, markets):
        now = datetime.now(timezone.utc)
        asset_data = getattr(self, '_asset_data', {})
        if len(asset_data) < 2:
            return

        # Count open spillover positions
        open_spill = sum(1 for t in self.spillover_trades.values() if t["status"] == "open")

        # --- RESOLVE expired spillover trades ---
        for tid, trade in list(self.spillover_trades.items()):
            if trade["status"] != "open":
                continue
            # Resolve by market match
            for mkt in markets:
                if mkt.get("conditionId", "") != trade.get("condition_id"):
                    continue
                time_left = self.get_time_remaining(mkt)
                if time_left > 0.5:
                    continue
                up_p, down_p = self.get_market_prices(mkt)
                if up_p is None or down_p is None:
                    continue
                price = up_p if trade["side"] == "UP" else down_p
                if price >= 0.95:
                    exit_val = trade["size_usd"] / trade["entry_price"]
                elif price <= 0.05:
                    exit_val = 0
                else:
                    exit_val = (trade["size_usd"] / trade["entry_price"]) * price
                trade["exit_price"] = price
                trade["exit_time"] = now.isoformat()
                trade["pnl"] = exit_val - trade["size_usd"]
                trade["status"] = "closed"
                won = trade["pnl"] > 0
                self.spillover_stats["wins" if won else "losses"] += 1
                self.spillover_stats["pnl"] += trade["pnl"]
                self.spillover_stats["trades"].append({
                    "side": trade["side"], "entry": trade["entry_price"],
                    "exit": price, "pnl": trade["pnl"],
                    "strategy": trade.get("strategy", "spillover"),
                    "asset": trade.get("asset", "?"), "time": trade["entry_time"],
                    "leader": trade.get("leader_asset", "?"),
                })
                open_spill -= 1
                w = self.spillover_stats["wins"]
                l = self.spillover_stats["losses"]
                wr = w / (w + l) * 100 if (w + l) > 0 else 0
                tag = "WIN" if won else "LOSS"
                print(f"[SPILL {tag}] {trade['asset']} {trade['side']} ${trade['pnl']:+.2f} | "
                      f"leader={trade.get('leader_asset','?')} z={trade.get('z_score',0):.1f} | "
                      f"spill: {w}W/{l}L {wr:.0f}%WR ${self.spillover_stats['pnl']:+.2f}")
                break

        # Resolve by age (>18 min)
        for tid, trade in list(self.spillover_trades.items()):
            if trade["status"] != "open":
                continue
            try:
                entry_dt = datetime.fromisoformat(trade["entry_time"])
                age_min = (now - entry_dt).total_seconds() / 60
            except:
                age_min = 999
            if age_min > 18:
                nid = trade.get("market_numeric_id")
                if nid:
                    try:
                        async with httpx.AsyncClient(timeout=8) as cl:
                            r = await cl.get(f"https://gamma-api.polymarket.com/markets/{nid}",
                                           headers={"User-Agent": "Mozilla/5.0"})
                            if r.status_code == 200:
                                rm = r.json()
                                up_p, down_p = self.get_market_prices(rm)
                                if up_p is not None:
                                    price = up_p if trade["side"] == "UP" else down_p
                                    if price >= 0.95:
                                        exit_val = trade["size_usd"] / trade["entry_price"]
                                    elif price <= 0.05:
                                        exit_val = 0
                                    else:
                                        exit_val = (trade["size_usd"] / trade["entry_price"]) * price
                                    trade["exit_price"] = price
                                    trade["exit_time"] = now.isoformat()
                                    trade["pnl"] = exit_val - trade["size_usd"]
                                    trade["status"] = "closed"
                                    won = trade["pnl"] > 0
                                    self.spillover_stats["wins" if won else "losses"] += 1
                                    self.spillover_stats["pnl"] += trade["pnl"]
                                    self.spillover_stats["trades"].append({
                                        "side": trade["side"], "entry": trade["entry_price"],
                                        "exit": price, "pnl": trade["pnl"],
                                        "strategy": trade.get("strategy", "spillover"),
                                        "asset": trade.get("asset", "?"), "time": trade["entry_time"],
                                        "leader": trade.get("leader_asset", "?"),
                                    })
                                    open_spill -= 1
                    except Exception:
                        pass
                else:
                    trade["status"] = "closed"
                    trade["pnl"] = -trade["size_usd"]
                    self.spillover_stats["losses"] += 1
                    self.spillover_stats["pnl"] += trade["pnl"]
                    open_spill -= 1

        # --- ENTRY: Detect cross-asset spillover opportunities ---
        if open_spill >= self.SPILLOVER_MAX_CONCURRENT:
            self._save()
            return

        betas, asset_returns = self._compute_asset_betas()
        if not betas:
            return

        # Check each asset as potential "leader" (the one that moved)
        for leader_asset, leader_data in asset_data.items():
            candles = leader_data.get("candles", [])
            if len(candles) < 12:
                continue
            price_now = candles[-1].close if hasattr(candles[-1], 'close') else float(candles[-1].get("c", 0))
            price_10m = candles[-10].close if hasattr(candles[-10], 'close') else float(candles[-10].get("c", 0))
            if price_10m == 0:
                continue
            delta_leader = (price_now - price_10m) / price_10m

            # Leader must move significantly
            if abs(delta_leader) < self.SPILLOVER_MIN_LEADER_MOVE:
                continue

            # Check each follower asset
            for follower_asset, follower_data in asset_data.items():
                if follower_asset == leader_asset:
                    continue

                pair_key = (leader_asset, follower_asset)
                if pair_key not in betas:
                    continue

                beta_info = betas[pair_key]
                beta = beta_info["beta"]
                sigma_j = beta_info["sigma_j"]
                if sigma_j < 1e-8:
                    continue

                # Expected move in follower (Binance spot)
                expected_delta = beta * delta_leader

                # Actual move in follower (Binance spot)
                f_candles = follower_data.get("candles", [])
                if len(f_candles) < 12:
                    continue
                f_now = f_candles[-1].close if hasattr(f_candles[-1], 'close') else float(f_candles[-1].get("c", 0))
                f_10m = f_candles[-10].close if hasattr(f_candles[-10], 'close') else float(f_candles[-10].get("c", 0))
                if f_10m == 0:
                    continue
                actual_delta = (f_now - f_10m) / f_10m

                # Residual: how much follower diverged from prediction
                residual = actual_delta - expected_delta
                z_score = residual / sigma_j

                if abs(z_score) < self.SPILLOVER_K_THRESHOLD:
                    continue

                # Follower is mispriced! Determine direction:
                # If expected_delta < 0 (should drop) but actual didn't drop enough (residual > 0):
                #   → follower odds are stale-high → buy DOWN
                # If expected_delta > 0 (should rise) but actual didn't rise enough (residual < 0):
                #   → follower odds are stale-low → buy UP
                if residual > 0:
                    predicted_side = "DOWN"  # Follower should have dropped more
                else:
                    predicted_side = "UP"    # Follower should have risen more

                # Find the follower's nearest eligible market
                for mkt in markets:
                    mkt_asset = mkt.get("_asset", "")
                    if mkt_asset != follower_asset:
                        continue
                    time_left = self.get_time_remaining(mkt)
                    if time_left < self.SPILLOVER_TIME_WINDOW[0] or time_left > self.SPILLOVER_TIME_WINDOW[1]:
                        continue
                    up_p, down_p = self.get_market_prices(mkt)
                    if up_p is None or down_p is None:
                        continue

                    entry_price = up_p if predicted_side == "UP" else down_p
                    if entry_price < 0.10 or entry_price > 0.60:
                        continue
                    entry_price = round(entry_price + 0.02, 2)  # Spread simulation

                    cid = mkt.get("conditionId", "")
                    nid = mkt.get("id")
                    question = mkt.get("question", "")
                    trade_key = f"spill_{cid}_{predicted_side}"

                    if trade_key in self.spillover_trades:
                        continue

                    self.spillover_trades[trade_key] = {
                        "side": predicted_side, "entry_price": entry_price,
                        "size_usd": self.SPILLOVER_SIZE,
                        "entry_time": now.isoformat(), "condition_id": cid,
                        "market_numeric_id": nid, "title": question,
                        "asset": follower_asset, "strategy": "spillover",
                        "leader_asset": leader_asset, "beta": round(beta, 4),
                        "delta_leader": round(delta_leader, 6),
                        "expected_delta": round(expected_delta, 6),
                        "actual_delta": round(actual_delta, 6),
                        "residual": round(residual, 6), "z_score": round(z_score, 2),
                        "status": "open", "pnl": 0.0,
                    }
                    open_spill += 1
                    print(f"[SPILLOVER ENTRY] {follower_asset} {predicted_side} ${entry_price:.2f} | "
                          f"leader={leader_asset} moved {delta_leader:+.3%} | "
                          f"beta={beta:.2f} expected={expected_delta:+.3%} actual={actual_delta:+.3%} | "
                          f"residual={residual:+.3%} z={z_score:.1f} | "
                          f"time={time_left:.1f}m | {question[:40]}")

                    if open_spill >= self.SPILLOVER_MAX_CONCURRENT:
                        break
                if open_spill >= self.SPILLOVER_MAX_CONCURRENT:
                    break
            if open_spill >= self.SPILLOVER_MAX_CONCURRENT:
                break

        self._save()

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

        self._last_signal = main_signal  # V3.18: Store for 5m TA scoring
        self.signals_count += 1

        # Sort markets by time remaining - trade nearest per asset
        # V3.16: 5m markets have tighter time windows (1-4 min sweet spot)
        eligible_markets = []
        for market in markets:
            up_price, down_price = self.get_market_prices(market)
            time_left = self.get_time_remaining(market)
            if up_price is None or down_price is None:
                continue
            tf = market.get("_timeframe", "15m")
            if tf == "5m":
                # 5m markets: trade 1-4 min before expiry (whale pattern: rapid entry near open)
                if time_left > 4.5 or time_left < 1.0:
                    continue
            else:
                # 15m markets: existing proven window
                if time_left > 9 or time_left < self.MIN_TIME_REMAINING:
                    continue
            eligible_markets.append((time_left, market, up_price, down_price))

        # Sort by time remaining (ascending) - nearest expiring first
        eligible_markets.sort(key=lambda x: x[0])

        if eligible_markets:
            print(f"[Filter] {len(eligible_markets)} markets in window, trading nearest ({eligible_markets[0][0]:.1f}min left)")

        # V3.16: Trade nearest expiring market PER ASSET+TIMEFRAME
        # Can trade BTC-5m AND BTC-15m simultaneously (different markets!)
        # Whale pattern: Canine-Commandment trades both 5m and 15m BTC concurrently
        nearest_per_asset = {}
        for time_left, market, up_price, down_price in eligible_markets:
            asset = market.get("_asset", "BTC")
            tf = market.get("_timeframe", "15m")
            asset_tf_key = f"{asset}_{tf}"  # e.g., "BTC_5m", "BTC_15m"
            if asset_tf_key not in nearest_per_asset:
                nearest_per_asset[asset_tf_key] = (time_left, market, up_price, down_price)

        # === HYDRA STRATEGY QUARANTINE SCAN (V3.14) ===
        # Run 10 adapted Hyperliquid strategies on each asset's candle data
        # Record shadow predictions for quarantine evaluation
        if HYDRA_AVAILABLE and hasattr(self, '_asset_data'):
            try:
                hydra_signals = scan_strategies(
                    {asset: data.get("candles", []) for asset, data in self._asset_data.items()}
                )
                if hydra_signals:
                    for hsig in hydra_signals:
                        sig_asset = hsig.asset
                        if not sig_asset:
                            continue
                        # V3.16: Find any market for this asset (prefer 15m, fallback 5m)
                        hydra_market_key = None
                        for tf_pref in ["15m", "5m"]:
                            candidate_key = f"{sig_asset}_{tf_pref}"
                            if candidate_key in nearest_per_asset:
                                hydra_market_key = candidate_key
                                break
                        if not hydra_market_key:
                            continue
                        mkt_time, mkt, mkt_up, mkt_down = nearest_per_asset[hydra_market_key]
                        mkt_id = mkt.get("conditionId", "")
                        pred_key = f"hydra_{hsig.name}_{sig_asset}_{hsig.direction}_{mkt_id}"
                        if pred_key in self.hydra_pending:
                            continue
                        # Record shadow prediction (both filtered and unfiltered for A/B)
                        self.hydra_pending[pred_key] = {
                            "strategy": hsig.name,
                            "asset": sig_asset,
                            "direction": hsig.direction,
                            "confidence": hsig.confidence,
                            "details": hsig.details,
                            "market_id": mkt_id,
                            "market_numeric_id": mkt.get("id"),
                            "market_title": mkt.get("question", "")[:80],
                            "up_price": mkt_up,
                            "down_price": mkt_down,
                            "entry_time": datetime.now(timezone.utc).isoformat(),
                            "time_left": mkt_time,
                            "status": "open",
                            "filtered": hsig.filtered,
                            "filter_reason": hsig.filter_reason,
                        }
                        tag = " [FILTERED]" if hsig.filtered else ""
                        reason = f" ({hsig.filter_reason})" if hsig.filter_reason else ""
                        print(f"[HYDRA] {hsig.name} -> {sig_asset} {hsig.direction} (conf={hsig.confidence:.0%}){tag}{reason} | {hsig.details}")
            except Exception as e:
                print(f"[HYDRA] Scan error: {e}")

        # === ARBITRAGE SCAN ===
        # Check ALL eligible markets for both-sides arb (UP + DOWN < $0.97)
        self._scan_arbitrage(eligible_markets)

        # === FLASH CRASH PRICE RECORDING (V3.20) ===
        # Record Polymarket odds for flash crash detection across all shadow systems
        self._record_market_prices(markets)

        # === 5M SHADOW PAPER TRADING (V3.18) ===
        # Separate momentum-based system for 5-minute markets
        # Bypasses the heavy 15m filter pipeline entirely
        await self._trade_5m_shadow(markets, candles, btc_price)

        # === 15M MOMENTUM SHADOW PAPER TRADING (V3.20) ===
        # Same momentum strategy as 5m, applied to 15m markets for A/B comparison
        await self._trade_15m_momentum_shadow(markets, candles, btc_price)

        # === CROSS-MARKET SPILLOVER ARBITRAGE (V3.21) ===
        # BTC moves → SOL/ETH odds should adjust. If they haven't, buy the lag.
        await self._trade_spillover_arb(markets)

        # === DIRECTIONAL TRADING (Phase 1: Collect candidates) ===
        # Gather all trades that pass filters, then allocate via multi-outcome Kelly
        trade_candidates = []
        for asset_tf_key, (time_left, market, up_price, down_price) in nearest_per_asset.items():
            # V3.16: Extract asset from composite key (e.g., "BTC_5m" -> "BTC")
            asset = market.get("_asset", asset_tf_key.split("_")[0])
            market_tf = market.get("_timeframe", "15m")
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

            # === V3.5 SYSTEMATIC FEATURE CALCULATIONS ===
            current_hour = datetime.now(timezone.utc).hour

            # Feature 1: BTC Overnight Seasonality — boost model confidence
            overnight_boost = self._check_overnight_seasonality(current_hour, asset, signal.side or "")
            if overnight_boost > 0 and signal.side == "UP":
                signal.model_up = min(0.95, signal.model_up + overnight_boost)
                print(f"[V3.5 OVERNIGHT] BTC UP boost +{overnight_boost:.0%} -> model_up={signal.model_up:.0%}")

            # Feature 3: Multi-Timeframe Confirmation (pre-compute for later)
            mtf_agrees = self._check_mtf_confirmation(asset_candles, signal.side or "UP")

            # Feature 4: Volume Spike (pre-compute for later)
            vol_spike_status = self._check_volume_spike(asset_candles)

            # Feature 5: Vol Regime Routing — boost/penalize confidence
            vol_regime = self._get_vol_regime(asset_candles)
            regime_value = signal.regime.value if hasattr(signal, 'regime') and signal.regime else "range_bound"
            volregime_boost = self._get_volregime_boost(vol_regime, regime_value)
            if volregime_boost != 0 and signal.side:
                if signal.side == "UP":
                    signal.model_up = max(0.01, min(0.99, signal.model_up + volregime_boost))
                else:
                    signal.model_down = max(0.01, min(0.99, signal.model_down + volregime_boost))
                if abs(volregime_boost) > 0.01:
                    print(f"[V3.5 VOLREGIME] {vol_regime} vol + {regime_value} -> boost {volregime_boost:+.0%}")

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
                # V3.5b: Block ALL UP < $0.20 — SOL UP@$0.21 lost -$20 (worst single loss)
                if up_price < 0.20:
                    skip_reason = f"UP_ultra_cheap_{up_price:.2f}_(V3.5b_block)"
                # TREND FILTER: Don't take ultra-cheap UP (<$0.15) during downtrends
                elif up_price < 0.15 and hasattr(signal, 'regime') and signal.regime.value == 'trend_down':
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
                # V3.17: FULL death zone restored — CSV $0.40-0.50 = -$183, -30% ROI
                if 0.38 <= down_price < 0.50:
                    skip_reason = f"DOWN_DEATH_ZONE_{down_price:.2f}_(CSV:-30%ROI)"
                # V3.17: Cheap DOWN block at $0.30 — CSV <$0.30 = 24% WR
                elif down_price < 0.30:
                    skip_reason = f"DOWN_cheap_{down_price:.2f}_(CSV:24%WR)"
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

            # === V3.5b: MULTI-TIMEFRAME CONFIRMATION GATE (Feature 3) ===
            # ML finding: MTF disagree = 0W/3L (-$17.23). HARD BLOCK, not just penalty.
            if signal.action == "ENTER" and signal.side and not mtf_agrees and self.MTF_ENABLED:
                print(f"[V3.5b MTF BLOCK] {asset} {signal.side} — short/long TFs disagree → SKIP (was 0W/3L)")
                signal.action = "NO_TRADE"
                signal.reason = "mtf_disagree_block"

            # Collect candidate if signal passes all filters
            if signal.action == "ENTER" and signal.side and trade_key:
                if trade_key not in self.trades:
                    mid_price = up_price if signal.side == "UP" else down_price
                    # Simulate realistic fill: offset +$0.03 toward ask to match live spread
                    entry_price = round(mid_price + 0.03, 2)

                    # V3.17: TIGHTENED to 0.45 — CSV $0.40-0.50 = -30% ROI, $0.50+ = 55% but thin edge
                    MAX_ACTUAL_ENTRY = 0.45
                    if entry_price > MAX_ACTUAL_ENTRY:
                        print(f"[V3.17] {asset} {signal.side} entry ${entry_price:.2f} > ${MAX_ACTUAL_ENTRY} after spread — SKIP")
                        continue

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

                    # === V3.19: BTC ML PREDICTOR (Binance-trained) ===
                    # Adds volatility regime and momentum quality to sizing
                    btc_ml_boost = 1.0
                    btc_ml_str = ""
                    if self._ml_initialized and self.ml_15m and self.ml_15m.is_trained:
                        try:
                            btc_pred = self.ml_15m.predict(self.ml_15m.candles_raw)
                            btc_ml_str = f"[BTC-ML:q={btc_pred.momentum_quality:.2f} v={btc_pred.vol_regime}]"

                            # Momentum quality boost/reduction
                            if btc_pred.momentum_quality > 0.6:
                                btc_ml_boost = 1.2  # High quality momentum → bigger position
                            elif btc_pred.momentum_quality < 0.3:
                                btc_ml_boost = 0.7  # Low quality → reduce exposure

                            # Vol regime adjustment
                            if btc_pred.vol_regime == "high":
                                btc_ml_boost *= 1.1  # Big moves = good for Polymarket
                            elif btc_pred.vol_regime == "low":
                                btc_ml_boost *= 0.8  # Low vol = prices stay near 0.50

                            ml_str += f" {btc_ml_str}"
                        except Exception:
                            pass

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

                    # === V3.5 SYSTEMATIC SIZING ADJUSTMENTS ===
                    v35_mults = []

                    # Feature 2: Inverse Volatility Sizing
                    invvol_scale = self._calc_inverse_vol_scale(asset_candles)
                    if invvol_scale != 1.0:
                        individual_size *= invvol_scale
                        v35_mults.append(f"InvVol:{invvol_scale:.2f}x")
                        self.systematic_stats["invvol"]["total_scale"] += invvol_scale
                        self.systematic_stats["invvol"]["trades_scaled"] += 1

                    # Feature 3: MTF Disagreement Penalty
                    if not mtf_agrees and self.MTF_ENABLED:
                        individual_size *= self.MTF_PENALTY
                        v35_mults.append(f"MTF:{self.MTF_PENALTY:.0%}")

                    # Feature 4: Volume Spike Boost/Penalty
                    if vol_spike_status == "spike":
                        individual_size *= self.VOLSPIKE_BOOST
                        v35_mults.append(f"VolSpike:{self.VOLSPIKE_BOOST:.1f}x")
                    elif vol_spike_status == "low":
                        individual_size *= self.VOLSPIKE_LOW_PENALTY
                        v35_mults.append(f"VolLow:{self.VOLSPIKE_LOW_PENALTY:.1f}x")

                    # Feature 5: Vol Regime Mismatch Penalty (confidence already boosted above)
                    volregime_match = (vol_regime == "high" and regime_value == "range_bound") or \
                                     (vol_regime == "low" and regime_value in ("trend_up", "trend_down")) or \
                                     vol_regime == "normal"
                    if not volregime_match and self.VOLREGIME_ENABLED:
                        individual_size *= self.VOLREGIME_MISMATCH_PENALTY
                        v35_mults.append(f"VolMismatch:{self.VOLREGIME_MISMATCH_PENALTY:.0%}")

                    # V3.16 Feature 6: Asset-Priority Sizing (whale-informed)
                    # Canine-Commandment: BTC=$800-2600, ETH=$150-830, SOL/XRP=$7-280
                    # Bidou28old: BTC heaviest, then ETH, then SOL
                    ASSET_SIZE_MULT = {"BTC": 1.3, "ETH": 1.0, "SOL": 0.8, "XRP": 0.7}
                    asset_mult = ASSET_SIZE_MULT.get(asset, 1.0)
                    if asset_mult != 1.0:
                        individual_size *= asset_mult
                        v35_mults.append(f"Asset:{asset}x{asset_mult}")

                    # V3.16 Feature 7: 5m market sizing (smaller due to shorter window, faster resolution)
                    if market_tf == "5m":
                        individual_size *= 0.7  # Conservative: 5m has less signal clarity
                        v35_mults.append("5m:0.7x")

                    # V3.19: BTC ML Predictor boost/reduction
                    if btc_ml_boost != 1.0:
                        individual_size *= btc_ml_boost
                        v35_mults.append(f"BTC-ML:{btc_ml_boost:.2f}x")

                    if v35_mults:
                        print(f"[V3.5] {' | '.join(v35_mults)} -> size=${individual_size:.2f}")

                    individual_size = max(5.0, individual_size)

                    # Build feature tags for this trade (for resolution tracking)
                    trade_feature_tags = {
                        "overnight": overnight_boost > 0,
                        "mtf_agree": mtf_agrees,
                        "volspike": vol_spike_status,
                        "volregime_match": volregime_match,
                        "vol_regime": vol_regime,
                        "invvol_scale": round(invvol_scale, 2),
                        "market_tf": market_tf,  # V3.16: 5m or 15m
                    }

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
                        'feature_tags': trade_feature_tags,
                        'market_tf': market_tf,  # V3.16: 5m or 15m
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

                        # === V3.5: UPDATE SYSTEMATIC FEATURE STATS ===
                        self._update_systematic_stats(tid, won, trade.pnl)
                        self._check_feature_revoke()

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
                        # V3.5: Update systematic feature stats
                        self._update_systematic_stats(tid, won2, trade.pnl)
                        self._check_feature_revoke()

                        result = "WIN" if won2 else "LOSS"
                        print(f"[{result}] {trade.side} PnL: ${trade.pnl:+.2f} | Day: ${self.daily_pnl:+.2f} | {trade.market_title[:35]}...")

                # Also resolve arb trades on this market
                for arb_key, arb in list(self.arb_trades.items()):
                    if arb["status"] == "open" and arb.get("condition_id") == mkt_id:
                        self._resolve_arb_trade(arb, up_p, down_p)

                # Resolve hydra strategy predictions on this market
                self._resolve_hydra_predictions(mkt_id, up_p, down_p)

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
                                # V3.5: Update systematic feature stats
                                self._update_systematic_stats(tid, won3, open_trade.pnl)
                                self._check_feature_revoke()

                                result = "WIN" if won3 else "LOSS"
                                print(f"[{result}] (expired) {open_trade.side} PnL: ${open_trade.pnl:+.2f} | Day: ${self.daily_pnl:+.2f} | {open_trade.market_title[:30]}...")
                    await asyncio.sleep(0.5)  # Rate limit between resolved market lookups
                except Exception as e:
                    print(f"[Expiry] Error resolving {tid[:20]}: {e}")

        # === HYDRA STALE PREDICTION RESOLUTION ===
        # Resolve hydra predictions older than 16 minutes via Gamma API
        for pkey, pred in list(self.hydra_pending.items()):
            if pred.get("status") != "open":
                continue
            try:
                entry_dt = datetime.fromisoformat(pred["entry_time"])
            except Exception:
                continue
            age_minutes = (now - entry_dt).total_seconds() / 60
            if age_minutes > 16:
                nid = pred.get("market_numeric_id")
                if not nid:
                    pred["status"] = "expired"
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
                                self._resolve_hydra_predictions(pred["market_id"], up_p, down_p)
                    await asyncio.sleep(0.3)
                except Exception:
                    pass

        # === HYDRA PENDING CLEANUP ===
        # Remove resolved/expired entries older than 1 hour to prevent memory bloat
        cleanup_cutoff = now - timedelta(hours=1)
        for pkey in [k for k, v in self.hydra_pending.items()
                     if v.get("status") in ("resolved", "expired")]:
            try:
                entry_dt = datetime.fromisoformat(self.hydra_pending[pkey]["entry_time"])
                if entry_dt < cleanup_cutoff:
                    del self.hydra_pending[pkey]
            except Exception:
                del self.hydra_pending[pkey]

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

                # Save V3.5 feature tags for this trade
                if 'feature_tags' in cand:
                    self._trade_features[cand['trade_key']] = cand['feature_tags']

                momentum_str = f"Mom:{cand['momentum']:+.2%}" if cand['momentum'] else ""
                nyu_info = ""
                if self.use_nyu_model:
                    nyu_r = self.nyu_model.calculate_volatility(cand['entry_price'], cand['time_left'])
                    nyu_info = f"NYU:{nyu_r.edge_score:.2f}"

                tf_tag = f"[{cand.get('market_tf', '15m')}]" if cand.get('market_tf') == '5m' else ""
                print(f"[NEW] {tf_tag}{side} @ ${cand['entry_price']:.4f} | Edge: {cand['edge']:.1%} | {momentum_str} | {nyu_info} | {meta_str} {kelly_str}")
                print(f"      Size: ${optimal_size:.2f} | KL: {cand['bregman_signal'].kl_divergence:.4f} | {cand['ml_str']} | T-{cand['time_left']:.1f}min | {cand['question'][:40]}...")

                cycle_exposure += optimal_size
                cycle_same_dir[side] = n_same + 1

            if len(trade_candidates) > 1:
                print(f"[KELLY] {len(trade_candidates)} candidates | Allocated ${cycle_exposure:.2f}/${self.MAX_CYCLE_BUDGET:.0f} budget | Sides: {dict(cycle_same_dir)}")

        self._save()
        return main_signal

    def _resolve_hydra_predictions(self, market_id: str, up_price: float, down_price: float):
        """Resolve hydra strategy shadow predictions for a settled market.
        V3.14d: A/B tracking — updates BOTH filtered (quarantine) and raw stats."""
        for pkey, pred in list(self.hydra_pending.items()):
            if pred.get("status") != "open":
                continue
            if pred.get("market_id") != market_id:
                continue

            direction = pred["direction"]
            strat_name = pred["strategy"]
            is_filtered = pred.get("filtered", False)
            filter_reason = pred.get("filter_reason", "")

            # Determine if prediction was correct
            if direction == "UP":
                correct = up_price >= 0.95
            else:
                correct = down_price >= 0.95

            if up_price < 0.95 and down_price < 0.95:
                continue

            # Calculate shadow PnL (as if we bet $10 at entry)
            entry_p = pred.get("up_price" if direction == "UP" else "down_price", 0.50)
            if entry_p <= 0:
                entry_p = 0.50
            shadow_pnl = (10.0 / entry_p) - 10.0 if correct else -10.0

            pred["status"] = "resolved"
            pred["correct"] = correct
            pred["shadow_pnl"] = shadow_pnl

            # === A/B TRACKING: Always update RAW stats (all signals) ===
            if strat_name in self.hydra_quarantine_raw:
                qr = self.hydra_quarantine_raw[strat_name]
                qr["predictions"] += 1
                if correct:
                    qr["correct"] += 1
                else:
                    qr["wrong"] += 1
                qr["pnl"] += shadow_pnl

            # === Update per-filter tracking ===
            if is_filtered and filter_reason:
                fkey = filter_reason.split("(")[0]  # e.g. "RSI_UP_VETO"
                if fkey not in self.hydra_filter_stats:
                    self.hydra_filter_stats[fkey] = {"blocked": 0, "would_win": 0, "would_lose": 0, "saved_pnl": 0.0}
                fs = self.hydra_filter_stats[fkey]
                fs["blocked"] += 1
                if correct:
                    fs["would_win"] += 1
                    fs["saved_pnl"] -= shadow_pnl  # Filter cost us a win (negative saved)
                else:
                    fs["would_lose"] += 1
                    fs["saved_pnl"] += abs(shadow_pnl)  # Filter saved us from a loss

            # === FILTERED stats: Only count non-filtered signals for promotion ===
            if not is_filtered and strat_name in self.hydra_quarantine:
                q = self.hydra_quarantine[strat_name]
                q["predictions"] += 1
                if correct:
                    q["correct"] += 1
                else:
                    q["wrong"] += 1
                q["pnl"] += shadow_pnl

                wr = q["correct"] / max(1, q["predictions"]) * 100
                result = "CORRECT" if correct else "WRONG"
                print(f"[HYDRA {result}] {strat_name} {pred['asset']} {direction} | Shadow: ${shadow_pnl:+.2f} | Running: {q['predictions']}T {wr:.0f}%WR ${q['pnl']:+.2f}")

                # Continuous promotion/demotion
                if q["predictions"] >= self.HYDRA_MIN_TRADES and q["predictions"] % 5 == 0:
                    wr_frac = q["correct"] / q["predictions"]
                    old_status = q["status"]
                    if wr_frac >= self.HYDRA_PROMOTE_WR and q["pnl"] > 0:
                        q["status"] = "PROMOTED"
                        if old_status != "PROMOTED":
                            print(f"[HYDRA PROMOTE] {strat_name} -> PROMOTED! ({q['predictions']}T {wr:.0f}%WR ${q['pnl']:+.2f})")
                    else:
                        q["status"] = "DEMOTED"
                        if old_status != "DEMOTED":
                            print(f"[HYDRA DEMOTE] {strat_name} -> DEMOTED ({q['predictions']}T {wr:.0f}%WR ${q['pnl']:+.2f})")
            elif is_filtered:
                # Log filtered signal result for A/B visibility
                result = "CORRECT" if correct else "WRONG"
                print(f"[HYDRA {result}] {strat_name} {pred['asset']} {direction} [FILTERED:{filter_reason}] | Shadow: ${shadow_pnl:+.2f} (A/B only)")

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

        # V3.5 Systematic Trading Features
        print(f"\nSYSTEMATIC FEATURES V3.5:")
        # Overnight
        ov = self.systematic_stats["overnight"]
        ov_wr = ov["wins"] / max(1, ov["trades"]) * 100 if ov["trades"] else 0
        ov_status = "ON" if self.OVERNIGHT_ENABLED else "REVOKED"
        print(f"  Overnight BTC UP: {ov_status} | {ov['trades']}T {ov_wr:.0f}%WR ${ov['pnl']:+.2f} | Boost: +{self.OVERNIGHT_BTC_BOOST:.0%}")
        # InvVol
        iv = self.systematic_stats["invvol"]
        iv_avg = iv["total_scale"] / max(1, iv["trades_scaled"])
        iv_status = "ON" if self.INVVOL_ENABLED else "REVOKED"
        print(f"  Inverse Vol Size: {iv_status} | {iv['trades_scaled']}T scaled | Avg scale: {iv_avg:.2f}x | Clamp: [{self.INVVOL_CLAMP[0]:.1f}, {self.INVVOL_CLAMP[1]:.1f}]")
        # MTF
        ma = self.systematic_stats["mtf_agree"]
        md = self.systematic_stats["mtf_disagree"]
        ma_wr = ma["wins"] / max(1, ma["trades"]) * 100 if ma["trades"] else 0
        md_wr = md["wins"] / max(1, md["trades"]) * 100 if md["trades"] else 0
        mtf_status = "ON" if self.MTF_ENABLED else "REVOKED"
        print(f"  MTF Confirm:  {mtf_status} | Agree: {ma['trades']}T {ma_wr:.0f}%WR ${ma['pnl']:+.2f} | Disagree: {md['trades']}T {md_wr:.0f}%WR ${md['pnl']:+.2f}")
        # VolSpike
        vs = self.systematic_stats["volspike"]
        vn = self.systematic_stats["volspike_normal"]
        vs_wr = vs["wins"] / max(1, vs["trades"]) * 100 if vs["trades"] else 0
        vn_wr = vn["wins"] / max(1, vn["trades"]) * 100 if vn["trades"] else 0
        volspike_status = "ON" if self.VOLSPIKE_ENABLED else "REVOKED"
        print(f"  Vol Spike:    {volspike_status} | Spike: {vs['trades']}T {vs_wr:.0f}%WR ${vs['pnl']:+.2f} | Normal: {vn['trades']}T {vn_wr:.0f}%WR ${vn['pnl']:+.2f}")
        # VolRegime
        vrm = self.systematic_stats["volregime_match"]
        vrmm = self.systematic_stats["volregime_mismatch"]
        vrm_wr = vrm["wins"] / max(1, vrm["trades"]) * 100 if vrm["trades"] else 0
        vrmm_wr = vrmm["wins"] / max(1, vrmm["trades"]) * 100 if vrmm["trades"] else 0
        volreg_status = "ON" if self.VOLREGIME_ENABLED else "REVOKED"
        print(f"  Vol Regime:   {volreg_status} | Match: {vrm['trades']}T {vrm_wr:.0f}%WR ${vrm['pnl']:+.2f} | Mismatch: {vrmm['trades']}T {vrmm_wr:.0f}%WR ${vrmm['pnl']:+.2f}")

        # Hydra Strategy Quarantine (V3.14d A/B tracking)
        hydra_open = sum(1 for p in self.hydra_pending.values() if p.get("status") == "open")
        hydra_filtered_total = sum(q["predictions"] for q in self.hydra_quarantine.values())
        hydra_raw_total = sum(q["predictions"] for q in self.hydra_quarantine_raw.values())
        print(f"\nHYDRA STRATEGY QUARANTINE ({len(self.HYDRA_STRATEGIES)} strategies, {hydra_filtered_total} filtered / {hydra_raw_total} raw, {hydra_open} pending):")
        print(f"  {'STRATEGY':25s} {'STATUS':12s} | {'FILTERED (acted on)':28s} | {'RAW (A/B comparison)':28s}")
        for name in self.HYDRA_STRATEGIES:
            q = self.hydra_quarantine.get(name, {"predictions": 0, "correct": 0, "wrong": 0, "pnl": 0.0, "status": "QUARANTINE"})
            qr = self.hydra_quarantine_raw.get(name, {"predictions": 0, "correct": 0, "wrong": 0, "pnl": 0.0})
            n = q["predictions"]
            nr = qr["predictions"]
            if n == 0 and nr == 0:
                print(f"  {name:25s} {'QUARANTINE':12s} | 0 predictions (waiting...)")
            else:
                # Filtered stats
                wr = q["correct"] / n * 100 if n > 0 else 0
                progress = f"{n}/{self.HYDRA_MIN_TRADES}" if q.get("status") == "QUARANTINE" else f"{n}T"
                f_str = f"{progress} {wr:.0f}%WR ${q['pnl']:+.2f}" if n > 0 else "no signals pass"
                # Raw stats
                wr_r = qr["correct"] / nr * 100 if nr > 0 else 0
                r_str = f"{nr}T {wr_r:.0f}%WR ${qr['pnl']:+.2f}" if nr > 0 else "-"
                print(f"  {name:25s} {q.get('status', 'Q'):12s} | {f_str:28s} | {r_str:28s}")

        # Filter effectiveness report
        if self.hydra_filter_stats:
            print(f"\n  FILTER A/B REPORT:")
            for fname, fs in sorted(self.hydra_filter_stats.items()):
                net = fs["saved_pnl"]
                verdict = "SAVING $$$" if net > 0 else "COSTING $$$"
                print(f"    {fname:25s} blocked={fs['blocked']} | would_win={fs['would_win']} would_lose={fs['would_lose']} | net=${net:+.2f} ({verdict})")

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

        # 5M Shadow Stats
        s5 = self.shadow_5m_stats
        open_5m = sum(1 for t in self.shadow_5m_trades.values() if t.get("status") == "open")
        total_5m = s5["wins"] + s5["losses"]
        wr_5m = s5["wins"] / total_5m * 100 if total_5m > 0 else 0
        ml_tag = ""
        if self._ml_initialized and self.ml_5m and self.ml_5m.is_trained:
            p = self.ml_5m
            ml_tag = f" | ML: {p.predictions_made}pred {p.correct_predictions}correct"
        print(f"\n5M SHADOW PAPER (V3.19 ML):")
        print(f"  {s5['wins']}W/{s5['losses']}L ({total_5m}T) | {wr_5m:.0f}% WR | PnL: ${s5['pnl']:+.2f} | {open_5m} open{ml_tag}")
        # Per-strategy breakdown
        strat_stats = {}
        for t in s5.get("trades", []):
            st = t.get("strategy", "unknown")
            if st not in strat_stats:
                strat_stats[st] = {"w": 0, "l": 0, "pnl": 0.0}
            if t.get("pnl", 0) > 0:
                strat_stats[st]["w"] += 1
            else:
                strat_stats[st]["l"] += 1
            strat_stats[st]["pnl"] += t.get("pnl", 0)
        for st, ss in sorted(strat_stats.items(), key=lambda x: -x[1]["pnl"]):
            tot = ss["w"] + ss["l"]
            wr = ss["w"] / tot * 100 if tot > 0 else 0
            print(f"    {st:20s} {ss['w']}W/{ss['l']}L {wr:.0f}%WR ${ss['pnl']:+.2f}")

        # 15M Momentum Shadow Stats (V3.20)
        s15m = self.shadow_15m_mom_stats
        open_15m = sum(1 for t in self.shadow_15m_mom_trades.values() if t.get("status") == "open")
        total_15m = s15m["wins"] + s15m["losses"]
        wr_15m = s15m["wins"] / total_15m * 100 if total_15m > 0 else 0
        print(f"\n15M MOMENTUM SHADOW (V3.20 A/B test vs 5m):")
        print(f"  {s15m['wins']}W/{s15m['losses']}L ({total_15m}T) | {wr_15m:.0f}% WR | PnL: ${s15m['pnl']:+.2f} | {open_15m} open")
        # Per-strategy breakdown
        strat_15m = {}
        for t in s15m.get("trades", []):
            st = t.get("strategy", "unknown")
            if st not in strat_15m:
                strat_15m[st] = {"w": 0, "l": 0, "pnl": 0.0}
            if t.get("pnl", 0) > 0:
                strat_15m[st]["w"] += 1
            else:
                strat_15m[st]["l"] += 1
            strat_15m[st]["pnl"] += t.get("pnl", 0)
        for st, ss in sorted(strat_15m.items(), key=lambda x: -x[1]["pnl"]):
            tot = ss["w"] + ss["l"]
            wr = ss["w"] / tot * 100 if tot > 0 else 0
            print(f"    {st:20s} {ss['w']}W/{ss['l']}L {wr:.0f}%WR ${ss['pnl']:+.2f}")

        # Spillover Arb Stats (V3.21)
        sp = self.spillover_stats
        open_sp = sum(1 for t in self.spillover_trades.values() if t.get("status") == "open")
        total_sp = sp["wins"] + sp["losses"]
        wr_sp = sp["wins"] / total_sp * 100 if total_sp > 0 else 0
        print(f"\nSPILLOVER ARB (V3.21 cross-market):")
        print(f"  {sp['wins']}W/{sp['losses']}L ({total_sp}T) | {wr_sp:.0f}% WR | PnL: ${sp['pnl']:+.2f} | {open_sp} open")
        # Per-leader breakdown
        leader_stats = {}
        for t in sp.get("trades", []):
            ldr = t.get("leader", "?")
            ast = t.get("asset", "?")
            key = f"{ldr}->{ast}"
            if key not in leader_stats:
                leader_stats[key] = {"w": 0, "l": 0, "pnl": 0.0}
            if t.get("pnl", 0) > 0:
                leader_stats[key]["w"] += 1
            else:
                leader_stats[key]["l"] += 1
            leader_stats[key]["pnl"] += t.get("pnl", 0)
        for key, ss in sorted(leader_stats.items(), key=lambda x: -x[1]["pnl"]):
            tot = ss["w"] + ss["l"]
            wr = ss["w"] / tot * 100 if tot > 0 else 0
            print(f"    {key:20s} {ss['w']}W/{ss['l']}L {wr:.0f}%WR ${ss['pnl']:+.2f}")

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
        print("TA + BREGMAN PAPER TRADER V3.5 - SYSTEMATIC TRADING FEATURES")
        print("=" * 70)
        print("Strategy: Momentum + ML ensemble + Meta-Label + Systematic V3.5")
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
        print("SYSTEMATIC V3.5 FEATURES (ML Auto-Revoke):")
        print(f"  1. BTC Overnight Seasonality: {'ON' if self.OVERNIGHT_ENABLED else 'OFF'} (22-23 UTC +{self.OVERNIGHT_BTC_BOOST:.0%} BTC UP)")
        print(f"  2. Inverse Vol Sizing: {'ON' if self.INVVOL_ENABLED else 'OFF'} (clamp [{self.INVVOL_CLAMP[0]:.1f}, {self.INVVOL_CLAMP[1]:.1f}])")
        print(f"  3. Multi-TF Confirm: {'ON' if self.MTF_ENABLED else 'OFF'} (disagree penalty: {self.MTF_PENALTY:.0%}x)")
        print(f"  4. Volume Spike: {'ON' if self.VOLSPIKE_ENABLED else 'OFF'} (threshold: {self.VOLSPIKE_THRESHOLD:.1f}x, boost: {self.VOLSPIKE_BOOST:.1f}x)")
        print(f"  5. Vol Regime Route: {'ON' if self.VOLREGIME_ENABLED else 'OFF'} (high+MR: +{self.VOLREGIME_HIGH_MR_BOOST:.0%}, low+trend: +{self.VOLREGIME_LOW_TREND_BOOST:.0%})")
        print()
        hydra_status = "ENABLED" if HYDRA_AVAILABLE else "DISABLED"
        hydra_total = sum(q["predictions"] for q in self.hydra_quarantine.values())
        hydra_raw_total = sum(q["predictions"] for q in self.hydra_quarantine_raw.values())
        promoted = [n for n, q in self.hydra_quarantine.items() if q["status"] == "PROMOTED"]
        print(f"HYDRA STRATEGY QUARANTINE V3.14d: {hydra_status} ({len(self.HYDRA_STRATEGIES)} strategies, {hydra_total} filtered / {hydra_raw_total} raw)")
        print(f"  Promotion threshold: {self.HYDRA_MIN_TRADES} trades @ {self.HYDRA_PROMOTE_WR:.0%} WR + positive PnL")
        print(f"  Filters: RSI>65 UP veto | TRENDLINE ema50+vol | Trend-beats-reversion conflict")
        print(f"  A/B tracking: ENABLED (filtered vs unfiltered for ML comparison)")
        if promoted:
            print(f"  PROMOTED: {', '.join(promoted)}")
        print()
        print("Scan: 2min | Update: 10min | ML Tune: 30min")
        print("=" * 70)
        print()

        # === V3.19: Initialize BTC ML Predictors ===
        if BTC_ML_AVAILABLE:
            try:
                self.ml_5m = BTCPredictor(timeframe='5m', bankroll=self.bankroll)
                self.ml_15m = BTCPredictor(timeframe='15m', bankroll=self.bankroll)
                ok_5m = await self.ml_5m.initialize()
                ok_15m = await self.ml_15m.initialize()
                self._ml_initialized = ok_5m or ok_15m
                print(f"\nBTC ML PREDICTORS V3.19: 5m={'OK' if ok_5m else 'FAIL'} | 15m={'OK' if ok_15m else 'FAIL'}")
                if ok_5m:
                    pred = self.ml_5m.predict(self.ml_5m.candles_raw)
                    print(f"  5m: {pred.direction} conf={pred.confidence:.1%} | "
                          f"vol={pred.vol_regime}({pred.vol_score:.2f}) | "
                          f"quality={pred.momentum_quality:.2f} | score={pred.trade_score:.2f}")
                if ok_15m:
                    pred = self.ml_15m.predict(self.ml_15m.candles_raw)
                    print(f"  15m: {pred.direction} conf={pred.confidence:.1%} | "
                          f"vol={pred.vol_regime}({pred.vol_score:.2f}) | "
                          f"quality={pred.momentum_quality:.2f} | score={pred.trade_score:.2f}")
            except Exception as e:
                print(f"[ML V3.19] Init error: {e}")
                self._ml_initialized = False
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

                # V3.19: Retrain BTC ML every 30 min
                if self._ml_initialized and self.ml_5m:
                    try:
                        retrained = await self.ml_5m.maybe_retrain()
                        if retrained:
                            print(f"[ML-5m] Retrained with fresh data")
                    except Exception:
                        pass
                if self._ml_initialized and self.ml_15m:
                    try:
                        retrained = await self.ml_15m.maybe_retrain()
                        if retrained:
                            print(f"[ML-15m] Retrained with fresh data")
                    except Exception:
                        pass

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
    from pid_lock import acquire_pid_lock, release_pid_lock
    acquire_pid_lock("ta_paper")
    try:
        trader = TAPaperTrader()
        asyncio.run(trader.run())
    finally:
        release_pid_lock("ta_paper")
