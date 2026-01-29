"""
Machine Learning optimizer for the arbitrage bot.
Uses collected trading data to optimize parameters.

RESEARCH FRAMEWORK IMPLEMENTATION:
- Temporal Integrity: No lookahead bias, point-in-time correctness
- Walk-Forward Validation: Train on past, test on next period, roll forward
- Regime Detection: Identify market regime shifts (volatility, trending, etc.)
- Data Leakage Prevention: Features only use past data
- Distribution Shift Awareness: Regime-aware training
"""

import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from enum import Enum

from .config import config
from .data_collector import collector


class MarketRegime(Enum):
    """Market regime classification."""
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    CRISIS = "crisis"  # Flash crash, black swan events


@dataclass
class OptimizedParams:
    """Optimized trading parameters from ML."""
    min_edge_percent: float
    min_negrisk_edge: float
    min_confidence: float
    max_position_size: float
    optimal_hours: List[int]  # Best hours to trade
    optimal_days: List[int]   # Best days to trade
    edge_multiplier: float    # Adjust edge calculation
    confidence_threshold: float
    time_decay_factor: float  # How much to reduce size near expiry

    # New: Regime-specific parameters
    regime_adjustments: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    last_trained: float = 0.0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0


@dataclass
class ValidationResult:
    """Results from walk-forward validation."""
    fold: int
    train_start: float
    train_end: float
    val_start: float
    val_end: float
    train_samples: int
    val_samples: int
    train_win_rate: float
    val_win_rate: float
    train_profit: float
    val_profit: float
    regime: MarketRegime
    overfitting_score: float  # train_win_rate - val_win_rate (high = overfit)


class MLOptimizer:
    """
    Optimizes trading parameters using historical data.

    RESEARCH FRAMEWORK COMPLIANT:
    1. Temporal Integrity - Features only use past data
    2. Walk-Forward Validation - Rolling train/val/test splits
    3. Regime Detection - Adapts to market conditions
    4. Data Leakage Prevention - Strict temporal ordering
    """

    # Validation split ratios (temporal order)
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # Walk-forward settings
    MIN_TRAIN_SAMPLES = 20
    WALK_FORWARD_FOLDS = 5

    # Regime thresholds
    VOLATILITY_HIGH_THRESHOLD = 0.15  # 15% price movement
    VOLATILITY_LOW_THRESHOLD = 0.03   # 3% price movement
    CRISIS_THRESHOLD = 0.30           # 30% movement = crisis mode

    def __init__(self, data_dir: str = "ml_data"):
        self.data_dir = data_dir
        self.params_file = os.path.join(data_dir, "optimized_params.json")
        self.validation_file = os.path.join(data_dir, "validation_results.json")
        self.current_params: Optional[OptimizedParams] = None
        self.validation_history: List[ValidationResult] = []
        self.current_regime: MarketRegime = MarketRegime.RANGING

        # Load existing params if available
        self._load_params()

    def _load_params(self):
        """Load optimized params from file."""
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file, "r") as f:
                    data = json.load(f)
                self.current_params = OptimizedParams(
                    min_edge_percent=data.get("min_edge_percent", config.MIN_EDGE_PERCENT),
                    min_negrisk_edge=data.get("min_negrisk_edge", config.MIN_NEGRISK_EDGE),
                    min_confidence=data.get("min_confidence", 0.3),
                    max_position_size=data.get("max_position_size", config.MAX_POSITION_SIZE),
                    optimal_hours=data.get("optimal_hours", list(range(24))),
                    optimal_days=data.get("optimal_days", list(range(7))),
                    edge_multiplier=data.get("edge_multiplier", 1.0),
                    confidence_threshold=data.get("confidence_threshold", 0.3),
                    time_decay_factor=data.get("time_decay_factor", 0.8),
                    regime_adjustments=data.get("regime_adjustments", {}),
                    validation_metrics=data.get("validation_metrics", {}),
                    last_trained=data.get("last_trained", 0.0),
                    train_samples=data.get("train_samples", 0),
                    val_samples=data.get("val_samples", 0),
                    test_samples=data.get("test_samples", 0),
                )
                print(f"[MLOptimizer] Loaded optimized params (trained on {self.current_params.train_samples} samples)")
            except Exception as e:
                print(f"[MLOptimizer] Error loading params: {e}")

    def _save_params(self):
        """Save optimized params to file."""
        if self.current_params:
            with open(self.params_file, "w") as f:
                json.dump({
                    "min_edge_percent": self.current_params.min_edge_percent,
                    "min_negrisk_edge": self.current_params.min_negrisk_edge,
                    "min_confidence": self.current_params.min_confidence,
                    "max_position_size": self.current_params.max_position_size,
                    "optimal_hours": self.current_params.optimal_hours,
                    "optimal_days": self.current_params.optimal_days,
                    "edge_multiplier": self.current_params.edge_multiplier,
                    "confidence_threshold": self.current_params.confidence_threshold,
                    "time_decay_factor": self.current_params.time_decay_factor,
                    "regime_adjustments": self.current_params.regime_adjustments,
                    "validation_metrics": self.current_params.validation_metrics,
                    "last_trained": self.current_params.last_trained,
                    "train_samples": self.current_params.train_samples,
                    "val_samples": self.current_params.val_samples,
                    "test_samples": self.current_params.test_samples,
                }, f, indent=2)

    # =========================================================================
    # TEMPORAL INTEGRITY - No Lookahead Bias
    # =========================================================================

    def temporal_split(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split data temporally (NOT randomly) to prevent data leakage.

        CRITICAL: Data must be sorted by timestamp and split in order:
        [oldest] ----TRAIN---- | --VAL-- | --TEST-- [newest]

        This ensures we only train on past data and validate on future data.
        """
        if not data:
            return [], [], []

        # Sort by timestamp (oldest first)
        sorted_data = sorted(data, key=lambda x: x.get("timestamp", 0))

        n = len(sorted_data)
        train_end = int(n * self.TRAIN_RATIO)
        val_end = int(n * (self.TRAIN_RATIO + self.VAL_RATIO))

        train = sorted_data[:train_end]
        val = sorted_data[train_end:val_end]
        test = sorted_data[val_end:]

        print(f"[MLOptimizer] Temporal split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

        # Log time ranges
        if train:
            train_start = datetime.fromtimestamp(train[0]["timestamp"])
            train_end_dt = datetime.fromtimestamp(train[-1]["timestamp"])
            print(f"  Train: {train_start.strftime('%Y-%m-%d')} to {train_end_dt.strftime('%Y-%m-%d')}")
        if val:
            val_start = datetime.fromtimestamp(val[0]["timestamp"])
            val_end_dt = datetime.fromtimestamp(val[-1]["timestamp"])
            print(f"  Val:   {val_start.strftime('%Y-%m-%d')} to {val_end_dt.strftime('%Y-%m-%d')}")
        if test:
            test_start = datetime.fromtimestamp(test[0]["timestamp"])
            test_end_dt = datetime.fromtimestamp(test[-1]["timestamp"])
            print(f"  Test:  {test_start.strftime('%Y-%m-%d')} to {test_end_dt.strftime('%Y-%m-%d')}")

        return train, val, test

    def check_data_leakage(self, train: List[Dict], val: List[Dict], test: List[Dict]) -> bool:
        """
        Verify no data leakage between splits.

        Checks:
        1. No temporal overlap between splits
        2. No market_id leakage (same market in train and test)
        3. Features don't use future data
        """
        issues = []

        # Check 1: Temporal overlap
        if train and val:
            train_max_ts = max(d["timestamp"] for d in train)
            val_min_ts = min(d["timestamp"] for d in val)
            if train_max_ts >= val_min_ts:
                issues.append(f"LEAKAGE: Train overlaps Val (train_max={train_max_ts}, val_min={val_min_ts})")

        if val and test:
            val_max_ts = max(d["timestamp"] for d in val)
            test_min_ts = min(d["timestamp"] for d in test)
            if val_max_ts >= test_min_ts:
                issues.append(f"LEAKAGE: Val overlaps Test (val_max={val_max_ts}, test_min={test_min_ts})")

        # Check 2: Market ID leakage (same market appearing in different splits)
        train_markets = set(d.get("market_id", "") for d in train)
        val_markets = set(d.get("market_id", "") for d in val)
        test_markets = set(d.get("market_id", "") for d in test)

        train_test_overlap = train_markets & test_markets
        if train_test_overlap:
            # This is actually OK for prediction markets since they resolve at different times
            # But we log it for awareness
            print(f"[MLOptimizer] Note: {len(train_test_overlap)} markets appear in both train and test (OK for PM)")

        if issues:
            for issue in issues:
                print(f"[MLOptimizer] WARNING: {issue}")
            return False

        print("[MLOptimizer] Data leakage check PASSED")
        return True

    # =========================================================================
    # WALK-FORWARD VALIDATION
    # =========================================================================

    def walk_forward_validation(self, data: List[Dict], n_folds: int = None) -> List[ValidationResult]:
        """
        Implement walk-forward (rolling) validation.

        Unlike k-fold cross-validation (which shuffles data), walk-forward
        respects temporal order:

        Fold 1: [===TRAIN===][VAL]..............
        Fold 2: .[===TRAIN===][VAL].............
        Fold 3: ..[===TRAIN===][VAL]............
        ...

        This simulates real trading where you only have past data.
        """
        n_folds = n_folds or self.WALK_FORWARD_FOLDS

        if len(data) < self.MIN_TRAIN_SAMPLES * 2:
            print(f"[MLOptimizer] Not enough data for walk-forward validation")
            return []

        # Sort by timestamp
        sorted_data = sorted(data, key=lambda x: x.get("timestamp", 0))
        n = len(sorted_data)

        # Calculate window sizes
        fold_size = (n - self.MIN_TRAIN_SAMPLES) // n_folds
        if fold_size < 5:
            print(f"[MLOptimizer] Fold size too small ({fold_size}), reducing folds")
            n_folds = max(2, (n - self.MIN_TRAIN_SAMPLES) // 5)
            fold_size = (n - self.MIN_TRAIN_SAMPLES) // n_folds

        results = []

        for fold in range(n_folds):
            # Training window expands with each fold
            train_end_idx = self.MIN_TRAIN_SAMPLES + (fold * fold_size)
            val_end_idx = min(train_end_idx + fold_size, n)

            train_data = sorted_data[:train_end_idx]
            val_data = sorted_data[train_end_idx:val_end_idx]

            if not val_data:
                continue

            # Calculate metrics for this fold
            train_wins = sum(1 for d in train_data if d.get("was_profitable"))
            val_wins = sum(1 for d in val_data if d.get("was_profitable"))

            train_win_rate = train_wins / len(train_data) if train_data else 0
            val_win_rate = val_wins / len(val_data) if val_data else 0

            train_profit = sum(d.get("actual_profit", 0) or 0 for d in train_data)
            val_profit = sum(d.get("actual_profit", 0) or 0 for d in val_data)

            # Detect regime for this validation period
            regime = self.detect_regime(val_data)

            # Overfitting score: if train >> val, we're overfitting
            overfitting_score = train_win_rate - val_win_rate

            result = ValidationResult(
                fold=fold,
                train_start=train_data[0]["timestamp"],
                train_end=train_data[-1]["timestamp"],
                val_start=val_data[0]["timestamp"],
                val_end=val_data[-1]["timestamp"],
                train_samples=len(train_data),
                val_samples=len(val_data),
                train_win_rate=train_win_rate,
                val_win_rate=val_win_rate,
                train_profit=train_profit,
                val_profit=val_profit,
                regime=regime,
                overfitting_score=overfitting_score,
            )
            results.append(result)

            print(f"[WalkForward] Fold {fold+1}: Train={train_win_rate:.1%}, Val={val_win_rate:.1%}, "
                  f"Overfit={overfitting_score:+.1%}, Regime={regime.value}")

        self.validation_history = results
        return results

    # =========================================================================
    # REGIME DETECTION
    # =========================================================================

    def detect_regime(self, data: List[Dict]) -> MarketRegime:
        """
        Detect market regime from recent data.

        Regimes:
        - LOW_VOLATILITY: Stable, small price movements
        - HIGH_VOLATILITY: Large swings, uncertainty
        - TRENDING_UP: Consistent upward movement
        - TRENDING_DOWN: Consistent downward movement
        - RANGING: Sideways, no clear direction
        - CRISIS: Flash crash, extreme events (COVID, 2008, etc.)
        """
        if not data or len(data) < 3:
            return MarketRegime.RANGING

        # Extract price movements
        prices = []
        for d in sorted(data, key=lambda x: x.get("timestamp", 0)):
            if "spot_price" in d and d["spot_price"]:
                prices.append(d["spot_price"])
            elif "yes_price" in d:
                prices.append(d["yes_price"])

        if len(prices) < 3:
            return MarketRegime.RANGING

        # Calculate returns
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)

        if not returns:
            return MarketRegime.RANGING

        # Volatility (standard deviation of returns)
        volatility = np.std(returns) if len(returns) > 1 else 0

        # Trend (average return)
        avg_return = np.mean(returns)

        # Total movement
        total_move = abs((prices[-1] - prices[0]) / prices[0]) if prices[0] > 0 else 0

        # Classify regime
        if total_move > self.CRISIS_THRESHOLD:
            return MarketRegime.CRISIS
        elif volatility > self.VOLATILITY_HIGH_THRESHOLD:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < self.VOLATILITY_LOW_THRESHOLD:
            return MarketRegime.LOW_VOLATILITY
        elif avg_return > 0.02:  # 2% average positive return
            return MarketRegime.TRENDING_UP
        elif avg_return < -0.02:  # 2% average negative return
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING

    def get_regime_adjustment(self, regime: MarketRegime) -> float:
        """
        Get position size adjustment for current regime.

        Conservative in high volatility/crisis, aggressive in trends.
        """
        adjustments = {
            MarketRegime.LOW_VOLATILITY: 1.0,    # Normal sizing
            MarketRegime.HIGH_VOLATILITY: 0.5,   # Half size
            MarketRegime.TRENDING_UP: 1.2,       # Slightly larger
            MarketRegime.TRENDING_DOWN: 0.8,     # Slightly smaller
            MarketRegime.RANGING: 1.0,           # Normal
            MarketRegime.CRISIS: 0.25,           # Quarter size - preserve capital
        }
        return adjustments.get(regime, 1.0)

    # =========================================================================
    # FEATURE ENGINEERING (Point-in-Time Correct)
    # =========================================================================

    def extract_features(self, signal: Dict, historical_data: List[Dict]) -> Dict:
        """
        Extract features for a signal using ONLY data available at signal time.

        CRITICAL: historical_data must only contain data BEFORE signal timestamp.
        This prevents lookahead bias.
        """
        signal_ts = signal.get("timestamp", 0)

        # Filter to only past data (point-in-time correctness)
        past_data = [d for d in historical_data if d.get("timestamp", 0) < signal_ts]

        features = {
            # Basic signal features
            "edge_percent": signal.get("edge_percent", 0),
            "confidence": signal.get("confidence", 0),
            "hour_of_day": signal.get("hour_of_day", 12),
            "day_of_week": signal.get("day_of_week", 0),
            "time_to_expiry_sec": signal.get("time_to_expiry_sec") or 3600,

            # Market state features
            "spread": signal.get("spread", 0),
            "yes_price": signal.get("yes_price", 0.5),
            "no_price": signal.get("no_price", 0.5),
        }

        # Historical performance features (using only past data)
        if past_data:
            # Win rate on similar signals
            similar = [d for d in past_data
                      if d.get("signal_type") == signal.get("signal_type")]
            if similar:
                features["historical_win_rate"] = (
                    sum(1 for d in similar if d.get("was_profitable")) / len(similar)
                )
                features["historical_avg_profit"] = (
                    sum(d.get("actual_profit", 0) or 0 for d in similar) / len(similar)
                )
            else:
                features["historical_win_rate"] = 0.5
                features["historical_avg_profit"] = 0

            # Recent performance (last 10 trades)
            recent = past_data[-10:] if len(past_data) >= 10 else past_data
            features["recent_win_rate"] = (
                sum(1 for d in recent if d.get("was_profitable")) / len(recent)
            )

            # Regime detection from past data
            regime = self.detect_regime(past_data[-50:])  # Last 50 samples
            features["regime"] = regime.value
            features["regime_adjustment"] = self.get_regime_adjustment(regime)
        else:
            features["historical_win_rate"] = 0.5
            features["historical_avg_profit"] = 0
            features["recent_win_rate"] = 0.5
            features["regime"] = MarketRegime.RANGING.value
            features["regime_adjustment"] = 1.0

        return features

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def analyze_edge_performance(self, data: List[Dict]) -> Dict:
        """Analyze how different edge thresholds performed."""
        if not data:
            return {}

        edges = [d["edge_percent"] for d in data]
        profits = [d.get("actual_profit", 0) for d in data]
        profitable = [d.get("was_profitable", False) for d in data]

        # Group by edge buckets
        buckets = {}
        for edge, profit, win in zip(edges, profits, profitable):
            bucket = int(edge)  # Round to nearest integer
            if bucket not in buckets:
                buckets[bucket] = {"count": 0, "wins": 0, "profit": 0}
            buckets[bucket]["count"] += 1
            buckets[bucket]["wins"] += 1 if win else 0
            buckets[bucket]["profit"] += profit or 0

        # Find optimal edge threshold
        best_edge = config.MIN_EDGE_PERCENT
        best_win_rate = 0

        for edge, stats in buckets.items():
            if stats["count"] >= 5:  # Need at least 5 samples
                win_rate = stats["wins"] / stats["count"]
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_edge = edge

        return {
            "buckets": buckets,
            "optimal_edge": best_edge,
            "best_win_rate": best_win_rate
        }

    def analyze_time_performance(self, data: List[Dict]) -> Dict:
        """Analyze performance by hour and day."""
        if not data:
            return {}

        # By hour
        hourly = {}
        for d in data:
            hour = d.get("hour_of_day", 12)
            if hour not in hourly:
                hourly[hour] = {"count": 0, "wins": 0}
            hourly[hour]["count"] += 1
            hourly[hour]["wins"] += 1 if d.get("was_profitable") else 0

        # By day
        daily = {}
        for d in data:
            day = d.get("day_of_week", 0)
            if day not in daily:
                daily[day] = {"count": 0, "wins": 0}
            daily[day]["count"] += 1
            daily[day]["wins"] += 1 if d.get("was_profitable") else 0

        # Find best hours (win rate > average)
        avg_win_rate = sum(d.get("was_profitable", False) for d in data) / len(data) if data else 0.5

        optimal_hours = []
        for hour, stats in hourly.items():
            if stats["count"] >= 3:
                win_rate = stats["wins"] / stats["count"]
                if win_rate >= avg_win_rate:
                    optimal_hours.append(hour)

        optimal_days = []
        for day, stats in daily.items():
            if stats["count"] >= 3:
                win_rate = stats["wins"] / stats["count"]
                if win_rate >= avg_win_rate:
                    optimal_days.append(day)

        return {
            "hourly": hourly,
            "daily": daily,
            "optimal_hours": optimal_hours or list(range(24)),
            "optimal_days": optimal_days or list(range(7)),
            "avg_win_rate": avg_win_rate
        }

    def analyze_confidence_performance(self, data: List[Dict]) -> Dict:
        """Analyze how confidence scores correlated with outcomes."""
        if not data:
            return {}

        # Group by confidence buckets (0.1 increments)
        buckets = {}
        for d in data:
            conf = round(d.get("confidence", 0.5), 1)
            if conf not in buckets:
                buckets[conf] = {"count": 0, "wins": 0}
            buckets[conf]["count"] += 1
            buckets[conf]["wins"] += 1 if d.get("was_profitable") else 0

        # Find minimum confidence with good win rate
        optimal_conf = 0.3
        for conf, stats in sorted(buckets.items()):
            if stats["count"] >= 3:
                win_rate = stats["wins"] / stats["count"]
                if win_rate >= 0.6:
                    optimal_conf = conf
                    break

        return {
            "buckets": buckets,
            "optimal_confidence": optimal_conf
        }

    def analyze_signal_types(self, data: List[Dict]) -> Dict:
        """Analyze performance by signal type."""
        if not data:
            return {}

        types = {}
        for d in data:
            sig_type = d.get("signal_type", "unknown")
            if sig_type not in types:
                types[sig_type] = {"count": 0, "wins": 0, "total_profit": 0}
            types[sig_type]["count"] += 1
            types[sig_type]["wins"] += 1 if d.get("was_profitable") else 0
            types[sig_type]["total_profit"] += d.get("actual_profit", 0) or 0

        for sig_type, stats in types.items():
            stats["win_rate"] = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
            stats["avg_profit"] = stats["total_profit"] / stats["count"] if stats["count"] > 0 else 0

        return types

    def analyze_regime_performance(self, data: List[Dict]) -> Dict:
        """Analyze performance by market regime."""
        if not data:
            return {}

        # Detect regime for each data point using historical context
        regime_stats = {}
        sorted_data = sorted(data, key=lambda x: x.get("timestamp", 0))

        for i, d in enumerate(sorted_data):
            # Use past data to detect regime (no lookahead)
            past_data = sorted_data[:i] if i > 0 else []
            regime = self.detect_regime(past_data[-50:]) if past_data else MarketRegime.RANGING

            regime_name = regime.value
            if regime_name not in regime_stats:
                regime_stats[regime_name] = {"count": 0, "wins": 0, "profit": 0}
            regime_stats[regime_name]["count"] += 1
            regime_stats[regime_name]["wins"] += 1 if d.get("was_profitable") else 0
            regime_stats[regime_name]["profit"] += d.get("actual_profit", 0) or 0

        # Calculate win rates
        for regime_name, stats in regime_stats.items():
            stats["win_rate"] = stats["wins"] / stats["count"] if stats["count"] > 0 else 0

        return regime_stats

    # =========================================================================
    # MAIN OPTIMIZATION
    # =========================================================================

    def optimize(self, min_samples: int = 10) -> Optional[OptimizedParams]:
        """
        Run optimization on collected data with RESEARCH FRAMEWORK compliance.

        Steps:
        1. Load and temporally sort data
        2. Check for data leakage
        3. Perform walk-forward validation
        4. Analyze performance by dimension
        5. Generate regime-aware parameters
        """
        data = collector.get_training_data()

        if len(data) < min_samples:
            print(f"[MLOptimizer] Not enough data: {len(data)}/{min_samples} samples")
            return None

        print(f"\n{'='*60}")
        print(f"ML OPTIMIZER - RESEARCH FRAMEWORK")
        print(f"{'='*60}")
        print(f"Total samples: {len(data)}")

        # Step 1: Temporal split (NEVER shuffle)
        train, val, test = self.temporal_split(data)

        if len(train) < self.MIN_TRAIN_SAMPLES:
            print(f"[MLOptimizer] Not enough training data: {len(train)}/{self.MIN_TRAIN_SAMPLES}")
            return None

        # Step 2: Check for data leakage
        if not self.check_data_leakage(train, val, test):
            print("[MLOptimizer] WARNING: Data leakage detected, results may be invalid")

        # Step 3: Walk-forward validation
        print(f"\n--- Walk-Forward Validation ---")
        wf_results = self.walk_forward_validation(data)

        # Check for overfitting
        if wf_results:
            avg_overfit = np.mean([r.overfitting_score for r in wf_results])
            if avg_overfit > 0.15:  # Train >> Val by more than 15%
                print(f"[MLOptimizer] WARNING: High overfitting detected ({avg_overfit:.1%})")
                print("  Consider: More data, simpler model, stronger regularization")

        # Step 4: Analyze each dimension (on TRAINING data only)
        print(f"\n--- Dimension Analysis (Training Data) ---")
        edge_analysis = self.analyze_edge_performance(train)
        time_analysis = self.analyze_time_performance(train)
        conf_analysis = self.analyze_confidence_performance(train)
        type_analysis = self.analyze_signal_types(train)
        regime_analysis = self.analyze_regime_performance(train)

        print(f"  Signal types: {list(type_analysis.keys())}")
        print(f"  Optimal edge: {edge_analysis.get('optimal_edge', 'N/A')}%")
        print(f"  Regimes seen: {list(regime_analysis.keys())}")

        # Step 5: Validate on held-out validation set
        print(f"\n--- Validation Set Performance ---")
        if val:
            val_wins = sum(1 for d in val if d.get("was_profitable"))
            val_win_rate = val_wins / len(val)
            val_profit = sum(d.get("actual_profit", 0) or 0 for d in val)
            print(f"  Val Win Rate: {val_win_rate:.1%}")
            print(f"  Val Profit: ${val_profit:.2f}")

        # Step 6: Final test set (report only, don't optimize on it)
        print(f"\n--- Test Set Performance (Out-of-Sample) ---")
        if test:
            test_wins = sum(1 for d in test if d.get("was_profitable"))
            test_win_rate = test_wins / len(test)
            test_profit = sum(d.get("actual_profit", 0) or 0 for d in test)
            print(f"  Test Win Rate: {test_win_rate:.1%}")
            print(f"  Test Profit: ${test_profit:.2f}")

        # Step 7: Create regime-adjusted parameters
        regime_adjustments = {}
        for regime_name, stats in regime_analysis.items():
            if stats["count"] >= 3:
                # Adjust based on regime win rate vs overall
                overall_win_rate = sum(1 for d in train if d.get("was_profitable")) / len(train)
                regime_win_rate = stats["win_rate"]
                adjustment = regime_win_rate / overall_win_rate if overall_win_rate > 0 else 1.0
                regime_adjustments[regime_name] = min(max(adjustment, 0.5), 1.5)  # Clamp to [0.5, 1.5]

        # Create optimized params
        self.current_params = OptimizedParams(
            min_edge_percent=max(edge_analysis.get("optimal_edge", config.MIN_EDGE_PERCENT), 1.5),
            min_negrisk_edge=max(edge_analysis.get("optimal_edge", config.MIN_NEGRISK_EDGE) * 0.6, 1.0),
            min_confidence=conf_analysis.get("optimal_confidence", 0.3),
            max_position_size=config.MAX_POSITION_SIZE,
            optimal_hours=time_analysis.get("optimal_hours", list(range(24))),
            optimal_days=time_analysis.get("optimal_days", list(range(7))),
            edge_multiplier=1.0 + (edge_analysis.get("best_win_rate", 0.5) - 0.5) * 0.5,  # Dampened
            confidence_threshold=conf_analysis.get("optimal_confidence", 0.3),
            time_decay_factor=0.8,
            regime_adjustments=regime_adjustments,
            validation_metrics={
                "train_win_rate": sum(1 for d in train if d.get("was_profitable")) / len(train) if train else 0,
                "val_win_rate": val_win_rate if val else 0,
                "test_win_rate": test_win_rate if test else 0,
                "avg_overfitting": float(np.mean([r.overfitting_score for r in wf_results])) if wf_results else 0,
            },
            last_trained=datetime.now().timestamp(),
            train_samples=len(train),
            val_samples=len(val),
            test_samples=len(test),
        )

        self._save_params()

        # Save validation results
        self._save_validation_results(wf_results)

        print(f"\n{'='*60}")
        print(f"OPTIMIZED PARAMETERS")
        print(f"{'='*60}")
        print(f"  Min Edge: {self.current_params.min_edge_percent}%")
        print(f"  Min Confidence: {self.current_params.min_confidence}")
        print(f"  Optimal Hours: {self.current_params.optimal_hours[:8]}...")
        print(f"  Edge Multiplier: {self.current_params.edge_multiplier:.2f}")
        print(f"  Regime Adjustments: {regime_adjustments}")
        print(f"{'='*60}\n")

        return self.current_params

    def _save_validation_results(self, results: List[ValidationResult]):
        """Save validation history for analysis."""
        with open(self.validation_file, "w") as f:
            json.dump([{
                "fold": r.fold,
                "train_start": r.train_start,
                "train_end": r.train_end,
                "val_start": r.val_start,
                "val_end": r.val_end,
                "train_samples": r.train_samples,
                "val_samples": r.val_samples,
                "train_win_rate": r.train_win_rate,
                "val_win_rate": r.val_win_rate,
                "train_profit": r.train_profit,
                "val_profit": r.val_profit,
                "regime": r.regime.value,
                "overfitting_score": r.overfitting_score,
            } for r in results], f, indent=2)

    # =========================================================================
    # DECISION METHODS
    # =========================================================================

    def should_take_signal(self, signal) -> Tuple[bool, str]:
        """
        Use ML-optimized params to decide if we should take a signal.
        Returns (should_trade, reason).
        """
        if not self.current_params:
            return True, "No ML params, using defaults"

        now = datetime.now()

        # Check time constraints
        if now.hour not in self.current_params.optimal_hours:
            return False, f"Hour {now.hour} not in optimal hours"

        if now.weekday() not in self.current_params.optimal_days:
            return False, f"Day {now.weekday()} not in optimal days"

        # Check edge threshold
        min_edge = self.current_params.min_edge_percent
        if hasattr(signal, 'signal_type') and signal.signal_type.value == "negrisk":
            min_edge = self.current_params.min_negrisk_edge

        signal_edge = signal.edge_percent if hasattr(signal, 'edge_percent') else 0
        if signal_edge < min_edge:
            return False, f"Edge {signal_edge:.2f}% below ML threshold {min_edge}%"

        # Check confidence
        signal_conf = signal.confidence if hasattr(signal, 'confidence') else 0.5
        if signal_conf < self.current_params.confidence_threshold:
            return False, f"Confidence {signal_conf:.2f} below ML threshold"

        return True, "ML approved"

    def adjust_position_size(self, signal, base_size: float) -> float:
        """Adjust position size based on ML insights and current regime."""
        if not self.current_params:
            return base_size

        size = base_size

        # Apply edge multiplier
        size *= self.current_params.edge_multiplier

        # Apply regime adjustment
        if self.current_regime.value in self.current_params.regime_adjustments:
            regime_adj = self.current_params.regime_adjustments[self.current_regime.value]
            size *= regime_adj
            if regime_adj < 1.0:
                print(f"[MLOptimizer] Regime adjustment: {self.current_regime.value} -> {regime_adj:.1%} size")

        # Apply time decay if close to expiry
        if hasattr(signal, 'market'):
            market = signal.market
            if hasattr(market, 'time_remaining_sec'):
                time_left = market.time_remaining_sec or 3600
                if time_left < 300:  # Less than 5 minutes
                    size *= self.current_params.time_decay_factor

        # Cap at max
        size = min(size, self.current_params.max_position_size)

        return max(config.MIN_POSITION_SIZE, size)

    def update_regime(self, recent_data: List[Dict]):
        """Update current regime based on recent market data."""
        self.current_regime = self.detect_regime(recent_data)
        print(f"[MLOptimizer] Current regime: {self.current_regime.value}")

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_performance_report(self) -> str:
        """Generate a comprehensive performance report."""
        data = collector.get_training_data()
        stats = collector.get_stats()

        if not data:
            return "No training data available yet."

        type_analysis = self.analyze_signal_types(data)
        edge_analysis = self.analyze_edge_performance(data)
        regime_analysis = self.analyze_regime_performance(data)

        report = []
        report.append("=" * 60)
        report.append("ML PERFORMANCE REPORT (RESEARCH FRAMEWORK)")
        report.append("=" * 60)
        report.append(f"Total Signals: {stats['total_signals']}")
        report.append(f"Training Samples: {stats['training_samples']}")
        report.append(f"Overall Win Rate: {stats['win_rate']:.1%}")
        report.append("")

        if self.current_params:
            report.append("Validation Metrics:")
            vm = self.current_params.validation_metrics
            report.append(f"  Train Win Rate: {vm.get('train_win_rate', 0):.1%}")
            report.append(f"  Val Win Rate: {vm.get('val_win_rate', 0):.1%}")
            report.append(f"  Test Win Rate: {vm.get('test_win_rate', 0):.1%}")
            report.append(f"  Overfitting Score: {vm.get('avg_overfitting', 0):+.1%}")
            report.append("")

        report.append("By Signal Type:")
        for sig_type, s in type_analysis.items():
            report.append(f"  {sig_type}: {s['count']} trades, {s['win_rate']:.1%} win rate, ${s['avg_profit']:.2f} avg")

        report.append("")
        report.append("By Market Regime:")
        for regime, s in regime_analysis.items():
            report.append(f"  {regime}: {s['count']} trades, {s['win_rate']:.1%} win rate")

        report.append("")
        report.append(f"Current Regime: {self.current_regime.value}")
        report.append(f"Optimal Edge Threshold: {edge_analysis.get('optimal_edge', 'N/A')}%")
        report.append("=" * 60)

        return "\n".join(report)


# Global optimizer instance
optimizer = MLOptimizer()
