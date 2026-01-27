"""
Machine Learning optimizer for the arbitrage bot.
Uses collected trading data to optimize parameters.
"""

import json
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from .config import config
from .data_collector import collector


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


class MLOptimizer:
    """
    Optimizes trading parameters using historical data.
    Uses simple statistical methods that work without heavy ML libraries.
    """

    def __init__(self, data_dir: str = "ml_data"):
        self.data_dir = data_dir
        self.params_file = os.path.join(data_dir, "optimized_params.json")
        self.current_params: Optional[OptimizedParams] = None

        # Load existing params if available
        self._load_params()

    def _load_params(self):
        """Load optimized params from file."""
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file, "r") as f:
                    data = json.load(f)
                self.current_params = OptimizedParams(**data)
                print(f"[MLOptimizer] Loaded optimized params")
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
                }, f, indent=2)

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

    def optimize(self, min_samples: int = 10) -> Optional[OptimizedParams]:
        """
        Run optimization on collected data.
        Returns optimized parameters if enough data is available.
        """
        data = collector.get_training_data()

        if len(data) < min_samples:
            print(f"[MLOptimizer] Not enough data: {len(data)}/{min_samples} samples")
            return None

        print(f"[MLOptimizer] Optimizing with {len(data)} samples...")

        # Analyze each dimension
        edge_analysis = self.analyze_edge_performance(data)
        time_analysis = self.analyze_time_performance(data)
        conf_analysis = self.analyze_confidence_performance(data)
        type_analysis = self.analyze_signal_types(data)

        # Create optimized params
        self.current_params = OptimizedParams(
            min_edge_percent=max(edge_analysis.get("optimal_edge", config.MIN_EDGE_PERCENT), 2.0),
            min_negrisk_edge=max(edge_analysis.get("optimal_edge", config.MIN_NEGRISK_EDGE) * 0.6, 1.0),
            min_confidence=conf_analysis.get("optimal_confidence", 0.3),
            max_position_size=config.MAX_POSITION_SIZE,
            optimal_hours=time_analysis.get("optimal_hours", list(range(24))),
            optimal_days=time_analysis.get("optimal_days", list(range(7))),
            edge_multiplier=1.0 + (edge_analysis.get("best_win_rate", 0.5) - 0.5),
            confidence_threshold=conf_analysis.get("optimal_confidence", 0.3),
            time_decay_factor=0.8,  # Reduce position size by 20% near expiry
        )

        self._save_params()

        print(f"[MLOptimizer] Optimized params:")
        print(f"  Min Edge: {self.current_params.min_edge_percent}%")
        print(f"  Min Confidence: {self.current_params.min_confidence}")
        print(f"  Optimal Hours: {self.current_params.optimal_hours[:5]}...")
        print(f"  Edge Multiplier: {self.current_params.edge_multiplier:.2f}")

        return self.current_params

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
        if signal.signal_type.value == "negrisk":
            min_edge = self.current_params.min_negrisk_edge

        if signal.edge_percent < min_edge:
            return False, f"Edge {signal.edge_percent:.2f}% below ML threshold {min_edge}%"

        # Check confidence
        if signal.confidence < self.current_params.confidence_threshold:
            return False, f"Confidence {signal.confidence:.2f} below ML threshold"

        return True, "ML approved"

    def adjust_position_size(self, signal, base_size: float) -> float:
        """Adjust position size based on ML insights."""
        if not self.current_params:
            return base_size

        size = base_size

        # Apply edge multiplier
        size *= self.current_params.edge_multiplier

        # Apply time decay if close to expiry
        if hasattr(signal.market, 'time_remaining_sec'):
            time_left = signal.market.time_remaining_sec or 3600
            if time_left < 300:  # Less than 5 minutes
                size *= self.current_params.time_decay_factor

        # Cap at max
        size = min(size, self.current_params.max_position_size)

        return max(config.MIN_POSITION_SIZE, size)

    def get_performance_report(self) -> str:
        """Generate a performance report."""
        data = collector.get_training_data()
        stats = collector.get_stats()

        if not data:
            return "No training data available yet."

        type_analysis = self.analyze_signal_types(data)
        edge_analysis = self.analyze_edge_performance(data)

        report = []
        report.append("=" * 50)
        report.append("ML PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append(f"Total Signals: {stats['total_signals']}")
        report.append(f"Training Samples: {stats['training_samples']}")
        report.append(f"Win Rate: {stats['win_rate']:.1%}")
        report.append("")
        report.append("By Signal Type:")
        for sig_type, s in type_analysis.items():
            report.append(f"  {sig_type}: {s['count']} trades, {s['win_rate']:.1%} win rate, ${s['avg_profit']:.2f} avg")
        report.append("")
        report.append(f"Optimal Edge Threshold: {edge_analysis.get('optimal_edge', 'N/A')}%")
        report.append("=" * 50)

        return "\n".join(report)


# Global optimizer instance
optimizer = MLOptimizer()
