"""
Advanced ML Engine for Profit Maximization

Implements:
1. Feature Engineering - Extract predictive signals
2. Gradient-Based Optimization - Find optimal parameters
3. Kelly Criterion - Optimal position sizing
4. Online Learning - Continuous improvement from outcomes
5. Ensemble Methods - Combine multiple signals
"""

import json
import math
import os
import statistics
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import random


@dataclass
class TradeFeatures:
    """Features extracted from a potential trade."""
    entry_price: float
    momentum: float  # (current - entry) / entry
    whale_pnl: float
    whale_pnl_pct: float
    time_to_expiry_hours: float
    outcome_is_yes: bool
    volume: float
    liquidity: float
    price_volatility: float
    whale_position_size: float

    def to_vector(self) -> List[float]:
        """Convert to feature vector for ML."""
        return [
            self.entry_price,
            self.momentum,
            self.whale_pnl / 10000,  # Normalize
            self.whale_pnl_pct / 100,
            min(self.time_to_expiry_hours / 168, 1.0),  # Cap at 1 week
            1.0 if self.outcome_is_yes else 0.0,
            min(self.volume / 1000000, 1.0),
            min(self.liquidity / 100000, 1.0),
            self.price_volatility,
            min(self.whale_position_size / 50000, 1.0),
        ]


@dataclass
class MLPrediction:
    """Prediction from the ML model."""
    win_probability: float
    expected_return: float
    confidence: float
    kelly_fraction: float
    recommended_size: float
    features_importance: Dict[str, float]
    reasoning: str


@dataclass
class TrainingExample:
    """A training example from past trades."""
    features: TradeFeatures
    outcome: int  # 1 = win, 0 = loss
    actual_return: float
    timestamp: str


class MLEngine:
    """
    Machine Learning engine for trade optimization.

    Uses a simple but effective approach:
    1. Logistic-like scoring function
    2. Online gradient updates
    3. Kelly criterion for sizing
    """

    FEATURE_NAMES = [
        "entry_price", "momentum", "whale_pnl", "whale_pnl_pct",
        "time_to_expiry", "is_yes", "volume", "liquidity",
        "volatility", "whale_size"
    ]

    def __init__(self, state_file: str = "ml_engine_state.json"):
        self.state_file = state_file

        # Model weights (learned from data)
        self.weights = self._initialize_weights()
        self.bias = 0.0

        # Learning parameters
        self.learning_rate = 0.01
        self.regularization = 0.001

        # Training history
        self.training_examples: List[TrainingExample] = []
        self.performance_history: List[Dict] = []

        # Optimal parameters (discovered)
        self.optimal_params = {
            "min_entry_price": 0.50,
            "max_entry_price": 0.85,
            "min_confidence": 0.60,
            "prefer_yes": True,
            "min_momentum": 0.0,
            "max_kelly_fraction": 0.25,
        }

        self._load_state()

    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize weights based on backtest insights."""
        return {
            "entry_price": 0.5,      # Higher entry = higher prob of win
            "momentum": 0.3,          # Positive momentum is good
            "whale_pnl": 0.2,         # Whale profiting = good signal
            "whale_pnl_pct": 0.15,    # Percentage gain matters
            "time_to_expiry": -0.1,   # Shorter time = more certainty
            "is_yes": 0.4,            # YES strongly preferred (backtest)
            "volume": 0.1,            # Higher volume = more reliable
            "liquidity": 0.1,         # Better liquidity = easier exit
            "volatility": -0.2,       # Lower volatility = more predictable
            "whale_size": 0.15,       # Larger whale position = more conviction
        }

    def _load_state(self):
        """Load saved state."""
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
                self.weights = data.get("weights", self.weights)
                self.bias = data.get("bias", self.bias)
                self.optimal_params = data.get("optimal_params", self.optimal_params)
                self.performance_history = data.get("performance_history", [])

                for ex_data in data.get("training_examples", []):
                    features = TradeFeatures(**ex_data["features"])
                    self.training_examples.append(TrainingExample(
                        features=features,
                        outcome=ex_data["outcome"],
                        actual_return=ex_data["actual_return"],
                        timestamp=ex_data["timestamp"]
                    ))

                print(f"[MLEngine] Loaded {len(self.training_examples)} training examples")
        except FileNotFoundError:
            print("[MLEngine] No saved state, starting fresh")
        except Exception as e:
            print(f"[MLEngine] Error loading state: {e}")

    def _save_state(self):
        """Save current state."""
        try:
            data = {
                "weights": self.weights,
                "bias": self.bias,
                "optimal_params": self.optimal_params,
                "performance_history": self.performance_history[-100:],
                "training_examples": [
                    {
                        "features": asdict(ex.features),
                        "outcome": ex.outcome,
                        "actual_return": ex.actual_return,
                        "timestamp": ex.timestamp
                    }
                    for ex in self.training_examples[-500:]
                ],
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[MLEngine] Error saving state: {e}")

    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

    def _compute_score(self, features: TradeFeatures) -> float:
        """Compute raw score from features."""
        vector = features.to_vector()
        score = self.bias

        for i, (name, weight) in enumerate(self.weights.items()):
            if i < len(vector):
                score += weight * vector[i]

        return score

    def predict(self, features: TradeFeatures, bankroll: float = 1000.0) -> MLPrediction:
        """
        Make a prediction for a potential trade.

        Returns win probability, expected return, and optimal size.
        """
        # Compute raw score
        raw_score = self._compute_score(features)

        # Convert to probability
        win_prob = self._sigmoid(raw_score)

        # Apply learned adjustments
        # Boost for YES (strong backtest signal)
        if features.outcome_is_yes:
            win_prob = min(0.95, win_prob * 1.15)
        else:
            win_prob = win_prob * 0.85  # Penalize NO

        # Boost for optimal entry range
        if 0.50 <= features.entry_price <= 0.85:
            win_prob = min(0.95, win_prob * 1.10)

        # Calculate expected return
        if win_prob > 0 and features.entry_price > 0:
            win_return = (1.0 / features.entry_price) - 1  # Return if win
            loss_return = -1.0  # Lose entire stake
            expected_return = (win_prob * win_return) + ((1 - win_prob) * loss_return)
        else:
            expected_return = -1.0

        # Kelly Criterion for optimal sizing
        # f* = (bp - q) / b where b=odds, p=win_prob, q=1-p
        if features.entry_price > 0 and features.entry_price < 1:
            odds = (1.0 / features.entry_price) - 1  # Potential profit per dollar
            kelly = (odds * win_prob - (1 - win_prob)) / odds if odds > 0 else 0
            kelly = max(0, min(kelly, self.optimal_params["max_kelly_fraction"]))
        else:
            kelly = 0

        # Recommended size
        recommended_size = kelly * bankroll

        # Confidence based on feature quality
        confidence = win_prob
        if features.momentum > 0.1:
            confidence = min(0.95, confidence + 0.05)
        if features.whale_pnl > 500:
            confidence = min(0.95, confidence + 0.05)

        # Feature importance for this prediction
        importance = {}
        vector = features.to_vector()
        for i, (name, weight) in enumerate(self.weights.items()):
            if i < len(vector):
                importance[name] = abs(weight * vector[i])

        # Normalize importance
        total_imp = sum(importance.values()) or 1
        importance = {k: v/total_imp for k, v in importance.items()}

        # Generate reasoning
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        reasoning = f"Win prob: {win_prob:.0%}. Top factors: "
        reasoning += ", ".join([f"{k}({v:.0%})" for k, v in top_features])

        return MLPrediction(
            win_probability=win_prob,
            expected_return=expected_return,
            confidence=confidence,
            kelly_fraction=kelly,
            recommended_size=recommended_size,
            features_importance=importance,
            reasoning=reasoning
        )

    def record_outcome(self, features: TradeFeatures, won: bool, actual_return: float):
        """
        Record trade outcome for learning.

        This updates the model weights using gradient descent.
        """
        example = TrainingExample(
            features=features,
            outcome=1 if won else 0,
            actual_return=actual_return,
            timestamp=datetime.now().isoformat()
        )
        self.training_examples.append(example)

        # Online learning update
        self._update_weights(features, 1 if won else 0)

        # Track performance
        self.performance_history.append({
            "timestamp": example.timestamp,
            "won": won,
            "return": actual_return,
            "entry_price": features.entry_price,
            "is_yes": features.outcome_is_yes,
        })

        self._save_state()

        # Periodic re-optimization
        if len(self.training_examples) % 10 == 0:
            self._optimize_parameters()

    def _update_weights(self, features: TradeFeatures, outcome: int):
        """Update weights using gradient descent."""
        # Predict
        raw_score = self._compute_score(features)
        predicted = self._sigmoid(raw_score)

        # Error
        error = outcome - predicted

        # Update weights
        vector = features.to_vector()
        for i, name in enumerate(self.weights.keys()):
            if i < len(vector):
                gradient = error * vector[i] - self.regularization * self.weights[name]
                self.weights[name] += self.learning_rate * gradient

        # Update bias
        self.bias += self.learning_rate * error

    def _optimize_parameters(self):
        """
        Re-optimize parameters based on training data.
        Uses simple grid search on key parameters.
        """
        if len(self.training_examples) < 10:
            return

        best_params = self.optimal_params.copy()
        best_score = self._evaluate_params(best_params)

        # Grid search
        for min_entry in [0.40, 0.50, 0.60, 0.70]:
            for max_entry in [0.75, 0.80, 0.85, 0.90]:
                if min_entry >= max_entry:
                    continue

                params = self.optimal_params.copy()
                params["min_entry_price"] = min_entry
                params["max_entry_price"] = max_entry

                score = self._evaluate_params(params)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()

        self.optimal_params = best_params
        print(f"[MLEngine] Optimized params: entry {best_params['min_entry_price']:.0%}-{best_params['max_entry_price']:.0%}")

    def _evaluate_params(self, params: Dict) -> float:
        """Evaluate parameters on training data."""
        wins = 0
        losses = 0
        total_return = 0

        for ex in self.training_examples:
            # Check if trade would have been taken with these params
            if not (params["min_entry_price"] <= ex.features.entry_price <= params["max_entry_price"]):
                continue
            if params["prefer_yes"] and not ex.features.outcome_is_yes:
                continue

            if ex.outcome == 1:
                wins += 1
            else:
                losses += 1
            total_return += ex.actual_return

        if wins + losses == 0:
            return 0

        win_rate = wins / (wins + losses)
        # Score = win_rate * (1 + avg_return)
        avg_return = total_return / (wins + losses)
        return win_rate * (1 + avg_return)

    def should_trade(self, features: TradeFeatures) -> Tuple[bool, str]:
        """
        Determine if we should take this trade.

        Returns (should_trade, reason)
        """
        # Apply hard filters first
        if not (self.optimal_params["min_entry_price"] <= features.entry_price <= self.optimal_params["max_entry_price"]):
            return False, f"Entry {features.entry_price:.0%} outside {self.optimal_params['min_entry_price']:.0%}-{self.optimal_params['max_entry_price']:.0%}"

        if self.optimal_params["prefer_yes"] and not features.outcome_is_yes:
            return False, "NO position - backtest shows 6% vs 43% win rate"

        if features.momentum < self.optimal_params["min_momentum"]:
            return False, f"Negative momentum: {features.momentum:.0%}"

        # Get ML prediction
        prediction = self.predict(features)

        if prediction.confidence < self.optimal_params["min_confidence"]:
            return False, f"Confidence {prediction.confidence:.0%} below {self.optimal_params['min_confidence']:.0%}"

        if prediction.expected_return < 0:
            return False, f"Negative expected return: {prediction.expected_return:.1%}"

        return True, prediction.reasoning

    def get_optimal_size(self, features: TradeFeatures, bankroll: float) -> float:
        """Get optimal position size using Kelly Criterion."""
        prediction = self.predict(features, bankroll)
        return prediction.recommended_size

    def get_stats(self) -> Dict:
        """Get ML engine statistics."""
        recent = self.performance_history[-50:] if self.performance_history else []

        wins = len([p for p in recent if p.get("won")])
        total = len(recent)

        return {
            "training_examples": len(self.training_examples),
            "recent_trades": total,
            "recent_wins": wins,
            "recent_win_rate": wins / total if total > 0 else 0,
            "optimal_entry_range": f"{self.optimal_params['min_entry_price']:.0%}-{self.optimal_params['max_entry_price']:.0%}",
            "prefer_yes": self.optimal_params["prefer_yes"],
            "top_weights": dict(sorted(self.weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]),
        }

    def print_status(self):
        """Print ML engine status."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("ML ENGINE STATUS")
        print("=" * 60)
        print(f"Training Examples: {stats['training_examples']}")
        print(f"Recent Win Rate: {stats['recent_win_rate']:.0%} ({stats['recent_wins']}/{stats['recent_trades']})")
        print(f"Optimal Entry: {stats['optimal_entry_range']}")
        print(f"Prefer YES: {stats['prefer_yes']}")
        print()
        print("Top Feature Weights:")
        for name, weight in stats["top_weights"].items():
            print(f"  {name}: {weight:+.3f}")
        print("=" * 60)


# Global ML engine instance
ml_engine = MLEngine()


def extract_features(position: Dict) -> TradeFeatures:
    """Extract ML features from a position dict."""
    entry_price = float(position.get("avgPrice", 0) or position.get("entry_price", 0) or 0)
    current_price = float(position.get("curPrice", 0) or position.get("current_price", 0) or entry_price)

    if entry_price > 0:
        momentum = (current_price - entry_price) / entry_price
    else:
        momentum = 0

    return TradeFeatures(
        entry_price=current_price,
        momentum=momentum,
        whale_pnl=float(position.get("cashPnl", 0) or position.get("pnl", 0) or 0),
        whale_pnl_pct=float(position.get("percentPnl", 0) or position.get("pnl_pct", 0) or 0),
        time_to_expiry_hours=72,  # Default
        outcome_is_yes=position.get("outcome", "Yes") == "Yes",
        volume=float(position.get("volume", 0) or 0),
        liquidity=float(position.get("liquidity", 0) or 0),
        price_volatility=0.1,  # Default
        whale_position_size=float(position.get("currentValue", 0) or position.get("size", 0) or 0),
    )


if __name__ == "__main__":
    # Test the ML engine
    engine = MLEngine()
    engine.print_status()

    # Test prediction
    test_features = TradeFeatures(
        entry_price=0.75,
        momentum=0.15,
        whale_pnl=1500,
        whale_pnl_pct=25,
        time_to_expiry_hours=48,
        outcome_is_yes=True,
        volume=500000,
        liquidity=50000,
        price_volatility=0.08,
        whale_position_size=10000,
    )

    should, reason = engine.should_trade(test_features)
    print(f"\nTest trade: {should} - {reason}")

    pred = engine.predict(test_features, bankroll=1000)
    print(f"Win prob: {pred.win_probability:.0%}")
    print(f"Expected return: {pred.expected_return:.1%}")
    print(f"Kelly size: ${pred.recommended_size:.2f}")
