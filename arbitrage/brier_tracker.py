"""Brier score tracking for prediction calibration."""
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path
import math
import logging

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A recorded prediction."""
    id: str
    market_id: str
    market_question: str
    predicted_side: str  # 'YES' or 'NO'
    confidence: float  # 0.0 to 1.0
    entry_price: float
    timestamp: str
    resolved: bool = False
    outcome: Optional[bool] = None  # True if our side won
    resolution_timestamp: Optional[str] = None
    brier_contribution: Optional[float] = None


@dataclass
class CalibrationBucket:
    """Calibration bucket for a confidence range."""
    range_start: float
    range_end: float
    predictions: int
    correct: int
    expected_rate: float
    actual_rate: float
    calibration_error: float


class BrierTracker:
    """Track predictions and calculate Brier scores for calibration."""

    def __init__(self, storage_path: str = None):
        if storage_path is None:
            storage_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "predictions.json"
            )
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.predictions: list[Prediction] = []
        self._load()

    def _load(self):
        """Load predictions from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    self.predictions = [Prediction(**p) for p in data]
            except Exception as e:
                logger.error(f"Failed to load predictions: {e}")
                self.predictions = []

    def _save(self):
        """Save predictions to storage."""
        try:
            with open(self.storage_path, "w") as f:
                json.dump([asdict(p) for p in self.predictions], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")

    def record_prediction(
        self,
        market_id: str,
        market_question: str,
        predicted_side: str,
        confidence: float,
        entry_price: float,
    ) -> Prediction:
        """Record a new prediction."""
        pred = Prediction(
            id=f"{market_id}_{datetime.now(timezone.utc).isoformat()}",
            market_id=market_id,
            market_question=market_question,
            predicted_side=predicted_side.upper(),
            confidence=min(max(confidence, 0.0), 1.0),
            entry_price=entry_price,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.predictions.append(pred)
        self._save()
        logger.info(f"Recorded prediction: {market_question[:50]} | {predicted_side} @ {confidence:.0%}")
        return pred

    def resolve_prediction(self, prediction_id: str, won: bool) -> Optional[Prediction]:
        """Resolve a prediction with the outcome."""
        for pred in self.predictions:
            if pred.id == prediction_id:
                pred.resolved = True
                pred.outcome = won
                pred.resolution_timestamp = datetime.now(timezone.utc).isoformat()

                # Calculate Brier contribution
                # Brier = (forecast - outcome)^2
                # forecast = confidence, outcome = 1 if won else 0
                outcome_val = 1.0 if won else 0.0
                pred.brier_contribution = (pred.confidence - outcome_val) ** 2

                self._save()
                logger.info(
                    f"Resolved: {pred.market_question[:40]} | "
                    f"{'WON' if won else 'LOST'} | Brier: {pred.brier_contribution:.3f}"
                )
                return pred

        return None

    def resolve_by_market(self, market_id: str, winning_side: str) -> list[Prediction]:
        """Resolve all predictions for a market."""
        resolved = []
        for pred in self.predictions:
            if pred.market_id == market_id and not pred.resolved:
                won = pred.predicted_side.upper() == winning_side.upper()
                self.resolve_prediction(pred.id, won)
                resolved.append(pred)
        return resolved

    def calculate_brier_score(self, resolved_only: bool = True) -> float:
        """Calculate overall Brier score."""
        preds = [p for p in self.predictions if p.resolved] if resolved_only else self.predictions

        if not preds:
            return 0.5  # Default (random guessing)

        total_brier = sum(p.brier_contribution or 0 for p in preds if p.brier_contribution is not None)
        return total_brier / len([p for p in preds if p.brier_contribution is not None])

    def get_calibration_buckets(self, num_buckets: int = 10) -> list[CalibrationBucket]:
        """Calculate calibration across confidence buckets."""
        resolved = [p for p in self.predictions if p.resolved]

        if not resolved:
            return []

        buckets = []
        bucket_size = 1.0 / num_buckets

        for i in range(num_buckets):
            start = i * bucket_size
            end = (i + 1) * bucket_size

            in_bucket = [p for p in resolved if start <= p.confidence < end]

            if in_bucket:
                predictions = len(in_bucket)
                correct = sum(1 for p in in_bucket if p.outcome)
                expected_rate = (start + end) / 2
                actual_rate = correct / predictions
                calibration_error = abs(expected_rate - actual_rate)

                buckets.append(CalibrationBucket(
                    range_start=start,
                    range_end=end,
                    predictions=predictions,
                    correct=correct,
                    expected_rate=expected_rate,
                    actual_rate=actual_rate,
                    calibration_error=calibration_error,
                ))

        return buckets

    def get_resolution_rate(self) -> float:
        """Get percentage of predictions that have been resolved."""
        if not self.predictions:
            return 0.0
        return len([p for p in self.predictions if p.resolved]) / len(self.predictions)

    def get_win_rate(self) -> float:
        """Get percentage of resolved predictions that won."""
        resolved = [p for p in self.predictions if p.resolved]
        if not resolved:
            return 0.0
        return len([p for p in resolved if p.outcome]) / len(resolved)

    def get_statistics(self) -> dict:
        """Get comprehensive statistics."""
        resolved = [p for p in self.predictions if p.resolved]

        stats = {
            "total_predictions": len(self.predictions),
            "resolved": len(resolved),
            "pending": len(self.predictions) - len(resolved),
            "wins": len([p for p in resolved if p.outcome]),
            "losses": len([p for p in resolved if not p.outcome]),
            "win_rate": self.get_win_rate(),
            "brier_score": self.calculate_brier_score(),
            "brier_rating": self._rate_brier_score(self.calculate_brier_score()),
        }

        # Add calibration summary
        buckets = self.get_calibration_buckets(5)
        if buckets:
            avg_calibration_error = sum(b.calibration_error for b in buckets) / len(buckets)
            stats["avg_calibration_error"] = avg_calibration_error
            stats["calibration_rating"] = "Good" if avg_calibration_error < 0.1 else "Needs work"

        return stats

    def _rate_brier_score(self, score: float) -> str:
        """Rate a Brier score."""
        if score < 0.15:
            return "Superforecaster"
        elif score < 0.20:
            return "Excellent"
        elif score < 0.25:
            return "Good"
        elif score < 0.33:
            return "Average"
        else:
            return "Below average"

    def get_kelly_adjustment(self) -> float:
        """Get Kelly criterion adjustment based on calibration."""
        brier = self.calculate_brier_score()

        # Better calibration = higher Kelly fraction
        # Superforecaster (0.15) -> 1.0x
        # Average (0.33) -> 0.5x
        # Poor (0.5) -> 0.25x
        if brier < 0.15:
            return 1.0
        elif brier < 0.25:
            return 0.75
        elif brier < 0.35:
            return 0.5
        else:
            return 0.25

    def print_report(self):
        """Print a formatted calibration report."""
        stats = self.get_statistics()

        print("=" * 60)
        print("BRIER SCORE & CALIBRATION REPORT")
        print("=" * 60)
        print()
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Resolved: {stats['resolved']} | Pending: {stats['pending']}")
        print(f"Wins: {stats['wins']} | Losses: {stats['losses']}")
        print(f"Win Rate: {stats['win_rate']*100:.1f}%")
        print()
        print(f"Brier Score: {stats['brier_score']:.3f}")
        print(f"Rating: {stats['brier_rating']}")
        print(f"Kelly Adjustment: {self.get_kelly_adjustment():.2f}x")
        print()

        buckets = self.get_calibration_buckets(5)
        if buckets:
            print("Calibration by Confidence Level:")
            print("-" * 50)
            for b in buckets:
                print(
                    f"  {b.range_start*100:.0f}-{b.range_end*100:.0f}%: "
                    f"{b.correct}/{b.predictions} ({b.actual_rate*100:.0f}% actual vs "
                    f"{b.expected_rate*100:.0f}% expected)"
                )


def main():
    """Test Brier tracker."""
    tracker = BrierTracker()

    # Show current stats
    tracker.print_report()

    # Example: record a prediction
    # tracker.record_prediction(
    #     market_id="seahawks_sb",
    #     market_question="Will the Seattle Seahawks win Super Bowl 2026?",
    #     predicted_side="YES",
    #     confidence=0.67,
    #     entry_price=0.674,
    # )


if __name__ == "__main__":
    main()
