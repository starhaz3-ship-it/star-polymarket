"""
Advanced Kelly Criterion Module

Based on 2024-2026 research:
- Fractional Kelly for volatility reduction
- Confidence-adjusted position sizing
- Bayesian Kelly for probability uncertainty
- Multi-asset portfolio optimization
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from scipy.optimize import minimize


@dataclass
class KellyResult:
    """Result of Kelly calculation."""
    full_kelly: float
    fractional_kelly: float
    recommended_size: float
    edge: float
    confidence: float
    expected_growth: float


class AdvancedKelly:
    """Advanced Kelly Criterion calculator."""

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position: float = 0.15,
        min_edge: float = 0.02,
        fee_rate: float = 0.02,
    ):
        """
        Initialize Kelly calculator.

        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
            max_position: Maximum position size as fraction of bankroll
            min_edge: Minimum edge required to trade (default 2% for Polymarket fees)
            fee_rate: Trading fee rate (Polymarket winner fee)
        """
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position
        self.min_edge = min_edge
        self.fee_rate = fee_rate

    def basic_kelly(self, win_prob: float, odds: float) -> float:
        """
        Basic Kelly formula: f* = (p * b - q) / b

        Args:
            win_prob: Probability of winning (0-1)
            odds: Odds received on win (e.g., 0.5 price = 1.0 odds for 2x return)

        Returns:
            Kelly fraction (can be negative if edge is negative)
        """
        if odds <= 0 or win_prob <= 0 or win_prob >= 1:
            return 0.0

        q = 1 - win_prob
        kelly = (win_prob * odds - q) / odds

        return kelly

    def kelly_from_price(self, win_prob: float, market_price: float) -> float:
        """
        Calculate Kelly from probability and market price.

        Args:
            win_prob: Estimated probability of YES winning
            market_price: Current YES price (0-1)

        Returns:
            Kelly fraction for betting on YES
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0

        # Odds = (1 - price) / price for YES bet
        # If you bet $1 at price 0.4, you get $2.50 back = $1.50 profit = 1.5 odds
        odds = (1 - market_price) / market_price

        # Adjust odds for fees (winner pays 2%)
        net_odds = odds * (1 - self.fee_rate)

        return self.basic_kelly(win_prob, net_odds)

    def fractional_kelly(
        self,
        win_prob: float,
        market_price: float,
        bankroll: float,
    ) -> KellyResult:
        """
        Calculate fractional Kelly position size.

        Research shows fractional Kelly (0.25-0.5) provides:
        - 70% less volatility
        - Similar long-term returns
        - Better survival probability
        """
        full_kelly = self.kelly_from_price(win_prob, market_price)
        edge = win_prob - market_price

        # No position if edge is below threshold
        if edge < self.min_edge:
            return KellyResult(
                full_kelly=full_kelly,
                fractional_kelly=0.0,
                recommended_size=0.0,
                edge=edge,
                confidence=1.0,
                expected_growth=0.0,
            )

        # Apply fractional Kelly
        fractional = full_kelly * self.kelly_fraction

        # Cap at maximum position
        fractional = min(fractional, self.max_position)
        fractional = max(fractional, 0.0)

        # Calculate expected log growth
        if fractional > 0:
            expected_growth = (
                win_prob * np.log(1 + fractional * ((1 - market_price) / market_price) * (1 - self.fee_rate))
                + (1 - win_prob) * np.log(1 - fractional)
            )
        else:
            expected_growth = 0.0

        return KellyResult(
            full_kelly=full_kelly,
            fractional_kelly=fractional,
            recommended_size=fractional * bankroll,
            edge=edge,
            confidence=1.0,
            expected_growth=expected_growth,
        )

    def confidence_adjusted_kelly(
        self,
        win_prob: float,
        market_price: float,
        confidence: float,
        bankroll: float,
    ) -> KellyResult:
        """
        Kelly adjusted by model confidence.

        Lower confidence -> more conservative sizing.

        Args:
            win_prob: Estimated win probability
            market_price: Current market price
            confidence: Model confidence (0-1)
            bankroll: Current bankroll
        """
        # Get base fractional Kelly
        base_result = self.fractional_kelly(win_prob, market_price, bankroll)

        # Scale by confidence
        adjusted_fraction = base_result.fractional_kelly * confidence

        return KellyResult(
            full_kelly=base_result.full_kelly,
            fractional_kelly=adjusted_fraction,
            recommended_size=adjusted_fraction * bankroll,
            edge=base_result.edge,
            confidence=confidence,
            expected_growth=base_result.expected_growth * confidence,
        )

    def bayesian_kelly(
        self,
        win_prob: float,
        prob_std: float,
        market_price: float,
        bankroll: float,
        n_samples: int = 1000,
        percentile: float = 25,
    ) -> KellyResult:
        """
        Bayesian Kelly that accounts for uncertainty in probability estimates.

        Samples from a distribution around the estimated probability and
        uses a conservative percentile of the Kelly distribution.

        Args:
            win_prob: Estimated probability (mean)
            prob_std: Standard deviation of probability estimate
            market_price: Current market price
            bankroll: Current bankroll
            n_samples: Number of Monte Carlo samples
            percentile: Percentile to use (lower = more conservative)
        """
        # Convert to beta distribution parameters
        # Using method of moments
        if prob_std <= 0:
            return self.fractional_kelly(win_prob, market_price, bankroll)

        # Clamp std to avoid numerical issues
        prob_std = min(prob_std, 0.2)

        # Beta distribution parameters from mean and variance
        mean = np.clip(win_prob, 0.01, 0.99)
        var = prob_std ** 2
        var = min(var, mean * (1 - mean) * 0.99)  # Must be less than mean*(1-mean)

        alpha = mean * (mean * (1 - mean) / var - 1)
        beta = (1 - mean) * (mean * (1 - mean) / var - 1)

        # Ensure valid parameters
        alpha = max(alpha, 0.5)
        beta = max(beta, 0.5)

        # Sample probabilities
        prob_samples = np.random.beta(alpha, beta, n_samples)

        # Calculate Kelly for each sample
        kelly_samples = []
        for p in prob_samples:
            k = self.kelly_from_price(p, market_price)
            kelly_samples.append(max(0, k))

        # Use conservative percentile
        conservative_kelly = np.percentile(kelly_samples, percentile)

        # Apply fractional Kelly and caps
        final_kelly = conservative_kelly * self.kelly_fraction
        final_kelly = min(final_kelly, self.max_position)

        return KellyResult(
            full_kelly=np.mean(kelly_samples),
            fractional_kelly=final_kelly,
            recommended_size=final_kelly * bankroll,
            edge=win_prob - market_price,
            confidence=1 - prob_std,  # Higher uncertainty = lower confidence
            expected_growth=0.0,  # Would need to calculate
        )


class MultiAssetKelly:
    """Kelly optimization for multiple concurrent positions."""

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_single_position: float = 0.15,
        max_total_leverage: float = 0.80,
        fee_rate: float = 0.02,
    ):
        self.kelly_fraction = kelly_fraction
        self.max_single_position = max_single_position
        self.max_total_leverage = max_total_leverage
        self.fee_rate = fee_rate

    def optimize_portfolio(
        self,
        opportunities: List[Dict],
        bankroll: float,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Optimize position sizes across multiple opportunities.

        Args:
            opportunities: List of dicts with 'prob', 'price', 'market_id', 'confidence'
            bankroll: Current bankroll
            correlation_matrix: Optional correlation matrix between outcomes

        Returns:
            List of position recommendations
        """
        if not opportunities:
            return []

        n = len(opportunities)

        # Calculate individual Kelly fractions
        kelly_calcs = []
        for opp in opportunities:
            prob = opp["prob"]
            price = opp["price"]
            edge = prob - price

            if abs(edge) < 0.02:  # Minimum edge
                kelly_calcs.append(0.0)
                continue

            if edge > 0:  # Bet YES
                odds = (1 - price) / price
                net_odds = odds * (1 - self.fee_rate)
                kelly = (prob * net_odds - (1 - prob)) / net_odds
            else:  # Bet NO
                no_prob = 1 - prob
                no_price = 1 - price
                odds = (1 - no_price) / no_price
                net_odds = odds * (1 - self.fee_rate)
                kelly = (no_prob * net_odds - prob) / net_odds

            kelly_calcs.append(max(0, kelly))

        # Apply fractional Kelly
        kelly_fractions = [k * self.kelly_fraction for k in kelly_calcs]

        # Cap individual positions
        kelly_fractions = [min(k, self.max_single_position) for k in kelly_fractions]

        # Scale if total exceeds max leverage
        total = sum(kelly_fractions)
        if total > self.max_total_leverage:
            scale = self.max_total_leverage / total
            kelly_fractions = [k * scale for k in kelly_fractions]

        # Build results
        results = []
        for i, opp in enumerate(opportunities):
            if kelly_fractions[i] <= 0:
                continue

            edge = opp["prob"] - opp["price"]
            direction = "YES" if edge > 0 else "NO"

            results.append({
                "market_id": opp.get("market_id", f"market_{i}"),
                "direction": direction,
                "position_fraction": kelly_fractions[i],
                "position_size": kelly_fractions[i] * bankroll,
                "edge": abs(edge),
                "confidence": opp.get("confidence", 1.0),
                "expected_return": abs(edge) * kelly_fractions[i],
            })

        # Sort by expected return
        results.sort(key=lambda x: x["expected_return"], reverse=True)

        return results

    def rebalance_portfolio(
        self,
        current_positions: List[Dict],
        opportunities: List[Dict],
        bankroll: float,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Rebalance existing portfolio with new opportunities.

        Returns:
            Tuple of (positions_to_close, positions_to_open)
        """
        # Get optimal new portfolio
        optimal = self.optimize_portfolio(opportunities, bankroll)
        optimal_ids = {p["market_id"] for p in optimal}

        # Positions to close (in current but not optimal, or direction changed)
        to_close = []
        for pos in current_positions:
            market_id = pos.get("market_id")
            optimal_pos = next((o for o in optimal if o["market_id"] == market_id), None)

            if not optimal_pos or optimal_pos["direction"] != pos.get("direction"):
                to_close.append(pos)

        # Positions to open (in optimal but not current)
        current_ids = {p.get("market_id") for p in current_positions}
        to_open = [o for o in optimal if o["market_id"] not in current_ids]

        return to_close, to_open


# Global instances
kelly = AdvancedKelly()
multi_kelly = MultiAssetKelly()


def calculate_position_size(
    win_prob: float,
    market_price: float,
    confidence: float,
    bankroll: float,
    use_bayesian: bool = False,
    prob_std: float = 0.05,
) -> KellyResult:
    """
    Convenience function to calculate optimal position size.

    Args:
        win_prob: Estimated probability of winning
        market_price: Current market price
        confidence: Model confidence (0-1)
        bankroll: Current bankroll
        use_bayesian: Whether to use Bayesian Kelly
        prob_std: Standard deviation for Bayesian Kelly
    """
    if use_bayesian:
        return kelly.bayesian_kelly(win_prob, prob_std, market_price, bankroll)
    else:
        return kelly.confidence_adjusted_kelly(win_prob, market_price, confidence, bankroll)
