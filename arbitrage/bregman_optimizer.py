"""
Bregman Divergence Optimizer for Prediction Markets

Based on the mathematical framework from the optimization proof:
- Guaranteed profit from optimal trade δ* is at least D(μ*||θ)
- D is the Bregman divergence between true probability μ* and market price θ
- For prediction markets with log scoring: D = KL divergence

Key insight: The edge is bounded by how much your belief differs from the market,
measured by information-theoretic divergence.

This is the strategy behind gabagool22's $739K profit.

Reference: Appendix A, Proof of Proposition 2.4
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum


class OptimalAction(Enum):
    """Optimal action based on divergence analysis."""
    BUY_YES = "buy_yes"
    BUY_NO = "buy_no"
    NO_TRADE = "no_trade"


@dataclass
class BregmanSignal:
    """Signal from Bregman divergence analysis."""
    # Probabilities
    model_prob: float  # Our estimated true probability μ*
    market_prob: float  # Current market price θ

    # Divergence metrics
    kl_divergence: float  # D(μ*||θ) - information gap
    reverse_kl: float  # D(θ||μ*) - for comparison
    jensen_shannon: float  # Symmetric divergence

    # Trading metrics
    guaranteed_profit: float  # Lower bound on expected profit
    optimal_edge: float  # Edge as percentage
    kelly_fraction: float  # Optimal bet fraction

    # Decision
    action: OptimalAction
    confidence: float  # 0-1 confidence in signal

    # Optimal trade sizing
    optimal_size_pct: float  # % of bankroll
    expected_log_growth: float  # Kelly growth rate


class BregmanOptimizer:
    """
    Calculates optimal trades using Bregman divergence theory.

    The key insight: profit ≥ D(μ*||θ) when you have better information
    than the market. This provides a mathematically optimal trading strategy.
    """

    # Minimum divergence to consider trading
    MIN_DIVERGENCE = 0.01  # ~1% information edge

    # Maximum Kelly fraction (for safety)
    MAX_KELLY = 0.25  # Never bet more than 25% of bankroll

    # Confidence thresholds
    HIGH_CONFIDENCE_DIV = 0.10  # 10% divergence = high confidence

    def __init__(self, bankroll: float = 1000.0):
        self.bankroll = bankroll

    @staticmethod
    def kl_divergence(p: float, q: float) -> float:
        """
        Calculate KL divergence D(P||Q) = p*log(p/q) + (1-p)*log((1-p)/(1-q))

        This measures how much information is lost when Q is used to approximate P.
        In trading terms: how mispriced the market is relative to truth.
        """
        # Avoid log(0) by clamping
        p = max(0.001, min(0.999, p))
        q = max(0.001, min(0.999, q))

        term1 = p * math.log(p / q) if p > 0 else 0
        term2 = (1 - p) * math.log((1 - p) / (1 - q)) if (1 - p) > 0 else 0

        return term1 + term2

    @staticmethod
    def jensen_shannon_divergence(p: float, q: float) -> float:
        """
        Calculate Jensen-Shannon divergence (symmetric version).
        JS(P||Q) = 0.5 * D(P||M) + 0.5 * D(Q||M) where M = (P+Q)/2
        """
        p = max(0.001, min(0.999, p))
        q = max(0.001, min(0.999, q))
        m = (p + q) / 2

        return 0.5 * BregmanOptimizer.kl_divergence(p, m) + 0.5 * BregmanOptimizer.kl_divergence(q, m)

    @staticmethod
    def kelly_criterion(p: float, odds: float) -> float:
        """
        Calculate Kelly criterion for optimal bet sizing.
        f* = (p * odds - (1-p)) / odds = p - (1-p)/odds

        For binary markets where you pay q to win 1:
        odds = (1-q)/q, so f* = (p - q) / (1 - q)
        """
        if odds <= 0:
            return 0

        f = (p * odds - (1 - p)) / odds
        return max(0, f)

    @staticmethod
    def expected_log_growth(p: float, f: float, odds: float) -> float:
        """
        Calculate expected log growth rate for Kelly betting.
        G = p * log(1 + f*odds) + (1-p) * log(1 - f)
        """
        if f <= 0 or f >= 1:
            return 0

        win_growth = math.log(1 + f * odds) if (1 + f * odds) > 0 else -10
        loss_growth = math.log(1 - f) if (1 - f) > 0 else -10

        return p * win_growth + (1 - p) * loss_growth

    def calculate_optimal_trade(
        self,
        model_prob: float,
        market_yes_price: float,
        market_no_price: Optional[float] = None
    ) -> BregmanSignal:
        """
        Calculate optimal trade using Bregman divergence theory.

        Args:
            model_prob: Our estimated true probability (μ*)
            market_yes_price: Current YES price (θ)
            market_no_price: Current NO price (if different from 1-YES)

        Returns:
            BregmanSignal with optimal trade parameters
        """
        # Normalize market price
        if market_no_price is not None:
            total = market_yes_price + market_no_price
            market_prob = market_yes_price / total if total > 0 else 0.5
        else:
            market_prob = market_yes_price

        # Clamp probabilities
        model_prob = max(0.01, min(0.99, model_prob))
        market_prob = max(0.01, min(0.99, market_prob))

        # Calculate divergences
        kl_div = self.kl_divergence(model_prob, market_prob)
        reverse_kl = self.kl_divergence(market_prob, model_prob)
        js_div = self.jensen_shannon_divergence(model_prob, market_prob)

        # The guaranteed profit is at least D(μ*||θ) per the proof
        guaranteed_profit = kl_div

        # Calculate edge as percentage
        optimal_edge = (model_prob - market_prob) * 100  # For YES side

        # Determine optimal action
        if abs(model_prob - market_prob) < 0.02:  # Less than 2% difference
            action = OptimalAction.NO_TRADE
            side_prob = model_prob
            side_price = market_prob
        elif model_prob > market_prob:
            action = OptimalAction.BUY_YES
            side_prob = model_prob
            side_price = market_prob
        else:
            action = OptimalAction.BUY_NO
            side_prob = 1 - model_prob
            side_price = 1 - market_prob

        # Calculate Kelly criterion for the chosen side
        if action != OptimalAction.NO_TRADE:
            # Odds for buying at price q to win $1: odds = (1-q)/q
            odds = (1 - side_price) / side_price if side_price > 0 else 0
            kelly_frac = self.kelly_criterion(side_prob, odds)
            kelly_frac = min(kelly_frac, self.MAX_KELLY)  # Safety cap

            # Expected log growth
            log_growth = self.expected_log_growth(side_prob, kelly_frac, odds)
        else:
            kelly_frac = 0
            log_growth = 0

        # Calculate confidence based on divergence magnitude
        if kl_div >= self.HIGH_CONFIDENCE_DIV:
            confidence = min(0.95, 0.5 + kl_div * 3)
        elif kl_div >= self.MIN_DIVERGENCE:
            confidence = 0.3 + kl_div * 5
        else:
            confidence = kl_div * 10

        return BregmanSignal(
            model_prob=model_prob,
            market_prob=market_prob,
            kl_divergence=kl_div,
            reverse_kl=reverse_kl,
            jensen_shannon=js_div,
            guaranteed_profit=guaranteed_profit,
            optimal_edge=optimal_edge,
            kelly_fraction=kelly_frac,
            action=action,
            confidence=confidence,
            optimal_size_pct=kelly_frac * 100,
            expected_log_growth=log_growth
        )

    def optimize_portfolio(
        self,
        opportunities: List[Tuple[str, float, float]]  # (market_id, model_prob, market_price)
    ) -> List[Tuple[str, BregmanSignal, float]]:
        """
        Optimize portfolio across multiple opportunities.

        Returns list of (market_id, signal, recommended_size) sorted by expected growth.
        """
        results = []

        for market_id, model_prob, market_price in opportunities:
            signal = self.calculate_optimal_trade(model_prob, market_price)

            if signal.action != OptimalAction.NO_TRADE:
                size = self.bankroll * signal.kelly_fraction
                results.append((market_id, signal, size))

        # Sort by expected log growth (Kelly optimality)
        results.sort(key=lambda x: x[1].expected_log_growth, reverse=True)

        return results

    def format_signal(self, signal: BregmanSignal, market_name: str = "") -> str:
        """Format signal for display."""
        lines = [
            "=" * 60,
            f"BREGMAN OPTIMIZATION SIGNAL{f' - {market_name}' if market_name else ''}",
            "=" * 60,
            f"Model Probability: {signal.model_prob:.1%}",
            f"Market Price: {signal.market_prob:.1%}",
            "",
            "DIVERGENCE METRICS:",
            f"  KL Divergence D(μ*||θ): {signal.kl_divergence:.4f}",
            f"  Reverse KL D(θ||μ*): {signal.reverse_kl:.4f}",
            f"  Jensen-Shannon: {signal.jensen_shannon:.4f}",
            "",
            "TRADING METRICS:",
            f"  Guaranteed Profit ≥ {signal.guaranteed_profit:.4f}",
            f"  Edge: {signal.optimal_edge:+.2f}%",
            f"  Kelly Fraction: {signal.kelly_fraction:.1%}",
            f"  Expected Log Growth: {signal.expected_log_growth:.4f}",
            "",
            "DECISION:",
            f"  Action: {signal.action.value.upper()}",
            f"  Confidence: {signal.confidence:.1%}",
            f"  Optimal Size: {signal.optimal_size_pct:.1f}% of bankroll",
            "=" * 60,
        ]
        return "\n".join(lines)


# Integration with TA Signals
def enhance_ta_signal_with_bregman(
    model_up_prob: float,
    market_yes_price: float,
    market_no_price: float,
    bankroll: float = 1000.0
) -> BregmanSignal:
    """
    Enhance TA signal with Bregman divergence optimization.

    Combines technical analysis probability estimate with
    information-theoretic optimal trading.
    """
    optimizer = BregmanOptimizer(bankroll=bankroll)
    return optimizer.calculate_optimal_trade(
        model_prob=model_up_prob,
        market_yes_price=market_yes_price,
        market_no_price=market_no_price
    )


# Global optimizer instance
bregman_optimizer = BregmanOptimizer()


if __name__ == "__main__":
    # Example usage
    optimizer = BregmanOptimizer(bankroll=1000)

    # Test case: Model thinks 70% YES, market at 55%
    signal = optimizer.calculate_optimal_trade(
        model_prob=0.70,
        market_yes_price=0.55
    )

    print(optimizer.format_signal(signal, "BTC Above $80K"))

    # Test case: Model thinks 30% YES (bearish), market at 50%
    signal2 = optimizer.calculate_optimal_trade(
        model_prob=0.30,
        market_yes_price=0.50
    )

    print()
    print(optimizer.format_signal(signal2, "BTC Up or Down"))
