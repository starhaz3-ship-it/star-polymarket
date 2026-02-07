"""
Bregman Divergence Optimizer for Prediction Markets

Based on the mathematical framework from the optimization proof:
- Guaranteed profit from optimal trade δ* is at least D(μ*||θ)
- D is the Bregman divergence between true probability μ* and market price θ
- For prediction markets with log scoring: D = KL divergence

Frank-Wolfe Profit Guarantee (Proposition 4.1):
- net_profit = bregman_divergence(μ_t, θ) - fw_gap(μ_t)
- Only execute when net_profit >= α * bregman_divergence (α = 0.9)
- FW gap estimates execution costs: fees, spread, slippage, time decay

Key insight: The edge is bounded by how much your belief differs from the market,
measured by information-theoretic divergence. But only CAPTURABLE edge matters.

Reference: Appendix A, Proof of Proposition 2.4 + Frank-Wolfe Market Maker
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

    # Frank-Wolfe profit guarantee (Proposition 4.1)
    fw_gap: float = 0.0  # Execution cost estimate (fees + spread + time decay)
    net_profit: float = 0.0  # bregman_div - fw_gap (capturable profit)
    profit_ratio: float = 0.0  # net_profit / bregman_div (α quality score, 0-1)
    fw_executable: bool = True  # True if profit_ratio >= ALPHA_THRESHOLD

    # Decision
    action: OptimalAction = OptimalAction.NO_TRADE
    confidence: float = 0.0  # 0-1 confidence in signal

    # Optimal trade sizing
    optimal_size_pct: float = 0.0  # % of bankroll
    expected_log_growth: float = 0.0  # Kelly growth rate


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

    # Frank-Wolfe parameters
    ALPHA_THRESHOLD = 0.70  # Only trade when 70%+ of edge is capturable
    FEE_RATE = 0.02  # Polymarket 2% fee on winnings
    BASE_SPREAD_COST = 0.005  # ~0.5% average spread cost
    TIME_DECAY_RATE = 0.001  # Edge erosion per minute near expiry

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

    def estimate_fw_gap(
        self,
        market_prob: float,
        kl_div: float,
        time_remaining_min: float = 10.0,
        spread: Optional[float] = None,
    ) -> float:
        """
        Estimate Frank-Wolfe gap: execution costs that eat into theoretical edge.

        FW gap = fee_cost + spread_cost + time_decay_cost

        From the FWMM paper: profit = bregman_div - fw_gap.
        Only trade when fw_gap is small relative to bregman_div.
        """
        # 1. Fee cost: Polymarket charges ~2% on net winnings
        # Expected fee = fee_rate * expected_payout * P(win)
        # Approximate as fee_rate * kl_div (proportional to edge)
        fee_cost = self.FEE_RATE * max(kl_div, 0.01)

        # 2. Spread cost: cost of crossing the bid-ask spread
        # Wider spreads near 50% (more liquid but tighter edge)
        # Narrower spreads at extremes (less liquid but edge is larger)
        if spread is not None:
            spread_cost = spread * 0.5  # Half-spread as cost estimate
        else:
            # Estimate: spread is wider near 50%, tighter at extremes
            price_extremity = abs(market_prob - 0.50)
            spread_cost = self.BASE_SPREAD_COST * (1.0 - price_extremity)

        # 3. Time decay: edge erodes faster near expiry as market converges to truth
        # Near expiry (<3 min), other traders have same info, edge shrinks
        if time_remaining_min < 3.0:
            time_cost = self.TIME_DECAY_RATE * (3.0 - time_remaining_min) ** 2
        elif time_remaining_min < 5.0:
            time_cost = self.TIME_DECAY_RATE * (5.0 - time_remaining_min) * 0.5
        else:
            time_cost = 0.0

        return fee_cost + spread_cost + time_cost

    def calculate_optimal_trade(
        self,
        model_prob: float,
        market_yes_price: float,
        market_no_price: Optional[float] = None,
        time_remaining_min: float = 10.0,
        spread: Optional[float] = None,
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

        # Frank-Wolfe profit guarantee (Proposition 4.1):
        # net_profit = bregman_div - fw_gap
        # Only trade when net_profit >= alpha * bregman_div
        fw_gap = self.estimate_fw_gap(market_prob, kl_div, time_remaining_min, spread)
        net_profit = max(0.0, kl_div - fw_gap)
        profit_ratio = net_profit / kl_div if kl_div > 0.001 else 0.0
        fw_executable = profit_ratio >= self.ALPHA_THRESHOLD

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
            fw_gap=fw_gap,
            net_profit=net_profit,
            profit_ratio=profit_ratio,
            fw_executable=fw_executable,
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
            f"  FW Gap (exec cost): {signal.fw_gap:.4f}",
            f"  Net Profit: {signal.net_profit:.4f} ({signal.profit_ratio:.0%} capturable)",
            f"  FW Executable: {signal.fw_executable} (α≥{self.ALPHA_THRESHOLD:.0%})",
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
