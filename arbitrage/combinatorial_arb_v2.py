"""
Advanced Combinatorial Arbitrage Detection

Based on research paper: "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"
(arXiv:2508.03474v1)

Key findings implemented:
1. Marginal Polytope - Valid outcome space defined by integer constraints
2. Dependency Detection - Logical relationships between markets
3. Frank-Wolfe Algorithm - Iterative projection onto arbitrage-free manifold
4. Multi-condition arbitrage - Beyond simple YES+NO=1

$39.7M extracted using these techniques (April 2024 - April 2025)
Top trader: $2M from 4,049 trades

Reference: gabagool22's $739K strategy
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
import numpy as np


class DependencyType(Enum):
    """Types of logical dependencies between markets."""
    IMPLIES = "implies"           # A → B: If A true, B must be true
    MUTEX = "mutex"               # A ⊕ B: Exactly one is true
    SUBSET = "subset"             # A ⊂ B: A is subset of B
    NEGATION = "negation"         # A = ¬B
    INDEPENDENT = "independent"   # No logical relationship


@dataclass
class MarketCondition:
    """A single condition/outcome in a market."""
    market_id: str
    condition_id: str
    description: str
    price: float
    volume: float = 0.0


@dataclass
class DependencyRelation:
    """Logical dependency between two conditions."""
    condition_a: str
    condition_b: str
    dependency_type: DependencyType
    confidence: float
    # For IMPLIES: if A then B
    # For MUTEX: exactly one of A, B is true


@dataclass
class ArbitrageOpportunity:
    """A detected arbitrage opportunity."""
    opportunity_type: str  # "single", "rebalancing", "combinatorial"
    conditions: List[MarketCondition]
    profit_per_dollar: float
    guaranteed_profit: float
    max_size: float  # Limited by liquidity
    dependencies: List[DependencyRelation] = field(default_factory=list)
    execution_risk: float = 0.0


class MarginalPolytopeChecker:
    """
    Check if prices lie in the marginal polytope (arbitrage-free region).

    For n conditions with k valid outcomes:
    - Naive: 2^n possible combinations
    - Valid: k outcomes (where k << 2^n due to constraints)

    Arbitrage exists when prices assume invalid outcome combinations.
    """

    @staticmethod
    def check_single_market(yes_price: float, no_price: float, tolerance: float = 0.02) -> Optional[ArbitrageOpportunity]:
        """
        Check single market: YES + NO should = 1.0

        Findings: 7,051 conditions (41%) had this type of arbitrage
        Median mispricing: $0.60 (40% error!)
        """
        total = yes_price + no_price

        if total < 1.0 - tolerance:
            # Buy both sides for guaranteed profit
            profit = 1.0 - total
            return ArbitrageOpportunity(
                opportunity_type="single_buy_both",
                conditions=[],
                profit_per_dollar=profit / total,
                guaranteed_profit=profit,
                max_size=1000,  # Would be from order book
            )
        elif total > 1.0 + tolerance:
            # Sell both sides
            profit = total - 1.0
            return ArbitrageOpportunity(
                opportunity_type="single_sell_both",
                conditions=[],
                profit_per_dollar=profit / total,
                guaranteed_profit=profit,
                max_size=1000,
            )

        return None

    @staticmethod
    def check_multi_outcome_market(
        outcome_prices: List[float],
        tolerance: float = 0.02
    ) -> Optional[ArbitrageOpportunity]:
        """
        Check multi-outcome market: sum of all prices should = 1.0

        From paper: 42% of multi-condition markets had rebalancing opportunities
        """
        total = sum(outcome_prices)

        if total < 1.0 - tolerance:
            # Buy all outcomes
            profit = 1.0 - total
            return ArbitrageOpportunity(
                opportunity_type="rebalancing_buy_all",
                conditions=[],
                profit_per_dollar=profit / total,
                guaranteed_profit=profit,
                max_size=min(p * 1000 for p in outcome_prices),  # Limited by smallest position
            )
        elif total > 1.0 + tolerance:
            # Sell all outcomes
            profit = total - 1.0
            return ArbitrageOpportunity(
                opportunity_type="rebalancing_sell_all",
                conditions=[],
                profit_per_dollar=profit / total,
                guaranteed_profit=profit,
                max_size=min(p * 1000 for p in outcome_prices),
            )

        return None


class DependencyDetector:
    """
    Detect logical dependencies between markets.

    From paper: Used DeepSeek-R1-Distill-Qwen-32B with 81.45% accuracy
    Found 1,576 dependent pairs from 46,360 possible pairs
    """

    # Common dependency patterns
    IMPLICATION_KEYWORDS = [
        ("if", "then"),
        ("requires", ""),
        ("only if", ""),
        ("win", "advance"),
    ]

    MUTEX_KEYWORDS = [
        ("vs", ""),
        ("or", ""),
        ("either", ""),
    ]

    @staticmethod
    def detect_implication(
        market_a: str,
        market_b: str
    ) -> Optional[DependencyRelation]:
        """
        Check if A → B (A implies B).

        Example: "Republicans win by 5+ points" → "Trump wins Pennsylvania"
        """
        a_lower = market_a.lower()
        b_lower = market_b.lower()

        # Check for subset relationships
        # e.g., "BTC > $100K" implies "BTC > $90K"

        # Extract numeric thresholds if present
        import re
        a_nums = re.findall(r'\$?([\d,]+)', a_lower)
        b_nums = re.findall(r'\$?([\d,]+)', b_lower)

        if a_nums and b_nums:
            a_val = float(a_nums[0].replace(',', ''))
            b_val = float(b_nums[0].replace(',', ''))

            # Check if same asset with different thresholds
            if "above" in a_lower and "above" in b_lower:
                # Higher threshold implies lower threshold
                if a_val > b_val:
                    return DependencyRelation(
                        condition_a=market_a,
                        condition_b=market_b,
                        dependency_type=DependencyType.IMPLIES,
                        confidence=0.9
                    )

            if "below" in a_lower and "below" in b_lower:
                # Lower threshold implies higher threshold
                if a_val < b_val:
                    return DependencyRelation(
                        condition_a=market_a,
                        condition_b=market_b,
                        dependency_type=DependencyType.IMPLIES,
                        confidence=0.9
                    )

        return None

    @staticmethod
    def check_dependency_arbitrage(
        market_a_yes: float,
        market_a_no: float,
        market_b_yes: float,
        market_b_no: float,
        dependency: DependencyRelation
    ) -> Optional[ArbitrageOpportunity]:
        """
        Check for arbitrage given a dependency relationship.

        If A → B, then P(A) ≤ P(B)
        If A → B and P(A) > P(B), arbitrage exists.
        """
        if dependency.dependency_type == DependencyType.IMPLIES:
            # A implies B means P(A) should be ≤ P(B)
            if market_a_yes > market_b_yes + 0.02:
                # Arbitrage: sell A_yes, buy B_yes
                profit = market_a_yes - market_b_yes
                return ArbitrageOpportunity(
                    opportunity_type="combinatorial_implication",
                    conditions=[],
                    profit_per_dollar=profit,
                    guaranteed_profit=profit,
                    max_size=500,
                    dependencies=[dependency],
                )

        elif dependency.dependency_type == DependencyType.MUTEX:
            # Exactly one is true: P(A) + P(B) should = 1
            total = market_a_yes + market_b_yes
            if abs(total - 1.0) > 0.05:
                profit = abs(1.0 - total)
                return ArbitrageOpportunity(
                    opportunity_type="combinatorial_mutex",
                    conditions=[],
                    profit_per_dollar=profit / max(total, 1.0),
                    guaranteed_profit=profit,
                    max_size=500,
                    dependencies=[dependency],
                )

        return None


class FrankWolfeProjector:
    """
    ALGORITHM 2: ProjectFW - Bregman Projection via Adaptive Fully-Corrective Frank-Wolfe

    From paper: "Unravelling the Probabilistic Forest" (arXiv:2508.03474v1)

    Input:
        - cost function C, state θ, partial outcome σ
        - IP constraints specified by A, b
        - approx ratio α ∈ (0,1), initial contraction ε₀ ∈ (0,1)
        - convergence threshold εD > 0

    Output:
        - extended partial outcome σ̂ ⊇ σ
        - state θ̂ whose price vector is approx Bregman projection of θ on M_σ

    Guarantees:
        1. p_σ(θ̂) ∈ M_σ and moving from θ to θ̂ guarantees profit of αD_σ̂(μ*||θ)
        2. θ̂ = θ and D_σ̂(μ*||θ) ≤ εD
        3. algorithm was interrupted; moving θ to θ̂ guarantees non-negative profit
    """

    def __init__(
        self,
        alpha: float = 0.9,           # Approximation ratio - extract at least 90% of arbitrage
        epsilon_init: float = 0.1,    # Initial contraction ε₀
        epsilon_d: float = 1e-6,      # Convergence threshold εD
        max_iterations: int = 150
    ):
        self.alpha = alpha
        self.epsilon_0 = epsilon_init
        self.epsilon_d = epsilon_d
        self.max_iter = max_iterations

    def _compute_interior_point(self, n: int) -> np.ndarray:
        """Initialize interior point u with all coordinates strictly in (0,1)."""
        return np.ones(n) / n  # Uniform distribution as interior point

    def _entropy(self, mu: np.ndarray) -> float:
        """Compute negative entropy R(μ) = Σ μᵢ·ln(μᵢ)"""
        mu_safe = np.clip(mu, 1e-10, 1.0)
        return np.sum(mu_safe * np.log(mu_safe))

    def _entropy_gradient(self, mu: np.ndarray) -> np.ndarray:
        """Compute ∇R(μ) = ln(μ) + 1"""
        mu_safe = np.clip(mu, 1e-10, 1.0)
        return np.log(mu_safe) + 1

    def _cost_function(self, theta: np.ndarray) -> float:
        """LMSR cost function C(θ) = ln(Σ exp(θᵢ))"""
        return np.log(np.sum(np.exp(np.clip(theta, -100, 100))))

    def _objective_F(self, mu: np.ndarray, theta: np.ndarray) -> float:
        """
        Objective function F(μ) = R_σ(μ) - θ·μ + C_σ(θ)
        For Bregman divergence: F(μ) = D(μ||θ)
        """
        return self._entropy(mu) - np.dot(theta, mu) + self._cost_function(theta)

    def _contract_active_set(
        self,
        active_set: List[np.ndarray],
        u: np.ndarray,
        epsilon: float
    ) -> List[np.ndarray]:
        """
        Contract active set: Z' = (1 - ε)Z + εu
        This ensures bounded gradients on contracted polytope.
        """
        return [(1 - epsilon) * z + epsilon * u for z in active_set]

    def _solve_over_convex_hull(
        self,
        contracted_set: List[np.ndarray],
        theta: np.ndarray
    ) -> np.ndarray:
        """
        Solve μ_t = argmin_{μ∈conv(Z')} F(μ)
        Simplified: weighted combination minimizing objective.
        """
        if len(contracted_set) == 1:
            return contracted_set[0].copy()

        # Use simple averaging weighted by inverse objective
        best_mu = contracted_set[0]
        best_val = self._objective_F(best_mu, theta)

        for z in contracted_set[1:]:
            val = self._objective_F(z, theta)
            if val < best_val:
                best_val = val
                best_mu = z

        return best_mu.copy()

    def _find_descent_vertex(
        self,
        theta_t: np.ndarray,
        theta: np.ndarray,
        valid_outcomes: List[np.ndarray]
    ) -> np.ndarray:
        """
        Call IP solver to find descent vertex:
        z_t = argmin_{z∈Z_σ} (θ_t - θ)·z

        In production, this would use Gurobi IP solver.
        """
        direction = theta_t - theta
        best_vertex = valid_outcomes[0]
        best_value = np.dot(direction, best_vertex)

        for z in valid_outcomes[1:]:
            value = np.dot(direction, z)
            if value < best_value:
                best_value = value
                best_vertex = z

        return best_vertex.copy()

    def _compute_fw_gap(
        self,
        theta_t: np.ndarray,
        theta: np.ndarray,
        mu_t: np.ndarray,
        z_t: np.ndarray
    ) -> float:
        """
        Compute Frank-Wolfe gap: g(μ_t) = (θ_t - θ)·(μ_t - z_t)
        The gap measures progress toward optimum.
        """
        return np.dot(theta_t - theta, mu_t - z_t)

    def project_to_simplex(self, prices: np.ndarray) -> np.ndarray:
        """Project prices onto probability simplex (sum = 1, all ≥ 0)."""
        n = len(prices)
        sorted_prices = np.sort(prices)[::-1]
        cumsum = np.cumsum(sorted_prices)
        rho = np.arange(1, n + 1)
        test = sorted_prices - (cumsum - 1) / rho
        indices = np.where(test > 0)[0]
        if len(indices) == 0:
            return np.ones(n) / n
        rho_star = indices[-1] + 1
        theta = (cumsum[rho_star - 1] - 1) / rho_star
        return np.maximum(prices - theta, 0)

    def bregman_projection(
        self,
        current_prices: np.ndarray,
        valid_outcomes: List[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, dict]:
        """
        Compute Bregman projection using Adaptive Fully-Corrective Frank-Wolfe.

        Returns: (projected_prices, guaranteed_profit, metadata)

        The guaranteed profit moving from θ to θ̂ is at least αD(μ*||θ).
        """
        theta = current_prices.copy()
        n = len(theta)

        if valid_outcomes is None or len(valid_outcomes) == 0:
            projected = self.project_to_simplex(theta)
            div = self._objective_F(projected, theta)
            return projected, max(0, div), {"iterations": 0, "converged": True}

        # Initialize interior point and active vertex set
        u = self._compute_interior_point(n)
        active_set = [valid_outcomes[0].copy()]  # Z_0
        epsilon_t = self.epsilon_0

        best_mu = active_set[0]
        best_F = self._objective_F(best_mu, theta)
        best_t = 0

        metadata = {"iterations": 0, "converged": False, "gaps": []}

        for t in range(1, self.max_iter + 1):
            # Contract active set: Z' = (1 - ε_{t-1})Z_{t-1} + ε_{t-1}u
            contracted_set = self._contract_active_set(active_set, u, epsilon_t)

            # Solve μ_t = argmin_{μ∈conv(Z')} F(μ)
            mu_t = self._solve_over_convex_hull(contracted_set, theta)

            # Compute gradient θ_t = ∇R_σ(μ_t)
            theta_t = self._entropy_gradient(mu_t)

            # Find descent vertex: z_t = argmin_{z∈Z_σ} (θ_t - θ)·z
            z_t = self._find_descent_vertex(theta_t, theta, valid_outcomes)

            # Add to active set: Z_t = Z_{t-1} ∪ {z_t}
            active_set.append(z_t)

            # Compute FW gap: g(μ_t) = (θ_t - θ)·(μ_t - z_t)
            gap = self._compute_fw_gap(theta_t, theta, mu_t, z_t)
            metadata["gaps"].append(gap)

            # Update best iterate: t* = argmax_{τ≤t} [F(μ_τ) - g(μ_τ)]
            F_t = self._objective_F(mu_t, theta)
            if F_t - gap > best_F - metadata["gaps"][best_t] if best_t < len(metadata["gaps"]) else best_F:
                best_mu = mu_t.copy()
                best_F = F_t
                best_t = t - 1

            # Check stopping conditions
            if gap <= (1 - self.alpha) * F_t:
                # Achieved α approximation
                metadata["converged"] = True
                metadata["stop_reason"] = "alpha_approximation"
                break

            if F_t <= self.epsilon_d:
                # Converged to threshold
                metadata["converged"] = True
                metadata["stop_reason"] = "threshold"
                break

            # Adapt contraction parameter
            g_u = np.dot(theta_t - theta, mu_t - u)
            if g_u < 0 and gap / (-4 * g_u) < epsilon_t:
                epsilon_t = min(gap / (-4 * g_u), epsilon_t / 2)

        metadata["iterations"] = t
        metadata["final_epsilon"] = epsilon_t

        # Guaranteed profit is αD(μ*||θ)
        guaranteed_profit = self.alpha * max(0, best_F)

        return best_mu, guaranteed_profit, metadata


class AdvancedArbitrageScanner:
    """
    Complete arbitrage scanning system.

    Combines:
    - Single market checks (41% of conditions)
    - Multi-outcome rebalancing (42% of markets)
    - Combinatorial cross-market (13 pairs found)
    """

    def __init__(self):
        self.polytope_checker = MarginalPolytopeChecker()
        self.dependency_detector = DependencyDetector()
        self.projector = FrankWolfeProjector()
        self.opportunities: List[ArbitrageOpportunity] = []

    def scan_single_markets(
        self,
        markets: List[Dict]
    ) -> List[ArbitrageOpportunity]:
        """Scan for single-market arbitrage."""
        opps = []

        for market in markets:
            yes_price = market.get("yes_price", 0.5)
            no_price = market.get("no_price", 0.5)

            opp = self.polytope_checker.check_single_market(yes_price, no_price)
            if opp:
                opp.conditions = [MarketCondition(
                    market_id=market.get("id", ""),
                    condition_id=market.get("condition_id", ""),
                    description=market.get("question", ""),
                    price=yes_price,
                )]
                opps.append(opp)

        return opps

    def scan_multi_outcome(
        self,
        markets: List[Dict]
    ) -> List[ArbitrageOpportunity]:
        """Scan multi-outcome markets for rebalancing opportunities."""
        opps = []

        for market in markets:
            outcomes = market.get("outcomes", [])
            prices = market.get("prices", [])

            if len(prices) > 2:
                opp = self.polytope_checker.check_multi_outcome_market(prices)
                if opp:
                    opps.append(opp)

        return opps

    def scan_cross_market(
        self,
        market_pairs: List[Tuple[Dict, Dict]]
    ) -> List[ArbitrageOpportunity]:
        """
        Scan for cross-market combinatorial arbitrage.

        From paper: 13 valid pairs found from 1,576 dependent pairs.
        """
        opps = []

        for market_a, market_b in market_pairs:
            q_a = market_a.get("question", "")
            q_b = market_b.get("question", "")

            # Check for implication dependency
            dep = self.dependency_detector.detect_implication(q_a, q_b)

            if dep:
                opp = self.dependency_detector.check_dependency_arbitrage(
                    market_a.get("yes_price", 0.5),
                    market_a.get("no_price", 0.5),
                    market_b.get("yes_price", 0.5),
                    market_b.get("no_price", 0.5),
                    dep
                )
                if opp:
                    opps.append(opp)

        return opps

    def full_scan(self, markets: List[Dict]) -> List[ArbitrageOpportunity]:
        """Run complete arbitrage scan."""
        all_opps = []

        # Single market
        all_opps.extend(self.scan_single_markets(markets))

        # Multi-outcome
        all_opps.extend(self.scan_multi_outcome(markets))

        # Cross-market (generate pairs)
        pairs = []
        for i, m1 in enumerate(markets):
            for m2 in markets[i+1:]:
                pairs.append((m1, m2))

        all_opps.extend(self.scan_cross_market(pairs[:100]))  # Limit pairs checked

        # Sort by profit
        all_opps.sort(key=lambda x: x.guaranteed_profit, reverse=True)

        self.opportunities = all_opps
        return all_opps


# Global scanner instance
advanced_scanner = AdvancedArbitrageScanner()


if __name__ == "__main__":
    # Test the scanner
    test_markets = [
        {
            "id": "btc_100k",
            "question": "Will Bitcoin be above $100,000?",
            "yes_price": 0.45,
            "no_price": 0.50,  # Sum = 0.95, arbitrage!
        },
        {
            "id": "btc_90k",
            "question": "Will Bitcoin be above $90,000?",
            "yes_price": 0.40,  # Should be > btc_100k!
            "no_price": 0.55,
        },
    ]

    scanner = AdvancedArbitrageScanner()
    opps = scanner.full_scan(test_markets)

    print("ARBITRAGE OPPORTUNITIES FOUND:")
    print("=" * 50)
    for opp in opps:
        print(f"Type: {opp.opportunity_type}")
        print(f"Profit: ${opp.guaranteed_profit:.4f} per $1")
        print(f"Max Size: ${opp.max_size:.2f}")
        print("-" * 50)
