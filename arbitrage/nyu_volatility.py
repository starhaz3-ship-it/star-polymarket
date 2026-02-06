"""
NYU Stern Two-Parameter Volatility Model for Prediction Markets

Based on research paper CeDER-08-07: "Modeling Volatility in Prediction Markets"
by Archak and Ipeirotis, NYU Stern School of Business.

Key insight: Contract volatility is FULLY determined by just two parameters:
1. Current price (π)
2. Time to expiration (T-t)

Formula: σ(t) = φ(N⁻¹(π)) / √(T-t)

Where:
- φ = standard normal PDF
- N⁻¹ = inverse normal CDF (probit function)
- π = current contract price (0 to 1)
- T-t = time to expiration

This simple model OUTPERFORMS complex GARCH models for prediction markets.
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional


# Standard normal PDF
def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


# Inverse standard normal CDF (probit function) - approximation
def norm_ppf(p: float) -> float:
    """
    Inverse standard normal CDF (probit function).
    Uses Abramowitz and Stegun approximation.
    """
    if p <= 0:
        return -10.0  # Clamp to avoid -inf
    if p >= 1:
        return 10.0   # Clamp to avoid +inf

    # For p < 0.5, use symmetry
    if p < 0.5:
        return -norm_ppf(1 - p)

    # Rational approximation for 0.5 <= p < 1
    t = math.sqrt(-2 * math.log(1 - p))

    # Coefficients for approximation
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    return t - (c0 + c1*t + c2*t*t) / (1 + d1*t + d2*t*t + d3*t*t*t)


@dataclass
class NYUVolatilityResult:
    """Result from NYU volatility model."""
    price: float
    time_to_expiry_min: float
    instantaneous_volatility: float
    probit_price: float  # N⁻¹(π)
    phi_value: float     # φ(N⁻¹(π))

    # Edge signals
    is_extreme_price: bool      # Price far from 0.50 (lower vol)
    is_near_expiry: bool        # Close to expiration (vol collapse)
    volatility_regime: str      # "low", "medium", "high"
    edge_score: float           # 0-1 score for trading edge
    recommended_action: str     # "TRADE", "CAUTION", "AVOID"


class NYUVolatilityModel:
    """
    Two-parameter volatility model for prediction markets.

    Key findings from NYU research:
    1. Volatility is HIGHEST at price = 0.50 (φ(0) = 0.399)
    2. Volatility DECREASES as price moves to extremes
    3. Volatility COLLAPSES as time to expiration → 0
    4. Historical data adds NO predictive value
    """

    # Volatility regime thresholds
    HIGH_VOL_THRESHOLD = 0.30
    LOW_VOL_THRESHOLD = 0.15

    # Price zones
    EXTREME_PRICE_LOW = 0.35   # Below this = low vol zone
    EXTREME_PRICE_HIGH = 0.65  # Above this = low vol zone

    # Time thresholds (minutes)
    NEAR_EXPIRY_MIN = 5.0      # Below this = vol collapse zone

    def calculate_volatility(self, price: float, time_to_expiry_min: float) -> NYUVolatilityResult:
        """
        Calculate instantaneous volatility using the NYU two-parameter model.

        Args:
            price: Current contract price (0 to 1)
            time_to_expiry_min: Time to expiration in minutes

        Returns:
            NYUVolatilityResult with volatility and trading signals
        """
        # Clamp price to valid range
        price = max(0.01, min(0.99, price))
        time_to_expiry_min = max(0.1, time_to_expiry_min)

        # Calculate probit (inverse normal CDF)
        probit = norm_ppf(price)

        # Calculate φ(probit) - standard normal PDF at probit value
        phi = norm_pdf(probit)

        # Calculate instantaneous volatility
        # σ = φ(N⁻¹(π)) / √(T-t)
        # Note: We use minutes, so scale appropriately
        time_factor = math.sqrt(time_to_expiry_min / 15.0)  # Normalize to 15-min window
        volatility = phi / max(0.1, time_factor)

        # Determine volatility regime
        if volatility < self.LOW_VOL_THRESHOLD:
            vol_regime = "low"
        elif volatility > self.HIGH_VOL_THRESHOLD:
            vol_regime = "high"
        else:
            vol_regime = "medium"

        # Check if price is extreme (lower volatility zone)
        is_extreme = price < self.EXTREME_PRICE_LOW or price > self.EXTREME_PRICE_HIGH

        # Check if near expiry (volatility collapse)
        is_near_expiry = time_to_expiry_min < self.NEAR_EXPIRY_MIN

        # Calculate edge score (0-1, higher = better edge)
        edge_score = self._calculate_edge_score(price, time_to_expiry_min, volatility)

        # Determine recommended action
        if edge_score >= 0.6:
            action = "TRADE"
        elif edge_score >= 0.4:
            action = "CAUTION"
        else:
            action = "AVOID"

        return NYUVolatilityResult(
            price=price,
            time_to_expiry_min=time_to_expiry_min,
            instantaneous_volatility=volatility,
            probit_price=probit,
            phi_value=phi,
            is_extreme_price=is_extreme,
            is_near_expiry=is_near_expiry,
            volatility_regime=vol_regime,
            edge_score=edge_score,
            recommended_action=action
        )

    def _calculate_edge_score(self, price: float, time_min: float, volatility: float) -> float:
        """
        Calculate trading edge score based on NYU model insights.

        Higher score = better edge because:
        1. Extreme prices have lower volatility (more predictable)
        2. Near expiry, prices converge to true value
        3. Lower volatility = less random noise
        """
        score = 0.0

        # Factor 1: Price extremity (0-0.4 points)
        # Prices near 0.50 have max volatility, extreme prices have less
        distance_from_center = abs(price - 0.50)
        price_score = min(0.4, distance_from_center * 0.8)
        score += price_score

        # Factor 2: Time to expiry (0-0.3 points)
        # Near expiry = volatility collapse = more predictable
        if time_min < 3:
            time_score = 0.3
        elif time_min < 5:
            time_score = 0.2
        elif time_min < 10:
            time_score = 0.1
        else:
            time_score = 0.0
        score += time_score

        # Factor 3: Low volatility bonus (0-0.3 points)
        if volatility < 0.15:
            vol_score = 0.3
        elif volatility < 0.25:
            vol_score = 0.2
        elif volatility < 0.35:
            vol_score = 0.1
        else:
            vol_score = 0.0
        score += vol_score

        return min(1.0, score)

    def get_optimal_entry_zone(self, side: str) -> Tuple[float, float]:
        """
        Get optimal entry price zone based on NYU model.

        For prediction markets, extreme prices have:
        - Lower volatility (more predictable)
        - Better risk/reward (cheap entry)

        Returns:
            (min_price, max_price) for optimal entry
        """
        if side == "UP" or side == "YES":
            # For UP/YES bets, buy when price is LOW (undervalued)
            return (0.20, 0.40)
        else:
            # For DOWN/NO bets, buy when price is LOW (DOWN is cheap)
            return (0.20, 0.40)

    def should_trade(self, price: float, time_min: float,
                     min_edge_score: float = 0.4) -> Tuple[bool, str]:
        """
        Simple decision: should we trade this contract?

        Based on NYU research - forget complex models, just use:
        1. Is price extreme enough? (away from 0.50)
        2. Is there enough time? (not too much, not too little)

        Returns:
            (should_trade, reason)
        """
        result = self.calculate_volatility(price, time_min)

        if result.edge_score >= min_edge_score:
            return True, f"Edge score {result.edge_score:.2f} >= {min_edge_score}"

        # Specific rejection reasons
        if 0.45 <= price <= 0.55:
            return False, "Price in 50% zone (max volatility, no edge)"

        if time_min > 12:
            return False, "Too far from expiry (high uncertainty)"

        if time_min < 2:
            return False, "Too close to expiry (insufficient time)"

        return False, f"Edge score {result.edge_score:.2f} < {min_edge_score}"


# Convenience function
def calculate_nyu_volatility(price: float, time_to_expiry_min: float) -> NYUVolatilityResult:
    """Quick volatility calculation using NYU model."""
    model = NYUVolatilityModel()
    return model.calculate_volatility(price, time_to_expiry_min)


def get_edge_recommendation(price: float, time_to_expiry_min: float) -> str:
    """Get simple trade recommendation based on NYU model."""
    result = calculate_nyu_volatility(price, time_to_expiry_min)
    return f"{result.recommended_action} (edge={result.edge_score:.2f}, vol={result.volatility_regime})"


# Test the model
if __name__ == "__main__":
    model = NYUVolatilityModel()

    print("NYU Two-Parameter Volatility Model")
    print("=" * 60)

    # Test different prices at 10 min to expiry
    print("\nVolatility by Price (10 min to expiry):")
    print("-" * 60)
    for price in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        result = model.calculate_volatility(price, 10.0)
        print(f"Price ${price:.2f}: Vol={result.instantaneous_volatility:.3f}, "
              f"Edge={result.edge_score:.2f}, Action={result.recommended_action}")

    # Test different times at 0.35 price
    print("\nVolatility by Time ($0.35 price):")
    print("-" * 60)
    for time_min in [2, 5, 8, 10, 12, 15]:
        result = model.calculate_volatility(0.35, time_min)
        print(f"Time {time_min:2d}min: Vol={result.instantaneous_volatility:.3f}, "
              f"Edge={result.edge_score:.2f}, Action={result.recommended_action}")
