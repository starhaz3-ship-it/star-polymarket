"""
Arbitrage opportunity detector.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .config import config, Config
from .polymarket_feed import BTCMarket, MultiOutcomeMarket


class SignalType(Enum):
    """Type of trading signal."""
    BUY_YES = "buy_yes"  # Spot price likely to be above strike
    BUY_NO = "buy_no"    # Spot price likely to be below strike
    BUY_BOTH = "buy_both"  # Same-market arbitrage (buy both sides)
    NEGRISK = "negrisk"  # Multi-outcome arbitrage (buy all outcomes)
    ENDGAME = "endgame"  # High-probability near resolution
    CROSS_MARKET = "cross_market"  # Cross-market dependency arbitrage (AFT 2025)


@dataclass
class ArbitrageSignal:
    """A detected arbitrage opportunity."""
    signal_type: SignalType
    market: BTCMarket
    spot_price: float
    edge_percent: float  # Expected edge after fees
    confidence: float    # 0-1 confidence score
    recommended_size: float  # Recommended position size in USD
    timestamp: float

    @property
    def is_profitable(self) -> bool:
        """Check if signal is expected to be profitable after fees."""
        return self.edge_percent > config.MIN_EDGE_PERCENT

    def __str__(self) -> str:
        if self.signal_type in (SignalType.NEGRISK, SignalType.ENDGAME):
            return (
                f"Signal: {self.signal_type.value} | "
                f"Edge: {self.edge_percent:.2f}% | "
                f"Outcomes: {getattr(self.market, 'outcome_count', 2)}"
            )
        return (
            f"Signal: {self.signal_type.value} | "
            f"Edge: {self.edge_percent:.2f}% | "
            f"Strike: ${self.market.strike_price:,.0f} | "
            f"Spot: ${self.spot_price:,.0f}"
        )


class ArbitrageDetector:
    """Detects arbitrage opportunities between spot and Polymarket."""

    def __init__(self):
        self.last_signals: List[ArbitrageSignal] = []

    def _calculate_implied_probability(
        self,
        spot_price: float,
        strike_price: float,
        time_remaining_sec: float
    ) -> float:
        """
        Calculate implied probability that BTC will be above strike at expiry.

        Simple model: Based on distance from strike and time remaining.
        More sophisticated models could use volatility estimates.
        """
        # Price difference as percentage
        diff_pct = (spot_price - strike_price) / strike_price

        # Time factor: less time = more certain
        time_factor = max(0.1, min(1.0, time_remaining_sec / 900))  # 15 min = 900 sec

        # Base probability from current price position
        if diff_pct > 0:
            # Above strike - higher probability of YES
            prob = 0.5 + min(0.45, diff_pct * 10 / time_factor)
        else:
            # Below strike - lower probability of YES
            prob = 0.5 + max(-0.45, diff_pct * 10 / time_factor)

        return max(0.01, min(0.99, prob))

    def _calculate_edge(
        self,
        implied_prob: float,
        market_price: float,
        is_yes: bool
    ) -> float:
        """
        Calculate edge (expected value) of a trade.

        Returns percentage edge (positive = profitable).
        """
        if is_yes:
            # Buying YES: profit if outcome is YES
            # EV = implied_prob * (1 - market_price) - (1 - implied_prob) * market_price
            # EV = implied_prob - market_price
            ev = implied_prob - market_price
        else:
            # Buying NO: profit if outcome is NO
            ev = (1 - implied_prob) - market_price

        # Account for fees
        fee = Config.estimate_fee(market_price)
        net_ev = ev - fee

        return net_ev * 100  # Return as percentage

    def detect_latency_arbitrage(
        self,
        market: BTCMarket,
        spot_price: float
    ) -> Optional[ArbitrageSignal]:
        """
        Detect latency arbitrage opportunity.

        This is when spot price has moved but Polymarket odds haven't caught up.
        """
        if not market.is_active:
            return None

        # Calculate what the probability "should" be
        implied_prob = self._calculate_implied_probability(
            spot_price,
            market.strike_price,
            market.time_remaining_sec
        )

        # Calculate edge for YES and NO
        yes_edge = self._calculate_edge(implied_prob, market.yes_ask, is_yes=True)
        no_edge = self._calculate_edge(1 - implied_prob, market.no_ask, is_yes=False)

        # Find best opportunity
        best_edge = max(yes_edge, no_edge)
        is_yes_better = yes_edge > no_edge

        if best_edge < config.MIN_EDGE_PERCENT:
            return None

        # Calculate confidence based on:
        # - Size of edge (larger = more confident)
        # - Time remaining (less time = more confident in price)
        # - Distance from strike (further = more confident)
        edge_factor = min(1.0, best_edge / 10)
        time_factor = max(0.3, 1 - market.time_remaining_sec / 900)
        distance_factor = min(1.0, abs(spot_price - market.strike_price) / market.strike_price * 20)

        confidence = (edge_factor + time_factor + distance_factor) / 3

        # Recommended size based on edge and confidence
        kelly_fraction = best_edge / 100 * confidence
        recommended_size = min(
            config.MAX_POSITION_SIZE,
            max(config.MIN_POSITION_SIZE, config.MAX_POSITION_SIZE * kelly_fraction)
        )

        return ArbitrageSignal(
            signal_type=SignalType.BUY_YES if is_yes_better else SignalType.BUY_NO,
            market=market,
            spot_price=spot_price,
            edge_percent=best_edge,
            confidence=confidence,
            recommended_size=recommended_size,
            timestamp=time.time()
        )

    def detect_same_market_arbitrage(
        self,
        market: BTCMarket
    ) -> Optional[ArbitrageSignal]:
        """
        Detect same-market arbitrage (buy both YES and NO).

        Profitable when: YES_ask + NO_ask < 1.0 - fees
        """
        if not market.is_active:
            return None

        # Calculate combined cost to buy both sides
        combined_cost = market.yes_ask + market.no_ask

        # Estimate fees for both sides
        yes_fee = Config.estimate_fee(market.yes_ask)
        no_fee = Config.estimate_fee(market.no_ask)
        total_fees = yes_fee + no_fee

        # Net profit per dollar
        gross_profit = 1.0 - combined_cost
        net_profit = gross_profit - total_fees

        edge_percent = net_profit * 100

        if edge_percent < config.MIN_EDGE_PERCENT:
            return None

        # Higher confidence for same-market arb (risk-free if executed)
        confidence = min(0.95, 0.5 + edge_percent / 10)

        return ArbitrageSignal(
            signal_type=SignalType.BUY_BOTH,
            market=market,
            spot_price=0,  # Not relevant for same-market arb
            edge_percent=edge_percent,
            confidence=confidence,
            recommended_size=config.MAX_POSITION_SIZE,  # Max size for guaranteed profit
            timestamp=time.time()
        )

    def detect_negrisk_arbitrage(
        self,
        market: MultiOutcomeMarket
    ) -> Optional[ArbitrageSignal]:
        """
        Detect NegRisk multi-outcome arbitrage.

        Profitable when: sum(all outcome asks) < 1.0 - fees
        This is the $29M strategy with 29x capital efficiency!
        """
        if not market.is_active:
            return None

        if market.outcome_count < config.NEGRISK_MIN_OUTCOMES:
            return None

        # Filter: only consider markets where outcomes are mutually exclusive
        # These should have total probability between 90% and 110%
        total_prob = market.total_probability
        if total_prob < 0.9 or total_prob > 1.1:
            return None  # Outcomes are not mutually exclusive

        # Calculate total cost to buy all outcomes
        total_ask = market.total_ask

        # Estimate total fees (sum of fees for each outcome)
        total_fees = sum(
            Config.estimate_fee(o.get("ask", 0.5))
            for o in market.outcomes
        )

        # Net profit per dollar
        gross_profit = 1.0 - total_ask
        net_profit = gross_profit - total_fees

        edge_percent = net_profit * 100

        if edge_percent < config.MIN_NEGRISK_EDGE:
            return None

        # Higher confidence for NegRisk (guaranteed if executed atomically)
        confidence = min(0.95, 0.6 + edge_percent / 10)

        # Risk adjustment based on outcome count and time
        if market.outcome_count > 10:
            confidence *= 0.9  # More outcomes = harder to execute

        if market.time_remaining_sec and market.time_remaining_sec < 7200:
            confidence *= 0.8  # Close to resolution = risky

        return ArbitrageSignal(
            signal_type=SignalType.NEGRISK,
            market=market,
            spot_price=0,  # Not relevant for NegRisk
            edge_percent=edge_percent,
            confidence=confidence,
            recommended_size=config.MAX_POSITION_SIZE,  # Max size for guaranteed profit
            timestamp=time.time()
        )

    def detect_endgame_arbitrage(
        self,
        market: MultiOutcomeMarket
    ) -> Optional[ArbitrageSignal]:
        """
        Detect endgame arbitrage opportunities.

        Targets high-probability outcomes (95%+) near resolution.
        Annualized returns can exceed 548%!
        """
        if not market.is_active:
            return None

        # Check time remaining
        time_remaining = market.time_remaining_sec
        if not time_remaining or time_remaining > config.ENDGAME_TIME_HOURS * 3600:
            return None

        # Find highest probability outcome
        top_outcome = market.highest_prob_outcome
        if not top_outcome:
            return None

        prob = top_outcome.get("price", 0)
        ask_price = top_outcome.get("ask", 1.0)

        # Must be high probability
        if prob < config.ENDGAME_PROB_THRESHOLD:
            return None

        # Calculate edge: we buy at ask, receive $1 if correct
        # Expected value = prob * 1.0 - ask_price
        fee = Config.estimate_fee(ask_price)
        expected_profit = prob - ask_price - fee
        edge_percent = (expected_profit / ask_price) * 100

        if edge_percent < config.MIN_EDGE_PERCENT:
            return None

        # Calculate annualized return for context
        hours_remaining = time_remaining / 3600
        if hours_remaining > 0:
            annualized = (edge_percent / 100) * (8760 / hours_remaining) * 100
        else:
            annualized = 0

        # Confidence based on probability and time
        confidence = prob * 0.9  # High prob = high confidence

        return ArbitrageSignal(
            signal_type=SignalType.ENDGAME,
            market=market,
            spot_price=annualized,  # Store annualized return here
            edge_percent=edge_percent,
            confidence=confidence,
            recommended_size=min(config.MAX_POSITION_SIZE, config.MAX_POSITION_SIZE * confidence),
            timestamp=time.time()
        )

    def scan_multi_outcome(
        self,
        markets: List[MultiOutcomeMarket]
    ) -> List[ArbitrageSignal]:
        """Scan multi-outcome markets for NegRisk and Endgame opportunities."""
        signals = []

        for market in markets:
            # Check NegRisk arbitrage
            negrisk_signal = self.detect_negrisk_arbitrage(market)
            if negrisk_signal and negrisk_signal.is_profitable:
                signals.append(negrisk_signal)

            # Check Endgame arbitrage
            endgame_signal = self.detect_endgame_arbitrage(market)
            if endgame_signal and endgame_signal.is_profitable:
                signals.append(endgame_signal)

        return signals

    def scan_all(
        self,
        markets: List[BTCMarket],
        spot_price: float
    ) -> List[ArbitrageSignal]:
        """
        Scan all markets for arbitrage opportunities.

        Returns list of signals sorted by edge (best first).
        """
        signals = []

        for market in markets:
            # Check latency arbitrage
            latency_signal = self.detect_latency_arbitrage(market, spot_price)
            if latency_signal and latency_signal.is_profitable:
                signals.append(latency_signal)

            # Check same-market arbitrage
            same_market_signal = self.detect_same_market_arbitrage(market)
            if same_market_signal and same_market_signal.is_profitable:
                signals.append(same_market_signal)

        # Sort by edge (best opportunities first)
        signals.sort(key=lambda s: s.edge_percent, reverse=True)

        self.last_signals = signals
        return signals


if __name__ == "__main__":
    # Test the detector
    from .polymarket_feed import PolymarketFeed
    from .spot_feed import get_btc_price

    spot = get_btc_price()
    print(f"Current BTC: ${spot:,.2f}")

    feed = PolymarketFeed()
    markets = feed.fetch_btc_markets()

    print(f"\nFound {len(markets)} active markets")

    detector = ArbitrageDetector()

    for market in markets:
        feed.update_market_prices(market)

    signals = detector.scan_all(markets, spot)

    print(f"\nFound {len(signals)} arbitrage opportunities:\n")
    for signal in signals[:5]:
        print(signal)
        print(f"  Market: {signal.market.question[:60]}...")
        print(f"  Confidence: {signal.confidence:.2f} | Size: ${signal.recommended_size:.2f}")
        print()
