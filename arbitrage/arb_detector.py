"""
Arbitrage Detection Module

Implements profitable strategies from research:
1. Rebalancing Arbitrage - Buy all outcomes when sum < 100%
2. Combinatorial Arbitrage - Exploit logical inconsistencies
3. Momentum Lag Arbitrage - BTC spot vs Polymarket odds
4. Cross-Platform Arbitrage - Polymarket vs Kalshi
"""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import httpx
import re


@dataclass
class ArbitrageOpportunity:
    """An arbitrage opportunity."""
    arb_type: str  # "rebalancing", "combinatorial", "momentum_lag", "cross_platform"
    market_id: str
    market_title: str
    profit_margin: float  # Expected profit as percentage
    confidence: float  # How confident we are this is real
    action: str  # Description of what to do
    details: Dict  # Additional details


class RebalancingDetector:
    """
    Detect rebalancing arbitrage opportunities.

    When sum of all outcome prices < 1 (minus fees), buy all outcomes
    for guaranteed profit.

    Research shows $40M extracted via this strategy (Apr 2024 - Apr 2025).
    """

    def __init__(self, min_profit: float = 0.01, fee_rate: float = 0.02):
        """
        Args:
            min_profit: Minimum profit margin to consider (default 1%)
            fee_rate: Winner fee rate (2% on Polymarket)
        """
        self.min_profit = min_profit
        self.fee_rate = fee_rate

    async def scan(self) -> List[ArbitrageOpportunity]:
        """Scan for rebalancing arbitrage opportunities."""
        opportunities = []

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Get active markets
                r = await client.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={"active": "true", "closed": "false", "limit": 200}
                )
                markets = r.json()

                # Group by event/condition
                events = {}
                for m in markets:
                    event_id = m.get("eventId") or m.get("conditionId")
                    if event_id not in events:
                        events[event_id] = []
                    events[event_id].append(m)

                # Check each event for arbitrage
                for event_id, event_markets in events.items():
                    opp = self._check_event(event_markets)
                    if opp:
                        opportunities.append(opp)

        except Exception as e:
            print(f"[RebalancingDetector] Scan error: {e}")

        return opportunities

    def _check_event(self, markets: List[Dict]) -> Optional[ArbitrageOpportunity]:
        """Check if an event has rebalancing arbitrage."""
        if len(markets) < 2:
            return None

        # Get YES prices for all outcomes
        outcomes = []
        for m in markets:
            try:
                prices = m.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    import json
                    prices = json.loads(prices)
                if prices:
                    yes_price = float(prices[0])
                    outcomes.append({
                        "market": m,
                        "price": yes_price,
                        "question": m.get("question", "")[:50],
                    })
            except:
                pass

        if len(outcomes) < 2:
            return None

        # Sum of prices
        total_price = sum(o["price"] for o in outcomes)

        # Profit after fees = 1 - total_price - (fee_rate * 1)
        # We pay total_price, receive 1 (guaranteed), pay fee on winnings
        net_profit = 1 - total_price - self.fee_rate

        if net_profit > self.min_profit:
            return ArbitrageOpportunity(
                arb_type="rebalancing",
                market_id=markets[0].get("conditionId", ""),
                market_title=markets[0].get("question", "")[:60],
                profit_margin=net_profit,
                confidence=0.95,  # High confidence - pure arbitrage
                action=f"Buy all {len(outcomes)} outcomes",
                details={
                    "total_price": total_price,
                    "outcomes": [(o["question"], o["price"]) for o in outcomes],
                    "investment_per_outcome": [1 / o["price"] for o in outcomes],
                }
            )

        return None


class MomentumLagDetector:
    """
    Detect momentum lag between BTC spot price and Polymarket odds.

    Research shows one bot turned $313 -> $414,000 exploiting this lag
    in BTC 15-minute up/down markets.
    """

    def __init__(self, min_edge: float = 0.10, lookback_seconds: int = 60):
        """
        Args:
            min_edge: Minimum edge (implied vs actual probability difference)
            lookback_seconds: How far back to look for momentum
        """
        self.min_edge = min_edge
        self.lookback_seconds = lookback_seconds
        self.btc_price_history: List[Tuple[float, float]] = []  # (timestamp, price)

    async def get_btc_price(self) -> float:
        """Get current BTC price."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    "https://api.binance.com/api/v3/ticker/price",
                    params={"symbol": "BTCUSDT"}
                )
                price = float(r.json()["price"])

                # Update history
                now = datetime.now().timestamp()
                self.btc_price_history.append((now, price))

                # Keep only recent history
                cutoff = now - 300  # 5 minutes
                self.btc_price_history = [
                    (t, p) for t, p in self.btc_price_history if t > cutoff
                ]

                return price
        except:
            return 0.0

    def _calculate_momentum(self) -> float:
        """Calculate BTC price momentum."""
        if len(self.btc_price_history) < 2:
            return 0.0

        now = datetime.now().timestamp()
        cutoff = now - self.lookback_seconds

        old_prices = [p for t, p in self.btc_price_history if t <= cutoff]
        if not old_prices:
            old_prices = [self.btc_price_history[0][1]]

        current_price = self.btc_price_history[-1][1]
        old_price = old_prices[-1]

        if old_price > 0:
            return (current_price - old_price) / old_price
        return 0.0

    async def scan(self) -> List[ArbitrageOpportunity]:
        """Scan for momentum lag opportunities."""
        opportunities = []
        btc_price = await self.get_btc_price()

        if btc_price <= 0:
            return []

        momentum = self._calculate_momentum()

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Get BTC markets
                r = await client.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={"active": "true", "closed": "false", "limit": 100}
                )
                markets = r.json()

                for m in markets:
                    question = m.get("question", "").upper()

                    # Check if it's a BTC price threshold market
                    if "BTC" not in question and "BITCOIN" not in question:
                        continue

                    # Extract threshold
                    match = re.search(r'\$?([\d,]+)', question)
                    if not match:
                        continue

                    threshold = float(match.group(1).replace(",", ""))

                    # Get current market price
                    try:
                        prices = m.get("outcomePrices", "[]")
                        if isinstance(prices, str):
                            import json
                            prices = json.loads(prices)
                        if not prices:
                            continue
                        yes_price = float(prices[0])
                    except:
                        continue

                    # Calculate implied vs actual probability
                    # Actual: based on current price and momentum
                    distance_pct = (btc_price - threshold) / threshold

                    # If BTC is 5% above threshold and rising, high probability
                    if distance_pct > 0.05 and momentum > 0:
                        actual_prob = 0.90 + min(momentum * 5, 0.09)
                    elif distance_pct > 0.02 and momentum > 0:
                        actual_prob = 0.80 + min(momentum * 5, 0.15)
                    elif distance_pct > 0:
                        actual_prob = 0.60 + distance_pct * 2 + momentum * 3
                    elif distance_pct > -0.02:
                        actual_prob = 0.50 + distance_pct * 10 + momentum * 5
                    else:
                        actual_prob = 0.30 + max(distance_pct * 5, -0.25) + momentum * 5

                    actual_prob = max(0.05, min(0.95, actual_prob))

                    # Edge
                    edge = actual_prob - yes_price

                    if abs(edge) >= self.min_edge:
                        direction = "YES" if edge > 0 else "NO"
                        opportunities.append(ArbitrageOpportunity(
                            arb_type="momentum_lag",
                            market_id=m.get("conditionId", ""),
                            market_title=m.get("question", "")[:60],
                            profit_margin=abs(edge),
                            confidence=min(abs(edge) / 0.30, 0.95),
                            action=f"Buy {direction} (edge: {edge:+.1%})",
                            details={
                                "btc_price": btc_price,
                                "threshold": threshold,
                                "distance_pct": distance_pct,
                                "momentum": momentum,
                                "market_price": yes_price,
                                "estimated_prob": actual_prob,
                                "edge": edge,
                            }
                        ))

        except Exception as e:
            print(f"[MomentumLagDetector] Scan error: {e}")

        # Sort by edge
        opportunities.sort(key=lambda x: x.profit_margin, reverse=True)
        return opportunities


class CombinatorialDetector:
    """
    Detect combinatorial arbitrage from logical inconsistencies.

    Example: P(X wins national) should >= P(X wins key state)
    If child probability > parent probability, arbitrage exists.
    """

    def __init__(self, min_edge: float = 0.03):
        self.min_edge = min_edge

    async def scan(self) -> List[ArbitrageOpportunity]:
        """Scan for combinatorial arbitrage."""
        opportunities = []

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Get events
                r = await client.get(
                    "https://gamma-api.polymarket.com/events",
                    params={"active": "true", "closed": "false", "limit": 100}
                )
                events = r.json()

                # Look for related markets within events
                for event in events:
                    event_opps = self._analyze_event(event)
                    opportunities.extend(event_opps)

        except Exception as e:
            print(f"[CombinatorialDetector] Scan error: {e}")

        return opportunities

    def _analyze_event(self, event: Dict) -> List[ArbitrageOpportunity]:
        """Analyze an event for combinatorial arbitrage."""
        opportunities = []
        markets = event.get("markets", [])

        if len(markets) < 2:
            return []

        # Extract prices
        market_data = []
        for m in markets:
            try:
                prices = m.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    import json
                    prices = json.loads(prices)
                if prices:
                    market_data.append({
                        "id": m.get("conditionId", ""),
                        "question": m.get("question", ""),
                        "yes_price": float(prices[0]),
                    })
            except:
                pass

        # Look for parent-child relationships
        # e.g., "X wins by March" should be <= "X wins by December"
        for i, m1 in enumerate(market_data):
            for m2 in market_data[i+1:]:
                # Check if one implies the other
                edge = self._check_implication(m1, m2)
                if edge and abs(edge) >= self.min_edge:
                    if edge > 0:
                        parent, child = m1, m2
                    else:
                        parent, child = m2, m1
                        edge = -edge

                    opportunities.append(ArbitrageOpportunity(
                        arb_type="combinatorial",
                        market_id=child["id"],
                        market_title=f"{child['question'][:30]} > {parent['question'][:30]}",
                        profit_margin=edge,
                        confidence=0.7,  # Lower confidence - requires interpretation
                        action=f"Sell {child['question'][:20]}, Buy {parent['question'][:20]}",
                        details={
                            "child_price": child["yes_price"],
                            "parent_price": parent["yes_price"],
                            "edge": edge,
                        }
                    ))

        return opportunities

    def _check_implication(self, m1: Dict, m2: Dict) -> Optional[float]:
        """
        Check if m1 implies m2 or vice versa.

        Returns edge if arbitrage exists, None otherwise.
        """
        q1 = m1["question"].lower()
        q2 = m2["question"].lower()

        # Check for date-based implications
        # "by March" implies "by December"
        months_order = ["jan", "feb", "mar", "apr", "may", "jun",
                       "jul", "aug", "sep", "oct", "nov", "dec"]

        m1_month = None
        m2_month = None
        for i, month in enumerate(months_order):
            if month in q1:
                m1_month = i
            if month in q2:
                m2_month = i

        if m1_month is not None and m2_month is not None:
            if m1_month < m2_month:
                # m1 is earlier, so m1 implies m2
                # P(by March) <= P(by December)
                # If P(March) > P(December), arbitrage
                if m1["yes_price"] > m2["yes_price"]:
                    return m1["yes_price"] - m2["yes_price"]
            elif m2_month < m1_month:
                if m2["yes_price"] > m1["yes_price"]:
                    return m2["yes_price"] - m1["yes_price"]

        return None


class ArbScanner:
    """Combined arbitrage scanner."""

    def __init__(self):
        self.rebalancing = RebalancingDetector()
        self.momentum = MomentumLagDetector()
        self.combinatorial = CombinatorialDetector()

    async def scan_all(self) -> List[ArbitrageOpportunity]:
        """Scan for all types of arbitrage."""
        results = []

        # Run scanners concurrently
        rebalancing_task = asyncio.create_task(self.rebalancing.scan())
        momentum_task = asyncio.create_task(self.momentum.scan())
        combinatorial_task = asyncio.create_task(self.combinatorial.scan())

        rebalancing_opps = await rebalancing_task
        momentum_opps = await momentum_task
        combinatorial_opps = await combinatorial_task

        results.extend(rebalancing_opps)
        results.extend(momentum_opps)
        results.extend(combinatorial_opps)

        # Sort by profit margin * confidence
        results.sort(key=lambda x: x.profit_margin * x.confidence, reverse=True)

        return results

    async def scan_and_print(self):
        """Scan and print results."""
        print("\n" + "=" * 70)
        print("ARBITRAGE SCAN")
        print("=" * 70)

        opps = await self.scan_all()

        if not opps:
            print("No arbitrage opportunities found")
            return []

        print(f"Found {len(opps)} opportunities:\n")

        for i, opp in enumerate(opps[:10], 1):
            print(f"{i}. [{opp.arb_type.upper()}] {opp.market_title}")
            print(f"   Profit: {opp.profit_margin:.1%} | Confidence: {opp.confidence:.0%}")
            print(f"   Action: {opp.action}")
            print()

        return opps


# Global scanner
arb_scanner = ArbScanner()


async def scan_arbitrage() -> List[ArbitrageOpportunity]:
    """Convenience function to scan for arbitrage."""
    return await arb_scanner.scan_all()
