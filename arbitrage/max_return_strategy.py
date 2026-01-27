"""
Maximum Return Strategy Module

Combines the highest-return strategies from research:
1. Copy Trading (kingofcoinflips) - Follow proven winners
2. Endgame Strategy - 1,800%+ annualized on 95%+ prob near resolution
3. High-Probability Bonds - Buy 85%+ positions for steady returns
4. NegRisk Arbitrage - Multi-outcome sum < 100%
5. Volatility Plays - Bet on price staying in range

Based on research:
- $40M+ extracted via arbitrage (April 2024 - April 2025)
- Top bot: $313 -> $414,000 in one month (98% win rate)
- NegRisk: $29M profit historically
"""

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import httpx

from .config import config


@dataclass
class StrategySignal:
    """A trading signal from any strategy."""
    strategy: str
    market_slug: str
    condition_id: str
    token_id: str
    side: str  # YES or NO
    probability: float
    recommended_size: float
    expected_return_pct: float
    confidence: float
    rationale: str
    expires: Optional[str] = None


class MaxReturnStrategy:
    """
    Combines multiple high-return strategies.
    """

    # kingofcoinflips wallet for copy trading
    COPY_TARGET = "0xe9c6312464b52aa3eff13d822b003282075995c9"

    def __init__(self):
        self.signals: List[StrategySignal] = []
        self.executed_trades: Dict[str, dict] = {}

    async def scan_all_strategies(self) -> List[StrategySignal]:
        """Run all strategies and collect signals."""
        self.signals = []

        async with httpx.AsyncClient(timeout=30) as client:
            # Run strategies in parallel
            results = await asyncio.gather(
                self._scan_copy_trading(client),
                self._scan_endgame(client),
                self._scan_high_prob_bonds(client),
                self._scan_negrisk(client),
                return_exceptions=True
            )

            for result in results:
                if isinstance(result, list):
                    self.signals.extend(result)
                elif isinstance(result, Exception):
                    print(f"[Strategy] Error: {result}")

        # Sort by expected return
        self.signals.sort(key=lambda x: x.expected_return_pct, reverse=True)
        return self.signals

    async def _scan_copy_trading(self, client: httpx.AsyncClient) -> List[StrategySignal]:
        """
        Strategy 1: Copy kingofcoinflips trades.
        They have $692K profit with 2,341 trades.
        """
        signals = []

        try:
            r = await client.get(
                "https://data-api.polymarket.com/positions",
                params={"user": self.COPY_TARGET, "sizeThreshold": 0}
            )
            positions = r.json()

            # Find their newest/best positions
            for pos in positions:
                value = float(pos.get("currentValue", 0))
                pnl = float(pos.get("cashPnl", 0))
                pnl_pct = float(pos.get("percentPnl", 0))

                # Copy positions that are:
                # 1. Currently profitable (they're right)
                # 2. Large enough to matter ($500+)
                # 3. Positive momentum
                if value > 500 and pnl > 0 and pnl_pct > 5:
                    title = pos.get("title", "")[:50]
                    outcome = pos.get("outcome", "")
                    cur_price = float(pos.get("curPrice", 0))
                    condition_id = pos.get("conditionId", "")
                    token_id = pos.get("asset", "")

                    if cur_price > 0:
                        # Calculate expected return if position wins
                        if cur_price < 1.0:
                            expected_return = ((1.0 / cur_price) - 1) * 100
                        else:
                            expected_return = 0

                        signals.append(StrategySignal(
                            strategy="COPY_TRADE",
                            market_slug=pos.get("slug", title),
                            condition_id=condition_id,
                            token_id=token_id,
                            side=outcome,
                            probability=cur_price,
                            recommended_size=min(value * 0.01, config.MAX_POSITION_SIZE),
                            expected_return_pct=expected_return,
                            confidence=0.8,  # High confidence in proven trader
                            rationale=f"Copy kingofcoinflips: +${pnl:.0f} ({pnl_pct:.0f}%)",
                            expires=pos.get("endDate"),
                        ))

        except Exception as e:
            print(f"[CopyTrading] Error: {e}")

        return signals

    async def _scan_endgame(self, client: httpx.AsyncClient) -> List[StrategySignal]:
        """
        Strategy 2: Endgame - Buy 90%+ probability near resolution.
        Historical: 1,800%+ annualized returns.
        """
        signals = []

        try:
            r = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={"closed": "false", "limit": 200}
            )
            markets = r.json()

            now = datetime.utcnow()

            for m in markets:
                try:
                    prices = m.get("outcomePrices", [])
                    end_date_str = m.get("endDate", "")
                    outcomes = m.get("outcomes", [])

                    if not prices or not end_date_str or len(prices) < 2:
                        continue

                    # Parse end date
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                        end_date = end_date.replace(tzinfo=None)
                    except:
                        continue

                    # Calculate time to expiry
                    time_to_expiry = end_date - now
                    hours_remaining = time_to_expiry.total_seconds() / 3600

                    # Endgame criteria: 90%+ probability, < 72 hours to resolution
                    price_list = [float(p) for p in prices]
                    max_prob = max(price_list)
                    max_idx = price_list.index(max_prob)

                    if max_prob >= 0.90 and 0 < hours_remaining < 72:
                        # Calculate annualized return
                        if max_prob < 1.0:
                            raw_return = (1.0 / max_prob) - 1
                            # Annualize based on time remaining
                            if hours_remaining > 0:
                                annualized = raw_return * (365 * 24 / hours_remaining) * 100
                            else:
                                annualized = raw_return * 100
                        else:
                            continue

                        # Cap annualized at reasonable level for display
                        annualized = min(annualized, 10000)

                        outcome_name = outcomes[max_idx] if max_idx < len(outcomes) else "YES"
                        tokens = m.get("clobTokenIds", [])
                        token_id = tokens[max_idx] if max_idx < len(tokens) else ""

                        signals.append(StrategySignal(
                            strategy="ENDGAME",
                            market_slug=m.get("slug", m.get("question", "")[:30]),
                            condition_id=m.get("conditionId", ""),
                            token_id=token_id,
                            side=outcome_name,
                            probability=max_prob,
                            recommended_size=config.MAX_POSITION_SIZE * 0.5,
                            expected_return_pct=annualized,
                            confidence=max_prob,  # Confidence = probability
                            rationale=f"{max_prob:.0%} prob, {hours_remaining:.0f}h to resolution",
                            expires=end_date_str,
                        ))

                except Exception:
                    continue

        except Exception as e:
            print(f"[Endgame] Error: {e}")

        return signals

    async def _scan_high_prob_bonds(self, client: httpx.AsyncClient) -> List[StrategySignal]:
        """
        Strategy 3: High-probability bonds.
        Buy positions with 80-95% probability for steady returns.
        Like buying bonds that pay out when resolved.
        """
        signals = []

        try:
            r = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={"closed": "false", "limit": 200}
            )
            markets = r.json()

            for m in markets:
                try:
                    prices = m.get("outcomePrices", [])
                    outcomes = m.get("outcomes", [])
                    volume = float(m.get("volume", 0))

                    if not prices or len(prices) < 2:
                        continue

                    # Need sufficient volume for liquidity
                    if volume < 10000:
                        continue

                    price_list = [float(p) for p in prices]
                    max_prob = max(price_list)
                    max_idx = price_list.index(max_prob)

                    # Bond criteria: 80-95% probability (not too high to be worthless)
                    if 0.80 <= max_prob <= 0.95:
                        raw_return = ((1.0 / max_prob) - 1) * 100

                        outcome_name = outcomes[max_idx] if max_idx < len(outcomes) else "YES"
                        tokens = m.get("clobTokenIds", [])
                        token_id = tokens[max_idx] if max_idx < len(tokens) else ""

                        signals.append(StrategySignal(
                            strategy="HIGH_PROB_BOND",
                            market_slug=m.get("slug", m.get("question", "")[:30]),
                            condition_id=m.get("conditionId", ""),
                            token_id=token_id,
                            side=outcome_name,
                            probability=max_prob,
                            recommended_size=config.MAX_POSITION_SIZE * 0.3,
                            expected_return_pct=raw_return,
                            confidence=max_prob * 0.9,  # Slight discount
                            rationale=f"Bond: {max_prob:.0%} -> {raw_return:.1f}% return if correct",
                            expires=m.get("endDate"),
                        ))

                except Exception:
                    continue

        except Exception as e:
            print(f"[HighProbBond] Error: {e}")

        return signals

    async def _scan_negrisk(self, client: httpx.AsyncClient) -> List[StrategySignal]:
        """
        Strategy 4: NegRisk multi-outcome arbitrage.
        Buy all outcomes when sum of best asks < 100%.
        Historical: $29M profit.
        """
        signals = []

        try:
            # Fetch events with multiple outcomes
            r = await client.get(
                "https://gamma-api.polymarket.com/events",
                params={"closed": "false", "limit": 100}
            )
            events = r.json()

            for event in events:
                markets = event.get("markets", [])
                if len(markets) < 3:  # Need 3+ outcomes for NegRisk
                    continue

                try:
                    # Calculate sum of YES prices across all outcomes
                    total_yes = 0
                    valid_markets = []

                    for market in markets:
                        prices = market.get("outcomePrices", [])
                        if prices:
                            yes_price = float(prices[0])
                            if yes_price > 0:
                                total_yes += yes_price
                                valid_markets.append({
                                    "market": market,
                                    "yes_price": yes_price,
                                })

                    # NegRisk opportunity if total < 100%
                    if len(valid_markets) >= 3 and total_yes < 0.98:
                        gap = 1.0 - total_yes
                        gap_pct = gap * 100

                        # Only signal if gap covers fees (>2%)
                        if gap_pct > 2.0:
                            event_title = event.get("title", "")[:40]

                            signals.append(StrategySignal(
                                strategy="NEGRISK_ARB",
                                market_slug=event.get("slug", event_title),
                                condition_id=event.get("id", ""),
                                token_id="",  # Multiple tokens
                                side="ALL_YES",
                                probability=total_yes,
                                recommended_size=config.MAX_POSITION_SIZE,
                                expected_return_pct=gap_pct,
                                confidence=0.95,  # High confidence - it's arbitrage
                                rationale=f"Sum={total_yes:.1%}, Gap={gap_pct:.1f}% across {len(valid_markets)} outcomes",
                                expires=None,
                            ))

                except Exception:
                    continue

        except Exception as e:
            print(f"[NegRisk] Error: {e}")

        return signals

    def get_top_signals(self, n: int = 10) -> List[StrategySignal]:
        """Get top N signals by expected return."""
        return self.signals[:n]

    def print_signals(self, limit: int = 15):
        """Print all signals in a formatted table."""
        print("\n" + "=" * 80)
        print("TOP SIGNALS BY EXPECTED RETURN")
        print("=" * 80)

        if not self.signals:
            print("No signals found.")
            return

        for i, s in enumerate(self.signals[:limit], 1):
            print(f"\n{i}. [{s.strategy}] {s.side} @ {s.probability:.0%}")
            print(f"   Market: {s.market_slug}")
            print(f"   Expected Return: {s.expected_return_pct:.1f}%")
            print(f"   Confidence: {s.confidence:.0%}")
            print(f"   Size: ${s.recommended_size:.2f}")
            print(f"   Rationale: {s.rationale}")

        print("\n" + "=" * 80)

        # Summary by strategy
        print("\nSIGNALS BY STRATEGY:")
        strategy_counts = {}
        for s in self.signals:
            strategy_counts[s.strategy] = strategy_counts.get(s.strategy, 0) + 1

        for strat, count in sorted(strategy_counts.items()):
            avg_return = sum(s.expected_return_pct for s in self.signals if s.strategy == strat) / count
            print(f"  {strat}: {count} signals, avg {avg_return:.1f}% expected return")


async def run_max_return_scan():
    """Run a full scan of all strategies."""
    strategy = MaxReturnStrategy()
    print("\n[MaxReturn] Scanning all strategies...")

    signals = await strategy.scan_all_strategies()
    strategy.print_signals()

    return signals


if __name__ == "__main__":
    asyncio.run(run_max_return_scan())
