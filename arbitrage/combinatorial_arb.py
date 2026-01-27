"""Combinatorial arbitrage detector for related markets."""
import asyncio
import httpx
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from itertools import combinations
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class CombinatorialOpportunity:
    """Combinatorial arbitrage opportunity."""
    markets: list[dict]
    total_probability: float
    arbitrage_type: str  # 'under_100' or 'over_100'
    profit_potential: float
    category: str
    description: str
    timestamp: datetime


class CombinatorialArbDetector:
    """Detect arbitrage across related markets."""

    # Categories of mutually exclusive markets
    EXCLUSIVE_CATEGORIES = {
        "super_bowl_winner": {
            "keywords": ["win Super Bowl", "Super Bowl 2026"],
            "must_contain": ["win"],
            "exclusive": True,  # Only one can win
        },
        "super_bowl_halftime": {
            "keywords": ["Super Bowl", "halftime"],
            "must_contain": ["perform", "halftime"],
            "exclusive": False,  # Multiple performers possible (but check)
        },
        "world_cup_winner": {
            "keywords": ["win", "FIFA World Cup", "2026"],
            "must_contain": ["win", "World Cup"],
            "exclusive": True,
        },
        "nba_champion": {
            "keywords": ["win", "NBA Finals", "NBA Championship"],
            "must_contain": ["win"],
            "exclusive": True,
        },
        "coach_of_year": {
            "keywords": ["Coach of the Year", "Coach of Year"],
            "must_contain": ["Coach"],
            "exclusive": True,
        },
        "president_2028": {
            "keywords": ["2028", "Presidential Election", "President"],
            "must_contain": ["2028", "President"],
            "exclusive": True,
        },
    }

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30)
        self.markets_cache: list[dict] = []
        self.opportunities: list[CombinatorialOpportunity] = []

    async def fetch_markets(self) -> list[dict]:
        """Fetch all active markets."""
        try:
            resp = await self.client.get(
                "https://gamma-api.polymarket.com/markets",
                params={"active": "true", "closed": "false", "limit": 500}
            )
            self.markets_cache = resp.json()
            return self.markets_cache
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    def categorize_market(self, market: dict) -> Optional[str]:
        """Determine which category a market belongs to."""
        question = market.get("question", "").lower()

        for category, config in self.EXCLUSIVE_CATEGORIES.items():
            # Check keywords
            keyword_match = any(kw.lower() in question for kw in config["keywords"])

            # Check must_contain
            must_match = all(mc.lower() in question for mc in config["must_contain"])

            if keyword_match and must_match:
                return category

        return None

    def extract_entity(self, question: str, category: str) -> Optional[str]:
        """Extract the entity (team, person) from market question."""
        question = question.strip()

        if category == "super_bowl_winner":
            # "Will the Seattle Seahawks win Super Bowl 2026?"
            match = re.search(r"Will (?:the )?([\w\s]+?) win", question, re.I)
            if match:
                return match.group(1).strip()

        elif category == "super_bowl_halftime":
            # "Will Drake perform during the Super Bowl LX halftime show?"
            match = re.search(r"Will ([\w\s]+?) perform", question, re.I)
            if match:
                return match.group(1).strip()

        elif category == "world_cup_winner":
            # "Will Brazil win the 2026 FIFA World Cup?"
            match = re.search(r"Will ([\w\s]+?) win.*World Cup", question, re.I)
            if match:
                return match.group(1).strip()

        elif category == "coach_of_year":
            # "Will Mike Vrabel win NFL Coach of the Year?"
            match = re.search(r"Will ([\w\s]+?) win.*Coach", question, re.I)
            if match:
                return match.group(1).strip()

        elif category == "president_2028":
            # "Will Marco Rubio win the 2028 US Presidential Election?"
            match = re.search(r"Will ([\w\s]+?) win.*President", question, re.I)
            if match:
                return match.group(1).strip()

        return None

    def get_yes_price(self, market: dict) -> float:
        """Extract YES price from market."""
        try:
            prices = market.get("outcomePrices", "[]")
            if isinstance(prices, str):
                prices = json.loads(prices)
            return float(prices[0]) if prices else 0
        except:
            return 0

    async def find_exclusive_groups(self) -> dict[str, list[dict]]:
        """Group markets by exclusive category."""
        if not self.markets_cache:
            await self.fetch_markets()

        groups = {cat: [] for cat in self.EXCLUSIVE_CATEGORIES}

        for market in self.markets_cache:
            category = self.categorize_market(market)
            if category:
                entity = self.extract_entity(market.get("question", ""), category)
                yes_price = self.get_yes_price(market)

                if yes_price > 0.01:  # Skip near-zero markets
                    groups[category].append({
                        "market": market,
                        "entity": entity,
                        "yes_price": yes_price,
                        "category": category,
                    })

        return groups

    async def scan_for_arbitrage(self) -> list[CombinatorialOpportunity]:
        """Scan for combinatorial arbitrage opportunities."""
        self.opportunities = []
        groups = await self.find_exclusive_groups()

        for category, markets in groups.items():
            if len(markets) < 2:
                continue

            config = self.EXCLUSIVE_CATEGORIES[category]

            if config["exclusive"]:
                # Sum of all YES prices should equal ~100%
                total_prob = sum(m["yes_price"] for m in markets)

                entities = [m["entity"] for m in markets]

                if total_prob < 0.95:
                    # Under 100% - can buy all YES and guarantee profit
                    profit = 1 - total_prob
                    self.opportunities.append(CombinatorialOpportunity(
                        markets=[m["market"] for m in markets],
                        total_probability=total_prob,
                        arbitrage_type="under_100",
                        profit_potential=profit,
                        category=category,
                        description=f"Buy all {len(markets)} YES positions for {category}. "
                                  f"Total cost: ${total_prob:.2f}, Guaranteed payout: $1.00. "
                                  f"Entities: {', '.join(str(e) for e in entities[:5])}",
                        timestamp=datetime.now(timezone.utc)
                    ))

                elif total_prob > 1.05:
                    # Over 100% - can buy all NO and guarantee profit
                    # NO cost = 1 - YES price for each
                    total_no_cost = sum(1 - m["yes_price"] for m in markets)
                    # Payout is (n-1) * $1 since one NO loses
                    payout = len(markets) - 1
                    if total_no_cost < payout:
                        profit = payout - total_no_cost
                        self.opportunities.append(CombinatorialOpportunity(
                            markets=[m["market"] for m in markets],
                            total_probability=total_prob,
                            arbitrage_type="over_100",
                            profit_potential=profit,
                            category=category,
                            description=f"Buy all {len(markets)} NO positions for {category}. "
                                      f"Total cost: ${total_no_cost:.2f}, Payout: ${payout:.2f}",
                            timestamp=datetime.now(timezone.utc)
                        ))

        # Sort by profit potential
        self.opportunities.sort(key=lambda x: -x.profit_potential)
        return self.opportunities

    async def get_category_summary(self) -> dict:
        """Get summary of each category's probability distribution."""
        groups = await self.find_exclusive_groups()
        summary = {}

        for category, markets in groups.items():
            if not markets:
                continue

            total_prob = sum(m["yes_price"] for m in markets)
            sorted_markets = sorted(markets, key=lambda x: -x["yes_price"])

            summary[category] = {
                "count": len(markets),
                "total_probability": total_prob,
                "deviation_from_100": abs(1 - total_prob),
                "top_3": [
                    {"entity": m["entity"], "prob": m["yes_price"]}
                    for m in sorted_markets[:3]
                ],
            }

        return summary


async def main():
    """Test combinatorial arbitrage detector."""
    detector = CombinatorialArbDetector()

    print("=" * 70)
    print("COMBINATORIAL ARBITRAGE SCANNER")
    print("=" * 70)

    # Get category summary
    summary = await detector.get_category_summary()

    print("\nCategory Probability Summary:\n")
    for category, data in summary.items():
        print(f"{category}:")
        print(f"  Markets: {data['count']}")
        print(f"  Total Probability: {data['total_probability']*100:.1f}%")
        print(f"  Deviation: {data['deviation_from_100']*100:.1f}%")
        if data['top_3']:
            top_str = ', '.join(f"{t['entity']}: {t['prob']*100:.0f}%" for t in data['top_3'])
            print(f"  Top 3: {top_str}")
        print()

    # Scan for arbitrage
    print("\nScanning for arbitrage opportunities...\n")
    opps = await detector.scan_for_arbitrage()

    if opps:
        print(f"Found {len(opps)} opportunities:\n")
        for opp in opps:
            print(f"Category: {opp.category}")
            print(f"  Type: {opp.arbitrage_type}")
            print(f"  Total Prob: {opp.total_probability*100:.1f}%")
            print(f"  Profit: ${opp.profit_potential:.2f} per $1 bet")
            print(f"  {opp.description}")
            print()
    else:
        print("No arbitrage opportunities found (markets are well-priced)")


if __name__ == "__main__":
    asyncio.run(main())
