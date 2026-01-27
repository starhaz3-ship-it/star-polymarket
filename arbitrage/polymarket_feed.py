"""
Polymarket BTC market monitor.
Supports both 15-minute and daily price markets.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import httpx

from .config import config


@dataclass
class BTCMarket:
    """A BTC prediction market."""
    condition_id: str
    question: str
    market_type: str  # "15min", "daily", "threshold"
    strike_price: Optional[float]  # For threshold markets
    end_time: Optional[datetime]
    yes_token_id: str
    no_token_id: str
    yes_price: float = 0.5
    no_price: float = 0.5
    yes_bid: float = 0.0
    yes_ask: float = 1.0
    no_bid: float = 0.0
    no_ask: float = 1.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    outcomes: List[str] = field(default_factory=lambda: ["Yes", "No"])

    @property
    def spread(self) -> float:
        """Combined spread (should be ~1.0 in efficient market)."""
        return self.yes_price + self.no_price

    @property
    def arbitrage_gap(self) -> float:
        """Gap available for same-market arbitrage."""
        return 1.0 - (self.yes_ask + self.no_ask)

    @property
    def time_remaining_sec(self) -> Optional[float]:
        """Seconds until market closes."""
        if self.end_time:
            now = datetime.now(timezone.utc)
            end = self.end_time if self.end_time.tzinfo else self.end_time.replace(tzinfo=timezone.utc)
            return (end - now).total_seconds()
        return None

    @property
    def is_active(self) -> bool:
        """Check if market is still active."""
        if self.time_remaining_sec is not None:
            return self.time_remaining_sec > 0
        return True


@dataclass
class MarketUpdate:
    """Update to market prices."""
    market: BTCMarket
    timestamp: float


@dataclass
class MultiOutcomeMarket:
    """A multi-outcome prediction market for NegRisk arbitrage."""
    condition_id: str
    event_slug: str
    question: str
    outcomes: List[dict]  # [{name, token_id, price, bid, ask}]
    end_time: Optional[datetime]
    volume_24h: float = 0.0
    liquidity: float = 0.0

    @property
    def outcome_count(self) -> int:
        return len(self.outcomes)

    @property
    def total_probability(self) -> float:
        """Sum of all outcome prices (should be ~1.0 in efficient market)."""
        return sum(o.get("price", 0) for o in self.outcomes)

    @property
    def total_ask(self) -> float:
        """Sum of all ask prices (cost to buy all outcomes)."""
        return sum(o.get("ask", 1.0) for o in self.outcomes)

    @property
    def negrisk_gap(self) -> float:
        """Gap available for NegRisk arbitrage (1.0 - total_ask)."""
        return 1.0 - self.total_ask

    @property
    def time_remaining_sec(self) -> Optional[float]:
        if self.end_time:
            now = datetime.now(timezone.utc)
            end = self.end_time if self.end_time.tzinfo else self.end_time.replace(tzinfo=timezone.utc)
            return (end - now).total_seconds()
        return None

    @property
    def is_active(self) -> bool:
        if self.time_remaining_sec is not None:
            return self.time_remaining_sec > 0
        return True

    @property
    def highest_prob_outcome(self) -> Optional[dict]:
        """Get the outcome with highest probability (for endgame strategy)."""
        if not self.outcomes:
            return None
        return max(self.outcomes, key=lambda o: o.get("price", 0))


class PolymarketFeed:
    """Monitor Polymarket BTC markets."""

    def __init__(self, on_update: Optional[Callable[[MarketUpdate], None]] = None):
        self.on_update = on_update
        self.markets: Dict[str, BTCMarket] = {}
        self._running = False
        self.client = httpx.Client(timeout=30.0)

    def _parse_strike_price(self, question: str) -> Optional[float]:
        """Extract strike price from market question."""
        patterns = [
            r"\$?([\d,]+(?:\.\d+)?)\s*(?:on|by|before|in)",
            r"(?:above|below|reach|hit)\s*\$?([\d,]+(?:\.\d+)?)",
            r"\$?([\d,]+)k",  # e.g., $100k
        ]
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                price_str = match.group(1).replace(",", "")
                price = float(price_str)
                # Handle "k" suffix
                if "k" in question.lower() and price < 1000:
                    price *= 1000
                return price
        return None

    def _parse_end_time(self, end_date_iso: str) -> Optional[datetime]:
        """Parse end time from ISO string."""
        try:
            if end_date_iso:
                # Handle various ISO formats
                end_date_iso = end_date_iso.replace("Z", "+00:00")
                if "+" not in end_date_iso and "-" not in end_date_iso[-6:]:
                    end_date_iso += "+00:00"
                return datetime.fromisoformat(end_date_iso)
        except Exception:
            pass
        return None

    def _classify_market(self, question: str, end_time: Optional[datetime]) -> str:
        """Classify market type."""
        q_lower = question.lower()

        # Check for 15-minute markets
        if "15" in q_lower and ("minute" in q_lower or "min" in q_lower):
            return "15min"
        if "up" in q_lower and "down" in q_lower:
            return "15min"

        # Check time remaining
        if end_time:
            now = datetime.now(timezone.utc)
            end = end_time if end_time.tzinfo else end_time.replace(tzinfo=timezone.utc)
            remaining = (end - now).total_seconds()
            if remaining < 3600:  # Less than 1 hour
                return "15min"
            if remaining < 86400:  # Less than 1 day
                return "daily"

        # Default to threshold market
        return "threshold"

    def fetch_btc_markets(self) -> List[BTCMarket]:
        """Fetch all active BTC markets."""
        markets = []
        btc_keywords = ["btc", "bitcoin"]

        try:
            # Fetch from Gamma API
            params = {
                "active": "true",
                "closed": "false",
                "limit": 100,
                "order": "volume24hr",
                "ascending": "false",
            }

            r = self.client.get(f"{config.GAMMA_API_URL}/markets", params=params)
            r.raise_for_status()
            data = r.json()

            for m in data:
                question = m.get("question", "")

                # Filter for BTC markets
                if not any(kw in question.lower() for kw in btc_keywords):
                    continue

                # Get token IDs
                token_ids = m.get("clobTokenIds", [])
                if isinstance(token_ids, str):
                    try:
                        token_ids = json.loads(token_ids)
                    except:
                        continue
                if len(token_ids) < 2:
                    continue

                # Parse prices
                prices = m.get("outcomePrices", [])
                if isinstance(prices, str):
                    try:
                        prices = json.loads(prices)
                    except:
                        prices = [0.5, 0.5]

                outcomes = m.get("outcomes", ["Yes", "No"])
                if isinstance(outcomes, str):
                    try:
                        outcomes = json.loads(outcomes)
                    except:
                        outcomes = ["Yes", "No"]

                end_time = self._parse_end_time(m.get("endDate", ""))
                strike = self._parse_strike_price(question)
                market_type = self._classify_market(question, end_time)

                market = BTCMarket(
                    condition_id=m.get("conditionId", ""),
                    question=question,
                    market_type=market_type,
                    strike_price=strike,
                    end_time=end_time,
                    yes_token_id=token_ids[0] if len(token_ids) > 0 else "",
                    no_token_id=token_ids[1] if len(token_ids) > 1 else "",
                    yes_price=float(prices[0]) if len(prices) > 0 else 0.5,
                    no_price=float(prices[1]) if len(prices) > 1 else 0.5,
                    volume_24h=float(m.get("volume24hr", 0) or 0),
                    liquidity=float(m.get("liquidity", 0) or 0),
                    outcomes=outcomes,
                )

                markets.append(market)
                self.markets[market.condition_id] = market

        except Exception as e:
            print(f"[PolyFeed] Error fetching markets: {e}")

        return markets

    def fetch_order_book(self, token_id: str, silent: bool = False) -> Dict:
        """Fetch order book for a specific token."""
        try:
            r = self.client.get(f"{config.CLOB_API_URL}/book", params={"token_id": token_id})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if not silent and "404" not in str(e):
                print(f"[PolyFeed] Order book error: {e}")
            return {}

    def update_market_prices(self, market: BTCMarket) -> BTCMarket:
        """Update a market's prices from the order book."""
        try:
            # Fetch YES order book
            yes_book = self.fetch_order_book(market.yes_token_id)
            if yes_book:
                bids = yes_book.get("bids", [])
                asks = yes_book.get("asks", [])
                if bids:
                    market.yes_bid = float(bids[0].get("price", 0))
                if asks:
                    market.yes_ask = float(asks[0].get("price", 1))
                if bids and asks:
                    market.yes_price = (market.yes_bid + market.yes_ask) / 2

            # Fetch NO order book
            no_book = self.fetch_order_book(market.no_token_id)
            if no_book:
                bids = no_book.get("bids", [])
                asks = no_book.get("asks", [])
                if bids:
                    market.no_bid = float(bids[0].get("price", 0))
                if asks:
                    market.no_ask = float(asks[0].get("price", 1))
                if bids and asks:
                    market.no_price = (market.no_bid + market.no_ask) / 2

        except Exception as e:
            print(f"[PolyFeed] Price update error: {e}")

        return market

    async def start(self, poll_interval_sec: float = 2.0):
        """Start polling for market updates."""
        self._running = True
        print("[PolyFeed] Starting Polymarket BTC monitor...")

        while self._running:
            try:
                markets = self.fetch_btc_markets()

                for market in markets:
                    market = self.update_market_prices(market)

                    if self.on_update:
                        update = MarketUpdate(
                            market=market,
                            timestamp=time.time()
                        )
                        self.on_update(update)

                await asyncio.sleep(poll_interval_sec)

            except Exception as e:
                print(f"[PolyFeed] Error: {e}")
                await asyncio.sleep(5)

    def stop(self):
        """Stop the feed."""
        self._running = False
        self.client.close()

    def get_active_markets(self) -> List[BTCMarket]:
        """Get all currently active markets."""
        return [m for m in self.markets.values() if m.is_active]

    def get_markets_by_type(self, market_type: str) -> List[BTCMarket]:
        """Get markets of a specific type."""
        return [m for m in self.markets.values() if m.market_type == market_type and m.is_active]

    def fetch_multi_outcome_markets(self, min_outcomes: int = 3, max_outcomes: int = 20) -> List[MultiOutcomeMarket]:
        """Fetch multi-outcome markets for NegRisk arbitrage."""
        multi_markets = []

        try:
            # Fetch events (which contain multiple markets/outcomes)
            params = {
                "active": "true",
                "closed": "false",
                "limit": 100,
                "order": "volume24hr",
                "ascending": "false",
            }

            r = self.client.get(f"{config.GAMMA_API_URL}/events", params=params)
            r.raise_for_status()
            events = r.json()

            for event in events:
                markets = event.get("markets", [])

                # Skip if not enough outcomes
                if len(markets) < min_outcomes or len(markets) > max_outcomes:
                    continue

                outcomes = []
                total_volume = 0
                total_liquidity = 0
                end_time = None

                for m in markets:
                    # Get token IDs
                    token_ids = m.get("clobTokenIds", [])
                    if isinstance(token_ids, str):
                        try:
                            token_ids = json.loads(token_ids)
                        except:
                            continue

                    # Get prices
                    prices = m.get("outcomePrices", [])
                    if isinstance(prices, str):
                        try:
                            prices = json.loads(prices)
                        except:
                            prices = [0.5, 0.5]

                    # For multi-outcome, we care about the YES price of each market
                    yes_price = float(prices[0]) if len(prices) > 0 else 0.5
                    yes_token = token_ids[0] if len(token_ids) > 0 else ""

                    outcome_name = m.get("groupItemTitle", m.get("question", "")[:30])

                    outcomes.append({
                        "name": outcome_name,
                        "token_id": yes_token,
                        "condition_id": m.get("conditionId", ""),
                        "price": yes_price,
                        "bid": 0.0,
                        "ask": yes_price,  # Will update with order book
                    })

                    total_volume += float(m.get("volume24hr", 0) or 0)
                    total_liquidity += float(m.get("liquidity", 0) or 0)

                    # Get end time from first market
                    if not end_time:
                        end_time = self._parse_end_time(m.get("endDate", ""))

                if outcomes:
                    multi_market = MultiOutcomeMarket(
                        condition_id=event.get("id", ""),
                        event_slug=event.get("slug", ""),
                        question=event.get("title", ""),
                        outcomes=outcomes,
                        end_time=end_time,
                        volume_24h=total_volume,
                        liquidity=total_liquidity,
                    )
                    multi_markets.append(multi_market)

        except Exception as e:
            print(f"[PolyFeed] Error fetching multi-outcome markets: {e}")

        return multi_markets

    def update_multi_market_prices(self, market: MultiOutcomeMarket) -> MultiOutcomeMarket:
        """Update prices for all outcomes in a multi-outcome market."""
        try:
            for outcome in market.outcomes:
                token_id = outcome.get("token_id", "")
                if not token_id:
                    continue

                book = self.fetch_order_book(token_id, silent=True)
                if book:
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])
                    if bids:
                        outcome["bid"] = float(bids[0].get("price", 0))
                    if asks:
                        outcome["ask"] = float(asks[0].get("price", 1))
                    if bids and asks:
                        outcome["price"] = (outcome["bid"] + outcome["ask"]) / 2

        except Exception as e:
            print(f"[PolyFeed] Multi-market price update error: {e}")

        return market


if __name__ == "__main__":
    # Test the feed
    def on_update(update: MarketUpdate):
        m = update.market
        print(f"{m.question[:55]}...")
        print(f"  Type: {m.market_type} | Strike: ${m.strike_price:,.0f if m.strike_price else 0}")
        print(f"  YES: {m.yes_price:.3f} | NO: {m.no_price:.3f} | Gap: {m.arbitrage_gap:.4f}")
        print()

    feed = PolymarketFeed(on_update=on_update)
    markets = feed.fetch_btc_markets()
    print(f"Found {len(markets)} BTC markets")
    print()

    for m in markets[:5]:
        feed.update_market_prices(m)
        on_update(MarketUpdate(market=m, timestamp=time.time()))
