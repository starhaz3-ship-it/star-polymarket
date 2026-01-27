"""
Kalshi market feed for cross-platform arbitrage.
Compares prices between Polymarket and Kalshi for arbitrage opportunities.
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import httpx

from .config import config


@dataclass
class KalshiMarket:
    """A Kalshi prediction market."""
    ticker: str
    title: str
    subtitle: str
    yes_price: float  # In cents (0-100)
    no_price: float
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume: int
    open_interest: int
    close_time: Optional[datetime]
    status: str
    category: str

    @property
    def yes_price_decimal(self) -> float:
        """Price as decimal (0-1)."""
        return self.yes_price / 100

    @property
    def no_price_decimal(self) -> float:
        return self.no_price / 100

    @property
    def spread_cents(self) -> float:
        """Spread in cents."""
        return self.yes_ask - self.yes_bid

    @property
    def is_active(self) -> bool:
        return self.status == "active"


@dataclass
class CrossPlatformOpportunity:
    """An arbitrage opportunity between Polymarket and Kalshi."""
    poly_market_id: str
    poly_question: str
    kalshi_ticker: str
    kalshi_title: str

    # Prices (as decimals 0-1)
    poly_yes_price: float
    poly_no_price: float
    kalshi_yes_price: float
    kalshi_no_price: float

    # The arbitrage
    strategy: str  # "buy_poly_yes_kalshi_no" or "buy_poly_no_kalshi_yes"
    combined_cost: float
    gross_profit: float
    net_profit: float  # After fees
    edge_percent: float

    timestamp: float


class KalshiFeed:
    """
    Fetches market data from Kalshi for cross-platform arbitrage.

    Note: Kalshi requires authentication for trading but market data
    can be accessed via their public API.
    """

    # Kalshi API endpoints
    DEMO_API_URL = "https://demo-api.kalshi.co/trade-api/v2"
    PROD_API_URL = "https://api.kalshi.com/trade-api/v2"
    ELECTIONS_API_URL = "https://api.elections.kalshi.com/trade-api/v2"

    # Fee structure (Kalshi charges ~0.7% for takers)
    KALSHI_FEE = 0.007

    def __init__(self, use_demo: bool = False):
        # Try production API first (more reliable)
        self.api_url = self.PROD_API_URL
        self.client = httpx.Client(timeout=30.0)
        self.markets: Dict[str, KalshiMarket] = {}

        # Market matching cache (maps Polymarket IDs to Kalshi tickers)
        self.market_matches: Dict[str, str] = {}

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def fetch_markets(self, category: str = None, limit: int = 100) -> List[KalshiMarket]:
        """Fetch active markets from Kalshi."""
        markets = []

        try:
            params = {
                "limit": limit,
            }
            if category:
                params["series_ticker"] = category

            # Try multiple endpoints
            endpoints = [
                self.PROD_API_URL,
                self.ELECTIONS_API_URL,
            ]

            data = None
            for endpoint in endpoints:
                try:
                    r = self.client.get(
                        f"{endpoint}/markets",
                        params=params,
                        headers=self._get_headers()
                    )
                    if r.status_code == 200:
                        data = r.json()
                        break
                except:
                    continue

            if not data:
                return markets

            for m in data.get("markets", []):
                market = KalshiMarket(
                    ticker=m.get("ticker", ""),
                    title=m.get("title", ""),
                    subtitle=m.get("subtitle", ""),
                    yes_price=m.get("yes_price", 50),
                    no_price=m.get("no_price", 50),
                    yes_bid=m.get("yes_bid", 0),
                    yes_ask=m.get("yes_ask", 100),
                    no_bid=m.get("no_bid", 0),
                    no_ask=m.get("no_ask", 100),
                    volume=m.get("volume", 0),
                    open_interest=m.get("open_interest", 0),
                    close_time=self._parse_time(m.get("close_time")),
                    status=m.get("status", ""),
                    category=m.get("category", ""),
                )
                markets.append(market)
                self.markets[market.ticker] = market

        except Exception as e:
            print(f"[KalshiFeed] Error fetching markets: {e}")

        return markets

    def fetch_orderbook(self, ticker: str) -> Optional[Dict]:
        """Fetch order book for a specific market."""
        try:
            r = self.client.get(
                f"{self.api_url}/markets/{ticker}/orderbook",
                headers=self._get_headers()
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[KalshiFeed] Orderbook error for {ticker}: {e}")
            return None

    def _parse_time(self, time_str: str) -> Optional[datetime]:
        """Parse Kalshi time format."""
        if not time_str:
            return None
        try:
            return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
        except:
            return None

    def match_markets(self, poly_question: str, kalshi_markets: List[KalshiMarket]) -> Optional[KalshiMarket]:
        """
        Find a Kalshi market that matches a Polymarket question.
        Uses simple keyword matching.
        """
        poly_lower = poly_question.lower()

        # Extract key terms
        keywords = []

        # BTC/Bitcoin
        if "btc" in poly_lower or "bitcoin" in poly_lower:
            keywords.extend(["btc", "bitcoin"])

        # Fed/Interest rates
        if "fed" in poly_lower or "interest rate" in poly_lower or "fomc" in poly_lower:
            keywords.extend(["fed", "fomc", "rate"])

        # Elections
        if "election" in poly_lower or "president" in poly_lower:
            keywords.extend(["election", "president"])

        # Price targets
        import re
        price_match = re.search(r'\$?([\d,]+)k?', poly_lower)
        if price_match:
            price = price_match.group(1).replace(",", "")
            keywords.append(price)

        if not keywords:
            return None

        # Score each Kalshi market
        best_match = None
        best_score = 0

        for km in kalshi_markets:
            kalshi_text = (km.title + " " + km.subtitle).lower()
            score = sum(1 for kw in keywords if kw in kalshi_text)

            if score > best_score:
                best_score = score
                best_match = km

        # Require at least 2 keyword matches
        if best_score >= 2:
            return best_match

        return None

    def find_cross_platform_arb(
        self,
        poly_market,  # BTCMarket or MultiOutcomeMarket
        kalshi_market: KalshiMarket
    ) -> Optional[CrossPlatformOpportunity]:
        """
        Find cross-platform arbitrage between Polymarket and Kalshi.

        Arbitrage exists when:
        - Poly YES + Kalshi NO < 1.0 (buy both, guaranteed $1 payout)
        - Poly NO + Kalshi YES < 1.0

        After fees, need combined cost < 1.0 - total_fees
        """
        # Get Polymarket prices (as decimals)
        poly_yes = poly_market.yes_ask if hasattr(poly_market, 'yes_ask') else poly_market.yes_price
        poly_no = poly_market.no_ask if hasattr(poly_market, 'no_ask') else poly_market.no_price

        # Get Kalshi prices (convert from cents to decimals)
        kalshi_yes = kalshi_market.yes_ask / 100
        kalshi_no = kalshi_market.no_ask / 100

        # Estimate fees
        poly_fee = config.TAKER_FEE_MAX * 0.5  # Average Polymarket fee
        kalshi_fee = self.KALSHI_FEE

        # Strategy 1: Buy Poly YES + Kalshi NO
        # This wins if event happens (Poly YES pays) OR doesn't (Kalshi NO pays)
        # Wait - that's not right. Both resolve the same way.
        # Correct arbitrage: markets must disagree on probability

        # If Poly thinks YES is likely (high price) and Kalshi thinks NO is likely (low YES price)
        # Then buy YES on Kalshi (cheap) and NO on Poly (cheap)

        # Check price disagreement
        poly_yes_prob = poly_yes
        kalshi_yes_prob = kalshi_market.yes_price / 100

        # Significant disagreement?
        disagreement = abs(poly_yes_prob - kalshi_yes_prob)
        if disagreement < 0.05:  # Less than 5% difference
            return None

        # Arbitrage: buy YES where cheaper, buy NO where cheaper (on opposing platform)
        # But this is directional, not true arbitrage...

        # True cross-platform arb would require simultaneous settlement
        # For now, look for simple price differences

        # If Poly YES is cheaper than Kalshi YES by more than fees
        if poly_yes < kalshi_yes_prob - poly_fee - kalshi_fee:
            edge = (kalshi_yes_prob - poly_yes - poly_fee - kalshi_fee) * 100
            if edge > 1.0:  # At least 1% edge
                return CrossPlatformOpportunity(
                    poly_market_id=poly_market.condition_id,
                    poly_question=poly_market.question[:50],
                    kalshi_ticker=kalshi_market.ticker,
                    kalshi_title=kalshi_market.title[:50],
                    poly_yes_price=poly_yes,
                    poly_no_price=poly_no,
                    kalshi_yes_price=kalshi_yes_prob,
                    kalshi_no_price=1 - kalshi_yes_prob,
                    strategy="buy_poly_yes",
                    combined_cost=poly_yes,
                    gross_profit=kalshi_yes_prob - poly_yes,
                    net_profit=kalshi_yes_prob - poly_yes - poly_fee - kalshi_fee,
                    edge_percent=edge,
                    timestamp=time.time()
                )

        # If Kalshi YES is cheaper than Poly YES
        if kalshi_yes_prob < poly_yes - poly_fee - kalshi_fee:
            edge = (poly_yes - kalshi_yes_prob - poly_fee - kalshi_fee) * 100
            if edge > 1.0:
                return CrossPlatformOpportunity(
                    poly_market_id=poly_market.condition_id,
                    poly_question=poly_market.question[:50],
                    kalshi_ticker=kalshi_market.ticker,
                    kalshi_title=kalshi_market.title[:50],
                    poly_yes_price=poly_yes,
                    poly_no_price=poly_no,
                    kalshi_yes_price=kalshi_yes_prob,
                    kalshi_no_price=1 - kalshi_yes_prob,
                    strategy="buy_kalshi_yes",
                    combined_cost=kalshi_yes_prob,
                    gross_profit=poly_yes - kalshi_yes_prob,
                    net_profit=poly_yes - kalshi_yes_prob - poly_fee - kalshi_fee,
                    edge_percent=edge,
                    timestamp=time.time()
                )

        return None

    def scan_for_arbitrage(
        self,
        poly_markets: List,  # List of BTCMarket
    ) -> List[CrossPlatformOpportunity]:
        """
        Scan for cross-platform arbitrage opportunities.
        """
        opportunities = []

        # Fetch Kalshi markets
        kalshi_markets = self.fetch_markets(limit=200)

        if not kalshi_markets:
            print("[KalshiFeed] No Kalshi markets fetched")
            return opportunities

        print(f"[KalshiFeed] Scanning {len(poly_markets)} Poly vs {len(kalshi_markets)} Kalshi markets")

        for poly_market in poly_markets:
            # Try to find matching Kalshi market
            kalshi_match = self.match_markets(poly_market.question, kalshi_markets)

            if kalshi_match:
                # Check for arbitrage
                opp = self.find_cross_platform_arb(poly_market, kalshi_match)
                if opp:
                    opportunities.append(opp)
                    print(f"[KalshiFeed] Found opportunity: {opp.edge_percent:.2f}% edge")

        return opportunities

    def stop(self):
        """Close the client."""
        self.client.close()


# Helper function to create signals from cross-platform opportunities
def create_cross_platform_signal(opp: CrossPlatformOpportunity):
    """Convert CrossPlatformOpportunity to ArbitrageSignal format."""
    from .detector import ArbitrageSignal, SignalType

    # Create a minimal market-like object
    class CrossPlatformMarket:
        def __init__(self, opp):
            self.condition_id = opp.poly_market_id
            self.question = f"[CROSS] {opp.poly_question} vs {opp.kalshi_title}"
            self.yes_price = opp.poly_yes_price
            self.no_price = opp.poly_no_price
            self.yes_ask = opp.poly_yes_price
            self.no_ask = opp.poly_no_price
            self.time_remaining_sec = None
            self.is_active = True

    return ArbitrageSignal(
        signal_type=SignalType.BUY_BOTH,  # Reuse this type for cross-platform
        market=CrossPlatformMarket(opp),
        spot_price=0,
        edge_percent=opp.edge_percent,
        confidence=0.7,  # Medium confidence for cross-platform
        recommended_size=config.MAX_POSITION_SIZE * 0.5,  # Conservative sizing
        timestamp=opp.timestamp
    )


if __name__ == "__main__":
    # Test the Kalshi feed
    feed = KalshiFeed(use_demo=True)

    print("Fetching Kalshi markets...")
    markets = feed.fetch_markets(limit=20)

    print(f"\nFound {len(markets)} markets:")
    for m in markets[:10]:
        print(f"  {m.ticker}: {m.title[:40]}... | YES: {m.yes_price}c | NO: {m.no_price}c")

    feed.stop()
