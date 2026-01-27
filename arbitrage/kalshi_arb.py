"""Cross-platform arbitrage: Polymarket vs Kalshi."""
import asyncio
import httpx
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ArbOpportunity:
    """Cross-platform arbitrage opportunity."""
    market_name: str
    polymarket_yes: float
    polymarket_no: float
    kalshi_yes: float
    kalshi_no: float
    spread: float
    strategy: str  # 'buy_poly_yes_kalshi_no' or 'buy_poly_no_kalshi_yes'
    expected_profit_pct: float
    poly_token_id: str
    kalshi_ticker: str
    timestamp: datetime


class KalshiClient:
    """Client for Kalshi API."""

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30)

    async def get_markets(self, status: str = "open", limit: int = 200) -> list:
        """Get active markets from Kalshi."""
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            resp = await self.client.get(
                f"{self.BASE_URL}/markets",
                params={"status": status, "limit": limit},
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("markets", [])
        except Exception as e:
            logger.warning(f"Kalshi API error: {e}")
            return []

    async def get_market(self, ticker: str) -> Optional[dict]:
        """Get specific market by ticker."""
        try:
            resp = await self.client.get(f"{self.BASE_URL}/markets/{ticker}")
            resp.raise_for_status()
            return resp.json().get("market")
        except Exception as e:
            logger.warning(f"Kalshi market lookup failed: {e}")
            return None

    async def get_orderbook(self, ticker: str) -> Optional[dict]:
        """Get orderbook for a market."""
        try:
            resp = await self.client.get(f"{self.BASE_URL}/markets/{ticker}/orderbook")
            resp.raise_for_status()
            return resp.json().get("orderbook")
        except Exception as e:
            logger.warning(f"Kalshi orderbook error: {e}")
            return None


class CrossPlatformArbitrage:
    """Detect arbitrage between Polymarket and Kalshi."""

    # Mapping of similar markets between platforms
    MARKET_MAPPINGS = {
        # BTC price markets
        "btc_above": {
            "poly_keywords": ["BTC", "Bitcoin", "above", "below"],
            "kalshi_prefix": "KXBTC",
        },
        # Election markets
        "president": {
            "poly_keywords": ["President", "Election", "2028"],
            "kalshi_prefix": "PRES",
        },
        # Fed rate markets
        "fed_rate": {
            "poly_keywords": ["Fed", "rate", "FOMC"],
            "kalshi_prefix": "FED",
        },
    }

    def __init__(self, kalshi_api_key: Optional[str] = None):
        self.kalshi = KalshiClient(kalshi_api_key)
        self.poly_client = httpx.AsyncClient(timeout=30)
        self.opportunities: list[ArbOpportunity] = []

    async def get_polymarket_markets(self) -> list:
        """Fetch active Polymarket markets."""
        try:
            resp = await self.poly_client.get(
                "https://gamma-api.polymarket.com/markets",
                params={"active": "true", "closed": "false", "limit": 500}
            )
            return resp.json()
        except Exception as e:
            logger.error(f"Polymarket API error: {e}")
            return []

    async def find_matching_markets(self) -> list[tuple]:
        """Find markets that exist on both platforms."""
        matches = []

        poly_markets = await self.get_polymarket_markets()
        kalshi_markets = await self.kalshi.get_markets()

        # Index Kalshi markets by keywords
        kalshi_index = {}
        for km in kalshi_markets:
            title = km.get("title", "").lower()
            ticker = km.get("ticker", "")
            kalshi_index[ticker] = {
                "market": km,
                "title": title,
                "yes_price": km.get("yes_bid", 0) / 100 if km.get("yes_bid") else 0,
                "no_price": km.get("no_bid", 0) / 100 if km.get("no_bid") else 0,
            }

        # Match Polymarket to Kalshi
        for pm in poly_markets:
            question = pm.get("question", "").lower()

            # BTC price matching
            if "btc" in question or "bitcoin" in question:
                for ticker, kdata in kalshi_index.items():
                    if ticker.startswith("KXBTC"):
                        # Check if same price level
                        if self._extract_price_level(question) == self._extract_price_level(kdata["title"]):
                            matches.append((pm, kdata["market"]))

        return matches

    def _extract_price_level(self, text: str) -> Optional[int]:
        """Extract price level from market title."""
        import re
        # Match patterns like "$90,000" or "90000" or "90k"
        patterns = [
            r'\$?([\d,]+)k',
            r'\$?([\d,]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text.replace(",", ""))
            if match:
                val = match.group(1).replace(",", "")
                if "k" in text.lower():
                    return int(float(val) * 1000)
                return int(val)
        return None

    async def scan_for_arbitrage(self, min_spread: float = 0.02) -> list[ArbOpportunity]:
        """Scan for cross-platform arbitrage opportunities."""
        self.opportunities = []

        matches = await self.find_matching_markets()
        logger.info(f"Found {len(matches)} matching markets")

        for poly_market, kalshi_market in matches:
            try:
                # Get Polymarket prices
                import json
                prices = poly_market.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    prices = json.loads(prices)

                poly_yes = float(prices[0]) if prices else 0
                poly_no = 1 - poly_yes

                # Get Kalshi prices
                kalshi_yes = kalshi_market.get("yes_ask", 0) / 100
                kalshi_no = kalshi_market.get("no_ask", 0) / 100

                if not all([poly_yes, kalshi_yes]):
                    continue

                # Check for arbitrage
                # Strategy 1: Buy YES on Poly, Buy NO on Kalshi
                cost1 = poly_yes + kalshi_no
                if cost1 < 1 - min_spread:
                    spread = 1 - cost1
                    self.opportunities.append(ArbOpportunity(
                        market_name=poly_market.get("question", ""),
                        polymarket_yes=poly_yes,
                        polymarket_no=poly_no,
                        kalshi_yes=kalshi_yes,
                        kalshi_no=kalshi_no,
                        spread=spread,
                        strategy="buy_poly_yes_kalshi_no",
                        expected_profit_pct=spread * 100,
                        poly_token_id=poly_market.get("clobTokenIds", [""])[0],
                        kalshi_ticker=kalshi_market.get("ticker", ""),
                        timestamp=datetime.now(timezone.utc)
                    ))

                # Strategy 2: Buy NO on Poly, Buy YES on Kalshi
                cost2 = poly_no + kalshi_yes
                if cost2 < 1 - min_spread:
                    spread = 1 - cost2
                    self.opportunities.append(ArbOpportunity(
                        market_name=poly_market.get("question", ""),
                        polymarket_yes=poly_yes,
                        polymarket_no=poly_no,
                        kalshi_yes=kalshi_yes,
                        kalshi_no=kalshi_no,
                        spread=spread,
                        strategy="buy_poly_no_kalshi_yes",
                        expected_profit_pct=spread * 100,
                        poly_token_id=poly_market.get("clobTokenIds", ["", ""])[1],
                        kalshi_ticker=kalshi_market.get("ticker", ""),
                        timestamp=datetime.now(timezone.utc)
                    ))

            except Exception as e:
                logger.warning(f"Error checking arb: {e}")
                continue

        # Sort by profit potential
        self.opportunities.sort(key=lambda x: -x.expected_profit_pct)
        return self.opportunities

    async def monitor_continuously(self, callback=None, interval: int = 30):
        """Continuously monitor for arbitrage opportunities."""
        while True:
            opps = await self.scan_for_arbitrage()

            if opps and callback:
                await callback(opps)
            elif opps:
                for opp in opps[:5]:
                    logger.info(
                        f"ARB: {opp.market_name[:40]} | "
                        f"Spread: {opp.spread*100:.1f}% | "
                        f"Strategy: {opp.strategy}"
                    )

            await asyncio.sleep(interval)


async def main():
    """Test cross-platform arbitrage scanner."""
    arb = CrossPlatformArbitrage()

    print("Scanning for cross-platform arbitrage...")
    opps = await arb.scan_for_arbitrage(min_spread=0.01)

    print(f"\nFound {len(opps)} opportunities:\n")
    for opp in opps[:10]:
        print(f"Market: {opp.market_name[:50]}")
        print(f"  Poly YES: ${opp.polymarket_yes:.2f} | Kalshi YES: ${opp.kalshi_yes:.2f}")
        print(f"  Spread: {opp.spread*100:.1f}% | Strategy: {opp.strategy}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
