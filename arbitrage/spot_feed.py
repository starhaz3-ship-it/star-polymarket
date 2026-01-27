"""
Real-time BTC spot price feed from Binance.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Callable, Optional

import httpx
import websockets

from .config import config


@dataclass
class PriceUpdate:
    """A price update from the exchange."""
    price: float
    timestamp: float
    source: str = "binance"


class BinanceSpotFeed:
    """Real-time BTC/USDT price feed from Binance."""

    def __init__(self, on_price: Optional[Callable[[PriceUpdate], None]] = None):
        self.on_price = on_price
        self.current_price: Optional[float] = None
        self.last_update: float = 0
        self._running = False
        self._ws = None

    def get_price_sync(self) -> Optional[float]:
        """Get current BTC price synchronously (REST API)."""
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(config.BINANCE_REST_URL)
                if r.status_code == 200:
                    data = r.json()
                    self.current_price = float(data["price"])
                    self.last_update = time.time()
                    return self.current_price
        except Exception as e:
            print(f"[SpotFeed] REST error: {e}")
        return None

    async def start(self):
        """Start the WebSocket price feed."""
        self._running = True
        print("[SpotFeed] Starting Binance BTC/USDT feed...")

        while self._running:
            try:
                async with websockets.connect(config.BINANCE_WS_URL) as ws:
                    self._ws = ws
                    print("[SpotFeed] Connected to Binance WebSocket")

                    async for message in ws:
                        if not self._running:
                            break

                        try:
                            data = json.loads(message)
                            price = float(data["p"])  # Trade price
                            self.current_price = price
                            self.last_update = time.time()

                            if self.on_price:
                                update = PriceUpdate(
                                    price=price,
                                    timestamp=self.last_update
                                )
                                self.on_price(update)

                        except (KeyError, ValueError) as e:
                            print(f"[SpotFeed] Parse error: {e}")

            except websockets.exceptions.ConnectionClosed:
                print("[SpotFeed] Connection closed, reconnecting...")
                await asyncio.sleep(1)
            except Exception as e:
                print(f"[SpotFeed] Error: {e}, reconnecting...")
                await asyncio.sleep(5)

    def stop(self):
        """Stop the price feed."""
        self._running = False
        if self._ws:
            asyncio.create_task(self._ws.close())

    def is_stale(self, max_age_sec: float = 5.0) -> bool:
        """Check if price data is stale."""
        if self.current_price is None:
            return True
        return (time.time() - self.last_update) > max_age_sec


# Convenience function
def get_btc_price() -> Optional[float]:
    """Get current BTC price (one-shot REST call)."""
    feed = BinanceSpotFeed()
    return feed.get_price_sync()


if __name__ == "__main__":
    # Test the feed
    def on_price(update: PriceUpdate):
        print(f"BTC: ${update.price:,.2f}")

    feed = BinanceSpotFeed(on_price=on_price)
    print(f"Current price (REST): ${feed.get_price_sync():,.2f}")

    asyncio.run(feed.start())
