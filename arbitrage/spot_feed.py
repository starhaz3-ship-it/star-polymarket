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

    async def _fetch_rest_price(self):
        """Fallback to REST API when WebSocket fails."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(config.BINANCE_REST_URL)
                if r.status_code == 200:
                    data = r.json()
                    self.current_price = float(data["price"])
                    self.last_update = time.time()
                    return self.current_price
        except Exception:
            pass
        return None

    async def start(self):
        """Start the WebSocket price feed with improved reconnection."""
        self._running = True
        print("[SpotFeed] Starting Binance BTC/USDT feed...")

        # Get initial price via REST
        await self._fetch_rest_price()
        if self.current_price:
            print(f"[SpotFeed] Initial price (REST): ${self.current_price:,.2f}")

        consecutive_failures = 0
        max_failures = 3

        while self._running:
            try:
                # Use shorter timeouts and ping interval for better connection
                async with websockets.connect(
                    config.BINANCE_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                    open_timeout=10,
                ) as ws:
                    self._ws = ws
                    print("[SpotFeed] Connected to Binance WebSocket")
                    consecutive_failures = 0

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
                            pass  # Silently ignore parse errors

            except websockets.exceptions.ConnectionClosed:
                consecutive_failures += 1
                print(f"[SpotFeed] Connection closed, reconnecting... ({consecutive_failures})")
                await asyncio.sleep(1)
            except Exception as e:
                consecutive_failures += 1
                # Fall back to REST if WebSocket keeps failing
                if consecutive_failures >= max_failures:
                    print(f"[SpotFeed] WebSocket unstable, using REST fallback")
                    await self._fetch_rest_price()
                    consecutive_failures = 0
                await asyncio.sleep(2)

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
