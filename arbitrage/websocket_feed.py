"""
Polymarket Real-Time Data Socket (RTDS) WebSocket feed.
Provides real-time price updates instead of polling.

WebSocket URL: wss://ws-live-data.polymarket.com
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed

from .config import config


@dataclass
class PriceUpdate:
    """Real-time price update from WebSocket."""
    token_id: str
    price: float
    timestamp: float


class PolymarketWebSocket:
    """
    Real-time WebSocket feed for Polymarket prices.

    Uses the RTDS (Real-Time Data Socket) at wss://ws-live-data.polymarket.com
    """

    WS_URL = "wss://ws-live-data.polymarket.com"
    PING_INTERVAL = 5.0  # Send ping every 5 seconds
    RECONNECT_DELAY = 3.0

    def __init__(
        self,
        on_price: Optional[Callable[[PriceUpdate], None]] = None,
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[], None]] = None,
    ):
        self.on_price = on_price
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect

        self._ws = None
        self._running = False
        self._subscribed_tokens: Set[str] = set()
        self._prices: Dict[str, float] = {}
        self._last_update: Dict[str, float] = {}

    async def connect(self):
        """Connect to the WebSocket."""
        try:
            self._ws = await websockets.connect(
                self.WS_URL,
                ping_interval=self.PING_INTERVAL,
                ping_timeout=10,
            )
            print(f"[WS] Connected to {self.WS_URL}")

            if self.on_connect:
                self.on_connect()

            return True
        except Exception as e:
            print(f"[WS] Connection error: {e}")
            return False

    async def subscribe(self, token_ids: List[str]):
        """Subscribe to price updates for specific tokens."""
        if not self._ws:
            print("[WS] Not connected, cannot subscribe")
            return False

        try:
            # RTDS subscription format
            message = {
                "action": "subscribe",
                "subscriptions": [
                    {
                        "topic": "market",
                        "type": "price",
                        "assets_ids": token_ids,
                    }
                ]
            }

            await self._ws.send(json.dumps(message))
            self._subscribed_tokens.update(token_ids)
            print(f"[WS] Subscribed to {len(token_ids)} tokens")
            return True

        except Exception as e:
            print(f"[WS] Subscribe error: {e}")
            return False

    async def unsubscribe(self, token_ids: List[str]):
        """Unsubscribe from token updates."""
        if not self._ws:
            return

        try:
            message = {
                "action": "unsubscribe",
                "subscriptions": [
                    {
                        "topic": "market",
                        "type": "price",
                        "assets_ids": token_ids,
                    }
                ]
            }

            await self._ws.send(json.dumps(message))
            self._subscribed_tokens -= set(token_ids)

        except Exception as e:
            print(f"[WS] Unsubscribe error: {e}")

    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            # Handle different message types
            msg_type = data.get("type", "")
            topic = data.get("topic", "")

            if topic == "market" and "payload" in data:
                payload = data["payload"]

                # Extract price update
                token_id = payload.get("asset_id", payload.get("token_id", ""))
                price = payload.get("price", payload.get("mid_price"))

                if token_id and price is not None:
                    self._prices[token_id] = float(price)
                    self._last_update[token_id] = time.time()

                    if self.on_price:
                        update = PriceUpdate(
                            token_id=token_id,
                            price=float(price),
                            timestamp=time.time()
                        )
                        self.on_price(update)

            elif msg_type == "error":
                print(f"[WS] Error: {data.get('message', data)}")

        except json.JSONDecodeError:
            pass  # Ignore non-JSON messages (pings, etc.)
        except Exception as e:
            print(f"[WS] Message handling error: {e}")

    async def _receive_loop(self):
        """Main receive loop."""
        while self._running and self._ws:
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=30.0
                )
                await self._handle_message(message)

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await self._ws.ping()
                except:
                    break

            except ConnectionClosed:
                print("[WS] Connection closed")
                break

            except Exception as e:
                print(f"[WS] Receive error: {e}")
                break

        if self.on_disconnect:
            self.on_disconnect()

    async def start(self, token_ids: Optional[List[str]] = None):
        """Start the WebSocket feed."""
        self._running = True

        while self._running:
            # Connect
            if not await self.connect():
                print(f"[WS] Reconnecting in {self.RECONNECT_DELAY}s...")
                await asyncio.sleep(self.RECONNECT_DELAY)
                continue

            # Subscribe to tokens if provided
            if token_ids:
                await self.subscribe(token_ids)
            elif self._subscribed_tokens:
                # Re-subscribe after reconnect
                await self.subscribe(list(self._subscribed_tokens))

            # Run receive loop
            await self._receive_loop()

            # Clean up
            if self._ws:
                await self._ws.close()
                self._ws = None

            if self._running:
                print(f"[WS] Reconnecting in {self.RECONNECT_DELAY}s...")
                await asyncio.sleep(self.RECONNECT_DELAY)

    def stop(self):
        """Stop the WebSocket feed."""
        self._running = False

    def get_price(self, token_id: str) -> Optional[float]:
        """Get cached price for a token."""
        return self._prices.get(token_id)

    def get_all_prices(self) -> Dict[str, float]:
        """Get all cached prices."""
        return self._prices.copy()

    def is_stale(self, token_id: str, max_age_sec: float = 10.0) -> bool:
        """Check if a price is stale."""
        last = self._last_update.get(token_id, 0)
        return (time.time() - last) > max_age_sec


# Convenience function for creating WebSocket-enabled feed
async def create_ws_feed(
    token_ids: List[str],
    on_price: Callable[[PriceUpdate], None]
) -> PolymarketWebSocket:
    """Create and start a WebSocket feed."""
    feed = PolymarketWebSocket(on_price=on_price)
    asyncio.create_task(feed.start(token_ids))
    return feed


if __name__ == "__main__":
    # Test the WebSocket feed
    import sys

    def on_price(update: PriceUpdate):
        print(f"[Price] {update.token_id[:20]}... = ${update.price:.4f}")

    def on_connect():
        print("[Event] Connected!")

    def on_disconnect():
        print("[Event] Disconnected!")

    async def main():
        feed = PolymarketWebSocket(
            on_price=on_price,
            on_connect=on_connect,
            on_disconnect=on_disconnect,
        )

        # Example token IDs (replace with real ones)
        # These would come from the Gamma API
        test_tokens = []

        if len(sys.argv) > 1:
            test_tokens = sys.argv[1:]

        print(f"Starting WebSocket feed with {len(test_tokens)} tokens...")
        await feed.start(test_tokens)

    asyncio.run(main())
