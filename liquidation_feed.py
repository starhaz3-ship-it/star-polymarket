"""
Binance Futures Liquidation Feed — real-time BTC long/short imbalance tracker.

Connects to public Binance Futures WebSocket (no API key needed).
Tracks BTC liquidation events in a rolling 10-minute window.
Used by both Polymarket bots as a directional confirmation signal.

Long liquidations (forced sells) = DOWN pressure on BTC
Short liquidations (forced buys) = UP pressure on BTC
"""

import asyncio
import json
import time
from collections import deque

try:
    import websockets
except ImportError:
    websockets = None

BINANCE_FUTURES_WS = "wss://fstream.binance.com/ws/!forceOrder@arr"


class BinanceLiqFeed:
    """Track BTC liquidation imbalance from Binance Futures public stream."""

    def __init__(self, window_sec: int = 600):
        self.window = window_sec
        self.events: deque = deque(maxlen=500)
        self._ws_task = None
        self._connected = False
        self._reconnect_delay = 5

    async def start(self):
        """Launch WebSocket listener as background task."""
        if websockets is None:
            print("[LIQ] websockets package not installed — liquidation feed disabled")
            return
        self._ws_task = asyncio.create_task(self._listen())

    async def _listen(self):
        """Connect and listen for forceOrder events."""
        while True:
            try:
                async with websockets.connect(BINANCE_FUTURES_WS,
                                              ping_interval=30,
                                              ping_timeout=10,
                                              close_timeout=5) as ws:
                    self._connected = True
                    self._reconnect_delay = 5
                    print("[LIQ] Connected to Binance Futures liquidation stream")
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                            order = data.get("o", {})
                            symbol = order.get("s", "")
                            if symbol != "BTCUSDT":
                                continue
                            # S="SELL" = long liquidated, S="BUY" = short liquidated
                            raw_side = order.get("S", "")
                            liq_side = "LONG" if raw_side == "SELL" else "SHORT"
                            qty = float(order.get("q", 0))
                            price = float(order.get("p", 0))
                            usd_value = qty * price
                            self.events.append((time.time(), liq_side, usd_value))
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            except Exception as e:
                self._connected = False
                print(f"[LIQ] Disconnected: {e}. Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    def get_imbalance(self) -> dict:
        """Return current BTC liquidation imbalance in the rolling window.

        Returns:
            dict with keys:
                long_pct: float (0-1) — fraction of liquidations that were longs
                short_pct: float (0-1)
                count: int — total events in window
                volume_usd: float — total USD liquidated
                signal: 'DOWN'|'UP'|None — DOWN if longs dominating, UP if shorts
        """
        cutoff = time.time() - self.window
        # Prune old events
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

        if not self.events:
            return {"long_pct": 0.5, "short_pct": 0.5, "count": 0,
                    "volume_usd": 0.0, "signal": None}

        long_vol = sum(usd for _, side, usd in self.events if side == "LONG")
        short_vol = sum(usd for _, side, usd in self.events if side == "SHORT")
        total_vol = long_vol + short_vol
        count = len(self.events)

        long_pct = long_vol / total_vol if total_vol > 0 else 0.5
        short_pct = 1.0 - long_pct

        signal = None
        if count >= 3:
            if long_pct > 0.60:
                signal = "DOWN"
            elif short_pct > 0.60:
                signal = "UP"

        return {
            "long_pct": round(long_pct, 3),
            "short_pct": round(short_pct, 3),
            "count": count,
            "volume_usd": round(total_vol, 2),
            "signal": signal,
        }

    @property
    def connected(self) -> bool:
        return self._connected
