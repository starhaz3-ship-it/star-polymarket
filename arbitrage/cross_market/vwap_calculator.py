"""
Real-Time VWAP Calculator with Sliding Window

Based on AFT paper using 950 blocks (~1 hour) for volume-weighted average price.
"""

import time
import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Deque
import httpx


@dataclass
class TradeRecord:
    """A single trade for VWAP calculation."""
    price: float
    size: float
    timestamp: float


@dataclass
class VWAPState:
    """State for calculating VWAP on a sliding window."""
    trades: Deque[TradeRecord] = field(default_factory=lambda: deque(maxlen=1000))
    last_updated: float = 0.0

    def add_trade(self, price: float, size: float, timestamp: float):
        """Add a trade to the window."""
        self.trades.append(TradeRecord(price=price, size=size, timestamp=timestamp))
        self.last_updated = time.time()

    def calculate_vwap(self, window_seconds: int) -> Optional[float]:
        """Calculate VWAP over the window."""
        if not self.trades:
            return None

        cutoff = time.time() - window_seconds
        total_volume = 0.0
        volume_weighted_sum = 0.0

        for trade in self.trades:
            if trade.timestamp >= cutoff:
                total_volume += trade.size
                volume_weighted_sum += trade.price * trade.size

        if total_volume == 0:
            return None

        return volume_weighted_sum / total_volume

    def prune_old(self, window_seconds: int):
        """Remove trades outside the window."""
        cutoff = time.time() - window_seconds
        while self.trades and self.trades[0].timestamp < cutoff:
            self.trades.popleft()


class VWAPCalculator:
    """
    Real-time VWAP calculator with sliding window.

    Features:
    - 1-hour default window (matching paper's 950 blocks)
    - Async seeding from data-api
    - Efficient sliding window pruning
    - Support for multiple tokens
    """

    DATA_API_URL = "https://data-api.polymarket.com"

    def __init__(self, window_seconds: int = 3600):
        """
        Initialize VWAP calculator.

        Args:
            window_seconds: Window size in seconds (default 1 hour)
        """
        self.window_seconds = window_seconds
        self.states: Dict[str, VWAPState] = {}
        self._lock = asyncio.Lock()

    def add_trade(
        self,
        token_id: str,
        price: float,
        size: float,
        timestamp: Optional[float] = None
    ):
        """
        Add a trade and update VWAP.

        Args:
            token_id: The token identifier
            price: Trade price (0-1)
            size: Trade size (number of shares)
            timestamp: Trade timestamp (defaults to now)
        """
        if token_id not in self.states:
            self.states[token_id] = VWAPState()

        ts = timestamp or time.time()
        self.states[token_id].add_trade(price, size, ts)

    def get_vwap(self, token_id: str) -> Optional[float]:
        """
        Get current VWAP for a token.

        Returns:
            VWAP or None if no data available
        """
        if token_id not in self.states:
            return None

        return self.states[token_id].calculate_vwap(self.window_seconds)

    def get_vwap_with_fallback(self, token_id: str, spot_price: float) -> float:
        """
        Get VWAP with fallback to spot price.

        Args:
            token_id: The token identifier
            spot_price: Fallback spot price

        Returns:
            VWAP if available, otherwise spot_price
        """
        vwap = self.get_vwap(token_id)
        return vwap if vwap is not None else spot_price

    def prune_all(self):
        """Prune old trades from all states."""
        for state in self.states.values():
            state.prune_old(self.window_seconds)

    async def seed_from_data_api(
        self,
        token_id: str,
        limit: int = 500
    ) -> int:
        """
        Seed VWAP state from data-api.polymarket.com historical trades.

        Args:
            token_id: Token to fetch trades for
            limit: Maximum number of trades to fetch

        Returns:
            Number of trades loaded
        """
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                # Fetch recent trades
                response = await client.get(
                    f"{self.DATA_API_URL}/trades",
                    params={
                        "asset_id": token_id,
                        "limit": limit,
                    }
                )

                if response.status_code != 200:
                    return 0

                trades = response.json()
                if not trades:
                    return 0

                # Initialize state if needed
                if token_id not in self.states:
                    self.states[token_id] = VWAPState()

                state = self.states[token_id]
                count = 0

                for trade in trades:
                    try:
                        price = float(trade.get("price", 0))
                        size = float(trade.get("size", 0))
                        # Parse timestamp - could be ISO format or unix
                        ts_str = trade.get("timestamp") or trade.get("created_at")
                        if isinstance(ts_str, str):
                            # Try parsing ISO format
                            from datetime import datetime
                            try:
                                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                                ts = dt.timestamp()
                            except:
                                ts = time.time()
                        else:
                            ts = float(ts_str) if ts_str else time.time()

                        if price > 0 and size > 0:
                            state.add_trade(price, size, ts)
                            count += 1
                    except (ValueError, TypeError):
                        continue

                return count

            except Exception as e:
                print(f"[VWAP] Error seeding from API: {e}")
                return 0

    async def seed_multiple(self, token_ids: List[str], limit: int = 500) -> Dict[str, int]:
        """
        Seed VWAP for multiple tokens concurrently.

        Args:
            token_ids: List of token IDs to seed
            limit: Max trades per token

        Returns:
            Dict mapping token_id to number of trades loaded
        """
        tasks = [self.seed_from_data_api(tid, limit) for tid in token_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            tid: (r if isinstance(r, int) else 0)
            for tid, r in zip(token_ids, results)
        }

    def get_all_vwaps(self) -> Dict[str, Optional[float]]:
        """Get VWAPs for all tracked tokens."""
        return {
            token_id: self.get_vwap(token_id)
            for token_id in self.states.keys()
        }

    def get_statistics(self) -> dict:
        """Get calculator statistics."""
        stats = {
            "tokens_tracked": len(self.states),
            "window_seconds": self.window_seconds,
            "tokens": {},
        }

        for token_id, state in self.states.items():
            stats["tokens"][token_id] = {
                "trade_count": len(state.trades),
                "vwap": self.get_vwap(token_id),
                "last_updated": state.last_updated,
            }

        return stats

    def clear(self, token_id: Optional[str] = None):
        """
        Clear VWAP state.

        Args:
            token_id: Specific token to clear, or None to clear all
        """
        if token_id:
            if token_id in self.states:
                del self.states[token_id]
        else:
            self.states.clear()


class VWAPManager:
    """
    Manager for VWAP calculations across multiple markets.

    Provides higher-level interface for cross-market arbitrage.
    """

    def __init__(self, window_seconds: int = 3600):
        self.calculator = VWAPCalculator(window_seconds)
        self._initialized_tokens: set = set()

    async def ensure_initialized(self, token_ids: List[str]):
        """Ensure tokens have VWAP data seeded."""
        new_tokens = [t for t in token_ids if t not in self._initialized_tokens]

        if new_tokens:
            results = await self.calculator.seed_multiple(new_tokens)
            for token_id, count in results.items():
                if count > 0:
                    self._initialized_tokens.add(token_id)

    def get_market_vwaps(
        self,
        yes_token_id: str,
        no_token_id: str,
        yes_spot: float,
        no_spot: float
    ) -> tuple[float, float]:
        """
        Get VWAPs for a market's YES and NO tokens.

        Returns:
            (yes_vwap, no_vwap) with fallbacks to spot prices
        """
        yes_vwap = self.calculator.get_vwap_with_fallback(yes_token_id, yes_spot)
        no_vwap = self.calculator.get_vwap_with_fallback(no_token_id, no_spot)
        return yes_vwap, no_vwap

    def add_trade(self, token_id: str, price: float, size: float, timestamp: float = None):
        """Add a trade update (e.g., from WebSocket)."""
        self.calculator.add_trade(token_id, price, size, timestamp)

    def prune(self):
        """Prune old trades from all tokens."""
        self.calculator.prune_all()
