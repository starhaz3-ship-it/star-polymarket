"""
Enhanced Feature Engineering Module

Implements advanced features based on 2024-2026 research:
- Order book imbalance (OBI)
- Microprice
- Volume/volatility features
- Time-based features
- Cross-market features
- Whale consensus signals
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import httpx


@dataclass
class OrderBookFeatures:
    """Features extracted from order book data."""
    bid_ask_spread: float = 0.0
    microprice: float = 0.0
    order_imbalance: float = 0.0  # -1 to 1
    depth_imbalance: float = 0.0
    volume_at_touch: float = 0.0


@dataclass
class VolumeFeatures:
    """Volume-based features."""
    volume_24h: float = 0.0
    volume_1h: float = 0.0
    volume_acceleration: float = 0.0  # Recent vs historical
    avg_trade_size: float = 0.0
    trade_count_1h: int = 0


@dataclass
class VolatilityFeatures:
    """Volatility-based features."""
    volatility_1h: float = 0.0
    volatility_24h: float = 0.0
    volatility_ratio: float = 0.0  # Short/long term
    price_range_1h: float = 0.0


@dataclass
class TimeFeatures:
    """Time-based features."""
    hours_to_expiry: float = 0.0
    time_decay_factor: float = 0.0  # 1/sqrt(hours+1)
    day_of_week: int = 0
    hour_of_day: int = 0
    is_weekend: bool = False
    time_since_whale_entry: float = 0.0  # hours


@dataclass
class MomentumFeatures:
    """Price momentum features."""
    momentum_5m: float = 0.0
    momentum_1h: float = 0.0
    momentum_24h: float = 0.0
    rsi_14: float = 50.0
    price_vs_vwap: float = 0.0


@dataclass
class WhaleFeatures:
    """Whale activity features."""
    whale_net_position: float = 0.0
    whale_volume_pct: float = 0.0
    whale_consensus: float = 0.0  # -1 to 1, agreement among whales
    whale_pnl_ratio: float = 0.0
    num_whales_in_market: int = 0
    whale_entry_price: float = 0.0
    whale_current_pnl: float = 0.0


@dataclass
class CrossMarketFeatures:
    """Cross-market correlation features."""
    btc_spot_price: float = 0.0
    btc_spot_momentum: float = 0.0
    spot_vs_market_diff: float = 0.0  # Implied vs actual
    cross_platform_spread: float = 0.0  # Polymarket vs Kalshi


@dataclass
class EnhancedFeatures:
    """Complete feature set for ML model."""
    # Core features
    entry_price: float = 0.0
    current_price: float = 0.0
    outcome_is_yes: bool = True

    # Order book
    order_book: OrderBookFeatures = field(default_factory=OrderBookFeatures)

    # Volume
    volume: VolumeFeatures = field(default_factory=VolumeFeatures)

    # Volatility
    volatility: VolatilityFeatures = field(default_factory=VolatilityFeatures)

    # Time
    time: TimeFeatures = field(default_factory=TimeFeatures)

    # Momentum
    momentum: MomentumFeatures = field(default_factory=MomentumFeatures)

    # Whale
    whale: WhaleFeatures = field(default_factory=WhaleFeatures)

    # Cross-market
    cross_market: CrossMarketFeatures = field(default_factory=CrossMarketFeatures)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model."""
        return np.array([
            # Core
            self.entry_price,
            self.current_price,
            1.0 if self.outcome_is_yes else 0.0,

            # Order book
            self.order_book.bid_ask_spread,
            self.order_book.microprice,
            self.order_book.order_imbalance,
            self.order_book.depth_imbalance,

            # Volume
            self.volume.volume_24h,
            self.volume.volume_1h,
            self.volume.volume_acceleration,
            self.volume.avg_trade_size,

            # Volatility
            self.volatility.volatility_1h,
            self.volatility.volatility_24h,
            self.volatility.volatility_ratio,

            # Time
            self.time.hours_to_expiry,
            self.time.time_decay_factor,
            self.time.day_of_week / 7.0,
            self.time.hour_of_day / 24.0,
            1.0 if self.time.is_weekend else 0.0,

            # Momentum
            self.momentum.momentum_5m,
            self.momentum.momentum_1h,
            self.momentum.momentum_24h,
            self.momentum.rsi_14 / 100.0,
            self.momentum.price_vs_vwap,

            # Whale
            self.whale.whale_net_position,
            self.whale.whale_volume_pct,
            self.whale.whale_consensus,
            self.whale.whale_pnl_ratio,
            self.whale.num_whales_in_market / 10.0,

            # Cross-market
            self.cross_market.btc_spot_momentum,
            self.cross_market.spot_vs_market_diff,
        ])

    @staticmethod
    def feature_names() -> List[str]:
        """Return feature names in order."""
        return [
            "entry_price", "current_price", "is_yes",
            "bid_ask_spread", "microprice", "order_imbalance", "depth_imbalance",
            "volume_24h", "volume_1h", "volume_acceleration", "avg_trade_size",
            "volatility_1h", "volatility_24h", "volatility_ratio",
            "hours_to_expiry", "time_decay_factor", "day_of_week", "hour_of_day", "is_weekend",
            "momentum_5m", "momentum_1h", "momentum_24h", "rsi_14", "price_vs_vwap",
            "whale_net_position", "whale_volume_pct", "whale_consensus", "whale_pnl_ratio", "num_whales",
            "btc_spot_momentum", "spot_vs_market_diff",
        ]


class FeatureExtractor:
    """Extracts enhanced features from market data."""

    def __init__(self):
        self.btc_price_cache = {"price": 0.0, "timestamp": 0}
        self.price_history: Dict[str, List[Tuple[float, float]]] = {}  # market_id -> [(timestamp, price)]

    async def get_btc_spot_price(self) -> float:
        """Get current BTC spot price from Binance."""
        now = datetime.now().timestamp()

        # Cache for 5 seconds
        if now - self.btc_price_cache["timestamp"] < 5:
            return self.btc_price_cache["price"]

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get("https://api.binance.com/api/v3/ticker/price",
                                     params={"symbol": "BTCUSDT"})
                price = float(r.json()["price"])
                self.btc_price_cache = {"price": price, "timestamp": now}
                return price
        except:
            return self.btc_price_cache["price"]

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI from price history."""
        if len(prices) < period + 1:
            return 50.0

        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [c if c > 0 else 0 for c in changes[-period:]]
        losses = [-c if c < 0 else 0 for c in changes[-period:]]

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_order_book_features(self, bids: List[Dict], asks: List[Dict]) -> OrderBookFeatures:
        """Calculate order book features from bid/ask data."""
        if not bids or not asks:
            return OrderBookFeatures()

        best_bid = float(bids[0].get("price", 0))
        best_ask = float(asks[0].get("price", 1))
        bid_vol = float(bids[0].get("size", 0))
        ask_vol = float(asks[0].get("size", 0))

        mid_price = (best_bid + best_ask) / 2

        # Bid-ask spread
        spread = (best_ask - best_bid) / mid_price if mid_price > 0 else 0

        # Microprice (volume-weighted)
        total_vol = bid_vol + ask_vol
        if total_vol > 0:
            microprice = (bid_vol * best_ask + ask_vol * best_bid) / total_vol
        else:
            microprice = mid_price

        # Order imbalance at best level
        if total_vol > 0:
            imbalance = (bid_vol - ask_vol) / total_vol
        else:
            imbalance = 0

        # Depth imbalance (first 5 levels)
        total_bid_vol = sum(float(b.get("size", 0)) for b in bids[:5])
        total_ask_vol = sum(float(a.get("size", 0)) for a in asks[:5])
        total_depth = total_bid_vol + total_ask_vol

        if total_depth > 0:
            depth_imbalance = (total_bid_vol - total_ask_vol) / total_depth
        else:
            depth_imbalance = 0

        return OrderBookFeatures(
            bid_ask_spread=spread,
            microprice=microprice,
            order_imbalance=imbalance,
            depth_imbalance=depth_imbalance,
            volume_at_touch=total_vol,
        )

    def calculate_time_features(self, end_date_str: str, whale_entry_time: Optional[str] = None) -> TimeFeatures:
        """Calculate time-based features."""
        now = datetime.now(timezone.utc)

        try:
            if end_date_str:
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                hours_to_expiry = max(0, (end_date - now).total_seconds() / 3600)
            else:
                hours_to_expiry = 168  # Default 1 week
        except:
            hours_to_expiry = 168

        time_decay = 1 / np.sqrt(hours_to_expiry + 1)

        # Time since whale entry
        time_since_whale = 0
        if whale_entry_time:
            try:
                whale_time = datetime.fromisoformat(whale_entry_time.replace("Z", "+00:00"))
                time_since_whale = (now - whale_time).total_seconds() / 3600
            except:
                pass

        return TimeFeatures(
            hours_to_expiry=hours_to_expiry,
            time_decay_factor=time_decay,
            day_of_week=now.weekday(),
            hour_of_day=now.hour,
            is_weekend=now.weekday() >= 5,
            time_since_whale_entry=time_since_whale,
        )

    def calculate_momentum_features(self, market_id: str, current_price: float) -> MomentumFeatures:
        """Calculate momentum features from price history."""
        now = datetime.now().timestamp()

        # Add current price to history
        if market_id not in self.price_history:
            self.price_history[market_id] = []

        self.price_history[market_id].append((now, current_price))

        # Keep only last 24 hours
        cutoff = now - 86400
        self.price_history[market_id] = [(t, p) for t, p in self.price_history[market_id] if t > cutoff]

        history = self.price_history[market_id]

        if len(history) < 2:
            return MomentumFeatures()

        prices = [p for _, p in history]

        # Calculate momentum at different windows
        def get_momentum(minutes: int) -> float:
            cutoff_ts = now - (minutes * 60)
            old_prices = [p for t, p in history if t <= cutoff_ts]
            if old_prices and prices[-1] > 0:
                old_price = old_prices[-1] if old_prices else prices[0]
                return (current_price - old_price) / old_price if old_price > 0 else 0
            return 0

        return MomentumFeatures(
            momentum_5m=get_momentum(5),
            momentum_1h=get_momentum(60),
            momentum_24h=get_momentum(1440),
            rsi_14=self.calculate_rsi(prices),
            price_vs_vwap=0,  # Would need volume data
        )

    async def extract_features(
        self,
        position_data: Dict,
        order_book: Optional[Dict] = None,
        whale_positions: Optional[List[Dict]] = None,
    ) -> EnhancedFeatures:
        """Extract all features from market data."""

        features = EnhancedFeatures()

        # Core features
        features.entry_price = float(position_data.get("avgPrice", 0) or 0)
        features.current_price = float(position_data.get("curPrice", 0) or 0)
        features.outcome_is_yes = position_data.get("outcome", "").lower() == "yes"

        market_id = position_data.get("conditionId", "")

        # Order book features
        if order_book:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            features.order_book = self.calculate_order_book_features(bids, asks)

        # Time features
        end_date = position_data.get("endDate", "")
        features.time = self.calculate_time_features(end_date)

        # Momentum features
        features.momentum = self.calculate_momentum_features(market_id, features.current_price)

        # Whale features
        if whale_positions:
            whale_yes = sum(float(p.get("currentValue", 0) or 0)
                          for p in whale_positions
                          if p.get("conditionId") == market_id and p.get("outcome") == "Yes")
            whale_no = sum(float(p.get("currentValue", 0) or 0)
                         for p in whale_positions
                         if p.get("conditionId") == market_id and p.get("outcome") == "No")

            total_whale = whale_yes + whale_no
            if total_whale > 0:
                features.whale.whale_net_position = (whale_yes - whale_no) / total_whale

            whale_pnl = float(position_data.get("cashPnl", 0) or 0)
            whale_cost = float(position_data.get("initialValue", 1) or 1)
            features.whale.whale_pnl_ratio = whale_pnl / whale_cost if whale_cost > 0 else 0
            features.whale.whale_entry_price = features.entry_price
            features.whale.whale_current_pnl = whale_pnl

        # Cross-market features
        btc_price = await self.get_btc_spot_price()
        features.cross_market.btc_spot_price = btc_price

        # Check if this is a BTC market
        title = position_data.get("title", "").upper()
        if "BTC" in title or "BITCOIN" in title:
            # Extract threshold from title
            import re
            match = re.search(r'\$?([\d,]+)', title)
            if match:
                threshold = float(match.group(1).replace(",", ""))
                implied_prob = features.current_price
                actual_prob = 1.0 if btc_price > threshold else 0.0
                features.cross_market.spot_vs_market_diff = actual_prob - implied_prob

        return features


# Global feature extractor instance
feature_extractor = FeatureExtractor()


async def extract_enhanced_features(position_data: Dict) -> EnhancedFeatures:
    """Convenience function to extract features."""
    return await feature_extractor.extract_features(position_data)
