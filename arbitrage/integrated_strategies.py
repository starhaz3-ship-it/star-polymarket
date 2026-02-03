"""
Integrated Trading Strategies

Combines strategies from multiple external sources:
- Kalshi cross-platform arbitrage
- Spike detection (price momentum)
- Flash crash detection
- Enhanced TA indicators

Sources:
- github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot
- github.com/Trust412/Polymarket-spike-bot-v1
- github.com/discountry/polymarket-trading-bot
- github.com/je-suis-tm/quant-trading
"""

import asyncio
import time
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from enum import Enum

import httpx


class SignalType(Enum):
    KALSHI_ARB = "kalshi_arb"
    SPIKE = "spike"
    FLASH_CRASH = "flash_crash"
    MOMENTUM = "momentum"


@dataclass
class IntegratedSignal:
    """Signal from integrated strategies."""
    signal_type: SignalType
    side: str  # "UP" or "DOWN" or "YES" or "NO"
    confidence: float  # 0-1
    edge: float  # Expected profit margin
    details: Dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ============================================================================
# KALSHI ARBITRAGE DETECTOR
# Source: github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot
# ============================================================================

class KalshiArbitrageDetector:
    """
    Detects cross-platform arbitrage between Polymarket and Kalshi.

    Logic:
    - Polymarket "Up" means Price >= Poly_Strike
    - Polymarket "Down" means Price < Poly_Strike
    - Kalshi "Yes" means Price >= Kalshi_Strike
    - Kalshi "No" means Price < Kalshi_Strike

    If combined cost < $1.00, risk-free profit exists.
    """

    KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2/markets"
    BINANCE_API = "https://api.binance.com/api/v3/ticker/price"

    def __init__(self):
        self.last_scan = 0
        self.scan_interval = 5  # seconds

    async def get_binance_price(self) -> Optional[float]:
        """Get current BTC price from Binance."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(self.BINANCE_API, params={"symbol": "BTCUSDT"})
                return float(r.json()["price"])
        except:
            return None

    async def get_kalshi_markets(self, event_ticker: str = "INXBTC") -> List[Dict]:
        """Fetch Kalshi BTC markets."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    self.KALSHI_API,
                    params={"limit": 100, "event_ticker": event_ticker}
                )
                markets = r.json().get('markets', [])

                parsed = []
                for m in markets:
                    strike = self._parse_strike(m.get('subtitle', ''))
                    if strike > 0:
                        parsed.append({
                            'strike': strike,
                            'yes_ask': m.get('yes_ask', 0) / 100.0,  # Convert cents to dollars
                            'no_ask': m.get('no_ask', 0) / 100.0,
                        })
                return sorted(parsed, key=lambda x: x['strike'])
        except:
            return []

    def _parse_strike(self, subtitle: str) -> float:
        """Parse strike price from Kalshi subtitle like '$96,250 or above'."""
        match = re.search(r'\$([\d,]+)', subtitle)
        if match:
            return float(match.group(1).replace(',', ''))
        return 0.0

    async def detect_arbitrage(
        self,
        poly_strike: float,
        poly_up_price: float,
        poly_down_price: float
    ) -> Optional[IntegratedSignal]:
        """
        Check for arbitrage between Polymarket and Kalshi.

        Returns signal if risk-free profit exists.
        """
        kalshi_markets = await self.get_kalshi_markets()
        if not kalshi_markets:
            return None

        for km in kalshi_markets:
            kalshi_strike = km['strike']
            kalshi_yes = km['yes_ask']
            kalshi_no = km['no_ask']

            # Skip if strikes too far apart
            if abs(kalshi_strike - poly_strike) > 5000:
                continue

            # Case 1: Poly Strike > Kalshi Strike
            # Strategy: Buy Poly DOWN + Kalshi YES
            if poly_strike > kalshi_strike:
                total_cost = poly_down_price + kalshi_yes
                if total_cost < 0.98:  # 2% minimum edge
                    margin = 1.0 - total_cost
                    return IntegratedSignal(
                        signal_type=SignalType.KALSHI_ARB,
                        side="DOWN",
                        confidence=min(margin * 2, 1.0),
                        edge=margin,
                        details={
                            "strategy": "Poly DOWN + Kalshi YES",
                            "poly_strike": poly_strike,
                            "kalshi_strike": kalshi_strike,
                            "poly_cost": poly_down_price,
                            "kalshi_cost": kalshi_yes,
                            "total_cost": total_cost,
                            "profit": margin,
                        }
                    )

            # Case 2: Poly Strike < Kalshi Strike
            # Strategy: Buy Poly UP + Kalshi NO
            elif poly_strike < kalshi_strike:
                total_cost = poly_up_price + kalshi_no
                if total_cost < 0.98:
                    margin = 1.0 - total_cost
                    return IntegratedSignal(
                        signal_type=SignalType.KALSHI_ARB,
                        side="UP",
                        confidence=min(margin * 2, 1.0),
                        edge=margin,
                        details={
                            "strategy": "Poly UP + Kalshi NO",
                            "poly_strike": poly_strike,
                            "kalshi_strike": kalshi_strike,
                            "poly_cost": poly_up_price,
                            "kalshi_cost": kalshi_no,
                            "total_cost": total_cost,
                            "profit": margin,
                        }
                    )

        return None


# ============================================================================
# SPIKE DETECTOR
# Source: github.com/Trust412/Polymarket-spike-bot-v1
# ============================================================================

class SpikeDetector:
    """
    Detects price spikes for momentum trading.

    When price moves significantly in short time, enter position
    expecting continuation or mean reversion.
    """

    def __init__(
        self,
        spike_threshold: float = 0.02,  # 2% price change
        lookback_seconds: int = 60,
        min_samples: int = 5
    ):
        self.spike_threshold = spike_threshold
        self.lookback_seconds = lookback_seconds
        self.min_samples = min_samples
        self.price_history: deque = deque(maxlen=1000)

    def add_price(self, price: float, timestamp: float = None):
        """Add price observation."""
        if timestamp is None:
            timestamp = time.time()
        self.price_history.append((timestamp, price))

    def detect_spike(self) -> Optional[IntegratedSignal]:
        """
        Detect if a price spike has occurred.

        Returns signal if spike detected.
        """
        if len(self.price_history) < self.min_samples:
            return None

        now = time.time()
        cutoff = now - self.lookback_seconds

        # Get recent prices
        recent = [(t, p) for t, p in self.price_history if t >= cutoff]
        if len(recent) < self.min_samples:
            return None

        # Calculate price change
        oldest_price = recent[0][1]
        newest_price = recent[-1][1]

        if oldest_price == 0:
            return None

        pct_change = (newest_price - oldest_price) / oldest_price

        # Check for spike
        if abs(pct_change) >= self.spike_threshold:
            side = "UP" if pct_change > 0 else "DOWN"
            return IntegratedSignal(
                signal_type=SignalType.SPIKE,
                side=side,
                confidence=min(abs(pct_change) / self.spike_threshold * 0.5, 1.0),
                edge=abs(pct_change),
                details={
                    "pct_change": pct_change,
                    "oldest_price": oldest_price,
                    "newest_price": newest_price,
                    "lookback_seconds": self.lookback_seconds,
                    "samples": len(recent),
                }
            )

        return None


# ============================================================================
# FLASH CRASH DETECTOR
# Source: github.com/discountry/polymarket-trading-bot
# ============================================================================

class FlashCrashDetector:
    """
    Detects flash crashes (sudden probability drops) for contrarian entry.

    When a market probability drops sharply, it may be an overreaction
    creating a buying opportunity.
    """

    def __init__(
        self,
        drop_threshold: float = 0.15,  # 15% probability drop
        lookback_seconds: int = 30,
        recovery_target: float = 0.10,  # Expect 10% recovery
    ):
        self.drop_threshold = drop_threshold
        self.lookback_seconds = lookback_seconds
        self.recovery_target = recovery_target
        self.prob_history: Dict[str, deque] = {}  # market_id -> history

    def add_probability(self, market_id: str, prob: float, timestamp: float = None):
        """Add probability observation for a market."""
        if timestamp is None:
            timestamp = time.time()

        if market_id not in self.prob_history:
            self.prob_history[market_id] = deque(maxlen=500)

        self.prob_history[market_id].append((timestamp, prob))

    def detect_flash_crash(self, market_id: str) -> Optional[IntegratedSignal]:
        """
        Detect flash crash in a specific market.

        Returns signal if crash detected and recovery expected.
        """
        if market_id not in self.prob_history:
            return None

        history = self.prob_history[market_id]
        if len(history) < 3:
            return None

        now = time.time()
        cutoff = now - self.lookback_seconds

        # Get recent observations
        recent = [(t, p) for t, p in history if t >= cutoff]
        if len(recent) < 2:
            return None

        # Find max and current
        max_prob = max(p for _, p in recent)
        current_prob = recent[-1][1]

        # Calculate drop
        drop = max_prob - current_prob

        if drop >= self.drop_threshold:
            # Flash crash detected - contrarian buy signal
            expected_recovery = min(drop * 0.5, self.recovery_target)

            return IntegratedSignal(
                signal_type=SignalType.FLASH_CRASH,
                side="YES",  # Buy the crashed asset
                confidence=min(drop / self.drop_threshold * 0.6, 1.0),
                edge=expected_recovery,
                details={
                    "max_prob": max_prob,
                    "current_prob": current_prob,
                    "drop": drop,
                    "expected_recovery": expected_recovery,
                    "market_id": market_id,
                }
            )

        return None


# ============================================================================
# MOMENTUM INDICATOR
# Source: github.com/je-suis-tm/quant-trading
# ============================================================================

class MomentumIndicator:
    """
    Multi-factor momentum indicator combining:
    - Price momentum
    - Volume momentum
    - RSI momentum (rate of change of RSI)
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.prices: deque = deque(maxlen=period * 2)
        self.volumes: deque = deque(maxlen=period * 2)

    def add_candle(self, close: float, volume: float):
        """Add price and volume data."""
        self.prices.append(close)
        self.volumes.append(volume)

    def calculate_momentum(self) -> Optional[IntegratedSignal]:
        """Calculate multi-factor momentum signal."""
        if len(self.prices) < self.period:
            return None

        prices = list(self.prices)
        volumes = list(self.volumes)

        # Price momentum (Rate of Change)
        price_roc = (prices[-1] - prices[-self.period]) / prices[-self.period]

        # Volume momentum
        recent_vol = sum(volumes[-self.period//2:]) / (self.period//2)
        older_vol = sum(volumes[-self.period:-self.period//2]) / (self.period//2)
        vol_ratio = recent_vol / older_vol if older_vol > 0 else 1.0

        # Combined score
        momentum_score = price_roc * vol_ratio

        if abs(momentum_score) > 0.01:  # 1% threshold
            side = "UP" if momentum_score > 0 else "DOWN"
            return IntegratedSignal(
                signal_type=SignalType.MOMENTUM,
                side=side,
                confidence=min(abs(momentum_score) * 10, 1.0),
                edge=abs(momentum_score),
                details={
                    "price_roc": price_roc,
                    "vol_ratio": vol_ratio,
                    "momentum_score": momentum_score,
                }
            )

        return None


# ============================================================================
# INTEGRATED STRATEGY MANAGER
# ============================================================================

class IntegratedStrategyManager:
    """
    Manages all integrated strategies and aggregates signals.
    """

    def __init__(self):
        self.kalshi_detector = KalshiArbitrageDetector()
        self.spike_detector = SpikeDetector()
        self.flash_crash_detector = FlashCrashDetector()
        self.momentum_indicator = MomentumIndicator()

        self.signals: List[IntegratedSignal] = []

    def add_price_data(
        self,
        btc_price: float,
        market_id: str = None,
        market_prob: float = None,
        volume: float = 0
    ):
        """Add price data to all detectors."""
        now = time.time()

        self.spike_detector.add_price(btc_price, now)
        self.momentum_indicator.add_candle(btc_price, volume)

        if market_id and market_prob is not None:
            self.flash_crash_detector.add_probability(market_id, market_prob, now)

    async def scan_all(
        self,
        poly_strike: float = None,
        poly_up_price: float = None,
        poly_down_price: float = None,
        market_id: str = None
    ) -> List[IntegratedSignal]:
        """
        Run all strategy detectors and return aggregated signals.
        """
        signals = []

        # 1. Kalshi Arbitrage
        if poly_strike and poly_up_price and poly_down_price:
            try:
                arb_signal = await self.kalshi_detector.detect_arbitrage(
                    poly_strike, poly_up_price, poly_down_price
                )
                if arb_signal:
                    signals.append(arb_signal)
            except:
                pass

        # 2. Spike Detection
        spike_signal = self.spike_detector.detect_spike()
        if spike_signal:
            signals.append(spike_signal)

        # 3. Flash Crash Detection
        if market_id:
            crash_signal = self.flash_crash_detector.detect_flash_crash(market_id)
            if crash_signal:
                signals.append(crash_signal)

        # 4. Momentum
        momentum_signal = self.momentum_indicator.calculate_momentum()
        if momentum_signal:
            signals.append(momentum_signal)

        self.signals = signals
        return signals

    def get_consensus(self) -> Optional[Tuple[str, float]]:
        """
        Get consensus signal from all strategies.

        Returns (side, confidence) if consensus exists.
        """
        if not self.signals:
            return None

        up_score = 0
        down_score = 0

        for sig in self.signals:
            if sig.side in ("UP", "YES"):
                up_score += sig.confidence * sig.edge
            elif sig.side in ("DOWN", "NO"):
                down_score += sig.confidence * sig.edge

        if up_score > down_score and up_score > 0.1:
            return ("UP", up_score)
        elif down_score > up_score and down_score > 0.1:
            return ("DOWN", down_score)

        return None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def quick_scan(
    btc_price: float,
    poly_up: float,
    poly_down: float,
    market_id: str = None
) -> List[IntegratedSignal]:
    """Quick scan using all strategies."""
    manager = IntegratedStrategyManager()
    manager.add_price_data(btc_price, market_id, poly_down)  # DOWN prob as market prob

    return await manager.scan_all(
        poly_strike=btc_price,
        poly_up_price=poly_up,
        poly_down_price=poly_down,
        market_id=market_id
    )


if __name__ == "__main__":
    # Test
    async def test():
        signals = await quick_scan(
            btc_price=75000,
            poly_up=0.45,
            poly_down=0.55,
            market_id="test_market"
        )
        for sig in signals:
            print(f"{sig.signal_type.value}: {sig.side} @ {sig.confidence:.1%} edge={sig.edge:.2%}")

    asyncio.run(test())
