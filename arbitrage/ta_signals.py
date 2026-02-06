"""
Technical Analysis Signal Generator for BTC 15-Minute Markets

Based on analysis of PolymarketBTC15mAssistant strategy:
- Heiken Ashi candlestick analysis
- RSI with slope tracking
- MACD with histogram delta
- Session VWAP with slope
- Direction scoring system
- Time-aware probability adjustment
- Phase-based decision thresholds
- Market regime detection

Created: 2026-02-03
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import time


class MarketRegime(Enum):
    """Market regime classification."""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    CHOP = "chop"


class TradePhase(Enum):
    """Time-based trading phase."""
    EARLY = "early"    # > 10 minutes remaining
    MID = "mid"        # 5-10 minutes remaining
    LATE = "late"      # < 5 minutes remaining


class SignalStrength(Enum):
    """Signal strength classification."""
    STRONG = "strong"      # Edge >= 20%
    GOOD = "good"          # Edge >= 10%
    OPTIONAL = "optional"  # Edge >= 5%
    NONE = "none"          # No trade


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class HeikenAshiCandle:
    """Heiken Ashi transformed candle."""
    open: float
    high: float
    low: float
    close: float
    is_green: bool
    body_size: float


@dataclass
class MACDResult:
    """MACD indicator values."""
    macd_line: float
    signal_line: float
    histogram: float
    hist_delta: Optional[float]  # Change from previous


@dataclass
class TTMSqueezeResult:
    """TTM Squeeze indicator values (Bollinger Band + Keltner Channel squeeze)."""
    squeeze_on: bool          # True = BB inside KC (volatility compression)
    momentum: float           # Oscillator value (-100 to +100 range)
    momentum_rising: bool     # True if momentum increasing
    squeeze_fired: bool       # True = just came out of squeeze


@dataclass
class OrderFlowData:
    """Order flow indicators from orderbook and trades."""
    # OBI - Order Book Imbalance
    obi: float = 0.0              # -1 to +1 (bid-ask volume ratio)
    obi_signal: str = "NEUTRAL"   # BULLISH / BEARISH / NEUTRAL

    # CVD - Cumulative Volume Delta
    cvd_1m: float = 0.0           # Net buy pressure over 1 minute
    cvd_5m: float = 0.0           # Net buy pressure over 5 minutes
    cvd_signal: str = "NEUTRAL"   # BULLISH / BEARISH / NEUTRAL

    # Orderbook walls
    bid_walls: int = 0            # Number of large bid orders
    ask_walls: int = 0            # Number of large ask orders

    # Depth
    bid_depth_usd: float = 0.0    # Total bid liquidity within 1%
    ask_depth_usd: float = 0.0    # Total ask liquidity within 1%


@dataclass
class DirectionScore:
    """Direction scoring result."""
    up_score: int
    down_score: int
    raw_up_probability: float
    scoring_breakdown: Dict[str, str] = field(default_factory=dict)


@dataclass
class TASignal:
    """Complete TA signal with all indicators."""
    # Indicators
    vwap: Optional[float] = None
    vwap_slope: Optional[float] = None
    rsi: Optional[float] = None
    rsi_slope: Optional[float] = None
    macd: Optional[MACDResult] = None
    squeeze: Optional[TTMSqueezeResult] = None  # TTM Squeeze (BB inside KC)
    heiken_color: Optional[str] = None
    heiken_count: int = 0

    # Order Flow (OBI + CVD)
    order_flow: Optional[OrderFlowData] = None

    # EMA Cross
    ema_fast: Optional[float] = None  # EMA(5)
    ema_slow: Optional[float] = None  # EMA(20)
    ema_cross: str = "NEUTRAL"        # GOLDEN / DEATH / NEUTRAL

    # Regime
    regime: MarketRegime = MarketRegime.CHOP
    regime_reason: str = ""

    # Direction scoring
    direction: DirectionScore = field(default_factory=lambda: DirectionScore(1, 1, 0.5))

    # Time-aware probability
    model_up: float = 0.5
    model_down: float = 0.5
    time_decay: float = 1.0

    # Edge calculation
    edge_up: Optional[float] = None
    edge_down: Optional[float] = None

    # Decision
    phase: TradePhase = TradePhase.EARLY
    action: str = "NO_TRADE"
    side: Optional[str] = None
    strength: SignalStrength = SignalStrength.NONE
    reason: str = ""

    # Raw data
    current_price: float = 0.0
    price_to_beat: Optional[float] = None
    time_remaining_min: float = 15.0


class TASignalGenerator:
    """Generates TA-based trading signals for BTC 15-minute markets."""

    # Indicator parameters (from PolymarketBTC15mAssistant)
    RSI_PERIOD = 14
    RSI_MA_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    VWAP_SLOPE_LOOKBACK = 10  # minutes

    # Phase thresholds (base - adjusted by price in decide())
    PHASE_THRESHOLDS = {
        TradePhase.EARLY: {"edge": 0.05, "min_prob": 0.52},
        TradePhase.MID: {"edge": 0.05, "min_prob": 0.52},
        TradePhase.LATE: {"edge": 0.08, "min_prob": 0.55},
    }

    def __init__(self):
        self.candle_cache: Dict[str, List[Candle]] = {}
        self.vwap_series: Dict[str, List[float]] = {}
        self.rsi_series: Dict[str, List[float]] = {}

    def compute_heiken_ashi(self, candles: List[Candle]) -> List[HeikenAshiCandle]:
        """Convert regular candles to Heiken Ashi candles."""
        if not candles:
            return []

        ha_candles = []

        for i, c in enumerate(candles):
            ha_close = (c.open + c.high + c.low + c.close) / 4

            if i == 0:
                ha_open = (c.open + c.close) / 2
            else:
                prev = ha_candles[i - 1]
                ha_open = (prev.open + prev.close) / 2

            ha_high = max(c.high, ha_open, ha_close)
            ha_low = min(c.low, ha_open, ha_close)

            ha_candles.append(HeikenAshiCandle(
                open=ha_open,
                high=ha_high,
                low=ha_low,
                close=ha_close,
                is_green=ha_close >= ha_open,
                body_size=abs(ha_close - ha_open)
            ))

        return ha_candles

    def count_consecutive_color(self, ha_candles: List[HeikenAshiCandle]) -> Tuple[str, int]:
        """Count consecutive candles of same color from the end."""
        if not ha_candles:
            return ("none", 0)

        last = ha_candles[-1]
        target = "green" if last.is_green else "red"

        count = 0
        for candle in reversed(ha_candles):
            color = "green" if candle.is_green else "red"
            if color != target:
                break
            count += 1

        return (target, count)

    def compute_session_vwap(self, candles: List[Candle]) -> Optional[float]:
        """Calculate session VWAP from candles."""
        if not candles:
            return None

        pv_sum = 0.0
        v_sum = 0.0

        for c in candles:
            typical_price = (c.high + c.low + c.close) / 3
            pv_sum += typical_price * c.volume
            v_sum += c.volume

        if v_sum == 0:
            return None

        return pv_sum / v_sum

    def compute_vwap_series(self, candles: List[Candle]) -> List[Optional[float]]:
        """Calculate cumulative VWAP series."""
        series = []
        for i in range(len(candles)):
            vwap = self.compute_session_vwap(candles[:i + 1])
            series.append(vwap)
        return series

    def compute_rsi(self, closes: List[float], period: int = 14) -> Optional[float]:
        """Calculate RSI from closing prices."""
        if len(closes) < period + 1:
            return None

        gains = 0.0
        losses = 0.0

        for i in range(len(closes) - period, len(closes)):
            diff = closes[i] - closes[i - 1]
            if diff > 0:
                gains += diff
            else:
                losses += -diff

        avg_gain = gains / period
        avg_loss = losses / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return max(0, min(100, rsi))

    def _ema(self, values: List[float], period: int) -> Optional[float]:
        """Calculate exponential moving average."""
        if len(values) < period:
            return None

        k = 2 / (period + 1)
        ema = values[0]

        for i in range(1, len(values)):
            ema = values[i] * k + ema * (1 - k)

        return ema

    def compute_macd(
        self,
        closes: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Optional[MACDResult]:
        """Calculate MACD with histogram delta."""
        if len(closes) < slow + signal:
            return None

        fast_ema = self._ema(closes, fast)
        slow_ema = self._ema(closes, slow)

        if fast_ema is None or slow_ema is None:
            return None

        macd_line = fast_ema - slow_ema

        # Build MACD series for signal line
        macd_series = []
        for i in range(slow - 1, len(closes)):
            sub = closes[:i + 1]
            f = self._ema(sub, fast)
            s = self._ema(sub, slow)
            if f is not None and s is not None:
                macd_series.append(f - s)

        signal_line = self._ema(macd_series, signal)
        if signal_line is None:
            return None

        histogram = macd_line - signal_line

        # Calculate histogram delta
        hist_delta = None
        if len(macd_series) >= signal + 1:
            prev_signal = self._ema(macd_series[:-1], signal)
            if prev_signal is not None:
                prev_hist = macd_series[-2] - prev_signal
                hist_delta = histogram - prev_hist

        return MACDResult(
            macd_line=macd_line,
            signal_line=signal_line,
            histogram=histogram,
            hist_delta=hist_delta
        )

    def slope_last(self, values: List[float], points: int = 3) -> Optional[float]:
        """Calculate slope over last N points."""
        if len(values) < points:
            return None

        slice_vals = values[-points:]
        return (slice_vals[-1] - slice_vals[0]) / (points - 1)

    def compute_ttm_squeeze(
        self,
        candles: List[Candle],
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5
    ) -> Optional[TTMSqueezeResult]:
        """
        Calculate TTM Squeeze indicator.

        Squeeze = Bollinger Bands inside Keltner Channels (volatility compression)
        - Squeeze ON: BB inside KC = price compression, breakout imminent
        - Squeeze OFF (fired): BB outside KC = breakout happening
        - Momentum oscillator shows direction of breakout

        From your trading folder: TTM Source Code.txt
        """
        if len(candles) < max(bb_length, kc_length) + 5:
            return None

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]

        # Bollinger Bands: SMA + 2*StdDev
        bb_closes = closes[-bb_length:]
        bb_sma = sum(bb_closes) / bb_length
        variance = sum((x - bb_sma) ** 2 for x in bb_closes) / bb_length
        bb_std = variance ** 0.5
        bb_upper = bb_sma + bb_mult * bb_std
        bb_lower = bb_sma - bb_mult * bb_std

        # Keltner Channels: EMA + mult*ATR
        kc_ema = self._ema(closes[-kc_length:], kc_length)
        if kc_ema is None:
            return None

        # ATR calculation
        trs = []
        for i in range(len(candles) - kc_length, len(candles)):
            if i == 0:
                tr = highs[i] - lows[i]
            else:
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1])
                )
            trs.append(tr)
        atr = sum(trs) / len(trs)

        kc_upper = kc_ema + kc_mult * atr
        kc_lower = kc_ema - kc_mult * atr

        # Squeeze detection: BB inside KC
        squeeze_on = (bb_lower > kc_lower) and (bb_upper < kc_upper)

        # Calculate momentum oscillator (linearized price deviation)
        # From TTM formula: osc = linreg(close - (highest + lowest)/2 + sma, length, 0)
        period_highs = highs[-bb_length:]
        period_lows = lows[-bb_length:]
        highest = max(period_highs)
        lowest = min(period_lows)
        midline = (highest + lowest) / 2 + bb_sma

        # Calculate momentum as deviation from midline
        momentum = (closes[-1] - midline / 2) / (bb_std if bb_std > 0 else 1) * 10

        # Check if momentum is rising (last 3 values)
        momentum_values = []
        for i in range(-3, 0):
            if len(closes) + i >= bb_length:
                idx = len(closes) + i
                prev_closes = closes[idx - bb_length + 1:idx + 1]
                if len(prev_closes) == bb_length:
                    prev_sma = sum(prev_closes) / bb_length
                    prev_highest = max(highs[idx - bb_length + 1:idx + 1])
                    prev_lowest = min(lows[idx - bb_length + 1:idx + 1])
                    prev_mid = (prev_highest + prev_lowest) / 2 + prev_sma
                    prev_var = sum((x - prev_sma) ** 2 for x in prev_closes) / bb_length
                    prev_std = prev_var ** 0.5 if prev_var > 0 else 1
                    momentum_values.append((closes[idx] - prev_mid / 2) / prev_std * 10)

        momentum_rising = len(momentum_values) >= 2 and momentum_values[-1] > momentum_values[0]

        # Check previous bar for squeeze to detect "fired" state
        squeeze_fired = False
        if len(candles) >= bb_length + 1:
            prev_closes = closes[-(bb_length + 1):-1]
            prev_sma = sum(prev_closes) / bb_length
            prev_var = sum((x - prev_sma) ** 2 for x in prev_closes) / bb_length
            prev_std = prev_var ** 0.5
            prev_bb_upper = prev_sma + bb_mult * prev_std
            prev_bb_lower = prev_sma - bb_mult * prev_std
            prev_kc_ema = self._ema(closes[-(kc_length + 1):-1], kc_length)
            if prev_kc_ema:
                prev_trs = trs[:-1] if len(trs) > 1 else trs
                prev_atr = sum(prev_trs) / len(prev_trs)
                prev_kc_upper = prev_kc_ema + kc_mult * prev_atr
                prev_kc_lower = prev_kc_ema - kc_mult * prev_atr
                prev_squeeze = (prev_bb_lower > prev_kc_lower) and (prev_bb_upper < prev_kc_upper)
                squeeze_fired = prev_squeeze and not squeeze_on

        return TTMSqueezeResult(
            squeeze_on=squeeze_on,
            momentum=momentum,
            momentum_rising=momentum_rising,
            squeeze_fired=squeeze_fired
        )

    def compute_obi(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        mid_price: float,
        band_pct: float = 1.0
    ) -> Tuple[float, str]:
        """
        Calculate Order Book Imbalance (OBI).

        OBI = (bid_volume - ask_volume) / total_volume within band around mid price.
        Returns: (obi_value, signal) where obi in [-1, +1], signal is BULLISH/BEARISH/NEUTRAL
        """
        if not bids or not asks or mid_price <= 0:
            return 0.0, "NEUTRAL"

        band = mid_price * band_pct / 100

        # Sum volume within band
        bid_vol = sum(qty for price, qty in bids if price >= mid_price - band)
        ask_vol = sum(qty for price, qty in asks if price <= mid_price + band)
        total = bid_vol + ask_vol

        if total == 0:
            return 0.0, "NEUTRAL"

        obi = (bid_vol - ask_vol) / total

        # Signal thresholds (from polymarket-assistant: ±10%)
        if obi > 0.10:
            return obi, "BULLISH"
        elif obi < -0.10:
            return obi, "BEARISH"
        else:
            return obi, "NEUTRAL"

    def compute_cvd(
        self,
        trades: List[Dict],
        window_seconds: int
    ) -> float:
        """
        Calculate Cumulative Volume Delta (CVD) over time window.

        CVD = sum of (qty * price * direction) where direction is +1 for buys, -1 for sells.
        Returns net buying pressure in USD terms.
        """
        if not trades:
            return 0.0

        cutoff = time.time() - window_seconds
        cvd = 0.0

        for trade in trades:
            trade_time = trade.get("t", trade.get("timestamp", 0))
            if trade_time >= cutoff:
                qty = trade.get("qty", trade.get("quantity", 0))
                price = trade.get("price", 0)
                is_buy = trade.get("is_buy", trade.get("side") == "buy")
                direction = 1 if is_buy else -1
                cvd += qty * price * direction

        return cvd

    def detect_walls(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        multiplier: float = 5.0
    ) -> Tuple[int, int]:
        """
        Detect orderbook walls (large orders > N times average).
        Returns: (bid_walls_count, ask_walls_count)
        """
        all_volumes = [qty for _, qty in bids] + [qty for _, qty in asks]
        if not all_volumes:
            return 0, 0

        avg_vol = sum(all_volumes) / len(all_volumes)
        threshold = avg_vol * multiplier

        bid_walls = sum(1 for _, qty in bids if qty >= threshold)
        ask_walls = sum(1 for _, qty in asks if qty >= threshold)

        return bid_walls, ask_walls

    def compute_order_flow(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        trades: List[Dict],
        mid_price: float
    ) -> OrderFlowData:
        """
        Compute all order flow indicators.

        Args:
            bids: List of (price, quantity) tuples
            asks: List of (price, quantity) tuples
            trades: List of trade dicts with t, qty, price, is_buy
            mid_price: Current mid price

        Returns: OrderFlowData with OBI, CVD, walls, depth
        """
        # OBI
        obi_val, obi_signal = self.compute_obi(bids, asks, mid_price)

        # CVD at multiple windows
        cvd_1m = self.compute_cvd(trades, 60)
        cvd_5m = self.compute_cvd(trades, 300)

        # CVD signal based on 5-minute CVD
        if cvd_5m > 0:
            cvd_signal = "BULLISH"
        elif cvd_5m < 0:
            cvd_signal = "BEARISH"
        else:
            cvd_signal = "NEUTRAL"

        # Walls
        bid_walls, ask_walls = self.detect_walls(bids, asks)

        # Depth within 1% of mid
        band = mid_price * 0.01
        bid_depth = sum(p * q for p, q in bids if p >= mid_price - band)
        ask_depth = sum(p * q for p, q in asks if p <= mid_price + band)

        return OrderFlowData(
            obi=obi_val,
            obi_signal=obi_signal,
            cvd_1m=cvd_1m,
            cvd_5m=cvd_5m,
            cvd_signal=cvd_signal,
            bid_walls=bid_walls,
            ask_walls=ask_walls,
            bid_depth_usd=bid_depth,
            ask_depth_usd=ask_depth
        )

    def compute_ema_cross(
        self,
        closes: List[float],
        fast_period: int = 5,
        slow_period: int = 20
    ) -> Tuple[Optional[float], Optional[float], str]:
        """
        Calculate EMA cross signal.
        Returns: (ema_fast, ema_slow, cross_signal)
        """
        ema_fast = self._ema(closes, fast_period)
        ema_slow = self._ema(closes, slow_period)

        if ema_fast is None or ema_slow is None:
            return None, None, "NEUTRAL"

        if ema_fast > ema_slow:
            return ema_fast, ema_slow, "GOLDEN"
        elif ema_fast < ema_slow:
            return ema_fast, ema_slow, "DEATH"
        else:
            return ema_fast, ema_slow, "NEUTRAL"

    def detect_regime(
        self,
        price: float,
        vwap: Optional[float],
        vwap_slope: Optional[float],
        vwap_cross_count: Optional[int],
        volume_recent: Optional[float],
        volume_avg: Optional[float]
    ) -> Tuple[MarketRegime, str]:
        """Detect market regime."""
        if price is None or vwap is None or vwap_slope is None:
            return (MarketRegime.CHOP, "missing_inputs")

        above_vwap = price > vwap

        # Check for low volume chop
        if volume_recent is not None and volume_avg is not None:
            low_volume = volume_recent < 0.6 * volume_avg
            near_vwap = abs((price - vwap) / vwap) < 0.001
            if low_volume and near_vwap:
                return (MarketRegime.CHOP, "low_volume_flat")

        # Trend detection
        if above_vwap and vwap_slope > 0:
            return (MarketRegime.TREND_UP, "price_above_vwap_slope_up")

        if not above_vwap and vwap_slope < 0:
            return (MarketRegime.TREND_DOWN, "price_below_vwap_slope_down")

        # Range detection
        if vwap_cross_count is not None and vwap_cross_count >= 3:
            return (MarketRegime.RANGE, "frequent_vwap_cross")

        return (MarketRegime.RANGE, "default")

    def count_vwap_crosses(
        self,
        closes: List[float],
        vwap_series: List[float],
        lookback: int = 20
    ) -> Optional[int]:
        """Count VWAP crosses in lookback period."""
        if len(closes) < lookback or len(vwap_series) < lookback:
            return None

        crosses = 0
        for i in range(len(closes) - lookback + 1, len(closes)):
            prev_diff = closes[i - 1] - vwap_series[i - 1]
            curr_diff = closes[i] - vwap_series[i]

            if prev_diff == 0:
                continue

            # Detect cross
            if (prev_diff > 0 and curr_diff < 0) or (prev_diff < 0 and curr_diff > 0):
                crosses += 1

        return crosses

    def score_direction(
        self,
        price: float,
        vwap: Optional[float],
        vwap_slope: Optional[float],
        rsi: Optional[float],
        rsi_slope: Optional[float],
        macd: Optional[MACDResult],
        heiken_color: Optional[str],
        heiken_count: int,
        failed_vwap_reclaim: bool = False,
        squeeze: Optional[TTMSqueezeResult] = None,
        order_flow: Optional[OrderFlowData] = None,
        ema_cross: str = "NEUTRAL"
    ) -> DirectionScore:
        """Calculate direction score based on TA indicators + order flow."""
        up = 1
        down = 1
        breakdown = {}

        # Price vs VWAP
        if price is not None and vwap is not None:
            if price > vwap:
                up += 2
                breakdown["vwap_position"] = "above (+2 UP)"
            elif price < vwap:
                down += 2
                breakdown["vwap_position"] = "below (+2 DOWN)"

        # VWAP slope
        if vwap_slope is not None:
            if vwap_slope > 0:
                up += 2
                breakdown["vwap_slope"] = "positive (+2 UP)"
            elif vwap_slope < 0:
                down += 2
                breakdown["vwap_slope"] = "negative (+2 DOWN)"

        # RSI with slope
        if rsi is not None and rsi_slope is not None:
            if rsi > 55 and rsi_slope > 0:
                up += 2
                breakdown["rsi"] = f"{rsi:.1f} rising (+2 UP)"
            elif rsi < 45 and rsi_slope < 0:
                down += 2
                breakdown["rsi"] = f"{rsi:.1f} falling (+2 DOWN)"

        # MACD histogram
        if macd is not None:
            hist = macd.histogram
            hist_delta = macd.hist_delta

            if hist is not None and hist_delta is not None:
                # Expanding green (bullish)
                if hist > 0 and hist_delta > 0:
                    up += 2
                    breakdown["macd_hist"] = "expanding green (+2 UP)"
                # Expanding red (bearish)
                elif hist < 0 and hist_delta < 0:
                    down += 2
                    breakdown["macd_hist"] = "expanding red (+2 DOWN)"

            # MACD line direction
            if macd.macd_line > 0:
                up += 1
                breakdown["macd_line"] = "positive (+1 UP)"
            elif macd.macd_line < 0:
                down += 1
                breakdown["macd_line"] = "negative (+1 DOWN)"

        # Heiken Ashi
        if heiken_color is not None and heiken_count >= 2:
            if heiken_color == "green":
                up += 1
                breakdown["heiken"] = f"green x{heiken_count} (+1 UP)"
            elif heiken_color == "red":
                down += 1
                breakdown["heiken"] = f"red x{heiken_count} (+1 DOWN)"

        # Failed VWAP reclaim (strong bearish signal)
        if failed_vwap_reclaim:
            down += 3
            breakdown["failed_reclaim"] = "failed VWAP reclaim (+3 DOWN)"

        # TTM Squeeze (volatility compression before breakout)
        # When squeeze fires (BB exits KC), momentum direction is key
        if squeeze is not None:
            if squeeze.squeeze_fired:
                # Squeeze just released - breakout happening!
                if squeeze.momentum > 0 and squeeze.momentum_rising:
                    up += 3
                    breakdown["squeeze"] = f"FIRED bullish ({squeeze.momentum:.1f} +3 UP)"
                elif squeeze.momentum < 0 and not squeeze.momentum_rising:
                    down += 3
                    breakdown["squeeze"] = f"FIRED bearish ({squeeze.momentum:.1f} +3 DOWN)"
            elif squeeze.squeeze_on:
                # Squeeze building - use momentum to predict direction
                if squeeze.momentum > 2 and squeeze.momentum_rising:
                    up += 1
                    breakdown["squeeze"] = f"ON building bullish (+1 UP)"
                elif squeeze.momentum < -2 and not squeeze.momentum_rising:
                    down += 1
                    breakdown["squeeze"] = f"ON building bearish (+1 DOWN)"

        # ORDER FLOW: OBI + CVD + Walls (from polymarket-assistant)
        if order_flow is not None:
            # OBI - Order Book Imbalance (±1 point)
            if order_flow.obi_signal == "BULLISH":
                up += 1
                breakdown["obi"] = f"OBI {order_flow.obi:+.1%} (+1 UP)"
            elif order_flow.obi_signal == "BEARISH":
                down += 1
                breakdown["obi"] = f"OBI {order_flow.obi:+.1%} (+1 DOWN)"

            # CVD - Cumulative Volume Delta (±1 point)
            if order_flow.cvd_signal == "BULLISH":
                up += 1
                breakdown["cvd"] = f"CVD buy pressure (+1 UP)"
            elif order_flow.cvd_signal == "BEARISH":
                down += 1
                breakdown["cvd"] = f"CVD sell pressure (+1 DOWN)"

            # Orderbook Walls (±2 points max)
            wall_score = min(order_flow.bid_walls, 2) - min(order_flow.ask_walls, 2)
            if wall_score > 0:
                up += wall_score
                breakdown["walls"] = f"{order_flow.bid_walls} bid walls (+{wall_score} UP)"
            elif wall_score < 0:
                down += abs(wall_score)
                breakdown["walls"] = f"{order_flow.ask_walls} ask walls (+{abs(wall_score)} DOWN)"

        # EMA Cross (±1 point)
        if ema_cross == "GOLDEN":
            up += 1
            breakdown["ema_cross"] = "golden cross (+1 UP)"
        elif ema_cross == "DEATH":
            down += 1
            breakdown["ema_cross"] = "death cross (+1 DOWN)"

        raw_up = up / (up + down)

        return DirectionScore(
            up_score=up,
            down_score=down,
            raw_up_probability=raw_up,
            scoring_breakdown=breakdown
        )

    def apply_time_awareness(
        self,
        raw_up: float,
        remaining_minutes: float,
        window_minutes: float = 15.0
    ) -> Tuple[float, float, float]:
        """Apply time awareness to probability estimate.
        Less time remaining = MORE confident (price action is clearer),
        not less. Use sqrt curve so signal strength grows as expiry nears.
        """
        time_ratio = max(0, min(1, remaining_minutes / window_minutes))
        # Invert: closer to expiry = stronger signal (sqrt for smooth curve)
        time_boost = max(0.75, 1.0 - (time_ratio * 0.25))  # 0.75 at 15min, 1.0 at 0min
        adjusted_up = max(0, min(1, 0.5 + (raw_up - 0.5) * time_boost))
        adjusted_down = 1 - adjusted_up

        return (time_boost, adjusted_up, adjusted_down)

    def compute_edge(
        self,
        model_up: float,
        model_down: float,
        market_yes: Optional[float],
        market_no: Optional[float]
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Calculate edge vs market prices."""
        if market_yes is None or market_no is None:
            return (None, None, None, None)

        total = market_yes + market_no
        if total <= 0:
            return (None, None, None, None)

        # Normalize market prices
        market_up = market_yes / total
        market_down = market_no / total

        # Calculate edge
        edge_up = model_up - market_up
        edge_down = model_down - market_down

        return (market_up, market_down, edge_up, edge_down)

    def decide(
        self,
        remaining_minutes: float,
        edge_up: Optional[float],
        edge_down: Optional[float],
        model_up: Optional[float] = None,
        model_down: Optional[float] = None,
        market_yes: Optional[float] = None,
        market_no: Optional[float] = None
    ) -> Tuple[str, Optional[str], TradePhase, str, SignalStrength]:
        """Make trading decision based on edge and phase.

        Price-aware thresholds: cheap entries (< $0.30) have massive payoff
        asymmetry, so we lower the probability bar. At $0.20, payoff is 5:1
        so even 25% probability is +EV.
        """
        # Determine phase
        if remaining_minutes > 10:
            phase = TradePhase.EARLY
        elif remaining_minutes > 5:
            phase = TradePhase.MID
        else:
            phase = TradePhase.LATE

        thresholds = self.PHASE_THRESHOLDS[phase]

        if edge_up is None or edge_down is None:
            return ("NO_TRADE", None, phase, "missing_market_data", SignalStrength.NONE)

        # Find best side
        best_side = "UP" if edge_up > edge_down else "DOWN"
        best_edge = edge_up if best_side == "UP" else edge_down
        best_model = model_up if best_side == "UP" else model_down

        # Price-aware probability threshold adjustment
        # Cheap entries have asymmetric payoff - lower the bar
        entry_price = market_yes if best_side == "UP" else market_no
        min_prob = thresholds["min_prob"]
        if entry_price is not None:
            if entry_price < 0.20:
                min_prob = 0.40  # 5:1 payoff, even 40% prob is hugely +EV
            elif entry_price < 0.30:
                min_prob = 0.45  # 3.3:1 payoff, 45% prob = strong edge
            elif entry_price < 0.40:
                min_prob = 0.48  # 2.5:1 payoff, slightly lower bar

        # Check edge threshold
        if best_edge < thresholds["edge"]:
            return ("NO_TRADE", None, phase, f"edge_below_{thresholds['edge']}", SignalStrength.NONE)

        # Check probability threshold (price-adjusted)
        if best_model is not None and best_model < min_prob:
            return ("NO_TRADE", None, phase, f"prob_below_{min_prob:.2f}", SignalStrength.NONE)

        # Determine strength
        if best_edge >= 0.20:
            strength = SignalStrength.STRONG
        elif best_edge >= 0.10:
            strength = SignalStrength.GOOD
        else:
            strength = SignalStrength.OPTIONAL

        return ("ENTER", best_side, phase, "signal_valid", strength)

    def generate_signal(
        self,
        market_id: str,
        candles: List[Candle],
        current_price: float,
        market_yes_price: Optional[float],
        market_no_price: Optional[float],
        time_remaining_min: float = 15.0,
        price_to_beat: Optional[float] = None,
        bids: Optional[List[Tuple[float, float]]] = None,
        asks: Optional[List[Tuple[float, float]]] = None,
        trades: Optional[List[Dict]] = None
    ) -> TASignal:
        """Generate complete TA signal for a market.

        Args:
            market_id: Market identifier
            candles: OHLCV candle data
            current_price: Current BTC price
            market_yes_price: Polymarket UP price
            market_no_price: Polymarket DOWN price
            time_remaining_min: Minutes until market expires
            price_to_beat: Reference price for comparison
            bids: Optional orderbook bids [(price, qty), ...]
            asks: Optional orderbook asks [(price, qty), ...]
            trades: Optional recent trades [{t, qty, price, is_buy}, ...]
        """
        signal = TASignal()
        signal.current_price = current_price
        signal.price_to_beat = price_to_beat
        signal.time_remaining_min = time_remaining_min

        if not candles or len(candles) < 5:
            signal.reason = "insufficient_data"
            return signal

        closes = [c.close for c in candles]

        # Calculate VWAP
        vwap_series = self.compute_vwap_series(candles)
        signal.vwap = vwap_series[-1] if vwap_series else None

        # VWAP slope
        if len(vwap_series) >= self.VWAP_SLOPE_LOOKBACK:
            valid_vwaps = [v for v in vwap_series if v is not None]
            if len(valid_vwaps) >= self.VWAP_SLOPE_LOOKBACK:
                signal.vwap_slope = self.slope_last(valid_vwaps, self.VWAP_SLOPE_LOOKBACK)

        # Calculate RSI
        signal.rsi = self.compute_rsi(closes, self.RSI_PERIOD)

        # RSI series and slope
        rsi_series = []
        for i in range(self.RSI_PERIOD, len(closes)):
            r = self.compute_rsi(closes[:i + 1], self.RSI_PERIOD)
            if r is not None:
                rsi_series.append(r)

        if len(rsi_series) >= 3:
            signal.rsi_slope = self.slope_last(rsi_series, 3)

        # Calculate MACD
        signal.macd = self.compute_macd(
            closes,
            self.MACD_FAST,
            self.MACD_SLOW,
            self.MACD_SIGNAL
        )

        # Calculate TTM Squeeze (volatility compression before breakout)
        signal.squeeze = self.compute_ttm_squeeze(candles)

        # Calculate EMA Cross (5/20)
        signal.ema_fast, signal.ema_slow, signal.ema_cross = self.compute_ema_cross(closes, 5, 20)

        # Compute Order Flow (OBI + CVD) if orderbook data provided
        if bids is not None and asks is not None:
            signal.order_flow = self.compute_order_flow(
                bids=bids,
                asks=asks,
                trades=trades or [],
                mid_price=current_price
            )

        # Heiken Ashi
        ha_candles = self.compute_heiken_ashi(candles)
        if ha_candles:
            color, count = self.count_consecutive_color(ha_candles)
            signal.heiken_color = color
            signal.heiken_count = count

        # VWAP cross count
        vwap_cross_count = None
        if signal.vwap is not None and vwap_series:
            valid_vwaps = [v if v is not None else closes[i] for i, v in enumerate(vwap_series)]
            vwap_cross_count = self.count_vwap_crosses(closes, valid_vwaps, 20)

        # Volume for regime detection
        recent_volume = sum(c.volume for c in candles[-20:]) if len(candles) >= 20 else None
        avg_volume = sum(c.volume for c in candles[-120:]) / 6 if len(candles) >= 120 else None

        # Detect failed VWAP reclaim
        failed_vwap_reclaim = False
        if signal.vwap is not None and len(closes) >= 3 and len(vwap_series) >= 3:
            curr_below = closes[-1] < signal.vwap
            prev_vwap = vwap_series[-2] if vwap_series[-2] else signal.vwap
            prev_above = closes[-2] > prev_vwap
            failed_vwap_reclaim = curr_below and prev_above

        # Detect regime
        signal.regime, signal.regime_reason = self.detect_regime(
            price=current_price,
            vwap=signal.vwap,
            vwap_slope=signal.vwap_slope,
            vwap_cross_count=vwap_cross_count,
            volume_recent=recent_volume,
            volume_avg=avg_volume
        )

        # Score direction (including order flow if available)
        signal.direction = self.score_direction(
            price=current_price,
            vwap=signal.vwap,
            vwap_slope=signal.vwap_slope,
            rsi=signal.rsi,
            rsi_slope=signal.rsi_slope,
            macd=signal.macd,
            heiken_color=signal.heiken_color,
            heiken_count=signal.heiken_count,
            failed_vwap_reclaim=failed_vwap_reclaim,
            squeeze=signal.squeeze,
            order_flow=signal.order_flow,
            ema_cross=signal.ema_cross
        )

        # Apply time awareness
        signal.time_decay, signal.model_up, signal.model_down = self.apply_time_awareness(
            signal.direction.raw_up_probability,
            time_remaining_min,
            15.0  # 15-minute window
        )

        # Compute edge
        _, _, signal.edge_up, signal.edge_down = self.compute_edge(
            signal.model_up,
            signal.model_down,
            market_yes_price,
            market_no_price
        )

        # Make decision (pass market prices for price-aware thresholds)
        signal.action, signal.side, signal.phase, signal.reason, signal.strength = self.decide(
            time_remaining_min,
            signal.edge_up,
            signal.edge_down,
            signal.model_up,
            signal.model_down,
            market_yes=market_yes_price,
            market_no=market_no_price
        )

        return signal

    def format_signal_summary(self, signal: TASignal) -> str:
        """Format signal for display."""
        lines = []

        lines.append("=" * 60)
        lines.append("TA SIGNAL SUMMARY")
        lines.append("=" * 60)

        # Indicators
        lines.append(f"VWAP: ${signal.vwap:,.2f}" if signal.vwap else "VWAP: N/A")
        lines.append(f"  Slope: {'UP' if signal.vwap_slope and signal.vwap_slope > 0 else 'DOWN' if signal.vwap_slope else 'FLAT'}")
        lines.append(f"RSI: {signal.rsi:.1f}" if signal.rsi else "RSI: N/A")
        lines.append(f"  Slope: {'UP' if signal.rsi_slope and signal.rsi_slope > 0 else 'DOWN' if signal.rsi_slope else 'FLAT'}")

        if signal.macd:
            hist_status = "expanding" if signal.macd.hist_delta and signal.macd.hist_delta * signal.macd.histogram > 0 else "contracting"
            direction = "bullish" if signal.macd.histogram > 0 else "bearish"
            lines.append(f"MACD: {direction} ({hist_status})")

        # TTM Squeeze
        if signal.squeeze:
            sq_status = "ON (compression)" if signal.squeeze.squeeze_on else "OFF"
            if signal.squeeze.squeeze_fired:
                sq_status = "FIRED!"
            mom_dir = "rising" if signal.squeeze.momentum_rising else "falling"
            lines.append(f"Squeeze: {sq_status} | Mom: {signal.squeeze.momentum:.1f} ({mom_dir})")

        lines.append(f"Heiken: {signal.heiken_color or 'N/A'} x{signal.heiken_count}")
        lines.append(f"Regime: {signal.regime.value} ({signal.regime_reason})")

        lines.append("-" * 60)

        # Scoring breakdown
        lines.append("Direction Score:")
        lines.append(f"  UP: {signal.direction.up_score} | DOWN: {signal.direction.down_score}")
        lines.append(f"  Raw: {signal.direction.raw_up_probability:.1%} UP")

        for indicator, desc in signal.direction.scoring_breakdown.items():
            lines.append(f"    {indicator}: {desc}")

        lines.append("-" * 60)

        # Model probability
        lines.append(f"Model: {signal.model_up:.1%} UP / {signal.model_down:.1%} DOWN")
        lines.append(f"Time Decay: {signal.time_decay:.2f}")

        if signal.edge_up is not None:
            lines.append(f"Edge: UP {signal.edge_up:+.1%} | DOWN {signal.edge_down:+.1%}")

        lines.append("-" * 60)

        # Decision
        lines.append(f"Phase: {signal.phase.value.upper()}")
        lines.append(f"Action: {signal.action}")
        if signal.side:
            lines.append(f"Side: {signal.side}")
            lines.append(f"Strength: {signal.strength.value.upper()}")
        lines.append(f"Reason: {signal.reason}")

        lines.append("=" * 60)

        return "\n".join(lines)


# Global instance
ta_signal_generator = TASignalGenerator()


def generate_btc_15m_signal(
    market_id: str,
    candles: List[Dict],
    current_price: float,
    market_yes_price: Optional[float],
    market_no_price: Optional[float],
    time_remaining_min: float = 15.0,
    price_to_beat: Optional[float] = None
) -> TASignal:
    """Convenience function to generate TA signal."""
    # Convert dict candles to Candle objects
    candle_objects = []
    for c in candles:
        candle_objects.append(Candle(
            timestamp=c.get("timestamp", 0),
            open=float(c.get("open", 0)),
            high=float(c.get("high", 0)),
            low=float(c.get("low", 0)),
            close=float(c.get("close", 0)),
            volume=float(c.get("volume", 0))
        ))

    return ta_signal_generator.generate_signal(
        market_id=market_id,
        candles=candle_objects,
        current_price=current_price,
        market_yes_price=market_yes_price,
        market_no_price=market_no_price,
        time_remaining_min=time_remaining_min,
        price_to_beat=price_to_beat
    )
