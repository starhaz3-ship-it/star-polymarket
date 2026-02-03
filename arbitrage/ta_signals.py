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
    heiken_color: Optional[str] = None
    heiken_count: int = 0

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

    # Phase thresholds
    PHASE_THRESHOLDS = {
        TradePhase.EARLY: {"edge": 0.05, "min_prob": 0.55},
        TradePhase.MID: {"edge": 0.10, "min_prob": 0.60},
        TradePhase.LATE: {"edge": 0.20, "min_prob": 0.65},
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
        failed_vwap_reclaim: bool = False
    ) -> DirectionScore:
        """Calculate direction score based on TA indicators."""
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
        """Apply time decay to probability estimate."""
        time_decay = max(0, min(1, remaining_minutes / window_minutes))
        adjusted_up = max(0, min(1, 0.5 + (raw_up - 0.5) * time_decay))
        adjusted_down = 1 - adjusted_up

        return (time_decay, adjusted_up, adjusted_down)

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
        model_down: Optional[float] = None
    ) -> Tuple[str, Optional[str], TradePhase, str, SignalStrength]:
        """Make trading decision based on edge and phase."""
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

        # Check edge threshold
        if best_edge < thresholds["edge"]:
            return ("NO_TRADE", None, phase, f"edge_below_{thresholds['edge']}", SignalStrength.NONE)

        # Check probability threshold
        if best_model is not None and best_model < thresholds["min_prob"]:
            return ("NO_TRADE", None, phase, f"prob_below_{thresholds['min_prob']}", SignalStrength.NONE)

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
        price_to_beat: Optional[float] = None
    ) -> TASignal:
        """Generate complete TA signal for a market."""
        signal = TASignal()
        signal.current_price = current_price
        signal.price_to_beat = price_to_beat
        signal.time_remaining_min = time_remaining_min

        if not candles or len(candles) < 30:
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

        # Score direction
        signal.direction = self.score_direction(
            price=current_price,
            vwap=signal.vwap,
            vwap_slope=signal.vwap_slope,
            rsi=signal.rsi,
            rsi_slope=signal.rsi_slope,
            macd=signal.macd,
            heiken_color=signal.heiken_color,
            heiken_count=signal.heiken_count,
            failed_vwap_reclaim=failed_vwap_reclaim
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

        # Make decision
        signal.action, signal.side, signal.phase, signal.reason, signal.strength = self.decide(
            time_remaining_min,
            signal.edge_up,
            signal.edge_down,
            signal.model_up,
            signal.model_down
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
