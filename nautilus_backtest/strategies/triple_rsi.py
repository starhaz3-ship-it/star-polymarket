"""
TRIPLE_RSI strategy -- Triple-timeframe RSI agreement for high-conviction reversals.

Processes BTC 1-min bars.  Computes three RSI indicators at different lookback
periods: RSI(7) fast, RSI(14) medium, RSI(28) slow.  A signal fires ONLY when
all three RSI values agree on an extreme condition (all < 30 oversold for UP,
all > 70 overbought for DOWN).  This multi-timeframe unanimity filter is very
selective but produces high-confidence signals when all periods align.

Volume must exceed 1.3x EMA(20) to confirm genuine participation behind the
extreme readings.
"""
from collections import deque
import math


def calc_ema(prev_ema, new_value, period):
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


class _RsiTracker:
    """Incremental RSI calculator for a single period."""

    def __init__(self, period):
        self.period = period
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._count = 0
        self.value = 50.0

    def update(self, prev_close, close):
        if prev_close is None:
            return
        delta = close - prev_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        self._count += 1

        if self._count <= self.period:
            self._avg_gain += gain / self.period
            self._avg_loss += loss / self.period
            if self._count == self.period:
                rs = self._avg_gain / max(self._avg_loss, 1e-10)
                self.value = 100.0 - (100.0 / (1.0 + rs))
        else:
            self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
            self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period
            rs = self._avg_gain / max(self._avg_loss, 1e-10)
            self.value = 100.0 - (100.0 / (1.0 + rs))

    @property
    def ready(self):
        return self._count >= self.period


class TripleRsi:
    """Triple-timeframe RSI agreement: all three must be oversold/overbought.

    Signal fires when:
    1. RSI(7), RSI(14), RSI(28) ALL < 30 --> UP (oversold on every timeframe)
    2. RSI(7), RSI(14), RSI(28) ALL > 70 --> DOWN (overbought on every timeframe)
    3. Volume > 1.3x EMA(20) for confirmation

    Very selective, high conviction when it fires.
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"TRIPLE_RSI_{horizon_bars}m"

        # ---- Three RSI trackers ----
        self.rsi_fast_period = params.get("rsi_fast_period", 7)
        self.rsi_med_period = params.get("rsi_med_period", 14)
        self.rsi_slow_period = params.get("rsi_slow_period", 28)

        self._rsi_fast = _RsiTracker(self.rsi_fast_period)
        self._rsi_med = _RsiTracker(self.rsi_med_period)
        self._rsi_slow = _RsiTracker(self.rsi_slow_period)

        # ---- RSI thresholds ----
        self.oversold = params.get("oversold", 30.0)
        self.overbought = params.get("overbought", 70.0)

        # ---- Volume EMA(20) ----
        self.vol_ema_period = params.get("vol_ema_period", 20)
        self.vol_spike_mult = params.get("vol_spike_mult", 1.3)
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=self.vol_ema_period)

        # ---- State ----
        self._prev_close = None
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = self.rsi_slow_period + self.vol_ema_period + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one 1-minute bar. Returns (direction|None, confidence)."""
        self._bar_count += 1

        # Update all three RSIs
        self._rsi_fast.update(self._prev_close, close)
        self._rsi_med.update(self._prev_close, close)
        self._rsi_slow.update(self._prev_close, close)

        # Update volume EMA
        self._update_volume(volume)

        # Warmup check
        if self._bar_count < self._warmup:
            self._prev_close = close
            return (None, 0.0)

        # All three RSIs must be ready
        if not (self._rsi_fast.ready and self._rsi_med.ready and self._rsi_slow.ready):
            self._prev_close = close
            return (None, 0.0)

        # Volume filter
        if self._vol_ema is None or self._vol_ema < 1e-10:
            self._prev_close = close
            return (None, 0.0)

        vol_ratio = volume / self._vol_ema
        if vol_ratio < self.vol_spike_mult:
            self._prev_close = close
            return (None, 0.0)

        # Cooldown
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            self._prev_close = close
            return (None, 0.0)

        rsi_f = self._rsi_fast.value
        rsi_m = self._rsi_med.value
        rsi_s = self._rsi_slow.value

        direction = None

        # All three oversold --> UP
        if rsi_f < self.oversold and rsi_m < self.oversold and rsi_s < self.oversold:
            direction = "UP"

        # All three overbought --> DOWN
        elif rsi_f > self.overbought and rsi_m > self.overbought and rsi_s > self.overbought:
            direction = "DOWN"

        if direction is None:
            self._prev_close = close
            return (None, 0.0)

        # ---- Confidence calculation ----
        confidence = 0.78

        # Depth of extremes: how far past thresholds
        if direction == "UP":
            avg_depth = ((self.oversold - rsi_f) + (self.oversold - rsi_m) + (self.oversold - rsi_s)) / 3.0
        else:
            avg_depth = ((rsi_f - self.overbought) + (rsi_m - self.overbought) + (rsi_s - self.overbought)) / 3.0

        # Depth bonus: deeper = more oversold/overbought = higher confidence
        depth_bonus = min(0.08, avg_depth * 0.006)
        confidence += depth_bonus

        # Volume spike bonus
        vol_bonus = min(0.04, (vol_ratio - self.vol_spike_mult) * 0.015)
        confidence += vol_bonus

        # Agreement tightness bonus: if all three RSIs are close together
        rsi_spread = max(rsi_f, rsi_m, rsi_s) - min(rsi_f, rsi_m, rsi_s)
        if rsi_spread < 8.0:
            confidence += 0.03  # very tight agreement

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        self._prev_close = close
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_volume(self, volume):
        self._vol_ema_seed.append(volume)
        if self._vol_ema is None:
            if len(self._vol_ema_seed) >= self.vol_ema_period:
                self._vol_ema = sum(self._vol_ema_seed) / self.vol_ema_period
        else:
            self._vol_ema = calc_ema(self._vol_ema, volume, self.vol_ema_period)
