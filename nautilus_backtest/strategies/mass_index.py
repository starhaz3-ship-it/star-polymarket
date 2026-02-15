"""
MASS_INDEX strategy -- Volatility reversal bulge detector.

Processes BTC 1-min bars.  The Mass Index (Donald Dorsey) measures the narrowing
and widening of the range between high and low prices.  It sums the ratio of
two EMAs of the high-low range over 25 bars:

    Single EMA = EMA(high - low, 9)
    Double EMA = EMA(Single EMA, 9)
    Mass Index = sum of (Single EMA / Double EMA) over 25 bars

A "reversal bulge" occurs when the MI rises above 27 (volatility expansion)
then falls below 26.5 (compression starting).  This signals that a trend
reversal is imminent.

After the bulge, EMA(9) vs EMA(18) crossover determines direction.
RSI(14) must be between 35-65 (the reversal hasn't fully played out yet).
"""
from collections import deque
import math


def calc_ema(prev_ema, new_value, period):
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


class MassIndex:
    """Mass Index reversal bulge with EMA crossover direction.

    Signal fires when:
    1. Mass Index rises above 27 then drops below 26.5 ("reversal bulge")
    2. After bulge: EMA(9) > EMA(18) --> UP, else --> DOWN
    3. RSI(14) between 35-65 (reversal hasn't happened yet)

    Detects volatility expansion/compression reversals.
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"MASS_INDEX_{horizon_bars}m"

        # ---- Mass Index params ----
        self.ema_range_period = params.get("ema_range_period", 9)
        self.mi_sum_period = params.get("mi_sum_period", 25)
        self.mi_bulge_hi = params.get("mi_bulge_hi", 27.0)
        self.mi_bulge_lo = params.get("mi_bulge_lo", 26.5)

        # EMA of (high - low)
        self._single_ema = None
        self._single_ema_count = 0

        # EMA of single_ema (double smoothing)
        self._double_ema = None
        self._double_ema_count = 0

        # Rolling ratio sum for MI
        self._ratio_hist = deque(maxlen=self.mi_sum_period)
        self._mass_index = None

        # Bulge state machine
        self._saw_above_27 = False
        self._bulge_fired = False  # True once MI crosses below 26.5 after being above 27

        # ---- EMA(9) and EMA(18) for direction ----
        self.ema_fast_period = params.get("ema_fast_period", 9)
        self.ema_slow_period = params.get("ema_slow_period", 18)
        self._ema_fast = None
        self._ema_fast_count = 0
        self._ema_slow = None
        self._ema_slow_count = 0

        # Seed buffers for initial SMA calculation
        self._ema_fast_seed = deque(maxlen=self.ema_fast_period)
        self._ema_slow_seed = deque(maxlen=self.ema_slow_period)
        self._single_ema_seed = deque(maxlen=self.ema_range_period)
        self._double_ema_seed = deque(maxlen=self.ema_range_period)

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_lo = params.get("rsi_lo", 35.0)
        self.rsi_hi = params.get("rsi_hi", 65.0)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._rsi_value = 50.0

        # ---- State ----
        self._prev_close = None
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = (self.ema_range_period * 2 + self.mi_sum_period +
                        max(self.ema_slow_period, self.rsi_period) + 5)

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one 1-minute bar. Returns (direction|None, confidence)."""
        self._bar_count += 1

        # Update all indicators
        self._update_mass_index(high, low)
        self._update_emas(close)
        self._update_rsi(close)

        # Save prev_close after RSI update
        self._prev_close = close

        # Warmup check
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # Need all indicators ready
        if (self._mass_index is None or self._ema_fast is None or
                self._ema_slow is None):
            return (None, 0.0)

        # ---- Bulge state machine ----
        if self._mass_index > self.mi_bulge_hi:
            self._saw_above_27 = True
            self._bulge_fired = False

        if self._saw_above_27 and self._mass_index < self.mi_bulge_lo:
            self._bulge_fired = True
            self._saw_above_27 = False  # reset for next bulge

        # No signal without a bulge
        if not self._bulge_fired:
            return (None, 0.0)

        # Consume the bulge (one signal per bulge)
        self._bulge_fired = False

        # Cooldown
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # RSI filter: must be in neutral zone (reversal hasn't happened yet)
        if self._rsi_value < self.rsi_lo or self._rsi_value > self.rsi_hi:
            return (None, 0.0)

        # Direction from EMA crossover
        if self._ema_fast > self._ema_slow:
            direction = "UP"
        else:
            direction = "DOWN"

        # ---- Confidence calculation ----
        confidence = 0.72

        # How high the MI peaked (stronger bulge = more energy released)
        # We can approximate by how much above 27 we were recently
        # Since we just detected the bulge, use the MI value context
        # MI closer to 26.5 from higher = stronger signal
        mi_bonus = 0.03  # base bonus for any valid bulge
        confidence += mi_bonus

        # EMA separation: wider gap = clearer trend direction
        if close > 0:
            ema_gap_pct = abs(self._ema_fast - self._ema_slow) / close
            ema_bonus = min(0.06, ema_gap_pct * 40.0)
            confidence += ema_bonus

        # RSI neutrality bonus: closer to 50 = more room for reversal
        rsi_dist_from_50 = abs(self._rsi_value - 50.0)
        if rsi_dist_from_50 < 10.0:
            confidence += 0.04  # very neutral RSI
        elif rsi_dist_from_50 < 15.0:
            confidence += 0.02

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_mass_index(self, high, low):
        hl_range = high - low

        # Single EMA of range
        self._single_ema_seed.append(hl_range)
        self._single_ema_count += 1

        if self._single_ema is None:
            if len(self._single_ema_seed) >= self.ema_range_period:
                self._single_ema = sum(self._single_ema_seed) / self.ema_range_period
        else:
            self._single_ema = calc_ema(self._single_ema, hl_range, self.ema_range_period)

        if self._single_ema is None:
            return

        # Double EMA (EMA of single EMA)
        self._double_ema_seed.append(self._single_ema)
        self._double_ema_count += 1

        if self._double_ema is None:
            if len(self._double_ema_seed) >= self.ema_range_period:
                self._double_ema = sum(self._double_ema_seed) / self.ema_range_period
        else:
            self._double_ema = calc_ema(self._double_ema, self._single_ema, self.ema_range_period)

        if self._double_ema is None or self._double_ema < 1e-10:
            return

        # Ratio for this bar
        ratio = self._single_ema / self._double_ema
        self._ratio_hist.append(ratio)

        # Mass Index = sum of ratios over 25 bars
        if len(self._ratio_hist) >= self.mi_sum_period:
            self._mass_index = sum(self._ratio_hist)

    def _update_emas(self, close):
        # EMA(9) fast
        self._ema_fast_seed.append(close)
        self._ema_fast_count += 1
        if self._ema_fast is None:
            if len(self._ema_fast_seed) >= self.ema_fast_period:
                self._ema_fast = sum(self._ema_fast_seed) / self.ema_fast_period
        else:
            self._ema_fast = calc_ema(self._ema_fast, close, self.ema_fast_period)

        # EMA(18) slow
        self._ema_slow_seed.append(close)
        self._ema_slow_count += 1
        if self._ema_slow is None:
            if len(self._ema_slow_seed) >= self.ema_slow_period:
                self._ema_slow = sum(self._ema_slow_seed) / self.ema_slow_period
        else:
            self._ema_slow = calc_ema(self._ema_slow, close, self.ema_slow_period)

    def _update_rsi(self, close):
        if self._prev_close is None:
            return
        delta = close - self._prev_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        self._rsi_count += 1

        if self._rsi_count <= self.rsi_period:
            self._rsi_avg_gain += gain / self.rsi_period
            self._rsi_avg_loss += loss / self.rsi_period
            if self._rsi_count == self.rsi_period:
                rs = self._rsi_avg_gain / max(self._rsi_avg_loss, 1e-10)
                self._rsi_value = 100.0 - (100.0 / (1.0 + rs))
        else:
            self._rsi_avg_gain = (self._rsi_avg_gain * (self.rsi_period - 1) + gain) / self.rsi_period
            self._rsi_avg_loss = (self._rsi_avg_loss * (self.rsi_period - 1) + loss) / self.rsi_period
            rs = self._rsi_avg_gain / max(self._rsi_avg_loss, 1e-10)
            self._rsi_value = 100.0 - (100.0 / (1.0 + rs))
