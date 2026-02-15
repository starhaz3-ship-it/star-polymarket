"""
ULTIMATE_OSC strategy -- Ultimate Oscillator multi-period momentum reversal.

Processes BTC 1-min bars.  The Ultimate Oscillator (Larry Williams) combines
buying pressure across three periods (7, 14, 28) with a weighted formula that
reduces false divergences inherent in single-period oscillators.

    BP = close - min(low, prev_close)
    TR = max(high, prev_close) - min(low, prev_close)
    UO = 100 * (4 * avg7(BP/TR) + 2 * avg14(BP/TR) + 1 * avg28(BP/TR)) / 7

When UO drops below 25 and turns up, buying pressure is returning after an
extreme sell-off --> UP.  When UO rises above 75 and turns down --> DOWN.

Volume > 1.2x EMA(20) required for confirmation.
"""
from collections import deque
import math


def calc_ema(prev_ema, new_value, period):
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


class UltimateOsc:
    """Ultimate Oscillator(7,14,28) reversal with volume confirmation.

    Signal fires when:
    1. UO < 25 AND UO > prev_UO (turning up) --> UP
    2. UO > 75 AND UO < prev_UO (turning down) --> DOWN
    3. Volume > 1.2x EMA(20)

    The triple-period weighting (4x short, 2x medium, 1x long) makes the UO
    more responsive than single-period oscillators while still filtering noise.
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"ULTIMATE_OSC_{horizon_bars}m"

        # ---- UO periods ----
        self.uo_short = params.get("uo_short", 7)
        self.uo_med = params.get("uo_med", 14)
        self.uo_long = params.get("uo_long", 28)

        # ---- UO thresholds ----
        self.uo_oversold = params.get("uo_oversold", 25.0)
        self.uo_overbought = params.get("uo_overbought", 75.0)

        # ---- BP and TR running sums for each period ----
        self._bp_short = deque(maxlen=self.uo_short)
        self._tr_short = deque(maxlen=self.uo_short)
        self._bp_med = deque(maxlen=self.uo_med)
        self._tr_med = deque(maxlen=self.uo_med)
        self._bp_long = deque(maxlen=self.uo_long)
        self._tr_long = deque(maxlen=self.uo_long)

        self._uo = None
        self._prev_uo = None

        # ---- Volume EMA(20) ----
        self.vol_ema_period = params.get("vol_ema_period", 20)
        self.vol_spike_mult = params.get("vol_spike_mult", 1.2)
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=self.vol_ema_period)

        # ---- State ----
        self._prev_close = None
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = self.uo_long + self.vol_ema_period + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one 1-minute bar. Returns (direction|None, confidence)."""
        self._bar_count += 1

        # Update UO components
        self._update_uo(high, low, close)

        # Update volume EMA
        self._update_volume(volume)

        # Save prev_close for next bar
        prev_close_for_save = self._prev_close

        # Warmup check
        if self._bar_count < self._warmup:
            self._prev_close = close
            return (None, 0.0)

        # Need UO ready with history
        if self._uo is None or self._prev_uo is None:
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

        direction = None

        # UO oversold and turning up
        if self._uo < self.uo_oversold and self._uo > self._prev_uo:
            direction = "UP"

        # UO overbought and turning down
        elif self._uo > self.uo_overbought and self._uo < self._prev_uo:
            direction = "DOWN"

        if direction is None:
            self._prev_close = close
            return (None, 0.0)

        # ---- Confidence calculation ----
        confidence = 0.72

        # UO depth bonus: further into extreme = higher conviction
        if direction == "UP":
            depth = self.uo_oversold - self._uo
        else:
            depth = self._uo - self.uo_overbought

        depth_bonus = min(0.08, max(0.0, depth * 0.005))
        confidence += depth_bonus

        # Turn strength: magnitude of UO reversal
        turn_size = abs(self._uo - self._prev_uo)
        turn_bonus = min(0.06, turn_size * 0.004)
        confidence += turn_bonus

        # Volume spike bonus
        vol_bonus = min(0.04, (vol_ratio - self.vol_spike_mult) * 0.02)
        confidence += max(0.0, vol_bonus)

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        self._prev_close = close
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_uo(self, high, low, close):
        if self._prev_close is None:
            self._prev_close = close
            return

        # Buying Pressure and True Range
        bp = close - min(low, self._prev_close)
        tr = max(high, self._prev_close) - min(low, self._prev_close)

        # Avoid division by zero in TR
        if tr < 1e-10:
            tr = 1e-10

        self._bp_short.append(bp)
        self._tr_short.append(tr)
        self._bp_med.append(bp)
        self._tr_med.append(tr)
        self._bp_long.append(bp)
        self._tr_long.append(tr)

        # Need all periods filled
        if (len(self._bp_short) < self.uo_short or
                len(self._bp_med) < self.uo_med or
                len(self._bp_long) < self.uo_long):
            self._prev_close = close
            return

        # Average ratios for each period
        sum_bp_short = sum(self._bp_short)
        sum_tr_short = sum(self._tr_short)
        sum_bp_med = sum(self._bp_med)
        sum_tr_med = sum(self._tr_med)
        sum_bp_long = sum(self._bp_long)
        sum_tr_long = sum(self._tr_long)

        avg_short = sum_bp_short / max(sum_tr_short, 1e-10)
        avg_med = sum_bp_med / max(sum_tr_med, 1e-10)
        avg_long = sum_bp_long / max(sum_tr_long, 1e-10)

        # UO = 100 * (4*short + 2*med + 1*long) / 7
        new_uo = 100.0 * (4.0 * avg_short + 2.0 * avg_med + 1.0 * avg_long) / 7.0

        self._prev_uo = self._uo
        self._uo = new_uo

        self._prev_close = close

    def _update_volume(self, volume):
        self._vol_ema_seed.append(volume)
        if self._vol_ema is None:
            if len(self._vol_ema_seed) >= self.vol_ema_period:
                self._vol_ema = sum(self._vol_ema_seed) / self.vol_ema_period
        else:
            self._vol_ema = calc_ema(self._vol_ema, volume, self.vol_ema_period)
