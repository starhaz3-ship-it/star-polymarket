"""
TRIX_CROSS strategy -- Triple-smoothed EMA rate of change crossover.

Processes BTC 1-min bars.  Computes TRIX(15) as the rate of change of a
triple-smoothed EMA, then generates signals when TRIX crosses its 9-period
signal line.  ATR% gate filters out unsuitable volatility regimes.
"""
from collections import deque
import math


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calc_ema(prev_ema, new_value, period):
    """Incremental EMA update.  Returns the new EMA value."""
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


# ---------------------------------------------------------------------------
# TRIX_CROSS Strategy
# ---------------------------------------------------------------------------

class TrixCross:
    """TRIX crossover strategy with ATR volatility gate.

    A signal fires when:
    1. TRIX(15) crosses above/below its 9-period signal line
    2. ATR% is between 0.12% and 2.8% (filters dead/insane markets)
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"TRIX_CROSS_{horizon_bars}m"

        # ---- TRIX(15) ----
        self.trix_period = params.get("trix_period", 15)
        self.trix_signal_period = params.get("trix_signal_period", 9)

        # Triple EMA state
        self._ema1 = None
        self._ema2 = None
        self._ema3 = None
        self._prev_ema3 = None

        # Seeding deques
        self._ema1_seed = deque(maxlen=self.trix_period)
        self._ema2_seed = deque(maxlen=self.trix_period)
        self._ema3_seed = deque(maxlen=self.trix_period)

        self._ema1_count = 0
        self._ema2_count = 0
        self._ema3_count = 0

        # TRIX value and signal line
        self._trix_value = 0.0
        self._prev_trix_value = 0.0
        self._trix_signal = None
        self._prev_trix_signal = None
        self._trix_signal_seed = deque(maxlen=self.trix_signal_period)

        # ---- ATR(14) ----
        self.atr_period = params.get("atr_period", 14)
        self._atr_value = 0.0
        self._atr_count = 0
        self._prev_close_atr = None
        self._tr_sum = 0.0

        self.atr_pct_min = params.get("atr_pct_min", 0.12)
        self.atr_pct_max = params.get("atr_pct_max", 2.8)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        # Need: trix_period * 3 for triple EMA + signal period + buffer
        self._warmup = self.trix_period * 3 + self.trix_signal_period + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Update ATR
        self._update_atr(high, low, close)

        # Update triple EMA -> TRIX
        self._update_trix(close)

        # Warmup check
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # Cooldown
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # Need signal line to be established
        if self._trix_signal is None or self._prev_trix_signal is None:
            return (None, 0.0)

        # ATR% gate
        if close < 1e-10:
            return (None, 0.0)
        atr_pct = (self._atr_value / close) * 100.0
        if atr_pct < self.atr_pct_min or atr_pct > self.atr_pct_max:
            return (None, 0.0)

        # ---- Signal conditions ----
        direction = None

        # TRIX crosses above signal line -> UP
        if (self._trix_value > self._trix_signal and
                self._prev_trix_value <= self._prev_trix_signal):
            direction = "UP"

        # TRIX crosses below signal line -> DOWN
        elif (self._trix_value < self._trix_signal and
              self._prev_trix_value >= self._prev_trix_signal):
            direction = "DOWN"

        if direction is None:
            return (None, 0.0)

        # ---- Compute confidence ----
        # Separation between TRIX and signal
        separation = abs(self._trix_value - self._trix_signal)
        sep_bonus = min(0.12, separation * 50.0)

        # ATR in sweet spot bonus (best around 0.5-1.5%)
        atr_mid = (self.atr_pct_min + self.atr_pct_max) / 2.0
        atr_dist = abs(atr_pct - atr_mid) / atr_mid
        atr_bonus = max(0.0, 0.06 * (1.0 - atr_dist))

        confidence = 0.64 + sep_bonus + atr_bonus
        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ------------------------------------------------------------------
    def _update_trix(self, close: float):
        """Update triple-smoothed EMA and compute TRIX."""
        # EMA1(close, period)
        self._ema1_count += 1
        self._ema1_seed.append(close)
        if self._ema1 is None:
            if len(self._ema1_seed) >= self.trix_period:
                self._ema1 = sum(self._ema1_seed) / self.trix_period
        else:
            self._ema1 = calc_ema(self._ema1, close, self.trix_period)

        if self._ema1 is None:
            return

        # EMA2(EMA1, period)
        self._ema2_seed.append(self._ema1)
        if self._ema2 is None:
            if len(self._ema2_seed) >= self.trix_period:
                self._ema2 = sum(self._ema2_seed) / self.trix_period
        else:
            self._ema2 = calc_ema(self._ema2, self._ema1, self.trix_period)

        if self._ema2 is None:
            return

        # EMA3(EMA2, period)
        self._ema3_seed.append(self._ema2)
        if self._ema3 is None:
            if len(self._ema3_seed) >= self.trix_period:
                self._ema3 = sum(self._ema3_seed) / self.trix_period
                self._prev_ema3 = self._ema3
        else:
            self._prev_ema3 = self._ema3
            self._ema3 = calc_ema(self._ema3, self._ema2, self.trix_period)

        if self._ema3 is None or self._prev_ema3 is None:
            return

        # TRIX = ROC of EMA3
        self._prev_trix_value = self._trix_value
        if abs(self._prev_ema3) > 1e-10:
            self._trix_value = ((self._ema3 - self._prev_ema3) / self._prev_ema3) * 100.0
        else:
            self._trix_value = 0.0

        # Signal line = EMA(TRIX, 9)
        self._trix_signal_seed.append(self._trix_value)
        self._prev_trix_signal = self._trix_signal
        if self._trix_signal is None:
            if len(self._trix_signal_seed) >= self.trix_signal_period:
                self._trix_signal = sum(self._trix_signal_seed) / self.trix_signal_period
        else:
            self._trix_signal = calc_ema(self._trix_signal, self._trix_value,
                                         self.trix_signal_period)

    def _update_atr(self, high: float, low: float, close: float):
        """Update ATR(14) using Wilder smoothing."""
        if self._prev_close_atr is None:
            self._prev_close_atr = close
            return

        tr = max(high - low,
                 abs(high - self._prev_close_atr),
                 abs(low - self._prev_close_atr))
        self._prev_close_atr = close

        self._atr_count += 1
        if self._atr_count <= self.atr_period:
            self._tr_sum += tr
            if self._atr_count == self.atr_period:
                self._atr_value = self._tr_sum / self.atr_period
        else:
            # Wilder smoothing
            self._atr_value = (self._atr_value * (self.atr_period - 1) + tr) / self.atr_period
