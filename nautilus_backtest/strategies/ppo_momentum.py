"""
PPO_MOMENTUM strategy -- Percentage Price Oscillator for Polymarket binary options.

Processes BTC 1-min bars.  Computes PPO = (EMA(12) - EMA(26)) / EMA(26) * 100,
a PPO signal line = EMA(PPO, 9), and the PPO histogram (PPO - signal).
Fires directional signals on PPO/signal crossovers confirmed by growing/shrinking
histogram, volume confirmation, and an ATR% volatility gate.
"""
from collections import deque
import math


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calc_ema(prev_ema, new_value, period):
    """Incremental EMA update."""
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


def calc_sma(values, period):
    """Simple moving average from the last *period* items."""
    vals = list(values)[-period:]
    if len(vals) < period:
        return None
    return sum(vals) / period


# ---------------------------------------------------------------------------
# PPO_MOMENTUM Strategy
# ---------------------------------------------------------------------------

class PpoMomentum:
    """Percentage Price Oscillator crossover with histogram momentum.

    A signal fires when:
    1. PPO crosses above its signal line AND the histogram is growing -> UP
    2. PPO crosses below its signal line AND the histogram is shrinking -> DOWN
    3. Volume must exceed 1.2x its EMA(20) for confirmation
    4. ATR% must be within the 0.12%-2.8% gate (not too dead, not too wild)
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"PPO_MOMENTUM_{horizon_bars}m"

        # ---- EMA periods for PPO ----
        self.fast_period = params.get("ppo_fast", 12)
        self.slow_period = params.get("ppo_slow", 26)
        self.signal_period = params.get("ppo_signal", 9)

        # EMA state for price
        self._ema_fast = None
        self._ema_fast_seed = deque(maxlen=self.fast_period)
        self._ema_slow = None
        self._ema_slow_seed = deque(maxlen=self.slow_period)

        # PPO signal line (EMA of PPO values)
        self._ppo_signal = None
        self._ppo_signal_seed = deque(maxlen=self.signal_period)

        # PPO and histogram tracking
        self._ppo_value = 0.0
        self._prev_ppo = 0.0
        self._ppo_signal_value = 0.0
        self._prev_ppo_signal = 0.0
        self._histogram = 0.0
        self._prev_histogram = 0.0

        # ---- Volume EMA(20) ----
        self.vol_period = params.get("vol_period", 20)
        self.vol_mult = params.get("vol_mult", 1.2)
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=self.vol_period)

        # ---- ATR(14) for volatility gate ----
        self.atr_period = params.get("atr_period", 14)
        self.atr_min_pct = params.get("atr_min_pct", 0.0012)  # 0.12%
        self.atr_max_pct = params.get("atr_max_pct", 0.028)   # 2.8%
        self._tr_hist = deque(maxlen=self.atr_period)
        self._prev_close = None
        self._atr_value = 0.0

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = self.slow_period + self.signal_period + 5  # ~40 bars

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Save previous values for crossover detection
        self._prev_ppo = self._ppo_value
        self._prev_ppo_signal = self._ppo_signal_value
        self._prev_histogram = self._histogram

        # ---- Update EMAs for PPO ----
        self._update_price_emas(close)

        # ---- Update ATR ----
        self._update_atr(high, low, close)
        self._prev_close = close

        # ---- Update volume EMA ----
        self._update_vol_ema(volume)

        # ---- Compute PPO ----
        if self._ema_fast is not None and self._ema_slow is not None and self._ema_slow > 0:
            self._ppo_value = (self._ema_fast - self._ema_slow) / self._ema_slow * 100.0
        else:
            return (None, 0.0)

        # ---- Update PPO signal line ----
        self._ppo_signal_seed.append(self._ppo_value)
        if self._ppo_signal is None:
            if len(self._ppo_signal_seed) >= self.signal_period:
                self._ppo_signal = sum(self._ppo_signal_seed) / self.signal_period
                self._ppo_signal_value = self._ppo_signal
        else:
            self._ppo_signal = calc_ema(self._ppo_signal, self._ppo_value, self.signal_period)
            self._ppo_signal_value = self._ppo_signal

        if self._ppo_signal is None:
            return (None, 0.0)

        # ---- Compute histogram ----
        self._histogram = self._ppo_value - self._ppo_signal_value

        # ---- Warmup check ----
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # ---- Cooldown ----
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ---- ATR% gate ----
        if close <= 0:
            return (None, 0.0)
        atr_pct = self._atr_value / close
        if atr_pct < self.atr_min_pct or atr_pct > self.atr_max_pct:
            return (None, 0.0)

        # ---- Volume confirmation ----
        if self._vol_ema is None or self._vol_ema <= 0:
            return (None, 0.0)
        if volume < self.vol_mult * self._vol_ema:
            return (None, 0.0)

        # ---- Crossover detection ----
        direction = None

        # PPO crosses above signal AND histogram growing (positive acceleration)
        ppo_crossed_above = (self._prev_ppo <= self._prev_ppo_signal and
                             self._ppo_value > self._ppo_signal_value)
        histogram_growing = self._histogram > self._prev_histogram

        # PPO crosses below signal AND histogram shrinking (negative acceleration)
        ppo_crossed_below = (self._prev_ppo >= self._prev_ppo_signal and
                             self._ppo_value < self._ppo_signal_value)
        histogram_shrinking = self._histogram < self._prev_histogram

        if ppo_crossed_above and histogram_growing:
            direction = "UP"
        elif ppo_crossed_below and histogram_shrinking:
            direction = "DOWN"

        if direction is None:
            return (None, 0.0)

        # ---- Confidence calculation ----
        confidence = 0.68

        # Histogram magnitude bonus (stronger momentum = higher confidence)
        hist_mag = abs(self._histogram)
        hist_bonus = min(0.08, hist_mag * 0.04)
        confidence += hist_bonus

        # PPO magnitude bonus (further from zero = stronger trend)
        ppo_mag = abs(self._ppo_value)
        ppo_bonus = min(0.06, ppo_mag * 0.02)
        confidence += ppo_bonus

        # Volume surge bonus
        vol_ratio = volume / max(self._vol_ema, 1e-10)
        vol_bonus = min(0.06, (vol_ratio - self.vol_mult) * 0.03)
        confidence += max(0.0, vol_bonus)

        # ATR sweet spot bonus (mid-range volatility is best)
        atr_mid = (self.atr_min_pct + self.atr_max_pct) / 2.0
        atr_dist = abs(atr_pct - atr_mid) / atr_mid
        atr_bonus = max(0.0, 0.04 * (1.0 - atr_dist))
        confidence += atr_bonus

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_price_emas(self, close: float):
        """Update fast and slow EMAs for PPO calculation."""
        # Fast EMA
        self._ema_fast_seed.append(close)
        if self._ema_fast is None:
            if len(self._ema_fast_seed) >= self.fast_period:
                self._ema_fast = sum(self._ema_fast_seed) / self.fast_period
        else:
            self._ema_fast = calc_ema(self._ema_fast, close, self.fast_period)

        # Slow EMA
        self._ema_slow_seed.append(close)
        if self._ema_slow is None:
            if len(self._ema_slow_seed) >= self.slow_period:
                self._ema_slow = sum(self._ema_slow_seed) / self.slow_period
        else:
            self._ema_slow = calc_ema(self._ema_slow, close, self.slow_period)

    def _update_atr(self, high: float, low: float, close: float):
        """Update ATR(14) incrementally."""
        if self._prev_close is not None:
            tr = max(high - low,
                     abs(high - self._prev_close),
                     abs(low - self._prev_close))
        else:
            tr = high - low
        self._tr_hist.append(tr)

        if len(self._tr_hist) >= self.atr_period:
            if self._atr_value == 0.0:
                self._atr_value = sum(self._tr_hist) / len(self._tr_hist)
            else:
                self._atr_value = (self._atr_value * (self.atr_period - 1) + tr) / self.atr_period

    def _update_vol_ema(self, volume: float):
        """Update volume EMA(20) incrementally."""
        self._vol_ema_seed.append(volume)
        if self._vol_ema is None:
            if len(self._vol_ema_seed) >= self.vol_period:
                self._vol_ema = sum(self._vol_ema_seed) / self.vol_period
        else:
            self._vol_ema = calc_ema(self._vol_ema, volume, self.vol_period)
