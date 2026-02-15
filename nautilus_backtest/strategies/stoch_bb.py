"""
STOCH_BB strategy -- Stochastic oscillator + Bollinger Band confluence.

Processes BTC 1-min bars and fires when the Stochastic(14,3) oscillator
reaches an extreme (oversold <15, overbought >85) while price simultaneously
touches the corresponding Bollinger Band(20,2). This double-filter identifies
moments where BOTH momentum and statistical price deviation agree on extreme
conditions.

Additional filters:
- Stochastic %K must cross %D in the signal direction (momentum confirmation)
- Volume spike > 1.3x EMA(20) (institutional footprint)

The Stochastic-BB combo is one of the most reliable mean-reversion setups
because each indicator catches a different dimension of "extreme": Stochastic
measures momentum position, while Bollinger Bands measure statistical deviation.
"""
from collections import deque
import math


def calc_ema(prev_ema, new_value, period):
    """Incremental EMA update. Returns the new EMA value."""
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


def calc_sma(values, period):
    """Simple moving average from the last *period* items of an iterable."""
    vals = list(values)[-period:]
    if len(vals) < period:
        return None
    return sum(vals) / period


def calc_stdev(values, period):
    """Population standard deviation of the last *period* items."""
    vals = list(values)[-period:]
    if len(vals) < period:
        return None
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))


class StochBb:
    """Stochastic(14,3) oversold/overbought at Bollinger Band(20,2) with volume spike.

    Signal fires when:
    1. Stoch %K < 15 AND close <= lower BB AND %K crosses above %D -> UP
    2. Stoch %K > 85 AND close >= upper BB AND %K crosses below %D -> DOWN
    3. Volume must be > 1.3x EMA(20) for confirmation
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"STOCH_BB_{horizon_bars}m"

        # ---- Stochastic(14,3) ----
        self.stoch_k_period = params.get("stoch_k_period", 14)
        self.stoch_d_period = params.get("stoch_d_period", 3)
        self.stoch_oversold = params.get("stoch_oversold", 15.0)
        self.stoch_overbought = params.get("stoch_overbought", 85.0)
        self._stoch_highs = deque(maxlen=self.stoch_k_period)
        self._stoch_lows = deque(maxlen=self.stoch_k_period)
        self._stoch_k_hist = deque(maxlen=self.stoch_d_period)
        self.stoch_k = 50.0
        self.stoch_d = 50.0
        self._prev_stoch_k = 50.0
        self._prev_stoch_d = 50.0

        # ---- Bollinger Bands(20,2) ----
        self.bb_period = params.get("bb_period", 20)
        self.bb_mult = params.get("bb_mult", 2.0)
        self._bb_closes = deque(maxlen=self.bb_period)
        self.bb_upper = 0.0
        self.bb_lower = 0.0
        self.bb_mid = 0.0

        # ---- Volume EMA(20) ----
        self.vol_ema_period = params.get("vol_ema_period", 20)
        self.vol_confirm_mult = params.get("vol_confirm_mult", 1.3)
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=self.vol_ema_period)

        # ---- Misc ----
        self._prev_close = None
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.stoch_k_period + self.stoch_d_period,
                           self.bb_period, self.vol_ema_period) + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar. Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Save previous stochastic values for crossover detection
        self._prev_stoch_k = self.stoch_k
        self._prev_stoch_d = self.stoch_d

        # Update indicators
        self._update_stochastic(high, low, close)
        self._update_bb(close)
        self._update_volume(volume)

        self._prev_close = close

        # Warmup check
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # Cooldown check
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # Volume confirmation
        if self._vol_ema is None or self._vol_ema <= 0:
            return (None, 0.0)
        if volume < self.vol_confirm_mult * self._vol_ema:
            return (None, 0.0)

        # BB must be valid (non-zero range)
        bb_range = self.bb_upper - self.bb_lower
        if bb_range < 1e-10:
            return (None, 0.0)

        direction = None
        confidence = 0.0

        # ---- Bullish: Stoch oversold + at lower BB + %K crosses above %D ----
        k_crosses_above_d = (self._prev_stoch_k <= self._prev_stoch_d and
                             self.stoch_k > self.stoch_d)

        if (self.stoch_k < self.stoch_oversold and
                close <= self.bb_lower and
                k_crosses_above_d):
            direction = "UP"

            # Confidence: how deep oversold + how far below BB
            stoch_depth = self.stoch_oversold - self.stoch_k
            bb_depth = (self.bb_lower - close) / bb_range  # fraction below lower band
            confidence = 0.68
            confidence += min(0.08, stoch_depth * 0.005)   # oversold depth bonus
            confidence += min(0.08, bb_depth * 0.80)       # BB penetration bonus

        # ---- Bearish: Stoch overbought + at upper BB + %K crosses below %D ----
        k_crosses_below_d = (self._prev_stoch_k >= self._prev_stoch_d and
                             self.stoch_k < self.stoch_d)

        if direction is None:
            if (self.stoch_k > self.stoch_overbought and
                    close >= self.bb_upper and
                    k_crosses_below_d):
                direction = "DOWN"

                stoch_depth = self.stoch_k - self.stoch_overbought
                bb_depth = (close - self.bb_upper) / bb_range
                confidence = 0.68
                confidence += min(0.08, stoch_depth * 0.005)
                confidence += min(0.08, bb_depth * 0.80)

        if direction is None:
            return (None, 0.0)

        # Volume spike bonus
        vol_ratio = volume / self._vol_ema
        if vol_ratio > 1.8:
            confidence += min(0.06, (vol_ratio - 1.8) * 0.04)

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_stochastic(self, high, low, close):
        """Stochastic %K and %D."""
        self._stoch_highs.append(high)
        self._stoch_lows.append(low)

        if len(self._stoch_highs) < self.stoch_k_period:
            return

        hh = max(self._stoch_highs)
        ll = min(self._stoch_lows)
        rng = hh - ll
        if rng < 1e-10:
            self.stoch_k = 50.0
        else:
            self.stoch_k = 100.0 * (close - ll) / rng

        self._stoch_k_hist.append(self.stoch_k)
        if len(self._stoch_k_hist) >= self.stoch_d_period:
            self.stoch_d = sum(self._stoch_k_hist) / len(self._stoch_k_hist)

    def _update_bb(self, close):
        """Bollinger Bands (SMA +/- mult * stdev)."""
        self._bb_closes.append(close)
        if len(self._bb_closes) < self.bb_period:
            return

        self.bb_mid = calc_sma(self._bb_closes, self.bb_period)
        std = calc_stdev(self._bb_closes, self.bb_period)
        if std is None or self.bb_mid is None:
            return
        self.bb_upper = self.bb_mid + self.bb_mult * std
        self.bb_lower = self.bb_mid - self.bb_mult * std

    def _update_volume(self, volume):
        self._vol_ema_seed.append(volume)
        if self._vol_ema is None:
            if len(self._vol_ema_seed) >= self.vol_ema_period:
                self._vol_ema = sum(self._vol_ema_seed) / self.vol_ema_period
        else:
            self._vol_ema = calc_ema(self._vol_ema, volume, self.vol_ema_period)
