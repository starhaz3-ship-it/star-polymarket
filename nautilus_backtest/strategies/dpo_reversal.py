"""
DPO_REVERSAL strategy -- Detrended Price Oscillator mean-reversion with Stochastic.

Processes BTC 1-min bars.  The Detrended Price Oscillator removes the trend from
price by subtracting SMA(20) from the current close, isolating the cyclical
component.  When DPO reaches an extreme (> 0.3% of price) and starts turning
back toward zero, it signals a mean-reversion opportunity.

Stochastic(14,3) provides confirmation: %K < 25 for UP signals, %K > 75 for
DOWN signals, ensuring momentum has already begun reversing.
"""
from collections import deque
import math


def calc_ema(prev_ema, new_value, period):
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


class DpoReversal:
    """DPO extreme + turning + Stochastic confirmation for mean reversion.

    Signal fires when:
    1. DPO(20) < -0.3% of price AND DPO is turning (closer to 0 than prev) --> UP
    2. DPO(20) > +0.3% of price AND DPO is turning --> DOWN
    3. Stochastic %K(14,3) < 25 for UP, > 75 for DOWN

    Targets cyclical exhaustion points where price has deviated from its
    moving average and is snapping back.
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"DPO_REVERSAL_{horizon_bars}m"

        # ---- DPO(20) = close - SMA(close, 20) ----
        self.dpo_period = params.get("dpo_period", 20)
        self.dpo_threshold_pct = params.get("dpo_threshold_pct", 0.003)  # 0.3%
        self._close_hist = deque(maxlen=self.dpo_period)
        self._dpo = None
        self._prev_dpo = None

        # ---- Stochastic(14,3) ----
        self.stoch_k_period = params.get("stoch_k_period", 14)
        self.stoch_smooth = params.get("stoch_smooth", 3)
        self.stoch_lo = params.get("stoch_lo", 25.0)
        self.stoch_hi = params.get("stoch_hi", 75.0)
        self._stoch_highs = deque(maxlen=self.stoch_k_period)
        self._stoch_lows = deque(maxlen=self.stoch_k_period)
        self._raw_k_hist = deque(maxlen=self.stoch_smooth)
        self._stoch_k = None

        # ---- State ----
        self._prev_close = None
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.dpo_period, self.stoch_k_period + self.stoch_smooth) + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one 1-minute bar. Returns (direction|None, confidence)."""
        self._bar_count += 1

        # Update DPO
        self._update_dpo(close)

        # Update Stochastic
        self._update_stochastic(high, low, close)

        # Warmup check
        if self._bar_count < self._warmup:
            self._prev_close = close
            return (None, 0.0)

        # Need both indicators ready
        if self._dpo is None or self._prev_dpo is None or self._stoch_k is None:
            self._prev_close = close
            return (None, 0.0)

        # Cooldown
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            self._prev_close = close
            return (None, 0.0)

        # DPO threshold relative to price
        threshold = close * self.dpo_threshold_pct

        direction = None

        # DPO at negative extreme AND turning up (getting closer to 0)
        if (self._dpo < -threshold and
                abs(self._dpo) < abs(self._prev_dpo) and
                self._stoch_k < self.stoch_lo):
            direction = "UP"

        # DPO at positive extreme AND turning down
        elif (self._dpo > threshold and
              abs(self._dpo) < abs(self._prev_dpo) and
              self._stoch_k > self.stoch_hi):
            direction = "DOWN"

        if direction is None:
            self._prev_close = close
            return (None, 0.0)

        # ---- Confidence calculation ----
        confidence = 0.72

        # DPO depth bonus: further from zero = stronger signal
        dpo_pct = abs(self._dpo) / close if close > 0 else 0
        depth_bonus = min(0.08, (dpo_pct - self.dpo_threshold_pct) * 15.0)
        confidence += max(0.0, depth_bonus)

        # Turn strength: how much DPO moved back toward zero
        if abs(self._prev_dpo) > 1e-10:
            turn_ratio = 1.0 - abs(self._dpo) / abs(self._prev_dpo)
            turn_bonus = min(0.06, turn_ratio * 0.10)
            confidence += max(0.0, turn_bonus)

        # Stochastic extremity bonus
        if direction == "UP":
            stoch_bonus = min(0.05, (self.stoch_lo - self._stoch_k) / 100.0)
        else:
            stoch_bonus = min(0.05, (self._stoch_k - self.stoch_hi) / 100.0)
        confidence += max(0.0, stoch_bonus)

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        self._prev_close = close
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_dpo(self, close):
        self._close_hist.append(close)
        if len(self._close_hist) < self.dpo_period:
            return

        sma = sum(self._close_hist) / self.dpo_period
        new_dpo = close - sma

        self._prev_dpo = self._dpo
        self._dpo = new_dpo

    def _update_stochastic(self, high, low, close):
        self._stoch_highs.append(high)
        self._stoch_lows.append(low)

        if len(self._stoch_highs) < self.stoch_k_period:
            return

        highest = max(self._stoch_highs)
        lowest = min(self._stoch_lows)
        hl_range = highest - lowest

        if hl_range < 1e-10:
            raw_k = 50.0
        else:
            raw_k = 100.0 * (close - lowest) / hl_range

        self._raw_k_hist.append(raw_k)

        if len(self._raw_k_hist) >= self.stoch_smooth:
            self._stoch_k = sum(self._raw_k_hist) / len(self._raw_k_hist)
