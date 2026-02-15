"""
AROON_CROSS strategy -- Aroon indicator crossover with ADX confirmation.

Processes BTC 1-min bars.  Computes Aroon Up/Down(25) to detect trend
emergence, generating signals when one Aroon line surges above 70 while the
other drops below 30.  ADX(14) > 18 confirms a trending environment.
"""
from collections import deque
import math


# ---------------------------------------------------------------------------
# AROON_CROSS Strategy
# ---------------------------------------------------------------------------

class AroonCross:
    """Aroon crossover strategy with ADX trend confirmation.

    A signal fires when:
    1. Aroon Up crosses above 70 AND Aroon Down drops below 30 -> UP
       Aroon Down crosses above 70 AND Aroon Up drops below 30 -> DOWN
    2. ADX(14) > 18 confirms a trending market
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"AROON_CROSS_{horizon_bars}m"

        # ---- Aroon(25) ----
        self.aroon_period = params.get("aroon_period", 25)
        self._highs = deque(maxlen=self.aroon_period + 1)
        self._lows = deque(maxlen=self.aroon_period + 1)
        self._aroon_up = 0.0
        self._aroon_down = 0.0
        self._prev_aroon_up = 0.0
        self._prev_aroon_down = 0.0

        # ---- ADX(14) ----
        self.adx_period = params.get("adx_period", 14)
        self.adx_threshold = params.get("adx_threshold", 18)
        self._adx_highs = deque(maxlen=2)
        self._adx_lows = deque(maxlen=2)
        self._adx_closes = deque(maxlen=2)
        self._smooth_plus_dm = 0.0
        self._smooth_minus_dm = 0.0
        self._smooth_tr = 0.0
        self._dx_hist = deque(maxlen=self.adx_period)
        self._adx_value = 0.0
        self._plus_di = 0.0
        self._minus_di = 0.0
        self._adx_count = 0

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.aroon_period + 2, self.adx_period * 2 + 5) + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Update Aroon
        self._update_aroon(high, low)

        # Update ADX
        self._update_adx(high, low, close)

        # Warmup check
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # Cooldown
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ADX gate: require trending market
        if self._adx_value < self.adx_threshold:
            return (None, 0.0)

        # ---- Signal conditions ----
        direction = None

        # Aroon Up crosses above 70 AND Aroon Down drops below 30 -> UP
        if (self._aroon_up > 70 and self._aroon_down < 30 and
                (self._prev_aroon_up <= 70 or self._prev_aroon_down >= 30)):
            direction = "UP"

        # Aroon Down crosses above 70 AND Aroon Up drops below 30 -> DOWN
        elif (self._aroon_down > 70 and self._aroon_up < 30 and
              (self._prev_aroon_down <= 70 or self._prev_aroon_up >= 30)):
            direction = "DOWN"

        if direction is None:
            return (None, 0.0)

        # ---- Compute confidence ----
        # Stronger Aroon separation = more confident
        aroon_sep = abs(self._aroon_up - self._aroon_down)
        aroon_bonus = min(0.12, (aroon_sep - 40.0) * 0.002)
        aroon_bonus = max(0.0, aroon_bonus)

        # ADX strength bonus
        adx_bonus = min(0.08, (self._adx_value - self.adx_threshold) * 0.003)
        adx_bonus = max(0.0, adx_bonus)

        # DI alignment bonus
        di_bonus = 0.0
        if direction == "UP" and self._plus_di > self._minus_di:
            di_bonus = min(0.06, (self._plus_di - self._minus_di) * 0.002)
        elif direction == "DOWN" and self._minus_di > self._plus_di:
            di_bonus = min(0.06, (self._minus_di - self._plus_di) * 0.002)

        confidence = 0.64 + aroon_bonus + adx_bonus + di_bonus
        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ------------------------------------------------------------------
    def _update_aroon(self, high: float, low: float):
        """Update Aroon Up and Down indicators."""
        self._prev_aroon_up = self._aroon_up
        self._prev_aroon_down = self._aroon_down

        self._highs.append(high)
        self._lows.append(low)

        if len(self._highs) < self.aroon_period + 1:
            return

        highs_list = list(self._highs)
        lows_list = list(self._lows)

        # Find bars since highest high in last aroon_period+1 bars
        max_val = highs_list[0]
        max_idx = 0
        for i in range(1, len(highs_list)):
            if highs_list[i] >= max_val:
                max_val = highs_list[i]
                max_idx = i

        # Find bars since lowest low
        min_val = lows_list[0]
        min_idx = 0
        for i in range(1, len(lows_list)):
            if lows_list[i] <= min_val:
                min_val = lows_list[i]
                min_idx = i

        # Bars since highest high (0 = most recent bar is highest)
        bars_since_high = len(highs_list) - 1 - max_idx
        bars_since_low = len(lows_list) - 1 - min_idx

        # Aroon Up = ((period - bars_since_high) / period) * 100
        self._aroon_up = ((self.aroon_period - bars_since_high) / self.aroon_period) * 100.0
        self._aroon_down = ((self.aroon_period - bars_since_low) / self.aroon_period) * 100.0

    def _update_adx(self, high: float, low: float, close: float):
        """Update ADX(14) using Wilder smoothing."""
        self._adx_highs.append(high)
        self._adx_lows.append(low)
        self._adx_closes.append(close)

        if len(self._adx_highs) < 2:
            return

        self._adx_count += 1

        h = list(self._adx_highs)
        l = list(self._adx_lows)
        c = list(self._adx_closes)

        # True range
        tr = max(h[-1] - l[-1], abs(h[-1] - c[-2]), abs(l[-1] - c[-2]))

        # Directional movement
        up_move = h[-1] - h[-2]
        down_move = l[-2] - l[-1]

        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        if self._adx_count <= self.adx_period:
            self._smooth_plus_dm += plus_dm
            self._smooth_minus_dm += minus_dm
            self._smooth_tr += tr
            if self._adx_count == self.adx_period:
                if self._smooth_tr > 0:
                    self._plus_di = 100.0 * self._smooth_plus_dm / self._smooth_tr
                    self._minus_di = 100.0 * self._smooth_minus_dm / self._smooth_tr
                di_sum = self._plus_di + self._minus_di
                if di_sum > 0:
                    dx = 100.0 * abs(self._plus_di - self._minus_di) / di_sum
                    self._dx_hist.append(dx)
        else:
            # Wilder smoothing
            self._smooth_plus_dm = self._smooth_plus_dm - (self._smooth_plus_dm / self.adx_period) + plus_dm
            self._smooth_minus_dm = self._smooth_minus_dm - (self._smooth_minus_dm / self.adx_period) + minus_dm
            self._smooth_tr = self._smooth_tr - (self._smooth_tr / self.adx_period) + tr

            if self._smooth_tr > 0:
                self._plus_di = 100.0 * self._smooth_plus_dm / self._smooth_tr
                self._minus_di = 100.0 * self._smooth_minus_dm / self._smooth_tr

            di_sum = self._plus_di + self._minus_di
            if di_sum > 0:
                dx = 100.0 * abs(self._plus_di - self._minus_di) / di_sum
                self._dx_hist.append(dx)

            if len(self._dx_hist) >= self.adx_period:
                if self._adx_value == 0.0:
                    self._adx_value = sum(list(self._dx_hist)[-self.adx_period:]) / self.adx_period
                else:
                    self._adx_value = (self._adx_value * (self.adx_period - 1) + dx) / self.adx_period
