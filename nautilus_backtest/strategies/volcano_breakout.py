"""
VOLCANO_BREAKOUT strategy -- BB/KC squeeze breakout for Polymarket binary options.

Processes BTC 1-min bars.  Tracks Bollinger Band / Keltner Channel squeeze
(BB inside KC) and triggers a momentum breakout signal when the squeeze
releases.  Direction is determined by price position relative to EMA(9)/EMA(21).
"""
from collections import deque
import math


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calc_sma(values, period):
    """Simple moving average from the last *period* items."""
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


def calc_ema(prev_ema, new_value, period):
    """Incremental EMA update."""
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


# ---------------------------------------------------------------------------
# VOLCANO_BREAKOUT Strategy
# ---------------------------------------------------------------------------

class VolcanoBreakout:
    """BB/KC squeeze breakout with EMA momentum confirmation.

    The squeeze is detected when Bollinger Bands contract inside the Keltner
    Channel.  The strategy tracks how many consecutive bars the squeeze has
    been active.  When the squeeze fires (was on, now off), it bets on the
    momentum direction determined by EMA(9) vs EMA(21) alignment and price
    position relative to the bands.
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"VOLCANO_BREAKOUT_{horizon_bars}m"

        # ---- BB(20,2) ----
        self.bb_period = params.get("bb_period", 20)
        self.bb_mult = params.get("bb_mult", 2.0)
        self._bb_closes = deque(maxlen=self.bb_period)
        self.bb_upper = 0.0
        self.bb_lower = 0.0
        self.bb_mid = 0.0

        # ---- KC(20,1.5) ----
        self.kc_period = params.get("kc_period", 20)
        self.kc_mult = params.get("kc_mult", 1.5)
        self._kc_closes = deque(maxlen=self.kc_period)
        self._kc_tr_hist = deque(maxlen=self.kc_period)
        self._prev_close = None
        self.kc_upper = 0.0
        self.kc_lower = 0.0
        self.kc_mid = 0.0

        # ---- EMA(9) and EMA(21) ----
        self._ema9 = None
        self._ema21 = None
        self._ema9_seed = deque(maxlen=9)
        self._ema21_seed = deque(maxlen=21)

        # ---- Squeeze tracking ----
        self._squeeze_on = False
        self._prev_squeeze_on = False
        self._squeeze_bars = 0
        self._max_squeeze_bars = 0  # peak squeeze length for current squeeze

        # ---- Momentum histogram (close - BB mid, smoothed) ----
        self._momentum_hist = deque(maxlen=5)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.bb_period, self.kc_period, 21) + 5
        self._min_squeeze_bars = params.get("min_squeeze_bars", 4)

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # ---- Update BB ----
        self._bb_closes.append(close)
        if len(self._bb_closes) >= self.bb_period:
            self.bb_mid = calc_sma(self._bb_closes, self.bb_period)
            std = calc_stdev(self._bb_closes, self.bb_period)
            if self.bb_mid is not None and std is not None:
                self.bb_upper = self.bb_mid + self.bb_mult * std
                self.bb_lower = self.bb_mid - self.bb_mult * std

        # ---- Update KC ----
        self._kc_closes.append(close)
        if self._prev_close is not None:
            tr = max(high - low,
                     abs(high - self._prev_close),
                     abs(low - self._prev_close))
        else:
            tr = high - low
        self._kc_tr_hist.append(tr)

        if len(self._kc_closes) >= self.kc_period and len(self._kc_tr_hist) >= self.kc_period:
            self.kc_mid = calc_sma(self._kc_closes, self.kc_period)
            atr = calc_sma(self._kc_tr_hist, self.kc_period)
            if self.kc_mid is not None and atr is not None:
                self.kc_upper = self.kc_mid + self.kc_mult * atr
                self.kc_lower = self.kc_mid - self.kc_mult * atr

        self._prev_close = close

        # ---- Update EMAs ----
        self._ema9_seed.append(close)
        if self._ema9 is None:
            if len(self._ema9_seed) >= 9:
                self._ema9 = sum(self._ema9_seed) / 9
        else:
            self._ema9 = calc_ema(self._ema9, close, 9)

        self._ema21_seed.append(close)
        if self._ema21 is None:
            if len(self._ema21_seed) >= 21:
                self._ema21 = sum(self._ema21_seed) / 21
        else:
            self._ema21 = calc_ema(self._ema21, close, 21)

        # ---- Momentum histogram ----
        if self.bb_mid > 0:
            self._momentum_hist.append(close - self.bb_mid)

        # ---- Squeeze detection ----
        self._prev_squeeze_on = self._squeeze_on
        if self.bb_upper > 0 and self.kc_upper > 0:
            self._squeeze_on = (self.bb_lower > self.kc_lower) and (self.bb_upper < self.kc_upper)
        else:
            self._squeeze_on = False

        if self._squeeze_on:
            self._squeeze_bars += 1
            self._max_squeeze_bars = max(self._max_squeeze_bars, self._squeeze_bars)
        else:
            if not self._prev_squeeze_on:
                # Not in a squeeze and not just released -- reset
                self._squeeze_bars = 0
                self._max_squeeze_bars = 0

        # ---- Wait for warmup ----
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # ---- Signal: squeeze just released ----
        squeeze_fire = self._prev_squeeze_on and not self._squeeze_on

        if not squeeze_fire:
            return (None, 0.0)

        # Minimum squeeze duration filter
        if self._max_squeeze_bars < self._min_squeeze_bars:
            self._max_squeeze_bars = 0
            self._squeeze_bars = 0
            return (None, 0.0)

        # ---- Direction from momentum ----
        direction = None

        # Primary: momentum histogram direction (last value positive = UP)
        if len(self._momentum_hist) >= 2:
            mom_now = self._momentum_hist[-1]
            mom_prev = self._momentum_hist[-2]
            mom_rising = mom_now > mom_prev

            if mom_now > 0 and mom_rising:
                direction = "UP"
            elif mom_now < 0 and not mom_rising:
                direction = "DOWN"

        # Fallback: EMA crossover
        if direction is None and self._ema9 is not None and self._ema21 is not None:
            if self._ema9 > self._ema21 and close > self._ema9:
                direction = "UP"
            elif self._ema9 < self._ema21 and close < self._ema9:
                direction = "DOWN"

        if direction is None:
            self._max_squeeze_bars = 0
            self._squeeze_bars = 0
            return (None, 0.0)

        # Cooldown
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ---- Confidence ----
        # Base 0.70 + squeeze duration bonus (longer = more energy stored)
        squeeze_bonus = min(0.12, self._max_squeeze_bars * 0.006)

        # EMA alignment bonus
        ema_bonus = 0.0
        if self._ema9 is not None and self._ema21 is not None:
            if direction == "UP" and self._ema9 > self._ema21:
                ema_bonus = 0.03
            elif direction == "DOWN" and self._ema9 < self._ema21:
                ema_bonus = 0.03

        confidence = 0.70 + squeeze_bonus + ema_bonus
        confidence = min(confidence, 0.90)

        # Reset squeeze tracking
        self._max_squeeze_bars = 0
        self._squeeze_bars = 0

        self._last_signal_bar = self._bar_count
        return (direction, confidence)
