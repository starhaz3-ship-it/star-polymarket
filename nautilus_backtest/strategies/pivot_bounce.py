"""
PIVOT_BOUNCE strategy -- Rolling pivot support/resistance bounce for Polymarket binary options.

Processes BTC 1-min bars.  Computes rolling pivot levels from the last 60 bars
(Pivot, S1, R1) and detects price bounces off these levels, confirmed by RSI
turning and volume surge.  Captures mean-reversion at key structural levels.
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


# ---------------------------------------------------------------------------
# PIVOT_BOUNCE Strategy
# ---------------------------------------------------------------------------

class PivotBounce:
    """Rolling pivot level bounce with RSI and volume confirmation.

    A signal fires when:
    1. Price touches S1 within 0.05% and then moves up -> UP (support bounce)
    2. Price touches R1 within 0.05% and then moves down -> DOWN (resistance bounce)
    3. RSI(14) must be turning (was <35 now >35 for UP, was >65 now <65 for DOWN)
    4. Volume must exceed 1.2x EMA(20) for confirmation

    Pivot levels are calculated from a rolling 60-bar window:
    - Pivot = (highest_high + lowest_low + last_close) / 3
    - S1 = 2 * Pivot - highest_high
    - R1 = 2 * Pivot - lowest_low
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"PIVOT_BOUNCE_{horizon_bars}m"

        # ---- Pivot parameters ----
        self.pivot_window = params.get("pivot_window", 60)
        self.touch_pct = params.get("touch_pct", 0.0005)  # 0.05% proximity threshold

        # Rolling window for pivot calculation
        self._highs = deque(maxlen=self.pivot_window)
        self._lows = deque(maxlen=self.pivot_window)
        self._closes = deque(maxlen=self.pivot_window)

        # Current pivot levels
        self._pivot = 0.0
        self._s1 = 0.0
        self._r1 = 0.0

        # Touch tracking: did price touch S1/R1 recently?
        self._touched_s1 = False
        self._touched_r1 = False
        self._touch_bar_s1 = -999
        self._touch_bar_r1 = -999
        self._touch_window = params.get("touch_window", 5)  # bars after touch to look for bounce

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._prev_close = None
        self.rsi_value = 50.0
        self._prev_rsi = 50.0

        # RSI thresholds for turning confirmation
        self._rsi_oversold = params.get("rsi_oversold", 35.0)
        self._rsi_overbought = params.get("rsi_overbought", 65.0)

        # ---- Volume EMA(20) ----
        self.vol_period = params.get("vol_period", 20)
        self.vol_mult = params.get("vol_mult", 1.2)
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=self.vol_period)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.pivot_window, self.rsi_period + 5, self.vol_period) + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Save previous RSI
        self._prev_rsi = self.rsi_value

        # Update rolling window
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

        # Update indicators
        self._update_rsi(close)
        self._update_vol_ema(volume)
        self._prev_close = close

        # ---- Calculate pivot levels ----
        if len(self._highs) >= self.pivot_window:
            highest_high = max(self._highs)
            lowest_low = min(self._lows)
            last_close = self._closes[-1]

            self._pivot = (highest_high + lowest_low + last_close) / 3.0
            self._s1 = 2.0 * self._pivot - highest_high
            self._r1 = 2.0 * self._pivot - lowest_low

        # ---- Warmup check ----
        if self._bar_count < self._warmup:
            return (None, 0.0)

        if self._pivot == 0.0:
            return (None, 0.0)

        # ---- Detect touches ----
        # Price touches S1 if low gets within touch_pct of S1
        if self._s1 > 0 and close > 0:
            s1_proximity = abs(low - self._s1) / close
            if s1_proximity <= self.touch_pct:
                self._touched_s1 = True
                self._touch_bar_s1 = self._bar_count

        # Price touches R1 if high gets within touch_pct of R1
        if self._r1 > 0 and close > 0:
            r1_proximity = abs(high - self._r1) / close
            if r1_proximity <= self.touch_pct:
                self._touched_r1 = True
                self._touch_bar_r1 = self._bar_count

        # Expire stale touches
        if self._bar_count - self._touch_bar_s1 > self._touch_window:
            self._touched_s1 = False
        if self._bar_count - self._touch_bar_r1 > self._touch_window:
            self._touched_r1 = False

        # ---- Cooldown ----
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ---- Volume confirmation ----
        if self._vol_ema is None or self._vol_ema <= 0:
            return (None, 0.0)
        if volume < self.vol_mult * self._vol_ema:
            return (None, 0.0)

        # ---- Bounce detection ----
        direction = None

        # S1 bounce: touched S1 recently AND RSI turning up (was <35, now >35)
        if self._touched_s1:
            rsi_turning_up = (self._prev_rsi < self._rsi_oversold and
                              self.rsi_value >= self._rsi_oversold)
            # Also accept: RSI was near oversold and price bouncing up
            rsi_recovering = (self._prev_rsi < self._rsi_oversold + 5 and
                              self.rsi_value > self._prev_rsi + 2.0)

            if rsi_turning_up or rsi_recovering:
                # Confirm price is bouncing (close above S1)
                if close > self._s1:
                    direction = "UP"

        # R1 bounce: touched R1 recently AND RSI turning down (was >65, now <65)
        if direction is None and self._touched_r1:
            rsi_turning_down = (self._prev_rsi > self._rsi_overbought and
                                self.rsi_value <= self._rsi_overbought)
            # Also accept: RSI was near overbought and price rejecting down
            rsi_fading = (self._prev_rsi > self._rsi_overbought - 5 and
                          self.rsi_value < self._prev_rsi - 2.0)

            if rsi_turning_down or rsi_fading:
                # Confirm price is bouncing down (close below R1)
                if close < self._r1:
                    direction = "DOWN"

        if direction is None:
            return (None, 0.0)

        # Reset touch state after signal
        if direction == "UP":
            self._touched_s1 = False
        else:
            self._touched_r1 = False

        # ---- Confidence calculation ----
        confidence = 0.68

        # RSI extremity bonus (more extreme = stronger reversal setup)
        if direction == "UP":
            rsi_bonus = min(0.08, max(0.0, (self._rsi_oversold - self._prev_rsi) / 60.0))
        else:
            rsi_bonus = min(0.08, max(0.0, (self._prev_rsi - self._rsi_overbought) / 60.0))
        confidence += rsi_bonus

        # Proximity bonus (closer touch to pivot level = stronger reaction)
        if direction == "UP" and self._s1 > 0:
            prox = abs(low - self._s1) / close
            prox_bonus = max(0.0, 0.06 * (1.0 - prox / self.touch_pct))
        elif direction == "DOWN" and self._r1 > 0:
            prox = abs(high - self._r1) / close
            prox_bonus = max(0.0, 0.06 * (1.0 - prox / self.touch_pct))
        else:
            prox_bonus = 0.0
        confidence += prox_bonus

        # Volume surge bonus
        vol_ratio = volume / max(self._vol_ema, 1e-10)
        vol_bonus = min(0.06, max(0.0, (vol_ratio - self.vol_mult) * 0.04))
        confidence += vol_bonus

        # Price position relative to pivot (bouncing from S1 towards pivot = good)
        if direction == "UP" and close < self._pivot:
            confidence += 0.02  # still below pivot = room to run
        elif direction == "DOWN" and close > self._pivot:
            confidence += 0.02  # still above pivot = room to fall

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_rsi(self, close: float):
        """Update RSI(14) incrementally using Wilder smoothing."""
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
                self.rsi_value = 100.0 - (100.0 / (1.0 + rs))
        else:
            self._rsi_avg_gain = ((self._rsi_avg_gain * (self.rsi_period - 1) + gain) /
                                  self.rsi_period)
            self._rsi_avg_loss = ((self._rsi_avg_loss * (self.rsi_period - 1) + loss) /
                                  self.rsi_period)
            rs = self._rsi_avg_gain / max(self._rsi_avg_loss, 1e-10)
            self.rsi_value = 100.0 - (100.0 / (1.0 + rs))

    def _update_vol_ema(self, volume: float):
        """Update volume EMA(20) incrementally."""
        self._vol_ema_seed.append(volume)
        if self._vol_ema is None:
            if len(self._vol_ema_seed) >= self.vol_period:
                self._vol_ema = sum(self._vol_ema_seed) / self.vol_period
        else:
            self._vol_ema = calc_ema(self._vol_ema, volume, self.vol_period)
