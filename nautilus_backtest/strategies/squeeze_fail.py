"""
SQUEEZE_FAIL strategy -- Failed breakout from volatility squeeze = reversal.

Processes BTC 1-min bars.  Tracks Bollinger Band(20,2) width relative to
Keltner Channel(20,1.5).  When BB contracts inside KC (squeeze), a breakout
attempt is expected.  But when that breakout FAILS (price pokes outside BB
then immediately comes back inside), it traps breakout traders and snaps
in the opposite direction.

Failed breakouts are among the highest-probability patterns in all of trading.
This strategy waits for the setup (squeeze), detects the failure, and fades it.
"""
from collections import deque
import math


def calc_sma(values, period):
    vals = list(values)[-period:]
    if len(vals) < period:
        return None
    return sum(vals) / period


def calc_stdev(values, period):
    vals = list(values)[-period:]
    if len(vals) < period:
        return None
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))


def calc_ema(prev_ema, new_value, period):
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


class SqueezeFail:
    """Trade failed breakouts from BB/KC squeezes.

    Signal fires when:
    1. BB was inside KC within last N bars (squeeze state)
    2. Price broke outside BB (breakout attempt)
    3. Price came back inside BB on the current bar (failure)
    4. Volume spike on the failure bar (trapped traders exiting)
    5. RSI NOT at extreme (not a genuine trend, just a fake-out)

    Direction: opposite of the failed breakout
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"SQUEEZE_FAIL_{horizon_bars}m"

        # ---- BB(20,2) ----
        self.bb_period = params.get("bb_period", 20)
        self.bb_mult = params.get("bb_mult", 2.0)
        self._bb_closes = deque(maxlen=self.bb_period)
        self.bb_upper = 0.0
        self.bb_lower = 0.0
        self.bb_mid = 0.0

        # ---- KC(20, 1.5) ----
        self.kc_period = params.get("kc_period", 20)
        self.kc_mult = params.get("kc_mult", 1.5)
        self._kc_closes = deque(maxlen=self.kc_period)
        self._atr_hist = deque(maxlen=self.kc_period)
        self._atr = None
        self.kc_upper = 0.0
        self.kc_lower = 0.0
        self.kc_mid = 0.0

        # ---- Volume EMA(20) ----
        self.vol_ema_period = params.get("vol_ema_period", 20)
        self.vol_spike_mult = params.get("vol_spike_mult", 1.3)
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=self.vol_ema_period)

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self.rsi_value = 50.0

        # ---- Squeeze tracking ----
        self._squeeze_bars = 0  # consecutive bars in squeeze
        self._min_squeeze_bars = params.get("min_squeeze_bars", 5)
        self._was_in_squeeze = False
        self._bars_since_squeeze = 0
        self._squeeze_lookback = params.get("squeeze_lookback", 8)  # how recent squeeze must be

        # ---- Breakout tracking ----
        self._prev_close = None
        self._prev_was_above_bb = False  # was outside upper BB
        self._prev_was_below_bb = False  # was outside lower BB
        self._prev_high = None
        self._prev_low = None

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.bb_period, self.kc_period, self.rsi_period,
                           self.vol_ema_period) + 10

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar. Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Update all indicators
        self._update_bb(close)
        self._update_kc(high, low, close)
        self._update_volume(volume)
        self._update_rsi(close)

        # Wait for warmup
        if self._bar_count < self._warmup or self.bb_upper == 0.0 or self.kc_upper == 0.0:
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            self._prev_was_above_bb = close > self.bb_upper if self.bb_upper > 0 else False
            self._prev_was_below_bb = close < self.bb_lower if self.bb_lower > 0 else False
            return (None, 0.0)

        if self._vol_ema is None or self._vol_ema < 1e-10:
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            return (None, 0.0)

        # ---- Check squeeze state ----
        bb_inside_kc = (self.bb_upper <= self.kc_upper and self.bb_lower >= self.kc_lower)

        if bb_inside_kc:
            self._squeeze_bars += 1
            if self._squeeze_bars >= self._min_squeeze_bars:
                self._was_in_squeeze = True
                self._bars_since_squeeze = 0
        else:
            if self._was_in_squeeze:
                self._bars_since_squeeze += 1
                if self._bars_since_squeeze > self._squeeze_lookback:
                    self._was_in_squeeze = False
            self._squeeze_bars = 0

        # ---- Detect failed breakout ----
        direction = None

        # We need: recent squeeze + prev bar was outside BB + current bar back inside
        if self._was_in_squeeze and self._prev_close is not None:
            # Failed UPSIDE breakout: prev bar closed above BB, current bar closes below
            if self._prev_was_above_bb and close <= self.bb_upper:
                # Price was above upper BB, came back inside = failed upside breakout
                direction = "DOWN"  # fade the failed breakout

            # Failed DOWNSIDE breakout: prev bar closed below BB, current bar closes above
            elif self._prev_was_below_bb and close >= self.bb_lower:
                direction = "UP"  # fade the failed breakout

            # Also check: prev bar's HIGH poked above BB but close was inside,
            # and now we confirm the failure
            elif (self._prev_high is not None and self._prev_high > self.bb_upper and
                  self._prev_close is not None and self._prev_close <= self.bb_upper and
                  close < self._prev_close):
                direction = "DOWN"

            elif (self._prev_low is not None and self._prev_low < self.bb_lower and
                  self._prev_close is not None and self._prev_close >= self.bb_lower and
                  close > self._prev_close):
                direction = "UP"

        if direction is None:
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            self._prev_was_above_bb = close > self.bb_upper
            self._prev_was_below_bb = close < self.bb_lower
            return (None, 0.0)

        # ---- Volume confirmation: should see activity on failure ----
        vol_ratio = volume / self._vol_ema
        if vol_ratio < self.vol_spike_mult:
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            self._prev_was_above_bb = close > self.bb_upper
            self._prev_was_below_bb = close < self.bb_lower
            return (None, 0.0)

        # ---- RSI guard: avoid fading genuine trends ----
        # If RSI is very extreme (< 20 or > 80), the move might be real, not a fake-out
        if self.rsi_value < 20.0 or self.rsi_value > 80.0:
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            self._prev_was_above_bb = close > self.bb_upper
            self._prev_was_below_bb = close < self.bb_lower
            return (None, 0.0)

        # ---- Cooldown ----
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            self._prev_was_above_bb = close > self.bb_upper
            self._prev_was_below_bb = close < self.bb_lower
            return (None, 0.0)

        # ---- Confidence calculation ----
        confidence = 0.72

        # Squeeze duration bonus (longer squeeze = more energy stored)
        squeeze_bonus = min(0.06, self._squeeze_bars * 0.003)
        confidence += squeeze_bonus

        # Volume spike bonus
        vol_bonus = min(0.06, (vol_ratio - self.vol_spike_mult) * 0.02)
        confidence += vol_bonus

        # How quickly the failure happened (fewer bars since squeeze = fresher)
        freshness_bonus = min(0.04, max(0, (self._squeeze_lookback - self._bars_since_squeeze)) * 0.005)
        confidence += freshness_bonus

        # RSI near midpoint = more room for the reversal
        rsi_mid_bonus = 0.0
        if 35.0 < self.rsi_value < 65.0:
            rsi_mid_bonus = 0.03
        confidence += rsi_mid_bonus

        confidence = max(0.65, min(confidence, 0.95))

        self._last_signal_bar = self._bar_count

        # Reset squeeze state after signal (don't double-fire)
        self._was_in_squeeze = False
        self._squeeze_bars = 0

        self._prev_close = close
        self._prev_high = high
        self._prev_low = low
        self._prev_was_above_bb = close > self.bb_upper
        self._prev_was_below_bb = close < self.bb_lower
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_bb(self, close):
        self._bb_closes.append(close)
        if len(self._bb_closes) < self.bb_period:
            return
        self.bb_mid = calc_sma(self._bb_closes, self.bb_period)
        std = calc_stdev(self._bb_closes, self.bb_period)
        if std is None or self.bb_mid is None:
            return
        self.bb_upper = self.bb_mid + self.bb_mult * std
        self.bb_lower = self.bb_mid - self.bb_mult * std

    def _update_kc(self, high, low, close):
        self._kc_closes.append(close)
        if self._prev_close is not None:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        else:
            tr = high - low
        self._atr_hist.append(tr)

        if len(self._kc_closes) < self.kc_period or len(self._atr_hist) < self.kc_period:
            return

        self.kc_mid = calc_sma(self._kc_closes, self.kc_period)
        if self._atr is None:
            self._atr = sum(self._atr_hist) / self.kc_period
        else:
            self._atr = (self._atr * (self.kc_period - 1) + self._atr_hist[-1]) / self.kc_period

        if self.kc_mid is None:
            return
        self.kc_upper = self.kc_mid + self.kc_mult * self._atr
        self.kc_lower = self.kc_mid - self.kc_mult * self._atr

    def _update_volume(self, volume):
        self._vol_ema_seed.append(volume)
        if self._vol_ema is None:
            if len(self._vol_ema_seed) >= self.vol_ema_period:
                self._vol_ema = sum(self._vol_ema_seed) / self.vol_ema_period
        else:
            self._vol_ema = calc_ema(self._vol_ema, volume, self.vol_ema_period)

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
                self.rsi_value = 100.0 - (100.0 / (1.0 + rs))
        else:
            self._rsi_avg_gain = (self._rsi_avg_gain * (self.rsi_period - 1) + gain) / self.rsi_period
            self._rsi_avg_loss = (self._rsi_avg_loss * (self.rsi_period - 1) + loss) / self.rsi_period
            rs = self._rsi_avg_gain / max(self._rsi_avg_loss, 1e-10)
            self.rsi_value = 100.0 - (100.0 / (1.0 + rs))
