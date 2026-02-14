"""
MULTI_EXTREME strategy -- Ultra-strict multi-indicator extreme convergence.

MEAN_REVERT_EXTREME on steroids.  Instead of 4 conditions, requires 6
simultaneous extreme readings.  The theory: when EVERYTHING is screaming
"oversold" or "overbought" at once, the snap-back is nearly guaranteed.

Trades are very rare (maybe 1-3 per day) but each one should have >65% WR.
This is quality over quantity -- only the most extreme extremes.
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


class MultiExtreme:
    """Ultra-strict multi-indicator extreme convergence.

    Signal fires when ALL 6 conditions are met simultaneously:
    1. RSI(14) at deep extreme: < 20 or > 80 (stricter than MRE's 25/75)
    2. Price beyond Bollinger Band(20,2)
    3. Volume spike > 2.0x EMA(20) (stricter than MRE's 1.5x)
    4. Stochastic %K at extreme: < 10 or > 90 (stricter than MRE's 20/80)
    5. MACD histogram diverging (momentum fading)
    6. Price > 1.5x ATR(14) from EMA(20) (extreme displacement)

    Plus anti-chase and deceleration filters.
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"MULTI_EXTREME_{horizon_bars}m"

        # ---- RSI(14) — stricter thresholds ----
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 20.0)
        self.rsi_overbought = params.get("rsi_overbought", 80.0)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._prev_close = None
        self.rsi_value = 50.0

        # ---- BB(20,2) ----
        self.bb_period = params.get("bb_period", 20)
        self.bb_mult = params.get("bb_mult", 2.0)
        self._bb_closes = deque(maxlen=self.bb_period)
        self.bb_upper = 0.0
        self.bb_lower = 0.0
        self.bb_mid = 0.0

        # ---- Volume EMA(20) — stricter multiplier ----
        self.vol_ema_period = params.get("vol_ema_period", 20)
        self.vol_spike_mult = params.get("vol_spike_mult", 2.0)
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=self.vol_ema_period)

        # ---- Stochastic(14,3) — stricter thresholds ----
        self.stoch_k_period = params.get("stoch_k", 14)
        self.stoch_d_period = params.get("stoch_d", 3)
        self._stoch_highs = deque(maxlen=self.stoch_k_period)
        self._stoch_lows = deque(maxlen=self.stoch_k_period)
        self._stoch_k_hist = deque(maxlen=self.stoch_d_period)
        self.stoch_k = 50.0
        self.stoch_d = 50.0
        self.stoch_exhaustion_lo = params.get("stoch_exhaustion_lo", 10.0)
        self.stoch_exhaustion_hi = params.get("stoch_exhaustion_hi", 90.0)

        # ---- MACD(12,26,9) for divergence ----
        self._ema12 = None
        self._ema26 = None
        self._ema12_seed = deque(maxlen=12)
        self._ema26_seed = deque(maxlen=26)
        self._macd_signal = None
        self._macd_hist = 0.0
        self._prev_macd_hist = 0.0
        self._macd_hist_queue = deque(maxlen=9)

        # ---- ATR(14) + EMA(20) displacement ----
        self.atr_period = params.get("atr_period", 14)
        self.displacement_mult = params.get("displacement_mult", 1.5)
        self._tr_hist = deque(maxlen=self.atr_period)
        self._atr = None
        self._ema20 = None
        self._ema20_seed = deque(maxlen=20)

        # ---- Anti-chase ----
        self._consecutive_oversold = 0
        self._consecutive_overbought = 0
        self._max_consecutive = params.get("max_consecutive", 6)

        # ---- RSI deceleration ----
        self._rsi_history = deque(maxlen=5)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = 40  # need MACD(26) + signal(9) + buffer

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar. Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Update all indicators
        self._update_rsi(close)
        self._update_bb(close)
        self._update_volume(volume)
        self._update_stochastic(high, low, close)
        self._update_macd(close)
        self._update_atr(high, low, close)
        self._update_ema(close)

        self._rsi_history.append(self.rsi_value)

        # Track consecutive extremes
        if self.rsi_value < self.rsi_oversold:
            self._consecutive_oversold += 1
            self._consecutive_overbought = 0
        elif self.rsi_value > self.rsi_overbought:
            self._consecutive_overbought += 1
            self._consecutive_oversold = 0
        else:
            self._consecutive_oversold = 0
            self._consecutive_overbought = 0

        self._prev_close = close

        # Wait for warmup
        if self._bar_count < self._warmup:
            return (None, 0.0)

        if (self.bb_upper == 0.0 or self._vol_ema is None or self._vol_ema < 1e-10 or
                self._atr is None or self._atr < 1e-10 or self._ema20 is None):
            return (None, 0.0)

        # ==================================================================
        # CHECK ALL 6 CONDITIONS
        # ==================================================================

        direction = None

        # 1. RSI at deep extreme
        rsi_oversold = self.rsi_value < self.rsi_oversold
        rsi_overbought = self.rsi_value > self.rsi_overbought

        # 2. Price beyond BB
        below_bb = close <= self.bb_lower
        above_bb = close >= self.bb_upper

        # 3. Volume spike
        vol_ratio = volume / self._vol_ema
        vol_spike = vol_ratio >= self.vol_spike_mult

        # 4. Stochastic extreme
        stoch_low = self.stoch_k < self.stoch_exhaustion_lo
        stoch_high = self.stoch_k > self.stoch_exhaustion_hi

        # 5. MACD histogram diverging (momentum fading at extreme)
        macd_diverging_bullish = (self._macd_hist > self._prev_macd_hist and
                                  self._macd_hist < 0)  # hist rising from below zero
        macd_diverging_bearish = (self._macd_hist < self._prev_macd_hist and
                                  self._macd_hist > 0)  # hist falling from above zero

        # 6. Price displacement from EMA(20)
        displacement = abs(close - self._ema20) / self._atr
        displaced = displacement >= self.displacement_mult

        # ---- OVERSOLD: all 6 bullish conditions ----
        if (rsi_oversold and below_bb and vol_spike and stoch_low and
                macd_diverging_bullish and displaced and close < self._ema20):
            direction = "UP"

        # ---- OVERBOUGHT: all 6 bearish conditions ----
        elif (rsi_overbought and above_bb and vol_spike and stoch_high and
              macd_diverging_bearish and displaced and close > self._ema20):
            direction = "DOWN"

        if direction is None:
            return (None, 0.0)

        # ---- Anti-chase filter ----
        if direction == "UP" and self._consecutive_oversold > self._max_consecutive:
            return (None, 0.0)
        if direction == "DOWN" and self._consecutive_overbought > self._max_consecutive:
            return (None, 0.0)

        # ---- RSI deceleration check ----
        if len(self._rsi_history) >= 3:
            rsi_vals = list(self._rsi_history)
            recent_delta = rsi_vals[-1] - rsi_vals[-2]
            prev_delta = rsi_vals[-2] - rsi_vals[-3]
            if direction == "UP":
                if recent_delta < prev_delta and recent_delta < -1.0:
                    return (None, 0.0)
            elif direction == "DOWN":
                if recent_delta > prev_delta and recent_delta > 1.0:
                    return (None, 0.0)

        # ---- Cooldown ----
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ---- Confidence calculation ----
        confidence = 0.78  # high base due to 6 confirmations

        # RSI depth bonus
        if direction == "UP":
            rsi_bonus = min(0.06, (self.rsi_oversold - self.rsi_value) / 40.0)
        else:
            rsi_bonus = min(0.06, (self.rsi_value - self.rsi_overbought) / 40.0)
        confidence += max(0, rsi_bonus)

        # Volume spike bonus
        vol_bonus = min(0.05, (vol_ratio - self.vol_spike_mult) * 0.015)
        confidence += vol_bonus

        # Displacement bonus
        disp_bonus = min(0.05, (displacement - self.displacement_mult) * 0.02)
        confidence += disp_bonus

        # Stochastic depth bonus
        if direction == "UP":
            stoch_bonus = min(0.04, (self.stoch_exhaustion_lo - self.stoch_k) / 50.0)
        else:
            stoch_bonus = min(0.04, (self.stoch_k - self.stoch_exhaustion_hi) / 50.0)
        confidence += max(0, stoch_bonus)

        confidence = max(0.72, min(confidence, 0.96))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

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

    def _update_volume(self, volume):
        self._vol_ema_seed.append(volume)
        if self._vol_ema is None:
            if len(self._vol_ema_seed) >= self.vol_ema_period:
                self._vol_ema = sum(self._vol_ema_seed) / self.vol_ema_period
        else:
            self._vol_ema = calc_ema(self._vol_ema, volume, self.vol_ema_period)

    def _update_stochastic(self, high, low, close):
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

    def _update_macd(self, close):
        # EMA(12)
        self._ema12_seed.append(close)
        if self._ema12 is None:
            if len(self._ema12_seed) >= 12:
                self._ema12 = sum(self._ema12_seed) / 12
        else:
            self._ema12 = calc_ema(self._ema12, close, 12)

        # EMA(26)
        self._ema26_seed.append(close)
        if self._ema26 is None:
            if len(self._ema26_seed) >= 26:
                self._ema26 = sum(self._ema26_seed) / 26
        else:
            self._ema26 = calc_ema(self._ema26, close, 26)

        if self._ema12 is None or self._ema26 is None:
            return

        macd_line = self._ema12 - self._ema26
        self._macd_hist_queue.append(macd_line)

        if self._macd_signal is None:
            if len(self._macd_hist_queue) >= 9:
                self._macd_signal = sum(self._macd_hist_queue) / 9
        else:
            self._macd_signal = calc_ema(self._macd_signal, macd_line, 9)

        if self._macd_signal is not None:
            self._prev_macd_hist = self._macd_hist
            self._macd_hist = macd_line - self._macd_signal

    def _update_atr(self, high, low, close):
        if self._prev_close is not None:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        else:
            tr = high - low
        self._tr_hist.append(tr)
        if len(self._tr_hist) >= self.atr_period:
            if self._atr is None:
                self._atr = sum(self._tr_hist) / self.atr_period
            else:
                self._atr = (self._atr * (self.atr_period - 1) + tr) / self.atr_period

    def _update_ema(self, close):
        self._ema20_seed.append(close)
        if self._ema20 is None:
            if len(self._ema20_seed) >= 20:
                self._ema20 = sum(self._ema20_seed) / 20
        else:
            self._ema20 = calc_ema(self._ema20, close, 20)
