"""
ALIGNMENT strategy -- 7-indicator confluence for Polymarket binary options.

Processes BTC 1-min bars and emits UP/DOWN signals when 5+ of 7 indicators
agree on direction. Uses RSI, MACD, Stochastic, Bollinger Band position,
dual-EMA crossover, close vs EMA(50), and ADX/DI.
"""
from collections import deque
import math


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

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


def calc_ema(prev_ema, new_value, period):
    """Incremental EMA update.  Returns the new EMA value."""
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


# ---------------------------------------------------------------------------
# ALIGNMENT Strategy
# ---------------------------------------------------------------------------

class Alignment:
    """7-indicator confluence strategy.

    Signal is emitted when at least 5 of the 7 indicators agree on a
    direction, subject to a Bollinger-Band guard and a per-signal cooldown.
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"ALIGNMENT_{horizon_bars}m"

        # ----- RSI(14) -----
        self.rsi_period = params.get("rsi_period", 14)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._prev_close = None
        self.rsi_value = 50.0

        # ----- MACD(12,26,9) -----
        self.macd_fast = params.get("macd_fast", 12)
        self.macd_slow = params.get("macd_slow", 26)
        self.macd_signal_p = params.get("macd_signal", 9)
        self._ema_fast = None
        self._ema_slow = None
        self._macd_line = 0.0
        self._macd_signal = None
        self._macd_histogram = 0.0
        self._macd_count = 0

        # ----- Stochastic(14,3) -----
        self.stoch_k_period = params.get("stoch_k", 14)
        self.stoch_d_period = params.get("stoch_d", 3)
        self._stoch_highs = deque(maxlen=self.stoch_k_period)
        self._stoch_lows = deque(maxlen=self.stoch_k_period)
        self._stoch_k_hist = deque(maxlen=self.stoch_d_period)
        self.stoch_k = 50.0
        self.stoch_d = 50.0

        # ----- Bollinger Bands(20,2) -----
        self.bb_period = params.get("bb_period", 20)
        self.bb_mult = params.get("bb_mult", 2.0)
        self._bb_closes = deque(maxlen=self.bb_period)
        self.bb_upper = 0.0
        self.bb_lower = 0.0
        self.bb_mid = 0.0
        self.bb_position = 0.5  # 0..1

        # ----- EMA(9) vs EMA(20) -----
        self.ema9_period = 9
        self.ema20_period = 20
        self._ema9 = None
        self._ema20 = None
        self._ema9_count = 0
        self._ema9_seed = deque(maxlen=self.ema9_period)
        self._ema20_seed = deque(maxlen=self.ema20_period)

        # ----- EMA(50) -----
        self.ema50_period = 50
        self._ema50 = None
        self._ema50_count = 0
        self._ema50_seed = deque(maxlen=self.ema50_period)

        # ----- ADX(14) / DI -----
        self.adx_period = params.get("adx_period", 14)
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

        # ----- Misc -----
        self._bar_count = 0
        self._last_signal_bar = -999
        # Warmup: need at least 50 bars for EMA50
        self._warmup = max(self.ema50_period, self.macd_slow + self.macd_signal_p,
                           self.bb_period, self.adx_period * 2) + 5

    # ------------------------------------------------------------------
    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        self._update_rsi(close)
        self._update_macd(close)
        self._update_stochastic(high, low, close)
        self._update_bb(close)
        self._update_emas(close)
        self._update_adx(high, low, close)

        self._prev_close = close

        if self._bar_count < self._warmup:
            return (None, 0.0)

        # ---- Score each indicator ----
        bull = 0
        bear = 0

        # 1. RSI
        if self.rsi_value > 55:
            bull += 1
        elif self.rsi_value < 45:
            bear += 1

        # 2. MACD histogram
        if self._macd_histogram > 0:
            bull += 1
        elif self._macd_histogram < 0:
            bear += 1

        # 3. Stochastic %K
        if self.stoch_k > 60:
            bull += 1
        elif self.stoch_k < 40:
            bear += 1

        # 4. BB position
        if self.bb_position > 0.6:
            bull += 1
        elif self.bb_position < 0.4:
            bear += 1

        # 5. EMA(9) vs EMA(20)
        if self._ema9 is not None and self._ema20 is not None:
            if self._ema9 > self._ema20:
                bull += 1
            elif self._ema9 < self._ema20:
                bear += 1

        # 6. Close vs EMA(50)
        if self._ema50 is not None:
            if close > self._ema50:
                bull += 1
            elif close < self._ema50:
                bear += 1

        # 7. ADX/DI
        if self._adx_value > 20:
            if self._plus_di > self._minus_di:
                bull += 1
            elif self._minus_di > self._plus_di:
                bear += 1

        # ---- Signal decision ----
        direction = None
        count = 0

        if bull >= 6:
            direction = "UP"
            count = bull
        elif bear >= 6:
            direction = "DOWN"
            count = bear
        else:
            return (None, 0.0)

        # Guard: BB position filter
        if direction == "UP" and self.bb_position >= 0.75:
            return (None, 0.0)
        if direction == "DOWN" and self.bb_position <= 0.25:
            return (None, 0.0)

        # Cooldown
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        confidence = 0.72 + (count - 6) * 0.08
        confidence = min(confidence, 0.90)

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_rsi(self, close: float):
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

    def _update_macd(self, close: float):
        self._macd_count += 1
        # Fast EMA
        if self._ema_fast is None:
            if self._macd_count == self.macd_fast:
                # seed: we haven't been collecting, use close as approximation
                self._ema_fast = close
            else:
                return
        else:
            self._ema_fast = calc_ema(self._ema_fast, close, self.macd_fast)

        # Slow EMA
        if self._ema_slow is None:
            if self._macd_count == self.macd_slow:
                self._ema_slow = close
            else:
                return
        else:
            self._ema_slow = calc_ema(self._ema_slow, close, self.macd_slow)

        self._macd_line = self._ema_fast - self._ema_slow

        # Signal EMA
        if self._macd_signal is None:
            self._macd_signal = self._macd_line  # seed
        else:
            self._macd_signal = calc_ema(self._macd_signal, self._macd_line, self.macd_signal_p)

        self._macd_histogram = self._macd_line - self._macd_signal

    def _update_stochastic(self, high: float, low: float, close: float):
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

    def _update_bb(self, close: float):
        self._bb_closes.append(close)
        if len(self._bb_closes) < self.bb_period:
            return

        self.bb_mid = calc_sma(self._bb_closes, self.bb_period)
        std = calc_stdev(self._bb_closes, self.bb_period)
        if std is None or self.bb_mid is None:
            return
        self.bb_upper = self.bb_mid + self.bb_mult * std
        self.bb_lower = self.bb_mid - self.bb_mult * std

        bb_range = self.bb_upper - self.bb_lower
        if bb_range > 1e-10:
            self.bb_position = (close - self.bb_lower) / bb_range
            self.bb_position = max(0.0, min(1.0, self.bb_position))
        else:
            self.bb_position = 0.5

    def _update_emas(self, close: float):
        # EMA(9)
        self._ema9_seed.append(close)
        if self._ema9 is None:
            if len(self._ema9_seed) >= self.ema9_period:
                self._ema9 = sum(self._ema9_seed) / self.ema9_period
        else:
            self._ema9 = calc_ema(self._ema9, close, self.ema9_period)

        # EMA(20)
        self._ema20_seed.append(close)
        if self._ema20 is None:
            if len(self._ema20_seed) >= self.ema20_period:
                self._ema20 = sum(self._ema20_seed) / self.ema20_period
        else:
            self._ema20 = calc_ema(self._ema20, close, self.ema20_period)

        # EMA(50)
        self._ema50_seed.append(close)
        if self._ema50 is None:
            if len(self._ema50_seed) >= self.ema50_period:
                self._ema50 = sum(self._ema50_seed) / self.ema50_period
        else:
            self._ema50 = calc_ema(self._ema50, close, self.ema50_period)

    def _update_adx(self, high: float, low: float, close: float):
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
                    # First ADX = SMA of DX
                    self._adx_value = sum(list(self._dx_hist)[-self.adx_period:]) / self.adx_period
                else:
                    self._adx_value = (self._adx_value * (self.adx_period - 1) + dx) / self.adx_period
