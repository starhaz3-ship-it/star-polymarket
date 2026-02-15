"""
ELDER_IMPULSE strategy -- Elder Impulse System with consecutive-bar confirmation.

Processes BTC 1-min bars using Dr. Alexander Elder's Impulse System, which combines
EMA(13) slope direction with MACD(12,26,9) histogram direction to classify each bar
as green (both rising), red (both falling), or blue (mixed).

A signal fires only after 3 consecutive same-color bars, filtered by ADX(14) > 20
to ensure a real trend exists (not choppy noise). Three consecutive green bars in
a trending market = strong bullish impulse. Three consecutive red bars = strong
bearish impulse.

The Elder Impulse System is specifically designed to catch the "impulse" -- the
moment when both trend (EMA) and momentum (MACD histogram) agree. Requiring
3 bars of agreement eliminates single-bar fakeouts.
"""
from collections import deque
import math


def calc_ema(prev_ema, new_value, period):
    """Incremental EMA update. Returns the new EMA value."""
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


class ElderImpulse:
    """Elder Impulse System: EMA(13) + MACD histogram direction with ADX filter.

    Signal fires when:
    1. 3 consecutive green bars (EMA rising + MACD hist rising) -> UP
    2. 3 consecutive red bars (EMA falling + MACD hist falling) -> DOWN
    3. ADX(14) must be > 20 (trend strength filter)
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"ELDER_IMPULSE_{horizon_bars}m"

        # ---- EMA(13) for trend direction ----
        self.ema_period = params.get("ema_period", 13)
        self._ema = None
        self._prev_ema = None
        self._ema_seed = deque(maxlen=self.ema_period)

        # ---- MACD(12,26,9) histogram ----
        self.macd_fast = params.get("macd_fast", 12)
        self.macd_slow = params.get("macd_slow", 26)
        self.macd_signal_p = params.get("macd_signal", 9)
        self._ema_fast = None
        self._ema_slow = None
        self._macd_line = 0.0
        self._macd_signal = None
        self._macd_histogram = 0.0
        self._prev_macd_histogram = 0.0
        self._macd_count = 0

        # ---- ADX(14) ----
        self.adx_period = params.get("adx_period", 14)
        self.adx_threshold = params.get("adx_threshold", 20.0)
        self._adx_highs = deque(maxlen=2)
        self._adx_lows = deque(maxlen=2)
        self._adx_closes = deque(maxlen=2)
        self._smooth_plus_dm = 0.0
        self._smooth_minus_dm = 0.0
        self._smooth_tr = 0.0
        self._dx_hist = deque(maxlen=self.adx_period)
        self._adx_value = 0.0
        self._adx_count = 0

        # ---- Consecutive bar tracking ----
        self.consec_required = params.get("consec_required", 3)
        self._consec_green = 0
        self._consec_red = 0

        # ---- Misc ----
        self._prev_close = None
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.ema_period,
                           self.macd_slow + self.macd_signal_p,
                           self.adx_period * 2) + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar. Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Save previous MACD histogram for direction comparison
        self._prev_macd_histogram = self._macd_histogram

        # Update indicators
        self._update_ema(close)
        self._update_macd(close)
        self._update_adx(high, low, close)

        self._prev_close = close

        # Warmup check
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # Determine bar color
        ema_rising = (self._ema is not None and self._prev_ema is not None and
                      self._ema > self._prev_ema)
        ema_falling = (self._ema is not None and self._prev_ema is not None and
                       self._ema < self._prev_ema)

        macd_hist_rising = self._macd_histogram > self._prev_macd_histogram
        macd_hist_falling = self._macd_histogram < self._prev_macd_histogram

        is_green = ema_rising and macd_hist_rising    # bullish impulse
        is_red = ema_falling and macd_hist_falling    # bearish impulse

        # Track consecutive bars
        if is_green:
            self._consec_green += 1
            self._consec_red = 0
        elif is_red:
            self._consec_red += 1
            self._consec_green = 0
        else:
            # Blue bar (mixed) -- reset both
            self._consec_green = 0
            self._consec_red = 0

        # Cooldown check
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ADX trend strength filter
        if self._adx_value < self.adx_threshold:
            return (None, 0.0)

        direction = None
        confidence = 0.0

        # ---- Bullish: 3+ consecutive green bars ----
        if self._consec_green >= self.consec_required:
            direction = "UP"
            # Confidence scales with streak length and ADX strength
            streak_bonus = min(0.08, (self._consec_green - self.consec_required) * 0.03)
            adx_bonus = min(0.08, (self._adx_value - self.adx_threshold) * 0.003)
            confidence = 0.68 + streak_bonus + adx_bonus

        # ---- Bearish: 3+ consecutive red bars ----
        elif self._consec_red >= self.consec_required:
            direction = "DOWN"
            streak_bonus = min(0.08, (self._consec_red - self.consec_required) * 0.03)
            adx_bonus = min(0.08, (self._adx_value - self.adx_threshold) * 0.003)
            confidence = 0.68 + streak_bonus + adx_bonus

        if direction is None:
            return (None, 0.0)

        # MACD histogram magnitude bonus
        if close > 0:
            hist_pct = abs(self._macd_histogram) / close
            confidence += min(0.06, hist_pct * 5000.0)

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_ema(self, close):
        """EMA(13) with SMA seed."""
        self._ema_seed.append(close)
        self._prev_ema = self._ema

        if self._ema is None:
            if len(self._ema_seed) >= self.ema_period:
                self._ema = sum(self._ema_seed) / self.ema_period
        else:
            self._ema = calc_ema(self._ema, close, self.ema_period)

    def _update_macd(self, close):
        """MACD(12,26,9) with incremental EMA."""
        self._macd_count += 1

        # Fast EMA
        if self._ema_fast is None:
            if self._macd_count == self.macd_fast:
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

    def _update_adx(self, high, low, close):
        """ADX(14) with Wilder smoothing."""
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
                    plus_di = 100.0 * self._smooth_plus_dm / self._smooth_tr
                    minus_di = 100.0 * self._smooth_minus_dm / self._smooth_tr
                    di_sum = plus_di + minus_di
                    if di_sum > 0:
                        dx = 100.0 * abs(plus_di - minus_di) / di_sum
                        self._dx_hist.append(dx)
        else:
            # Wilder smoothing
            self._smooth_plus_dm = (self._smooth_plus_dm -
                                    (self._smooth_plus_dm / self.adx_period) + plus_dm)
            self._smooth_minus_dm = (self._smooth_minus_dm -
                                     (self._smooth_minus_dm / self.adx_period) + minus_dm)
            self._smooth_tr = (self._smooth_tr -
                               (self._smooth_tr / self.adx_period) + tr)

            if self._smooth_tr > 0:
                plus_di = 100.0 * self._smooth_plus_dm / self._smooth_tr
                minus_di = 100.0 * self._smooth_minus_dm / self._smooth_tr

                di_sum = plus_di + minus_di
                if di_sum > 0:
                    dx = 100.0 * abs(plus_di - minus_di) / di_sum
                    self._dx_hist.append(dx)

                if len(self._dx_hist) >= self.adx_period:
                    if self._adx_value == 0.0:
                        # First ADX = SMA of DX
                        self._adx_value = (sum(list(self._dx_hist)[-self.adx_period:]) /
                                           self.adx_period)
                    else:
                        self._adx_value = ((self._adx_value * (self.adx_period - 1) + dx) /
                                           self.adx_period)
