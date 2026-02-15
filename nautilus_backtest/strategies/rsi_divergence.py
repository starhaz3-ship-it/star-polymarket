"""
RSI_DIVERGENCE strategy -- Classic RSI divergence detector for BTC 5-min binary options.

Processes BTC 1-min bars and detects divergence between price action and RSI(14).
A bullish divergence occurs when price makes a lower low but RSI makes a higher
low, indicating weakening bearish momentum. A bearish divergence occurs when price
makes a higher high but RSI makes a lower high, indicating weakening bullish momentum.

Confirmation requires volume > 1.2x EMA(20) to filter out low-conviction setups.
Swing points are tracked over a 20-bar lookback window using local min/max detection
with a 3-bar shoulder on each side.

Expected edge: divergences on 5-min BTC are well-documented mean-reversion signals.
Volume confirmation filters out noise divergences in thin markets.
"""
from collections import deque
import math


def calc_ema(prev_ema, new_value, period):
    """Incremental EMA update. Returns the new EMA value."""
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


class RsiDivergence:
    """RSI divergence strategy with volume confirmation.

    Signal fires when:
    1. Bullish: price makes lower low but RSI makes higher low over last 20 bars -> UP
    2. Bearish: price makes higher high but RSI makes lower high over last 20 bars -> DOWN
    3. Volume must exceed 1.2x EMA(20) for confirmation
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"RSI_DIVERGENCE_{horizon_bars}m"

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._prev_close = None
        self.rsi_value = 50.0

        # ---- Price and RSI history for swing detection ----
        self.swing_lookback = params.get("swing_lookback", 20)
        self.swing_shoulder = params.get("swing_shoulder", 3)
        self._close_hist = deque(maxlen=self.swing_lookback + self.swing_shoulder + 1)
        self._low_hist = deque(maxlen=self.swing_lookback + self.swing_shoulder + 1)
        self._high_hist = deque(maxlen=self.swing_lookback + self.swing_shoulder + 1)
        self._rsi_hist = deque(maxlen=self.swing_lookback + self.swing_shoulder + 1)

        # ---- Volume EMA(20) ----
        self.vol_ema_period = params.get("vol_ema_period", 20)
        self.vol_confirm_mult = params.get("vol_confirm_mult", 1.2)
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=self.vol_ema_period)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = self.swing_lookback + self.swing_shoulder + self.rsi_period + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar. Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Update indicators
        self._update_rsi(close)
        self._update_volume(volume)

        # Store history
        self._close_hist.append(close)
        self._low_hist.append(low)
        self._high_hist.append(high)
        self._rsi_hist.append(self.rsi_value)

        # Save prev close for RSI delta
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

        # Find swing lows and swing highs in the lookback window
        swing_lows = self._find_swing_lows()
        swing_highs = self._find_swing_highs()

        direction = None
        confidence = 0.0

        # ---- Bullish divergence: price lower low, RSI higher low ----
        if len(swing_lows) >= 2:
            # Most recent two swing lows
            prev_sw = swing_lows[-2]
            curr_sw = swing_lows[-1]

            prev_price_low = prev_sw[0]
            curr_price_low = curr_sw[0]
            prev_rsi_low = prev_sw[1]
            curr_rsi_low = curr_sw[1]

            if curr_price_low < prev_price_low and curr_rsi_low > prev_rsi_low:
                # Bullish divergence confirmed
                direction = "UP"
                # Confidence scales with RSI divergence magnitude
                rsi_diff = curr_rsi_low - prev_rsi_low
                confidence = 0.68 + min(0.16, rsi_diff * 0.008)

        # ---- Bearish divergence: price higher high, RSI lower high ----
        if direction is None and len(swing_highs) >= 2:
            prev_sw = swing_highs[-2]
            curr_sw = swing_highs[-1]

            prev_price_high = prev_sw[0]
            curr_price_high = curr_sw[0]
            prev_rsi_high = prev_sw[1]
            curr_rsi_high = curr_sw[1]

            if curr_price_high > prev_price_high and curr_rsi_high < prev_rsi_high:
                # Bearish divergence confirmed
                direction = "DOWN"
                rsi_diff = prev_rsi_high - curr_rsi_high
                confidence = 0.68 + min(0.16, rsi_diff * 0.008)

        if direction is None:
            return (None, 0.0)

        # Volume spike bonus
        vol_ratio = volume / self._vol_ema if self._vol_ema > 0 else 1.0
        if vol_ratio > 1.5:
            confidence += min(0.06, (vol_ratio - 1.5) * 0.04)

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ==================================================================
    # Swing detection helpers
    # ==================================================================

    def _find_swing_lows(self):
        """Find swing lows in the history buffer.

        A swing low is a bar whose low is lower than the lows of
        `swing_shoulder` bars on each side.

        Returns list of (price_low, rsi_value) tuples.
        """
        lows = list(self._low_hist)
        rsis = list(self._rsi_hist)
        n = len(lows)
        shoulder = self.swing_shoulder
        swings = []

        # Don't check the very last `shoulder` bars (need right shoulder)
        for i in range(shoulder, n - shoulder):
            is_swing = True
            for j in range(1, shoulder + 1):
                if lows[i] > lows[i - j] or lows[i] > lows[i + j]:
                    is_swing = False
                    break
            if is_swing:
                swings.append((lows[i], rsis[i]))

        return swings

    def _find_swing_highs(self):
        """Find swing highs in the history buffer.

        A swing high is a bar whose high is higher than the highs of
        `swing_shoulder` bars on each side.

        Returns list of (price_high, rsi_value) tuples.
        """
        highs = list(self._high_hist)
        rsis = list(self._rsi_hist)
        n = len(highs)
        shoulder = self.swing_shoulder
        swings = []

        for i in range(shoulder, n - shoulder):
            is_swing = True
            for j in range(1, shoulder + 1):
                if highs[i] < highs[i - j] or highs[i] < highs[i + j]:
                    is_swing = False
                    break
            if is_swing:
                swings.append((highs[i], rsis[i]))

        return swings

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

    def _update_volume(self, volume):
        self._vol_ema_seed.append(volume)
        if self._vol_ema is None:
            if len(self._vol_ema_seed) >= self.vol_ema_period:
                self._vol_ema = sum(self._vol_ema_seed) / self.vol_ema_period
        else:
            self._vol_ema = calc_ema(self._vol_ema, volume, self.vol_ema_period)
