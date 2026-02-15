"""
OBV_DIVERGENCE strategy -- On-Balance Volume divergence for Polymarket binary options.

Processes BTC 1-min bars.  Tracks On-Balance Volume (OBV), detects swing
highs/lows in both price and OBV over a rolling window, and identifies
bullish/bearish divergences confirmed by RSI exhaustion readings.
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
# OBV_DIVERGENCE Strategy
# ---------------------------------------------------------------------------

class ObvDivergence:
    """On-Balance Volume divergence with RSI exhaustion confirmation.

    A signal fires when:
    1. Bullish divergence: price makes a lower low but OBV makes a higher low -> UP
       - Confirmed by RSI < 35 (exhausted sellers)
    2. Bearish divergence: price makes a higher high but OBV makes a lower high -> DOWN
       - Confirmed by RSI > 65 (exhausted buyers)

    OBV is computed incrementally: OBV += volume if close > prev_close, else OBV -= volume.
    Swing highs/lows are detected over a rolling 20-bar window.
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"OBV_DIVERGENCE_{horizon_bars}m"

        # ---- OBV state ----
        self._obv = 0.0
        self._prev_close = None

        # ---- Swing detection window ----
        self.swing_window = params.get("swing_window", 20)
        self._price_hist = deque(maxlen=self.swing_window)
        self._obv_hist = deque(maxlen=self.swing_window)
        self._high_hist = deque(maxlen=self.swing_window)
        self._low_hist = deque(maxlen=self.swing_window)

        # Swing tracking: store (bar_number, value) for recent swings
        self._swing_lookback = params.get("swing_lookback", 5)  # bars to confirm a swing

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._rsi_prev_close = None
        self.rsi_value = 50.0

        # RSI exhaustion thresholds
        self._rsi_oversold = params.get("rsi_oversold", 35.0)
        self._rsi_overbought = params.get("rsi_overbought", 65.0)

        # ---- Volume EMA(20) for bonus ----
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=20)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.swing_window, self.rsi_period + 5) + 10

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # ---- Update OBV ----
        if self._prev_close is not None:
            if close > self._prev_close:
                self._obv += volume
            elif close < self._prev_close:
                self._obv -= volume
            # If close == prev_close, OBV stays the same

        # ---- Update histories ----
        self._price_hist.append(close)
        self._obv_hist.append(self._obv)
        self._high_hist.append(high)
        self._low_hist.append(low)

        # ---- Update RSI ----
        self._update_rsi(close)

        # ---- Update volume EMA ----
        self._update_vol_ema(volume)

        self._prev_close = close

        # ---- Warmup check ----
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # ---- Cooldown ----
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ---- Need enough history ----
        if len(self._price_hist) < self.swing_window:
            return (None, 0.0)

        # ---- Detect divergences ----
        direction = self._detect_divergence()

        if direction is None:
            return (None, 0.0)

        # ---- RSI exhaustion confirmation ----
        if direction == "UP" and self.rsi_value >= self._rsi_oversold:
            return (None, 0.0)  # Not exhausted enough for bullish divergence
        if direction == "DOWN" and self.rsi_value <= self._rsi_overbought:
            return (None, 0.0)  # Not exhausted enough for bearish divergence

        # ---- Confidence calculation ----
        confidence = 0.68

        # RSI extremity bonus (more extreme = stronger exhaustion = better divergence)
        if direction == "UP":
            rsi_bonus = min(0.08, max(0.0, (self._rsi_oversold - self.rsi_value) / 50.0))
        else:
            rsi_bonus = min(0.08, max(0.0, (self.rsi_value - self._rsi_overbought) / 50.0))
        confidence += rsi_bonus

        # OBV divergence magnitude bonus
        obv_list = list(self._obv_hist)
        price_list = list(self._price_hist)
        div_strength = self._divergence_strength(price_list, obv_list, direction)
        div_bonus = min(0.08, div_strength * 0.04)
        confidence += div_bonus

        # Volume surge bonus
        if self._vol_ema is not None and self._vol_ema > 0:
            vol_ratio = volume / self._vol_ema
            if vol_ratio > 1.2:
                vol_bonus = min(0.04, (vol_ratio - 1.2) * 0.04)
                confidence += vol_bonus

        # Multiple swing confirmation bonus (if we see 2+ diverging swings)
        multi_bonus = self._multi_swing_bonus(price_list, obv_list, direction)
        confidence += multi_bonus

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ==================================================================
    # Divergence detection
    # ==================================================================

    def _detect_divergence(self):
        """Check for price/OBV divergence in the current window.

        Returns 'UP' for bullish divergence, 'DOWN' for bearish, or None.
        """
        price_list = list(self._price_hist)
        obv_list = list(self._obv_hist)
        high_list = list(self._high_hist)
        low_list = list(self._low_hist)
        n = len(price_list)

        if n < self.swing_window:
            return None

        lb = self._swing_lookback

        # Find swing lows in price and OBV (for bullish divergence)
        price_swing_lows = []
        obv_at_price_lows = []
        for i in range(lb, n - lb):
            # A swing low: low[i] is the minimum in its neighborhood
            neighborhood_lows = low_list[i - lb: i + lb + 1]
            if low_list[i] <= min(neighborhood_lows):
                price_swing_lows.append((i, low_list[i]))
                obv_at_price_lows.append((i, obv_list[i]))

        # Find swing highs in price and OBV (for bearish divergence)
        price_swing_highs = []
        obv_at_price_highs = []
        for i in range(lb, n - lb):
            neighborhood_highs = high_list[i - lb: i + lb + 1]
            if high_list[i] >= max(neighborhood_highs):
                price_swing_highs.append((i, high_list[i]))
                obv_at_price_highs.append((i, obv_list[i]))

        # ---- Bullish divergence: price lower low, OBV higher low ----
        if len(price_swing_lows) >= 2:
            # Compare last two swing lows
            prev_low = price_swing_lows[-2]
            curr_low = price_swing_lows[-1]

            prev_obv_low = obv_at_price_lows[-2]
            curr_obv_low = obv_at_price_lows[-1]

            # Price made lower low
            price_lower_low = curr_low[1] < prev_low[1]
            # OBV made higher low
            obv_higher_low = curr_obv_low[1] > prev_obv_low[1]

            if price_lower_low and obv_higher_low:
                # Current bar should be near the recent swing low (within 3 bars)
                if abs(curr_low[0] - (n - 1)) <= 3:
                    return "UP"

        # ---- Bearish divergence: price higher high, OBV lower high ----
        if len(price_swing_highs) >= 2:
            prev_high = price_swing_highs[-2]
            curr_high = price_swing_highs[-1]

            prev_obv_high = obv_at_price_highs[-2]
            curr_obv_high = obv_at_price_highs[-1]

            # Price made higher high
            price_higher_high = curr_high[1] > prev_high[1]
            # OBV made lower high
            obv_lower_high = curr_obv_high[1] < prev_obv_high[1]

            if price_higher_high and obv_lower_high:
                # Current bar should be near the recent swing high
                if abs(curr_high[0] - (n - 1)) <= 3:
                    return "DOWN"

        return None

    def _divergence_strength(self, price_list, obv_list, direction):
        """Estimate divergence strength (0.0 to 2.0+).

        Looks at how strongly price and OBV are diverging.
        """
        n = len(price_list)
        if n < 10:
            return 0.0

        half = n // 2

        if direction == "UP":
            # Price trend: compare first-half avg to second-half avg
            first_half_price = sum(price_list[:half]) / half
            second_half_price = sum(price_list[half:]) / (n - half)
            first_half_obv = sum(obv_list[:half]) / half
            second_half_obv = sum(obv_list[half:]) / (n - half)

            price_falling = first_half_price > second_half_price
            obv_rising = second_half_obv > first_half_obv

            if price_falling and obv_rising:
                price_drop = (first_half_price - second_half_price) / max(first_half_price, 1e-10)
                obv_rise = (second_half_obv - first_half_obv) / max(abs(first_half_obv), 1e-10)
                return (price_drop + abs(obv_rise)) * 10.0
        else:
            first_half_price = sum(price_list[:half]) / half
            second_half_price = sum(price_list[half:]) / (n - half)
            first_half_obv = sum(obv_list[:half]) / half
            second_half_obv = sum(obv_list[half:]) / (n - half)

            price_rising = second_half_price > first_half_price
            obv_falling = second_half_obv < first_half_obv

            if price_rising and obv_falling:
                price_rise = (second_half_price - first_half_price) / max(first_half_price, 1e-10)
                obv_drop = (first_half_obv - second_half_obv) / max(abs(first_half_obv), 1e-10)
                return (price_rise + abs(obv_drop)) * 10.0

        return 0.0

    def _multi_swing_bonus(self, price_list, obv_list, direction):
        """Bonus confidence if multiple swing points show divergence."""
        n = len(price_list)
        lb = self._swing_lookback
        low_list = list(self._low_hist)
        high_list = list(self._high_hist)

        if direction == "UP":
            # Count how many consecutive swing lows show bullish divergence
            swing_lows = []
            for i in range(lb, n - lb):
                neighborhood = low_list[i - lb: i + lb + 1]
                if low_list[i] <= min(neighborhood):
                    swing_lows.append((low_list[i], obv_list[i]))

            if len(swing_lows) >= 3:
                # Check if last 3 lows show consistent divergence
                p1, o1 = swing_lows[-3]
                p2, o2 = swing_lows[-2]
                p3, o3 = swing_lows[-1]
                if p3 < p2 < p1 and o3 > o2 > o1:
                    return 0.04  # Triple divergence
            return 0.0
        else:
            swing_highs = []
            for i in range(lb, n - lb):
                neighborhood = high_list[i - lb: i + lb + 1]
                if high_list[i] >= max(neighborhood):
                    swing_highs.append((high_list[i], obv_list[i]))

            if len(swing_highs) >= 3:
                p1, o1 = swing_highs[-3]
                p2, o2 = swing_highs[-2]
                p3, o3 = swing_highs[-1]
                if p3 > p2 > p1 and o3 < o2 < o1:
                    return 0.04  # Triple divergence
            return 0.0

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_rsi(self, close: float):
        """Update RSI(14) incrementally using Wilder smoothing."""
        if self._rsi_prev_close is None:
            self._rsi_prev_close = close
            return
        delta = close - self._rsi_prev_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        self._rsi_count += 1
        self._rsi_prev_close = close

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
            if len(self._vol_ema_seed) >= 20:
                self._vol_ema = sum(self._vol_ema_seed) / 20
        else:
            self._vol_ema = calc_ema(self._vol_ema, volume, 20)
