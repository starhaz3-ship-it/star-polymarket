"""
DIVERGENCE_HUNTER strategy -- Price/RSI divergence for Polymarket binary options.

Processes BTC 1-min bars.  Detects classic divergence patterns where price
makes a new high/low but RSI fails to confirm, signaling weakening momentum.
Confirmed with MACD histogram direction for additional conviction.

Bearish divergence: price new 20-bar high, RSI < previous 20-bar RSI high
Bullish divergence: price new 20-bar low, RSI > previous 20-bar RSI low

This is a well-known pattern that captures momentum exhaustion before reversals.
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
# DIVERGENCE_HUNTER Strategy
# ---------------------------------------------------------------------------

class DivergenceHunter:
    """Price/RSI divergence with MACD histogram confirmation.

    Signal conditions:

    Bearish divergence (-> DOWN):
    1. Price makes a new 20-bar high
    2. RSI(14) at this new high is LOWER than RSI was at the previous 20-bar high
    3. RSI divergence gap > min_divergence threshold
    4. MACD histogram is declining (momentum fading)

    Bullish divergence (-> UP):
    1. Price makes a new 20-bar low
    2. RSI(14) at this new low is HIGHER than RSI was at the previous 20-bar low
    3. RSI divergence gap > min_divergence threshold
    4. MACD histogram is rising (momentum fading)

    Additional filters:
    - Minimum bars between pivots (avoid noise)
    - EMA(20) trend context (divergence against trend is stronger)
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"DIVERGENCE_HUNTER_{horizon_bars}m"

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._prev_close = None
        self.rsi_value = 50.0

        # ---- MACD(12,26,9) ----
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

        # ---- Rolling window for price highs/lows and RSI at those pivots ----
        self.pivot_lookback = params.get("pivot_lookback", 20)
        self._price_highs = deque(maxlen=self.pivot_lookback)
        self._price_lows = deque(maxlen=self.pivot_lookback)
        self._rsi_at_bar = deque(maxlen=self.pivot_lookback)
        self._close_history = deque(maxlen=self.pivot_lookback)

        # Track previous pivot points (price and RSI at those pivots)
        self._prev_high_price = None
        self._prev_high_rsi = None
        self._prev_high_bar = -999
        self._prev_low_price = None
        self._prev_low_rsi = None
        self._prev_low_bar = -999

        # Minimum divergence in RSI units to trigger signal
        self.min_divergence = params.get("min_divergence", 3.0)
        # Minimum bars between pivots (avoid noise)
        self.min_pivot_gap = params.get("min_pivot_gap", 5)

        # ---- EMA(20) for trend context ----
        self._ema20 = None
        self._ema20_seed = deque(maxlen=20)

        # ---- Volume EMA(20) for confirmation ----
        self.vol_ema_period = params.get("vol_ema_period", 20)
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=self.vol_ema_period)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.pivot_lookback, self.macd_slow + self.macd_signal_p,
                           self.rsi_period, 50) + 10

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Save previous MACD histogram
        self._prev_macd_histogram = self._macd_histogram

        # Update all indicators
        self._update_rsi(close)
        self._update_macd(close)
        self._update_ema(close)
        self._update_volume(volume)

        self._prev_close = close

        # Track rolling price and RSI
        self._price_highs.append(high)
        self._price_lows.append(low)
        self._close_history.append(close)
        self._rsi_at_bar.append(self.rsi_value)

        # Wait for warmup
        if self._bar_count < self._warmup:
            return (None, 0.0)

        if len(self._price_highs) < self.pivot_lookback:
            return (None, 0.0)

        # ---- Detect new highs/lows within the rolling window ----
        direction = None
        divergence_strength = 0.0

        highs_list = list(self._price_highs)
        lows_list = list(self._price_lows)
        rsi_list = list(self._rsi_at_bar)

        current_high = high
        current_low = low
        current_rsi = self.rsi_value
        window_max = max(highs_list)
        window_min = min(lows_list)

        # ---- BEARISH DIVERGENCE: new price high but RSI lower ----
        if current_high >= window_max:
            # Price made a new window high -- check if RSI diverges
            # Find the previous window high (excluding last few bars)
            prev_high_idx = -1
            prev_high_val = -1e30
            for i in range(len(highs_list) - self.min_pivot_gap):
                if highs_list[i] > prev_high_val:
                    prev_high_val = highs_list[i]
                    prev_high_idx = i

            if prev_high_idx >= 0 and prev_high_val > 0:
                prev_rsi_at_high = rsi_list[prev_high_idx]
                rsi_gap = prev_rsi_at_high - current_rsi  # positive = bearish divergence

                if rsi_gap >= self.min_divergence:
                    # MACD histogram must be declining (momentum fading)
                    macd_declining = self._macd_histogram < self._prev_macd_histogram
                    if macd_declining:
                        direction = "DOWN"
                        divergence_strength = rsi_gap

        # ---- BULLISH DIVERGENCE: new price low but RSI higher ----
        if direction is None and current_low <= window_min:
            # Price made a new window low -- check if RSI diverges
            prev_low_idx = -1
            prev_low_val = 1e30
            for i in range(len(lows_list) - self.min_pivot_gap):
                if lows_list[i] < prev_low_val:
                    prev_low_val = lows_list[i]
                    prev_low_idx = i

            if prev_low_idx >= 0 and prev_low_val < 1e30:
                prev_rsi_at_low = rsi_list[prev_low_idx]
                rsi_gap = current_rsi - prev_rsi_at_low  # positive = bullish divergence

                if rsi_gap >= self.min_divergence:
                    # MACD histogram must be rising (momentum fading)
                    macd_rising = self._macd_histogram > self._prev_macd_histogram
                    if macd_rising:
                        direction = "UP"
                        divergence_strength = rsi_gap

        if direction is None:
            return (None, 0.0)

        # ---- RSI range filter: avoid divergence signals when RSI is near 50 ----
        # Divergence is more meaningful when RSI is in an extended zone
        if direction == "DOWN" and current_rsi < 45:
            # RSI already quite low for a bearish divergence -- weak signal
            return (None, 0.0)
        if direction == "UP" and current_rsi > 55:
            # RSI already quite high for a bullish divergence -- weak signal
            return (None, 0.0)

        # ---- Volume check: prefer divergence with above-average volume ----
        vol_confirms = False
        if self._vol_ema is not None and self._vol_ema > 1e-10:
            vol_ratio = volume / self._vol_ema
            if vol_ratio >= 1.0:
                vol_confirms = True
        else:
            vol_confirms = True  # No volume data yet, skip filter

        if not vol_confirms:
            return (None, 0.0)

        # ---- Cooldown ----
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ---- Confidence calculation ----
        confidence = 0.66

        # Divergence strength bonus: larger RSI gap = stronger divergence
        div_bonus = min(0.10, divergence_strength / 40.0)
        confidence += div_bonus

        # MACD histogram magnitude bonus: bigger histogram move = more momentum shift
        macd_shift = abs(self._macd_histogram - self._prev_macd_histogram)
        macd_bonus = min(0.06, macd_shift * 100.0)  # Scale for BTC price magnitudes
        confidence += macd_bonus

        # EMA(20) trend context bonus: divergence against the trend is more powerful
        ema_bonus = 0.0
        if self._ema20 is not None:
            if direction == "DOWN" and close > self._ema20:
                # Bearish divergence while price is above EMA = against trend = stronger
                ema_bonus = 0.04
            elif direction == "UP" and close < self._ema20:
                # Bullish divergence while price is below EMA = against trend = stronger
                ema_bonus = 0.04
        confidence += ema_bonus

        # Volume spike bonus
        if self._vol_ema is not None and self._vol_ema > 1e-10:
            vol_ratio = volume / self._vol_ema
            vol_bonus = min(0.04, (vol_ratio - 1.0) * 0.02)
            confidence += max(0, vol_bonus)

        confidence = max(0.60, min(confidence, 0.92))

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
            self._macd_signal = self._macd_line
        else:
            self._macd_signal = calc_ema(self._macd_signal, self._macd_line, self.macd_signal_p)

        self._macd_histogram = self._macd_line - self._macd_signal

    def _update_ema(self, close: float):
        self._ema20_seed.append(close)
        if self._ema20 is None:
            if len(self._ema20_seed) >= 20:
                self._ema20 = sum(self._ema20_seed) / 20
        else:
            self._ema20 = calc_ema(self._ema20, close, 20)

    def _update_volume(self, volume: float):
        self._vol_ema_seed.append(volume)
        if self._vol_ema is None:
            if len(self._vol_ema_seed) >= self.vol_ema_period:
                self._vol_ema = sum(self._vol_ema_seed) / self.vol_ema_period
        else:
            self._vol_ema = calc_ema(self._vol_ema, volume, self.vol_ema_period)
