"""
KAMA_TREND strategy -- Kaufman Adaptive Moving Average trend follower.

Processes BTC 1-min bars.  Computes KAMA(10, fast=2, slow=30) which adapts
its smoothing speed based on the market's efficiency ratio.  Generates signals
when price crosses KAMA while KAMA is trending, confirmed by RSI(14) in the
neutral zone (40-60).
"""
from collections import deque
import math


# ---------------------------------------------------------------------------
# KAMA_TREND Strategy
# ---------------------------------------------------------------------------

class KamaTrend:
    """Kaufman Adaptive Moving Average trend following strategy.

    A signal fires when:
    1. Price crosses above KAMA AND KAMA is rising -> UP
       Price crosses below KAMA AND KAMA is falling -> DOWN
    2. RSI(14) between 40-60 (momentum not at extremes)
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"KAMA_TREND_{horizon_bars}m"

        # ---- KAMA(10, fast=2, slow=30) ----
        self.kama_er_period = params.get("kama_er_period", 10)
        self.kama_fast = params.get("kama_fast", 2)
        self.kama_slow = params.get("kama_slow", 30)

        # Precompute fast and slow smoothing constants
        self._fast_sc = 2.0 / (self.kama_fast + 1)
        self._slow_sc = 2.0 / (self.kama_slow + 1)

        self._closes = deque(maxlen=self.kama_er_period + 1)
        self._kama_value = None
        self._prev_kama_value = None
        self._prev_prev_kama_value = None  # for trend direction

        # Track previous close vs KAMA for crossover detection
        self._prev_close = None

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._prev_close_rsi = None
        self.rsi_value = 50.0

        self.rsi_low = params.get("rsi_low", 40)
        self.rsi_high = params.get("rsi_high", 60)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.kama_er_period + 5, self.rsi_period + 2) + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Update RSI
        self._update_rsi(close)

        # Update KAMA
        self._update_kama(close)

        prev_close = self._prev_close
        self._prev_close = close

        # Warmup check
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # Cooldown
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # Need KAMA history for trend direction
        if (self._kama_value is None or self._prev_kama_value is None or
                self._prev_prev_kama_value is None or prev_close is None):
            return (None, 0.0)

        # ---- Signal conditions ----

        # KAMA trend direction
        kama_rising = self._kama_value > self._prev_kama_value
        kama_falling = self._kama_value < self._prev_kama_value

        # Price crossover
        price_crossed_above = (prev_close <= self._prev_kama_value and
                               close > self._kama_value)
        price_crossed_below = (prev_close >= self._prev_kama_value and
                               close < self._kama_value)

        direction = None

        # Price crosses above KAMA AND KAMA is rising -> UP
        if price_crossed_above and kama_rising:
            # RSI between 40-60 (not at extremes)
            if self.rsi_low <= self.rsi_value <= self.rsi_high:
                direction = "UP"

        # Price crosses below KAMA AND KAMA is falling -> DOWN
        elif price_crossed_below and kama_falling:
            # RSI between 40-60 (not at extremes)
            if self.rsi_low <= self.rsi_value <= self.rsi_high:
                direction = "DOWN"

        if direction is None:
            return (None, 0.0)

        # ---- Compute confidence ----

        # KAMA slope strength: steeper = more confident
        if self._prev_kama_value > 1e-10:
            kama_slope_pct = abs(self._kama_value - self._prev_kama_value) / self._prev_kama_value * 100.0
        else:
            kama_slope_pct = 0.0
        slope_bonus = min(0.10, kama_slope_pct * 5.0)

        # Price separation from KAMA after cross
        if self._kama_value > 1e-10:
            sep_pct = abs(close - self._kama_value) / self._kama_value * 100.0
        else:
            sep_pct = 0.0
        sep_bonus = min(0.08, sep_pct * 2.0)

        # RSI centrality bonus: closer to 50 = better
        rsi_centrality = 1.0 - abs(self.rsi_value - 50.0) / 10.0
        rsi_bonus = max(0.0, min(0.06, rsi_centrality * 0.06))

        # Efficiency ratio bonus (if available from KAMA calculation)
        er_bonus = min(0.06, self._last_er * 0.06) if hasattr(self, '_last_er') else 0.0

        confidence = 0.63 + slope_bonus + sep_bonus + rsi_bonus + er_bonus
        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ------------------------------------------------------------------
    def _update_kama(self, close: float):
        """Update Kaufman Adaptive Moving Average."""
        self._closes.append(close)

        if len(self._closes) < self.kama_er_period + 1:
            return

        closes_list = list(self._closes)

        # Efficiency Ratio = abs(close - close[er_period]) / sum(abs(close[i] - close[i-1]))
        direction_change = abs(closes_list[-1] - closes_list[0])

        volatility = 0.0
        for i in range(1, len(closes_list)):
            volatility += abs(closes_list[i] - closes_list[i - 1])

        if volatility > 1e-10:
            er = direction_change / volatility
        else:
            er = 0.0

        self._last_er = er

        # Smoothing Constant: SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
        sc = (er * (self._fast_sc - self._slow_sc) + self._slow_sc) ** 2

        # KAMA = prev_KAMA + SC * (close - prev_KAMA)
        self._prev_prev_kama_value = self._prev_kama_value
        self._prev_kama_value = self._kama_value

        if self._kama_value is None:
            self._kama_value = close
        else:
            self._kama_value = self._kama_value + sc * (close - self._kama_value)

    def _update_rsi(self, close: float):
        if self._prev_close_rsi is None:
            self._prev_close_rsi = close
            return
        delta = close - self._prev_close_rsi
        self._prev_close_rsi = close
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
