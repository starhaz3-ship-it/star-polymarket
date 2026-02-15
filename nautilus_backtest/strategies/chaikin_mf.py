"""
CHAIKIN_MF strategy -- Chaikin Money Flow crossover with RSI and SMA filter.

Processes BTC 1-min bars.  Computes CMF(20) to measure accumulation/distribution
pressure, then generates signals when CMF crosses key thresholds.  RSI(14) and
proximity to SMA(20) provide confirmation filters.
"""
from collections import deque
import math


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calc_ema(prev_ema, new_value, period):
    """Incremental EMA update.  Returns the new EMA value."""
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


# ---------------------------------------------------------------------------
# CHAIKIN_MF Strategy
# ---------------------------------------------------------------------------

class ChaikinMf:
    """Chaikin Money Flow crossover strategy.

    A signal fires when:
    1. CMF(20) crosses from negative to positive (above 0.05) -> UP
       CMF(20) crosses from positive to negative (below -0.05) -> DOWN
    2. RSI(14) confirms: RSI < 55 for UP bounce, RSI > 45 for DOWN
    3. Price within 0.3% of SMA(20) — not overextended
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"CHAIKIN_MF_{horizon_bars}m"

        # ---- CMF(20) ----
        self.cmf_period = params.get("cmf_period", 20)
        self.cmf_up_threshold = params.get("cmf_up_threshold", 0.05)
        self.cmf_down_threshold = params.get("cmf_down_threshold", -0.05)

        # Rolling window of MFV and volume for CMF
        self._mfv_window = deque(maxlen=self.cmf_period)
        self._vol_window = deque(maxlen=self.cmf_period)
        self._cmf_value = 0.0
        self._prev_cmf_value = 0.0

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._prev_close = None
        self.rsi_value = 50.0

        # ---- SMA(20) ----
        self.sma_period = params.get("sma_period", 20)
        self._sma_closes = deque(maxlen=self.sma_period)
        self._sma_value = None
        self.price_proximity_pct = params.get("price_proximity_pct", 0.3)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.cmf_period, self.rsi_period, self.sma_period) + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Update CMF
        self._update_cmf(high, low, close, volume)

        # Update RSI
        self._update_rsi(close)
        self._prev_close = close

        # Update SMA
        self._sma_closes.append(close)
        if len(self._sma_closes) >= self.sma_period:
            self._sma_value = sum(self._sma_closes) / self.sma_period

        # Warmup check
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # Cooldown
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        if self._sma_value is None:
            return (None, 0.0)

        # ---- Price proximity to SMA(20): within 0.3% ----
        if self._sma_value > 1e-10:
            price_dist_pct = abs(close - self._sma_value) / self._sma_value * 100.0
        else:
            return (None, 0.0)

        if price_dist_pct > self.price_proximity_pct:
            return (None, 0.0)

        # ---- Signal conditions ----
        direction = None

        # CMF crosses from negative to positive (above threshold) -> UP
        if (self._cmf_value > self.cmf_up_threshold and
                self._prev_cmf_value <= 0.0):
            # RSI confirmation: RSI < 55 for UP bounce (not overbought)
            if self.rsi_value < 55:
                direction = "UP"

        # CMF crosses from positive to negative (below threshold) -> DOWN
        elif (self._cmf_value < self.cmf_down_threshold and
              self._prev_cmf_value >= 0.0):
            # RSI confirmation: RSI > 45 for DOWN (not oversold)
            if self.rsi_value > 45:
                direction = "DOWN"

        if direction is None:
            return (None, 0.0)

        # ---- Compute confidence ----
        cmf_strength = abs(self._cmf_value)
        cmf_bonus = min(0.12, cmf_strength * 0.5)

        # Closer to SMA = better setup
        proximity_bonus = min(0.06, (self.price_proximity_pct - price_dist_pct) * 0.15)

        # RSI near midpoint bonus (40-60 is neutral zone = better reversal setup)
        rsi_mid_dist = abs(self.rsi_value - 50.0)
        rsi_bonus = max(0.0, 0.06 * (1.0 - rsi_mid_dist / 20.0))

        confidence = 0.64 + cmf_bonus + proximity_bonus + rsi_bonus
        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ------------------------------------------------------------------
    def _update_cmf(self, high: float, low: float, close: float, volume: float):
        """Update Chaikin Money Flow over rolling window."""
        self._prev_cmf_value = self._cmf_value

        # Money Flow Multiplier: (2*close - high - low) / (high - low)
        hl_range = high - low
        if hl_range > 1e-10:
            mf_multiplier = (2.0 * close - high - low) / hl_range
        else:
            mf_multiplier = 0.0

        # Money Flow Volume
        mfv = mf_multiplier * volume

        self._mfv_window.append(mfv)
        self._vol_window.append(volume)

        if len(self._mfv_window) >= self.cmf_period:
            total_vol = sum(self._vol_window)
            if total_vol > 1e-10:
                self._cmf_value = sum(self._mfv_window) / total_vol
            else:
                self._cmf_value = 0.0

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
