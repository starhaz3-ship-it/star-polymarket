"""
ROC_EXTREME strategy -- Rate of Change mean reversion at extremes.

Processes BTC 1-min bars.  Detects when ROC(12) reaches extreme levels
(>1.5% or <-1.5%) and begins reverting, confirmed by RSI exhaustion and
a climactic volume spike.  Generates counter-trend signals for Polymarket
binary options.
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
# ROC_EXTREME Strategy
# ---------------------------------------------------------------------------

class RocExtreme:
    """Rate of Change extreme reversal strategy.

    A signal fires when:
    1. ROC(12) reached an extreme (|ROC| > 1.5%) and starts reverting
    2. RSI(14) confirms exhaustion (<30 for UP, >70 for DOWN)
    3. Volume exceeds 1.5x the 20-bar EMA of volume (climactic move)
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"ROC_EXTREME_{horizon_bars}m"

        # ---- ROC(12) ----
        self.roc_period = params.get("roc_period", 12)
        self.roc_extreme_pct = params.get("roc_extreme_pct", 1.5)
        self._closes = deque(maxlen=self.roc_period + 1)
        self._roc_value = 0.0
        self._prev_roc_value = 0.0

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._prev_close = None
        self.rsi_value = 50.0

        # ---- Volume EMA(20) ----
        self.vol_ema_period = params.get("vol_ema_period", 20)
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=self.vol_ema_period)
        self.vol_spike_mult = params.get("vol_spike_mult", 1.5)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.roc_period + 2, self.rsi_period + 2,
                           self.vol_ema_period + 2) + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Store closes for ROC
        self._closes.append(close)

        # Update RSI
        self._update_rsi(close)
        self._prev_close = close

        # Update Volume EMA
        self._vol_ema_seed.append(volume)
        if self._vol_ema is None:
            if len(self._vol_ema_seed) >= self.vol_ema_period:
                self._vol_ema = sum(self._vol_ema_seed) / self.vol_ema_period
        else:
            self._vol_ema = calc_ema(self._vol_ema, volume, self.vol_ema_period)

        # Compute ROC
        self._prev_roc_value = self._roc_value
        if len(self._closes) > self.roc_period:
            past_close = self._closes[-(self.roc_period + 1)]
            if past_close > 1e-10:
                self._roc_value = ((close - past_close) / past_close) * 100.0
            else:
                self._roc_value = 0.0
        else:
            self._roc_value = 0.0

        # Warmup check
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # Cooldown
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        if self._vol_ema is None or self._vol_ema < 1e-10:
            return (None, 0.0)

        # ---- Signal conditions ----

        # 1. Volume spike: volume > 1.5x EMA(20)
        vol_ratio = volume / self._vol_ema
        if vol_ratio < self.vol_spike_mult:
            return (None, 0.0)

        direction = None

        # 2. ROC extreme + reverting
        # ROC was < -1.5% (oversold extreme) and starts rising -> mean reversion UP
        if (self._prev_roc_value < -self.roc_extreme_pct and
                self._roc_value > self._prev_roc_value):
            # 3. RSI confirms exhaustion: RSI < 30 for UP
            if self.rsi_value < 30:
                direction = "UP"

        # ROC was > 1.5% (overbought extreme) and starts declining -> mean reversion DOWN
        elif (self._prev_roc_value > self.roc_extreme_pct and
              self._roc_value < self._prev_roc_value):
            # 3. RSI confirms exhaustion: RSI > 70 for DOWN
            if self.rsi_value > 70:
                direction = "DOWN"

        if direction is None:
            return (None, 0.0)

        # ---- Compute confidence ----
        roc_extremity = abs(self._prev_roc_value) - self.roc_extreme_pct
        roc_bonus = min(0.10, roc_extremity * 0.04)

        vol_bonus = min(0.08, (vol_ratio - self.vol_spike_mult) * 0.03)

        # RSI extremity bonus
        if direction == "UP":
            rsi_bonus = min(0.06, (30.0 - self.rsi_value) * 0.003)
        else:
            rsi_bonus = min(0.06, (self.rsi_value - 70.0) * 0.003)

        confidence = 0.65 + roc_bonus + vol_bonus + rsi_bonus
        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ------------------------------------------------------------------
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
