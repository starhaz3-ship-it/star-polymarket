"""
FISHER_CASCADE strategy -- Fisher Transform reversal + volume spike + RSI.

Processes BTC 1-min bars.  Detects Fisher Transform extreme crossovers
confirmed by a volume spike and RSI agreement to generate reversal signals
for Polymarket binary options.
"""
from collections import deque
import math

from nautilus_backtest.indicators.fisher import FisherTransform


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calc_sma(values, period):
    """Simple moving average from the last *period* items."""
    vals = list(values)[-period:]
    if len(vals) < period:
        return None
    return sum(vals) / period


def calc_ema(prev_ema, new_value, period):
    """Incremental EMA update."""
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


# ---------------------------------------------------------------------------
# FISHER_CASCADE Strategy
# ---------------------------------------------------------------------------

class FisherCascade:
    """Fisher Transform extreme crossover + volume spike + RSI confirmation.

    A signal fires when:
    1. Fisher Transform reaches an extreme (|value| > 1.2) and crosses its trigger
    2. Current volume exceeds 1.5x the 20-bar EMA of volume
    3. RSI(14) confirms the direction (oversold for UP, overbought for DOWN)
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"FISHER_CASCADE_{horizon_bars}m"

        # ---- Fisher Transform ----
        self.fisher = FisherTransform(period=params.get("fisher_period", 10))
        self.fisher_extreme = params.get("fisher_extreme", 0.8)

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
        self.vol_spike_mult = params.get("vol_spike_mult", 1.2)

        # ---- Previous Fisher for crossover detection ----
        self._prev_fisher_value = 0.0
        self._prev_fisher_trigger = 0.0

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.rsi_period, self.vol_ema_period, 15) + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Save previous Fisher state for crossover detection
        self._prev_fisher_value = self.fisher.value
        self._prev_fisher_trigger = self.fisher.trigger

        # Update Fisher Transform
        self.fisher.update(high, low, close)

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

        # Wait for warmup
        if self._bar_count < self._warmup:
            return (None, 0.0)

        if not self.fisher.initialized:
            return (None, 0.0)

        if self._vol_ema is None or self._vol_ema < 1e-10:
            return (None, 0.0)

        # ---- Signal conditions ----

        # 1. Fisher extreme crossover
        fisher_val = self.fisher.value
        fisher_trig = self.fisher.trigger

        # Bullish: Fisher was below -extreme, now crosses above trigger
        bullish_cross = (fisher_val < -self.fisher_extreme and
                         fisher_val > fisher_trig and
                         self._prev_fisher_value <= self._prev_fisher_trigger)

        # Bearish: Fisher was above +extreme, now crosses below trigger
        bearish_cross = (fisher_val > self.fisher_extreme and
                         fisher_val < fisher_trig and
                         self._prev_fisher_value >= self._prev_fisher_trigger)

        if not bullish_cross and not bearish_cross:
            return (None, 0.0)

        # 2. Volume spike
        vol_ratio = volume / self._vol_ema
        if vol_ratio < self.vol_spike_mult:
            return (None, 0.0)

        # 3. RSI confirmation (loose — just not contradicting)
        if bullish_cross and self.rsi_value > 65:
            # RSI strongly overbought contradicts bullish reversal
            return (None, 0.0)
        if bearish_cross and self.rsi_value < 35:
            # RSI strongly oversold contradicts bearish reversal
            return (None, 0.0)

        # Cooldown
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ---- Compute direction and confidence ----
        if bullish_cross:
            direction = "UP"
            fisher_extremity = max(0, abs(fisher_val) - self.fisher_extreme) / 2.0
        else:
            direction = "DOWN"
            fisher_extremity = max(0, abs(fisher_val) - self.fisher_extreme) / 2.0

        # Volume bonus (capped)
        vol_bonus = min(0.08, (vol_ratio - self.vol_spike_mult) * 0.02)

        # Fisher extremity bonus
        fisher_bonus = min(0.10, fisher_extremity * 0.05)

        confidence = 0.65 + fisher_bonus + vol_bonus
        confidence = min(confidence, 0.90)

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
