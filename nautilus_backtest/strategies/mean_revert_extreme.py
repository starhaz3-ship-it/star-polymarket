"""
MEAN_REVERT_EXTREME strategy -- Rubber-band snap at extremes for Polymarket binary options.

Processes BTC 1-min bars.  Fires reversal signals when RSI(14) hits extreme
readings (<25 or >75) while price is at or beyond Bollinger Band(20,2) and
a volume spike (1.5x 20-bar EMA) confirms climactic activity.

This captures the "rubber band" snap after overextension -- very selective,
targeting only the most stretched conditions with volume confirmation from
smart money climax activity.
"""
from collections import deque
import math


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calc_sma(values, period):
    """Simple moving average from the last *period* items."""
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
    """Incremental EMA update."""
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


# ---------------------------------------------------------------------------
# MEAN_REVERT_EXTREME Strategy
# ---------------------------------------------------------------------------

class MeanRevertExtreme:
    """Pure mean reversion at extreme readings with volume climax confirmation.

    A signal fires when ALL three conditions are met simultaneously:
    1. RSI(14) at extremes: < 25 (oversold) or > 75 (overbought)
    2. Price at or beyond Bollinger Band(20,2): close <= lower BB or close >= upper BB
    3. Volume spike: current volume >= 1.5x the 20-bar EMA of volume

    Direction:
    - Oversold (RSI < 25 + below lower BB + vol spike) -> UP (mean reversion)
    - Overbought (RSI > 75 + above upper BB + vol spike) -> DOWN (mean reversion)

    Additional filters:
    - Stochastic(14,3) must confirm exhaustion (%K < 20 or %K > 80)
    - Consecutive extreme bars counter prevents chasing extended moves
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"MEAN_REVERT_EXTREME_{horizon_bars}m"

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 25.0)
        self.rsi_overbought = params.get("rsi_overbought", 75.0)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._prev_close = None
        self.rsi_value = 50.0

        # ---- Bollinger Bands(20,2) ----
        self.bb_period = params.get("bb_period", 20)
        self.bb_mult = params.get("bb_mult", 2.0)
        self._bb_closes = deque(maxlen=self.bb_period)
        self.bb_upper = 0.0
        self.bb_lower = 0.0
        self.bb_mid = 0.0

        # ---- Volume EMA(20) ----
        self.vol_ema_period = params.get("vol_ema_period", 20)
        self.vol_spike_mult = params.get("vol_spike_mult", 1.5)
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=self.vol_ema_period)

        # ---- Stochastic(14,3) for exhaustion confirmation ----
        self.stoch_k_period = params.get("stoch_k", 14)
        self.stoch_d_period = params.get("stoch_d", 3)
        self._stoch_highs = deque(maxlen=self.stoch_k_period)
        self._stoch_lows = deque(maxlen=self.stoch_k_period)
        self._stoch_k_hist = deque(maxlen=self.stoch_d_period)
        self.stoch_k = 50.0
        self.stoch_d = 50.0
        self.stoch_exhaustion_lo = params.get("stoch_exhaustion_lo", 20.0)
        self.stoch_exhaustion_hi = params.get("stoch_exhaustion_hi", 80.0)

        # ---- Consecutive extreme bars tracking ----
        # If price has been extreme for too long, the move is persistent, not reverting
        self._consecutive_oversold = 0
        self._consecutive_overbought = 0
        self._max_consecutive_extreme = params.get("max_consecutive_extreme", 8)

        # ---- Rate of change check: RSI should be decelerating at extreme ----
        self._rsi_history = deque(maxlen=5)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.bb_period, self.rsi_period, self.vol_ema_period,
                           self.stoch_k_period) + 10

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Update all indicators
        self._update_rsi(close)
        self._update_bb(close)
        self._update_volume(volume)
        self._update_stochastic(high, low, close)

        self._prev_close = close

        # Track RSI history for deceleration check
        self._rsi_history.append(self.rsi_value)

        # Track consecutive extreme bars
        if self.rsi_value < self.rsi_oversold:
            self._consecutive_oversold += 1
            self._consecutive_overbought = 0
        elif self.rsi_value > self.rsi_overbought:
            self._consecutive_overbought += 1
            self._consecutive_oversold = 0
        else:
            self._consecutive_oversold = 0
            self._consecutive_overbought = 0

        # Wait for warmup
        if self._bar_count < self._warmup:
            return (None, 0.0)

        if self.bb_upper == 0.0 or self._vol_ema is None or self._vol_ema < 1e-10:
            return (None, 0.0)

        # ---- Check for extreme conditions ----
        direction = None
        rsi_extremity = 0.0

        # OVERSOLD: RSI < 25, price at/below lower BB
        is_oversold = (self.rsi_value < self.rsi_oversold and close <= self.bb_lower)
        # OVERBOUGHT: RSI > 75, price at/above upper BB
        is_overbought = (self.rsi_value > self.rsi_overbought and close >= self.bb_upper)

        if is_oversold:
            direction = "UP"
            rsi_extremity = self.rsi_oversold - self.rsi_value  # positive, larger = more extreme
        elif is_overbought:
            direction = "DOWN"
            rsi_extremity = self.rsi_value - self.rsi_overbought

        if direction is None:
            return (None, 0.0)

        # ---- Volume spike confirmation ----
        vol_ratio = volume / self._vol_ema
        if vol_ratio < self.vol_spike_mult:
            return (None, 0.0)

        # ---- Stochastic exhaustion confirmation ----
        if direction == "UP" and self.stoch_k > self.stoch_exhaustion_lo:
            return (None, 0.0)
        if direction == "DOWN" and self.stoch_k < self.stoch_exhaustion_hi:
            return (None, 0.0)

        # ---- Anti-chase filter: reject if extreme has persisted too long ----
        # Long-running extremes suggest a persistent trend, not a snap-back
        if direction == "UP" and self._consecutive_oversold > self._max_consecutive_extreme:
            return (None, 0.0)
        if direction == "DOWN" and self._consecutive_overbought > self._max_consecutive_extreme:
            return (None, 0.0)

        # ---- RSI deceleration check: RSI should be flattening or reversing ----
        # If RSI is still accelerating into the extreme, the move may continue
        if len(self._rsi_history) >= 3:
            rsi_vals = list(self._rsi_history)
            recent_delta = rsi_vals[-1] - rsi_vals[-2]
            prev_delta = rsi_vals[-2] - rsi_vals[-3]
            if direction == "UP":
                # For UP reversal, RSI should stop falling (delta getting less negative)
                if recent_delta < prev_delta and recent_delta < -1.0:
                    return (None, 0.0)  # Still accelerating down
            elif direction == "DOWN":
                # For DOWN reversal, RSI should stop rising
                if recent_delta > prev_delta and recent_delta > 1.0:
                    return (None, 0.0)  # Still accelerating up

        # ---- Cooldown ----
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ---- Confidence calculation ----
        confidence = 0.70

        # RSI extremity bonus: deeper extreme = higher confidence in snap-back
        rsi_bonus = min(0.10, rsi_extremity / 50.0)
        confidence += rsi_bonus

        # Volume spike strength bonus: larger spike = more climactic
        vol_bonus = min(0.06, (vol_ratio - self.vol_spike_mult) * 0.02)
        confidence += vol_bonus

        # BB penetration bonus: further beyond band = more stretched
        if direction == "UP":
            bb_range = self.bb_upper - self.bb_lower
            if bb_range > 1e-10:
                penetration = (self.bb_lower - close) / bb_range
                bb_bonus = min(0.06, max(0, penetration) * 0.3)
                confidence += bb_bonus
        elif direction == "DOWN":
            bb_range = self.bb_upper - self.bb_lower
            if bb_range > 1e-10:
                penetration = (close - self.bb_upper) / bb_range
                bb_bonus = min(0.06, max(0, penetration) * 0.3)
                confidence += bb_bonus

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

    def _update_volume(self, volume: float):
        self._vol_ema_seed.append(volume)
        if self._vol_ema is None:
            if len(self._vol_ema_seed) >= self.vol_ema_period:
                self._vol_ema = sum(self._vol_ema_seed) / self.vol_ema_period
        else:
            self._vol_ema = calc_ema(self._vol_ema, volume, self.vol_ema_period)

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
