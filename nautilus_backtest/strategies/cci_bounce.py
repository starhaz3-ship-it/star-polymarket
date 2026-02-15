"""
CCI_BOUNCE strategy -- CCI extreme reversal with RSI + volume confirmation.

Processes BTC 1-min bars and fires when the Commodity Channel Index (CCI-20)
crosses back from an extreme zone (beyond +/-200). A CCI reading beyond +/-200
is a 2-sigma event on the typical price distribution -- when it snaps back inside,
it signals a high-probability mean reversion.

Triple confirmation required:
1. CCI(20) crosses back from extreme (was beyond +/-200, now inside)
2. RSI(14) confirms direction (below 35 for UP, above 65 for DOWN)
3. Volume exceeds 1.3x EMA(20) -- institutional participation

This triple gate is deliberately strict to keep signal count low and win rate high.
"""
from collections import deque
import math


def calc_ema(prev_ema, new_value, period):
    """Incremental EMA update. Returns the new EMA value."""
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


def calc_sma(values, period):
    """Simple moving average from the last *period* items of an iterable."""
    vals = list(values)[-period:]
    if len(vals) < period:
        return None
    return sum(vals) / period


def calc_mean_dev(values, mean, period):
    """Mean absolute deviation of the last *period* items from mean."""
    vals = list(values)[-period:]
    if len(vals) < period or mean is None:
        return None
    return sum(abs(x - mean) for x in vals) / period


class CciBounce:
    """CCI(20) reversal from extreme with RSI and volume confirmation.

    Signal fires when:
    1. CCI was below -200 and crosses above -200 -> UP (oversold bounce)
    2. CCI was above +200 and crosses below +200 -> DOWN (overbought reversal)
    3. RSI(14) must confirm: < 35 for UP, > 65 for DOWN
    4. Volume must be > 1.3x EMA(20) for institutional confirmation
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"CCI_BOUNCE_{horizon_bars}m"

        # ---- CCI(20) ----
        self.cci_period = params.get("cci_period", 20)
        self.cci_extreme = params.get("cci_extreme", 200.0)
        self._tp_hist = deque(maxlen=self.cci_period)  # typical prices
        self.cci_value = 0.0
        self._prev_cci = 0.0

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_confirm_lo = params.get("rsi_confirm_lo", 35.0)
        self.rsi_confirm_hi = params.get("rsi_confirm_hi", 65.0)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._prev_close = None
        self.rsi_value = 50.0

        # ---- Volume EMA(20) ----
        self.vol_ema_period = params.get("vol_ema_period", 20)
        self.vol_confirm_mult = params.get("vol_confirm_mult", 1.3)
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=self.vol_ema_period)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.cci_period, self.rsi_period, self.vol_ema_period) + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar. Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Save previous CCI before updating
        self._prev_cci = self.cci_value

        # Update indicators
        self._update_cci(high, low, close)
        self._update_rsi(close)
        self._update_volume(volume)

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

        direction = None
        confidence = 0.0

        # ---- Bullish: CCI crosses UP through -200 (was below, now above) ----
        if self._prev_cci < -self.cci_extreme and self.cci_value >= -self.cci_extreme:
            # RSI must confirm oversold
            if self.rsi_value < self.rsi_confirm_lo:
                direction = "UP"
                # Confidence scales with how deep the CCI was
                cci_depth = abs(self._prev_cci) - self.cci_extreme
                confidence = 0.68
                confidence += min(0.10, cci_depth * 0.001)

                # RSI depth bonus
                rsi_bonus = (self.rsi_confirm_lo - self.rsi_value) / 35.0
                confidence += min(0.06, rsi_bonus * 0.06)

        # ---- Bearish: CCI crosses DOWN through +200 (was above, now below) ----
        elif self._prev_cci > self.cci_extreme and self.cci_value <= self.cci_extreme:
            # RSI must confirm overbought
            if self.rsi_value > self.rsi_confirm_hi:
                direction = "DOWN"
                cci_depth = abs(self._prev_cci) - self.cci_extreme
                confidence = 0.68
                confidence += min(0.10, cci_depth * 0.001)

                rsi_bonus = (self.rsi_value - self.rsi_confirm_hi) / 35.0
                confidence += min(0.06, rsi_bonus * 0.06)

        if direction is None:
            return (None, 0.0)

        # Volume spike bonus
        vol_ratio = volume / self._vol_ema
        if vol_ratio > 1.8:
            confidence += min(0.06, (vol_ratio - 1.8) * 0.04)

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_cci(self, high, low, close):
        """CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)."""
        tp = (high + low + close) / 3.0
        self._tp_hist.append(tp)

        if len(self._tp_hist) < self.cci_period:
            return

        sma = calc_sma(self._tp_hist, self.cci_period)
        if sma is None:
            return

        mean_dev = calc_mean_dev(self._tp_hist, sma, self.cci_period)
        if mean_dev is None or mean_dev < 1e-10:
            self.cci_value = 0.0
            return

        self.cci_value = (tp - sma) / (0.015 * mean_dev)

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
