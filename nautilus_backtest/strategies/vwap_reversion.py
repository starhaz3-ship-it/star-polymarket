"""
VWAP_REVERSION strategy -- Snap back to institutional fair value.

Processes BTC 1-min bars.  Computes rolling session VWAP (volume-weighted
average price) and standard deviation bands.  Signals when price deviates
beyond 2 standard deviations from VWAP and shows signs of reverting back.

VWAP is THE institutional benchmark -- when price overshoots it significantly
and then turns, the snap-back is high probability.  This strategy requires
RSI confirmation and volume spike to ensure we catch genuine reversals.
"""
from collections import deque
import math


def calc_ema(prev_ema, new_value, period):
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


class VwapReversion:
    """Mean reversion to VWAP at extreme deviations.

    Signal fires when:
    1. Price > VWAP + 2σ or < VWAP - 2σ (extreme deviation)
    2. Previous bar was more extreme than current (price turning back)
    3. RSI(14) confirms extreme (< 30 or > 70)
    4. Volume > 1.5x EMA(20) (climactic activity at extreme)

    Direction: back toward VWAP (mean reversion)
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"VWAP_REVERSION_{horizon_bars}m"

        # ---- VWAP params ----
        self.vwap_period = params.get("vwap_period", 120)  # 2-hour rolling window
        self.vwap_band_mult = params.get("vwap_band_mult", 2.0)
        self._tp_vol_hist = deque(maxlen=self.vwap_period)  # (typical_price, volume)
        self.vwap = 0.0
        self.vwap_upper = 0.0
        self.vwap_lower = 0.0
        self.vwap_std = 0.0

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_extreme_lo = params.get("rsi_extreme_lo", 30.0)
        self.rsi_extreme_hi = params.get("rsi_extreme_hi", 70.0)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._prev_close = None
        self.rsi_value = 50.0

        # ---- Volume EMA(20) ----
        self.vol_ema_period = params.get("vol_ema_period", 20)
        self.vol_spike_mult = params.get("vol_spike_mult", 1.5)
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=self.vol_ema_period)

        # ---- Price turning detection ----
        self._prev_deviation = 0.0  # prev bar's (close - vwap) / std
        self._prev_high = None
        self._prev_low = None

        # ---- EMA(20) for trend context ----
        self._ema20 = None
        self._ema20_seed = deque(maxlen=20)

        # ---- Consecutive extreme bars ----
        self._consecutive_extreme = 0
        self._max_consecutive = params.get("max_consecutive", 10)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.vwap_period, self.rsi_period, self.vol_ema_period) + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar. Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Update all indicators
        self._update_vwap(high, low, close, volume)
        self._update_rsi(close)
        self._update_volume(volume)
        self._update_ema(close)

        # Wait for warmup
        if self._bar_count < self._warmup:
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            return (None, 0.0)

        if self.vwap_std < 1e-10 or self._vol_ema is None or self._vol_ema < 1e-10:
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            return (None, 0.0)

        # ---- Calculate deviation from VWAP ----
        deviation = (close - self.vwap) / self.vwap_std

        # Track consecutive extreme bars
        if abs(deviation) > self.vwap_band_mult:
            self._consecutive_extreme += 1
        else:
            self._consecutive_extreme = 0

        # ---- Check for extreme deviation ----
        direction = None
        dev_extremity = 0.0

        if deviation < -self.vwap_band_mult:
            # Price below lower VWAP band -- potential UP reversal
            direction = "UP"
            dev_extremity = abs(deviation) - self.vwap_band_mult
        elif deviation > self.vwap_band_mult:
            # Price above upper VWAP band -- potential DOWN reversal
            direction = "DOWN"
            dev_extremity = abs(deviation) - self.vwap_band_mult

        if direction is None:
            self._prev_deviation = deviation
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            return (None, 0.0)

        # ---- Price must be turning back toward VWAP ----
        # Current deviation should be LESS extreme than previous
        turning = False
        if self._prev_deviation != 0.0:
            if direction == "UP":
                # Deviation should be getting less negative (price rising)
                turning = deviation > self._prev_deviation
            elif direction == "DOWN":
                # Deviation should be getting less positive (price falling)
                turning = deviation < self._prev_deviation

        if not turning:
            self._prev_deviation = deviation
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            return (None, 0.0)

        # ---- RSI confirmation ----
        if direction == "UP" and self.rsi_value > self.rsi_extreme_lo:
            self._prev_deviation = deviation
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            return (None, 0.0)
        if direction == "DOWN" and self.rsi_value < self.rsi_extreme_hi:
            self._prev_deviation = deviation
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            return (None, 0.0)

        # ---- Volume confirmation ----
        vol_ratio = volume / self._vol_ema
        if vol_ratio < self.vol_spike_mult:
            self._prev_deviation = deviation
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            return (None, 0.0)

        # ---- Anti-chase: reject if extreme has persisted too long ----
        if self._consecutive_extreme > self._max_consecutive:
            self._prev_deviation = deviation
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            return (None, 0.0)

        # ---- Cooldown ----
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            self._prev_deviation = deviation
            self._prev_close = close
            self._prev_high = high
            self._prev_low = low
            return (None, 0.0)

        # ---- Confidence calculation ----
        confidence = 0.72

        # Deviation severity bonus
        dev_bonus = min(0.08, dev_extremity * 0.04)
        confidence += dev_bonus

        # Volume spike strength bonus
        vol_bonus = min(0.06, (vol_ratio - self.vol_spike_mult) * 0.02)
        confidence += vol_bonus

        # RSI extremity bonus
        if direction == "UP":
            rsi_bonus = min(0.06, (self.rsi_extreme_lo - self.rsi_value) / 60.0)
        else:
            rsi_bonus = min(0.06, (self.rsi_value - self.rsi_extreme_hi) / 60.0)
        confidence += max(0, rsi_bonus)

        # EMA alignment bonus (price reverting toward EMA = stronger)
        if self._ema20 is not None:
            if direction == "UP" and close < self._ema20:
                confidence += 0.03  # below EMA, reverting up = strong
            elif direction == "DOWN" and close > self._ema20:
                confidence += 0.03

        confidence = max(0.65, min(confidence, 0.95))

        self._last_signal_bar = self._bar_count
        self._prev_deviation = deviation
        self._prev_close = close
        self._prev_high = high
        self._prev_low = low
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_vwap(self, high, low, close, volume):
        typical_price = (high + low + close) / 3.0
        self._tp_vol_hist.append((typical_price, volume))

        if len(self._tp_vol_hist) < 20:  # need at least 20 bars
            return

        total_tpv = sum(tp * v for tp, v in self._tp_vol_hist)
        total_vol = sum(v for _, v in self._tp_vol_hist)

        if total_vol < 1e-10:
            return

        self.vwap = total_tpv / total_vol

        # Standard deviation of typical prices from VWAP
        variance = sum(v * (tp - self.vwap) ** 2 for tp, v in self._tp_vol_hist) / total_vol
        self.vwap_std = math.sqrt(variance) if variance > 0 else 0.0

        self.vwap_upper = self.vwap + self.vwap_band_mult * self.vwap_std
        self.vwap_lower = self.vwap - self.vwap_band_mult * self.vwap_std

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

    def _update_ema(self, close):
        self._ema20_seed.append(close)
        if self._ema20 is None:
            if len(self._ema20_seed) >= 20:
                self._ema20 = sum(self._ema20_seed) / 20
        else:
            self._ema20 = calc_ema(self._ema20, close, 20)
