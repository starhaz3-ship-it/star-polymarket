"""
ADX_DI_CROSS strategy -- ADX trend strength + DI crossover for Polymarket binary options.

Processes BTC 1-min bars.  Computes the Average Directional Index (ADX) and
the Plus/Minus Directional Indicators (+DI/-DI) using Wilder smoothing.
Fires directional signals when ADX confirms a strong trend (>25) and a
DI crossover occurs.
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
# ADX_DI_CROSS Strategy
# ---------------------------------------------------------------------------

class AdxDiCross:
    """ADX trend filter with +DI/-DI crossover signals.

    A signal fires when:
    1. ADX(14) > 25  (confirms strong, trending market)
    2. +DI crosses above -DI  -> UP  (bullish momentum taking control)
    3. -DI crosses above +DI  -> DOWN (bearish momentum taking control)

    The ADX threshold ensures we only trade in trending conditions where
    directional crossovers are meaningful rather than noise.
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"ADX_DI_CROSS_{horizon_bars}m"

        # ---- ADX parameters ----
        self.adx_period = params.get("adx_period", 14)
        self.adx_threshold = params.get("adx_threshold", 25.0)

        # ---- Wilder-smoothed ADX state ----
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None

        self._smooth_plus_dm = 0.0
        self._smooth_minus_dm = 0.0
        self._smooth_tr = 0.0

        self._plus_di = 0.0
        self._minus_di = 0.0
        self._prev_plus_di = 0.0
        self._prev_minus_di = 0.0

        self._dx_hist = deque(maxlen=self.adx_period)
        self._adx_value = 0.0
        self._adx_count = 0  # bars processed for ADX accumulation

        # ---- EMA(20) for trend confirmation ----
        self._ema20 = None
        self._ema20_seed = deque(maxlen=20)

        # ---- EMA(50) for higher-timeframe alignment ----
        self._ema50 = None
        self._ema50_seed = deque(maxlen=50)

        # ---- Volume EMA(20) for optional confirmation ----
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=20)

        # ---- ATR(14) for volatility context ----
        self._tr_hist = deque(maxlen=self.adx_period)
        self._atr_value = 0.0

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = self.adx_period * 3 + 10  # ADX needs ~52 bars to stabilize

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Save previous DI values for crossover detection
        self._prev_plus_di = self._plus_di
        self._prev_minus_di = self._minus_di

        # Update all indicators
        self._update_adx(high, low, close)
        self._update_atr(high, low, close)
        self._update_emas(close)
        self._update_vol_ema(volume)

        # Store current bar for next iteration
        self._prev_high = high
        self._prev_low = low
        self._prev_close = close

        # ---- Warmup check ----
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # ---- Cooldown ----
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ---- ADX must confirm strong trend ----
        if self._adx_value < self.adx_threshold:
            return (None, 0.0)

        # ---- DI crossover detection ----
        direction = None

        # +DI crosses above -DI -> bullish
        plus_crossed_above = (self._prev_plus_di <= self._prev_minus_di and
                              self._plus_di > self._minus_di)
        # -DI crosses above +DI -> bearish
        minus_crossed_above = (self._prev_minus_di <= self._prev_plus_di and
                               self._minus_di > self._plus_di)

        if plus_crossed_above:
            direction = "UP"
        elif minus_crossed_above:
            direction = "DOWN"

        if direction is None:
            return (None, 0.0)

        # ---- EMA alignment guard ----
        # Price should be on the correct side of EMA(20) for the signal direction
        if self._ema20 is not None:
            if direction == "UP" and close < self._ema20:
                return (None, 0.0)
            if direction == "DOWN" and close > self._ema20:
                return (None, 0.0)

        # ---- Confidence calculation ----
        confidence = 0.68

        # ADX strength bonus: stronger trend = more reliable crossover
        adx_bonus = min(0.08, (self._adx_value - self.adx_threshold) / 100.0)
        confidence += adx_bonus

        # DI separation bonus: wider gap after cross = stronger conviction
        di_gap = abs(self._plus_di - self._minus_di)
        di_bonus = min(0.06, di_gap / 80.0)
        confidence += di_bonus

        # EMA(50) alignment bonus
        if self._ema50 is not None:
            if direction == "UP" and close > self._ema50:
                confidence += 0.04
            elif direction == "DOWN" and close < self._ema50:
                confidence += 0.04

        # Volume confirmation bonus
        if self._vol_ema is not None and self._vol_ema > 0:
            vol_ratio = volume / self._vol_ema
            if vol_ratio > 1.2:
                vol_bonus = min(0.04, (vol_ratio - 1.2) * 0.04)
                confidence += vol_bonus

        # ADX acceleration bonus (ADX itself is rising = trend strengthening)
        # We approximate this by checking if ADX > 30 (well above threshold)
        if self._adx_value > 35.0:
            confidence += 0.02

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_adx(self, high: float, low: float, close: float):
        """Update ADX, +DI, -DI using Wilder smoothing."""
        if self._prev_high is None:
            return

        self._adx_count += 1

        # True range
        tr = max(high - low,
                 abs(high - self._prev_close),
                 abs(low - self._prev_close))

        # Directional movement
        up_move = high - self._prev_high
        down_move = self._prev_low - low

        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        if self._adx_count <= self.adx_period:
            # Accumulation phase
            self._smooth_plus_dm += plus_dm
            self._smooth_minus_dm += minus_dm
            self._smooth_tr += tr

            if self._adx_count == self.adx_period:
                # First DI values
                if self._smooth_tr > 0:
                    self._plus_di = 100.0 * self._smooth_plus_dm / self._smooth_tr
                    self._minus_di = 100.0 * self._smooth_minus_dm / self._smooth_tr
                di_sum = self._plus_di + self._minus_di
                if di_sum > 0:
                    dx = 100.0 * abs(self._plus_di - self._minus_di) / di_sum
                    self._dx_hist.append(dx)
        else:
            # Wilder smoothing
            self._smooth_plus_dm = (self._smooth_plus_dm -
                                    self._smooth_plus_dm / self.adx_period + plus_dm)
            self._smooth_minus_dm = (self._smooth_minus_dm -
                                     self._smooth_minus_dm / self.adx_period + minus_dm)
            self._smooth_tr = (self._smooth_tr -
                               self._smooth_tr / self.adx_period + tr)

            if self._smooth_tr > 0:
                self._plus_di = 100.0 * self._smooth_plus_dm / self._smooth_tr
                self._minus_di = 100.0 * self._smooth_minus_dm / self._smooth_tr

            di_sum = self._plus_di + self._minus_di
            if di_sum > 0:
                dx = 100.0 * abs(self._plus_di - self._minus_di) / di_sum
                self._dx_hist.append(dx)

            if len(self._dx_hist) >= self.adx_period:
                if self._adx_value == 0.0:
                    # First ADX = SMA of DX values
                    self._adx_value = sum(list(self._dx_hist)[-self.adx_period:]) / self.adx_period
                else:
                    # Wilder smoothing of ADX
                    self._adx_value = ((self._adx_value * (self.adx_period - 1) + dx) /
                                       self.adx_period)

    def _update_atr(self, high: float, low: float, close: float):
        """Update ATR(14) incrementally."""
        if self._prev_close is not None:
            tr = max(high - low,
                     abs(high - self._prev_close),
                     abs(low - self._prev_close))
        else:
            tr = high - low
        self._tr_hist.append(tr)

        if len(self._tr_hist) >= self.adx_period:
            if self._atr_value == 0.0:
                self._atr_value = sum(self._tr_hist) / len(self._tr_hist)
            else:
                self._atr_value = ((self._atr_value * (self.adx_period - 1) + tr) /
                                   self.adx_period)

    def _update_emas(self, close: float):
        """Update EMA(20) and EMA(50) incrementally."""
        # EMA(20)
        self._ema20_seed.append(close)
        if self._ema20 is None:
            if len(self._ema20_seed) >= 20:
                self._ema20 = sum(self._ema20_seed) / 20
        else:
            self._ema20 = calc_ema(self._ema20, close, 20)

        # EMA(50)
        self._ema50_seed.append(close)
        if self._ema50 is None:
            if len(self._ema50_seed) >= 50:
                self._ema50 = sum(self._ema50_seed) / 50
        else:
            self._ema50 = calc_ema(self._ema50, close, 50)

    def _update_vol_ema(self, volume: float):
        """Update volume EMA(20) incrementally."""
        self._vol_ema_seed.append(volume)
        if self._vol_ema is None:
            if len(self._vol_ema_seed) >= 20:
                self._vol_ema = sum(self._vol_ema_seed) / 20
        else:
            self._vol_ema = calc_ema(self._vol_ema, volume, 20)
