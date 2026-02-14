"""
MOMENTUM_REGIME strategy -- Pullback-in-trend for Polymarket binary options.

Processes BTC 1-min bars.  Only trades when ADX(14) confirms a strong trend
(ADX > 25 with clear DI separation > 5), then waits for RSI(14) to pull back
into the 40-60 neutral zone and cross back out in the trend direction.

This captures the highest-probability pattern: buying pullbacks in established
trends.  Very selective -- only fires in strong trends with confirmed pullbacks.
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


def calc_ema(prev_ema, new_value, period):
    """Incremental EMA update."""
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


# ---------------------------------------------------------------------------
# MOMENTUM_REGIME Strategy
# ---------------------------------------------------------------------------

class MomentumRegime:
    """Trade pullbacks in established trends using ADX + DI + RSI.

    A signal fires when:
    1. ADX(14) > 25 (strong trend regime)
    2. +DI/-DI separation > 5 (clear directional bias)
    3. RSI(14) has pulled back into the 40-60 neutral zone
    4. RSI crosses back out in the trend direction (leaves the neutral zone)

    Direction:
    - Bullish trend (+DI > -DI): RSI crosses above 60 after visiting 40-60
    - Bearish trend (-DI > +DI): RSI crosses below 40 after visiting 40-60
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"MOMENTUM_REGIME_{horizon_bars}m"

        # ---- ADX(14) / DI parameters ----
        self.adx_period = params.get("adx_period", 14)
        self.adx_threshold = params.get("adx_threshold", 25.0)
        self.di_separation = params.get("di_separation", 5.0)

        # ADX state
        self._adx_highs = deque(maxlen=2)
        self._adx_lows = deque(maxlen=2)
        self._adx_closes = deque(maxlen=2)
        self._smooth_plus_dm = 0.0
        self._smooth_minus_dm = 0.0
        self._smooth_tr = 0.0
        self._dx_hist = deque(maxlen=self.adx_period)
        self._adx_value = 0.0
        self._plus_di = 0.0
        self._minus_di = 0.0
        self._adx_count = 0

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._prev_close = None
        self.rsi_value = 50.0
        self._prev_rsi = 50.0

        # RSI pullback tracking
        self._rsi_was_in_neutral = False  # RSI has been in the 40-60 zone
        self._neutral_lo = params.get("rsi_neutral_lo", 40.0)
        self._neutral_hi = params.get("rsi_neutral_hi", 60.0)
        self._neutral_bar_count = 0  # how many bars RSI spent in neutral zone
        self._min_neutral_bars = params.get("min_neutral_bars", 3)  # require 3+ bars

        # ---- EMA(20) for trend confirmation ----
        self._ema20 = None
        self._ema20_seed = deque(maxlen=20)

        # ---- EMA(50) for higher-timeframe trend ----
        self._ema50 = None
        self._ema50_seed = deque(maxlen=50)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = self.adx_period * 3 + 10  # ADX needs ~42 bars to stabilize

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Save previous RSI for crossover detection
        self._prev_rsi = self.rsi_value

        # Update all indicators
        self._update_adx(high, low, close)
        self._update_rsi(close)
        self._update_emas(close)

        self._prev_close = close

        # Track RSI time in neutral zone
        in_neutral = self._neutral_lo <= self.rsi_value <= self._neutral_hi
        if in_neutral:
            self._neutral_bar_count += 1
            self._rsi_was_in_neutral = True
        else:
            # RSI left the neutral zone -- check for signal, then reset
            pass

        # Wait for warmup
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # ---- Regime filter: ADX must show strong trend ----
        if self._adx_value < self.adx_threshold:
            # Not in a strong trend -- reset pullback state and skip
            if not in_neutral:
                self._rsi_was_in_neutral = False
                self._neutral_bar_count = 0
            return (None, 0.0)

        # ---- DI separation must be clear ----
        di_gap = abs(self._plus_di - self._minus_di)
        if di_gap < self.di_separation:
            if not in_neutral:
                self._rsi_was_in_neutral = False
                self._neutral_bar_count = 0
            return (None, 0.0)

        # ---- Determine trend direction from DI ----
        bullish_trend = self._plus_di > self._minus_di
        bearish_trend = self._minus_di > self._plus_di

        # ---- Check for RSI pullback completion (crossover out of neutral) ----
        if not self._rsi_was_in_neutral:
            return (None, 0.0)

        if self._neutral_bar_count < self._min_neutral_bars:
            # Pullback was too brief -- not a real pullback
            if not in_neutral:
                self._rsi_was_in_neutral = False
                self._neutral_bar_count = 0
            return (None, 0.0)

        direction = None

        # Bullish: RSI crosses above 60 (leaving neutral zone upward) in uptrend
        if bullish_trend:
            crossed_up = self._prev_rsi <= self._neutral_hi and self.rsi_value > self._neutral_hi
            if crossed_up:
                direction = "UP"

        # Bearish: RSI crosses below 40 (leaving neutral zone downward) in downtrend
        if bearish_trend:
            crossed_down = self._prev_rsi >= self._neutral_lo and self.rsi_value < self._neutral_lo
            if crossed_down:
                direction = "DOWN"

        if direction is None:
            # Still waiting for the crossover, or RSI went the wrong way
            if not in_neutral:
                # RSI left neutral in the wrong direction -- reset
                self._rsi_was_in_neutral = False
                self._neutral_bar_count = 0
            return (None, 0.0)

        # Reset pullback tracking after signal
        self._rsi_was_in_neutral = False
        self._neutral_bar_count = 0

        # ---- EMA alignment guard: price should be on correct side of EMA(20) ----
        if self._ema20 is not None:
            if direction == "UP" and close < self._ema20:
                return (None, 0.0)
            if direction == "DOWN" and close > self._ema20:
                return (None, 0.0)

        # ---- Cooldown ----
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ---- Confidence calculation ----
        # Base confidence
        confidence = 0.68

        # ADX strength bonus: stronger trend = higher confidence
        adx_bonus = min(0.08, (self._adx_value - self.adx_threshold) / 100.0)
        confidence += adx_bonus

        # DI separation bonus: clearer direction = higher confidence
        di_bonus = min(0.06, (di_gap - self.di_separation) / 150.0)
        confidence += di_bonus

        # EMA(50) alignment bonus: price on correct side of EMA(50) = stronger trend
        ema50_bonus = 0.0
        if self._ema50 is not None:
            if direction == "UP" and close > self._ema50:
                ema50_bonus = 0.04
            elif direction == "DOWN" and close < self._ema50:
                ema50_bonus = 0.04
        confidence += ema50_bonus

        # Pullback depth bonus: longer time in neutral = deeper pullback = better setup
        pullback_bonus = min(0.06, self._neutral_bar_count * 0.005)
        confidence += pullback_bonus

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

    def _update_adx(self, high: float, low: float, close: float):
        self._adx_highs.append(high)
        self._adx_lows.append(low)
        self._adx_closes.append(close)

        if len(self._adx_highs) < 2:
            return

        self._adx_count += 1

        h = list(self._adx_highs)
        l = list(self._adx_lows)
        c = list(self._adx_closes)

        # True range
        tr = max(h[-1] - l[-1], abs(h[-1] - c[-2]), abs(l[-1] - c[-2]))

        # Directional movement
        up_move = h[-1] - h[-2]
        down_move = l[-2] - l[-1]

        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

        if self._adx_count <= self.adx_period:
            self._smooth_plus_dm += plus_dm
            self._smooth_minus_dm += minus_dm
            self._smooth_tr += tr
            if self._adx_count == self.adx_period:
                if self._smooth_tr > 0:
                    self._plus_di = 100.0 * self._smooth_plus_dm / self._smooth_tr
                    self._minus_di = 100.0 * self._smooth_minus_dm / self._smooth_tr
                di_sum = self._plus_di + self._minus_di
                if di_sum > 0:
                    dx = 100.0 * abs(self._plus_di - self._minus_di) / di_sum
                    self._dx_hist.append(dx)
        else:
            # Wilder smoothing
            self._smooth_plus_dm = self._smooth_plus_dm - (self._smooth_plus_dm / self.adx_period) + plus_dm
            self._smooth_minus_dm = self._smooth_minus_dm - (self._smooth_minus_dm / self.adx_period) + minus_dm
            self._smooth_tr = self._smooth_tr - (self._smooth_tr / self.adx_period) + tr

            if self._smooth_tr > 0:
                self._plus_di = 100.0 * self._smooth_plus_dm / self._smooth_tr
                self._minus_di = 100.0 * self._smooth_minus_dm / self._smooth_tr

            di_sum = self._plus_di + self._minus_di
            if di_sum > 0:
                dx = 100.0 * abs(self._plus_di - self._minus_di) / di_sum
                self._dx_hist.append(dx)

            if len(self._dx_hist) >= self.adx_period:
                if self._adx_value == 0.0:
                    self._adx_value = sum(list(self._dx_hist)[-self.adx_period:]) / self.adx_period
                else:
                    self._adx_value = (self._adx_value * (self.adx_period - 1) + dx) / self.adx_period

    def _update_emas(self, close: float):
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
