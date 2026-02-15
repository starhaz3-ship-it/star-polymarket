"""
WILLIAMS_VWAP strategy -- Williams %R + VWAP deviation mean reversion.

Processes BTC 1-min bars and fires when Williams %R(14) reaches an extreme
(<-90 oversold or >-10 overbought) simultaneously with price deviating from
the rolling 60-bar VWAP by at least 0.15%. This double-filter catches
high-conviction mean-reversion setups where both momentum and price-relative-
to-fair-value agree the move is overdone.

An ATR% gate (0.12%-2.8%) excludes dead markets (no edge) and extreme
volatility (stop-loss risk too high for binary options sizing).

Expected edge: VWAP is institutional fair value; Williams %R extremes confirm
momentum exhaustion. Combining both greatly reduces false signals vs either alone.
"""
from collections import deque
import math


def calc_ema(prev_ema, new_value, period):
    """Incremental EMA update. Returns the new EMA value."""
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


class WilliamsVwap:
    """Williams %R(14) at extreme + VWAP deviation mean-reversion strategy.

    Signal fires when:
    1. Williams %R < -90 AND price < VWAP - 0.15% -> UP (oversold + below fair value)
    2. Williams %R > -10 AND price > VWAP + 0.15% -> DOWN (overbought + above fair value)
    3. ATR% must be between 0.12% and 2.8% (volatility gate)
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"WILLIAMS_VWAP_{horizon_bars}m"

        # ---- Williams %R(14) ----
        self.wr_period = params.get("wr_period", 14)
        self.wr_oversold = params.get("wr_oversold", -90.0)
        self.wr_overbought = params.get("wr_overbought", -10.0)
        self._wr_highs = deque(maxlen=self.wr_period)
        self._wr_lows = deque(maxlen=self.wr_period)
        self.wr_value = -50.0

        # ---- Rolling VWAP(60) ----
        self.vwap_period = params.get("vwap_period", 60)
        self.vwap_deviation_pct = params.get("vwap_deviation_pct", 0.0015)  # 0.15%
        self._vwap_tp_vol = deque(maxlen=self.vwap_period)   # (typical_price * volume)
        self._vwap_vol = deque(maxlen=self.vwap_period)      # volume
        self.vwap_value = 0.0

        # ---- ATR(14) for volatility gate ----
        self.atr_period = params.get("atr_period", 14)
        self.atr_min_pct = params.get("atr_min_pct", 0.0012)   # 0.12%
        self.atr_max_pct = params.get("atr_max_pct", 0.028)    # 2.8%
        self._tr_hist = deque(maxlen=self.atr_period)
        self._atr = None

        # ---- Misc ----
        self._prev_close = None
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.wr_period, self.vwap_period, self.atr_period) + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar. Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Update all indicators
        self._update_williams_r(high, low, close)
        self._update_vwap(high, low, close, volume)
        self._update_atr(high, low, close)

        self._prev_close = close

        # Warmup check
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # Cooldown check
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ATR% volatility gate
        if self._atr is None or close <= 0:
            return (None, 0.0)
        atr_pct = self._atr / close
        if atr_pct < self.atr_min_pct or atr_pct > self.atr_max_pct:
            return (None, 0.0)

        # VWAP check
        if self.vwap_value <= 0:
            return (None, 0.0)

        vwap_lower = self.vwap_value * (1.0 - self.vwap_deviation_pct)
        vwap_upper = self.vwap_value * (1.0 + self.vwap_deviation_pct)

        direction = None
        confidence = 0.0

        # ---- Bullish: oversold %R + price below VWAP ----
        if self.wr_value <= self.wr_oversold and close <= vwap_lower:
            direction = "UP"
            # Confidence scales with how extreme the readings are
            wr_depth = abs(self.wr_value - self.wr_oversold)
            vwap_dist = (self.vwap_value - close) / self.vwap_value  # positive when below
            confidence = 0.68
            confidence += min(0.08, wr_depth * 0.008)       # %R depth bonus
            confidence += min(0.08, vwap_dist * 200.0)      # VWAP distance bonus

        # ---- Bearish: overbought %R + price above VWAP ----
        elif self.wr_value >= self.wr_overbought and close >= vwap_upper:
            direction = "DOWN"
            wr_depth = abs(self.wr_value - self.wr_overbought)
            vwap_dist = (close - self.vwap_value) / self.vwap_value  # positive when above
            confidence = 0.68
            confidence += min(0.08, wr_depth * 0.008)
            confidence += min(0.08, vwap_dist * 200.0)

        if direction is None:
            return (None, 0.0)

        # ATR sweet-spot bonus: mid-range volatility is best
        atr_mid = (self.atr_min_pct + self.atr_max_pct) / 2.0
        atr_score = 1.0 - abs(atr_pct - atr_mid) / (atr_mid - self.atr_min_pct)
        confidence += max(0.0, min(0.04, atr_score * 0.04))

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_williams_r(self, high, low, close):
        """Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100."""
        self._wr_highs.append(high)
        self._wr_lows.append(low)

        if len(self._wr_highs) < self.wr_period:
            return

        hh = max(self._wr_highs)
        ll = min(self._wr_lows)
        rng = hh - ll

        if rng < 1e-10:
            self.wr_value = -50.0
        else:
            self.wr_value = ((hh - close) / rng) * -100.0

    def _update_vwap(self, high, low, close, volume):
        """Rolling VWAP = sum(TP * volume) / sum(volume) over lookback window."""
        typical_price = (high + low + close) / 3.0
        self._vwap_tp_vol.append(typical_price * volume)
        self._vwap_vol.append(volume)

        total_vol = sum(self._vwap_vol)
        if total_vol > 0 and len(self._vwap_vol) >= self.vwap_period:
            self.vwap_value = sum(self._vwap_tp_vol) / total_vol
        elif total_vol > 0:
            self.vwap_value = sum(self._vwap_tp_vol) / total_vol

    def _update_atr(self, high, low, close):
        if self._prev_close is not None:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        else:
            tr = high - low
        self._tr_hist.append(tr)

        if len(self._tr_hist) >= self.atr_period:
            if self._atr is None:
                self._atr = sum(self._tr_hist) / self.atr_period
            else:
                self._atr = (self._atr * (self.atr_period - 1) + tr) / self.atr_period
