"""
VWAP_ST_SR strategy -- VWAP pullback + Supertrend regime + StochRSI trigger.

Processes BTC 1-min bars. Three-layer confirmation:
1. Supertrend (ATR-based) defines the regime (bullish/bearish)
2. Price pulls back to VWAP against the trend (mean-reversion entry)
3. StochRSI K/D cross from oversold/overbought zone triggers entry WITH trend

This is a trend-following pullback strategy: only enter in the direction of
the Supertrend regime, but wait for a pullback to VWAP and a StochRSI cross
to confirm the pullback is over and trend is resuming.
"""
from collections import deque
import math


def _ema_update(prev, value, period):
    alpha = 2.0 / (period + 1)
    return alpha * value + (1 - alpha) * prev


class VwapSupertrendStochRSI:
    """VWAP pullback + Supertrend regime + StochRSI cross trigger.

    Signal fires when:
    1. Supertrend is bullish (UP) or bearish (DOWN) -- defines regime
    2. Price has pulled back to/past VWAP against the trend
       (in uptrend: price <= VWAP - dev_threshold)
       (in downtrend: price >= VWAP + dev_threshold)
    3. StochRSI K crosses D from oversold (<20 for longs) or overbought (>80 for shorts)
    4. ATR% is within sane range (not too flat, not too volatile)

    Direction: WITH the Supertrend trend (trend continuation after pullback)
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"VWAP_ST_SR_{horizon_bars}m"

        # VWAP params
        self.vwap_len = params.get("vwap_len", 48)  # ~48 bars = 48 min rolling window
        self._tp_vol_hist = deque(maxlen=self.vwap_len)

        # Supertrend params
        self.st_atr_len = params.get("st_atr_len", 10)
        self.st_mult = params.get("st_mult", 2.5)
        self._st_upper = None
        self._st_lower = None
        self._st_trend = 0  # 1 = bullish, -1 = bearish
        self._st_line = None

        # ATR for supertrend (SMA-based)
        self._tr_hist = deque(maxlen=self.st_atr_len)

        # ATR for volatility filter
        self.atr_len = params.get("atr_len", 14)
        self._atr_hist = deque(maxlen=self.atr_len)
        self.atrp_min = params.get("atrp_min", 0.0012)
        self.atrp_max = params.get("atrp_max", 0.028)

        # StochRSI params
        self.rsi_len = params.get("rsi_len", 14)
        self.stoch_len = params.get("stoch_len", 14)
        self.k_smooth = params.get("k_smooth", 3)
        self.d_smooth = params.get("d_smooth", 3)

        # RSI state
        self._rsi_gains = deque(maxlen=self.rsi_len)
        self._rsi_losses = deque(maxlen=self.rsi_len)
        self._rsi_val = 50.0
        self._rsi_hist = deque(maxlen=self.stoch_len)

        # StochRSI K/D state
        self._k_hist = deque(maxlen=self.k_smooth)
        self._k_val = 50.0
        self._k_prev = 50.0
        self._d_hist = deque(maxlen=self.d_smooth)
        self._d_val = 50.0
        self._d_prev = 50.0

        # Entry thresholds
        self.vwap_dev_pull = params.get("vwap_dev_pull", 0.0020)  # 0.20% pullback required
        self.k_cross_long = params.get("k_cross_long", 20.0)
        self.k_cross_short = params.get("k_cross_short", 80.0)

        # Previous close for TR calculation
        self._prev_close = None
        self._bar_count = 0

    def _compute_tr(self, high, low, close):
        if self._prev_close is None:
            return high - low
        return max(
            high - low,
            abs(high - self._prev_close),
            abs(low - self._prev_close),
        )

    def _update_rsi(self, close):
        if self._prev_close is None:
            return
        delta = close - self._prev_close
        gain = max(0, delta)
        loss = max(0, -delta)
        self._rsi_gains.append(gain)
        self._rsi_losses.append(loss)
        if len(self._rsi_gains) >= self.rsi_len:
            avg_gain = sum(self._rsi_gains) / self.rsi_len
            avg_loss = sum(self._rsi_losses) / self.rsi_len
            if avg_loss == 0:
                self._rsi_val = 100.0
            else:
                rs = avg_gain / avg_loss
                self._rsi_val = 100.0 - (100.0 / (1.0 + rs))

    def _update_stoch_rsi(self):
        self._rsi_hist.append(self._rsi_val)
        if len(self._rsi_hist) < self.stoch_len:
            return
        lo = min(self._rsi_hist)
        hi = max(self._rsi_hist)
        if hi == lo:
            raw_k = 50.0
        else:
            raw_k = 100.0 * (self._rsi_val - lo) / (hi - lo)

        # Smooth K (SMA)
        self._k_hist.append(raw_k)
        self._k_prev = self._k_val
        self._k_val = sum(self._k_hist) / len(self._k_hist)

        # Smooth D (SMA of K)
        self._d_hist.append(self._k_val)
        self._d_prev = self._d_val
        self._d_val = sum(self._d_hist) / len(self._d_hist)

    def _update_supertrend(self, high, low, close):
        tr = self._compute_tr(high, low, close)
        self._tr_hist.append(tr)
        if len(self._tr_hist) < self.st_atr_len:
            return

        atr_val = sum(self._tr_hist) / self.st_atr_len
        hl2 = (high + low) / 2.0
        upper = hl2 + self.st_mult * atr_val
        lower = hl2 - self.st_mult * atr_val

        # Band locking
        if self._st_upper is not None:
            if upper > self._st_upper and self._prev_close is not None and self._prev_close <= self._st_upper:
                upper = self._st_upper
        if self._st_lower is not None:
            if lower < self._st_lower and self._prev_close is not None and self._prev_close >= self._st_lower:
                lower = self._st_lower

        # Trend direction
        if self._st_trend == 0:
            # Initialize
            self._st_trend = 1
            self._st_line = lower
        elif self._st_trend == 1:
            if close < lower:
                self._st_trend = -1
                self._st_line = upper
            else:
                self._st_line = lower
        else:  # -1
            if close > upper:
                self._st_trend = 1
                self._st_line = lower
            else:
                self._st_line = upper

        self._st_upper = upper
        self._st_lower = lower

    def _compute_vwap(self):
        if len(self._tp_vol_hist) < self.vwap_len:
            return None
        total_tpv = sum(tp * v for tp, v in self._tp_vol_hist)
        total_v = sum(v for _, v in self._tp_vol_hist)
        if total_v <= 0:
            return None
        return total_tpv / total_v

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar. Returns (direction|None, confidence)."""
        self._bar_count += 1

        # Update VWAP
        tp = (high + low + close) / 3.0
        self._tp_vol_hist.append((tp, volume))

        # Update RSI + StochRSI
        self._update_rsi(close)
        self._update_stoch_rsi()

        # Update Supertrend
        self._update_supertrend(high, low, close)

        # Update ATR for vol filter
        atr_tr = self._compute_tr(high, low, close)
        self._atr_hist.append(atr_tr)

        # Save prev close for next bar
        prev = self._prev_close
        self._prev_close = close

        # Need enough data
        if self._bar_count < max(self.vwap_len, self.st_atr_len + 5, self.rsi_len + self.stoch_len + 5):
            return None, 0.0

        # Volatility filter
        if len(self._atr_hist) >= self.atr_len:
            atr_val = sum(self._atr_hist) / self.atr_len
            atrp = atr_val / max(1e-9, close)
            if atrp < self.atrp_min or atrp > self.atrp_max:
                return None, 0.0

        # VWAP
        vwap = self._compute_vwap()
        if vwap is None:
            return None, 0.0

        # Deviation from VWAP
        dev = (close - vwap) / max(1e-9, close)

        # Regime from Supertrend
        regime_up = self._st_trend == 1
        regime_dn = self._st_trend == -1

        # Pullback check: price deviates AGAINST trend toward/past VWAP
        pullback_long = regime_up and (dev <= -self.vwap_dev_pull)
        pullback_short = regime_dn and (dev >= self.vwap_dev_pull)

        # StochRSI cross
        cross_up = (self._k_prev <= self._d_prev) and (self._k_val > self._d_val) and (self._k_prev <= self.k_cross_long)
        cross_dn = (self._k_prev >= self._d_prev) and (self._k_val < self._d_val) and (self._k_prev >= self.k_cross_short)

        direction = None
        conf = 0.0

        if pullback_long and cross_up:
            direction = "UP"
            conf = 0.72 + min(0.15, abs(dev) / (self.vwap_dev_pull * 4.0)) + min(0.05, (self.k_cross_long - self._k_prev) / 100.0)
        elif pullback_short and cross_dn:
            direction = "DOWN"
            conf = 0.72 + min(0.15, abs(dev) / (self.vwap_dev_pull * 4.0)) + min(0.05, (self._k_prev - self.k_cross_short) / 100.0)

        if direction is None:
            return None, 0.0

        conf = min(0.92, max(0.70, conf))
        return direction, conf
