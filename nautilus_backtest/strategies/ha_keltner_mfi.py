"""
HA_KELTNER_MFI strategy -- Heikin-Ashi trend + Keltner Channel breakout + MFI confirmation.

Processes BTC 1-min bars. Three-layer confirmation:
1. Heikin-Ashi trend (2 consecutive bullish/bearish HA candles)
2. Price breaks above/below Keltner Channel (volatility-adaptive breakout)
3. MFI (Money Flow Index) confirms momentum (>55 for longs, <45 for shorts)
4. ATR% gate filters dead chop and extreme spikes

This is a momentum/breakout strategy: enter when HA trend aligns with KC
breakout and volume-weighted momentum (MFI) confirms.
"""
from collections import deque
import math


def _ema_update(prev, value, period):
    alpha = 2.0 / (period + 1)
    return alpha * value + (1 - alpha) * prev


class HaKeltnerMfi:
    """Heikin-Ashi trend + Keltner Channel breakout + MFI.

    Signal fires when:
    1. HA close > HA open for 2 consecutive bars (UP) or vice versa (DOWN)
    2. Close breaks above KC upper (UP) or below KC lower (DOWN)
    3. MFI >= 55 (UP) or MFI <= 45 (DOWN)
    4. ATR% between 0.12% and 2.8%

    Direction: breakout continuation
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"HA_KELTNER_MFI_{horizon_bars}m"

        # Keltner Channel params
        self.kc_ema_len = params.get("kc_ema_len", 20)
        self.kc_atr_len = params.get("kc_atr_len", 10)
        self.kc_mult = params.get("kc_mult", 1.5)

        # KC state
        self._kc_ema = None  # EMA of close
        self._kc_tr_hist = deque(maxlen=self.kc_atr_len)

        # MFI params
        self.mfi_len = params.get("mfi_len", 14)
        self.mfi_long_min = params.get("mfi_long_min", 55.0)
        self.mfi_short_max = params.get("mfi_short_max", 45.0)

        # MFI state: track typical price * volume flows
        self._tp_prev = None
        self._pos_flow = deque(maxlen=self.mfi_len)
        self._neg_flow = deque(maxlen=self.mfi_len)

        # ATR gate
        self.atr_len = params.get("atr_len", 14)
        self._atr_hist = deque(maxlen=self.atr_len)
        self.atrp_min = params.get("atrp_min", 0.0012)
        self.atrp_max = params.get("atrp_max", 0.028)

        # Heikin-Ashi state (rolling, no numpy needed)
        self._ha_open = None
        self._ha_close = None
        self._ha_prev_open = None
        self._ha_prev_close = None
        # We need 2 bars of HA trend, so track prev-prev too
        self._ha_pp_open = None
        self._ha_pp_close = None

        # Previous close for TR
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

    def _update_ha(self, open_price, high, low, close):
        """Update Heikin-Ashi candles. open_price approximated as prev_close."""
        # Shift history
        self._ha_pp_open = self._ha_prev_open
        self._ha_pp_close = self._ha_prev_close
        self._ha_prev_open = self._ha_open
        self._ha_prev_close = self._ha_close

        # Compute new HA candle
        ha_close = (open_price + high + low + close) / 4.0
        if self._ha_prev_open is not None and self._ha_prev_close is not None:
            ha_open = (self._ha_prev_open + self._ha_prev_close) / 2.0
        else:
            ha_open = (open_price + close) / 2.0

        self._ha_open = ha_open
        self._ha_close = ha_close

    def _update_kc(self, high, low, close):
        """Update Keltner Channel (EMA mid + ATR bands)."""
        # EMA of close
        if self._kc_ema is None:
            self._kc_ema = close
        else:
            self._kc_ema = _ema_update(self._kc_ema, close, self.kc_ema_len)

        # ATR for KC
        tr = self._compute_tr(high, low, close)
        self._kc_tr_hist.append(tr)

        if len(self._kc_tr_hist) < self.kc_atr_len:
            return None, None, None

        kc_atr = sum(self._kc_tr_hist) / self.kc_atr_len
        upper = self._kc_ema + self.kc_mult * kc_atr
        lower = self._kc_ema - self.kc_mult * kc_atr
        return upper, lower, kc_atr

    def _update_mfi(self, high, low, close, volume):
        """Update Money Flow Index."""
        tp = (high + low + close) / 3.0
        mf = tp * volume

        if self._tp_prev is not None:
            if tp > self._tp_prev:
                self._pos_flow.append(mf)
                self._neg_flow.append(0.0)
            elif tp < self._tp_prev:
                self._pos_flow.append(0.0)
                self._neg_flow.append(mf)
            else:
                self._pos_flow.append(0.0)
                self._neg_flow.append(0.0)
        else:
            self._pos_flow.append(0.0)
            self._neg_flow.append(0.0)

        self._tp_prev = tp

        if len(self._pos_flow) < self.mfi_len:
            return None

        pos_sum = sum(self._pos_flow)
        neg_sum = sum(self._neg_flow)

        if neg_sum == 0:
            return 100.0
        mr = pos_sum / neg_sum
        return 100.0 - (100.0 / (1.0 + mr))

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar. Returns (direction|None, confidence)."""
        self._bar_count += 1

        # Approximate open as previous close (very close for 1-min BTC)
        open_price = self._prev_close if self._prev_close is not None else close

        # Update Heikin-Ashi
        self._update_ha(open_price, high, low, close)

        # Update Keltner Channel
        kc_upper, kc_lower, kc_atr = self._update_kc(high, low, close)

        # Update MFI
        mfi_val = self._update_mfi(high, low, close, volume)

        # Update ATR for vol gate
        tr = self._compute_tr(high, low, close)
        self._atr_hist.append(tr)

        # Save prev close
        self._prev_close = close

        # Need enough data
        if self._bar_count < max(self.kc_ema_len + 5, self.mfi_len + 2, self.kc_atr_len + 5):
            return None, 0.0

        # Check all indicators are ready
        if kc_upper is None or mfi_val is None:
            return None, 0.0
        if self._ha_prev_open is None or self._ha_prev_close is None:
            return None, 0.0
        if self._ha_pp_open is None or self._ha_pp_close is None:
            return None, 0.0

        # ATR% gate
        if len(self._atr_hist) >= self.atr_len:
            atr_val = sum(self._atr_hist) / self.atr_len
            atrp = atr_val / max(1e-9, close)
            if atrp < self.atrp_min or atrp > self.atrp_max:
                return None, 0.0

        # HA trend: 2 consecutive bars in same direction
        # Current bar = self._ha_open/close, previous = self._ha_prev_open/close
        ha_up_curr = self._ha_close > self._ha_open
        ha_up_prev = self._ha_prev_close > self._ha_prev_open
        ha_dn_curr = self._ha_close < self._ha_open
        ha_dn_prev = self._ha_prev_close < self._ha_prev_open

        ha_up_2 = ha_up_curr and ha_up_prev
        ha_dn_2 = ha_dn_curr and ha_dn_prev

        # KC breakout
        broke_up = close > kc_upper
        broke_dn = close < kc_lower

        # Signal logic
        direction = None
        conf = 0.0

        if ha_up_2 and broke_up and mfi_val >= self.mfi_long_min:
            direction = "UP"
            band_push = (close - kc_upper) / max(1e-9, close)
            conf = 0.72 + min(0.14, band_push / 0.0020) + min(0.06, (mfi_val - self.mfi_long_min) / 60.0)
        elif ha_dn_2 and broke_dn and mfi_val <= self.mfi_short_max:
            direction = "DOWN"
            band_push = (kc_lower - close) / max(1e-9, close)
            conf = 0.72 + min(0.14, band_push / 0.0020) + min(0.06, (self.mfi_short_max - mfi_val) / 60.0)

        if direction is None:
            return None, 0.0

        conf = min(0.92, max(0.70, conf))
        return direction, conf
