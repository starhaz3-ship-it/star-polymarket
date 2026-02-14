"""
TDI_SQUEEZE strategy -- TDI extreme + BB/KC squeeze for Polymarket binary options.

Processes BTC 1-min bars.  Detects Bollinger Band / Keltner Channel squeeze
conditions and combines with TDI extreme readings (RSI outside TDI bands)
to generate high-confidence reversal signals on squeeze release.
"""
from collections import deque
import math

from nautilus_backtest.indicators.tdi import TradersIndex


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
# TDI_SQUEEZE Strategy
# ---------------------------------------------------------------------------

class TdiSqueeze:
    """TDI extreme + Bollinger / Keltner squeeze release.

    When the Bollinger Bands contract inside the Keltner Channel, volatility
    is compressed (a *squeeze*).  When the squeeze releases while TDI shows
    an extreme condition (RSI outside TDI bands), a reversal signal fires.
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"TDI_SQUEEZE_{horizon_bars}m"

        # ---- TDI ----
        self.tdi = TradersIndex(
            rsi_period=params.get("tdi_rsi", 13),
            bb_period=params.get("tdi_bb", 34),
            bb_mult=params.get("tdi_bb_mult", 1.6185),
            fast_period=params.get("tdi_fast", 7),
            slow_period=params.get("tdi_slow", 34),
        )

        # ---- BB(20,2) on price ----
        self.bb_period = params.get("bb_period", 20)
        self.bb_mult = params.get("bb_mult", 2.0)
        self._bb_closes = deque(maxlen=self.bb_period)
        self.bb_upper = 0.0
        self.bb_lower = 0.0
        self.bb_mid = 0.0

        # ---- Keltner Channel(20, 1.5) ----
        self.kc_period = params.get("kc_period", 20)
        self.kc_mult = params.get("kc_mult", 1.5)
        self._kc_closes = deque(maxlen=self.kc_period)
        self._kc_tr_hist = deque(maxlen=self.kc_period)
        self._prev_close_kc = None
        self.kc_upper = 0.0
        self.kc_lower = 0.0
        self.kc_mid = 0.0

        # ---- Squeeze state ----
        self._squeeze_on = False
        self._prev_squeeze_on = False
        self._squeeze_bars = 0

        # ---- EMA(9) for momentum direction ----
        self._ema9 = None
        self._ema9_seed = deque(maxlen=9)

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.bb_period, self.kc_period, 48) + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Update TDI
        self.tdi.update(close)

        # Update BB
        self._bb_closes.append(close)
        if len(self._bb_closes) >= self.bb_period:
            self.bb_mid = calc_sma(self._bb_closes, self.bb_period)
            std = calc_stdev(self._bb_closes, self.bb_period)
            if self.bb_mid is not None and std is not None:
                self.bb_upper = self.bb_mid + self.bb_mult * std
                self.bb_lower = self.bb_mid - self.bb_mult * std

        # Update KC
        self._kc_closes.append(close)
        if self._prev_close_kc is not None:
            tr = max(high - low, abs(high - self._prev_close_kc), abs(low - self._prev_close_kc))
        else:
            tr = high - low
        self._kc_tr_hist.append(tr)
        self._prev_close_kc = close

        if len(self._kc_closes) >= self.kc_period and len(self._kc_tr_hist) >= self.kc_period:
            self.kc_mid = calc_sma(self._kc_closes, self.kc_period)
            atr = calc_sma(self._kc_tr_hist, self.kc_period)
            if self.kc_mid is not None and atr is not None:
                self.kc_upper = self.kc_mid + self.kc_mult * atr
                self.kc_lower = self.kc_mid - self.kc_mult * atr

        # Update EMA(9) for direction
        self._ema9_seed.append(close)
        if self._ema9 is None:
            if len(self._ema9_seed) >= 9:
                self._ema9 = sum(self._ema9_seed) / 9
        else:
            self._ema9 = calc_ema(self._ema9, close, 9)

        # Squeeze detection: BB inside KC
        self._prev_squeeze_on = self._squeeze_on
        if self.bb_upper > 0 and self.kc_upper > 0:
            self._squeeze_on = (self.bb_lower > self.kc_lower) and (self.bb_upper < self.kc_upper)
        else:
            self._squeeze_on = False

        if self._squeeze_on:
            self._squeeze_bars += 1
        else:
            if not self._prev_squeeze_on:
                self._squeeze_bars = 0

        if self._bar_count < self._warmup:
            return (None, 0.0)

        if not self.tdi.initialized:
            return (None, 0.0)

        # ---- Signal logic ----
        # Squeeze just released (was on, now off)
        squeeze_releasing = self._prev_squeeze_on and not self._squeeze_on
        if not squeeze_releasing:
            return (None, 0.0)

        # TDI extreme check
        tdi_oversold = self.tdi.extreme_oversold
        tdi_overbought = self.tdi.extreme_overbought

        if not tdi_oversold and not tdi_overbought:
            return (None, 0.0)

        # Direction: TDI oversold = reversal UP, TDI overbought = reversal DOWN
        direction = None
        if tdi_oversold:
            direction = "UP"
        elif tdi_overbought:
            direction = "DOWN"

        if direction is None:
            return (None, 0.0)

        # Cooldown
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # Confidence: base 0.68 + extremity bonus
        extremity = 0.0
        if tdi_oversold:
            extremity = max(0, self.tdi.lower_band - self.tdi.rsi_value) / 20.0
        elif tdi_overbought:
            extremity = max(0, self.tdi.rsi_value - self.tdi.upper_band) / 20.0

        # Squeeze duration bonus (longer squeeze = more energy stored)
        squeeze_bonus = min(0.08, self._squeeze_bars * 0.005)

        confidence = 0.68 + min(0.10, extremity) + squeeze_bonus
        confidence = min(confidence, 0.90)

        self._last_signal_bar = self._bar_count
        return (direction, confidence)
