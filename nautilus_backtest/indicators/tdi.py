"""Traders Dynamic Index (TDI) — RSI with Bollinger Bands and EMAs applied to RSI."""
import numpy as np
from collections import deque


class TradersIndex:
    """TDI: RSI → BB on RSI → Fast/Slow EMA on RSI."""

    def __init__(self, rsi_period=13, bb_period=34, bb_mult=1.6185,
                 fast_period=7, slow_period=34):
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_mult = bb_mult
        self.fast_period = fast_period
        self.slow_period = slow_period

        # RSI state
        self._closes = deque(maxlen=rsi_period + 1)
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._rsi_count = 0

        # RSI series for BB + EMA
        self._rsi_history = deque(maxlen=max(bb_period, slow_period) + 1)

        # Outputs
        self.rsi_value = 50.0
        self.upper_band = 70.0
        self.lower_band = 30.0
        self.fast_line = 50.0
        self.slow_line = 50.0
        self.initialized = False
        self._count = 0

    def update(self, close: float):
        self._closes.append(close)
        self._count += 1

        if len(self._closes) < 2:
            return

        # RSI calculation (Wilder smoothing)
        delta = self._closes[-1] - self._closes[-2]
        gain = max(delta, 0)
        loss = max(-delta, 0)

        self._rsi_count += 1
        if self._rsi_count <= self.rsi_period:
            self._avg_gain += gain / self.rsi_period
            self._avg_loss += loss / self.rsi_period
            if self._rsi_count == self.rsi_period:
                rs = self._avg_gain / max(self._avg_loss, 1e-10)
                self.rsi_value = 100.0 - (100.0 / (1.0 + rs))
        else:
            self._avg_gain = (self._avg_gain * (self.rsi_period - 1) + gain) / self.rsi_period
            self._avg_loss = (self._avg_loss * (self.rsi_period - 1) + loss) / self.rsi_period
            rs = self._avg_gain / max(self._avg_loss, 1e-10)
            self.rsi_value = 100.0 - (100.0 / (1.0 + rs))

        self._rsi_history.append(self.rsi_value)

        # BB on RSI
        if len(self._rsi_history) >= self.bb_period:
            arr = list(self._rsi_history)[-self.bb_period:]
            mean = sum(arr) / len(arr)
            std = (sum((x - mean) ** 2 for x in arr) / len(arr)) ** 0.5
            self.upper_band = mean + self.bb_mult * std
            self.lower_band = mean - self.bb_mult * std

        # Fast EMA on RSI
        if len(self._rsi_history) >= self.fast_period:
            alpha_f = 2.0 / (self.fast_period + 1)
            if self._count == self.rsi_period + self.fast_period:
                self.fast_line = sum(list(self._rsi_history)[-self.fast_period:]) / self.fast_period
            else:
                self.fast_line = alpha_f * self.rsi_value + (1 - alpha_f) * self.fast_line

        # Slow EMA on RSI
        if len(self._rsi_history) >= self.slow_period:
            alpha_s = 2.0 / (self.slow_period + 1)
            if self._count == self.rsi_period + self.slow_period:
                self.slow_line = sum(list(self._rsi_history)[-self.slow_period:]) / self.slow_period
            else:
                self.slow_line = alpha_s * self.rsi_value + (1 - alpha_s) * self.slow_line

        if self._count >= self.rsi_period + self.bb_period:
            self.initialized = True

    @property
    def extreme_oversold(self) -> bool:
        return self.rsi_value < self.lower_band

    @property
    def extreme_overbought(self) -> bool:
        return self.rsi_value > self.upper_band
