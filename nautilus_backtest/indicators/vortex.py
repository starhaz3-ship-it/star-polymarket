"""Vortex Indicator — trend direction via +VI/-VI crossovers."""
from collections import deque


class VortexIndicator:
    """Vortex Indicator: +VI and -VI showing bullish/bearish trend strength."""

    def __init__(self, period: int = 14):
        self.period = period
        self._highs = deque(maxlen=period + 1)
        self._lows = deque(maxlen=period + 1)
        self._closes = deque(maxlen=period + 1)

        self.plus_vi = 0.5
        self.minus_vi = 0.5
        self.initialized = False
        self._count = 0

    def update(self, high: float, low: float, close: float):
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._count += 1

        if len(self._highs) < self.period + 1:
            return

        h = list(self._highs)
        l = list(self._lows)
        c = list(self._closes)

        vm_plus = 0.0
        vm_minus = 0.0
        tr_sum = 0.0

        for i in range(1, len(h)):
            vm_plus += abs(h[i] - l[i - 1])
            vm_minus += abs(l[i] - h[i - 1])
            tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
            tr_sum += tr

        if tr_sum > 0:
            self.plus_vi = vm_plus / tr_sum
            self.minus_vi = vm_minus / tr_sum

        if self._count >= self.period + 1:
            self.initialized = True

    @property
    def bullish(self) -> bool:
        return self.plus_vi > self.minus_vi

    @property
    def bearish(self) -> bool:
        return self.minus_vi > self.plus_vi

    @property
    def strength(self) -> float:
        return abs(self.plus_vi - self.minus_vi)
