"""Fisher Transform indicator — converts price to Gaussian distribution."""
import math
from collections import deque


class FisherTransform:
    """Fisher Transform with trigger line for crossover detection."""

    def __init__(self, period: int = 10):
        self.period = period
        self._highs = deque(maxlen=period)
        self._lows = deque(maxlen=period)

        self.value = 0.0
        self.trigger = 0.0
        self._prev_value = 0.0
        self._norm = 0.0
        self.initialized = False
        self._count = 0

    def update(self, high: float, low: float, close: float):
        self._highs.append(high)
        self._lows.append(low)
        self._count += 1

        if len(self._highs) < self.period:
            return

        hh = max(self._highs)
        ll = min(self._lows)
        rng = hh - ll
        if rng < 1e-10:
            return

        # Normalize price to [-1, 1]
        mid = (high + low) / 2.0
        raw = 2.0 * ((mid - ll) / rng) - 1.0
        # Smooth
        self._norm = 0.33 * raw + 0.67 * self._norm
        # Clamp to prevent log(0)
        clamped = max(-0.999, min(0.999, self._norm))

        # Fisher Transform
        self.trigger = self._prev_value
        self.value = 0.5 * math.log((1 + clamped) / (1 - clamped))
        self.value = 0.5 * self.value + 0.5 * self._prev_value  # Smooth
        self._prev_value = self.value

        if self._count >= self.period + 5:
            self.initialized = True

    @property
    def bullish_cross(self) -> bool:
        """Fisher crossed above trigger from below."""
        return self.value > self.trigger and self._prev_value <= self.trigger

    @property
    def bearish_cross(self) -> bool:
        """Fisher crossed below trigger from above."""
        return self.value < self.trigger and self._prev_value >= self.trigger
