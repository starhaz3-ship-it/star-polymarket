"""Simplified Wyckoff phase detection — spring and upthrust patterns."""
from collections import deque


class WyckoffPhase:
    """Detects Wyckoff spring (false breakdown) and upthrust (false breakout)."""

    def __init__(self, lookback: int = 20, spring_pct: float = 0.002):
        self.lookback = lookback
        self.spring_pct = spring_pct  # How far beyond support/resistance to count as spring

        self._highs = deque(maxlen=lookback + 5)
        self._lows = deque(maxlen=lookback + 5)
        self._closes = deque(maxlen=lookback + 5)
        self._volumes = deque(maxlen=lookback + 5)

        self.spring_detected = False
        self.upthrust_detected = False
        self.phase_value = 0.0  # -1 to +1 (distribution to accumulation)
        self.initialized = False
        self._count = 0

    def update(self, high: float, low: float, close: float, volume: float):
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._volumes.append(volume)
        self._count += 1

        self.spring_detected = False
        self.upthrust_detected = False

        if len(self._closes) < self.lookback + 2:
            return

        self.initialized = True
        h = list(self._highs)
        l = list(self._lows)
        c = list(self._closes)
        v = list(self._volumes)

        # Support/resistance from lookback (excluding last 2 bars)
        lookback_lows = l[-(self.lookback + 2):-2]
        lookback_highs = h[-(self.lookback + 2):-2]
        support = min(lookback_lows)
        resistance = max(lookback_highs)
        rng = resistance - support
        if rng < 1e-10:
            return

        # Volume trend (declining = institutional patience)
        vol_recent = sum(v[-5:]) / 5
        vol_older = sum(v[-(self.lookback):-5]) / max(1, len(v[-(self.lookback):-5]))
        vol_declining = vol_recent < vol_older * 0.9

        # SPRING: price pierced below support then recovered above it
        prev_low = l[-2]
        curr_close = c[-1]
        spring_threshold = support * (1.0 - self.spring_pct)
        if prev_low < spring_threshold and curr_close > support:
            self.spring_detected = True
            depth = (support - prev_low) / rng
            self.phase_value = min(1.0, 0.5 + depth * 2.0)
            if vol_declining:
                self.phase_value = min(1.0, self.phase_value + 0.1)

        # UPTHRUST: price pierced above resistance then fell below it
        prev_high = h[-2]
        upthrust_threshold = resistance * (1.0 + self.spring_pct)
        if prev_high > upthrust_threshold and curr_close < resistance:
            self.upthrust_detected = True
            depth = (prev_high - resistance) / rng
            self.phase_value = max(-1.0, -0.5 - depth * 2.0)
            if vol_declining:
                self.phase_value = max(-1.0, self.phase_value - 0.1)
