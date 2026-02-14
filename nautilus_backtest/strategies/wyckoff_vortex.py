"""
WYCKOFF_VORTEX strategy -- Wyckoff spring/upthrust + Vortex crossover.

Processes BTC 1-min bars.  Detects Wyckoff accumulation/distribution
patterns (spring, upthrust) confirmed by Vortex Indicator trend direction
and RSI(14) for Polymarket binary options.
"""
from collections import deque

from nautilus_backtest.indicators.wyckoff import WyckoffPhase
from nautilus_backtest.indicators.vortex import VortexIndicator


# ---------------------------------------------------------------------------
# WYCKOFF_VORTEX Strategy
# ---------------------------------------------------------------------------

class WyckoffVortex:
    """Wyckoff spring/upthrust + Vortex cross + RSI confirmation.

    Signal conditions:
    - Spring detected (false breakdown) + Vortex bullish (+VI > -VI) -> UP
    - Upthrust detected (false breakout) + Vortex bearish (-VI > +VI) -> DOWN
    - RSI(14) adds confirmation bonus (oversold for UP, overbought for DOWN)
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"WYCKOFF_VORTEX_{horizon_bars}m"

        # ---- Wyckoff Phase Detector ----
        self.wyckoff = WyckoffPhase(
            lookback=params.get("wyckoff_lookback", 20),
            spring_pct=params.get("wyckoff_spring_pct", 0.001),
        )

        # ---- Vortex Indicator ----
        self.vortex = VortexIndicator(
            period=params.get("vortex_period", 14),
        )

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._prev_close = None
        self.rsi_value = 50.0

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(
            params.get("wyckoff_lookback", 20) + 5,
            params.get("vortex_period", 14) + 2,
            self.rsi_period + 2,
        )

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Update indicators
        self.wyckoff.update(high, low, close, volume)
        self.vortex.update(high, low, close)
        self._update_rsi(close)
        self._prev_close = close

        # Wait for warmup
        if self._bar_count < self._warmup:
            return (None, 0.0)

        if not self.wyckoff.initialized or not self.vortex.initialized:
            return (None, 0.0)

        # ---- Signal detection ----
        direction = None
        vortex_confirms = False

        # Spring (false breakdown) -> UP (Vortex bullish = bonus, not required)
        if self.wyckoff.spring_detected:
            direction = "UP"
            vortex_confirms = self.vortex.bullish

        # Upthrust (false breakout) -> DOWN (Vortex bearish = bonus, not required)
        elif self.wyckoff.upthrust_detected:
            direction = "DOWN"
            vortex_confirms = self.vortex.bearish

        if direction is None:
            return (None, 0.0)

        # Cooldown
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ---- Confidence calculation ----
        # Base confidence (higher if Vortex confirms)
        confidence = 0.70 if vortex_confirms else 0.62

        # RSI bonus: deeper oversold/overbought = more confidence in reversal
        rsi_bonus = 0.0
        if direction == "UP":
            if self.rsi_value < 35:
                rsi_bonus = min(0.08, (35.0 - self.rsi_value) / 100.0)
            elif self.rsi_value < 45:
                rsi_bonus = 0.02
        elif direction == "DOWN":
            if self.rsi_value > 65:
                rsi_bonus = min(0.08, (self.rsi_value - 65.0) / 100.0)
            elif self.rsi_value > 55:
                rsi_bonus = 0.02

        # Vortex strength bonus (larger gap between +VI and -VI = stronger trend)
        vortex_bonus = min(0.08, self.vortex.strength * 0.2)

        # Wyckoff phase depth bonus (|phase_value| closer to 1.0 = deeper spring/upthrust)
        wyckoff_bonus = min(0.06, abs(self.wyckoff.phase_value) * 0.06)

        confidence += rsi_bonus + vortex_bonus + wyckoff_bonus
        confidence = min(confidence, 0.90)

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ------------------------------------------------------------------
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
