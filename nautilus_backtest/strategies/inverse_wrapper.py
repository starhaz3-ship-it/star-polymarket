"""
InverseStrategy — wraps any strategy and flips its signals.

If the base strategy says UP, inverse says DOWN (and vice versa).
In binary markets, a consistent loser becomes a consistent winner.

Theory: If HA_KELTNER_MFI has 45.1% WR (-$171), inverse has 54.9% WR.
The spread eats both sides ($0.03 each way), so inverse PnL != |original PnL|,
but any strategy below ~47% WR becomes profitable when inverted.
"""


class InverseStrategy:
    """Wraps a strategy and flips UP↔DOWN."""

    def __init__(self, base_strategy, suffix: str = "INV"):
        self.base = base_strategy
        self.horizon_bars = base_strategy.horizon_bars
        # Replace the horizon suffix with _INV suffix
        base_name = base_strategy.name
        if base_name.endswith(f"_{self.horizon_bars}m"):
            base_name = base_name[: -len(f"_{self.horizon_bars}m")]
        self.name = f"{base_name}_{suffix}_{self.horizon_bars}m"

    def update(self, high: float, low: float, close: float, volume: float):
        direction, confidence = self.base.update(high, low, close, volume)
        if direction is None:
            return None, 0.0
        # Flip
        flipped = "DOWN" if direction == "UP" else "UP"
        return flipped, confidence
