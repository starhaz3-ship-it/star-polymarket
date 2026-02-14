"""
EXHAUSTION_BAR strategy -- Catches the exact death of a big move.

Processes BTC 1-min bars.  Fires when a massive bar (range > 2x ATR(14))
appears with climactic volume (> 2x EMA(20)), AND the very next bar reverses
direction (closes opposite).  This is the "exhaustion" pattern -- a move that
went too far too fast and snaps back.

Ultra-selective: requires simultaneous extreme range, extreme volume, AND
immediate price reversal.  Targets >65% WR by only trading the highest-
conviction exhaustion events.
"""
from collections import deque
import math


def calc_ema(prev_ema, new_value, period):
    alpha = 2.0 / (period + 1)
    return alpha * new_value + (1 - alpha) * prev_ema


class ExhaustionBar:
    """Reversal after exhaustion bars (extreme range + volume + reversal).

    Signal fires when:
    1. Previous bar range > 2.0x ATR(14) -- massive move
    2. Previous bar volume > 2.0x volume EMA(20) -- climactic volume
    3. Current bar closes in OPPOSITE direction of prev bar
    4. Current bar close retraces at least 40% of prev bar range
    5. RSI(14) was at extreme on the big bar (< 30 or > 70)

    Direction: opposite of the exhaustion bar
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"EXHAUSTION_BAR_{horizon_bars}m"

        # ---- ATR(14) ----
        self.atr_period = params.get("atr_period", 14)
        self._tr_hist = deque(maxlen=self.atr_period)
        self._atr = None

        # ---- Volume EMA(20) ----
        self.vol_ema_period = params.get("vol_ema_period", 20)
        self.vol_spike_mult = params.get("vol_spike_mult", 2.0)
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=self.vol_ema_period)

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_extreme_lo = params.get("rsi_extreme_lo", 30.0)
        self.rsi_extreme_hi = params.get("rsi_extreme_hi", 70.0)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self.rsi_value = 50.0

        # ---- Exhaustion detection params ----
        self.range_mult = params.get("range_mult", 2.0)  # bar range > Nx ATR
        self.retrace_pct = params.get("retrace_pct", 0.40)  # must retrace 40%+

        # ---- Previous bar state ----
        self._prev_high = None
        self._prev_low = None
        self._prev_close = None
        self._prev_open = None  # approximated as prev_close of bar before
        self._prev_volume = 0.0
        self._prev_range = 0.0
        self._prev_rsi = 50.0
        self._prev_was_exhaustion = False
        self._prev_bar_direction = 0  # +1 up, -1 down

        # ---- Two bars ago close (for open approximation) ----
        self._two_ago_close = None

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = max(self.atr_period, self.vol_ema_period, self.rsi_period) + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar. Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Update indicators
        prev_rsi = self.rsi_value
        self._update_atr(high, low, close)
        self._update_volume(volume)
        self._update_rsi(close)

        # Check if PREVIOUS bar was an exhaustion bar and THIS bar reverses
        signal = (None, 0.0)

        if (self._bar_count > self._warmup and
                self._prev_was_exhaustion and
                self._prev_close is not None):

            prev_dir = self._prev_bar_direction
            prev_range = self._prev_range

            if prev_dir != 0 and prev_range > 0:
                # Current bar must close in opposite direction
                if self._prev_close is not None:
                    current_move = close - self._prev_close

                    # Reversal check: current bar goes opposite
                    is_reversal = False
                    retrace_ratio = 0.0

                    if prev_dir > 0 and current_move < 0:
                        # Prev was up, current is down
                        retrace_ratio = abs(current_move) / prev_range
                        is_reversal = True
                    elif prev_dir < 0 and current_move > 0:
                        # Prev was down, current is up
                        retrace_ratio = abs(current_move) / prev_range
                        is_reversal = True

                    if is_reversal and retrace_ratio >= self.retrace_pct:
                        # Cooldown
                        if self._bar_count - self._last_signal_bar >= self.horizon_bars:
                            direction = "DOWN" if prev_dir > 0 else "UP"

                            # Confidence based on exhaustion severity
                            confidence = 0.72

                            # Range severity bonus
                            if self._atr and self._atr > 0:
                                range_ratio = prev_range / self._atr
                                range_bonus = min(0.08, (range_ratio - self.range_mult) * 0.02)
                                confidence += range_bonus

                            # Retrace depth bonus
                            retrace_bonus = min(0.06, (retrace_ratio - self.retrace_pct) * 0.10)
                            confidence += retrace_bonus

                            # RSI extremity bonus
                            if direction == "UP" and self._prev_rsi < self.rsi_extreme_lo:
                                rsi_bonus = min(0.06, (self.rsi_extreme_lo - self._prev_rsi) / 60.0)
                                confidence += rsi_bonus
                            elif direction == "DOWN" and self._prev_rsi > self.rsi_extreme_hi:
                                rsi_bonus = min(0.06, (self._prev_rsi - self.rsi_extreme_hi) / 60.0)
                                confidence += rsi_bonus

                            confidence = max(0.65, min(confidence, 0.95))
                            self._last_signal_bar = self._bar_count
                            signal = (direction, confidence)

        # ---- Determine if THIS bar is an exhaustion bar (for next bar's check) ----
        self._prev_was_exhaustion = False
        self._prev_bar_direction = 0
        self._prev_range = high - low

        if (self._atr is not None and self._atr > 0 and
                self._vol_ema is not None and self._vol_ema > 0 and
                self._prev_close is not None):

            bar_range = high - low
            bar_direction = 1 if close > self._prev_close else (-1 if close < self._prev_close else 0)

            is_big_range = bar_range > self.range_mult * self._atr
            is_big_volume = volume > self.vol_spike_mult * self._vol_ema

            # RSI at extreme
            is_rsi_extreme = (self.rsi_value < self.rsi_extreme_lo or
                              self.rsi_value > self.rsi_extreme_hi)

            if is_big_range and is_big_volume and is_rsi_extreme and bar_direction != 0:
                self._prev_was_exhaustion = True
                self._prev_bar_direction = bar_direction
                self._prev_range = bar_range

        # Save state for next bar
        self._prev_rsi = prev_rsi
        self._two_ago_close = self._prev_close
        self._prev_high = high
        self._prev_low = low
        self._prev_close = close
        self._prev_volume = volume

        return signal

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

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

    def _update_volume(self, volume):
        self._vol_ema_seed.append(volume)
        if self._vol_ema is None:
            if len(self._vol_ema_seed) >= self.vol_ema_period:
                self._vol_ema = sum(self._vol_ema_seed) / self.vol_ema_period
        else:
            self._vol_ema = calc_ema(self._vol_ema, volume, self.vol_ema_period)

    def _update_rsi(self, close):
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
