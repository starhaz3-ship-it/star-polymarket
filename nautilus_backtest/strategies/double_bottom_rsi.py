"""
DOUBLE_BOTTOM_RSI strategy -- RSI double-bottom/top pattern recognition.

Processes BTC 1-min bars.  Tracks RSI(14) over a 30-bar window looking for
classic double-bottom and double-top patterns in RSI space:

Double bottom (UP):
  RSI dips below 25, bounces, dips below 25 again (second dip >= first dip),
  then rises above 30.  The second dip holding at or above the first indicates
  buying support, and the break above 30 confirms the reversal.

Double top (DOWN):
  RSI rises above 75, pulls back, rises above 75 again (second peak <= first),
  then drops below 70.  Mirrors the logic for overbought conditions.

Price direction must confirm the RSI pattern.
"""
from collections import deque
import math


class DoubleBottomRsi:
    """RSI double-bottom/top pattern detector.

    Signal fires when:
    1. Double bottom: RSI hits <25, bounces, hits <25 again (2nd >= 1st), rises >30 --> UP
    2. Double top: RSI hits >75, drops, hits >75 again (2nd <= 1st), drops <70 --> DOWN
    3. Price must confirm direction (close > prev_close for UP, < for DOWN)

    Scans a rolling 30-bar RSI window for the pattern.
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"DOUBLE_BOTTOM_RSI_{horizon_bars}m"

        # ---- RSI(14) ----
        self.rsi_period = params.get("rsi_period", 14)
        self._rsi_avg_gain = 0.0
        self._rsi_avg_loss = 0.0
        self._rsi_count = 0
        self._rsi_value = 50.0

        # ---- Pattern detection ----
        self.scan_window = params.get("scan_window", 30)
        self.oversold_thresh = params.get("oversold_thresh", 25.0)
        self.oversold_exit = params.get("oversold_exit", 30.0)
        self.overbought_thresh = params.get("overbought_thresh", 75.0)
        self.overbought_exit = params.get("overbought_exit", 70.0)
        self._rsi_hist = deque(maxlen=self.scan_window)

        # ---- State ----
        self._prev_close = None
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = self.rsi_period + self.scan_window + 5

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one 1-minute bar. Returns (direction|None, confidence)."""
        self._bar_count += 1

        # Update RSI
        self._update_rsi(close)
        self._rsi_hist.append(self._rsi_value)

        # Warmup check
        if self._bar_count < self._warmup:
            self._prev_close = close
            return (None, 0.0)

        # Need enough RSI history
        if len(self._rsi_hist) < self.scan_window:
            self._prev_close = close
            return (None, 0.0)

        # Cooldown
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            self._prev_close = close
            return (None, 0.0)

        # Price confirmation requires prev_close
        if self._prev_close is None:
            self._prev_close = close
            return (None, 0.0)

        rsi_window = list(self._rsi_hist)
        current_rsi = rsi_window[-1]

        direction = None
        pattern_strength = 0.0

        # ---- Check for double bottom ----
        db_result = self._detect_double_bottom(rsi_window)
        if db_result is not None and close > self._prev_close:
            direction = "UP"
            pattern_strength = db_result

        # ---- Check for double top ----
        if direction is None:
            dt_result = self._detect_double_top(rsi_window)
            if dt_result is not None and close < self._prev_close:
                direction = "DOWN"
                pattern_strength = dt_result

        if direction is None:
            self._prev_close = close
            return (None, 0.0)

        # ---- Confidence calculation ----
        confidence = 0.72

        # Pattern clarity bonus (how clean the double bottom/top is)
        pattern_bonus = min(0.08, pattern_strength * 0.04)
        confidence += pattern_bonus

        # RSI confirmation: current RSI just crossed the exit threshold
        if direction == "UP":
            exit_margin = current_rsi - self.oversold_exit
            if 0 < exit_margin < 5.0:
                confidence += 0.04  # fresh breakout above 30
        else:
            exit_margin = self.overbought_exit - current_rsi
            if 0 < exit_margin < 5.0:
                confidence += 0.04  # fresh breakdown below 70

        # Price momentum bonus
        if self._prev_close > 0:
            price_move_pct = abs(close - self._prev_close) / self._prev_close
            momentum_bonus = min(0.05, price_move_pct * 25.0)
            confidence += momentum_bonus

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        self._prev_close = close
        return (direction, confidence)

    # ==================================================================
    # Pattern detection
    # ==================================================================

    def _detect_double_bottom(self, rsi_window):
        """Scan for RSI double bottom pattern.

        Returns pattern strength (float) or None if no pattern found.
        Pattern: dip1 < 25, bounce, dip2 < 25 (dip2 >= dip1), then RSI > 30.
        """
        n = len(rsi_window)
        current = rsi_window[-1]

        # Current RSI must have just crossed above 30 (confirmation)
        if current < self.oversold_exit:
            return None

        # Recent bar must have been below exit to confirm fresh cross
        found_recent_below = False
        for i in range(n - 2, max(n - 6, -1), -1):
            if i >= 0 and rsi_window[i] < self.oversold_exit:
                found_recent_below = True
                break
        if not found_recent_below:
            return None

        # Find two dips below oversold threshold
        dips = []  # (index, value)
        in_dip = False
        dip_min_val = 999.0
        dip_min_idx = -1

        for i in range(n):
            if rsi_window[i] < self.oversold_thresh:
                if not in_dip:
                    in_dip = True
                    dip_min_val = rsi_window[i]
                    dip_min_idx = i
                else:
                    if rsi_window[i] < dip_min_val:
                        dip_min_val = rsi_window[i]
                        dip_min_idx = i
            else:
                if in_dip:
                    dips.append((dip_min_idx, dip_min_val))
                    in_dip = False
                    dip_min_val = 999.0
                    dip_min_idx = -1

        # If we ended in a dip, record it
        if in_dip:
            dips.append((dip_min_idx, dip_min_val))

        if len(dips) < 2:
            return None

        # Use last two dips
        dip1_idx, dip1_val = dips[-2]
        dip2_idx, dip2_val = dips[-1]

        # Second dip must be at same level or higher (support holding)
        if dip2_val < dip1_val - 2.0:
            return None

        # Must have a bounce between dips (RSI went above oversold threshold)
        had_bounce = False
        for i in range(dip1_idx + 1, dip2_idx):
            if rsi_window[i] >= self.oversold_thresh:
                had_bounce = True
                break
        if not had_bounce:
            return None

        # Pattern strength: how well the second dip held + bounce height
        hold_strength = max(0.0, dip2_val - dip1_val)  # higher second dip = stronger
        bounce_height = 0.0
        for i in range(dip1_idx + 1, dip2_idx):
            if rsi_window[i] > bounce_height:
                bounce_height = rsi_window[i]
        bounce_strength = max(0.0, bounce_height - self.oversold_thresh)

        return hold_strength + bounce_strength * 0.5

    def _detect_double_top(self, rsi_window):
        """Scan for RSI double top pattern.

        Returns pattern strength (float) or None if no pattern found.
        Pattern: peak1 > 75, dip, peak2 > 75 (peak2 <= peak1), then RSI < 70.
        """
        n = len(rsi_window)
        current = rsi_window[-1]

        # Current RSI must have just crossed below 70 (confirmation)
        if current > self.overbought_exit:
            return None

        # Recent bar must have been above exit to confirm fresh cross
        found_recent_above = False
        for i in range(n - 2, max(n - 6, -1), -1):
            if i >= 0 and rsi_window[i] > self.overbought_exit:
                found_recent_above = True
                break
        if not found_recent_above:
            return None

        # Find two peaks above overbought threshold
        peaks = []  # (index, value)
        in_peak = False
        peak_max_val = -999.0
        peak_max_idx = -1

        for i in range(n):
            if rsi_window[i] > self.overbought_thresh:
                if not in_peak:
                    in_peak = True
                    peak_max_val = rsi_window[i]
                    peak_max_idx = i
                else:
                    if rsi_window[i] > peak_max_val:
                        peak_max_val = rsi_window[i]
                        peak_max_idx = i
            else:
                if in_peak:
                    peaks.append((peak_max_idx, peak_max_val))
                    in_peak = False
                    peak_max_val = -999.0
                    peak_max_idx = -1

        if in_peak:
            peaks.append((peak_max_idx, peak_max_val))

        if len(peaks) < 2:
            return None

        # Use last two peaks
        peak1_idx, peak1_val = peaks[-2]
        peak2_idx, peak2_val = peaks[-1]

        # Second peak must be at same level or lower (resistance holding)
        if peak2_val > peak1_val + 2.0:
            return None

        # Must have a pullback between peaks
        had_pullback = False
        for i in range(peak1_idx + 1, peak2_idx):
            if rsi_window[i] <= self.overbought_thresh:
                had_pullback = True
                break
        if not had_pullback:
            return None

        # Pattern strength
        hold_strength = max(0.0, peak1_val - peak2_val)  # lower second peak = stronger
        pullback_depth = 0.0
        for i in range(peak1_idx + 1, peak2_idx):
            depth = self.overbought_thresh - rsi_window[i]
            if depth > pullback_depth:
                pullback_depth = depth

        return hold_strength + pullback_depth * 0.5

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

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
                self._rsi_value = 100.0 - (100.0 / (1.0 + rs))
        else:
            self._rsi_avg_gain = (self._rsi_avg_gain * (self.rsi_period - 1) + gain) / self.rsi_period
            self._rsi_avg_loss = (self._rsi_avg_loss * (self.rsi_period - 1) + loss) / self.rsi_period
            rs = self._rsi_avg_gain / max(self._rsi_avg_loss, 1e-10)
            self._rsi_value = 100.0 - (100.0 / (1.0 + rs))
