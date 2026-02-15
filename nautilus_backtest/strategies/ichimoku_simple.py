"""
ICHIMOKU_SIMPLE strategy -- Simplified Ichimoku Cloud for Polymarket binary options.

Processes BTC 1-min bars.  Computes Tenkan-sen (conversion line), Kijun-sen
(base line), and the Kumo cloud (Senkou Span A / Senkou Span B) using
rolling high/low windows.  Fires directional signals on Tenkan/Kijun
crossovers confirmed by price position relative to the cloud.
"""
from collections import deque
import math


# ---------------------------------------------------------------------------
# ICHIMOKU_SIMPLE Strategy
# ---------------------------------------------------------------------------

class IchimokuSimple:
    """Simplified Ichimoku Cloud: Tenkan/Kijun cross + cloud position.

    A signal fires when:
    1. Tenkan (conversion) crosses above Kijun (base) AND price is above the cloud -> UP
    2. Tenkan crosses below Kijun AND price is below the cloud -> DOWN

    Lines computed:
    - Tenkan = (highest_high(9) + lowest_low(9)) / 2
    - Kijun  = (highest_high(26) + lowest_low(26)) / 2
    - Senkou A (cloud top)    = (Tenkan + Kijun) / 2
    - Senkou B (cloud bottom) = (highest_high(52) + lowest_low(52)) / 2
    """

    def __init__(self, horizon_bars: int = 5, **params):
        self.horizon_bars = horizon_bars
        self.name = f"ICHIMOKU_SIMPLE_{horizon_bars}m"

        # ---- Ichimoku periods ----
        self.tenkan_period = params.get("tenkan_period", 9)
        self.kijun_period = params.get("kijun_period", 26)
        self.senkou_b_period = params.get("senkou_b_period", 52)

        # Rolling high/low windows
        self._highs_tenkan = deque(maxlen=self.tenkan_period)
        self._lows_tenkan = deque(maxlen=self.tenkan_period)

        self._highs_kijun = deque(maxlen=self.kijun_period)
        self._lows_kijun = deque(maxlen=self.kijun_period)

        self._highs_senkou = deque(maxlen=self.senkou_b_period)
        self._lows_senkou = deque(maxlen=self.senkou_b_period)

        # Current line values
        self._tenkan = 0.0
        self._kijun = 0.0
        self._senkou_a = 0.0  # cloud top
        self._senkou_b = 0.0  # cloud bottom

        # Previous values for crossover detection
        self._prev_tenkan = 0.0
        self._prev_kijun = 0.0

        # ---- Volume EMA(20) for confirmation ----
        self._vol_ema = None
        self._vol_ema_seed = deque(maxlen=20)

        # ---- ATR(14) for volatility context ----
        self.atr_period = params.get("atr_period", 14)
        self._tr_hist = deque(maxlen=self.atr_period)
        self._atr_value = 0.0
        self._prev_close = None

        # ---- Misc ----
        self._bar_count = 0
        self._last_signal_bar = -999
        self._warmup = self.senkou_b_period + 5  # need 52 bars for Senkou B

    def update(self, high: float, low: float, close: float, volume: float):
        """Process one bar.  Returns (direction, confidence) or (None, 0.0)."""
        self._bar_count += 1

        # Save previous line values for crossover detection
        self._prev_tenkan = self._tenkan
        self._prev_kijun = self._kijun

        # ---- Update rolling windows ----
        self._highs_tenkan.append(high)
        self._lows_tenkan.append(low)
        self._highs_kijun.append(high)
        self._lows_kijun.append(low)
        self._highs_senkou.append(high)
        self._lows_senkou.append(low)

        # ---- Compute Ichimoku lines ----
        if len(self._highs_tenkan) >= self.tenkan_period:
            self._tenkan = (max(self._highs_tenkan) + min(self._lows_tenkan)) / 2.0

        if len(self._highs_kijun) >= self.kijun_period:
            self._kijun = (max(self._highs_kijun) + min(self._lows_kijun)) / 2.0

        # Senkou A = midpoint of Tenkan and Kijun
        if self._tenkan > 0 and self._kijun > 0:
            self._senkou_a = (self._tenkan + self._kijun) / 2.0

        # Senkou B = midpoint of 52-period high/low
        if len(self._highs_senkou) >= self.senkou_b_period:
            self._senkou_b = (max(self._highs_senkou) + min(self._lows_senkou)) / 2.0

        # ---- Update ATR ----
        self._update_atr(high, low, close)
        self._prev_close = close

        # ---- Update volume EMA ----
        self._update_vol_ema(volume)

        # ---- Warmup check ----
        if self._bar_count < self._warmup:
            return (None, 0.0)

        # ---- Cooldown ----
        if self._bar_count - self._last_signal_bar < self.horizon_bars:
            return (None, 0.0)

        # ---- Cloud boundaries ----
        cloud_top = max(self._senkou_a, self._senkou_b)
        cloud_bottom = min(self._senkou_a, self._senkou_b)

        if cloud_top == 0 or cloud_bottom == 0:
            return (None, 0.0)

        # ---- Crossover detection ----
        direction = None

        # Tenkan crosses above Kijun
        tenkan_crossed_above = (self._prev_tenkan <= self._prev_kijun and
                                self._tenkan > self._kijun)
        # Tenkan crosses below Kijun
        tenkan_crossed_below = (self._prev_tenkan >= self._prev_kijun and
                                self._tenkan < self._kijun)

        # Price position relative to cloud
        price_above_cloud = close > cloud_top
        price_below_cloud = close < cloud_bottom
        price_in_cloud = cloud_bottom <= close <= cloud_top

        # Bullish: Tenkan crosses above Kijun AND price is above the cloud
        if tenkan_crossed_above and price_above_cloud:
            direction = "UP"

        # Bearish: Tenkan crosses below Kijun AND price is below the cloud
        if tenkan_crossed_below and price_below_cloud:
            direction = "DOWN"

        if direction is None:
            return (None, 0.0)

        # ---- Confidence calculation ----
        confidence = 0.68

        # Tenkan/Kijun separation bonus (wider gap after cross = stronger signal)
        tk_gap = abs(self._tenkan - self._kijun)
        if close > 0:
            tk_gap_pct = tk_gap / close
            tk_bonus = min(0.06, tk_gap_pct * 200.0)
            confidence += tk_bonus

        # Distance from cloud bonus (further from cloud = stronger trend)
        if direction == "UP" and close > cloud_top:
            cloud_dist_pct = (close - cloud_top) / close
            cloud_bonus = min(0.06, cloud_dist_pct * 100.0)
            confidence += cloud_bonus
        elif direction == "DOWN" and close < cloud_bottom:
            cloud_dist_pct = (cloud_bottom - close) / close
            cloud_bonus = min(0.06, cloud_dist_pct * 100.0)
            confidence += cloud_bonus

        # Cloud thickness bonus (thin cloud = weaker support, thick = stronger)
        cloud_thickness = cloud_top - cloud_bottom
        if close > 0:
            thickness_pct = cloud_thickness / close
            thickness_bonus = min(0.04, thickness_pct * 50.0)
            confidence += thickness_bonus

        # Volume confirmation bonus
        if self._vol_ema is not None and self._vol_ema > 0:
            vol_ratio = volume / self._vol_ema
            if vol_ratio > 1.2:
                vol_bonus = min(0.04, (vol_ratio - 1.2) * 0.04)
                confidence += vol_bonus

        # Kijun slope bonus (Kijun trending in signal direction)
        if self._prev_kijun > 0:
            kijun_delta = self._kijun - self._prev_kijun
            if direction == "UP" and kijun_delta > 0:
                confidence += 0.02
            elif direction == "DOWN" and kijun_delta < 0:
                confidence += 0.02

        confidence = max(0.60, min(confidence, 0.92))

        self._last_signal_bar = self._bar_count
        return (direction, confidence)

    # ==================================================================
    # Indicator update helpers
    # ==================================================================

    def _update_atr(self, high: float, low: float, close: float):
        """Update ATR incrementally."""
        if self._prev_close is not None:
            tr = max(high - low,
                     abs(high - self._prev_close),
                     abs(low - self._prev_close))
        else:
            tr = high - low
        self._tr_hist.append(tr)

        if len(self._tr_hist) >= self.atr_period:
            if self._atr_value == 0.0:
                self._atr_value = sum(self._tr_hist) / len(self._tr_hist)
            else:
                self._atr_value = ((self._atr_value * (self.atr_period - 1) +
                                    self._tr_hist[-1]) / self.atr_period)

    def _update_vol_ema(self, volume: float):
        """Update volume EMA(20) incrementally."""
        alpha = 2.0 / 21.0
        self._vol_ema_seed.append(volume)
        if self._vol_ema is None:
            if len(self._vol_ema_seed) >= 20:
                self._vol_ema = sum(self._vol_ema_seed) / 20
        else:
            self._vol_ema = alpha * volume + (1 - alpha) * self._vol_ema
