"""
Experimental Paper Trader - Multi-Strategy Testing Lab

Tests 7 strategies simultaneously using new indicators:
- Bollinger Bands squeeze/breakout
- Stochastic RSI momentum
- ADX trend strength filter
- Williams %R overbought/oversold
- OBV (On-Balance Volume) direction
- MACD-V (Spiroglou 2022) volatility-normalized momentum + ATR compression
- Combined "Kitchen Sink" strategy

Auto-promotes winning strategies (75%+ WR over 20+ trades) to live trader.
"""
import sys
import asyncio
import json
import time
import math
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from functools import partial

import httpx

print = partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
from arbitrage.ta_signals import TASignalGenerator, Candle, SignalStrength, TASignal


# =============================================================================
# NEW INDICATORS
# =============================================================================

def compute_bollinger_bands(closes: List[float], period: int = 10, std_dev: float = 1.5):
    """Bollinger Bands optimized for crypto day trading (SMA10, 1.5 SD).
    Returns (upper, middle, lower, bandwidth, %b)
    """
    if len(closes) < period:
        return None
    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = math.sqrt(variance)
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bandwidth = (upper - lower) / middle if middle > 0 else 0
    pct_b = (closes[-1] - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    return {
        'upper': upper, 'middle': middle, 'lower': lower,
        'bandwidth': bandwidth, 'pct_b': pct_b, 'std': std,
    }


def detect_bb_squeeze(bandwidths: List[float], lookback: int = 20):
    """Detect Bollinger Band squeeze (low bandwidth = consolidation).
    Returns squeeze_active, squeeze_duration.
    """
    if len(bandwidths) < lookback:
        return False, 0
    avg_bw = sum(bandwidths[-lookback:]) / lookback
    threshold = avg_bw * 0.75  # Squeeze when BW < 75% of average
    squeeze_count = 0
    for bw in reversed(bandwidths[-lookback:]):
        if bw < threshold:
            squeeze_count += 1
        else:
            break
    return squeeze_count >= 3, squeeze_count


def compute_stochastic_rsi(closes: List[float], rsi_period: int = 14,
                            stoch_period: int = 14, k_smooth: int = 3):
    """Stochastic RSI - applies stochastic formula to RSI values.
    Returns (%K, %D) where overbought > 80, oversold < 20.
    """
    if len(closes) < rsi_period + stoch_period + k_smooth:
        return None
    # Calculate RSI series
    rsi_values = []
    for i in range(rsi_period, len(closes)):
        gains = losses = 0
        for j in range(i - rsi_period + 1, i + 1):
            diff = closes[j] - closes[j - 1]
            if diff > 0:
                gains += diff
            else:
                losses -= diff
        avg_gain = gains / rsi_period
        avg_loss = losses / rsi_period
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))

    if len(rsi_values) < stoch_period:
        return None
    # Stochastic of RSI
    stoch_k_raw = []
    for i in range(stoch_period - 1, len(rsi_values)):
        window = rsi_values[i - stoch_period + 1:i + 1]
        high = max(window)
        low = min(window)
        if high == low:
            stoch_k_raw.append(50.0)
        else:
            stoch_k_raw.append((rsi_values[i] - low) / (high - low) * 100)

    if len(stoch_k_raw) < k_smooth:
        return None
    # Smooth %K
    k = sum(stoch_k_raw[-k_smooth:]) / k_smooth
    # %D = SMA of %K (use last 3)
    if len(stoch_k_raw) >= k_smooth + 2:
        d_values = [sum(stoch_k_raw[i:i + k_smooth]) / k_smooth
                    for i in range(len(stoch_k_raw) - 3, len(stoch_k_raw))]
        d = sum(d_values) / len(d_values)
    else:
        d = k
    return {'k': k, 'd': d}


def compute_adx(candles: List[Candle], period: int = 14):
    """Average Directional Index - measures trend strength.
    ADX > 25 = trending, < 20 = ranging/choppy.
    Returns (adx, plus_di, minus_di).
    """
    if len(candles) < period * 2 + 1:
        return None
    # Calculate +DM, -DM, TR
    plus_dm = []
    minus_dm = []
    tr_list = []
    for i in range(1, len(candles)):
        high_diff = candles[i].high - candles[i - 1].high
        low_diff = candles[i - 1].low - candles[i].low
        pdm = high_diff if high_diff > low_diff and high_diff > 0 else 0
        mdm = low_diff if low_diff > high_diff and low_diff > 0 else 0
        plus_dm.append(pdm)
        minus_dm.append(mdm)
        tr = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close)
        )
        tr_list.append(tr)

    if len(tr_list) < period:
        return None
    # Smoothed averages (Wilder's smoothing)
    atr = sum(tr_list[:period]) / period
    plus_di_smooth = sum(plus_dm[:period]) / period
    minus_di_smooth = sum(minus_dm[:period]) / period
    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
        plus_di_smooth = (plus_di_smooth * (period - 1) + plus_dm[i]) / period
        minus_di_smooth = (minus_di_smooth * (period - 1) + minus_dm[i]) / period

    if atr == 0:
        return None
    plus_di = (plus_di_smooth / atr) * 100
    minus_di = (minus_di_smooth / atr) * 100
    di_sum = plus_di + minus_di
    if di_sum == 0:
        return None
    dx_values = []
    # Build DX series for ADX smoothing
    temp_atr = sum(tr_list[:period]) / period
    temp_pdm = sum(plus_dm[:period]) / period
    temp_mdm = sum(minus_dm[:period]) / period
    for i in range(period, len(tr_list)):
        temp_atr = (temp_atr * (period - 1) + tr_list[i]) / period
        temp_pdm = (temp_pdm * (period - 1) + plus_dm[i]) / period
        temp_mdm = (temp_mdm * (period - 1) + minus_dm[i]) / period
        if temp_atr == 0:
            continue
        pdi = (temp_pdm / temp_atr) * 100
        mdi = (temp_mdm / temp_atr) * 100
        di_s = pdi + mdi
        if di_s > 0:
            dx_values.append(abs(pdi - mdi) / di_s * 100)

    if len(dx_values) < period:
        return None
    adx = sum(dx_values[:period]) / period
    for dx in dx_values[period:]:
        adx = (adx * (period - 1) + dx) / period

    return {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}


def compute_williams_r(candles: List[Candle], period: int = 14):
    """Williams %R - momentum oscillator.
    > -20 = overbought, < -80 = oversold.
    """
    if len(candles) < period:
        return None
    window = candles[-period:]
    highest = max(c.high for c in window)
    lowest = min(c.low for c in window)
    if highest == lowest:
        return -50.0
    close = candles[-1].close
    return ((highest - close) / (highest - lowest)) * -100


def compute_obv_slope(candles: List[Candle], lookback: int = 10):
    """On-Balance Volume slope over lookback period.
    Positive slope = buying pressure, negative = selling pressure.
    """
    if len(candles) < lookback + 1:
        return None
    obv = 0
    obv_series = []
    start = max(0, len(candles) - lookback - 20)
    for i in range(start + 1, len(candles)):
        if candles[i].close > candles[i - 1].close:
            obv += candles[i].volume
        elif candles[i].close < candles[i - 1].close:
            obv -= candles[i].volume
        obv_series.append(obv)

    if len(obv_series) < lookback:
        return None
    recent = obv_series[-lookback:]
    slope = (recent[-1] - recent[0]) / lookback if lookback > 0 else 0
    return slope


def compute_atr(candles: List[Candle], period: int = 14):
    """Average True Range."""
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        tr = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close)
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def compute_ema(values: List[float], period: int) -> Optional[float]:
    """Exponential Moving Average."""
    if len(values) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = (v - ema) * multiplier + ema
    return ema


def compute_macd_v(candles: List[Candle], fast: int = 12, slow: int = 26) -> Optional[Dict]:
    """MACD-V: Volatility Normalized Momentum (Spiroglou 2022).
    MACD-V = (EMA_fast - EMA_slow) / ATR(slow) * 100
    Signal = EMA(9) of MACD-V series
    Returns dict with macd_v, signal, histogram.
    """
    if len(candles) < slow + 10:
        return None
    closes = [c.close for c in candles]
    atr = compute_atr(candles, slow)
    if not atr or atr <= 0:
        return None
    ema_fast = compute_ema(closes, fast)
    ema_slow = compute_ema(closes, slow)
    if ema_fast is None or ema_slow is None:
        return None
    macd_v = ((ema_fast - ema_slow) / atr) * 100
    # Build MACD-V series for signal line
    macd_v_series = []
    for i in range(slow + 5, len(candles)):
        sub = candles[:i + 1]
        sub_closes = [c.close for c in sub]
        ef = compute_ema(sub_closes, fast)
        es = compute_ema(sub_closes, slow)
        a = compute_atr(sub, slow)
        if ef and es and a and a > 0:
            macd_v_series.append(((ef - es) / a) * 100)
    signal = compute_ema(macd_v_series, 9) if len(macd_v_series) >= 9 else macd_v
    histogram = macd_v - signal if signal else 0
    return {'macd_v': macd_v, 'signal': signal, 'histogram': histogram}


def is_atr_compressing(candles: List[Candle], short: int = 20, long: int = 30) -> bool:
    """ATR Compression: short-term ATR < long-term ATR = coiled volatility."""
    atr_short = compute_atr(candles, short)
    atr_long = compute_atr(candles, long)
    if atr_short is None or atr_long is None:
        return False
    return atr_short < atr_long


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

@dataclass
class StrategySignal:
    """Signal from a strategy."""
    side: Optional[str] = None  # "UP" or "DOWN" or None
    confidence: float = 0.0     # 0-1
    reason: str = ""


def strategy_baseline(signal: TASignal, candles, price, **kw) -> StrategySignal:
    """Baseline: original TA signals only (control group)."""
    if signal.action != "ENTER" or not signal.side:
        return StrategySignal(reason="no_signal")
    edge = signal.edge_up if signal.side == "UP" else signal.edge_down
    if edge is None or edge < 0.05:
        return StrategySignal(reason="low_edge")
    model = signal.model_up if signal.side == "UP" else signal.model_down
    return StrategySignal(side=signal.side, confidence=model, reason="baseline")


def strategy_bb_squeeze(signal: TASignal, candles, price, **kw) -> StrategySignal:
    """Bollinger Band squeeze breakout + direction from TA."""
    closes = [c.close for c in candles]
    if len(closes) < 30:
        return StrategySignal(reason="insufficient_data")
    # Compute BB
    bb = compute_bollinger_bands(closes, period=10, std_dev=1.5)
    if not bb:
        return StrategySignal(reason="no_bb")
    # Compute bandwidth history for squeeze detection
    bw_history = []
    for i in range(20, len(closes)):
        b = compute_bollinger_bands(closes[:i + 1], 10, 1.5)
        if b:
            bw_history.append(b['bandwidth'])
    squeeze, duration = detect_bb_squeeze(bw_history)
    # Position in bands
    pct_b = bb['pct_b']
    # Trade signal: after squeeze, direction from band position + TA
    if not signal.side:
        return StrategySignal(reason="no_ta_side")
    edge = signal.edge_up if signal.side == "UP" else signal.edge_down
    if edge is None or edge < 0.08:
        return StrategySignal(reason="low_edge")
    # BB confirmation
    if signal.side == "UP" and pct_b > 0.5:  # Price above middle band = bullish
        conf = min(1.0, pct_b * 0.8 + (0.2 if squeeze else 0))
        return StrategySignal(side="UP", confidence=conf, reason=f"bb_up pctb={pct_b:.2f} sq={squeeze}")
    elif signal.side == "DOWN" and pct_b < 0.5:  # Below middle = bearish
        conf = min(1.0, (1 - pct_b) * 0.8 + (0.2 if squeeze else 0))
        return StrategySignal(side="DOWN", confidence=conf, reason=f"bb_down pctb={pct_b:.2f} sq={squeeze}")
    return StrategySignal(reason=f"bb_no_confirm pctb={pct_b:.2f}")


def strategy_stoch_rsi(signal: TASignal, candles, price, **kw) -> StrategySignal:
    """Stochastic RSI momentum confirmation."""
    closes = [c.close for c in candles]
    stoch = compute_stochastic_rsi(closes)
    if not stoch:
        return StrategySignal(reason="no_stoch")
    k, d = stoch['k'], stoch['d']
    if not signal.side:
        return StrategySignal(reason="no_ta_side")
    edge = signal.edge_up if signal.side == "UP" else signal.edge_down
    if edge is None or edge < 0.08:
        return StrategySignal(reason="low_edge")
    # UP: StochRSI should be rising from oversold or in momentum zone
    if signal.side == "UP" and k < 80 and k > d:  # Rising momentum, not overbought
        return StrategySignal(side="UP", confidence=min(1.0, k / 100), reason=f"srsi_up k={k:.0f} d={d:.0f}")
    # DOWN: StochRSI should be falling from overbought or in bearish zone
    if signal.side == "DOWN" and k > 20 and k < d:  # Falling momentum, not oversold
        return StrategySignal(side="DOWN", confidence=min(1.0, (100 - k) / 100), reason=f"srsi_dn k={k:.0f} d={d:.0f}")
    return StrategySignal(reason=f"srsi_no_confirm k={k:.0f} d={d:.0f}")


def strategy_adx_trend(signal: TASignal, candles, price, **kw) -> StrategySignal:
    """ADX trend strength filter - only trade when trending (ADX > 20)."""
    adx_data = compute_adx(candles)
    if not adx_data:
        return StrategySignal(reason="no_adx")
    adx = adx_data['adx']
    plus_di = adx_data['plus_di']
    minus_di = adx_data['minus_di']
    if not signal.side:
        return StrategySignal(reason="no_ta_side")
    edge = signal.edge_up if signal.side == "UP" else signal.edge_down
    if edge is None or edge < 0.08:
        return StrategySignal(reason="low_edge")
    # Only trade in trending markets
    if adx < 20:
        return StrategySignal(reason=f"adx_low={adx:.0f}")
    # DI confirms direction
    if signal.side == "UP" and plus_di > minus_di:
        return StrategySignal(side="UP", confidence=min(1.0, adx / 50), reason=f"adx_up adx={adx:.0f} +di={plus_di:.0f}")
    if signal.side == "DOWN" and minus_di > plus_di:
        return StrategySignal(side="DOWN", confidence=min(1.0, adx / 50), reason=f"adx_dn adx={adx:.0f} -di={minus_di:.0f}")
    return StrategySignal(reason=f"adx_no_confirm +di={plus_di:.0f} -di={minus_di:.0f}")


def strategy_williams_obv(signal: TASignal, candles, price, **kw) -> StrategySignal:
    """Williams %R + OBV double confirmation."""
    wr = compute_williams_r(candles)
    obv_slope = compute_obv_slope(candles)
    if wr is None or obv_slope is None:
        return StrategySignal(reason="missing_data")
    if not signal.side:
        return StrategySignal(reason="no_ta_side")
    edge = signal.edge_up if signal.side == "UP" else signal.edge_down
    if edge is None or edge < 0.08:
        return StrategySignal(reason="low_edge")
    # UP: Williams not overbought + OBV rising
    if signal.side == "UP" and wr < -20 and obv_slope > 0:
        return StrategySignal(side="UP", confidence=0.7, reason=f"wr_obv_up wr={wr:.0f} obv={obv_slope:.0f}")
    # DOWN: Williams not oversold + OBV falling
    if signal.side == "DOWN" and wr > -80 and obv_slope < 0:
        return StrategySignal(side="DOWN", confidence=0.7, reason=f"wr_obv_dn wr={wr:.0f} obv={obv_slope:.0f}")
    return StrategySignal(reason=f"wr_obv_no wr={wr:.0f} obv={obv_slope:.0f}")


def strategy_macd_v(signal: TASignal, candles, price, **kw) -> StrategySignal:
    """MACD-V (Spiroglou 2022) + ATR Compression breakout strategy.
    Uses volatility-normalized momentum + coiled volatility detection.
    """
    mv = compute_macd_v(candles)
    if not mv:
        return StrategySignal(reason="no_macd_v")
    if not signal.side:
        return StrategySignal(reason="no_ta_side")
    edge = signal.edge_up if signal.side == "UP" else signal.edge_down
    if edge is None or edge < 0.08:
        return StrategySignal(reason="low_edge")
    macd_v = mv['macd_v']
    histogram = mv['histogram']
    compressed = is_atr_compressing(candles)
    # DOWN: MACD-V negative (bearish momentum), histogram falling
    if signal.side == "DOWN":
        if macd_v > 50:
            return StrategySignal(reason=f"macdv_oppose_dn mv={macd_v:.0f}")
        if macd_v < 0 or histogram < 0:
            conf = min(1.0, abs(macd_v) / 100 + (0.15 if compressed else 0))
            return StrategySignal(
                side="DOWN", confidence=max(0.3, conf),
                reason=f"macdv_dn mv={macd_v:.0f} h={histogram:.0f} comp={compressed}"
            )
    # UP: MACD-V positive or crossing up, ATR compression = breakout setup
    if signal.side == "UP":
        if macd_v < -50:
            return StrategySignal(reason=f"macdv_oppose_up mv={macd_v:.0f}")
        if macd_v > 0 or (compressed and histogram > 0):
            conf = min(1.0, abs(macd_v) / 100 + (0.20 if compressed else 0))
            return StrategySignal(
                side="UP", confidence=max(0.3, conf),
                reason=f"macdv_up mv={macd_v:.0f} h={histogram:.0f} comp={compressed}"
            )
    return StrategySignal(reason=f"macdv_no_confirm mv={macd_v:.0f} h={histogram:.0f}")


def strategy_kitchen_sink(signal: TASignal, candles, price, **kw) -> StrategySignal:
    """Triple confirmation: BB + StochRSI + ADX. Highest WR target."""
    closes = [c.close for c in candles]
    bb = compute_bollinger_bands(closes, 10, 1.5)
    stoch = compute_stochastic_rsi(closes)
    adx_data = compute_adx(candles)
    if not bb or not stoch or not adx_data:
        return StrategySignal(reason="missing_indicators")
    if not signal.side:
        return StrategySignal(reason="no_ta_side")
    edge = signal.edge_up if signal.side == "UP" else signal.edge_down
    if edge is None or edge < 0.10:
        return StrategySignal(reason="low_edge")
    adx = adx_data['adx']
    k = stoch['k']
    d = stoch['d']
    pct_b = bb['pct_b']
    confirms = 0
    # ADX trending
    if adx >= 20:
        confirms += 1
    # BB position confirms
    if signal.side == "UP" and pct_b > 0.5:
        confirms += 1
    elif signal.side == "DOWN" and pct_b < 0.5:
        confirms += 1
    # StochRSI confirms
    if signal.side == "UP" and k > d and k < 80:
        confirms += 1
    elif signal.side == "DOWN" and k < d and k > 20:
        confirms += 1
    if confirms >= 2:
        return StrategySignal(
            side=signal.side,
            confidence=min(1.0, confirms / 3),
            reason=f"ks_{confirms}/3 adx={adx:.0f} k={k:.0f} pctb={pct_b:.2f}"
        )
    return StrategySignal(reason=f"ks_only_{confirms}/3")


# Strategy registry
STRATEGIES = {
    "baseline":     strategy_baseline,
    "bb_squeeze":   strategy_bb_squeeze,
    "stoch_rsi":    strategy_stoch_rsi,
    "adx_trend":    strategy_adx_trend,
    "williams_obv": strategy_williams_obv,
    "macd_v":       strategy_macd_v,
    "kitchen_sink": strategy_kitchen_sink,
}


# =============================================================================
# MULTI-STRATEGY PAPER TRADER
# =============================================================================

ASSETS = {
    "BTC": {"symbol": "BTCUSDT", "keywords": ["bitcoin", "btc"]},
    "ETH": {"symbol": "ETHUSDT", "keywords": ["ethereum", "eth"]},
    "SOL": {"symbol": "SOLUSDT", "keywords": ["solana", "sol"]},
}

OUTPUT_FILE = Path(__file__).parent / "experiment_results.json"


@dataclass
class VirtualTrade:
    strategy: str
    asset: str
    side: str
    entry_price: float
    entry_time: str
    market_id: str
    market_title: str
    confidence: float
    reason: str
    exit_price: Optional[float] = None
    pnl: float = 0.0
    status: str = "open"


class ExperimentRunner:
    def __init__(self):
        self.generator = TASignalGenerator()
        self.trades: Dict[str, List[VirtualTrade]] = {s: [] for s in STRATEGIES}
        self.stats: Dict[str, Dict] = {s: {'wins': 0, 'losses': 0, 'pnl': 0.0} for s in STRATEGIES}
        self.cycle = 0
        self._load()

    def _load(self):
        if OUTPUT_FILE.exists():
            try:
                data = json.load(open(OUTPUT_FILE))
                for s in STRATEGIES:
                    if s in data.get('trades', {}):
                        self.trades[s] = [VirtualTrade(**t) for t in data['trades'][s]]
                    if s in data.get('stats', {}):
                        self.stats[s] = data['stats'][s]
                total = sum(s['wins'] + s['losses'] for s in self.stats.values())
                print(f"[Experiment] Loaded {total} total trades across {len(STRATEGIES)} strategies")
            except Exception as e:
                print(f"[Experiment] Load error: {e}")

    def _save(self):
        data = {
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'trades': {s: [asdict(t) for t in trades] for s, trades in self.trades.items()},
            'stats': self.stats,
        }
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def print_scoreboard(self):
        print()
        print("=" * 78)
        print(f"EXPERIMENT SCOREBOARD - {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
        print("=" * 78)
        print(f"{'Strategy':<15} {'W/L':>7} {'WR%':>6} {'PnL':>10} {'Open':>5} {'Last':>20}")
        print("-" * 78)
        for s in STRATEGIES:
            st = self.stats[s]
            w, l = st['wins'], st['losses']
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            pnl = st['pnl']
            open_count = sum(1 for t in self.trades[s] if t.status == 'open')
            last = ""
            closed = [t for t in self.trades[s] if t.status == 'closed']
            if closed:
                last_t = closed[-1]
                result = "W" if last_t.pnl > 0 else "L"
                last = f"{result} ${last_t.pnl:+.2f}"
            marker = " **" if wr >= 75 and total >= 20 else ""
            print(f"{s:<15} {w:3d}/{l:<3d} {wr:5.1f}% ${pnl:+9.2f} {open_count:5d} {last:>20}{marker}")
        print("=" * 78)
        # Flag strategies ready for promotion
        for s in STRATEGIES:
            st = self.stats[s]
            total = st['wins'] + st['losses']
            if total >= 20 and st['wins'] / total >= 0.75:
                print(f"  >>> {s} READY FOR LIVE PROMOTION ({st['wins']}/{total} = {st['wins']/total*100:.0f}% WR)")
        print()

    async def run_cycle(self):
        """Run one cycle across all assets and strategies."""
        self.cycle += 1
        asset_data = {}

        async with httpx.AsyncClient(timeout=15) as client:
            for asset, cfg in ASSETS.items():
                try:
                    r = await client.get(
                        "https://api.binance.com/api/v3/klines",
                        params={"symbol": cfg["symbol"], "interval": "1m", "limit": 240}
                    )
                    klines = r.json()
                    pr = await client.get(
                        "https://api.binance.com/api/v3/ticker/price",
                        params={"symbol": cfg["symbol"]}
                    )
                    price = float(pr.json()["price"])
                    candles = [
                        Candle(k[0] / 1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
                        for k in klines
                    ]
                    asset_data[asset] = {"candles": candles, "price": price, "markets": []}
                    await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"[API] {asset} error: {e}")

            await asyncio.sleep(1.0)
            try:
                mr = await client.get(
                    "https://gamma-api.polymarket.com/events",
                    params={"tag_slug": "15M", "active": "true", "closed": "false", "limit": 50},
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                if mr.status_code == 200:
                    for event in mr.json():
                        title = event.get("title", "").lower()
                        for asset, cfg in ASSETS.items():
                            if any(kw in title for kw in cfg["keywords"]) and asset in asset_data:
                                for m in event.get("markets", []):
                                    if not m.get("closed", True):
                                        if not m.get("question"):
                                            m["question"] = event.get("title", "")
                                        asset_data[asset]["markets"].append(m)
                                break
            except Exception as e:
                print(f"[API] Markets error: {e}")

        # Process each asset
        for asset, data in asset_data.items():
            candles = data["candles"]
            price = data["price"]
            markets = data["markets"]
            if not candles or not markets:
                continue

            recent = candles[-15:]

            # Find nearest eligible market
            best_market = None
            best_time = 999
            for mkt in markets:
                end = mkt.get("endDate")
                if not end:
                    continue
                try:
                    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
                    t_left = (end_dt - datetime.now(timezone.utc)).total_seconds() / 60
                    if 3.0 <= t_left <= 14.0 and t_left < best_time:
                        best_time = t_left
                        best_market = mkt
                except:
                    pass

            if not best_market:
                continue

            # Get market prices
            outcomes = best_market.get("outcomes", [])
            prices = best_market.get("outcomePrices", [])
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            if isinstance(prices, str):
                prices = json.loads(prices)

            up_price = down_price = None
            for i, o in enumerate(outcomes):
                if i < len(prices):
                    p = float(prices[i])
                    if str(o).lower() == "up":
                        up_price = p
                    elif str(o).lower() == "down":
                        down_price = p

            if up_price is None or down_price is None:
                continue

            market_id = best_market.get("conditionId", "")
            question = best_market.get("question", "")

            # Generate base TA signal
            signal = self.generator.generate_signal(
                market_id=market_id, candles=recent, current_price=price,
                market_yes_price=up_price, market_no_price=down_price,
                time_remaining_min=best_time
            )

            # Run each strategy
            for strat_name, strat_fn in STRATEGIES.items():
                trade_key = f"{market_id}_{strat_name}"
                # Skip if already have open trade on this market
                if any(t.status == "open" and t.market_id == market_id for t in self.trades[strat_name]):
                    continue

                sig = strat_fn(signal, candles, price)
                if sig.side:
                    entry = up_price if sig.side == "UP" else down_price
                    if entry and entry <= 0.55:
                        trade = VirtualTrade(
                            strategy=strat_name, asset=asset, side=sig.side,
                            entry_price=entry, entry_time=datetime.now(timezone.utc).isoformat(),
                            market_id=market_id, market_title=f"[{asset}] {question[:50]}",
                            confidence=sig.confidence, reason=sig.reason,
                        )
                        self.trades[strat_name].append(trade)
                        print(f"  [{strat_name}] {asset} {sig.side} @ ${entry:.2f} | {sig.reason}")

            # Resolve expired trades
            now = datetime.now(timezone.utc)
            for strat_name in STRATEGIES:
                for trade in self.trades[strat_name]:
                    if trade.status != "open":
                        continue
                    entry_dt = datetime.fromisoformat(trade.entry_time)
                    age = (now - entry_dt).total_seconds() / 60
                    if age < 16:
                        continue
                    # Resolve via Gamma API
                    try:
                        async with httpx.AsyncClient(timeout=10) as c:
                            r = await c.get(
                                "https://gamma-api.polymarket.com/markets",
                                params={"condition_id": trade.market_id, "limit": "1"},
                                headers={"User-Agent": "Mozilla/5.0"}
                            )
                            if r.status_code == 200 and r.json():
                                mkt_data = r.json()[0]
                                res_outcomes = mkt_data.get("outcomes", [])
                                res_prices = mkt_data.get("outcomePrices", [])
                                if isinstance(res_outcomes, str):
                                    res_outcomes = json.loads(res_outcomes)
                                if isinstance(res_prices, str):
                                    res_prices = json.loads(res_prices)
                                res_price = None
                                for i, o in enumerate(res_outcomes):
                                    if str(o).lower() == trade.side.lower() and i < len(res_prices):
                                        res_price = float(res_prices[i])
                                        break
                                if res_price is not None:
                                    shares = 5.0 / trade.entry_price  # $5 virtual bet
                                    if res_price >= 0.95:
                                        exit_val = shares
                                    elif res_price <= 0.05:
                                        exit_val = 0
                                    else:
                                        exit_val = shares * res_price
                                    trade.pnl = exit_val - 5.0
                                    trade.exit_price = res_price
                                    trade.status = "closed"
                                    won = trade.pnl > 0
                                    if won:
                                        self.stats[strat_name]['wins'] += 1
                                    else:
                                        self.stats[strat_name]['losses'] += 1
                                    self.stats[strat_name]['pnl'] += trade.pnl
                                    result = "WIN" if won else "LOSS"
                                    print(f"  [{strat_name}] {result} {trade.side} ${trade.pnl:+.2f} | {trade.market_title[:40]}")
                                else:
                                    trade.status = "closed"
                                    trade.pnl = 0
                    except:
                        pass

        self._save()

    async def run(self):
        print("=" * 78)
        print("EXPERIMENTAL PAPER TRADER - Multi-Strategy Lab")
        print("=" * 78)
        print(f"Strategies: {', '.join(STRATEGIES.keys())}")
        print(f"Assets: {', '.join(ASSETS.keys())}")
        print(f"Virtual bet: $5 per trade")
        print(f"Auto-promote to live at 75%+ WR with 20+ trades")
        print("=" * 78)

        last_report = 0
        while True:
            try:
                await self.run_cycle()
                now = time.time()
                if now - last_report >= 600:  # Every 10 min
                    self.print_scoreboard()
                    last_report = now
                await asyncio.sleep(60)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(60)

        self.print_scoreboard()


if __name__ == "__main__":
    runner = ExperimentRunner()
    asyncio.run(runner.run())
