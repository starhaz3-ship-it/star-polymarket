"""
Hydra Strategy Signals for Polymarket 15-Minute Binary Markets
Adapted from top 10 Hyperliquid strategies for direction prediction.

Each strategy returns: (direction, confidence, strategy_name)
- direction: "UP" or "DOWN" or None
- confidence: 0.0-1.0 (higher = more confident)
- strategy_name: for tracking

V3.14: Initial implementation — quarantine paper trading
V3.14d: Fine-tuning from 37-trade analysis + A/B filter tracking
  - RSI > 65 veto on UP (95% WR in 40-60 vs 57% when >60)
  - TRENDLINE_BREAK: require below_ema50 + vol > 0.3
  - Trend-beats-reversion conflict resolution
  - Loosen MFI_DIVERGENCE + SHORT_TERM_REVERSAL thresholds
  - Remove RETURN_ASYMMETRY (42% WR design, zero signals)
  - All signals still emitted (filtered=True for blocked ones) for A/B comparison
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Strategy classification for conflict resolution
TREND_STRATEGIES = {"MULTI_SMA_TREND", "TRENDLINE_BREAK", "DC03_KALMAN_ADX"}
REVERSION_STRATEGIES = {"CONNORS_RSI", "CONSEC_CANDLE_REVERSAL", "SHORT_TERM_REVERSAL", "MFI_DIVERGENCE",
                        "BB_BOUNCE", "EXTREME_MOMENTUM", "RUBBER_BAND"}

# V3.16: KILLED — 0% WR in paper, all net negative. Do not generate signals.
KILLED_STRATEGIES = {"CONNORS_RSI", "CONSEC_CANDLE_REVERSAL", "SHORT_TERM_REVERSAL", "MFI_DIVERGENCE"}


@dataclass
class StrategySignal:
    name: str
    direction: str  # "UP" or "DOWN"
    confidence: float
    details: str = ""
    asset: str = ""  # "BTC", "ETH", "SOL"
    filtered: bool = False  # True = signal would be blocked by filters (A/B tracking)
    filter_reason: str = ""  # Why it was filtered


def _ema(data: list, period: int) -> list:
    """Calculate EMA series."""
    if len(data) < period:
        return data[:]
    mult = 2 / (period + 1)
    ema_val = sum(data[:period]) / period
    result = [0.0] * (period - 1) + [ema_val]
    for i in range(period, len(data)):
        ema_val = data[i] * mult + ema_val * (1 - mult)
        result.append(ema_val)
    return result


def _sma(data: list, period: int) -> list:
    """Calculate SMA series."""
    result = []
    for i in range(len(data)):
        if i < period - 1:
            result.append(sum(data[:i+1]) / (i+1))
        else:
            result.append(sum(data[i-period+1:i+1]) / period)
    return result


def _rsi(closes: list, period: int = 14) -> float:
    """Calculate RSI."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _stoch(highs: list, lows: list, closes: list, k_period: int = 14) -> float:
    """Calculate Stochastic %K."""
    if len(closes) < k_period:
        return 50.0
    h = max(highs[-k_period:])
    l = min(lows[-k_period:])
    if h == l:
        return 50.0
    return ((closes[-1] - l) / (h - l)) * 100


def _bb(closes: list, period: int = 20, std_mult: float = 2.0) -> Tuple[float, float, float, float]:
    """Calculate Bollinger Bands. Returns (upper, middle, lower, bb_position)."""
    if len(closes) < period:
        return closes[-1], closes[-1], closes[-1], 0.5
    sma_val = sum(closes[-period:]) / period
    std = (sum((c - sma_val) ** 2 for c in closes[-period:]) / period) ** 0.5
    upper = sma_val + std_mult * std
    lower = sma_val - std_mult * std
    bb_pos = (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5
    return upper, sma_val, lower, bb_pos


def _adx(highs: list, lows: list, closes: list, period: int = 14) -> Tuple[float, float, float]:
    """Calculate ADX, +DI, -DI."""
    if len(closes) < period + 1:
        return 20.0, 50.0, 50.0
    plus_dm = []
    minus_dm = []
    tr_list = []
    for i in range(1, len(closes)):
        h_diff = highs[i] - highs[i-1]
        l_diff = lows[i-1] - lows[i]
        plus_dm.append(max(h_diff, 0) if h_diff > l_diff else 0)
        minus_dm.append(max(l_diff, 0) if l_diff > h_diff else 0)
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_list.append(tr)
    if len(tr_list) < period:
        return 20.0, 50.0, 50.0
    # Smoothed averages
    atr = sum(tr_list[:period]) / period
    smooth_plus = sum(plus_dm[:period]) / period
    smooth_minus = sum(minus_dm[:period]) / period
    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
        smooth_plus = (smooth_plus * (period - 1) + plus_dm[i]) / period
        smooth_minus = (smooth_minus * (period - 1) + minus_dm[i]) / period
    plus_di = (smooth_plus / atr * 100) if atr > 0 else 0
    minus_di = (smooth_minus / atr * 100) if atr > 0 else 0
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
    return dx, plus_di, minus_di


def _mfi(highs: list, lows: list, closes: list, volumes: list, period: int = 14) -> float:
    """Calculate Money Flow Index."""
    if len(closes) < period + 1:
        return 50.0
    typical = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    pos_flow = 0
    neg_flow = 0
    for i in range(-period, 0):
        mf = typical[i] * volumes[i]
        if typical[i] > typical[i-1]:
            pos_flow += mf
        else:
            neg_flow += mf
    if neg_flow == 0:
        return 100.0
    mfi_ratio = pos_flow / neg_flow
    return 100 - (100 / (1 + mfi_ratio))


def _connors_rsi(closes: list) -> float:
    """Calculate Connors RSI (3-period RSI + streak RSI + ROC percentile)."""
    if len(closes) < 100:
        return 50.0
    # Component 1: 3-period RSI
    rsi3 = _rsi(closes, 3)
    # Component 2: Streak RSI (2-period RSI of up/down streak)
    streaks = []
    streak = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            streak = streak + 1 if streak > 0 else 1
        elif closes[i] < closes[i-1]:
            streak = streak - 1 if streak < 0 else -1
        else:
            streak = 0
        streaks.append(streak)
    streak_rsi = _rsi([s + 50 for s in streaks[-20:]], 2) if len(streaks) >= 20 else 50
    # Component 3: Percentile rank of 1-period ROC
    roc = (closes[-1] - closes[-2]) / closes[-2] * 100 if closes[-2] != 0 else 0
    rocs = [(closes[i] - closes[i-1]) / closes[i-1] * 100 for i in range(1, len(closes)) if closes[i-1] != 0]
    if len(rocs) > 10:
        pct_rank = sum(1 for r in rocs[-100:] if r < roc) / len(rocs[-100:]) * 100
    else:
        pct_rank = 50
    return (rsi3 + streak_rsi + pct_rank) / 3


def scan_strategies(candles_data: dict) -> List[StrategySignal]:
    """
    Run all 10 Hydra strategies on candle data for each asset.

    candles_data: dict with keys like 'BTC', 'SOL', 'ETH',
                  values are lists of Candle objects (1-min candles)

    Returns list of StrategySignal objects that fired.
    """
    signals = []

    for asset, candles in candles_data.items():
        if not candles or len(candles) < 30:
            continue

        _sig_start = len(signals)  # Track new signals for this asset

        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]

        rsi_val = _rsi(closes, 14)
        bb_upper, bb_mid, bb_lower, bb_pos = _bb(closes, 20)
        adx_val, plus_di, minus_di = _adx(highs, lows, closes, 14)

        # EMA values
        ema_9 = _ema(closes, 9)[-1] if len(closes) >= 9 else closes[-1]
        ema_20 = _ema(closes, 20)[-1] if len(closes) >= 20 else closes[-1]
        ema_50 = _ema(closes, 50)[-1] if len(closes) >= 50 else closes[-1]

        price = closes[-1]

        # Volume ratio
        vol_avg = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes) if volumes else 1
        vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1.0

        # MACD
        ema_12 = _ema(closes, 12)
        ema_26 = _ema(closes, 26)
        if len(ema_12) >= 26 and len(ema_26) >= 26:
            macd_line = ema_12[-1] - ema_26[-1]
            macd_signal_line = _ema([ema_12[i] - ema_26[i] for i in range(25, len(ema_12))], 9)
            macd_hist = macd_line - macd_signal_line[-1] if macd_signal_line else 0
        else:
            macd_line, macd_hist = 0, 0

        # ============================================================
        # STRATEGY 1: TRENDLINE_BREAK (SHORT-only, 70% backtest WR)
        # ============================================================
        try:
            tl_lookback = min(100, len(closes) - 5)
            if tl_lookback > 30:
                tl_close = closes[-tl_lookback:]
                tl_low = lows[-tl_lookback:]
                tl_vol = volumes[-tl_lookback:]
                tl_n = len(tl_close)

                # Find pivot lows
                pivot_lows = []
                for pi in range(5, tl_n - 3):
                    is_pl = all(tl_low[pi] < tl_low[pi - j] for j in range(1, 6))
                    if is_pl:
                        is_pl = all(tl_low[pi] < tl_low[pi + j] for j in range(1, 4))
                    if is_pl:
                        pivot_lows.append((pi, tl_low[pi]))

                # Check ascending trendline break
                tl_break = False
                for pl_i in range(max(0, len(pivot_lows) - 6), len(pivot_lows)):
                    if tl_break:
                        break
                    p2_idx, p2_price = pivot_lows[pl_i]
                    for prev_i in range(max(0, pl_i - 4), pl_i):
                        p1_idx, p1_price = pivot_lows[prev_i]
                        if p2_price <= p1_price or p2_idx - p1_idx < 5:
                            continue
                        slope = (p2_price - p1_price) / (p2_idx - p1_idx)
                        if slope <= 0:
                            continue
                        intercept = p1_price - slope * p1_idx
                        tl_val = slope * (tl_n - 1) + intercept
                        if tl_close[-1] < tl_val * 0.9995:  # 0.05% below
                            tl_break = True
                            break

                if tl_break and rsi_val < 50:
                    conf = 0.78
                    if vol_ratio > 1.3:
                        conf += 0.04
                    if price < ema_50:
                        conf += 0.03
                    if macd_hist < 0:
                        conf += 0.02
                    if rsi_val < 40:
                        conf += 0.03
                    signals.append(StrategySignal(
                        "TRENDLINE_BREAK", "DOWN", min(0.94, conf),
                        f"RSI={rsi_val:.0f} vol={vol_ratio:.1f}x below_ema50={price < ema_50}"
                    ))
        except Exception:
            pass

        # ============================================================
        # STRATEGY 2: MFI_DIVERGENCE (75% WR on Hydra)
        # ============================================================
        try:
            if len(closes) >= 30:
                mfi_val = _mfi(highs, lows, closes, volumes, 14)
                # Check for divergence over last 20 bars
                mid = len(closes) // 2
                price_low1 = min(lows[-20:-10]) if len(lows) >= 20 else min(lows)
                price_low2 = min(lows[-10:])
                mfi_at_low1 = _mfi(highs[:-10], lows[:-10], closes[:-10], volumes[:-10], 14) if len(closes) > 24 else 50
                mfi_at_low2 = mfi_val

                # Bullish divergence: lower price low + higher MFI low
                # V3.14d: Loosened from MFI<35/RSI<45 (zero signals in 1.5hr)
                if price_low2 < price_low1 and mfi_at_low2 > mfi_at_low1 and mfi_val < 40 and rsi_val < 50:
                    conf = 0.75 + (0.03 if vol_ratio > 1.2 else 0) + (0.02 if rsi_val < 35 else 0)
                    signals.append(StrategySignal(
                        "MFI_DIVERGENCE", "UP", min(0.90, conf),
                        f"MFI={mfi_val:.0f} RSI={rsi_val:.0f} bullish_div"
                    ))

                # Bearish divergence: higher price high + lower MFI high
                price_hi1 = max(highs[-20:-10]) if len(highs) >= 20 else max(highs)
                price_hi2 = max(highs[-10:])
                mfi_hi1 = _mfi(highs[:-10], lows[:-10], closes[:-10], volumes[:-10], 14) if len(closes) > 24 else 50

                # V3.14d: Loosened from MFI>65/RSI>55
                if price_hi2 > price_hi1 and mfi_val < mfi_hi1 and mfi_val > 60 and rsi_val > 50:
                    conf = 0.75 + (0.03 if vol_ratio > 1.2 else 0) + (0.02 if rsi_val > 65 else 0)
                    signals.append(StrategySignal(
                        "MFI_DIVERGENCE", "DOWN", min(0.90, conf),
                        f"MFI={mfi_val:.0f} RSI={rsi_val:.0f} bearish_div"
                    ))
        except Exception:
            pass

        # ============================================================
        # STRATEGY 3: CONNORS_RSI (documented 75% WR, mean reversion)
        # ============================================================
        try:
            if len(closes) >= 100:
                crsi = _connors_rsi(closes)
                # Extreme oversold -> UP
                if crsi < 10 and rsi_val < 40:
                    conf = 0.78 + (0.04 if crsi < 5 else 0) + (0.03 if bb_pos < 0.1 else 0)
                    signals.append(StrategySignal(
                        "CONNORS_RSI", "UP", min(0.92, conf),
                        f"CRSI={crsi:.1f} RSI={rsi_val:.0f} deeply_oversold"
                    ))
                # Extreme overbought -> DOWN
                elif crsi > 90 and rsi_val > 60:
                    conf = 0.78 + (0.04 if crsi > 95 else 0) + (0.03 if bb_pos > 0.9 else 0)
                    signals.append(StrategySignal(
                        "CONNORS_RSI", "DOWN", min(0.92, conf),
                        f"CRSI={crsi:.1f} RSI={rsi_val:.0f} deeply_overbought"
                    ))
        except Exception:
            pass

        # ============================================================
        # STRATEGY 4: MULTI_SMA_TREND (12-SMA consensus, 100% WR)
        # ============================================================
        try:
            sma_periods = [5, 8, 10, 13, 15, 20, 25, 30, 35, 40, 50, 60]
            above_count = 0
            valid_smas = 0
            for p in sma_periods:
                if len(closes) >= p:
                    sma_val = sum(closes[-p:]) / p
                    valid_smas += 1
                    if price > sma_val:
                        above_count += 1

            if valid_smas >= 8:
                score = above_count / valid_smas
                # 5-bar momentum
                mom_5 = (price - closes[-6]) / closes[-6] * 100 if len(closes) >= 6 else 0

                if score >= 0.85 and mom_5 > 0.1 and 45 < rsi_val < 75:
                    conf = 0.80 + min(0.10, (score - 0.85) * 2)
                    signals.append(StrategySignal(
                        "MULTI_SMA_TREND", "UP", conf,
                        f"SMA_score={score:.2f} mom={mom_5:.2f}% RSI={rsi_val:.0f}"
                    ))
                elif score <= 0.15 and mom_5 < -0.1 and 25 < rsi_val < 55:
                    conf = 0.80 + min(0.10, (0.15 - score) * 2)
                    signals.append(StrategySignal(
                        "MULTI_SMA_TREND", "DOWN", conf,
                        f"SMA_score={score:.2f} mom={mom_5:.2f}% RSI={rsi_val:.0f}"
                    ))
        except Exception:
            pass

        # ============================================================
        # STRATEGY 5: ALIGNMENT (5/7 indicator confluence)
        # ============================================================
        try:
            bull_count = 0
            bear_count = 0
            # 1. RSI direction
            if rsi_val > 55:
                bull_count += 1
            elif rsi_val < 45:
                bear_count += 1
            # 2. MACD histogram
            if macd_hist > 0:
                bull_count += 1
            elif macd_hist < 0:
                bear_count += 1
            # 3. Stochastic
            stoch_k = _stoch(highs, lows, closes)
            if stoch_k > 60:
                bull_count += 1
            elif stoch_k < 40:
                bear_count += 1
            # 4. BB position
            if bb_pos > 0.6:
                bull_count += 1
            elif bb_pos < 0.4:
                bear_count += 1
            # 5. EMA 9/20 cross
            if ema_9 > ema_20:
                bull_count += 1
            elif ema_9 < ema_20:
                bear_count += 1
            # 6. Price vs EMA50
            if price > ema_50:
                bull_count += 1
            elif price < ema_50:
                bear_count += 1
            # 7. ADX + DI
            if adx_val > 20:
                if plus_di > minus_di:
                    bull_count += 1
                else:
                    bear_count += 1

            if bull_count >= 5 and bb_pos < 0.7:
                conf = 0.75 + min(0.15, (bull_count - 5) * 0.05)
                signals.append(StrategySignal(
                    "ALIGNMENT", "UP", conf,
                    f"{bull_count}/7 bullish RSI={rsi_val:.0f} ADX={adx_val:.0f}"
                ))
            elif bear_count >= 5 and bb_pos > 0.3:
                conf = 0.75 + min(0.15, (bear_count - 5) * 0.05)
                signals.append(StrategySignal(
                    "ALIGNMENT", "DOWN", conf,
                    f"{bear_count}/7 bearish RSI={rsi_val:.0f} ADX={adx_val:.0f}"
                ))
        except Exception:
            pass

        # ============================================================
        # STRATEGY 6: DC03_KALMAN_ADX (67% WR, PROMOTED)
        # ============================================================
        try:
            # Simplified Kalman filter (exponential smoothing with adaptive gain)
            if len(closes) >= 30:
                kalman = closes[0]
                kalman_vel = 0
                kalman_gain = 0.1
                for c in closes[1:]:
                    pred = kalman + kalman_vel
                    err = c - pred
                    kalman = pred + kalman_gain * err
                    kalman_vel = kalman_vel + 0.01 * err

                kalman_dir = "UP" if kalman_vel > 0 else "DOWN"

                if adx_val > 25:
                    if kalman_dir == "UP" and plus_di > minus_di and bb_pos < 0.7:
                        conf = 0.72 + (0.03 if adx_val > 35 else 0) + (0.02 if vol_ratio > 1.2 else 0)
                        signals.append(StrategySignal(
                            "DC03_KALMAN_ADX", "UP", min(0.88, conf),
                            f"Kalman_vel={kalman_vel:.4f} ADX={adx_val:.0f} +DI={plus_di:.0f}"
                        ))
                    elif kalman_dir == "DOWN" and minus_di > plus_di and bb_pos > 0.3:
                        conf = 0.72 + (0.03 if adx_val > 35 else 0) + (0.02 if vol_ratio > 1.2 else 0)
                        signals.append(StrategySignal(
                            "DC03_KALMAN_ADX", "DOWN", min(0.88, conf),
                            f"Kalman_vel={kalman_vel:.4f} ADX={adx_val:.0f} -DI={minus_di:.0f}"
                        ))
        except Exception:
            pass

        # ============================================================
        # STRATEGY 7: CONSEC_CANDLE_REVERSAL (exhaustion mean reversion)
        # ============================================================
        try:
            if len(closes) >= 10:
                # Count consecutive bearish/bullish candles
                bear_streak = 0
                bull_streak = 0
                for i in range(len(closes) - 1, max(len(closes) - 10, 0), -1):
                    if closes[i] < closes[i-1]:
                        if bull_streak > 0:
                            break
                        bear_streak += 1
                    elif closes[i] > closes[i-1]:
                        if bear_streak > 0:
                            break
                        bull_streak += 1
                    else:
                        break

                # 4+ bearish candles + recovering = UP
                if bear_streak >= 4 and rsi_val < 35 and closes[-1] > lows[-1]:
                    conf = 0.72 + min(0.12, (bear_streak - 4) * 0.03)
                    if bb_pos < 0.1:
                        conf += 0.03
                    signals.append(StrategySignal(
                        "CONSEC_CANDLE_REVERSAL", "UP", min(0.90, conf),
                        f"{bear_streak}_bearish_candles RSI={rsi_val:.0f} BB={bb_pos:.2f}"
                    ))
                # 4+ bullish candles + dropping = DOWN
                elif bull_streak >= 4 and rsi_val > 65 and closes[-1] < highs[-1]:
                    conf = 0.72 + min(0.12, (bull_streak - 4) * 0.03)
                    if bb_pos > 0.9:
                        conf += 0.03
                    signals.append(StrategySignal(
                        "CONSEC_CANDLE_REVERSAL", "DOWN", min(0.90, conf),
                        f"{bull_streak}_bullish_candles RSI={rsi_val:.0f} BB={bb_pos:.2f}"
                    ))
        except Exception:
            pass

        # ============================================================
        # STRATEGY 8: SHORT_TERM_REVERSAL (academic, 1hr overreaction)
        # ============================================================
        try:
            if len(closes) >= 60:
                # 60-bar (1hr on 1m candles) return
                ret_60 = (closes[-1] - closes[-60]) / closes[-60] * 100

                # Oversold reversal -> UP
                # V3.14d: Loosened from -1.0%/RSI<35/BB<0.2 (zero signals)
                if ret_60 < -0.5 and rsi_val < 40 and bb_pos < 0.25:
                    conf = 0.73 + min(0.12, abs(ret_60 - (-0.5)) * 0.03)
                    signals.append(StrategySignal(
                        "SHORT_TERM_REVERSAL", "UP", min(0.90, conf),
                        f"1hr_ret={ret_60:.2f}% RSI={rsi_val:.0f} BB={bb_pos:.2f}"
                    ))
                # Overbought reversal -> DOWN
                # V3.14d: Loosened from 1.0%/RSI>65/BB>0.8
                elif ret_60 > 0.5 and rsi_val > 60 and bb_pos > 0.75:
                    conf = 0.73 + min(0.12, (ret_60 - 0.5) * 0.03)
                    signals.append(StrategySignal(
                        "SHORT_TERM_REVERSAL", "DOWN", min(0.90, conf),
                        f"1hr_ret={ret_60:.2f}% RSI={rsi_val:.0f} BB={bb_pos:.2f}"
                    ))
        except Exception:
            pass

        # ============================================================
        # STRATEGY 9: FVG_RETEST (Fair Value Gap, 50% WR +$2.71)
        # ============================================================
        try:
            if len(closes) >= 20:
                # Detect FVGs in last 20 bars
                bullish_fvg = None
                bearish_fvg = None
                for i in range(len(closes) - 18, len(closes) - 2):
                    if i < 2:
                        continue
                    # Bullish FVG: candle[i+1] low > candle[i-1] high (gap up)
                    if lows[i+1] > highs[i-1]:
                        gap_size = (lows[i+1] - highs[i-1]) / closes[i] * 100
                        if gap_size > 0.05:  # Minimum gap size
                            bullish_fvg = (highs[i-1], lows[i+1], i)
                    # Bearish FVG: candle[i+1] high < candle[i-1] low (gap down)
                    if highs[i+1] < lows[i-1]:
                        gap_size = (lows[i-1] - highs[i+1]) / closes[i] * 100
                        if gap_size > 0.05:
                            bearish_fvg = (highs[i+1], lows[i-1], i)

                # Price retesting bullish FVG from above -> UP
                if bullish_fvg and price > bullish_fvg[0] and price < bullish_fvg[1] * 1.002:
                    conf = 0.70 + (0.03 if rsi_val < 45 else 0) + (0.02 if vol_ratio > 1.2 else 0)
                    signals.append(StrategySignal(
                        "FVG_RETEST", "UP", min(0.85, conf),
                        f"bullish_fvg_retest RSI={rsi_val:.0f}"
                    ))
                # Price retesting bearish FVG from below -> DOWN
                if bearish_fvg and price < bearish_fvg[1] and price > bearish_fvg[0] * 0.998:
                    conf = 0.70 + (0.03 if rsi_val > 55 else 0) + (0.02 if vol_ratio > 1.2 else 0)
                    signals.append(StrategySignal(
                        "FVG_RETEST", "DOWN", min(0.85, conf),
                        f"bearish_fvg_retest RSI={rsi_val:.0f}"
                    ))
        except Exception:
            pass

        # ============================================================
        # STRATEGY 10: BB_BOUNCE (Bollinger Band mean reversion, ideal for 5m)
        # Buy at lower BB touch, sell at upper BB touch
        # BB_SCALPER on Hydra: 42% WR but +$237 — high R:R scalps
        # ============================================================
        try:
            if bb_lower and bb_upper and bb_mid:
                # Price touching/below lower BB + RSI oversold area
                if price <= bb_lower * 1.002 and rsi_val < 40:
                    conf = 0.60 + (40 - rsi_val) / 100  # More oversold = higher conf
                    signals.append(StrategySignal(
                        "BB_BOUNCE", "UP", min(0.92, conf),
                        f"BB_lower_touch RSI={rsi_val:.0f} bb_pos={bb_pos:.2f}"
                    ))
                # Price touching/above upper BB + RSI overbought area
                elif price >= bb_upper * 0.998 and rsi_val > 60:
                    conf = 0.60 + (rsi_val - 60) / 100
                    signals.append(StrategySignal(
                        "BB_BOUNCE", "DOWN", min(0.92, conf),
                        f"BB_upper_touch RSI={rsi_val:.0f} bb_pos={bb_pos:.2f}"
                    ))
        except Exception:
            pass

        # ============================================================
        # STRATEGY 11: EXTREME_MOMENTUM (RSI extreme + MACD confirmation)
        # Catches strong momentum after oversold/overbought extremes
        # Hydra: 90% WR backtest, 16:1 R:R
        # ============================================================
        try:
            if rsi_val < 25 and macd_hist > 0:
                # Extremely oversold + MACD turning bullish = snap-back UP
                conf = 0.70 + (25 - rsi_val) / 50
                signals.append(StrategySignal(
                    "EXTREME_MOMENTUM", "UP", min(0.95, conf),
                    f"RSI_extreme_low={rsi_val:.0f} MACD_bull={macd_hist:.4f}"
                ))
            elif rsi_val > 75 and macd_hist < 0:
                # Extremely overbought + MACD turning bearish = snap-back DOWN
                conf = 0.70 + (rsi_val - 75) / 50
                signals.append(StrategySignal(
                    "EXTREME_MOMENTUM", "DOWN", min(0.95, conf),
                    f"RSI_extreme_high={rsi_val:.0f} MACD_bear={macd_hist:.4f}"
                ))
        except Exception:
            pass

        # ============================================================
        # STRATEGY 12: RUBBER_BAND (Z-score extreme snap-back)
        # Price deviates >2 std from 20-bar SMA = rubber band snap
        # Hydra: 10:1 R:R, mean reversion after extreme deviation
        # ============================================================
        try:
            if len(closes) >= 20:
                sma_20 = np.mean(closes[-20:])
                std_20 = np.std(closes[-20:])
                if std_20 > 0:
                    zscore = (price - sma_20) / std_20
                    if zscore < -2.0 and rsi_val < 35:
                        # Extreme downside deviation = snap UP
                        conf = 0.65 + min(0.25, abs(zscore) / 10)
                        signals.append(StrategySignal(
                            "RUBBER_BAND", "UP", min(0.90, conf),
                            f"z={zscore:.2f} RSI={rsi_val:.0f} mean_revert"
                        ))
                    elif zscore > 2.0 and rsi_val > 65:
                        # Extreme upside deviation = snap DOWN
                        conf = 0.65 + min(0.25, abs(zscore) / 10)
                        signals.append(StrategySignal(
                            "RUBBER_BAND", "DOWN", min(0.90, conf),
                            f"z={zscore:.2f} RSI={rsi_val:.0f} mean_revert"
                        ))
        except Exception:
            pass

        # STRATEGY 13: RETURN_ASYMMETRY — REMOVED V3.14d
        # 42% WR by design, zero signals in 1.5hr quarantine. Not worth monitoring.

        # Tag all signals from this asset
        for sig in signals[_sig_start:]:
            sig.asset = asset

    # === V3.14d POST-PROCESSING FILTERS ===
    # All signals are kept (for A/B tracking) but filtered ones are marked
    _apply_filters(signals)

    # V3.16: Remove killed strategies (0% WR in paper — CONNORS_RSI, CONSEC_CANDLE_REVERSAL, SHORT_TERM_REVERSAL, MFI_DIVERGENCE)
    signals = [s for s in signals if s.name not in KILLED_STRATEGIES]

    return signals


def _apply_filters(signals: List[StrategySignal]):
    """Apply data-driven filters to signals. Marks filtered=True instead of removing.
    This allows A/B comparison: filtered (what we act on) vs unfiltered (what would have happened)."""

    # Build per-asset lookup for conflict detection
    asset_signals: Dict[str, List[StrategySignal]] = {}
    for s in signals:
        asset_signals.setdefault(s.asset, []).append(s)

    for sig in signals:
        # --- REMOVED FILTER 1 (RSI_UP_VETO): A/B data showed 84% would_win (36W/7L, -$92.56). Killed 2026-02-12. ---
        # --- REMOVED FILTER 2 (TL_EMA50_FILTER + TL_VOL_FILTER): A/B data 55-59% would_win (-$49.72 combined). Killed 2026-02-12. ---

        # --- FILTER 3: Trend-beats-reversion conflict resolution ---
        # Data: When trend vs reversion disagree, trend wins 100%
        # If a trend strategy fires opposite direction on same asset, suppress reversion
        if sig.name in REVERSION_STRATEGIES:
            same_asset = asset_signals.get(sig.asset, [])
            for other in same_asset:
                if (other.name in TREND_STRATEGIES and
                    other.direction != sig.direction and
                    not other.filtered):
                    sig.filtered = True
                    sig.filter_reason = f"TREND_CONFLICT({other.name}_{other.direction}_overrides)"
                    break


def get_strategy_consensus(signals: List[StrategySignal], direction: str) -> Tuple[int, float, List[str]]:
    """
    Count how many strategies agree on a direction.
    Returns: (count, avg_confidence, strategy_names)
    """
    matching = [s for s in signals if s.direction == direction]
    if not matching:
        return 0, 0.0, []
    avg_conf = sum(s.confidence for s in matching) / len(matching)
    names = [s.name for s in matching]
    return len(matching), avg_conf, names
