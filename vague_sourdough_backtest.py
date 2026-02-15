"""
Reverse-engineer @vague-sourdough's Polymarket BTC 5-minute strategy.

Profile: 441 trades, $1.95M volume, +$38,711.74 PnL (Feb 11-14 2026)
Pattern: EXCLUSIVELY trades BTC 5-min Up/Down markets
Strategy: Gabagool-style - buys BOTH Up AND Down, leans directionally

This script:
1. Fetches 7 days of BTC 1-minute data from Binance
2. Aggregates into 5-minute windows
3. Tests 15+ indicator strategies for direction prediction
4. Simulates Polymarket-style PnL (buy at ~0.52-0.56, win pays $1)
5. Finds which indicator best mirrors their ~$38K PnL pattern
"""

import numpy as np
import pandas as pd
import requests
import time
import json
from datetime import datetime, timedelta


def fetch_binance_klines(symbol='BTCUSDT', interval='1m', days=7):
    """Fetch historical klines from Binance."""
    url = 'https://api.binance.com/api/v3/klines'
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 60 * 60 * 1000)

    all_klines = []
    current = start_ms

    while current < end_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current,
            'limit': 1000
        }
        resp = requests.get(url, params=params)
        data = resp.json()

        if not data:
            break

        all_klines.extend(data)
        current = data[-1][0] + 1  # next ms after last candle

        if len(data) < 1000:
            break
        time.sleep(0.1)

    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_vol',
        'taker_buy_quote_vol', 'ignore'
    ])

    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                'taker_buy_vol', 'taker_buy_quote_vol']:
        df[col] = df[col].astype(float)

    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)

    return df


def aggregate_to_5min(df_1m):
    """Aggregate 1-minute data to 5-minute bars."""
    df_5m = df_1m.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'quote_volume': 'sum',
        'trades': 'sum',
        'taker_buy_vol': 'sum',
        'taker_buy_quote_vol': 'sum'
    }).dropna()

    return df_5m


def calc_ema(data, period):
    """Exponential Moving Average."""
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


def calc_rsi(close, period=14):
    """RSI calculation."""
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))

    if len(gains) < period:
        return np.full(len(close), 50.0)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period

    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(close, fast=12, slow=26, signal=9):
    """MACD calculation."""
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    sma = pd.Series(close).rolling(period).mean().values
    std = pd.Series(close).rolling(period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, sma, lower


def calc_atr(high, low, close, period=14):
    """Average True Range."""
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ))
    tr[0] = high[0] - low[0]
    atr = calc_ema(tr, period)
    return atr


def calc_stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    """Stochastic RSI."""
    rsi = calc_rsi(close, rsi_period)
    rsi_series = pd.Series(rsi)
    min_rsi = rsi_series.rolling(stoch_period).min()
    max_rsi = rsi_series.rolling(stoch_period).max()
    stoch_rsi = np.where(max_rsi - min_rsi > 0,
                          (rsi - min_rsi) / (max_rsi - min_rsi) * 100, 50)
    k = pd.Series(stoch_rsi).rolling(k_smooth).mean().values
    d = pd.Series(k).rolling(d_smooth).mean().values
    return k, d


def calc_cvd(taker_buy_vol, total_vol):
    """Cumulative Volume Delta."""
    sell_vol = total_vol - taker_buy_vol
    delta = taker_buy_vol - sell_vol
    cvd = np.cumsum(delta)
    return cvd, delta


def calc_vwap(high, low, close, volume):
    """VWAP (reset each day - simplified as rolling)."""
    typical = (high + low + close) / 3
    cum_tp_vol = np.cumsum(typical * volume)
    cum_vol = np.cumsum(volume)
    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, close)
    return vwap


def calc_momentum(close, period=5):
    """Simple momentum (rate of change)."""
    mom = np.zeros_like(close)
    mom[period:] = (close[period:] - close[:-period]) / close[:-period] * 100
    return mom


def calc_williams_r(high, low, close, period=14):
    """Williams %R."""
    highest = pd.Series(high).rolling(period).max().values
    lowest = pd.Series(low).rolling(period).min().values
    wr = np.where(highest - lowest > 0,
                   (highest - close) / (highest - lowest) * -100, -50)
    return wr


def calc_obv(close, volume):
    """On Balance Volume."""
    obv = np.zeros_like(close)
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    return obv


def calc_adx(high, low, close, period=14):
    """ADX (simplified)."""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

    atr = calc_ema(tr, period)
    plus_di = np.where(atr > 0, calc_ema(plus_dm, period) / atr * 100, 0)
    minus_di = np.where(atr > 0, calc_ema(minus_dm, period) / atr * 100, 0)

    dx = np.where(plus_di + minus_di > 0,
                   np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = calc_ema(dx, period)

    return adx, plus_di, minus_di


# ============================================================
# STRATEGY DEFINITIONS
# Each returns: array of predictions (-1=DOWN, 0=NEUTRAL, +1=UP)
# Prediction is made at bar i for the NEXT 5-min bar direction
# ============================================================

def strategy_ema_cross(df, fast=5, slow=15):
    """EMA crossover: fast > slow = UP prediction."""
    close = df['close'].values
    ema_f = calc_ema(close, fast)
    ema_s = calc_ema(close, slow)
    signals = np.where(ema_f > ema_s, 1, -1)
    return signals

def strategy_rsi_extremes(df, period=14, os=30, ob=70):
    """RSI extremes: <30 = UP, >70 = DOWN."""
    rsi = calc_rsi(df['close'].values, period)
    signals = np.zeros(len(df))
    signals[rsi < os] = 1
    signals[rsi > ob] = -1
    # Neutral when in middle - use trend
    signals[signals == 0] = np.where(rsi[signals == 0] > 50, 1, -1)
    return signals

def strategy_rsi_momentum(df, period=6):
    """Short RSI momentum: RSI>55=UP, RSI<45=DOWN."""
    rsi = calc_rsi(df['close'].values, period)
    signals = np.where(rsi > 55, 1, np.where(rsi < 45, -1, 0))
    # Fill neutrals with last known
    for i in range(1, len(signals)):
        if signals[i] == 0:
            signals[i] = signals[i-1]
    return signals

def strategy_macd(df, fast=12, slow=26, sig=9):
    """MACD: histogram positive = UP."""
    macd_line, signal_line, hist = calc_macd(df['close'].values, fast, slow, sig)
    signals = np.where(hist > 0, 1, -1)
    return signals

def strategy_macd_cross(df, fast=8, slow=21, sig=5):
    """MACD line cross signal: MACD > signal = UP."""
    macd_line, signal_line, _ = calc_macd(df['close'].values, fast, slow, sig)
    signals = np.where(macd_line > signal_line, 1, -1)
    return signals

def strategy_momentum(df, period=3):
    """Simple momentum: positive = UP."""
    mom = calc_momentum(df['close'].values, period)
    signals = np.where(mom > 0, 1, -1)
    return signals

def strategy_momentum_5bar(df):
    """5-bar momentum with threshold."""
    mom = calc_momentum(df['close'].values, 5)
    signals = np.where(mom > 0.02, 1, np.where(mom < -0.02, -1, 0))
    for i in range(1, len(signals)):
        if signals[i] == 0:
            signals[i] = signals[i-1]
    return signals

def strategy_bb_mean_revert(df, period=20, std=2.0):
    """BB mean reversion: below lower=UP, above upper=DOWN."""
    close = df['close'].values
    upper, mid, lower = calc_bollinger(close, period, std)
    signals = np.zeros(len(df))
    signals[close < lower] = 1   # oversold -> UP
    signals[close > upper] = -1  # overbought -> DOWN
    # Fill neutrals with trend direction
    for i in range(1, len(signals)):
        if signals[i] == 0:
            signals[i] = 1 if close[i] > mid[i] else -1
    return signals

def strategy_bb_breakout(df, period=20, std=2.0):
    """BB breakout: above upper=UP (momentum), below lower=DOWN."""
    close = df['close'].values
    upper, mid, lower = calc_bollinger(close, period, std)
    signals = np.where(close > upper, 1, np.where(close < lower, -1, 0))
    for i in range(1, len(signals)):
        if signals[i] == 0:
            signals[i] = 1 if close[i] > mid[i] else -1
    return signals

def strategy_vwap(df):
    """VWAP: price above VWAP = UP."""
    vwap = calc_vwap(df['high'].values, df['low'].values,
                     df['close'].values, df['volume'].values)
    signals = np.where(df['close'].values > vwap, 1, -1)
    return signals

def strategy_cvd(df):
    """CVD: positive delta trend = UP."""
    cvd, delta = calc_cvd(df['taker_buy_vol'].values, df['volume'].values)
    # Use EMA of delta for smoothing
    delta_ema = calc_ema(delta, 5)
    signals = np.where(delta_ema > 0, 1, -1)
    return signals

def strategy_cvd_divergence(df):
    """CVD divergence: price down + CVD up = UP (bullish divergence)."""
    close = df['close'].values
    cvd, _ = calc_cvd(df['taker_buy_vol'].values, df['volume'].values)

    price_mom = calc_momentum(close, 3)
    cvd_mom = np.zeros_like(cvd)
    cvd_mom[3:] = cvd[3:] - cvd[:-3]

    signals = np.zeros(len(df))
    # Bullish divergence: price down, CVD up
    signals[(price_mom < -0.01) & (cvd_mom > 0)] = 1
    # Bearish divergence: price up, CVD down
    signals[(price_mom > 0.01) & (cvd_mom < 0)] = -1
    # Fill neutrals
    for i in range(1, len(signals)):
        if signals[i] == 0:
            signals[i] = signals[i-1] if signals[i-1] != 0 else (1 if close[i] > close[i-1] else -1)
    return signals

def strategy_obv_trend(df, period=10):
    """OBV trend: OBV rising = UP."""
    obv = calc_obv(df['close'].values, df['volume'].values)
    obv_ema = calc_ema(obv, period)
    signals = np.where(obv > obv_ema, 1, -1)
    return signals

def strategy_stoch_rsi(df, rsi_period=14, stoch_period=14):
    """Stochastic RSI: K<20=UP (oversold bounce), K>80=DOWN."""
    k, d = calc_stoch_rsi(df['close'].values, rsi_period, stoch_period)
    signals = np.zeros(len(df))
    signals[k < 20] = 1
    signals[k > 80] = -1
    # Fill neutrals with K vs D
    for i in range(len(signals)):
        if signals[i] == 0:
            if not np.isnan(k[i]) and not np.isnan(d[i]):
                signals[i] = 1 if k[i] > d[i] else -1
    return signals

def strategy_williams_r(df, period=14):
    """Williams %R: < -80 = UP (oversold), > -20 = DOWN (overbought)."""
    wr = calc_williams_r(df['high'].values, df['low'].values,
                         df['close'].values, period)
    signals = np.zeros(len(df))
    signals[wr < -80] = 1
    signals[wr > -20] = -1
    for i in range(1, len(signals)):
        if signals[i] == 0:
            signals[i] = signals[i-1] if signals[i-1] != 0 else 1
    return signals

def strategy_adx_trend(df, period=10, threshold=20):
    """ADX trending: ADX>threshold, follow DI direction."""
    adx, plus_di, minus_di = calc_adx(df['high'].values, df['low'].values,
                                       df['close'].values, period)
    signals = np.where(plus_di > minus_di, 1, -1)
    # When ADX < threshold (no trend), use momentum instead
    mom = calc_momentum(df['close'].values, 3)
    no_trend = adx < threshold
    signals[no_trend] = np.where(mom[no_trend] > 0, 1, -1)
    return signals

def strategy_taker_ratio(df):
    """Taker buy ratio: >55% = UP (buyers dominating)."""
    ratio = np.where(df['volume'].values > 0,
                      df['taker_buy_vol'].values / df['volume'].values, 0.5)
    signals = np.where(ratio > 0.55, 1, np.where(ratio < 0.45, -1, 0))
    for i in range(1, len(signals)):
        if signals[i] == 0:
            signals[i] = signals[i-1] if signals[i-1] != 0 else 1
    return signals

def strategy_price_action(df):
    """Pure price action: higher high + higher low = UP."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values

    signals = np.zeros(len(df))
    for i in range(2, len(df)):
        hh = high[i] > high[i-1]  # higher high
        hl = low[i] > low[i-1]    # higher low
        lh = high[i] < high[i-1]  # lower high
        ll = low[i] < low[i-1]    # lower low

        if hh and hl:
            signals[i] = 1   # UP trend
        elif lh and ll:
            signals[i] = -1  # DOWN trend
        else:
            signals[i] = signals[i-1]

    return signals

def strategy_candle_body(df):
    """Candle body direction: bullish candle = UP prediction for next bar."""
    close = df['close'].values
    open_p = df['open'].values
    signals = np.where(close > open_p, 1, -1)
    return signals

def strategy_multi_confluence(df):
    """Multi-indicator confluence: EMA + RSI + MACD agreement."""
    close = df['close'].values

    # EMA trend
    ema_fast = calc_ema(close, 5)
    ema_slow = calc_ema(close, 15)
    ema_sig = np.where(ema_fast > ema_slow, 1, -1)

    # RSI momentum
    rsi = calc_rsi(close, 6)
    rsi_sig = np.where(rsi > 55, 1, np.where(rsi < 45, -1, 0))

    # MACD
    _, _, hist = calc_macd(close, 8, 21, 5)
    macd_sig = np.where(hist > 0, 1, -1)

    # CVD
    _, delta = calc_cvd(df['taker_buy_vol'].values, df['volume'].values)
    delta_ema = calc_ema(delta, 3)
    cvd_sig = np.where(delta_ema > 0, 1, -1)

    # Sum and threshold
    total = ema_sig + rsi_sig + macd_sig + cvd_sig
    signals = np.where(total >= 2, 1, np.where(total <= -2, -1, 0))
    for i in range(1, len(signals)):
        if signals[i] == 0:
            signals[i] = signals[i-1] if signals[i-1] != 0 else 1
    return signals


# ============================================================
# POLYMARKET PnL SIMULATION
# ============================================================

def simulate_polymarket_pnl(predictions, actual_directions,
                            buy_price_winner=0.53, buy_price_loser=0.53,
                            lean_pct=0.65, size_per_trade=500,
                            confidence_threshold=0):
    """
    Simulate Polymarket-style PnL for 5-min BTC Up/Down markets.

    @vague-sourdough pattern:
    - Buys BOTH Up and Down each round (gabagool)
    - Leans directionally (puts more $ on predicted winner)
    - Winner side pays $1, loser side pays $0

    Args:
        predictions: array of -1/+1 predictions
        actual_directions: array of -1/+1 actual outcomes
        buy_price_winner: avg price paid for the side they lean toward
        buy_price_loser: avg price paid for the other side
        lean_pct: % of capital on predicted winner side
        size_per_trade: total $ per round (split between both sides)
        confidence_threshold: minimum signal strength to trade
    """
    n = len(predictions)
    pnl_per_trade = []
    total_volume = 0

    for i in range(n):
        pred = predictions[i]
        actual = actual_directions[i]

        if pred == 0:
            continue

        # Split capital: lean% on predicted winner, rest on other side
        winner_size = size_per_trade * lean_pct
        loser_size = size_per_trade * (1 - lean_pct)

        total_volume += size_per_trade

        if pred == actual:
            # Prediction correct:
            # - Winner side: bought at buy_price_winner, pays $1
            # - Loser side: bought at buy_price_loser, pays $0
            pnl = winner_size * (1.0 / buy_price_winner - 1) - loser_size
            pnl_per_trade.append(pnl)
        else:
            # Prediction wrong:
            # - Winner side (which lost): bought at buy_price_winner, pays $0
            # - Loser side (which won): bought at buy_price_loser, pays $1
            pnl = loser_size * (1.0 / buy_price_loser - 1) - winner_size
            pnl_per_trade.append(pnl)

    return pnl_per_trade, total_volume


def simulate_directional_pnl(predictions, actual_directions,
                               size_per_trade=500, buy_price=0.53):
    """
    Simpler model: just buy the predicted side at avg price.
    Win: pay buy_price, receive $1 (per share)
    Lose: pay buy_price, receive $0
    """
    pnl_per_trade = []
    total_volume = 0

    for i in range(len(predictions)):
        if predictions[i] == 0:
            continue

        shares = size_per_trade / buy_price
        total_volume += size_per_trade

        if predictions[i] == actual_directions[i]:
            pnl = shares * 1.0 - size_per_trade  # win
        else:
            pnl = -size_per_trade  # lose all

        pnl_per_trade.append(pnl)

    return pnl_per_trade, total_volume


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("REVERSE-ENGINEERING @vague-sourdough BTC 5-MINUTE STRATEGY")
    print("=" * 70)
    print()

    # Fetch data
    print("[1/4] Fetching 7 days of BTC 1-minute data from Binance...")
    df_1m = fetch_binance_klines('BTCUSDT', '1m', days=7)
    print(f"  Fetched {len(df_1m)} 1-minute bars")
    print(f"  Range: {df_1m.index[0]} to {df_1m.index[-1]}")

    # Aggregate to 5-min
    print("\n[2/4] Aggregating to 5-minute bars...")
    df_5m = aggregate_to_5min(df_1m)
    print(f"  {len(df_5m)} 5-minute bars")

    # Compute actual direction (next bar)
    close = df_5m['close'].values
    actual_dir = np.zeros(len(df_5m))
    actual_dir[:-1] = np.where(close[1:] > close[:-1], 1, -1)
    actual_dir[-1] = 0  # unknown

    # Warmup: skip first 50 bars for indicator warmup
    warmup = 50
    actual_dir_test = actual_dir[warmup:-1]  # exclude last (unknown)

    # Natural up/down distribution
    up_pct = np.sum(actual_dir_test == 1) / len(actual_dir_test) * 100
    print(f"  Natural UP frequency: {up_pct:.1f}%")
    print(f"  Natural DOWN frequency: {100-up_pct:.1f}%")

    # Define all strategies to test
    strategies = {
        'EMA_CROSS_5_15':      lambda d: strategy_ema_cross(d, 5, 15),
        'EMA_CROSS_3_8':       lambda d: strategy_ema_cross(d, 3, 8),
        'EMA_CROSS_8_21':      lambda d: strategy_ema_cross(d, 8, 21),
        'RSI_14':              lambda d: strategy_rsi_extremes(d, 14, 30, 70),
        'RSI_6_MOMENTUM':      lambda d: strategy_rsi_momentum(d, 6),
        'MACD_HISTOGRAM':      lambda d: strategy_macd(d),
        'MACD_CROSS_8_21':     lambda d: strategy_macd_cross(d, 8, 21, 5),
        'MOMENTUM_3BAR':       lambda d: strategy_momentum(d, 3),
        'MOMENTUM_5BAR':       lambda d: strategy_momentum_5bar(d),
        'BB_MEAN_REVERT':      lambda d: strategy_bb_mean_revert(d),
        'BB_BREAKOUT':         lambda d: strategy_bb_breakout(d),
        'VWAP':                lambda d: strategy_vwap(d),
        'CVD':                 lambda d: strategy_cvd(d),
        'CVD_DIVERGENCE':      lambda d: strategy_cvd_divergence(d),
        'OBV_TREND':           lambda d: strategy_obv_trend(d),
        'STOCH_RSI':           lambda d: strategy_stoch_rsi(d),
        'WILLIAMS_R':          lambda d: strategy_williams_r(d),
        'ADX_TREND':           lambda d: strategy_adx_trend(d),
        'TAKER_RATIO':         lambda d: strategy_taker_ratio(d),
        'PRICE_ACTION':        lambda d: strategy_price_action(d),
        'CANDLE_BODY':         lambda d: strategy_candle_body(d),
        'MULTI_CONFLUENCE':    lambda d: strategy_multi_confluence(d),
    }

    # Test each strategy
    print(f"\n[3/4] Testing {len(strategies)} strategies for 5-min direction prediction...")
    print(f"  Test period: {len(actual_dir_test)} bars ({len(actual_dir_test)*5/60:.0f} hours)")
    print()

    results = []

    for name, strat_fn in strategies.items():
        try:
            signals = strat_fn(df_5m)
            signals_test = signals[warmup:-1]  # align with actual_dir_test

            # Accuracy
            correct = np.sum(signals_test == actual_dir_test)
            accuracy = correct / len(actual_dir_test) * 100

            # Simulate PnL (Gabagool-style)
            pnl_trades, volume = simulate_polymarket_pnl(
                signals_test, actual_dir_test,
                buy_price_winner=0.53, buy_price_loser=0.53,
                lean_pct=0.65, size_per_trade=4500  # ~$1.95M / 441 trades
            )

            total_pnl = sum(pnl_trades)

            # Directional PnL (simpler model)
            dir_pnl_trades, dir_volume = simulate_directional_pnl(
                signals_test, actual_dir_test,
                size_per_trade=4500, buy_price=0.53
            )
            dir_total_pnl = sum(dir_pnl_trades)

            # Win rate on directional
            dir_wins = sum(1 for p in dir_pnl_trades if p > 0)
            dir_wr = dir_wins / len(dir_pnl_trades) * 100 if dir_pnl_trades else 0

            # Streak analysis
            max_win_streak = 0
            max_loss_streak = 0
            cur_win = 0
            cur_loss = 0
            for p in pnl_trades:
                if p > 0:
                    cur_win += 1
                    cur_loss = 0
                    max_win_streak = max(max_win_streak, cur_win)
                else:
                    cur_loss += 1
                    cur_win = 0
                    max_loss_streak = max(max_loss_streak, cur_loss)

            results.append({
                'name': name,
                'accuracy': accuracy,
                'trades': len(pnl_trades),
                'gabagool_pnl': total_pnl,
                'directional_pnl': dir_total_pnl,
                'directional_wr': dir_wr,
                'volume': volume,
                'max_win_streak': max_win_streak,
                'max_loss_streak': max_loss_streak,
                'avg_pnl_per_trade': total_pnl / len(pnl_trades) if pnl_trades else 0,
            })

        except Exception as e:
            print(f"  ERROR {name}: {e}")

    # Sort by accuracy (direction prediction quality)
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    # Print results table
    print(f"{'Strategy':<22} {'Accuracy':>8} {'Trades':>7} {'Gabagool PnL':>13} {'Dir PnL':>12} {'Dir WR':>7} {'Avg/Trade':>10}")
    print("-" * 85)

    for r in results:
        print(f"{r['name']:<22} {r['accuracy']:>7.1f}% {r['trades']:>7} "
              f"${r['gabagool_pnl']:>11,.0f} ${r['directional_pnl']:>10,.0f} "
              f"{r['directional_wr']:>6.1f}% ${r['avg_pnl_per_trade']:>8,.0f}")

    # Analyze best strategy
    print()
    print("=" * 70)
    best = results[0]
    print(f"BEST DIRECTION PREDICTOR: {best['name']}")
    print(f"  Accuracy: {best['accuracy']:.1f}%")
    print(f"  Gabagool PnL: ${best['gabagool_pnl']:,.0f}")
    print(f"  Directional PnL: ${best['directional_pnl']:,.0f}")
    print(f"  Avg PnL/trade: ${best['avg_pnl_per_trade']:,.0f}")
    print(f"  Win streak: {best['max_win_streak']} | Loss streak: {best['max_loss_streak']}")

    # Compare to random
    random_accuracy = 50.0
    edge = best['accuracy'] - random_accuracy
    print(f"\n  Edge over random: +{edge:.1f}% (random = 50%)")

    # @vague-sourdough benchmarks
    print()
    print("=" * 70)
    print("@vague-sourdough BENCHMARKS:")
    print(f"  PnL target: $38,711.74 over ~441 trades")
    print(f"  Volume: $1.95M")
    print(f"  Avg PnL/trade: ${38711.74/441:,.0f}")
    print(f"  Implied accuracy: ~{50 + 38711.74/(441*4500)*100:.1f}%+ (rough estimate)")

    # Strategy parameter sweep on top 3
    print()
    print("=" * 70)
    print("[4/4] PARAMETER SWEEP on top strategies...")
    print()

    # Sweep EMA periods
    best_ema = None
    best_ema_acc = 0
    for fast in [2, 3, 4, 5, 6, 8]:
        for slow in [8, 10, 12, 15, 20, 25]:
            if fast >= slow:
                continue
            signals = strategy_ema_cross(df_5m, fast, slow)
            signals_test = signals[warmup:-1]
            acc = np.sum(signals_test == actual_dir_test) / len(actual_dir_test) * 100
            if acc > best_ema_acc:
                best_ema_acc = acc
                best_ema = (fast, slow)

    print(f"  Best EMA cross: ({best_ema[0]}, {best_ema[1]}) = {best_ema_acc:.1f}%")

    # Sweep RSI periods
    best_rsi = None
    best_rsi_acc = 0
    for period in [3, 4, 5, 6, 8, 10, 14]:
        for thresh in [45, 50, 55, 60]:
            signals = strategy_rsi_momentum(df_5m, period)
            signals_test = signals[warmup:-1]
            rsi = calc_rsi(df_5m['close'].values, period)
            rsi_test = rsi[warmup:-1]
            # Custom threshold
            signals_custom = np.where(rsi_test > thresh, 1, np.where(rsi_test < (100-thresh), -1, 0))
            for j in range(1, len(signals_custom)):
                if signals_custom[j] == 0:
                    signals_custom[j] = signals_custom[j-1]
            acc = np.sum(signals_custom == actual_dir_test) / len(actual_dir_test) * 100
            if acc > best_rsi_acc:
                best_rsi_acc = acc
                best_rsi = (period, thresh)

    print(f"  Best RSI momentum: period={best_rsi[0]}, threshold={best_rsi[1]} = {best_rsi_acc:.1f}%")

    # Sweep MACD
    best_macd = None
    best_macd_acc = 0
    for fast in [6, 8, 10, 12]:
        for slow in [16, 21, 26, 30]:
            for sig in [3, 5, 7, 9]:
                if fast >= slow:
                    continue
                signals = strategy_macd_cross(df_5m, fast, slow, sig)
                signals_test = signals[warmup:-1]
                acc = np.sum(signals_test == actual_dir_test) / len(actual_dir_test) * 100
                if acc > best_macd_acc:
                    best_macd_acc = acc
                    best_macd = (fast, slow, sig)

    print(f"  Best MACD cross: ({best_macd[0]}, {best_macd[1]}, {best_macd[2]}) = {best_macd_acc:.1f}%")

    # Sweep momentum period
    best_mom = None
    best_mom_acc = 0
    for period in [1, 2, 3, 4, 5, 6, 8, 10]:
        signals = strategy_momentum(df_5m, period)
        signals_test = signals[warmup:-1]
        acc = np.sum(signals_test == actual_dir_test) / len(actual_dir_test) * 100
        if acc > best_mom_acc:
            best_mom_acc = acc
            best_mom = period

    print(f"  Best momentum: period={best_mom} = {best_mom_acc:.1f}%")

    # Final combined best
    print()
    print("=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)

    overall_best = max([
        ('EMA_CROSS', best_ema_acc, f"fast={best_ema[0]}, slow={best_ema[1]}"),
        ('RSI_MOMENTUM', best_rsi_acc, f"period={best_rsi[0]}, thresh={best_rsi[1]}"),
        ('MACD_CROSS', best_macd_acc, f"fast={best_macd[0]}, slow={best_macd[1]}, sig={best_macd[2]}"),
        ('MOMENTUM', best_mom_acc, f"period={best_mom}"),
    ], key=lambda x: x[1])

    print(f"  Best overall: {overall_best[0]} ({overall_best[2]}) = {overall_best[1]:.1f}%")

    # Simulate PnL with best params
    if overall_best[0] == 'EMA_CROSS':
        best_signals = strategy_ema_cross(df_5m, best_ema[0], best_ema[1])
    elif overall_best[0] == 'RSI_MOMENTUM':
        best_signals = strategy_rsi_momentum(df_5m, best_rsi[0])
    elif overall_best[0] == 'MACD_CROSS':
        best_signals = strategy_macd_cross(df_5m, best_macd[0], best_macd[1], best_macd[2])
    else:
        best_signals = strategy_momentum(df_5m, best_mom)

    best_signals_test = best_signals[warmup:-1]

    # Different sizing scenarios
    print()
    print("  PnL at different position sizes:")
    for size in [500, 1000, 2500, 4500, 10000]:
        pnl_trades, vol = simulate_polymarket_pnl(
            best_signals_test, actual_dir_test,
            buy_price_winner=0.53, buy_price_loser=0.53,
            lean_pct=0.65, size_per_trade=size
        )
        total = sum(pnl_trades)
        print(f"    ${size:>6}/trade: PnL=${total:>10,.0f} | Volume=${vol:>12,.0f} | Trades={len(pnl_trades)}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'data_range': f"{df_1m.index[0]} to {df_1m.index[-1]}",
        'bars_1m': len(df_1m),
        'bars_5m': len(df_5m),
        'test_bars': len(actual_dir_test),
        'natural_up_pct': float(up_pct),
        'strategies': results,
        'param_sweep': {
            'best_ema': {'fast': best_ema[0], 'slow': best_ema[1], 'accuracy': float(best_ema_acc)},
            'best_rsi': {'period': best_rsi[0], 'threshold': best_rsi[1], 'accuracy': float(best_rsi_acc)},
            'best_macd': {'fast': best_macd[0], 'slow': best_macd[1], 'signal': best_macd[2], 'accuracy': float(best_macd_acc)},
            'best_momentum': {'period': best_mom, 'accuracy': float(best_mom_acc)},
        },
        'overall_best': {
            'strategy': overall_best[0],
            'params': overall_best[2],
            'accuracy': float(overall_best[1]),
        },
        'benchmark': {
            'target_pnl': 38711.74,
            'target_trades': 441,
            'target_volume': 1950000,
        }
    }

    output_path = 'C:/Users/Star/.local/bin/star-polymarket/vague_sourdough_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to {output_path}")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
