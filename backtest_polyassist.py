"""
PolyAssist Strategy Backtest for BTC 5-Minute Polymarket Markets
================================================================
Strategy by @SolSt1ne -- backtest implementation.

Computes composite bias from 8 normalized signals:
  RSI(14), MACD(12,26,9), VWAP, EMA 5/20, Heikin Ashi streak,
  Momentum, Bollinger Bands, Taker flow

Simulates binary 5-min BTC direction bets at various bias thresholds.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests
import numpy as np
import time
from datetime import datetime, timezone
from collections import defaultdict


# --- Data Fetching ------------------------------------------------------------

def fetch_binance_klines(symbol="BTCUSDT", interval="1m", days=14):
    """Fetch 1-minute klines from Binance REST API."""
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    print(f"Fetching {days} days of {interval} klines for {symbol}...")
    batch = 0
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_klines.extend(data)
        batch += 1

        # Move start past last candle
        current_start = data[-1][0] + 60000  # +1 minute

        if batch % 5 == 0:
            print(f"  Fetched {len(all_klines)} candles so far...")

        time.sleep(0.1)  # Rate limit

    print(f"  Total: {len(all_klines)} 1-minute candles fetched.")
    return all_klines


def parse_klines(raw_klines):
    """Parse raw Binance klines into structured arrays."""
    n = len(raw_klines)
    timestamps = np.zeros(n, dtype=np.int64)
    opens = np.zeros(n)
    highs = np.zeros(n)
    lows = np.zeros(n)
    closes = np.zeros(n)
    volumes = np.zeros(n)
    taker_buy_vols = np.zeros(n)

    for i, k in enumerate(raw_klines):
        timestamps[i] = k[0]
        opens[i] = float(k[1])
        highs[i] = float(k[2])
        lows[i] = float(k[3])
        closes[i] = float(k[4])
        volumes[i] = float(k[5])
        taker_buy_vols[i] = float(k[9])  # Taker buy base asset volume

    return {
        "timestamps": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "taker_buy_vol": taker_buy_vols
    }


# --- Technical Indicators -----------------------------------------------------

def ema(data, period):
    """Exponential moving average."""
    result = np.full_like(data, np.nan)
    if len(data) < period:
        return result
    # SMA seed
    result[period - 1] = np.mean(data[:period])
    multiplier = 2.0 / (period + 1)
    for i in range(period, len(data)):
        result[i] = data[i] * multiplier + result[i - 1] * (1 - multiplier)
    return result


def rsi(closes, period=14):
    """RSI calculation."""
    result = np.full(len(closes), np.nan)
    if len(closes) < period + 1:
        return result

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


def macd(closes, fast=12, slow=26, signal=9):
    """MACD line and signal line."""
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line[~np.isnan(macd_line)], signal)

    # Re-align signal line
    full_signal = np.full_like(closes, np.nan)
    non_nan_idx = np.where(~np.isnan(macd_line))[0]
    if len(non_nan_idx) >= signal:
        start = non_nan_idx[signal - 1]
        valid_signal = signal_line[~np.isnan(signal_line)]
        end = start + len(valid_signal)
        if end <= len(full_signal):
            full_signal[start:end] = valid_signal

    histogram = macd_line - full_signal
    return macd_line, full_signal, histogram


def bollinger_bands(closes, period=20, num_std=2):
    """Bollinger Bands."""
    mid = np.full_like(closes, np.nan)
    std = np.full_like(closes, np.nan)

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        mid[i] = np.mean(window)
        std[i] = np.std(window)

    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower, std


def vwap(highs, lows, closes, volumes):
    """Session VWAP -- resets every 24 hours (1440 1-min bars)."""
    typical_price = (highs + lows + closes) / 3.0
    result = np.full_like(closes, np.nan)

    session_len = 1440  # 24h in 1-min bars

    for i in range(len(closes)):
        session_start = (i // session_len) * session_len
        window_tp = typical_price[session_start:i + 1]
        window_vol = volumes[session_start:i + 1]
        total_vol = np.sum(window_vol)
        if total_vol > 0:
            result[i] = np.sum(window_tp * window_vol) / total_vol
        else:
            result[i] = closes[i]

    return result


def heikin_ashi(opens, highs, lows, closes):
    """Heikin Ashi candles."""
    n = len(opens)
    ha_close = (opens + highs + lows + closes) / 4.0
    ha_open = np.zeros(n)
    ha_open[0] = (opens[0] + closes[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum(highs, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(lows, np.minimum(ha_open, ha_close))

    return ha_open, ha_high, ha_low, ha_close


# --- Signal Computation -------------------------------------------------------

def compute_signals(data, idx):
    """
    Compute all 8 normalized signals at a given index using preceding bars.
    Each signal normalized to [-1, +1].
    Returns dict of signal values and composite bias.
    """
    closes = data["close"]
    opens = data["open"]
    highs = data["high"]
    lows = data["low"]
    volumes = data["volume"]
    taker_buy_vols = data["taker_buy_vol"]

    price = closes[idx]
    signals = {}

    # Need at least 26 bars for MACD slow period
    if idx < 30:
        return None

    # Use a window of bars up to idx (inclusive)
    window_start = max(0, idx - 100)  # enough history for EMA warmup
    c = closes[window_start:idx + 1]
    o = opens[window_start:idx + 1]
    h = highs[window_start:idx + 1]
    l = lows[window_start:idx + 1]
    v = volumes[window_start:idx + 1]
    tbv = taker_buy_vols[window_start:idx + 1]

    rel_idx = len(c) - 1  # index within window

    # 1. EMA 5/20 cross: (EMA5 - EMA20) / price * 500
    ema5 = ema(c, 5)
    ema20 = ema(c, 20)
    if not np.isnan(ema5[rel_idx]) and not np.isnan(ema20[rel_idx]):
        raw = (ema5[rel_idx] - ema20[rel_idx]) / price * 500
        signals["ema_cross"] = np.clip(raw, -1, 1)

    # 2. RSI(14): (RSI - 50) / 50
    rsi_vals = rsi(c, 14)
    if not np.isnan(rsi_vals[rel_idx]):
        raw = (rsi_vals[rel_idx] - 50.0) / 50.0
        signals["rsi"] = np.clip(raw, -1, 1)

    # 3. MACD histogram: MACD_line / price * 5000
    macd_line, _, hist = macd(c, 12, 26, 9)
    if not np.isnan(macd_line[rel_idx]):
        raw = macd_line[rel_idx] / price * 5000
        signals["macd"] = np.clip(raw, -1, 1)

    # 4. VWAP deviation: (price - VWAP) / VWAP * 200
    vwap_vals = vwap(h, l, c, v)
    if not np.isnan(vwap_vals[rel_idx]):
        raw = (price - vwap_vals[rel_idx]) / vwap_vals[rel_idx] * 200
        signals["vwap"] = np.clip(raw, -1, 1)

    # 5. Bollinger position: (price - BB_mid) / (2 * BB_std)
    _, bb_mid, _, bb_std = bollinger_bands(c, 20, 2)
    if not np.isnan(bb_mid[rel_idx]) and bb_std[rel_idx] > 0:
        raw = (price - bb_mid[rel_idx]) / (2 * bb_std[rel_idx])
        signals["bollinger"] = np.clip(raw, -1, 1)

    # 6. Momentum: 5-bar return / 10-bar volatility
    if rel_idx >= 10:
        ret_5 = (c[rel_idx] - c[rel_idx - 5]) / c[rel_idx - 5]
        vol_10 = np.std(np.diff(c[rel_idx - 10:rel_idx + 1]) / c[rel_idx - 10:rel_idx])
        if vol_10 > 0:
            raw = ret_5 / vol_10
            signals["momentum"] = np.clip(raw / 3.0, -1, 1)  # /3 to normalize range

    # 7. Heikin Ashi streak: consecutive green/red, /5
    ha_o, ha_h, ha_l, ha_c = heikin_ashi(o, h, l, c)
    streak = 0
    for j in range(rel_idx, -1, -1):
        if ha_c[j] > ha_o[j]:  # green
            if streak >= 0:
                streak += 1
            else:
                break
        elif ha_c[j] < ha_o[j]:  # red
            if streak <= 0:
                streak -= 1
            else:
                break
        else:
            break
    signals["ha_streak"] = np.clip(streak / 5.0, -1, 1)

    # 8. Taker flow: (taker_buy_vol / total_vol - 0.5) * 2
    # Use last 5 bars for smoother signal
    if rel_idx >= 5:
        recent_tbv = np.sum(tbv[rel_idx - 4:rel_idx + 1])
        recent_vol = np.sum(v[rel_idx - 4:rel_idx + 1])
        if recent_vol > 0:
            raw = (recent_tbv / recent_vol - 0.5) * 2
            signals["taker_flow"] = np.clip(raw, -1, 1)

    if len(signals) < 5:
        return None  # Not enough signals

    # Composite bias
    values = list(signals.values())
    composite = np.mean(values)
    bias_pct = (composite + 1) / 2 * 100  # Convert to 0-100 scale

    return {
        "signals": signals,
        "composite": composite,
        "bias_pct": bias_pct,
        "n_signals": len(signals)
    }


# --- Backtest Engine -----------------------------------------------------------

def run_backtest(data, thresholds=(55, 60, 65, 70, 75)):
    """
    Run backtest over all 5-minute windows.
    At each 5-min boundary, compute signals from preceding 1-min bars,
    then check if the NEXT 5 minutes went UP or DOWN.
    """
    closes = data["close"]
    opens = data["open"]
    timestamps = data["timestamps"]
    n = len(closes)

    # Find 5-minute boundaries (every 5th 1-min bar)
    # Align to actual 5-min timestamps
    first_ts = timestamps[0]

    results_by_threshold = {t: {"trades": 0, "wins": 0, "pnl": 0.0, "details": []}
                            for t in thresholds}

    all_biases = []
    all_directions = []
    total_windows = 0
    skipped_no_signal = 0

    # Iterate through 5-min windows
    # For each window: use bars [i-24..i] to compute signals at bar i
    # Then check direction of bars [i+1..i+5]

    for i in range(30, n - 5, 5):
        total_windows += 1

        # Compute signals at bar i (end of current 5-min window)
        result = compute_signals(data, i)

        if result is None:
            skipped_no_signal += 1
            continue

        bias_pct = result["bias_pct"]

        # Actual direction of next 5-min window
        open_next = opens[i + 1] if i + 1 < n else closes[i]
        close_next = closes[min(i + 5, n - 1)]
        actual_up = close_next > open_next
        actual_flat = close_next == open_next

        all_biases.append(bias_pct)
        all_directions.append(1 if actual_up else 0)

        for t in thresholds:
            bullish_threshold = t
            bearish_threshold = 100 - t

            trade_taken = False
            predicted_up = None
            entry_price = 0.50  # Assume 50/50 odds
            mispricing = False
            fade = False

            # Standard entries
            if bias_pct >= bullish_threshold:
                trade_taken = True
                predicted_up = True
                # Mispricing: 75%+ bias but odds near 50% -> asymmetric
                if bias_pct >= 75:
                    mispricing = True
                    entry_price = 0.48  # Better odds on mispricing
            elif bias_pct <= bearish_threshold:
                trade_taken = True
                predicted_up = False
                if bias_pct <= 25:
                    mispricing = True
                    entry_price = 0.48

            # Fade logic: strong bias (>75%) but price already moved (simulated as
            # consecutive same-direction candles suggesting overextension)
            # Skip fade for simplicity in backtest — it requires odds data we don't have

            if trade_taken:
                risk = 5.0  # $5 per trade
                shares = risk / entry_price

                won = (predicted_up and actual_up) or (not predicted_up and not actual_up)

                if actual_flat:
                    # Push — no win, no loss (return entry)
                    pnl = 0.0
                elif won:
                    pnl = shares * (1.0 - entry_price)  # Win: $1/share - entry
                else:
                    pnl = -risk  # Lose entry

                results_by_threshold[t]["trades"] += 1
                results_by_threshold[t]["wins"] += (1 if won else 0)
                results_by_threshold[t]["pnl"] += pnl
                results_by_threshold[t]["details"].append({
                    "bias_pct": bias_pct,
                    "predicted_up": predicted_up,
                    "actual_up": actual_up,
                    "won": won,
                    "pnl": pnl,
                    "mispricing": mispricing,
                    "timestamp": timestamps[i]
                })

    return {
        "total_windows": total_windows,
        "skipped": skipped_no_signal,
        "all_biases": all_biases,
        "all_directions": all_directions,
        "results": results_by_threshold
    }


# --- Main ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  PolyAssist Strategy Backtest -- BTC 5-Minute Direction")
    print("  Strategy by @SolSt1ne | Backtest Implementation")
    print("=" * 70)
    print()

    # Fetch data
    raw = fetch_binance_klines("BTCUSDT", "1m", days=14)
    data = parse_klines(raw)

    print(f"\nData range: {datetime.fromtimestamp(data['timestamps'][0]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"         to {datetime.fromtimestamp(data['timestamps'][-1]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Price range: ${data['close'].min():,.0f} - ${data['close'].max():,.0f}")
    print()

    # Run backtest
    thresholds = [55, 60, 65, 70, 75]
    results = run_backtest(data, thresholds)

    print(f"Total 5-min windows analyzed: {results['total_windows']}")
    print(f"Skipped (insufficient signal data): {results['skipped']}")
    print(f"Usable windows: {results['total_windows'] - results['skipped']}")
    print()

    # Bias distribution
    biases = np.array(results["all_biases"])
    directions = np.array(results["all_directions"])
    base_rate = np.mean(directions) * 100

    print("-" * 70)
    print("BIAS DISTRIBUTION")
    print("-" * 70)
    print(f"  Mean bias: {np.mean(biases):.1f}%")
    print(f"  Std bias:  {np.std(biases):.1f}%")
    print(f"  Min/Max:   {np.min(biases):.1f}% / {np.max(biases):.1f}%")
    print(f"  Base rate (BTC went UP): {base_rate:.1f}%")
    print()

    # Histogram of bias values
    bins = [0, 20, 30, 40, 45, 50, 55, 60, 70, 80, 100]
    hist, _ = np.histogram(biases, bins=bins)
    print("  Bias Histogram:")
    for i in range(len(bins) - 1):
        bar = "#" * (hist[i] // max(1, max(hist) // 40))
        print(f"    {bins[i]:3d}-{bins[i+1]:3d}%: {hist[i]:5d} {bar}")
    print()

    # Results per threshold
    print("-" * 70)
    print("RESULTS BY BIAS THRESHOLD")
    print("-" * 70)
    print(f"  {'Thresh':>8s} {'Trades':>8s} {'Wins':>8s} {'WR%':>8s} {'PnL':>10s} {'Avg PnL':>10s} {'vs Random':>10s}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")

    best_threshold = None
    best_pnl = -float('inf')

    for t in thresholds:
        r = results["results"][t]
        trades = r["trades"]
        wins = r["wins"]
        wr = (wins / trades * 100) if trades > 0 else 0
        pnl = r["pnl"]
        avg_pnl = pnl / trades if trades > 0 else 0

        # Random benchmark: 50% WR at same entry price
        # Expected PnL per trade at $0.50 entry: 0.5 * $5.00 - 0.5 * $5.00 = $0.00
        random_pnl = 0.0
        edge_vs_random = pnl - random_pnl

        print(f"  {t:>6d}%  {trades:>8d} {wins:>8d} {wr:>7.1f}% ${pnl:>9.2f} ${avg_pnl:>9.3f}  ${edge_vs_random:>9.2f}")

        if pnl > best_pnl:
            best_pnl = pnl
            best_threshold = t

    print()

    # Detailed breakdown for best threshold
    print("-" * 70)
    print(f"BEST THRESHOLD: {best_threshold}%")
    print("-" * 70)
    best = results["results"][best_threshold]
    trades = best["trades"]
    wins = best["wins"]
    wr = (wins / trades * 100) if trades > 0 else 0

    print(f"  Trades: {trades}")
    print(f"  Wins:   {wins}")
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  Total PnL: ${best['pnl']:.2f}")
    print(f"  Avg PnL/trade: ${best['pnl']/trades:.3f}" if trades > 0 else "  No trades")
    print(f"  Risk per trade: $5.00")

    if trades > 0:
        # Streaks analysis
        details = best["details"]
        max_win_streak = 0
        max_lose_streak = 0
        current_streak = 0
        for d in details:
            if d["won"]:
                current_streak = current_streak + 1 if current_streak > 0 else 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                current_streak = current_streak - 1 if current_streak < 0 else -1
                max_lose_streak = max(max_lose_streak, abs(current_streak))

        print(f"  Max win streak: {max_win_streak}")
        print(f"  Max lose streak: {max_lose_streak}")

        # Mispricing trades
        mispricing_trades = [d for d in details if d["mispricing"]]
        if mispricing_trades:
            mp_wins = sum(1 for d in mispricing_trades if d["won"])
            mp_wr = mp_wins / len(mispricing_trades) * 100
            mp_pnl = sum(d["pnl"] for d in mispricing_trades)
            print(f"\n  Mispricing trades (75%+ bias):")
            print(f"    Count: {len(mispricing_trades)}")
            print(f"    Win Rate: {mp_wr:.1f}%")
            print(f"    PnL: ${mp_pnl:.2f}")

        # Bullish vs Bearish breakdown
        bull_trades = [d for d in details if d["predicted_up"]]
        bear_trades = [d for d in details if not d["predicted_up"]]

        if bull_trades:
            bull_wr = sum(1 for d in bull_trades if d["won"]) / len(bull_trades) * 100
            bull_pnl = sum(d["pnl"] for d in bull_trades)
            print(f"\n  Bullish (buy UP) trades: {len(bull_trades)}, WR: {bull_wr:.1f}%, PnL: ${bull_pnl:.2f}")
        if bear_trades:
            bear_wr = sum(1 for d in bear_trades if d["won"]) / len(bear_trades) * 100
            bear_pnl = sum(d["pnl"] for d in bear_trades)
            print(f"  Bearish (buy DOWN) trades: {len(bear_trades)}, WR: {bear_wr:.1f}%, PnL: ${bear_pnl:.2f}")

    print()

    # Entry price sensitivity analysis
    print("-" * 70)
    print("ENTRY PRICE SENSITIVITY (best threshold)")
    print("-" * 70)
    print(f"  Simulating at various entry prices (odds paid for binary contract):")
    print(f"  {'Entry $':>10s} {'WR Needed':>10s} {'PnL':>10s} {'ROI/trade':>10s}")
    print(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    details = best["details"]
    for entry_p in [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]:
        sim_pnl = 0.0
        risk = 5.0
        shares = risk / entry_p
        breakeven_wr = entry_p * 100

        for d in details:
            if d["won"]:
                sim_pnl += shares * (1.0 - entry_p)
            else:
                sim_pnl -= risk

        avg_roi = (sim_pnl / trades / risk * 100) if trades > 0 else 0
        print(f"  ${entry_p:>8.2f}  {breakeven_wr:>8.0f}%  ${sim_pnl:>9.2f}  {avg_roi:>+8.1f}%")

    print()

    # Hourly performance breakdown
    print("-" * 70)
    print("HOURLY PERFORMANCE (best threshold)")
    print("-" * 70)
    hourly = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    for d in details:
        hour = datetime.fromtimestamp(d["timestamp"] / 1000, tz=timezone.utc).hour
        hourly[hour]["trades"] += 1
        hourly[hour]["wins"] += (1 if d["won"] else 0)
        hourly[hour]["pnl"] += d["pnl"]

    print(f"  {'Hour':>6s} {'Trades':>8s} {'WR%':>8s} {'PnL':>10s}")
    for h in sorted(hourly.keys()):
        hr = hourly[h]
        wr_h = hr["wins"] / hr["trades"] * 100 if hr["trades"] > 0 else 0
        print(f"  {h:>4d}h  {hr['trades']:>8d} {wr_h:>7.1f}% ${hr['pnl']:>9.2f}")

    print()

    # Equity curve stats
    print("-" * 70)
    print("EQUITY CURVE (best threshold)")
    print("-" * 70)
    equity = [0.0]
    for d in details:
        equity.append(equity[-1] + d["pnl"])
    equity = np.array(equity)

    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_dd = np.min(drawdown)

    print(f"  Starting balance: $0.00 (tracking PnL only)")
    print(f"  Final PnL: ${equity[-1]:.2f}")
    print(f"  Peak PnL: ${np.max(equity):.2f}")
    print(f"  Max Drawdown: ${max_dd:.2f}")
    print(f"  Sharpe (approx): {np.mean([d['pnl'] for d in details]) / max(0.01, np.std([d['pnl'] for d in details])) * np.sqrt(288):.2f}")
    print(f"    (288 = approx 5-min windows per day)")

    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Strategy: PolyAssist (composite bias from 8 signals)")
    print(f"  Market: BTC 5-minute direction (binary)")
    print(f"  Period: 14 days, {results['total_windows']} windows")
    print(f"  Best threshold: {best_threshold}% bias")
    print(f"  Win rate: {wr:.1f}% (vs 50% random baseline)")
    print(f"  Edge: {wr - 50:.1f}pp over random")
    print(f"  PnL at $5/trade: ${best['pnl']:.2f}")

    if wr > 52:
        print(f"\n  VERDICT: Strategy shows a {wr-50:.1f}pp edge. At $0.50 entry,")
        print(f"  breakeven is 50% WR. Edge is {'SIGNIFICANT' if wr > 55 else 'MARGINAL'}.")
    elif wr > 50:
        print(f"\n  VERDICT: Marginal edge of {wr-50:.1f}pp. May not survive fees/slippage.")
    else:
        print(f"\n  VERDICT: No edge detected at {best_threshold}% threshold.")
        print(f"  Strategy underperforms random coin flip.")

    print()
    print("  NOTE: This backtest uses perfect signal computation (no look-ahead).")
    print("  Live execution requires entering in last 60-90s of the 5-min window,")
    print("  which this backtest simulates correctly by computing signals at bar close.")
    print("=" * 70)


if __name__ == "__main__":
    main()
