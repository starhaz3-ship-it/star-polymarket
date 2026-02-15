"""
@vague-sourdough V2: Test INTRA-BAR signals using 1-minute data.

Key insight from V1: No standard indicator on 5-min bars achieves >51.3% accuracy.
Hypothesis: The edge comes from SHORT-TERM momentum — checking Binance price
just before the Polymarket 5-min round resolves.

This tests:
1. Last N minutes of momentum as predictor for 5-min bar close direction
2. Selective entry: only trade when signal is strong
3. Oracles: Binance price leads Polymarket by seconds
"""

import numpy as np
import pandas as pd
import requests
import time
import json
from datetime import datetime


def fetch_binance_klines(symbol='BTCUSDT', interval='1m', days=7):
    """Fetch historical klines from Binance."""
    url = 'https://api.binance.com/api/v3/klines'
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 60 * 60 * 1000)
    all_klines = []
    current = start_ms
    while current < end_ms:
        params = {'symbol': symbol, 'interval': interval,
                  'startTime': current, 'limit': 1000}
        resp = requests.get(url, params=params)
        data = resp.json()
        if not data:
            break
        all_klines.extend(data)
        current = data[-1][0] + 1
        if len(data) < 1000:
            break
        time.sleep(0.1)

    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_vol',
        'taker_buy_quote_vol', 'ignore'])
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                'taker_buy_vol', 'taker_buy_quote_vol']:
        df[col] = df[col].astype(float)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def calc_ema(data, period):
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


def calc_rsi(close, period=14):
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


def main():
    print("=" * 70)
    print("@vague-sourdough V2: INTRA-BAR MOMENTUM ANALYSIS")
    print("=" * 70)

    # Fetch 1-minute data
    print("\n[1/4] Fetching 7 days of BTC 1-minute data...")
    df_1m = fetch_binance_klines('BTCUSDT', '1m', days=7)
    print(f"  {len(df_1m)} bars: {df_1m.index[0]} to {df_1m.index[-1]}")

    close_1m = df_1m['close'].values
    volume_1m = df_1m['volume'].values
    taker_buy_1m = df_1m['taker_buy_vol'].values

    # Group into 5-minute windows
    # Each 5-min window has bars [0,1,2,3,4] where bar 4 = last minute
    n_windows = len(df_1m) // 5
    print(f"  {n_windows} complete 5-minute windows")

    # For each window, compute the 5-min bar's open and close
    window_open = np.array([close_1m[i*5] for i in range(n_windows)])  # open of first 1m bar
    window_close = np.array([close_1m[i*5 + 4] for i in range(n_windows)])  # close of last 1m bar
    actual_dir = np.where(window_close > window_open, 1, -1)

    # Skip first window for warmup
    warmup = 1

    print(f"\n[2/4] Testing INTRA-BAR momentum strategies...")
    print(f"  {n_windows - warmup} windows to evaluate\n")

    results = []

    # ================================================================
    # Strategy 1: Last-minute momentum
    # Use the direction of the last 1-3 minutes WITHIN the 5-min bar
    # as predictor for the NEXT 5-min bar
    # ================================================================
    for lookback in [1, 2, 3, 4]:
        label = f"LAST_{lookback}MIN_MOM"
        predictions = np.zeros(n_windows)
        for i in range(warmup, n_windows):
            # Last N minutes of current 5-min window
            start_idx = i * 5 + (5 - lookback)
            end_idx = i * 5 + 4
            mom = close_1m[end_idx] - close_1m[start_idx]
            predictions[i] = 1 if mom > 0 else -1

        pred_test = predictions[warmup:]
        actual_test = actual_dir[warmup:]
        acc = np.sum(pred_test == actual_test) / len(actual_test) * 100
        results.append((label, acc, pred_test, actual_test))

    # ================================================================
    # Strategy 2: Intra-bar volume imbalance
    # If taker buys > 55% of volume in last N minutes, predict UP
    # ================================================================
    for lookback in [1, 2, 3]:
        label = f"LAST_{lookback}MIN_TAKER"
        predictions = np.zeros(n_windows)
        for i in range(warmup, n_windows):
            start_idx = i * 5 + (5 - lookback)
            end_idx = i * 5 + 5
            total_vol = np.sum(volume_1m[start_idx:end_idx])
            buy_vol = np.sum(taker_buy_1m[start_idx:end_idx])
            ratio = buy_vol / total_vol if total_vol > 0 else 0.5
            predictions[i] = 1 if ratio > 0.52 else -1

        pred_test = predictions[warmup:]
        actual_test = actual_dir[warmup:]
        acc = np.sum(pred_test == actual_test) / len(actual_test) * 100
        results.append((label, acc, pred_test, actual_test))

    # ================================================================
    # Strategy 3: Momentum continuation — if this bar was UP, next UP
    # (serial correlation test)
    # ================================================================
    label = "SERIAL_CORR"
    predictions = np.zeros(n_windows)
    for i in range(warmup, n_windows):
        predictions[i] = actual_dir[i - 1]  # Same as previous bar

    pred_test = predictions[warmup:]
    actual_test = actual_dir[warmup:]
    acc = np.sum(pred_test == actual_test) / len(actual_test) * 100
    results.append((label, acc, pred_test, actual_test))

    # ================================================================
    # Strategy 4: Mean reversion — if this bar was UP, next DOWN
    # ================================================================
    label = "MEAN_REVERT"
    predictions = np.zeros(n_windows)
    for i in range(warmup, n_windows):
        predictions[i] = -actual_dir[i - 1]  # Opposite of previous bar

    pred_test = predictions[warmup:]
    actual_test = actual_dir[warmup:]
    acc = np.sum(pred_test == actual_test) / len(actual_test) * 100
    results.append((label, acc, pred_test, actual_test))

    # ================================================================
    # Strategy 5: Candle shape prediction
    # Long lower wick = buy pressure = next bar UP
    # ================================================================
    label = "WICK_SIGNAL"
    predictions = np.zeros(n_windows)
    for i in range(warmup, n_windows):
        idx = i * 5  # first bar of window
        o = df_1m['open'].values[idx]
        h = max(df_1m['high'].values[idx:idx+5])
        l = min(df_1m['low'].values[idx:idx+5])
        c = close_1m[idx + 4]

        body = abs(c - o)
        full_range = h - l
        if full_range < 1e-8:
            predictions[i] = predictions[i-1] if i > warmup else 1
            continue

        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)

        if lower_wick > upper_wick * 1.5:
            predictions[i] = 1  # Buy pressure (hammer)
        elif upper_wick > lower_wick * 1.5:
            predictions[i] = -1  # Sell pressure (shooting star)
        else:
            predictions[i] = 1 if c > o else -1

    pred_test = predictions[warmup:]
    actual_test = actual_dir[warmup:]
    acc = np.sum(pred_test == actual_test) / len(actual_test) * 100
    results.append((label, acc, pred_test, actual_test))

    # ================================================================
    # Strategy 6: RSI on 1-minute data (fast RSI)
    # ================================================================
    rsi_1m = calc_rsi(close_1m, 6)
    for label_suffix, threshold in [("55", 55), ("52", 52), ("50", 50)]:
        label = f"RSI6_1M_{label_suffix}"
        predictions = np.zeros(n_windows)
        for i in range(warmup, n_windows):
            rsi_val = rsi_1m[i * 5 + 4]  # RSI at last minute of window
            predictions[i] = 1 if rsi_val > int(threshold) else -1

        pred_test = predictions[warmup:]
        actual_test = actual_dir[warmup:]
        acc = np.sum(pred_test == actual_test) / len(actual_test) * 100
        results.append((label, acc, pred_test, actual_test))

    # ================================================================
    # Strategy 7: EMA trend on 1-minute data
    # ================================================================
    ema5 = calc_ema(close_1m, 5)
    ema15 = calc_ema(close_1m, 15)
    label = "EMA_5_15_1M"
    predictions = np.zeros(n_windows)
    for i in range(warmup, n_windows):
        idx = i * 5 + 4
        predictions[i] = 1 if ema5[idx] > ema15[idx] else -1

    pred_test = predictions[warmup:]
    actual_test = actual_dir[warmup:]
    acc = np.sum(pred_test == actual_test) / len(actual_test) * 100
    results.append((label, acc, pred_test, actual_test))

    # ================================================================
    # Strategy 8: Combined — last-minute momentum + taker ratio
    # ================================================================
    label = "COMBO_MOM_TAKER"
    predictions = np.zeros(n_windows)
    for i in range(warmup, n_windows):
        # Last 2 minutes momentum
        mom = close_1m[i*5 + 4] - close_1m[i*5 + 2]
        mom_sig = 1 if mom > 0 else -1

        # Taker buy ratio last 2 min
        total_vol = np.sum(volume_1m[i*5+3:i*5+5])
        buy_vol = np.sum(taker_buy_1m[i*5+3:i*5+5])
        ratio = buy_vol / total_vol if total_vol > 0 else 0.5
        taker_sig = 1 if ratio > 0.52 else -1

        # Agree = trade, disagree = use momentum
        if mom_sig == taker_sig:
            predictions[i] = mom_sig
        else:
            predictions[i] = mom_sig

    pred_test = predictions[warmup:]
    actual_test = actual_dir[warmup:]
    acc = np.sum(pred_test == actual_test) / len(actual_test) * 100
    results.append((label, acc, pred_test, actual_test))

    # ================================================================
    # Strategy 9: SELECTIVE ENTRY — only trade when signal is strong
    # Test: momentum > threshold = confident, trade with lean
    # ================================================================
    print("\n[3/4] Testing SELECTIVE ENTRY strategies...")
    print("  (Only trade when confidence is high)\n")

    selective_results = []

    for mom_bars in [1, 2, 3]:
        for threshold_bps in [5, 10, 15, 20, 30, 50]:  # basis points
            threshold = threshold_bps / 10000.0  # Convert to fraction
            label = f"SELECT_{mom_bars}M_{threshold_bps}bp"
            predictions = []
            actuals = []

            for i in range(warmup, n_windows - 1):
                start_idx = i * 5 + (5 - mom_bars)
                end_idx = i * 5 + 4
                mom = (close_1m[end_idx] - close_1m[start_idx]) / close_1m[start_idx]

                if abs(mom) > threshold:
                    pred = 1 if mom > 0 else -1
                    predictions.append(pred)
                    actuals.append(actual_dir[i + 1])  # NEXT bar direction

            if len(predictions) < 10:
                continue

            predictions = np.array(predictions)
            actuals = np.array(actuals)
            acc = np.sum(predictions == actuals) / len(actuals) * 100
            trade_pct = len(predictions) / (n_windows - warmup - 1) * 100

            # PnL simulation — pure directional at $4400/trade, buy at 0.53
            wins = np.sum(predictions == actuals)
            losses = len(predictions) - wins
            pnl = wins * (4400 / 0.53 - 4400) - losses * 4400  # win: get shares*$1 - cost, lose: -cost

            selective_results.append({
                'label': label,
                'accuracy': acc,
                'trades': len(predictions),
                'trade_pct': trade_pct,
                'pnl': pnl,
                'avg_pnl': pnl / len(predictions),
                'mom_bars': mom_bars,
                'threshold_bps': threshold_bps,
            })

    # Also test selective with taker ratio
    for ratio_thresh in [0.54, 0.56, 0.58, 0.60]:
        label = f"SELECT_TAKER_{int(ratio_thresh*100)}"
        predictions = []
        actuals = []

        for i in range(warmup, n_windows - 1):
            total_vol = np.sum(volume_1m[i*5:i*5+5])
            buy_vol = np.sum(taker_buy_1m[i*5:i*5+5])
            ratio = buy_vol / total_vol if total_vol > 0 else 0.5

            if abs(ratio - 0.5) > (ratio_thresh - 0.5):
                pred = 1 if ratio > 0.5 else -1
                predictions.append(pred)
                actuals.append(actual_dir[i + 1])

        if len(predictions) < 10:
            continue

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        acc = np.sum(predictions == actuals) / len(actuals) * 100
        trade_pct = len(predictions) / (n_windows - warmup - 1) * 100

        wins = np.sum(predictions == actuals)
        losses = len(predictions) - wins
        pnl = wins * (4400 / 0.53 - 4400) - losses * 4400

        selective_results.append({
            'label': label,
            'accuracy': acc,
            'trades': len(predictions),
            'trade_pct': trade_pct,
            'pnl': pnl,
            'avg_pnl': pnl / len(predictions),
        })

    # ================================================================
    # Strategy 10: PREDICTION OF CURRENT BAR (not next bar)
    # Key insight: Polymarket 5-min markets resolve based on whether
    # BTC goes UP or DOWN in that 5-min window. The prediction is made
    # DURING the window. So if you can predict current bar direction
    # from the first 1-2 minutes, that's the edge.
    # ================================================================
    print("\n[4/4] Testing CURRENT-BAR prediction (first N minutes -> final direction)...")
    print("  This is the actual Polymarket model: predict THIS bar from early data\n")

    current_bar_results = []

    for look_mins in [1, 2, 3]:
        label = f"EARLY_{look_mins}MIN"
        predictions = np.zeros(n_windows)
        for i in range(n_windows):
            # Direction of first N minutes predicts full 5-min direction
            early_close = close_1m[i*5 + look_mins - 1]
            bar_open = close_1m[i*5] if i*5 > 0 else close_1m[0]
            # Actually use the open of the 5-min bar
            bar_open = df_1m['open'].values[i*5]
            mom = early_close - bar_open
            predictions[i] = 1 if mom > 0 else -1

        acc = np.sum(predictions == actual_dir) / len(actual_dir) * 100
        current_bar_results.append((label, acc, len(actual_dir)))

    # Selective current-bar
    for look_mins in [1, 2, 3]:
        for threshold_bps in [5, 10, 15, 20, 30, 50]:
            threshold = threshold_bps / 10000.0
            label = f"EARLY_{look_mins}M_{threshold_bps}bp"
            correct = 0
            total = 0

            for i in range(n_windows):
                bar_open = df_1m['open'].values[i*5]
                early_close = close_1m[i*5 + look_mins - 1]
                mom = (early_close - bar_open) / bar_open

                if abs(mom) > threshold:
                    pred = 1 if mom > 0 else -1
                    if pred == actual_dir[i]:
                        correct += 1
                    total += 1

            if total < 10:
                continue

            acc = correct / total * 100
            trade_pct = total / n_windows * 100

            # PnL (buy during window when confident)
            wins = correct
            losses = total - correct
            pnl = wins * (4400 / 0.53 - 4400) - losses * 4400

            current_bar_results.append((label, acc, total, trade_pct, pnl))

    # ================================================================
    # Print all results
    # ================================================================
    print("=" * 70)
    print("RESULTS: NEXT-BAR PREDICTION (standard indicators on 1m data)")
    print("=" * 70)

    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'Strategy':<22} {'Accuracy':>8} {'Edge':>6}")
    print("-" * 40)
    for name, acc, _, _ in results:
        edge = acc - 50
        marker = " ***" if acc > 51 else ""
        print(f"{name:<22} {acc:>7.1f}% {edge:>+5.1f}%{marker}")

    print()
    print("=" * 70)
    print("RESULTS: SELECTIVE NEXT-BAR (only trade when confident)")
    print("=" * 70)

    selective_results.sort(key=lambda x: x['accuracy'], reverse=True)
    print(f"\n{'Strategy':<28} {'Acc':>6} {'Trades':>7} {'%Traded':>8} {'PnL':>12} {'$/Trade':>8}")
    print("-" * 75)
    for r in selective_results[:20]:
        marker = " ***" if r['accuracy'] > 53 else ""
        print(f"{r['label']:<28} {r['accuracy']:>5.1f}% {r['trades']:>7} {r['trade_pct']:>7.1f}% "
              f"${r['pnl']:>10,.0f} ${r['avg_pnl']:>6,.0f}{marker}")

    print()
    print("=" * 70)
    print("RESULTS: CURRENT-BAR PREDICTION (first N minutes -> 5m close)")
    print("=" * 70)
    print("  This is the ACTUAL Polymarket model\n")

    print(f"{'Strategy':<28} {'Acc':>6} {'Trades':>7} {'%Traded':>8} {'PnL':>12}")
    print("-" * 70)

    # Simple current-bar
    for item in current_bar_results:
        if len(item) == 3:
            label, acc, trades = item
            edge = acc - 50
            marker = " ***" if acc > 55 else ""
            print(f"{label:<28} {acc:>5.1f}% {trades:>7} {'100.0%':>8} {'N/A':>12}{marker}")
        else:
            label, acc, trades, trade_pct, pnl = item
            marker = " ***" if acc > 55 else ""
            print(f"{label:<28} {acc:>5.1f}% {trades:>7} {trade_pct:>7.1f}% ${pnl:>10,.0f}{marker}")

    # ================================================================
    # SYNTHESIS
    # ================================================================
    print()
    print("=" * 70)
    print("SYNTHESIS: HOW @vague-sourdough LIKELY PROFITS")
    print("=" * 70)

    # Find best current-bar result
    best_current = None
    best_current_acc = 0
    for item in current_bar_results:
        if len(item) >= 3:
            if item[1] > best_current_acc:
                best_current_acc = item[1]
                best_current = item

    print(f"""
  FINDINGS:
  1. Standard 5-min indicators: MAX {results[0][1]:.1f}% accuracy (almost random)
  2. Selective next-bar: Best {selective_results[0]['accuracy']:.1f}% ({selective_results[0]['trades']} trades)
  3. Current-bar prediction: Best {best_current_acc:.1f}%

  KEY INSIGHT: The edge is likely from CURRENT-BAR prediction:
  - Watch Binance BTC price for 1-2 minutes after a 5-min round starts
  - If BTC is already moving UP in the first 1-2 minutes, buy UP on Polymarket
  - The first 1-2 minutes of a 5-min bar predict the close {best_current_acc:.0f}% of the time
  - This is a MOMENTUM CONTINUATION effect within the bar

  PROFIT MODEL:
  - They trade ~441 rounds out of ~864 (3 days = 51% selectivity)
  - They skip rounds where early momentum is unclear
  - They buy at ~$0.52-0.53 (market price during early window)
  - With {best_current_acc:.0f}% accuracy and $4,400/trade:
    Win: $4,400/0.53 * $1 - $4,400 = ${4400/0.53 - 4400:,.0f} profit
    Lose: -$4,400
    Expected/trade: ${(best_current_acc/100) * (4400/0.53 - 4400) - (1 - best_current_acc/100) * 4400:,.0f}

  REPLICATION:
  1. When a new Polymarket BTC 5-min market opens, wait 1-2 minutes
  2. Check Binance BTC price vs the round's opening price
  3. If momentum > {selective_results[0].get('threshold_bps', 10)}bp, buy the direction on Polymarket
  4. Skip if momentum is unclear
  5. Use $500-1000/trade to start (their $4,400/trade requires $50K+ capital)
""")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'next_bar_strategies': [{'name': r[0], 'accuracy': r[1]} for r in results],
        'selective_strategies': selective_results[:20],
        'current_bar_strategies': [
            {'name': item[0], 'accuracy': item[1], 'trades': item[2],
             'trade_pct': item[3] if len(item) > 3 else 100,
             'pnl': item[4] if len(item) > 4 else None}
            for item in current_bar_results
        ],
    }

    with open('C:/Users/Star/.local/bin/star-polymarket/vague_sourdough_v2_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("  Results saved to vague_sourdough_v2_results.json")


if __name__ == '__main__':
    main()
