"""
FIB_CONFLUENCE_BOUNCE Backtest for Polymarket 5M Directional
Tests on 1-minute, 5-minute, and combined (1M signal + 5M confirmation)

Polymarket binary: buy UP or DOWN token at ~$0.48-0.49
Win = payout $1.00/share. Lose = $0.
Breakeven WR ~48% at $0.48 entry.
"""
import numpy as np
import httpx
import time
from functools import partial as fn_partial

print = fn_partial(print, flush=True)


# === FETCH CANDLES ===
def fetch_candles(symbol, interval, days=14):
    all_candles = []
    end_ms = int(time.time() * 1000)
    ms_per_bar = {"1m": 60_000, "5m": 300_000}[interval]
    needed = int(days * 86_400_000 / ms_per_bar) + 200
    batches = (needed // 1000) + 2

    for _ in range(batches):
        start_ms = end_ms - 1000 * ms_per_bar
        r = httpx.get("https://api.binance.com/api/v3/klines",
                      params={"symbol": symbol, "interval": interval,
                              "startTime": start_ms, "endTime": end_ms, "limit": 1000},
                      timeout=15)
        for b in r.json():
            all_candles.append({
                "time": int(b[0]), "open": float(b[1]), "high": float(b[2]),
                "low": float(b[3]), "close": float(b[4]), "volume": float(b[5])
            })
        end_ms = start_ms
        time.sleep(0.25)
        if len(all_candles) >= needed:
            break

    all_candles.sort(key=lambda x: x["time"])
    seen = set()
    return [c for c in all_candles if c["time"] not in seen and not seen.add(c["time"])]


# === INDICATORS ===
def calc_atr(high, low, close, period=14):
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


# === FIB_CONFLUENCE_BOUNCE SIGNAL ===
def signal_fib_confluence(close, high, low, open_, end_idx):
    """
    Evaluate FIB_CONFLUENCE_BOUNCE on data[:end_idx].
    Returns (side, conf) or (None, 0).
    """
    if end_idx < 55:
        return None, 0

    c = close[:end_idx]
    h = high[:end_idx]
    l = low[:end_idx]
    o = open_[:end_idx]
    n = len(c)

    fc_pass = False
    fc_side = None
    strong_confirm = False
    bars_after_touch_val = 0
    bars_after_val = 0
    swing_range_val = 0.0
    swing_high_val_g = 0.0

    # --- LONG setup ---
    for swing_lb in [30, 50]:
        if fc_pass or n < swing_lb + 5:
            break

        window_h = h[-(swing_lb + 1):-1]
        window_l = l[-(swing_lb + 1):-1]

        sh = np.max(window_h)
        sl_val = np.min(window_l)
        sr = sh - sl_val

        if sr <= 0 or sh <= 0:
            continue
        if (sr / sh) < 0.0015:
            continue

        fib_786 = sh - 0.786 * sr
        fib_618 = sh - 0.618 * sr
        fib_500 = sh - 0.500 * sr
        tol = sh * 0.0015

        # 0.786 touch scan (last 15 bars, not current)
        touched = False
        touch_bar = -1
        scan_s = max(0, n - 16)
        scan_e = n - 2

        for k in range(scan_e, scan_s - 1, -1):
            if l[k] <= fib_786 + tol and l[k] >= fib_786 - tol * 2:
                if c[k] > fib_786:
                    touched = True
                    touch_bar = k
                    break

        if not touched:
            for k in range(scan_e, scan_s - 1, -1):
                if l[k] <= fib_786 + tol and c[k] > fib_618:
                    touched = True
                    touch_bar = k
                    break

        if not touched:
            continue

        # 0.618 higher-low pivot after touch
        pivot = False
        bat = n - 1 - touch_bar
        if bat >= 2:
            recent_lows = l[touch_bar + 1:]
            lowest = np.min(recent_lows) if len(recent_lows) > 0 else 0
            if lowest >= fib_618 - tol:
                pivot = True
            elif lowest >= fib_786 - tol and lowest <= fib_618 + tol:
                pivot = True

        if not pivot:
            continue

        if (c[-1] > o[-1]) and (c[-1] > fib_618):
            fc_side = 'LONG'
            fc_pass = True
            strong_confirm = c[-1] > fib_500
            bars_after_touch_val = bat
            swing_range_val = sr
            swing_high_val_g = sh
            break

    # --- SHORT setup ---
    if not fc_pass:
        for swing_lb in [30, 50]:
            if fc_pass or n < swing_lb + 5:
                break

            window_h = h[-(swing_lb + 1):-1]
            window_l = l[-(swing_lb + 1):-1]

            sh = np.max(window_h)
            sl_val = np.min(window_l)
            sr = sh - sl_val

            if sr <= 0 or sl_val <= 0:
                continue
            if (sr / sl_val) < 0.0015:
                continue

            fib_786 = sl_val + 0.786 * sr
            fib_618 = sl_val + 0.618 * sr
            tol = sl_val * 0.0015

            touched = False
            touch_bar = -1
            scan_s = max(0, n - 16)
            scan_e = n - 2

            for k in range(scan_e, scan_s - 1, -1):
                if h[k] >= fib_786 - tol and h[k] <= fib_786 + tol * 2:
                    if c[k] < fib_786:
                        touched = True
                        touch_bar = k
                        break

            if not touched:
                for k in range(scan_e, scan_s - 1, -1):
                    if h[k] >= fib_786 - tol and c[k] < fib_618:
                        touched = True
                        touch_bar = k
                        break

            if not touched:
                continue

            pivot = False
            ba = n - 1 - touch_bar
            if ba >= 2:
                recent_highs = h[touch_bar + 1:]
                highest = np.max(recent_highs) if len(recent_highs) > 0 else float('inf')
                if highest <= fib_618 + tol:
                    pivot = True
                elif highest <= fib_786 + tol and highest >= fib_618 - tol:
                    pivot = True

            if not pivot:
                continue

            if (c[-1] < o[-1]) and (c[-1] < fib_618):
                fc_side = 'SHORT'
                fc_pass = True
                bars_after_val = ba
                swing_range_val = sr
                swing_high_val_g = sh
                break

    # ATR gate
    if fc_pass:
        atr = calc_atr(h, l, c, 14)
        atr_pct = (atr[-1] / c[-1] * 100) if c[-1] > 0 else 0
        if atr_pct < 0.15 or atr_pct > 8.0:
            fc_pass = False

    if fc_pass and fc_side:
        conf = 0.75
        if fc_side == 'LONG' and strong_confirm:
            conf += 0.05
        if swing_high_val_g > 0 and swing_range_val / swing_high_val_g <= 0.005:
            conf += 0.03
        if fc_side == 'LONG' and bars_after_touch_val >= 4:
            conf += 0.04
        elif fc_side == 'SHORT' and bars_after_val >= 4:
            conf += 0.04
        return fc_side, min(0.92, conf)

    return None, 0


# === BACKTEST ENGINE ===
def run_backtest(candles, outcome_bars, label):
    """
    For each bar, check for fib signal. If triggered, check if price moves
    in predicted direction over next outcome_bars.
    Polymarket pricing: UP entry $0.48, DOWN entry $0.49, $10 position.
    """
    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)
    open_ = np.array([c["open"] for c in candles], dtype=float)

    WARMUP = 60
    trades = []
    last_signal_bar = -999  # cooldown: no signal within 3 bars of last

    total_bars = len(candles) - outcome_bars
    pct_marks = set()

    for i in range(WARMUP, total_bars):
        # Progress
        pct = int((i - WARMUP) / max(1, total_bars - WARMUP) * 100)
        if pct % 20 == 0 and pct not in pct_marks:
            pct_marks.add(pct)
            print(f"  [{label}] {pct}% ({i - WARMUP}/{total_bars - WARMUP} bars scanned, {len(trades)} signals so far)")

        # Cooldown: skip if signal fired within last 3 bars
        if i - last_signal_bar < 3:
            continue

        side, conf = signal_fib_confluence(close, high, low, open_, i + 1)
        if side is None:
            continue

        last_signal_bar = i

        # Outcome: price direction over next outcome_bars
        entry_price = close[i]
        exit_price = close[i + outcome_bars]
        actual_up = exit_price > entry_price

        if side == 'LONG':
            entry_cost = 0.48
            correct = actual_up
        else:
            entry_cost = 0.49
            correct = not actual_up

        # PnL on $10 position
        if correct:
            trade_pnl = (10.0 / entry_cost) - 10.0
        else:
            trade_pnl = -10.0

        trades.append({
            'bar': i,
            'time': candles[i]["time"],
            'side': side,
            'conf': conf,
            'correct': correct,
            'pnl': trade_pnl,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'move_pct': (exit_price - entry_price) / entry_price * 100,
        })

    return trades


def print_results(trades, label, days):
    if not trades:
        print(f"\n{'='*70}")
        print(f"  {label}: NO TRADES")
        print(f"{'='*70}")
        return

    total = len(trades)
    wins = sum(1 for t in trades if t['correct'])
    losses = total - wins
    wr = wins / total * 100
    total_pnl = sum(t['pnl'] for t in trades)
    avg_pnl = total_pnl / total

    longs = [t for t in trades if t['side'] == 'LONG']
    shorts = [t for t in trades if t['side'] == 'SHORT']
    long_wins = sum(1 for t in longs if t['correct'])
    short_wins = sum(1 for t in shorts if t['correct'])

    win_pnls = [t['pnl'] for t in trades if t['correct']]
    loss_pnls = [t['pnl'] for t in trades if not t['correct']]
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else -1

    # Drawdown
    cumsum = np.cumsum([t['pnl'] for t in trades])
    peak = np.maximum.accumulate(cumsum)
    max_dd = float(np.max(peak - cumsum))

    daily = total_pnl / max(1, days)
    monthly = daily * 30
    tpd = total / max(1, days)

    # Consecutive losses
    streaks = []
    streak = 0
    for t in trades:
        if not t['correct']:
            streak += 1
        else:
            if streak > 0:
                streaks.append(streak)
            streak = 0
    if streak > 0:
        streaks.append(streak)

    # High confidence subset
    hc = [t for t in trades if t['conf'] >= 0.80]
    hc_wins = sum(1 for t in hc if t['correct'])

    # Confidence buckets
    conf_buckets = [
        ("0.75-0.78", 0.75, 0.79),
        ("0.79-0.82", 0.79, 0.83),
        ("0.83-0.87", 0.83, 0.88),
        ("0.88-0.92", 0.88, 0.93),
    ]

    # Breakeven: at $0.48 entry, win=$10.83, loss=-$10. BE=10/20.83=48.0%
    be_wr = 10.0 / (10.0 + 10.0 / 0.48 - 10.0) * 100

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Trades: {total} ({tpd:.1f}/day) | Days: {days:.1f}")
    print(f"  Win Rate: {wins}/{total} = {wr:.1f}%")
    print(f"  Total PnL: ${total_pnl:.2f} | Avg: ${avg_pnl:.2f}/trade")
    print(f"  Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}")
    print(f"  LONG: {long_wins}/{len(longs)} ({long_wins/max(1,len(longs))*100:.1f}%)")
    print(f"  SHORT: {short_wins}/{len(shorts)} ({short_wins/max(1,len(shorts))*100:.1f}%)")
    print(f"  MaxDD: ${max_dd:.2f} | $/day: ${daily:.2f} | $/month: ${monthly:.0f}")
    print(f"  Max consecutive losses: {max(streaks) if streaks else 0}")
    print(f"  Breakeven WR: {be_wr:.1f}% -> {'PROFITABLE' if wr > be_wr else 'UNPROFITABLE'}")

    if hc:
        print(f"\n  High conf (>=0.80): {hc_wins}/{len(hc)} ({hc_wins/max(1,len(hc))*100:.1f}%) PnL: ${sum(t['pnl'] for t in hc):.2f}")

    print(f"\n  Confidence Breakdown:")
    for bucket_label, lo, hi in conf_buckets:
        bucket = [t for t in trades if lo <= t['conf'] < hi]
        if bucket:
            bw = sum(1 for t in bucket if t['correct'])
            bp = sum(t['pnl'] for t in bucket)
            print(f"    {bucket_label}: {bw}/{len(bucket)} ({bw/len(bucket)*100:.1f}%) PnL: ${bp:.2f}")

    # Move size analysis (how big were the BTC moves on wins vs losses?)
    win_moves = [abs(t['move_pct']) for t in trades if t['correct']]
    loss_moves = [abs(t['move_pct']) for t in trades if not t['correct']]
    if win_moves and loss_moves:
        print(f"\n  BTC move on wins: avg {np.mean(win_moves):.3f}% | losses: avg {np.mean(loss_moves):.3f}%")


# ========== MAIN ==========
print("=" * 70)
print("FIB_CONFLUENCE_BOUNCE — Polymarket 5M Directional Backtest")
print("=" * 70)

# Fetch data
print("\nFetching 14 days of 1M BTC candles...")
candles_1m = fetch_candles("BTCUSDT", "1m", days=14)
days_1m = len(candles_1m) / 60 / 24
print(f"Got {len(candles_1m)} candles ({days_1m:.1f} days)")

time.sleep(1)

print("\nFetching 14 days of 5M BTC candles...")
candles_5m = fetch_candles("BTCUSDT", "5m", days=14)
days_5m = len(candles_5m) * 5 / 60 / 24
print(f"Got {len(candles_5m)} candles ({days_5m:.1f} days)")

# === TEST 1: 1M candles -> predict next 5 minutes ===
print("\n\n>>> TEST 1: 1-MINUTE candles, predict next 5 min direction")
trades_1m = run_backtest(candles_1m, outcome_bars=5, label="1M")
print_results(trades_1m, "TEST 1: 1-MINUTE CANDLES -> 5-min outcome", days_1m)

# === TEST 2: 5M candles -> predict next bar ===
print("\n\n>>> TEST 2: 5-MINUTE candles, predict next bar direction")
trades_5m = run_backtest(candles_5m, outcome_bars=1, label="5M")
print_results(trades_5m, "TEST 2: 5-MINUTE CANDLES -> next-bar outcome", days_5m)

# === TEST 3: COMBINED — 1M signal + 5M fib confirmation ===
print("\n\n>>> TEST 3: COMBINED (1M signal confirmed by 5M fib)")

# Pre-compute 5M signals for lookup
print("  Building 5M signal map...")
close_5m = np.array([c["close"] for c in candles_5m], dtype=float)
high_5m = np.array([c["high"] for c in candles_5m], dtype=float)
low_5m = np.array([c["low"] for c in candles_5m], dtype=float)
open_5m = np.array([c["open"] for c in candles_5m], dtype=float)

five_min_signals = {}
for idx in range(60, len(candles_5m)):
    side_5m, _ = signal_fib_confluence(close_5m, high_5m, low_5m, open_5m, idx + 1)
    if side_5m:
        five_min_signals[candles_5m[idx]["time"]] = side_5m

print(f"  5M signal map: {len(five_min_signals)} bars with signals")

# Filter 1M trades: keep only where 5M agrees
combined = []
for t in trades_1m:
    bar_time = candles_1m[t['bar']]["time"]
    aligned_5m = (bar_time // 300_000) * 300_000
    trend_5m = five_min_signals.get(aligned_5m)
    if trend_5m and trend_5m == t['side']:
        combined.append(t)

print_results(combined, "TEST 3: COMBINED (1M fib + 5M fib confirmation)", days_1m)

# === TEST 4: 1M with RELAXED confirmation (5M trend direction, not fib) ===
print("\n\n>>> TEST 4: 1M signal + 5M simple trend (EMA direction)")

# Simple 5M trend: EMA9 > EMA21 = UP, else DOWN
def calc_ema(arr, n):
    out = np.empty(len(arr))
    a = 2.0 / (n + 1.0)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out

ema9_5m = calc_ema(close_5m, 9)
ema21_5m = calc_ema(close_5m, 21)

five_min_trend = {}
for idx in range(21, len(candles_5m)):
    trend = "LONG" if ema9_5m[idx] > ema21_5m[idx] else "SHORT"
    five_min_trend[candles_5m[idx]["time"]] = trend

combined_trend = []
for t in trades_1m:
    bar_time = candles_1m[t['bar']]["time"]
    aligned_5m = (bar_time // 300_000) * 300_000
    trend_5m = five_min_trend.get(aligned_5m)
    if trend_5m and trend_5m == t['side']:
        combined_trend.append(t)

print_results(combined_trend, "TEST 4: 1M fib + 5M EMA trend confirmation", days_1m)


# ========== FINAL SUMMARY ==========
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print(f"{'='*70}")
be_wr = 10.0 / (10.0 + 10.0 / 0.48 - 10.0) * 100
print(f"  Breakeven WR: {be_wr:.1f}%\n")

configs = [
    ("1M candles only", trades_1m, days_1m),
    ("5M candles only", trades_5m, days_5m),
    ("1M + 5M fib confirm", combined, days_1m),
    ("1M + 5M EMA trend", combined_trend, days_1m),
]

print(f"  {'Config':<25} {'Trades':>7} {'WR':>7} {'PnL':>10} {'$/day':>8} {'$/mo':>8} {'Verdict':>12}")
print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*10} {'-'*8} {'-'*8} {'-'*12}")

best_daily = -9999
for name, trades, d in configs:
    if trades:
        pnl = sum(t['pnl'] for t in trades)
        daily = pnl / max(1, d)
        if daily > best_daily:
            best_daily = daily

for name, trades, d in configs:
    if not trades:
        print(f"  {name:<25} {'0':>7} {'N/A':>7} {'$0.00':>10} {'$0.00':>8} {'$0':>8} {'NO DATA':>12}")
        continue
    total = len(trades)
    wins = sum(1 for t in trades if t['correct'])
    wr = wins / total * 100
    pnl = sum(t['pnl'] for t in trades)
    daily = pnl / max(1, d)
    monthly = daily * 30
    verdict = "BEST" if daily == best_daily else ("OK" if wr > be_wr else "FAIL")
    marker = " ***" if daily == best_daily else ""
    print(f"  {name:<25} {total:>7} {wr:>6.1f}% ${pnl:>8.2f} ${daily:>6.2f} ${monthly:>6.0f} {verdict:>12}{marker}")

print("\nDone.")
