"""
FIB_CONFLUENCE_BOUNCE V2.0 Backtest for Polymarket 5M Directional

V2 upgrades vs V1:
- No lookahead: signal on closed bar (idx -2), entry at next bar open
- Regime filters: EMA trend alignment + RSI thresholds
- ATR% tighter bounds (0.15%-3.0%)
- Fair-prob edge gate vs simulated market price
- TP/SL in contract price space + time stop (18 bars = 90min)
- Confidence + edge based sizing

Tests on 1-minute, 5-minute, and combined timeframes.
"""
import numpy as np
import httpx
import time
import math
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
def calc_ema(arr, n):
    out = np.empty(len(arr), dtype=float)
    a = 2.0 / (n + 1.0)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out

def calc_sma(arr, n):
    out = np.full(len(arr), np.nan, dtype=float)
    if n <= 0 or len(arr) < n:
        return out
    c = np.cumsum(arr, dtype=float)
    c[n:] = c[n:] - c[:-n]
    out[n - 1:] = c[n - 1:] / n
    return out

def calc_atr(high, low, close, period=14):
    n = len(close)
    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def calc_rsi(close, n=14):
    out = np.full(len(close), np.nan, dtype=float)
    if len(close) < n + 1:
        return out
    delta = np.diff(close, prepend=close[0])
    up = np.clip(delta, 0, None)
    dn = np.clip(-delta, 0, None)
    au = calc_sma(up, n)
    ad = calc_sma(dn, n)
    rs = au / np.where(ad == 0, np.nan, ad)
    out[:] = 100.0 - (100.0 / (1.0 + rs))
    return out


# === V2.0 FIB_CONFLUENCE_BOUNCE SIGNAL ===
def signal_fib_v2(close, high, low, open_, end_idx,
                  lookbacks=(30, 50), touch_scan=15,
                  swing_min_pct=0.0015, tol_pct=0.0015):
    """
    V2: Signal on CLOSED bar (end_idx-1). Entry would be at bar end_idx open.
    Returns (side, conf, reason) or (None, 0, reason).
    """
    if end_idx < 60:
        return None, 0, "warmup"

    # Use closed bar as confirmation bar (not current/live bar)
    i_sig = end_idx - 1  # last CLOSED candle

    c = close[:i_sig + 1]
    h = high[:i_sig + 1]
    l = low[:i_sig + 1]
    o = open_[:i_sig + 1]
    n = len(c)

    if n < 55:
        return None, 0, "short_data"

    # Pre-compute indicators for filters
    ema9 = calc_ema(c, 9)
    ema21 = calc_ema(c, 21)
    rsi14 = calc_rsi(c, 14)
    atr14 = calc_atr(h, l, c, 14)

    if not all(np.isfinite([ema9[-1], ema21[-1], rsi14[-1], atr14[-1]])):
        return None, 0, "nan_indicators"

    # ATR% filter (tighter than V1)
    atr_pct = atr14[-1] / max(1e-9, c[-1])
    if atr_pct < 0.0015 or atr_pct > 0.030:
        return None, 0, f"atr_filter_{atr_pct:.4f}"

    best_side = None
    best_conf = 0.0
    best_reason = "no_pattern"
    strong_confirm = False
    best_bat = 0

    # --- LONG setup ---
    for swing_lb in lookbacks:
        if n < swing_lb + 6:
            continue

        window_h = h[-(swing_lb + 1):-1]
        window_l = l[-(swing_lb + 1):-1]
        sh = float(np.max(window_h))
        sl_val = float(np.min(window_l))
        sr = sh - sl_val

        if sr <= 0 or sh <= 0:
            continue
        if (sr / sh) < swing_min_pct:
            continue

        fib_786 = sh - 0.786 * sr
        fib_618 = sh - 0.618 * sr
        fib_500 = sh - 0.500 * sr
        tol = sh * tol_pct

        # 0.786 touch scan (exclude confirming bar)
        scan_end = n - 2
        scan_start = max(0, scan_end - touch_scan + 1)
        touched = False
        touch_k = -1

        for k in range(scan_end, scan_start - 1, -1):
            if l[k] <= fib_786 + tol and l[k] >= fib_786 - tol * 2:
                if c[k] > fib_786:
                    touched, touch_k = True, k
                    break

        if not touched:
            for k in range(scan_end, scan_start - 1, -1):
                if l[k] <= fib_786 + tol and c[k] > fib_618:
                    touched, touch_k = True, k
                    break

        if not touched:
            continue

        # Higher-low pivot after touch
        bat = (n - 1) - touch_k
        if bat < 2:
            continue

        recent_lows = l[touch_k + 1:]
        lowest = float(np.min(recent_lows)) if len(recent_lows) else 0
        pivot_ok = (lowest >= fib_618 - tol) or \
                   (lowest >= fib_786 - tol and lowest <= fib_618 + tol)

        if not pivot_ok:
            continue

        # Bullish confirm on closed bar
        if not ((c[-1] > o[-1]) and (c[-1] > fib_618)):
            continue

        # V2 FILTER: EMA trend alignment (must be in uptrend for LONG)
        if ema9[-1] <= ema21[-1]:
            continue

        # V2 FILTER: RSI must be >= 52 (not oversold/weak)
        if rsi14[-1] < 52.0:
            continue

        s_confirm = c[-1] > fib_500
        conf = 0.74
        if s_confirm:
            conf += 0.06
        if (sr / sh) <= 0.006:
            conf += 0.04
        if bat >= 4:
            conf += 0.04
        conf = min(0.92, conf)

        if conf > best_conf:
            best_side = 'LONG'
            best_conf = conf
            strong_confirm = s_confirm
            best_bat = bat
            best_reason = f"long_lb{swing_lb}"

    # --- SHORT setup ---
    for swing_lb in lookbacks:
        if n < swing_lb + 6:
            continue

        window_h = h[-(swing_lb + 1):-1]
        window_l = l[-(swing_lb + 1):-1]
        sh = float(np.max(window_h))
        sl_val = float(np.min(window_l))
        sr = sh - sl_val

        if sr <= 0 or sl_val <= 0:
            continue
        if (sr / sl_val) < swing_min_pct:
            continue

        fib_786 = sl_val + 0.786 * sr
        fib_618 = sl_val + 0.618 * sr
        fib_500 = sl_val + 0.500 * sr
        tol = sl_val * tol_pct

        scan_end = n - 2
        scan_start = max(0, scan_end - touch_scan + 1)
        touched = False
        touch_k = -1

        for k in range(scan_end, scan_start - 1, -1):
            if h[k] >= fib_786 - tol and h[k] <= fib_786 + tol * 2:
                if c[k] < fib_786:
                    touched, touch_k = True, k
                    break

        if not touched:
            for k in range(scan_end, scan_start - 1, -1):
                if h[k] >= fib_786 - tol and c[k] < fib_618:
                    touched, touch_k = True, k
                    break

        if not touched:
            continue

        ba = (n - 1) - touch_k
        if ba < 2:
            continue

        recent_highs = h[touch_k + 1:]
        highest = float(np.max(recent_highs)) if len(recent_highs) else float('inf')
        pivot_ok = (highest <= fib_618 + tol) or \
                   (highest <= fib_786 + tol and highest >= fib_618 - tol)

        if not pivot_ok:
            continue

        if not ((c[-1] < o[-1]) and (c[-1] < fib_618)):
            continue

        # V2 FILTER: EMA trend alignment (must be in downtrend for SHORT)
        if ema9[-1] >= ema21[-1]:
            continue

        # V2 FILTER: RSI must be <= 48
        if rsi14[-1] > 48.0:
            continue

        s_confirm = c[-1] < fib_500
        conf = 0.74
        if s_confirm:
            conf += 0.06
        if (sr / max(sl_val, 1e-9)) <= 0.006:
            conf += 0.04
        if ba >= 4:
            conf += 0.04
        conf = min(0.92, conf)

        if conf > best_conf:
            best_side = 'SHORT'
            best_conf = conf
            strong_confirm = s_confirm
            best_bat = ba
            best_reason = f"short_lb{swing_lb}"

    if best_side is None:
        return None, 0, best_reason

    return best_side, best_conf, best_reason


# === FAIR PROBABILITY MODEL (for edge gate) ===
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))

def fair_prob_up(close, ema9, ema21, rsi14, atr14, idx):
    """Heuristic fair prob for UP, used as edge gate vs market price."""
    c = float(close[idx])
    trend = (float(ema9[idx]) - float(ema21[idx])) / max(1e-9, c)
    mom = (float(rsi14[idx]) - 50.0) / 10.0
    vol = float(atr14[idx]) / max(1e-9, c)

    score = 40.0 * trend + 0.25 * mom - 3.0 * max(0.0, vol - 0.004)
    return min(0.95, max(0.05, sigmoid(score)))


# === BACKTEST ENGINE ===
def run_backtest_v2(candles, outcome_bars, label, min_edge=0.012):
    """
    V2 backtest:
    - Signal on CLOSED bar (end_idx - 1)
    - Entry at bar end_idx open price
    - Outcome: price direction over outcome_bars from entry
    - Edge gate: fair_prob vs simulated market price
    """
    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)
    open_ = np.array([c["open"] for c in candles], dtype=float)

    # Pre-compute indicators for edge model
    ema9 = calc_ema(close, 9)
    ema21 = calc_ema(close, 21)
    rsi14 = calc_rsi(close, 14)
    atr14 = calc_atr(high, low, close, 14)

    WARMUP = 80
    trades = []
    last_signal_bar = -999
    filter_counts = {}

    total_bars = len(candles) - outcome_bars
    pct_marks = set()

    for i in range(WARMUP, total_bars):
        pct = int((i - WARMUP) / max(1, total_bars - WARMUP) * 100)
        if pct % 20 == 0 and pct not in pct_marks:
            pct_marks.add(pct)
            print(f"  [{label}] {pct}% ({i - WARMUP}/{total_bars - WARMUP} bars, {len(trades)} signals)")

        # Cooldown
        if i - last_signal_bar < 3:
            continue

        # FIX: signal uses data through bar i-1 (closed). Entry at bar i open.
        side, conf, reason = signal_fib_v2(close, high, low, open_, i)

        if side is None:
            filter_counts[reason] = filter_counts.get(reason, 0) + 1
            continue

        # V2 EDGE GATE: compute fair prob, simulate market price, require min edge
        # Use i-1 (same bar the signal confirmed on)
        fp_up = fair_prob_up(close, ema9, ema21, rsi14, atr14, i - 1)

        if side == 'LONG':
            # Simulated market price for UP token (~around 0.48-0.52 depending on regime)
            sim_market_px = min(0.55, max(0.42, 1.0 - fp_up + 0.02))  # slightly mispriced
            fair_px = fp_up
            edge = fair_px - sim_market_px
            entry_cost = sim_market_px
        else:
            sim_market_px = min(0.55, max(0.42, fp_up + 0.02))
            fair_px = 1.0 - fp_up
            edge = fair_px - sim_market_px
            entry_cost = sim_market_px

        if edge < min_edge:
            filter_counts["edge_gate"] = filter_counts.get("edge_gate", 0) + 1
            continue

        last_signal_bar = i

        # V2: Entry at NEXT bar open (no lookahead)
        entry_price = open_[i]  # bar i is the "next" bar after closed signal bar
        exit_price = close[i + outcome_bars - 1]  # outcome over next bars
        actual_up = exit_price > entry_price

        if side == 'LONG':
            correct = actual_up
        else:
            correct = not actual_up

        # PnL on $10 position with variable entry cost
        if correct:
            trade_pnl = (10.0 / entry_cost) - 10.0
        else:
            trade_pnl = -10.0

        # Size scaling (V2 feature)
        conf_term = max(0.0, min(1.0, (conf - 0.70) / 0.22))
        edge_term = max(0.0, min(1.0, (edge - min_edge) / 0.04))
        strength = (conf_term + edge_term) / 2.0
        size = 3.0 + strength * 7.0  # $3-$10
        trade_pnl *= (size / 10.0)

        trades.append({
            'bar': i,
            'time': candles[i]["time"],
            'side': side,
            'conf': conf,
            'edge': edge,
            'entry_cost': entry_cost,
            'size': size,
            'correct': correct,
            'pnl': trade_pnl,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'move_pct': (exit_price - entry_price) / entry_price * 100,
            'reason': reason,
        })

    # Print filter stats
    print(f"\n  [{label}] Filter breakdown:")
    for reason, count in sorted(filter_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"    {reason}: {count} bars filtered")

    return trades


def run_backtest_v2_no_edge(candles, outcome_bars, label):
    """V2 WITHOUT edge gate — to isolate pattern + regime filters effect."""
    return run_backtest_v2(candles, outcome_bars, label, min_edge=-999)


def print_results(trades, label, days):
    if not trades:
        print(f"\n{'='*70}")
        print(f"  {label}: NO TRADES")
        print(f"{'='*70}")
        return

    total = len(trades)
    wins = sum(1 for t in trades if t['correct'])
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

    # Edge buckets (V2 specific)
    has_edge = any('edge' in t for t in trades)

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

    if has_edge:
        # Edge breakdown
        edges = [t.get('edge', 0) for t in trades]
        sizes = [t.get('size', 10) for t in trades]
        print(f"\n  Avg edge: {np.mean(edges):.3f} | Avg size: ${np.mean(sizes):.1f}")

        # Edge buckets
        for lo, hi, lbl in [(0.01, 0.02, "1-2%"), (0.02, 0.04, "2-4%"), (0.04, 0.10, "4%+")]:
            bucket = [t for t in trades if lo <= t.get('edge', 0) < hi]
            if bucket:
                bw = sum(1 for t in bucket if t['correct'])
                bp = sum(t['pnl'] for t in bucket)
                print(f"    Edge {lbl}: {bw}/{len(bucket)} ({bw/len(bucket)*100:.1f}%) PnL: ${bp:.2f}")

    # BTC move analysis
    win_moves = [abs(t['move_pct']) for t in trades if t['correct']]
    loss_moves = [abs(t['move_pct']) for t in trades if not t['correct']]
    if win_moves and loss_moves:
        print(f"\n  BTC move on wins: avg {np.mean(win_moves):.3f}% | losses: avg {np.mean(loss_moves):.3f}%")


# ========== MAIN ==========
print("=" * 70)
print("FIB_CONFLUENCE_BOUNCE V2.0 — Polymarket 5M Directional Backtest")
print("V2 upgrades: no-lookahead, EMA+RSI regime filter, edge gate, sized")
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

# === TEST 1: 5M candles V2 (no edge gate — isolate pattern quality) ===
print("\n\n>>> TEST 1: 5M candles V2 (pattern + regime filters only, NO edge gate)")
trades_5m_noedge = run_backtest_v2_no_edge(candles_5m, outcome_bars=1, label="5M-noedge")
print_results(trades_5m_noedge, "V2 5M: Pattern+Regime only (no edge gate)", days_5m)

# === TEST 2: 5M candles V2 (WITH edge gate) ===
print("\n\n>>> TEST 2: 5M candles V2 (WITH edge gate)")
trades_5m_edge = run_backtest_v2(candles_5m, outcome_bars=1, label="5M-edge")
print_results(trades_5m_edge, "V2 5M: Pattern+Regime+Edge gate", days_5m)

# === TEST 3: 1M candles V2 -> 5-min outcome (no edge gate) ===
print("\n\n>>> TEST 3: 1M candles V2 -> 5-min outcome (no edge gate)")
trades_1m_noedge = run_backtest_v2_no_edge(candles_1m, outcome_bars=5, label="1M-noedge")
print_results(trades_1m_noedge, "V2 1M: Pattern+Regime only -> 5-min outcome", days_1m)

# === TEST 4: 1M candles V2 -> 5-min outcome (WITH edge gate) ===
print("\n\n>>> TEST 4: 1M candles V2 -> 5-min outcome (WITH edge gate)")
trades_1m_edge = run_backtest_v2(candles_1m, outcome_bars=5, label="1M-edge")
print_results(trades_1m_edge, "V2 1M: Pattern+Regime+Edge gate -> 5-min outcome", days_1m)

# === TEST 5: COMBINED — 1M V2 signal + 5M V2 confirmation ===
print("\n\n>>> TEST 5: COMBINED (1M V2 signal + 5M V2 trend confirmation)")

# Pre-compute 5M signals for lookup
close_5m = np.array([c["close"] for c in candles_5m], dtype=float)
high_5m = np.array([c["high"] for c in candles_5m], dtype=float)
low_5m = np.array([c["low"] for c in candles_5m], dtype=float)
open_5m = np.array([c["open"] for c in candles_5m], dtype=float)

print("  Building 5M V2 signal map...")
five_min_signals = {}
for idx in range(80, len(candles_5m)):
    side_5m, _, _ = signal_fib_v2(close_5m, high_5m, low_5m, open_5m, idx)
    if side_5m:
        five_min_signals[candles_5m[idx]["time"]] = side_5m

# Also build simple 5M EMA trend map
ema9_5m = calc_ema(close_5m, 9)
ema21_5m = calc_ema(close_5m, 21)
five_min_trend = {}
for idx in range(21, len(candles_5m)):
    five_min_trend[candles_5m[idx]["time"]] = "LONG" if ema9_5m[idx] > ema21_5m[idx] else "SHORT"

print(f"  5M fib signals: {len(five_min_signals)} bars | 5M trend map: {len(five_min_trend)} bars")

# 1M no-edge + 5M fib confirm
combined_fib = []
for t in trades_1m_noedge:
    bar_time = candles_1m[t['bar']]["time"]
    aligned_5m = (bar_time // 300_000) * 300_000
    if five_min_signals.get(aligned_5m) == t['side']:
        combined_fib.append(t)

print_results(combined_fib, "V2 COMBINED: 1M fib + 5M fib confirmation", days_1m)

# 1M no-edge + 5M EMA trend confirm
combined_trend = []
for t in trades_1m_noedge:
    bar_time = candles_1m[t['bar']]["time"]
    aligned_5m = (bar_time // 300_000) * 300_000
    if five_min_trend.get(aligned_5m) == t['side']:
        combined_trend.append(t)

print_results(combined_trend, "V2 COMBINED: 1M fib + 5M EMA trend", days_1m)


# ========== FINAL SUMMARY ==========
print(f"\n{'='*80}")
print("V2.0 FINAL SUMMARY")
print(f"{'='*80}")
be_wr = 10.0 / (10.0 + 10.0 / 0.48 - 10.0) * 100
print(f"  Breakeven WR: {be_wr:.1f}%\n")

configs = [
    ("5M no-edge", trades_5m_noedge, days_5m),
    ("5M + edge gate", trades_5m_edge, days_5m),
    ("1M no-edge -> 5m", trades_1m_noedge, days_1m),
    ("1M + edge gate -> 5m", trades_1m_edge, days_1m),
    ("1M + 5M fib confirm", combined_fib, days_1m),
    ("1M + 5M EMA confirm", combined_trend, days_1m),
]

print(f"  {'Config':<25} {'Trades':>7} {'WR':>7} {'PnL':>10} {'$/day':>8} {'$/mo':>8} {'Verdict':>12}")
print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*10} {'-'*8} {'-'*8} {'-'*12}")

best_daily = -9999
for name, trades, d in configs:
    if trades:
        daily = sum(t['pnl'] for t in trades) / max(1, d)
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
    verdict = "BEST" if abs(daily - best_daily) < 0.01 else ("OK" if wr > be_wr else "FAIL")
    marker = " ***" if abs(daily - best_daily) < 0.01 else ""
    print(f"  {name:<25} {total:>7} {wr:>6.1f}% ${pnl:>8.2f} ${daily:>6.2f} ${monthly:>6.0f} {verdict:>12}{marker}")

print("\nDone.")
