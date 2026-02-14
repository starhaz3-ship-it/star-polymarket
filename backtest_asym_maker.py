"""
Asymmetric Maker Backtest

For each historical 5M BTC candle:
1. Compute directional signals (what would the TA predict?)
2. Determine actual outcome (UP or DOWN based on close vs open)
3. Compare symmetric ($10/$10) vs asymmetric (tilted) maker PnL
"""
import numpy as np
import httpx
import time
from functools import partial as fn_partial

print = fn_partial(print, flush=True)

# Fetch 7 days of 5M BTC candles from Binance
print("Fetching 7 days of 5M BTC candles...")
all_candles = []
end_ms = int(time.time() * 1000)
for batch in range(8):
    start_ms = end_ms - 1000 * 5 * 60 * 1000
    r = httpx.get("https://api.binance.com/api/v3/klines",
                  params={"symbol": "BTCUSDT", "interval": "5m",
                          "startTime": start_ms, "endTime": end_ms, "limit": 1000},
                  timeout=15)
    bars = r.json()
    for b in bars:
        all_candles.append({
            "time": int(b[0]), "open": float(b[1]), "high": float(b[2]),
            "low": float(b[3]), "close": float(b[4]), "volume": float(b[5])
        })
    end_ms = start_ms
    time.sleep(0.3)

all_candles.sort(key=lambda x: x["time"])
seen = set()
candles = []
for c in all_candles:
    if c["time"] not in seen:
        seen.add(c["time"])
        candles.append(c)

days = len(candles) * 5 / 60 / 24
print(f"Got {len(candles)} candles ({days:.1f} days)")


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

def calc_rsi(close, n=14):
    out = np.full(len(close), np.nan, dtype=float)
    delta = np.diff(close, prepend=close[0])
    up = np.clip(delta, 0, None)
    dn = np.clip(-delta, 0, None)
    au = calc_sma(up, n)
    ad = calc_sma(dn, n)
    rs = au / np.where(ad == 0, np.nan, ad)
    out[:] = 100.0 - (100.0 / (1.0 + rs))
    return out


# === DIRECTIONAL SIGNAL ===
def predict_direction(close, high, low, vol):
    if len(close) < 30:
        return None, 0

    e9 = calc_ema(close, 9)
    e21 = calc_ema(close, 21)
    r = calc_rsi(close, 14)

    i = len(close) - 1
    if not all(np.isfinite([e9[i], e21[i], r[i]])):
        return None, 0

    ema_gap = (e9[i] - e21[i]) / max(1e-9, close[i])
    mom = (close[i] - close[i - 3]) / max(1e-9, close[i])
    rsi_bias = (r[i] - 50) / 100.0

    score = ema_gap * 500 + mom * 200 + rsi_bias * 0.3

    if abs(score) < 0.05:
        return None, 0

    conf = min(1.0, abs(score) / 0.5)
    return ("UP" if score > 0 else "DOWN"), conf


# === BUILD SIGNALS ===
print("\nComputing signals...")
close_arr = np.array([c["close"] for c in candles], dtype=float)
high_arr = np.array([c["high"] for c in candles], dtype=float)
low_arr = np.array([c["low"] for c in candles], dtype=float)
vol_arr = np.array([c["volume"] for c in candles], dtype=float)

outcomes = ["UP" if c["close"] > c["open"] else "DOWN" for c in candles]

LOOKBACK = 60
signals = []
for i in range(LOOKBACK, len(candles)):
    direction, conf = predict_direction(
        close_arr[i - LOOKBACK:i], high_arr[i - LOOKBACK:i],
        low_arr[i - LOOKBACK:i], vol_arr[i - LOOKBACK:i])
    signals.append((i, direction, conf, outcomes[i]))

total_signals = sum(1 for _, d, c, _ in signals if d is not None)
correct = sum(1 for _, d, c, a in signals if d is not None and d == a)
accuracy = correct / total_signals * 100 if total_signals > 0 else 0

up_pred = sum(1 for _, d, _, _ in signals if d == "UP")
up_correct = sum(1 for _, d, _, a in signals if d == "UP" and a == "UP")
dn_pred = sum(1 for _, d, _, _ in signals if d == "DOWN")
dn_correct = sum(1 for _, d, _, a in signals if d == "DOWN" and a == "DOWN")

print(f"\nSignal accuracy: {correct}/{total_signals} = {accuracy:.1f}%")
print(f"  UP predicted: {up_correct}/{up_pred} = {up_correct/max(1,up_pred)*100:.1f}%")
print(f"  DOWN predicted: {dn_correct}/{dn_pred} = {dn_correct/max(1,dn_pred)*100:.1f}%")
print(f"  No signal: {len(signals) - total_signals}/{len(signals)}")

# Actual UP/DOWN distribution
actual_up = sum(1 for o in outcomes[LOOKBACK:] if o == "UP")
actual_dn = len(outcomes) - LOOKBACK - actual_up
print(f"\nActual distribution: UP={actual_up} ({actual_up/(actual_up+actual_dn)*100:.1f}%) DOWN={actual_dn} ({actual_dn/(actual_up+actual_dn)*100:.1f}%)")


# === SIMULATE MAKER PNL ===
UP_BID = 0.48
DN_BID = 0.49

tilt_configs = [
    ("Symmetric $10/$10", 10, 10),
    ("Light $12/$8", 12, 8),
    ("Medium $14/$6", 14, 6),
    ("Heavy $16/$4", 16, 4),
    ("Max $18/$2", 18, 2),
]

conf_thresholds = [0.0, 0.2, 0.4, 0.6]

print(f"\n{'=' * 95}")
print(f"BACKTEST RESULTS  |  {len(signals)} markets  |  {days:.0f} days  |  Signal accuracy: {accuracy:.1f}%")
print(f"{'=' * 95}")

results = []

for conf_thresh in conf_thresholds:
    filtered = [(i, d, c, a) for i, d, c, a in signals if d is not None and c >= conf_thresh]
    no_signal = [(i, d, c, a) for i, d, c, a in signals if d is None or c < conf_thresh]

    # Signal accuracy at this threshold
    ct = len(filtered)
    cc = sum(1 for _, d, _, a in filtered if d == a)
    ca = cc / ct * 100 if ct > 0 else 0

    print(f"\n--- Conf >= {conf_thresh:.1f} | {ct} tilted trades ({ca:.1f}% acc) | {len(no_signal)} symmetric ---")
    print(f"  {'Config':<22} {'Trades':>6} {'Total PnL':>10} {'Avg/T':>8} {'WR':>6} {'MaxDD':>8} {'$/day':>8} {'$/mo':>8}")

    for name, tilt_size, hedge_size in tilt_configs:
        total_deployed = tilt_size + hedge_size
        pnl_list = []

        for idx, direction, conf, actual in filtered:
            if direction == "UP":
                up_s, dn_s = tilt_size, hedge_size
            else:
                up_s, dn_s = hedge_size, tilt_size

            if actual == "UP":
                pnl = (up_s / UP_BID) - total_deployed
            else:
                pnl = (dn_s / DN_BID) - total_deployed
            pnl_list.append(pnl)

        for idx, direction, conf, actual in no_signal:
            if actual == "UP":
                pnl = (10 / UP_BID) - total_deployed
            else:
                pnl = (10 / DN_BID) - total_deployed
            pnl_list.append(pnl)

        total_pnl = sum(pnl_list)
        wins = sum(1 for p in pnl_list if p > 0)
        total = len(pnl_list)
        wr = wins / total * 100 if total > 0 else 0
        avg = total_pnl / total if total > 0 else 0

        cumsum = np.cumsum(pnl_list)
        peak = np.maximum.accumulate(cumsum)
        dd = peak - cumsum
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0

        daily = total_pnl / days
        monthly = daily * 30

        results.append((conf_thresh, name, total_pnl, daily, monthly, max_dd, wr, ca))

        marker = " <-- BEST" if daily == max(r[3] for r in results) else ""
        print(f"  {name:<22} {total:>6} ${total_pnl:>8.2f} ${avg:>6.3f} {wr:>5.1f}% ${max_dd:>6.2f} ${daily:>6.2f} ${monthly:>6.0f}{marker}")


# === FIND OPTIMAL ===
print(f"\n{'=' * 95}")
print("TOP 5 CONFIGURATIONS")
print(f"{'=' * 95}")

results.sort(key=lambda r: r[3], reverse=True)
for i, (ct, name, tot, daily, monthly, mdd, wr, acc) in enumerate(results[:5]):
    print(f"  {i+1}. {name} @ conf>={ct:.1f} | ${daily:.2f}/day (${monthly:.0f}/mo) | "
          f"WR={wr:.1f}% | MaxDD=${mdd:.2f} | SigAcc={acc:.1f}%")


# === RISK ANALYSIS ===
print(f"\n{'=' * 95}")
print("RISK ANALYSIS: Best config stress test")
print(f"{'=' * 95}")

# Use best config
best = results[0]
best_ct, best_name = best[0], best[1]
# Find tilt values
tilt_map = {name: (ts, hs) for name, ts, hs in tilt_configs}
bt, bh = tilt_map.get(best_name, (10, 10))

filtered = [(i, d, c, a) for i, d, c, a in signals if d is not None and c >= best_ct]
no_signal = [(i, d, c, a) for i, d, c, a in signals if d is None or c < best_ct]

all_pnl = []
for idx, direction, conf, actual in filtered:
    if direction == "UP":
        up_s, dn_s = bt, bh
    else:
        up_s, dn_s = bh, bt
    if actual == "UP":
        all_pnl.append((up_s / UP_BID) - (bt + bh))
    else:
        all_pnl.append((dn_s / DN_BID) - (bt + bh))

for idx, direction, conf, actual in no_signal:
    if actual == "UP":
        all_pnl.append((10 / UP_BID) - 20)
    else:
        all_pnl.append((10 / DN_BID) - 20)

pnl_arr = np.array(all_pnl)
print(f"\n  Avg PnL/trade: ${pnl_arr.mean():.4f}")
print(f"  Median PnL/trade: ${np.median(pnl_arr):.4f}")
print(f"  Std dev: ${pnl_arr.std():.4f}")
print(f"  Best trade: ${pnl_arr.max():.2f}")
print(f"  Worst trade: ${pnl_arr.min():.2f}")
print(f"  Sharpe (daily): {(pnl_arr.mean() / max(1e-9, pnl_arr.std())) * np.sqrt(288):.2f}")

# Consecutive losses
streaks = []
streak = 0
for p in pnl_arr:
    if p < 0:
        streak += 1
    else:
        if streak > 0:
            streaks.append(streak)
        streak = 0
if streak > 0:
    streaks.append(streak)
print(f"  Max consecutive losses: {max(streaks) if streaks else 0}")
print(f"  Avg loss streak: {np.mean(streaks):.1f}" if streaks else "")

# What % of days are profitable?
daily_pnl = []
trades_per_day = len(pnl_arr) / days
chunk = int(trades_per_day)
for d in range(0, len(pnl_arr) - chunk, chunk):
    daily_pnl.append(sum(pnl_arr[d:d + chunk]))
profitable_days = sum(1 for d in daily_pnl if d > 0)
total_days = len(daily_pnl)
print(f"  Profitable days: {profitable_days}/{total_days} ({profitable_days/max(1,total_days)*100:.0f}%)")
print(f"  Best day: ${max(daily_pnl):.2f}" if daily_pnl else "")
print(f"  Worst day: ${min(daily_pnl):.2f}" if daily_pnl else "")

print("\nDone.")
