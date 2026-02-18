#!/usr/bin/env python3
"""
optimize_signal_threshold.py
Finds the optimal Binance signal strength threshold for the 5-min BTC sniper bot.

Methodology:
  1. Load all resolved trades from sniper_5m_results.json
  2. For each trade, fetch Binance 1-min OHLCV candles covering the 5-min window
  3. Reconstruct "signal strength" = % change from window-open price to price at entry time
  4. Sweep thresholds 0.01%-0.25% to find the optimal filter
  5. Also sweep using Binance signal for DIRECTION (not just filter)
  6. Report all results and save to signal_threshold_analysis.json
"""

import json
import time
import math
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("C:/Users/Star/.local/bin/star-polymarket")
INPUT_FILE  = BASE_DIR / "sniper_5m_results.json"
OUTPUT_FILE = BASE_DIR / "signal_threshold_analysis.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ms(dt: datetime) -> int:
    """Convert aware datetime to milliseconds epoch."""
    return int(dt.timestamp() * 1000)


def parse_iso(s: str) -> datetime:
    """Parse ISO-8601 string (with or without microseconds / Z suffix)."""
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def fetch_binance_klines(start_ms: int, end_ms: int, symbol="BTCUSDT", interval="1m", limit=10):
    """
    Fetch 1-min klines from Binance REST API.
    Returns list of [open_time, open, high, low, close, ...] lists.
    Raises on HTTP/network error.
    """
    params = urllib.parse.urlencode({
        "symbol":    symbol,
        "interval":  interval,
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     limit,
    })
    url = f"https://api.binance.com/api/v3/klines?{params}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Step 1 – Load trades
# ---------------------------------------------------------------------------
print("=" * 70)
print("  Binance Signal Threshold Optimizer — Sniper 5M")
print("=" * 70)
print()
print(f"Loading trades from {INPUT_FILE} …")

with open(INPUT_FILE) as f:
    raw = json.load(f)

trades = raw.get("resolved", [])
print(f"  {len(trades)} resolved trades found.")

# ---------------------------------------------------------------------------
# Step 2 – Reconstruct Binance signal for each trade
# ---------------------------------------------------------------------------
print()
print("Fetching Binance 1-min candles for each trade window …")
print("(sleeping 0.1 s between requests to respect rate limits)")
print()

dataset = []          # list of dicts
skipped_fetch = 0
skipped_data  = 0

for i, t in enumerate(trades):
    # Parse times
    try:
        end_dt    = parse_iso(t["end_dt"])      # 5-min window close
        entry_dt  = parse_iso(t["entry_time"])  # actual entry timestamp
    except Exception as e:
        print(f"  [SKIP #{i}] bad timestamps: {e}")
        skipped_data += 1
        continue

    # The 5-min window open is exactly 5 minutes before end_dt
    window_open_dt = end_dt.replace(second=0, microsecond=0)
    window_open_dt = window_open_dt.replace(minute=((window_open_dt.minute // 5) * 5))
    # Simpler: subtract 300 s from end_dt (which is always at :XX:00)
    import datetime as _dt
    window_open_dt = end_dt - _dt.timedelta(seconds=300)

    start_ms_val = ms(window_open_dt)
    end_ms_val   = ms(entry_dt)          # candles up to entry time

    # Fetch candles
    try:
        klines = fetch_binance_klines(start_ms_val, end_ms_val, limit=10)
        time.sleep(0.1)
    except Exception as e:
        print(f"  [FETCH ERR #{i}] {t.get('question','?')[:50]}: {e}")
        skipped_fetch += 1
        continue

    if not klines:
        print(f"  [NO DATA #{i}] {t.get('question','?')[:50]}")
        skipped_data += 1
        continue

    # Window open price = open of the first candle
    window_open_price = float(klines[0][1])

    # Price at entry = close of the candle nearest to entry_time
    # The last candle in the list is the one closest to entry
    nearest_candle_close = float(klines[-1][4])

    # Signal strength = % change from window open to entry price
    if window_open_price == 0:
        skipped_data += 1
        continue

    signal_pct = (nearest_candle_close - window_open_price) / window_open_price * 100.0

    # Determine Binance signal direction
    if signal_pct > 0:
        binance_direction = "UP"
    elif signal_pct < 0:
        binance_direction = "DOWN"
    else:
        binance_direction = "FLAT"

    # Trade fields
    side         = t.get("side", "")
    entry_price  = t.get("entry_price", 0.5)
    shares       = t.get("shares", round(2.5 / max(entry_price, 0.01), 4))
    cost         = t.get("cost", shares * entry_price)
    pnl          = t.get("pnl", 0.0)
    result       = t.get("result", "")        # WIN / LOSS
    market_outcome = t.get("market_outcome", "")

    # Does Binance direction agree with trade side?
    binance_agrees_with_trade = (binance_direction == side)

    # Does Binance direction predict actual outcome?
    binance_correct = (binance_direction == market_outcome)

    record = {
        "idx":                       i,
        "question":                  t.get("question", ""),
        "side":                      side,
        "entry_price":               entry_price,
        "shares":                    shares,
        "cost":                      cost,
        "pnl":                       pnl,
        "result":                    result,
        "market_outcome":            market_outcome,
        "time_remaining_sec":        t.get("time_remaining_sec", 0),
        "signal_pct":                round(signal_pct, 4),
        "signal_abs":                round(abs(signal_pct), 4),
        "binance_direction":         binance_direction,
        "window_open_price":         window_open_price,
        "entry_candle_close":        nearest_candle_close,
        "binance_agrees_with_trade": binance_agrees_with_trade,
        "binance_correct":           binance_correct,
    }
    dataset.append(record)

    # Progress every 10 trades
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{len(trades)} trades …")

print()
print(f"Dataset built: {len(dataset)} trades with signal data")
print(f"  Skipped (fetch error): {skipped_fetch}")
print(f"  Skipped (bad data):    {skipped_data}")

if not dataset:
    print("ERROR: No data to analyze. Exiting.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Step 3 – Baseline stats
# ---------------------------------------------------------------------------
total_trades = len(dataset)
wins  = sum(1 for r in dataset if r["result"] == "WIN")
losses = total_trades - wins
total_pnl = sum(r["pnl"] for r in dataset)
wr = wins / total_trades * 100 if total_trades else 0
avg_pnl = total_pnl / total_trades if total_trades else 0

print()
print("=" * 70)
print("  BASELINE (no filter — all trades taken as-is)")
print("=" * 70)
print(f"  Trades:    {total_trades}")
print(f"  Wins:      {wins}  |  Losses: {losses}")
print(f"  Win Rate:  {wr:.1f}%")
print(f"  Total PnL: ${total_pnl:.2f}")
print(f"  PnL/Trade: ${avg_pnl:.4f}")

# Signal stats
flat_count = sum(1 for r in dataset if r["binance_direction"] == "FLAT")
up_count   = sum(1 for r in dataset if r["binance_direction"] == "UP")
dn_count   = sum(1 for r in dataset if r["binance_direction"] == "DOWN")
print()
print("  Binance signal distribution:")
print(f"    UP:   {up_count} ({up_count/total_trades*100:.1f}%)")
print(f"    DOWN: {dn_count} ({dn_count/total_trades*100:.1f}%)")
print(f"    FLAT: {flat_count} ({flat_count/total_trades*100:.1f}%)")

agree_count = sum(1 for r in dataset if r["binance_agrees_with_trade"])
correct_count = sum(1 for r in dataset if r["binance_correct"])
print()
print(f"  Binance agrees with trade side:   {agree_count}/{total_trades} ({agree_count/total_trades*100:.1f}%)")
print(f"  Binance direction predicts outcome: {correct_count}/{total_trades} ({correct_count/total_trades*100:.1f}%)")

# ---------------------------------------------------------------------------
# Step 4 – Threshold sweep (FILTER mode)
# ---------------------------------------------------------------------------
# Filter mode: only take a trade if abs(signal_pct) > T AND binance direction
# matches trade side. Trades not matching are "skipped".

thresholds = [round(t * 0.01, 4) for t in range(1, 26)]   # 0.01 to 0.25 (%)

filter_results = []

for T in thresholds:
    taken  = [r for r in dataset if r["signal_abs"] > T and r["binance_direction"] == r["side"]]
    skipped = [r for r in dataset if not (r["signal_abs"] > T and r["binance_direction"] == r["side"])]

    n_taken  = len(taken)
    n_skip   = len(skipped)

    if n_taken == 0:
        filter_results.append({
            "threshold": T,
            "n_taken":   0,
            "n_skipped": n_skip,
            "wins":      0,
            "losses":    0,
            "wr_pct":    0.0,
            "total_pnl": 0.0,
            "pnl_per_trade": 0.0,
        })
        continue

    w = sum(1 for r in taken if r["result"] == "WIN")
    l = n_taken - w
    pnl = sum(r["pnl"] for r in taken)
    wr_pct = w / n_taken * 100
    pnl_per = pnl / n_taken

    filter_results.append({
        "threshold":     T,
        "n_taken":       n_taken,
        "n_skipped":     n_skip,
        "wins":          w,
        "losses":        l,
        "wr_pct":        round(wr_pct, 2),
        "total_pnl":     round(pnl, 4),
        "pnl_per_trade": round(pnl_per, 4),
    })

# ---------------------------------------------------------------------------
# Step 5 – Threshold sweep (DIRECTION mode)
# ---------------------------------------------------------------------------
# Direction mode: if abs(signal_pct) > T, take the trade in Binance's direction
# (ignoring CLOB side). Win if Binance direction matches actual market_outcome.
# For PnL: we use the original cost/shares but recalculate win/loss based on
# whether Binance direction matched market_outcome.

direction_results = []

for T in thresholds:
    # Only trades where signal is strong enough
    signal_trades = [r for r in dataset if r["signal_abs"] > T and r["binance_direction"] != "FLAT"]
    n = len(signal_trades)

    if n == 0:
        direction_results.append({
            "threshold":     T,
            "n_taken":       0,
            "n_skipped":     total_trades - 0,
            "wins":          0,
            "losses":        0,
            "wr_pct":        0.0,
            "total_pnl":     0.0,
            "pnl_per_trade": 0.0,
        })
        continue

    wins_dir = 0
    pnl_dir  = 0.0

    for r in signal_trades:
        if r["binance_correct"]:   # Binance direction == market_outcome
            wins_dir += 1
            # Win: payout = shares * $1.00 - cost
            # But we don't know what price we'd have bought at if using Binance direction
            # (entry price reflects CLOB confidence, not our hypothetical side).
            # Use actual cost/shares as proxy — if binance direction wins, use actual pnl sign.
            # Approximate: use the SAME entry price (we'd still pay the CLOB price for that side).
            # If binance_direction == trade side AND won: same as actual win
            # If binance_direction != trade side but binance_correct: outcome is opposite
            # Simplest: win PnL = shares * 1.00 - cost; loss PnL = -cost
            pnl_dir += r["shares"] * 1.0 - r["cost"]
        else:
            pnl_dir += -r["cost"]

    losses_dir = n - wins_dir
    wr_pct = wins_dir / n * 100
    pnl_per = pnl_dir / n

    direction_results.append({
        "threshold":     T,
        "n_taken":       n,
        "n_skipped":     total_trades - n,
        "wins":          wins_dir,
        "losses":        losses_dir,
        "wr_pct":        round(wr_pct, 2),
        "total_pnl":     round(pnl_dir, 4),
        "pnl_per_trade": round(pnl_per, 4),
    })

# ---------------------------------------------------------------------------
# Step 6 – Print tables
# ---------------------------------------------------------------------------

def print_table(title, results, baseline_pnl, baseline_wr):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"  {'Thresh':>7}  {'Taken':>6}  {'Skip':>6}  {'Wins':>5}  {'WR%':>6}  {'PnL':>8}  {'PnL/Tr':>8}  {'vs Base':>9}")
    print("  " + "-" * 66)
    for r in results:
        vs = r["total_pnl"] - baseline_pnl
        marker = ""
        if r["n_taken"] == 0:
            row = f"  {r['threshold']:>7.2f}%  {r['n_taken']:>6}  {r['n_skipped']:>6}  {'—':>5}  {'—':>6}  {'—':>8}  {'—':>8}  {'—':>9}"
        else:
            row = (f"  {r['threshold']:>7.2f}%"
                   f"  {r['n_taken']:>6}"
                   f"  {r['n_skipped']:>6}"
                   f"  {r['wins']:>5}"
                   f"  {r['wr_pct']:>5.1f}%"
                   f"  ${r['total_pnl']:>7.2f}"
                   f"  ${r['pnl_per_trade']:>7.4f}"
                   f"  {vs:>+9.2f}")
            if r["total_pnl"] == max(x["total_pnl"] for x in results if x["n_taken"] > 0):
                marker = "  << BEST PnL"
        print(row + marker)
    print()

print_table(
    "FILTER MODE  (only take trades where Binance direction matches CLOB side)",
    filter_results,
    total_pnl,
    wr
)

print_table(
    "DIRECTION MODE  (use Binance direction as trade signal)",
    direction_results,
    total_pnl,
    wr
)

# ---------------------------------------------------------------------------
# Step 7 – Find optimal threshold
# ---------------------------------------------------------------------------

def best_result(results, key="total_pnl"):
    valid = [r for r in results if r["n_taken"] > 0]
    if not valid:
        return None
    return max(valid, key=lambda x: x[key])


filter_best_pnl  = best_result(filter_results, "total_pnl")
filter_best_wr   = best_result(filter_results, "wr_pct")
filter_best_eff  = best_result(filter_results, "pnl_per_trade")

dir_best_pnl     = best_result(direction_results, "total_pnl")
dir_best_wr      = best_result(direction_results, "wr_pct")
dir_best_eff     = best_result(direction_results, "pnl_per_trade")

CURRENT_THRESHOLD = 0.08   # live bot as of 2026-02-18

# Find current threshold row
def get_row(results, T):
    for r in results:
        if abs(r["threshold"] - T) < 0.001:
            return r
    return None

filter_current  = get_row(filter_results, CURRENT_THRESHOLD)
dir_current     = get_row(direction_results, CURRENT_THRESHOLD)

print()
print("=" * 70)
print("  OPTIMAL THRESHOLD RECOMMENDATIONS")
print("=" * 70)
print()
print(f"  Current live threshold: {CURRENT_THRESHOLD:.2f}%")
print()

print("  -- FILTER MODE (keep Binance direction aligned with CLOB side) --")
if filter_current and filter_current["n_taken"] > 0:
    print(f"  Current ({CURRENT_THRESHOLD:.2f}%): {filter_current['n_taken']} trades, "
          f"{filter_current['wr_pct']:.1f}% WR, ${filter_current['total_pnl']:.2f} PnL")
if filter_best_pnl:
    print(f"  Best PnL  -> threshold={filter_best_pnl['threshold']:.2f}%  "
          f"({filter_best_pnl['n_taken']} trades, {filter_best_pnl['wr_pct']:.1f}% WR, "
          f"${filter_best_pnl['total_pnl']:.2f} PnL)")
if filter_best_wr:
    print(f"  Best WR   -> threshold={filter_best_wr['threshold']:.2f}%  "
          f"({filter_best_wr['n_taken']} trades, {filter_best_wr['wr_pct']:.1f}% WR, "
          f"${filter_best_wr['total_pnl']:.2f} PnL)")
if filter_best_eff:
    print(f"  Best Eff  -> threshold={filter_best_eff['threshold']:.2f}%  "
          f"({filter_best_eff['n_taken']} trades, {filter_best_eff['wr_pct']:.1f}% WR, "
          f"${filter_best_eff['pnl_per_trade']:.4f}/trade)")

print()
print("  -- DIRECTION MODE (use Binance signal as trade direction) --")
if dir_current and dir_current["n_taken"] > 0:
    print(f"  Current ({CURRENT_THRESHOLD:.2f}%): {dir_current['n_taken']} trades, "
          f"{dir_current['wr_pct']:.1f}% WR, ${dir_current['total_pnl']:.2f} PnL")
if dir_best_pnl:
    print(f"  Best PnL  -> threshold={dir_best_pnl['threshold']:.2f}%  "
          f"({dir_best_pnl['n_taken']} trades, {dir_best_pnl['wr_pct']:.1f}% WR, "
          f"${dir_best_pnl['total_pnl']:.2f} PnL)")
if dir_best_wr:
    print(f"  Best WR   -> threshold={dir_best_wr['threshold']:.2f}%  "
          f"({dir_best_wr['n_taken']} trades, {dir_best_wr['wr_pct']:.1f}% WR, "
          f"${dir_best_wr['total_pnl']:.2f} PnL)")
if dir_best_eff:
    print(f"  Best Eff  -> threshold={dir_best_eff['threshold']:.2f}%  "
          f"({dir_best_eff['n_taken']} trades, {dir_best_eff['wr_pct']:.1f}% WR, "
          f"${dir_best_eff['pnl_per_trade']:.4f}/trade)")

# ---------------------------------------------------------------------------
# Step 8 – Print recommendation
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("  FINAL RECOMMENDATION")
print("=" * 70)

# Primary recommendation: filter mode best PnL if it beats baseline
if filter_best_pnl and filter_best_pnl["total_pnl"] > total_pnl:
    print(f"  Use FILTER MODE at threshold = {filter_best_pnl['threshold']:.2f}%")
    print(f"  Expected: {filter_best_pnl['n_taken']} trades, "
          f"{filter_best_pnl['wr_pct']:.1f}% WR, ${filter_best_pnl['total_pnl']:.2f} PnL "
          f"(+${filter_best_pnl['total_pnl'] - total_pnl:.2f} vs no filter)")
elif filter_best_pnl:
    print(f"  Filter mode DOES NOT improve on baseline (no Binance filter = better PnL).")
    print(f"  Consider setting threshold = 0.00% (take all CLOB-aligned trades)")
    if dir_best_pnl and dir_best_pnl["total_pnl"] > total_pnl:
        print(f"  ALTERNATIVE: Direction mode at {dir_best_pnl['threshold']:.2f}% "
              f"earns ${dir_best_pnl['total_pnl']:.2f} "
              f"({dir_best_pnl['wr_pct']:.1f}% WR, {dir_best_pnl['n_taken']} trades)")

print()
# Signal strength distribution
buckets = {}
for r in dataset:
    bucket = round(math.floor(r["signal_abs"] / 0.05) * 0.05, 2)
    buckets.setdefault(bucket, {"count": 0, "wins": 0, "pnl": 0.0})
    buckets[bucket]["count"] += 1
    if r["result"] == "WIN":
        buckets[bucket]["wins"] += 1
    buckets[bucket]["pnl"] += r["pnl"]

print("  Signal strength distribution (buckets of 0.05%):")
print(f"  {'Bucket':>10}  {'Count':>6}  {'WR%':>6}  {'PnL':>8}")
for b in sorted(buckets.keys()):
    bk = buckets[b]
    bwr = bk["wins"] / bk["count"] * 100 if bk["count"] else 0
    print(f"  {b:.2f}%-{b+0.05:.2f}%  {bk['count']:>6}  {bwr:>5.1f}%  ${bk['pnl']:>7.2f}")

# ---------------------------------------------------------------------------
# Step 9 – Save JSON
# ---------------------------------------------------------------------------
output = {
    "generated_at":      datetime.now(timezone.utc).isoformat(),
    "source_file":       str(INPUT_FILE),
    "total_trades_raw":  len(trades),
    "total_trades_with_signal": total_trades,
    "skipped_fetch":     skipped_fetch,
    "skipped_data":      skipped_data,
    "baseline": {
        "trades":    total_trades,
        "wins":      wins,
        "losses":    losses,
        "wr_pct":    round(wr, 2),
        "total_pnl": round(total_pnl, 4),
        "pnl_per_trade": round(avg_pnl, 4),
    },
    "current_live_threshold": CURRENT_THRESHOLD,
    "filter_mode_results":    filter_results,
    "direction_mode_results": direction_results,
    "optimal": {
        "filter_best_pnl":  filter_best_pnl,
        "filter_best_wr":   filter_best_wr,
        "filter_best_eff":  filter_best_eff,
        "dir_best_pnl":     dir_best_pnl,
        "dir_best_wr":      dir_best_wr,
        "dir_best_eff":     dir_best_eff,
    },
    "per_trade_dataset": dataset,
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print()
print(f"  Full results saved to: {OUTPUT_FILE}")
print()
print("=" * 70)
print("  Done.")
print("=" * 70)
