#!/usr/bin/env python3
"""
Hourly Edge Analysis: BTC Up/Down Markets
Combines paper trading data + Binance ground truth to find optimal trading hours.

Part 1: Paper bot data from 4 sources
Part 2: Binance 15M candle ground truth (14 days)
Part 3: Combined recommendation table with TRADE/SKIP/WATCH per hour
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_entry_time(raw):
    """Parse entry_time from ISO string or float timestamp. Returns UTC datetime or None."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(raw, tz=timezone.utc)
        except (OSError, ValueError, OverflowError):
            return None
    if isinstance(raw, str):
        # Try ISO parse
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
        ]:
            try:
                dt = datetime.strptime(raw, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
        # Maybe it's a numeric string
        try:
            ts = float(raw)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OSError):
            pass
    return None


def load_json(path):
    """Load JSON file, return dict or None."""
    if not os.path.exists(path):
        print(f"  [WARN] File not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] Failed to load {path}: {e}")
        return None


def extract_trades(data, source_name):
    """Extract list of (utc_hour, pnl, is_win, source) from various data formats."""
    trades = []
    if data is None:
        return trades

    resolved = []
    if isinstance(data, dict):
        if "resolved" in data:
            resolved = data["resolved"]
        elif "trades" in data:
            resolved = data["trades"]
    elif isinstance(data, list):
        resolved = data

    for t in resolved:
        if not isinstance(t, dict):
            continue
        # Skip non-closed trades
        status = t.get("status", "closed")
        if status not in ("closed", "resolved", "settled"):
            continue

        entry_time = t.get("entry_time")
        dt = parse_entry_time(entry_time)
        if dt is None:
            continue

        # Get PnL
        pnl = t.get("pnl")
        if pnl is None:
            continue
        try:
            pnl = float(pnl)
        except (ValueError, TypeError):
            continue

        # Determine win
        result = t.get("result", "")
        if result == "WIN":
            is_win = True
        elif result == "LOSS":
            is_win = False
        else:
            is_win = pnl > 0

        utc_hour = dt.hour
        trades.append((utc_hour, pnl, is_win, source_name))

    return trades


# ---------------------------------------------------------------------------
# Part 1: Paper / Live Bot Data
# ---------------------------------------------------------------------------

BASE = r"C:\Users\Star\.local\bin\star-polymarket"

print("=" * 80)
print("  HOURLY EDGE ANALYSIS: BTC Up/Down Markets")
print("=" * 80)
print()

sources = [
    ("fifteen_min_strategies_results.json", "15M Strategies (paper)"),
    ("sniper_5m_results.json", "Sniper 5M (paper)"),
    ("sniper_5m_live_results.json", "Sniper 5M (live)"),
    ("momentum_15m_results.json", "Momentum 15M (live)"),
]

all_trades = []
for fname, label in sources:
    path = os.path.join(BASE, fname)
    data = load_json(path)
    trades = extract_trades(data, label)
    print(f"  Loaded {len(trades):>4d} trades from {label}")
    all_trades.extend(trades)

print(f"\n  TOTAL: {len(all_trades)} trades loaded from all sources")
print()

# Group by hour
hourly = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
for utc_hour, pnl, is_win, _ in all_trades:
    hourly[utc_hour]["trades"] += 1
    hourly[utc_hour]["pnl"] += pnl
    if is_win:
        hourly[utc_hour]["wins"] += 1

# Also group by source x hour for breakdown
source_hourly = defaultdict(lambda: defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0}))
for utc_hour, pnl, is_win, source in all_trades:
    source_hourly[source][utc_hour]["trades"] += 1
    source_hourly[source][utc_hour]["pnl"] += pnl
    if is_win:
        source_hourly[source][utc_hour]["wins"] += 1

CURRENT_SKIP = {7, 11, 12, 13, 14, 15}

print("-" * 80)
print("  PART 1: PAPER + LIVE BOT DATA  --  Hourly Win Rates (UTC)")
print("-" * 80)
print()
print(f"  {'Hour':>4s}  {'UTC':>5s}  {'MST':>7s}  {'Trades':>6s}  {'Wins':>4s}  {'WR%':>6s}  {'PnL':>8s}  {'Avg PnL':>8s}  {'Skip?':>5s}")
print(f"  {'----':>4s}  {'-----':>5s}  {'-------':>7s}  {'------':>6s}  {'----':>4s}  {'------':>6s}  {'--------':>8s}  {'--------':>8s}  {'-----':>5s}")

for h in range(24):
    d = hourly.get(h, {"trades": 0, "wins": 0, "pnl": 0.0})
    wr = (d["wins"] / d["trades"] * 100) if d["trades"] > 0 else 0
    avg_pnl = d["pnl"] / d["trades"] if d["trades"] > 0 else 0
    # MST = UTC - 7
    mst_h = (h - 7) % 24
    mst_str = f"{mst_h:02d}:00"
    skip_marker = "SKIP" if h in CURRENT_SKIP else ""

    # Color indicator
    if d["trades"] >= 3:
        if wr >= 65:
            indicator = "++"
        elif wr >= 55:
            indicator = " +"
        elif wr < 45:
            indicator = "--"
        else:
            indicator = "  "
    else:
        indicator = "  "

    print(f"  {h:4d}  {h:02d}:00  {mst_str:>7s}  {d['trades']:6d}  {d['wins']:4d}  {wr:5.1f}%  ${d['pnl']:+7.2f}  ${avg_pnl:+7.2f}  {skip_marker:>5s} {indicator}")

# Print per-source breakdown
print()
print("-" * 80)
print("  PART 1b: Per-Source Hourly Breakdown")
print("-" * 80)
for source in sorted(source_hourly.keys()):
    sh = source_hourly[source]
    total_t = sum(v["trades"] for v in sh.values())
    total_w = sum(v["wins"] for v in sh.values())
    total_p = sum(v["pnl"] for v in sh.values())
    total_wr = (total_w / total_t * 100) if total_t > 0 else 0
    print(f"\n  {source} -- {total_t} trades, {total_wr:.1f}% WR, ${total_p:+.2f}")
    print(f"    {'Hour':>4s}  {'Trades':>6s}  {'Wins':>4s}  {'WR%':>6s}  {'PnL':>8s}")
    for h in range(24):
        d = sh.get(h)
        if d is None or d["trades"] == 0:
            continue
        wr = d["wins"] / d["trades"] * 100
        print(f"    {h:4d}  {d['trades']:6d}  {d['wins']:4d}  {wr:5.1f}%  ${d['pnl']:+7.2f}")


# ---------------------------------------------------------------------------
# Part 2: Binance Ground Truth
# ---------------------------------------------------------------------------

print()
print()
print("-" * 80)
print("  PART 2: BINANCE GROUND TRUTH  --  BTC 15M Candles (14 days)")
print("-" * 80)
print()

try:
    import requests
    r = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": "BTCUSDT", "interval": "15m", "limit": 1000},
        timeout=15,
    )
    r.raise_for_status()
    klines = r.json()
    print(f"  Fetched {len(klines)} candles from Binance")
except Exception as e:
    print(f"  [ERROR] Binance fetch failed: {e}")
    print("  Attempting Binance.US fallback...")
    try:
        r = requests.get(
            "https://api.binance.us/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "15m", "limit": 1000},
            timeout=15,
        )
        r.raise_for_status()
        klines = r.json()
        print(f"  Fetched {len(klines)} candles from Binance.US")
    except Exception as e2:
        print(f"  [ERROR] Binance.US also failed: {e2}")
        klines = []

binance_hourly = defaultdict(lambda: {"up": 0, "down": 0, "total": 0})

for k in klines:
    open_ts_ms = k[0]
    open_price = float(k[1])
    close_price = float(k[4])
    dt = datetime.fromtimestamp(open_ts_ms / 1000, tz=timezone.utc)
    utc_hour = dt.hour
    outcome = "up" if close_price >= open_price else "down"
    binance_hourly[utc_hour][outcome] += 1
    binance_hourly[utc_hour]["total"] += 1

if klines:
    first_dt = datetime.fromtimestamp(klines[0][0] / 1000, tz=timezone.utc)
    last_dt = datetime.fromtimestamp(klines[-1][0] / 1000, tz=timezone.utc)
    print(f"  Range: {first_dt.strftime('%Y-%m-%d %H:%M')} to {last_dt.strftime('%Y-%m-%d %H:%M')} UTC")
    print()

print(f"  {'Hour':>4s}  {'UTC':>5s}  {'Candles':>7s}  {'UP':>4s}  {'DOWN':>4s}  {'UP%':>6s}  {'DOWN%':>6s}  {'Majority':>8s}  {'Maj WR':>6s}")
print(f"  {'----':>4s}  {'-----':>5s}  {'-------':>7s}  {'----':>4s}  {'----':>4s}  {'------':>6s}  {'------':>6s}  {'--------':>8s}  {'------':>6s}")

binance_up_pct = {}
binance_majority_wr = {}

for h in range(24):
    d = binance_hourly.get(h, {"up": 0, "down": 0, "total": 0})
    if d["total"] > 0:
        up_pct = d["up"] / d["total"] * 100
        down_pct = d["down"] / d["total"] * 100
        majority = "UP" if d["up"] >= d["down"] else "DOWN"
        maj_wr = max(up_pct, down_pct)
    else:
        up_pct = down_pct = 0
        majority = "N/A"
        maj_wr = 0

    binance_up_pct[h] = up_pct
    binance_majority_wr[h] = maj_wr

    bias = ""
    if maj_wr >= 58:
        bias = " <<<"
    elif maj_wr <= 52:
        bias = " (coin flip)"

    print(f"  {h:4d}  {h:02d}:00  {d['total']:7d}  {d['up']:4d}  {d['down']:4d}  {up_pct:5.1f}%  {down_pct:5.1f}%  {majority:>8s}  {maj_wr:5.1f}%{bias}")

# Overall binance stats
total_up = sum(d["up"] for d in binance_hourly.values())
total_down = sum(d["down"] for d in binance_hourly.values())
total_candles = total_up + total_down
if total_candles > 0:
    print(f"\n  Overall: {total_candles} candles, UP {total_up} ({total_up/total_candles*100:.1f}%), DOWN {total_down} ({total_down/total_candles*100:.1f}%)")

    # Best-hour strategy backtest
    correct = 0
    for h in range(24):
        d = binance_hourly.get(h, {"up": 0, "down": 0, "total": 0})
        correct += max(d["up"], d["down"])
    print(f"  'Always bet majority' strategy: {correct}/{total_candles} = {correct/total_candles*100:.1f}% WR")


# ---------------------------------------------------------------------------
# Part 3: Combined Recommendation
# ---------------------------------------------------------------------------

print()
print()
print("=" * 80)
print("  PART 3: COMBINED RECOMMENDATION")
print("=" * 80)
print()

print(f"  {'Hour':>4s}  {'UTC':>5s}  {'MST':>7s}  {'PaperWR':>8s}  {'PaperN':>7s}  {'PaperPnL':>9s}  {'BinUP%':>7s}  {'BinBias':>8s}  {'Action':>8s}  {'Reason'}")
print(f"  {'----':>4s}  {'-----':>5s}  {'-------':>7s}  {'--------':>8s}  {'-------':>7s}  {'---------':>9s}  {'-------':>7s}  {'--------':>8s}  {'--------':>8s}  {'------'}")

recommendations = {}
for h in range(24):
    d = hourly.get(h, {"trades": 0, "wins": 0, "pnl": 0.0})
    paper_wr = (d["wins"] / d["trades"] * 100) if d["trades"] > 0 else None
    paper_n = d["trades"]
    paper_pnl = d["pnl"]

    b_up = binance_up_pct.get(h, 50)
    b_maj_wr = binance_majority_wr.get(h, 50)
    b_dir = "UP" if b_up >= 50 else "DOWN"

    mst_h = (h - 7) % 24
    mst_str = f"{mst_h:02d}:00"

    # Recommendation logic
    reasons = []
    action = "WATCH"

    # Strong paper evidence
    if paper_n >= 3 and paper_wr is not None:
        if paper_wr > 55:
            reasons.append(f"paper {paper_wr:.0f}%WR({paper_n}t)")
            action = "TRADE"
        elif paper_wr < 45:
            reasons.append(f"paper LOSING {paper_wr:.0f}%WR({paper_n}t)")
            action = "SKIP"

    # Binance bias
    if b_maj_wr >= 58:
        reasons.append(f"binance {b_dir} bias {b_maj_wr:.0f}%")
        if action != "SKIP":
            action = "TRADE"
    elif 48 <= b_maj_wr <= 52:
        reasons.append(f"binance coin flip {b_maj_wr:.0f}%")
        if action == "WATCH":
            action = "SKIP"

    # No paper data + no binance edge
    if paper_n == 0 and b_maj_wr < 55:
        reasons.append("no data")
        action = "WATCH"
    elif paper_n == 0 and b_maj_wr >= 55:
        reasons.append(f"binance only: {b_dir} {b_maj_wr:.0f}%")

    if not reasons:
        reasons.append("low confidence")

    recommendations[h] = action

    paper_wr_str = f"{paper_wr:5.1f}%" if paper_wr is not None else "  N/A "

    # Mark current skip hours
    current = " [CURRENTLY SKIPPED]" if h in CURRENT_SKIP else ""

    print(f"  {h:4d}  {h:02d}:00  {mst_str:>7s}  {paper_wr_str:>8s}  {paper_n:7d}  ${paper_pnl:+8.2f}  {b_up:5.1f}%  {b_dir:>8s}  {action:>8s}  {'; '.join(reasons)}{current}")

# ---------------------------------------------------------------------------
# Part 4: Skip-Hour Recommendations
# ---------------------------------------------------------------------------

print()
print()
print("=" * 80)
print("  SKIP-HOUR RECOMMENDATIONS")
print("=" * 80)
print()
print(f"  Currently skipping: {sorted(CURRENT_SKIP)} UTC")
print()

# Hours that should be added to skip
add_skip = []
for h in range(24):
    if h not in CURRENT_SKIP and recommendations[h] == "SKIP":
        d = hourly.get(h, {"trades": 0, "wins": 0, "pnl": 0.0})
        paper_wr = (d["wins"] / d["trades"] * 100) if d["trades"] > 0 else None
        add_skip.append((h, paper_wr, d["trades"], d["pnl"]))

# Hours that should be removed from skip (re-opened)
remove_skip = []
for h in CURRENT_SKIP:
    if recommendations[h] == "TRADE":
        d = hourly.get(h, {"trades": 0, "wins": 0, "pnl": 0.0})
        paper_wr = (d["wins"] / d["trades"] * 100) if d["trades"] > 0 else None
        remove_skip.append((h, paper_wr, d["trades"], d["pnl"]))

# Hours in skip that should stay
keep_skip = []
for h in CURRENT_SKIP:
    if recommendations[h] != "TRADE":
        d = hourly.get(h, {"trades": 0, "wins": 0, "pnl": 0.0})
        paper_wr = (d["wins"] / d["trades"] * 100) if d["trades"] > 0 else None
        keep_skip.append((h, paper_wr, d["trades"], d["pnl"], recommendations[h]))

if add_skip:
    print("  ADD to skip list:")
    for h, wr, n, pnl in add_skip:
        mst = (h - 7) % 24
        wr_str = f"{wr:.0f}%" if wr is not None else "N/A"
        print(f"    Hour {h:02d} UTC ({mst:02d}:00 MST) -- {n} trades, {wr_str} WR, ${pnl:+.2f}")
else:
    print("  ADD to skip list: (none)")

print()
if remove_skip:
    print("  REMOVE from skip list (re-open for trading):")
    for h, wr, n, pnl in remove_skip:
        mst = (h - 7) % 24
        wr_str = f"{wr:.0f}%" if wr is not None else "N/A"
        print(f"    Hour {h:02d} UTC ({mst:02d}:00 MST) -- {n} trades, {wr_str} WR, ${pnl:+.2f}")
else:
    print("  REMOVE from skip list: (none)")

print()
print("  KEEP skipped (confirmed bad):")
for h, wr, n, pnl, act in keep_skip:
    mst = (h - 7) % 24
    wr_str = f"{wr:.0f}%" if wr is not None else "N/A"
    print(f"    Hour {h:02d} UTC ({mst:02d}:00 MST) -- {n} trades, {wr_str} WR, ${pnl:+.2f} [{act}]")

# Proposed new skip set
new_skip = set()
for h in CURRENT_SKIP:
    if recommendations[h] != "TRADE":
        new_skip.add(h)
for h in range(24):
    if h not in CURRENT_SKIP and recommendations[h] == "SKIP":
        new_skip.add(h)

print()
print(f"  PROPOSED skip set: {sorted(new_skip)} UTC")
print(f"  Current skip set:  {sorted(CURRENT_SKIP)} UTC")
diff_add = new_skip - CURRENT_SKIP
diff_rm = CURRENT_SKIP - new_skip
if diff_add:
    print(f"  + Adding:  {sorted(diff_add)}")
if diff_rm:
    print(f"  - Removing: {sorted(diff_rm)}")
if not diff_add and not diff_rm:
    print(f"  No changes needed.")

# Summary
print()
print("=" * 80)
print("  SUMMARY")
print("=" * 80)
trade_hours = [h for h in range(24) if recommendations[h] == "TRADE"]
skip_hours = [h for h in range(24) if recommendations[h] == "SKIP"]
watch_hours = [h for h in range(24) if recommendations[h] == "WATCH"]
print(f"  TRADE hours: {trade_hours} ({len(trade_hours)} hours)")
print(f"  SKIP  hours: {skip_hours} ({len(skip_hours)} hours)")
print(f"  WATCH hours: {watch_hours} ({len(watch_hours)} hours)")
print()

# Paper PnL if we only traded TRADE hours
trade_pnl = sum(hourly.get(h, {"pnl": 0})["pnl"] for h in trade_hours)
skip_pnl = sum(hourly.get(h, {"pnl": 0})["pnl"] for h in skip_hours)
watch_pnl = sum(hourly.get(h, {"pnl": 0})["pnl"] for h in watch_hours)
total_pnl = sum(hourly.get(h, {"pnl": 0})["pnl"] for h in range(24))

print(f"  Paper PnL in TRADE hours: ${trade_pnl:+.2f}")
print(f"  Paper PnL in SKIP  hours: ${skip_pnl:+.2f}")
print(f"  Paper PnL in WATCH hours: ${watch_pnl:+.2f}")
print(f"  Total paper PnL:          ${total_pnl:+.2f}")
print(f"  PnL saved by skipping:    ${-skip_pnl:+.2f}")
print()
print("=" * 80)
