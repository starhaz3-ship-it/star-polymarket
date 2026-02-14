"""Analyze shadow trades to find filters worth loosening."""
import json, sys, io
from datetime import datetime, timezone
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('C:/Users/Star/.local/bin/star-polymarket/ta_live_results.json') as f:
    data = json.load(f)

# shadow_trades is a dict keyed by condition_id+side+filter
shadow_dict = data.get('shadow_trades', {})
shadows = list(shadow_dict.values())
resolved = data.get('resolved', [])
now = datetime.now(timezone.utc)

# Only use shadows that have pnl
shadows_with_pnl = [s for s in shadows if s.get('pnl') is not None]
shadows_no_pnl = [s for s in shadows if s.get('pnl') is None]

print("=" * 90)
print(f"  SHADOW TRADE ANALYSIS  |  {len(shadows)} total ({len(shadows_with_pnl)} resolved, {len(shadows_no_pnl)} pending)")
print(f"  Live trades: {len(resolved)}")
print("=" * 90)

# --- BY FILTER ---
by_filter = defaultdict(lambda: {'w': 0, 'l': 0, 'pnl': 0.0})
for s in shadows_with_pnl:
    f = s.get('filter_name') or s.get('reason', 'unknown')
    # Normalize filter name
    if f.startswith('filter_'):
        f = f[7:]
    p = s.get('pnl', 0)
    by_filter[f]['pnl'] += p
    if p > 0:
        by_filter[f]['w'] += 1
    else:
        by_filter[f]['l'] += 1

print(f"\n  BY FILTER (sorted by shadow PnL):")
print(f"  {'Filter':<42} {'T':>4} {'W':>4} {'L':>4} {'WR':>5} {'PnL':>10} {'Avg':>8}")
print(f"  {'-' * 82}")
total_shadow_pnl = 0
for f in sorted(by_filter, key=lambda x: -by_filter[x]['pnl']):
    s = by_filter[f]
    total = s['w'] + s['l']
    wr = s['w'] / total * 100 if total else 0
    avg = s['pnl'] / total if total else 0
    total_shadow_pnl += s['pnl']
    flag = ' <<<' if s['pnl'] > 5 and wr >= 50 and total >= 3 else ''
    print(f"  {f:<42} {total:>4} {s['w']:>4} {s['l']:>4} {wr:>4.0f}% ${s['pnl']:>+9.2f} ${avg:>+7.2f}{flag}")
print(f"  {'-' * 82}")
print(f"  TOTAL SHADOW PnL: ${total_shadow_pnl:+.2f}")

# --- BY FILTER + ASSET + SIDE ---
by_detail = defaultdict(lambda: {'w': 0, 'l': 0, 'pnl': 0.0})
for s in shadows_with_pnl:
    f = s.get('filter_name') or s.get('reason', 'unknown')
    if f.startswith('filter_'):
        f = f[7:]
    asset = s.get('asset', '?')
    side = s.get('side', '?')
    key = f"{f} | {asset} {side}"
    p = s.get('pnl', 0)
    by_detail[key]['pnl'] += p
    if p > 0:
        by_detail[key]['w'] += 1
    else:
        by_detail[key]['l'] += 1

print(f"\n  BY FILTER + ASSET + SIDE (2+ trades, sorted by PnL):")
print(f"  {'Key':<58} {'T':>3} {'W':>3} {'L':>3} {'WR':>5} {'PnL':>9}")
print(f"  {'-' * 85}")
for k in sorted(by_detail, key=lambda x: -by_detail[x]['pnl']):
    s = by_detail[k]
    total = s['w'] + s['l']
    if total < 2:
        continue
    wr = s['w'] / total * 100 if total else 0
    print(f"  {k:<58} {total:>3} {s['w']:>3} {s['l']:>3} {wr:>4.0f}% ${s['pnl']:>+8.2f}")

# --- TIME-BASED (last 12h, 24h, all) ---
def get_exit_time(s):
    t = s.get('exit_time') or s.get('entry_time', '2026-01-01T00:00:00+00:00')
    return datetime.fromisoformat(t)

recent_24h = [s for s in shadows_with_pnl if (now - get_exit_time(s)).total_seconds() < 86400]
recent_12h = [s for s in shadows_with_pnl if (now - get_exit_time(s)).total_seconds() < 43200]
recent_6h = [s for s in shadows_with_pnl if (now - get_exit_time(s)).total_seconds() < 21600]

def summarize(trades, label):
    if not trades:
        print(f"\n  {label}: no trades")
        return
    print(f"\n  {label} ({len(trades)} trades):")
    by_f = defaultdict(lambda: {'w': 0, 'l': 0, 'pnl': 0})
    for s in trades:
        f = s.get('filter_name') or s.get('reason', '?')
        if f.startswith('filter_'):
            f = f[7:]
        p = s.get('pnl', 0)
        by_f[f]['pnl'] += p
        if p > 0:
            by_f[f]['w'] += 1
        else:
            by_f[f]['l'] += 1
    for f in sorted(by_f, key=lambda x: -by_f[x]['pnl']):
        s = by_f[f]
        t = s['w'] + s['l']
        wr = s['w'] / t * 100 if t else 0
        print(f"    {f:<42} {t:>3}T {s['w']:>2}W/{s['l']:>2}L {wr:>4.0f}% ${s['pnl']:>+8.2f}")
    total_pnl = sum(s['pnl'] for s in by_f.values())
    total_t = sum(s['w'] + s['l'] for s in by_f.values())
    total_w = sum(s['w'] for s in by_f.values())
    total_wr = total_w / total_t * 100 if total_t else 0
    print(f"    {'TOTAL':<42} {total_t:>3}T        {total_wr:>4.0f}% ${total_pnl:>+8.2f}")

summarize(shadows_with_pnl, "ALL TIME SHADOWS")
summarize(recent_24h, "LAST 24H SHADOWS")
summarize(recent_12h, "LAST 12H SHADOWS")
summarize(recent_6h, "LAST 6H SHADOWS")

# --- LIVE vs SHADOW comparison ---
live_pnl = sum(t.get('pnl', 0) for t in resolved)
live_wins = sum(1 for t in resolved if t.get('pnl', 0) > 0)
live_total = len(resolved)
live_wr = live_wins / live_total * 100 if live_total else 0

print(f"\n  {'=' * 85}")
print(f"  COMPARISON:")
print(f"    LIVE:     {live_total:>4}T | {live_wr:.0f}% WR | ${live_pnl:+.2f}")
print(f"    SHADOW:   {len(shadows_with_pnl):>4}T |        | ${total_shadow_pnl:+.2f}")
print(f"    COMBINED: {live_total + len(shadows_with_pnl):>4}T |        | ${live_pnl + total_shadow_pnl:+.2f}")
print(f"  {'=' * 85}")

# --- RECOMMENDATION ---
print(f"\n  RECOMMENDATIONS:")
print(f"  {'-' * 85}")
unblock = []
consider = []
keep = []
for f in sorted(by_filter, key=lambda x: -by_filter[x]['pnl']):
    s = by_filter[f]
    total = s['w'] + s['l']
    wr = s['w'] / total * 100 if total else 0
    if total >= 5 and wr >= 55 and s['pnl'] > 3:
        unblock.append((f, total, wr, s['pnl']))
    elif total >= 3 and wr >= 50 and s['pnl'] > 0:
        consider.append((f, total, wr, s['pnl']))
    elif total >= 5 and (wr < 40 or s['pnl'] < -5):
        keep.append((f, total, wr, s['pnl']))

if unblock:
    print(f"  UNBLOCK (55%+ WR, 5+ trades, $3+ PnL):")
    for f, t, wr, pnl in unblock:
        print(f"    >> {f} ({t}T, {wr:.0f}% WR, ${pnl:+.2f})")
else:
    print(f"  UNBLOCK: None yet meet criteria (55%+ WR, 5+ trades, $3+ PnL)")

if consider:
    print(f"  WATCH (50%+ WR, 3+ trades, positive PnL - need more data):")
    for f, t, wr, pnl in consider:
        print(f"    ~~ {f} ({t}T, {wr:.0f}% WR, ${pnl:+.2f})")

if keep:
    print(f"  KEEP BLOCKED (<40% WR or heavy losses):")
    for f, t, wr, pnl in keep:
        print(f"    XX {f} ({t}T, {wr:.0f}% WR, ${pnl:+.2f})")

print()
