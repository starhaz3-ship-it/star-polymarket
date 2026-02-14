"""Maker Bot Dashboard - PnL breakdown by 5m/15m with chart."""
import json
from datetime import datetime, timezone

with open('C:/Users/Star/.local/bin/star-polymarket/maker_results.json') as f:
    data = json.load(f)

resolved = data.get('resolved', [])
active = data.get('active', {})
stats = data.get('stats', {})

# Separate by duration
m5_paired = [r for r in resolved if r.get('paired') and r.get('duration_min', 15) == 5]
m5_partial = [r for r in resolved if r.get('partial') and r.get('duration_min', 15) == 5]
m15_paired = [r for r in resolved if r.get('paired') and r.get('duration_min', 15) != 5]
m15_partial = [r for r in resolved if r.get('partial') and r.get('duration_min', 15) != 5]

def calc(paired, partial):
    p_pnl = sum(r['pnl'] for r in paired)
    pt_pnl = sum(r['pnl'] for r in partial)
    wins = sum(1 for r in paired if r['pnl'] > 0)
    wr = (wins / len(paired) * 100) if paired else 0
    avg = p_pnl / len(paired) if paired else 0
    return p_pnl, pt_pnl, p_pnl + pt_pnl, len(paired), len(partial), wr, avg

p15, pt15, net15, n15, npt15, wr15, avg15 = calc(m15_paired, m15_partial)
p5, pt5, net5, n5, npt5, wr5, avg5 = calc(m5_paired, m5_partial)
total_net = net15 + net5

# Runtime
start = datetime.fromisoformat(stats.get('start_time', datetime.now(timezone.utc).isoformat()))
now = datetime.now(timezone.utc)
hours = (now - start).total_seconds() / 3600

print()
print("=" * 62)
print(f"  MAKER BOT DASHBOARD  |  {now.strftime('%H:%M UTC %b %d')}  |  {hours:.0f}h runtime")
print("=" * 62)

# Chart - cumulative PnL over time
all_trades = sorted(resolved, key=lambda r: r.get('resolved_at', r.get('created_at', '')))
cum_pnl = 0
cum_5m = 0
cum_15m = 0
chart_points_all = []
chart_points_5m = []
chart_points_15m = []

for r in all_trades:
    pnl = r.get('pnl', 0)
    cum_pnl += pnl
    chart_points_all.append(cum_pnl)
    if r.get('duration_min', 15) == 5:
        cum_5m += pnl
    else:
        cum_15m += pnl
    chart_points_5m.append(cum_5m)
    chart_points_15m.append(cum_15m)

# ASCII chart
if chart_points_all:
    chart_w = 55
    chart_h = 12
    all_vals = chart_points_all + chart_points_5m + chart_points_15m
    y_min = min(min(all_vals), 0)
    y_max = max(all_vals)
    y_range = y_max - y_min if y_max != y_min else 1

    def resample(points, width):
        if len(points) <= width:
            return points
        step = len(points) / width
        return [points[int(i * step)] for i in range(width)]

    pts_all = resample(chart_points_all, chart_w)
    pts_5m = resample(chart_points_5m, chart_w)
    pts_15m = resample(chart_points_15m, chart_w)

    print()
    print("  CUMULATIVE PnL CHART")
    print(f"  {'=' * chart_w}")

    for row in range(chart_h, -1, -1):
        y_val = y_min + (row / chart_h) * y_range
        label = f"${y_val:>6.1f} |"
        line = list(" " * chart_w)

        for x in range(len(pts_all)):
            # Map each series to row
            def at_row(val):
                r_pos = int((val - y_min) / y_range * chart_h + 0.5)
                return r_pos == row

            if x < len(pts_15m) and at_row(pts_15m[x]):
                line[x] = "o"  # 15m
            if x < len(pts_5m) and at_row(pts_5m[x]):
                line[x] = "x"  # 5m
            if at_row(pts_all[x]):
                line[x] = "#"  # combined

        # Zero line
        zero_row = int((0 - y_min) / y_range * chart_h + 0.5)
        if row == zero_row:
            for i in range(chart_w):
                if line[i] == " ":
                    line[i] = "-"

        print(f"  {label}{''.join(line)}")

    print(f"  {'':>8}{'=' * chart_w}")
    print(f"  {'':>8}Trade #1{' ' * (chart_w - 16)}#{len(all_trades)}")
    print(f"  {'':>8}# = Combined  o = 15m  x = 5m")

# Stats table
print()
print("  " + "-" * 58)
print(f"  {'':>12} | {'15-MIN':>12} | {'5-MIN':>12} | {'COMBINED':>12}")
print("  " + "-" * 58)
print(f"  {'Paired':>12} | {n15:>12} | {n5:>12} | {n15+n5:>12}")
print(f"  {'Partial':>12} | {npt15:>12} | {npt5:>12} | {npt15+npt5:>12}")
print(f"  {'Win Rate':>12} | {wr15:>11.0f}% | {wr5:>11.0f}% | {((sum(1 for r in m15_paired+m5_paired if r['pnl']>0))/(n15+n5)*100 if n15+n5 else 0):>11.0f}%")
print(f"  {'Paired PnL':>12} | ${p15:>+10.2f} | ${p5:>+10.2f} | ${p15+p5:>+10.2f}")
print(f"  {'Partial PnL':>12} | ${pt15:>+10.2f} | ${pt5:>+10.2f} | ${pt15+pt5:>+10.2f}")
print(f"  {'NET PnL':>12} | ${net15:>+10.2f} | ${net5:>+10.2f} | ${total_net:>+10.2f}")
print(f"  {'Avg/Paired':>12} | ${avg15:>+10.2f} | ${avg5:>+10.2f} | ${(p15+p5)/(n15+n5) if n15+n5 else 0:>+10.2f}")
print("  " + "-" * 58)

# Per hour rate
if hours > 0:
    print(f"  {'$/hour':>12} | ${net15/hours:>+10.2f} | ${net5/hours:>+10.2f} | ${total_net/hours:>+10.2f}")
    print(f"  {'$/day (proj)':>12} | ${net15/hours*24:>+10.2f} | ${net5/hours*24:>+10.2f} | ${total_net/hours*24:>+10.2f}")
    print("  " + "-" * 58)

# By asset
print()
print("  BY ASSET:")
by_asset = {}
for r in resolved:
    a = r.get('asset', '?')
    if a not in by_asset:
        by_asset[a] = {'paired': 0, 'partial': 0, 'pnl': 0}
    by_asset[a]['pnl'] += r.get('pnl', 0)
    if r.get('paired'):
        by_asset[a]['paired'] += 1
    elif r.get('partial'):
        by_asset[a]['partial'] += 1

for a, s in sorted(by_asset.items(), key=lambda x: -x[1]['pnl']):
    total_trades = s['paired'] + s['partial']
    rate = s['paired'] / total_trades * 100 if total_trades > 0 else 0
    bar_len = max(0, int(s['pnl'] / 2))
    bar = "+" * min(bar_len, 30)
    if s['pnl'] < 0:
        bar_len = max(0, int(abs(s['pnl']) / 2))
        bar = "-" * min(bar_len, 30)
    print(f"    {a:>4}: {s['paired']:>3}P/{s['partial']:>2}pt ({rate:>3.0f}%) ${s['pnl']:>+7.2f} |{bar}")

# Active positions
print()
print(f"  ACTIVE: {len(active)} positions")
for cid, pos in active.items():
    dur = pos.get('duration_min', 15)
    up = "FILLED" if pos.get('up_filled') else "open"
    dn = "FILLED" if pos.get('down_filled') else "open"
    paired = "PAIRED" if pos.get('paired') else ""
    print(f"    {pos.get('asset', '?')} {dur}m | UP:{up} DN:{dn} {paired}")

# Last 5 resolved
print()
print("  LAST 5 TRADES:")
for r in resolved[-5:]:
    dur = r.get('duration_min', 15)
    typ = "PAIR" if r.get('paired') else "PART"
    pnl = r.get('pnl', 0)
    print(f"    {r.get('asset', '?'):>4} {dur:>2}m {typ} ${pnl:>+6.2f} | {r.get('question', '?')[:40]}")

print()
print("=" * 62)
