"""Maker Bot Earnings Projections - $5 and $10 sizing with charts."""
import json, sys, io
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('C:/Users/Star/.local/bin/star-polymarket/maker_results.json') as f:
    data = json.load(f)

stats = data.get('stats', {})
resolved = data.get('resolved', [])
paired = [r for r in resolved if r.get('paired')]
partial = [r for r in resolved if r.get('partial')]

start = datetime.fromisoformat(stats['start_time'])
now = datetime.now(timezone.utc)
hours = (now - start).total_seconds() / 3600

CURRENT_SIZE = 5.0

paired_pnl = sum(r['pnl'] for r in paired)
partial_pnl = sum(r['pnl'] for r in partial)
net_pnl = paired_pnl + partial_pnl
wins = sum(1 for r in paired if r['pnl'] > 0)
wr = wins / len(paired) * 100 if paired else 0

hourly_base = net_pnl / hours

sizes = [
    (5, "CURRENT"),
    (10, "MONDAY SCALE-UP"),
]

print()
print("=" * 72)
print(f"  MAKER BOT EARNINGS PROJECTIONS")
print(f"  Based on {hours:.0f}h live data | {len(paired)} paired ({wr:.0f}% WR) | {len(partial)} partial")
print(f"  Current: ${CURRENT_SIZE}/side | Net: ${net_pnl:+.2f}")
print("=" * 72)

for size, label in sizes:
    scale = size / CURRENT_SIZE
    hourly = hourly_base * scale
    daily = hourly * 24
    weekly = daily * 7
    monthly = daily * 30

    capital = size * 2 * 8 * 1.5
    daily_roi = daily / capital * 100 if capital > 0 else 0

    worst_partial = -size
    avg_paired = (paired_pnl / len(paired) * scale) if paired else 0

    print()
    print(f"  {'=' * 68}")
    print(f"  ${size}/SIDE ({label})")
    print(f"  {'=' * 68}")
    print(f"  Capital needed: ~${capital:.0f}")
    print()

    # HOURLY
    print(f"  HOURLY: ${hourly:+.2f}/hr")
    bar_h = max(1, int(hourly * 10))
    print(f"  |{'#' * min(bar_h, 60)}")

    # DAILY - 7-day accumulation
    print()
    print(f"  DAILY:  ${daily:+.2f}/day  (ROI: {daily_roi:.1f}%/day)")
    print(f"  {'-' * 55}")
    for d in range(1, 8):
        cum = daily * d
        bar_len = min(int(cum / (daily * 7) * 45), 45)
        bar = "#" * bar_len
        print(f"   Day {d} |{bar} ${cum:>8.2f}")
    print(f"  {'-' * 55}")

    # MONTHLY - 6-month accumulation
    print()
    print(f"  MONTHLY: ${monthly:+,.2f}/month")
    print(f"  {'-' * 60}")
    for m in range(1, 7):
        cum = monthly * m
        bar_len = min(int(cum / (monthly * 6) * 40), 40)
        bar = "#" * bar_len
        print(f"   Month {m} |{bar} ${cum:>10,.2f}")
    print(f"  {'-' * 60}")

    # Summary box
    print()
    print(f"  +-------------------------------------------+")
    print(f"  |  HOURLY:    ${hourly:>8.2f}                  |")
    print(f"  |  DAILY:     ${daily:>8.2f}                  |")
    print(f"  |  WEEKLY:    ${weekly:>8.2f}                  |")
    print(f"  |  MONTHLY:   ${monthly:>8,.2f}                |")
    print(f"  |  YEARLY:    ${monthly*12:>10,.2f}              |")
    print(f"  |                                           |")
    print(f"  |  Avg paired:  ${avg_paired:>+6.2f}/trade            |")
    print(f"  |  Worst partial: ${worst_partial:>+6.2f}                |")
    print(f"  |  Trades/day:  ~{len(paired)/hours*24:.0f} paired              |")
    print(f"  +-------------------------------------------+")

# Side by side comparison
h5 = hourly_base
h10 = hourly_base * 2

print()
print("=" * 72)
print("  SIDE-BY-SIDE COMPARISON: $5 vs $10")
print("=" * 72)
print(f"  {'':>15} | {'$5/side':>12} | {'$10/side':>12} | {'Diff':>10}")
print(f"  {'-' * 56}")
print(f"  {'Hourly':>15} | ${h5:>10.2f} | ${h10:>10.2f} | ${h10-h5:>+8.2f}")
print(f"  {'Daily':>15} | ${h5*24:>10.2f} | ${h10*24:>10.2f} | ${(h10-h5)*24:>+8.2f}")
print(f"  {'Weekly':>15} | ${h5*168:>10.2f} | ${h10*168:>10.2f} | ${(h10-h5)*168:>+8.2f}")
print(f"  {'Monthly':>15} | ${h5*720:>10,.2f} | ${h10*720:>10,.2f} | ${(h10-h5)*720:>+8,.2f}")
print(f"  {'Yearly':>15} | ${h5*8760:>10,.2f} | ${h10*8760:>10,.2f} | ${(h10-h5)*8760:>+8,.2f}")
print(f"  {'-' * 56}")
print(f"  {'Capital':>15} | ${120:>10.0f} | ${240:>10.0f} |")
print(f"  {'Daily ROI':>15} | {h5*24/120*100:>10.1f}% | {h10*24/240*100:>10.1f}% |")

# Visual comparison bar chart
print()
print("  EARNINGS COMPARISON (daily)")
print(f"  $5/side  |{'#' * min(int(h5*24), 50)} ${h5*24:.2f}")
print(f"  $10/side |{'#' * min(int(h10*24), 50)} ${h10*24:.2f}")

print()
print("  EARNINGS COMPARISON (monthly)")
print(f"  $5/side  |{'#' * min(int(h5*720/50), 50)} ${h5*720:,.2f}")
print(f"  $10/side |{'#' * min(int(h10*720/50), 50)} ${h10*720:,.2f}")

print()
print("=" * 72)
print("  CAVEATS:")
print("  1. Assumes same fill rates (larger orders harder to fill)")
print("  2. Partial losses scale: $5->-$5, $10->-$10 per bad fill")
print("  3. 5M markets not yet pairing (projections are 15M only)")
print("  4. Past performance does not guarantee future results")
print("  5. Monday review (Feb 17): scale to $10 if still profitable")
print("=" * 72)
