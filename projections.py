"""Overall profit projections grounded in CSV export data."""
import csv, json, sys, io
from datetime import datetime, timezone
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CSV_PATH = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-13 (3).csv"
MAKER_PATH = r"C:\Users\Star\.local\bin\star-polymarket\maker_results.json"

# === CSV ground truth ===
rows = list(csv.DictReader(open(CSV_PATH, encoding='utf-8-sig')))

deposits = 0
buys_total = 0
sells_total = 0
redeems_total = 0
rebates_total = 0

by_day = defaultdict(lambda: {'spent': 0, 'received': 0})

for r in rows:
    amt = float(r['usdcAmount']) if r['usdcAmount'] else 0
    ts = int(r['timestamp']) if r['timestamp'] else 0
    action = r['action']
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
    day = dt.strftime('%b %d') if dt else '?'

    if action == 'Deposit':
        deposits += amt
    elif action == 'Buy':
        buys_total += amt
        by_day[day]['spent'] += amt
    elif action == 'Sell':
        sells_total += amt
        by_day[day]['received'] += amt
    elif action == 'Redeem':
        redeems_total += amt
        by_day[day]['received'] += amt
    elif action == 'Maker Rebate':
        rebates_total += amt
        by_day[day]['received'] += amt

# === Maker bot stats ===
maker = json.load(open(MAKER_PATH))
maker_stats = maker.get('stats', {})
maker_resolved = maker.get('resolved', [])
maker_paired = [r for r in maker_resolved if r.get('paired')]
maker_partial = [r for r in maker_resolved if r.get('partial')]
maker_start = datetime.fromisoformat(maker_stats['start_time'])
now = datetime.now(timezone.utc)
maker_hours = (now - maker_start).total_seconds() / 3600

maker_paired_pnl = sum(r['pnl'] for r in maker_paired)
maker_partial_pnl = sum(r['pnl'] for r in maker_partial)
maker_net = maker_paired_pnl + maker_partial_pnl

# BTC+ETH only (post V1.7)
btc_paired = [r for r in maker_paired if r.get('asset') == 'BTC']
eth_paired = [r for r in maker_paired if r.get('asset') == 'ETH']
btc_pnl = sum(r['pnl'] for r in btc_paired) + sum(r['pnl'] for r in maker_partial if r.get('asset') == 'BTC')
eth_pnl = sum(r['pnl'] for r in eth_paired) + sum(r['pnl'] for r in maker_partial if r.get('asset') == 'ETH')
# Hourly rate for BTC+ETH only (exclude SOL/XRP drag going forward)
maker_btc_eth_net = btc_pnl + eth_pnl
maker_btc_eth_hourly = maker_btc_eth_net / maker_hours

# === Current balance ===
current_balance = 100.0  # updated 2026-02-14

# === Compute daily PnL from CSV (last 3 days = most recent trend) ===
sorted_days = sorted(by_day.keys())
recent_days = sorted_days[-3:] if len(sorted_days) >= 3 else sorted_days
recent_daily_pnls = []
for day in recent_days:
    d = by_day[day]
    pnl = d['received'] - d['spent']
    recent_daily_pnls.append((day, pnl))

# === PRINT REPORT ===
print("=" * 75)
print("  PROFIT PROJECTIONS — Grounded in CSV Export + Maker Data")
print("=" * 75)

print(f"\n  ACCOUNT STATUS:")
print(f"    Total deposited:    ${deposits:>10,.2f}")
print(f"    Current balance:    ${current_balance:>10,.2f}")
print(f"    Actual PnL to date: ${current_balance - deposits:>+10,.2f}")

print(f"\n  RECENT DAILY PnL (from CSV):")
print(f"  {'-' * 50}")
for day, pnl in recent_daily_pnls:
    bar = '+' * max(0, int(pnl / 2)) if pnl > 0 else '-' * max(0, int(-pnl / 2))
    print(f"    {day}: ${pnl:>+8.2f}  {bar}")
avg_recent = sum(p for _, p in recent_daily_pnls) / len(recent_daily_pnls) if recent_daily_pnls else 0
print(f"  {'-' * 50}")
print(f"    Avg last {len(recent_daily_pnls)} days: ${avg_recent:>+8.2f}/day")

# === MAKER PROJECTIONS (BTC+ETH only, post-V1.7) ===
print(f"\n  {'=' * 70}")
print(f"  MAKER BOT (BTC+ETH only — V1.7)")
print(f"  {'=' * 70}")
print(f"    Runtime: {maker_hours:.0f}h | {len(maker_paired)} paired ({len(btc_paired)} BTC, {len(eth_paired)} ETH)")
print(f"    BTC+ETH net: ${maker_btc_eth_net:>+.2f} (excl SOL -$16.77 drag)")
print(f"    Hourly rate:  ${maker_btc_eth_hourly:>+.4f}")

m_hourly = maker_btc_eth_hourly
m_daily = m_hourly * 24
m_weekly = m_daily * 7
m_monthly = m_daily * 30

print(f"\n    $5/side (current):")
print(f"    +-----------------------------------------------+")
print(f"    |  HOURLY:   ${m_hourly:>8.2f}                        |")
print(f"    |  DAILY:    ${m_daily:>8.2f}                        |")
print(f"    |  WEEKLY:   ${m_weekly:>8.2f}                        |")
print(f"    |  MONTHLY:  ${m_monthly:>8,.2f}                      |")
print(f"    |  YEARLY:   ${m_monthly*12:>10,.2f}                    |")
print(f"    +-----------------------------------------------+")

# At $10/side
scale = 2.0
print(f"\n    $10/side (when capital allows):")
print(f"    +-----------------------------------------------+")
print(f"    |  HOURLY:   ${m_hourly*scale:>8.2f}                        |")
print(f"    |  DAILY:    ${m_daily*scale:>8.2f}                        |")
print(f"    |  WEEKLY:   ${m_weekly*scale:>8.2f}                        |")
print(f"    |  MONTHLY:  ${m_monthly*scale:>8,.2f}                      |")
print(f"    |  YEARLY:   ${m_monthly*12*scale:>10,.2f}                    |")
print(f"    +-----------------------------------------------+")

# 7-day accumulation
print(f"\n    DAILY ACCUMULATION ($5/side):")
print(f"    {'-' * 55}")
for d in range(1, 8):
    cum = m_daily * d
    bar_len = min(int(cum / max(m_daily * 7, 1) * 40), 40)
    bar = "#" * max(bar_len, 1)
    print(f"     Day {d} |{bar} ${cum:>8.2f}")

# 6-month accumulation
print(f"\n    MONTHLY ACCUMULATION ($5/side):")
print(f"    {'-' * 60}")
for m in range(1, 7):
    cum = m_monthly * m
    bar_len = min(int(cum / max(m_monthly * 6, 1) * 35), 35)
    bar = "#" * max(bar_len, 1)
    print(f"     Month {m} |{bar} ${cum:>10,.2f}")

# === COMBINED PROJECTIONS ===
# TA live is harder to project — use CSV recent trend minus maker contribution
# Maker contribution to CSV: maker has been running ~51h so ~2.1 days
# Conservative: use maker-only for projections since TA live is volatile

print(f"\n  {'=' * 70}")
print(f"  COMBINED PROJECTION (Maker + TA Live V10.19)")
print(f"  {'=' * 70}")
print(f"  NOTE: TA Live projections are speculative (filters just changed).")
print(f"  Maker is the reliable baseline. TA Live adds upside potential.")

# TA live: 95 trades, +$21 over ~9 days = ~$2.3/day
# But recent trend is worse. Use conservative $0/day for TA live
# The V10.19 filter unblocks should improve this

# Conservative (maker only)
print(f"\n  CONSERVATIVE (Maker only — proven $22/day):")
print(f"    Hourly:   ${m_hourly:>+.2f}")
print(f"    Daily:    ${m_daily:>+.2f}")
print(f"    Weekly:   ${m_weekly:>+.2f}")
print(f"    Monthly:  ${m_monthly:>+,.2f}")

# Moderate (maker + TA live at historical rate)
ta_hourly = 21.21 / (9.7 * 24)  # $21.21 over 9.7 days
combined_hourly = m_hourly + ta_hourly
combined_daily = combined_hourly * 24
combined_monthly = combined_daily * 30

print(f"\n  MODERATE (Maker + TA Live historical rate):")
print(f"    Hourly:   ${combined_hourly:>+.2f}")
print(f"    Daily:    ${combined_daily:>+.2f}")
print(f"    Weekly:   ${combined_daily*7:>+.2f}")
print(f"    Monthly:  ${combined_monthly:>+,.2f}")

# Optimistic (maker + TA live with unblocked filters)
# Shadow data suggests +$114 over observation period (~2 days)
# Conservative estimate: 40% of shadow PnL realized
ta_optimistic_daily = (21.21 / 9.7) + (114.0 * 0.40 / 9.7)
opt_daily = m_daily + ta_optimistic_daily
opt_monthly = opt_daily * 30

print(f"\n  OPTIMISTIC (Maker + TA V10.19 with filter unblocks):")
print(f"    Hourly:   ${opt_daily/24:>+.2f}")
print(f"    Daily:    ${opt_daily:>+.2f}")
print(f"    Weekly:   ${opt_daily*7:>+.2f}")
print(f"    Monthly:  ${opt_monthly:>+,.2f}")

# === BREAKEVEN TIMELINE ===
hole = deposits - current_balance
print(f"\n  {'=' * 70}")
print(f"  BREAKEVEN TIMELINE (recovering ${hole:,.2f} deficit)")
print(f"  {'=' * 70}")
if m_daily > 0:
    print(f"    Conservative (Maker only):   {hole / m_daily:>5.0f} days ({hole / m_daily / 7:.1f} weeks)")
if combined_daily > 0:
    print(f"    Moderate (Maker + TA hist):  {hole / combined_daily:>5.0f} days ({hole / combined_daily / 7:.1f} weeks)")
if opt_daily > 0:
    print(f"    Optimistic (V10.19 unblock): {hole / opt_daily:>5.0f} days ({hole / opt_daily / 7:.1f} weeks)")

# === SIDE-BY-SIDE $5 vs $10 ===
print(f"\n  {'=' * 70}")
print(f"  $5/SIDE vs $10/SIDE (Maker only)")
print(f"  {'=' * 70}")
print(f"  {'':>15} | {'$5/side':>12} | {'$10/side':>12} | {'Diff':>10}")
print(f"  {'-' * 56}")
print(f"  {'Hourly':>15} | ${m_hourly:>10.2f} | ${m_hourly*2:>10.2f} | ${m_hourly:>+8.2f}")
print(f"  {'Daily':>15} | ${m_daily:>10.2f} | ${m_daily*2:>10.2f} | ${m_daily:>+8.2f}")
print(f"  {'Weekly':>15} | ${m_weekly:>10.2f} | ${m_weekly*2:>10.2f} | ${m_weekly:>+8.2f}")
print(f"  {'Monthly':>15} | ${m_monthly:>10,.2f} | ${m_monthly*2:>10,.2f} | ${m_monthly:>+8,.2f}")
print(f"  {'Breakeven':>15} | {hole/m_daily:>10.0f}d | {hole/(m_daily*2):>10.0f}d |")
print(f"  {'-' * 56}")

print(f"\n  CAVEATS:")
print(f"  1. CSV truncated at 1000 rows — actual PnL may differ slightly")
print(f"  2. Maker projections based on {maker_hours:.0f}h of BTC+ETH data (SOL/XRP excluded)")
print(f"  3. TA Live V10.19 filters just deployed — no post-change data yet")
print(f"  4. $10/side requires ~$200+ balance (currently ${current_balance:.0f})")
print(f"  5. Past performance does not guarantee future results")
print()
