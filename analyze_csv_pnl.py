"""Analyze Polymarket CSV export for actual PnL."""
import csv, sys, io
from datetime import datetime, timezone
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CSV_PATH = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-13 (3).csv"

rows = []
with open(CSV_PATH, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f"Total rows: {len(rows)}")
print(f"Columns: {list(rows[0].keys())}")

# Count by action type
actions = defaultdict(int)
for r in rows:
    actions[r['action']] += 1
print(f"\nBy action: {dict(actions)}")

# Parse all rows with amounts
buys = []
sells = []
redeems = []

for r in rows:
    amt = float(r['usdcAmount']) if r['usdcAmount'] else 0
    tokens = float(r['tokenAmount']) if r['tokenAmount'] else 0
    ts = int(r['timestamp']) if r['timestamp'] else 0
    market = r['marketName']
    action = r['action']
    token = r['tokenName']

    entry = {
        'market': market,
        'action': action,
        'usdc': amt,
        'tokens': tokens,
        'token': token,
        'timestamp': ts,
        'dt': datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None,
        'hash': r['hash']
    }

    if action == 'Buy':
        buys.append(entry)
    elif action == 'Sell':
        sells.append(entry)
    elif action == 'Redeem':
        redeems.append(entry)

print(f"\nBuys: {len(buys)} (${sum(b['usdc'] for b in buys):,.2f} spent)")
print(f"Sells: {len(sells)} (${sum(s['usdc'] for s in sells):,.2f} received)")
print(f"Redeems: {len(redeems)} (${sum(r['usdc'] for r in redeems):,.2f} received)")

total_spent = sum(b['usdc'] for b in buys)
total_received = sum(s['usdc'] for s in sells) + sum(r['usdc'] for r in redeems)
net_pnl = total_received - total_spent

print(f"\n{'='*70}")
print(f"  POLYMARKET ACTUAL PnL (from CSV export)")
print(f"{'='*70}")
print(f"  Total spent (buys):     ${total_spent:>10,.2f}")
print(f"  Total received (sells): ${sum(s['usdc'] for s in sells):>10,.2f}")
print(f"  Total received (redeem):${sum(r['usdc'] for r in redeems):>10,.2f}")
print(f"  Total received:         ${total_received:>10,.2f}")
print(f"  ---")
print(f"  NET PnL:                ${net_pnl:>+10,.2f}")
print(f"{'='*70}")

# Time range
all_entries = buys + sells + redeems
all_entries.sort(key=lambda x: x['timestamp'])
if all_entries:
    first = all_entries[0]['dt']
    last = all_entries[-1]['dt']
    hours = (last - first).total_seconds() / 3600
    print(f"\n  Period: {first.strftime('%b %d %H:%M')} -> {last.strftime('%b %d %H:%M')} UTC")
    print(f"  Duration: {hours:.1f} hours ({hours/24:.1f} days)")
    if hours > 0:
        print(f"  $/hour: ${net_pnl/hours:+.2f}")
        print(f"  $/day:  ${net_pnl/hours*24:+.2f}")

# Categorize by market type
def classify_market(name):
    name_lower = name.lower()
    # 5m vs 15m up/down
    if 'up or down' in name_lower:
        # Check for 5-min pattern (times 5 min apart)
        if 'bitcoin' in name_lower or 'btc' in name_lower:
            asset = 'BTC'
        elif 'ethereum' in name_lower or 'eth' in name_lower:
            asset = 'ETH'
        elif 'solana' in name_lower or 'sol' in name_lower:
            asset = 'SOL'
        elif 'xrp' in name_lower:
            asset = 'XRP'
        else:
            asset = 'OTHER'
        return f"{asset} UpDown"
    elif 'temperature' in name_lower or 'weather' in name_lower:
        return 'Weather'
    elif 'super bowl' in name_lower:
        return 'SuperBowl'
    else:
        return 'Other'

by_category = defaultdict(lambda: {'spent': 0, 'received': 0, 'buys': 0, 'sells': 0, 'redeems': 0})
for b in buys:
    cat = classify_market(b['market'])
    by_category[cat]['spent'] += b['usdc']
    by_category[cat]['buys'] += 1
for s in sells:
    cat = classify_market(s['market'])
    by_category[cat]['received'] += s['usdc']
    by_category[cat]['sells'] += 1
for r in redeems:
    cat = classify_market(r['market'])
    by_category[cat]['received'] += r['usdc']
    by_category[cat]['redeems'] += 1

print(f"\n  BY CATEGORY:")
print(f"  {'Category':<20} {'Buys':>5} {'Sells':>5} {'Redeem':>6} {'Spent':>10} {'Recv':>10} {'PnL':>10}")
print(f"  {'-'*72}")
for cat in sorted(by_category, key=lambda x: -(by_category[x]['received'] - by_category[x]['spent'])):
    c = by_category[cat]
    pnl = c['received'] - c['spent']
    print(f"  {cat:<20} {c['buys']:>5} {c['sells']:>5} {c['redeems']:>6} ${c['spent']:>9,.2f} ${c['received']:>9,.2f} ${pnl:>+9,.2f}")

# By asset for UpDown markets specifically
print(f"\n  BY ASSET (Up/Down markets only):")
by_asset = defaultdict(lambda: {'spent': 0, 'received': 0, 'buy_count': 0})
for b in buys:
    cat = classify_market(b['market'])
    if 'UpDown' in cat:
        asset = cat.split()[0]
        by_asset[asset]['spent'] += b['usdc']
        by_asset[asset]['buy_count'] += 1
for s in sells:
    cat = classify_market(s['market'])
    if 'UpDown' in cat:
        asset = cat.split()[0]
        by_asset[asset]['received'] += s['usdc']
for r in redeems:
    cat = classify_market(r['market'])
    if 'UpDown' in cat:
        asset = cat.split()[0]
        by_asset[asset]['received'] += r['usdc']

print(f"  {'Asset':<8} {'Trades':>6} {'Spent':>10} {'Recv':>10} {'PnL':>10}")
print(f"  {'-'*50}")
for a in sorted(by_asset, key=lambda x: -(by_asset[x]['received'] - by_asset[x]['spent'])):
    s = by_asset[a]
    pnl = s['received'] - s['spent']
    print(f"  {a:<8} {s['buy_count']:>6} ${s['spent']:>9,.2f} ${s['received']:>9,.2f} ${pnl:>+9,.2f}")

# Daily breakdown
by_day = defaultdict(lambda: {'spent': 0, 'received': 0})
for b in buys:
    if b['dt']:
        day = b['dt'].strftime('%b %d')
        by_day[day]['spent'] += b['usdc']
for s in sells:
    if s['dt']:
        day = s['dt'].strftime('%b %d')
        by_day[day]['received'] += s['usdc']
for r in redeems:
    if r['dt']:
        day = r['dt'].strftime('%b %d')
        by_day[day]['received'] += r['usdc']

print(f"\n  DAILY BREAKDOWN:")
print(f"  {'Day':<10} {'Spent':>10} {'Recv':>10} {'PnL':>10}")
print(f"  {'-'*45}")
for day in sorted(by_day.keys()):
    d = by_day[day]
    pnl = d['received'] - d['spent']
    bar = '+' * max(0, int(pnl / 2)) if pnl > 0 else '-' * max(0, int(-pnl / 2))
    print(f"  {day:<10} ${d['spent']:>9,.2f} ${d['received']:>9,.2f} ${pnl:>+9,.2f} {bar}")

print()
