import csv
from datetime import datetime, timezone
from collections import defaultdict

rows = []
with open('C:/Users/Star/Downloads/Polymarket-History-2026-02-15 (5).csv', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# We reset PnL at ~22:44 UTC. Look at everything after 22:40 UTC
cutoff = datetime(2026, 2, 15, 22, 40, tzinfo=timezone.utc)
recent = []
for r in rows:
    ts = int(r.get('timestamp', 0) or 0)
    if ts > cutoff.timestamp():
        r['_dt'] = datetime.fromtimestamp(ts, tz=timezone.utc)
        recent.append(r)

recent.sort(key=lambda x: x['_dt'])

print(f"Trades since reset (22:40 UTC): {len(recent)}")
print()

total_buys = 0
total_sells = 0
total_redeems = 0

for r in recent:
    action = r['action']
    usdc = float(r.get('usdcAmount', 0) or 0)
    tokens = float(r.get('tokenAmount', 0) or 0)
    token = r.get('tokenName', '')[:6]
    dt = r['_dt'].strftime('%H:%M:%S')
    market = r.get('marketName', '')[:60]

    if action == 'Buy': total_buys += usdc
    elif action == 'Sell': total_sells += usdc
    elif action == 'Redeem': total_redeems += usdc

    print(f"{dt} {action:8s} ${usdc:7.2f} {tokens:7.1f}tk {token:6s} | {market}")

print()
print(f"Buys:    ${total_buys:.2f}")
print(f"Sells:   ${total_sells:.2f}")
print(f"Redeems: ${total_redeems:.2f}")
print(f"Net flow: ${total_sells + total_redeems - total_buys:.2f}")
print()

# Group by market
markets = defaultdict(lambda: {'buys': 0, 'sells': 0, 'redeems': 0, 'name': '', 'trades': []})
for r in recent:
    mid = r.get('conditionId', '') or r.get('marketName', '')
    mname = r.get('marketName', '')[:60]
    action = r['action']
    usdc = float(r.get('usdcAmount', 0) or 0)
    token = r.get('tokenName', '')[:6]
    markets[mid]['name'] = mname
    markets[mid]['trades'].append((r['_dt'].strftime('%H:%M'), action, usdc, token))
    if action == 'Buy': markets[mid]['buys'] += usdc
    elif action == 'Sell': markets[mid]['sells'] += usdc
    elif action == 'Redeem': markets[mid]['redeems'] += usdc

print("=== PER-MARKET BREAKDOWN ===")
for mid, m in sorted(markets.items(), key=lambda x: x[1]['name']):
    net = m['sells'] + m['redeems'] - m['buys']
    status = 'RESOLVED' if m['redeems'] > 0 else ('SOLD' if m['sells'] > 0 else 'HOLDING')
    print(f"\n{m['name']}")
    print(f"  Bought: ${m['buys']:.2f} | Sold: ${m['sells']:.2f} | Redeemed: ${m['redeems']:.2f} | Net: ${net:+.2f} [{status}]")
    for t in m['trades']:
        print(f"    {t[0]} {t[1]:8s} ${t[2]:6.2f} {t[3]}")

# Summary stats
print("\n=== SUMMARY ===")
wins = 0
losses = 0
for mid, m in markets.items():
    net = m['sells'] + m['redeems'] - m['buys']
    if m['redeems'] > 0 or m['sells'] > 0:  # resolved
        if net > 0: wins += 1
        else: losses += 1
print(f"Resolved: {wins}W / {losses}L ({wins/(wins+losses)*100:.0f}% WR)" if wins+losses > 0 else "No resolved yet")
print(f"Still holding: {sum(1 for m in markets.values() if m['redeems']==0 and m['sells']==0)}")
