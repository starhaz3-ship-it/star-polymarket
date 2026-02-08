import csv
from datetime import datetime

trades = []
with open(r'C:\Users\Star\Downloads\Polymarket-History-2026-02-08 (3).csv', 'r', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        trades.append(row)

print(f'Total CSV rows: {len(trades)}')

buys = [t for t in trades if t['action'] == 'Buy']
sells = [t for t in trades if t['action'] == 'Sell']
redeems = [t for t in trades if t['action'] == 'Redeem']
deposits = [t for t in trades if t['action'] == 'Deposit']

buy_total = sum(float(t['usdcAmount']) for t in buys)
sell_total = sum(float(t['usdcAmount']) for t in sells)
redeem_total = sum(float(t['usdcAmount']) for t in redeems)
dep_total = sum(float(t['usdcAmount']) for t in deposits)

print(f'Buys: {len(buys)} = ${buy_total:.2f}')
print(f'Sells: {len(sells)} = ${sell_total:.2f}')
print(f'Redeems: {len(redeems)} = ${redeem_total:.2f}')
print(f'Deposits: {len(deposits)} = ${dep_total:.2f}')
print(f'Net P&L: ${sell_total + redeem_total - buy_total:.2f}')
print(f'Expected balance: ${dep_total + sell_total + redeem_total - buy_total:.2f}')
print(f'User says balance: $45.82')
print()

print('=== RECENT (newest first) ===')
for t in trades[:20]:
    dt = datetime.utcfromtimestamp(int(t['timestamp'])).strftime('%H:%M')
    usd = float(t['usdcAmount'])
    tok = float(t['tokenAmount']) if t['tokenAmount'] else 0
    nm = t.get('tokenName', '')
    print(f'{dt} {t["action"]:7s} ${usd:>8.2f} {tok:>7.1f} {nm:>5s} | {t["marketName"][:52]}')

print()

# Group buys by market
mkt_buys = {}
for t in buys:
    m = t['marketName']
    if m not in mkt_buys:
        mkt_buys[m] = {'usd': 0, 'tokens': 0, 'side': t.get('tokenName', ''), 'ts': int(t['timestamp'])}
    mkt_buys[m]['usd'] += float(t['usdcAmount'])
    mkt_buys[m]['tokens'] += float(t['tokenAmount'])

print('=== POSITIONS (buys grouped, newest first) ===')
sorted_mkts = sorted(mkt_buys.items(), key=lambda x: x[1]['ts'], reverse=True)
for m, d in sorted_mkts[:15]:
    dt = datetime.utcfromtimestamp(d['ts']).strftime('%H:%M')
    print(f'{dt} ${d["usd"]:>7.2f} {d["tokens"]:>7.1f} {d["side"]:>5s} | {m[:55]}')

print()

# Open positions (bought but no redeem/sell)
redeemed = set()
for t in redeems:
    if float(t['usdcAmount']) > 0:
        redeemed.add(t['marketName'])
sold = set(t['marketName'] for t in sells)
closed = redeemed | sold

# Also count zero-redeems as closed (lost positions)
for t in redeems:
    if float(t['usdcAmount']) == 0:
        redeemed.add(t['marketName'])
closed = redeemed | sold

print('=== LIKELY OPEN (no redeem/sell found) ===')
open_total = 0
for m, d in sorted_mkts:
    if m not in closed:
        open_total += d['usd']
        print(f'  ${d["usd"]:>7.2f} {d["side"]:>5s} | {m[:60]}')

print(f'\nTotal in open positions: ${open_total:.2f}')
print(f'Expected free balance: ${dep_total + sell_total + redeem_total - buy_total + open_total:.2f}')
