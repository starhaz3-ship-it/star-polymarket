import csv
from datetime import datetime

path = r'C:\Users\Star\Downloads\Polymarket-History-2026-02-14 (8).csv'

# Handle BOM in header
with open(path, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

buys = [r for r in rows if r['action'] == 'Buy']
sells = [r for r in rows if r['action'] == 'Sell']
redeems = [r for r in rows if r['action'] == 'Redeem']

print(f'Total rows: {len(rows)}')
print(f'Buys: {len(buys)}, Sells: {len(sells)}, Redeems: {len(redeems)}')

total_buy = sum(float(r['usdcAmount']) for r in buys)
total_sell = sum(float(r['usdcAmount']) for r in sells)
total_redeem = sum(float(r['usdcAmount']) for r in redeems)
print(f'Total bought: ${total_buy:.2f}')
print(f'Total sold: ${total_sell:.2f}')
print(f'Total redeemed: ${total_redeem:.2f}')
print(f'Net PnL: ${total_sell + total_redeem - total_buy:.2f}')
print()

# All sells — these are kill-sells
print('=== ALL SELLS (kill-sells / partial kills) ===')
for r in sells:
    amt = float(r['usdcAmount'])
    tokens = float(r['tokenAmount'])
    price = amt / tokens if tokens > 0 else 0
    name = r.get('marketName', r.get('\ufeff"marketName"', ''))
    token = r.get('tokenName', '')
    print(f"  {name[:60]:60s} | ${amt:7.2f} | {tokens:5.0f} {token:4s} @ ${price:.2f}")

# Per-market PnL
print()
print('=== PER-MARKET PnL ===')
markets = {}
for r in rows:
    mkt = r.get('marketName', '')
    if not mkt:
        continue
    if mkt not in markets:
        markets[mkt] = {'buy': 0, 'sell': 0, 'redeem': 0, 'buy_tokens': 0, 'sell_tokens': 0}
    amt = float(r['usdcAmount'])
    tokens = float(r['tokenAmount'])
    if r['action'] == 'Buy':
        markets[mkt]['buy'] += amt
        markets[mkt]['buy_tokens'] += tokens
    elif r['action'] == 'Sell':
        markets[mkt]['sell'] += amt
        markets[mkt]['sell_tokens'] += tokens
    elif r['action'] == 'Redeem':
        markets[mkt]['redeem'] += amt

losers = []
for mkt, v in markets.items():
    pnl = v['sell'] + v['redeem'] - v['buy']
    if v['buy'] > 0:
        losers.append((pnl, mkt, v))

losers.sort()
print(f'\nTop 20 LOSERS:')
for pnl, mkt, v in losers[:20]:
    print(f'  ${pnl:+7.2f} | buy=${v["buy"]:6.2f} sell=${v["sell"]:6.2f} redeem=${v["redeem"]:6.2f} | {mkt[:55]}')

print(f'\nTop 10 WINNERS:')
for pnl, mkt, v in losers[-10:]:
    print(f'  ${pnl:+7.2f} | buy=${v["buy"]:6.2f} sell=${v["sell"]:6.2f} redeem=${v["redeem"]:6.2f} | {mkt[:55]}')

# Losses from sells specifically (kill-sells)
sell_loss = 0
for r in sells:
    # Each sell is a kill-sell where we bought at ~0.49 and sold lower
    sell_loss += float(r['usdcAmount'])
buy_for_sells = 0  # approximate: sells * ~0.49 buy price * tokens

print(f'\n=== SELL SUMMARY ===')
print(f'Total sell revenue: ${total_sell:.2f}')
print(f'Number of kill-sells: {len(sells)}')
for r in sells:
    amt = float(r['usdcAmount'])
    tokens = float(r['tokenAmount'])
    price = amt / tokens if tokens > 0 else 0
    name = r.get('marketName', '')
    token = r.get('tokenName', '')
    # Estimate loss: bought at ~0.48-0.49, sold at sell_price
    buy_price = 0.49 if token == 'Up' else 0.48
    est_loss = (price - buy_price) * tokens
    print(f"  {name[:50]:50s} | {token:4s} | sold {tokens:.0f} @ ${price:.2f} | est loss: ${est_loss:+.2f}")

# Recent activity timeline
print(f'\n=== LAST 30 TRANSACTIONS ===')
for r in rows[:30]:
    ts = int(r['timestamp'])
    dt = datetime.utcfromtimestamp(ts)
    amt = float(r['usdcAmount'])
    tokens = float(r['tokenAmount'])
    name = r.get('marketName', '')
    token = r.get('tokenName', '')
    print(f'  {dt.strftime("%m/%d %H:%M")} | {r["action"]:7s} | ${amt:7.2f} | {tokens:5.0f} {token:4s} | {name[:45]}')
