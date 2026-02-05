"""Analyze actual Polymarket trade history CSV."""
import csv
from collections import defaultdict

rows = []
with open(r'C:\Users\Star\Downloads\Polymarket-History-2026-02-04.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

print(f'Total rows: {len(rows)}')
print()

# Count by action type
actions = defaultdict(int)
action_usd = defaultdict(float)
for r in rows:
    act = r['action']
    amt = float(r['usdcAmount'])
    actions[act] += 1
    action_usd[act] += amt

print('=== ACTION SUMMARY ===')
for act in sorted(actions.keys()):
    print(f'  {act}: {actions[act]} txns, ${action_usd[act]:.2f} total')

print()

# Separate buys and redeems
buys = [r for r in rows if r['action'] == 'Buy']
redeems = [r for r in rows if r['action'] == 'Redeem']
sells = [r for r in rows if r['action'] == 'Sell']
rebates = [r for r in rows if 'rebate' in r['action'].lower() or 'rebate' in r['marketName'].lower()]

total_bought = sum(float(r['usdcAmount']) for r in buys)
total_redeemed = sum(float(r['usdcAmount']) for r in redeems)
total_sold = sum(float(r['usdcAmount']) for r in sells)
total_rebates = sum(float(r['usdcAmount']) for r in rebates)

print(f'Total spent on buys: ${total_bought:.2f}')
print(f'Total redeemed: ${total_redeemed:.2f}')
print(f'Total sold: ${total_sold:.2f}')
print(f'Maker rebates: ${total_rebates:.2f}')

pnl = total_redeemed + total_sold + total_rebates - total_bought
print(f'\n=== ACTUAL PnL: ${pnl:+.2f} ===')

# Group buys by market
print()
print('=== ACTUAL FILLS PER MARKET (buys, top 25 by size) ===')
market_buys = defaultdict(lambda: {'total_usd': 0, 'total_tokens': 0, 'side': '', 'count': 0})
for r in buys:
    mkt = r['marketName']
    side = r['tokenName']
    usd = float(r['usdcAmount'])
    tokens = float(r['tokenAmount'])
    market_buys[mkt]['total_usd'] += usd
    market_buys[mkt]['total_tokens'] += tokens
    market_buys[mkt]['side'] = side
    market_buys[mkt]['count'] += 1

for i, (mkt, data) in enumerate(sorted(market_buys.items(), key=lambda x: -x[1]['total_usd'])):
    avg_price = data['total_usd'] / data['total_tokens'] if data['total_tokens'] > 0 else 0
    print(f'  [{data["side"]}] ${data["total_usd"]:.2f} ({data["count"]} fills, avg ${avg_price:.4f}) | {mkt[:65]}')
    if i >= 24:
        print(f'  ... and {len(market_buys) - 25} more markets')
        break

# Detect unfilled / partial fills
buy_markets = set(market_buys.keys())
redeem_markets = set(r['marketName'] for r in redeems)
sell_markets = set(r['marketName'] for r in sells)

no_resolution = buy_markets - redeem_markets - sell_markets
print(f'\n=== MARKETS WITH NO REDEEM/SELL ({len(no_resolution)}) - LOSSES ===')
total_lost = 0
for mkt in sorted(no_resolution):
    data = market_buys[mkt]
    total_lost += data['total_usd']
    print(f'  [{data["side"]}] ${data["total_usd"]:.2f} | {mkt[:65]}')
print(f'Total lost on these: ${total_lost:.2f}')

# Win rate
print(f'\nAvg bet size per market: ${sum(d["total_usd"] for d in market_buys.values()) / len(market_buys):.2f}')
print(f'Unique markets traded: {len(market_buys)}')

wins = len(buy_markets & redeem_markets)
losses = len(no_resolution)
sold = len(buy_markets & sell_markets - redeem_markets)
print(f'\n=== ACTUAL WIN/LOSS ===')
print(f'Won (redeemed): {wins}')
print(f'Lost (no redeem): {losses}')
print(f'Sold: {sold}')
total = wins + losses + sold
if total > 0:
    print(f'Win rate: {wins/total*100:.1f}%')

# Win by side
print(f'\n=== WIN RATE BY SIDE ===')
for side in ['Up', 'Down']:
    side_buys = {mkt for mkt, d in market_buys.items() if d['side'] == side}
    side_wins = len(side_buys & redeem_markets)
    side_losses = len(side_buys & no_resolution)
    side_total = side_wins + side_losses
    wr = side_wins / side_total * 100 if side_total > 0 else 0
    side_spent = sum(market_buys[m]['total_usd'] for m in side_buys)
    side_redeemed = sum(float(r['usdcAmount']) for r in redeems if r['marketName'] in side_buys)
    print(f'  {side}: {side_wins}W/{side_losses}L ({wr:.1f}% WR) | Spent: ${side_spent:.2f} | Redeemed: ${side_redeemed:.2f} | PnL: ${side_redeemed - side_spent:+.2f}')

# Win by asset
print(f'\n=== WIN RATE BY ASSET ===')
for asset in ['Bitcoin', 'Ethereum', 'Solana', 'XRP']:
    asset_buys = {mkt for mkt in buy_markets if asset.lower() in mkt.lower()}
    asset_wins = len(asset_buys & redeem_markets)
    asset_losses = len(asset_buys & no_resolution)
    asset_total = asset_wins + asset_losses
    wr = asset_wins / asset_total * 100 if asset_total > 0 else 0
    asset_spent = sum(market_buys[m]['total_usd'] for m in asset_buys)
    asset_redeemed = sum(float(r['usdcAmount']) for r in redeems if r['marketName'] in asset_buys)
    print(f'  {asset}: {asset_wins}W/{asset_losses}L ({wr:.1f}% WR) | Spent: ${asset_spent:.2f} | Redeemed: ${asset_redeemed:.2f} | PnL: ${asset_redeemed - asset_spent:+.2f}')

# Check for the copy trader XRP mess
print(f'\n=== XRP COPY TRADER DAMAGE ===')
xrp_buys = {mkt: d for mkt, d in market_buys.items() if 'xrp' in mkt.lower()}
for mkt, data in sorted(xrp_buys.items(), key=lambda x: -x[1]['total_usd']):
    print(f'  [{data["side"]}] ${data["total_usd"]:.2f} ({data["count"]} fills) | {mkt[:65]}')
xrp_total_spent = sum(d['total_usd'] for d in xrp_buys.values())
xrp_redeemed = sum(float(r['usdcAmount']) for r in redeems if 'xrp' in r['marketName'].lower())
print(f'XRP total spent: ${xrp_total_spent:.2f} | Redeemed: ${xrp_redeemed:.2f} | PnL: ${xrp_redeemed - xrp_total_spent:+.2f}')
