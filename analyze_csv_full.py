"""Full analysis of Polymarket CSV trade history."""
import csv
import sys
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

csv_path = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-04 (1).csv"

# Parse all trades
buys = defaultdict(lambda: {'side': None, 'cost': 0, 'shares': 0, 'trades': []})
redeems = defaultdict(float)

with open(csv_path, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        market = row['marketName']
        action = row['action']
        usdc = float(row['usdcAmount']) if row['usdcAmount'] else 0
        tokens = float(row['tokenAmount']) if row['tokenAmount'] else 0
        side = row['tokenName']

        if action == 'Buy':
            buys[market]['side'] = side
            buys[market]['cost'] += usdc
            buys[market]['shares'] += tokens
            buys[market]['trades'].append({'cost': usdc, 'shares': tokens})
        elif action == 'Redeem' and usdc > 0:
            redeems[market] += usdc

# Categorize markets
crypto_15m = []
other = []

for market, data in buys.items():
    is_crypto = any(x in market.lower() for x in ['bitcoin', 'ethereum', 'solana', 'xrp'])
    is_15m = 'Up or Down' in market and any(x in market for x in ['AM', 'PM'])

    result = {
        'market': market,
        'side': data['side'],
        'cost': data['cost'],
        'shares': data['shares'],
        'redeemed': redeems.get(market, 0),
        'resolved': market in redeems and redeems[market] > 0,
        'pnl': redeems.get(market, 0) - data['cost'] if market in redeems else -data['cost'],
    }
    result['won'] = result['redeemed'] > result['cost'] if result['resolved'] else None

    # Extract asset
    if 'bitcoin' in market.lower() or 'btc' in market.lower():
        result['asset'] = 'BTC'
    elif 'ethereum' in market.lower() or 'eth' in market.lower():
        result['asset'] = 'ETH'
    elif 'solana' in market.lower() or 'sol' in market.lower():
        result['asset'] = 'SOL'
    elif 'xrp' in market.lower():
        result['asset'] = 'XRP'
    else:
        result['asset'] = 'OTHER'

    if is_crypto and is_15m:
        crypto_15m.append(result)
    else:
        other.append(result)

print("=" * 70)
print("POLYMARKET CSV TRADE ANALYSIS")
print("=" * 70)

# Overall stats
all_trades = crypto_15m + other
resolved = [t for t in all_trades if t['resolved']]
pending = [t for t in all_trades if not t['resolved']]

wins = sum(1 for t in resolved if t['won'])
losses = sum(1 for t in resolved if not t['won'])
total_cost = sum(t['cost'] for t in all_trades)
total_redeemed = sum(t['redeemed'] for t in resolved)
realized_pnl = sum(t['pnl'] for t in resolved)
pending_cost = sum(t['cost'] for t in pending)

print(f"\n=== OVERALL STATS ===")
print(f"Total markets traded: {len(all_trades)}")
print(f"Resolved: {len(resolved)} | Pending: {len(pending)}")
print(f"Wins: {wins} | Losses: {losses}")
print(f"Win Rate: {wins/(wins+losses)*100 if wins+losses else 0:.1f}%")
print(f"Total invested: ${total_cost:.2f}")
print(f"Total redeemed: ${total_redeemed:.2f}")
print(f"Realized PnL: ${realized_pnl:+.2f}")
print(f"Pending exposure: ${pending_cost:.2f}")

# Crypto 15m breakdown
print(f"\n=== CRYPTO 15-MINUTE MARKETS ===")
c_resolved = [t for t in crypto_15m if t['resolved']]
c_pending = [t for t in crypto_15m if not t['resolved']]
c_wins = sum(1 for t in c_resolved if t['won'])
c_losses = sum(1 for t in c_resolved if not t['won'])
c_pnl = sum(t['pnl'] for t in c_resolved)
c_pending_cost = sum(t['cost'] for t in c_pending)

print(f"Resolved: {c_wins}W/{c_losses}L = {c_wins/(c_wins+c_losses)*100 if c_wins+c_losses else 0:.1f}% WR")
print(f"Realized PnL: ${c_pnl:+.2f}")
print(f"Pending: {len(c_pending)} markets (${c_pending_cost:.2f} at risk)")

# By side
print(f"\n=== BY SIDE (Resolved Only) ===")
for side in ['Up', 'Down']:
    side_trades = [t for t in c_resolved if t['side'] == side]
    side_wins = sum(1 for t in side_trades if t['won'])
    side_losses = len(side_trades) - side_wins
    side_pnl = sum(t['pnl'] for t in side_trades)
    wr = side_wins / len(side_trades) * 100 if side_trades else 0
    print(f"{side:5}: {side_wins:3}W/{side_losses:2}L = {wr:5.1f}% WR | PnL: ${side_pnl:+.2f}")

# By asset
print(f"\n=== BY ASSET (Resolved Only) ===")
for asset in ['BTC', 'ETH', 'SOL', 'XRP']:
    asset_trades = [t for t in c_resolved if t['asset'] == asset]
    if not asset_trades:
        continue
    asset_wins = sum(1 for t in asset_trades if t['won'])
    asset_losses = len(asset_trades) - asset_wins
    asset_pnl = sum(t['pnl'] for t in asset_trades)
    wr = asset_wins / len(asset_trades) * 100 if asset_trades else 0
    print(f"{asset:4}: {asset_wins:3}W/{asset_losses:2}L = {wr:5.1f}% WR | PnL: ${asset_pnl:+.2f}")

# Pending breakdown
print(f"\n=== PENDING TRADES (Not Yet Resolved) ===")
for side in ['Up', 'Down']:
    side_pending = [t for t in c_pending if t['side'] == side]
    side_cost = sum(t['cost'] for t in side_pending)
    print(f"{side:5}: {len(side_pending):3} markets | ${side_cost:.2f} at risk")

# Show big pending positions
print(f"\n=== LARGEST PENDING POSITIONS ===")
c_pending.sort(key=lambda x: -x['cost'])
for t in c_pending[:10]:
    short_market = t['market'][:50]
    print(f"  ${t['cost']:6.2f} | {t['side']:4} | {short_market}")

# Other markets (non-crypto or non-15m)
if other:
    print(f"\n=== OTHER MARKETS (Non-Crypto 15m) ===")
    o_resolved = [t for t in other if t['resolved']]
    o_pending = [t for t in other if not t['resolved']]
    o_wins = sum(1 for t in o_resolved if t['won'])
    o_losses = len(o_resolved) - o_wins
    o_pnl = sum(t['pnl'] for t in o_resolved)
    o_pending_cost = sum(t['cost'] for t in o_pending)
    print(f"Resolved: {o_wins}W/{o_losses}L = {o_wins/(o_wins+o_losses)*100 if o_wins+o_losses else 0:.1f}% WR | PnL: ${o_pnl:+.2f}")
    print(f"Pending: {len(o_pending)} markets (${o_pending_cost:.2f})")
    for t in o_pending[:5]:
        print(f"  ${t['cost']:6.2f} | {t['market'][:55]}")
