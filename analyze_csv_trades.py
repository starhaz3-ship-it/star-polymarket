"""Analyze Polymarket trade history CSV for win rate and PnL."""
import csv
import sys
from collections import defaultdict
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

csv_path = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-04 (1).csv"

# Parse trades
buys = []  # (market, side, cost, shares, ts)
redeems = defaultdict(float)  # market -> total redeemed

with open(csv_path, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        market = row['marketName']
        action = row['action']
        usdc = float(row['usdcAmount']) if row['usdcAmount'] else 0
        tokens = float(row['tokenAmount']) if row['tokenAmount'] else 0
        side = row['tokenName']
        ts = int(row['timestamp']) if row['timestamp'] else 0

        # Only 15-minute crypto Up/Down markets
        if 'Up or Down' not in market:
            continue
        if 'February 4' not in market:
            continue

        if action == 'Buy':
            buys.append({
                'market': market,
                'side': side,
                'cost': usdc,
                'shares': tokens,
                'ts': ts,
            })
        elif action == 'Redeem' and usdc > 0:
            redeems[market] += usdc

# Group buys by market
market_buys = defaultdict(list)
for b in buys:
    market_buys[b['market']].append(b)

# Calculate per-market results
results = []
for market, trades in market_buys.items():
    total_cost = sum(t['cost'] for t in trades)
    total_shares = sum(t['shares'] for t in trades)
    sides = list(set(t['side'] for t in trades if t['side']))
    side = sides[0] if len(sides) == 1 else '/'.join(sides)

    redeemed = redeems.get(market, 0)
    pnl = redeemed - total_cost
    won = redeemed > total_cost

    # Extract asset
    asset = 'BTC' if 'Bitcoin' in market else 'ETH' if 'Ethereum' in market else 'SOL' if 'Solana' in market else '?'

    results.append({
        'market': market,
        'asset': asset,
        'side': side,
        'cost': total_cost,
        'shares': total_shares,
        'redeemed': redeemed,
        'pnl': pnl,
        'won': won,
        'pending': redeemed == 0 and total_cost > 0,
    })

# Sort by timestamp (extract time from market name)
results.sort(key=lambda x: x['market'], reverse=True)

# Summary stats
wins = sum(1 for r in results if r['won'] and not r['pending'])
losses = sum(1 for r in results if not r['won'] and not r['pending'])
pending = sum(1 for r in results if r['pending'])
total_pnl = sum(r['pnl'] for r in results if not r['pending'])
total_cost = sum(r['cost'] for r in results)
total_redeemed = sum(r['redeemed'] for r in results)

print("=" * 80)
print("POLYMARKET TRADE HISTORY - February 4, 2026")
print("=" * 80)

# By side
up_trades = [r for r in results if r['side'] == 'Up' and not r['pending']]
down_trades = [r for r in results if r['side'] == 'Down' and not r['pending']]
up_wins = sum(1 for r in up_trades if r['won'])
down_wins = sum(1 for r in down_trades if r['won'])

print(f"\n=== BY SIDE ===")
print(f"UP trades:   {len(up_trades):3d} | Wins: {up_wins:3d} | WR: {up_wins/len(up_trades)*100 if up_trades else 0:.1f}% | PnL: ${sum(r['pnl'] for r in up_trades):+.2f}")
print(f"DOWN trades: {len(down_trades):3d} | Wins: {down_wins:3d} | WR: {down_wins/len(down_trades)*100 if down_trades else 0:.1f}% | PnL: ${sum(r['pnl'] for r in down_trades):+.2f}")

# By asset
print(f"\n=== BY ASSET ===")
for asset in ['BTC', 'ETH', 'SOL']:
    asset_trades = [r for r in results if r['asset'] == asset and not r['pending']]
    asset_wins = sum(1 for r in asset_trades if r['won'])
    asset_pnl = sum(r['pnl'] for r in asset_trades)
    wr = asset_wins/len(asset_trades)*100 if asset_trades else 0
    print(f"{asset}: {len(asset_trades):3d} trades | Wins: {asset_wins:3d} | WR: {wr:.1f}% | PnL: ${asset_pnl:+.2f}")

# Recent trades detail
print(f"\n=== RECENT TRADES (most recent first) ===")
print(f"{'Market':<55} {'Side':>5} {'Cost':>8} {'Redeem':>8} {'PnL':>8} {'Result':>8}")
print("-" * 95)
for r in results[:30]:
    short = r['market'].replace('Up or Down - ', '').replace(', ', ' ')[:52]
    result = 'WIN' if r['won'] else 'LOSS' if not r['pending'] else 'OPEN'
    print(f"{short:<55} {r['side']:>5} ${r['cost']:>6.2f} ${r['redeemed']:>6.2f} ${r['pnl']:>+6.2f} {result:>8}")

# Summary
print(f"\n=== SUMMARY ===")
print(f"Closed trades: {wins + losses}")
print(f"Wins: {wins} | Losses: {losses}")
print(f"Win Rate: {wins/(wins+losses)*100 if wins+losses else 0:.1f}%")
print(f"Total Cost: ${total_cost:.2f}")
print(f"Total Redeemed: ${total_redeemed:.2f}")
print(f"Net PnL: ${total_pnl:+.2f}")
print(f"Pending (open): {pending} markets")

# ROI
if total_cost > 0:
    roi = (total_redeemed / total_cost - 1) * 100
    print(f"ROI: {roi:+.1f}%")
