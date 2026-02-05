"""Analyze ONLY today's (Feb 4-5) trades from CSV."""
import csv
from datetime import datetime, timezone
from collections import defaultdict

csv_path = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-04 (4).csv"

# Parse trades - only February 4 and 5
buys = defaultdict(lambda: {'side': None, 'cost': 0, 'shares': 0})
redeems = defaultdict(float)

with open(csv_path, encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        market = row['marketName']
        action = row['action']
        usdc = float(row['usdcAmount']) if row['usdcAmount'] else 0
        tokens = float(row['tokenAmount']) if row['tokenAmount'] else 0
        side = row['tokenName']
        ts = int(row['timestamp']) if row['timestamp'] else 0

        # Only February 4-5 trades (timestamp > Feb 4 midnight UTC)
        # Feb 4 2026 00:00 UTC = 1770163200
        if ts < 1770163200:
            continue

        # Only crypto 15m markets
        if 'Up or Down' not in market:
            continue
        if 'February 4' not in market and 'February 5' not in market:
            continue

        if action == 'Buy':
            buys[market]['side'] = side
            buys[market]['cost'] += usdc
            buys[market]['shares'] += tokens
        elif action == 'Redeem' and usdc > 0:
            redeems[market] += usdc

# Calculate results
wins = losses = 0
total_cost = total_redeemed = 0
pending_cost = 0
pending_markets = []

print("=" * 70)
print("TODAY'S TRADES (Feb 4-5, 2026)")
print("=" * 70)

for market, data in sorted(buys.items(), key=lambda x: x[0]):
    cost = data['cost']
    side = data['side']
    redeemed = redeems.get(market, 0)
    total_cost += cost

    if redeemed > 0:
        total_redeemed += redeemed
        pnl = redeemed - cost
        won = pnl > 0
        if won:
            wins += 1
            result = "WIN"
        else:
            losses += 1
            result = "LOSS"
        print(f"{result:4} | {side:4} | cost ${cost:>6.2f} | redeem ${redeemed:>6.2f} | pnl ${pnl:>+6.2f} | {market[:45]}")
    else:
        pending_cost += cost
        pending_markets.append((market, side, cost))

print(f"\n=== RESOLVED TRADES ===")
print(f"Wins: {wins} | Losses: {losses}")
wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
print(f"Win Rate: {wr:.1f}%")
print(f"Total Cost: ${total_cost:.2f}")
print(f"Total Redeemed: ${total_redeemed:.2f}")
print(f"Realized PnL: ${total_redeemed - (total_cost - pending_cost):+.2f}")

print(f"\n=== PENDING (still open) ===")
print(f"Count: {len(pending_markets)}")
print(f"At Risk: ${pending_cost:.2f}")
for market, side, cost in pending_markets[:10]:
    print(f"  ${cost:>6.2f} | {side:>4} | {market[:50]}")

print(f"\n=== BY SIDE (resolved only) ===")
up_w = up_l = down_w = down_l = 0
up_pnl = down_pnl = 0
for market, data in buys.items():
    redeemed = redeems.get(market, 0)
    if redeemed == 0:
        continue
    cost = data['cost']
    side = data['side']
    pnl = redeemed - cost
    if side == 'Up':
        if pnl > 0:
            up_w += 1
        else:
            up_l += 1
        up_pnl += pnl
    elif side == 'Down':
        if pnl > 0:
            down_w += 1
        else:
            down_l += 1
        down_pnl += pnl

up_total = up_w + up_l
down_total = down_w + down_l
print(f"UP:   {up_w}W/{up_l}L = {up_w/up_total*100 if up_total else 0:.0f}% WR | PnL: ${up_pnl:+.2f}")
print(f"DOWN: {down_w}W/{down_l}L = {down_w/down_total*100 if down_total else 0:.0f}% WR | PnL: ${down_pnl:+.2f}")
