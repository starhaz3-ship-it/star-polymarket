"""Analyze REAL losses - pending positions that went to zero."""
import csv
from collections import defaultdict

csv_path = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-04 (4).csv"

# Parse ALL buys and redeems for today
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

        # Only Feb 4-5
        if ts < 1770163200:
            continue
        if 'Up or Down' not in market:
            continue
        if 'February 4' not in market and 'February 5' not in market:
            continue

        if action == 'Buy':
            buys[market]['side'] = side
            buys[market]['cost'] += usdc
            buys[market]['shares'] += tokens
        elif action == 'Redeem':
            # Even $0 redeems mean the market resolved
            redeems[market] = max(redeems.get(market, 0), usdc)

# Now check: if a market has a redeem entry (even $0), it's resolved
# $0 redeem = LOSS (we bet wrong side)
wins = losses = 0
win_pnl = loss_pnl = 0
pending_count = 0
pending_cost = 0

print("=" * 70)
print("REAL LOSS ANALYSIS - Today (Feb 4-5)")
print("=" * 70)

print("\n=== LOSSES (redeemed $0 or less than cost) ===")
for market, data in sorted(buys.items()):
    cost = data['cost']
    side = data['side']

    if market in redeems:
        redeemed = redeems[market]
        pnl = redeemed - cost
        if pnl < 0:
            losses += 1
            loss_pnl += pnl
            print(f"LOSS ${pnl:>+7.2f} | {side:>4} | cost ${cost:>6.2f} | got ${redeemed:>6.2f} | {market[:45]}")
        else:
            wins += 1
            win_pnl += pnl
    else:
        pending_count += 1
        pending_cost += cost

print(f"\n=== SUMMARY ===")
print(f"Wins: {wins} (PnL: ${win_pnl:+.2f})")
print(f"Losses: {losses} (PnL: ${loss_pnl:+.2f})")
total = wins + losses
if total > 0:
    print(f"Win Rate: {wins/total*100:.1f}%")
print(f"Net Realized PnL: ${win_pnl + loss_pnl:+.2f}")
print(f"\nStill Pending: {pending_count} markets (${pending_cost:.2f} at risk)")

# Check for $0 redeems specifically
print(f"\n=== $0 REDEMPTIONS (total losses) ===")
zero_redeems = 0
for market in redeems:
    if redeems[market] == 0 and market in buys:
        zero_redeems += 1
        cost = buys[market]['cost']
        side = buys[market]['side']
        print(f"  ${cost:>6.2f} lost | {side:>4} | {market[:50]}")
print(f"Total $0 redeems: {zero_redeems}")
