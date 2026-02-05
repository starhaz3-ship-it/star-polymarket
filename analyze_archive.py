"""Analyze the permanent archive to calculate actual PnL."""
import json
import httpx
import os
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Load the archive
with open('ta_live_archive.json') as f:
    data = json.load(f)

trades = data['trades']
print(f'Analyzing {len(trades)} trades...')

# Fetch current positions to find dead ($0) positions
proxy = os.getenv('POLYMARKET_PROXY_ADDRESS', '')
dead_positions = set()  # condition_ids of $0 positions

if proxy:
    try:
        client = httpx.Client(timeout=30)
        r = client.get('https://data-api.polymarket.com/positions',
                       params={'user': proxy, 'sizeThreshold': 0})
        positions = r.json()

        for p in positions:
            size = float(p.get('size', 0))
            price = float(p.get('curPrice', p.get('price', 0)))
            cid = p.get('conditionId', '')
            title = p.get('title', '')

            # Dead position = has shares but worth $0
            if size > 0 and price == 0 and 'Up or Down' in title:
                dead_positions.add(cid)

        print(f'Found {len(dead_positions)} dead positions (resolved to $0)')
    except Exception as e:
        print(f'Warning: Could not fetch positions: {e}')

print('='*70)

# Group by market (condition_id)
by_market = defaultdict(list)
for t in trades:
    by_market[t['condition_id']].append(t)

# Analyze each market
wins = losses = pending = 0
win_pnl = loss_pnl = pending_cost = 0
up_w = up_l = down_w = down_l = 0
up_pnl = down_pnl = 0

results = []
for cid, market_trades in by_market.items():
    buys = [t for t in market_trades if t['side'] == 'BUY']
    redeems = [t for t in market_trades if not t['side']]  # Redemptions have no side

    if not buys:
        continue

    # Calculate cost
    total_cost = sum(t['usdc'] for t in buys)
    total_shares = sum(t['size'] for t in buys)
    outcome = buys[0]['outcome']
    market = buys[0]['market']

    # Calculate redemption value
    redemption = sum(t['usdc'] for t in redeems)

    pnl = redemption - total_cost

    # A trade is only a WIN if we got back MORE than we staked
    # Getting $0.01 back on a $5 stake = LOSS
    # Dead positions ($0 in wallet) = LOSS even without redemption record
    if redeems:  # Has redemption record (resolved)
        if redemption > total_cost:  # Actually profitable
            wins += 1
            win_pnl += pnl
            status = 'WIN'
            if outcome == 'Up':
                up_w += 1
                up_pnl += pnl
            else:
                down_w += 1
                down_pnl += pnl
        else:  # Redemption <= cost = LOSS (includes $0.01 redemptions)
            losses += 1
            loss_pnl += pnl
            status = 'LOSS'
            if outcome == 'Up':
                up_l += 1
                up_pnl += pnl
            else:
                down_l += 1
                down_pnl += pnl
    elif cid in dead_positions:  # Dead position - resolved to $0, no redemption
        losses += 1
        pnl = -total_cost  # Lost entire stake
        loss_pnl += pnl
        status = 'DEAD'
        if outcome == 'Up':
            up_l += 1
            up_pnl += pnl
        else:
            down_l += 1
            down_pnl += pnl
    else:
        pending += 1
        pending_cost += total_cost
        status = 'PENDING'

    results.append({
        'market': market,
        'outcome': outcome,
        'cost': total_cost,
        'redemption': redemption,
        'pnl': pnl,
        'status': status
    })

# Print summary
print(f'\n=== RESOLVED TRADES ===')
print(f'Wins: {wins} (PnL: ${win_pnl:+.2f})')
print(f'Losses: {losses} (PnL: ${loss_pnl:+.2f})')
total = wins + losses
if total > 0:
    print(f'Win Rate: {wins/total*100:.1f}%')
print(f'Net Realized PnL: ${win_pnl + loss_pnl:+.2f}')
print(f'\nPending: {pending} trades (${pending_cost:.2f} at risk)')

print(f'\n=== BY SIDE ===')
up_total = up_w + up_l
down_total = down_w + down_l
if up_total > 0:
    print(f'UP:   {up_w}W/{up_l}L = {up_w/up_total*100:.0f}% WR | PnL: ${up_pnl:+.2f}')
if down_total > 0:
    print(f'DOWN: {down_w}W/{down_l}L = {down_w/down_total*100:.0f}% WR | PnL: ${down_pnl:+.2f}')

# Show dead positions (unredeemed $0 losses)
dead_trades = [r for r in results if r['status'] == 'DEAD']
if dead_trades:
    print(f'\n=== DEAD POSITIONS (unredeemed $0 losses) ===')
    print(f'Count: {len(dead_trades)} | Total Lost: ${sum(abs(r["pnl"]) for r in dead_trades):.2f}')
    for r in dead_trades[:10]:
        print(f"  DEAD ${r['pnl']:+.2f} | {r['outcome']:>4} | cost ${r['cost']:.2f} | {r['market'][:45]}")
    if len(dead_trades) > 10:
        print(f"  ... and {len(dead_trades) - 10} more")

# Show redeemed losses
print(f'\n=== REDEEMED LOSSES ===')
loss_trades = [r for r in results if r['status'] == 'LOSS'][:10]
for r in loss_trades:
    print(f"  LOSS ${r['pnl']:+.2f} | {r['outcome']:>4} | cost ${r['cost']:.2f} | {r['market'][:45]}")

# Show some wins
print(f'\n=== SAMPLE WINS ===')
win_trades = [r for r in results if r['status'] == 'WIN'][:10]
for r in win_trades:
    print(f"  WIN  ${r['pnl']:+.2f} | {r['outcome']:>4} | cost ${r['cost']:.2f} | {r['market'][:45]}")
