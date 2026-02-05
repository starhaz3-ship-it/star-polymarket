"""Analyze recent losses to find patterns."""
import json

d = json.load(open('ta_live_results.json'))
trades = d.get('trades', [])

# Get recent losses
losses = [t for t in trades if t.get('pnl', 0) < 0]
wins = [t for t in trades if t.get('pnl', 0) > 0]

print(f"=== LOSS ANALYSIS ({len(losses)} losses) ===\n")

# Group by side
up_losses = [t for t in losses if t.get('side') == 'UP']
down_losses = [t for t in losses if t.get('side') == 'DOWN']
print(f"UP losses: {len(up_losses)}")
print(f"DOWN losses: {len(down_losses)}")

# Group by asset
for asset in ['BTC', 'ETH', 'SOL', 'XRP']:
    asset_losses = [t for t in losses if asset in t.get('market', '')]
    asset_wins = [t for t in wins if asset in t.get('market', '')]
    if asset_losses or asset_wins:
        total = len(asset_losses) + len(asset_wins)
        wr = len(asset_wins) / total * 100 if total else 0
        print(f"{asset}: {len(asset_wins)}W/{len(asset_losses)}L = {wr:.0f}% WR")

print(f"\n=== RECENT 10 TRADES ===")
for t in trades[-10:]:
    market = t.get('market', '?')[:45]
    side = t.get('side', '?')
    pnl = t.get('pnl', 0)
    entry = t.get('entry_price', 0)
    result = 'WIN' if pnl > 0 else 'LOSS'
    print(f"{result:4} | {side:4} @ ${entry:.2f} | {market}")

print(f"\n=== LOSS DETAILS ===")
for t in losses[-7:]:
    market = t.get('market', '?')[:50]
    side = t.get('side', '?')
    pnl = t.get('pnl', 0)
    entry = t.get('entry_price', 0)
    confidence = t.get('confidence', 0)
    edge = t.get('edge', 0)
    print(f"LOSS ${pnl:.2f} | {side} @ ${entry:.2f} | conf={confidence:.0%} edge={edge:.0%}")
    print(f"  {market}")
