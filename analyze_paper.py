"""Analyze paper trader for insights to apply to live."""
import json
from collections import defaultdict

with open('ta_paper_results.json', 'r') as f:
    data = json.load(f)

trades = data.get('trades', {})
closed = [t for t in trades.values() if t.get('status') == 'closed']

print(f"=== PAPER TRADER ANALYSIS ===")
print(f"Total closed trades: {len(closed)}")

wins = [t for t in closed if t.get('pnl', 0) > 0]
losses = [t for t in closed if t.get('pnl', 0) <= 0]
print(f"Wins: {len(wins)} | Losses: {len(losses)}")
print(f"Win Rate: {len(wins)/len(closed)*100:.1f}%")
print(f"Total PnL: ${sum(t.get('pnl', 0) for t in closed):.2f}")

# By side
print(f"\n=== BY SIDE ===")
for side in ['UP', 'DOWN']:
    side_trades = [t for t in closed if t.get('side') == side]
    side_wins = [t for t in side_trades if t.get('pnl', 0) > 0]
    side_pnl = sum(t.get('pnl', 0) for t in side_trades)
    wr = len(side_wins) / len(side_trades) * 100 if side_trades else 0
    print(f"{side}: {len(side_wins)}W/{len(side_trades)-len(side_wins)}L = {wr:.1f}% WR | PnL: ${side_pnl:+.2f}")

# By entry price bucket
print(f"\n=== BY ENTRY PRICE ===")
for low, high in [(0, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 1.0)]:
    bucket = [t for t in closed if low <= t.get('entry_price', 0) < high]
    if not bucket:
        continue
    bucket_wins = [t for t in bucket if t.get('pnl', 0) > 0]
    bucket_pnl = sum(t.get('pnl', 0) for t in bucket)
    wr = len(bucket_wins) / len(bucket) * 100 if bucket else 0
    print(f"${low:.2f}-${high:.2f}: {len(bucket_wins)}W/{len(bucket)-len(bucket_wins)}L = {wr:.1f}% WR | PnL: ${bucket_pnl:+.2f}")

# By edge bucket
print(f"\n=== BY EDGE ===")
for low, high in [(0, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 1.0)]:
    bucket = [t for t in closed if low <= t.get('edge_at_entry', 0) < high]
    if not bucket:
        continue
    bucket_wins = [t for t in bucket if t.get('pnl', 0) > 0]
    bucket_pnl = sum(t.get('pnl', 0) for t in bucket)
    wr = len(bucket_wins) / len(bucket) * 100 if bucket else 0
    print(f"{low*100:.0f}%-{high*100:.0f}%: {len(bucket_wins)}W/{len(bucket)-len(bucket_wins)}L = {wr:.1f}% WR | PnL: ${bucket_pnl:+.2f}")

# By KL divergence
print(f"\n=== BY KL DIVERGENCE ===")
for low, high in [(0, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 1.0)]:
    bucket = [t for t in closed if low <= t.get('kl_divergence', 0) < high]
    if not bucket:
        continue
    bucket_wins = [t for t in bucket if t.get('pnl', 0) > 0]
    bucket_pnl = sum(t.get('pnl', 0) for t in bucket)
    wr = len(bucket_wins) / len(bucket) * 100 if bucket else 0
    print(f"KL {low:.2f}-{high:.2f}: {len(bucket_wins)}W/{len(bucket)-len(bucket_wins)}L = {wr:.1f}% WR | PnL: ${bucket_pnl:+.2f}")

# Best trades
print(f"\n=== TOP 5 WINNING PATTERNS ===")
wins_sorted = sorted(wins, key=lambda x: x.get('pnl', 0), reverse=True)[:5]
for t in wins_sorted:
    print(f"  +${t.get('pnl', 0):.2f} | {t.get('side')} @ ${t.get('entry_price', 0):.2f} | Edge: {t.get('edge_at_entry', 0)*100:.0f}% | KL: {t.get('kl_divergence', 0):.3f}")

# Worst trades
print(f"\n=== TOP 5 LOSING PATTERNS ===")
losses_sorted = sorted(losses, key=lambda x: x.get('pnl', 0))[:5]
for t in losses_sorted:
    print(f"  ${t.get('pnl', 0):.2f} | {t.get('side')} @ ${t.get('entry_price', 0):.2f} | Edge: {t.get('edge_at_entry', 0)*100:.0f}% | KL: {t.get('kl_divergence', 0):.3f}")

# Recommendations
print(f"\n=== RECOMMENDATIONS ===")
up_trades = [t for t in closed if t.get('side') == 'UP']
down_trades = [t for t in closed if t.get('side') == 'DOWN']
up_wr = len([t for t in up_trades if t.get('pnl', 0) > 0]) / len(up_trades) * 100 if up_trades else 0
down_wr = len([t for t in down_trades if t.get('pnl', 0) > 0]) / len(down_trades) * 100 if down_trades else 0

if down_wr > up_wr + 5:
    print(f"1. DOWN trades outperform UP by {down_wr - up_wr:.0f}% - favor DOWN")
else:
    print(f"1. Both sides similar ({up_wr:.0f}% vs {down_wr:.0f}%) - trade both")

# Find optimal entry price
best_entry = 0
best_wr = 0
for max_entry in [0.40, 0.45, 0.50, 0.55]:
    bucket = [t for t in closed if t.get('entry_price', 0) <= max_entry]
    if len(bucket) >= 10:
        wr = len([t for t in bucket if t.get('pnl', 0) > 0]) / len(bucket) * 100
        if wr > best_wr:
            best_wr = wr
            best_entry = max_entry
print(f"2. Optimal max entry price: ${best_entry:.2f} ({best_wr:.0f}% WR)")

# Find optimal edge
best_edge = 0
best_wr = 0
for min_edge in [0.10, 0.15, 0.20, 0.25, 0.30]:
    bucket = [t for t in closed if t.get('edge_at_entry', 0) >= min_edge]
    if len(bucket) >= 10:
        wr = len([t for t in bucket if t.get('pnl', 0) > 0]) / len(bucket) * 100
        if wr > best_wr:
            best_wr = wr
            best_edge = min_edge
print(f"3. Optimal min edge: {best_edge*100:.0f}% ({best_wr:.0f}% WR)")
