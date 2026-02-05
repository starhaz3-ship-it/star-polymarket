"""
Backtest filter combinations against actual trade data to find optimal settings.
Uses ta_live_results.json which has features recorded for each trade.
Tests different ATR, RSI, edge, volatility, side filters to maximize win rate.
"""
import json
from collections import defaultdict
from itertools import product

with open('ta_live_results.json', 'r') as f:
    data = json.load(f)

trades = list(data['trades'].values())
# Only use closed trades with actual outcomes (non-zero PnL)
closed = [t for t in trades if t['status'] == 'closed' and t['pnl'] != 0]
print(f"Backtesting against {len(closed)} closed trades with outcomes\n")

def backtest(trades, filters):
    """Apply filters and return stats."""
    passed = []
    for t in trades:
        f = t.get('features', {})
        side = t['side']
        edge = t.get('edge_at_entry', 0)
        strength = t.get('signal_strength', '')
        rsi = f.get('rsi', 50)
        vol = f.get('btc_volatility', 0)
        model_conf = f.get('model_confidence', 0)
        time_rem = f.get('time_remaining', 15)

        # Parse entry hour
        try:
            from datetime import datetime
            entry_hour = datetime.fromisoformat(t['entry_time']).hour
        except:
            entry_hour = 12

        # Apply filters
        skip = False

        # Skip hours filter
        if 'skip_hours' in filters and entry_hour in filters['skip_hours']:
            skip = True

        # Edge minimum
        if 'min_edge' in filters and edge < filters['min_edge']:
            skip = True

        # Volatility cap
        if 'max_vol' in filters and vol > filters['max_vol']:
            skip = True

        # RSI filter for DOWN
        if 'down_max_rsi' in filters and side == 'DOWN' and rsi >= filters['down_max_rsi']:
            skip = True

        # RSI filter for UP
        if 'up_min_rsi' in filters and side == 'UP' and rsi <= filters['up_min_rsi']:
            skip = True

        # UP minimum edge
        if 'up_min_edge' in filters and side == 'UP' and edge < filters['up_min_edge']:
            skip = True

        # Block UP entirely
        if filters.get('down_only') and side == 'UP':
            skip = True

        # Model confidence minimum
        if 'min_model_conf' in filters and model_conf < filters['min_model_conf']:
            skip = True

        # Max entry price
        if 'max_entry' in filters and t.get('entry_price', 0) > filters['max_entry']:
            skip = True

        if not skip:
            passed.append(t)

    if not passed:
        return None

    wins = len([t for t in passed if t['pnl'] > 0])
    losses = len([t for t in passed if t['pnl'] < 0])
    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    pnl = sum(t['pnl'] for t in passed)
    avg_pnl = pnl / total if total > 0 else 0
    trades_per_hour = total / 27  # ~27 hours of data

    return {
        'wr': wr, 'wins': wins, 'losses': losses, 'total': total,
        'pnl': pnl, 'avg_pnl': avg_pnl, 'per_hour': trades_per_hour
    }

# ============================================================
# SYSTEMATIC GRID SEARCH
# ============================================================
print("=" * 80)
print("GRID SEARCH: Finding optimal filter combination for max WR + volume")
print("=" * 80)

# Parameter grid
edge_options = [0.08, 0.10, 0.15, 0.20, 0.25]
vol_options = [0.10, 0.12, 0.15, 0.20, 999]  # 999 = no filter
down_rsi_options = [45, 50, 55, 60, 999]  # max RSI for DOWN trades
up_mode_options = ['allow', 'strict', 'block']  # how to handle UP trades
skip_hours_options = [
    set(),                          # no skip
    {6, 7, 8},                      # original
    {6, 7, 8, 14, 15, 16, 20},     # expanded
    {6, 7, 8, 14, 15, 16, 19, 20}, # more aggressive
]
entry_price_options = [0.48, 0.50, 0.52, 0.55]

results = []

for edge_min in edge_options:
    for max_vol in vol_options:
        for down_rsi in down_rsi_options:
            for up_mode in up_mode_options:
                for skip_hrs in skip_hours_options:
                    for max_entry in entry_price_options:
                        filters = {
                            'min_edge': edge_min,
                            'max_vol': max_vol if max_vol != 999 else 999,
                            'down_max_rsi': down_rsi if down_rsi != 999 else 999,
                            'skip_hours': skip_hrs,
                            'max_entry': max_entry,
                        }

                        if up_mode == 'block':
                            filters['down_only'] = True
                        elif up_mode == 'strict':
                            filters['up_min_edge'] = 0.30
                            filters['up_min_rsi'] = 55

                        r = backtest(closed, filters)
                        if r and r['total'] >= 15:  # Need at least 15 trades for significance
                            results.append({
                                'edge': edge_min, 'vol': max_vol, 'rsi': down_rsi,
                                'up': up_mode, 'skip': len(skip_hrs),
                                'entry': max_entry,
                                **r
                            })

# Sort by win rate, then by total trades (prefer more volume)
results.sort(key=lambda x: (x['wr'], x['total']), reverse=True)

print(f"\nTested {len(results)} filter combinations\n")

# Top 20 by win rate (with min 15 trades)
print("=== TOP 20 HIGHEST WIN RATE (min 15 trades) ===")
print(f"{'WR%':>5} {'W/L':>7} {'PnL':>8} {'$/hr':>5} {'Edge':>5} {'Vol':>5} {'RSI':>4} {'UP':>7} {'Skip':>4} {'Entry':>5}")
print("-" * 70)
for r in results[:20]:
    print(f"{r['wr']:5.1f} {r['wins']:3d}/{r['losses']:<3d} ${r['pnl']:+7.2f} {r['per_hour']:5.1f} {r['edge']:5.2f} {r['vol']:5.2f} {r['rsi']:4d} {r['up']:>7} {r['skip']:4d} {r['entry']:5.2f}")

# Top 20 by PnL
print("\n=== TOP 20 HIGHEST PnL (min 15 trades) ===")
pnl_results = sorted(results, key=lambda x: x['pnl'], reverse=True)
print(f"{'WR%':>5} {'W/L':>7} {'PnL':>8} {'$/hr':>5} {'Edge':>5} {'Vol':>5} {'RSI':>4} {'UP':>7} {'Skip':>4} {'Entry':>5}")
print("-" * 70)
for r in pnl_results[:20]:
    print(f"{r['wr']:5.1f} {r['wins']:3d}/{r['losses']:<3d} ${r['pnl']:+7.2f} {r['per_hour']:5.1f} {r['edge']:5.2f} {r['vol']:5.2f} {r['rsi']:4d} {r['up']:>7} {r['skip']:4d} {r['entry']:5.2f}")

# Best balanced (WR >= 65% AND most trades)
print("\n=== BEST BALANCED: WR >= 65% with MOST trades ===")
balanced = [r for r in results if r['wr'] >= 65]
balanced.sort(key=lambda x: x['total'], reverse=True)
print(f"{'WR%':>5} {'W/L':>7} {'PnL':>8} {'$/hr':>5} {'Edge':>5} {'Vol':>5} {'RSI':>4} {'UP':>7} {'Skip':>4} {'Entry':>5}")
print("-" * 70)
for r in balanced[:15]:
    print(f"{r['wr']:5.1f} {r['wins']:3d}/{r['losses']:<3d} ${r['pnl']:+7.2f} {r['per_hour']:5.1f} {r['edge']:5.2f} {r['vol']:5.2f} {r['rsi']:4d} {r['up']:>7} {r['skip']:4d} {r['entry']:5.2f}")

# 70%+ WR
print("\n=== 70%+ WIN RATE (with most trades) ===")
high_wr = [r for r in results if r['wr'] >= 70]
high_wr.sort(key=lambda x: x['total'], reverse=True)
print(f"{'WR%':>5} {'W/L':>7} {'PnL':>8} {'$/hr':>5} {'Edge':>5} {'Vol':>5} {'RSI':>4} {'UP':>7} {'Skip':>4} {'Entry':>5}")
print("-" * 70)
for r in high_wr[:15]:
    print(f"{r['wr']:5.1f} {r['wins']:3d}/{r['losses']:<3d} ${r['pnl']:+7.2f} {r['per_hour']:5.1f} {r['edge']:5.2f} {r['vol']:5.2f} {r['rsi']:4d} {r['up']:>7} {r['skip']:4d} {r['entry']:5.2f}")

# 75%+ WR
print("\n=== 75%+ WIN RATE ===")
v_high = [r for r in results if r['wr'] >= 75]
v_high.sort(key=lambda x: x['total'], reverse=True)
if v_high:
    print(f"{'WR%':>5} {'W/L':>7} {'PnL':>8} {'$/hr':>5} {'Edge':>5} {'Vol':>5} {'RSI':>4} {'UP':>7} {'Skip':>4} {'Entry':>5}")
    print("-" * 70)
    for r in v_high[:15]:
        print(f"{r['wr']:5.1f} {r['wins']:3d}/{r['losses']:<3d} ${r['pnl']:+7.2f} {r['per_hour']:5.1f} {r['edge']:5.2f} {r['vol']:5.2f} {r['rsi']:4d} {r['up']:>7} {r['skip']:4d} {r['entry']:5.2f}")
else:
    print("No combinations achieved 75%+ WR with min 15 trades")

# Show current settings performance
print("\n=== CURRENT V2 SETTINGS PERFORMANCE ===")
current = backtest(closed, {
    'min_edge': 0.20,
    'max_vol': 0.15,
    'down_max_rsi': 55,
    'up_min_edge': 0.25,
    'up_min_rsi': 55,
    'skip_hours': {6, 7, 8, 14, 15, 16, 20},
    'max_entry': 0.50,
})
if current:
    print(f"Win Rate: {current['wr']:.1f}%")
    print(f"Trades: {current['wins']}W/{current['losses']}L ({current['total']} total)")
    print(f"PnL: ${current['pnl']:+.2f}")
    print(f"Trades/hour: {current['per_hour']:.1f}")
else:
    print("Too few trades pass current filters")

# Baseline (no filters)
print("\n=== BASELINE (no filters) ===")
baseline = backtest(closed, {})
if baseline:
    print(f"Win Rate: {baseline['wr']:.1f}%")
    print(f"Trades: {baseline['wins']}W/{baseline['losses']}L ({baseline['total']} total)")
    print(f"PnL: ${baseline['pnl']:+.2f}")
