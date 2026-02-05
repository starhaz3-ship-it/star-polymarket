"""Full reset - clean slate with ML learning preserved."""
import json

with open('ta_live_results.json', 'r') as f:
    data = json.load(f)

print(f"OLD STATE:")
print(f"  W/L: {data.get('wins', 0)}/{data.get('losses', 0)}")
print(f"  PnL: ${data.get('total_pnl', 0):.2f}")
print(f"  Bankroll: ${data.get('bankroll', 0):.2f}")
print(f"  Trades tracked: {len(data.get('trades', {}))}")

# Keep ML weights for learning, clear everything else
ml_win = data.get('ml_win_features', [])
ml_loss = data.get('ml_loss_features', [])
ml_weights = data.get('ml_weights', {})

# Fresh start
data = {
    'start_time': __import__('datetime').datetime.now().isoformat(),
    'last_updated': __import__('datetime').datetime.now().isoformat(),
    'dry_run': False,
    'trades': {},  # Clear all trades
    'total_pnl': 0.0,
    'wins': 0,
    'losses': 0,
    'signals_count': 0,
    'ml_rejections': 0,
    'consecutive_wins': 0,
    'consecutive_losses': 0,
    'bankroll': 73.63,  # From screenshot
    'ml_win_features': ml_win,  # Keep for learning
    'ml_loss_features': ml_loss,  # Keep for learning
    'ml_weights': ml_weights,  # Keep for learning
}

with open('ta_live_results.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"\nNEW STATE:")
print(f"  W/L: 0/0")
print(f"  PnL: $0.00")
print(f"  Bankroll: $73.63")
print(f"  Trades: CLEARED")
print(f"  ML features preserved: {len(ml_win)} wins, {len(ml_loss)} losses")
print(f"\nREADY FOR FRESH START")
