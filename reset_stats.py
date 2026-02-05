"""Reset live trading stats to zero with $78 bankroll."""
import json

with open('ta_live_results.json', 'r') as f:
    data = json.load(f)

print(f"OLD: {data.get('wins', 0)}W/{data.get('losses', 0)}L | PnL: ${data.get('total_pnl', 0):.2f} | Bankroll: ${data.get('bankroll', 0):.2f}")

# Keep trades history for ML learning but reset counters
data['bankroll'] = 78.0
data['total_pnl'] = 0.0
data['wins'] = 0
data['losses'] = 0
data['consecutive_wins'] = 0
data['consecutive_losses'] = 0
data['signals_count'] = 0
data['ml_rejections'] = 0

with open('ta_live_results.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"NEW: 0W/0L | PnL: $0.00 | Bankroll: $78.00")
print("Stats reset. ML feature weights preserved for learning.")
