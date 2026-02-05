"""Reset to actual portfolio value."""
import json

with open('ta_live_results.json', 'r') as f:
    data = json.load(f)

print(f"OLD: {data.get('wins', 0)}W/{data.get('losses', 0)}L | PnL: ${data.get('total_pnl', 0):.2f} | Bankroll: ${data.get('bankroll', 0):.2f}")

# Reset to actual portfolio value from screenshot
data['bankroll'] = 60.65  # Actual portfolio value
data['total_pnl'] = 0.0
data['wins'] = 0
data['losses'] = 0
data['consecutive_wins'] = 0
data['consecutive_losses'] = 0

with open('ta_live_results.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"NEW: 0W/0L | PnL: $0.00 | Bankroll: $60.65")
print("DOWN ONLY MODE ENABLED - No more UP trades until profitable")
