"""Reset to actual portfolio - $73.63."""
import json

with open('ta_live_results.json', 'r') as f:
    data = json.load(f)

print(f"OLD: {data.get('wins', 0)}W/{data.get('losses', 0)}L | PnL: ${data.get('total_pnl', 0):.2f} | Bankroll: ${data.get('bankroll', 0):.2f}")

# Reset to actual portfolio from screenshot
data['bankroll'] = 73.63
data['total_pnl'] = 0.0
data['wins'] = 0
data['losses'] = 0
data['consecutive_wins'] = 0
data['consecutive_losses'] = 0

with open('ta_live_results.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"NEW: 0W/0L | PnL: $0.00 | Bankroll: $73.63")
print("HARD CAP: $3 max per trade (no more $10 bets)")
print("DOWN ONLY MODE: ON")
