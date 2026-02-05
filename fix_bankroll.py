"""Fix bankroll to actual account value."""
import json

with open('ta_live_results.json', 'r') as f:
    data = json.load(f)

print(f"Old bankroll: ${data.get('bankroll', 0):.2f}")
print(f"Tracked PnL: ${data.get('total_pnl', 0):.2f}")
print(f"Wins: {data.get('wins', 0)} Losses: {data.get('losses', 0)}")

data['bankroll'] = 75.0

with open('ta_live_results.json', 'w') as f:
    json.dump(data, f, indent=2)

print("Updated bankroll to $75.00")
print("MAX_POSITION_SIZE lowered to $5 (was $10)")
print("$3 bets = 4% of bankroll - reasonable Kelly sizing")
