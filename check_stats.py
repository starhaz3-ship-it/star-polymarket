"""Quick stats check."""
import json
d = json.load(open('ta_live_results.json'))
print(f"W/L: {d.get('wins',0)}/{d.get('losses',0)}")
print(f"PnL: ${d.get('total_pnl',0):.2f}")
print(f"Bankroll: ${d.get('bankroll',0):.2f}")
print(f"Consecutive: {d.get('consecutive_wins',0)}W / {d.get('consecutive_losses',0)}L")
