"""Quick live trader status check."""
import json

data = json.load(open("ta_live_results.json"))
trades = data.get("trades", {})
o = sum(1 for t in trades.values() if t.get("status") == "open")
c = sum(1 for t in trades.values() if t.get("status") == "closed")
print("=== LIVE TRADER ===")
print(f"Bankroll: ${data.get('bankroll', 0):.2f}")
print(f"Trades: {len(trades)} (Open: {o}, Closed: {c})")
print(f"PnL: ${data.get('total_pnl', 0):+.2f} | Wins: {data.get('wins', 0)} Losses: {data.get('losses', 0)}")
cl = data.get("asset_consecutive_losses", {})
cd = data.get("asset_cooldown_start", {})
if cl:
    print(f"Asset loss streaks: {cl}")
if cd:
    print(f"Cooldown timers: {cd}")
closed_list = [(k, v) for k, v in trades.items() if v.get("status") == "closed"]
if closed_list:
    closed_list.sort(key=lambda x: x[1].get("exit_time", ""), reverse=True)
    print(f"\nRecent closed:")
    for k, v in closed_list[:5]:
        result = "WIN" if v.get("pnl", 0) > 0 else "LOSS"
        print(f"  [{result}] {v.get('side','?')} ${v.get('pnl',0):+.2f} | {v.get('market_title','')[:50]}")
open_t = [(k, v) for k, v in trades.items() if v.get("status") == "open"]
if open_t:
    print(f"\nOpen trades:")
    for k, v in open_t:
        print(f"  {v.get('side','?')} @ ${v.get('entry_price',0):.4f} | {v.get('market_title','')[:50]}")
else:
    print("\nNo open trades")
print(f"\nSignals seen: {data.get('signals_count', 0)}")
# Shadow stats
ss = data.get("shadow_stats", {})
if ss:
    st = ss.get("total", 0)
    sw = ss.get("wins", 0)
    swr = sw / max(1, st) * 100
    print(f"Shadow trades: {st} ({sw}W, {swr:.0f}%WR, ${ss.get('pnl', 0):+.2f})")
