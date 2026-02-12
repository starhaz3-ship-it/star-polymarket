import json
from pathlib import Path
p = Path(__file__).parent / "maker_results.json"
if p.exists():
    d = json.loads(p.read_text())
    print(f"Active: {len(d.get('active_positions', []))}")
    print(f"Resolved: {len(d.get('resolved', []))}")
    stats = d.get("stats", {})
    print(f"Pairs attempted: {stats.get('pairs_attempted', 0)}")
    print(f"Pairs completed: {stats.get('pairs_completed', 0)}")
    print(f"Pairs partial: {stats.get('pairs_partial', 0)}")
    print(f"Fills UP: {stats.get('fills_up', 0)}")
    print(f"Fills DN: {stats.get('fills_down', 0)}")
    print(f"Total PnL: ${stats.get('total_pnl', 0):.4f}")
    for pos in d.get("active_positions", []):
        q = pos.get("question", "")[:50]
        print(f"  {pos.get('asset')} | {pos.get('status')} | up={pos.get('up_filled')} dn={pos.get('down_filled')} | {q}")
    for r in d.get("resolved", []):
        print(f"  RESOLVED: {r.get('asset')} | {r.get('outcome')} | paired={r.get('paired')} | PnL=${r.get('pnl', 0):.4f}")
else:
    print("No results file")
