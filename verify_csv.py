"""Reconcile internal trades against CSV by market name + side."""
import json, csv
from collections import defaultdict

# Load internal trades
with open("momentum_15m_results.json") as f:
    data = json.load(f)
resolved = data["resolved"]

# Load CSV into lookup by market name
csv_by_market = defaultdict(list)
with open("C:/Users/Star/Downloads/Polymarket-History-2026-02-17 (3).csv", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        csv_by_market[row["marketName"].strip()].append(row)

print(f"CSV: {sum(len(v) for v in csv_by_market.values())} rows, {len(csv_by_market)} unique markets")
print(f"Internal: {len(resolved)} resolved trades")
print()

# Match by market name
matched = 0
phantom = 0
phantom_pnl = 0.0
matched_pnl = 0.0
phantom_list = []

for t in resolved:
    title = t.get("title", "").strip()
    side = t.get("side", "")
    pnl = t.get("pnl", 0)
    size = t.get("size_usd", 0)
    strategy = t.get("strategy", "")

    csv_rows = csv_by_market.get(title, [])
    # Look for a Buy on the same side (tokenName matches side)
    found = False
    for cr in csv_rows:
        if cr["action"] == "Buy":
            token = cr["tokenName"].strip()
            usdc = float(cr["usdcAmount"])
            # Token should match: "Up" matches side "UP", "Down" matches "DOWN"
            if token.upper() == side.upper() and abs(usdc - size) < 2.0:
                found = True
                break
            # Also try without amount matching
            if token.upper() == side.upper():
                found = True
                break

    if found:
        matched += 1
        matched_pnl += pnl
    else:
        phantom += 1
        phantom_pnl += pnl
        result = "WIN" if pnl > 0 else "LOSS"
        has_csv = "HAS CSV ROWS" if csv_rows else "NO CSV ROWS"
        csv_actions = ", ".join(f"{r['action']} {r['tokenName']} ${r['usdcAmount']}" for r in csv_rows[:3])
        phantom_list.append((result, pnl, side, strategy, title[:55], has_csv, csv_actions))

# Print phantoms
for result, pnl, side, strategy, title, has_csv, csv_actions in phantom_list:
    print(f"  PHANTOM {result} ${pnl:>+7.2f} {side:4} {strategy:18} | {has_csv}")
    if csv_actions:
        print(f"    CSV rows: {csv_actions}")

print()
matched_wins = sum(1 for t in resolved if t.get("pnl", 0) > 0) - sum(1 for r, p, *_ in phantom_list if r == "WIN")
matched_losses = sum(1 for t in resolved if t.get("pnl", 0) <= 0) - sum(1 for r, p, *_ in phantom_list if r == "LOSS")
print(f"Matched: {matched} ({matched_wins}W/{matched_losses}L, PnL: ${matched_pnl:+.2f})")
print(f"Phantom: {phantom} (PnL: ${phantom_pnl:+.2f})")
print(f"Total internal PnL: ${matched_pnl + phantom_pnl:+.2f}")
