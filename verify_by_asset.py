"""Break down on-chain verified PnL by asset (BTC vs ETH)."""
import json
from collections import defaultdict

# Load internal trades
with open("momentum_15m_results.json") as f:
    data = json.load(f)
resolved = data["resolved"]

# Load on-chain activity
with open("onchain_activity.json") as f:
    activity = json.load(f)

# Group on-chain by conditionId
by_cond = defaultdict(list)
for a in activity:
    by_cond[a["conditionId"]].append(a)

# Compute on-chain PnL per condition
onchain = {}
for cid, acts in by_cond.items():
    buys = [a for a in acts if a["type"] == "TRADE"]
    reds = [a for a in acts if a["type"] == "REDEEM"]
    total_spent = sum(a["usdcSize"] for a in buys)
    total_redeemed = sum(a["usdcSize"] for a in reds)
    title = acts[0].get("title", "?")
    outcome = buys[0]["outcome"] if buys else "?"
    onchain[cid] = {
        "title": title,
        "outcome": outcome,
        "spent": total_spent,
        "redeemed": total_redeemed,
        "pnl": total_redeemed - total_spent,
        "status": "REDEEMED" if reds else "OPEN",
    }

# Match internal trades to on-chain, group by asset
btc = {"wins": 0, "losses": 0, "pnl": 0.0, "phantom_w": 0, "phantom_l": 0, "phantom_pnl": 0.0, "trades": []}
eth = {"wins": 0, "losses": 0, "pnl": 0.0, "phantom_w": 0, "phantom_l": 0, "phantom_pnl": 0.0, "trades": []}
other = {"wins": 0, "losses": 0, "pnl": 0.0, "phantom_w": 0, "phantom_l": 0, "phantom_pnl": 0.0, "trades": []}

for t in resolved:
    cid = t.get("condition_id", "")
    title = t.get("title", "")
    side = t.get("side", "")
    internal_pnl = t.get("pnl", 0)
    size = t.get("size_usd", 0)
    strategy = t.get("strategy", "")

    # Determine asset
    if "Bitcoin" in title or "BTC" in title:
        bucket = btc
    elif "Ethereum" in title or "ETH" in title:
        bucket = eth
    else:
        bucket = other

    oc = onchain.get(cid)
    if oc and oc["spent"] > 0:
        real_pnl = oc["pnl"]
        bucket["pnl"] += real_pnl
        if real_pnl > 0:
            bucket["wins"] += 1
        else:
            bucket["losses"] += 1
        bucket["trades"].append({
            "title": title[:50],
            "side": side,
            "spent": oc["spent"],
            "redeemed": oc["redeemed"],
            "pnl": real_pnl,
            "status": oc["status"],
            "strategy": strategy,
        })
    else:
        # Phantom
        if internal_pnl > 0:
            bucket["phantom_w"] += 1
        else:
            bucket["phantom_l"] += 1
        bucket["phantom_pnl"] += internal_pnl

for name, b in [("BTC", btc), ("ETH", eth), ("OTHER", other)]:
    total = b["wins"] + b["losses"]
    if total == 0:
        continue
    wr = b["wins"] / total * 100 if total > 0 else 0
    print(f"=== {name} (on-chain verified) ===")
    print(f"  Record: {b['wins']}W/{b['losses']}L ({wr:.0f}% WR)")
    print(f"  PnL:    ${b['pnl']:+.2f}")
    print(f"  Phantom: {b['phantom_w']}W/{b['phantom_l']}L (fake PnL: ${b['phantom_pnl']:+.2f})")
    print()

    # Show each trade
    for tr in b["trades"]:
        result = "W" if tr["pnl"] > 0 else "L"
        print(f"    {result} {tr['side']:4} ${tr['pnl']:>+7.2f} spent=${tr['spent']:>6.2f} {tr['status']:8} | {tr['title']}")
    print()
