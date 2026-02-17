"""Reconcile internal trades against Polymarket data-api (on-chain truth)."""
import json, urllib.request
from collections import defaultdict

PROXY = "0x4f3456c5b05b14D8aFc40E0299d6d68DE0fF7d7e"

# Load internal trades
with open("momentum_15m_results.json") as f:
    data = json.load(f)
resolved = data["resolved"]
active = data.get("active", {})

# Load on-chain activity (fetched via curl)
with open("onchain_activity.json") as f:
    activity = json.load(f)

# Group by conditionId
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

# Match internal trades to on-chain
print("=" * 80)
print("ON-CHAIN RECONCILIATION vs INTERNAL TRADES")
print("=" * 80)
print()

matched = 0
phantom = 0
matched_wins = 0
matched_losses = 0
matched_pnl = 0.0
phantom_pnl = 0.0
phantom_list = []

for t in resolved:
    cid = t.get("condition_id", "")
    title = t.get("title", "")[:55]
    side = t.get("side", "")
    internal_pnl = t.get("pnl", 0)
    size = t.get("size_usd", 0)

    oc = onchain.get(cid)
    if oc and oc["spent"] > 0:
        matched += 1
        # Use on-chain PnL
        real_pnl = oc["pnl"]
        matched_pnl += real_pnl
        if real_pnl > 0:
            matched_wins += 1
        else:
            matched_losses += 1
        # Flag discrepancies
        if abs(internal_pnl - real_pnl) > 1.0:
            print(f"  MISMATCH {side:4} internal=${internal_pnl:>+7.2f} onchain=${real_pnl:>+7.2f} | {title}")
    else:
        phantom += 1
        phantom_pnl += internal_pnl
        result = "WIN" if internal_pnl > 0 else "LOSS"
        phantom_list.append((result, internal_pnl, side, title))

if phantom_list:
    print()
    print(f"--- PHANTOM TRADES (no on-chain activity) ---")
    for result, pnl, side, title in phantom_list:
        print(f"  PHANTOM {result} ${pnl:>+7.2f} {side:4} | {title}")

print()
print(f"--- SUMMARY ---")
print(f"On-chain matched: {matched} ({matched_wins}W/{matched_losses}L)")
print(f"On-chain PnL:     ${matched_pnl:+.2f}")
print(f"Phantom trades:   {phantom} (fake PnL: ${phantom_pnl:+.2f})")
print(f"Internal claims:  ${matched_pnl + phantom_pnl:+.2f}")
print(f"REAL PnL:         ${matched_pnl:+.2f}")
print()

# Also show positions that are OPEN (unredeemed losses)
open_positions = {cid: oc for cid, oc in onchain.items() if oc["status"] == "OPEN" and oc["spent"] > 0}
if open_positions:
    print(f"--- OPEN POSITIONS (unredeemed, need redeem) ---")
    open_total = 0.0
    for cid, oc in sorted(open_positions.items(), key=lambda x: x[1]["spent"], reverse=True):
        print(f"  {oc['outcome']:5} spent=${oc['spent']:>7.2f} | {oc['title'][:60]}")
        open_total += oc["spent"]
    print(f"  Total locked in open positions: ${open_total:.2f}")

# Count all on-chain trades (not just momentum bot)
all_trades = {cid: oc for cid, oc in onchain.items() if oc["spent"] > 0}
total_onchain_spent = sum(oc["spent"] for oc in all_trades.values())
total_onchain_redeemed = sum(oc["redeemed"] for oc in all_trades.values())
total_onchain_pnl = total_onchain_redeemed - total_onchain_spent
print()
print(f"--- FULL ACCOUNT (all on-chain activity, {len(all_trades)} positions) ---")
print(f"Total spent:    ${total_onchain_spent:.2f}")
print(f"Total redeemed: ${total_onchain_redeemed:.2f}")
print(f"Net PnL:        ${total_onchain_pnl:+.2f}")
