"""Deep analysis of whale trading signal — are they hedging or directional?"""
import json
from collections import defaultdict
from datetime import datetime, timezone

for label, path in [("WHALE-1 (0xe1DF)", "whale1_activity.json"), ("WHALE-2 (0x8151)", "whale2_activity.json")]:
    with open(path) as f:
        activity = json.load(f)

    trades = [a for a in activity if a["type"] == "TRADE"]

    print(f"\n{'='*70}")
    print(f"{label} — TRADE-LEVEL DETAIL")
    print(f"{'='*70}")

    # Group trades by conditionId to see if they buy BOTH sides
    by_cond = defaultdict(list)
    for t in trades:
        by_cond[t["conditionId"]].append(t)

    for cid, tlist in sorted(by_cond.items(), key=lambda x: max(t["timestamp"] for t in x[1]), reverse=True)[:15]:
        title = tlist[0].get("title", "?")[:55]
        sides = defaultdict(lambda: {"shares": 0, "cost": 0.0, "count": 0, "prices": []})
        for t in tlist:
            outcome = t.get("outcome", "?")
            sides[outcome]["shares"] += t["size"]
            sides[outcome]["cost"] += t["usdcSize"]
            sides[outcome]["count"] += 1
            sides[outcome]["prices"].append(t.get("price", 0))

        # Entry timing relative to market close
        ts = max(t["timestamp"] for t in tlist)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)

        side_strs = []
        for outcome, data in sorted(sides.items()):
            avg_p = data["cost"] / data["shares"] if data["shares"] > 0 else 0
            prices = [f"{p:.3f}" for p in data["prices"][:3]]
            side_strs.append(f"{outcome}: {data['shares']:.0f}sh ${data['cost']:.2f} avg={avg_p:.3f} ({data['count']} fills)")

        both_sides = len(sides) > 1
        marker = " ** BOTH SIDES **" if both_sides else ""
        combined_cost = sum(s["cost"] for s in sides.values())

        print(f"\n  {dt.strftime('%H:%M UTC')} | {title}{marker}")
        for s in side_strs:
            print(f"    {s}")
        if both_sides:
            total_shares_up = sides.get("Up", {"shares": 0})["shares"]
            total_shares_dn = sides.get("Down", {"shares": 0})["shares"]
            cost_up = sides.get("Up", {"cost": 0})["cost"]
            cost_dn = sides.get("Down", {"cost": 0})["cost"]
            hedged = min(total_shares_up, total_shares_dn)
            print(f"    Combined cost: ${combined_cost:.2f} | Hedged: {hedged:.0f}sh | Unhedged: {abs(total_shares_up-total_shares_dn):.0f}sh")
            if hedged > 0:
                cost_per_hedged_pair = combined_cost / hedged
                print(f"    Cost per hedged pair: ${cost_per_hedged_pair:.3f} (profit if <$1.00)")
