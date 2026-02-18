"""Analyze fill timing for pairs arb — when do both sides fill?"""
import json
from collections import defaultdict

for label, path in [
    ("WHALE-1 (0xe1DF)", "whale1_activity.json"),
    ("WHALE-2 (0x8151)", "whale2_activity.json"),
]:
    with open(path) as f:
        activity = json.load(f)

    trades = [a for a in activity if a["type"] == "TRADE"]

    by_cond = defaultdict(list)
    for t in trades:
        by_cond[t["conditionId"]].append(t)

    print(f"\n{'='*70}")
    print(f"{label} — FILL TIMING ANALYSIS")
    print(f"{'='*70}")

    # For each market, track when each fill arrives
    hedged_fill_times = []  # seconds from first fill to second side's first fill
    all_fill_gaps = []  # gap from first fill to EACH subsequent fill

    for cid, tlist in by_cond.items():
        up_trades = sorted([t for t in tlist if t.get("outcome") == "Up"], key=lambda t: t["timestamp"])
        dn_trades = sorted([t for t in tlist if t.get("outcome") == "Down"], key=lambda t: t["timestamp"])

        if not up_trades or not dn_trades:
            continue  # One side missing

        first_fill = min(tlist[0]["timestamp"] for tlist in [up_trades, dn_trades])
        up_first = up_trades[0]["timestamp"]
        dn_first = dn_trades[0]["timestamp"]
        second_side_first = max(up_first, dn_first)
        gap = second_side_first - first_fill

        hedged_fill_times.append(gap)

        # Track every individual fill time from market entry
        for t in sorted(tlist, key=lambda t: t["timestamp"]):
            all_fill_gaps.append(t["timestamp"] - first_fill)

    print(f"\n--- TIME TO GET BOTH SIDES (seconds from 1st fill to 2nd side's 1st fill) ---")
    if hedged_fill_times:
        hedged_fill_times.sort()
        print(f"  Count: {len(hedged_fill_times)} hedged markets")
        print(f"  Min:    {min(hedged_fill_times):>5}s")
        print(f"  25th %: {hedged_fill_times[len(hedged_fill_times)//4]:>5}s")
        print(f"  Median: {hedged_fill_times[len(hedged_fill_times)//2]:>5}s")
        print(f"  75th %: {hedged_fill_times[3*len(hedged_fill_times)//4]:>5}s")
        print(f"  Max:    {max(hedged_fill_times):>5}s")
        print(f"  Avg:    {sum(hedged_fill_times)/len(hedged_fill_times):>5.0f}s ({sum(hedged_fill_times)/len(hedged_fill_times)/60:.1f}min)")

        # Cumulative fill probability
        print(f"\n--- CUMULATIVE FILL PROBABILITY ---")
        for cutoff in [30, 60, 90, 120, 180, 240, 300, 420, 600, 900]:
            filled = sum(1 for t in hedged_fill_times if t <= cutoff)
            pct = filled / len(hedged_fill_times) * 100
            print(f"  Within {cutoff:>3}s ({cutoff/60:>5.1f}min): {filled}/{len(hedged_fill_times)} = {pct:.0f}%")

        # What if we bail at X seconds?
        print(f"\n--- BAIL-OUT ANALYSIS ---")
        print(f"  If we set a deadline and market-sell the exposed side if 2nd side doesn't fill:")
        print(f"  (Assumes market-sell loses ~5c/share = $0.40 exit on $0.45 entry)")
        for bail_s in [60, 120, 180, 240, 300]:
            filled_in_time = sum(1 for t in hedged_fill_times if t <= bail_s)
            missed = len(hedged_fill_times) + (len([c for c in by_cond.values() if len(set(t.get("outcome") for t in c)) == 1]) ) - filled_in_time
            # Actually just count: how many would we catch vs miss
            caught = sum(1 for t in hedged_fill_times if t <= bail_s)
            late = sum(1 for t in hedged_fill_times if t > bail_s)
            total_markets = len(hedged_fill_times) + len([1 for c in by_cond.values() if len(set(t.get("outcome") for t in c)) == 1])

            # Get one-sided markets (never filled second side)
            one_sided = len([1 for c in by_cond.values() if len(set(t.get("outcome") for t in c)) == 1])

            # Profit from caught: $0.10 per pair * shares
            # Loss from bailing: ~$0.05 per share (sell at $0.40 what we bought at $0.45)
            # Loss from late fills we would have bailed on: same $0.05
            arb_profit_per = 0.10  # guaranteed per hedged pair
            bail_loss_per = 0.05   # estimated loss from market-selling exposed side

            hedged_profit = caught * arb_profit_per
            bail_cost = (late + one_sided) * bail_loss_per
            net = hedged_profit - bail_cost

            print(f"  Bail at {bail_s:>3}s ({bail_s/60:.0f}min): {caught} hedged (+${hedged_profit:.2f}) | {late} late + {one_sided} missed = {late+one_sided} bails (-${bail_cost:.2f}) | NET: ${net:+.2f} per share")

    # Individual fill timeline for last 5 markets
    print(f"\n--- FILL TIMELINE (last 5 hedged markets) ---")
    count = 0
    for cid, tlist in sorted(by_cond.items(), key=lambda x: max(t["timestamp"] for t in x[1]), reverse=True):
        up_trades = [t for t in tlist if t.get("outcome") == "Up"]
        dn_trades = [t for t in tlist if t.get("outcome") == "Down"]
        if not up_trades or not dn_trades:
            continue

        title = tlist[0].get("title", "?")[:50]
        first_ts = min(t["timestamp"] for t in tlist)
        print(f"\n  {title}")
        for t in sorted(tlist, key=lambda t: t["timestamp"]):
            elapsed = t["timestamp"] - first_ts
            print(f"    +{elapsed:>4}s  {t['outcome']:4} {t['size']:>6.0f}sh @ ${t['price']:.3f} = ${t['usdcSize']:.2f}")

        count += 1
        if count >= 5:
            break
