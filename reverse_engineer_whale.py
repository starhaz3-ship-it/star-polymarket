"""Reverse-engineer the pairs arb whale strategy."""
import json
from collections import defaultdict
from datetime import datetime, timezone

for label, path, size_label in [
    ("WHALE-1 (0xe1DF) — $9/trade", "whale1_activity.json", "small"),
    ("WHALE-2 (0x8151) — $50-100/trade", "whale2_activity.json", "big"),
]:
    with open(path) as f:
        activity = json.load(f)

    trades = [a for a in activity if a["type"] == "TRADE"]
    redeems = [a for a in activity if a["type"] == "REDEEM"]

    print(f"\n{'='*80}")
    print(f"REVERSE ENGINEERING: {label}")
    print(f"{'='*80}")

    # Group trades by conditionId
    by_cond = defaultdict(list)
    for t in trades:
        by_cond[t["conditionId"]].append(t)

    # Analyze each market
    markets = []
    for cid, tlist in by_cond.items():
        title = tlist[0].get("title", "?")

        up_trades = [t for t in tlist if t.get("outcome") == "Up"]
        dn_trades = [t for t in tlist if t.get("outcome") == "Down"]

        up_shares = sum(t["size"] for t in up_trades)
        dn_shares = sum(t["size"] for t in dn_trades)
        up_cost = sum(t["usdcSize"] for t in up_trades)
        dn_cost = sum(t["usdcSize"] for t in dn_trades)

        up_fills = len(up_trades)
        dn_fills = len(dn_trades)

        up_prices = [t["price"] for t in up_trades]
        dn_prices = [t["price"] for t in dn_trades]

        # Timestamps
        all_ts = [t["timestamp"] for t in tlist]
        first_ts = min(all_ts)
        last_ts = max(all_ts)
        duration_s = last_ts - first_ts

        # Which side was bought first?
        up_first_ts = min(t["timestamp"] for t in up_trades) if up_trades else 999999999999
        dn_first_ts = min(t["timestamp"] for t in dn_trades) if dn_trades else 999999999999
        first_side = "UP" if up_first_ts < dn_first_ts else "DOWN" if dn_first_ts < up_first_ts else "SAME"
        time_between_sides = abs(up_first_ts - dn_first_ts)

        # Hedged?
        hedged_shares = min(up_shares, dn_shares)
        unhedged_shares = abs(up_shares - dn_shares)
        combined_cost = up_cost + dn_cost
        both_sides = up_shares > 0 and dn_shares > 0

        # Redeems for this market
        market_redeems = [a for a in redeems if a["conditionId"] == cid]
        redeemed = sum(a["usdcSize"] for a in market_redeems)
        pnl = redeemed - combined_cost

        # Parse market close time from title to get entry timing
        entry_dt = datetime.fromtimestamp(first_ts, tz=timezone.utc)

        markets.append({
            "title": title[:60],
            "both_sides": both_sides,
            "up_shares": up_shares,
            "dn_shares": dn_shares,
            "up_cost": up_cost,
            "dn_cost": dn_cost,
            "up_fills": up_fills,
            "dn_fills": dn_fills,
            "up_prices": up_prices,
            "dn_prices": dn_prices,
            "hedged": hedged_shares,
            "unhedged": unhedged_shares,
            "combined_cost": combined_cost,
            "cost_per_pair": combined_cost / hedged_shares if hedged_shares > 0 else 0,
            "first_side": first_side,
            "time_between_sides_s": time_between_sides,
            "duration_s": duration_s,
            "first_ts": first_ts,
            "entry_time": entry_dt.strftime("%H:%M:%S UTC"),
            "pnl": pnl,
            "redeemed": redeemed,
            "total_fills": up_fills + dn_fills,
        })

    markets.sort(key=lambda m: m["first_ts"], reverse=True)

    # === TIMING ANALYSIS ===
    print(f"\n--- TIMING ---")
    first_sides = [m["first_side"] for m in markets if m["both_sides"]]
    up_first = first_sides.count("UP")
    dn_first = first_sides.count("DOWN")
    same = first_sides.count("SAME")
    print(f"  First side bought: UP first={up_first}, DOWN first={dn_first}, simultaneous={same}")

    gaps = [m["time_between_sides_s"] for m in markets if m["both_sides"]]
    if gaps:
        print(f"  Time between UP/DOWN fills: avg={sum(gaps)/len(gaps):.1f}s, min={min(gaps)}s, max={max(gaps)}s")

    durations = [m["duration_s"] for m in markets]
    print(f"  Total fill duration: avg={sum(durations)/len(durations):.1f}s, min={min(durations)}s, max={max(durations)}s")

    # === FILL ANALYSIS ===
    print(f"\n--- FILL PATTERNS ---")
    both = [m for m in markets if m["both_sides"]]
    one_side = [m for m in markets if not m["both_sides"]]
    print(f"  Both sides filled: {len(both)} markets ({len(both)/len(markets)*100:.0f}%)")
    print(f"  One side only: {len(one_side)} markets ({len(one_side)/len(markets)*100:.0f}%)")

    if one_side:
        for m in one_side:
            side = "UP only" if m["up_shares"] > 0 else "DOWN only"
            print(f"    {side} ${m['combined_cost']:.2f} pnl=${m['pnl']:+.2f} | {m['title']}")

    total_fills_list = [m["total_fills"] for m in both]
    if total_fills_list:
        print(f"  Fills per market (hedged): avg={sum(total_fills_list)/len(total_fills_list):.1f}, min={min(total_fills_list)}, max={max(total_fills_list)}")

    # === PRICING ANALYSIS ===
    print(f"\n--- PRICING ---")
    all_up_prices = [p for m in markets for p in m["up_prices"]]
    all_dn_prices = [p for m in markets for p in m["dn_prices"]]
    if all_up_prices:
        print(f"  UP prices: avg={sum(all_up_prices)/len(all_up_prices):.4f}, min={min(all_up_prices):.4f}, max={max(all_up_prices):.4f}")
    if all_dn_prices:
        print(f"  DOWN prices: avg={sum(all_dn_prices)/len(all_dn_prices):.4f}, min={min(all_dn_prices):.4f}, max={max(all_dn_prices):.4f}")

    costs_per_pair = [m["cost_per_pair"] for m in both if m["cost_per_pair"] > 0]
    if costs_per_pair:
        print(f"  Cost per hedged pair: avg=${sum(costs_per_pair)/len(costs_per_pair):.4f}, min=${min(costs_per_pair):.4f}, max=${max(costs_per_pair):.4f}")
        arb_profit = [1.0 - c for c in costs_per_pair]
        print(f"  Arb profit per pair: avg=${sum(arb_profit)/len(arb_profit):.4f} ({sum(arb_profit)/len(arb_profit)/sum(costs_per_pair)*len(costs_per_pair)*100:.1f}% ROI)")

    # === P&L ANALYSIS ===
    print(f"\n--- P&L ---")
    hedged_pnl = sum(m["pnl"] for m in both if m["redeemed"] > 0)
    exposed_pnl = sum(m["pnl"] for m in one_side if m["redeemed"] > 0)
    open_exposure = sum(m["combined_cost"] for m in markets if m["redeemed"] == 0)
    print(f"  Hedged trades PnL: ${hedged_pnl:+.2f} (guaranteed profit when both sides fill)")
    print(f"  Exposed trades PnL: ${exposed_pnl:+.2f} (one side missed)")
    print(f"  Open/unredeemed exposure: ${open_exposure:.2f}")
    total_invested = sum(m["combined_cost"] for m in markets)
    total_returned = sum(m["redeemed"] for m in markets)
    print(f"  Total invested: ${total_invested:.2f}")
    print(f"  Total returned: ${total_returned:.2f}")
    print(f"  Net PnL: ${total_returned - total_invested:+.2f}")

    # === ENTRY TIMING (minutes before close) ===
    print(f"\n--- ENTRY TIMING (when do they enter?) ---")
    for m in markets[:10]:
        hedged_str = "HEDGED" if m["both_sides"] else "EXPOSED"
        fills = f"fills: UP={m['up_fills']} DN={m['dn_fills']}"
        gap = f"gap={m['time_between_sides_s']}s" if m["both_sides"] else ""
        pnl_str = f"pnl=${m['pnl']:+.2f}" if m["redeemed"] > 0 else "OPEN"
        print(f"  {m['entry_time']} | {hedged_str:7} {m['first_side']:4}first | {fills} {gap} | {pnl_str} | {m['title'][:45]}")

    # === FREQUENCY ===
    print(f"\n--- FREQUENCY ---")
    print(f"  Total markets traded: {len(markets)}")
    if len(markets) >= 2:
        ts_sorted = sorted(m["first_ts"] for m in markets)
        gaps_between = [ts_sorted[i+1] - ts_sorted[i] for i in range(len(ts_sorted)-1)]
        avg_gap = sum(gaps_between) / len(gaps_between)
        print(f"  Avg time between markets: {avg_gap:.0f}s ({avg_gap/60:.1f}min)")
        print(f"  Trading every 15-min window: {'YES' if avg_gap < 1000 else 'NO'}")
