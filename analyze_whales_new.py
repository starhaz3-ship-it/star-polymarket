"""Analyze two new whale wallets for trading patterns."""
import json
from collections import defaultdict
from datetime import datetime, timezone

for label, path in [("WHALE-1 (0xe1DF)", "whale1_activity.json"), ("WHALE-2 (0x8151)", "whale2_activity.json")]:
    with open(path) as f:
        activity = json.load(f)

    if not activity:
        print(f"\n=== {label}: NO ACTIVITY ===")
        continue

    trades = [a for a in activity if a["type"] == "TRADE"]
    redeems = [a for a in activity if a["type"] == "REDEEM"]

    # Group by conditionId
    by_cond = defaultdict(list)
    for a in activity:
        by_cond[a["conditionId"]].append(a)

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"Activity entries: {len(activity)} | Trades: {len(trades)} | Redeems: {len(redeems)}")

    # Per-position breakdown
    wins = 0
    losses = 0
    total_pnl = 0.0
    total_spent = 0.0

    positions = []
    for cid, acts in sorted(by_cond.items(), key=lambda x: max(a["timestamp"] for a in x[1]), reverse=True):
        buys = [a for a in acts if a["type"] == "TRADE"]
        reds = [a for a in acts if a["type"] == "REDEEM"]
        spent = sum(a["usdcSize"] for a in buys)
        redeemed = sum(a["usdcSize"] for a in reds)
        pnl = redeemed - spent
        title = acts[0].get("title", "?")[:55]
        outcome = buys[0]["outcome"] if buys else "?"
        side = buys[0].get("side", "?") if buys else "?"
        avg_price = spent / sum(a["size"] for a in buys) if buys and sum(a["size"] for a in buys) > 0 else 0
        ts = max(a["timestamp"] for a in acts)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)

        status = "REDEEMED" if reds else "OPEN"
        if spent > 0:
            total_spent += spent
            total_pnl += pnl
            if status == "REDEEMED":
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

            positions.append({
                "title": title,
                "outcome": outcome,
                "spent": spent,
                "redeemed": redeemed,
                "pnl": pnl,
                "status": status,
                "avg_price": avg_price,
                "time": dt.strftime("%m/%d %H:%M UTC"),
            })

    print(f"Resolved: {wins}W/{losses}L ({wins/(wins+losses)*100:.0f}% WR)" if wins+losses > 0 else "No resolved trades")
    print(f"Total spent: ${total_spent:.2f} | PnL: ${total_pnl:+.2f}")
    print()

    for p in positions[:20]:
        result = "W" if p["pnl"] > 0 else "L" if p["pnl"] < 0 else "?"
        print(f"  {p['status']:8} {result} {p['outcome']:5} ${p['pnl']:>+8.2f} spent=${p['spent']:>8.2f} avg=${p['avg_price']:.3f} {p['time']} | {p['title']}")
