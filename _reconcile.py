"""
Polymarket CSV vs ALL Bots Reconciliation (Feb 17-19, 2026)
Compares on-chain CSV against: momentum_15m, sniper_5m, pairs_arb_15m, pairs_arb_5m, maker
Baseline: V5.2 deploy Feb 17 ~12:00AM ET (ts > 1771308000). Starting balance: $118.89
"""

import csv
import json
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-19 (1).csv"
BOT_DIR = r"C:\Users\Star\.local\bin\star-polymarket"
BASELINE_TS = 1771308000  # V5.2 deploy Feb 17 ~12:00AM ET
STARTING_BALANCE = 118.89

RESULT_FILES = {
    "momentum_15m": "momentum_15m_results.json",
    "sniper_5m": "sniper_5m_live_results.json",
    "pairs_arb_15m": "pairs_arb_results.json",
    "pairs_arb_5m": "pairs_arb_5m_results.json",
    "maker": "maker_results.json",
}

print("=" * 90)
print("  POLYMARKET FULL RECONCILIATION — CSV vs ALL BOTS")
print("  Baseline: V5.2 deploy Feb 17 ~12:00AM ET | Starting balance: $118.89")
print("=" * 90)
print()

# ── 1. Parse CSV ────────────────────────────────────────────────────────────
csv_trades = []
with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ts = int(row["timestamp"])
        if ts >= BASELINE_TS:
            csv_trades.append({
                "market": row["marketName"],
                "action": row["action"],
                "usdc": float(row["usdcAmount"]),
                "tokens": float(row["tokenAmount"]),
                "token_name": row["tokenName"],
                "timestamp": ts,
                "hash": row["hash"],
                "dt": datetime.fromtimestamp(ts, tz=timezone.utc),
            })

# Sort by timestamp ascending
csv_trades.sort(key=lambda x: x["timestamp"])

# Aggregate
buys = [t for t in csv_trades if t["action"] == "Buy"]
sells = [t for t in csv_trades if t["action"] == "Sell"]
redeems = [t for t in csv_trades if t["action"] == "Redeem"]
rebates = [t for t in csv_trades if t["action"] == "Maker Rebate"]

total_buy = sum(t["usdc"] for t in buys)
total_sell = sum(t["usdc"] for t in sells)
total_redeem = sum(t["usdc"] for t in redeems)
total_rebate = sum(t["usdc"] for t in rebates)
csv_net = total_sell + total_redeem + total_rebate - total_buy

print(f"[CSV] Post-baseline trades: {len(csv_trades)}")
print(f"  Buys:    {len(buys):>4} txns  ${total_buy:>10,.2f}")
print(f"  Sells:   {len(sells):>4} txns  ${total_sell:>10,.2f}")
print(f"  Redeems: {len(redeems):>4} txns  ${total_redeem:>10,.2f}")
print(f"  Rebates: {len(rebates):>4} txns  ${total_rebate:>10,.2f}")
print(f"  NET PnL: ${csv_net:>10,.2f}")
print(f"  Implied balance: ${STARTING_BALANCE + csv_net:.2f}")
print()

# ── 2. Daily breakdown ─────────────────────────────────────────────────────
daily = defaultdict(lambda: {"buys": 0.0, "sells": 0.0, "redeems": 0.0, "rebates": 0.0, "buy_count": 0})
for t in csv_trades:
    day = t["dt"].strftime("%Y-%m-%d")
    if t["action"] == "Buy":
        daily[day]["buys"] += t["usdc"]
        daily[day]["buy_count"] += 1
    elif t["action"] == "Sell":
        daily[day]["sells"] += t["usdc"]
    elif t["action"] == "Redeem":
        daily[day]["redeems"] += t["usdc"]
    elif t["action"] == "Maker Rebate":
        daily[day]["rebates"] += t["usdc"]

print("-" * 90)
print("DAILY BREAKDOWN (UTC)")
print(f"{'Date':<14} {'Buys':>10} {'Sells':>10} {'Redeems':>10} {'Rebates':>10} {'Net PnL':>10} {'#Buys':>6}")
print("-" * 90)
for day in sorted(daily.keys()):
    d = daily[day]
    net = d["sells"] + d["redeems"] + d["rebates"] - d["buys"]
    print(f"{day:<14} ${d['buys']:>9,.2f} ${d['sells']:>9,.2f} ${d['redeems']:>9,.2f} ${d['rebates']:>9,.2f} ${net:>9,.2f} {d['buy_count']:>6}")
print()

# ── 3. Buy size clustering (identify which bot) ────────────────────────────
print("-" * 90)
print("BUY SIZE CLUSTERING (identify bot source)")
print("-" * 90)

def classify_buy(usdc):
    if 2.40 <= usdc <= 2.60:
        return "momentum_15m ($2.50)"
    elif 2.90 <= usdc <= 3.10:
        return "sniper_5m ($3.00)"
    elif 3.40 <= usdc <= 3.60:
        return "pairs_arb ($3.50/leg)"
    elif 4.40 <= usdc <= 5.10:
        return "deep_bid/pairs ($4.50-$5)"
    elif 1.90 <= usdc <= 2.10:
        return "maker ($2/side)"
    elif 0.90 <= usdc <= 1.10:
        return "maker_small ($1/side)"
    else:
        return f"other (${usdc:.2f})"

clusters = defaultdict(lambda: {"count": 0, "total": 0.0})
for t in buys:
    cat = classify_buy(t["usdc"])
    clusters[cat]["count"] += 1
    clusters[cat]["total"] += t["usdc"]

for cat in sorted(clusters.keys(), key=lambda c: -clusters[c]["total"]):
    c = clusters[cat]
    print(f"  {cat:<30} {c['count']:>4} trades  ${c['total']:>10,.2f}")
print()

# ── 4. Load ALL internal bot results ───────────────────────────────────────
print("-" * 90)
print("INTERNAL BOT RESULTS (post-baseline)")
print("-" * 90)

def parse_entry_time(val):
    """Parse entry_time which can be ISO string or Unix float."""
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val)
            return dt.timestamp()
        except:
            return 0
    return 0

bot_data = {}
total_internal_pnl = 0.0

for bot_name, filename in RESULT_FILES.items():
    filepath = os.path.join(BOT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"  {bot_name}: FILE NOT FOUND")
        continue

    with open(filepath) as fh:
        data = json.load(fh)

    resolved = data.get("resolved", [])
    stats = data.get("stats", {})

    # Filter post-baseline
    post = []
    for r in resolved:
        et = r.get("entry_time", 0)
        ts = parse_entry_time(et)
        # For maker, use resolved_at if available
        if bot_name == "maker" and "resolved_at" in r:
            try:
                ts = datetime.fromisoformat(r["resolved_at"]).timestamp()
            except:
                pass
        if ts >= BASELINE_TS:
            r["_parsed_ts"] = ts
            post.append(r)

    pnl = sum(r.get("pnl", 0) for r in post)
    total_internal_pnl += pnl

    wins = sum(1 for r in post if r.get("pnl", 0) > 0)
    losses = sum(1 for r in post if r.get("pnl", 0) < 0)
    zeros = sum(1 for r in post if r.get("pnl", 0) == 0)

    bot_data[bot_name] = {"resolved": post, "stats": stats, "pnl": pnl, "wins": wins, "losses": losses}

    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    print(f"  {bot_name:<18} {len(post):>4} trades  {wins}W/{losses}L{f'/{zeros}Z' if zeros else ''} ({wr:.0f}% WR)  PnL: ${pnl:>8,.2f}")

print(f"\n  TOTAL INTERNAL PnL: ${total_internal_pnl:>10,.2f}")
print(f"  CSV PnL:            ${csv_net:>10,.2f}")
print(f"  GAP:                ${csv_net - total_internal_pnl:>10,.2f}")
print()

# ── 5. Per-market CSV breakdown ─────────────────────────────────────────────
market_csv = defaultdict(lambda: {"buys": 0.0, "sells": 0.0, "redeems": 0.0, "rebates": 0.0,
                                   "buy_count": 0, "buy_trades": []})
for t in csv_trades:
    m = market_csv[t["market"]]
    if t["action"] == "Buy":
        m["buys"] += t["usdc"]
        m["buy_count"] += 1
        m["buy_trades"].append(t)
    elif t["action"] == "Sell":
        m["sells"] += t["usdc"]
    elif t["action"] == "Redeem":
        m["redeems"] += t["usdc"]
    elif t["action"] == "Maker Rebate":
        m["rebates"] += t["usdc"]

# ── 6. Invisible losses (buy with no sell/redeem) ──────────────────────────
print("-" * 90)
print("INVISIBLE LOSSES (Buy with no Sell/Redeem = total loss of position)")
print("-" * 90)

invisible = []
for mname, md in market_csv.items():
    if md["buys"] > 0 and md["sells"] == 0 and md["redeems"] == 0:
        invisible.append((mname, md["buys"], md["buy_count"]))

invisible.sort(key=lambda x: -x[1])
inv_total = sum(x[1] for x in invisible)

if invisible:
    for name, amt, cnt in invisible[:30]:
        short = name[:70] if len(name) > 70 else name
        print(f"  ${amt:>7.2f} ({cnt}B)  {short}")
    if len(invisible) > 30:
        print(f"  ... and {len(invisible) - 30} more")
    print(f"\n  Total invisible losses: ${inv_total:.2f} across {len(invisible)} markets")
else:
    print("  None detected")
print()

# ── 7. Winners (markets with redeems > buys) ───────────────────────────────
print("-" * 90)
print("WINNING MARKETS (Redeems > Buys)")
print("-" * 90)

winners = []
for mname, md in market_csv.items():
    net = md["sells"] + md["redeems"] + md["rebates"] - md["buys"]
    if net > 0:
        winners.append((mname, net, md["buy_count"]))

winners.sort(key=lambda x: -x[1])
win_total = sum(x[1] for x in winners)

for name, net, cnt in winners[:20]:
    short = name[:70] if len(name) > 70 else name
    print(f"  +${net:>7.2f} ({cnt}B)  {short}")
if len(winners) > 20:
    print(f"  ... and {len(winners) - 20} more")
print(f"\n  Total winning PnL: +${win_total:.2f} across {len(winners)} markets")
print()

# ── 8. Biggest losers ──────────────────────────────────────────────────────
print("-" * 90)
print("BIGGEST LOSING MARKETS")
print("-" * 90)

losers = []
for mname, md in market_csv.items():
    net = md["sells"] + md["redeems"] + md["rebates"] - md["buys"]
    if net < 0:
        losers.append((mname, net, md["buy_count"]))

losers.sort(key=lambda x: x[1])
lose_total = sum(x[1] for x in losers)

for name, net, cnt in losers[:20]:
    short = name[:70] if len(name) > 70 else name
    print(f"  ${net:>8.2f} ({cnt}B)  {short}")
print(f"\n  Total losing PnL: ${lose_total:.2f} across {len(losers)} markets")
print()

# ── 9. Final summary ───────────────────────────────────────────────────────
print("=" * 90)
print("  FINAL RECONCILIATION")
print("=" * 90)
print(f"""
  ON-CHAIN (CSV):
    Spent (Buys):       ${total_buy:>10,.2f}  ({len(buys)} txns)
    Received (S+R+Rb):  ${total_sell + total_redeem + total_rebate:>10,.2f}  ({len(sells)}S + {len(redeems)}R + {len(rebates)}Rb)
    CSV Net PnL:        ${csv_net:>10,.2f}
    Implied Balance:    ${STARTING_BALANCE + csv_net:>10,.2f}  (from $118.89)

  INTERNAL BOTS:""")
for bot_name in RESULT_FILES:
    if bot_name in bot_data:
        bd = bot_data[bot_name]
        print(f"    {bot_name:<18} ${bd['pnl']:>10,.2f}  ({bd['wins']}W/{bd['losses']}L)")
print(f"""    {'TOTAL':<18} ${total_internal_pnl:>10,.2f}

  RECONCILIATION:
    CSV PnL:            ${csv_net:>10,.2f}
    Internal PnL:       ${total_internal_pnl:>10,.2f}
    GAP:                ${csv_net - total_internal_pnl:>10,.2f}
    (Gap = untracked trades, fees, old positions resolving, rounding)

  MARKET SUMMARY:
    Total CSV markets:  {len(market_csv)}
    Winning markets:    {len(winners)}  (+${win_total:.2f})
    Losing markets:     {len(losers)}  (${lose_total:.2f})
    Invisible losses:   {len(invisible)} markets (${inv_total:.2f})
""")
print("=" * 90)
