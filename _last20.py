import json, csv
from datetime import datetime, timezone, timedelta
from collections import defaultdict

CSV_PATH = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-19 (1).csv"

all_trades = []
with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_trades.append({
            "market": row["marketName"],
            "action": row["action"],
            "usdc": float(row["usdcAmount"]),
            "tokens": float(row["tokenAmount"]),
            "token_name": row["tokenName"],
            "ts": int(row["timestamp"]),
        })

all_trades.sort(key=lambda x: x["ts"], reverse=True)

# Build market-level outcomes
market_recv = defaultdict(float)
for t in all_trades:
    if t["action"] in ("Redeem", "Sell"):
        market_recv[t["market"]] += t["usdc"]

# Get last 20 buys
buys = [t for t in all_trades if t["action"] == "Buy"][:20]

# Group by market
seen = []
market_data = defaultdict(lambda: {"cost": 0.0, "sides": set(), "first_ts": 0, "count": 0})
for b in buys:
    m = b["market"]
    if m not in [s[0] for s in seen]:
        seen.append((m, b["ts"]))
    market_data[m]["cost"] += b["usdc"]
    market_data[m]["sides"].add(b["token_name"])
    market_data[m]["count"] += 1
    if market_data[m]["first_ts"] == 0:
        market_data[m]["first_ts"] = b["ts"]

print("=" * 100)
print("  LAST 20 ON-CHAIN BUYS - GROUPED BY MARKET")
print("=" * 100)

total_pnl = 0.0
w = 0
l = 0
total_cost = 0.0
total_recv_all = 0.0

for mkt, ts in seen:
    md = market_data[mkt]
    recv = market_recv.get(mkt, 0.0)
    pnl = recv - md["cost"]
    total_pnl += pnl
    total_cost += md["cost"]
    total_recv_all += recv

    if pnl > 0:
        tag = "WIN"
        w += 1
    else:
        tag = "LOSS"
        l += 1

    mst_dt = datetime.fromtimestamp(ts, tz=timezone.utc) - timedelta(hours=7)
    time_str = mst_dt.strftime("%m/%d %I:%M%p")
    sides = "+".join(sorted(md["sides"]))
    short_mkt = mkt[:52] if len(mkt) > 52 else mkt
    buys_str = str(md["count"]) + "B"

    print("  %s %-5s %-8s %-3s cost=$%6.2f  recv=$%6.2f  pnl=$%+7.2f  %s" % (
        time_str.ljust(15), tag, sides, buys_str, md["cost"], recv, pnl, short_mkt))

wr = w / (w + l) * 100 if (w + l) > 0 else 0
print("-" * 100)
print("  TOTALS: %d markets | %dW/%dL (%.0f%% WR) | Cost: $%.2f | Recv: $%.2f | Net: $%+.2f" % (
    len(seen), w, l, wr, total_cost, total_recv_all, total_pnl))
print()

# Internal bot records
print("=" * 100)
print("  INTERNAL BOT RECORDS - MOST RECENT TRADES")
print("=" * 100)

with open("momentum_15m_results.json") as f:
    mom = json.load(f)
print("\nMomentum 15M: %d resolved" % len(mom["resolved"]))
for r in mom["resolved"][-5:]:
    et = str(r.get("entry_time", ""))[:19]
    pnl = r.get("pnl", 0)
    strat = r.get("strategy", "")
    tier = r.get("tier", "?")
    title = str(r.get("title", ""))[:45]
    print("  %s  pnl=$%+6.2f  %s  tier=%s  %s" % (et, pnl, strat, tier, title))

with open("sniper_5m_live_results.json") as f:
    snp = json.load(f)
print("\nSniper 5M Live: %d resolved" % len(snp["resolved"]))
for r in snp["resolved"][-5:]:
    et = str(r.get("entry_time", ""))[:19]
    pnl = r.get("pnl", 0)
    conf = r.get("confidence", 0)
    q = str(r.get("question", ""))[:45]
    print("  %s  pnl=$%+6.2f  conf=%.2f  %s" % (et, pnl, conf, q))

mom_s = mom.get("stats", {})
snp_s = snp.get("stats", {})
print("\nMomentum stats: W=%s L=%s PnL=$%.2f" % (mom_s.get("wins", 0), mom_s.get("losses", 0), mom_s.get("pnl", 0)))
print("Sniper stats: W=%s L=%s PnL=$%.2f" % (snp_s.get("wins", 0), snp_s.get("losses", 0), snp_s.get("pnl", 0)))

# Feb 19 only analysis
print("\n" + "=" * 100)
print("  FEB 19 ONLY (after killing all losers)")
print("=" * 100)
feb19_start = 1771459200  # Feb 19 00:00 UTC
feb19_buys = [t for t in all_trades if t["action"] == "Buy" and t["ts"] >= feb19_start]
feb19_recv = [t for t in all_trades if t["action"] in ("Redeem", "Sell", "Maker Rebate") and t["ts"] >= feb19_start]
f19_cost = sum(t["usdc"] for t in feb19_buys)
f19_recv = sum(t["usdc"] for t in feb19_recv)
f19_net = f19_recv - f19_cost
print("  Buys: %d ($%.2f) | Recv: %d ($%.2f) | Net: $%+.2f" % (
    len(feb19_buys), f19_cost, len(feb19_recv), f19_recv, f19_net))

# Per-market for Feb 19
f19_mkt_cost = defaultdict(float)
f19_mkt_recv = defaultdict(float)
f19_mkt_sides = defaultdict(set)
for t in feb19_buys:
    f19_mkt_cost[t["market"]] += t["usdc"]
    f19_mkt_sides[t["market"]].add(t["token_name"])
for t in feb19_recv:
    f19_mkt_recv[t["market"]] += t["usdc"]

f19w = 0
f19l = 0
for mkt in sorted(f19_mkt_cost.keys(), key=lambda m: f19_mkt_cost[m], reverse=True):
    cost = f19_mkt_cost[mkt]
    recv = f19_mkt_recv.get(mkt, 0.0)
    pnl = recv - cost
    sides = "+".join(sorted(f19_mkt_sides[mkt]))
    tag = "WIN" if pnl > 0 else "LOSS"
    if pnl > 0: f19w += 1
    else: f19l += 1
    short = mkt[:55]
    print("  %-5s %-6s cost=$%6.2f recv=$%6.2f pnl=$%+7.2f  %s" % (tag, sides, cost, recv, pnl, short))

if f19w + f19l > 0:
    print("\n  Feb 19: %dW/%dL (%.0f%% WR) | Net: $%+.2f" % (f19w, f19l, f19w/(f19w+f19l)*100, f19_net))
