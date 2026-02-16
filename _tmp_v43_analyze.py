"""Analyze V4.3 CSV performance for profit projections."""
import csv
from collections import defaultdict
from datetime import datetime

csv_path = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-15 (6).csv"

buys = []
redeems = []
sells = []
rebates = []
total_bought = 0.0
total_redeemed = 0.0
total_sold = 0.0
total_rebates = 0.0

markets = defaultdict(lambda: {"buys": [], "redeems": [], "sells": []})

with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        action = row['action']
        amount = float(row['usdcAmount'])
        tokens = float(row['tokenAmount'])
        market = row['marketName']
        token = row['tokenName']
        ts = int(row['timestamp'])

        if action == 'Buy':
            buys.append({"market": market, "amount": amount, "tokens": tokens, "token": token, "ts": ts})
            total_bought += amount
            markets[market]["buys"].append({"amount": amount, "tokens": tokens, "token": token, "ts": ts})
        elif action == 'Redeem':
            redeems.append({"market": market, "amount": amount, "ts": ts})
            total_redeemed += amount
            markets[market]["redeems"].append({"amount": amount, "ts": ts})
        elif action == 'Sell':
            sells.append({"market": market, "amount": amount, "tokens": tokens, "token": token, "ts": ts})
            total_sold += amount
            markets[market]["sells"].append({"amount": amount, "tokens": tokens, "token": token, "ts": ts})
        elif 'Rebate' in action or 'rebate' in action.lower():
            rebates.append({"amount": amount, "ts": ts})
            total_rebates += amount

# Time range
all_ts = [b['ts'] for b in buys] + [r['ts'] for r in redeems]
if all_ts:
    min_ts = min(all_ts)
    max_ts = max(all_ts)
    duration_sec = max_ts - min_ts
    duration_hrs = duration_sec / 3600
else:
    duration_hrs = 0

# Market-level P&L
print("=" * 70)
print("V4.3 TAKER HEDGE MAKER — PERFORMANCE ANALYSIS")
print("=" * 70)

# Count unique markets
unique_markets = set()
for b in buys:
    unique_markets.add(b['market'])

# Count pairs vs partials
market_stats = {}
for mkt, data in markets.items():
    buy_tokens = defaultdict(float)
    buy_cost = 0.0
    for b in data["buys"]:
        buy_tokens[b["token"]] += b["tokens"]
        buy_cost += b["amount"]

    redeem_total = sum(r["amount"] for r in data["redeems"])
    sell_total = sum(s["amount"] for s in data["sells"])

    has_up = buy_tokens.get("Up", 0) > 0
    has_down = buy_tokens.get("Down", 0) > 0
    is_paired = has_up and has_down

    pnl = redeem_total + sell_total - buy_cost

    market_stats[mkt] = {
        "buy_cost": buy_cost,
        "redeem": redeem_total,
        "sell": sell_total,
        "pnl": pnl,
        "paired": is_paired,
        "up_tokens": buy_tokens.get("Up", 0),
        "down_tokens": buy_tokens.get("Down", 0),
    }

paired_count = sum(1 for s in market_stats.values() if s["paired"])
partial_count = sum(1 for s in market_stats.values() if not s["paired"])
total_markets = len(market_stats)

paired_pnl = sum(s["pnl"] for s in market_stats.values() if s["paired"])
partial_pnl = sum(s["pnl"] for s in market_stats.values() if not s["paired"])

# Separate wins/losses
paired_wins = sum(1 for s in market_stats.values() if s["paired"] and s["pnl"] > 0)
paired_losses = sum(1 for s in market_stats.values() if s["paired"] and s["pnl"] <= 0)
partial_wins = sum(1 for s in market_stats.values() if not s["paired"] and s["pnl"] > 0)
partial_losses = sum(1 for s in market_stats.values() if not s["paired"] and s["pnl"] <= 0)

# Resolved vs unresolved
resolved = sum(1 for s in market_stats.values() if s["redeem"] > 0 or s["sell"] > 0)
unresolved = total_markets - resolved

net_pnl = total_redeemed + total_sold + total_rebates - total_bought

print(f"\nTime window: {duration_hrs:.1f} hours")
print(f"Total markets entered: {total_markets}")
print(f"  Paired (both sides): {paired_count} ({paired_count/total_markets*100:.0f}%)")
print(f"  Partial (one side):  {partial_count} ({partial_count/total_markets*100:.0f}%)")
print(f"  Resolved: {resolved} | Unresolved: {unresolved}")

print(f"\n{'='*40}")
print(f"CASH FLOW")
print(f"{'='*40}")
print(f"Total bought:    -${total_bought:.2f}")
print(f"Total redeemed:  +${total_redeemed:.2f}")
print(f"Total sold:      +${total_sold:.2f}")
print(f"Maker rebates:   +${total_rebates:.2f}")
print(f"{'='*40}")
print(f"NET P&L:         ${net_pnl:+.2f}")

print(f"\n{'='*40}")
print(f"P&L BREAKDOWN")
print(f"{'='*40}")
print(f"Paired P&L:   ${paired_pnl:+.2f} ({paired_wins}W/{paired_losses}L)")
print(f"Partial P&L:  ${partial_pnl:+.2f} ({partial_wins}W/{partial_losses}L)")
print(f"Rebates:      ${total_rebates:+.2f}")

# Per-asset breakdown
asset_pnl = defaultdict(float)
asset_count = defaultdict(int)
for mkt, stats in market_stats.items():
    for asset in ["Bitcoin", "Ethereum", "Solana", "XRP"]:
        if asset in mkt:
            asset_pnl[asset] += stats["pnl"]
            asset_count[asset] += 1
            break

print(f"\n{'='*40}")
print(f"PER-ASSET P&L")
print(f"{'='*40}")
for asset in sorted(asset_pnl.keys()):
    print(f"  {asset}: ${asset_pnl[asset]:+.2f} ({asset_count[asset]} markets)")

# Hourly rate
if duration_hrs > 0:
    hourly_rate = net_pnl / duration_hrs
    daily_rate = hourly_rate * 24
    weekly_rate = daily_rate * 7
    monthly_rate = daily_rate * 30

    print(f"\n{'='*40}")
    print(f"PROFIT PROJECTIONS (at $3/side)")
    print(f"{'='*40}")
    print(f"  Hourly:  ${hourly_rate:+.2f}/hr")
    print(f"  Daily:   ${daily_rate:+.2f}/day (24h)")
    print(f"  Weekly:  ${weekly_rate:+.2f}/week")
    print(f"  Monthly: ${monthly_rate:+.2f}/month")

    # At $5/side (1.67x scale)
    scale = 5.0 / 3.0
    print(f"\n{'='*40}")
    print(f"PROFIT PROJECTIONS (at $5/side = 1.67x)")
    print(f"{'='*40}")
    print(f"  Hourly:  ${hourly_rate * scale:+.2f}/hr")
    print(f"  Daily:   ${daily_rate * scale:+.2f}/day (24h)")
    print(f"  Weekly:  ${weekly_rate * scale:+.2f}/week")
    print(f"  Monthly: ${monthly_rate * scale:+.2f}/month")

# Top 10 best/worst markets
print(f"\n{'='*40}")
print(f"TOP 5 BEST MARKETS")
print(f"{'='*40}")
sorted_markets = sorted(market_stats.items(), key=lambda x: x[1]["pnl"], reverse=True)
for mkt, stats in sorted_markets[:5]:
    pair_tag = "PAIR" if stats["paired"] else "PART"
    print(f"  [{pair_tag}] ${stats['pnl']:+.2f} | {mkt[:55]}")

print(f"\n{'='*40}")
print(f"TOP 5 WORST MARKETS")
print(f"{'='*40}")
for mkt, stats in sorted_markets[-5:]:
    pair_tag = "PAIR" if stats["paired"] else "PART"
    print(f"  [{pair_tag}] ${stats['pnl']:+.2f} | {mkt[:55]}")

# Pair rate analysis
print(f"\n{'='*40}")
print(f"KEY METRICS")
print(f"{'='*40}")
pair_rate = paired_count / total_markets * 100 if total_markets > 0 else 0
avg_paired_pnl = paired_pnl / paired_count if paired_count > 0 else 0
avg_partial_pnl = partial_pnl / partial_count if partial_count > 0 else 0
print(f"  Pair rate: {pair_rate:.1f}%")
print(f"  Avg paired P&L: ${avg_paired_pnl:+.2f}")
print(f"  Avg partial P&L: ${avg_partial_pnl:+.2f}")
print(f"  Trades/hour: {total_markets / duration_hrs:.1f}" if duration_hrs > 0 else "")

# Taker hedge analysis - look for markets where we bought AFTER initial fill
# (hedge buys have slightly different timing patterns)
print(f"\n{'='*40}")
print(f"TAKER HEDGE EFFECTIVENESS")
print(f"{'='*40}")
hedge_success = 0
hedge_attempt = 0
for mkt, data in markets.items():
    buy_ts_list = sorted([b["ts"] for b in data["buys"]])
    if len(buy_ts_list) >= 2:
        # Multiple buy timestamps = likely hedge attempt
        up_buys = [b for b in data["buys"] if b["token"] == "Up"]
        down_buys = [b for b in data["buys"] if b["token"] == "Down"]
        if up_buys and down_buys:
            up_ts = min(b["ts"] for b in up_buys)
            down_ts = min(b["ts"] for b in down_buys)
            time_diff = abs(up_ts - down_ts)
            if time_diff > 5:  # >5 seconds apart = likely hedge
                hedge_attempt += 1
                if market_stats[mkt]["paired"]:
                    hedge_success += 1

print(f"  Likely hedge attempts: {hedge_attempt}")
print(f"  Successful hedges: {hedge_success}")
if hedge_attempt > 0:
    print(f"  Hedge success rate: {hedge_success/hedge_attempt*100:.0f}%")
