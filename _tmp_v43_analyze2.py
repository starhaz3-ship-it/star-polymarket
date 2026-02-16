"""Analyze V4.3 era ONLY — filter by $3/side trades (post-fix)."""
import csv
from collections import defaultdict

csv_path = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-15 (6).csv"

rows = []
with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "market": row['marketName'],
            "action": row['action'],
            "amount": float(row['usdcAmount']),
            "tokens": float(row['tokenAmount']),
            "token": row['tokenName'],
            "ts": int(row['timestamp']),
        })

# Sort by timestamp ascending
rows.sort(key=lambda x: x['ts'])

# Find the V4.3 era: trades with $3/side (individual buys ~$2.5-$3.1)
# V4.1/4.2 used $5/side (individual buys ~$4.5-$5.1)
# V4.2 deep bids had some at $5/side too
# Best separator: look at when $3/side trades started

# Group buys by market to identify per-market cost
markets_all = defaultdict(list)
for r in rows:
    markets_all[r['market']].append(r)

# For each market, calculate total buy cost and identify era
# V4.3 markets: total buy cost per side ~$2.5-3.1
# V4.1/4.2 markets: total buy cost per side ~$4.5-5.1

# Actually, let's split by timestamp. V4.3 moderate offsets started around 7PM ET on Feb 15
# Based on summary: V4.3 was deployed after killing the deep bid bot
# Looking at the data: $2.58-$2.94 buys started appearing around ts 1771200000+

# Let me identify the cutoff by looking at buy sizes
print("=" * 70)
print("FINDING V4.3 START POINT")
print("=" * 70)

buy_rows = [r for r in rows if r['action'] == 'Buy']
for b in buy_rows[:20]:
    # Show first 20 buys (oldest first)
    from datetime import datetime, timezone
    dt = datetime.fromtimestamp(b['ts'], tz=timezone.utc)
    print(f"  {dt.strftime('%H:%M:%S')} UTC | ${b['amount']:.2f} | {b['tokens']:.1f} {b['token']} | {b['market'][:40]}")

print(f"\n...skipping middle...\n")

# Find the transition point: last big buy ($4+) then first small buy ($3-)
big_buys = [b for b in buy_rows if b['amount'] >= 4.0]
small_buys = [b for b in buy_rows if b['amount'] < 3.5 and b['amount'] > 0.5]

if big_buys and small_buys:
    last_big = max(b['ts'] for b in big_buys)
    # Find first small buy after last big
    first_small_after = min((b['ts'] for b in small_buys if b['ts'] > last_big), default=None)

    # Also find the gap where V4.2 was killed and V4.3 started
    all_buy_ts = sorted(set(b['ts'] for b in buy_rows))
    gaps = []
    for i in range(1, len(all_buy_ts)):
        gap = all_buy_ts[i] - all_buy_ts[i-1]
        if gap > 120:  # >2 min gap
            from datetime import datetime, timezone
            dt1 = datetime.fromtimestamp(all_buy_ts[i-1], tz=timezone.utc)
            dt2 = datetime.fromtimestamp(all_buy_ts[i], tz=timezone.utc)
            gaps.append((gap, all_buy_ts[i-1], all_buy_ts[i]))

    print("Significant gaps (>2 min between buys):")
    for gap, t1, t2 in sorted(gaps, key=lambda x: x[0], reverse=True)[:10]:
        from datetime import datetime, timezone
        dt1 = datetime.fromtimestamp(t1, tz=timezone.utc)
        dt2 = datetime.fromtimestamp(t2, tz=timezone.utc)
        print(f"  {gap/60:.0f} min gap: {dt1.strftime('%H:%M')} -> {dt2.strftime('%H:%M')} UTC")

# Now analyze by era
# V4.3 with $3/side: markets where max single buy < $3.5
print(f"\n{'='*70}")
print(f"ERA ANALYSIS")
print(f"{'='*70}")

era_v41 = {"bought": 0, "redeemed": 0, "sold": 0, "markets": 0, "paired": 0, "partial": 0}
era_v43 = {"bought": 0, "redeemed": 0, "sold": 0, "markets": 0, "paired": 0, "partial": 0}

for mkt, mkt_rows in markets_all.items():
    buy_rows_m = [r for r in mkt_rows if r['action'] == 'Buy']
    if not buy_rows_m:
        continue

    max_single_buy = max(r['amount'] for r in buy_rows_m)
    total_buy = sum(r['amount'] for r in buy_rows_m)
    total_redeem = sum(r['amount'] for r in mkt_rows if r['action'] == 'Redeem')
    total_sell = sum(r['amount'] for r in mkt_rows if r['action'] == 'Sell')

    tokens_by_side = defaultdict(float)
    for b in buy_rows_m:
        if b['token']:
            tokens_by_side[b['token']] += b['tokens']
    is_paired = len(tokens_by_side) >= 2

    pnl = total_redeem + total_sell - total_buy

    # V4.1/V4.2: max single buy >= $4.0 (was $5/side)
    # V4.3: max single buy < $4.0 (is $3/side)
    if max_single_buy >= 4.0:
        era = era_v41
    else:
        era = era_v43

    era["bought"] += total_buy
    era["redeemed"] += total_redeem
    era["sold"] += total_sell
    era["markets"] += 1
    if is_paired:
        era["paired"] += 1
    else:
        era["partial"] += 1

for name, era in [("V4.1/V4.2 ($5/side)", era_v41), ("V4.3 ($3/side)", era_v43)]:
    pnl = era["redeemed"] + era["sold"] - era["bought"]
    pair_rate = era["paired"] / era["markets"] * 100 if era["markets"] > 0 else 0
    print(f"\n  {name}:")
    print(f"    Markets: {era['markets']} (paired: {era['paired']}, partial: {era['partial']})")
    print(f"    Pair rate: {pair_rate:.0f}%")
    print(f"    Bought: ${era['bought']:.2f} | Redeemed: ${era['redeemed']:.2f} | Sold: ${era['sold']:.2f}")
    print(f"    P&L: ${pnl:+.2f}")

# Now detailed V4.3 analysis
print(f"\n{'='*70}")
print(f"V4.3 DETAILED ANALYSIS ($3/side era)")
print(f"{'='*70}")

v43_markets = {}
v43_min_ts = float('inf')
v43_max_ts = 0

for mkt, mkt_rows in markets_all.items():
    buy_rows_m = [r for r in mkt_rows if r['action'] == 'Buy']
    if not buy_rows_m:
        continue
    max_single_buy = max(r['amount'] for r in buy_rows_m)
    if max_single_buy >= 4.0:
        continue  # Skip V4.1/V4.2 era

    total_buy = sum(r['amount'] for r in buy_rows_m)
    total_redeem = sum(r['amount'] for r in mkt_rows if r['action'] == 'Redeem')
    total_sell = sum(r['amount'] for r in mkt_rows if r['action'] == 'Sell')

    tokens_by_side = defaultdict(float)
    for b in buy_rows_m:
        if b['token']:
            tokens_by_side[b['token']] += b['tokens']
    is_paired = len(tokens_by_side) >= 2

    pnl = total_redeem + total_sell - total_buy
    resolved = total_redeem > 0 or total_sell > 0

    all_ts = [r['ts'] for r in mkt_rows]
    v43_min_ts = min(v43_min_ts, min(all_ts))
    v43_max_ts = max(v43_max_ts, max(all_ts))

    v43_markets[mkt] = {
        "buy": total_buy, "redeem": total_redeem, "sell": total_sell,
        "pnl": pnl, "paired": is_paired, "resolved": resolved,
        "up": tokens_by_side.get("Up", 0), "down": tokens_by_side.get("Down", 0),
    }

v43_duration = (v43_max_ts - v43_min_ts) / 3600
v43_total_pnl = sum(s["pnl"] for s in v43_markets.values())
v43_resolved = sum(1 for s in v43_markets.values() if s["resolved"])
v43_unresolved = sum(1 for s in v43_markets.values() if not s["resolved"])
v43_paired = sum(1 for s in v43_markets.values() if s["paired"])
v43_partial = sum(1 for s in v43_markets.values() if not s["paired"])

paired_wins = sum(1 for s in v43_markets.values() if s["paired"] and s["pnl"] > 0)
paired_losses = sum(1 for s in v43_markets.values() if s["paired"] and s["pnl"] <= 0)
partial_wins = sum(1 for s in v43_markets.values() if not s["paired"] and s["pnl"] > 0)
partial_losses = sum(1 for s in v43_markets.values() if not s["paired"] and s["pnl"] <= 0)

paired_pnl = sum(s["pnl"] for s in v43_markets.values() if s["paired"])
partial_pnl = sum(s["pnl"] for s in v43_markets.values() if not s["paired"])

print(f"Duration: {v43_duration:.1f} hours")
print(f"Markets: {len(v43_markets)} (resolved: {v43_resolved}, unresolved: {v43_unresolved})")
print(f"Pair rate: {v43_paired / len(v43_markets) * 100:.0f}% ({v43_paired} paired, {v43_partial} partial)")
print(f"\nPaired: ${paired_pnl:+.2f} ({paired_wins}W/{paired_losses}L)")
print(f"Partial: ${partial_pnl:+.2f} ({partial_wins}W/{partial_losses}L)")
print(f"TOTAL P&L: ${v43_total_pnl:+.2f}")

if v43_duration > 0:
    hourly = v43_total_pnl / v43_duration
    print(f"\n--- PROJECTIONS ($3/side) ---")
    print(f"  Per hour:  ${hourly:+.2f}")
    print(f"  Per day:   ${hourly * 24:+.2f}")
    print(f"  Per week:  ${hourly * 168:+.2f}")
    print(f"  Per month: ${hourly * 720:+.2f}")

    scale = 5.0 / 3.0
    print(f"\n--- PROJECTIONS ($5/side = 1.67x) ---")
    print(f"  Per hour:  ${hourly * scale:+.2f}")
    print(f"  Per day:   ${hourly * scale * 24:+.2f}")
    print(f"  Per week:  ${hourly * scale * 168:+.2f}")
    print(f"  Per month: ${hourly * scale * 720:+.2f}")

# Best/worst V4.3 markets
sorted_v43 = sorted(v43_markets.items(), key=lambda x: x[1]["pnl"], reverse=True)
print(f"\nTop 5 V4.3 winners:")
for mkt, s in sorted_v43[:5]:
    tag = "PAIR" if s["paired"] else "PART"
    print(f"  [{tag}] ${s['pnl']:+.2f} | {mkt[:50]}")

print(f"\nTop 5 V4.3 losers:")
for mkt, s in sorted_v43[-5:]:
    tag = "PAIR" if s["paired"] else "PART"
    print(f"  [{tag}] ${s['pnl']:+.2f} | {mkt[:50]}")

# Win rate by type
print(f"\nWin rates:")
if v43_paired > 0:
    print(f"  Paired WR: {paired_wins}/{v43_paired} = {paired_wins/v43_paired*100:.0f}%")
if v43_partial > 0:
    print(f"  Partial WR: {partial_wins}/{v43_partial} = {partial_wins/v43_partial*100:.0f}%")
overall_wins = paired_wins + partial_wins
print(f"  Overall WR: {overall_wins}/{len(v43_markets)} = {overall_wins/len(v43_markets)*100:.0f}%")

# Average win/loss size
wins = [s["pnl"] for s in v43_markets.values() if s["pnl"] > 0]
losses = [s["pnl"] for s in v43_markets.values() if s["pnl"] < 0]
if wins:
    print(f"\n  Avg win:  ${sum(wins)/len(wins):+.2f} ({len(wins)} trades)")
if losses:
    print(f"  Avg loss: ${sum(losses)/len(losses):+.2f} ({len(losses)} trades)")
if wins and losses:
    print(f"  W:L ratio: {abs(sum(wins)/len(wins)) / abs(sum(losses)/len(losses)):.2f}x")
