"""
Polymarket CSV vs Maker Bot Reconciliation Script
Compares Polymarket trade history CSV against maker_results.json for last 48 hours.
"""

import csv
import json
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# ── File paths ──────────────────────────────────────────────────────────────
CSV_PATH = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-14 (5).csv"
MAKER_PATH = r"C:\Users\Star\.local\bin\star-polymarket\maker_results.json"

# ── Time window: last 48 hours from ~Feb 15 00:45 UTC ─────────────────────
NOW_UTC = datetime(2026, 2, 15, 0, 55, 0, tzinfo=timezone.utc)
CUTOFF_UTC = NOW_UTC - timedelta(hours=48)
CUTOFF_TS = int(CUTOFF_UTC.timestamp())

print("=" * 80)
print("POLYMARKET CSV vs MAKER BOT RECONCILIATION")
print("=" * 80)
print(f"Time window: {CUTOFF_UTC.strftime('%Y-%m-%d %H:%M UTC')} to {NOW_UTC.strftime('%Y-%m-%d %H:%M UTC')}")
print(f"Cutoff timestamp: {CUTOFF_TS}")
print()

# ── 1. Parse CSV ───────────────────────────────────────────────────────────
csv_trades = []
with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ts = int(row["timestamp"])
        if ts >= CUTOFF_TS:
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

print(f"CSV: {len(csv_trades)} trades in last 48h (out of 999 data rows)")
if csv_trades:
    print(f"  Earliest: {csv_trades[-1]['dt'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Latest:   {csv_trades[0]['dt'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
print()

# ── 2. Group CSV by market ─────────────────────────────────────────────────
market_data = defaultdict(lambda: {
    "buys_usdc": 0.0, "sells_usdc": 0.0, "redeems_usdc": 0.0,
    "buy_count": 0, "sell_count": 0, "redeem_count": 0,
    "buy_tokens_up": 0.0, "buy_tokens_down": 0.0,
    "first_ts": None, "last_ts": None,
})

total_buys = 0.0
total_sells = 0.0
total_redeems = 0.0
total_buy_count = 0
total_sell_count = 0
total_redeem_count = 0

for t in csv_trades:
    m = market_data[t["market"]]
    action = t["action"]

    if action == "Buy":
        m["buys_usdc"] += t["usdc"]
        m["buy_count"] += 1
        total_buys += t["usdc"]
        total_buy_count += 1
        if t["token_name"] == "Up":
            m["buy_tokens_up"] += t["tokens"]
        elif t["token_name"] == "Down":
            m["buy_tokens_down"] += t["tokens"]
    elif action == "Sell":
        m["sells_usdc"] += t["usdc"]
        m["sell_count"] += 1
        total_sells += t["usdc"]
        total_sell_count += 1
    elif action == "Redeem":
        m["redeems_usdc"] += t["usdc"]
        m["redeem_count"] += 1
        total_redeems += t["usdc"]
        total_redeem_count += 1

    if m["first_ts"] is None or t["timestamp"] < m["first_ts"]:
        m["first_ts"] = t["timestamp"]
    if m["last_ts"] is None or t["timestamp"] > m["last_ts"]:
        m["last_ts"] = t["timestamp"]

csv_net_pnl = total_sells + total_redeems - total_buys

print("-" * 80)
print("CSV SUMMARY (Last 48h)")
print("-" * 80)
print(f"  Markets traded:    {len(market_data)}")
print(f"  Total Buys:        ${total_buys:,.2f}  ({total_buy_count} txns)")
print(f"  Total Sells:       ${total_sells:,.2f}  ({total_sell_count} txns)")
print(f"  Total Redeems:     ${total_redeems:,.2f}  ({total_redeem_count} txns)")
print(f"  NET PnL (CSV):     ${csv_net_pnl:,.2f}")
print(f"  NOTE: Losing positions have NO Redeem row -- invisible losses!")
print()

# ── 3. Parse maker_results.json ────────────────────────────────────────────
with open(MAKER_PATH, "r", encoding="utf-8") as f:
    maker = json.load(f)

maker_stats = maker.get("stats", {})
maker_resolved = maker.get("resolved", [])
maker_active = maker.get("active_positions", [])

# Filter resolved to last 48h
maker_resolved_48h = []
for r in maker_resolved:
    resolved_at = datetime.fromisoformat(r["resolved_at"])
    if resolved_at >= CUTOFF_UTC:
        maker_resolved_48h.append(r)

# Parse active positions
maker_active_48h = []
for a in maker_active:
    created_at = datetime.fromisoformat(a["created_at"])
    if created_at >= CUTOFF_UTC:
        maker_active_48h.append(a)

maker_total_pnl = sum(r["pnl"] for r in maker_resolved_48h)
maker_total_cost = sum(r["combined_cost"] for r in maker_resolved_48h)
maker_paired = [r for r in maker_resolved_48h if r.get("paired")]
maker_unpaired = [r for r in maker_resolved_48h if not r.get("paired")]

print("-" * 80)
print("MAKER BOT INTERNAL SUMMARY (Last 48h)")
print("-" * 80)
print(f"  Stats start_time:  {maker_stats.get('start_time', 'N/A')}")
print(f"  Resolved total:    {len(maker_resolved_48h)}")
print(f"    Paired:          {len(maker_paired)}")
print(f"    Unpaired/Zero:   {len(maker_unpaired)}")
print(f"  Active positions:  {len(maker_active_48h)}")
print(f"  Total cost:        ${maker_total_cost:,.2f}")
print(f"  Total PnL:         ${maker_total_pnl:,.4f}")
print(f"  Stats total_pnl:   ${maker_stats.get('total_pnl', 0):,.4f}")
print(f"  Stats volume:      ${maker_stats.get('total_volume', 0):,.2f}")
print()

# ── 4. Per-market reconciliation ───────────────────────────────────────────
print("-" * 80)
print("PER-MARKET RECONCILIATION")
print("-" * 80)

# Build maker lookup by question (market name)
maker_by_question = {}
for r in maker_resolved_48h:
    q = r["question"]
    if q not in maker_by_question:
        maker_by_question[q] = []
    maker_by_question[q].append(r)

for a in maker_active_48h:
    q = a["question"]
    if q not in maker_by_question:
        maker_by_question[q] = []
    maker_by_question[q].append({**a, "_active": True})

# Track matched/unmatched
csv_matched = set()
maker_matched = set()

# Categorize CSV markets
csv_markets_with_buys_only = []   # bought but no sell/redeem = likely loss
csv_markets_with_redeem = []       # redeemed = win
csv_markets_with_sell = []         # sold = early exit

print()
print(f"{'Market':<60} {'CSV Buy':>10} {'CSV Sell':>10} {'CSV Redm':>10} {'CSV Net':>10} {'Maker PnL':>10} {'Match':>8}")
print("-" * 128)

# Sort markets by first timestamp
sorted_markets = sorted(market_data.keys(), key=lambda m: market_data[m]["first_ts"] or 0)

discrepancies = []
csv_market_count = 0
csv_market_matched = 0

for market_name in sorted_markets:
    md = market_data[market_name]
    csv_net = md["sells_usdc"] + md["redeems_usdc"] - md["buys_usdc"]
    csv_market_count += 1

    # Find matching maker entry
    maker_entry = maker_by_question.get(market_name, [])

    if maker_entry:
        maker_pnl = sum(e.get("pnl", 0) for e in maker_entry)
        is_active = any(e.get("_active") for e in maker_entry)
        maker_cost = sum(e.get("combined_cost", 0) for e in maker_entry)
        paired_any = any(e.get("paired") for e in maker_entry)

        status = "ACTIVE" if is_active else "OK"

        # Check if PnL roughly matches
        if not is_active and paired_any and abs(csv_net - maker_pnl) > 0.10:
            status = "DIFF"
            discrepancies.append({
                "market": market_name,
                "csv_net": csv_net,
                "maker_pnl": maker_pnl,
                "diff": csv_net - maker_pnl,
            })

        maker_matched.add(market_name)
        csv_matched.add(market_name)
        csv_market_matched += 1

        marker = f"${maker_pnl:>9.2f}"
    else:
        marker = "    N/A"
        status = "NO-MKR"

    # Truncate long market names
    short_name = market_name[:58] if len(market_name) > 58 else market_name

    print(f"{short_name:<60} ${md['buys_usdc']:>9.2f} ${md['sells_usdc']:>9.2f} ${md['redeems_usdc']:>9.2f} ${csv_net:>9.2f} {marker:>10} {status:>8}")

print("-" * 128)

# ── 5. Maker entries NOT in CSV ────────────────────────────────────────────
print()
print("-" * 80)
print("MAKER ENTRIES NOT FOUND IN CSV (Last 48h)")
print("-" * 80)

maker_only = []
for q, entries in maker_by_question.items():
    if q not in csv_matched:
        for e in entries:
            maker_only.append(e)

if maker_only:
    for e in maker_only:
        is_active = e.get("_active", False)
        status = "ACTIVE" if is_active else "RESOLVED"
        paired = e.get("paired", False)
        pnl = e.get("pnl", 0)
        cost = e.get("combined_cost", 0)
        print(f"  [{status}] {e['question']}")
        print(f"    Outcome: {e.get('outcome', 'pending')}, Paired: {paired}, Cost: ${cost:.2f}, PnL: ${pnl:.2f}")
else:
    print("  None -- all maker entries found in CSV")

# ── 6. CSV markets NOT in maker (non-maker trades) ────────────────────────
print()
print("-" * 80)
print("CSV MARKETS NOT IN MAKER (non-maker trades or different bot)")
print("-" * 80)

csv_only = [m for m in sorted_markets if m not in maker_matched]
non_maker_pnl = 0.0
if csv_only:
    for m in csv_only:
        md = market_data[m]
        net = md["sells_usdc"] + md["redeems_usdc"] - md["buys_usdc"]
        non_maker_pnl += net
        action_summary = []
        if md["buy_count"]: action_summary.append(f"{md['buy_count']}B")
        if md["sell_count"]: action_summary.append(f"{md['sell_count']}S")
        if md["redeem_count"]: action_summary.append(f"{md['redeem_count']}R")
        print(f"  {m[:70]}")
        print(f"    Actions: {'+'.join(action_summary)}, Buy: ${md['buys_usdc']:.2f}, Sell: ${md['sells_usdc']:.2f}, Redeem: ${md['redeems_usdc']:.2f}, Net: ${net:+.2f}")
    print(f"\n  Total non-maker PnL: ${non_maker_pnl:+.2f}")
else:
    print("  None -- all CSV markets accounted for in maker")

# ── 7. Invisible losses analysis ───────────────────────────────────────────
print()
print("-" * 80)
print("INVISIBLE LOSS DETECTION")
print("-" * 80)
print("Markets where USDC was spent (Buy) but NO Sell and NO Redeem = total loss")
print()

invisible_losses = []
for market_name in sorted_markets:
    md = market_data[market_name]
    if md["buys_usdc"] > 0 and md["sells_usdc"] == 0 and md["redeems_usdc"] == 0:
        invisible_losses.append((market_name, md["buys_usdc"]))

if invisible_losses:
    total_invisible = 0
    for name, amount in invisible_losses:
        total_invisible += amount
        print(f"  LOSS: ${amount:>8.2f}  {name[:65]}")
    print(f"\n  Total invisible losses: ${total_invisible:.2f} ({len(invisible_losses)} markets)")
else:
    print("  No invisible losses detected in this window")

# ── 8. Paired trade analysis (maker specific) ─────────────────────────────
print()
print("-" * 80)
print("MAKER PAIRED TRADE ANALYSIS")
print("-" * 80)

for r in maker_resolved_48h:
    q = r["question"]
    paired = r.get("paired", False)
    cost = r.get("combined_cost", 0)
    pnl = r.get("pnl", 0)
    outcome = r.get("outcome", "?")
    resolved_at = r.get("resolved_at", "?")
    up_p = r.get("up_price", 0)
    dn_p = r.get("down_price", 0)
    spread = (1.0 - up_p - dn_p) if (up_p and dn_p) else 0

    status = "PAIRED" if paired else "SINGLE"

    # Cross-reference with CSV
    csv_md = market_data.get(q)
    csv_net = (csv_md["sells_usdc"] + csv_md["redeems_usdc"] - csv_md["buys_usdc"]) if csv_md else None
    csv_str = f"${csv_net:+.2f}" if csv_net is not None else "N/A"

    print(f"  [{status}] {q}")
    print(f"    Outcome: {outcome}, Up: {up_p}, Down: {dn_p}, Spread: {spread:.2f}")
    print(f"    Cost: ${cost:.2f}, Maker PnL: ${pnl:+.4f}, CSV Net: {csv_str}")
    if csv_md and paired and csv_net is not None:
        diff = csv_net - pnl
        if abs(diff) > 0.01:
            print(f"    *** DISCREPANCY: ${diff:+.2f} (CSV - Maker)")
    print()

# ── 9. Active position check ──────────────────────────────────────────────
print("-" * 80)
print("ACTIVE POSITIONS (Open in Maker)")
print("-" * 80)

for a in maker_active_48h:
    q = a["question"]
    up_filled = a.get("up_filled", False)
    down_filled = a.get("down_filled", False)
    cost = a.get("combined_cost", 0)
    status = a.get("status", "?")

    csv_md = market_data.get(q)
    csv_buy = csv_md["buys_usdc"] if csv_md else 0

    print(f"  {q}")
    print(f"    Status: {status}, Up filled: {up_filled}, Down filled: {down_filled}")
    print(f"    Combined cost: ${cost:.2f}, CSV buys: ${csv_buy:.2f}")
    print()

# ── 10. Final reconciliation ──────────────────────────────────────────────
print("=" * 80)
print("FINAL RECONCILIATION SUMMARY")
print("=" * 80)

# Maker-only PnL from CSV (markets that ARE in maker)
maker_csv_pnl = 0.0
for market_name in sorted_markets:
    if market_name in maker_matched:
        md = market_data[market_name]
        maker_csv_pnl += md["sells_usdc"] + md["redeems_usdc"] - md["buys_usdc"]

print(f"""
  CSV TOTALS (all trades, last 48h):
    Total Buy:           ${total_buys:>10,.2f}  ({total_buy_count} txns)
    Total Sell:          ${total_sells:>10,.2f}  ({total_sell_count} txns)
    Total Redeem:        ${total_redeems:>10,.2f}  ({total_redeem_count} txns)
    CSV Net PnL:         ${csv_net_pnl:>10,.2f}

  MAKER BOT TOTALS (resolved, last 48h):
    Resolved pairs:      {len(maker_resolved_48h)}
    Paired trades:       {len(maker_paired)}
    Total cost:          ${maker_total_cost:>10,.2f}
    Maker PnL:           ${maker_total_pnl:>10,.4f}
    Stats PnL (all):     ${maker_stats.get('total_pnl', 0):>10,.4f}

  CROSS-CHECK:
    CSV PnL (maker markets only): ${maker_csv_pnl:>10,.2f}
    Maker internal PnL:           ${maker_total_pnl:>10,.4f}
    Difference:                   ${maker_csv_pnl - maker_total_pnl:>10,.4f}

  NON-MAKER CSV PnL:              ${non_maker_pnl:>10,.2f}
    (Markets in CSV but not in maker_results.json)

  MARKETS:
    Total CSV markets (48h):      {csv_market_count}
    Matched to maker:             {csv_market_matched}
    CSV-only (non-maker):         {len(csv_only)}
    Maker-only (not in CSV):      {len(maker_only)}
    Invisible losses:             {len(invisible_losses)} markets, ${sum(a for _, a in invisible_losses) if invisible_losses else 0:.2f}
""")

if discrepancies:
    print("  DISCREPANCIES (>$0.10 diff between CSV and Maker):")
    for d in discrepancies:
        print(f"    {d['market'][:60]}")
        print(f"      CSV: ${d['csv_net']:+.2f}, Maker: ${d['maker_pnl']:+.4f}, Diff: ${d['diff']:+.4f}")
else:
    print("  No significant discrepancies found between CSV and Maker PnL.")

print()
print("=" * 80)
print("NOTES:")
print("  - Polymarket losses have NO Redeem row; buys with no sell/redeem = total loss")
print("  - Maker bot buys BOTH sides (Up+Down), so one side always loses")
print("  - Maker PnL = winning side redeem - combined cost of both sides")
print("  - CSV 'Redeem 0.00' rows are the losing side of maker pairs")
print("  - Non-maker trades may be from TA live/paper or manual trades")
print("=" * 80)
