"""Deep statistical analysis of paper trade results - ALL 12 DIMENSIONS."""
import json
from datetime import datetime
from collections import defaultdict
import statistics
import re

with open(r"C:\Users\Star\.local\bin\star-polymarket\ta_paper_results.json", "r") as f:
    data = json.load(f)

trades_raw = data.get("trades", {})

# Parse trades into list
trades = []
for tid, t in trades_raw.items():
    if t.get("status") != "closed":
        continue
    if t.get("pnl") is None:
        continue

    # Detect asset from market_title
    title = t.get("market_title", "")
    if "Bitcoin" in title:
        asset = "BTC"
    elif "Ethereum" in title:
        asset = "ETH"
    elif "Solana" in title:
        asset = "SOL"
    else:
        asset = "OTHER"

    entry_time = datetime.fromisoformat(t["entry_time"])
    hour_utc = entry_time.hour

    entry_price = t.get("entry_price", 0)
    edge = t.get("edge_at_entry", 0)
    kl = t.get("kl_divergence", 0)
    signal_strength = t.get("signal_strength", "")

    pnl = t["pnl"]
    win = 1 if pnl > 0 else 0

    # Time remaining: 15-min windows, compute from entry minute within window
    minute_of_quarter = entry_time.minute % 15
    time_remaining_min = 15 - minute_of_quarter

    # Features dict (may contain model_confidence, _with_trend, etc.)
    features = t.get("features", {})
    model_conf = features.get("model_confidence", t.get("model_confidence", None))
    with_trend = features.get("_with_trend", t.get("with_trend", None))

    trades.append({
        "side": t["side"],
        "asset": asset,
        "hour": hour_utc,
        "entry_price": entry_price,
        "edge": edge,
        "kl": kl,
        "pnl": pnl,
        "win": win,
        "model_confidence": model_conf,
        "with_trend": with_trend,
        "signal_strength": signal_strength,
        "entry_time": entry_time,
        "entry_time_str": t["entry_time"],
        "size_usd": t.get("size_usd", 10),
        "kelly_fraction": t.get("kelly_fraction", 0),
        "features": features,
        "title": title,
        "exit_price": t.get("exit_price", 0),
    })

# Count open/pending
open_trades = sum(1 for tid, t in trades_raw.items() if t.get("status") != "closed")

print(f"=" * 90)
print(f"  DEEP STATISTICAL ANALYSIS - PAPER TRADE RESULTS")
print(f"=" * 90)
print(f"Total CLOSED trades analyzed: {len(trades)}")
print(f"Open/pending trades (excluded): {open_trades}")
total_pnl = sum(t["pnl"] for t in trades)
total_wins = sum(t["win"] for t in trades)
total_losses = len(trades) - total_wins
print(f"Overall: {total_wins}W / {total_losses}L ({100*total_wins/len(trades):.1f}% WR)")
print(f"Total PnL: ${total_pnl:.2f}")
print(f"Avg PnL/trade: ${total_pnl/len(trades):.2f}")
print(f"Date range: {min(t['entry_time'] for t in trades).strftime('%Y-%m-%d %H:%M')} to {max(t['entry_time'] for t in trades).strftime('%Y-%m-%d %H:%M')} UTC")
print()

# Helper
def calc_stats(group):
    n = len(group)
    if n == 0:
        return {"N": 0, "Wins": 0, "Losses": 0, "WR%": 0, "PnL": 0, "Avg_PnL": 0, "Avg_Win": 0, "Avg_Loss": 0}
    wins = sum(t["win"] for t in group)
    losses = n - wins
    total_pnl = sum(t["pnl"] for t in group)
    avg_pnl = total_pnl / n
    win_pnls = [t["pnl"] for t in group if t["win"] == 1]
    loss_pnls = [t["pnl"] for t in group if t["win"] == 0]
    avg_win = statistics.mean(win_pnls) if win_pnls else 0
    avg_loss = statistics.mean(loss_pnls) if loss_pnls else 0
    return {
        "N": n,
        "Wins": wins,
        "Losses": losses,
        "WR%": round(wins/n*100, 1),
        "PnL": round(total_pnl, 2),
        "Avg_PnL": round(avg_pnl, 2),
        "Avg_Win": round(avg_win, 2),
        "Avg_Loss": round(avg_loss, 2),
    }

def print_table(title, rows, headers):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    if not rows:
        print("  (no data)")
        return
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 1 for i, h in enumerate(headers)]
    header_line = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))))

# ========================================================================
#  1. WIN RATE BY ASSET
# ========================================================================
assets = defaultdict(list)
for t in trades:
    assets[t["asset"]].append(t)

rows = []
for asset in ["BTC", "ETH", "SOL", "OTHER"]:
    if asset not in assets:
        continue
    s = calc_stats(assets[asset])
    rows.append([asset, s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}", f"${s['Avg_Win']}", f"${s['Avg_Loss']}"])
print_table("1. WIN RATE BY ASSET", rows, ["Asset", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL", "Avg Win", "Avg Loss"])

# ========================================================================
#  2. WIN RATE BY SIDE (UP vs DOWN)
# ========================================================================
sides = defaultdict(list)
for t in trades:
    sides[t["side"]].append(t)

rows = []
for side in ["UP", "DOWN"]:
    if side not in sides:
        continue
    s = calc_stats(sides[side])
    rows.append([side, s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}", f"${s['Avg_Win']}", f"${s['Avg_Loss']}"])
print_table("2. WIN RATE BY SIDE", rows, ["Side", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL", "Avg Win", "Avg Loss"])

# ========================================================================
#  3. WIN RATE BY ENTRY PRICE BUCKET (fine-grained)
# ========================================================================
def price_bucket(p):
    if p < 0.05: return "A: <$0.05"
    if p < 0.10: return "B: $0.05-0.10"
    if p < 0.15: return "C: $0.10-0.15"
    if p < 0.20: return "D: $0.15-0.20"
    if p < 0.25: return "E: $0.20-0.25"
    if p < 0.30: return "F: $0.25-0.30"
    if p < 0.35: return "G: $0.30-0.35"
    if p < 0.40: return "H: $0.35-0.40"
    if p < 0.45: return "I: $0.40-0.45"
    if p < 0.50: return "J: $0.45-0.50"
    if p < 0.55: return "K: $0.50-0.55"
    return "L: $0.55+"

price_groups = defaultdict(list)
for t in trades:
    price_groups[price_bucket(t["entry_price"])].append(t)

rows = []
for b in sorted(price_groups.keys()):
    s = calc_stats(price_groups[b])
    rows.append([b, s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}", f"${s['Avg_Win']}", f"${s['Avg_Loss']}"])
print_table("3. WIN RATE BY ENTRY PRICE BUCKET", rows, ["Bucket", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL", "Avg Win", "Avg Loss"])

# Price bucket x Side cross-tab
for side in ["UP", "DOWN"]:
    rows = []
    for b in sorted(price_groups.keys()):
        group = [t for t in price_groups[b] if t["side"] == side]
        if not group:
            continue
        s = calc_stats(group)
        rows.append([b, s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}"])
    if rows:
        print_table(f"3b. PRICE BUCKET - {side} ONLY", rows, ["Bucket", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL"])

# ========================================================================
#  4. WIN RATE BY TIME REMAINING AT ENTRY
# ========================================================================
def time_bucket(minutes_remaining):
    if minutes_remaining <= 3: return "A: 0-3 min"
    if minutes_remaining <= 5: return "B: 3-5 min"
    if minutes_remaining <= 7: return "C: 5-7 min"
    if minutes_remaining <= 9: return "D: 7-9 min"
    if minutes_remaining <= 11: return "E: 9-11 min"
    if minutes_remaining <= 13: return "F: 11-13 min"
    return "G: 13-15 min"

time_groups = defaultdict(list)
for t in trades:
    minute_of_quarter = t["entry_time"].minute % 15
    tr = 15 - minute_of_quarter
    time_groups[time_bucket(tr)].append(t)

rows = []
for b in sorted(time_groups.keys()):
    s = calc_stats(time_groups[b])
    rows.append([b, s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}"])
print_table("4. WIN RATE BY TIME REMAINING AT ENTRY", rows, ["Time Left", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL"])

# ========================================================================
#  5. WIN RATE BY HOUR UTC
# ========================================================================
hour_groups = defaultdict(list)
for t in trades:
    hour_groups[t["hour"]].append(t)

rows = []
for h in sorted(hour_groups.keys()):
    s = calc_stats(hour_groups[h])
    rows.append([f"{h:02d}:00 UTC", s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}"])
print_table("5. WIN RATE BY HOUR UTC", rows, ["Hour", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL"])

# Best/worst hours (min 3 trades)
qualifying_hours = {h: g for h, g in hour_groups.items() if len(g) >= 3}
if qualifying_hours:
    best_hour = max(qualifying_hours.keys(), key=lambda h: calc_stats(qualifying_hours[h])["WR%"])
    worst_hour = min(qualifying_hours.keys(), key=lambda h: calc_stats(qualifying_hours[h])["WR%"])
    bs = calc_stats(qualifying_hours[best_hour])
    ws = calc_stats(qualifying_hours[worst_hour])
    print(f"\n  BEST hour (3+ trades): {best_hour:02d}:00 UTC - {bs['WR%']}% WR, {bs['N']} trades, ${bs['PnL']} PnL")
    print(f"  WORST hour (3+ trades): {worst_hour:02d}:00 UTC - {ws['WR%']}% WR, {ws['N']} trades, ${ws['PnL']} PnL")

# ========================================================================
#  6. WIN RATE BY MODEL CONFIDENCE BUCKET
# ========================================================================
def conf_bucket(c):
    if c is None: return None
    c_pct = c * 100 if c <= 1 else c
    if c_pct < 55: return "A: <55%"
    if c_pct < 60: return "B: 55-60%"
    if c_pct < 65: return "C: 60-65%"
    if c_pct < 70: return "D: 65-70%"
    if c_pct < 75: return "E: 70-75%"
    if c_pct < 80: return "F: 75-80%"
    return "G: 80%+"

conf_groups = defaultdict(list)
has_conf = 0
for t in trades:
    b = conf_bucket(t["model_confidence"])
    if b is not None:
        conf_groups[b].append(t)
        has_conf += 1

if conf_groups:
    rows = []
    for b in sorted(conf_groups.keys()):
        s = calc_stats(conf_groups[b])
        rows.append([b, s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}"])
    print_table(f"6. WIN RATE BY MODEL CONFIDENCE ({has_conf} trades have data)", rows, ["Confidence", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL"])
else:
    print(f"\n{'='*90}")
    print(f"  6. WIN RATE BY MODEL CONFIDENCE")
    print(f"{'='*90}")
    print(f"  NOTE: {len(trades) - has_conf} trades have NO model_confidence field.")
    print(f"  Using edge_at_entry as proxy for model confidence instead.")
    # Use edge as proxy
    def edge_conf_bucket(e):
        e_pct = e * 100
        if e_pct < 15: return "A: <15%"
        if e_pct < 25: return "B: 15-25%"
        if e_pct < 35: return "C: 25-35%"
        if e_pct < 45: return "D: 35-45%"
        if e_pct < 55: return "E: 45-55%"
        return "F: 55%+"

    econf_groups = defaultdict(list)
    for t in trades:
        econf_groups[edge_conf_bucket(t["edge"])].append(t)
    rows = []
    for b in sorted(econf_groups.keys()):
        s = calc_stats(econf_groups[b])
        rows.append([b, s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}"])
    print_table(f"6. EDGE_AT_ENTRY AS PROXY FOR CONFIDENCE", rows, ["Edge Bucket", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL"])

# ========================================================================
#  7. WIN RATE BY SIGNAL STRENGTH
# ========================================================================
sig_groups = defaultdict(list)
for t in trades:
    sig_groups[t["signal_strength"] or "none"].append(t)

rows = []
for sig in sorted(sig_groups.keys()):
    s = calc_stats(sig_groups[sig])
    rows.append([sig, s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}", f"${s['Avg_Win']}", f"${s['Avg_Loss']}"])
print_table("7. WIN RATE BY SIGNAL STRENGTH", rows, ["Strength", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL", "Avg Win", "Avg Loss"])

# ========================================================================
#  8. AVERAGE PnL PER TRADE (already in every bucket above - this is the summary)
# ========================================================================
print(f"\n{'='*90}")
print(f"  8. PnL DISTRIBUTION SUMMARY")
print(f"{'='*90}")
pnls = sorted([t["pnl"] for t in trades])
win_pnls = [p for p in pnls if p > 0]
loss_pnls = [p for p in pnls if p <= 0]
print(f"  Min PnL:       ${min(pnls):.2f}")
print(f"  Max PnL:       ${max(pnls):.2f}")
print(f"  Mean PnL:      ${statistics.mean(pnls):.2f}")
print(f"  Median PnL:    ${statistics.median(pnls):.2f}")
if len(pnls) >= 2:
    print(f"  Std Dev PnL:   ${statistics.stdev(pnls):.2f}")
print(f"  Mean Win:      ${statistics.mean(win_pnls):.2f}" if win_pnls else "  Mean Win: N/A")
print(f"  Mean Loss:     ${statistics.mean(loss_pnls):.2f}" if loss_pnls else "  Mean Loss: N/A")
print(f"  Median Win:    ${statistics.median(win_pnls):.2f}" if win_pnls else "  Median Win: N/A")
print(f"  Median Loss:   ${statistics.median(loss_pnls):.2f}" if loss_pnls else "  Median Loss: N/A")
big_wins = [p for p in pnls if p > 15]
big_losses = [p for p in pnls if p < -5]
print(f"  Big wins (>$15):   {len(big_wins)} trades, total ${sum(big_wins):.2f}")
print(f"  Big losses (<-$5): {len(big_losses)} trades, total ${sum(big_losses):.2f}")
if win_pnls and loss_pnls:
    pf = sum(win_pnls) / abs(sum(loss_pnls))
    print(f"  Profit factor:     {pf:.2f}")

# ========================================================================
#  9. WIN RATE BY WITH_TREND
# ========================================================================
trend_groups = defaultdict(list)
trend_count = 0
for t in trades:
    wt = t["with_trend"]
    if wt is True or wt == 1 or wt == "true" or wt == "True":
        trend_groups["WITH_TREND"].append(t)
        trend_count += 1
    elif wt is False or wt == 0 or wt == "false" or wt == "False":
        trend_groups["COUNTER_TREND"].append(t)
        trend_count += 1
    # Also check features for other trend keys
    feats = t.get("features", {})
    for fk in feats:
        if "trend" in fk.lower() and fk != "_with_trend":
            pass  # just noting existence

if trend_groups:
    rows = []
    for label in ["WITH_TREND", "COUNTER_TREND"]:
        if label in trend_groups:
            s = calc_stats(trend_groups[label])
            rows.append([label, s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}", f"${s['Avg_Win']}", f"${s['Avg_Loss']}"])
    no_trend = len(trades) - trend_count
    print_table(f"9. WIN RATE BY WITH_TREND ({trend_count} trades have data, {no_trend} missing)", rows,
                ["Trend", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL", "Avg Win", "Avg Loss"])
else:
    print(f"\n{'='*90}")
    print(f"  9. WIN RATE BY WITH_TREND")
    print(f"{'='*90}")
    print(f"  NO with_trend data found in features or top-level fields.")
    print(f"  Checking all available feature keys...")
    all_feature_keys = set()
    for t in trades:
        all_feature_keys.update(t["features"].keys())
    if all_feature_keys:
        print(f"  Available feature keys: {sorted(all_feature_keys)}")
    else:
        print(f"  No features dict found on any trades.")

# ========================================================================
#  10. WIN RATE BY EDGE AT ENTRY BUCKET
# ========================================================================
def edge_bucket(e):
    e_pct = e * 100
    if e_pct < 10: return "A: <10%"
    if e_pct < 20: return "B: 10-20%"
    if e_pct < 30: return "C: 20-30%"
    if e_pct < 40: return "D: 30-40%"
    if e_pct < 50: return "E: 40-50%"
    return "F: 50%+"

edge_groups = defaultdict(list)
for t in trades:
    edge_groups[edge_bucket(t["edge"])].append(t)

rows = []
for b in sorted(edge_groups.keys()):
    s = calc_stats(edge_groups[b])
    rows.append([b, s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}", f"${s['Avg_Win']}", f"${s['Avg_Loss']}"])
print_table("10. WIN RATE BY EDGE AT ENTRY", rows, ["Edge Bucket", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL", "Avg Win", "Avg Loss"])

# ========================================================================
#  11. WIN RATE BY KL DIVERGENCE BUCKET
# ========================================================================
def kl_bucket(k):
    if k < 0.05: return "A: <0.05"
    if k < 0.10: return "B: 0.05-0.10"
    if k < 0.20: return "C: 0.10-0.20"
    if k < 0.30: return "D: 0.20-0.30"
    if k < 0.40: return "E: 0.30-0.40"
    if k < 0.50: return "F: 0.40-0.50"
    return "G: 0.50+"

kl_groups = defaultdict(list)
for t in trades:
    kl_groups[kl_bucket(t["kl"])].append(t)

rows = []
for b in sorted(kl_groups.keys()):
    s = calc_stats(kl_groups[b])
    rows.append([b, s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}", f"${s['Avg_Win']}", f"${s['Avg_Loss']}"])
print_table("11. WIN RATE BY KL DIVERGENCE", rows, ["KL Bucket", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL", "Avg Win", "Avg Loss"])

# ========================================================================
#  12. ADDITIONAL PATTERNS
# ========================================================================

# 12a. Asset x Side cross-tab
rows = []
for asset in ["BTC", "ETH", "SOL"]:
    for side in ["UP", "DOWN"]:
        group = [t for t in trades if t["asset"] == asset and t["side"] == side]
        if not group:
            continue
        s = calc_stats(group)
        rows.append([f"{asset}_{side}", s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}"])
print_table("12a. ASSET x SIDE CROSS-TAB", rows, ["Combo", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL"])

# 12b. Win rate by date
date_groups = defaultdict(list)
for t in trades:
    date_groups[t["entry_time"].strftime("%Y-%m-%d")].append(t)

rows = []
for d in sorted(date_groups.keys()):
    s = calc_stats(date_groups[d])
    rows.append([d, s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}"])
print_table("12b. WIN RATE BY DATE", rows, ["Date", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL"])

# 12c. Streak analysis
print(f"\n{'='*90}")
print(f"  12c. STREAK ANALYSIS")
print(f"{'='*90}")
sorted_trades = sorted(trades, key=lambda t: t["entry_time"])
max_win_streak = 0
max_loss_streak = 0
cur_win = 0
cur_loss = 0
for t in sorted_trades:
    if t["win"]:
        cur_win += 1
        cur_loss = 0
        max_win_streak = max(max_win_streak, cur_win)
    else:
        cur_loss += 1
        cur_win = 0
        max_loss_streak = max(max_loss_streak, cur_loss)
print(f"  Max consecutive wins:   {max_win_streak}")
print(f"  Max consecutive losses: {max_loss_streak}")

# 12d. Cumulative PnL progression
print(f"\n  Cumulative PnL progression (every 10 trades):")
cum_pnl = 0
for i, t in enumerate(sorted_trades):
    cum_pnl += t["pnl"]
    if (i + 1) % 10 == 0 or i == len(sorted_trades) - 1:
        wr_so_far = sum(tt["win"] for tt in sorted_trades[:i+1]) / (i+1) * 100
        print(f"    Trade {i+1:>4}: Cumulative PnL = ${cum_pnl:>8.2f} | Running WR = {wr_so_far:.1f}%")

# 12e. Correlations
print(f"\n{'='*90}")
print(f"  12e. CORRELATIONS")
print(f"{'='*90}")

def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx)**2 for x in xs) ** 0.5
    dy = sum((y - my)**2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)

edge_vals = [t["edge"] for t in trades]
win_vals = [t["win"] for t in trades]
pnl_vals = [t["pnl"] for t in trades]
kl_vals = [t["kl"] for t in trades]
price_vals = [t["entry_price"] for t in trades]

r = pearson(edge_vals, win_vals)
print(f"  edge_at_entry vs win:       r = {r:.4f}" if r else "  edge_at_entry vs win:       insufficient data")
r = pearson(edge_vals, pnl_vals)
print(f"  edge_at_entry vs PnL:       r = {r:.4f}" if r else "  edge_at_entry vs PnL:       insufficient data")
r = pearson(kl_vals, win_vals)
print(f"  kl_divergence vs win:       r = {r:.4f}" if r else "  kl_divergence vs win:       insufficient data")
r = pearson(kl_vals, pnl_vals)
print(f"  kl_divergence vs PnL:       r = {r:.4f}" if r else "  kl_divergence vs PnL:       insufficient data")
r = pearson(price_vals, win_vals)
print(f"  entry_price vs win:         r = {r:.4f}" if r else "  entry_price vs win:         insufficient data")
r = pearson(price_vals, pnl_vals)
print(f"  entry_price vs PnL:         r = {r:.4f}" if r else "  entry_price vs PnL:         insufficient data")

# If model_confidence data exists
conf_vals = [t["model_confidence"] for t in trades if t["model_confidence"] is not None]
if len(conf_vals) >= 3:
    conf_wins = [t["win"] for t in trades if t["model_confidence"] is not None]
    conf_pnls = [t["pnl"] for t in trades if t["model_confidence"] is not None]
    r = pearson(conf_vals, conf_wins)
    print(f"  model_confidence vs win:    r = {r:.4f}" if r else "  model_confidence vs win:    insufficient data")
    r = pearson(conf_vals, conf_pnls)
    print(f"  model_confidence vs PnL:    r = {r:.4f}" if r else "  model_confidence vs PnL:    insufficient data")

# 12f. Best and worst combos (side + asset + price bucket)
print(f"\n{'='*90}")
print(f"  12f. BEST AND WORST COMBOS (Side-Asset-PriceBucket)")
print(f"{'='*90}")

all_combos = defaultdict(list)
for t in trades:
    combo = f"{t['side']}-{t['asset']}-{price_bucket(t['entry_price'])[3:]}"
    all_combos[combo].append(t)

print(f"\n  TOP 10 MOST PROFITABLE COMBOS (by total PnL):")
for combo, group in sorted(all_combos.items(), key=lambda x: -sum(t['pnl'] for t in x[1]))[:10]:
    s = calc_stats(group)
    print(f"    {combo:<35} {s['N']:>3} trades | {s['WR%']:>5.1f}% WR | ${s['PnL']:>8.2f} PnL | ${s['Avg_PnL']:>6.2f} avg")

print(f"\n  TOP 10 WORST COMBOS (by total PnL):")
for combo, group in sorted(all_combos.items(), key=lambda x: sum(t['pnl'] for t in x[1]))[:10]:
    s = calc_stats(group)
    print(f"    {combo:<35} {s['N']:>3} trades | {s['WR%']:>5.1f}% WR | ${s['PnL']:>8.2f} PnL | ${s['Avg_PnL']:>6.2f} avg")

# 12g. Top 10 best and worst individual trades
print(f"\n{'='*90}")
print(f"  12g. TOP 10 BEST TRADES")
print(f"{'='*90}")
sorted_by_pnl = sorted(trades, key=lambda t: t["pnl"], reverse=True)
for i, t in enumerate(sorted_by_pnl[:10]):
    print(f"  #{i+1:>2}: {t['asset']} {t['side']:<4} @ ${t['entry_price']:.3f} -> ${t['exit_price']:.3f} | PnL: ${t['pnl']:>7.2f} | Edge: {t['edge']:.3f} | KL: {t['kl']:.3f} | Sig: {t['signal_strength']:<6} | {t['entry_time'].strftime('%m-%d %H:%M')}")

print(f"\n{'='*90}")
print(f"  12h. TOP 10 WORST TRADES")
print(f"{'='*90}")
for i, t in enumerate(sorted_by_pnl[-10:][::-1]):
    print(f"  #{i+1:>2}: {t['asset']} {t['side']:<4} @ ${t['entry_price']:.3f} -> ${t['exit_price']:.3f} | PnL: ${t['pnl']:>7.2f} | Edge: {t['edge']:.3f} | KL: {t['kl']:.3f} | Sig: {t['signal_strength']:<6} | {t['entry_time'].strftime('%m-%d %H:%M')}")

# 12i. Kelly fraction analysis
kelly_groups = defaultdict(list)
for t in trades:
    kf = t["kelly_fraction"]
    if kf <= 0:
        kelly_groups["A: 0%"].append(t)
    elif kf < 0.10:
        kelly_groups["B: 0-10%"].append(t)
    elif kf < 0.15:
        kelly_groups["C: 10-15%"].append(t)
    elif kf < 0.20:
        kelly_groups["D: 15-20%"].append(t)
    elif kf < 0.25:
        kelly_groups["E: 20-25%"].append(t)
    else:
        kelly_groups["F: 25%"].append(t)

rows = []
for b in sorted(kelly_groups.keys()):
    s = calc_stats(kelly_groups[b])
    rows.append([b, s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}"])
print_table("12i. WIN RATE BY KELLY FRACTION", rows, ["Kelly", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL"])

# 12j. Position size analysis
size_groups = defaultdict(list)
for t in trades:
    size_groups[f"${t['size_usd']:.0f}"].append(t)

rows = []
for b in sorted(size_groups.keys()):
    s = calc_stats(size_groups[b])
    rows.append([b, s["N"], s["Wins"], s["Losses"], f"{s['WR%']}%", f"${s['PnL']}", f"${s['Avg_PnL']}"])
print_table("12j. WIN RATE BY POSITION SIZE", rows, ["Size", "Trades", "Wins", "Losses", "WR%", "Total PnL", "Avg PnL"])

# 12k. Feature keys inventory
print(f"\n{'='*90}")
print(f"  12k. ALL FEATURE KEYS FOUND IN TRADES")
print(f"{'='*90}")
all_feature_keys = set()
for t in trades:
    all_feature_keys.update(t["features"].keys())
if all_feature_keys:
    for k in sorted(all_feature_keys):
        vals = [t["features"].get(k) for t in trades if k in t["features"]]
        sample = vals[0] if vals else "N/A"
        if isinstance(sample, float):
            sample = f"{sample:.4f}"
        print(f"  {k:<30} {len(vals):>4} trades | sample: {sample}")
else:
    print(f"  No features dict found in trades.")

# 12l. Entry price vs exit price (did we buy cheap and sell high?)
print(f"\n{'='*90}")
print(f"  12l. PRICE MOVEMENT ANALYSIS")
print(f"{'='*90}")
up_moves = [t for t in trades if t["exit_price"] > t["entry_price"]]
down_moves = [t for t in trades if t["exit_price"] < t["entry_price"]]
flat = [t for t in trades if t["exit_price"] == t["entry_price"]]
print(f"  Price went UP from entry to exit:   {len(up_moves)} trades ({100*len(up_moves)/len(trades):.1f}%)")
print(f"  Price went DOWN from entry to exit: {len(down_moves)} trades ({100*len(down_moves)/len(trades):.1f}%)")
print(f"  Price was FLAT:                     {len(flat)} trades ({100*len(flat)/len(trades):.1f}%)")
# For DOWN trades, price going UP = win; for UP trades, price going UP = also means win...
# Actually for prediction markets: you buy a contract and if it resolves to $1 you win
# So for DOWN, if price goes from 0.30 to 0.95 = big win (DOWN resolved correctly)
# For UP, if price goes from 0.30 to 0.95 = big win (UP resolved correctly)
avg_price_change = statistics.mean([t["exit_price"] - t["entry_price"] for t in trades])
print(f"  Avg price change (exit-entry):      ${avg_price_change:.4f}")

# ========================================================================
#  EXECUTIVE SUMMARY
# ========================================================================
print(f"\n{'='*90}")
print(f"  EXECUTIVE SUMMARY")
print(f"{'='*90}")
total_invested = sum(t["size_usd"] for t in trades)
wins_list = [t for t in trades if t["win"] == 1]
losses_list = [t for t in trades if t["win"] == 0]
print(f"\n  Total closed trades:    {len(trades)}")
print(f"  Win/Loss:               {total_wins}W / {total_losses}L")
print(f"  Win Rate:               {100*total_wins/len(trades):.1f}%")
print(f"  Total PnL:              ${total_pnl:.2f}")
print(f"  Capital deployed:       ${total_invested:.2f}")
print(f"  ROI on capital:         {100*total_pnl/total_invested:.2f}%")
if wins_list:
    print(f"  Avg winning trade:      ${statistics.mean([t['pnl'] for t in wins_list]):.2f}")
if losses_list:
    print(f"  Avg losing trade:       ${statistics.mean([t['pnl'] for t in losses_list]):.2f}")
if wins_list and losses_list:
    avg_w = statistics.mean([t['pnl'] for t in wins_list])
    avg_l = abs(statistics.mean([t['pnl'] for t in losses_list]))
    print(f"  Win/Loss ratio:         {avg_w/avg_l:.2f}x")
    pf = sum(t['pnl'] for t in wins_list) / abs(sum(t['pnl'] for t in losses_list))
    print(f"  Profit factor:          {pf:.2f}")
print(f"  Expectancy per trade:   ${total_pnl/len(trades):.2f}")
print(f"  Avg position size:      ${statistics.mean([t['size_usd'] for t in trades]):.2f}")
print()

# KEY ACTIONABLE INSIGHTS
print(f"{'='*90}")
print(f"  KEY ACTIONABLE INSIGHTS")
print(f"{'='*90}")

# Find best performing segments
best_segments = []

# Best asset
for asset, group in assets.items():
    s = calc_stats(group)
    if s["N"] >= 5:
        best_segments.append((f"Asset={asset}", s["WR%"], s["N"], s["PnL"]))

# Best side
for side, group in sides.items():
    s = calc_stats(group)
    if s["N"] >= 5:
        best_segments.append((f"Side={side}", s["WR%"], s["N"], s["PnL"]))

# Best edge bucket
for b, group in edge_groups.items():
    s = calc_stats(group)
    if s["N"] >= 3:
        best_segments.append((f"Edge={b[3:]}", s["WR%"], s["N"], s["PnL"]))

# Best price bucket
for b, group in price_groups.items():
    s = calc_stats(group)
    if s["N"] >= 3:
        best_segments.append((f"Price={b[3:]}", s["WR%"], s["N"], s["PnL"]))

# Best signal
for sig, group in sig_groups.items():
    s = calc_stats(group)
    if s["N"] >= 3:
        best_segments.append((f"Signal={sig}", s["WR%"], s["N"], s["PnL"]))

print(f"\n  HIGHEST WIN RATE SEGMENTS (3+ trades):")
for seg, wr, n, pnl in sorted(best_segments, key=lambda x: -x[1])[:10]:
    print(f"    {seg:<35} {wr:>5.1f}% WR | {n:>3} trades | ${pnl:>8.2f} PnL")

print(f"\n  LOWEST WIN RATE SEGMENTS (3+ trades):")
for seg, wr, n, pnl in sorted(best_segments, key=lambda x: x[1])[:10]:
    print(f"    {seg:<35} {wr:>5.1f}% WR | {n:>3} trades | ${pnl:>8.2f} PnL")

print(f"\n  HIGHEST PnL SEGMENTS:")
for seg, wr, n, pnl in sorted(best_segments, key=lambda x: -x[3])[:10]:
    print(f"    {seg:<35} {wr:>5.1f}% WR | {n:>3} trades | ${pnl:>8.2f} PnL")
print()
