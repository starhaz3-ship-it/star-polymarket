"""Comprehensive paper trade analysis for Star Polymarket bot."""
import json
from datetime import datetime
from collections import defaultdict
import statistics

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
    model_conf = t.get("model_confidence", None)
    signal_strength = t.get("signal_strength", "")

    pnl = t["pnl"]
    win = 1 if pnl > 0 else 0

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
        "signal_strength": signal_strength,
        "entry_time": entry_time,
        "size_usd": t.get("size_usd", 10),
        "kelly_fraction": t.get("kelly_fraction", 0),
    })

print(f"Total closed trades: {len(trades)}")
print(f"Overall WR: {sum(t['win'] for t in trades)}/{len(trades)} = {sum(t['win'] for t in trades)/len(trades)*100:.1f}%")
print(f"Overall PnL: ${sum(t['pnl'] for t in trades):.2f}")
print(f"Date range: {min(t['entry_time'] for t in trades).strftime('%Y-%m-%d %H:%M')} to {max(t['entry_time'] for t in trades).strftime('%Y-%m-%d %H:%M')} UTC")
print()

# Helper
def calc_stats(group):
    n = len(group)
    if n == 0:
        return {"N": 0, "Wins": 0, "WR%": 0, "PnL": 0, "Avg_PnL": 0, "Avg_Win": 0, "Avg_Loss": 0}
    wins = sum(t["win"] for t in group)
    total_pnl = sum(t["pnl"] for t in group)
    avg_pnl = total_pnl / n
    win_pnls = [t["pnl"] for t in group if t["win"] == 1]
    loss_pnls = [t["pnl"] for t in group if t["win"] == 0]
    avg_win = statistics.mean(win_pnls) if win_pnls else 0
    avg_loss = statistics.mean(loss_pnls) if loss_pnls else 0
    return {
        "N": n,
        "Wins": wins,
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
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    header_line = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))))

# ========== 1. BY SIDE ==========
sides = defaultdict(list)
for t in trades:
    sides[t["side"]].append(t)

rows = []
for side in ["UP", "DOWN"]:
    s = calc_stats(sides[side])
    rows.append([side, s["N"], s["Wins"], s["WR%"], f"${s['PnL']}", f"${s['Avg_PnL']}", f"${s['Avg_Win']}", f"${s['Avg_Loss']}"])
print_table("1. BY SIDE", rows, ["Side", "N", "Wins", "WR%", "Total PnL", "Avg PnL", "Avg Win", "Avg Loss"])

# ========== 2. BY ASSET ==========
assets = defaultdict(list)
for t in trades:
    assets[t["asset"]].append(t)

rows = []
for asset in ["BTC", "ETH", "SOL", "OTHER"]:
    if asset not in assets:
        continue
    s = calc_stats(assets[asset])
    rows.append([asset, s["N"], s["Wins"], s["WR%"], f"${s['PnL']}", f"${s['Avg_PnL']}", f"${s['Avg_Win']}", f"${s['Avg_Loss']}"])
print_table("2. BY ASSET", rows, ["Asset", "N", "Wins", "WR%", "Total PnL", "Avg PnL", "Avg Win", "Avg Loss"])

# Also by asset + side
rows = []
for asset in ["BTC", "ETH", "SOL"]:
    for side in ["UP", "DOWN"]:
        group = [t for t in trades if t["asset"] == asset and t["side"] == side]
        if not group:
            continue
        s = calc_stats(group)
        rows.append([f"{asset}-{side}", s["N"], s["Wins"], s["WR%"], f"${s['PnL']}", f"${s['Avg_PnL']}"])
print_table("2b. BY ASSET x SIDE", rows, ["Asset-Side", "N", "Wins", "WR%", "Total PnL", "Avg PnL"])

# ========== 3. BY HOUR (UTC) ==========
hours = defaultdict(list)
for t in trades:
    hours[t["hour"]].append(t)

rows = []
for h in sorted(hours.keys()):
    s = calc_stats(hours[h])
    rows.append([f"{h:02d}:00", s["N"], s["Wins"], s["WR%"], f"${s['PnL']}", f"${s['Avg_PnL']}"])
print_table("3. BY HOUR (UTC)", rows, ["Hour", "N", "Wins", "WR%", "Total PnL", "Avg PnL"])

# Best/worst hours (min 3 trades)
qualifying_hours = {h: g for h, g in hours.items() if len(g) >= 3}
if qualifying_hours:
    best_hour = max(qualifying_hours.keys(), key=lambda h: calc_stats(qualifying_hours[h])["WR%"])
    worst_hour = min(qualifying_hours.keys(), key=lambda h: calc_stats(qualifying_hours[h])["WR%"])
    print(f"\n  Best hour (3+ trades): {best_hour:02d}:00 UTC ({calc_stats(hours[best_hour])['WR%']}% WR, {calc_stats(hours[best_hour])['N']} trades)")
    print(f"  Worst hour (3+ trades): {worst_hour:02d}:00 UTC ({calc_stats(hours[worst_hour])['WR%']}% WR, {calc_stats(hours[worst_hour])['N']} trades)")

# ========== 4. BY ENTRY PRICE BUCKET ==========
def price_bucket(p):
    if p < 0.15:
        return "$0.00-0.15"
    elif p < 0.25:
        return "$0.15-0.25"
    elif p < 0.35:
        return "$0.25-0.35"
    elif p < 0.45:
        return "$0.35-0.45"
    elif p < 0.55:
        return "$0.45-0.55"
    else:
        return "$0.55+"

buckets = defaultdict(list)
for t in trades:
    buckets[price_bucket(t["entry_price"])].append(t)

bucket_order = ["$0.00-0.15", "$0.15-0.25", "$0.25-0.35", "$0.35-0.45", "$0.45-0.55", "$0.55+"]
rows = []
for b in bucket_order:
    if b not in buckets:
        continue
    s = calc_stats(buckets[b])
    rows.append([b, s["N"], s["Wins"], s["WR%"], f"${s['PnL']}", f"${s['Avg_PnL']}", f"${s['Avg_Win']}", f"${s['Avg_Loss']}"])
print_table("4. BY ENTRY PRICE BUCKET", rows, ["Bucket", "N", "Wins", "WR%", "Total PnL", "Avg PnL", "Avg Win", "Avg Loss"])

# Price bucket by side
for side in ["UP", "DOWN"]:
    rows = []
    for b in bucket_order:
        group = [t for t in buckets.get(b, []) if t["side"] == side]
        if not group:
            continue
        s = calc_stats(group)
        rows.append([b, s["N"], s["Wins"], s["WR%"], f"${s['PnL']}", f"${s['Avg_PnL']}"])
    if rows:
        print_table(f"4b. PRICE BUCKET x {side}", rows, ["Bucket", "N", "Wins", "WR%", "Total PnL", "Avg PnL"])

# ========== 5. BY EDGE QUARTILE ==========
edges = sorted([t["edge"] for t in trades])
if edges:
    q1 = edges[len(edges)//4]
    q2 = edges[len(edges)//2]
    q3 = edges[3*len(edges)//4]

    def edge_quartile(e):
        if e <= q1:
            return f"Q1 (<= {q1:.3f})"
        elif e <= q2:
            return f"Q2 ({q1:.3f}-{q2:.3f})"
        elif e <= q3:
            return f"Q3 ({q2:.3f}-{q3:.3f})"
        else:
            return f"Q4 (> {q3:.3f})"

    edge_groups = defaultdict(list)
    for t in trades:
        edge_groups[edge_quartile(t["edge"])].append(t)

    rows = []
    for q in sorted(edge_groups.keys()):
        s = calc_stats(edge_groups[q])
        rows.append([q, s["N"], s["Wins"], s["WR%"], f"${s['PnL']}", f"${s['Avg_PnL']}"])
    print_table("5. BY EDGE QUARTILE (edge_at_entry)", rows, ["Quartile", "N", "Wins", "WR%", "Total PnL", "Avg PnL"])
    print(f"  Edge range: {min(edges):.4f} to {max(edges):.4f}, median: {q2:.4f}")

# ========== 6. BY KL DIVERGENCE QUARTILE ==========
kls = sorted([t["kl"] for t in trades])
if kls:
    kq1 = kls[len(kls)//4]
    kq2 = kls[len(kls)//2]
    kq3 = kls[3*len(kls)//4]

    def kl_quartile(k):
        if k <= kq1:
            return f"Q1 (<= {kq1:.3f})"
        elif k <= kq2:
            return f"Q2 ({kq1:.3f}-{kq2:.3f})"
        elif k <= kq3:
            return f"Q3 ({kq2:.3f}-{kq3:.3f})"
        else:
            return f"Q4 (> {kq3:.3f})"

    kl_groups = defaultdict(list)
    for t in trades:
        kl_groups[kl_quartile(t["kl"])].append(t)

    rows = []
    for q in sorted(kl_groups.keys()):
        s = calc_stats(kl_groups[q])
        rows.append([q, s["N"], s["Wins"], s["WR%"], f"${s['PnL']}", f"${s['Avg_PnL']}"])
    print_table("6. BY KL DIVERGENCE QUARTILE", rows, ["Quartile", "N", "Wins", "WR%", "Total PnL", "Avg PnL"])
    print(f"  KL range: {min(kls):.4f} to {max(kls):.4f}, median: {kq2:.4f}")

# ========== 7. BY TIME REMAINING ==========
time_buckets = defaultdict(list)
for t in trades:
    et = t["entry_time"]
    minute_of_quarter = et.minute % 15
    minutes_remaining = 15 - minute_of_quarter
    if minutes_remaining <= 3:
        bucket = "0-3 min"
    elif minutes_remaining <= 5:
        bucket = "3-5 min"
    elif minutes_remaining <= 8:
        bucket = "5-8 min"
    elif minutes_remaining <= 12:
        bucket = "8-12 min"
    else:
        bucket = "12-15 min"
    time_buckets[bucket].append(t)

bucket_order_time = ["0-3 min", "3-5 min", "5-8 min", "8-12 min", "12-15 min"]
rows = []
for b in bucket_order_time:
    if b not in time_buckets:
        continue
    s = calc_stats(time_buckets[b])
    rows.append([b, s["N"], s["Wins"], s["WR%"], f"${s['PnL']}", f"${s['Avg_PnL']}"])
print_table("7. BY TIME REMAINING (minutes left in 15-min window)", rows, ["Time Left", "N", "Wins", "WR%", "Total PnL", "Avg PnL"])

# ========== 8. LOSING PATTERNS ==========
losses = [t for t in trades if t["win"] == 0]
wins_list = [t for t in trades if t["win"] == 1]
print(f"\n{'='*90}")
print(f"  8. LOSING PATTERNS ({len(losses)} losses)")
print(f"{'='*90}")

# Side distribution of losses
loss_sides = defaultdict(int)
for t in losses:
    loss_sides[t["side"]] += 1
print(f"\n  Side distribution of losses:")
for side, count in sorted(loss_sides.items()):
    total_side = len([t for t in trades if t["side"] == side])
    print(f"    {side}: {count}/{total_side} ({count/total_side*100:.1f}% loss rate)")

# Asset distribution of losses
loss_assets = defaultdict(int)
for t in losses:
    loss_assets[t["asset"]] += 1
print(f"\n  Asset distribution of losses:")
for asset, count in sorted(loss_assets.items()):
    total_asset = len([t for t in trades if t["asset"] == asset])
    print(f"    {asset}: {count}/{total_asset} ({count/total_asset*100:.1f}% loss rate)")

# Price distribution of losses
print(f"\n  Entry price of losses:")
loss_prices = [t["entry_price"] for t in losses]
print(f"    Mean: ${statistics.mean(loss_prices):.3f}")
print(f"    Median: ${statistics.median(loss_prices):.3f}")
print(f"    Min: ${min(loss_prices):.3f}, Max: ${max(loss_prices):.3f}")

# Edge of losses
print(f"\n  Edge at entry of losses:")
loss_edges = [t["edge"] for t in losses]
print(f"    Mean: {statistics.mean(loss_edges):.4f}")
print(f"    Median: {statistics.median(loss_edges):.4f}")

# Compare to wins
print(f"\n  Edge at entry of wins:")
win_edges = [t["edge"] for t in wins_list]
print(f"    Mean: {statistics.mean(win_edges):.4f}")
print(f"    Median: {statistics.median(win_edges):.4f}")

# Hour distribution of losses
loss_hours = defaultdict(int)
for t in losses:
    loss_hours[t["hour"]] += 1
print(f"\n  Hour distribution of losses (top 5):")
for h, count in sorted(loss_hours.items(), key=lambda x: -x[1])[:5]:
    total_hour = len([t for t in trades if t["hour"] == h])
    print(f"    {h:02d}:00 UTC: {count} losses / {total_hour} trades ({count/total_hour*100:.1f}% loss rate)")

# Biggest losses
print(f"\n  Top 10 biggest losses:")
sorted_losses = sorted(losses, key=lambda t: t["pnl"])
for i, t in enumerate(sorted_losses[:10]):
    print(f"    {i+1}. ${t['pnl']:.2f} | {t['side']} {t['asset']} @ ${t['entry_price']:.3f} | edge={t['edge']:.3f} | hour={t['hour']:02d} | sig={t['signal_strength']}")

# Common patterns in losses
print(f"\n  Loss pattern combinations (top 10):")
loss_combos = defaultdict(list)
for t in losses:
    combo = f"{t['side']}-{t['asset']}-{price_bucket(t['entry_price'])}"
    loss_combos[combo].append(t)
for combo, group in sorted(loss_combos.items(), key=lambda x: -len(x[1]))[:10]:
    total_pnl = sum(t["pnl"] for t in group)
    print(f"    {combo}: {len(group)} losses, ${total_pnl:.2f} PnL")

# ========== 9. WINNING PATTERNS ==========
print(f"\n{'='*90}")
print(f"  9. WINNING PATTERNS ({len(wins_list)} wins)")
print(f"{'='*90}")

# Top 10 biggest wins
print(f"\n  Top 10 biggest winners:")
sorted_wins = sorted(wins_list, key=lambda t: -t["pnl"])
for i, t in enumerate(sorted_wins[:10]):
    print(f"    {i+1}. +${t['pnl']:.2f} | {t['side']} {t['asset']} @ ${t['entry_price']:.3f} | edge={t['edge']:.3f} | hour={t['hour']:02d} | sig={t['signal_strength']}")

# Side distribution of wins
win_sides = defaultdict(int)
for t in wins_list:
    win_sides[t["side"]] += 1
print(f"\n  Side distribution of wins:")
for side, count in sorted(win_sides.items()):
    total_side = len([t for t in trades if t["side"] == side])
    print(f"    {side}: {count}/{total_side} ({count/total_side*100:.1f}% win rate)")

# Entry price of big winners (top 25%)
big_winners = sorted_wins[:max(1, len(sorted_wins)//4)]
print(f"\n  Big winners (top 25% by PnL, N={len(big_winners)}) characteristics:")
print(f"    Mean entry price: ${statistics.mean([t['entry_price'] for t in big_winners]):.3f}")
print(f"    Median entry price: ${statistics.median([t['entry_price'] for t in big_winners]):.3f}")
print(f"    Mean edge: {statistics.mean([t['edge'] for t in big_winners]):.4f}")
print(f"    Side split: UP={sum(1 for t in big_winners if t['side']=='UP')}, DOWN={sum(1 for t in big_winners if t['side']=='DOWN')}")
print(f"    Asset split: BTC={sum(1 for t in big_winners if t['asset']=='BTC')}, ETH={sum(1 for t in big_winners if t['asset']=='ETH')}, SOL={sum(1 for t in big_winners if t['asset']=='SOL')}")

# Winning combos
print(f"\n  Most profitable combos (by total PnL):")
all_combos = defaultdict(list)
for t in trades:
    combo = f"{t['side']}-{t['asset']}-{price_bucket(t['entry_price'])}"
    all_combos[combo].append(t)
for combo, group in sorted(all_combos.items(), key=lambda x: -sum(t['pnl'] for t in x[1]))[:10]:
    total_pnl = sum(t["pnl"] for t in group)
    wr = sum(t["win"] for t in group) / len(group) * 100
    print(f"    {combo}: {len(group)} trades, {wr:.0f}% WR, ${total_pnl:.2f} PnL")

# ========== 10. MODEL CONFIDENCE CORRELATION ==========
print(f"\n{'='*90}")
print(f"  10. MODEL CONFIDENCE CORRELATION")
print(f"{'='*90}")

has_conf = [t for t in trades if t["model_confidence"] is not None]
no_conf = [t for t in trades if t["model_confidence"] is None]
print(f"\n  Trades with model_confidence: {len(has_conf)}")
print(f"  Trades without model_confidence: {len(no_conf)}")

if len(has_conf) >= 10:
    confs = sorted([t["model_confidence"] for t in has_conf])
    cq1 = confs[len(confs)//4]
    cq2 = confs[len(confs)//2]
    cq3 = confs[3*len(confs)//4]

    def conf_quartile(c):
        if c <= cq1:
            return f"Q1 (<= {cq1:.3f})"
        elif c <= cq2:
            return f"Q2 ({cq1:.3f}-{cq2:.3f})"
        elif c <= cq3:
            return f"Q3 ({cq2:.3f}-{cq3:.3f})"
        else:
            return f"Q4 (> {cq3:.3f})"

    conf_groups = defaultdict(list)
    for t in has_conf:
        conf_groups[conf_quartile(t["model_confidence"])].append(t)

    rows = []
    for q in sorted(conf_groups.keys()):
        s = calc_stats(conf_groups[q])
        rows.append([q, s["N"], s["Wins"], s["WR%"], f"${s['PnL']}", f"${s['Avg_PnL']}"])
    print_table("10. MODEL CONFIDENCE QUARTILES", rows, ["Quartile", "N", "Wins", "WR%", "Total PnL", "Avg PnL"])

    # Simple correlation
    conf_vals = [t["model_confidence"] for t in has_conf]
    win_vals = [t["win"] for t in has_conf]
    mean_c = statistics.mean(conf_vals)
    mean_w = statistics.mean(win_vals)
    numerator = sum((c - mean_c) * (w - mean_w) for c, w in zip(conf_vals, win_vals))
    denom_c = sum((c - mean_c)**2 for c in conf_vals) ** 0.5
    denom_w = sum((w - mean_w)**2 for w in win_vals) ** 0.5
    if denom_c > 0 and denom_w > 0:
        corr = numerator / (denom_c * denom_w)
        print(f"\n  Pearson correlation (model_confidence vs win): {corr:.4f}")

    pnl_vals = [t["pnl"] for t in has_conf]
    mean_p = statistics.mean(pnl_vals)
    numerator_p = sum((c - mean_c) * (p - mean_p) for c, p in zip(conf_vals, pnl_vals))
    denom_p = sum((p - mean_p)**2 for p in pnl_vals) ** 0.5
    if denom_c > 0 and denom_p > 0:
        corr_p = numerator_p / (denom_c * denom_p)
        print(f"  Pearson correlation (model_confidence vs PnL): {corr_p:.4f}")
else:
    print("\n  Insufficient model_confidence data. Using edge_at_entry as proxy.")

# Always show edge correlation
print(f"\n  Edge-to-outcome correlation (all trades):")
edge_vals = [t["edge"] for t in trades]
win_vals = [t["win"] for t in trades]
mean_e = statistics.mean(edge_vals)
mean_w = statistics.mean(win_vals)
numerator = sum((e - mean_e) * (w - mean_w) for e, w in zip(edge_vals, win_vals))
denom_e = sum((e - mean_e)**2 for e in edge_vals) ** 0.5
denom_w = sum((w - mean_w)**2 for w in win_vals) ** 0.5
if denom_e > 0 and denom_w > 0:
    corr = numerator / (denom_e * denom_w)
    print(f"  Pearson correlation (edge_at_entry vs win): {corr:.4f}")

pnl_vals = [t["pnl"] for t in trades]
mean_p = statistics.mean(pnl_vals)
numerator_p = sum((e - mean_e) * (p - mean_p) for e, p in zip(edge_vals, pnl_vals))
denom_p = sum((p - mean_p)**2 for p in pnl_vals) ** 0.5
if denom_e > 0 and denom_p > 0:
    corr_p = numerator_p / (denom_e * denom_p)
    print(f"  Pearson correlation (edge_at_entry vs PnL): {corr_p:.4f}")

# KL to outcome correlation
kl_vals = [t["kl"] for t in trades]
mean_k = statistics.mean(kl_vals)
numerator_k = sum((k - mean_k) * (w - mean_w) for k, w in zip(kl_vals, win_vals))
denom_k = sum((k - mean_k)**2 for k in kl_vals) ** 0.5
if denom_k > 0 and denom_w > 0:
    corr_k = numerator_k / (denom_k * denom_w)
    print(f"  Pearson correlation (kl_divergence vs win): {corr_k:.4f}")

# ========== SIGNAL STRENGTH ANALYSIS ==========
sig_groups = defaultdict(list)
for t in trades:
    sig_groups[t["signal_strength"] or "none"].append(t)

rows = []
for sig in sorted(sig_groups.keys()):
    s = calc_stats(sig_groups[sig])
    rows.append([sig, s["N"], s["Wins"], s["WR%"], f"${s['PnL']}", f"${s['Avg_PnL']}"])
print_table("BONUS: BY SIGNAL STRENGTH", rows, ["Strength", "N", "Wins", "WR%", "Total PnL", "Avg PnL"])

# ========== SUMMARY ==========
print(f"\n{'='*90}")
print(f"  EXECUTIVE SUMMARY")
print(f"{'='*90}")

print(f"\n  Total trades: {len(trades)}")
print(f"  Overall: {sum(t['win'] for t in trades)}/{len(trades)} = {sum(t['win'] for t in trades)/len(trades)*100:.1f}% WR | ${sum(t['pnl'] for t in trades):.2f} PnL")
if wins_list:
    print(f"  Win avg: +${statistics.mean([t['pnl'] for t in wins_list]):.2f} | Loss avg: ${statistics.mean([t['pnl'] for t in losses]):.2f}" if losses else f"  Win avg: +${statistics.mean([t['pnl'] for t in wins_list]):.2f} | No losses!")
if losses and wins_list:
    pf = sum(t['pnl'] for t in wins_list) / abs(sum(t['pnl'] for t in losses))
    print(f"  Profit factor: {pf:.2f}")
print(f"  Average position size: ${statistics.mean([t['size_usd'] for t in trades]):.2f}")
print(f"  Average kelly_fraction: {statistics.mean([t['kelly_fraction'] for t in trades]):.4f}")
print()
