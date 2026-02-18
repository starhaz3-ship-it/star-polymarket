"""
sniper_ml_optimize.py
=====================
Trains a GradientBoostingClassifier on sniper_5m_results.json paper trades
and outputs optimal settings for the sniper bot.

Features: time_remaining_sec, entry_price, ask_gap, hour_utc, side
Target:   1 = WIN, 0 = LOSS
"""

import json
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.inspection import permutation_importance
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
DATA_PATH = "C:/Users/Star/.local/bin/star-polymarket/sniper_5m_results.json"
OUT_PATH = "C:/Users/Star/.local/bin/star-polymarket/sniper_ml_config.json"

print("=" * 70)
print("SNIPER ML OPTIMIZER")
print("=" * 70)

with open(DATA_PATH) as f:
    raw = json.load(f)

resolved = raw.get("resolved", [])
# Keep only trades with a WIN/LOSS result
trades = [t for t in resolved if t.get("result") in ("WIN", "LOSS")]
print(f"\nLoaded {len(resolved)} resolved trades, {len(trades)} with WIN/LOSS result")

if len(trades) < 10:
    print("ERROR: Not enough resolved trades for ML (need >=10). Exiting.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Feature extraction
# ---------------------------------------------------------------------------
X_rows = []
y_labels = []
meta = []  # keep raw row for window simulation

for t in trades:
    entry_dt = datetime.fromisoformat(t["entry_time"])
    hour_utc = entry_dt.hour
    side_enc = 1 if t["side"] == "UP" else 0
    ask_gap = abs(t["up_ask"] - t["down_ask"])

    X_rows.append([
        t["time_remaining_sec"],   # 0
        t["entry_price"],          # 1
        ask_gap,                   # 2
        hour_utc,                  # 3
        side_enc,                  # 4
    ])
    y_labels.append(1 if t["result"] == "WIN" else 0)
    meta.append({
        "time_remaining_sec": t["time_remaining_sec"],
        "entry_price": t["entry_price"],
        "ask_gap": ask_gap,
        "hour_utc": hour_utc,
        "side": side_enc,
        "pnl": t["pnl"],
        "result": t["result"],
    })

X = np.array(X_rows, dtype=float)
y = np.array(y_labels, dtype=int)

FEATURE_NAMES = ["time_remaining_sec", "entry_price", "ask_gap", "hour_utc", "side"]

wins = int(y.sum())
losses = int(len(y) - wins)
wr_overall = wins / len(y) * 100
print(f"  Wins: {wins}  Losses: {losses}  WR: {wr_overall:.1f}%")

# ---------------------------------------------------------------------------
# 3. Train GradientBoostingClassifier
# ---------------------------------------------------------------------------
print("\n--- Training GradientBoostingClassifier ---")

clf = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)
clf.fit(X, y)

# Cross-val with leave-one-out style (small dataset)
try:
    cv_scores = cross_val_score(clf, X, y, cv=min(5, len(trades) // 5), scoring="accuracy")
    print(f"  CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
except Exception as e:
    print(f"  CV skipped ({e})")

train_acc = (clf.predict(X) == y).mean()
print(f"  Train accuracy: {train_acc:.3f}")

# ---------------------------------------------------------------------------
# 4. Feature importances
# ---------------------------------------------------------------------------
print("\n--- Feature Importances ---")
importances = clf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
for rank, idx in enumerate(sorted_idx, 1):
    print(f"  {rank}. {FEATURE_NAMES[idx]:<22} {importances[idx]:.4f}  ({importances[idx]*100:.1f}%)")

# ---------------------------------------------------------------------------
# 5. SNIPER_WINDOW simulation
# ---------------------------------------------------------------------------
print("\n--- SNIPER_WINDOW Simulation ---")
print(f"  {'Window':>8} | {'Trades':>6} | {'Wins':>5} | {'Losses':>6} | {'WR%':>6} | {'PnL':>8}")
print("  " + "-" * 55)

window_results = {}
for W in [61, 62, 63, 64, 65]:
    subset = [m for m in meta if m["time_remaining_sec"] <= W]
    if len(subset) == 0:
        continue
    w_wins = sum(1 for m in subset if m["result"] == "WIN")
    w_losses = sum(1 for m in subset if m["result"] == "LOSS")
    w_wr = w_wins / len(subset) * 100 if subset else 0
    w_pnl = sum(m["pnl"] for m in subset)
    window_results[W] = {
        "trades": len(subset), "wins": w_wins, "losses": w_losses,
        "wr": w_wr, "pnl": w_pnl
    }
    print(f"  W={W:>3}s   | {len(subset):>6} | {w_wins:>5} | {w_losses:>6} | {w_wr:>5.1f}% | ${w_pnl:>7.2f}")

# Best window by PnL/trade ratio
best_window = max(window_results, key=lambda w: window_results[w]["wr"])
print(f"\n  Best window by WR: {best_window}s  (WR={window_results[best_window]['wr']:.1f}%)")

# ---------------------------------------------------------------------------
# 6. Optimal entry timing via model probability scan
# ---------------------------------------------------------------------------
print("\n--- Entry Timing Sweet Spot (model probability scan) ---")

# Grid-search time_remaining_sec 51..65 with typical feature values
# Use median entry_price and ask_gap from wins
win_meta = [m for m in meta if m["result"] == "WIN"]
med_price = float(np.median([m["entry_price"] for m in win_meta]))
med_gap = float(np.median([m["ask_gap"] for m in win_meta]))
med_hour = float(np.median([m["hour_utc"] for m in win_meta]))

print(f"  Median win: price={med_price:.2f}  gap={med_gap:.3f}  hour={med_hour:.0f}")
print(f"\n  {'time_rem':>8} | {'P(win)':>7}")
print("  " + "-" * 22)

time_probs = []
for t_rem in np.arange(51.0, 66.0, 0.5):
    feat = np.array([[t_rem, med_price, med_gap, med_hour, 1]])
    prob = clf.predict_proba(feat)[0][1]
    time_probs.append((t_rem, prob))
    bar = "#" * int(prob * 30)
    print(f"  {t_rem:>7.1f}s | {prob:>6.3f}  {bar}")

# Sweet spot = contiguous range where P(win) >= 0.80
threshold = 0.80
sweet_times = [t for t, p in time_probs if p >= threshold]
if sweet_times:
    sweet_min = min(sweet_times)
    sweet_max = max(sweet_times)
    print(f"\n  Sweet spot (P>=0.80): [{sweet_min:.1f}s, {sweet_max:.1f}s]")
else:
    # Fallback — top quartile
    sorted_probs = sorted(time_probs, key=lambda x: x[1], reverse=True)
    top_times = [t for t, p in sorted_probs[:len(sorted_probs)//4 + 1]]
    sweet_min, sweet_max = min(top_times), max(top_times)
    print(f"\n  Sweet spot (top quartile): [{sweet_min:.1f}s, {sweet_max:.1f}s]")

# ---------------------------------------------------------------------------
# 7. Dangerous hours, prices, and gap threshold
# ---------------------------------------------------------------------------
print("\n--- Danger Zone Analysis ---")

# Danger hours: WR < 75% with >=3 trades
hour_stats = defaultdict(lambda: {"W": 0, "L": 0, "pnl": 0.0})
for m in meta:
    hour_stats[m["hour_utc"]]["W" if m["result"] == "WIN" else "L"] += 1
    hour_stats[m["hour_utc"]]["pnl"] += m["pnl"]

danger_hours = []
print("  Hour stats:")
for h in sorted(hour_stats.keys()):
    s = hour_stats[h]
    total = s["W"] + s["L"]
    wr = s["W"] / total * 100 if total else 0
    flag = " <-- DANGER" if wr < 75 and total >= 3 else ""
    print(f"    UTC{h:02d}: {total:>3} trades  WR={wr:>5.1f}%  PnL=${s['pnl']:>6.2f}{flag}")
    if wr < 75 and total >= 3:
        danger_hours.append(h)

# Danger prices: entry prices that only appear in losses
loss_prices = set(round(m["entry_price"], 2) for m in meta if m["result"] == "LOSS")
win_prices = set(round(m["entry_price"], 2) for m in meta if m["result"] == "WIN")
pure_loss_prices = sorted(loss_prices - win_prices)
mixed_bad_prices = sorted(p for p in (loss_prices & win_prices)
                          if sum(1 for m in meta if round(m["entry_price"], 2) == p and m["result"] == "LOSS") >
                             sum(1 for m in meta if round(m["entry_price"], 2) == p and m["result"] == "WIN"))

danger_prices = sorted(set(pure_loss_prices + mixed_bad_prices))
print(f"\n  Pure loss prices: {pure_loss_prices}")
print(f"  Mixed but net-loss prices: {mixed_bad_prices}")
print(f"  DANGER_PRICES: {danger_prices}")

# Min ask gap: find threshold that cuts most losses while keeping most wins
print("\n  Ask gap threshold analysis:")
best_gap_threshold = 0.0
best_gap_score = -999.0
for gap_thresh in np.arange(0.0, 0.75, 0.05):
    filtered = [m for m in meta if m["ask_gap"] >= gap_thresh]
    if len(filtered) == 0:
        continue
    f_wins = sum(1 for m in filtered if m["result"] == "WIN")
    f_losses = sum(1 for m in filtered if m["result"] == "LOSS")
    f_wr = f_wins / len(filtered) * 100 if filtered else 0
    f_pnl = sum(m["pnl"] for m in filtered)
    lost_wins = wins - f_wins
    eliminated_losses = losses - f_losses
    # Score: reward loss elimination, penalise win loss
    score = eliminated_losses * 2.5 - lost_wins * 0.3
    flag = " <-- BEST" if score > best_gap_score else ""
    if score > best_gap_score:
        best_gap_score = score
        best_gap_threshold = gap_thresh
    print(f"    gap>={gap_thresh:.2f}: {len(filtered):>3} trades  WR={f_wr:>5.1f}%"
          f"  PnL=${f_pnl:>6.2f}  cut_losses={eliminated_losses}  cut_wins={lost_wins}{flag}")

print(f"\n  Optimal MIN_ASK_GAP: {best_gap_threshold:.2f}")

# ---------------------------------------------------------------------------
# 8. Per-bucket time analysis (1s bins) for reporting
# ---------------------------------------------------------------------------
print("\n--- Time Bucket Detail (1s bins) ---")
time_buckets = defaultdict(lambda: {"W": 0, "L": 0, "pnl": 0.0})
for m in meta:
    b = int(m["time_remaining_sec"])
    time_buckets[b]["W" if m["result"] == "WIN" else "L"] += 1
    time_buckets[b]["pnl"] += m["pnl"]

for b in sorted(time_buckets.keys()):
    s = time_buckets[b]
    total = s["W"] + s["L"]
    wr = s["W"] / total * 100
    bar = "W" * s["W"] + "L" * s["L"]
    print(f"  {b}s: {bar:<25} WR={wr:>5.1f}%  PnL=${s['pnl']:>6.2f}")

# ---------------------------------------------------------------------------
# 9. Build final config
# ---------------------------------------------------------------------------
print("\n--- Recommended Config ---")

# Use the empirically best SNIPER_WINDOW (best WR among windows)
rec_window = best_window

# Entry sweet spot: use model-derived range
rec_sweet_min = sweet_min
rec_sweet_max = sweet_max

# If model can't distinguish well (small data), trust empirical 1s-bin analysis
# Sweet spots from data: 55s and 57s bins are 100% WR
empirical_sweet = [b for b in sorted(time_buckets.keys())
                   if time_buckets[b]["L"] == 0 and time_buckets[b]["W"] >= 5]
if empirical_sweet:
    emp_min = float(min(empirical_sweet))
    emp_max = float(max(empirical_sweet)) + 0.9  # include full second
    print(f"  Empirical 0-loss buckets: {empirical_sweet}s  range=[{emp_min},{emp_max:.1f}]")

# Final sweet spot = intersection or union, bias toward empirical on small data
# Use empirical if it makes sense (covers enough trades)
if empirical_sweet:
    rec_sweet_min = emp_min
    rec_sweet_max = emp_max

# Recalculate SNIPER_WINDOW to be rec_sweet_max + 1 (ceiling)
rec_window = int(rec_sweet_max) + 1

config = {
    "SNIPER_WINDOW": rec_window,
    "ENTRY_SWEET_SPOT_MIN": round(rec_sweet_min, 1),
    "ENTRY_SWEET_SPOT_MAX": round(rec_sweet_max, 1),
    "SKIP_HOURS_UTC": sorted(danger_hours),
    "DANGER_PRICES": [round(p, 2) for p in danger_prices],
    "MIN_ASK_GAP": round(best_gap_threshold, 2),
    "_meta": {
        "generated": datetime.utcnow().isoformat() + "Z",
        "n_trades": len(trades),
        "overall_wr_pct": round(wr_overall, 1),
        "train_accuracy": round(float(train_acc), 4),
        "feature_importances": {
            FEATURE_NAMES[i]: round(float(importances[i]), 4)
            for i in range(len(FEATURE_NAMES))
        },
    }
}

print(f"\n  SNIPER_WINDOW         : {config['SNIPER_WINDOW']}s")
print(f"  ENTRY_SWEET_SPOT_MIN  : {config['ENTRY_SWEET_SPOT_MIN']}s (most time left)")
print(f"  ENTRY_SWEET_SPOT_MAX  : {config['ENTRY_SWEET_SPOT_MAX']}s (least time left)")
print(f"  SKIP_HOURS_UTC        : {config['SKIP_HOURS_UTC']}")
print(f"  DANGER_PRICES         : {config['DANGER_PRICES']}")
print(f"  MIN_ASK_GAP           : {config['MIN_ASK_GAP']}")

with open(OUT_PATH, "w") as f:
    json.dump(config, f, indent=2)

print(f"\nConfig saved to: {OUT_PATH}")
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
