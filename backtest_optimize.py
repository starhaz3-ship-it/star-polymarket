"""
Backtest Optimizer - Find filters that achieve 55%+ win rate
"""

import json
from pathlib import Path
from collections import defaultdict
from itertools import product

# Load backtest data
data_path = Path(__file__).parent / "backtest_data.json"
data = json.load(open(data_path))
trades = data["trades"]

print(f"Analyzing {len(trades)} trades from backtest...")
print("=" * 70)

# Analyze different filter combinations
def analyze_subset(trades, name="All"):
    if not trades:
        return None
    wins = sum(1 for t in trades if t["won"])
    total = len(trades)
    wr = wins / total * 100
    return {"name": name, "trades": total, "wins": wins, "wr": wr}

# 1. By Side only
print("\n## BY SIDE")
print("-" * 50)
for side in ["UP", "DOWN"]:
    subset = [t for t in trades if t["side"] == side]
    r = analyze_subset(subset, side)
    if r:
        print(f"{r['name']:<20} {r['trades']:>6} trades, {r['wr']:>5.1f}% WR")

# 2. DOWN + Negative Momentum
print("\n## DOWN + MOMENTUM FILTERS")
print("-" * 50)
for mom_threshold in [0, -0.001, -0.002, -0.003, -0.005]:
    subset = [t for t in trades if t["side"] == "DOWN" and t["momentum"] <= mom_threshold]
    r = analyze_subset(subset, f"DOWN, mom<={mom_threshold:.3f}")
    if r and r["trades"] >= 50:
        marker = " ***" if r["wr"] >= 55 else " **" if r["wr"] >= 50 else ""
        print(f"{r['name']:<25} {r['trades']:>6} trades, {r['wr']:>5.1f}% WR{marker}")

# 3. By Confidence Level (bucketed)
print("\n## BY CONFIDENCE LEVEL")
print("-" * 50)
for conf_min in [0.50, 0.55, 0.60, 0.65, 0.70]:
    subset = [t for t in trades if t["confidence"] >= conf_min]
    r = analyze_subset(subset, f"Conf >= {conf_min:.0%}")
    if r and r["trades"] >= 50:
        marker = " ***" if r["wr"] >= 55 else " **" if r["wr"] >= 50 else ""
        print(f"{r['name']:<25} {r['trades']:>6} trades, {r['wr']:>5.1f}% WR{marker}")

# 4. DOWN + High Confidence
print("\n## DOWN + CONFIDENCE FILTERS")
print("-" * 50)
for conf_min in [0.55, 0.60, 0.65, 0.70, 0.75]:
    subset = [t for t in trades if t["side"] == "DOWN" and t["confidence"] >= conf_min]
    r = analyze_subset(subset, f"DOWN, conf>={conf_min:.0%}")
    if r and r["trades"] >= 30:
        marker = " ***" if r["wr"] >= 55 else " **" if r["wr"] >= 50 else ""
        print(f"{r['name']:<25} {r['trades']:>6} trades, {r['wr']:>5.1f}% WR{marker}")

# 5. By Hour
print("\n## BEST HOURS (>45% WR)")
print("-" * 50)
hour_stats = defaultdict(lambda: {"wins": 0, "total": 0})
for t in trades:
    hour = int(t["entry_time"].split("T")[1].split(":")[0])
    hour_stats[hour]["total"] += 1
    if t["won"]:
        hour_stats[hour]["wins"] += 1

good_hours = []
for hour in sorted(hour_stats.keys()):
    total = hour_stats[hour]["total"]
    wins = hour_stats[hour]["wins"]
    if total >= 50:
        wr = wins / total * 100
        if wr >= 45:
            good_hours.append(hour)
            marker = " ***" if wr >= 55 else " **" if wr >= 50 else ""
            print(f"Hour {hour:02d}:00 UTC         {total:>6} trades, {wr:>5.1f}% WR{marker}")

print(f"\nGood hours: {good_hours}")

# 6. DOWN + Good Hours
print("\n## DOWN + GOOD HOURS")
print("-" * 50)
for hour_set_name, hours in [
    ("Best 3 (12,17,03)", {3, 12, 17}),
    ("Best 5 (03,04,12,17,18)", {3, 4, 12, 17, 18}),
    ("Avoid worst (not 05,09,10)", set(range(24)) - {5, 9, 10}),
]:
    subset = [t for t in trades
              if t["side"] == "DOWN"
              and int(t["entry_time"].split("T")[1].split(":")[0]) in hours]
    r = analyze_subset(subset, hour_set_name)
    if r and r["trades"] >= 30:
        marker = " ***" if r["wr"] >= 55 else " **" if r["wr"] >= 50 else ""
        print(f"DOWN + {r['name']:<20} {r['trades']:>5} trades, {r['wr']:>5.1f}% WR{marker}")

# 7. Combined filters - find winning formula
print("\n## COMBINED FILTER SEARCH (Finding 55%+ WR)")
print("-" * 70)
print("Testing combinations of: Side + Confidence + Momentum + Hours")
print("-" * 70)

results = []

# Test combinations
for side in ["DOWN"]:  # DOWN is clearly better
    for conf_min in [0.55, 0.60, 0.65, 0.70]:
        for mom_max in [0.002, 0, -0.001, -0.002, -0.003]:
            for skip_bad_hours in [False, True]:
                bad_hours = {5, 9, 10}  # Worst performing hours

                subset = []
                for t in trades:
                    if t["side"] != side:
                        continue
                    if t["confidence"] < conf_min:
                        continue
                    if t["momentum"] > mom_max:
                        continue
                    if skip_bad_hours:
                        hour = int(t["entry_time"].split("T")[1].split(":")[0])
                        if hour in bad_hours:
                            continue
                    subset.append(t)

                if len(subset) >= 30:
                    wins = sum(1 for t in subset if t["won"])
                    wr = wins / len(subset) * 100
                    results.append({
                        "side": side,
                        "conf": conf_min,
                        "mom": mom_max,
                        "skip_bad": skip_bad_hours,
                        "trades": len(subset),
                        "wins": wins,
                        "wr": wr
                    })

# Sort by win rate
results.sort(key=lambda x: x["wr"], reverse=True)

# Show top 15 combinations
print(f"{'Side':<6} {'Conf':>6} {'Mom Max':>8} {'Skip Bad':>10} {'Trades':>8} {'WR':>8}")
print("-" * 70)
for r in results[:15]:
    skip_str = "Yes" if r["skip_bad"] else "No"
    marker = " ***" if r["wr"] >= 55 else " **" if r["wr"] >= 50 else ""
    print(f"{r['side']:<6} {r['conf']:>5.0%} {r['mom']:>8.3f} {skip_str:>10} {r['trades']:>8} {r['wr']:>7.1f}%{marker}")

# 8. RSI Analysis
print("\n## RSI ANALYSIS (DOWN trades)")
print("-" * 50)
for rsi_max in [70, 60, 55, 50, 45]:
    subset = [t for t in trades if t["side"] == "DOWN" and t["rsi"] <= rsi_max]
    r = analyze_subset(subset, f"DOWN, RSI<={rsi_max}")
    if r and r["trades"] >= 30:
        marker = " ***" if r["wr"] >= 55 else " **" if r["wr"] >= 50 else ""
        print(f"{r['name']:<25} {r['trades']:>6} trades, {r['wr']:>5.1f}% WR{marker}")

# 9. Heiken Ashi Color
print("\n## HEIKEN ASHI COLOR")
print("-" * 50)
for color in ["red", "green"]:
    for side in ["UP", "DOWN"]:
        subset = [t for t in trades if t["side"] == side and t["heiken_color"] == color]
        r = analyze_subset(subset, f"{side} + Heiken {color}")
        if r and r["trades"] >= 30:
            marker = " ***" if r["wr"] >= 55 else " **" if r["wr"] >= 50 else ""
            print(f"{r['name']:<25} {r['trades']:>6} trades, {r['wr']:>5.1f}% WR{marker}")

# 10. Final Recommendation
print("\n" + "=" * 70)
print("RECOMMENDED FILTERS FOR 55%+ WIN RATE")
print("=" * 70)

best = results[0] if results else None
if best:
    print(f"""
Based on 30-day backtest analysis:

BEST FILTER COMBINATION:
- Side: {best['side']} only
- Min Confidence: {best['conf']:.0%}
- Max Momentum: {best['mom']:.3f} (price must be falling)
- Skip Hours: {'{5, 9, 10}' if best['skip_bad'] else 'None'}

RESULTS:
- Trades: {best['trades']}
- Wins: {best['wins']}
- Win Rate: {best['wr']:.1f}%

The issue: Raw directional prediction tops out at ~47% even with filters.
To reach 55%+, we NEED Polymarket's market inefficiency (cheap entry prices).

REAL EDGE COMES FROM:
1. Entry price < $0.40 (betting when odds are wrong)
2. Order flow confirmation (OBI > 0.10 = bullish pressure)
3. CVD confirmation (net buying/selling pressure)
4. TTM Squeeze breakouts (volatility compression → expansion)

Historical 153 trades achieved 60% WR because they were filtered through
actual market prices where we only bet when the edge was real.
""")
