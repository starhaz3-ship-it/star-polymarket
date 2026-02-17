"""Analyze on-chain verified BTC trades to find edge patterns."""
import json
from collections import defaultdict
from datetime import datetime, timezone

# Load internal trades (for metadata: RSI, momentum, entry_price, time)
with open("momentum_15m_results.json") as f:
    data = json.load(f)
resolved = data["resolved"]

# Load on-chain activity
with open("onchain_activity.json") as f:
    activity = json.load(f)

# Group on-chain by conditionId
by_cond = defaultdict(list)
for a in activity:
    by_cond[a["conditionId"]].append(a)

onchain = {}
for cid, acts in by_cond.items():
    buys = [a for a in acts if a["type"] == "TRADE"]
    reds = [a for a in acts if a["type"] == "REDEEM"]
    total_spent = sum(a["usdcSize"] for a in buys)
    total_redeemed = sum(a["usdcSize"] for a in reds)
    onchain[cid] = {
        "spent": total_spent,
        "redeemed": total_redeemed,
        "pnl": total_redeemed - total_spent,
        "status": "REDEEMED" if reds else "OPEN",
    }

# Merge: only BTC, only on-chain confirmed
trades = []
for t in resolved:
    title = t.get("title", "")
    if "Bitcoin" not in title and "BTC" not in title:
        continue
    cid = t.get("condition_id", "")
    oc = onchain.get(cid)
    if not oc or oc["spent"] == 0:
        continue  # phantom

    entry_price = t.get("entry_price", 0)
    rsi = t.get("rsi", 0)
    momentum_10m = t.get("momentum_10m", 0) or t.get("momentum_pct", 0)
    momentum_5m = t.get("momentum_5m", 0)
    side = t.get("side", "")
    strategy = t.get("strategy", "")
    size = t.get("size_usd", 0)
    ts = t.get("timestamp", 0) or t.get("entry_time", 0)

    # Parse hour from title
    hour_utc = None
    try:
        # Title like "Bitcoin Up or Down - February 17, 1:45AM-2:00AM ET"
        # Extract the start time
        parts = title.split(",")
        if len(parts) >= 2:
            time_part = parts[-1].strip()  # "1:45AM-2:00AM ET"
            start_time = time_part.split("-")[0].strip()
            # Parse ET hour
            if "PM" in start_time:
                h = int(start_time.split(":")[0])
                if h != 12:
                    h += 12
                hour_et = h
            else:
                h = int(start_time.split(":")[0])
                if h == 12:
                    h = 0
                hour_et = h
            hour_utc = (hour_et + 5) % 24  # ET to UTC
    except:
        pass

    real_pnl = oc["pnl"]
    won = real_pnl > 0

    trades.append({
        "title": title[:55],
        "side": side,
        "entry_price": entry_price,
        "rsi": rsi,
        "momentum_10m": momentum_10m,
        "momentum_5m": momentum_5m,
        "strategy": strategy,
        "size": size,
        "real_pnl": real_pnl,
        "won": won,
        "hour_utc": hour_utc,
        "hour_et": hour_utc - 5 if hour_utc and hour_utc >= 5 else (hour_utc + 19 if hour_utc is not None else None),
    })

print(f"Analyzing {len(trades)} on-chain verified BTC trades")
print()

# === ANALYSIS ===

# 1. By side
print("=== BY SIDE ===")
for side in ["UP", "DOWN"]:
    st = [t for t in trades if t["side"] == side]
    w = sum(1 for t in st if t["won"])
    l = len(st) - w
    pnl = sum(t["real_pnl"] for t in st)
    wr = w / len(st) * 100 if st else 0
    print(f"  {side:4}: {w}W/{l}L ({wr:.0f}% WR) PnL: ${pnl:+.2f}")
print()

# 2. By entry price range
print("=== BY ENTRY PRICE ===")
for lo, hi, label in [(0, 0.40, "<$0.40"), (0.40, 0.50, "$0.40-0.50"), (0.50, 0.60, "$0.50-0.60"), (0.60, 1.0, ">$0.60")]:
    st = [t for t in trades if lo <= t["entry_price"] < hi]
    if not st:
        continue
    w = sum(1 for t in st if t["won"])
    l = len(st) - w
    pnl = sum(t["real_pnl"] for t in st)
    wr = w / len(st) * 100
    avg_price = sum(t["entry_price"] for t in st) / len(st)
    print(f"  {label:10}: {w}W/{l}L ({wr:.0f}% WR) PnL: ${pnl:+.2f} | avg entry: ${avg_price:.3f}")
print()

# 3. By RSI
print("=== BY RSI ===")
for lo, hi, label in [(0, 25, "RSI <25"), (25, 35, "RSI 25-35"), (35, 50, "RSI 35-50"), (50, 65, "RSI 50-65"), (65, 75, "RSI 65-75"), (75, 100, "RSI >75")]:
    st = [t for t in trades if t["rsi"] and lo <= t["rsi"] < hi]
    if not st:
        continue
    w = sum(1 for t in st if t["won"])
    l = len(st) - w
    pnl = sum(t["real_pnl"] for t in st)
    wr = w / len(st) * 100
    print(f"  {label:10}: {w}W/{l}L ({wr:.0f}% WR) PnL: ${pnl:+.2f}")
print()

# 4. By hour (ET)
print("=== BY HOUR (ET) ===")
hour_buckets = defaultdict(list)
for t in trades:
    if t["hour_et"] is not None:
        hour_buckets[t["hour_et"]].append(t)
for h in sorted(hour_buckets.keys()):
    st = hour_buckets[h]
    w = sum(1 for t in st if t["won"])
    l = len(st) - w
    pnl = sum(t["real_pnl"] for t in st)
    wr = w / len(st) * 100
    marker = " *** LOSING" if wr < 50 else ""
    print(f"  {h:2d}:00 ET: {w}W/{l}L ({wr:.0f}% WR) PnL: ${pnl:+.2f}{marker}")
print()

# 5. By momentum strength
print("=== BY MOMENTUM 10M ===")
for lo, hi, label in [(0, 0.1, "<0.10%"), (0.1, 0.2, "0.10-0.20%"), (0.2, 0.5, "0.20-0.50%"), (0.5, 10, ">0.50%")]:
    st = [t for t in trades if t["momentum_10m"] and lo <= abs(t["momentum_10m"]) < hi]
    if not st:
        continue
    w = sum(1 for t in st if t["won"])
    l = len(st) - w
    pnl = sum(t["real_pnl"] for t in st)
    wr = w / len(st) * 100
    print(f"  {label:12}: {w}W/{l}L ({wr:.0f}% WR) PnL: ${pnl:+.2f}")
print()

# 6. Show every trade with all metadata
print("=== ALL TRADES (chronological) ===")
for t in trades:
    result = "W" if t["won"] else "L"
    h = f"{t['hour_et']}ET" if t['hour_et'] is not None else "?"
    print(f"  {result} {t['side']:4} ${t['real_pnl']:>+7.2f} entry=${t['entry_price']:.3f} RSI={t['rsi']:>5.1f} mom10={t['momentum_10m']:>+.4f} mom5={t['momentum_5m']:>+.4f} {h:>5} | {t['title']}")
