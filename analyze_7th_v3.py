"""Deep analysis of 7thStaircase: what are they doing RIGHT that we can copy?"""
import httpx
import json
from collections import defaultdict
from datetime import datetime, timezone
import statistics

wallet = "0x0ac97e4f5c542cd98c226ae8e1736ae78b489641"

# Fetch all activity (up to 2500)
all_trades = []
offset = 0
while offset < 5000:
    r = httpx.get(
        f"https://data-api.polymarket.com/activity?user={wallet}&limit=100&offset={offset}",
        timeout=15,
    )
    batch = r.json()
    if not batch:
        break
    all_trades.extend(batch)
    offset += 100
    if len(batch) < 100:
        break

print(f"Total activity entries: {len(all_trades)}")
# Filter out non-dict entries
all_trades = [t for t in all_trades if isinstance(t, dict) and t.get("type") == "TRADE"]
print(f"Valid trade entries: {len(all_trades)}")

# Group by event
events = defaultdict(lambda: {
    "up_fills": [], "down_fills": [], "title": "", "slug": "",
    "timestamps": [], "is_5m": False, "is_15m": False,
})
for t in all_trades:
    slug = t.get("eventSlug", "")
    side = t.get("outcome", "")
    price = float(t.get("price", 0))
    cost = float(t.get("usdcSize", 0))
    shares = float(t.get("size", 0))
    ts = t.get("timestamp", 0)
    title = t.get("title", "")

    e = events[slug]
    e["title"] = title
    e["slug"] = slug
    e["timestamps"].append(ts)
    e["is_5m"] = "5m" in slug or "5M" in slug
    e["is_15m"] = "15m" in slug or "15M" in slug

    fill = {"price": price, "cost": cost, "shares": shares, "ts": ts}
    if side == "Up":
        e["up_fills"].append(fill)
    elif side == "Down":
        e["down_fills"].append(fill)

print(f"Unique events: {len(events)}")

# === ANALYSIS 1: Entry timing relative to market close ===
print("\n" + "=" * 70)
print("ANALYSIS 1: WHEN DO THEY ENTER RELATIVE TO MARKET CLOSE?")
print("=" * 70)

# For 5m markets, slug contains the start timestamp
# e.g. btc-updown-5m-1771547100 -> market starts at 1771547100
# 5m market closes at start + 300s, 15m at start + 900s
entry_timings = {"5m": [], "15m": []}
for slug, e in events.items():
    if not e["timestamps"]:
        continue
    # Extract market start time from slug
    parts = slug.split("-")
    try:
        market_start = int(parts[-1])
    except (ValueError, IndexError):
        continue

    if e["is_5m"]:
        market_end = market_start + 300  # 5 min
        tf = "5m"
    elif e["is_15m"]:
        market_end = market_start + 900  # 15 min
        tf = "15m"
    else:
        continue

    first_fill = min(e["timestamps"])
    last_fill = max(e["timestamps"])
    secs_before_close_first = market_end - first_fill
    secs_before_close_last = market_end - last_fill
    secs_after_open_first = first_fill - market_start

    entry_timings[tf].append({
        "first_entry_after_open": secs_after_open_first,
        "first_entry_before_close": secs_before_close_first,
        "last_entry_before_close": secs_before_close_last,
        "num_fills": len(e["up_fills"]) + len(e["down_fills"]),
        "total_cost": sum(f["cost"] for f in e["up_fills"]) + sum(f["cost"] for f in e["down_fills"]),
    })

for tf in ["5m", "15m"]:
    timings = entry_timings[tf]
    if not timings:
        continue
    avg_first = statistics.mean([t["first_entry_after_open"] for t in timings])
    avg_last = statistics.mean([t["last_entry_before_close"] for t in timings])
    avg_fills = statistics.mean([t["num_fills"] for t in timings])
    avg_cost = statistics.mean([t["total_cost"] for t in timings])
    print(f"\n{tf} markets ({len(timings)} events):")
    print(f"  First fill: {avg_first:.0f}s after open (avg)")
    print(f"  Last fill: {avg_last:.0f}s before close (avg)")
    print(f"  Avg fills per event: {avg_fills:.0f}")
    print(f"  Avg volume per event: ${avg_cost:.2f}")

    # Distribution of first entry timing
    buckets = defaultdict(int)
    for t in timings:
        bucket = int(t["first_entry_after_open"] // 30) * 30
        buckets[bucket] += 1
    print(f"  First entry timing (seconds after open):")
    for b in sorted(buckets.keys()):
        bar = "#" * buckets[b]
        print(f"    {b:>4d}-{b+30:>4d}s: {buckets[b]:>3d} {bar}")


# === ANALYSIS 2: Size allocation UP vs DOWN ===
print("\n" + "=" * 70)
print("ANALYSIS 2: DIRECTIONAL BIAS PATTERN")
print("=" * 70)

biases = []
for slug, e in events.items():
    up_cost = sum(f["cost"] for f in e["up_fills"])
    dn_cost = sum(f["cost"] for f in e["down_fills"])
    total = up_cost + dn_cost
    if total < 5:
        continue
    up_pct = up_cost / total * 100
    dn_pct = dn_cost / total * 100
    bias = "UP" if up_cost > dn_cost else "DN"
    ratio = max(up_cost, dn_cost) / min(up_cost, dn_cost) if min(up_cost, dn_cost) > 0 else 99
    biases.append({"up_pct": up_pct, "dn_pct": dn_pct, "bias": bias, "ratio": ratio, "total": total})

if biases:
    up_biased = sum(1 for b in biases if b["bias"] == "UP")
    dn_biased = sum(1 for b in biases if b["bias"] == "DN")
    avg_ratio = statistics.mean([b["ratio"] for b in biases])
    avg_up_pct = statistics.mean([b["up_pct"] for b in biases])
    print(f"UP-biased events: {up_biased} | DOWN-biased: {dn_biased}")
    print(f"Avg allocation: {avg_up_pct:.1f}% UP / {100-avg_up_pct:.1f}% DOWN")
    print(f"Avg bias ratio: {avg_ratio:.1f}x toward dominant side")

    # Ratio distribution
    print(f"Bias ratio distribution:")
    for thresh in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        count = sum(1 for b in biases if b["ratio"] >= thresh)
        print(f"  >= {thresh:.1f}x: {count} events")


# === ANALYSIS 3: Fill price distribution — are they getting better prices? ===
print("\n" + "=" * 70)
print("ANALYSIS 3: FILL PRICE QUALITY")
print("=" * 70)

all_up_prices = []
all_dn_prices = []
for slug, e in events.items():
    for f in e["up_fills"]:
        if f["price"] > 0:
            all_up_prices.append(f["price"])
    for f in e["down_fills"]:
        if f["price"] > 0:
            all_dn_prices.append(f["price"])

if all_up_prices:
    print(f"UP fills: {len(all_up_prices)} orders")
    print(f"  Mean: ${statistics.mean(all_up_prices):.3f}")
    print(f"  Median: ${statistics.median(all_up_prices):.3f}")
    print(f"  <$0.30: {sum(1 for p in all_up_prices if p < 0.30)} ({100*sum(1 for p in all_up_prices if p < 0.30)/len(all_up_prices):.0f}%)")
    print(f"  $0.30-0.50: {sum(1 for p in all_up_prices if 0.30 <= p < 0.50)} ({100*sum(1 for p in all_up_prices if 0.30 <= p < 0.50)/len(all_up_prices):.0f}%)")
    print(f"  >$0.50: {sum(1 for p in all_up_prices if p >= 0.50)} ({100*sum(1 for p in all_up_prices if p >= 0.50)/len(all_up_prices):.0f}%)")

if all_dn_prices:
    print(f"\nDOWN fills: {len(all_dn_prices)} orders")
    print(f"  Mean: ${statistics.mean(all_dn_prices):.3f}")
    print(f"  Median: ${statistics.median(all_dn_prices):.3f}")
    print(f"  <$0.30: {sum(1 for p in all_dn_prices if p < 0.30)} ({100*sum(1 for p in all_dn_prices if p < 0.30)/len(all_dn_prices):.0f}%)")
    print(f"  $0.30-0.50: {sum(1 for p in all_dn_prices if 0.30 <= p < 0.50)} ({100*sum(1 for p in all_dn_prices if 0.30 <= p < 0.50)/len(all_dn_prices):.0f}%)")
    print(f"  >$0.50: {sum(1 for p in all_dn_prices if p >= 0.50)} ({100*sum(1 for p in all_dn_prices if p >= 0.50)/len(all_dn_prices):.0f}%)")


# === ANALYSIS 4: Order splitting — fill size distribution ===
print("\n" + "=" * 70)
print("ANALYSIS 4: ORDER SPLITTING PATTERN")
print("=" * 70)

all_fill_sizes = [float(t.get("usdcSize", 0)) for t in all_trades if float(t.get("usdcSize", 0)) > 0]
if all_fill_sizes:
    print(f"Total fills: {len(all_fill_sizes)}")
    print(f"Mean fill: ${statistics.mean(all_fill_sizes):.2f}")
    print(f"Median fill: ${statistics.median(all_fill_sizes):.2f}")
    print(f"Max fill: ${max(all_fill_sizes):.2f}")
    print(f"< $5: {sum(1 for s in all_fill_sizes if s < 5)} ({100*sum(1 for s in all_fill_sizes if s < 5)/len(all_fill_sizes):.0f}%)")
    print(f"$5-20: {sum(1 for s in all_fill_sizes if 5 <= s < 20)} ({100*sum(1 for s in all_fill_sizes if 5 <= s < 20)/len(all_fill_sizes):.0f}%)")
    print(f"$20-100: {sum(1 for s in all_fill_sizes if 20 <= s < 100)} ({100*sum(1 for s in all_fill_sizes if 20 <= s < 100)/len(all_fill_sizes):.0f}%)")
    print(f">$100: {sum(1 for s in all_fill_sizes if s >= 100)} ({100*sum(1 for s in all_fill_sizes if s >= 100)/len(all_fill_sizes):.0f}%)")


# === ANALYSIS 5: What % of markets do they skip? ===
print("\n" + "=" * 70)
print("ANALYSIS 5: MARKET COVERAGE")
print("=" * 70)

# Count unique 5m and 15m events
m5_events = set()
m15_events = set()
for slug, e in events.items():
    if e["is_5m"]:
        m5_events.add(slug)
    elif e["is_15m"]:
        m15_events.add(slug)

# How many 5m and 15m markets existed during their trading hours?
if all_trades:
    first_ts = min(t.get("timestamp", 0) for t in all_trades)
    last_ts = max(t.get("timestamp", 0) for t in all_trades)
    hours_active = (last_ts - first_ts) / 3600
    possible_5m = hours_active * 12  # 12 5m markets per hour
    possible_15m = hours_active * 4  # 4 15m markets per hour
    print(f"Trading window: {hours_active:.1f} hours")
    print(f"5M markets traded: {len(m5_events)} (possible ~{possible_5m:.0f}, coverage {100*len(m5_events)/possible_5m:.0f}%)")
    print(f"15M markets traded: {len(m15_events)} (possible ~{possible_15m:.0f}, coverage {100*len(m15_events)/possible_15m:.0f}%)")
    print(f"Total: {len(m5_events) + len(m15_events)} markets in {hours_active:.1f} hours")


# === ANALYSIS 6: Combined price analysis — actual cost basis ===
print("\n" + "=" * 70)
print("ANALYSIS 6: COMBINED COST BASIS (both-sides)")
print("=" * 70)

combined_prices = []
for slug, e in events.items():
    up_cost = sum(f["cost"] for f in e["up_fills"])
    dn_cost = sum(f["cost"] for f in e["down_fills"])
    up_shares = sum(f["shares"] for f in e["up_fills"])
    dn_shares = sum(f["shares"] for f in e["down_fills"])

    if up_shares > 0 and dn_shares > 0:
        up_avg = up_cost / up_shares
        dn_avg = dn_cost / dn_shares
        combined = up_avg + dn_avg
        combined_prices.append(combined)

if combined_prices:
    print(f"Markets with both sides: {len(combined_prices)}")
    print(f"Combined price mean: ${statistics.mean(combined_prices):.3f}")
    print(f"Combined price median: ${statistics.median(combined_prices):.3f}")
    print(f"< $1.00 (arb): {sum(1 for p in combined_prices if p < 1.0)}")
    print(f"$1.00-$1.20: {sum(1 for p in combined_prices if 1.0 <= p < 1.2)}")
    print(f"$1.20-$1.50: {sum(1 for p in combined_prices if 1.2 <= p < 1.5)}")
    print(f"> $1.50: {sum(1 for p in combined_prices if p >= 1.5)}")

    print(f"\nThe excess above $1.00 is the 'directional premium' they pay for hedging.")
    print(f"They profit when their biased side wins by more than this premium.")


# === ANALYSIS 7: Compare their approach to ours ===
print("\n" + "=" * 70)
print("COMPARISON: 7thStaircase vs Our Bots")
print("=" * 70)
print(f"""
                         7thStaircase      Our Momentum    Our Sniper
Markets/day:             ~{len(events):>3d}              ~3-5            ~2-3
Avg volume/market:       ${statistics.mean([sum(f['cost'] for f in e['up_fills'])+sum(f['cost'] for f in e['down_fills']) for e in events.values() if sum(f['cost'] for f in e['up_fills'])+sum(f['cost'] for f in e['down_fills']) > 0]):.0f}           $3              $3
Both-sides:              YES               NO              NO
Order splitting:         60-160 fills       1 FOK           1 FOK
Entry timing:            At market open     2-6min before   63sec before
Holds to resolution:     YES               YES             YES
Directional signal:      Price bias         RSI/FRAMA       Binance spike
Daily volume:            $41,328            ~$15            ~$10
""")
