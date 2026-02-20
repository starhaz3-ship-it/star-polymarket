"""Deep analysis of 7thStaircase trading pattern."""
import httpx
import json
from collections import defaultdict
from datetime import datetime, timezone

wallet = "0x0ac97e4f5c542cd98c226ae8e1736ae78b489641"

# Fetch all activity
all_trades = []
offset = 0
while offset < 2500:
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

today = [
    t
    for t in all_trades
    if datetime.fromtimestamp(t.get("timestamp", 0), tz=timezone.utc).day == 19
    and datetime.fromtimestamp(t.get("timestamp", 0), tz=timezone.utc).month == 2
]

# Check BUY vs SELL
buy_count = sum(1 for t in today if t.get("side") == "BUY")
sell_count = sum(1 for t in today if t.get("side") == "SELL")
print(f"Today: {buy_count} BUYs, {sell_count} SELLs (total {len(today)})")

# Find the event with most fills
slugs = defaultdict(int)
for t in today:
    slugs[t.get("eventSlug", "")] += 1

target = max(slugs, key=slugs.get)
target_trades = [t for t in today if t.get("eventSlug", "") == target]
target_trades.sort(key=lambda x: x.get("timestamp", 0))

print(f"\nDeep dive: {target_trades[0].get('title','')} ({len(target_trades)} fills)")
for t in target_trades[:50]:
    ts = datetime.fromtimestamp(t["timestamp"], tz=timezone.utc).strftime("%H:%M:%S")
    side = t.get("outcome", "?")[:4]
    action = t.get("side", "?")
    price = float(t.get("price", 0))
    cost = float(t.get("usdcSize", 0))
    shares = float(t.get("size", 0))
    print(f"  {ts} {action:4s} {side:4s} @{price:.4f} | {shares:>8.2f}sh | ${cost:>8.2f}")

# Sells analysis
sells = [t for t in today if t.get("side") == "SELL"]
if sells:
    total_sell_rev = sum(float(t.get("usdcSize", 0)) for t in sells)
    print(f"\nSell revenue: ${total_sell_rev:,.2f} ({len(sells)} sells)")
    for t in sells[:15]:
        ts = datetime.fromtimestamp(t["timestamp"], tz=timezone.utc).strftime("%H:%M:%S")
        side = t.get("outcome", "?")
        price = float(t.get("price", 0))
        cost = float(t.get("usdcSize", 0))
        title = t.get("title", "")[:50]
        print(f"  {ts} SELL {side:4s} @{price:.3f} ${cost:.2f} | {title}")
else:
    print("\nNo sells found - holds everything to resolution!")

# Hourly distribution
print("\nHourly distribution (UTC):")
hourly = defaultdict(lambda: {"buys": 0, "sells": 0, "volume": 0})
for t in today:
    h = datetime.fromtimestamp(t["timestamp"], tz=timezone.utc).hour
    hourly[h]["volume"] += float(t.get("usdcSize", 0))
    if t.get("side") == "BUY":
        hourly[h]["buys"] += 1
    else:
        hourly[h]["sells"] += 1
for h in sorted(hourly):
    e = hourly[h]
    print(
        f"  {h:02d}:00 UTC | {e['buys']:>4d} buys, {e['sells']:>4d} sells | ${e['volume']:>8,.2f}"
    )

# Price evolution: when do they buy cheap vs expensive?
print("\nFill price evolution (first 30 min of trading):")
if today:
    first_ts = min(t["timestamp"] for t in today)
    for t in sorted(today, key=lambda x: x["timestamp"])[:30]:
        elapsed = t["timestamp"] - first_ts
        mins = elapsed // 60
        secs = elapsed % 60
        side = t.get("outcome", "?")[:4]
        action = t.get("side", "?")
        price = float(t.get("price", 0))
        cost = float(t.get("usdcSize", 0))
        title = t.get("title", "")[:40]
        print(f"  +{mins:>3d}m{secs:02d}s {action:4s} {side:4s} @{price:.3f} ${cost:.2f} | {title}")
