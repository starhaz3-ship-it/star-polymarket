"""Check open orders and recent activity on Polymarket."""
import httpx, os
from functools import partial
from dotenv import load_dotenv

print = partial(print, flush=True)
load_dotenv()

proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
print(f"Proxy: {proxy}")

# Check recent activity
r2 = httpx.get(
    "https://data-api.polymarket.com/activity",
    params={"user": proxy, "limit": "10"},
    headers={"User-Agent": "Mozilla/5.0"},
    timeout=15
)
if r2.status_code == 200:
    acts = r2.json()
    print(f"\nRecent activity ({len(acts)}):")
    for a in acts:
        atype = a.get("type", "?")
        side = a.get("side", "?")
        usd = float(a.get("usdcSize", 0))
        title = str(a.get("title", ""))[:55]
        ts = str(a.get("timestamp", ""))[:19]
        print(f"  {atype:6} {side:4} ${usd:>8.2f} | {title} | {ts}")

# Check positions
r3 = httpx.get(
    "https://data-api.polymarket.com/positions",
    params={"user": proxy, "sizeThreshold": "0.1"},
    headers={"User-Agent": "Mozilla/5.0"},
    timeout=15
)
if r3.status_code == 200:
    positions = r3.json()
    print(f"\nAll positions: {len(positions)}")
    for p in positions:
        title = str(p.get("title", ""))[:50]
        size = p.get("size", 0)
        price = p.get("curPrice", 0)
        outcome = p.get("outcome", "?")
        value = float(size) * float(price)
        if value > 0.01:
            print(f"  {outcome} | {size} shares @ ${price} = ${value:.2f} | {title}")
