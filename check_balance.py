"""Full account check: balance, all activity, P&L summary."""
import httpx
import os
from dotenv import load_dotenv
from functools import partial

print = partial(print, flush=True)
load_dotenv()

proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
print(f"Proxy wallet: {proxy}")

# Get ALL recent activity
offset = 0
all_activity = []
while True:
    r = httpx.get(
        "https://data-api.polymarket.com/activity",
        params={"user": proxy, "limit": "100", "offset": str(offset)},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15
    )
    if r.status_code != 200:
        break
    batch = r.json()
    if not batch:
        break
    all_activity.extend(batch)
    if len(batch) < 100:
        break
    offset += 100

print(f"\nTotal activity records: {len(all_activity)}")

# Filter to February 3 BTC 15m trades only
feb3_trades = []
for a in all_activity:
    title = str(a.get("title", ""))
    if "February 3" in title and "Bitcoin" in title:
        feb3_trades.append(a)

print(f"February 3 BTC trades: {len(feb3_trades)}")

# Summarize
total_bought = 0
total_sold = 0
total_redeemed = 0
buy_count = 0
sell_count = 0
redeem_count = 0

print("\nFEBRUARY 3 BTC ACTIVITY:")
for a in feb3_trades:
    atype = a.get("type", "?")
    side = a.get("side", "?")
    usd = float(a.get("usdcSize", 0))
    title = str(a.get("title", ""))[:55]
    ts = str(a.get("timestamp", ""))[:19]

    if atype == "TRADE" and side == "BUY":
        total_bought += usd
        buy_count += 1
        print(f"  BUY  ${usd:>8.2f} | {title}")
    elif atype == "TRADE" and side == "SELL":
        total_sold += usd
        sell_count += 1
        print(f"  SELL ${usd:>8.2f} | {title}")
    elif atype == "REDEEM":
        total_redeemed += usd
        redeem_count += 1
        print(f"  WIN  ${usd:>8.2f} | {title}")
    else:
        print(f"  {atype} {side} ${usd:.2f} | {title}")

print(f"\n{'='*60}")
print(f"SUMMARY - February 3 BTC 15m Trades:")
print(f"  Buys:     {buy_count} trades, ${total_bought:.2f} spent")
print(f"  Sells:    {sell_count} trades, ${total_sold:.2f} received back")
print(f"  Redeems:  {redeem_count} wins, ${total_redeemed:.2f} collected")
print(f"  {'='*50}")
net = total_sold + total_redeemed - total_bought
print(f"  NET P&L:  ${net:+.2f}")
print(f"  Capital deployed: ${total_bought:.2f}")
print(f"  Capital returned: ${total_sold + total_redeemed:.2f}")

# Current open positions
print(f"\n{'='*60}")
r3 = httpx.get(
    "https://data-api.polymarket.com/positions",
    params={"user": proxy, "sizeThreshold": "0.1"},
    headers={"User-Agent": "Mozilla/5.0"},
    timeout=15
)
if r3.status_code == 200:
    positions = r3.json()
    active = [p for p in positions if float(p.get("curPrice", 0)) > 0]
    print(f"ACTIVE POSITIONS (non-zero price): {len(active)}")
    for p in active:
        title = str(p.get("title", ""))[:50]
        size = p.get("size", 0)
        price = p.get("curPrice", 0)
        outcome = p.get("outcome", "?")
        value = float(size) * float(price)
        print(f"  {outcome} | {size} shares @ ${price} = ${value:.2f} | {title}")
