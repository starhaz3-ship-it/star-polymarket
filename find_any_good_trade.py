"""Find any good non-crypto trade - relaxed criteria."""
import httpx
import json
from datetime import datetime, timezone

print("=" * 70)
print("SCANNING ALL NON-CRYPTO MARKETS (RELAXED CRITERIA)")
print("=" * 70)

r = httpx.get("https://gamma-api.polymarket.com/markets", params={
    "active": "true",
    "closed": "false",
    "limit": 500
})
markets = r.json()

now = datetime.now(timezone.utc)
crypto_keywords = ["BTC", "BITCOIN", "ETH", "ETHER", "CRYPTO", "SOL", "DOGE", "XRP", "MEMECOIN"]

opportunities = []

for m in markets:
    q = m.get("question", "")
    q_upper = q.upper()

    if any(kw in q_upper for kw in crypto_keywords):
        continue

    try:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            prices = json.loads(prices)
        if not prices:
            continue
        yes_price = float(prices[0])
    except:
        continue

    if yes_price <= 0.05 or yes_price >= 0.95:
        continue

    end_str = m.get("endDate", "")
    if not end_str:
        continue

    try:
        end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        hours_left = (end_date - now).total_seconds() / 3600
    except:
        continue

    # Up to 30 days
    if hours_left < 12 or hours_left > 720:
        continue

    volume = float(m.get("volume", 0) or 0)
    no_price = 1 - yes_price

    for side, price in [("YES", yes_price), ("NO", no_price)]:
        if price < 0.55:
            continue

        ret = (1 / price - 1) * 100

        opportunities.append({
            "question": q,
            "condition_id": m.get("conditionId", ""),
            "tokens": m.get("clobTokenIds", ""),
            "side": side,
            "price": price,
            "return": ret,
            "hours": hours_left,
            "volume": volume,
        })

# Sort by probability then return
opportunities.sort(key=lambda x: (x["price"], x["return"]), reverse=True)

print(f"\nFound {len(opportunities)} opportunities\n")

# Group by probability tier
high_prob = [o for o in opportunities if o["price"] >= 0.80]
med_prob = [o for o in opportunities if 0.65 <= o["price"] < 0.80]
low_prob = [o for o in opportunities if o["price"] < 0.65]

if high_prob:
    print("### HIGH PROBABILITY (80%+) ###\n")
    for o in high_prob[:5]:
        days = o["hours"] / 24
        print(f"{o['side']} @ ${o['price']:.2f} | +{o['return']:.0f}% | {days:.0f}d | Vol: ${o['volume']:,.0f}")
        print(f"  {o['question'][:60]}")
        print()

if med_prob:
    print("\n### MEDIUM PROBABILITY (65-80%) ###\n")
    for o in med_prob[:5]:
        days = o["hours"] / 24
        print(f"{o['side']} @ ${o['price']:.2f} | +{o['return']:.0f}% | {days:.0f}d | Vol: ${o['volume']:,.0f}")
        print(f"  {o['question'][:60]}")
        print()

# Best trade recommendation
print("=" * 70)
print("MY RECOMMENDATION")
print("=" * 70)

# Look for high volume + high probability
best = None
for o in opportunities:
    if o["price"] >= 0.70 and o["volume"] >= 1000:
        best = o
        break

if not best and opportunities:
    best = opportunities[0]

if best:
    days = best["hours"] / 24
    print(f"\nMarket: {best['question']}")
    print(f"Action: BUY {best['side']}")
    print(f"Price: ${best['price']:.3f} ({best['price']*100:.0f}% probability)")
    print(f"Return if wins: +{best['return']:.0f}%")
    print(f"Expires: {days:.0f} days")
    print(f"Volume: ${best['volume']:,.0f}")
    print(f"\nCondition ID: {best['condition_id']}")

    tokens = best.get("tokens", "")
    if tokens:
        try:
            token_list = json.loads(tokens) if isinstance(tokens, str) else tokens
            idx = 0 if best["side"] == "YES" else 1
            if len(token_list) > idx:
                print(f"Token ID: {token_list[idx]}")
        except:
            pass
else:
    print("\nNo suitable non-crypto trades found.")
