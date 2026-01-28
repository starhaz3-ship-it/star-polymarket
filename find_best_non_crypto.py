"""Find best non-crypto trade with real analysis."""
import httpx
import json
from datetime import datetime, timezone

print("=" * 70)
print("FINDING BEST NON-CRYPTO TRADE")
print("=" * 70)

# Get markets
r = httpx.get("https://gamma-api.polymarket.com/markets", params={
    "active": "true",
    "closed": "false",
    "limit": 500
})
markets = r.json()

now = datetime.now(timezone.utc)
crypto_keywords = ["BTC", "BITCOIN", "ETH", "ETHER", "CRYPTO", "SOL", "DOGE", "XRP"]

opportunities = []

for m in markets:
    q = m.get("question", "")
    q_upper = q.upper()

    # Skip crypto
    if any(kw in q_upper for kw in crypto_keywords):
        continue

    # Get prices
    try:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            prices = json.loads(prices)
        if not prices:
            continue
        yes_price = float(prices[0])
    except:
        continue

    # Skip settled/illiquid
    if yes_price <= 0.05 or yes_price >= 0.95:
        continue

    # End date
    end_str = m.get("endDate", "")
    if not end_str:
        continue

    try:
        end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        hours_left = (end_date - now).total_seconds() / 3600
    except:
        continue

    # Only 1-7 days
    if hours_left < 24 or hours_left > 168:
        continue

    volume = float(m.get("volume", 0) or 0)

    # Need decent volume
    if volume < 5000:
        continue

    no_price = 1 - yes_price

    # Check both sides
    for side, price in [("YES", yes_price), ("NO", no_price)]:
        if price < 0.60:  # Need 60%+ probability
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

# Score by: probability, volume, short duration
for o in opportunities:
    o["score"] = o["price"] * (o["volume"] ** 0.3) / (o["hours"] ** 0.5)

opportunities.sort(key=lambda x: x["score"], reverse=True)

print(f"\nFound {len(opportunities)} opportunities (60%+ prob, $5K+ volume, 1-7 days)\n")

for i, o in enumerate(opportunities[:10], 1):
    days = o["hours"] / 24
    print(f"{i}. {o['question'][:60]}")
    print(f"   {o['side']} @ ${o['price']:.2f} ({o['price']*100:.0f}% prob)")
    print(f"   Return: +{o['return']:.0f}% | Days: {days:.1f} | Volume: ${o['volume']:,.0f}")

    if i == 1:
        tokens = o.get("tokens", "")
        if tokens:
            try:
                token_list = json.loads(tokens) if isinstance(tokens, str) else tokens
                idx = 0 if o["side"] == "YES" else 1
                if len(token_list) > idx:
                    print(f"   Token: {token_list[idx]}")
            except:
                pass
        print(f"   Condition: {o['condition_id']}")
    print()

# RECOMMENDATION
if opportunities:
    best = opportunities[0]
    print("=" * 70)
    print("RECOMMENDED TRADE")
    print("=" * 70)
    print(f"\nMarket: {best['question']}")
    print(f"Action: BUY {best['side']}")
    print(f"Price: ${best['price']:.3f}")
    print(f"Win Probability: {best['price']*100:.0f}%")
    print(f"Return if wins: +{best['return']:.0f}%")
    print(f"Expires in: {best['hours']/24:.1f} days")
    print(f"Volume: ${best['volume']:,.0f}")

    tokens = best.get("tokens", "")
    if tokens:
        try:
            token_list = json.loads(tokens) if isinstance(tokens, str) else tokens
            idx = 0 if best["side"] == "YES" else 1
            if len(token_list) > idx:
                print(f"\nToken ID: {token_list[idx]}")
        except:
            pass
    print(f"Condition ID: {best['condition_id']}")
