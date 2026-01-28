"""Find halftime markets."""
import httpx
import json

r = httpx.get("https://gamma-api.polymarket.com/markets", params={
    "active": "true",
    "limit": 500
})
markets = r.json()

print(f"Total markets: {len(markets)}")
print()

for m in markets:
    q = m.get("question", "")
    if "HALFTIME" in q.upper() or "HALF TIME" in q.upper():
        prices = m.get("outcomePrices", "")
        try:
            price_list = json.loads(prices) if isinstance(prices, str) else prices
            yes = float(price_list[0]) if price_list else 0
            no = 1 - yes
        except:
            yes = 0
            no = 0

        tokens = m.get('clobTokenIds', '')
        try:
            token_list = json.loads(tokens) if isinstance(tokens, str) else tokens
        except:
            token_list = []

        volume = float(m.get("volume", 0) or 0)

        # Only show markets with decent volume and NO price around 0.77
        if volume > 10000 and 0.70 <= no <= 0.85:
            print(f"Market: {q}")
            print(f"  YES: ${yes:.2f} | NO: ${no:.2f}")
            print(f"  Volume: ${volume:,.0f}")
            print(f"  Condition: {m.get('conditionId')}")
            if len(token_list) > 1:
                print(f"  NO Token: {token_list[1]}")
            print()
