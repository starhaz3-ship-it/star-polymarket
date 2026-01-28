"""Check Super Bowl halftime show market."""
import httpx
import json

print("=" * 60)
print("SUPER BOWL / ENTERTAINMENT MARKETS")
print("=" * 60)

# Get events
r = httpx.get('https://gamma-api.polymarket.com/events', params={
    'active': 'true',
    'closed': 'false',
    'limit': 100
})

events = r.json()
for event in events:
    title = event.get('title', '')
    if 'SUPER BOWL' in title.upper() or 'HALFTIME' in title.upper():
        print(f"\nEvent: {title}")
        for m in event.get('markets', [])[:10]:
            q = m.get('question', '')
            prices = m.get('outcomePrices', '')
            volume = m.get('volume', 0)
            tokens = m.get('clobTokenIds', '')
            cond = m.get('conditionId', '')

            try:
                price_list = json.loads(prices) if isinstance(prices, str) else prices
                yes = float(price_list[0]) if price_list else 0
            except:
                yes = 0

            print(f"\n  {q[:55]}")
            print(f"  YES: ${yes:.2f} | Volume: ${float(volume or 0):,.0f}")
            print(f"  Cond: {cond[:30]}...")
            if tokens:
                try:
                    token_list = json.loads(tokens) if isinstance(tokens, str) else tokens
                    if token_list:
                        print(f"  YES Token: {token_list[0][:30]}...")
                except:
                    pass

# Search for Cardi B specifically
print("\n" + "=" * 60)
print("SEARCHING FOR CARDI B MARKET")
print("=" * 60)

r = httpx.get('https://gamma-api.polymarket.com/markets', params={
    'active': 'true',
    'limit': 200
})

markets = r.json()
for m in markets:
    q = m.get('question', '')
    if 'CARDI' in q.upper():
        prices = m.get('outcomePrices', '')
        volume = m.get('volume', 0)
        tokens = m.get('clobTokenIds', '')
        cond = m.get('conditionId', '')
        end = m.get('endDate', '')

        try:
            price_list = json.loads(prices) if isinstance(prices, str) else prices
            yes = float(price_list[0]) if price_list else 0
        except:
            yes = 0

        print(f"\nMarket: {q}")
        print(f"YES Price: ${yes:.3f} ({yes*100:.0f}%)")
        print(f"Return if wins: +{(1/yes - 1)*100:.0f}%")
        print(f"Volume: ${float(volume or 0):,.0f}")
        print(f"Expires: {end[:16] if end else 'N/A'}")
        print(f"Condition ID: {cond}")

        if tokens:
            try:
                token_list = json.loads(tokens) if isinstance(tokens, str) else tokens
                if token_list:
                    print(f"YES Token: {token_list[0]}")
            except:
                pass
