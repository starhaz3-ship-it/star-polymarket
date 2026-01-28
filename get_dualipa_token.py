"""Get Dua Lipa market token."""
import httpx
import json

r = httpx.get("https://gamma-api.polymarket.com/markets", params={
    "active": "true",
    "limit": 300
})
markets = r.json()

for m in markets:
    q = m.get("question", "")
    if "DUA LIPA" in q.upper() and "SUPER BOWL" in q.upper():
        print(f"Found: {q}")
        print(f"Condition: {m.get('conditionId')}")
        print(f"Prices: {m.get('outcomePrices')}")

        tokens = m.get('clobTokenIds', '')
        if tokens:
            token_list = json.loads(tokens) if isinstance(tokens, str) else tokens
            print(f"YES Token: {token_list[0]}")
            if len(token_list) > 1:
                print(f"NO Token: {token_list[1]}")

        # Also check order book
        if token_list and len(token_list) > 1:
            no_token = token_list[1]
            r2 = httpx.get("https://clob.polymarket.com/price", params={"token_id": no_token})
            print(f"NO current price: {r2.json()}")
