"""Find current daily crypto Up/Down markets with full details."""
import httpx
import json
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

client = httpx.Client(timeout=15, headers={'User-Agent': 'Mozilla/5.0'})

for crypto in ['bitcoin', 'ethereum', 'solana']:
    for day in range(5, 10):
        slug = f'{crypto}-up-or-down-on-february-{day}'
        r = client.get(f'https://gamma-api.polymarket.com/events?slug={slug}')
        if r.status_code == 200 and r.json():
            event = r.json()[0]
            title = event.get('title', '')
            markets = event.get('markets', [])
            open_m = [m for m in markets if not m.get('closed', True)]
            if open_m:
                m = open_m[0]
                outcomes = m.get('outcomes', '[]')
                prices = m.get('outcomePrices', '[]')
                if isinstance(outcomes, str): outcomes = json.loads(outcomes)
                if isinstance(prices, str): prices = json.loads(prices)
                cid = m.get('conditionId', '')
                end_date = m.get('endDate', '')
                tokens = m.get('clobTokenIds', '[]')
                if isinstance(tokens, str): tokens = json.loads(tokens)
                print(f'{title}')
                print(f'  outcomes: {outcomes}')
                print(f'  prices: {prices}')
                print(f'  conditionId: {cid}')
                print(f'  endDate: {end_date}')
                if tokens:
                    print(f'  UP token: {tokens[0][:30]}...')
                    if len(tokens) > 1:
                        print(f'  DOWN token: {tokens[1][:30]}...')
                print()

# Also check: what are king's non-crypto trades? (the "above $X" markets)
print("=" * 60)
print("KING'S OTHER STRATEGY: Price target markets")
print("=" * 60)
for crypto in ['bitcoin', 'ethereum', 'solana']:
    for price_target in ['70000', '76000', '80000', '90', '100', '2100', '2500']:
        slug = f'will-the-price-of-{crypto}-be-above-{price_target}'
        r = client.get(f'https://gamma-api.polymarket.com/events',
                       params={'slug_contains': slug, 'active': 'true', 'limit': 5})
        if r.status_code == 200 and r.json():
            for event in r.json()[:2]:
                title = event.get('title', '')
                markets = event.get('markets', [])
                open_m = [m for m in markets if not m.get('closed', True)]
                if open_m:
                    m = open_m[0]
                    prices = m.get('outcomePrices', '[]')
                    if isinstance(prices, str): prices = json.loads(prices)
                    print(f'  {title} | prices={prices}')
