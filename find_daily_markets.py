"""Find daily crypto Up/Down markets on Polymarket."""
import httpx
import json
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

client = httpx.Client(timeout=15, headers={'User-Agent': 'Mozilla/5.0'})

# Get lots of events
all_events = []
for offset in [0, 100, 200, 300, 400]:
    r = client.get('https://gamma-api.polymarket.com/events',
                   params={'active': 'true', 'closed': 'false', 'limit': 100, 'offset': offset})
    if r.status_code == 200:
        all_events.extend(r.json())

print(f'Total events fetched: {len(all_events)}')

# Find crypto Up or Down
updown = [e for e in all_events if 'Up or Down' in e.get('title', '')]
print(f'\n"Up or Down" events: {len(updown)}')
for e in updown[:30]:
    title = e.get('title', '')[:75]
    tags = e.get('tags', [])
    markets = e.get('markets', [])
    open_m = [m for m in markets if not m.get('closed', True)]
    print(f'  {title} | tags={tags[:3]} | {len(open_m)} open mkts')

# Also search for "on February" pattern (daily)
daily = [e for e in all_events if 'on February' in e.get('title', '') and ('Bitcoin' in e.get('title', '') or 'Ethereum' in e.get('title', '') or 'Solana' in e.get('title', ''))]
print(f'\n"on February" crypto events: {len(daily)}')
for e in daily[:15]:
    title = e.get('title', '')[:75]
    tags = e.get('tags', [])
    markets = e.get('markets', [])
    open_m = [m for m in markets if not m.get('closed', True)]
    if open_m:
        prices = open_m[0].get('outcomePrices', '[]')
        if isinstance(prices, str):
            prices = json.loads(prices)
        print(f'  {title} | tags={tags[:3]} | prices={prices}')

# Check for "above" or "below" price target markets
price_target = [e for e in all_events if ('above' in e.get('title', '').lower() or 'below' in e.get('title', '').lower()) and any(c in e.get('title', '') for c in ['Bitcoin', 'Ethereum', 'Solana', 'XRP'])]
print(f'\nPrice target crypto events: {len(price_target)}')
for e in price_target[:15]:
    title = e.get('title', '')[:75]
    tags = e.get('tags', [])
    markets = e.get('markets', [])
    open_m = [m for m in markets if not m.get('closed', True)]
    if open_m:
        prices = open_m[0].get('outcomePrices', '[]')
        if isinstance(prices, str):
            prices = json.loads(prices)
        outcomes = open_m[0].get('outcomes', '[]')
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        cid = open_m[0].get('conditionId', '')[:20]
        print(f'  {title}')
        print(f'    {outcomes} = {prices} | {len(open_m)} mkts | cid={cid}')

# Check ALL tag slugs we can find
all_tags = set()
for e in all_events:
    for t in e.get('tags', []):
        if isinstance(t, dict):
            all_tags.add(t.get('slug', '') or t.get('label', ''))
        else:
            all_tags.add(str(t))
crypto_tags = [t for t in sorted(all_tags) if any(x in t.lower() for x in ['crypto', 'btc', 'eth', 'bitcoin', 'daily', 'up', 'down', 'price', '15', 'hourly'])]
print(f'\nRelevant tags: {crypto_tags[:30]}')
