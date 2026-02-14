"""Direct order book depth query for active updown markets."""
import requests, json, sys, io, time
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

now = datetime.now(timezone.utc)
print(f"Time: {now.strftime('%H:%M:%S')} UTC\n")

for tag in ['15M']:
    r = requests.get('https://gamma-api.polymarket.com/events',
        params={'tag_slug': tag, 'active': 'true', 'closed': 'false', 'limit': 10},
        headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
    events = r.json()

    for event in events:
        title = event.get('title', '').lower()
        if 'bitcoin' not in title and 'ethereum' not in title:
            continue

        for m in event.get('markets', []):
            if m.get('closed', True):
                continue

            end_str = m.get('endDate', '')
            try:
                end_dt = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                mins = (end_dt - now).total_seconds() / 60
            except:
                continue
            if mins < 0 or mins > 60:
                continue

            q = m.get('question', '')[:65]
            asset = 'BTC' if 'bitcoin' in title else 'ETH'
            print(f"  {asset} | {q} | {mins:.0f}m left")

            tokens = m.get('tokens', [])
            for tok in tokens:
                tid = tok.get('token_id', '')
                outcome = tok.get('outcome', '?')
                if not tid:
                    continue

                try:
                    book = requests.get('https://clob.polymarket.com/book',
                                       params={'token_id': tid}, timeout=10).json()
                    bids = book.get('bids', [])
                    asks = book.get('asks', [])

                    if not bids:
                        print(f"    {outcome}: EMPTY")
                        continue

                    best_bid = float(bids[0]['price'])
                    best_ask = float(asks[0]['price']) if asks else 1.0
                    total_bid = sum(float(b['size']) for b in bids)
                    near_3c = sum(float(b['size']) for b in bids if float(b['price']) >= best_bid - 0.03)
                    near_5c = sum(float(b['size']) for b in bids if float(b['price']) >= best_bid - 0.05)
                    total_ask = sum(float(a['size']) for a in asks)

                    print(f"    {outcome}: bid=${best_bid:.2f} ask=${best_ask:.2f}")
                    print(f"      Bids: {near_3c:.0f}sh(3c) / {near_5c:.0f}sh(5c) / {total_bid:.0f}sh total")
                    print(f"      Asks: {total_ask:.0f}sh total")
                    print(f"      Levels:")
                    for b in bids[:8]:
                        sz = float(b['size'])
                        bar = '#' * min(int(sz / 3), 50)
                        print(f"        ${b['price']} x {sz:>6.0f} {bar}")

                    print(f"      MAX SIZING (% of near-bid depth):")
                    for test_usd in [5, 10, 15, 20, 25, 50]:
                        test_shares = test_usd / best_bid if best_bid > 0 else 0
                        pct = test_shares / near_3c * 100 if near_3c > 0 else 999
                        flag = '<-- current' if test_usd == 5 else ''
                        flag = '<-- SWEET SPOT' if pct < 25 and test_usd > 5 and not flag else flag
                        print(f"        ${test_usd:>3}/side ({test_shares:>5.0f}sh) = {pct:>5.1f}% of near-bid {flag}")

                except Exception as e:
                    print(f"    {outcome}: Error: {e}")

                time.sleep(0.3)
            print()

print("=" * 70)
print("  ANALYSIS")
print("=" * 70)
print("  As a MAKER (placing limit bids), our fill depends on:")
print("  1. Taker flow (sellers hitting our bid price)")
print("  2. Competition (other bids at same level)")
print("  3. Time window (15m markets give ~12min to fill)")
print()
print("  Rule of thumb for maker orders:")
print("  - <10% of near-bid: invisible, fills easily")
print("  - 10-25%: noticeable but manageable")
print("  - 25-50%: may slow fills, partial risk increases")
print("  - >50%: likely to move the market, high partial risk")
print()
