"""Query live CLOB order books for active BTC/ETH updown markets."""
import requests, json, sys, io, time
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

now = datetime.now(timezone.utc)

for tag in ['15M', '5M']:
    r = requests.get('https://gamma-api.polymarket.com/events',
        params={'tag_slug': tag, 'active': 'true', 'closed': 'false',
                'limit': 30, 'order': 'endDate', 'ascending': 'false'},
        headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
    events = r.json()
    print(f"\n{'='*75}")
    print(f"  {tag} MARKETS ({len(events)} events)")
    print(f"{'='*75}")

    checked = 0
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
            if mins < 0 or mins > 30:
                continue

            q = m.get('question', '')[:65]
            asset = 'BTC' if 'bitcoin' in q.lower() else 'ETH'
            print(f"\n  {asset} | {q} | {mins:.0f}m left")
            print(f"  {'-'*70}")

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
                        print(f"    {outcome}: EMPTY BOOK")
                        continue

                    best_bid = float(bids[0]['price'])
                    best_ask = float(asks[0]['price']) if asks else 1.0
                    total_bid_shares = sum(float(b['size']) for b in bids)
                    near_bid_3c = sum(float(b['size']) for b in bids if float(b['price']) >= best_bid - 0.03)
                    near_bid_5c = sum(float(b['size']) for b in bids if float(b['price']) >= best_bid - 0.05)
                    total_ask_shares = sum(float(a['size']) for a in asks)

                    print(f"    {outcome}: bid=${best_bid:.2f} ask=${best_ask:.2f} spread=${best_ask-best_bid:.2f}")
                    print(f"      Bid depth: {near_bid_3c:.0f}sh (3c) / {near_bid_5c:.0f}sh (5c) / {total_bid_shares:.0f}sh total")
                    print(f"      Ask depth: {total_ask_shares:.0f}sh total")
                    print(f"      Top bid levels:")
                    for b in bids[:5]:
                        sz = float(b['size'])
                        bar = '#' * min(int(sz / 5), 50)
                        print(f"        ${b['price']} x {sz:>6.0f} {bar}")

                    # Sizing analysis
                    print(f"      Our size vs book:")
                    for test_shares in [10, 20, 30, 50, 100]:
                        test_usd = test_shares * best_bid
                        pct_near = test_shares / near_bid_3c * 100 if near_bid_3c > 0 else 999
                        pct_total = test_shares / total_bid_shares * 100 if total_bid_shares > 0 else 999
                        print(f"        {test_shares:>4}sh (${test_usd:>5.0f}) = {pct_near:>5.1f}% of near / {pct_total:>5.1f}% of total")

                except Exception as e:
                    print(f"    {outcome}: Error: {e}")

                time.sleep(0.2)

            checked += 1
            if checked >= 3:
                break
        if checked >= 3:
            break

print()
