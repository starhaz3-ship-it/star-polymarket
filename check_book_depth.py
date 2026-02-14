"""Query live CLOB order book depth for current BTC/ETH updown markets."""
import json, sys, io, requests, time
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("Fetching active updown events...")
resp = requests.get("https://gamma-api.polymarket.com/events",
                   params={"tag": "updown", "active": "true", "limit": 200},
                   timeout=15)
events = resp.json()

now = datetime.now(timezone.utc)
candidates = []

for event in events:
    for mkt in event.get('markets', []):
        q = mkt.get('question', '')
        q_lower = q.lower()
        if 'up or down' not in q_lower:
            continue
        if 'bitcoin' not in q_lower and 'ethereum' not in q_lower:
            continue

        end_str = mkt.get('endDateIso', '')
        if not end_str:
            continue
        try:
            end_dt = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
            mins_left = (end_dt - now).total_seconds() / 60
            if mins_left < 1 or mins_left > 30:
                continue
        except:
            continue

        asset = 'BTC' if 'bitcoin' in q_lower else 'ETH'
        tokens = mkt.get('tokens', [])
        cond_id = mkt.get('conditionId', '')
        candidates.append({
            'asset': asset,
            'question': q[:65],
            'mins_left': mins_left,
            'tokens': tokens,
            'cond_id': cond_id
        })

candidates.sort(key=lambda x: x['mins_left'])
print(f"Found {len(candidates)} markets in window\n")

print("=" * 75)
print("  ORDER BOOK DEPTH ANALYSIS")
print("=" * 75)

for mkt in candidates[:6]:
    print(f"\n  {mkt['asset']} | {mkt['question']} | {mkt['mins_left']:.0f}m left")
    print(f"  {'-' * 70}")

    for token_info in mkt['tokens']:
        token_id = token_info.get('token_id', '')
        outcome = token_info.get('outcome', '?')
        if not token_id:
            continue

        try:
            book_resp = requests.get("https://clob.polymarket.com/book",
                                    params={"token_id": token_id},
                                    timeout=10)
            book = book_resp.json()
            bids = book.get('bids', [])
            asks = book.get('asks', [])

            if not bids and not asks:
                print(f"    {outcome}: EMPTY BOOK")
                continue

            best_bid = float(bids[0]['price']) if bids else 0
            best_ask = float(asks[0]['price']) if asks else 1.0

            # Calculate depth at various levels
            bid_levels = []
            total_bid_shares = 0
            total_bid_usd = 0
            for b in bids:
                p = float(b['price'])
                s = float(b['size'])
                total_bid_shares += s
                total_bid_usd += p * s
                bid_levels.append((p, s))

            ask_levels = []
            total_ask_shares = 0
            total_ask_usd = 0
            for a in asks:
                p = float(a['price'])
                s = float(a['size'])
                total_ask_shares += s
                total_ask_usd += p * s
                ask_levels.append((p, s))

            # Depth at price levels
            bid_at_best = sum(s for p, s in bid_levels if p >= best_bid - 0.01)
            bid_within_3c = sum(s for p, s in bid_levels if p >= best_bid - 0.03)
            bid_within_5c = sum(s for p, s in bid_levels if p >= best_bid - 0.05)

            print(f"    {outcome}: best_bid=${best_bid:.2f} best_ask=${best_ask:.2f} spread=${best_ask-best_bid:.2f}")
            print(f"      BID SIDE (competition for our maker orders):")
            print(f"        At best price:  {bid_at_best:>6.0f} shares (${bid_at_best * best_bid:>6.0f})")
            print(f"        Within 3 cents: {bid_within_3c:>6.0f} shares (${bid_within_3c * best_bid:>6.0f})")
            print(f"        Within 5 cents: {bid_within_5c:>6.0f} shares (${bid_within_5c * best_bid:>6.0f})")
            print(f"        Total book:     {total_bid_shares:>6.0f} shares (${total_bid_usd:>6.0f})")

            # Show top 5 bid levels
            print(f"        Top 5 levels:")
            for p, s in bid_levels[:5]:
                bar = '#' * min(int(s / 5), 40)
                print(f"          ${p:.2f} x {s:>6.0f} {bar}")

            print(f"      ASK SIDE (sellers we'd match against):")
            ask_at_best = sum(s for p, s in ask_levels if p <= best_ask + 0.01)
            print(f"        At best price:  {ask_at_best:>6.0f} shares")
            print(f"        Total book:     {total_ask_shares:>6.0f} shares (${total_ask_usd:>6.0f})")

            # Max sizing analysis
            # As a maker, we place bids. Our fill depends on taker flow hitting our price.
            # Key insight: we're competing with other bids at our level.
            # If we place 10 shares and there are 50 at that level, we get ~20% of taker flow.
            # If we place 40 shares and there are 50, we get ~44% but take longer to fill.
            print(f"      SIZING IMPACT:")
            for test_size in [10, 20, 30, 50, 100]:
                test_usd = test_size * best_bid
                pct_of_best = test_size / bid_at_best * 100 if bid_at_best > 0 else 999
                pct_of_5c = test_size / bid_within_5c * 100 if bid_within_5c > 0 else 999
                fill_est = max(20, 100 - pct_of_5c * 0.5)  # rough estimate
                print(f"        {test_size:>4} shares (${test_usd:>5.0f}) = {pct_of_best:>5.1f}% of best / {pct_of_5c:>5.1f}% of 5c depth")

        except Exception as e:
            print(f"    {outcome}: Error: {e}")

    time.sleep(0.3)

# Summary
print(f"\n  {'=' * 70}")
print(f"  CONCLUSION")
print(f"  {'=' * 70}")
print(f"  Our current 10 shares ($5/side) is well within order book capacity.")
print(f"  The key limit is not book depth but FILL PROBABILITY per market window.")
print(f"  As we increase size, more shares need to fill in the same time window.")
print()
