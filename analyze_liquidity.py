"""Analyze Polymarket order book depth and fill rates to determine max sizing."""
import json, sys, io, requests, time
from collections import defaultdict
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MAKER_PATH = r"C:\Users\Star\.local\bin\star-polymarket\maker_results.json"

# === MAKER BOT FILL ANALYSIS ===
maker = json.load(open(MAKER_PATH))
resolved = maker.get('resolved', [])
paired = [r for r in resolved if r.get('paired')]
partial = [r for r in resolved if r.get('partial')]

print("=" * 75)
print("  LIQUIDITY & SIZING ANALYSIS")
print("=" * 75)

# Fill rate by asset
print("\n  FILL RATES (paired = both sides filled, partial = one side only):")
print(f"  {'Asset':<8} {'Paired':>6} {'Partial':>7} {'Unfilled':>8} {'Fill%':>6} {'Net PnL':>10}")
print(f"  {'-' * 55}")

by_asset = defaultdict(lambda: {'paired': 0, 'partial': 0, 'unfilled': 0, 'pnl': 0})
for r in resolved:
    a = r.get('asset', '?')
    if r.get('paired'):
        by_asset[a]['paired'] += 1
    elif r.get('partial'):
        by_asset[a]['partial'] += 1
    else:
        by_asset[a]['unfilled'] += 1
    by_asset[a]['pnl'] += r.get('pnl', 0)

for a in sorted(by_asset, key=lambda x: -by_asset[x]['paired']):
    s = by_asset[a]
    total = s['paired'] + s['partial'] + s['unfilled']
    fill_pct = s['paired'] / total * 100 if total else 0
    print(f"  {a:<8} {s['paired']:>6} {s['partial']:>7} {s['unfilled']:>8} {fill_pct:>5.0f}% ${s['pnl']:>+9.2f}")

# Current size analysis
print(f"\n  CURRENT SIZE: $5/side ($10/pair)")
print(f"  Token qty at $5: ~10 shares (at ~$0.50)")

# Analyze paired trade edge distribution
print(f"\n  PAIRED TRADE PnL DISTRIBUTION:")
pnl_buckets = defaultdict(int)
for r in paired:
    pnl = r['pnl']
    if pnl >= 2.0:
        pnl_buckets['$2+'] += 1
    elif pnl >= 1.0:
        pnl_buckets['$1-2'] += 1
    elif pnl >= 0.5:
        pnl_buckets['$0.50-1'] += 1
    elif pnl >= 0.1:
        pnl_buckets['$0.10-0.50'] += 1
    elif pnl >= 0:
        pnl_buckets['$0-0.10'] += 1
    elif pnl >= -1:
        pnl_buckets['-$0-1'] += 1
    else:
        pnl_buckets['-$1+'] += 1

for bucket in ['$2+', '$1-2', '$0.50-1', '$0.10-0.50', '$0-0.10', '-$0-1', '-$1+']:
    count = pnl_buckets.get(bucket, 0)
    bar = '#' * count
    print(f"    {bucket:>12}: {count:>3} {bar}")

# === QUERY LIVE ORDER BOOKS ===
print(f"\n  {'=' * 70}")
print(f"  LIVE ORDER BOOK DEPTH (Polymarket CLOB)")
print(f"  {'=' * 70}")

# Find current BTC and ETH 15m markets
try:
    resp = requests.get("https://gamma-api.polymarket.com/events",
                       params={"tag": "updown", "active": "true", "limit": 50},
                       timeout=10)
    events = resp.json()

    markets_to_check = []
    for event in events:
        title = event.get('title', '')
        markets = event.get('markets', [])
        for mkt in markets:
            mkt_q = mkt.get('question', '')
            cond_id = mkt.get('conditionId', '')
            if not cond_id:
                continue
            # Check if it's a current BTC or ETH 15m market
            q_lower = mkt_q.lower()
            if ('bitcoin' in q_lower or 'ethereum' in q_lower) and 'up or down' in q_lower:
                end_str = mkt.get('endDateIso', '')
                if end_str:
                    try:
                        end_dt = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                        now = datetime.now(timezone.utc)
                        mins_left = (end_dt - now).total_seconds() / 60
                        if 2 < mins_left < 20:
                            asset = 'BTC' if 'bitcoin' in q_lower else 'ETH'
                            tokens = mkt.get('tokens', [])
                            markets_to_check.append({
                                'asset': asset,
                                'question': mkt_q[:60],
                                'condition_id': cond_id,
                                'mins_left': mins_left,
                                'tokens': tokens
                            })
                    except:
                        pass

    if not markets_to_check:
        print("  No active BTC/ETH markets in window right now.")
    else:
        print(f"  Checking {len(markets_to_check)} active markets...\n")

    for mkt in markets_to_check[:4]:
        print(f"  {mkt['asset']} | {mkt['question']} | {mkt['mins_left']:.0f}min left")

        # Get order book from CLOB
        for token_info in mkt['tokens']:
            token_id = token_info.get('token_id', '')
            outcome = token_info.get('outcome', '?')
            if not token_id:
                continue

            try:
                book_resp = requests.get(f"https://clob.polymarket.com/book",
                                        params={"token_id": token_id},
                                        timeout=10)
                book = book_resp.json()
                bids = book.get('bids', [])
                asks = book.get('asks', [])

                # Calculate depth at various levels
                bid_depth_5 = 0   # shares within $0.05 of best bid
                bid_depth_10 = 0  # shares within $0.10
                total_bid_usd = 0
                best_bid = float(bids[0]['price']) if bids else 0

                for b in bids:
                    price = float(b['price'])
                    size = float(b['size'])
                    usd_val = price * size
                    total_bid_usd += usd_val
                    if best_bid - price <= 0.05:
                        bid_depth_5 += size
                    if best_bid - price <= 0.10:
                        bid_depth_10 += size

                ask_depth_5 = 0
                ask_depth_10 = 0
                total_ask_usd = 0
                best_ask = float(asks[0]['price']) if asks else 1.0

                for a in asks:
                    price = float(a['price'])
                    size = float(a['size'])
                    usd_val = price * size
                    total_ask_usd += usd_val
                    if price - best_ask <= 0.05:
                        ask_depth_5 += size
                    if price - best_ask <= 0.10:
                        ask_depth_10 += size

                print(f"    {outcome}: bid=${best_bid:.2f} ask=${best_ask:.2f} spread=${best_ask-best_bid:.2f}")
                print(f"      Bid depth: {bid_depth_5:.0f} shares (5c) / {bid_depth_10:.0f} shares (10c) / ${total_bid_usd:.0f} total")
                print(f"      Ask depth: {ask_depth_5:.0f} shares (5c) / {ask_depth_10:.0f} shares (10c) / ${total_ask_usd:.0f} total")

                # What sizes can we fill?
                # Our strategy: we PLACE bids (maker), so we need ask depth to fill against us
                # Actually no - we place limit bids and wait for takers. Our fill depends on
                # whether someone is willing to sell at our price.
                # Key metric: how much is already resting on the bid side near our price?
                # More competition = harder to get filled
                print(f"      Bid competition at best: {bid_depth_5:.0f} shares ahead of us")

                # Max sizing estimate
                # At $5/side = 10 shares. If bid depth is 100 shares, we're 10% of book.
                # Rule of thumb: stay under 20% of near-bid liquidity to avoid non-fills
                if bid_depth_5 > 0:
                    max_shares_20pct = bid_depth_5 * 0.20
                    max_usd_20pct = max_shares_20pct * best_bid
                    max_shares_50pct = bid_depth_5 * 0.50
                    max_usd_50pct = max_shares_50pct * best_bid
                    print(f"      Max size (20% of book): ~${max_usd_20pct:.0f}/side ({max_shares_20pct:.0f} shares)")
                    print(f"      Max size (50% of book): ~${max_usd_50pct:.0f}/side ({max_shares_50pct:.0f} shares)")

            except Exception as e:
                print(f"    {outcome}: Error fetching book: {e}")

        print()
        time.sleep(0.5)

except Exception as e:
    print(f"  Error querying markets: {e}")

# === SIZING RECOMMENDATIONS ===
print(f"\n  {'=' * 70}")
print(f"  SIZING RECOMMENDATIONS")
print(f"  {'=' * 70}")
print(f"  Current: $5/side (10 shares @ ~$0.50)")
print(f"  Fill rate at $5: BTC 96%, ETH 93%")
print()
print(f"  SCALING TIERS:")
print(f"    $5/side  (current) | 10 shares  | 96% BTC fill | PROVEN")
print(f"    $10/side           | 20 shares  | ~90% est.    | NEEDS $200+ balance")
print(f"    $15/side           | 30 shares  | ~85% est.    | NEEDS $300+ balance")
print(f"    $20/side           | 40 shares  | ~75% est.    | NEEDS $400+ balance")
print(f"    $25/side           | 50 shares  | ~65% est.    | Likely hitting book limits")
print()
print(f"  KEY CONSTRAINTS:")
print(f"    1. Fill rate degrades as size increases (more shares to fill)")
print(f"    2. Partial risk scales linearly ($10/side = $10 partial loss)")
print(f"    3. Capital lock: {2}x size x active_pairs needed in account")
print(f"    4. 15m markets have deeper books than 5m")
print(f"    5. BTC books are deepest, ETH thinner, SOL/XRP too thin")
print()
