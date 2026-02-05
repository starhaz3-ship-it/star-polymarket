"""Part 2: kingofcoinflips market type analysis."""
import sqlite3
from datetime import datetime, timezone
conn = sqlite3.connect('whale_watcher.db')
conn.row_factory = sqlite3.Row

# Unique markets
crypto = conn.execute("""
    SELECT DISTINCT market_title, condition_id FROM whale_trades
    WHERE whale_name = 'kingofcoinflips'
    AND market_title LIKE '%Up or Down%'
    ORDER BY market_title
""").fetchall()
print('=== UNIQUE MARKETS TRADED ===')
for r in crypto:
    cid = r['condition_id'] or '?'
    print(f"  {r['market_title'][:80]}  |  cid: {cid[:20]}...")

# Compare to 15-minute market titles
our_markets = conn.execute("""
    SELECT DISTINCT market_title FROM whale_trades
    WHERE market_title LIKE '%15%' AND market_title LIKE '%Up or Down%'
    LIMIT 5
""").fetchall()
print(f'\n=== 15-MIN MARKET TITLES FOR COMPARISON ===')
for r in our_markets:
    print(f"  {r['market_title'][:80]}")

# Hourly market titles
hourly = conn.execute("""
    SELECT DISTINCT market_title FROM whale_trades
    WHERE market_title LIKE '%Up or Down%' AND market_title LIKE '%AM%'
    LIMIT 5
""").fetchall()
print(f'\n=== HOURLY MARKET TITLES FOR COMPARISON ===')
for r in hourly:
    print(f"  {r['market_title'][:80]}")

# King's categories
cats = conn.execute("""
    SELECT market_category, COUNT(*) as cnt FROM whale_trades
    WHERE whale_name = 'kingofcoinflips'
    GROUP BY market_category
""").fetchall()
print(f'\n=== KING MARKET CATEGORIES ===')
for c in cats:
    print(f"  {c['market_category']}: {c['cnt']}")

# Trades per unique market (accumulation pattern)
per_market = conn.execute("""
    SELECT market_title, condition_id, COUNT(*) as cnt,
           SUM(cost_usd) as total_cost, GROUP_CONCAT(outcome, ',') as outcomes
    FROM whale_trades
    WHERE whale_name = 'kingofcoinflips' AND market_title LIKE '%Up or Down%'
    GROUP BY condition_id
    ORDER BY cnt DESC
""").fetchall()
print(f'\n=== TRADES PER MARKET (position accumulation) ===')
for r in per_market:
    outcomes = r['outcomes'] or ''
    up_cnt = outcomes.count('Up')
    down_cnt = outcomes.count('Down')
    total = r['total_cost'] or 0
    print(f"  {r['cnt']:3d} entries  ${total:>8.0f}  Up:{up_cnt} Down:{down_cnt}  | {r['market_title'][:55]}")

# Timing analysis - how spread out are entries on a single market?
print(f'\n=== ENTRY SPREAD ON SINGLE MARKETS ===')
for r in per_market[:5]:
    cid = r['condition_id']
    entries = conn.execute("""
        SELECT timestamp, outcome, price, cost_usd FROM whale_trades
        WHERE whale_name = 'kingofcoinflips' AND condition_id = ?
        ORDER BY timestamp
    """, (cid,)).fetchall()
    print(f"\n  --- {r['market_title'][:60]} ({len(entries)} entries) ---")
    for e in entries:
        ts = datetime.fromtimestamp(e['timestamp'], tz=timezone.utc).strftime('%m/%d %H:%M') if e['timestamp'] else '?'
        price = e['price'] or 0
        cost = e['cost_usd'] or 0
        print(f"    {ts}  {e['outcome']:>4} @${price:.3f}  ${cost:>8.2f}")

conn.close()
