"""Deep analysis of kingofcoinflips trading strategy."""
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone

conn = sqlite3.connect('whale_watcher.db')
conn.row_factory = sqlite3.Row

crypto = conn.execute("""
    SELECT * FROM whale_trades
    WHERE whale_name = 'kingofcoinflips'
    AND market_title LIKE '%Up or Down%'
    ORDER BY timestamp DESC
""").fetchall()

print("=== SAMPLE WINNING TRADES (most recent) ===")
wins = [r for r in crypto if r['status'] == 'resolved' and (r['paper_pnl'] or 0) > 0]
for r in wins[:20]:
    ts = datetime.fromtimestamp(r['timestamp'], tz=timezone.utc).strftime('%m/%d %H:%M') if r['timestamp'] else '?'
    title = (r['market_title'] or '')[:65]
    pnl = r['paper_pnl'] or 0
    price = r['price'] or 0
    cost = r['cost_usd'] or 0
    print(f"  {ts} {r['outcome']:>4} @${price:.3f} ${cost:>8.2f} PnL:${pnl:>+7.2f} | {title}")

print(f"\n=== MARKET TIME WINDOWS ===")
time_windows = []
for r in crypto:
    title = r['market_title'] or ''
    tl = title.lower()
    if '15' in tl and ('min' in tl):
        time_windows.append('15min')
    elif any(x in title for x in ['1AM', '2AM', '3AM', '4AM', '5AM', '6AM', '7AM', '8AM', '9AM', '10AM', '11AM', '12AM',
                                    '1PM', '2PM', '3PM', '4PM', '5PM', '6PM', '7PM', '8PM', '9PM', '10PM', '11PM', '12PM']):
        time_windows.append('hourly')
    else:
        time_windows.append('other')
print(Counter(time_windows))

print(f"\n=== UNIQUE MARKET TITLE SAMPLES ===")
titles = set()
for r in crypto:
    titles.add((r['market_title'] or '')[:90])
for t in sorted(titles)[:20]:
    print(f"  {t}")

print(f"\n=== UP TRADE ENTRY PRICE DISTRIBUTION ===")
up_trades = [r for r in crypto if r['outcome'] == 'Up']
up_prices = [r['price'] for r in up_trades if r['price'] and r['price'] > 0]
for label, lo, hi in [('<$0.35', 0, 0.35), ('$0.35-0.45', 0.35, 0.45), ('$0.45-0.50', 0.45, 0.50),
                       ('$0.50-0.55', 0.50, 0.55), ('$0.55+', 0.55, 1.0)]:
    cnt = len([p for p in up_prices if lo <= p < hi])
    pct = cnt/len(up_prices)*100 if up_prices else 0
    print(f"  {label}: {cnt} ({pct:.0f}%)")

print(f"\n=== DOWN TRADE ENTRY PRICE DISTRIBUTION ===")
down_trades = [r for r in crypto if r['outcome'] == 'Down']
down_prices = [r['price'] for r in down_trades if r['price'] and r['price'] > 0]
for label, lo, hi in [('<$0.50', 0, 0.50), ('$0.50-0.70', 0.50, 0.70), ('$0.70-0.90', 0.70, 0.90),
                       ('$0.90+', 0.90, 1.0)]:
    cnt = len([p for p in down_prices if lo <= p < hi])
    pct = cnt/len(down_prices)*100 if down_prices else 0
    print(f"  {label}: {cnt} ({pct:.0f}%)")

print(f"\n=== TIMING: WHEN DURING THE HOUR DO THEY ENTER? ===")
# Check minutes within the hour (are they entering early or late in the market window?)
for r in wins[:30]:
    ts = datetime.fromtimestamp(r['timestamp'], tz=timezone.utc)
    title = (r['market_title'] or '')[:55]
    price = r['price'] or 0
    cost = r['cost_usd'] or 0
    pnl = r['paper_pnl'] or 0
    print(f"  {ts.strftime('%m/%d %H:%M:%S')} {r['outcome']:>4} @${price:.3f} ${cost:>7.2f} PnL:${pnl:>+.2f} | {title}")

print(f"\n=== NON-CRYPTO TRADES (what else does king trade?) ===")
other = conn.execute("""
    SELECT * FROM whale_trades
    WHERE whale_name = 'kingofcoinflips'
    AND market_title NOT LIKE '%Up or Down%'
    ORDER BY timestamp DESC LIMIT 20
""").fetchall()
for r in other[:15]:
    ts = datetime.fromtimestamp(r['timestamp'], tz=timezone.utc).strftime('%m/%d %H:%M') if r['timestamp'] else '?'
    title = (r['market_title'] or '')[:70]
    price = r['price'] or 0
    cost = r['cost_usd'] or 0
    print(f"  {ts} {r['outcome']:>5} @${price:.3f} ${cost:>8.2f} [{r['status']}] | {title}")

print(f"\n=== KEY INSIGHT: ENTRY TIMING vs MARKET RESOLUTION ===")
# For hourly markets, how far before resolution do they enter?
# Market titles often contain time like "5PM" - try to extract
import re
for r in wins[:15]:
    title = r['market_title'] or ''
    # Try to find market resolution time in title
    time_match = re.search(r'(\d{1,2})(AM|PM)', title)
    entry_ts = datetime.fromtimestamp(r['timestamp'], tz=timezone.utc)
    price = r['price'] or 0
    cost = r['cost_usd'] or 0
    pnl = r['paper_pnl'] or 0
    mkt_time = time_match.group() if time_match else "?"
    print(f"  Entry: {entry_ts.strftime('%H:%M')} UTC | Market: {mkt_time} | {r['outcome']} @${price:.3f} ${cost:.0f} PnL:${pnl:+.0f} | {title[:50]}")

conn.close()
