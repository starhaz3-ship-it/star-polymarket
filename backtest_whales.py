"""
Backtest whale crypto trades to validate and optimize live trading system.
Analyzes whale BTC/ETH/SOL 15m trades from whale_watcher.db.
"""
import sqlite3
from collections import defaultdict
from datetime import datetime

conn = sqlite3.connect('whale_watcher.db')
conn.row_factory = sqlite3.Row

# Get all crypto 15m trades
rows = conn.execute("""
    SELECT * FROM whale_trades
    WHERE market_category = 'crypto_15m'
    AND status = 'closed'
    AND paper_pnl IS NOT NULL
    ORDER BY timestamp
""").fetchall()

print(f"=== WHALE CRYPTO 15M TRADES: {len(rows)} closed trades ===\n")

if not rows:
    # Try broader query
    rows = conn.execute("""
        SELECT * FROM whale_trades
        WHERE (market_title LIKE '%Bitcoin%' OR market_title LIKE '%Ethereum%' OR market_title LIKE '%Solana%')
        AND (market_title LIKE '%Up or Down%')
        AND status = 'closed'
        AND paper_pnl IS NOT NULL
        ORDER BY timestamp
    """).fetchall()
    print(f"(Broader query found {len(rows)} trades)\n")

if not rows:
    # Even broader - check what categories exist
    cats = conn.execute("SELECT market_category, COUNT(*) as cnt FROM whale_trades GROUP BY market_category ORDER BY cnt DESC").fetchall()
    print("Market categories in DB:")
    for c in cats:
        print(f"  {c['market_category']}: {c['cnt']} trades")

    # Check statuses
    stats = conn.execute("SELECT status, COUNT(*) as cnt FROM whale_trades GROUP BY status").fetchall()
    print("\nTrade statuses:")
    for s in stats:
        print(f"  {s['status']}: {s['cnt']}")

    # Sample some trades to understand data
    samples = conn.execute("SELECT whale_name, market_title, outcome, side, price, cost_usd, paper_pnl, status, market_category FROM whale_trades LIMIT 20").fetchall()
    print("\nSample trades:")
    for s in samples:
        print(f"  {s['whale_name']:<15} {s['side']} {s['outcome']} @ ${s['price']:.2f} ${s['cost_usd']:.2f} PnL:${s['paper_pnl'] or 0:.2f} [{s['status']}] {s['market_category']} | {s['market_title'][:50]}")

# Get ALL crypto trades regardless of status
print("\n" + "=" * 80)
all_crypto = conn.execute("""
    SELECT * FROM whale_trades
    WHERE (market_title LIKE '%Bitcoin%Up or Down%'
        OR market_title LIKE '%Ethereum%Up or Down%'
        OR market_title LIKE '%Solana%Up or Down%'
        OR market_title LIKE '%XRP%Up or Down%')
    ORDER BY timestamp
""").fetchall()
print(f"\nAll crypto Up/Down trades: {len(all_crypto)}")

# Analyze by status
status_counts = defaultdict(int)
for r in all_crypto:
    status_counts[r['status']] += 1
print("By status:", dict(status_counts))

# Filter to ones with PnL data
with_pnl = [r for r in all_crypto if r['paper_pnl'] is not None and r['paper_pnl'] != 0]
print(f"With non-zero PnL: {len(with_pnl)}")

if with_pnl:
    wins = [r for r in with_pnl if r['paper_pnl'] > 0]
    losses = [r for r in with_pnl if r['paper_pnl'] < 0]
    print(f"\nWins: {len(wins)} | Losses: {len(losses)} | WR: {len(wins)/(len(wins)+len(losses))*100:.1f}%")
    print(f"Total PnL: ${sum(r['paper_pnl'] for r in with_pnl):.2f}")

# Analyze ALL crypto trades by whale
print("\n=== WHALE CRYPTO TRADE STATS ===")
whale_stats = defaultdict(lambda: {'buys': 0, 'sells': 0, 'total_cost': 0, 'outcomes': defaultdict(int), 'assets': defaultdict(int)})
for r in all_crypto:
    name = r['whale_name']
    whale_stats[name]['buys' if r['side'] == 'BUY' else 'sells'] += 1
    whale_stats[name]['total_cost'] += r['cost_usd'] or 0
    whale_stats[name]['outcomes'][r['outcome']] += 1
    # Detect asset
    title = r['market_title'].lower()
    if 'bitcoin' in title or 'btc' in title:
        whale_stats[name]['assets']['BTC'] += 1
    elif 'ethereum' in title or 'eth' in title:
        whale_stats[name]['assets']['ETH'] += 1
    elif 'solana' in title or 'sol' in title:
        whale_stats[name]['assets']['SOL'] += 1
    elif 'xrp' in title:
        whale_stats[name]['assets']['XRP'] += 1

print(f"{'Whale':<20} {'Buys':>5} {'Sells':>5} {'$Volume':>10} {'Up':>4} {'Down':>4} {'Assets'}")
print("-" * 80)
for name in sorted(whale_stats.keys(), key=lambda n: -whale_stats[n]['buys']):
    s = whale_stats[name]
    if s['buys'] + s['sells'] < 3:
        continue
    assets = ', '.join(f"{a}:{c}" for a, c in sorted(s['assets'].items(), key=lambda x: -x[1]))
    print(f"{name:<20} {s['buys']:5d} {s['sells']:5d} ${s['total_cost']:9.0f} {s['outcomes'].get('Up',0):4d} {s['outcomes'].get('Down',0):4d} {assets}")

# Analyze entry prices
print("\n=== WHALE ENTRY PRICE ANALYSIS ===")
buy_trades = [r for r in all_crypto if r['side'] == 'BUY']
if buy_trades:
    prices = [r['price'] for r in buy_trades if r['price'] and r['price'] > 0]
    if prices:
        avg_price = sum(prices) / len(prices)
        under_50 = len([p for p in prices if p < 0.50])
        at_50 = len([p for p in prices if 0.48 <= p <= 0.52])
        over_55 = len([p for p in prices if p > 0.55])
        print(f"Avg entry price: ${avg_price:.3f}")
        print(f"Under $0.50: {under_50} ({under_50/len(prices)*100:.0f}%)")
        print(f"Near $0.50: {at_50} ({at_50/len(prices)*100:.0f}%)")
        print(f"Over $0.55: {over_55} ({over_55/len(prices)*100:.0f}%)")

# Analyze UP vs DOWN preference
print("\n=== UP vs DOWN PREFERENCE ===")
up_trades = [r for r in buy_trades if r['outcome'] == 'Up']
down_trades = [r for r in buy_trades if r['outcome'] == 'Down']
print(f"Buy UP: {len(up_trades)} trades (${sum(r['cost_usd'] or 0 for r in up_trades):.0f})")
print(f"Buy DOWN: {len(down_trades)} trades (${sum(r['cost_usd'] or 0 for r in down_trades):.0f})")

# Time of day analysis
print("\n=== TRADE TIMING (UTC HOUR) ===")
hour_counts = defaultdict(int)
for r in buy_trades:
    if r['timestamp']:
        h = datetime.utcfromtimestamp(r['timestamp']).hour
        hour_counts[h] += 1
for h in sorted(hour_counts.keys()):
    bar = "#" * (hour_counts[h] // 2)
    print(f"  {h:02d}:00 UTC: {hour_counts[h]:4d} trades {bar}")

# Trade size analysis
print("\n=== WHALE BET SIZES ===")
sizes = [r['cost_usd'] for r in buy_trades if r['cost_usd'] and r['cost_usd'] > 0]
if sizes:
    sizes.sort()
    print(f"Min: ${min(sizes):.2f} | Median: ${sizes[len(sizes)//2]:.2f} | Max: ${max(sizes):.2f}")
    print(f"Avg: ${sum(sizes)/len(sizes):.2f}")
    for label, lo, hi in [('<$10', 0, 10), ('$10-50', 10, 50), ('$50-200', 50, 200), ('$200-1K', 200, 1000), ('$1K+', 1000, 1e9)]:
        cnt = len([s for s in sizes if lo <= s < hi])
        print(f"  {label}: {cnt} trades ({cnt/len(sizes)*100:.0f}%)")

# Per-asset entry price analysis
print("\n=== ENTRY PRICE BY ASSET ===")
for asset_name, kws in [('BTC', ['bitcoin', 'btc']), ('ETH', ['ethereum', 'eth']), ('SOL', ['solana', 'sol'])]:
    asset_buys = [r for r in buy_trades if any(kw in (r['market_title'] or '').lower() for kw in kws)]
    if not asset_buys:
        continue
    ap = [r['price'] for r in asset_buys if r['price'] and r['price'] > 0]
    up_count = len([r for r in asset_buys if r['outcome'] == 'Up'])
    down_count = len([r for r in asset_buys if r['outcome'] == 'Down'])
    if ap:
        print(f"{asset_name}: {len(asset_buys)} trades | avg entry ${sum(ap)/len(ap):.3f} | UP:{up_count} DOWN:{down_count} ({down_count/(up_count+down_count)*100:.0f}% DOWN)")

conn.close()
