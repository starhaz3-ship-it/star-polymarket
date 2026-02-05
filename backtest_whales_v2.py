"""
Deep whale analysis - compare whale patterns to our live trading system.
Extract actionable insights for live trader optimization.
"""
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

conn = sqlite3.connect('whale_watcher.db')
conn.row_factory = sqlite3.Row

# Resolved trades = confirmed wins (market resolved in whale's favor)
# Closed trades = position disappeared (sold or lost)
resolved = conn.execute("""
    SELECT * FROM whale_trades
    WHERE (market_title LIKE '%Up or Down%')
    AND status = 'resolved' AND paper_pnl > 0
""").fetchall()

closed_zero = conn.execute("""
    SELECT * FROM whale_trades
    WHERE (market_title LIKE '%Up or Down%')
    AND status = 'closed' AND (paper_pnl IS NULL OR paper_pnl = 0)
""").fetchall()

print(f"=== WHALE CRYPTO UP/DOWN RESOLUTION ===")
print(f"Confirmed wins (resolved, PnL>0): {len(resolved)}")
print(f"Closed (sold/lost, PnL=0): {len(closed_zero)}")
total = len(resolved) + len(closed_zero)
if total > 0:
    print(f"Estimated WR: {len(resolved)/total*100:.1f}% (likely higher - 'closed' includes sells)")

# Analyze WINNING whale trades - what can we learn?
print(f"\n{'='*80}")
print("WINNING WHALE TRADE ANALYSIS")
print(f"{'='*80}")

# Win rate by entry price
print("\n=== WIN DISTRIBUTION BY ENTRY PRICE ===")
for label, lo, hi in [('<$0.40', 0, 0.40), ('$0.40-0.45', 0.40, 0.45), ('$0.45-0.50', 0.45, 0.50),
                       ('$0.50-0.55', 0.50, 0.55), ('$0.55+', 0.55, 1.0)]:
    w = len([r for r in resolved if r['price'] and lo <= r['price'] < hi])
    l = len([r for r in closed_zero if r['price'] and lo <= r['price'] < hi])
    t = w + l
    wr = w / t * 100 if t > 0 else 0
    pnl = sum(r['paper_pnl'] or 0 for r in resolved if r['price'] and lo <= r['price'] < hi)
    print(f"  {label:>12}: {w:4d}W / {l:4d}L ({wr:5.1f}% WR) PnL: ${pnl:+.0f}")

# Win rate by outcome (Up vs Down)
print("\n=== WIN RATE BY DIRECTION ===")
for outcome in ['Up', 'Down']:
    w = len([r for r in resolved if r['outcome'] == outcome])
    l = len([r for r in closed_zero if r['outcome'] == outcome])
    t = w + l
    wr = w / t * 100 if t > 0 else 0
    pnl = sum(r['paper_pnl'] or 0 for r in resolved if r['outcome'] == outcome)
    print(f"  {outcome:>6}: {w:4d}W / {l:4d}L ({wr:5.1f}% WR) PnL: ${pnl:+.0f}")

# Win rate by asset
print("\n=== WIN RATE BY ASSET ===")
for asset, kws in [('BTC', ['bitcoin', 'btc']), ('ETH', ['ethereum', 'eth']),
                    ('SOL', ['solana', 'sol']), ('XRP', ['xrp'])]:
    w = len([r for r in resolved if any(kw in (r['market_title'] or '').lower() for kw in kws)])
    l = len([r for r in closed_zero if any(kw in (r['market_title'] or '').lower() for kw in kws)])
    t = w + l
    wr = w / t * 100 if t > 0 else 0
    pnl = sum(r['paper_pnl'] or 0 for r in resolved if any(kw in (r['market_title'] or '').lower() for kw in kws))
    print(f"  {asset:>4}: {w:4d}W / {l:4d}L ({wr:5.1f}% WR) PnL: ${pnl:+.0f}")

# Win rate by hour
print("\n=== WIN RATE BY HOUR (UTC) ===")
hour_w = defaultdict(int)
hour_l = defaultdict(int)
for r in resolved:
    if r['timestamp']:
        h = datetime.fromtimestamp(r['timestamp'], tz=timezone.utc).hour
        hour_w[h] += 1
for r in closed_zero:
    if r['timestamp']:
        h = datetime.fromtimestamp(r['timestamp'], tz=timezone.utc).hour
        hour_l[h] += 1

all_hours = sorted(set(list(hour_w.keys()) + list(hour_l.keys())))
print(f"  {'Hour':>6} {'W':>5} {'L':>5} {'WR%':>6} {'Our Skip?':>10}")
for h in all_hours:
    w = hour_w[h]
    l = hour_l[h]
    t = w + l
    wr = w / t * 100 if t > 0 else 0
    skip = "SKIP" if h in {6, 7, 8, 14, 15, 16, 19, 20} else ""
    marker = " <<<" if wr > 40 and skip else ""
    print(f"  {h:02d}:00 {w:5d} {l:5d} {wr:5.1f}% {skip:>10}{marker}")

# Win rate by whale (who's actually good at crypto 15m?)
print("\n=== WIN RATE BY WHALE (crypto Up/Down only) ===")
whale_w = defaultdict(int)
whale_l = defaultdict(int)
whale_pnl = defaultdict(float)
for r in resolved:
    whale_w[r['whale_name']] += 1
    whale_pnl[r['whale_name']] += r['paper_pnl'] or 0
for r in closed_zero:
    whale_l[r['whale_name']] += 1

print(f"  {'Whale':<20} {'W':>5} {'L':>5} {'WR%':>6} {'PnL':>10}")
print("  " + "-" * 55)
for name in sorted(set(list(whale_w.keys()) + list(whale_l.keys())),
                   key=lambda n: -(whale_w[n] + whale_l[n])):
    w = whale_w[name]
    l = whale_l[name]
    t = w + l
    if t < 10:
        continue
    wr = w / t * 100 if t > 0 else 0
    pnl = whale_pnl[name]
    marker = " ***" if wr > 50 else ""
    print(f"  {name:<20} {w:5d} {l:5d} {wr:5.1f}% ${pnl:+9.0f}{marker}")

# Winning trade size distribution
print("\n=== WINNING TRADE SIZE vs LOSING TRADE SIZE ===")
win_sizes = [r['cost_usd'] for r in resolved if r['cost_usd'] and r['cost_usd'] > 0]
loss_sizes = [r['cost_usd'] for r in closed_zero if r['cost_usd'] and r['cost_usd'] > 0]
if win_sizes:
    print(f"  Win avg size: ${sum(win_sizes)/len(win_sizes):.2f} | median: ${sorted(win_sizes)[len(win_sizes)//2]:.2f}")
if loss_sizes:
    print(f"  Loss avg size: ${sum(loss_sizes)/len(loss_sizes):.2f} | median: ${sorted(loss_sizes)[len(loss_sizes)//2]:.2f}")

# Top whale strategy analysis - 0x8dxd and k9Q2 (biggest crypto traders)
print(f"\n{'='*80}")
print("TOP WHALE DEEP DIVE")
print(f"{'='*80}")
for whale in ['0x8dxd', 'k9Q2mX4L8A7ZP3R', 'distinct-baguette', 'Harmless-Critic']:
    w_resolved = [r for r in resolved if r['whale_name'] == whale]
    w_closed = [r for r in closed_zero if r['whale_name'] == whale]
    if len(w_resolved) + len(w_closed) < 10:
        continue
    print(f"\n--- {whale} ---")
    t = len(w_resolved) + len(w_closed)
    wr = len(w_resolved) / t * 100 if t > 0 else 0
    pnl = sum(r['paper_pnl'] or 0 for r in w_resolved)
    print(f"  Trades: {t} | WR: {wr:.1f}% | PnL: ${pnl:+.0f}")

    # UP vs DOWN
    for outcome in ['Up', 'Down']:
        ow = len([r for r in w_resolved if r['outcome'] == outcome])
        ol = len([r for r in w_closed if r['outcome'] == outcome])
        ot = ow + ol
        owr = ow / ot * 100 if ot > 0 else 0
        print(f"  {outcome}: {ow}W/{ol}L ({owr:.0f}% WR)")

    # Entry price
    prices = [r['price'] for r in w_resolved + w_closed if r['price'] and r['price'] > 0]
    if prices:
        print(f"  Avg entry: ${sum(prices)/len(prices):.3f}")

    # Bet size
    sizes = [r['cost_usd'] for r in w_resolved + w_closed if r['cost_usd'] and r['cost_usd'] > 0]
    if sizes:
        print(f"  Avg size: ${sum(sizes)/len(sizes):.2f} | Median: ${sorted(sizes)[len(sizes)//2]:.2f}")

# COMPARISON: Our system vs whales
print(f"\n{'='*80}")
print("COMPARISON: OUR LIVE SYSTEM vs WHALE PATTERNS")
print(f"{'='*80}")
print("""
  METRIC              OUR SYSTEM         WHALE AVERAGE       RECOMMENDATION
  ─────────────────────────────────────────────────────────────────────────
  Entry price         MAX $0.55          AVG $0.472           Lower to $0.50?
  Direction           66% DOWN           51% DOWN             Our DOWN bias correct
  BTC entry           ~$0.49             $0.454               Whales get cheaper BTC
  Bet size            $3-$10             median $4.70         Our sizing is good
  Skip hours          8 hours            Trade all hours      Whales don't skip
""")

conn.close()
