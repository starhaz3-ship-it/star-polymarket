import json

with open('C:/Users/Star/.local/bin/star-polymarket/ta_live_results.json') as f:
    data = json.load(f)

trades = data.get('trades', {})
total_pnl = 0
wins = 0
losses = 0
by_asset = {}
by_tf = {"5m": {"w": 0, "l": 0, "pnl": 0}, "15m": {"w": 0, "l": 0, "pnl": 0}}
recent = []

for tid, t in trades.items():
    pnl = t.get('pnl')
    if pnl is None or t.get('status') != 'closed':
        continue
    total_pnl += pnl
    if pnl > 0:
        wins += 1
    else:
        losses += 1

    asset = t.get('asset', 'BTC')
    if asset not in by_asset:
        by_asset[asset] = {'w': 0, 'l': 0, 'pnl': 0}
    by_asset[asset]['pnl'] += pnl
    by_asset[asset]['w' if pnl > 0 else 'l'] += 1

    # Detect timeframe from title
    title = t.get('market_title', t.get('title', ''))
    tf = "15m"
    # 5m markets have 5-min spans like 12:00AM-12:05AM
    if any(f":{m:02d}" in title for m in [5, 10, 20, 25, 35, 40, 50, 55]):
        tf = "5m"
    by_tf[tf]['pnl'] += pnl
    by_tf[tf]['w' if pnl > 0 else 'l'] += 1

    recent.append(t)

total = wins + losses
wr = wins / total * 100 if total else 0

print("=" * 60)
print("TA LIVE TRADER (Directional) - Performance Report")
print("=" * 60)
print(f"  Total: {total} trades | {wins}W/{losses}L | {wr:.1f}% WR")
print(f"  PnL: ${total_pnl:+.2f}")
if total:
    print(f"  Avg: ${total_pnl/total:+.2f}/trade")
print()

print("BY TIMEFRAME:")
for tf, s in by_tf.items():
    t2 = s['w'] + s['l']
    if t2 == 0:
        continue
    wr2 = s['w'] / t2 * 100
    print(f"  {tf}: {s['w']}W/{s['l']}L ({wr2:.0f}%) ${s['pnl']:+.2f}")

print()
print("BY ASSET:")
for a, s in sorted(by_asset.items(), key=lambda x: -x[1]['pnl']):
    t2 = s['w'] + s['l']
    wr2 = s['w'] / t2 * 100 if t2 else 0
    print(f"  {a}: {s['w']}W/{s['l']}L ({wr2:.0f}%) ${s['pnl']:+.2f}")

# Last 10
recent.sort(key=lambda x: x.get('exit_time', x.get('entry_time', '')))
print()
print("LAST 10:")
for t in recent[-10:]:
    tag = 'W' if t.get('pnl', 0) > 0 else 'L'
    side = t.get('side', '?')
    pnl = t.get('pnl', 0)
    ep = t.get('entry_price', 0)
    xp = t.get('exit_price', 0)
    title = t.get('market_title', t.get('title', '?'))[:45]
    print(f"  [{tag}] {side:>4} ${pnl:+.2f} | in=${ep:.2f} out=${xp:.2f} | {title}")

print("=" * 60)
