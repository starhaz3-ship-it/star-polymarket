import json, time
from datetime import datetime, timezone

with open('sniper_5m_live_results.json') as f:
    data = json.load(f)

now = time.time()
cutoff = now - 86400

resolved = data.get('resolved', [])
recent = [t for t in resolved if (t.get('entry_ts') or t.get('timestamp') or 0) > cutoff]

print("=== SNIPER 5M LIVE - LAST 24 HOURS ===")
print(f"Total resolved: {len(resolved)} | Last 24h: {len(recent)}")
print()

wins = losses = 0
total_pnl = 0.0
for t in recent:
    pnl = t.get('pnl', t.get('pnl_usd', 0))
    won = t.get('won', pnl > 0)
    side = t.get('side', '?')
    entry_p = t.get('entry_price', 0)
    size = t.get('trade_size', t.get('size', 0))
    ts = t.get('entry_ts', t.get('timestamp', 0))
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%H:%M UTC') if ts else '?'
    if won:
        wins += 1
    else:
        losses += 1
    total_pnl += pnl
    marker = 'W' if won else 'L'
    print(f"  [{marker}] {dt} | {side:4s} @ ${entry_p:.2f} | ${size:.2f} | PnL: ${pnl:+.2f}")

print()
total = wins + losses
wr = wins / total * 100 if total > 0 else 0
avg_w = sum(t.get('pnl', 0) for t in recent if t.get('won', t.get('pnl', 0) > 0)) / max(wins, 1)
avg_l = sum(t.get('pnl', 0) for t in recent if not t.get('won', t.get('pnl', 0) > 0)) / max(losses, 1)
print(f"Record: {wins}W/{losses}L ({wr:.0f}% WR)")
print(f"PnL: ${total_pnl:+.2f}")
print(f"Avg win: ${avg_w:+.2f} | Avg loss: ${avg_l:+.2f}")
if avg_l != 0:
    print(f"W:L ratio: {abs(avg_w / avg_l):.2f}x")

all_pnl = sum(t.get('pnl', t.get('pnl_usd', 0)) for t in resolved)
all_w = sum(1 for t in resolved if t.get('won', t.get('pnl', 0) > 0))
all_l = len(resolved) - all_w
all_wr = all_w / (all_w + all_l) * 100 if (all_w + all_l) > 0 else 0
print(f"\nAll-time: {all_w}W/{all_l}L ({all_wr:.0f}% WR) | PnL: ${all_pnl:+.2f}")
