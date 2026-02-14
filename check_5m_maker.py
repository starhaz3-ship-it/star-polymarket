import json

with open('C:/Users/Star/.local/bin/star-polymarket/maker_results.json') as f:
    data = json.load(f)

resolved = data.get('resolved', [])
active = data.get('active', {})

m5_all = [r for r in resolved if r.get('duration_min', 15) == 5]
m5_paired = [r for r in m5_all if r.get('paired')]
m5_partial = [r for r in m5_all if r.get('partial')]
m5_cancelled = [r for r in m5_all if not r.get('paired') and not r.get('partial')]

print("5-MINUTE MAKER BOT BREAKDOWN")
print("=" * 50)
print(f"Total 5m resolved: {len(m5_all)}")
print(f"  Paired:    {len(m5_paired)}")
print(f"  Partial:   {len(m5_partial)}")
print(f"  Cancelled: {len(m5_cancelled)}")

if m5_paired:
    pnl = sum(r['pnl'] for r in m5_paired)
    wr = sum(1 for r in m5_paired if r['pnl'] > 0) / len(m5_paired) * 100
    print(f"  Paired PnL: ${pnl:+.2f} ({wr:.0f}% WR)")
if m5_partial:
    pnl = sum(r['pnl'] for r in m5_partial)
    print(f"  Partial PnL: ${pnl:+.2f}")

total_5m_pnl = sum(r.get('pnl', 0) for r in m5_all)
print(f"  NET 5m PnL: ${total_5m_pnl:+.2f}")

# Active 5m
active_5m = [p for p in active.values() if p.get('duration_min', 15) == 5]
print(f"\n  Active 5m positions: {len(active_5m)}")
for p in active_5m:
    up = "FILLED" if p.get('up_filled') else "open"
    dn = "FILLED" if p.get('down_filled') else "open"
    paired = "PAIRED" if p.get('paired') else ""
    q = p.get('question', '?')[:50]
    print(f"    BTC | UP:{up} DN:{dn} {paired} | {q}")

# All 5m resolved
if m5_all:
    print(f"\nALL 5m RESOLVED ({len(m5_all)}):")
    for r in m5_all:
        typ = "PAIR" if r.get('paired') else ("PART" if r.get('partial') else "CANC")
        print(f"  {r.get('asset','?')} {typ} ${r.get('pnl',0):+.2f} | {r.get('question','?')[:50]}")
else:
    print("\nNo 5m trades resolved yet.")

# Compare with 15m
m15_all = [r for r in resolved if r.get('duration_min', 15) != 5]
m15_paired = [r for r in m15_all if r.get('paired')]
print(f"\nCOMPARISON:")
print(f"  15m: {len(m15_paired)} paired, ${sum(r['pnl'] for r in m15_paired):+.2f}")
print(f"   5m: {len(m5_paired)} paired, ${sum(r['pnl'] for r in m5_paired):+.2f}")
