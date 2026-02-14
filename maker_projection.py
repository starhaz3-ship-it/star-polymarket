import json
from datetime import datetime, timezone

with open('C:/Users/Star/.local/bin/star-polymarket/maker_results.json') as f:
    data = json.load(f)

stats = data['stats']
resolved = data['resolved']
paired = [r for r in resolved if r.get('paired')]
partial = [r for r in resolved if r.get('partial')]

start = datetime.fromisoformat(stats['start_time'])
now = datetime.now(timezone.utc)
hours = (now - start).total_seconds() / 3600

CURRENT_SIZE = 3.0
sizes = [3, 10, 100]

print('=' * 72)
print(f'MAKER BOT SCALING PROJECTIONS (based on {hours:.0f}h paper data)')
print(f'{len(paired)} paired trades | {len(partial)} partial | 86.3% paired WR')
print('=' * 72)
print()

for size in sizes:
    scale = size / CURRENT_SIZE

    paired_pnl = sum(r['pnl'] for r in paired) * scale
    partial_pnl = sum(r['pnl'] for r in partial) * scale
    total_pnl = paired_pnl + partial_pnl

    avg_paired = paired_pnl / len(paired) if paired else 0
    avg_partial = partial_pnl / len(partial) if partial else 0

    pnl_per_hour = total_pnl / hours
    daily = pnl_per_hour * 24
    monthly = daily * 30
    yearly = daily * 365

    best_pair = stats['best_pair_pnl'] * scale
    worst_pair = stats['worst_pair_pnl'] * scale
    max_exposure = size * 2 * 5  # 2 sides x 5 max concurrent pairs
    capital_needed = max_exposure * 1.5

    daily_roi = (daily / capital_needed * 100) if capital_needed > 0 else 0
    monthly_roi = (monthly / capital_needed * 100) if capital_needed > 0 else 0

    print(f'--- ${size}/side (${size*2}/pair) ---')
    print(f'  Scale factor:     {scale:.1f}x from current')
    print(f'  Capital needed:   ~${capital_needed:.0f} (max exposure ${max_exposure:.0f})')
    print()
    print(f'  Paired PnL:       ${paired_pnl:+.2f} (avg ${avg_paired:+.2f}/trade)')
    print(f'  Partial PnL:      ${partial_pnl:+.2f} (avg ${avg_partial:+.2f}/trade)')
    print(f'  NET PnL ({hours:.0f}h):    ${total_pnl:+.2f}')
    print()
    print(f'  Hourly:           ${pnl_per_hour:+.2f}/hr')
    print(f'  DAILY:            ${daily:+.2f}/day')
    print(f'  MONTHLY:          ${monthly:+,.2f}/month')
    print(f'  YEARLY:           ${yearly:+,.2f}/year')
    print()
    print(f'  Daily ROI:        {daily_roi:.1f}%')
    print(f'  Monthly ROI:      {monthly_roi:.0f}%')
    print(f'  Best single pair: ${best_pair:+.2f}')
    print(f'  Worst single pair:${worst_pair:+.2f}')
    print(f'  Worst partial:    ${-size * 1.0:+.2f} (full loss on one side)')
    print()

print('=' * 72)
print('CAVEATS:')
print('  1. Linear scaling assumes same fill rates at larger sizes')
print('     (likely WORSE - bigger orders harder to fill passively)')
print('  2. Partial losses scale too - one bad partial at $100 = -$100')
print('  3. Polymarket 15m orderbooks are THIN - $100/side may not fill')
print('  4. Current account balance: ~$38 CLOB (can only run $3-5/side)')
print('  5. Slippage increases with size - paper fills are optimistic')
print('=' * 72)
