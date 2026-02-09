"""Audit all live trades and recalculate real PnL."""
import json

with open('ta_live_results.json') as f:
    data = json.load(f)

trades = data.get('trades', {})
sorted_trades = sorted(
    [(k, t) for k, t in trades.items() if t.get('status') == 'closed'],
    key=lambda x: x[1].get('entry_time', '')
)

# Recalculate PnL using ACTUAL resolution (wins=$1.00, losses=$0.00)
# A trade 'wins' if exit_price > entry_price
print('=== RECALCULATED PnL (actual resolution $1.00/$0.00) ===')
header = f"{'#':>3} | {'Side':>5} | {'Entry':>6} | {'Size':>6} | {'Shrs':>5} | {'Old PnL':>8} | {'Real PnL':>8} | {'Diff':>6}"
print(header)
print('-' * len(header))

total_old = 0
total_real = 0
total_fees = 0

for i, (k, t) in enumerate(sorted_trades, 1):
    entry = t['entry_price']
    exit_p = t.get('exit_price', 0)
    size = t['size_usd']
    old_pnl = t.get('pnl', 0)
    shares = size / entry if entry > 0 else 0

    won = exit_p > entry  # price went our way

    if won:
        # Shares resolve to $1.00, we get shares * $1.00
        gross_return = shares * 1.00
        real_pnl = gross_return - size  # profit = return - cost
        # Polymarket charges ~2% fee on winnings
        fee = real_pnl * 0.02
        real_pnl_after_fee = real_pnl - fee
    else:
        # Shares resolve to $0.00, total loss
        real_pnl = -size
        real_pnl_after_fee = -size
        fee = 0

    total_old += old_pnl
    total_real += real_pnl_after_fee
    total_fees += fee
    diff = real_pnl_after_fee - old_pnl

    wl = 'W' if won else 'L'
    print(f"{i:>3} | {t['side']:>5} | ${entry:.3f} | ${size:>5.2f} | {shares:>4.1f} | ${old_pnl:>+7.2f} | ${real_pnl_after_fee:>+7.2f} | ${diff:>+5.2f} {wl}")

print('-' * len(header))
print(f"Old total PnL:  ${total_old:.2f}")
print(f"Real total PnL: ${total_real:.2f} (after 2% fees on wins)")
print(f"Total fees:     ${total_fees:.2f}")
print(f"Difference:     ${total_real - total_old:.2f}")
print()
print(f"Initial deposit: $74.48")
print(f"Expected with OLD PnL: ${74.48 + total_old:.2f}")
print(f"Expected with REAL PnL: ${74.48 + total_real:.2f}")
print(f"User says actual: $73.14")
print(f"Gap (real vs user): ${74.48 + total_real - 73.14:.2f}")
