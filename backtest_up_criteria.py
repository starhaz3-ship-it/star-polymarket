"""Backtest UP trade criteria using whale data."""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Load whale trades
try:
    with open('whale_watcher.db', 'rb') as f:
        pass
    import sqlite3
    conn = sqlite3.connect('whale_watcher.db')
    cursor = conn.cursor()

    # Get all crypto 15m trades
    cursor.execute("""
        SELECT whale_name, outcome, price, size, paper_pnl, status
        FROM whale_trades
        WHERE market_category = 'crypto_15m'
        AND status = 'closed'
    """)
    trades = cursor.fetchall()
    conn.close()

    if not trades:
        print("No closed trades in whale_watcher.db yet")
        sys.exit(0)

    up_trades = [(name, price, pnl) for name, outcome, price, size, pnl, status in trades if outcome and outcome.lower() == 'up']
    down_trades = [(name, price, pnl) for name, outcome, price, size, pnl, status in trades if outcome and outcome.lower() == 'down']

    print(f"=== WHALE CRYPTO 15M BACKTEST ===")
    print(f"Total trades: {len(trades)}")
    print(f"UP trades: {len(up_trades)}")
    print(f"DOWN trades: {len(down_trades)}")

    # UP win rate by entry price
    print(f"\n=== UP TRADES BY ENTRY PRICE ===")
    for price_max in [0.35, 0.40, 0.45, 0.50]:
        filtered = [t for t in up_trades if t[1] <= price_max]
        wins = sum(1 for t in filtered if t[2] and t[2] > 0)
        total = len(filtered)
        wr = wins / total * 100 if total else 0
        print(f"Entry <= ${price_max:.2f}: {wins}/{total} = {wr:.1f}% WR")

    # DOWN win rate by entry price
    print(f"\n=== DOWN TRADES BY ENTRY PRICE ===")
    for price_max in [0.35, 0.40, 0.45, 0.50]:
        filtered = [t for t in down_trades if t[1] <= price_max]
        wins = sum(1 for t in filtered if t[2] and t[2] > 0)
        total = len(filtered)
        wr = wins / total * 100 if total else 0
        print(f"Entry <= ${price_max:.2f}: {wins}/{total} = {wr:.1f}% WR")

except Exception as e:
    print(f"Error: {e}")
    print("\nUsing CSV trade history instead...")

# Also check CSV if available
try:
    import csv
    csv_path = r"C:\Users\Star\Downloads\Polymarket-History-2026-02-04 (1).csv"

    up_wins = up_losses = down_wins = down_losses = 0
    up_pnl = down_pnl = 0

    # Parse and match buys to redeems
    buys = {}  # market -> {side, cost}
    redeems = {}  # market -> amount

    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            market = row['marketName']
            if 'Up or Down' not in market:
                continue
            action = row['action']
            usdc = float(row['usdcAmount']) if row['usdcAmount'] else 0
            side = row['tokenName']

            if action == 'Buy':
                if market not in buys:
                    buys[market] = {'side': side, 'cost': 0}
                buys[market]['cost'] += usdc
            elif action == 'Redeem' and usdc > 0:
                redeems[market] = redeems.get(market, 0) + usdc

    # Calculate win rates
    for market, buy in buys.items():
        if market not in redeems:
            continue  # Still open
        cost = buy['cost']
        redeemed = redeems[market]
        won = redeemed > cost
        side = buy['side']

        if side == 'Up':
            if won:
                up_wins += 1
                up_pnl += redeemed - cost
            else:
                up_losses += 1
                up_pnl += redeemed - cost
        elif side == 'Down':
            if won:
                down_wins += 1
                down_pnl += redeemed - cost
            else:
                down_losses += 1
                down_pnl += redeemed - cost

    print(f"\n=== YOUR ACTUAL TRADE HISTORY ===")
    up_total = up_wins + up_losses
    down_total = down_wins + down_losses

    up_wr = up_wins / up_total * 100 if up_total else 0
    down_wr = down_wins / down_total * 100 if down_total else 0

    print(f"UP:   {up_wins}W/{up_losses}L = {up_wr:.1f}% WR | PnL: ${up_pnl:+.2f}")
    print(f"DOWN: {down_wins}W/{down_losses}L = {down_wr:.1f}% WR | PnL: ${down_pnl:+.2f}")

    print(f"\n=== RECOMMENDATION ===")
    if down_wr > 60 and up_wr < 50:
        print("DOWN ONLY MODE is correct - UP trades are hurting you")
        print(f"Keep UP disabled until you have {10}+ DOWN wins at 70%+ WR")
    elif up_wr >= 50:
        print("UP trades are performing OK - consider enabling with strict filters")

except Exception as e:
    print(f"CSV analysis error: {e}")
