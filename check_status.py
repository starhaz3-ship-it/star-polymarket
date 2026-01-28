"""Check current market status and positions."""
import httpx

# Get BTC price
r = httpx.get('https://api.binance.com/api/v3/ticker/price', params={'symbol': 'BTCUSDT'})
btc = float(r.json()['price'])
print(f'Current BTC: ${btc:,.2f}')
print()

# Check our real trade - BTC > $86K Jan 28 YES
TOKEN_ID = '65224904822079901363762017933560954830891511836817593442796381106577645373684'
try:
    r = httpx.get('https://clob.polymarket.com/price', params={'token_id': TOKEN_ID})
    data = r.json()
    current = float(data.get('price', 0))

    print('YOUR REAL TRADE:')
    print('-' * 40)
    print('Market: BTC > $86,000 on January 28')
    print('Side: YES')

    entry = 0.835
    shares = 5.988
    cost = 5.0

    print(f'Entry Price: ${entry:.3f}')
    print(f'Current Price: ${current:.3f}')
    print(f'Shares: {shares:.3f}')
    print(f'Cost: ${cost:.2f}')

    unrealized = (current - entry) * shares
    print(f'Unrealized P&L: ${unrealized:+.2f}')

    if btc > 86000:
        print(f'Status: BTC (${btc:,.0f}) is ABOVE $86K - WINNING')
        potential_profit = shares - cost
        print(f'If market settles YES: ${potential_profit:+.2f} profit')
    else:
        print(f'Status: BTC (${btc:,.0f}) is BELOW $86K - AT RISK')
except Exception as e:
    print(f'Error checking trade: {e}')

print()
print('PAPER TRADING SUMMARY:')
print('-' * 40)

# Summary of all systems
systems = {
    'ML Trader V2': {'positions': 4, 'invested': 18.55, 'pnl': 0},
    'ML Trader V1': {'positions': 4, 'invested': 29.00, 'pnl': 0},
    'Forward Tester': {'positions': 5, 'invested': 50.00, 'pnl': 0},
}

total_invested = sum(s['invested'] for s in systems.values())
total_pnl = sum(s['pnl'] for s in systems.values())

for name, data in systems.items():
    print(f'{name}: {data["positions"]} positions, ${data["invested"]:.2f} invested')

print()
print(f'Total Paper Positions: {sum(s["positions"] for s in systems.values())}')
print(f'Total Paper Invested: ${total_invested:.2f}')
print(f'Total Paper P&L: ${total_pnl:+.2f}')
print()
print('Note: Markets have not resolved yet. P&L will update when BTC markets settle.')
