"""Execute batch of high-probability trades."""
import os
import sys
sys.path.insert(0, 'C:/Users/Star/.local/bin/star-polymarket')

from dotenv import load_dotenv
load_dotenv('C:/Users/Star/.local/bin/star-polymarket/.env')

from crypto_utils import decrypt_key as dk

def decrypt_key():
    key = os.getenv('POLYMARKET_PRIVATE_KEY', '')
    salt = os.getenv('POLYMARKET_KEY_SALT', '')
    password = '49pIMj@*3aDwzlbqN2MqDMaX'
    if key.startswith('ENC:'):
        return dk(key[4:], salt, password)
    return key

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

def main():
    private_key = decrypt_key()
    proxy_address = os.getenv('POLYMARKET_PROXY_ADDRESS', '')

    client = ClobClient(
        host='https://clob.polymarket.com',
        key=private_key,
        chain_id=137,
        signature_type=1,
        funder=proxy_address,
    )

    creds = client.derive_api_key()
    client.set_api_creds(creds)
    print('Client initialized')
    print()

    # Define trades with correct token IDs
    trades = [
        {
            'name': 'Cardi B at Super Bowl halftime - YES',
            'token': '40188011078723717028570277979743079995042968731892406119337197516891041207608',
            'price': 0.61,
            'size': 5.0,
        },
        {
            'name': 'Patriots NOT winning Super Bowl - NO',
            'token': '49985214708919646661175099546558878672426321417556753698510266808095631910814',
            'price': 0.68,
            'size': 5.0,
        },
    ]

    results = []

    for t in trades:
        print('=' * 60)
        print(f"TRADE: {t['name']}")
        print('=' * 60)

        shares = t['size'] / t['price']
        print(f"Buying {shares:.2f} shares @ ${t['price']}")

        try:
            order = OrderArgs(
                price=t['price'],
                size=shares,
                side=BUY,
                token_id=t['token'],
            )

            signed = client.create_order(order)
            resp = client.post_order(signed, OrderType.GTC)

            success = resp.get('success', False)
            status = resp.get('status', 'unknown')

            print(f"Success: {success}")
            print(f"Status: {status}")

            if resp.get('errorMsg'):
                print(f"Error: {resp.get('errorMsg')}")

            if success:
                profit = shares - t['size']
                print(f"Payout if wins: ${shares:.2f} | Profit: +${profit:.2f} (+{(profit/t['size'])*100:.0f}%)")

            results.append({
                'name': t['name'],
                'success': success,
                'status': status,
                'size': t['size'],
                'price': t['price'],
            })

        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'name': t['name'],
                'success': False,
                'status': 'error',
                'error': str(e),
            })

        print()

    # Summary
    print('=' * 60)
    print('EXECUTION SUMMARY')
    print('=' * 60)

    filled = [r for r in results if r.get('status') == 'matched']
    live = [r for r in results if r.get('status') == 'live']
    failed = [r for r in results if not r.get('success')]

    print(f"Filled: {len(filled)}")
    print(f"Live (pending): {len(live)}")
    print(f"Failed: {len(failed)}")
    print()

    total_invested = sum(r['size'] for r in results if r.get('success'))
    print(f"Total invested: ${total_invested:.2f}")

if __name__ == '__main__':
    main()
