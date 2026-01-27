"""Execute Dua Lipa Super Bowl halftime NO trade."""
import os
import sys
sys.path.insert(0, 'C:/Users/Star/.local/bin/star-polymarket')

from dotenv import load_dotenv
load_dotenv('C:/Users/Star/.local/bin/star-polymarket/.env')

# Trade parameters
# Dua Lipa NOT performing at Super Bowl halftime - NO
TOKEN_ID = "88226159985644028236116842517792605031005401519713246281229752375427444280657"
TRADE_SIZE_USD = 9.0
PRICE = 0.77

def decrypt_key():
    """Decrypt the private key."""
    from crypto_utils import decrypt_key as dk
    key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    salt = os.getenv("POLYMARKET_KEY_SALT", "")
    password = "49pIMj@*3aDwzlbqN2MqDMaX"
    if key.startswith("ENC:"):
        return dk(key[4:], salt, password)
    return key

def main():
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY
    import httpx

    print("=" * 60)
    print("EXECUTING TRADE: Dua Lipa Super Bowl Halftime")
    print("=" * 60)
    print()
    print(f"Market: Will Dua Lipa perform at Super Bowl LX halftime?")
    print(f"Action: BUY NO (betting she will NOT perform)")
    print(f"Size: ${TRADE_SIZE_USD}")
    print(f"Price: ${PRICE}")
    print(f"Potential return: +{(1/PRICE - 1)*100:.0f}%")
    print()

    private_key = decrypt_key()
    proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")

    # Initialize client
    client = ClobClient(
        host="https://clob.polymarket.com",
        key=private_key,
        chain_id=137,
        signature_type=1,
        funder=proxy_address,
    )

    creds = client.derive_api_key()
    client.set_api_creds(creds)
    print("Client initialized")

    # Get current price
    try:
        price_data = client.get_last_trade_price(TOKEN_ID)
        current_price = float(price_data.get('price', PRICE))
        print(f"Current market price: ${current_price}")

        if abs(current_price - PRICE) > 0.03:
            print(f"Price moved! Using ${current_price}")
            actual_price = current_price
        else:
            actual_price = PRICE
    except Exception as e:
        print(f"Price check: {e}")
        actual_price = PRICE

    # Calculate shares
    shares = TRADE_SIZE_USD / actual_price
    print(f"Buying {shares:.4f} shares @ ${actual_price}")

    # Create order
    order_args = OrderArgs(
        price=actual_price,
        size=shares,
        side=BUY,
        token_id=TOKEN_ID,
    )

    print("\nCreating order...")
    signed_order = client.create_order(order_args)

    print("Posting order...")
    response = client.post_order(signed_order, OrderType.GTC)

    print(f"\nOrder Response:")
    print(f"  Success: {response.get('success')}")
    print(f"  Order ID: {response.get('orderID')}")
    print(f"  Status: {response.get('status')}")

    if response.get('errorMsg'):
        print(f"  Error: {response.get('errorMsg')}")

    if response.get('success'):
        payout = shares
        profit = payout - TRADE_SIZE_USD
        print(f"\n*** TRADE EXECUTED ***")
        print(f"  Market: Dua Lipa Super Bowl Halftime - NO")
        print(f"  Size: ${TRADE_SIZE_USD}")
        print(f"  Price: ${actual_price}")
        print(f"  Shares: {shares:.4f}")
        print(f"  Payout if wins: ${payout:.2f}")
        print(f"  Profit if wins: ${profit:.2f} (+{(profit/TRADE_SIZE_USD)*100:.0f}%)")

    return response

if __name__ == "__main__":
    main()
