"""Execute a single trade on Polymarket."""
import os
import sys
sys.path.insert(0, 'C:/Users/Star/.local/bin/star-polymarket')

from dotenv import load_dotenv
load_dotenv('C:/Users/Star/.local/bin/star-polymarket/.env')

# Trade parameters
# BTC > $86,000 on January 28 - YES
TOKEN_ID = "65224904822079901363762017933560954830891511836817593442796381106577645373684"
TRADE_SIZE_USD = 5.0
PRICE = 0.835  # Current YES price

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

    private_key = decrypt_key()
    proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")

    print(f"Proxy address: {proxy_address}")
    print(f"Trade: ${TRADE_SIZE_USD} on BTC > $86K Jan 28 YES @ {PRICE}")

    # Initialize client with Magic wallet settings
    client = ClobClient(
        host="https://clob.polymarket.com",
        key=private_key,
        chain_id=137,  # Polygon
        signature_type=1,  # Magic wallet
        funder=proxy_address,
    )

    # Derive and set API credentials
    creds = client.derive_api_key()
    client.set_api_creds(creds)

    print("Client initialized")

    # Get current price
    try:
        price_data = client.get_last_trade_price(TOKEN_ID)
        print(f"Current price: {price_data}")
    except Exception as e:
        print(f"Price check error: {e}")

    # Calculate shares
    shares = TRADE_SIZE_USD / PRICE
    print(f"Buying {shares:.4f} shares @ ${PRICE}")

    # Create order
    order_args = OrderArgs(
        price=PRICE,
        size=shares,
        side=BUY,
        token_id=TOKEN_ID,
    )

    print("Creating order...")
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
        print(f"\n*** TRADE EXECUTED ***")
        print(f"  Market: BTC > $86,000 on January 28")
        print(f"  Side: YES")
        print(f"  Size: ${TRADE_SIZE_USD}")
        print(f"  Price: {PRICE}")
        print(f"  Shares: {shares:.4f}")
        print(f"  Payout if wins: ${shares:.2f}")

    return response

if __name__ == "__main__":
    main()
