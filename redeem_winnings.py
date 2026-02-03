"""
Auto-redeem winning positions from resolved Polymarket markets.

Uses the Conditional Tokens Framework (CTF) contract on Polygon
to redeem winning shares for USDC.
"""
import os
import sys
import json
import httpx
from pathlib import Path
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

# Contract addresses on Polygon
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEG_RISK_ADAPTER_ADDRESS = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
PROXY_FACTORY_ADDRESS = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"

# Minimal ABIs
CTF_ABI = [
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]

PROXY_FACTORY_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"name": "to", "type": "address"},
                    {"name": "typeCode", "type": "uint256"},
                    {"name": "data", "type": "bytes"},
                    {"name": "value", "type": "uint256"},
                ],
                "name": "_txns",
                "type": "tuple[]",
            }
        ],
        "name": "proxy",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]


def decrypt_key() -> str:
    """Decrypt private key from env."""
    key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if key.startswith("ENC:"):
        sys.path.insert(0, str(Path(__file__).parent))
        from crypto_utils import decrypt_key as dk
        salt = os.getenv("POLYMARKET_KEY_SALT", "")
        password = os.getenv("POLYMARKET_PASSWORD", "")
        if not password:
            import getpass
            password = getpass.getpass("Enter wallet password: ")
        return dk(key[4:], salt, password)
    return key


def get_redeemable_positions(proxy_address: str) -> list:
    """Fetch all redeemable positions from Polymarket data API."""
    r = httpx.get(
        "https://data-api.polymarket.com/positions",
        params={"user": proxy_address, "sizeThreshold": "0"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    positions = r.json()
    redeemable = [p for p in positions if p.get("redeemable", False)]
    return redeemable


def main():
    proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
    if not proxy_address:
        print("ERROR: POLYMARKET_PROXY_ADDRESS not set")
        return

    # Get redeemable positions
    positions = get_redeemable_positions(proxy_address)
    if not positions:
        print("No redeemable positions found.")
        return

    print(f"Found {len(positions)} redeemable positions:\n")
    total_value = 0
    for p in positions:
        title = p.get("title", "")[:60]
        outcome = p.get("outcome", "")
        size = float(p.get("size", 0) or 0)
        value = float(p.get("currentValue", 0) or 0)
        cond_id = p.get("conditionId", "")
        total_value += value
        print(f"  {outcome}: {size:.1f} shares = ${value:.2f} | {title}")
        print(f"    conditionId: {cond_id[:20]}...")

    print(f"\nTotal redeemable value: ${total_value:.2f}")

    # Connect to Polygon
    w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))
    if not w3.is_connected():
        # Fallback RPC
        w3 = Web3(Web3.HTTPProvider("https://rpc.ankr.com/polygon"))
    if not w3.is_connected():
        print("ERROR: Could not connect to Polygon RPC")
        return

    print(f"\nConnected to Polygon (block: {w3.eth.block_number})")

    # Decrypt private key
    private_key = decrypt_key()
    if not private_key:
        print("ERROR: Could not decrypt private key")
        return

    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    account = w3.eth.account.from_key(private_key)
    print(f"EOA wallet: {account.address}")
    print(f"Proxy wallet: {proxy_address}")

    # Check MATIC balance for gas
    balance = w3.eth.get_balance(account.address)
    matic = w3.from_wei(balance, "ether")
    print(f"MATIC balance: {matic:.4f}")
    if matic < 0.01:
        print("WARNING: Low MATIC balance, may not have enough gas")

    # Set up contracts
    ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF_ADDRESS), abi=CTF_ABI)
    factory = w3.eth.contract(
        address=Web3.to_checksum_address(PROXY_FACTORY_ADDRESS), abi=PROXY_FACTORY_ABI
    )

    # Process each redeemable position
    redeemed = 0
    failed = 0
    condition_ids_done = set()

    for p in positions:
        cond_id = p.get("conditionId", "")
        if not cond_id or cond_id in condition_ids_done:
            continue
        condition_ids_done.add(cond_id)

        title = p.get("title", "")[:50]
        outcome = p.get("outcome", "")
        size = float(p.get("size", 0) or 0)

        print(f"\nRedeeming: {outcome} {size:.1f} shares | {title}...")

        try:
            # Encode the CTF redeemPositions call
            parent_collection_id = bytes(32)  # HashZero
            condition_id_bytes = Web3.to_bytes(hexstr=cond_id)
            index_sets = [1, 2]  # Binary market: YES + NO

            redeem_data = ctf.encode_abi(
                "redeemPositions",
                args=[
                    Web3.to_checksum_address(USDC_ADDRESS),
                    parent_collection_id,
                    condition_id_bytes,
                    index_sets,
                ],
            )

            # Wrap in proxy transaction
            proxy_txn = (
                Web3.to_checksum_address(CTF_ADDRESS),  # to
                1,  # typeCode (1 for proxy wallet)
                Web3.to_bytes(hexstr=redeem_data),  # data
                0,  # value
            )

            # Build and send the proxy transaction
            nonce = w3.eth.get_transaction_count(account.address)
            gas_price = w3.eth.gas_price

            txn = factory.functions.proxy([proxy_txn]).build_transaction(
                {
                    "from": account.address,
                    "nonce": nonce,
                    "gasPrice": min(gas_price * 2, w3.to_wei(100, "gwei")),
                    "gas": 300000,
                }
            )

            # Sign and send
            signed = w3.eth.account.sign_transaction(txn, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  TX sent: {tx_hash.hex()}")

            # Wait for confirmation
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            if receipt["status"] == 1:
                print(f"  SUCCESS (gas used: {receipt['gasUsed']})")
                redeemed += 1
            else:
                print(f"  FAILED (reverted)")
                failed += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"REDEMPTION COMPLETE")
    print(f"  Redeemed: {redeemed}")
    print(f"  Failed: {failed}")
    print(f"  Total value claimed: ~${total_value:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
