"""
Swap POL (native) -> USDC on Polygon via KyberSwap Aggregator.
No API key needed. Keeps 5 POL for emergency gas, swaps the rest.

Usage:
  python swap_pol_to_usdc.py          # Show quote only (dry run)
  python swap_pol_to_usdc.py --execute  # Actually execute the swap
"""
import os
import sys
import httpx
from web3 import Web3
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

# --- Config ---
RPC_URL = "https://polygon-bor.publicnode.com"
SIGNER = "0xD375494Fd97366F543DAB3CB88684EFE738DCd40"
NATIVE_TOKEN = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
USDC_POLYGON = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"  # Native USDC
KEEP_POL = 5  # Keep 5 POL for emergency gas
SLIPPAGE_BPS = 100  # 1%
KYBER_BASE = "https://aggregator-api.kyberswap.com"
CLIENT_ID = "star-polymarket-bot"


def get_private_key():
    from crypto_utils import decrypt_key
    password = os.getenv("POLYMARKET_PASSWORD", "")
    if not password:
        print("ERROR: POLYMARKET_PASSWORD not set")
        sys.exit(1)
    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if pk.startswith("ENC:"):
        pk = decrypt_key(pk[4:], os.getenv("POLYMARKET_KEY_SALT", ""), password)
    return pk


def main():
    execute = "--execute" in sys.argv
    w3 = Web3(Web3.HTTPProvider(RPC_URL))

    # Check balance
    bal_wei = w3.eth.get_balance(SIGNER)
    bal_pol = bal_wei / 1e18
    print(f"Wallet: {SIGNER}")
    print(f"Balance: {bal_pol:.4f} POL")
    print(f"Keeping: {KEEP_POL} POL for gas")

    swap_pol = bal_pol - KEEP_POL
    if swap_pol < 1:
        print(f"Not enough to swap ({swap_pol:.2f} POL after reserve)")
        return

    swap_wei = str(int(swap_pol * 1e18))
    print(f"Swapping: {swap_pol:.4f} POL -> USDC")
    print()

    # Step 1: Get route
    headers = {"X-Client-Id": CLIENT_ID}
    print("Getting best route from KyberSwap...")
    r = httpx.get(
        f"{KYBER_BASE}/polygon/api/v1/routes",
        headers=headers,
        params={
            "tokenIn": NATIVE_TOKEN,
            "tokenOut": USDC_POLYGON,
            "amountIn": swap_wei,
        },
        timeout=15,
    )
    r.raise_for_status()
    resp = r.json()
    if resp.get("code") != 0:
        print(f"Route error: {resp}")
        return

    route_data = resp["data"]
    route_summary = route_data["routeSummary"]
    router_address = route_data["routerAddress"]

    amount_out = int(route_summary["amountOut"]) / 1e6
    print(f"Quote: {swap_pol:.2f} POL -> {amount_out:.2f} USDC")
    print(f"Router: {router_address}")

    if not execute:
        print()
        print("DRY RUN - add --execute to swap for real")
        return

    # Step 2: Build transaction
    print("Building swap transaction...")
    build_r = httpx.post(
        f"{KYBER_BASE}/polygon/api/v1/route/build",
        headers=headers,
        json={
            "routeSummary": route_summary,
            "sender": SIGNER,
            "recipient": SIGNER,
            "slippageTolerance": SLIPPAGE_BPS,
        },
        timeout=15,
    )
    build_r.raise_for_status()
    build_resp = build_r.json()
    if build_resp.get("code") != 0:
        print(f"Build error: {build_resp}")
        return

    build_data = build_resp["data"]

    # Step 3: Sign and send
    pk = get_private_key()
    tx = {
        "from": SIGNER,
        "to": Web3.to_checksum_address(router_address),
        "data": build_data["data"],
        "value": int(swap_wei),
        "gas": 500_000,
        "gasPrice": w3.eth.gas_price,
        "nonce": w3.eth.get_transaction_count(SIGNER),
        "chainId": 137,
    }

    # Estimate gas
    try:
        est_gas = w3.eth.estimate_gas(tx)
        tx["gas"] = int(est_gas * 1.2)  # 20% buffer
        print(f"Estimated gas: {est_gas} (using {tx['gas']})")
    except Exception as e:
        print(f"Gas estimate failed ({e}), using 500k default")

    signed = w3.eth.account.sign_transaction(tx, private_key=pk)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"TX sent: https://polygonscan.com/tx/{tx_hash.hex()}")

    # Wait for confirmation
    print("Waiting for confirmation...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    if receipt["status"] == 1:
        new_bal = w3.eth.get_balance(SIGNER) / 1e18
        print(f"SUCCESS! Gas used: {receipt['gasUsed']}")
        print(f"Remaining POL: {new_bal:.4f}")

        # Check USDC balance
        usdc_abi = [{"constant": True, "inputs": [{"name": "", "type": "address"}],
                     "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
                     "type": "function"}]
        usdc = w3.eth.contract(address=Web3.to_checksum_address(USDC_POLYGON), abi=usdc_abi)
        usdc_bal = usdc.functions.balanceOf(SIGNER).call() / 1e6
        print(f"USDC in signer wallet: {usdc_bal:.2f}")
        print()
        print("NOTE: USDC is in your signer wallet, not Polymarket.")
        print("To deposit to Polymarket, transfer to your proxy wallet or use the deposit flow.")
    else:
        print("FAILED! Transaction reverted.")
        print(f"TX: https://polygonscan.com/tx/{tx_hash.hex()}")


if __name__ == "__main__":
    main()
