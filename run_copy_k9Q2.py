"""
Live copy trader for k9Q2mX4L8A7ZP3R
$422K PnL whale - crypto hourly Up/Down specialist
"""

import asyncio
import os
import sys
import time
from datetime import datetime

# Set password before imports
os.environ["POLYMARKET_PASSWORD"] = "49pIMj@*3aDwzlbqN2MqDMaX"

from arbitrage.copy_trader import CopyTrader
from arbitrage.executor import Executor
from arbitrage.config import config

# Configuration for k9Q2mX4L8A7ZP3R
TARGET_WALLET = "0xd0d6053c3c37e727402d84c14069780d360993aa"
TARGET_NAME = "k9Q2mX4L8A7ZP3R"
MAX_TRADE_SIZE = 100.0  # $100 max per trade
MIN_TRADE_SIZE = 5.0    # $5 minimum
SCALE_FACTOR = 0.05     # 5% of their size (they trade larger)
POLL_INTERVAL = 15      # Check every 15 seconds


def place_copy_order(executor: Executor, token_id: str, amount_usd: float, price: float) -> dict:
    """Place a market order for copy trading."""
    if not executor._initialized:
        return {"success": False, "error": "Executor not initialized"}

    try:
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        # Calculate size in shares
        size = amount_usd / price if price > 0 else amount_usd

        order_args = OrderArgs(
            price=price,
            size=size,
            side=BUY,
            token_id=token_id,
        )

        print(f"[Copy] Placing order: {size:.2f} shares @ ${price:.4f} = ${amount_usd:.2f}")

        signed_order = executor.client.create_order(order_args)
        response = executor.client.post_order(signed_order, OrderType.GTC)

        return {
            "success": response.get("success", False),
            "order_id": response.get("orderID", ""),
            "status": response.get("status", ""),
            "error": response.get("errorMsg", ""),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def run_live_copy():
    """Run live copy trading for k9Q2mX4L8A7ZP3R."""

    print(f"[{datetime.now().strftime('%H:%M:%S')}] LIVE COPY TRADER - {TARGET_NAME}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Max: ${MAX_TRADE_SIZE} | Min: ${MIN_TRADE_SIZE} | Scale: {int(SCALE_FACTOR*100)}%")

    # Initialize executor
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing executor...")
    executor = Executor()

    if not executor._initialized:
        print("ERROR: Failed to initialize executor")
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Executor READY - monitoring for trades...")

    # Initialize copy trader
    trader = CopyTrader(
        target_wallet=TARGET_WALLET,
        scale_factor=SCALE_FACTOR,
        min_position_usd=50.0,  # Only copy positions > $50
        state_file=f"copy_trader_{TARGET_NAME}_state.json",
    )

    # Override config for this trader
    config.MAX_POSITION_SIZE = MAX_TRADE_SIZE
    config.MIN_POSITION_SIZE = MIN_TRADE_SIZE

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(trader.executed_copies)} existing trades")

    while True:
        try:
            signals = await trader.scan_for_signals()

            for signal in signals:
                if signal.signal_type == "ENTER":
                    # Cap our size
                    our_size = min(signal.our_size, MAX_TRADE_SIZE)
                    our_size = max(our_size, MIN_TRADE_SIZE)

                    print(f"\n{'='*50}")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] COPY SIGNAL: BUY {signal.outcome}")
                    print(f"Market: {signal.market_slug}")
                    print(f"Target: {signal.target_size:.0f} shares @ ${signal.target_price:.2f}")
                    print(f"Our size: ${our_size:.2f}")
                    print(f"{'='*50}")

                    # Execute the trade
                    try:
                        result = place_copy_order(
                            executor=executor,
                            token_id=signal.token_id,
                            amount_usd=our_size,
                            price=signal.target_price if signal.target_price > 0 else 0.50,
                        )

                        if result.get("success"):
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] ORDER PLACED: {result['order_id']}")
                            trader.mark_executed(signal.condition_id)
                        else:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] ORDER FAILED: {result.get('error')}")

                    except Exception as e:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error executing: {e}")

                elif signal.signal_type == "EXIT":
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] EXIT SIGNAL: {signal.market_slug}")
                    # For now, log exits but don't auto-sell
                    # Markets resolve automatically anyway

            # Status update every 5 minutes
            if int(time.time()) % 300 < POLL_INTERVAL:
                status = trader.get_status()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status['tracked_positions']} positions, "
                      f"${status['total_target_value']:,.0f} target value, {status['executed_copies']} copied")

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")

        await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"LIVE COPY TRADER - {TARGET_NAME}")
    print(f"Wallet: {TARGET_WALLET[:10]}...")
    print(f"Max trade: ${MAX_TRADE_SIZE} | Scale: {int(SCALE_FACTOR*100)}%")
    print(f"{'='*60}\n")

    if "--confirm" not in sys.argv and "-y" not in sys.argv:
        confirm = input("Start LIVE trading with REAL money? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    asyncio.run(run_live_copy())
