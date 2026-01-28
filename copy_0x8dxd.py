"""
Copy trader for 0x8dxd - Crypto hourly Up/Down specialist.

$885K PnL, trades BTC/ETH/SOL/XRP hourly markets.
Budget: $100 max per trade.

Usage:
    python copy_0x8dxd.py [--live]
"""

import asyncio
import os
import sys
import time
from datetime import datetime

import httpx

# 0x8dxd wallet
TARGET_WALLET = "0x63ce342161250d705dc0b16df89036c8e5f9ba9a"

# Copy settings
MAX_COPY_AMOUNT = 200.0  # Max $200 per trade
SCALE_FACTOR = 0.10  # Copy at 10% of their size (they bet $1-3K)
POLL_INTERVAL = 10  # Check every 10 seconds (fast for hourly markets)

# Track what we've seen
seen_trades = set()


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


async def get_recent_activity(client: httpx.AsyncClient) -> list:
    """Get 0x8dxd's recent trading activity."""
    try:
        r = await client.get(
            "https://data-api.polymarket.com/activity",
            params={"user": TARGET_WALLET, "limit": 20}
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log(f"Error fetching activity: {e}")
    return []


async def get_positions(client: httpx.AsyncClient) -> list:
    """Get 0x8dxd's current positions."""
    try:
        r = await client.get(
            "https://data-api.polymarket.com/positions",
            params={"user": TARGET_WALLET}
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log(f"Error fetching positions: {e}")
    return []


def format_trade(trade: dict) -> str:
    """Format a trade for display."""
    title = trade.get("title", "Unknown")[:45]
    outcome = trade.get("outcome", "?")
    price = float(trade.get("price", 0) or 0)
    size = float(trade.get("size", 0) or 0)
    cost = price * size
    return f"{title} | {outcome} @ {price*100:.0f}% | ${cost:.2f}"


async def copy_trade(trade: dict, live: bool = False):
    """Execute a copy trade."""
    title = trade.get("title", "Unknown")
    outcome = trade.get("outcome", "?")
    price = float(trade.get("price", 0) or 0)
    size = float(trade.get("size", 0) or 0)
    condition_id = trade.get("conditionId", "")
    token_id = trade.get("asset", "")

    # Calculate our copy size - use minimal size up to MAX
    their_cost = price * size
    our_cost = min(their_cost * SCALE_FACTOR, MAX_COPY_AMOUNT)
    # Ensure minimum $5 trade
    our_cost = max(our_cost, 5.0) if our_cost > 0 else 0
    our_size = our_cost / price if price > 0 else 0

    log("=" * 60)
    log("COPY SIGNAL DETECTED!")
    log(f"Market: {title}")
    log(f"Outcome: {outcome}")
    log(f"0x8dxd: {size:.0f} shares @ ${price:.3f} = ${their_cost:.2f}")
    log(f"Our copy: {our_size:.0f} shares @ ${price:.3f} = ${our_cost:.2f}")
    log(f"Token: {token_id[:30]}...")
    log("=" * 60)

    if live:
        log("EXECUTING LIVE TRADE...")
        try:
            # Import and use executor
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from arbitrage.executor import Executor

            executor = Executor()
            if executor.client:
                from py_clob_client.order_builder.constants import BUY

                # Build and place order
                order = executor.client.create_order(
                    token_id=token_id,
                    price=price,
                    size=our_size,
                    side=BUY,
                )
                result = executor.client.post_order(order)

                log(f"ORDER PLACED: {result}")
                return True
            else:
                log("ERROR: Executor not initialized (check wallet password)")
                return False
        except Exception as e:
            log(f"TRADE ERROR: {e}")
            return False
    else:
        log("[DRY RUN] Would execute trade")

    return True


async def monitor_loop(live: bool = False):
    """Main monitoring loop."""
    global seen_trades

    log("=" * 60)
    log("0x8dxd COPY TRADER")
    log("=" * 60)
    log(f"Target: {TARGET_WALLET[:20]}...")
    log(f"Max copy: ${MAX_COPY_AMOUNT}")
    log(f"Scale: {SCALE_FACTOR*100:.0f}%")
    log(f"Mode: {'LIVE' if live else 'DRY RUN'}")
    log(f"Poll interval: {POLL_INTERVAL}s")
    log("=" * 60)
    log("")

    async with httpx.AsyncClient(timeout=30) as client:
        # Initialize with current activity
        log("Initializing - loading existing trades...")
        initial = await get_recent_activity(client)
        for trade in initial:
            tx = trade.get("transactionHash", "")
            if tx:
                seen_trades.add(tx)
        log(f"Loaded {len(seen_trades)} existing trades")
        log("")
        log("Monitoring for new trades...")
        log("")

        while True:
            try:
                # Check for new activity
                activity = await get_recent_activity(client)

                for trade in activity:
                    tx = trade.get("transactionHash", "")
                    action = trade.get("action", "")

                    # New trade?
                    if tx and tx not in seen_trades and action == "TRADE":
                        seen_trades.add(tx)

                        # Only copy BUY trades on crypto up/down markets
                        title = trade.get("title", "").lower()
                        side = trade.get("side", "")

                        if side == "BUY" and ("up or down" in title or "bitcoin" in title or "ethereum" in title):
                            await copy_trade(trade, live=live)
                        else:
                            log(f"Skipped: {format_trade(trade)}")

                # Also check positions for new entries
                positions = await get_positions(client)
                open_count = sum(1 for p in positions
                                if 0.05 < float(p.get("curPrice", 0) or 0) < 0.95)

                # Status update every minute
                if int(time.time()) % 60 < POLL_INTERVAL:
                    log(f"Status: {len(seen_trades)} trades seen | {open_count} open positions")

                await asyncio.sleep(POLL_INTERVAL)

            except KeyboardInterrupt:
                log("Shutting down...")
                break
            except Exception as e:
                log(f"Error: {e}")
                await asyncio.sleep(30)


def main():
    live = "--live" in sys.argv

    if live:
        print("\n" + "!" * 60)
        print("WARNING: LIVE MODE - REAL MONEY WILL BE USED")
        print("!" * 60)
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Aborted.")
            return

    try:
        asyncio.run(monitor_loop(live=live))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
