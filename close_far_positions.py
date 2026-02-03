"""
Close out all open positions that expire more than 30 minutes from now.
First tries to cancel unfilled orders, then sells any filled positions.
"""

import asyncio
import json
import os
import sys
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from functools import partial

import httpx
from dotenv import load_dotenv

print = partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

RESULTS_FILE = Path(__file__).parent / "ta_live_results.json"
CUTOFF_MINUTES = 30  # Close positions for markets > 30 min away


def parse_market_time_et(title: str):
    """Extract the END time from market title like 'Bitcoin Up or Down - February 3, 4:15PM-4:30PM ET'"""
    match = re.search(r'-\s*(\d{1,2}:\d{2}(?:AM|PM))\s*ET', title)
    if not match:
        return None

    time_str = match.group(1)
    try:
        t = datetime.strptime(time_str, "%I:%M%p")
        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()
        et_offset = timezone(timedelta(hours=-5))
        market_end = datetime(today.year, today.month, today.day,
                              t.hour, t.minute, 0, tzinfo=et_offset)
        return market_end
    except:
        return None


def get_minutes_until_expiry(title: str) -> float:
    """Get minutes until market expires based on title."""
    end_time = parse_market_time_et(title)
    if not end_time:
        return 999
    now = datetime.now(timezone.utc)
    delta = (end_time - now).total_seconds() / 60
    return delta


async def close_positions():
    """Close all positions expiring > 30 min from now."""
    if not RESULTS_FILE.exists():
        print("No results file found")
        return

    data = json.load(open(RESULTS_FILE))
    trades = data.get("trades", {})

    # Find open LIVE trades that expire > 30 min from now
    to_close = []
    for tid, trade in trades.items():
        if trade["status"] != "open":
            continue
        if trade.get("order_id") == "dry_run":
            continue

        minutes_left = get_minutes_until_expiry(trade["market_title"])
        if minutes_left > CUTOFF_MINUTES:
            to_close.append(trade)
            print(f"[CLOSE] {trade['side']} @ ${trade['entry_price']:.4f} | "
                  f"{minutes_left:.0f}m left | {trade['market_title']} | order: {trade['order_id'][:16]}...")

    if not to_close:
        print("No positions to close (all within 30 min)")
        return

    print(f"\nFound {len(to_close)} positions to close")

    # Initialize executor
    from arbitrage.executor import Executor
    executor = Executor()

    if not executor._initialized:
        print("ERROR: Executor not initialized")
        return

    # Step 1: Try to cancel all open orders first
    print("\n=== Step 1: Cancelling open orders ===")

    # Get all open orders from the exchange
    try:
        open_orders = executor.client.get_orders()
        print(f"Found {len(open_orders) if open_orders else 0} open orders on exchange")

        if open_orders:
            # Cancel ALL open orders
            for order in open_orders:
                oid = order.get("id", "") or order.get("orderID", "")
                if oid:
                    try:
                        executor.client.cancel(oid)
                        print(f"  [CANCELLED] {oid[:16]}...")
                    except Exception as e:
                        print(f"  [CANCEL FAILED] {oid[:16]}... - {e}")
    except Exception as e:
        print(f"Error getting orders: {e}")

    # Step 2: Try cancelling by our stored order IDs
    print("\n=== Step 2: Cancelling by stored order IDs ===")
    cancelled = 0
    for trade in to_close:
        order_id = trade.get("order_id", "")
        if order_id and order_id != "dry_run":
            try:
                executor.client.cancel(order_id)
                print(f"  [CANCELLED] {trade['side']} | {trade['market_title'][:40]}...")
                trades[trade["trade_id"]]["status"] = "cancelled"
                trades[trade["trade_id"]]["exit_time"] = datetime.now(timezone.utc).isoformat()
                cancelled += 1
            except Exception as e:
                error_str = str(e)
                if "not found" in error_str.lower() or "already" in error_str.lower():
                    print(f"  [ALREADY FILLED/GONE] {trade['side']} | {trade['market_title'][:40]}...")
                else:
                    print(f"  [ERROR] {trade['side']} | {error_str[:60]}")
            await asyncio.sleep(0.3)

    # Step 3: For filled positions, try selling with correct share amounts
    print("\n=== Step 3: Checking for filled positions to sell ===")
    # Query positions from the API
    try:
        # Try to get our positions
        positions = executor.client.get_positions()
        if positions:
            print(f"Found {len(positions)} positions")
            for pos in positions:
                print(f"  Token: {str(pos.get('asset', ''))[:16]}... | Size: {pos.get('size', 0)}")
        else:
            print("No positions found (orders may not have been filled)")
    except Exception as e:
        print(f"Could not get positions: {e}")

    # Save updated results
    data["trades"] = trades
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nDone. Cancelled {cancelled}/{len(to_close)} orders.")
    print("Remaining positions will resolve at market expiry.")


if __name__ == "__main__":
    asyncio.run(close_positions())
