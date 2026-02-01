"""
Copy trading script for aenews2 whale.

aenews2: 0x44c1dfe43260c94ed4f1d00de2e1f80fb113ebc1
- 72% win rate on current positions
- $1,034,266 in open positions
- 48 winning / 19 losing positions
- Avg entry price: 0.56

FILTER: Only copies trades that resolve within 4 months
"""

import asyncio
import sys
sys.path.insert(0, '.')

from arbitrage.copy_trader import CopyTrader, CopySignal
from arbitrage.config import config
from datetime import datetime, timedelta
import httpx
import time

# aenews2 whale wallet
AENEWS2_WALLET = "0x44c1dfe43260c94ed4f1d00de2e1f80fb113ebc1"

# Resolution time limit (4 months)
MAX_RESOLUTION_DAYS = 120


async def get_market_end_date(condition_id: str) -> datetime | None:
    """Fetch market end date from Polymarket API."""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            # Try gamma-api first
            resp = await client.get(
                f"https://gamma-api.polymarket.com/markets/{condition_id}"
            )
            if resp.status_code == 200:
                data = resp.json()
                end_date_str = data.get("endDate") or data.get("end_date_iso")
                if end_date_str:
                    return datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        except Exception as e:
            print(f"[Filter] Error fetching market end date: {e}")
    return None


async def is_within_resolution_limit(condition_id: str, max_days: int = MAX_RESOLUTION_DAYS) -> tuple[bool, str]:
    """
    Check if market resolves within the allowed time limit.

    Returns:
        (is_valid, reason_string)
    """
    end_date = await get_market_end_date(condition_id)

    if end_date is None:
        # If we can't determine end date, be conservative and skip
        return False, "Could not determine end date"

    now = datetime.now(end_date.tzinfo) if end_date.tzinfo else datetime.now()
    days_until_resolution = (end_date - now).days

    if days_until_resolution <= max_days:
        return True, f"Resolves in {days_until_resolution} days"
    else:
        return False, f"Resolves in {days_until_resolution} days (exceeds {max_days} day limit)"


async def run_copy_aenews2(
    dry_run: bool = True,
    scale_factor: float = 0.05,  # 5% of their positions
    min_position: float = 500,   # Only copy positions > $500
    max_position: float = 200,   # Max $200 per copy trade
    poll_interval: int = 60,
    max_resolution_days: int = MAX_RESOLUTION_DAYS,  # 4 months
):
    """
    Run copy trading for aenews2.

    Args:
        dry_run: If True, paper trade only (no real money)
        scale_factor: Copy at this % of their position size
        min_position: Only copy positions larger than this
        max_position: Max size for our copy trades
        poll_interval: Seconds between scans
        max_resolution_days: Only copy markets resolving within this many days
    """
    print("=" * 60)
    print("COPY TRADER - aenews2")
    print("=" * 60)
    print(f"Wallet: {AENEWS2_WALLET}")
    print(f"Mode: {'PAPER TRADING' if dry_run else 'LIVE TRADING'}")
    print(f"Scale: {scale_factor*100:.0f}% of whale positions")
    print(f"Min position to copy: ${min_position}")
    print(f"Max copy size: ${max_position}")
    print(f"Max resolution: {max_resolution_days} days ({max_resolution_days/30:.1f} months)")
    print(f"Poll interval: {poll_interval}s")
    print("=" * 60)
    print()

    trader = CopyTrader(
        target_wallet=AENEWS2_WALLET,
        scale_factor=scale_factor,
        min_position_usd=min_position,
        state_file="copy_trader_aenews2_state.json",
    )

    # Override max position
    original_max = config.MAX_POSITION_SIZE
    config.MAX_POSITION_SIZE = max_position

    trader.print_status()

    print(f"\n[CopyTrader] Starting aenews2 copy trader...")
    print(f"[CopyTrader] Polling every {poll_interval}s")
    print(f"[CopyTrader] Only copying markets that resolve within {max_resolution_days} days")
    print(f"[CopyTrader] Press Ctrl+C to stop\n")

    iteration = 0
    skipped_markets = set()  # Track markets we've skipped due to resolution time

    while True:
        try:
            iteration += 1
            signals = await trader.scan_for_signals()

            for signal in signals:
                if signal.signal_type == "ENTER":
                    # Check if market resolves within our limit
                    is_valid, reason = await is_within_resolution_limit(
                        signal.condition_id,
                        max_resolution_days
                    )

                    if not is_valid:
                        if signal.condition_id not in skipped_markets:
                            print(f"\n[SKIP] {signal.market_slug}")
                            print(f"       Reason: {reason}")
                            skipped_markets.add(signal.condition_id)
                        # Mark as executed so we don't keep trying to copy it
                        trader.mark_executed(signal.condition_id)
                        continue

                    print(f"\n{'='*60}")
                    print(f"COPY SIGNAL: BUY {signal.outcome}")
                    print(f"{'='*60}")
                    print(f"Market: {signal.market_slug}")
                    print(f"Resolution: {reason}")
                    print(f"aenews2 bought: {signal.target_size:.0f} shares @ ${signal.target_price:.2f}")
                    print(f"Our copy size: ${signal.our_size:.2f}")
                    print(f"Token ID: {signal.token_id[:20]}..." if signal.token_id else "")

                    if dry_run:
                        print(f"\n[PAPER TRADE] Recording copy trade...")
                        trader.mark_executed(signal.condition_id)
                        print(f"[PAPER TRADE] Copied {signal.outcome} @ ${signal.target_price:.2f}")
                    else:
                        print(f"\n[LIVE] Would execute real trade here")
                        trader.mark_executed(signal.condition_id)

                    print(f"{'='*60}\n")

                elif signal.signal_type == "EXIT":
                    print(f"\n{'='*60}")
                    print(f"EXIT SIGNAL: CLOSE {signal.outcome}")
                    print(f"{'='*60}")
                    print(f"Market: {signal.market_slug}")
                    print(f"aenews2 closed their position")

                    if dry_run:
                        print(f"[PAPER TRADE] Would close our copy position")
                    else:
                        print(f"[LIVE] Would close position")

                    print(f"{'='*60}\n")

            # Status update every 5 minutes
            if iteration % (300 // poll_interval) == 0:
                trader.print_status()
                if skipped_markets:
                    print(f"[Filter] Skipped {len(skipped_markets)} markets due to resolution time limit")

        except KeyboardInterrupt:
            print("\n[CopyTrader] Stopping...")
            break
        except Exception as e:
            print(f"[CopyTrader] Error: {e}")

        await asyncio.sleep(poll_interval)

    # Restore config
    config.MAX_POSITION_SIZE = original_max

    print("\n[CopyTrader] Stopped. Final status:")
    trader.print_status()
    print(f"[Filter] Total skipped due to resolution limit: {len(skipped_markets)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Copy trade aenews2 whale")
    parser.add_argument("--live", action="store_true", help="Enable live trading (real money!)")
    parser.add_argument("--scale", type=float, default=0.05, help="Scale factor (default 5%%)")
    parser.add_argument("--min-pos", type=float, default=500, help="Min position to copy (default $500)")
    parser.add_argument("--max-pos", type=float, default=200, help="Max copy size (default $200)")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval seconds (default 60)")
    parser.add_argument("--max-days", type=int, default=120, help="Max days until resolution (default 120 = 4 months)")

    args = parser.parse_args()

    if args.live:
        print("\n" + "!" * 60)
        print("WARNING: LIVE TRADING MODE - REAL MONEY AT RISK!")
        print("!" * 60)
        confirm = input("Type 'YES' to confirm: ")
        if confirm != "YES":
            print("Aborted.")
            sys.exit(0)

    asyncio.run(run_copy_aenews2(
        dry_run=not args.live,
        scale_factor=args.scale,
        min_position=args.min_pos,
        max_position=args.max_pos,
        poll_interval=args.interval,
        max_resolution_days=args.max_days,
    ))
