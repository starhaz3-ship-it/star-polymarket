#!/usr/bin/env python3
"""
Run the Correlation Lag Arbitrage Strategy (Paper Trading)

Based on research from @arndxt_xo and @the_smart_ape:
- Markets correlated at 0.8-0.99 have exploitable lag
- One market moves first, the other follows minutes later
- Trade the lagging market before it catches up

Usage:
    python run_correlation_lag.py [-y]

Options:
    -y    Auto-confirm start
"""

import sys
import asyncio

# Add project to path
sys.path.insert(0, "C:/Users/Star/.local/bin/star-polymarket")

from arbitrage.correlation_lag import run_correlation_lag_strategy


def main():
    # Check for auto-confirm flag
    auto_confirm = "-y" in sys.argv

    print("\n" + "="*60)
    print("CORRELATION LAG ARBITRAGE - PAPER TRADING")
    print("="*60)
    print("\nStrategy: Trade lagging correlated markets")
    print("Source: @arndxt_xo & @the_smart_ape research")
    print("\nParameters:")
    print("  - Correlation threshold: 0.80 - 0.99")
    print("  - Min move to trigger: 2%")
    print("  - Max lag window: 5 minutes")
    print("  - Position size: $25 paper")
    print("  - Starting balance: $1,000 paper")
    print("="*60)

    if not auto_confirm:
        confirm = input("\nStart paper trading? [y/N]: ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return

    print("\nStarting strategy...")
    asyncio.run(run_correlation_lag_strategy(
        dry_run=True,  # Paper trading
        poll_interval=15,  # Check every 15 seconds
        report_interval=3600,  # Hourly reports
    ))


if __name__ == "__main__":
    main()
