"""
Copy trading module - mirrors trades from successful traders.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set
import httpx

from .config import config


@dataclass
class TrackedPosition:
    """A position being tracked from target wallet."""
    condition_id: str
    token_id: str
    market_slug: str
    outcome: str  # YES or NO
    size: float
    avg_price: float
    current_value: float
    first_seen: float
    last_updated: float


@dataclass
class CopySignal:
    """Signal to copy a trade."""
    condition_id: str
    token_id: str
    market_slug: str
    outcome: str
    target_size: float
    target_price: float
    our_size: float  # Scaled to our budget
    signal_type: str  # "ENTER" or "EXIT"
    timestamp: float


class CopyTrader:
    """
    Monitors a target wallet and generates copy trading signals.
    """

    # Known whale wallets
    WHALES = {
        "kingofcoinflips": "0xe9c6312464b52aa3eff13d822b003282075995c9",  # $692K PnL, crypto-focused
        "swisstony": "0x204f72f35326db932158cba6adff0b9a1da95e14",  # $3.6M PnL, sports-focused
        "RN1": "0x2005d16a84ceefa912d4e380cd32e7ff827875ea",  # $4.2M PnL, sports-focused
        "XPredicter1": "0x6c16abad96d6989efe1b0333cb9af9158f548bfa",  # $75K PnL, sports/politics
        "0x8dxd": "0x63ce342161250d705dc0b16df89036c8e5f9ba9a",  # $885K PnL, crypto hourly up/down
        "Gametheory": "0x8b3234f9027f4e994e949df4b48b90ab79015950",  # $244K PnL, diversified
        "Account88888": "0x7f69983eb28245bba0d5083502a78744a8f66162",  # $645K PnL, $118M volume
        "k9Q2mX4L8A7ZP3R": "0xd0d6053c3c37e727402d84c14069780d360993aa",  # $422K PnL, $52M volume
        "B8a2eF07B847626688": "0x3c27b6ef2d914abbc3c96948a942e295b2d11b56",  # $83K PnL, $1M portfolio, 60 trades
        "Maskache": "0xc981e9d3b977dfc69188889f979f5cd36555a75d",  # $8.5K PnL, 1,517 trades, $3M volume
        "bobe2": "0xed107a85a4585a381e48c7f7ca4144909e7dd2e5",  # $1.36M PnL, 1,091 trades, $113M volume
        "kch123": "0x6a72f61820b26b1fe4d956e17b6dc2a1ea3033ee",  # $8.79M PnL, 1,769 trades, $222M volume - TOP WHALE
        "WickRick": "0x1c12abb42d0427e70e144ae67c92951b232f79d9",  # $21K PnL, 1 trade, $75K portfolio
        "BoshBashBish": "0x29bc82f761749e67fa00d62896bc6855097b683c",  # $147K PnL, 9,621 trades, $29M volume, 7 weeks old
        "NeverYES": "0xe55108581aec68af7686bd4cbc4f53ef93680a67",  # $107K portfolio, 17 trades
        "Harmless-Critic": "0x1461cC6e1A05e20710c416307Db62C28f1D122d8",  # $737K portfolio, $4.5M volume, 192 markets
        "distinct-baguette": "0xe00740bce98a594e26861838885ab310ec3b548c",  # $521K PnL, 30K trades, $47M vol - MARKET MAKER, DCA ladders both sides
        "DrPufferfish": "0xdb27bf2ac5d428a9c63dbc914611036855a6c56e",
        "PBot1": "0x88f46b9e5d86b4fb85be55ab0ec4004264b9d4db",  # $107K PnL, 6,098 trades, $7.6M volume
        "0x93C2": "0x93C22116E4402C9332Ee6Db578050e688934C072",  # User-added whale
        "0xE594": "0xE594336603F4fB5d3ba4125a67021ab3B4347052",  # User-added whale
    }

    DEFAULT_TARGET = WHALES["kingofcoinflips"]

    def __init__(
        self,
        target_wallet: str = None,
        scale_factor: float = 0.01,  # Copy at 1% of their size by default
        min_position_usd: float = 100.0,  # Only copy positions > $100
        state_file: str = "copy_trader_state.json",
    ):
        self.target_wallet = target_wallet or self.DEFAULT_TARGET
        self.scale_factor = scale_factor
        self.min_position_usd = min_position_usd
        self.state_file = state_file

        # Track positions
        self.known_positions: Dict[str, TrackedPosition] = {}
        self.pending_signals: List[CopySignal] = []
        self.executed_copies: Set[str] = set()  # condition_ids we've copied

        self._load_state()

    def _load_state(self):
        """Load previous state."""
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
                self.executed_copies = set(data.get("executed_copies", []))

                for pos_data in data.get("known_positions", []):
                    pos = TrackedPosition(**pos_data)
                    self.known_positions[pos.condition_id] = pos

                print(f"[CopyTrader] Loaded {len(self.known_positions)} tracked positions")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[CopyTrader] Error loading state: {e}")

    def _save_state(self):
        """Save current state."""
        try:
            data = {
                "executed_copies": list(self.executed_copies),
                "known_positions": [
                    {
                        "condition_id": p.condition_id,
                        "token_id": p.token_id,
                        "market_slug": p.market_slug,
                        "outcome": p.outcome,
                        "size": p.size,
                        "avg_price": p.avg_price,
                        "current_value": p.current_value,
                        "first_seen": p.first_seen,
                        "last_updated": p.last_updated,
                    }
                    for p in self.known_positions.values()
                ],
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[CopyTrader] Error saving state: {e}")

    async def fetch_target_positions(self) -> List[dict]:
        """Fetch current positions from target wallet."""
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                r = await client.get(
                    "https://data-api.polymarket.com/positions",
                    params={
                        "user": self.target_wallet,
                        "sizeThreshold": 0,
                    }
                )
                r.raise_for_status()
                return r.json()
            except Exception as e:
                print(f"[CopyTrader] Error fetching positions: {e}")
                return []

    async def scan_for_signals(self) -> List[CopySignal]:
        """
        Scan target wallet for new/changed positions.
        Returns list of copy signals.
        """
        positions = await self.fetch_target_positions()
        if not positions:
            return []

        signals = []
        current_ids = set()
        now = time.time()

        for pos in positions:
            condition_id = pos.get("conditionId", "")
            current_ids.add(condition_id)

            # Parse position data
            size = float(pos.get("size", 0))
            avg_price = float(pos.get("avgPrice", 0))
            current_value = float(pos.get("currentValue", 0))
            token_id = pos.get("asset", "")
            market_slug = pos.get("slug", pos.get("title", "unknown")[:50])
            outcome = pos.get("outcome", "")

            # Skip small positions
            if current_value < self.min_position_usd:
                continue

            # Check if this is a new position
            if condition_id not in self.known_positions:
                # New position detected!
                # Get title for display
                title = pos.get("title", market_slug)[:50]

                tracked = TrackedPosition(
                    condition_id=condition_id,
                    token_id=token_id,
                    market_slug=title,
                    outcome=outcome,
                    size=size,
                    avg_price=avg_price,
                    current_value=current_value,
                    first_seen=now,
                    last_updated=now,
                )
                self.known_positions[condition_id] = tracked

                # Generate copy signal if not already copied
                if condition_id not in self.executed_copies:
                    our_size = current_value * self.scale_factor
                    if our_size >= config.MIN_POSITION_SIZE:
                        signal = CopySignal(
                            condition_id=condition_id,
                            token_id=token_id,
                            market_slug=market_slug,
                            outcome=outcome,
                            target_size=size,
                            target_price=avg_price,
                            our_size=min(our_size, config.MAX_POSITION_SIZE),
                            signal_type="ENTER",
                            timestamp=now,
                        )
                        signals.append(signal)
                        print(f"[CopyTrader] NEW POSITION: {market_slug} {outcome}")
                        print(f"  Target: {size:.0f} shares @ ${avg_price:.2f} = ${current_value:.2f}")
                        print(f"  Copy size: ${signal.our_size:.2f}")
            else:
                # Update existing position
                tracked = self.known_positions[condition_id]

                # Check for significant size increase (adding to position)
                size_increase = size - tracked.size
                if size_increase > tracked.size * 0.1:  # >10% increase
                    print(f"[CopyTrader] POSITION INCREASED: {market_slug}")
                    print(f"  From {tracked.size:.0f} to {size:.0f} shares (+{size_increase:.0f})")

                tracked.size = size
                tracked.avg_price = avg_price
                tracked.current_value = current_value
                tracked.last_updated = now

        # Check for closed positions (exits)
        for condition_id in list(self.known_positions.keys()):
            if condition_id not in current_ids:
                tracked = self.known_positions[condition_id]
                print(f"[CopyTrader] POSITION CLOSED: {tracked.market_slug}")

                # Generate exit signal if we copied this
                if condition_id in self.executed_copies:
                    signal = CopySignal(
                        condition_id=condition_id,
                        token_id=tracked.token_id,
                        market_slug=tracked.market_slug,
                        outcome=tracked.outcome,
                        target_size=0,
                        target_price=0,
                        our_size=0,
                        signal_type="EXIT",
                        timestamp=now,
                    )
                    signals.append(signal)

                del self.known_positions[condition_id]

        self._save_state()
        return signals

    def mark_executed(self, condition_id: str):
        """Mark a copy trade as executed."""
        self.executed_copies.add(condition_id)
        self._save_state()

    def get_status(self) -> dict:
        """Get current copy trading status."""
        total_target_value = sum(p.current_value for p in self.known_positions.values())

        return {
            "target_wallet": self.target_wallet[:10] + "...",
            "tracked_positions": len(self.known_positions),
            "total_target_value": total_target_value,
            "executed_copies": len(self.executed_copies),
            "scale_factor": self.scale_factor,
            "min_position": self.min_position_usd,
        }

    def print_status(self):
        """Print current status."""
        status = self.get_status()
        print("\n" + "=" * 50)
        print("COPY TRADER STATUS")
        print("=" * 50)
        print(f"Target: {status['target_wallet']}")
        print(f"Tracking: {status['tracked_positions']} positions")
        print(f"Target Value: ${status['total_target_value']:,.2f}")
        print(f"Copied: {status['executed_copies']} trades")
        print(f"Scale: {status['scale_factor']*100:.1f}%")
        print("=" * 50)

        # Show top positions
        if self.known_positions:
            print("\nTop Positions Being Tracked:")
            sorted_pos = sorted(
                self.known_positions.values(),
                key=lambda p: p.current_value,
                reverse=True
            )[:5]
            for p in sorted_pos:
                print(f"  {p.outcome}: ${p.current_value:,.0f} - {p.market_slug}")
        print()


async def run_copy_monitor(
    dry_run: bool = True,
    poll_interval: int = 60,
    executor=None,
):
    """
    Run the copy trading monitor.

    Args:
        dry_run: If True, only log signals without executing
        poll_interval: Seconds between position checks
        executor: ArbitrageExecutor instance for placing orders
    """
    trader = CopyTrader()
    trader.print_status()

    print(f"\n[CopyTrader] Starting monitor (dry_run={dry_run})")
    print(f"[CopyTrader] Polling every {poll_interval}s")

    while True:
        try:
            signals = await trader.scan_for_signals()

            for signal in signals:
                if signal.signal_type == "ENTER":
                    print(f"\n{'='*50}")
                    print(f"COPY SIGNAL: BUY {signal.outcome}")
                    print(f"Market: {signal.market_slug}")
                    print(f"Target bought: {signal.target_size:.0f} @ ${signal.target_price:.2f}")
                    print(f"Our size: ${signal.our_size:.2f}")
                    print(f"{'='*50}")

                    if not dry_run and executor:
                        # Execute the copy trade
                        # This would integrate with the executor
                        pass
                    else:
                        print("[DRY RUN] Would execute copy trade")

                    trader.mark_executed(signal.condition_id)

                elif signal.signal_type == "EXIT":
                    print(f"\n{'='*50}")
                    print(f"EXIT SIGNAL: SELL {signal.outcome}")
                    print(f"Market: {signal.market_slug}")
                    print(f"{'='*50}")

                    if not dry_run and executor:
                        # Execute exit
                        pass
                    else:
                        print("[DRY RUN] Would exit position")

            # Print status periodically
            if int(time.time()) % 300 < poll_interval:  # Every ~5 min
                trader.print_status()

        except Exception as e:
            print(f"[CopyTrader] Error in monitor loop: {e}")

        await asyncio.sleep(poll_interval)


if __name__ == "__main__":
    asyncio.run(run_copy_monitor(dry_run=True, poll_interval=30))
