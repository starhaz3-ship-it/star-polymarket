"""
Paper Trade All Whales - Track and save their trades/PnL for review.

Monitors all tracked whales concurrently and saves:
- All positions detected
- Entry/exit signals
- Simulated PnL if we had copied
- Performance metrics per whale

AUTO-SYNCS with whale_tracker.py - any new whales added there will be tracked.
"""

import asyncio
import json
import sys
import time
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import httpx

sys.path.insert(0, '.')


def load_whales_from_tracker() -> Dict[str, str]:
    """Load whale list from whale_tracker.py to stay in sync."""
    whales = {}

    try:
        # Import whale_tracker module
        spec = importlib.util.spec_from_file_location(
            "whale_tracker",
            Path(__file__).parent / "arbitrage" / "whale_tracker.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get DEFAULT_WHALES from WhaleTracker class
        for whale in module.WhaleTracker.DEFAULT_WHALES:
            whales[whale["name"]] = whale["address"]

        print(f"[Sync] Loaded {len(whales)} whales from whale_tracker.py")

    except Exception as e:
        print(f"[Sync] Error loading from whale_tracker.py: {e}")
        print("[Sync] Using fallback whale list")

        # Fallback list
        whales = {
            "kingofcoinflips": "0xe9c6312464b52aa3eff13d822b003282075995c9",
            "DrPufferfish": "0xdb27bf2ac5d428a9c63dbc914611036855a6c56e",
            "aenews2": "0x44c1dfe43260c94ed4f1d00de2e1f80fb113ebc1",
            "ImJustKen": "0x9d84ce0306f8551e02efef1680475fc0f1dc1344",
            "LDSIADAS": "0xd830027529b0baca2a52fd8a4dee43d366a9a592",
            "k9Q2mX4L8A7ZP3R": "0xd0d6053c3c37e727402d84c14069780d360993aa",
            "432614799197": "0xdc876e6873772d38716fda7f2452a78d426d7ab6",
            "Candid-Closure": "0x93c22116e4402c9332ee6db578050e688934c072",
            "0xCB3143ee": "0xcb3143ee858e14d0b3fe40ffeaea78416e646b02",
            "AiBird": "0xa58d4f278d7953cd38eeb929f7e242bfc7c0b9b8",
            "willi": "0xcb2952fd655813ad0f9ee893122c54298e1a975e",
            "I-do-stupid-bets": "0x79bf43cbd005a51e25bafa4abe3f51a3c8ba96a8",
            "automatedAItradingbot": "0xd8f8c13644ea84d62e1ec88c5d1215e436eb0f11",
            "WickRick": "0x1c12abb42d0427e70e144ae67c92951b232f79d9",
            "0x8dxd": "0x63ce342161250d705dc0b16df89036c8e5f9ba9a",
            "distinct-baguette": "0xe00740bce98a594e26861838885ab310ec3b548c",
            "NeverYES": "0xe55108581aec68af7686bd4cbc4f53ef93680a67",
        }

    # Also add whales from copy_trader.py
    try:
        spec = importlib.util.spec_from_file_location(
            "copy_trader",
            Path(__file__).parent / "arbitrage" / "copy_trader.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for name, address in module.CopyTrader.WHALES.items():
            if name not in whales:
                whales[name] = address
                print(f"[Sync] Added {name} from copy_trader.py")

    except Exception as e:
        pass  # copy_trader may not have additional whales

    return whales


# Paper trading settings
COPY_SCALE = 0.05  # 5% of whale position
MAX_COPY_SIZE = 200  # Max $200 per trade
MIN_POSITION_USD = 100  # Only track positions > $100

# Output file
OUTPUT_FILE = "whale_paper_trades.json"


@dataclass
class TrackedPosition:
    """A position being tracked."""
    whale_name: str
    whale_address: str
    condition_id: str
    token_id: str
    market_title: str
    outcome: str
    whale_size: float
    whale_avg_price: float
    whale_value: float
    our_copy_size: float
    our_entry_price: float
    first_seen: str
    last_updated: str
    current_price: float
    our_pnl: float
    our_pnl_pct: float
    status: str  # "open", "closed", "resolved"


@dataclass
class WhaleStats:
    """Stats for a single whale."""
    name: str
    address: str
    positions_tracked: int
    total_value: float
    winning_positions: int
    losing_positions: int
    our_total_pnl: float
    our_total_invested: float
    last_scan: str


class AllWhalesPaperTrader:
    """Paper trade all whales and save results."""

    def __init__(self, output_file: str = OUTPUT_FILE):
        self.output_file = Path(output_file)
        self.positions: Dict[str, TrackedPosition] = {}  # key: condition_id + whale_address
        self.whale_stats: Dict[str, WhaleStats] = {}
        self.trade_history: List[dict] = []
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.whales: Dict[str, str] = {}

        self._load_state()

    def _load_state(self):
        """Load previous state if exists."""
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r') as f:
                    data = json.load(f)

                    for key, pos_data in data.get("positions", {}).items():
                        self.positions[key] = TrackedPosition(**pos_data)

                    for name, stats_data in data.get("whale_stats", {}).items():
                        self.whale_stats[name] = WhaleStats(**stats_data)

                    self.trade_history = data.get("trade_history", [])
                    self.start_time = data.get("start_time", self.start_time)

                print(f"[PaperTrader] Loaded {len(self.positions)} positions from previous session")
            except Exception as e:
                print(f"[PaperTrader] Error loading state: {e}")

    def _save_state(self):
        """Save current state to file."""
        data = {
            "start_time": self.start_time,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "whales_tracked": list(self.whales.keys()),
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "whale_stats": {k: asdict(v) for k, v in self.whale_stats.items()},
            "trade_history": self.trade_history,
            "summary": self._get_summary(),
        }

        with open(self.output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_summary(self) -> dict:
        """Get overall summary stats."""
        total_pnl = sum(p.our_pnl for p in self.positions.values())
        total_invested = sum(p.our_copy_size for p in self.positions.values() if p.status == "open")
        winning = sum(1 for p in self.positions.values() if p.our_pnl > 0)
        losing = sum(1 for p in self.positions.values() if p.our_pnl < 0)

        return {
            "total_positions": len(self.positions),
            "open_positions": sum(1 for p in self.positions.values() if p.status == "open"),
            "total_pnl": total_pnl,
            "total_invested": total_invested,
            "winning_positions": winning,
            "losing_positions": losing,
            "win_rate": winning / (winning + losing) * 100 if (winning + losing) > 0 else 0,
            "whales_tracked": len(self.whales),
        }

    def refresh_whale_list(self):
        """Refresh whale list from tracker files."""
        old_count = len(self.whales)
        self.whales = load_whales_from_tracker()
        new_count = len(self.whales)

        if new_count > old_count:
            print(f"[Sync] Whale list updated: {old_count} -> {new_count} whales")

    async def fetch_whale_positions(self, name: str, address: str) -> List[dict]:
        """Fetch positions for a single whale."""
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.get(
                    "https://data-api.polymarket.com/positions",
                    params={"user": address, "sizeThreshold": 0}
                )
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                pass  # Silently fail for individual whales
        return []

    async def scan_whale(self, name: str, address: str) -> List[TrackedPosition]:
        """Scan a single whale for positions."""
        positions = await self.fetch_whale_positions(name, address)
        now = datetime.now(timezone.utc).isoformat()

        new_positions = []
        current_ids = set()

        whale_value = 0
        whale_winning = 0
        whale_losing = 0
        our_pnl = 0
        our_invested = 0

        for pos in positions:
            condition_id = pos.get("conditionId", "")
            current_value = float(pos.get("currentValue", 0) or 0)

            if current_value < MIN_POSITION_USD:
                continue

            key = f"{condition_id}_{address}"
            current_ids.add(key)

            whale_size = float(pos.get("size", 0) or 0)
            whale_avg_price = float(pos.get("avgPrice", 0) or 0)
            current_price = float(pos.get("curPrice", whale_avg_price) or whale_avg_price)
            pct_pnl = float(pos.get("percentPnl", 0) or 0)

            whale_value += current_value
            if pct_pnl > 0:
                whale_winning += 1
            elif pct_pnl < 0:
                whale_losing += 1

            if key not in self.positions:
                # New position
                our_size = min(current_value * COPY_SCALE, MAX_COPY_SIZE)

                tracked = TrackedPosition(
                    whale_name=name,
                    whale_address=address,
                    condition_id=condition_id,
                    token_id=pos.get("asset", ""),
                    market_title=pos.get("title", "")[:80],
                    outcome=pos.get("outcome", ""),
                    whale_size=whale_size,
                    whale_avg_price=whale_avg_price,
                    whale_value=current_value,
                    our_copy_size=our_size,
                    our_entry_price=whale_avg_price,
                    first_seen=now,
                    last_updated=now,
                    current_price=current_price,
                    our_pnl=0,
                    our_pnl_pct=0,
                    status="open",
                )

                self.positions[key] = tracked
                new_positions.append(tracked)

                # Record trade
                self.trade_history.append({
                    "timestamp": now,
                    "whale": name,
                    "action": "ENTER",
                    "market": tracked.market_title,
                    "outcome": tracked.outcome,
                    "entry_price": whale_avg_price,
                    "our_size": our_size,
                })

                print(f"[{name}] NEW: {tracked.outcome} {tracked.market_title[:50]}")
                print(f"         Entry: ${whale_avg_price:.2f} | Our size: ${our_size:.2f}")
            else:
                # Update existing position
                tracked = self.positions[key]
                tracked.whale_size = whale_size
                tracked.whale_value = current_value
                tracked.current_price = current_price
                tracked.last_updated = now

                # Calculate our PnL
                if tracked.our_entry_price > 0:
                    price_change = current_price - tracked.our_entry_price
                    # Shares we would have = our_size / entry_price
                    our_shares = tracked.our_copy_size / tracked.our_entry_price
                    tracked.our_pnl = our_shares * price_change
                    tracked.our_pnl_pct = (price_change / tracked.our_entry_price) * 100

            our_pnl += self.positions[key].our_pnl
            our_invested += self.positions[key].our_copy_size

        # Check for closed positions
        for key in list(self.positions.keys()):
            if key.endswith(f"_{address}") and key not in current_ids:
                tracked = self.positions[key]
                if tracked.status == "open":
                    tracked.status = "closed"
                    tracked.last_updated = now

                    self.trade_history.append({
                        "timestamp": now,
                        "whale": name,
                        "action": "EXIT",
                        "market": tracked.market_title,
                        "outcome": tracked.outcome,
                        "entry_price": tracked.our_entry_price,
                        "exit_price": tracked.current_price,
                        "our_pnl": tracked.our_pnl,
                    })

                    print(f"[{name}] EXIT: {tracked.market_title[:50]}")
                    print(f"         PnL: ${tracked.our_pnl:.2f} ({tracked.our_pnl_pct:.1f}%)")

        # Update whale stats
        self.whale_stats[name] = WhaleStats(
            name=name,
            address=address,
            positions_tracked=sum(1 for k in self.positions if k.endswith(f"_{address}")),
            total_value=whale_value,
            winning_positions=whale_winning,
            losing_positions=whale_losing,
            our_total_pnl=our_pnl,
            our_total_invested=our_invested,
            last_scan=now,
        )

        return new_positions

    async def scan_all_whales(self):
        """Scan all whales concurrently."""
        tasks = [
            self.scan_whale(name, address)
            for name, address in self.whales.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_new = 0
        for result in results:
            if isinstance(result, list):
                total_new += len(result)

        return total_new

    def print_summary(self):
        """Print current summary."""
        summary = self._get_summary()

        print()
        print("=" * 70)
        print("WHALE PAPER TRADING SUMMARY")
        print("=" * 70)
        print(f"Running since: {self.start_time}")
        print(f"Whales tracked: {summary['whales_tracked']}")
        print(f"Total positions: {summary['total_positions']} ({summary['open_positions']} open)")
        print(f"Win/Loss: {summary['winning_positions']}/{summary['losing_positions']} ({summary['win_rate']:.0f}%)")
        print(f"Total PnL: ${summary['total_pnl']:.2f}")
        print(f"Invested: ${summary['total_invested']:.2f}")
        print("=" * 70)

        # Top whales by PnL
        print("\nTop Whales by PnL:")
        sorted_whales = sorted(
            self.whale_stats.values(),
            key=lambda w: w.our_total_pnl,
            reverse=True
        )
        for i, w in enumerate(sorted_whales[:10], 1):
            wr = w.winning_positions / (w.winning_positions + w.losing_positions) * 100 if (w.winning_positions + w.losing_positions) > 0 else 0
            print(f"  {i}. {w.name:<20} PnL: ${w.our_total_pnl:>8.2f} | Win Rate: {wr:.0f}% | Positions: {w.positions_tracked}")

        print()

    async def run(self, poll_interval: int = 60, refresh_whales_every: int = 10):
        """
        Run the paper trading monitor.

        Args:
            poll_interval: Seconds between scans
            refresh_whales_every: Refresh whale list every N iterations
        """
        # Initial whale list load
        self.refresh_whale_list()

        print("=" * 70)
        print("WHALE PAPER TRADER - ALL WHALES (Auto-Sync)")
        print("=" * 70)
        print(f"Tracking {len(self.whales)} whales")
        print(f"Copy scale: {COPY_SCALE*100:.0f}%")
        print(f"Max copy size: ${MAX_COPY_SIZE}")
        print(f"Poll interval: {poll_interval}s")
        print(f"Whale list refresh: every {refresh_whales_every} scans")
        print(f"Output file: {self.output_file}")
        print("=" * 70)
        print()

        iteration = 0
        while True:
            try:
                iteration += 1

                # Refresh whale list periodically
                if iteration % refresh_whales_every == 0:
                    self.refresh_whale_list()

                print(f"\n[Scan {iteration}] Scanning {len(self.whales)} whales...")

                new_positions = await self.scan_all_whales()

                if new_positions > 0:
                    print(f"[Scan {iteration}] Found {new_positions} new positions")

                # Save state
                self._save_state()

                # Print summary every 5 minutes
                if iteration % (300 // poll_interval) == 0:
                    self.print_summary()

            except KeyboardInterrupt:
                print("\n[PaperTrader] Stopping...")
                break
            except Exception as e:
                print(f"[PaperTrader] Error: {e}")

            await asyncio.sleep(poll_interval)

        # Final save and summary
        self._save_state()
        self.print_summary()
        print(f"\nResults saved to: {self.output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paper trade all whales")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval (default 60s)")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output file")
    parser.add_argument("--refresh", type=int, default=10, help="Refresh whale list every N scans")

    args = parser.parse_args()

    trader = AllWhalesPaperTrader(output_file=args.output)
    asyncio.run(trader.run(poll_interval=args.interval, refresh_whales_every=args.refresh))
