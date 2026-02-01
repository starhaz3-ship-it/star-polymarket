"""
Enhanced Whale Tracking Module

Based on 2024-2026 research:
- Track multiple profitable whales
- Optimal copy timing (30-60 sec delay)
- Context-dependent signal interpretation
- Whale consensus detection
"""

import asyncio
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set
from datetime import datetime, timezone
from collections import defaultdict
import httpx


@dataclass
class WhaleProfile:
    """Profile of a tracked whale."""
    address: str
    name: str
    trades_count: int
    win_rate: float
    total_pnl: float
    avg_position_size: float
    sharpe_ratio: float
    last_active: str


@dataclass
class WhaleSignal:
    """A signal from whale activity."""
    whale_address: str
    whale_name: str
    market_id: str
    market_title: str
    direction: str  # "YES" or "NO"
    entry_price: float
    position_size: float
    whale_pnl: float
    signal_strength: float  # 0-1
    is_contrarian: bool  # Against retail sentiment
    whale_consensus: float  # How many whales agree
    timestamp: str


class WhaleTracker:
    """
    Track and analyze whale trading behavior.
    """

    # Known profitable whales from research
    DEFAULT_WHALES = [
        {
            "address": "0xe9c6312464b52aa3eff13d822b003282075995c9",
            "name": "kingofcoinflips",
            "reputation": 0.8,
        },
        {
            "address": "0x3b50e70e15d5f1bb1f5c5b93b0d31c26f8f4b5c6",
            "name": "whale_alpha",
            "reputation": 0.7,
        },
        {
            "address": "0xdb27bf2ac5d428a9c63dbc914611036855a6c56e",
            "name": "DrPufferfish",
            "reputation": 0.75,
        },
        {
            "address": "0x44c1dfe43260c94ed4f1d00de2e1f80fb113ebc1",
            "name": "aenews2",
            "reputation": 0.8,
        },
        {
            "address": "0x9d84ce0306f8551e02efef1680475fc0f1dc1344",
            "name": "ImJustKen",
            "reputation": 0.75,
        },
        {
            "address": "0xd830027529b0baca2a52fd8a4dee43d366a9a592",
            "name": "LDSIADAS",
            "reputation": 0.75,
        },
        {
            "address": "0xd0d6053c3c37e727402d84c14069780d360993aa",
            "name": "k9Q2mX4L8A7ZP3R",
            "reputation": 0.8,
        },
        {
            "address": "0xdc876e6873772d38716fda7f2452a78d426d7ab6",
            "name": "432614799197",
            "reputation": 0.75,
        },
        {
            "address": "0x93c22116e4402c9332ee6db578050e688934c072",
            "name": "Candid-Closure",
            "reputation": 0.75,
        },
        {
            "address": "0xcb3143ee858e14d0b3fe40ffeaea78416e646b02",
            "name": "0xCB3143ee858E14d0B3FE40fFeaEa78416e646B0",
            "reputation": 0.75,
        },
        {
            "address": "0xa58d4f278d7953cd38eeb929f7e242bfc7c0b9b8",
            "name": "AiBird",
            "reputation": 0.75,
        },
        {
            "address": "0xcb2952fd655813ad0f9ee893122c54298e1a975e",
            "name": "willi",
            "reputation": 0.75,
        },
        {
            "address": "0x79bf43cbd005a51e25bafa4abe3f51a3c8ba96a8",
            "name": "I-do-stupid-bets",
            "reputation": 0.75,
        },
        {
            "address": "0xd8f8c13644ea84d62e1ec88c5d1215e436eb0f11",
            "name": "automatedAItradingbot",
            "reputation": 0.75,
        },
        {
            "address": "0x1c12abb42d0427e70e144ae67c92951b232f79d9",
            "name": "WickRick",
            "reputation": 0.75,
        },
        {
            "address": "0x63ce342161250d705dc0b16df89036c8e5f9ba9a",
            "name": "0x8dxd",
            "reputation": 0.85,  # $885K PnL, crypto hourly up/down specialist
        },
    ]

    def __init__(self, state_file: str = "whale_tracker_state.json"):
        self.state_file = state_file
        self.whales: Dict[str, WhaleProfile] = {}
        self.whale_positions: Dict[str, List[Dict]] = {}
        self.position_history: Dict[str, List[Dict]] = defaultdict(list)
        self.copy_delay_seconds = 30  # Optimal delay from research

        self._load_state()
        self._init_default_whales()

    def _load_state(self):
        """Load previous state."""
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                for addr, profile_data in data.get("whales", {}).items():
                    self.whales[addr] = WhaleProfile(**profile_data)
                self.position_history = defaultdict(list, data.get("position_history", {}))
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[WhaleTracker] Error loading state: {e}")

    def _save_state(self):
        """Save current state."""
        try:
            data = {
                "whales": {addr: asdict(profile) for addr, profile in self.whales.items()},
                "position_history": dict(self.position_history),
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[WhaleTracker] Error saving state: {e}")

    def _init_default_whales(self):
        """Initialize default whale profiles."""
        for whale in self.DEFAULT_WHALES:
            addr = whale["address"]
            if addr not in self.whales:
                self.whales[addr] = WhaleProfile(
                    address=addr,
                    name=whale["name"],
                    trades_count=0,
                    win_rate=0.5,
                    total_pnl=0.0,
                    avg_position_size=10000.0,
                    sharpe_ratio=whale.get("reputation", 0.5),
                    last_active="",
                )

    async def fetch_positions(self, whale_address: str) -> List[Dict]:
        """Fetch current positions for a whale."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(
                    "https://data-api.polymarket.com/positions",
                    params={"user": whale_address, "sizeThreshold": 0}
                )
                positions = r.json()
                self.whale_positions[whale_address] = positions
                return positions
        except Exception as e:
            print(f"[WhaleTracker] Error fetching positions for {whale_address[:10]}...: {e}")
            return []

    async def detect_new_positions(self, whale_address: str) -> List[Dict]:
        """Detect new positions since last check."""
        old_positions = self.position_history.get(whale_address, [])
        old_ids = {p.get("conditionId") for p in old_positions}

        new_positions = await self.fetch_positions(whale_address)

        # Find new positions
        new_entries = []
        for pos in new_positions:
            cond_id = pos.get("conditionId")
            if cond_id not in old_ids:
                new_entries.append(pos)

        # Update history
        self.position_history[whale_address] = new_positions
        self._save_state()

        return new_entries

    async def scan_all_whales(self) -> List[WhaleSignal]:
        """Scan all tracked whales for signals."""
        signals = []

        # Fetch all whale positions concurrently
        tasks = []
        for addr in self.whales.keys():
            tasks.append(self.fetch_positions(addr))

        all_positions = await asyncio.gather(*tasks)

        # Analyze positions across all whales
        market_whale_positions: Dict[str, List[Dict]] = defaultdict(list)

        for addr, positions in zip(self.whales.keys(), all_positions):
            whale = self.whales[addr]

            for pos in positions:
                market_id = pos.get("conditionId", "")
                current_price = float(pos.get("curPrice", 0) or 0)
                pnl = float(pos.get("cashPnl", 0) or 0)
                value = float(pos.get("currentValue", 0) or 0)

                # Only consider active positions with meaningful size
                if value < 500 or current_price <= 0:
                    continue

                market_whale_positions[market_id].append({
                    "whale_address": addr,
                    "whale_name": whale.name,
                    "outcome": pos.get("outcome"),
                    "entry_price": float(pos.get("avgPrice", 0) or 0),
                    "current_price": current_price,
                    "position_size": value,
                    "pnl": pnl,
                    "title": pos.get("title", "")[:60],
                    "sharpe": whale.sharpe_ratio,
                })

        # Generate signals for markets with whale activity
        for market_id, whale_pos_list in market_whale_positions.items():
            if not whale_pos_list:
                continue

            # Check for whale consensus
            yes_whales = [p for p in whale_pos_list if p["outcome"] == "Yes"]
            no_whales = [p for p in whale_pos_list if p["outcome"] == "No"]

            yes_value = sum(p["position_size"] for p in yes_whales)
            no_value = sum(p["position_size"] for p in no_whales)
            total_value = yes_value + no_value

            if total_value > 0:
                consensus = (yes_value - no_value) / total_value  # -1 to 1
            else:
                consensus = 0

            # Generate signal for dominant direction
            if abs(consensus) > 0.3:  # Minimum consensus threshold
                if consensus > 0:
                    dominant_positions = yes_whales
                    direction = "YES"
                else:
                    dominant_positions = no_whales
                    direction = "NO"

                # Weight by sharpe ratio
                weighted_sum = sum(p["position_size"] * p["sharpe"] for p in dominant_positions)
                total_weight = sum(p["position_size"] for p in dominant_positions)

                if total_weight > 0:
                    avg_sharpe = weighted_sum / total_weight
                else:
                    avg_sharpe = 0.5

                # Signal strength based on consensus and whale quality
                signal_strength = abs(consensus) * avg_sharpe

                # Use best whale's entry
                best_whale = max(dominant_positions, key=lambda p: p["sharpe"])

                signals.append(WhaleSignal(
                    whale_address=best_whale["whale_address"],
                    whale_name=best_whale["whale_name"],
                    market_id=market_id,
                    market_title=best_whale["title"],
                    direction=direction,
                    entry_price=best_whale["current_price"],
                    position_size=total_value,
                    whale_pnl=sum(p["pnl"] for p in dominant_positions),
                    signal_strength=signal_strength,
                    is_contrarian=False,  # Would need retail data
                    whale_consensus=abs(consensus),
                    timestamp=datetime.now().isoformat(),
                ))

        # Sort by signal strength
        signals.sort(key=lambda x: x.signal_strength, reverse=True)
        return signals

    async def get_copy_signals(
        self,
        min_signal_strength: float = 0.4,
        min_whale_pnl: float = 0,
        max_signals: int = 10,
    ) -> List[WhaleSignal]:
        """
        Get filtered copy trading signals.

        Args:
            min_signal_strength: Minimum signal strength (0-1)
            min_whale_pnl: Minimum whale P&L to copy
            max_signals: Maximum signals to return
        """
        all_signals = await self.scan_all_whales()

        filtered = [
            s for s in all_signals
            if s.signal_strength >= min_signal_strength
            and s.whale_pnl >= min_whale_pnl
        ]

        return filtered[:max_signals]

    def calculate_position_size(
        self,
        signal: WhaleSignal,
        bankroll: float,
        max_position: float = 0.10,
    ) -> float:
        """
        Calculate recommended position size for a copy trade.

        Based on:
        - Signal strength
        - Whale consensus
        - Kelly sizing principles
        """
        base_fraction = 0.02  # Start with 2%

        # Scale by signal strength
        size_fraction = base_fraction * signal.signal_strength

        # Boost for strong consensus
        if signal.whale_consensus > 0.7:
            size_fraction *= 1.3

        # Cap at maximum
        size_fraction = min(size_fraction, max_position)

        return size_fraction * bankroll

    def update_whale_stats(self, whale_address: str, won: bool, pnl: float):
        """Update whale performance statistics."""
        if whale_address not in self.whales:
            return

        whale = self.whales[whale_address]

        # Update win rate with exponential moving average
        alpha = 0.1
        new_win = 1.0 if won else 0.0
        whale.win_rate = alpha * new_win + (1 - alpha) * whale.win_rate

        whale.total_pnl += pnl
        whale.trades_count += 1
        whale.last_active = datetime.now().isoformat()

        # Update sharpe estimate (simplified)
        # Would need return history for proper calculation
        if whale.trades_count > 10:
            whale.sharpe_ratio = whale.win_rate * 2 - 1  # Rough proxy

        self._save_state()

    def print_status(self):
        """Print whale tracking status."""
        print("\n" + "=" * 60)
        print("WHALE TRACKER STATUS")
        print("=" * 60)

        for addr, whale in self.whales.items():
            print(f"\n{whale.name} ({addr[:10]}...)")
            print(f"  Trades: {whale.trades_count}")
            print(f"  Win Rate: {whale.win_rate:.0%}")
            print(f"  Total P&L: ${whale.total_pnl:+,.0f}")
            print(f"  Quality Score: {whale.sharpe_ratio:.2f}")

            # Show current positions
            positions = self.whale_positions.get(addr, [])
            if positions:
                print(f"  Active Positions: {len(positions)}")
                for pos in positions[:3]:
                    pnl = float(pos.get("cashPnl", 0) or 0)
                    outcome = pos.get("outcome", "")
                    title = pos.get("title", "")[:30]
                    print(f"    - {outcome} ${pnl:+,.0f} | {title}")

        print("=" * 60)


# Global tracker
whale_tracker = WhaleTracker()


async def get_whale_signals() -> List[WhaleSignal]:
    """Convenience function to get whale signals."""
    return await whale_tracker.get_copy_signals()
