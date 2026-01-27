"""
Forward Testing System with Optimized Parameters

Based on backtest findings:
- Entry: 50-85% probability (sweet spot)
- Side: YES only (43% win vs 6% for NO)
- Momentum: Look for positive price movement
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import httpx

from .config import config


@dataclass
class ForwardTestTrade:
    """A forward test trade."""
    id: str
    timestamp: str
    market: str
    condition_id: str
    token_id: str
    side: str  # Always YES per backtest
    entry_price: float
    target_exit: float  # 1.0 if win
    shares: float
    cost: float
    status: str  # OPEN, WON, LOST, EXPIRED
    strategy: str
    confidence: float
    entry_reason: str
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
    closed_at: Optional[str] = None


class ForwardTester:
    """
    Forward tests optimized strategies in real-time.
    """

    # Optimized parameters from backtest
    MIN_ENTRY_PROB = 0.50
    MAX_ENTRY_PROB = 0.85
    PREFERRED_SIDE = "Yes"  # YES only - 43% win vs 6% for NO
    MIN_MOMENTUM = 0.0  # Positive momentum preferred

    def __init__(self, state_file: str = "forward_test_state.json"):
        self.state_file = state_file
        self.trades: Dict[str, ForwardTestTrade] = {}
        self.total_invested = 0.0
        self.realized_pnl = 0.0
        self._load_state()

    def _load_state(self):
        """Load previous state."""
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
                for trade_data in data.get("trades", []):
                    trade = ForwardTestTrade(**trade_data)
                    self.trades[trade.id] = trade
                self.realized_pnl = data.get("realized_pnl", 0.0)
                print(f"[ForwardTest] Loaded {len(self.trades)} trades, P&L: ${self.realized_pnl:+.2f}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[ForwardTest] Error loading state: {e}")

    def _save_state(self):
        """Save current state."""
        try:
            data = {
                "trades": [asdict(t) for t in self.trades.values()],
                "realized_pnl": self.realized_pnl,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ForwardTest] Error saving state: {e}")

    async def scan_for_opportunities(self) -> List[Dict]:
        """
        Scan markets for opportunities matching optimized criteria.
        """
        opportunities = []

        async with httpx.AsyncClient(timeout=30) as client:
            # Get whale positions (copy trading with filter)
            r = await client.get(
                "https://data-api.polymarket.com/positions",
                params={
                    "user": "0xe9c6312464b52aa3eff13d822b003282075995c9",
                    "sizeThreshold": 0
                }
            )
            whale_positions = r.json()

            for pos in whale_positions:
                outcome = pos.get("outcome", "")
                entry_price = float(pos.get("avgPrice", 0) or 0)
                current_price = float(pos.get("curPrice", 0) or 0)
                pnl = float(pos.get("cashPnl", 0) or 0)
                value = float(pos.get("currentValue", 0) or 0)
                condition_id = pos.get("conditionId", "")

                # APPLY OPTIMIZED FILTERS
                # 1. YES only (backtest: 43% win vs 6% for NO)
                if outcome != self.PREFERRED_SIDE:
                    continue

                # 2. Entry price 50-85% (backtest sweet spot: 86% win rate)
                if not (self.MIN_ENTRY_PROB <= current_price <= self.MAX_ENTRY_PROB):
                    continue

                # 3. Positive momentum (backtest: winners have +21% momentum)
                if entry_price > 0:
                    momentum = (current_price - entry_price) / entry_price
                    if momentum < self.MIN_MOMENTUM:
                        continue
                else:
                    momentum = 0

                # 4. Position is currently profitable
                if pnl <= 0:
                    continue

                # 5. Minimum size to ensure liquidity
                if value < 500:
                    continue

                # 6. Not already in our portfolio
                if condition_id in [t.condition_id for t in self.trades.values() if t.status == "OPEN"]:
                    continue

                # Calculate expected return
                expected_return = ((1.0 / current_price) - 1) * 100 if current_price > 0 else 0

                # Confidence score based on matching optimal parameters
                confidence = 0.5  # Base
                confidence += 0.2 if 0.60 <= current_price <= 0.80 else 0.1  # Sweet spot
                confidence += 0.1 if momentum > 0.10 else 0  # Strong momentum
                confidence += 0.1 if pnl > 1000 else 0.05  # Whale is winning big
                confidence = min(confidence, 0.95)

                opportunities.append({
                    "market": pos.get("title", "")[:60],
                    "slug": pos.get("slug", ""),
                    "condition_id": condition_id,
                    "token_id": pos.get("asset", ""),
                    "outcome": outcome,
                    "current_price": current_price,
                    "whale_entry": entry_price,
                    "whale_pnl": pnl,
                    "whale_value": value,
                    "momentum": momentum,
                    "expected_return": expected_return,
                    "confidence": confidence,
                    "end_date": pos.get("endDate", ""),
                })

            # Sort by confidence and expected return
            opportunities.sort(key=lambda x: (x["confidence"], x["expected_return"]), reverse=True)

        return opportunities

    def open_trade(self, opp: Dict, size_usd: float = 10.0) -> ForwardTestTrade:
        """Open a new forward test trade."""
        shares = size_usd / opp["current_price"] if opp["current_price"] > 0 else 0

        trade = ForwardTestTrade(
            id=f"FT_{opp['condition_id'][:8]}_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            market=opp["market"],
            condition_id=opp["condition_id"],
            token_id=opp["token_id"],
            side=opp["outcome"],
            entry_price=opp["current_price"],
            target_exit=1.0,
            shares=shares,
            cost=size_usd,
            status="OPEN",
            strategy="OPTIMIZED_COPY",
            confidence=opp["confidence"],
            entry_reason=f"Whale PnL: ${opp['whale_pnl']:+.0f}, Momentum: {opp['momentum']:+.0%}",
            current_price=opp["current_price"],
            unrealized_pnl=0.0,
        )

        self.trades[trade.id] = trade
        self.total_invested += size_usd
        self._save_state()

        print(f"\n[ForwardTest] OPENED: {trade.side} @ ${trade.entry_price:.3f}")
        print(f"  Market: {trade.market}")
        print(f"  Size: ${size_usd:.2f} ({shares:.1f} shares)")
        print(f"  Confidence: {trade.confidence:.0%}")
        print(f"  Reason: {trade.entry_reason}")
        print(f"  Expected Return: {((1/trade.entry_price)-1)*100:.1f}% if wins")

        return trade

    async def update_positions(self):
        """Update all open positions with current prices."""
        async with httpx.AsyncClient(timeout=30) as client:
            for trade_id, trade in list(self.trades.items()):
                if trade.status != "OPEN":
                    continue

                try:
                    # Check if market has settled
                    r = await client.get(
                        f"https://clob.polymarket.com/price",
                        params={"token_id": trade.token_id}
                    )
                    data = r.json()
                    current_price = float(data.get("price", trade.entry_price))

                    trade.current_price = current_price
                    trade.unrealized_pnl = (current_price - trade.entry_price) * trade.shares

                    # Check for settlement (price = 0 or 1)
                    if current_price <= 0.001:
                        # Lost
                        trade.status = "LOST"
                        trade.realized_pnl = -trade.cost
                        trade.closed_at = datetime.now().isoformat()
                        self.realized_pnl += trade.realized_pnl
                        print(f"\n[ForwardTest] LOST: {trade.market[:40]}")
                        print(f"  P&L: ${trade.realized_pnl:.2f}")

                    elif current_price >= 0.999:
                        # Won
                        trade.status = "WON"
                        trade.realized_pnl = trade.shares - trade.cost
                        trade.closed_at = datetime.now().isoformat()
                        self.realized_pnl += trade.realized_pnl
                        print(f"\n[ForwardTest] WON: {trade.market[:40]}")
                        print(f"  P&L: ${trade.realized_pnl:+.2f}")

                except Exception as e:
                    pass

        self._save_state()

    def get_summary(self) -> Dict:
        """Get portfolio summary."""
        open_trades = [t for t in self.trades.values() if t.status == "OPEN"]
        won_trades = [t for t in self.trades.values() if t.status == "WON"]
        lost_trades = [t for t in self.trades.values() if t.status == "LOST"]

        total_invested = sum(t.cost for t in open_trades)
        unrealized_pnl = sum(t.unrealized_pnl or 0 for t in open_trades)

        total_closed = len(won_trades) + len(lost_trades)
        win_rate = len(won_trades) / total_closed if total_closed > 0 else 0

        return {
            "open_positions": len(open_trades),
            "won": len(won_trades),
            "lost": len(lost_trades),
            "win_rate": win_rate,
            "total_invested": total_invested,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": unrealized_pnl + self.realized_pnl,
        }

    def print_status(self):
        """Print current status."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("FORWARD TEST - OPTIMIZED STRATEGY")
        print("=" * 60)
        print(f"Parameters: {self.PREFERRED_SIDE} only, {self.MIN_ENTRY_PROB:.0%}-{self.MAX_ENTRY_PROB:.0%} entry")
        print("-" * 60)
        print(f"Open: {summary['open_positions']} | Won: {summary['won']} | Lost: {summary['lost']}")
        print(f"Win Rate: {summary['win_rate']:.0%}")
        print(f"Invested: ${summary['total_invested']:.2f}")
        print(f"Unrealized P&L: ${summary['unrealized_pnl']:+.2f}")
        print(f"Realized P&L: ${summary['realized_pnl']:+.2f}")
        print(f"TOTAL P&L: ${summary['total_pnl']:+.2f}")
        print("-" * 60)

        if summary['open_positions'] > 0:
            print("\nOPEN POSITIONS:")
            for t in self.trades.values():
                if t.status == "OPEN":
                    pnl_str = f"${t.unrealized_pnl:+.2f}" if t.unrealized_pnl else "N/A"
                    print(f"  {t.side} @ ${t.entry_price:.2f} | {pnl_str} | {t.market[:35]}")

        print("=" * 60)


async def run_forward_test(
    trade_size: float = 10.0,
    max_positions: int = 10,
    scan_interval: int = 120,
):
    """
    Run forward testing with optimized parameters.
    """
    tester = ForwardTester()

    print("\n" + "=" * 60)
    print("FORWARD TESTING - OPTIMIZED STRATEGY")
    print("=" * 60)
    print(f"Entry: {tester.MIN_ENTRY_PROB:.0%}-{tester.MAX_ENTRY_PROB:.0%} probability")
    print(f"Side: {tester.PREFERRED_SIDE} only (backtest: 86% win rate)")
    print(f"Trade Size: ${trade_size}")
    print(f"Max Positions: {max_positions}")
    print("=" * 60 + "\n")

    while True:
        try:
            # Update existing positions
            await tester.update_positions()

            # Scan for new opportunities
            summary = tester.get_summary()
            if summary["open_positions"] < max_positions:
                opps = await tester.scan_for_opportunities()

                print(f"\n[ForwardTest] Found {len(opps)} opportunities matching criteria")

                # Open top opportunities
                for opp in opps[:3]:
                    if summary["open_positions"] >= max_positions:
                        break

                    tester.open_trade(opp, trade_size)
                    summary["open_positions"] += 1

            # Print status
            tester.print_status()

        except Exception as e:
            print(f"[ForwardTest] Error: {e}")

        await asyncio.sleep(scan_interval)


if __name__ == "__main__":
    asyncio.run(run_forward_test(trade_size=10.0, max_positions=10, scan_interval=120))
