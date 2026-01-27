"""
Paper Trading Tracker

Tracks all paper trades and real positions with P&L monitoring.
Runs multiple strategies simultaneously and records performance.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional
import httpx

from .config import config
from .max_return_strategy import MaxReturnStrategy, StrategySignal


@dataclass
class PaperTrade:
    """A paper trade record."""
    id: str
    timestamp: str
    strategy: str
    market: str
    condition_id: str
    token_id: str
    side: str
    entry_price: float
    shares: float
    cost: float
    status: str  # OPEN, WON, LOST, SOLD
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    notes: str = ""


class PaperTrader:
    """
    Manages paper trading across all strategies.
    """

    def __init__(self, state_file: str = "paper_trades.json"):
        self.state_file = state_file
        self.trades: Dict[str, PaperTrade] = {}
        self.strategy = MaxReturnStrategy()
        self._load_state()

    def _load_state(self):
        """Load previous state."""
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
                for trade_data in data.get("trades", []):
                    trade = PaperTrade(**trade_data)
                    self.trades[trade.id] = trade
                print(f"[PaperTrader] Loaded {len(self.trades)} trades")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[PaperTrader] Error loading state: {e}")

    def _save_state(self):
        """Save current state."""
        try:
            data = {
                "trades": [asdict(t) for t in self.trades.values()],
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[PaperTrader] Error saving state: {e}")

    def record_paper_trade(self, signal: StrategySignal, size_usd: float = 10.0) -> PaperTrade:
        """Record a new paper trade from a signal."""
        shares = size_usd / signal.probability if signal.probability > 0 else 0

        trade = PaperTrade(
            id=f"{signal.strategy}_{signal.condition_id[:8]}_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            strategy=signal.strategy,
            market=signal.market_slug,
            condition_id=signal.condition_id,
            token_id=signal.token_id,
            side=signal.side,
            entry_price=signal.probability,
            shares=shares,
            cost=size_usd,
            status="OPEN",
            notes=signal.rationale,
        )

        self.trades[trade.id] = trade
        self._save_state()

        print(f"[PaperTrade] Opened: {signal.strategy} {signal.side} @ ${signal.probability:.3f}")
        print(f"  Market: {signal.market_slug}")
        print(f"  Cost: ${size_usd:.2f} for {shares:.2f} shares")

        return trade

    async def update_positions(self):
        """Update all open positions with current prices."""
        async with httpx.AsyncClient(timeout=30) as client:
            for trade_id, trade in list(self.trades.items()):
                if trade.status != "OPEN":
                    continue

                try:
                    # Get current price from CLOB
                    r = await client.get(
                        f"https://clob.polymarket.com/price",
                        params={"token_id": trade.token_id}
                    )
                    data = r.json()
                    current_price = float(data.get("price", trade.entry_price))

                    # Calculate unrealized P&L
                    current_value = trade.shares * current_price
                    unrealized_pnl = current_value - trade.cost

                    trade.pnl = unrealized_pnl

                except Exception as e:
                    pass  # Keep last known P&L

        self._save_state()

    def close_trade(self, trade_id: str, won: bool, exit_price: float = None):
        """Close a paper trade as won or lost."""
        if trade_id not in self.trades:
            return

        trade = self.trades[trade_id]

        if won:
            trade.status = "WON"
            trade.exit_price = 1.0  # Full payout
            trade.pnl = trade.shares - trade.cost
        else:
            trade.status = "LOST"
            trade.exit_price = 0.0
            trade.pnl = -trade.cost

        self._save_state()

        print(f"[PaperTrade] Closed: {trade.strategy} {trade.market}")
        print(f"  Result: {trade.status} | P&L: ${trade.pnl:+.2f}")

    def get_summary(self) -> dict:
        """Get portfolio summary."""
        open_trades = [t for t in self.trades.values() if t.status == "OPEN"]
        closed_trades = [t for t in self.trades.values() if t.status in ("WON", "LOST")]

        total_invested = sum(t.cost for t in open_trades)
        unrealized_pnl = sum(t.pnl or 0 for t in open_trades)
        realized_pnl = sum(t.pnl or 0 for t in closed_trades)

        wins = len([t for t in closed_trades if t.status == "WON"])
        losses = len([t for t in closed_trades if t.status == "LOST"])
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        # By strategy
        by_strategy = {}
        for t in self.trades.values():
            if t.strategy not in by_strategy:
                by_strategy[t.strategy] = {"count": 0, "pnl": 0, "invested": 0}
            by_strategy[t.strategy]["count"] += 1
            by_strategy[t.strategy]["pnl"] += t.pnl or 0
            if t.status == "OPEN":
                by_strategy[t.strategy]["invested"] += t.cost

        return {
            "open_positions": len(open_trades),
            "closed_positions": len(closed_trades),
            "total_invested": total_invested,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "total_pnl": unrealized_pnl + realized_pnl,
            "win_rate": win_rate,
            "by_strategy": by_strategy,
        }

    def print_status(self):
        """Print current portfolio status."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("PAPER TRADING PORTFOLIO")
        print("=" * 60)
        print(f"Open Positions: {summary['open_positions']}")
        print(f"Closed: {summary['closed_positions']} (Win Rate: {summary['win_rate']:.0%})")
        print(f"Total Invested: ${summary['total_invested']:.2f}")
        print(f"Unrealized P&L: ${summary['unrealized_pnl']:+.2f}")
        print(f"Realized P&L: ${summary['realized_pnl']:+.2f}")
        print(f"Total P&L: ${summary['total_pnl']:+.2f}")
        print()

        print("BY STRATEGY:")
        for strat, data in summary['by_strategy'].items():
            print(f"  {strat}: {data['count']} trades, ${data['pnl']:+.2f} P&L")

        print()
        print("OPEN POSITIONS:")
        for t in self.trades.values():
            if t.status == "OPEN":
                pnl_str = f"${t.pnl:+.2f}" if t.pnl else "N/A"
                print(f"  [{t.strategy}] {t.side} @ ${t.entry_price:.3f} | {pnl_str} | {t.market[:35]}")

        print("=" * 60)


async def run_paper_trading(
    strategies_enabled: List[str] = None,
    trade_size: float = 10.0,
    max_positions: int = 10,
    scan_interval: int = 300,
):
    """
    Run automated paper trading.

    Args:
        strategies_enabled: List of strategies to use (None = all)
        trade_size: USD per trade
        max_positions: Max open positions
        scan_interval: Seconds between scans
    """
    trader = PaperTrader()
    strategy = MaxReturnStrategy()

    print("\n" + "=" * 60)
    print("AUTOMATED PAPER TRADING")
    print("=" * 60)
    print(f"Trade Size: ${trade_size}")
    print(f"Max Positions: {max_positions}")
    print(f"Scan Interval: {scan_interval}s")
    print(f"Strategies: {strategies_enabled or 'ALL'}")
    print("=" * 60 + "\n")

    while True:
        try:
            # Update existing positions
            await trader.update_positions()

            # Check if we can open new positions
            summary = trader.get_summary()
            if summary["open_positions"] >= max_positions:
                print(f"[PaperTrader] Max positions reached ({max_positions})")
            else:
                # Scan for signals
                signals = await strategy.scan_all_strategies()

                # Filter by enabled strategies
                if strategies_enabled:
                    signals = [s for s in signals if s.strategy in strategies_enabled]

                # Filter out already-traded markets
                traded_conditions = {t.condition_id for t in trader.trades.values() if t.status == "OPEN"}
                signals = [s for s in signals if s.condition_id not in traded_conditions]

                # Take top signals
                for signal in signals[:3]:
                    if summary["open_positions"] >= max_positions:
                        break

                    # Open paper trade
                    trader.record_paper_trade(signal, trade_size)
                    summary["open_positions"] += 1

            # Print status
            trader.print_status()

        except Exception as e:
            print(f"[PaperTrader] Error: {e}")

        await asyncio.sleep(scan_interval)


if __name__ == "__main__":
    asyncio.run(run_paper_trading(
        strategies_enabled=["COPY_TRADE", "ENDGAME", "HIGH_PROB_BOND"],
        trade_size=10.0,
        max_positions=10,
        scan_interval=300
    ))
