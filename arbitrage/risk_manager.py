"""
Risk management for the arbitrage bot.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional

from .config import config
from .detector import ArbitrageSignal
from .executor import ExecutionResult


@dataclass
class Position:
    """An open position."""
    market_id: str
    market_question: str
    side: str  # YES, NO, or BOTH
    size: float
    entry_price: float
    entry_time: float
    token_id: str

    @property
    def cost_basis(self) -> float:
        return self.size * self.entry_price


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: str
    trades_executed: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    gross_profit: float = 0.0
    fees_paid: float = 0.0
    net_profit: float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.trades_won + self.trades_lost
        return self.trades_won / total if total > 0 else 0.0


class RiskManager:
    """Manages risk and position tracking."""

    def __init__(self, state_file: str = "bot_state.json"):
        self.state_file = state_file
        self.positions: Dict[str, Position] = {}
        self.daily_stats: DailyStats = DailyStats(date=str(date.today()))
        self.trade_history: List[dict] = []
        self._load_state()

    def _load_state(self):
        """Load state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)

                # Load daily stats
                stats_data = data.get("daily_stats", {})
                if stats_data.get("date") == str(date.today()):
                    self.daily_stats = DailyStats(**stats_data)
                else:
                    # New day, reset stats
                    self.daily_stats = DailyStats(date=str(date.today()))

                # Load positions
                for pos_data in data.get("positions", []):
                    pos = Position(**pos_data)
                    self.positions[pos.market_id] = pos

                # Load trade history (last 100)
                self.trade_history = data.get("trade_history", [])[-100:]

                print(f"[Risk] Loaded state: {len(self.positions)} positions, "
                      f"${self.daily_stats.net_profit:.2f} P&L today")

            except Exception as e:
                print(f"[Risk] Error loading state: {e}")

    def _save_state(self):
        """Save state to file."""
        try:
            data = {
                "daily_stats": {
                    "date": self.daily_stats.date,
                    "trades_executed": self.daily_stats.trades_executed,
                    "trades_won": self.daily_stats.trades_won,
                    "trades_lost": self.daily_stats.trades_lost,
                    "gross_profit": self.daily_stats.gross_profit,
                    "fees_paid": self.daily_stats.fees_paid,
                    "net_profit": self.daily_stats.net_profit,
                },
                "positions": [
                    {
                        "market_id": p.market_id,
                        "market_question": p.market_question,
                        "side": p.side,
                        "size": p.size,
                        "entry_price": p.entry_price,
                        "entry_time": p.entry_time,
                        "token_id": p.token_id,
                    }
                    for p in self.positions.values()
                ],
                "trade_history": self.trade_history[-100:],
            }

            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"[Risk] Error saving state: {e}")

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed based on risk limits."""
        # Check daily loss limit
        if self.daily_stats.net_profit < -config.MAX_DAILY_LOSS:
            return False, f"Daily loss limit reached: ${self.daily_stats.net_profit:.2f}"

        # Check max open positions
        if len(self.positions) >= config.MAX_OPEN_POSITIONS:
            return False, f"Max positions reached: {len(self.positions)}"

        return True, "OK"

    def should_take_signal(self, signal: ArbitrageSignal) -> tuple[bool, str]:
        """Evaluate if we should take a trading signal."""
        # Check basic trading allowance
        can, reason = self.can_trade()
        if not can:
            return False, reason

        # Check if we already have a position in this market
        if signal.market.condition_id in self.positions:
            return False, "Already have position in this market"

        # Check minimum edge
        if signal.edge_percent < config.MIN_EDGE_PERCENT:
            return False, f"Edge too low: {signal.edge_percent:.2f}%"

        # Check confidence threshold
        if signal.confidence < 0.3:
            return False, f"Confidence too low: {signal.confidence:.2f}"

        # Check market time remaining
        if signal.market.time_remaining_sec < 60:
            return False, "Market closing too soon"

        return True, "OK"

    def calculate_position_size(self, signal: ArbitrageSignal) -> float:
        """Calculate optimal position size using Kelly criterion."""
        # Kelly fraction: edge / odds
        # Simplified: edge_pct * confidence * max_size

        kelly = signal.edge_percent / 100 * signal.confidence

        # Apply fractional Kelly (more conservative)
        kelly = kelly * 0.25

        size = kelly * config.MAX_POSITION_SIZE

        # Apply limits
        size = max(config.MIN_POSITION_SIZE, min(config.MAX_POSITION_SIZE, size))

        # Don't risk more than we can afford to lose today
        remaining_budget = config.MAX_DAILY_LOSS + self.daily_stats.net_profit
        size = min(size, remaining_budget * 0.5)

        return max(config.MIN_POSITION_SIZE, size)

    def record_trade(
        self,
        signal: ArbitrageSignal,
        result: ExecutionResult
    ):
        """Record a completed trade."""
        self.daily_stats.trades_executed += 1

        if result.success:
            # Add position
            pos = Position(
                market_id=signal.market.condition_id,
                market_question=signal.market.question,
                side=signal.signal_type.value,
                size=result.filled_size,
                entry_price=result.filled_price,
                entry_time=time.time(),
                token_id=signal.market.yes_token_id if "yes" in signal.signal_type.value else signal.market.no_token_id,
            )
            self.positions[signal.market.condition_id] = pos

            # Update fees
            self.daily_stats.fees_paid += result.fees_paid

        # Record in history
        self.trade_history.append({
            "timestamp": time.time(),
            "market": signal.market.question[:50],
            "signal": signal.signal_type.value,
            "edge": signal.edge_percent,
            "size": result.filled_size,
            "price": result.filled_price,
            "success": result.success,
            "error": result.error_message,
        })

        self._save_state()

    def record_settlement(
        self,
        market_id: str,
        won: bool,
        payout: float
    ):
        """Record a market settlement."""
        if market_id not in self.positions:
            return

        pos = self.positions[market_id]

        if won:
            self.daily_stats.trades_won += 1
            profit = payout - pos.cost_basis
        else:
            self.daily_stats.trades_lost += 1
            profit = -pos.cost_basis

        self.daily_stats.gross_profit += profit
        self.daily_stats.net_profit = self.daily_stats.gross_profit - self.daily_stats.fees_paid

        # Remove position
        del self.positions[market_id]

        self._save_state()

        print(f"[Risk] Settlement: {'WON' if won else 'LOST'} | "
              f"P&L: ${profit:.2f} | Daily: ${self.daily_stats.net_profit:.2f}")

    def get_status(self) -> dict:
        """Get current risk status."""
        return {
            "positions": len(self.positions),
            "daily_trades": self.daily_stats.trades_executed,
            "daily_pnl": self.daily_stats.net_profit,
            "win_rate": self.daily_stats.win_rate,
            "can_trade": self.can_trade()[0],
        }

    def print_status(self):
        """Print current status."""
        status = self.get_status()
        print("\n" + "=" * 50)
        print("RISK STATUS")
        print("=" * 50)
        print(f"Open Positions: {status['positions']}/{config.MAX_OPEN_POSITIONS}")
        print(f"Daily Trades: {status['daily_trades']}")
        print(f"Daily P&L: ${status['daily_pnl']:.2f}")
        print(f"Win Rate: {status['win_rate']*100:.1f}%")
        print(f"Can Trade: {'YES' if status['can_trade'] else 'NO'}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    rm = RiskManager()
    rm.print_status()
