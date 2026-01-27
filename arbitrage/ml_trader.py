"""
ML-Powered Live Trading System

Uses the trained ML engine to make trading decisions in real-time.
Continuously learns from outcomes to improve predictions.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional
import httpx

from .config import config
from .ml_engine import MLEngine, TradeFeatures, extract_features, ml_engine


@dataclass
class MLTrade:
    """An ML-driven trade."""
    id: str
    timestamp: str
    market: str
    condition_id: str
    token_id: str
    side: str
    entry_price: float
    shares: float
    cost: float
    status: str  # OPEN, WON, LOST
    ml_win_prob: float
    ml_confidence: float
    kelly_fraction: float
    features_snapshot: Dict
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
    closed_at: Optional[str] = None


class MLTrader:
    """
    ML-powered trading system.

    Uses machine learning to:
    1. Score potential trades
    2. Determine optimal position size (Kelly)
    3. Filter for high-probability setups
    4. Learn from outcomes
    """

    # kingofcoinflips wallet for copy trading
    WHALE_WALLET = "0xe9c6312464b52aa3eff13d822b003282075995c9"

    def __init__(self, state_file: str = "ml_trader_state.json", bankroll: float = 100.0):
        self.state_file = state_file
        self.bankroll = bankroll
        self.trades: Dict[str, MLTrade] = {}
        self.ml = ml_engine  # Use global ML engine
        self._load_state()

    def _load_state(self):
        """Load previous state."""
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
                self.bankroll = data.get("bankroll", self.bankroll)
                for trade_data in data.get("trades", []):
                    trade = MLTrade(**trade_data)
                    self.trades[trade.id] = trade
                print(f"[MLTrader] Loaded {len(self.trades)} trades, bankroll: ${self.bankroll:.2f}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[MLTrader] Error loading state: {e}")

    def _save_state(self):
        """Save current state."""
        try:
            data = {
                "bankroll": self.bankroll,
                "trades": [asdict(t) for t in self.trades.values()],
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[MLTrader] Error saving state: {e}")

    async def scan_opportunities(self) -> List[Dict]:
        """
        Scan for ML-approved opportunities.
        """
        approved = []

        async with httpx.AsyncClient(timeout=30) as client:
            # Get whale positions
            r = await client.get(
                "https://data-api.polymarket.com/positions",
                params={"user": self.WHALE_WALLET, "sizeThreshold": 0}
            )
            positions = r.json()

            for pos in positions:
                # Extract ML features
                features = extract_features(pos)

                # Check with ML engine
                should_trade, reason = self.ml.should_trade(features)

                if not should_trade:
                    continue

                # Get ML prediction
                prediction = self.ml.predict(features, self.bankroll)

                # Skip if already have position
                condition_id = pos.get("conditionId", "")
                if condition_id in [t.condition_id for t in self.trades.values() if t.status == "OPEN"]:
                    continue

                # Skip if not profitable for whale (bad signal)
                whale_pnl = float(pos.get("cashPnl", 0) or 0)
                if whale_pnl <= 0:
                    continue

                approved.append({
                    "market": pos.get("title", "")[:60],
                    "slug": pos.get("slug", ""),
                    "condition_id": condition_id,
                    "token_id": pos.get("asset", ""),
                    "outcome": pos.get("outcome", ""),
                    "current_price": features.entry_price,
                    "features": features,
                    "prediction": prediction,
                    "whale_pnl": whale_pnl,
                    "reason": reason,
                })

            # Sort by expected return
            approved.sort(key=lambda x: x["prediction"].expected_return, reverse=True)

        return approved

    def open_trade(self, opp: Dict) -> MLTrade:
        """Open a new ML-driven trade."""
        prediction = opp["prediction"]
        features = opp["features"]

        # Use Kelly-optimal size (capped)
        size_usd = min(prediction.recommended_size, self.bankroll * 0.10, config.MAX_POSITION_SIZE)
        size_usd = max(size_usd, config.MIN_POSITION_SIZE)

        shares = size_usd / features.entry_price if features.entry_price > 0 else 0

        trade = MLTrade(
            id=f"ML_{opp['condition_id'][:8]}_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            market=opp["market"],
            condition_id=opp["condition_id"],
            token_id=opp["token_id"],
            side=opp["outcome"],
            entry_price=features.entry_price,
            shares=shares,
            cost=size_usd,
            status="OPEN",
            ml_win_prob=prediction.win_probability,
            ml_confidence=prediction.confidence,
            kelly_fraction=prediction.kelly_fraction,
            features_snapshot=asdict(features),
            current_price=features.entry_price,
            unrealized_pnl=0.0,
        )

        self.trades[trade.id] = trade
        self.bankroll -= size_usd
        self._save_state()

        print(f"\n[MLTrader] OPENED: {trade.side} @ ${trade.entry_price:.3f}")
        print(f"  Market: {trade.market}")
        print(f"  ML Win Prob: {trade.ml_win_prob:.0%}")
        print(f"  Kelly Size: ${size_usd:.2f} ({trade.kelly_fraction:.0%} of bankroll)")
        print(f"  Expected Return: {prediction.expected_return:.1%}")
        print(f"  Reason: {opp['reason']}")

        return trade

    async def update_positions(self):
        """Update all open positions and check for settlements."""
        async with httpx.AsyncClient(timeout=30) as client:
            for trade_id, trade in list(self.trades.items()):
                if trade.status != "OPEN":
                    continue

                try:
                    # Get current price
                    r = await client.get(
                        f"https://clob.polymarket.com/price",
                        params={"token_id": trade.token_id}
                    )
                    data = r.json()
                    current_price = float(data.get("price", trade.entry_price))

                    trade.current_price = current_price
                    trade.unrealized_pnl = (current_price - trade.entry_price) * trade.shares

                    # Check for settlement
                    if current_price <= 0.001:
                        # LOST
                        self._close_trade(trade, won=False)

                    elif current_price >= 0.999:
                        # WON
                        self._close_trade(trade, won=True)

                except Exception:
                    pass

        self._save_state()

    def _close_trade(self, trade: MLTrade, won: bool):
        """Close a trade and update ML engine."""
        if won:
            trade.status = "WON"
            trade.realized_pnl = trade.shares - trade.cost
            self.bankroll += trade.shares  # Full payout
        else:
            trade.status = "LOST"
            trade.realized_pnl = -trade.cost
            # Bankroll already reduced

        trade.closed_at = datetime.now().isoformat()

        # Update ML engine with outcome
        features = TradeFeatures(**trade.features_snapshot)
        actual_return = trade.realized_pnl / trade.cost if trade.cost > 0 else 0
        self.ml.record_outcome(features, won, actual_return)

        print(f"\n[MLTrader] {'WON' if won else 'LOST'}: {trade.market[:40]}")
        print(f"  P&L: ${trade.realized_pnl:+.2f}")
        print(f"  New Bankroll: ${self.bankroll:.2f}")

        self._save_state()

    def get_summary(self) -> Dict:
        """Get trading summary."""
        open_trades = [t for t in self.trades.values() if t.status == "OPEN"]
        won_trades = [t for t in self.trades.values() if t.status == "WON"]
        lost_trades = [t for t in self.trades.values() if t.status == "LOST"]

        total_invested = sum(t.cost for t in open_trades)
        unrealized = sum(t.unrealized_pnl or 0 for t in open_trades)
        realized = sum(t.realized_pnl or 0 for t in won_trades + lost_trades)

        total_closed = len(won_trades) + len(lost_trades)
        win_rate = len(won_trades) / total_closed if total_closed > 0 else 0

        return {
            "bankroll": self.bankroll,
            "open_positions": len(open_trades),
            "total_invested": total_invested,
            "won": len(won_trades),
            "lost": len(lost_trades),
            "win_rate": win_rate,
            "unrealized_pnl": unrealized,
            "realized_pnl": realized,
            "total_pnl": unrealized + realized,
        }

    def print_status(self):
        """Print current status."""
        summary = self.get_summary()
        ml_stats = self.ml.get_stats()

        print("\n" + "=" * 70)
        print("ML-POWERED TRADING SYSTEM")
        print("=" * 70)
        print(f"Bankroll: ${summary['bankroll']:.2f}")
        print(f"Open: {summary['open_positions']} | Won: {summary['won']} | Lost: {summary['lost']}")
        print(f"Win Rate: {summary['win_rate']:.0%}")
        print(f"Invested: ${summary['total_invested']:.2f}")
        print(f"Unrealized P&L: ${summary['unrealized_pnl']:+.2f}")
        print(f"Realized P&L: ${summary['realized_pnl']:+.2f}")
        print(f"TOTAL P&L: ${summary['total_pnl']:+.2f}")
        print("-" * 70)
        print(f"ML Training Samples: {ml_stats['training_examples']}")
        print(f"ML Recent Win Rate: {ml_stats['recent_win_rate']:.0%}")
        print(f"ML Optimal Entry: {ml_stats['optimal_entry_range']}")
        print("-" * 70)

        if summary['open_positions'] > 0:
            print("\nOPEN POSITIONS:")
            for t in self.trades.values():
                if t.status == "OPEN":
                    pnl = f"${t.unrealized_pnl:+.2f}" if t.unrealized_pnl else "N/A"
                    print(f"  {t.side} @ ${t.entry_price:.2f} | ML:{t.ml_win_prob:.0%} | {pnl} | {t.market[:30]}")

        print("=" * 70)


async def run_ml_trading(
    bankroll: float = 100.0,
    max_positions: int = 8,
    scan_interval: int = 120,
):
    """
    Run ML-powered trading system.
    """
    trader = MLTrader(bankroll=bankroll)

    print("\n" + "=" * 70)
    print("ML-POWERED TRADING SYSTEM - STARTING")
    print("=" * 70)
    print(f"Bankroll: ${bankroll}")
    print(f"Max Positions: {max_positions}")
    print(f"Scan Interval: {scan_interval}s")
    print("=" * 70 + "\n")

    # Train ML on historical data first
    print("[MLTrader] Loading historical data for training...")
    try:
        with open('C:/Users/Star/.local/bin/star-polymarket/data/btc_backtest_compact.json', 'r') as f:
            data = json.load(f)

        for trade in data.get('whale_trades', []):
            features = extract_features(trade)
            pnl = float(trade.get('pnl', 0) or 0)
            trader.ml.record_outcome(features, pnl > 0, pnl / 10000)

        print(f"[MLTrader] Trained on {len(data.get('whale_trades', []))} historical trades")
    except Exception as e:
        print(f"[MLTrader] Could not load training data: {e}")

    while True:
        try:
            # Update existing positions
            await trader.update_positions()

            # Scan for new opportunities
            summary = trader.get_summary()
            if summary["open_positions"] < max_positions:
                opps = await trader.scan_opportunities()

                if opps:
                    print(f"\n[MLTrader] Found {len(opps)} ML-approved opportunities")

                # Open top opportunities
                for opp in opps[:2]:  # Max 2 per scan
                    if summary["open_positions"] >= max_positions:
                        break
                    if summary["bankroll"] < config.MIN_POSITION_SIZE:
                        print("[MLTrader] Insufficient bankroll")
                        break

                    trader.open_trade(opp)
                    summary["open_positions"] += 1

            # Print status
            trader.print_status()

        except Exception as e:
            print(f"[MLTrader] Error: {e}")

        await asyncio.sleep(scan_interval)


if __name__ == "__main__":
    asyncio.run(run_ml_trading(bankroll=100.0, max_positions=8, scan_interval=120))
