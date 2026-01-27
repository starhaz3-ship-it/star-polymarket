"""
Data collector for ML training.
Collects paper trading signals, market conditions, and outcomes.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any

from .config import config


@dataclass
class TradeSignal:
    """A recorded trading signal for ML training."""
    timestamp: float
    signal_type: str
    market_id: str
    market_question: str

    # Market conditions at signal time
    spot_price: float
    yes_price: float
    no_price: float
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    spread: float

    # Signal metrics
    edge_percent: float
    confidence: float
    recommended_size: float

    # Time factors
    time_to_expiry_sec: Optional[float]
    hour_of_day: int
    day_of_week: int

    # Outcome (filled in later)
    executed: bool = False
    outcome_price: Optional[float] = None  # Final settlement price
    actual_profit: Optional[float] = None
    was_profitable: Optional[bool] = None


@dataclass
class MarketSnapshot:
    """Point-in-time market data for analysis."""
    timestamp: float
    market_id: str
    spot_price: float
    yes_price: float
    no_price: float
    volume_24h: float
    liquidity: float
    time_to_expiry_sec: Optional[float]


class DataCollector:
    """Collects and stores trading data for ML training."""

    def __init__(self, data_dir: str = "ml_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.signals_file = os.path.join(data_dir, "signals.jsonl")
        self.snapshots_file = os.path.join(data_dir, "snapshots.jsonl")
        self.outcomes_file = os.path.join(data_dir, "outcomes.jsonl")

        # In-memory cache for current session
        self.session_signals: List[TradeSignal] = []
        self.session_snapshots: List[MarketSnapshot] = []

    def record_signal(self, signal: TradeSignal):
        """Record a trading signal."""
        self.session_signals.append(signal)

        # Append to file
        with open(self.signals_file, "a") as f:
            f.write(json.dumps(asdict(signal)) + "\n")

        print(f"[DataCollector] Recorded signal: {signal.signal_type} | Edge: {signal.edge_percent:.2f}%")

    def record_snapshot(self, snapshot: MarketSnapshot):
        """Record a market snapshot."""
        self.session_snapshots.append(snapshot)

        # Append to file (less verbose)
        with open(self.snapshots_file, "a") as f:
            f.write(json.dumps(asdict(snapshot)) + "\n")

    def record_outcome(self, market_id: str, settlement_price: float, profit: float):
        """Record the outcome of a market."""
        outcome = {
            "timestamp": time.time(),
            "market_id": market_id,
            "settlement_price": settlement_price,
            "profit": profit,
            "was_profitable": profit > 0
        }

        with open(self.outcomes_file, "a") as f:
            f.write(json.dumps(outcome) + "\n")

        print(f"[DataCollector] Recorded outcome: {market_id} | Profit: ${profit:.2f}")

    def load_all_signals(self) -> List[Dict]:
        """Load all recorded signals for training."""
        signals = []
        if os.path.exists(self.signals_file):
            with open(self.signals_file, "r") as f:
                for line in f:
                    try:
                        signals.append(json.loads(line.strip()))
                    except:
                        continue
        return signals

    def load_all_outcomes(self) -> Dict[str, Dict]:
        """Load all outcomes indexed by market_id."""
        outcomes = {}
        if os.path.exists(self.outcomes_file):
            with open(self.outcomes_file, "r") as f:
                for line in f:
                    try:
                        o = json.loads(line.strip())
                        outcomes[o["market_id"]] = o
                    except:
                        continue
        return outcomes

    def get_training_data(self) -> List[Dict]:
        """Get signals with their outcomes for ML training."""
        signals = self.load_all_signals()
        outcomes = self.load_all_outcomes()

        training_data = []
        for signal in signals:
            market_id = signal.get("market_id")
            if market_id in outcomes:
                signal["outcome"] = outcomes[market_id]
                signal["was_profitable"] = outcomes[market_id].get("was_profitable")
                signal["actual_profit"] = outcomes[market_id].get("profit")
                training_data.append(signal)

        return training_data

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics on collected data."""
        signals = self.load_all_signals()
        outcomes = self.load_all_outcomes()
        training_data = self.get_training_data()

        profitable = [d for d in training_data if d.get("was_profitable")]

        return {
            "total_signals": len(signals),
            "total_outcomes": len(outcomes),
            "training_samples": len(training_data),
            "profitable_trades": len(profitable),
            "win_rate": len(profitable) / len(training_data) if training_data else 0,
            "session_signals": len(self.session_signals),
        }


# Global collector instance
collector = DataCollector()


def create_signal_from_arb(signal, spot_price: float = 0) -> TradeSignal:
    """Create a TradeSignal from an ArbitrageSignal."""
    now = datetime.now()
    market = signal.market

    # Handle different market types
    if hasattr(market, 'yes_price'):
        yes_price = market.yes_price
        no_price = market.no_price
        yes_bid = getattr(market, 'yes_bid', 0)
        yes_ask = getattr(market, 'yes_ask', 1)
        no_bid = getattr(market, 'no_bid', 0)
        no_ask = getattr(market, 'no_ask', 1)
        spread = market.spread if hasattr(market, 'spread') else yes_price + no_price
    else:
        # Multi-outcome market
        yes_price = market.total_probability if hasattr(market, 'total_probability') else 0.5
        no_price = 1 - yes_price
        yes_bid = yes_ask = no_bid = no_ask = 0.5
        spread = 1.0

    return TradeSignal(
        timestamp=time.time(),
        signal_type=signal.signal_type.value,
        market_id=market.condition_id if hasattr(market, 'condition_id') else "",
        market_question=market.question if hasattr(market, 'question') else "",
        spot_price=spot_price,
        yes_price=yes_price,
        no_price=no_price,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        spread=spread,
        edge_percent=signal.edge_percent,
        confidence=signal.confidence,
        recommended_size=signal.recommended_size,
        time_to_expiry_sec=market.time_remaining_sec if hasattr(market, 'time_remaining_sec') else None,
        hour_of_day=now.hour,
        day_of_week=now.weekday(),
        executed=False,
    )
