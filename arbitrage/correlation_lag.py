"""
Correlation Lag Arbitrage Strategy

Based on research from @arndxt_xo and @the_smart_ape:
- Markets correlated at 0.8-0.99 have exploitable lag
- One market moves first, the other follows minutes later
- Trade the lagging market before it catches up
- Negative correlations (<-0.85) work for inverse bets

Sources:
- https://x.com/arndxt_xo/status/2008579579812999656
- https://x.com/the_smart_ape/status/2008556048282509548
- polymarketcorrelation.com (tool by the_smart_ape)
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import httpx
import numpy as np

from .config import config


class CorrelationType(Enum):
    """Type of correlation between markets."""
    POSITIVE = "positive"  # 0.8 to 0.99 - same direction
    NEGATIVE = "negative"  # -0.85 to -0.99 - inverse direction
    NONE = "none"  # No tradeable correlation


@dataclass
class MarketPrice:
    """A single price observation for a market."""
    timestamp: float
    yes_price: float
    no_price: float
    market_id: str
    market_slug: str


@dataclass
class CorrelationPair:
    """A pair of correlated markets."""
    leader_id: str
    leader_slug: str
    follower_id: str
    follower_slug: str
    correlation: float
    correlation_type: CorrelationType
    avg_lag_seconds: float
    last_updated: float


@dataclass
class LagSignal:
    """Signal generated when lag is detected."""
    follower_market_id: str
    follower_slug: str
    direction: str  # "YES" or "NO"
    leader_move_pct: float
    expected_move_pct: float
    confidence: float
    timestamp: float
    leader_slug: str
    correlation: float


@dataclass
class PaperTrade:
    """A paper trade for backtesting."""
    signal: LagSignal
    entry_price: float
    entry_time: float
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "OPEN"


class CorrelationLagDetector:
    """
    Detects and trades correlation lag between Polymarket markets.

    Strategy Parameters (Conservative - 86% ROI in backtest):
    - CORRELATION_THRESHOLD: 0.80 (minimum correlation to consider)
    - CORRELATION_MAX: 0.99 (too perfect = no opportunity)
    - MIN_MOVE_PCT: 2.0 (minimum leader move to trigger signal)
    - MAX_LAG_SECONDS: 300 (5 minutes max lag window)
    - OBSERVATION_WINDOW: 120 (watch first 2 minutes of each round)
    """

    # Strategy parameters
    CORRELATION_MIN = 0.80
    CORRELATION_MAX = 0.99
    NEGATIVE_CORRELATION_THRESHOLD = -0.85
    MIN_MOVE_PCT = 2.0  # Minimum % move in leader to trigger
    MAX_LAG_SECONDS = 300  # 5 minute max lag
    OBSERVATION_WINDOW = 120  # Watch first 2 minutes
    PRICE_HISTORY_SIZE = 100  # Keep last 100 observations per market

    # Paper trading parameters
    PAPER_POSITION_SIZE = 25.0  # $25 per paper trade
    PAPER_MAX_POSITIONS = 5

    def __init__(self, state_file: str = "correlation_lag_state.json"):
        self.state_file = state_file

        # Price history for correlation calculation
        self.price_history: Dict[str, deque] = {}

        # Known correlation pairs
        self.correlation_pairs: Dict[str, CorrelationPair] = {}

        # Active signals
        self.pending_signals: List[LagSignal] = []

        # Paper trading
        self.paper_trades: List[PaperTrade] = []
        self.paper_balance = 1000.0  # Start with $1000
        self.paper_starting_balance = 1000.0

        # Stats
        self.stats = {
            "signals_generated": 0,
            "trades_opened": 0,
            "trades_closed": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "start_time": time.time(),
        }

        self._load_state()

    def _load_state(self):
        """Load previous state."""
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
                self.paper_balance = data.get("paper_balance", 1000.0)
                self.stats = data.get("stats", self.stats)

                # Load paper trades
                for trade_data in data.get("paper_trades", []):
                    signal = LagSignal(**trade_data["signal"])
                    trade = PaperTrade(
                        signal=signal,
                        entry_price=trade_data["entry_price"],
                        entry_time=trade_data["entry_time"],
                        exit_price=trade_data.get("exit_price"),
                        exit_time=trade_data.get("exit_time"),
                        pnl=trade_data.get("pnl"),
                        status=trade_data.get("status", "OPEN"),
                    )
                    self.paper_trades.append(trade)

                print(f"[CorrelationLag] Loaded state: ${self.paper_balance:.2f} balance, {len(self.paper_trades)} trades")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[CorrelationLag] Error loading state: {e}")

    def _save_state(self):
        """Save current state."""
        try:
            data = {
                "paper_balance": self.paper_balance,
                "stats": self.stats,
                "paper_trades": [
                    {
                        "signal": {
                            "follower_market_id": t.signal.follower_market_id,
                            "follower_slug": t.signal.follower_slug,
                            "direction": t.signal.direction,
                            "leader_move_pct": t.signal.leader_move_pct,
                            "expected_move_pct": t.signal.expected_move_pct,
                            "confidence": t.signal.confidence,
                            "timestamp": t.signal.timestamp,
                            "leader_slug": t.signal.leader_slug,
                            "correlation": t.signal.correlation,
                        },
                        "entry_price": t.entry_price,
                        "entry_time": t.entry_time,
                        "exit_price": t.exit_price,
                        "exit_time": t.exit_time,
                        "pnl": t.pnl,
                        "status": t.status,
                    }
                    for t in self.paper_trades[-50:]  # Keep last 50 trades
                ],
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[CorrelationLag] Error saving state: {e}")

    def record_price(self, market_id: str, market_slug: str, yes_price: float, no_price: float):
        """Record a price observation for a market."""
        if market_id not in self.price_history:
            self.price_history[market_id] = deque(maxlen=self.PRICE_HISTORY_SIZE)

        observation = MarketPrice(
            timestamp=time.time(),
            yes_price=yes_price,
            no_price=no_price,
            market_id=market_id,
            market_slug=market_slug,
        )
        self.price_history[market_id].append(observation)

    def calculate_correlation(self, market_a: str, market_b: str) -> Tuple[float, CorrelationType]:
        """Calculate correlation between two markets."""
        if market_a not in self.price_history or market_b not in self.price_history:
            return 0.0, CorrelationType.NONE

        history_a = list(self.price_history[market_a])
        history_b = list(self.price_history[market_b])

        if len(history_a) < 10 or len(history_b) < 10:
            return 0.0, CorrelationType.NONE

        # Align timestamps (find common time range)
        prices_a = []
        prices_b = []

        for obs_a in history_a:
            # Find closest observation in B
            closest_b = min(history_b, key=lambda x: abs(x.timestamp - obs_a.timestamp))
            if abs(closest_b.timestamp - obs_a.timestamp) < 60:  # Within 1 minute
                prices_a.append(obs_a.yes_price)
                prices_b.append(closest_b.yes_price)

        if len(prices_a) < 10:
            return 0.0, CorrelationType.NONE

        # Calculate Pearson correlation
        try:
            correlation = np.corrcoef(prices_a, prices_b)[0, 1]
        except:
            return 0.0, CorrelationType.NONE

        if np.isnan(correlation):
            return 0.0, CorrelationType.NONE

        # Determine correlation type
        if self.CORRELATION_MIN <= correlation <= self.CORRELATION_MAX:
            return correlation, CorrelationType.POSITIVE
        elif correlation <= self.NEGATIVE_CORRELATION_THRESHOLD:
            return correlation, CorrelationType.NEGATIVE
        else:
            return correlation, CorrelationType.NONE

    def detect_leader_move(self, market_id: str) -> Optional[Tuple[float, str]]:
        """
        Detect if a market has made a significant move.

        Returns (move_pct, direction) or None.
        """
        if market_id not in self.price_history:
            return None

        history = list(self.price_history[market_id])
        if len(history) < 3:
            return None

        # Compare current price to price from OBSERVATION_WINDOW seconds ago
        current = history[-1]
        now = current.timestamp

        # Find observation from ~2 minutes ago
        reference = None
        for obs in reversed(history):
            if now - obs.timestamp >= self.OBSERVATION_WINDOW:
                reference = obs
                break

        if not reference:
            return None

        # Calculate move
        move_pct = (current.yes_price - reference.yes_price) / reference.yes_price * 100

        if abs(move_pct) >= self.MIN_MOVE_PCT:
            direction = "YES" if move_pct > 0 else "NO"
            return (move_pct, direction)

        return None

    def detect_lag_opportunity(
        self,
        leader_id: str,
        follower_id: str,
        correlation: float,
        correlation_type: CorrelationType
    ) -> Optional[LagSignal]:
        """
        Detect if there's a lag opportunity between two correlated markets.
        """
        # Check if leader has moved
        leader_move = self.detect_leader_move(leader_id)
        if not leader_move:
            return None

        move_pct, direction = leader_move

        # Check if follower has caught up
        follower_history = self.price_history.get(follower_id, deque())
        if len(follower_history) < 2:
            return None

        follower_current = follower_history[-1]
        follower_reference = None
        for obs in reversed(list(follower_history)):
            if follower_current.timestamp - obs.timestamp >= self.OBSERVATION_WINDOW:
                follower_reference = obs
                break

        if not follower_reference:
            return None

        follower_move_pct = (follower_current.yes_price - follower_reference.yes_price) / follower_reference.yes_price * 100

        # Determine expected move based on correlation
        if correlation_type == CorrelationType.POSITIVE:
            expected_move = move_pct * correlation
            trade_direction = direction
        else:  # NEGATIVE
            expected_move = move_pct * correlation  # Will be negative
            trade_direction = "NO" if direction == "YES" else "YES"

        # Check if follower hasn't caught up yet
        lag_remaining = abs(expected_move) - abs(follower_move_pct)

        if lag_remaining < self.MIN_MOVE_PCT / 2:
            return None  # Follower has already caught up

        # Calculate confidence
        confidence = min(0.9, abs(correlation) * (lag_remaining / abs(expected_move)))

        leader_slug = self.price_history[leader_id][-1].market_slug
        follower_slug = follower_current.market_slug

        return LagSignal(
            follower_market_id=follower_id,
            follower_slug=follower_slug,
            direction=trade_direction,
            leader_move_pct=move_pct,
            expected_move_pct=expected_move,
            confidence=confidence,
            timestamp=time.time(),
            leader_slug=leader_slug,
            correlation=correlation,
        )

    def scan_for_signals(self) -> List[LagSignal]:
        """Scan all tracked markets for lag opportunities."""
        signals = []
        market_ids = list(self.price_history.keys())

        # Check all pairs
        for i, market_a in enumerate(market_ids):
            for market_b in market_ids[i+1:]:
                correlation, corr_type = self.calculate_correlation(market_a, market_b)

                if corr_type == CorrelationType.NONE:
                    continue

                # Check A leads B
                signal = self.detect_lag_opportunity(market_a, market_b, correlation, corr_type)
                if signal:
                    signals.append(signal)

                # Check B leads A
                signal = self.detect_lag_opportunity(market_b, market_a, correlation, corr_type)
                if signal:
                    signals.append(signal)

        self.stats["signals_generated"] += len(signals)
        return signals

    def execute_paper_trade(self, signal: LagSignal) -> Optional[PaperTrade]:
        """Execute a paper trade based on a signal."""
        # Check position limits
        open_trades = [t for t in self.paper_trades if t.status == "OPEN"]
        if len(open_trades) >= self.PAPER_MAX_POSITIONS:
            return None

        # Check if we already have a position in this market
        for trade in open_trades:
            if trade.signal.follower_market_id == signal.follower_market_id:
                return None

        # Get current price
        history = self.price_history.get(signal.follower_market_id, deque())
        if not history:
            return None

        current_price = history[-1].yes_price if signal.direction == "YES" else history[-1].no_price

        trade = PaperTrade(
            signal=signal,
            entry_price=current_price,
            entry_time=time.time(),
        )

        self.paper_trades.append(trade)
        self.stats["trades_opened"] += 1

        print(f"\n[PAPER TRADE] OPENED: {signal.direction} on {signal.follower_slug}")
        print(f"  Entry: ${current_price:.4f} | Size: ${self.PAPER_POSITION_SIZE:.2f}")
        print(f"  Leader: {signal.leader_slug} moved {signal.leader_move_pct:+.2f}%")
        print(f"  Correlation: {signal.correlation:.2f} | Confidence: {signal.confidence:.2f}")

        self._save_state()
        return trade

    def update_paper_trades(self):
        """Update and close paper trades."""
        for trade in self.paper_trades:
            if trade.status != "OPEN":
                continue

            # Check if trade should be closed (5 minute timeout or target reached)
            age = time.time() - trade.entry_time

            history = self.price_history.get(trade.signal.follower_market_id, deque())
            if not history:
                continue

            current_price = history[-1].yes_price if trade.signal.direction == "YES" else history[-1].no_price

            # Calculate unrealized PnL
            if trade.signal.direction == "YES":
                unrealized_pnl = (current_price - trade.entry_price) / trade.entry_price * self.PAPER_POSITION_SIZE
            else:
                unrealized_pnl = (trade.entry_price - current_price) / trade.entry_price * self.PAPER_POSITION_SIZE

            # Close conditions:
            # 1. Timeout (5 minutes)
            # 2. Hit target (expected move realized)
            # 3. Stop loss (-5% of position)

            should_close = False
            close_reason = ""

            if age >= self.MAX_LAG_SECONDS:
                should_close = True
                close_reason = "TIMEOUT"
            elif unrealized_pnl >= self.PAPER_POSITION_SIZE * 0.03:  # 3% profit
                should_close = True
                close_reason = "TARGET"
            elif unrealized_pnl <= -self.PAPER_POSITION_SIZE * 0.05:  # 5% loss
                should_close = True
                close_reason = "STOP_LOSS"

            if should_close:
                trade.exit_price = current_price
                trade.exit_time = time.time()
                trade.pnl = unrealized_pnl
                trade.status = "CLOSED"

                self.paper_balance += unrealized_pnl
                self.stats["trades_closed"] += 1
                self.stats["total_pnl"] += unrealized_pnl

                if unrealized_pnl > 0:
                    self.stats["wins"] += 1
                else:
                    self.stats["losses"] += 1

                print(f"\n[PAPER TRADE] CLOSED ({close_reason}): {trade.signal.direction} on {trade.signal.follower_slug}")
                print(f"  Entry: ${trade.entry_price:.4f} -> Exit: ${trade.exit_price:.4f}")
                print(f"  PnL: ${unrealized_pnl:+.2f} | Balance: ${self.paper_balance:.2f}")

                self._save_state()

    def get_hourly_report(self) -> str:
        """Generate hourly performance report."""
        runtime_hours = (time.time() - self.stats["start_time"]) / 3600

        open_trades = [t for t in self.paper_trades if t.status == "OPEN"]
        closed_trades = [t for t in self.paper_trades if t.status == "CLOSED"]

        win_rate = self.stats["wins"] / max(1, self.stats["trades_closed"]) * 100
        roi = (self.paper_balance - self.paper_starting_balance) / self.paper_starting_balance * 100

        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        for trade in open_trades:
            history = self.price_history.get(trade.signal.follower_market_id, deque())
            if history:
                current = history[-1].yes_price if trade.signal.direction == "YES" else history[-1].no_price
                if trade.signal.direction == "YES":
                    unrealized_pnl += (current - trade.entry_price) / trade.entry_price * self.PAPER_POSITION_SIZE
                else:
                    unrealized_pnl += (trade.entry_price - current) / trade.entry_price * self.PAPER_POSITION_SIZE

        report = f"""
{'='*60}
CORRELATION LAG STRATEGY - HOURLY REPORT
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

PAPER TRADING PERFORMANCE
--------------------------
Starting Balance: ${self.paper_starting_balance:.2f}
Current Balance:  ${self.paper_balance:.2f}
Unrealized PnL:   ${unrealized_pnl:+.2f}
Total ROI:        {roi:+.2f}%

TRADE STATISTICS
--------------------------
Runtime:          {runtime_hours:.1f} hours
Signals Generated:{self.stats['signals_generated']}
Trades Opened:    {self.stats['trades_opened']}
Trades Closed:    {self.stats['trades_closed']}
Win Rate:         {win_rate:.1f}%
Total PnL:        ${self.stats['total_pnl']:+.2f}

OPEN POSITIONS ({len(open_trades)})
--------------------------"""

        for trade in open_trades:
            history = self.price_history.get(trade.signal.follower_market_id, deque())
            if history:
                current = history[-1].yes_price if trade.signal.direction == "YES" else history[-1].no_price
                age_min = (time.time() - trade.entry_time) / 60
                report += f"\n  {trade.signal.direction}: {trade.signal.follower_slug[:30]}"
                report += f"\n    Entry: ${trade.entry_price:.4f} | Current: ${current:.4f} | Age: {age_min:.1f}m"

        report += f"""

MARKET CORRELATIONS TRACKED
--------------------------
Markets in history: {len(self.price_history)}

{'='*60}
"""
        return report


async def fetch_crypto_markets() -> List[dict]:
    """Fetch current crypto up/down markets from whale positions."""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            crypto_markets = []
            seen_conditions = set()

            # Get markets from active whale positions (0x8dxd)
            whale_wallets = [
                "0x63ce342161250d705dc0b16df89036c8e5f9ba9a",  # 0x8dxd
                "0xd0d6053c3c37e727402d84c14069780d360993aa",  # k9Q2mX4L8A7ZP3R
                "0xdb27bf2ac5d428a9c63dbc914611036855a6c56e",  # DrPufferfish
                "0x44c1dfe43260c94ed4f1d00de2e1f80fb113ebc1",  # aenews2
                "0x9d84ce0306f8551e02efef1680475fc0f1dc1344",  # ImJustKen
                "0xd830027529b0baca2a52fd8a4dee43d366a9a592",  # LDSIADAS
                "0xdc876e6873772d38716fda7f2452a78d426d7ab6",  # 432614799197
                "0x93c22116e4402c9332ee6db578050e688934c072",  # Candid-Closure
                "0xcb3143ee858e14d0b3fe40ffeaea78416e646b02",  # 0xCB3143ee...
                "0xa58d4f278d7953cd38eeb929f7e242bfc7c0b9b8",  # AiBird
                "0xcb2952fd655813ad0f9ee893122c54298e1a975e",  # willi
                "0x79bf43cbd005a51e25bafa4abe3f51a3c8ba96a8",  # I-do-stupid-bets
            ]

            for wallet in whale_wallets:
                try:
                    r = await client.get(
                        "https://data-api.polymarket.com/positions",
                        params={"user": wallet, "sizeThreshold": 0}
                    )
                    if r.status_code == 200:
                        positions = r.json()
                        for pos in positions:
                            condition_id = pos.get("conditionId", "")
                            slug = pos.get("slug", "")

                            # Only include unique crypto up/down markets
                            if condition_id and condition_id not in seen_conditions:
                                slug_lower = slug.lower()
                                if ("bitcoin" in slug_lower or "btc" in slug_lower or
                                    "ethereum" in slug_lower or "eth" in slug_lower or
                                    "solana" in slug_lower or "sol" in slug_lower or
                                    "xrp" in slug_lower):
                                    if "up" in slug_lower or "down" in slug_lower:
                                        seen_conditions.add(condition_id)
                                        # Build market-like dict from position
                                        market = {
                                            "conditionId": condition_id,
                                            "slug": slug,
                                            "tokens": [{
                                                "outcome": pos.get("outcome", "YES"),
                                                "price": float(pos.get("curPrice", 0.5)),
                                            }],
                                        }
                                        crypto_markets.append(market)
                except Exception as e:
                    print(f"[CorrelationLag] Error fetching {wallet[:10]}: {e}", flush=True)
                    continue

            print(f"[CorrelationLag] Found {len(crypto_markets)} crypto up/down markets", flush=True)
            return crypto_markets
        except Exception as e:
            print(f"[CorrelationLag] Error fetching markets: {e}", flush=True)
            return []


async def run_correlation_lag_strategy(
    dry_run: bool = True,
    poll_interval: int = 15,
    report_interval: int = 3600,
):
    """
    Run the correlation lag arbitrage strategy.

    Args:
        dry_run: Paper trading only (no real trades)
        poll_interval: Seconds between market scans
        report_interval: Seconds between hourly reports
    """
    detector = CorrelationLagDetector()
    last_report = time.time()

    print("\n" + "="*60)
    print("CORRELATION LAG ARBITRAGE STRATEGY")
    print("Based on @arndxt_xo and @the_smart_ape research")
    print("="*60)
    print(f"Mode: {'PAPER TRADING' if dry_run else 'LIVE'}")
    print(f"Poll interval: {poll_interval}s")
    print(f"Report interval: {report_interval}s")
    print("="*60 + "\n")

    while True:
        try:
            # Fetch current markets
            markets = await fetch_crypto_markets()

            # Record prices
            for market in markets:
                condition_id = market.get("conditionId", "")
                slug = market.get("slug", "unknown")

                # Get prices from tokens
                tokens = market.get("tokens", [])
                yes_price = 0.5
                no_price = 0.5

                for token in tokens:
                    outcome = token.get("outcome", "").upper()
                    price = float(token.get("price", 0.5))
                    if outcome == "YES":
                        yes_price = price
                    elif outcome == "NO":
                        no_price = price

                if condition_id:
                    detector.record_price(condition_id, slug, yes_price, no_price)

            # Scan for signals
            signals = detector.scan_for_signals()

            for signal in signals:
                if signal.confidence >= 0.5:
                    print(f"\n[SIGNAL] Lag detected!")
                    print(f"  Leader: {signal.leader_slug} moved {signal.leader_move_pct:+.2f}%")
                    print(f"  Follower: {signal.follower_slug} (trade {signal.direction})")
                    print(f"  Correlation: {signal.correlation:.2f} | Confidence: {signal.confidence:.2f}")

                    if dry_run:
                        detector.execute_paper_trade(signal)

            # Update open trades
            detector.update_paper_trades()

            # Hourly report
            if time.time() - last_report >= report_interval:
                print(detector.get_hourly_report())
                last_report = time.time()

        except Exception as e:
            print(f"[CorrelationLag] Error in main loop: {e}")

        await asyncio.sleep(poll_interval)


if __name__ == "__main__":
    asyncio.run(run_correlation_lag_strategy(dry_run=True, poll_interval=15))
