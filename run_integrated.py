"""
Integrated Strategy Runner

Combines ALL strategies into one unified system:
- TA Signals (RSI, VWAP, MACD, Heiken Ashi)
- Bregman Divergence Optimization
- Kalshi Cross-Platform Arbitrage
- Spike Detection
- Flash Crash Detection
- Momentum Indicators
- ML Optimization

This is the ultimate BTC 15m trading system.
"""

import asyncio
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
from functools import partial

import httpx
from dotenv import load_dotenv

# Force unbuffered output
print = partial(print, flush=True)

# Add path
sys.path.insert(0, str(Path(__file__).parent))

from arbitrage.ta_signals import TASignalGenerator, Candle, SignalStrength
from arbitrage.bregman_optimizer import BregmanOptimizer
from arbitrage.integrated_strategies import (
    IntegratedStrategyManager,
    IntegratedSignal,
    SignalType,
)

load_dotenv()


@dataclass
class UnifiedTrade:
    """Trade record with all signal sources."""
    trade_id: str
    market_title: str
    side: str
    entry_price: float
    entry_time: str
    size_usd: float
    # Signal sources
    ta_signal: str = ""
    ta_strength: str = ""
    bregman_kl: float = 0.0
    bregman_kelly: float = 0.0
    integrated_signals: List[str] = field(default_factory=list)
    consensus_score: float = 0.0
    # Results
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: float = 0.0
    status: str = "open"


class UnifiedTrader:
    """
    Ultimate unified trading system combining all strategies.
    """

    OUTPUT_FILE = Path(__file__).parent / "unified_results.json"
    POSITION_SIZE = 25.0  # $25 per trade

    def __init__(self, bankroll: float = 1000.0, dry_run: bool = True):
        self.dry_run = dry_run
        self.bankroll = bankroll

        # Core components
        self.ta_generator = TASignalGenerator()
        self.bregman = BregmanOptimizer(bankroll=bankroll)
        self.integrated = IntegratedStrategyManager()

        # State
        self.trades: Dict[str, UnifiedTrade] = {}
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.signals_count = 0
        self.start_time = datetime.now(timezone.utc).isoformat()

        # Signal counts by type
        self.signal_counts = {
            "ta": 0,
            "kalshi_arb": 0,
            "spike": 0,
            "flash_crash": 0,
            "momentum": 0,
        }

        self._load()

    def _load(self):
        """Load previous state."""
        if self.OUTPUT_FILE.exists():
            try:
                data = json.load(open(self.OUTPUT_FILE))
                for tid, t in data.get("trades", {}).items():
                    signals = t.pop("integrated_signals", [])
                    self.trades[tid] = UnifiedTrade(**t, integrated_signals=signals)
                self.total_pnl = data.get("total_pnl", 0)
                self.wins = data.get("wins", 0)
                self.losses = data.get("losses", 0)
                print(f"[Unified] Loaded {len(self.trades)} trades")
            except Exception as e:
                print(f"[Unified] Load error: {e}")

    def _save(self):
        """Save state."""
        data = {
            "start_time": self.start_time,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "dry_run": self.dry_run,
            "trades": {tid: asdict(t) for tid, t in self.trades.items()},
            "total_pnl": self.total_pnl,
            "wins": self.wins,
            "losses": self.losses,
            "signals_count": self.signals_count,
            "signal_counts": self.signal_counts,
        }
        with open(self.OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    async def fetch_data(self):
        """Fetch BTC candles and Polymarket markets."""
        async with httpx.AsyncClient(timeout=15) as client:
            # Binance candles
            r = await client.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "1m", "limit": 240}
            )
            klines = r.json()

            # Current price
            pr = await client.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": "BTCUSDT"}
            )
            btc_price = float(pr.json()["price"])

            await asyncio.sleep(1.5)  # Rate limit

            # Polymarket BTC 15m markets
            markets = []
            try:
                mr = await client.get(
                    "https://gamma-api.polymarket.com/events",
                    params={"tag_slug": "15M", "active": "true", "closed": "false", "limit": 50},
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                if mr.status_code == 200:
                    events = mr.json()
                    for event in events:
                        title = event.get("title", "").lower()
                        if "bitcoin" not in title and "btc" not in title:
                            continue
                        for m in event.get("markets", []):
                            if not m.get("closed", True):
                                if not m.get("question"):
                                    m["question"] = event.get("title", "")
                                markets.append(m)
            except Exception as e:
                print(f"[API] Error: {e}")

        candles = [
            Candle(k[0]/1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
            for k in klines
        ]

        return candles, btc_price, markets

    def get_market_prices(self, market: Dict) -> tuple:
        """Extract UP/DOWN prices."""
        outcomes = market.get("outcomes", [])
        prices = market.get("outcomePrices", [])

        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(prices, str):
            prices = json.loads(prices)

        up_price, down_price = None, None
        for i, o in enumerate(outcomes):
            if i < len(prices):
                p = float(prices[i])
                if str(o).lower() == "up":
                    up_price = p
                elif str(o).lower() == "down":
                    down_price = p

        return up_price, down_price

    def get_time_remaining(self, market: Dict) -> float:
        """Get minutes remaining until market expires."""
        end = market.get("endDate")
        if not end:
            return 15.0
        try:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return max(0, (end_dt - now).total_seconds() / 60)
        except:
            return 15.0

    async def run_cycle(self):
        """Run one trading cycle with all strategies."""
        candles, btc_price, markets = await self.fetch_data()

        if not candles:
            return None

        # Feed price to integrated strategies
        self.integrated.add_price_data(
            btc_price=btc_price,
            volume=candles[-1].volume if candles else 0
        )

        # Generate main TA signal
        main_signal = self.ta_generator.generate_signal(
            market_id="btc_main",
            candles=candles,
            current_price=btc_price,
            market_yes_price=0.5,
            market_no_price=0.5,
            time_remaining_min=10.0
        )
        self.signals_count += 1
        self.signal_counts["ta"] += 1

        print(f"[Markets] Found {len(markets)} BTC 15m market(s)")

        # Process each market
        for market in markets:
            market_id = market.get("conditionId", "")
            question = market.get("question", "")
            up_price, down_price = self.get_market_prices(market)
            time_left = self.get_time_remaining(market)

            if up_price is None or down_price is None:
                continue

            # Only trade markets expiring within 15 minutes (high turnover)
            if time_left > 15:
                continue  # Skip markets too far in the future
            if time_left < 2:
                continue  # Skip markets about to expire

            # TA Signal for this market
            signal = self.ta_generator.generate_signal(
                market_id=market_id,
                candles=candles,
                current_price=btc_price,
                market_yes_price=up_price,
                market_no_price=down_price,
                time_remaining_min=time_left
            )

            # Bregman optimization
            bregman_signal = self.bregman.calculate_optimal_trade(
                model_prob=signal.model_up,
                market_yes_price=up_price,
                market_no_price=down_price
            )

            # Integrated strategies scan
            integrated_signals = await self.integrated.scan_all(
                poly_strike=btc_price,
                poly_up_price=up_price,
                poly_down_price=down_price,
                market_id=market_id
            )

            # Count signal types
            for isig in integrated_signals:
                self.signal_counts[isig.signal_type.value] = \
                    self.signal_counts.get(isig.signal_type.value, 0) + 1

            # Get consensus
            consensus = self.integrated.get_consensus()

            # Decision logic: combine all signals
            should_trade = False
            trade_side = None
            confidence = 0

            # Priority 1: Kalshi arbitrage (risk-free)
            kalshi_arb = [s for s in integrated_signals if s.signal_type == SignalType.KALSHI_ARB]
            if kalshi_arb:
                should_trade = True
                trade_side = kalshi_arb[0].side
                confidence = 1.0
                print(f"[ARB] Kalshi arbitrage found: {trade_side} | Profit: {kalshi_arb[0].edge:.2%}")

            # Priority 2: Strong TA + Bregman agreement
            elif signal.action == "ENTER" and bregman_signal.kl_divergence > 0.2:
                should_trade = True
                trade_side = signal.side
                confidence = bregman_signal.kelly_fraction

            # Priority 3: Consensus from integrated strategies
            elif consensus and consensus[1] > 0.3:
                should_trade = True
                trade_side = consensus[0]
                confidence = min(consensus[1], 1.0)

            # Execute trade
            trade_key = f"{market_id}_{trade_side}" if trade_side else None

            if should_trade and trade_side and trade_key:
                if trade_key not in self.trades:
                    entry_price = up_price if trade_side == "UP" else down_price
                    edge = signal.edge_up if trade_side == "UP" else signal.edge_down

                    trade = UnifiedTrade(
                        trade_id=trade_key,
                        market_title=question[:80],
                        side=trade_side,
                        entry_price=entry_price,
                        entry_time=datetime.now(timezone.utc).isoformat(),
                        size_usd=min(self.POSITION_SIZE * confidence, self.POSITION_SIZE),
                        ta_signal=signal.action,
                        ta_strength=signal.strength.value,
                        bregman_kl=bregman_signal.kl_divergence,
                        bregman_kelly=bregman_signal.kelly_fraction,
                        integrated_signals=[s.signal_type.value for s in integrated_signals],
                        consensus_score=confidence,
                    )
                    self.trades[trade_key] = trade

                    mode = "LIVE" if not self.dry_run else "DRY"
                    sig_sources = ", ".join(trade.integrated_signals) if trade.integrated_signals else "TA"
                    print(f"[{mode}] {trade_side} @ ${entry_price:.4f} | Edge: {edge:.1%} | Sources: {sig_sources}")

        self._save()
        return main_signal

    def print_update(self, signal):
        """Print status update."""
        now = datetime.now(timezone.utc)
        open_trades = [t for t in self.trades.values() if t.status == "open"]
        closed_trades = [t for t in self.trades.values() if t.status == "closed"]

        print()
        print("=" * 70)
        mode = "LIVE" if not self.dry_run else "DRY RUN"
        print(f"UNIFIED STRATEGY UPDATE ({mode}) - {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 70)

        if signal:
            print(f"\nCURRENT SIGNAL:")
            print(f"  BTC: ${signal.current_price:,.2f}")
            print(f"  Model: {signal.model_up:.1%} UP / {signal.model_down:.1%} DOWN")

        print(f"\nSIGNAL SOURCES:")
        for src, count in self.signal_counts.items():
            print(f"  {src}: {count}")

        print(f"\nTRADING STATS:")
        print(f"  Position Size: ${self.POSITION_SIZE}")
        print(f"  Total trades: {len(closed_trades)}")
        print(f"  Win/Loss: {self.wins}/{self.losses}")
        win_rate = (self.wins / max(1, self.wins + self.losses)) * 100
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Total PnL: ${self.total_pnl:+.2f}")

        print(f"\nOPEN TRADES ({len(open_trades)}):")
        for t in open_trades[:5]:
            sources = ", ".join(t.integrated_signals[:2]) if t.integrated_signals else "TA"
            print(f"  {t.side} @ ${t.entry_price:.4f} | {sources} | {t.market_title[:30]}...")

        print("=" * 70)

    async def run(self):
        """Main loop."""
        print("=" * 70)
        mode = "LIVE" if not self.dry_run else "DRY RUN"
        print(f"UNIFIED STRATEGY TRADER ({mode}) - All Signals Combined")
        print("=" * 70)
        print("Strategies: TA + Bregman + Kalshi Arb + Spike + Flash Crash + Momentum")
        print(f"Position Size: ${self.POSITION_SIZE}")
        print("Scan interval: 2 minutes")
        print("=" * 70)

        last_update = 0
        cycle = 0

        while True:
            try:
                cycle += 1
                now = time.time()

                signal = await self.run_cycle()

                if signal:
                    print(f"[Scan {cycle}] BTC ${signal.current_price:,.2f} | {signal.regime.value} | Model: {signal.model_up:.0%} UP")

                if now - last_update >= 600:
                    self.print_update(signal)
                    last_update = now

                await asyncio.sleep(120)

            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(60)

        self.print_update(None)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    args = parser.parse_args()

    trader = UnifiedTrader(dry_run=not args.live)
    asyncio.run(trader.run())
