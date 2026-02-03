"""
TA-Based Paper Trader for BTC 15-Minute Markets

Paper trades based on technical analysis signals from ta_signals.py.
Updates and reports every 10 minutes.

Based on PolymarketBTC15mAssistant strategy analysis.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import httpx

from .ta_signals import (
    TASignal, TASignalGenerator, Candle, SignalStrength,
    generate_btc_15m_signal, ta_signal_generator
)
from .config import config


@dataclass
class PaperTrade:
    """A paper trade record."""
    trade_id: str
    market_id: str
    market_title: str
    side: str  # "UP" or "DOWN"
    entry_price: float
    entry_time: str
    size_usd: float
    signal_strength: str
    edge_at_entry: float
    model_prob_at_entry: float
    # Exit info
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = "open"  # open, closed, expired


@dataclass
class TradingStats:
    """Cumulative trading statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    avg_entry_edge: float = 0.0
    strong_signals: int = 0
    good_signals: int = 0
    optional_signals: int = 0


class BinanceCandleFetcher:
    """Fetches candle data from Binance."""

    async def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        limit: int = 240
    ) -> List[Candle]:
        """Fetch klines from Binance."""
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url, params=params)
                data = r.json()

                candles = []
                for k in data:
                    candles.append(Candle(
                        timestamp=k[0] / 1000,
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5])
                    ))

                return candles
        except Exception as e:
            print(f"[Binance] Error fetching klines: {e}")
            return []

    async def get_current_price(self) -> Optional[float]:
        """Get current BTC price."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    "https://api.binance.com/api/v3/ticker/price",
                    params={"symbol": "BTCUSDT"}
                )
                return float(r.json()["price"])
        except:
            return None


class PolymarketBTC15mFetcher:
    """Fetches BTC 15-minute markets from Polymarket."""

    SERIES_ID = "btc-15m"  # BTC 15-minute markets series

    async def fetch_live_markets(self) -> List[Dict]:
        """Fetch active BTC 15-minute markets."""
        markets = []

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # Search for BTC up/down markets
                r = await client.get(
                    f"{config.GAMMA_API_URL}/markets",
                    params={
                        "active": "true",
                        "closed": "false",
                        "limit": 50,
                    }
                )

                all_markets = r.json()

                # Filter for BTC 15-minute markets
                for m in all_markets:
                    question = m.get("question", "").lower()
                    if ("bitcoin" in question or "btc" in question) and "up or down" in question:
                        if "15" in question or "15m" in question or ":00" in question or ":15" in question or ":30" in question or ":45" in question:
                            markets.append(m)

        except Exception as e:
            print(f"[Polymarket] Error fetching markets: {e}")

        return markets

    def extract_market_prices(self, market: Dict) -> tuple:
        """Extract UP/DOWN prices from market."""
        outcomes = market.get("outcomes", [])
        prices = market.get("outcomePrices", [])

        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(prices, str):
            prices = json.loads(prices)

        up_price = None
        down_price = None

        for i, outcome in enumerate(outcomes):
            outcome_lower = str(outcome).lower()
            if i < len(prices):
                price = float(prices[i])
                if outcome_lower == "up":
                    up_price = price
                elif outcome_lower == "down":
                    down_price = price

        return (up_price, down_price)

    def get_time_remaining(self, market: Dict) -> float:
        """Get minutes remaining until market resolution."""
        end_date = market.get("endDate")
        if not end_date:
            return 15.0

        try:
            end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            remaining_sec = (end - now).total_seconds()
            return max(0, remaining_sec / 60)
        except:
            return 15.0


class TAPaperTrader:
    """Paper trades BTC 15-minute markets using TA signals."""

    OUTPUT_FILE = Path(__file__).parent.parent / "ta_paper_trades.json"
    UPDATE_INTERVAL = 600  # 10 minutes
    POSITION_SIZE = 100  # $100 per trade

    def __init__(self):
        self.binance = BinanceCandleFetcher()
        self.polymarket = PolymarketBTC15mFetcher()
        self.trades: Dict[str, PaperTrade] = {}
        self.stats = TradingStats()
        self.signals_generated = 0
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.last_update = time.time()

        self._load_state()

    def _load_state(self):
        """Load previous state if exists."""
        if self.OUTPUT_FILE.exists():
            try:
                with open(self.OUTPUT_FILE, 'r') as f:
                    data = json.load(f)
                    for tid, trade_data in data.get("trades", {}).items():
                        self.trades[tid] = PaperTrade(**trade_data)
                    stats_data = data.get("stats", {})
                    self.stats = TradingStats(**stats_data) if stats_data else TradingStats()
                    self.start_time = data.get("start_time", self.start_time)
                print(f"[TAPaper] Loaded {len(self.trades)} trades from previous session")
            except Exception as e:
                print(f"[TAPaper] Error loading state: {e}")

    def _save_state(self):
        """Save current state."""
        data = {
            "start_time": self.start_time,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "trades": {tid: asdict(t) for tid, t in self.trades.items()},
            "stats": asdict(self.stats),
            "signals_generated": self.signals_generated,
        }

        with open(self.OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def _update_stats(self):
        """Recalculate statistics."""
        closed = [t for t in self.trades.values() if t.status == "closed"]

        self.stats.total_trades = len(closed)
        self.stats.winning_trades = sum(1 for t in closed if t.pnl > 0)
        self.stats.losing_trades = sum(1 for t in closed if t.pnl < 0)
        self.stats.total_pnl = sum(t.pnl for t in closed)

        if closed:
            self.stats.best_trade_pnl = max(t.pnl for t in closed)
            self.stats.worst_trade_pnl = min(t.pnl for t in closed)
            self.stats.avg_entry_edge = sum(t.edge_at_entry for t in closed) / len(closed)

        # Count signal strengths
        self.stats.strong_signals = sum(1 for t in self.trades.values()
                                        if t.signal_strength == "strong")
        self.stats.good_signals = sum(1 for t in self.trades.values()
                                      if t.signal_strength == "good")
        self.stats.optional_signals = sum(1 for t in self.trades.values()
                                          if t.signal_strength == "optional")

    async def process_market(self, market: Dict, candles: List[Candle], btc_price: float) -> Optional[TASignal]:
        """Process a single market and generate/update trades."""
        market_id = market.get("conditionId", "")
        question = market.get("question", "")

        up_price, down_price = self.polymarket.extract_market_prices(market)
        time_remaining = self.polymarket.get_time_remaining(market)

        if up_price is None or down_price is None:
            return None

        # Generate TA signal
        signal = ta_signal_generator.generate_signal(
            market_id=market_id,
            candles=candles,
            current_price=btc_price,
            market_yes_price=up_price,
            market_no_price=down_price,
            time_remaining_min=time_remaining
        )

        self.signals_generated += 1

        # Check for existing trade
        trade_key = f"{market_id}_{signal.side}" if signal.side else None

        # Enter new trade if signal is ENTER
        if signal.action == "ENTER" and signal.side and trade_key:
            if trade_key not in self.trades:
                entry_price = up_price if signal.side == "UP" else down_price
                best_edge = signal.edge_up if signal.side == "UP" else signal.edge_down

                trade = PaperTrade(
                    trade_id=trade_key,
                    market_id=market_id,
                    market_title=question[:80],
                    side=signal.side,
                    entry_price=entry_price,
                    entry_time=datetime.now(timezone.utc).isoformat(),
                    size_usd=self.POSITION_SIZE,
                    signal_strength=signal.strength.value,
                    edge_at_entry=best_edge if best_edge else 0,
                    model_prob_at_entry=signal.model_up if signal.side == "UP" else signal.model_down,
                )

                self.trades[trade_key] = trade
                print(f"[NEW TRADE] {signal.side} on {question[:50]}...")
                print(f"  Entry: ${entry_price:.4f} | Edge: {best_edge:.1%} | Strength: {signal.strength.value}")

        # Update open trades for this market
        for tid, trade in list(self.trades.items()):
            if trade.market_id == market_id and trade.status == "open":
                # Check if market is resolved (time remaining near zero)
                if time_remaining < 0.5:
                    # Simulate resolution
                    current_price = up_price if trade.side == "UP" else down_price

                    # Calculate PnL
                    shares = trade.size_usd / trade.entry_price
                    if current_price >= 0.95:  # Won
                        exit_value = shares * 1.0  # $1 per share
                    elif current_price <= 0.05:  # Lost
                        exit_value = 0
                    else:
                        exit_value = shares * current_price

                    trade.exit_price = current_price
                    trade.exit_time = datetime.now(timezone.utc).isoformat()
                    trade.pnl = exit_value - trade.size_usd
                    trade.pnl_pct = (trade.pnl / trade.size_usd) * 100
                    trade.status = "closed"

                    result = "WIN" if trade.pnl > 0 else "LOSS"
                    print(f"[{result}] {trade.side} {trade.market_title[:40]}...")
                    print(f"  PnL: ${trade.pnl:+.2f} ({trade.pnl_pct:+.1f}%)")

        return signal

    def print_update(self):
        """Print 10-minute update."""
        self._update_stats()

        now = datetime.now(timezone.utc)
        open_trades = [t for t in self.trades.values() if t.status == "open"]
        closed_trades = [t for t in self.trades.values() if t.status == "closed"]

        print()
        print("=" * 70)
        print(f"TA PAPER TRADING UPDATE - {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 70)

        # Overall stats
        print(f"\nRunning since: {self.start_time}")
        print(f"Signals generated: {self.signals_generated}")
        print(f"Total trades: {self.stats.total_trades}")
        print(f"Win/Loss: {self.stats.winning_trades}/{self.stats.losing_trades}")

        win_rate = (self.stats.winning_trades / max(1, self.stats.total_trades)) * 100
        print(f"Win rate: {win_rate:.1f}%")
        print(f"Total PnL: ${self.stats.total_pnl:+.2f}")

        if self.stats.total_trades > 0:
            print(f"Best trade: ${self.stats.best_trade_pnl:+.2f}")
            print(f"Worst trade: ${self.stats.worst_trade_pnl:+.2f}")
            print(f"Avg entry edge: {self.stats.avg_entry_edge:.1%}")

        # Signal breakdown
        print(f"\nSignal Breakdown:")
        print(f"  STRONG: {self.stats.strong_signals}")
        print(f"  GOOD: {self.stats.good_signals}")
        print(f"  OPTIONAL: {self.stats.optional_signals}")

        # Open trades
        print(f"\nOpen Trades ({len(open_trades)}):")
        if open_trades:
            for t in open_trades[:5]:
                print(f"  {t.side} {t.market_title[:40]}...")
                print(f"    Entry: ${t.entry_price:.4f} | Edge: {t.edge_at_entry:.1%}")
        else:
            print("  No open trades")

        # Recent closed trades
        print(f"\nRecent Closed Trades:")
        recent = sorted(closed_trades, key=lambda x: x.exit_time or "", reverse=True)[:5]
        if recent:
            for t in recent:
                result = "WIN" if t.pnl > 0 else "LOSS"
                print(f"  [{result}] {t.side} ${t.pnl:+.2f} ({t.pnl_pct:+.1f}%) - {t.market_title[:30]}...")
        else:
            print("  No closed trades yet")

        print("=" * 70)
        print()

    async def run(self):
        """Main trading loop."""
        print("=" * 70)
        print("TA PAPER TRADER - BTC 15-Minute Markets")
        print("=" * 70)
        print(f"Strategy: PolymarketBTC15mAssistant (edge-based TA)")
        print(f"Position size: ${self.POSITION_SIZE}")
        print(f"Update interval: {self.UPDATE_INTERVAL}s (10 minutes)")
        print(f"Output file: {self.OUTPUT_FILE}")
        print("=" * 70)
        print()

        iteration = 0

        while True:
            try:
                iteration += 1
                now = time.time()

                # Fetch data
                candles = await self.binance.fetch_klines(limit=240)
                btc_price = await self.binance.get_current_price()
                markets = await self.polymarket.fetch_live_markets()

                if not candles or not btc_price:
                    print(f"[Scan {iteration}] No data available, retrying...")
                    await asyncio.sleep(30)
                    continue

                print(f"[Scan {iteration}] BTC: ${btc_price:,.2f} | Markets: {len(markets)}")

                # Process each market
                for market in markets:
                    try:
                        signal = await self.process_market(market, candles, btc_price)

                        if signal and signal.action == "ENTER":
                            print(f"  Signal: {signal.side} | Edge: UP {signal.edge_up:+.1%} DOWN {signal.edge_down:+.1%}")
                            print(f"  Model: {signal.model_up:.1%} UP | Regime: {signal.regime.value}")

                    except Exception as e:
                        pass  # Skip individual market errors

                # Save state
                self._save_state()

                # Print update every 10 minutes
                if now - self.last_update >= self.UPDATE_INTERVAL:
                    self.print_update()
                    self.last_update = now

                await asyncio.sleep(60)  # Scan every minute

            except KeyboardInterrupt:
                print("\n[TAPaper] Stopping...")
                break
            except Exception as e:
                print(f"[TAPaper] Error: {e}")
                await asyncio.sleep(30)

        # Final update
        self.print_update()
        self._save_state()
        print(f"Results saved to: {self.OUTPUT_FILE}")


async def main():
    """Entry point."""
    trader = TAPaperTrader()
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
