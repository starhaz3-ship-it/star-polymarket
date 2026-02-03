"""
TA Paper Trading Runner with 10-minute Updates

Standalone script that runs TA-based paper trading for BTC 15m markets.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import httpx

# Import from arbitrage package
import sys
sys.path.insert(0, str(Path(__file__).parent))

from arbitrage.ta_signals import TASignalGenerator, Candle, SignalStrength
from arbitrage.bregman_optimizer import BregmanOptimizer, enhance_ta_signal_with_bregman


@dataclass
class TAPaperTrade:
    """A paper trade record."""
    trade_id: str
    market_title: str
    side: str
    entry_price: float
    entry_time: str
    size_usd: float = 100.0
    signal_strength: str = ""
    edge_at_entry: float = 0.0
    # Bregman optimization metrics
    kl_divergence: float = 0.0
    kelly_fraction: float = 0.0
    guaranteed_profit: float = 0.0
    # Exit
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: float = 0.0
    status: str = "open"


class TAPaperTrader:
    """Paper trades based on TA signals."""

    OUTPUT_FILE = Path(__file__).parent / "ta_paper_results.json"
    POSITION_SIZE = 100.0

    def __init__(self, bankroll: float = 1000.0):
        self.generator = TASignalGenerator()
        self.bregman = BregmanOptimizer(bankroll=bankroll)
        self.bankroll = bankroll
        self.trades: Dict[str, TAPaperTrade] = {}
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.signals_count = 0
        self.bregman_signals = 0
        self.start_time = datetime.now(timezone.utc).isoformat()
        self._load()

    def _load(self):
        if self.OUTPUT_FILE.exists():
            try:
                data = json.load(open(self.OUTPUT_FILE))
                for tid, t in data.get("trades", {}).items():
                    self.trades[tid] = TAPaperTrade(**t)
                self.total_pnl = data.get("total_pnl", 0)
                self.wins = data.get("wins", 0)
                self.losses = data.get("losses", 0)
                self.signals_count = data.get("signals_count", 0)
                self.start_time = data.get("start_time", self.start_time)
                print(f"Loaded {len(self.trades)} trades from previous session")
            except Exception as e:
                print(f"Error loading state: {e}")

    def _save(self):
        data = {
            "start_time": self.start_time,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "trades": {tid: asdict(t) for tid, t in self.trades.items()},
            "total_pnl": self.total_pnl,
            "wins": self.wins,
            "losses": self.losses,
            "signals_count": self.signals_count,
        }
        with open(self.OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    async def fetch_data(self):
        """Fetch BTC candles and price with rate limiting."""
        async with httpx.AsyncClient(timeout=15) as client:
            # Candles from Binance
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

            # Rate limit: delay before Polymarket call
            await asyncio.sleep(1.5)

            # BTC 15m markets - use EVENTS endpoint with tag_slug=15M
            # This returns all crypto 15m up/down markets, then filter for BTC
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
                        # Filter for Bitcoin markets only
                        if "bitcoin" not in title and "btc" not in title:
                            continue
                        for m in event.get("markets", []):
                            if not m.get("closed", True):
                                # Copy event title to market question if missing
                                if not m.get("question"):
                                    m["question"] = event.get("title", "")
                                markets.append(m)
                    # Cache for rate limit fallback
                    if markets:
                        self._market_cache = markets
                else:
                    print(f"[API] Status {mr.status_code}, using cache")
                    markets = getattr(self, '_market_cache', [])
            except Exception as e:
                print(f"[API] Error: {e}, using cache")
                markets = getattr(self, '_market_cache', [])

        candles = [
            Candle(k[0]/1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
            for k in klines
        ]

        btc_markets = markets  # Already filtered by series slug

        if btc_markets:
            print(f"[Markets] Found {len(btc_markets)} BTC 15m Up/Down market(s)")
        else:
            print("[Markets] No active BTC 15m markets found")

        return candles, btc_price, btc_markets

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
        """Get minutes remaining."""
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
        """Run one trading cycle."""
        candles, btc_price, markets = await self.fetch_data()

        if not candles:
            return None

        # Generate main signal (without market prices for general direction)
        main_signal = self.generator.generate_signal(
            market_id="btc_main",
            candles=candles,
            current_price=btc_price,
            market_yes_price=0.5,
            market_no_price=0.5,
            time_remaining_min=10.0
        )

        self.signals_count += 1

        # Process each market
        for market in markets:
            market_id = market.get("conditionId", "")
            question = market.get("question", "")
            up_price, down_price = self.get_market_prices(market)
            time_left = self.get_time_remaining(market)

            if up_price is None or down_price is None:
                continue  # Skip markets without Up/Down outcomes

            # Only trade markets expiring within 30 minutes (nearest markets)
            if time_left > 30:
                continue  # Skip markets too far in the future
            if time_left < 2:
                continue  # Skip markets about to expire

            # Generate signal for this market
            signal = self.generator.generate_signal(
                market_id=market_id,
                candles=candles,
                current_price=btc_price,
                market_yes_price=up_price,
                market_no_price=down_price,
                time_remaining_min=time_left
            )

            trade_key = f"{market_id}_{signal.side}" if signal.side else None

            # Debug: show why no trade
            if signal.action != "ENTER":
                edge = max(signal.edge_up or 0, signal.edge_down or 0)
                print(f"[Skip] {question[:35]}... | UP:{up_price:.2f} DOWN:{down_price:.2f} | Edge:{edge:.1%} | {signal.reason}")

            # Enter trade if signal - use Bregman optimization for sizing
            if signal.action == "ENTER" and signal.side and trade_key:
                if trade_key not in self.trades:
                    entry_price = up_price if signal.side == "UP" else down_price
                    edge = signal.edge_up if signal.side == "UP" else signal.edge_down

                    # Bregman divergence optimization
                    bregman_signal = self.bregman.calculate_optimal_trade(
                        model_prob=signal.model_up,
                        market_yes_price=up_price,
                        market_no_price=down_price
                    )
                    self.bregman_signals += 1

                    # Use Kelly fraction for position sizing (capped)
                    optimal_size = min(self.POSITION_SIZE, self.bankroll * bregman_signal.kelly_fraction)
                    optimal_size = max(10.0, optimal_size)  # Minimum $10

                    trade = TAPaperTrade(
                        trade_id=trade_key,
                        market_title=question[:80],
                        side=signal.side,
                        entry_price=entry_price,
                        entry_time=datetime.now(timezone.utc).isoformat(),
                        size_usd=optimal_size,
                        signal_strength=signal.strength.value,
                        edge_at_entry=edge if edge else 0,
                        kl_divergence=bregman_signal.kl_divergence,
                        kelly_fraction=bregman_signal.kelly_fraction,
                        guaranteed_profit=bregman_signal.guaranteed_profit,
                    )
                    self.trades[trade_key] = trade
                    print(f"[NEW] {signal.side} @ ${entry_price:.4f} | Edge: {edge:.1%} | KL: {bregman_signal.kl_divergence:.4f} | Kelly: {bregman_signal.kelly_fraction:.1%}")
                    print(f"      Size: ${optimal_size:.2f} | {question[:50]}...")

            # Check for resolved markets
            if time_left < 0.5:
                for tid, trade in list(self.trades.items()):
                    if trade.status == "open" and market_id in tid:
                        price = up_price if trade.side == "UP" else down_price

                        # Determine outcome
                        if price >= 0.95:
                            exit_val = self.POSITION_SIZE / trade.entry_price
                        elif price <= 0.05:
                            exit_val = 0
                        else:
                            exit_val = (self.POSITION_SIZE / trade.entry_price) * price

                        trade.exit_price = price
                        trade.exit_time = datetime.now(timezone.utc).isoformat()
                        trade.pnl = exit_val - self.POSITION_SIZE
                        trade.status = "closed"

                        self.total_pnl += trade.pnl
                        if trade.pnl > 0:
                            self.wins += 1
                        else:
                            self.losses += 1

                        result = "WIN" if trade.pnl > 0 else "LOSS"
                        print(f"[{result}] {trade.side} PnL: ${trade.pnl:+.2f} | {trade.market_title[:40]}...")

        self._save()
        return main_signal

    def print_update(self, signal):
        """Print 10-minute update."""
        now = datetime.now(timezone.utc)
        open_trades = [t for t in self.trades.values() if t.status == "open"]
        closed_trades = [t for t in self.trades.values() if t.status == "closed"]

        print()
        print("=" * 70)
        print(f"TA PAPER TRADING UPDATE - {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 70)

        # Signal info
        if signal:
            print(f"\nCURRENT TA SIGNAL:")
            print(f"  BTC: ${signal.current_price:,.2f}")
            print(f"  VWAP: ${signal.vwap:,.2f}" if signal.vwap else "  VWAP: N/A")
            print(f"  RSI: {signal.rsi:.1f}" if signal.rsi else "  RSI: N/A")
            print(f"  Heiken: {signal.heiken_color} x{signal.heiken_count}")
            print(f"  Regime: {signal.regime.value}")
            print(f"  Model: {signal.model_up:.1%} UP / {signal.model_down:.1%} DOWN")

        # Stats
        print(f"\nTRADING STATS:")
        print(f"  Running since: {self.start_time[:19]}")
        print(f"  TA Signals: {self.signals_count} | Bregman Signals: {self.bregman_signals}")
        print(f"  Total trades: {len(closed_trades)}")
        print(f"  Win/Loss: {self.wins}/{self.losses}")
        win_rate = (self.wins / max(1, self.wins + self.losses)) * 100
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Total PnL: ${self.total_pnl:+.2f}")

        # Bregman stats
        if closed_trades:
            avg_kl = sum(t.kl_divergence for t in closed_trades) / len(closed_trades)
            avg_kelly = sum(t.kelly_fraction for t in closed_trades) / len(closed_trades)
            print(f"\nBREGMAN OPTIMIZATION:")
            print(f"  Avg KL Divergence: {avg_kl:.4f}")
            print(f"  Avg Kelly Fraction: {avg_kelly:.1%}")

        # Open trades
        print(f"\nOPEN TRADES ({len(open_trades)}):")
        for t in open_trades[:5]:
            print(f"  {t.side} @ ${t.entry_price:.4f} | KL: {t.kl_divergence:.4f} | {t.market_title[:35]}...")

        # Recent closed
        print(f"\nRECENT CLOSED:")
        recent = sorted(closed_trades, key=lambda x: x.exit_time or "", reverse=True)[:5]
        for t in recent:
            result = "WIN" if t.pnl > 0 else "LOSS"
            print(f"  [{result}] {t.side} ${t.pnl:+.2f} | {t.market_title[:35]}...")

        print("=" * 70)
        print()

    async def run(self):
        """Main loop with 10-minute updates."""
        print("=" * 70)
        print("TA + BREGMAN PAPER TRADER - BTC 15-Minute Markets")
        print("=" * 70)
        print("Strategy: PolymarketBTC15mAssistant + Bregman Divergence")
        print("Optimization: Kelly criterion with KL divergence edge")
        print(f"Bankroll: ${self.bankroll} | Max Position: ${self.POSITION_SIZE}")
        print("Scan interval: 2 minutes | Update interval: 10 minutes")
        print("=" * 70)
        print()

        last_update = 0
        cycle = 0

        while True:
            try:
                cycle += 1
                now = time.time()

                signal = await self.run_cycle()

                if signal:
                    print(f"[Scan {cycle}] BTC ${signal.current_price:,.2f} | {signal.regime.value} | Model: {signal.model_up:.0%} UP")

                # 10-minute update
                if now - last_update >= 600:
                    self.print_update(signal)
                    last_update = now

                await asyncio.sleep(120)  # 2 minutes between scans to avoid rate limiting

            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

        self.print_update(None)
        print(f"Results saved to: {self.OUTPUT_FILE}")


if __name__ == "__main__":
    trader = TAPaperTrader()
    asyncio.run(trader.run())
