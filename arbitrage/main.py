"""
Main arbitrage bot - 15-minute BTC markets.

Usage:
    python -m arbitrage.main [--dry-run] [--verbose]
"""

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime

from .config import config
from .spot_feed import BinanceSpotFeed, PriceUpdate
from .polymarket_feed import PolymarketFeed, BTCMarket
from .detector import ArbitrageDetector, ArbitrageSignal
from .executor import Executor, OrderStatus
from .risk_manager import RiskManager
from .data_collector import collector, create_signal_from_arb
from .ml_optimizer import optimizer
from .kalshi_feed import KalshiFeed


class ArbitrageBot:
    """15-minute BTC arbitrage bot."""

    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose

        # Components
        self.spot_feed = BinanceSpotFeed(on_price=self._on_spot_price)
        self.poly_feed = PolymarketFeed()
        self.kalshi_feed = KalshiFeed(use_demo=False)
        self.detector = ArbitrageDetector()
        self.executor = Executor() if not dry_run else None
        self.risk_manager = RiskManager()

        # ML components
        self.use_ml = True  # Enable ML optimization

        # State
        self.current_spot_price: float = 0
        self.last_scan_time: float = 0
        self.running = False

        # Stats
        self.opportunities_found = 0
        self.trades_attempted = 0
        self.negrisk_found = 0
        self.endgame_found = 0
        self.cross_platform_found = 0
        self.ml_rejections = 0

    def _on_spot_price(self, update: PriceUpdate):
        """Handle spot price updates."""
        self.current_spot_price = update.price

        if self.verbose:
            print(f"\r[Spot] BTC: ${update.price:,.2f}", end="", flush=True)

    def _log(self, msg: str):
        """Log a message with timestamp."""
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}")

    async def _scan_loop(self):
        """Main scanning loop."""
        self._log("Starting scan loop...")

        while self.running:
            try:
                # Check if we can trade
                can_trade, reason = self.risk_manager.can_trade()
                if not can_trade:
                    if self.verbose:
                        self._log(f"Trading paused: {reason}")
                    await asyncio.sleep(60)
                    continue

                # Get current spot price
                if self.current_spot_price == 0:
                    self.current_spot_price = self.spot_feed.get_price_sync() or 0
                    if self.current_spot_price == 0:
                        self._log("Waiting for spot price...")
                        await asyncio.sleep(1)
                        continue

                # Fetch active 15-min markets
                markets = self.poly_feed.fetch_btc_markets()

                if self.verbose:
                    self._log(f"Scanning {len(markets)} markets | BTC: ${self.current_spot_price:,.2f}")

                # Update market prices
                for market in markets:
                    self.poly_feed.update_market_prices(market)

                # Scan for BTC market opportunities
                signals = self.detector.scan_all(markets, self.current_spot_price)

                # Also scan multi-outcome markets for NegRisk/Endgame
                multi_markets = self.poly_feed.fetch_multi_outcome_markets()
                if multi_markets:
                    if self.verbose:
                        self._log(f"Scanning {len(multi_markets)} multi-outcome markets...")

                    # Update prices for top markets by volume
                    for mm in multi_markets[:20]:
                        self.poly_feed.update_multi_market_prices(mm)

                    multi_signals = self.detector.scan_multi_outcome(multi_markets[:20])

                    # Track NegRisk/Endgame separately
                    for s in multi_signals:
                        if s.signal_type.value == "negrisk":
                            self.negrisk_found += 1
                        elif s.signal_type.value == "endgame":
                            self.endgame_found += 1

                    signals.extend(multi_signals)

                # Scan for cross-platform arbitrage (Kalshi)
                try:
                    cross_opps = self.kalshi_feed.scan_for_arbitrage(markets)
                    if cross_opps:
                        self.cross_platform_found += len(cross_opps)
                        for opp in cross_opps:
                            from .kalshi_feed import create_cross_platform_signal
                            signals.append(create_cross_platform_signal(opp))
                except Exception as e:
                    if self.verbose:
                        self._log(f"Kalshi scan error: {e}")

                if signals:
                    self.opportunities_found += len(signals)
                    self._log(f"Found {len(signals)} opportunities!")

                    for signal in signals:
                        await self._process_signal(signal)

                self.last_scan_time = time.time()

                # Wait before next scan
                await asyncio.sleep(config.POLL_INTERVAL_MS / 1000)

            except Exception as e:
                self._log(f"Scan error: {e}")
                await asyncio.sleep(5)

    async def _process_signal(self, signal: ArbitrageSignal):
        """Process a trading signal."""
        self._log(str(signal))

        # Handle different market types
        if signal.signal_type.value == "negrisk":
            self._log(f"  Event: {signal.market.question[:55]}...")
            self._log(f"  Outcomes: {signal.market.outcome_count} | "
                     f"Total Ask: ${signal.market.total_ask:.3f} | "
                     f"Gap: {signal.market.negrisk_gap:.4f}")
        elif signal.signal_type.value == "endgame":
            top = signal.market.highest_prob_outcome
            self._log(f"  Event: {signal.market.question[:55]}...")
            self._log(f"  Top Outcome: {top.get('name', '?')[:30]} @ {top.get('price', 0):.1%}")
            self._log(f"  Annualized Return: {signal.spot_price:.0f}%")
        else:
            self._log(f"  Market: {signal.market.question[:60]}...")

        # Check with risk manager
        should_trade, reason = self.risk_manager.should_take_signal(signal)

        if not should_trade:
            self._log(f"  Skipped: {reason}")
            return

        # Check with ML optimizer
        if self.use_ml:
            ml_approved, ml_reason = optimizer.should_take_signal(signal)
            if not ml_approved:
                self.ml_rejections += 1
                if self.verbose:
                    self._log(f"  ML Rejected: {ml_reason}")
                return

        # Record signal for ML training
        trade_signal = create_signal_from_arb(signal, self.current_spot_price)
        collector.record_signal(trade_signal)

        # Calculate position size (with ML adjustment)
        base_size = self.risk_manager.calculate_position_size(signal)
        if self.use_ml:
            size = optimizer.adjust_position_size(signal, base_size)
        else:
            size = base_size
        signal.recommended_size = size

        self._log(f"  Size: ${size:.2f} | Confidence: {signal.confidence:.2f}")

        if self.dry_run:
            self._log("  [DRY RUN] Would execute trade")
            return

        # Execute the trade
        self.trades_attempted += 1
        result = self.executor.execute_signal(signal)

        if result.success:
            self._log(f"  EXECUTED: {result.status.value} | "
                     f"Filled: {result.filled_size:.2f} @ ${result.filled_price:.4f}")
            self.risk_manager.record_trade(signal, result)
        else:
            self._log(f"  FAILED: {result.error_message}")

    async def _monitor_positions(self):
        """Monitor open positions for settlement."""
        while self.running:
            try:
                for market_id, position in list(self.risk_manager.positions.items()):
                    # Check if market has settled
                    if market_id in self.poly_feed.markets:
                        market = self.poly_feed.markets[market_id]

                        if not market.is_active:
                            # Market has closed - check settlement
                            # In production, you'd query the settlement result
                            self._log(f"Market settled: {position.market_question[:40]}...")

                await asyncio.sleep(10)

            except Exception as e:
                self._log(f"Monitor error: {e}")
                await asyncio.sleep(30)

    async def _status_loop(self):
        """Print periodic status updates."""
        while self.running:
            await asyncio.sleep(60)

            if self.verbose:
                self.risk_manager.print_status()
                self._log(f"Stats: {self.opportunities_found} opps | "
                         f"NegRisk: {self.negrisk_found} | "
                         f"Endgame: {self.endgame_found} | "
                         f"Cross-Platform: {self.cross_platform_found} | "
                         f"ML Rejections: {self.ml_rejections}")

                # Print ML stats
                ml_stats = collector.get_stats()
                self._log(f"ML Data: {ml_stats['total_signals']} signals | "
                         f"{ml_stats['training_samples']} training samples")

            # Periodically run ML optimization
            if self.use_ml and self.opportunities_found > 0 and self.opportunities_found % 50 == 0:
                optimizer.optimize(min_samples=10)

    async def run(self):
        """Run the bot."""
        self.running = True

        self._log("=" * 60)
        self._log("STAR POLYMARKET - ML-OPTIMIZED ARBITRAGE BOT")
        self._log("=" * 60)
        self._log(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        self._log(f"Strategies: BTC Latency | NegRisk | Endgame | Cross-Platform")
        self._log(f"ML Optimization: {'ENABLED' if self.use_ml else 'DISABLED'}")
        self._log(f"Min Edge: {config.MIN_EDGE_PERCENT}% (NegRisk: {config.MIN_NEGRISK_EDGE}%)")
        self._log(f"Max Position: ${config.MAX_POSITION_SIZE}")
        self._log(f"Max Daily Loss: ${config.MAX_DAILY_LOSS}")
        self._log("=" * 60)

        # Initial spot price
        self.current_spot_price = self.spot_feed.get_price_sync() or 0
        self._log(f"Initial BTC price: ${self.current_spot_price:,.2f}")

        # Start all tasks
        tasks = [
            asyncio.create_task(self.spot_feed.start()),
            asyncio.create_task(self._scan_loop()),
            asyncio.create_task(self._monitor_positions()),
            asyncio.create_task(self._status_loop()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            self._log("Bot shutting down...")
        finally:
            self.stop()

    def stop(self):
        """Stop the bot."""
        self.running = False
        self.spot_feed.stop()
        self.poly_feed.stop()
        self.kalshi_feed.stop()

        # Print final ML report
        if self.use_ml:
            report = optimizer.get_performance_report()
            print(report)

        self._log("Bot stopped")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="15-min BTC Arbitrage Bot")
    parser.add_argument("--dry-run", action="store_true", help="Run without executing trades")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    bot = ArbitrageBot(dry_run=args.dry_run, verbose=args.verbose)

    # Handle shutdown gracefully
    def shutdown(sig, frame):
        print("\nShutdown requested...")
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Run the bot
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
