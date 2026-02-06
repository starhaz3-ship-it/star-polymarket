"""
Strategy Backtest - 30 Days of BTC, ETH, SOL
Tests TA + ML signal generation on historical data.
"""

import asyncio
import json
import httpx
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))

from arbitrage.ta_signals import TASignalGenerator, Candle

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Warning] matplotlib not installed - will skip charts")


@dataclass
class BacktestTrade:
    """A simulated trade."""
    asset: str
    side: str  # UP or DOWN
    entry_price: float  # Asset price at entry
    entry_time: datetime
    market_price: float  # Simulated market entry price (0.40-0.55)
    confidence: float
    edge: float
    # Outcome
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    won: bool = False


class StrategyBacktester:
    """Backtest the TA strategy on historical data."""

    # Strategy parameters (from ML tuner)
    UP_MAX_PRICE = 0.46
    UP_MIN_CONFIDENCE = 0.64
    DOWN_MAX_PRICE = 0.50
    DOWN_MIN_CONFIDENCE = 0.65
    MIN_EDGE = 0.06
    SKIP_HOURS_UTC = {0, 6, 7, 8, 14, 15, 19, 20, 21, 23}

    # Backtest settings
    POSITION_SIZE = 10.0
    WINDOW_MINUTES = 15  # 15-minute markets

    def __init__(self):
        self.generator = TASignalGenerator()
        self.trades: List[BacktestTrade] = []
        self.signals_generated = 0
        self.signals_skipped = 0

    async def fetch_historical_candles(self, symbol: str, days: int = 30) -> List[Candle]:
        """Fetch historical 1-minute candles from Binance."""
        print(f"[{symbol}] Fetching {days} days of 1m candles...")

        candles = []
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        async with httpx.AsyncClient(timeout=30) as client:
            current = start_time
            while current < end_time:
                # Binance limit is 1000 candles per request
                start_ms = int(current.timestamp() * 1000)

                try:
                    r = await client.get(
                        "https://api.binance.com/api/v3/klines",
                        params={
                            "symbol": symbol,
                            "interval": "1m",
                            "startTime": start_ms,
                            "limit": 1000
                        }
                    )

                    if r.status_code == 200:
                        data = r.json()
                        for k in data:
                            candles.append(Candle(
                                timestamp=k[0] / 1000,
                                open=float(k[1]),
                                high=float(k[2]),
                                low=float(k[3]),
                                close=float(k[4]),
                                volume=float(k[5])
                            ))

                        if data:
                            # Move to next batch
                            last_ts = data[-1][0] / 1000
                            current = datetime.fromtimestamp(last_ts, tz=timezone.utc) + timedelta(minutes=1)
                        else:
                            break
                    else:
                        print(f"[{symbol}] API error: {r.status_code}")
                        break

                except Exception as e:
                    print(f"[{symbol}] Fetch error: {e}")
                    break

                # Rate limit
                await asyncio.sleep(0.2)

        print(f"[{symbol}] Fetched {len(candles)} candles ({len(candles)/1440:.1f} days)")
        return candles

    def simulate_market_prices(self, model_up: float, model_down: float) -> Tuple[float, float]:
        """
        Simulate Polymarket prices based on model probabilities.
        Add some noise to simulate real market conditions.
        """
        # Markets typically have some spread around fair value
        noise = np.random.uniform(-0.05, 0.05)

        # Polymarket prices tend to be slightly inefficient
        up_price = max(0.05, min(0.95, model_up + noise))
        down_price = max(0.05, min(0.95, 1 - up_price + np.random.uniform(-0.02, 0.02)))

        return up_price, down_price

    def get_price_momentum(self, candles: List[Candle], lookback: int = 5) -> float:
        """Calculate recent price momentum."""
        if len(candles) < lookback:
            return 0.0
        old_price = candles[-lookback].close
        new_price = candles[-1].close
        return (new_price - old_price) / old_price

    def determine_outcome(self, candles: List[Candle], entry_idx: int,
                         side: str, window_minutes: int = 15) -> Tuple[bool, float]:
        """
        Determine if a trade would have won based on price action.

        For UP: wins if price is higher at window end
        For DOWN: wins if price is lower at window end
        """
        if entry_idx + window_minutes >= len(candles):
            return False, 0.0

        entry_price = candles[entry_idx].close
        exit_price = candles[entry_idx + window_minutes].close

        price_change = (exit_price - entry_price) / entry_price

        if side == "UP":
            won = exit_price > entry_price
        else:  # DOWN
            won = exit_price < entry_price

        return won, price_change

    def run_backtest(self, asset: str, candles: List[Candle]) -> Dict:
        """Run backtest on candles for a single asset."""
        print(f"\n[{asset}] Running backtest on {len(candles)} candles...")

        trades = []
        stats = {
            "asset": asset,
            "total_candles": len(candles),
            "windows_analyzed": 0,
            "signals_generated": 0,
            "trades_taken": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "by_side": {"UP": {"trades": 0, "wins": 0, "pnl": 0.0},
                       "DOWN": {"trades": 0, "wins": 0, "pnl": 0.0}},
            "by_hour": defaultdict(lambda: {"trades": 0, "wins": 0}),
            "by_confidence": defaultdict(lambda: {"trades": 0, "wins": 0}),
        }

        # Process in 15-minute windows
        lookback = 60  # Need 60 candles for TA calculations

        for i in range(lookback, len(candles) - self.WINDOW_MINUTES, self.WINDOW_MINUTES):
            stats["windows_analyzed"] += 1

            # Get window candles
            window_candles = candles[i - lookback:i]
            current_price = candles[i].close
            current_time = datetime.fromtimestamp(candles[i].timestamp, tz=timezone.utc)

            # Skip hours filter
            if current_time.hour in self.SKIP_HOURS_UTC:
                continue

            # Generate signal
            signal = self.generator.generate_signal(
                market_id=f"{asset}_backtest",
                candles=window_candles[-15:],  # Recent 15 candles
                current_price=current_price,
                market_yes_price=0.5,
                market_no_price=0.5,
                time_remaining_min=10.0
            )

            stats["signals_generated"] += 1

            # Simulate market prices
            up_price, down_price = self.simulate_market_prices(signal.model_up, signal.model_down)

            # Calculate momentum
            momentum = self.get_price_momentum(window_candles, lookback=5)

            # Apply strategy filters
            take_trade = False
            side = None
            entry_price = None
            confidence = None
            edge = None

            if signal.side == "UP":
                confidence = signal.model_up
                edge = signal.edge_up or 0

                if (confidence >= self.UP_MIN_CONFIDENCE and
                    up_price <= self.UP_MAX_PRICE and
                    edge >= self.MIN_EDGE and
                    momentum >= -0.001):  # Not falling
                    take_trade = True
                    side = "UP"
                    entry_price = up_price

            elif signal.side == "DOWN":
                confidence = signal.model_down
                edge = signal.edge_down or 0

                if (confidence >= self.DOWN_MIN_CONFIDENCE and
                    down_price <= self.DOWN_MAX_PRICE and
                    edge >= self.MIN_EDGE and
                    momentum <= -0.002):  # Must be falling
                    take_trade = True
                    side = "DOWN"
                    entry_price = down_price

            if take_trade and side:
                # Determine outcome
                won, price_change = self.determine_outcome(candles, i, side, self.WINDOW_MINUTES)

                # Calculate PnL
                if won:
                    # Win: get $1 per share, minus entry cost
                    shares = self.POSITION_SIZE / entry_price
                    pnl = shares - self.POSITION_SIZE
                else:
                    # Loss: lose entire position
                    pnl = -self.POSITION_SIZE

                trade = BacktestTrade(
                    asset=asset,
                    side=side,
                    entry_price=current_price,
                    entry_time=current_time,
                    market_price=entry_price,
                    confidence=confidence,
                    edge=edge,
                    exit_price=candles[i + self.WINDOW_MINUTES].close,
                    exit_time=datetime.fromtimestamp(candles[i + self.WINDOW_MINUTES].timestamp, tz=timezone.utc),
                    pnl=pnl,
                    won=won
                )
                trades.append(trade)

                # Update stats
                stats["trades_taken"] += 1
                stats["total_pnl"] += pnl
                if won:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1

                stats["by_side"][side]["trades"] += 1
                stats["by_side"][side]["pnl"] += pnl
                if won:
                    stats["by_side"][side]["wins"] += 1

                stats["by_hour"][current_time.hour]["trades"] += 1
                if won:
                    stats["by_hour"][current_time.hour]["wins"] += 1

                # Confidence bucket
                conf_bucket = f"{int(confidence * 10) * 10}%"
                stats["by_confidence"][conf_bucket]["trades"] += 1
                if won:
                    stats["by_confidence"][conf_bucket]["wins"] += 1

        stats["trades"] = trades
        self.trades.extend(trades)

        return stats

    def generate_report(self, results: Dict[str, Dict]) -> str:
        """Generate text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("STRATEGY BACKTEST REPORT - 30 DAYS")
        lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        lines.append("=" * 80)

        # Overall summary
        total_trades = sum(r["trades_taken"] for r in results.values())
        total_wins = sum(r["wins"] for r in results.values())
        total_pnl = sum(r["total_pnl"] for r in results.values())

        lines.append("\n## OVERALL SUMMARY")
        lines.append(f"Total Trades: {total_trades}")
        lines.append(f"Wins/Losses: {total_wins}/{total_trades - total_wins}")
        lines.append(f"Win Rate: {total_wins / max(1, total_trades) * 100:.1f}%")
        lines.append(f"Total PnL: ${total_pnl:+.2f}")
        lines.append(f"Avg PnL/Trade: ${total_pnl / max(1, total_trades):+.2f}")

        # Per-asset breakdown
        lines.append("\n## BY ASSET")
        lines.append("-" * 60)
        lines.append(f"{'Asset':<10} {'Trades':>8} {'Wins':>8} {'WR':>8} {'PnL':>12}")
        lines.append("-" * 60)

        for asset, stats in results.items():
            trades = stats["trades_taken"]
            wins = stats["wins"]
            wr = wins / max(1, trades) * 100
            pnl = stats["total_pnl"]
            lines.append(f"{asset:<10} {trades:>8} {wins:>8} {wr:>7.1f}% ${pnl:>+10.2f}")

        # By side breakdown
        lines.append("\n## BY SIDE (All Assets)")
        lines.append("-" * 60)

        up_trades = sum(r["by_side"]["UP"]["trades"] for r in results.values())
        up_wins = sum(r["by_side"]["UP"]["wins"] for r in results.values())
        up_pnl = sum(r["by_side"]["UP"]["pnl"] for r in results.values())

        down_trades = sum(r["by_side"]["DOWN"]["trades"] for r in results.values())
        down_wins = sum(r["by_side"]["DOWN"]["wins"] for r in results.values())
        down_pnl = sum(r["by_side"]["DOWN"]["pnl"] for r in results.values())

        lines.append(f"{'Side':<10} {'Trades':>8} {'Wins':>8} {'WR':>8} {'PnL':>12}")
        lines.append("-" * 60)
        lines.append(f"{'UP':<10} {up_trades:>8} {up_wins:>8} {up_wins/max(1,up_trades)*100:>7.1f}% ${up_pnl:>+10.2f}")
        lines.append(f"{'DOWN':<10} {down_trades:>8} {down_wins:>8} {down_wins/max(1,down_trades)*100:>7.1f}% ${down_pnl:>+10.2f}")

        # By hour breakdown (aggregate)
        lines.append("\n## BY HOUR (UTC)")
        lines.append("-" * 60)

        hour_stats = defaultdict(lambda: {"trades": 0, "wins": 0})
        for r in results.values():
            for hour, data in r["by_hour"].items():
                hour_stats[hour]["trades"] += data["trades"]
                hour_stats[hour]["wins"] += data["wins"]

        lines.append(f"{'Hour':<6} {'Trades':>8} {'Wins':>8} {'WR':>8}")
        lines.append("-" * 60)
        for hour in sorted(hour_stats.keys()):
            data = hour_stats[hour]
            if data["trades"] > 0:
                wr = data["wins"] / data["trades"] * 100
                lines.append(f"{hour:02d}:00  {data['trades']:>8} {data['wins']:>8} {wr:>7.1f}%")

        # Strategy parameters used
        lines.append("\n## STRATEGY PARAMETERS")
        lines.append("-" * 60)
        lines.append(f"UP Max Price: ${self.UP_MAX_PRICE}")
        lines.append(f"UP Min Confidence: {self.UP_MIN_CONFIDENCE:.0%}")
        lines.append(f"DOWN Max Price: ${self.DOWN_MAX_PRICE}")
        lines.append(f"DOWN Min Confidence: {self.DOWN_MIN_CONFIDENCE:.0%}")
        lines.append(f"Min Edge: {self.MIN_EDGE:.0%}")
        lines.append(f"Skip Hours: {sorted(self.SKIP_HOURS_UTC)}")
        lines.append(f"Position Size: ${self.POSITION_SIZE}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def generate_charts(self, results: Dict[str, Dict], output_dir: Path):
        """Generate visualization charts."""
        if not HAS_MATPLOTLIB:
            print("[Charts] Skipping - matplotlib not available")
            return

        print("[Charts] Generating visualizations...")

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Strategy Backtest Results - 30 Days', fontsize=14, fontweight='bold')

        # 1. Win Rate by Asset (bar chart)
        ax1 = axes[0, 0]
        assets = list(results.keys())
        win_rates = [results[a]["wins"] / max(1, results[a]["trades_taken"]) * 100 for a in assets]
        colors = ['green' if wr >= 55 else 'orange' if wr >= 50 else 'red' for wr in win_rates]
        bars = ax1.bar(assets, win_rates, color=colors, edgecolor='black', alpha=0.8)
        ax1.axhline(y=55, color='green', linestyle='--', alpha=0.7, label='Target (55%)')
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Breakeven')
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_title('Win Rate by Asset')
        ax1.legend()
        for bar, wr in zip(bars, win_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{wr:.1f}%', ha='center', va='bottom', fontsize=10)

        # 2. PnL by Asset
        ax2 = axes[0, 1]
        pnls = [results[a]["total_pnl"] for a in assets]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        bars = ax2.bar(assets, pnls, color=colors, edgecolor='black', alpha=0.8)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_ylabel('PnL ($)')
        ax2.set_title('Total PnL by Asset')
        for bar, pnl in zip(bars, pnls):
            y_pos = bar.get_height() + (5 if pnl >= 0 else -15)
            ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'${pnl:+.0f}', ha='center', va='bottom' if pnl >= 0 else 'top', fontsize=10)

        # 3. UP vs DOWN comparison
        ax3 = axes[1, 0]
        up_trades = sum(r["by_side"]["UP"]["trades"] for r in results.values())
        up_wins = sum(r["by_side"]["UP"]["wins"] for r in results.values())
        down_trades = sum(r["by_side"]["DOWN"]["trades"] for r in results.values())
        down_wins = sum(r["by_side"]["DOWN"]["wins"] for r in results.values())

        x = np.arange(2)
        width = 0.35
        trades_bars = ax3.bar(x - width/2, [up_trades, down_trades], width,
                              label='Trades', color='steelblue', alpha=0.8)
        wins_bars = ax3.bar(x + width/2, [up_wins, down_wins], width,
                           label='Wins', color='green', alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['UP', 'DOWN'])
        ax3.set_ylabel('Count')
        ax3.set_title('UP vs DOWN Performance')
        ax3.legend()

        # Add win rate annotations
        for i, (trades, wins) in enumerate([(up_trades, up_wins), (down_trades, down_wins)]):
            if trades > 0:
                wr = wins / trades * 100
                ax3.annotate(f'WR: {wr:.1f}%', xy=(i, max(trades, wins) + 5),
                            ha='center', fontsize=10, fontweight='bold')

        # 4. Cumulative PnL over time
        ax4 = axes[1, 1]
        all_trades = []
        for r in results.values():
            all_trades.extend(r["trades"])

        if all_trades:
            all_trades.sort(key=lambda t: t.entry_time)
            times = [t.entry_time for t in all_trades]
            cum_pnl = np.cumsum([t.pnl for t in all_trades])

            ax4.plot(times, cum_pnl, color='blue', linewidth=2)
            ax4.fill_between(times, cum_pnl, alpha=0.3,
                            color=['green' if v >= 0 else 'red' for v in cum_pnl])
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Cumulative PnL ($)')
            ax4.set_title('Cumulative PnL Over Time')
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax4.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # Save chart
        chart_path = output_dir / "backtest_results.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"[Charts] Saved to: {chart_path}")
        plt.close()

        # Additional chart: Win rate by hour
        fig2, ax = plt.subplots(figsize=(12, 5))

        hour_stats = defaultdict(lambda: {"trades": 0, "wins": 0})
        for r in results.values():
            for hour, data in r["by_hour"].items():
                hour_stats[hour]["trades"] += data["trades"]
                hour_stats[hour]["wins"] += data["wins"]

        hours = sorted(hour_stats.keys())
        win_rates = [hour_stats[h]["wins"] / max(1, hour_stats[h]["trades"]) * 100 for h in hours]
        trade_counts = [hour_stats[h]["trades"] for h in hours]

        # Bar chart for win rate
        colors = ['green' if wr >= 55 else 'orange' if wr >= 50 else 'red' for wr in win_rates]
        bars = ax.bar(hours, win_rates, color=colors, alpha=0.8, edgecolor='black')
        ax.axhline(y=55, color='green', linestyle='--', alpha=0.7, label='Target (55%)')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Breakeven')

        # Add trade count as text
        for bar, count, wr in zip(bars, trade_counts, win_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'n={count}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Hour (UTC)')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate by Hour of Day')
        ax.set_xticks(hours)
        ax.set_xticklabels([f'{h:02d}' for h in hours])
        ax.legend()

        chart_path2 = output_dir / "backtest_hourly.png"
        plt.savefig(chart_path2, dpi=150, bbox_inches='tight')
        print(f"[Charts] Saved to: {chart_path2}")
        plt.close()


async def main():
    """Run the backtest."""
    print("=" * 80)
    print("STRATEGY BACKTEST - 30 DAYS")
    print("Testing: BTC, ETH, SOL on 15-minute prediction markets")
    print("=" * 80)

    backtester = StrategyBacktester()
    output_dir = Path(__file__).parent

    # Assets to test
    assets = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT"
    }

    results = {}

    # Fetch data and run backtests
    for name, symbol in assets.items():
        candles = await backtester.fetch_historical_candles(symbol, days=30)
        if candles:
            stats = backtester.run_backtest(name, candles)
            results[name] = stats
        await asyncio.sleep(1)  # Rate limit

    # Generate report
    report = backtester.generate_report(results)
    print("\n" + report)

    # Save report
    report_path = output_dir / "backtest_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n[Report] Saved to: {report_path}")

    # Generate charts
    backtester.generate_charts(results, output_dir)

    # Save raw data
    data_path = output_dir / "backtest_data.json"
    export_data = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "summary": {
            asset: {k: v for k, v in stats.items() if k != "trades"}
            for asset, stats in results.items()
        },
        "trades": [asdict(t) for t in backtester.trades]
    }
    # Convert datetime objects
    for t in export_data["trades"]:
        t["entry_time"] = t["entry_time"].isoformat() if t["entry_time"] else None
        t["exit_time"] = t["exit_time"].isoformat() if t["exit_time"] else None

    with open(data_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    print(f"[Data] Saved to: {data_path}")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
