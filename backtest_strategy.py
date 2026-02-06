"""
Strategy Backtest - 30 Days of BTC, ETH, SOL
Tests TA signal directional accuracy on historical data.

Focus: Does the TA signal correctly predict if price will go UP or DOWN
in the next 15 minutes?
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
    confidence: float  # Model confidence
    edge: float  # Calculated edge
    momentum: float  # Price momentum at entry
    rsi: float
    heiken_color: str
    # Outcome
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    price_change: float = 0.0
    won: bool = False
    pnl: float = 0.0


class StrategyBacktester:
    """Backtest the TA strategy on historical data."""

    # Strategy parameters (from ML tuner - current settings)
    UP_MIN_CONFIDENCE = 0.55  # Relaxed for more signals
    DOWN_MIN_CONFIDENCE = 0.55
    MIN_DIRECTION_SCORE = 3  # From ta_signals.py

    # Skip hours - same as live trading
    SKIP_HOURS_UTC = {0, 6, 7, 8, 14, 15, 19, 20, 21, 23}

    # Backtest settings
    POSITION_SIZE = 10.0
    WINDOW_MINUTES = 15  # 15-minute markets
    SIMULATED_ENTRY_PRICE = 0.45  # Assume we get in at good prices

    def __init__(self):
        self.generator = TASignalGenerator()
        self.trades: List[BacktestTrade] = []
        self.all_signals = []

    async def fetch_historical_candles(self, symbol: str, days: int = 30) -> List[Candle]:
        """Fetch historical 1-minute candles from Binance."""
        print(f"[{symbol}] Fetching {days} days of 1m candles...")

        candles = []
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        async with httpx.AsyncClient(timeout=30) as client:
            current = start_time
            while current < end_time:
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

                await asyncio.sleep(0.2)

        print(f"[{symbol}] Fetched {len(candles)} candles ({len(candles)/1440:.1f} days)")
        return candles

    def get_price_momentum(self, candles: List[Candle], lookback: int = 5) -> float:
        """Calculate recent price momentum."""
        if len(candles) < lookback:
            return 0.0
        old_price = candles[-lookback].close
        new_price = candles[-1].close
        return (new_price - old_price) / old_price

    def determine_outcome(self, candles: List[Candle], entry_idx: int,
                         window_minutes: int = 15) -> Tuple[str, float]:
        """
        Determine actual market direction based on price action.
        Returns (actual_direction, price_change)
        """
        if entry_idx + window_minutes >= len(candles):
            return "UNKNOWN", 0.0

        entry_price = candles[entry_idx].close
        exit_price = candles[entry_idx + window_minutes].close
        price_change = (exit_price - entry_price) / entry_price

        # Small threshold to avoid noise
        if abs(price_change) < 0.0001:  # Less than 0.01% = essentially flat
            return "FLAT", price_change
        elif price_change > 0:
            return "UP", price_change
        else:
            return "DOWN", price_change

    def run_backtest(self, asset: str, candles: List[Candle]) -> Dict:
        """Run backtest on candles for a single asset."""
        print(f"\n[{asset}] Running backtest on {len(candles)} candles...")

        trades = []
        stats = {
            "asset": asset,
            "total_candles": len(candles),
            "windows_analyzed": 0,
            "signals_up": 0,
            "signals_down": 0,
            "signals_skip": 0,
            "trades_taken": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "by_side": {"UP": {"trades": 0, "wins": 0, "pnl": 0.0},
                       "DOWN": {"trades": 0, "wins": 0, "pnl": 0.0}},
            "by_hour": defaultdict(lambda: {"trades": 0, "wins": 0}),
            "by_confidence": defaultdict(lambda: {"trades": 0, "wins": 0}),
            "by_momentum": {"positive": {"trades": 0, "wins": 0},
                           "negative": {"trades": 0, "wins": 0}},
        }

        # Process in 15-minute windows (like real market cadence)
        lookback = 60  # Need 60 candles for TA calculations

        for i in range(lookback, len(candles) - self.WINDOW_MINUTES, self.WINDOW_MINUTES):
            stats["windows_analyzed"] += 1

            window_candles = candles[i - lookback:i]
            current_price = candles[i].close
            current_time = datetime.fromtimestamp(candles[i].timestamp, tz=timezone.utc)

            # Skip hours filter (same as live)
            if current_time.hour in self.SKIP_HOURS_UTC:
                continue

            # Generate signal using recent 15 candles (like live)
            signal = self.generator.generate_signal(
                market_id=f"{asset}_backtest",
                candles=window_candles[-15:],
                current_price=current_price,
                market_yes_price=0.5,  # Neutral market
                market_no_price=0.5,
                time_remaining_min=10.0
            )

            # Track all signals
            momentum = self.get_price_momentum(window_candles, lookback=5)

            # Determine actual outcome
            actual_direction, price_change = self.determine_outcome(candles, i, self.WINDOW_MINUTES)

            if actual_direction == "UNKNOWN":
                continue

            # Only take trades with strong direction score
            take_trade = False
            side = None
            confidence = 0.0

            if signal.side == "UP" and signal.model_up >= self.UP_MIN_CONFIDENCE:
                side = "UP"
                confidence = signal.model_up
                take_trade = True
                stats["signals_up"] += 1
            elif signal.side == "DOWN" and signal.model_down >= self.DOWN_MIN_CONFIDENCE:
                side = "DOWN"
                confidence = signal.model_down
                take_trade = True
                stats["signals_down"] += 1
            else:
                stats["signals_skip"] += 1

            if take_trade and side:
                # Check if prediction was correct
                if actual_direction == "FLAT":
                    won = False  # Flat = no winner
                else:
                    won = (side == actual_direction)

                # Calculate PnL (simulated at $0.45 entry)
                entry_price_sim = self.SIMULATED_ENTRY_PRICE
                if won:
                    shares = self.POSITION_SIZE / entry_price_sim
                    pnl = shares - self.POSITION_SIZE  # Win: $1 per share
                else:
                    pnl = -self.POSITION_SIZE  # Loss: lose stake

                trade = BacktestTrade(
                    asset=asset,
                    side=side,
                    entry_price=current_price,
                    entry_time=current_time,
                    confidence=confidence,
                    edge=signal.edge_up if side == "UP" else signal.edge_down or 0,
                    momentum=momentum,
                    rsi=signal.rsi or 50,
                    heiken_color=signal.heiken_color or "gray",
                    exit_price=candles[i + self.WINDOW_MINUTES].close,
                    exit_time=datetime.fromtimestamp(candles[i + self.WINDOW_MINUTES].timestamp, tz=timezone.utc),
                    price_change=price_change,
                    won=won,
                    pnl=pnl
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

                # Momentum bucket
                mom_key = "positive" if momentum > 0 else "negative"
                stats["by_momentum"][mom_key]["trades"] += 1
                if won:
                    stats["by_momentum"][mom_key]["wins"] += 1

        stats["trades"] = trades
        self.trades.extend(trades)

        # Print quick summary
        if stats["trades_taken"] > 0:
            wr = stats["wins"] / stats["trades_taken"] * 100
            print(f"[{asset}] {stats['trades_taken']} trades, {stats['wins']} wins ({wr:.1f}% WR), PnL: ${stats['total_pnl']:+.2f}")

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
        total_windows = sum(r["windows_analyzed"] for r in results.values())

        lines.append("\n## OVERALL SUMMARY")
        lines.append(f"Windows Analyzed: {total_windows}")
        lines.append(f"Total Trades: {total_trades}")
        lines.append(f"Wins/Losses: {total_wins}/{total_trades - total_wins}")
        if total_trades > 0:
            lines.append(f"Win Rate: {total_wins / total_trades * 100:.1f}%")
            lines.append(f"Total PnL: ${total_pnl:+.2f}")
            lines.append(f"Avg PnL/Trade: ${total_pnl / total_trades:+.2f}")

            # Statistical significance
            import math
            p_win = total_wins / total_trades
            se = math.sqrt(p_win * (1 - p_win) / total_trades)
            ci_low = (p_win - 1.96 * se) * 100
            ci_high = (p_win + 1.96 * se) * 100
            lines.append(f"95% CI: [{ci_low:.1f}%, {ci_high:.1f}%]")

        # Per-asset breakdown
        lines.append("\n## BY ASSET")
        lines.append("-" * 70)
        lines.append(f"{'Asset':<8} {'Trades':>8} {'Wins':>8} {'Losses':>8} {'WR':>10} {'PnL':>12}")
        lines.append("-" * 70)

        for asset, stats in results.items():
            trades = stats["trades_taken"]
            wins = stats["wins"]
            losses = trades - wins
            wr = wins / max(1, trades) * 100
            pnl = stats["total_pnl"]
            lines.append(f"{asset:<8} {trades:>8} {wins:>8} {losses:>8} {wr:>9.1f}% ${pnl:>+10.2f}")

        # By side breakdown
        lines.append("\n## BY SIDE (All Assets)")
        lines.append("-" * 70)

        up_trades = sum(r["by_side"]["UP"]["trades"] for r in results.values())
        up_wins = sum(r["by_side"]["UP"]["wins"] for r in results.values())
        up_pnl = sum(r["by_side"]["UP"]["pnl"] for r in results.values())

        down_trades = sum(r["by_side"]["DOWN"]["trades"] for r in results.values())
        down_wins = sum(r["by_side"]["DOWN"]["wins"] for r in results.values())
        down_pnl = sum(r["by_side"]["DOWN"]["pnl"] for r in results.values())

        lines.append(f"{'Side':<8} {'Trades':>8} {'Wins':>8} {'Losses':>8} {'WR':>10} {'PnL':>12}")
        lines.append("-" * 70)
        lines.append(f"{'UP':<8} {up_trades:>8} {up_wins:>8} {up_trades-up_wins:>8} {up_wins/max(1,up_trades)*100:>9.1f}% ${up_pnl:>+10.2f}")
        lines.append(f"{'DOWN':<8} {down_trades:>8} {down_wins:>8} {down_trades-down_wins:>8} {down_wins/max(1,down_trades)*100:>9.1f}% ${down_pnl:>+10.2f}")

        # By confidence breakdown
        lines.append("\n## BY CONFIDENCE LEVEL")
        lines.append("-" * 70)

        conf_stats = defaultdict(lambda: {"trades": 0, "wins": 0})
        for r in results.values():
            for conf, data in r["by_confidence"].items():
                conf_stats[conf]["trades"] += data["trades"]
                conf_stats[conf]["wins"] += data["wins"]

        lines.append(f"{'Confidence':<12} {'Trades':>8} {'Wins':>8} {'WR':>10}")
        lines.append("-" * 70)
        for conf in sorted(conf_stats.keys(), key=lambda x: int(x.replace('%', ''))):
            data = conf_stats[conf]
            if data["trades"] > 0:
                wr = data["wins"] / data["trades"] * 100
                lines.append(f"{conf:<12} {data['trades']:>8} {data['wins']:>8} {wr:>9.1f}%")

        # By momentum
        lines.append("\n## BY MOMENTUM (at entry)")
        lines.append("-" * 70)

        pos_trades = sum(r["by_momentum"]["positive"]["trades"] for r in results.values())
        pos_wins = sum(r["by_momentum"]["positive"]["wins"] for r in results.values())
        neg_trades = sum(r["by_momentum"]["negative"]["trades"] for r in results.values())
        neg_wins = sum(r["by_momentum"]["negative"]["wins"] for r in results.values())

        lines.append(f"{'Momentum':<12} {'Trades':>8} {'Wins':>8} {'WR':>10}")
        lines.append("-" * 70)
        lines.append(f"{'Positive':<12} {pos_trades:>8} {pos_wins:>8} {pos_wins/max(1,pos_trades)*100:>9.1f}%")
        lines.append(f"{'Negative':<12} {neg_trades:>8} {neg_wins:>8} {neg_wins/max(1,neg_trades)*100:>9.1f}%")

        # By hour breakdown
        lines.append("\n## BY HOUR (UTC) - Top 10 by trades")
        lines.append("-" * 70)

        hour_stats = defaultdict(lambda: {"trades": 0, "wins": 0})
        for r in results.values():
            for hour, data in r["by_hour"].items():
                hour_stats[hour]["trades"] += data["trades"]
                hour_stats[hour]["wins"] += data["wins"]

        lines.append(f"{'Hour':<8} {'Trades':>8} {'Wins':>8} {'WR':>10}")
        lines.append("-" * 70)
        sorted_hours = sorted(hour_stats.items(), key=lambda x: x[1]["trades"], reverse=True)[:10]
        for hour, data in sorted_hours:
            if data["trades"] > 0:
                wr = data["wins"] / data["trades"] * 100
                lines.append(f"{hour:02d}:00    {data['trades']:>8} {data['wins']:>8} {wr:>9.1f}%")

        # Strategy parameters used
        lines.append("\n## STRATEGY PARAMETERS")
        lines.append("-" * 70)
        lines.append(f"UP Min Confidence: {self.UP_MIN_CONFIDENCE:.0%}")
        lines.append(f"DOWN Min Confidence: {self.DOWN_MIN_CONFIDENCE:.0%}")
        lines.append(f"Min Direction Score: {self.MIN_DIRECTION_SCORE}")
        lines.append(f"Skip Hours: {sorted(self.SKIP_HOURS_UTC)}")
        lines.append(f"Window: {self.WINDOW_MINUTES} minutes")
        lines.append(f"Simulated Entry Price: ${self.SIMULATED_ENTRY_PRICE}")
        lines.append(f"Position Size: ${self.POSITION_SIZE}")

        # Key findings
        lines.append("\n## KEY FINDINGS")
        lines.append("-" * 70)
        if total_trades > 0:
            overall_wr = total_wins / total_trades * 100
            up_wr = up_wins / max(1, up_trades) * 100
            down_wr = down_wins / max(1, down_trades) * 100

            if overall_wr >= 55:
                lines.append(f"[OK] Overall win rate {overall_wr:.1f}% exceeds 55% target")
            else:
                lines.append(f"[!!] Overall win rate {overall_wr:.1f}% below 55% target")

            if down_wr > up_wr:
                lines.append(f"[OK] DOWN ({down_wr:.1f}%) outperforms UP ({up_wr:.1f}%) - matches historical data")
            else:
                lines.append(f"[!!] UP ({up_wr:.1f}%) outperforms DOWN ({down_wr:.1f}%) - differs from historical")

            # Best performing config
            best_conf = max(conf_stats.items(), key=lambda x: x[1]["wins"]/max(1,x[1]["trades"]) if x[1]["trades"] >= 5 else 0)
            if best_conf[1]["trades"] >= 5:
                lines.append(f"[OK] Best confidence: {best_conf[0]} with {best_conf[1]['wins']/best_conf[1]['trades']*100:.1f}% WR")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def generate_charts(self, results: Dict[str, Dict], output_dir: Path):
        """Generate visualization charts."""
        if not HAS_MATPLOTLIB:
            print("[Charts] Skipping - matplotlib not available")
            return

        print("[Charts] Generating visualizations...")

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Strategy Backtest Results - 30 Days (BTC, ETH, SOL)', fontsize=14, fontweight='bold')

        # 1. Win Rate by Asset
        ax1 = axes[0, 0]
        assets = list(results.keys())
        win_rates = [results[a]["wins"] / max(1, results[a]["trades_taken"]) * 100 for a in assets]
        trade_counts = [results[a]["trades_taken"] for a in assets]
        colors = ['green' if wr >= 55 else 'orange' if wr >= 50 else 'red' for wr in win_rates]

        bars = ax1.bar(assets, win_rates, color=colors, edgecolor='black', alpha=0.8)
        ax1.axhline(y=55, color='green', linestyle='--', alpha=0.7, label='Target (55%)')
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Breakeven')
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_title('Win Rate by Asset')
        ax1.legend(loc='lower right')
        ax1.set_ylim(0, 100)

        for bar, wr, n in zip(bars, win_rates, trade_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{wr:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=10)

        # 2. UP vs DOWN comparison
        ax2 = axes[0, 1]
        up_trades = sum(r["by_side"]["UP"]["trades"] for r in results.values())
        up_wins = sum(r["by_side"]["UP"]["wins"] for r in results.values())
        down_trades = sum(r["by_side"]["DOWN"]["trades"] for r in results.values())
        down_wins = sum(r["by_side"]["DOWN"]["wins"] for r in results.values())

        up_wr = up_wins / max(1, up_trades) * 100
        down_wr = down_wins / max(1, down_trades) * 100

        x = np.arange(2)
        colors_side = ['green' if up_wr >= 55 else 'orange' if up_wr >= 50 else 'red',
                       'green' if down_wr >= 55 else 'orange' if down_wr >= 50 else 'red']

        bars = ax2.bar(['UP', 'DOWN'], [up_wr, down_wr], color=colors_side, edgecolor='black', alpha=0.8)
        ax2.axhline(y=55, color='green', linestyle='--', alpha=0.7, label='Target')
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('UP vs DOWN Win Rate')
        ax2.legend(loc='lower right')
        ax2.set_ylim(0, 100)

        for bar, wr, trades, wins in zip(bars, [up_wr, down_wr], [up_trades, down_trades], [up_wins, down_wins]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{wr:.1f}%\n({wins}/{trades})', ha='center', va='bottom', fontsize=10)

        # 3. Win Rate by Confidence Level
        ax3 = axes[1, 0]
        conf_stats = defaultdict(lambda: {"trades": 0, "wins": 0})
        for r in results.values():
            for conf, data in r["by_confidence"].items():
                conf_stats[conf]["trades"] += data["trades"]
                conf_stats[conf]["wins"] += data["wins"]

        confs = sorted([c for c in conf_stats.keys() if conf_stats[c]["trades"] >= 3],
                      key=lambda x: int(x.replace('%', '')))
        conf_wrs = [conf_stats[c]["wins"] / conf_stats[c]["trades"] * 100 for c in confs]
        conf_counts = [conf_stats[c]["trades"] for c in confs]

        colors_conf = ['green' if wr >= 55 else 'orange' if wr >= 50 else 'red' for wr in conf_wrs]
        bars = ax3.bar(confs, conf_wrs, color=colors_conf, edgecolor='black', alpha=0.8)
        ax3.axhline(y=55, color='green', linestyle='--', alpha=0.7)
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Model Confidence')
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_title('Win Rate by Confidence Level')
        ax3.set_ylim(0, 100)

        for bar, n in zip(bars, conf_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'n={n}', ha='center', va='bottom', fontsize=8)

        # 4. Cumulative PnL over time
        ax4 = axes[1, 1]
        all_trades = []
        for r in results.values():
            all_trades.extend(r["trades"])

        if all_trades:
            all_trades.sort(key=lambda t: t.entry_time)
            times = [t.entry_time for t in all_trades]
            cum_pnl = np.cumsum([t.pnl for t in all_trades])

            ax4.plot(times, cum_pnl, color='blue', linewidth=2, label='Cumulative PnL')
            ax4.fill_between(times, 0, cum_pnl, alpha=0.3,
                            where=[p >= 0 for p in cum_pnl], color='green')
            ax4.fill_between(times, 0, cum_pnl, alpha=0.3,
                            where=[p < 0 for p in cum_pnl], color='red')
            ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Cumulative PnL ($)')
            ax4.set_title(f'Cumulative PnL (Final: ${cum_pnl[-1]:+.2f})')
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax4.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            ax4.legend()

        plt.tight_layout()

        chart_path = output_dir / "backtest_results.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"[Charts] Saved to: {chart_path}")
        plt.close()

        # Second chart: Hourly analysis
        fig2, (ax_hour, ax_pnl_side) = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle('Detailed Analysis', fontsize=12, fontweight='bold')

        # Win rate by hour
        hour_stats = defaultdict(lambda: {"trades": 0, "wins": 0})
        for r in results.values():
            for hour, data in r["by_hour"].items():
                hour_stats[hour]["trades"] += data["trades"]
                hour_stats[hour]["wins"] += data["wins"]

        hours = sorted(hour_stats.keys())
        hour_wrs = [hour_stats[h]["wins"] / max(1, hour_stats[h]["trades"]) * 100 for h in hours]
        hour_counts = [hour_stats[h]["trades"] for h in hours]

        colors_hour = ['green' if wr >= 55 else 'orange' if wr >= 50 else 'red' for wr in hour_wrs]
        bars = ax_hour.bar(hours, hour_wrs, color=colors_hour, alpha=0.8, edgecolor='black')
        ax_hour.axhline(y=55, color='green', linestyle='--', alpha=0.7)
        ax_hour.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax_hour.set_xlabel('Hour (UTC)')
        ax_hour.set_ylabel('Win Rate (%)')
        ax_hour.set_title('Win Rate by Hour')
        ax_hour.set_xticks(hours)
        ax_hour.set_xticklabels([f'{h:02d}' for h in hours], fontsize=8)
        ax_hour.set_ylim(0, 100)

        # PnL by side for each asset
        x = np.arange(len(assets))
        width = 0.35
        up_pnls = [results[a]["by_side"]["UP"]["pnl"] for a in assets]
        down_pnls = [results[a]["by_side"]["DOWN"]["pnl"] for a in assets]

        bars1 = ax_pnl_side.bar(x - width/2, up_pnls, width, label='UP', color='steelblue', alpha=0.8)
        bars2 = ax_pnl_side.bar(x + width/2, down_pnls, width, label='DOWN', color='coral', alpha=0.8)
        ax_pnl_side.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax_pnl_side.set_xlabel('Asset')
        ax_pnl_side.set_ylabel('PnL ($)')
        ax_pnl_side.set_title('PnL by Side and Asset')
        ax_pnl_side.set_xticks(x)
        ax_pnl_side.set_xticklabels(assets)
        ax_pnl_side.legend()

        plt.tight_layout()

        chart_path2 = output_dir / "backtest_detailed.png"
        plt.savefig(chart_path2, dpi=150, bbox_inches='tight')
        print(f"[Charts] Saved to: {chart_path2}")
        plt.close()


async def main():
    """Run the backtest."""
    print("=" * 80)
    print("STRATEGY BACKTEST - 30 DAYS")
    print("Testing: BTC, ETH, SOL direction prediction for 15-minute windows")
    print("=" * 80)

    backtester = StrategyBacktester()
    output_dir = Path(__file__).parent

    assets = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT"
    }

    results = {}

    for name, symbol in assets.items():
        candles = await backtester.fetch_historical_candles(symbol, days=30)
        if candles:
            stats = backtester.run_backtest(name, candles)
            results[name] = stats
        await asyncio.sleep(1)

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
