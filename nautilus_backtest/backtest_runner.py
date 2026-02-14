"""Nautilus Backtest Runner - simple bar-by-bar loop for binary outcome backtesting.

Does NOT use nautilus_trader's BacktestEngine. Instead, fetches BTC 1-min candles
from Binance and loops through each bar, feeding data to 8 strategies at 2 horizons
(5-min and 15-min) for 16 strategy instances total. Signals are scored as binary
outcomes (Polymarket-style PnL).

Usage:
    python -m nautilus_backtest.backtest_runner
    python -m nautilus_backtest.backtest_runner --days 7
    python nautilus_backtest/backtest_runner.py --days 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

# Ensure parent directory is on sys.path for package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nautilus_backtest.data_pipeline import fetch_binance_klines
from nautilus_backtest.actors.binary_scorer import BinaryOutcomeScorer
from nautilus_backtest.strategies.alignment import Alignment
from nautilus_backtest.strategies.tdi_squeeze import TdiSqueeze
from nautilus_backtest.strategies.fisher_cascade import FisherCascade
from nautilus_backtest.strategies.volcano_breakout import VolcanoBreakout
from nautilus_backtest.strategies.wyckoff_vortex import WyckoffVortex
from nautilus_backtest.strategies.momentum_regime import MomentumRegime
from nautilus_backtest.strategies.mean_revert_extreme import MeanRevertExtreme
from nautilus_backtest.strategies.divergence_hunter import DivergenceHunter
from nautilus_backtest.config import DEFAULT_ENTRY_PRICE, DEFAULT_SPREAD


# ── Formatting Helpers ───────────────────────────────────────────────────────

def fmt_pnl(val: float) -> str:
    """Format PnL with color-like prefix."""
    sign = "+" if val >= 0 else ""
    return f"{sign}${val:.2f}"


def fmt_wr(wr: float) -> str:
    return f"{wr:.1f}%"


def print_divider(char: str = "=", width: int = 90) -> None:
    print(char * width)


def print_header(title: str, width: int = 90) -> None:
    print()
    print_divider("=", width)
    padding = (width - len(title) - 2) // 2
    print(f"{'=' * padding} {title} {'=' * padding}")
    print_divider("=", width)


def ascii_chart(series: dict[str, list[float]], width: int = 60, height: int = 15) -> str:
    """Render an ASCII cumulative PnL chart for multiple strategies.

    Each strategy gets a single character marker. The chart shows the
    final cumulative PnL progression for each strategy.
    """
    if not series or all(len(v) == 0 for v in series.values()):
        return "  (no data to chart)\n"

    # Assign markers to strategies
    markers = "ATFVWMRD*#@%&^~+"
    strat_names = sorted(series.keys())
    marker_map = {}
    for idx, name in enumerate(strat_names):
        marker_map[name] = markers[idx % len(markers)]

    # Find global min/max PnL across all strategies
    all_vals = []
    for vals in series.values():
        if vals:
            all_vals.extend(vals)
    if not all_vals:
        return "  (no data to chart)\n"

    vmin = min(all_vals)
    vmax = max(all_vals)
    if abs(vmax - vmin) < 0.01:
        vmax = vmin + 1.0  # Prevent division by zero

    lines = []

    # Legend
    legend_parts = []
    for name in strat_names:
        short = name[:12]
        legend_parts.append(f"  {marker_map[name]}={short}")
    lines.append("  Legend:" + "".join(legend_parts))
    lines.append("")

    # Build chart grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Zero line
    if vmin <= 0 <= vmax:
        zero_row = int((vmax - 0) / (vmax - vmin) * (height - 1))
        zero_row = max(0, min(height - 1, zero_row))
        for c in range(width):
            if grid[zero_row][c] == " ":
                grid[zero_row][c] = "-"

    # Plot each strategy
    for name, vals in series.items():
        if not vals:
            continue
        m = marker_map[name]
        n_points = len(vals)
        # Resample to fit width
        for col in range(width):
            idx = int(col / width * n_points)
            idx = min(idx, n_points - 1)
            v = vals[idx]
            row = int((vmax - v) / (vmax - vmin) * (height - 1))
            row = max(0, min(height - 1, row))
            grid[row][col] = m

    # Render with Y-axis labels
    for r in range(height):
        val = vmax - (r / (height - 1)) * (vmax - vmin)
        label = f"{val:+7.2f}"
        row_str = "".join(grid[r])
        lines.append(f"  {label} |{row_str}|")

    # X-axis
    lines.append(f"  {'':>7} +{'-' * width}+")
    lines.append(f"  {'':>7}  {'Start':<{width // 2}}{'End':>{width // 2}}")

    return "\n".join(lines) + "\n"


# ── Strategy Instance Factory ────────────────────────────────────────────────

def create_strategy_instances() -> list:
    """Create all 16 strategy instances (8 strategies x 2 horizons)."""
    strategies = []

    # Each strategy class, instantiated at 5-min and 15-min horizons
    strategy_classes = [
        Alignment,
        TdiSqueeze,
        FisherCascade,
        VolcanoBreakout,
        WyckoffVortex,
        MomentumRegime,
        MeanRevertExtreme,
        DivergenceHunter,
    ]

    for cls in strategy_classes:
        for horizon in [5, 15]:
            strategies.append(cls(horizon_bars=horizon))

    return strategies


# ── Main Backtest Loop ───────────────────────────────────────────────────────

def run_backtest(
    days: int = 14,
    entry_cost: float = DEFAULT_ENTRY_PRICE,
    spread: float = DEFAULT_SPREAD,
) -> None:
    """Run the full backtest: fetch data, loop bars, resolve signals, print results."""

    print_header(f"NAUTILUS BINARY BACKTEST  |  {days} days  |  BTC/USDT 1m")

    # 1. Fetch data
    t0 = time.time()
    klines = fetch_binance_klines(days=days)
    if not klines:
        print("[ERROR] No kline data fetched. Exiting.")
        return
    fetch_time = time.time() - t0
    print(f"[DATA] {len(klines)} candles fetched in {fetch_time:.1f}s")

    # 2. Initialize strategies and scorer
    strategies = create_strategy_instances()
    scorer = BinaryOutcomeScorer(entry_cost=entry_cost, spread=spread)

    print(f"[INIT] {len(strategies)} strategy instances:")
    for s in strategies:
        print(f"       - {s.name} ({s.horizon_bars}-bar horizon)")

    # 3. Bar-by-bar loop
    print(f"\n[RUN]  Processing {len(klines)} bars...")
    t1 = time.time()
    signals_generated = 0

    for i, k in enumerate(klines):
        open_time = int(k[0])
        high = float(k[2])
        low = float(k[3])
        close = float(k[4])
        volume = float(k[5])

        for strategy in strategies:
            direction, confidence = strategy.update(high, low, close, volume)
            if direction:
                scorer.add_signal(
                    strategy=strategy.name,
                    bar_index=i,
                    direction=direction,
                    confidence=confidence,
                    entry_price=close,
                    horizon_bars=strategy.horizon_bars,
                    open_time_ms=open_time,
                )
                signals_generated += 1

        scorer.check_resolution(i, close)

        # Progress
        if (i + 1) % 5000 == 0:
            print(f"       {i + 1}/{len(klines)} bars | {signals_generated} signals | "
                  f"{len(scorer.resolved)} resolved")

    # Force-resolve any signals still pending at end of data
    if klines:
        last_close = float(klines[-1][4])
        pending_count = len(scorer.pending)
        if pending_count > 0:
            scorer.force_resolve_pending(last_close)
            print(f"[WARN] Force-resolved {pending_count} pending signals at data end")

    loop_time = time.time() - t1
    print(f"[DONE] {len(klines)} bars in {loop_time:.1f}s | "
          f"{signals_generated} signals | {len(scorer.resolved)} resolved")

    # 4. Print results
    if not scorer.resolved:
        print("\n[WARN] No signals resolved. Strategies may need more warmup data.")
        print("       Try --days 30 for more data.")
        return

    print_results(scorer, klines)


# ── Results Printer ──────────────────────────────────────────────────────────

def print_results(scorer: BinaryOutcomeScorer, klines: list) -> None:
    """Print comprehensive backtest results."""

    # ── Per-Strategy Summary Table ──
    print_header("PER-STRATEGY SUMMARY")

    summary = scorer.summary()
    overall = scorer.overall_summary()

    # Table header
    print(f"  {'Strategy':<28} {'Horizon':>7} {'Trades':>7} {'Wins':>6} "
          f"{'Loss':>6} {'WR%':>7} {'PnL':>9} {'AvgConf':>8}")
    print(f"  {'-' * 28} {'-' * 7} {'-' * 7} {'-' * 6} {'-' * 6} "
          f"{'-' * 7} {'-' * 9} {'-' * 8}")

    sorted_strats = sorted(summary.items(), key=lambda x: x[1]["pnl"], reverse=True)

    for name, stats in sorted_strats:
        # Extract horizon from strategy name (last part after _)
        # Name format: "StrategyName_5m" or "StrategyName_15m"
        if "_5m" in name:
            horizon_str = "5m"
            base_name = name.replace("_5m", "")
        elif "_15m" in name:
            horizon_str = "15m"
            base_name = name.replace("_15m", "")
        else:
            horizon_str = "?"
            base_name = name

        print(f"  {base_name:<28} {horizon_str:>7} {stats['trades']:>7} "
              f"{stats['wins']:>6} {stats['losses']:>6} "
              f"{fmt_wr(stats['wr']):>7} {fmt_pnl(stats['pnl']):>9} "
              f"{stats['avg_confidence']:>8.3f}")

    # Totals row
    print(f"  {'-' * 28} {'-' * 7} {'-' * 7} {'-' * 6} {'-' * 6} "
          f"{'-' * 7} {'-' * 9} {'-' * 8}")
    print(f"  {'TOTAL':<28} {'':>7} {overall['total_trades']:>7} "
          f"{overall['wins']:>6} {overall['losses']:>6} "
          f"{fmt_wr(overall['wr']):>7} {fmt_pnl(overall['total_pnl']):>9}")

    # ── Best / Worst ──
    print_header("BEST / WORST")
    if sorted_strats:
        best_name, best = sorted_strats[0]
        worst_name, worst = sorted_strats[-1]
        print(f"  BEST:  {best_name:<30} WR={fmt_wr(best['wr'])}  PnL={fmt_pnl(best['pnl'])}  "
              f"({best['trades']} trades)")
        print(f"  WORST: {worst_name:<30} WR={fmt_wr(worst['wr'])}  PnL={fmt_pnl(worst['pnl'])}  "
              f"({worst['trades']} trades)")

    # Breakeven WR for these odds
    entry = scorer.entry_cost
    spread = scorer.spread
    win_payout = 1.0 - entry - spread
    loss_cost = entry + spread
    breakeven_wr = loss_cost / (win_payout + loss_cost) * 100
    print(f"\n  Breakeven WR: {breakeven_wr:.1f}% "
          f"(win=${win_payout:.2f}, loss=${loss_cost:.2f})")

    # ── Hourly Breakdown ──
    print_header("HOUR-OF-DAY BREAKDOWN (UTC)")

    hourly = scorer.hourly_breakdown()
    print(f"  {'Hour':>4}  {'Trades':>7}  {'Wins':>5}  {'WR%':>7}  {'PnL':>9}  Bar")
    print(f"  {'-' * 4}  {'-' * 7}  {'-' * 5}  {'-' * 7}  {'-' * 9}  {'-' * 20}")
    for hour in range(24):
        if hour in hourly:
            h = hourly[hour]
            bar_len = int(h["trades"] / max(1, max(hh["trades"] for hh in hourly.values())) * 20)
            bar = "#" * bar_len
            wr_marker = " <<<" if h["wr"] >= 55 and h["trades"] >= 10 else ""
            print(f"  {hour:>4}  {h['trades']:>7}  {h['wins']:>5}  "
                  f"{fmt_wr(h['wr']):>7}  {fmt_pnl(h['pnl']):>9}  {bar}{wr_marker}")
        else:
            print(f"  {hour:>4}  {'0':>7}  {'0':>5}  {'  n/a':>7}  {'$0.00':>9}")

    # ── Consensus Analysis ──
    print_header("CONSENSUS ANALYSIS (2+ strategies agree)")

    consensus = scorer.consensus_analysis()
    c = consensus["consensus"]
    s = consensus["solo"]
    print(f"  CONSENSUS (2+ agree): {c['trades']} trades, "
          f"{c['wins']} wins, WR={fmt_wr(c['wr'])}, PnL={fmt_pnl(c['pnl'])}")
    print(f"  SOLO (1 strategy):    {s['trades']} trades, "
          f"{s['wins']} wins, WR={fmt_wr(s['wr'])}, PnL={fmt_pnl(s['pnl'])}")
    if c["trades"] > 0 and s["trades"] > 0:
        edge = c["wr"] - s["wr"]
        edge_sign = "+" if edge >= 0 else ""
        print(f"  CONSENSUS EDGE:       {edge_sign}{edge:.1f}% WR improvement")

    # ── Confidence Calibration ──
    print_header("CONFIDENCE CALIBRATION")

    conf_buckets = {}
    for sig in scorer.resolved:
        bucket = round(sig.confidence, 1)
        if bucket not in conf_buckets:
            conf_buckets[bucket] = {"trades": 0, "wins": 0, "pnl": 0.0}
        conf_buckets[bucket]["trades"] += 1
        if sig.correct:
            conf_buckets[bucket]["wins"] += 1
        conf_buckets[bucket]["pnl"] += sig.pnl

    print(f"  {'Conf':>5}  {'Trades':>7}  {'Wins':>5}  {'WR%':>7}  {'PnL':>9}")
    print(f"  {'-' * 5}  {'-' * 7}  {'-' * 5}  {'-' * 7}  {'-' * 9}")
    for bucket in sorted(conf_buckets.keys()):
        b = conf_buckets[bucket]
        wr = (b["wins"] / b["trades"] * 100) if b["trades"] else 0
        marker = " <<<" if wr >= 53 and b["trades"] >= 20 else ""
        print(f"  {bucket:>5.1f}  {b['trades']:>7}  {b['wins']:>5}  "
              f"{fmt_wr(wr):>7}  {fmt_pnl(b['pnl']):>9}{marker}")

    # ── Per-Strategy x Horizon x Hour Deep Dive (TDI_SQUEEZE only) ──
    print_header("TDI_SQUEEZE_15m HOUR BREAKDOWN")

    tdi_15m_sigs = [s for s in scorer.resolved if s.strategy == "TDI_SQUEEZE_15m"]
    if tdi_15m_sigs:
        tdi_hourly = {}
        for sig in tdi_15m_sigs:
            if sig.open_time_ms > 0:
                h = datetime.fromtimestamp(sig.open_time_ms / 1000.0, tz=timezone.utc).hour
            else:
                h = -1
            if h not in tdi_hourly:
                tdi_hourly[h] = {"trades": 0, "wins": 0, "pnl": 0.0}
            tdi_hourly[h]["trades"] += 1
            if sig.correct:
                tdi_hourly[h]["wins"] += 1
            tdi_hourly[h]["pnl"] += sig.pnl

        print(f"  {'Hour':>4}  {'Trades':>7}  {'Wins':>5}  {'WR%':>7}  {'PnL':>9}")
        print(f"  {'-' * 4}  {'-' * 7}  {'-' * 5}  {'-' * 7}  {'-' * 9}")
        for hour in range(24):
            if hour in tdi_hourly:
                h = tdi_hourly[hour]
                wr = (h["wins"] / h["trades"] * 100) if h["trades"] else 0
                marker = " <<<" if wr >= 55 and h["trades"] >= 5 else ""
                print(f"  {hour:>4}  {h['trades']:>7}  {h['wins']:>5}  "
                      f"{fmt_wr(wr):>7}  {fmt_pnl(h['pnl']):>9}{marker}")
    else:
        print("  (no TDI_SQUEEZE_15m signals)")

    # ── Skip-Hour Filtered Results ──
    print_header("FILTERED: SKIP BAD HOURS (12,17,19,20,21)")

    skip_hours = {12, 17, 19, 20, 21}
    filtered_by_strat = {}
    for sig in scorer.resolved:
        if sig.open_time_ms > 0:
            h = datetime.fromtimestamp(sig.open_time_ms / 1000.0, tz=timezone.utc).hour
            if h in skip_hours:
                continue
        name = sig.strategy
        if name not in filtered_by_strat:
            filtered_by_strat[name] = {"trades": 0, "wins": 0, "pnl": 0.0}
        filtered_by_strat[name]["trades"] += 1
        if sig.correct:
            filtered_by_strat[name]["wins"] += 1
        filtered_by_strat[name]["pnl"] += sig.pnl

    sorted_filtered = sorted(filtered_by_strat.items(), key=lambda x: x[1]["pnl"], reverse=True)
    print(f"  {'Strategy':<28} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'PnL':>9}")
    print(f"  {'-' * 28} {'-' * 7} {'-' * 6} {'-' * 7} {'-' * 9}")
    total_ft, total_fw, total_fpnl = 0, 0, 0.0
    for name, stats in sorted_filtered:
        wr = (stats["wins"] / stats["trades"] * 100) if stats["trades"] else 0
        marker = " <<<" if wr >= 53 else ""
        print(f"  {name:<28} {stats['trades']:>7} {stats['wins']:>6} "
              f"{fmt_wr(wr):>7} {fmt_pnl(stats['pnl']):>9}{marker}")
        total_ft += stats["trades"]
        total_fw += stats["wins"]
        total_fpnl += stats["pnl"]
    total_fwr = (total_fw / total_ft * 100) if total_ft else 0
    print(f"  {'-' * 28} {'-' * 7} {'-' * 6} {'-' * 7} {'-' * 9}")
    print(f"  {'FILTERED TOTAL':<28} {total_ft:>7} {total_fw:>6} "
          f"{fmt_wr(total_fwr):>7} {fmt_pnl(total_fpnl):>9}")

    # ── ASCII Cumulative PnL Chart ──
    print_header("CUMULATIVE PnL CHART")

    cum_pnl = scorer.cumulative_pnl_by_strategy()
    chart = ascii_chart(cum_pnl, width=60, height=18)
    print(chart)

    # ── Overall Summary ──
    print_header("OVERALL SUMMARY")
    print(f"  Total Trades:    {overall['total_trades']}")
    print(f"  Wins / Losses:   {overall['wins']} / {overall['losses']}")
    print(f"  Win Rate:        {fmt_wr(overall['wr'])}")
    print(f"  Total PnL:       {fmt_pnl(overall['total_pnl'])}")
    print(f"  Avg PnL/Trade:   {fmt_pnl(overall['avg_pnl'])}")

    # Data range
    if klines:
        start_dt = datetime.fromtimestamp(int(klines[0][0]) / 1000, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(int(klines[-1][0]) / 1000, tz=timezone.utc)
        print(f"  Data Range:      {start_dt:%Y-%m-%d %H:%M} to {end_dt:%Y-%m-%d %H:%M} UTC")
        print(f"  Candles:         {len(klines)}")

    print()
    print_divider("=")
    print()


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Nautilus Binary Outcome Backtester - BTC directional signals vs Polymarket PnL"
    )
    parser.add_argument(
        "--days", type=int, default=14,
        help="Number of days of BTC 1-min data to fetch (default: 14)"
    )
    parser.add_argument(
        "--entry-cost", type=float, default=None,
        help=f"Simulated entry cost (default: {DEFAULT_ENTRY_PRICE})"
    )
    parser.add_argument(
        "--spread", type=float, default=None,
        help=f"Simulated spread (default: {DEFAULT_SPREAD})"
    )
    args = parser.parse_args()

    entry = args.entry_cost if args.entry_cost is not None else DEFAULT_ENTRY_PRICE
    spread = args.spread if args.spread is not None else DEFAULT_SPREAD

    if args.entry_cost is not None or args.spread is not None:
        print(f"[CONFIG] Entry cost: {entry}")
        print(f"[CONFIG] Spread: {spread}")

    run_backtest(days=args.days, entry_cost=entry, spread=spread)


if __name__ == "__main__":
    main()
