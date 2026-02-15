"""Quick backtest: inverse the top 5 losers and see if they become winners."""
import sys
import os
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nautilus_backtest.data_pipeline import fetch_binance_klines
from nautilus_backtest.actors.binary_scorer import BinaryOutcomeScorer
from nautilus_backtest.strategies.inverse_wrapper import InverseStrategy

# Import the 5 biggest losers
from nautilus_backtest.strategies.ha_keltner_mfi import HaKeltnerMfi
from nautilus_backtest.strategies.elder_impulse import ElderImpulse
from nautilus_backtest.strategies.kama_trend import KamaTrend
from nautilus_backtest.strategies.alignment import Alignment
from nautilus_backtest.strategies.aroon_cross import AroonCross

# Also test inverse of mid-range losers
from nautilus_backtest.strategies.ichimoku_simple import IchimokuSimple
from nautilus_backtest.strategies.wyckoff_vortex import WyckoffVortex
from nautilus_backtest.strategies.adx_di_cross import AdxDiCross
from nautilus_backtest.strategies.chaikin_mf import ChaikinMf
from nautilus_backtest.strategies.volcano_breakout import VolcanoBreakout

# And original top winners for comparison
from nautilus_backtest.strategies.williams_vwap import WilliamsVwap
from nautilus_backtest.strategies.stoch_bb import StochBb
from nautilus_backtest.strategies.mean_revert_extreme import MeanRevertExtreme


def main():
    print("=" * 90)
    print("INVERSE STRATEGY BACKTEST — Flip the losers, find new winners")
    print("=" * 90)

    days = 14
    klines = fetch_binance_klines(days=days)
    if not klines:
        print("No data!")
        return
    print(f"[DATA] {len(klines)} candles, {days} days")

    # Create strategies: inverse losers + original winners for comparison
    strategies = [
        # Inverse of top 5 losers
        InverseStrategy(HaKeltnerMfi(horizon_bars=5)),
        InverseStrategy(ElderImpulse(horizon_bars=5)),
        InverseStrategy(KamaTrend(horizon_bars=5)),
        InverseStrategy(Alignment(horizon_bars=5)),
        InverseStrategy(AroonCross(horizon_bars=5)),
        # Inverse of mid-range losers
        InverseStrategy(IchimokuSimple(horizon_bars=5)),
        InverseStrategy(WyckoffVortex(horizon_bars=5)),
        InverseStrategy(AdxDiCross(horizon_bars=5)),
        InverseStrategy(ChaikinMf(horizon_bars=5)),
        InverseStrategy(VolcanoBreakout(horizon_bars=5)),
        # Original winners for comparison
        WilliamsVwap(horizon_bars=5),
        StochBb(horizon_bars=5),
        MeanRevertExtreme(horizon_bars=5),
    ]

    scorer = BinaryOutcomeScorer(entry_cost=0.50, spread=0.03)

    print(f"[INIT] {len(strategies)} strategies:")
    for s in strategies:
        print(f"       {s.name}")

    print(f"\n[RUN] Processing {len(klines)} bars...")
    t0 = time.time()
    signals = 0

    for i, k in enumerate(klines):
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
                    open_time_ms=int(k[0]),
                )
                signals += 1

        scorer.check_resolution(i, close)

    if klines:
        last_close = float(klines[-1][4])
        pending = len(scorer.pending)
        if pending:
            scorer.force_resolve_pending(last_close)

    elapsed = time.time() - t0
    print(f"[DONE] {elapsed:.1f}s | {signals} signals | {len(scorer.resolved)} resolved\n")

    # Results
    summary = scorer.summary()
    breakeven_wr = 53.0  # 0.53 / (0.47 + 0.53) * 100

    print(f"{'Strategy':<35} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'PnL':>10} {'$/Trade':>8}  Status")
    print(f"{'-'*35} {'-'*7} {'-'*6} {'-'*7} {'-'*10} {'-'*8}  {'-'*12}")

    sorted_strats = sorted(summary.items(), key=lambda x: x[1]["pnl"], reverse=True)

    for name, stats in sorted_strats:
        avg_pnl = stats["pnl"] / stats["trades"] if stats["trades"] else 0
        status = "PROFITABLE" if stats["wr"] > breakeven_wr else "LOSING"
        marker = " <<<" if stats["wr"] > 55 and stats["trades"] >= 50 else ""
        print(f"{name:<35} {stats['trades']:>7} {stats['wins']:>6} "
              f"{stats['wr']:>6.1f}% ${stats['pnl']:>+8.2f} ${avg_pnl:>+7.4f}  {status}{marker}")

    # Summary
    overall = scorer.overall_summary()
    print(f"\n{'='*90}")
    print(f"TOTAL: {overall['total_trades']} trades, {overall['wins']}W/{overall['losses']}L, "
          f"{overall['wr']:.1f}% WR, PnL ${overall['total_pnl']:+.2f}")
    print(f"Breakeven WR: {breakeven_wr:.1f}%")

    # Compare inverse vs original
    print(f"\n{'='*90}")
    print("INVERSE vs ORIGINAL COMPARISON")
    print(f"{'='*90}")
    inv_trades = sum(s["trades"] for n, s in summary.items() if "_INV_" in n)
    inv_wins = sum(s["wins"] for n, s in summary.items() if "_INV_" in n)
    inv_pnl = sum(s["pnl"] for n, s in summary.items() if "_INV_" in n)
    inv_wr = inv_wins / inv_trades * 100 if inv_trades else 0

    orig_trades = sum(s["trades"] for n, s in summary.items() if "_INV_" not in n)
    orig_wins = sum(s["wins"] for n, s in summary.items() if "_INV_" not in n)
    orig_pnl = sum(s["pnl"] for n, s in summary.items() if "_INV_" not in n)
    orig_wr = orig_wins / orig_trades * 100 if orig_trades else 0

    print(f"  INVERSE strategies:  {inv_trades:>6} trades, {inv_wr:.1f}% WR, PnL ${inv_pnl:+.2f}")
    print(f"  ORIGINAL winners:    {orig_trades:>6} trades, {orig_wr:.1f}% WR, PnL ${orig_pnl:+.2f}")

    # Hourly breakdown for ALL profitable inverse strategies
    profitable_inv = [(n, s) for n, s in sorted_strats if "_INV_" in n and s["wr"] > breakeven_wr]
    for inv_name, inv_stats in profitable_inv:
        print(f"\n{'='*90}")
        print(f"HOURLY BREAKDOWN: {inv_name} ({inv_stats['trades']}T, {inv_stats['wr']:.1f}%WR, ${inv_stats['pnl']:+.2f})")
        print(f"{'='*90}")
        hourly = {}
        for sig in scorer.resolved:
            if sig.strategy != inv_name:
                continue
            h = datetime.fromtimestamp(sig.open_time_ms / 1000.0, tz=timezone.utc).hour if sig.open_time_ms else -1
            if h not in hourly:
                hourly[h] = {"t": 0, "w": 0, "pnl": 0.0}
            hourly[h]["t"] += 1
            if sig.correct:
                hourly[h]["w"] += 1
            hourly[h]["pnl"] += sig.pnl

        print(f"  {'Hour':>4}  {'Trades':>7}  {'Wins':>5}  {'WR%':>7}  {'PnL':>9}")
        entries = []
        for hour in range(24):
            if hour in hourly:
                h = hourly[hour]
                wr = h["w"] / h["t"] * 100 if h["t"] else 0
                marker = " <<<" if wr > 55 and h["t"] >= 10 else ""
                print(f"  {hour:>4}  {h['t']:>7}  {h['w']:>5}  {wr:>6.1f}%  ${h['pnl']:>+8.2f}{marker}")
                entries.append((hour, wr, h["t"], h["pnl"]))
        # Show top 5 hours
        top5 = sorted(entries, key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  TOP 5 HOURS: {[f'UTC{h}({wr:.0f}%)' for h, wr, _, _ in top5]}")


if __name__ == "__main__":
    main()
