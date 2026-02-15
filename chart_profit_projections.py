"""
Speculative profit projections for the Maker Bot — all strategies combined.

Strategies:
  1. Maker spread (both-sides market making)
  2. V3.0 Momentum (5m BTC) — 89.3% WR backtest
  3. V12a Order Flow (5m BTC, non-overlapping) — 62.1% WR backtest
  4. Late Entry (15m BTC) — 93.2% WR backtest

Two scenarios: Conservative vs Optimistic
Two lot sizes: $5 and $10
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path


# ============================================================
# STRATEGY PARAMETERS (from backtests)
# ============================================================

# Backtest win rates
WR_V30 = 0.893       # V3.0 momentum 5m
WR_V12A = 0.621      # V12a unique (non-overlapping with V3.0)
WR_LATE = 0.932      # Late entry 15m

# Trades per day (from 7-day backtests)
TPD_V30 = 46.9
TPD_V12A = 9.4       # unique only (non-overlapping)
TPD_LATE = 44.3

# Maker spread: estimated from live data (~$0.10-0.20 per paired trade)
MAKER_PAIRS_PER_DAY = 40  # ~40 paired trades/day


def calc_pnl_per_trade(lot_size, buy_price=0.51):
    """Calculate win/loss PnL for a directional binary bet."""
    import math
    shares = max(1, math.floor(lot_size / buy_price))
    cost = buy_price * shares
    win_pnl = shares - cost  # pay $1/share on win
    loss_pnl = -cost
    return win_pnl, loss_pnl


def project(wr, tpd, win_pnl, loss_pnl, fill_rate=1.0):
    """Calculate expected PnL per hour/day/week/month."""
    effective_tpd = tpd * fill_rate
    avg_pnl = wr * win_pnl + (1 - wr) * loss_pnl
    daily = effective_tpd * avg_pnl
    return {
        "hour": daily / 24,
        "day": daily,
        "week": daily * 7,
        "month": daily * 30,
        "trades_day": effective_tpd,
        "avg_pnl": avg_pnl,
        "wr": wr,
    }


def build_scenario(lot_size, wr_adj, fill_rate, maker_daily, label):
    """Build a full scenario across all strategies."""
    win_pnl, loss_pnl = calc_pnl_per_trade(lot_size)

    v30 = project(WR_V30 + wr_adj, TPD_V30, win_pnl, loss_pnl, fill_rate)
    v12a = project(WR_V12A + wr_adj, TPD_V12A, win_pnl, loss_pnl, fill_rate)
    late = project(WR_LATE + wr_adj, TPD_LATE, win_pnl, loss_pnl, fill_rate)

    # Maker spread scales with lot size
    maker_scale = lot_size / 5.0
    maker = {
        "hour": maker_daily * maker_scale / 24,
        "day": maker_daily * maker_scale,
        "week": maker_daily * maker_scale * 7,
        "month": maker_daily * maker_scale * 30,
    }

    combined = {}
    for tf in ["hour", "day", "week", "month"]:
        combined[tf] = v30[tf] + v12a[tf] + late[tf] + maker[tf]

    return {
        "label": label,
        "lot_size": lot_size,
        "v30": v30, "v12a": v12a, "late": late, "maker": maker,
        "combined": combined,
        "total_trades_day": v30["trades_day"] + v12a["trades_day"] + late["trades_day"] + MAKER_PAIRS_PER_DAY * fill_rate,
    }


def main():
    # Build 4 scenarios
    scenarios = {
        "cons_5": build_scenario(
            lot_size=5, wr_adj=-0.08, fill_rate=0.75, maker_daily=2.0,
            label="Conservative $5"),
        "opt_5": build_scenario(
            lot_size=5, wr_adj=0.0, fill_rate=0.95, maker_daily=8.0,
            label="Optimistic $5"),
        "cons_10": build_scenario(
            lot_size=10, wr_adj=-0.08, fill_rate=0.75, maker_daily=2.0,
            label="Conservative $10"),
        "opt_10": build_scenario(
            lot_size=10, wr_adj=0.0, fill_rate=0.95, maker_daily=8.0,
            label="Optimistic $10"),
    }

    # Print table
    print("=" * 90)
    print("  MAKER BOT PROFIT PROJECTIONS — ALL STRATEGIES COMBINED")
    print("=" * 90)

    for key in ["cons_5", "opt_5", "cons_10", "opt_10"]:
        s = scenarios[key]
        c = s["combined"]
        print(f"\n  {s['label']}  (WR adj: {'-8%' if 'cons' in key else 'backtest'}, "
              f"fill: {'75%' if 'cons' in key else '95%'})")
        print(f"  {'-' * 60}")
        print(f"    Trades/day: {s['total_trades_day']:.0f}")
        print(f"    V3.0:  {s['v30']['wr']:.1%} WR, ${s['v30']['day']:+.2f}/day")
        print(f"    V12a:  {s['v12a']['wr']:.1%} WR, ${s['v12a']['day']:+.2f}/day")
        print(f"    Late:  {s['late']['wr']:.1%} WR, ${s['late']['day']:+.2f}/day")
        print(f"    Maker: ${s['maker']['day']:+.2f}/day")
        print(f"    ----------------------------")
        print(f"    TOTAL: ${c['hour']:+.2f}/hr | ${c['day']:+.2f}/day | "
              f"${c['week']:+.2f}/wk | ${c['month']:+.2f}/mo")

    # ============================================================
    # BUILD CHART
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Maker Bot — Speculative Profit Projections\nAll Strategies Combined (Maker + V3.0 + V12a + Late Entry)",
                 fontsize=14, fontweight="bold", y=0.98)

    timeframes = ["hour", "day", "week", "month"]
    tf_labels = ["Per Hour", "Per Day", "Per Week", "Per Month"]

    # Color scheme
    colors = {
        "cons_5": "#4A90D9",   # blue
        "opt_5": "#2E7D32",    # green
        "cons_10": "#E65100",  # orange
        "opt_10": "#C62828",   # red
    }
    labels_map = {
        "cons_5": "Conservative $5",
        "opt_5": "Optimistic $5",
        "cons_10": "Conservative $10",
        "opt_10": "Optimistic $10",
    }

    for idx, (tf, tf_label) in enumerate(zip(timeframes, tf_labels)):
        ax = axes[idx // 2][idx % 2]

        # Strategy breakdown as stacked bars
        strat_names = ["Maker\nSpread", "V3.0\nMomentum\n5m", "V12a\nOrder Flow\n5m", "Late Entry\n15m"]
        scenario_keys = ["cons_5", "opt_5", "cons_10", "opt_10"]
        x = np.arange(len(strat_names))
        width = 0.18

        for i, sk in enumerate(scenario_keys):
            s = scenarios[sk]
            vals = [s["maker"][tf], s["v30"][tf], s["v12a"][tf], s["late"][tf]]
            bars = ax.bar(x + i * width - 1.5 * width, vals, width,
                         label=labels_map[sk], color=colors[sk], alpha=0.85,
                         edgecolor="white", linewidth=0.5)

            # Value labels on bars
            for bar, val in zip(bars, vals):
                if abs(val) >= 0.01:
                    va = "bottom" if val >= 0 else "top"
                    fmt = f"${val:.2f}" if abs(val) < 10 else f"${val:.0f}"
                    if abs(val) >= 100:
                        fmt = f"${val:,.0f}"
                    ax.text(bar.get_x() + bar.get_width() / 2, val,
                           fmt, ha="center", va=va, fontsize=6.5,
                           fontweight="bold", color=colors[sk])

        ax.set_title(tf_label, fontsize=13, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(strat_names, fontsize=9)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))

        if idx == 0:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    plt.tight_layout(rect=[0, 0.08, 1, 0.94])

    # Add combined totals as a text box at the bottom
    summary_lines = []
    for sk in ["cons_5", "opt_5", "cons_10", "opt_10"]:
        s = scenarios[sk]
        c = s["combined"]
        summary_lines.append(
            f"{labels_map[sk]:20s}:  "
            f"${c['hour']:>+6.2f}/hr  |  ${c['day']:>+7.2f}/day  |  "
            f"${c['week']:>+8.2f}/wk  |  ${c['month']:>+9.2f}/mo"
        )

    fig.text(0.5, 0.02,
             "COMBINED TOTALS\n" + "\n".join(summary_lines),
             ha="center", va="bottom", fontsize=9,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor="#999", alpha=0.9))

    # Add assumptions box
    fig.text(0.01, 0.94,
             "Conservative: WR −8%, 75% fill rate, $2/day maker spread\n"
             "Optimistic: Backtest WR, 95% fill rate, $8/day maker spread",
             fontsize=8, va="top", color="#666",
             fontstyle="italic")

    out = Path(__file__).parent / "chart_profit_projections.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved to {out}")
    plt.close()

    # ============================================================
    # SECOND CHART: Combined totals bar chart (cleaner overview)
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    fig2.suptitle("Maker Bot — Combined Profit Projections\nAll Strategies (Maker + V3.0 5m + V12a 5m + Late Entry 15m)",
                  fontsize=14, fontweight="bold")

    tf_labels2 = ["1 Hour", "1 Day", "1 Week", "1 Month"]
    x2 = np.arange(len(tf_labels2))
    width2 = 0.18

    for i, sk in enumerate(["cons_5", "opt_5", "cons_10", "opt_10"]):
        s = scenarios[sk]
        vals = [s["combined"][tf] for tf in timeframes]
        bars = ax2.bar(x2 + i * width2 - 1.5 * width2, vals, width2,
                      label=labels_map[sk], color=colors[sk], alpha=0.9,
                      edgecolor="white", linewidth=0.8)

        for bar, val in zip(bars, vals):
            fmt = f"${val:,.0f}" if abs(val) >= 100 else f"${val:.2f}"
            ax2.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.01,
                    fmt, ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color=colors[sk])

    ax2.set_xticks(x2)
    ax2.set_xticklabels(tf_labels2, fontsize=12, fontweight="bold")
    ax2.set_ylabel("Projected Profit ($)", fontsize=12)
    ax2.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(y=0, color="gray", linewidth=0.5)

    # Format y-axis
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Add trade count and WR info
    info_text = (
        "Trade Frequency (backtest):  V3.0: ~47/day  |  V12a: ~9/day  |  Late Entry: ~44/day  |  Maker: ~40 pairs/day\n"
        "Win Rates (backtest):  V3.0: 89.3%  |  V12a: 62.1% (unique)  |  Late Entry: 93.2%\n"
        "Conservative: WR −8%, 75% fill rate  |  Optimistic: Backtest WR, 95% fill rate"
    )
    ax2.text(0.5, -0.12, info_text, transform=ax2.transAxes,
            ha="center", va="top", fontsize=9, color="#555",
            fontstyle="italic", fontfamily="monospace")

    plt.tight_layout()
    out2 = Path(__file__).parent / "chart_profit_projections_combined.png"
    fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
    print(f"  Combined chart saved to {out2}")
    plt.close()


if __name__ == "__main__":
    main()
