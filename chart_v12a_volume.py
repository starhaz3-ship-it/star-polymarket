"""
Chart: V12a Order Flow performance at different volume surge thresholds.
Shows WR, PnL, trades, and other stats for 1.5x, 2x, 3x volume.
"""

import json
import math
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# DATA FETCHING
# ============================================================

def fetch_binance_klines(symbol="BTCUSDT", interval="1m", days=7):
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    current = start_ms
    while current < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": current, "limit": 1000}
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if not data:
            break
        all_klines.extend(data)
        current = data[-1][0] + 60000
        time.sleep(0.15)
    bars = []
    for k in all_klines:
        total_vol = float(k[5])
        taker_buy = float(k[9])
        bars.append({
            "ts": k[0], "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]),
            "volume": total_vol, "taker_buy": taker_buy,
            "taker_sell": total_vol - taker_buy,
        })
    return bars


def aggregate_5m(bars_1m):
    bars_5m = []
    for i in range(0, len(bars_1m) - 4, 5):
        chunk = bars_1m[i:i + 5]
        if len(chunk) < 5:
            break
        bars_5m.append({
            "ts": chunk[0]["ts"],
            "open": chunk[0]["open"],
            "high": max(c["high"] for c in chunk),
            "low": min(c["low"] for c in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(c["volume"] for c in chunk),
            "taker_buy": sum(c["taker_buy"] for c in chunk),
            "taker_sell": sum(c["taker_sell"] for c in chunk),
            "minutes": chunk,
        })
    return bars_5m


def get_v30_bars(bars_5m):
    v30 = set()
    for i, bar in enumerate(bars_5m):
        mins = bar["minutes"]
        if len(mins) < 5:
            continue
        op = mins[0]["open"]
        p2 = mins[1]["close"]
        mom = (p2 - op) / op * 10000
        if abs(mom) >= 15.0:
            v30.add(i)
    return v30


def generate_all_flow_trades(bars_5m, bars_1m):
    n1 = len(bars_1m)
    vols = np.array([b["volume"] for b in bars_1m], dtype=float)
    vol_sma = np.full_like(vols, np.nan)
    for idx in range(19, len(vols)):
        vol_sma[idx] = np.mean(vols[idx - 19:idx + 1])

    trades = []
    for i, bar in enumerate(bars_5m):
        mins = bar["minutes"]
        if len(mins) < 5:
            continue
        m0, m1 = mins[0], mins[1]
        total_vol = m0["volume"] + m1["volume"]
        total_buy = m0["taker_buy"] + m1["taker_buy"]
        if total_vol < 1e-9:
            continue
        buy_ratio = total_buy / total_vol
        m1_global = i * 5 + 1
        if m1_global >= n1 or not np.isfinite(vol_sma[m1_global]) or vol_sma[m1_global] <= 0:
            continue
        avg_2min_vol = vol_sma[m1_global] * 2
        vol_ratio = total_vol / max(1e-9, avg_2min_vol)
        m0_br = m0["taker_buy"] / max(1e-9, m0["volume"])
        m1_br = m1["taker_buy"] / max(1e-9, m1["volume"])
        flow_consistent = (m0_br > 0.5 and m1_br > 0.5) or (m0_br < 0.5 and m1_br < 0.5)
        actual_up = bar["close"] > bar["open"]
        trades.append({
            "bar_idx": i, "buy_ratio": buy_ratio, "vol_ratio": vol_ratio,
            "flow_consistent": flow_consistent, "actual_up": actual_up,
        })
    return trades


def compute_stats(all_trades, v30_bars, buy_t, vol_t, consist):
    BUY_PRICE = 0.51
    SIZE = 5.0
    sell_t = 1.0 - buy_t
    sh = max(1, math.floor(SIZE / BUY_PRICE))

    unique_trades = []
    all_filtered = []

    for t in all_trades:
        if t["vol_ratio"] < vol_t:
            continue
        if consist and not t["flow_consistent"]:
            continue
        if t["buy_ratio"] >= buy_t:
            side_up = True
        elif t["buy_ratio"] <= sell_t:
            side_up = False
        else:
            continue

        win = (side_up and t["actual_up"]) or (not side_up and not t["actual_up"])
        cost = BUY_PRICE * sh
        pnl = (sh - cost) if win else -cost

        entry = {"win": win, "pnl": pnl, "cost": cost, "bar_idx": t["bar_idx"],
                 "direction": "UP" if side_up else "DOWN"}
        all_filtered.append(entry)
        if t["bar_idx"] not in v30_bars:
            unique_trades.append(entry)

    def stats(trades):
        if not trades:
            return {"trades": 0, "wins": 0, "wr": 0, "pnl": 0, "roi": 0,
                    "max_dd": 0, "tpd": 0, "max_ws": 0, "max_ls": 0,
                    "avg_pnl": 0, "up_trades": 0, "up_wr": 0, "dn_trades": 0, "dn_wr": 0}
        total = len(trades)
        wins = sum(1 for t in trades if t["win"])
        wr = wins / total * 100
        total_pnl = sum(t["pnl"] for t in trades)
        total_cost = sum(t["cost"] for t in trades)
        roi = total_pnl / total_cost * 100 if total_cost > 0 else 0
        cum = peak = max_dd = 0
        for t in trades:
            cum += t["pnl"]
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)
        mws = mls = cw = cl = 0
        for t in trades:
            if t["win"]:
                cw += 1; cl = 0
            else:
                cl += 1; cw = 0
            mws = max(mws, cw)
            mls = max(mls, cl)
        ups = [t for t in trades if t["direction"] == "UP"]
        dns = [t for t in trades if t["direction"] == "DOWN"]
        up_wr = sum(1 for t in ups if t["win"]) / len(ups) * 100 if ups else 0
        dn_wr = sum(1 for t in dns if t["win"]) / len(dns) * 100 if dns else 0
        return {"trades": total, "wins": wins, "wr": wr, "pnl": total_pnl,
                "roi": roi, "max_dd": max_dd, "tpd": total / 7.0,
                "max_ws": mws, "max_ls": mls, "avg_pnl": total_pnl / total,
                "up_trades": len(ups), "up_wr": up_wr,
                "dn_trades": len(dns), "dn_wr": dn_wr}

    return stats(unique_trades), stats(all_filtered)


def main():
    print("Fetching 7 days of BTC 1-min data from Binance...")
    bars_1m = fetch_binance_klines(days=7)
    bars_5m = aggregate_5m(bars_1m)
    v30_bars = get_v30_bars(bars_5m)
    all_trades = generate_all_flow_trades(bars_5m, bars_1m)
    print(f"  {len(bars_5m)} 5m bars, {len(v30_bars)} V3.0 bars, {len(all_trades)} flow candidates")

    # Compute stats at each volume level (using best buy_ratio per sweep results)
    vol_levels = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    buy_t = 0.56  # best from sweep
    consist = False

    unique_stats = []
    all_stats = []

    for vt in vol_levels:
        u, a = compute_stats(all_trades, v30_bars, buy_t, vt, consist)
        unique_stats.append(u)
        all_stats.append(a)
        print(f"  Vol>={vt:.1f}x: Unique {u['trades']}T {u['wr']:.1f}%WR ${u['pnl']:+.2f} | "
              f"All {a['trades']}T {a['wr']:.1f}%WR ${a['pnl']:+.2f}")

    # ============================================================
    # CHART
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("V12a Order Flow: Performance by Volume Surge Threshold\n"
                 f"(buy_ratio >= {buy_t}, NON-OVERLAPPING with V3.0 only)\n"
                 "7-day BTC backtest | $5 sizing",
                 fontsize=14, fontweight='bold')

    labels = [f"{v:.1f}x" for v in vol_levels]
    x = np.arange(len(vol_levels))
    width = 0.55

    colors_wr = ['#e74c3c' if s['wr'] < 60 else '#f39c12' if s['wr'] < 70 else
                 '#27ae60' if s['wr'] < 80 else '#2ecc71' for s in unique_stats]
    colors_pnl = ['#e74c3c' if s['pnl'] < 0 else '#27ae60' for s in unique_stats]

    # 1. Win Rate
    ax = axes[0, 0]
    wrs = [s['wr'] for s in unique_stats]
    bars_plot = ax.bar(x, wrs, width, color=colors_wr, edgecolor='white', linewidth=0.5)
    ax.axhline(y=80, color='#2ecc71', linestyle='--', linewidth=1.5, alpha=0.7, label='80% target')
    ax.axhline(y=70, color='#f39c12', linestyle='--', linewidth=1.5, alpha=0.7, label='70% target')
    ax.axhline(y=51.3, color='#e74c3c', linestyle=':', linewidth=1, alpha=0.5, label='Coin flip (51.3%)')
    for i, (bar, wr) in enumerate(zip(bars_plot, wrs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{wr:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylabel('Win Rate %', fontsize=12)
    ax.set_title('Win Rate', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Volume Surge Threshold')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    # 2. Total PnL
    ax = axes[0, 1]
    pnls = [s['pnl'] for s in unique_stats]
    bars_plot = ax.bar(x, pnls, width, color=colors_pnl, edgecolor='white', linewidth=0.5)
    for i, (bar, pnl) in enumerate(zip(bars_plot, pnls)):
        y_pos = bar.get_height() + 2 if pnl >= 0 else bar.get_height() - 8
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'${pnl:+.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylabel('PnL ($)', fontsize=12)
    ax.set_title('Total PnL (unique trades)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Volume Surge Threshold')
    ax.axhline(y=0, color='gray', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

    # 3. Trade Count
    ax = axes[0, 2]
    trades_u = [s['trades'] for s in unique_stats]
    trades_a = [s['trades'] for s in all_stats]
    b1 = ax.bar(x - width/3, trades_a, width*0.6, label='All trades', color='#3498db',
                edgecolor='white', linewidth=0.5, alpha=0.7)
    b2 = ax.bar(x + width/3, trades_u, width*0.6, label='Unique (non-V3.0)', color='#e67e22',
                edgecolor='white', linewidth=0.5)
    for bar, val in zip(b1, trades_a):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha='center', va='bottom', fontsize=9, color='#3498db')
    for bar, val in zip(b2, trades_u):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#e67e22')
    ax.set_ylabel('Trade Count', fontsize=12)
    ax.set_title('Trade Frequency (7 days)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Volume Surge Threshold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 4. ROI %
    ax = axes[1, 0]
    rois = [s['roi'] for s in unique_stats]
    colors_roi = ['#e74c3c' if r < 0 else '#27ae60' for r in rois]
    bars_plot = ax.bar(x, rois, width, color=colors_roi, edgecolor='white', linewidth=0.5)
    for bar, roi in zip(bars_plot, rois):
        y_pos = bar.get_height() + 1 if roi >= 0 else bar.get_height() - 3
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{roi:+.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylabel('ROI %', fontsize=12)
    ax.set_title('Return on Investment', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Volume Surge Threshold')
    ax.axhline(y=0, color='gray', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

    # 5. Max Drawdown
    ax = axes[1, 1]
    dds = [s['max_dd'] for s in unique_stats]
    bars_plot = ax.bar(x, dds, width, color='#e74c3c', edgecolor='white', linewidth=0.5, alpha=0.8)
    for bar, dd in zip(bars_plot, dds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'${dd:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax.set_ylabel('Max Drawdown ($)', fontsize=12)
    ax.set_title('Max Drawdown', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Volume Surge Threshold')
    ax.grid(axis='y', alpha=0.3)

    # 6. Trades per Day + Avg PnL/trade (dual axis)
    ax = axes[1, 2]
    tpds = [s['tpd'] for s in unique_stats]
    avg_pnls = [s['avg_pnl'] for s in unique_stats]

    color1 = '#3498db'
    color2 = '#e67e22'
    b1 = ax.bar(x - width/3, tpds, width*0.6, color=color1, edgecolor='white',
                linewidth=0.5, label='Trades/day')
    for bar, val in zip(b1, tpds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, color=color1, fontweight='bold')
    ax.set_ylabel('Trades / Day', fontsize=12, color=color1)
    ax.set_xlabel('Volume Surge Threshold')
    ax.set_title('Frequency vs Avg PnL/Trade', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='y', labelcolor=color1)

    ax2 = ax.twinx()
    b2 = ax2.bar(x + width/3, avg_pnls, width*0.6, color=color2, edgecolor='white',
                 linewidth=0.5, alpha=0.8, label='Avg $/trade')
    for bar, val in zip(b2, avg_pnls):
        y_pos = bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.15
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'${val:+.2f}', ha='center', va='bottom', fontsize=9, color=color2, fontweight='bold')
    ax2.set_ylabel('Avg PnL / Trade ($)', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.axhline(y=0, color='gray', linewidth=0.5)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = Path(__file__).parent / "chart_v12a_volume_threshold.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n  Chart saved to {out_path}")
    plt.close()

    # Also print summary table
    print("\n" + "=" * 95)
    print("  V12a Order Flow — NON-OVERLAPPING trades by Volume Surge Threshold")
    print("  (buy_ratio >= 0.56, 7-day BTC backtest, $5 sizing)")
    print("=" * 95)
    print(f"  {'Vol':>5} | {'Trades':>7} {'T/day':>6} | {'WR%':>6} {'Wins':>5} {'Loss':>5} | "
          f"{'PnL':>8} {'ROI%':>7} | {'MaxDD':>7} {'AvgPnL':>7} | {'WS':>3} {'LS':>3}")
    print(f"  {'-'*5} | {'-'*7} {'-'*6} | {'-'*6} {'-'*5} {'-'*5} | "
          f"{'-'*8} {'-'*7} | {'-'*7} {'-'*7} | {'-'*3} {'-'*3}")
    for vt, s in zip(vol_levels, unique_stats):
        flag = " <-- BEST" if s['wr'] >= 80 else ""
        print(f"  {vt:>4.1f}x | {s['trades']:>7} {s['tpd']:>5.1f} | {s['wr']:>5.1f}% {s['wins']:>5} "
              f"{s['trades']-s['wins']:>5} | ${s['pnl']:>+7.2f} {s['roi']:>+6.1f}% | "
              f"${s['max_dd']:>6.2f} ${s['avg_pnl']:>+6.2f} | {s['max_ws']:>3} {s['max_ls']:>3}{flag}")


if __name__ == "__main__":
    main()
