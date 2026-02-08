"""
Backtest: CBC Gate + BB+RSI + Volume Spike Filters
Analyzes historical paper trades to show PnL impact WITH vs WITHOUT new filters.

CBC Flip: close > prev_high = bullish, close < prev_low = bearish
BB+RSI Combo: Price near BB lower + RSI < 35 = oversold signal (buy UP),
              Price near BB upper + RSI > 65 = overbought signal (buy DOWN)
Volume Spike: volume > 1.5x MA(volume, 20) = confirmed move
"""

import json
import time
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    import requests
except ImportError:
    print("pip install requests")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available, skipping charts")


# ============================================================
# INDICATOR CALCULATIONS
# ============================================================

def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[0]
    for i in range(1, len(values)):
        e = values[i] * k + e * (1 - k)
    return e


def compute_bollinger(closes: List[float], period: int = 20, mult: float = 2.0):
    """Returns (upper, middle, lower, pct_b)"""
    if len(closes) < period:
        return None, None, None, None
    window = closes[-period:]
    sma = sum(window) / period
    variance = sum((x - sma) ** 2 for x in window) / period
    std = variance ** 0.5
    upper = sma + mult * std
    lower = sma - mult * std
    if upper == lower:
        pct_b = 0.5
    else:
        pct_b = (closes[-1] - lower) / (upper - lower)
    return upper, sma, lower, pct_b


def compute_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(len(closes) - period, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses += -diff
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return max(0, min(100, 100 - (100 / (1 + rs))))


def compute_cbc_flip(candles: List[dict]) -> str:
    """
    Candle Body Confirmation (CBC) Flip.
    close > prev_high = BULLISH flip
    close < prev_low = BEARISH flip
    Otherwise = NONE
    """
    if len(candles) < 2:
        return "NONE"
    curr = candles[-1]
    prev = candles[-2]
    if curr["close"] > prev["high"]:
        return "BULLISH"
    elif curr["close"] < prev["low"]:
        return "BEARISH"
    return "NONE"


def compute_volume_spike(volumes: List[float], period: int = 20, threshold: float = 1.5) -> bool:
    """Volume > threshold * MA(volume, period)"""
    if len(volumes) < period + 1:
        return False
    vol_ma = sum(volumes[-(period + 1):-1]) / period
    if vol_ma == 0:
        return False
    return volumes[-1] > threshold * vol_ma


def check_bb_rsi_confirmation(side: str, pct_b: float, rsi: float) -> bool:
    """
    BB+RSI combo: block contradictory entries.
    - UP trade at overbought zone (pct_b > 0.8 + RSI > 65) = BLOCKED
    - DOWN trade at oversold zone (pct_b < 0.2 + RSI < 35) = BLOCKED
    - Everything else passes
    """
    if side == "UP":
        if pct_b > 0.8 and rsi > 65:
            return False
        return True
    elif side == "DOWN":
        if pct_b < 0.2 and rsi < 35:
            return False
        return True
    return True


def check_cbc_confirmation(side: str, cbc_flip: str) -> bool:
    """
    CBC gate: trade direction must align with CBC flip.
    UP needs BULLISH flip, DOWN needs BEARISH flip.
    NONE = no confirmation = BLOCKED.
    """
    if cbc_flip == "NONE":
        return False
    if side == "UP" and cbc_flip == "BULLISH":
        return True
    if side == "DOWN" and cbc_flip == "BEARISH":
        return True
    return False


# ============================================================
# DATA FETCHING
# ============================================================

def get_binance_symbol(asset: str) -> str:
    return {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}.get(asset, "BTCUSDT")


def fetch_candles(asset: str, end_time_iso: str, count: int = 60) -> List[dict]:
    """Fetch 1-minute candles from Binance ending at end_time."""
    symbol = get_binance_symbol(asset)
    end_dt = datetime.fromisoformat(end_time_iso.replace("Z", "+00:00"))
    end_ms = int(end_dt.timestamp() * 1000)
    start_ms = end_ms - count * 60 * 1000

    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": count,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        candles = []
        for k in data:
            candles.append({
                "timestamp": k[0] / 1000,
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        return candles
    except Exception as e:
        print(f"  [API Error] {asset}: {e}")
        return []


# ============================================================
# MAIN BACKTEST
# ============================================================

def run_backtest():
    results_path = Path(__file__).parent / "ta_paper_results.json"
    with open(results_path) as f:
        data = json.load(f)

    trades = data.get("trades", {})
    closed = [t for t in trades.values() if t.get("status") == "closed"]
    closed.sort(key=lambda t: t.get("entry_time", ""))

    print(f"\n{'='*70}")
    print(f"BACKTEST: CBC + BB+RSI + Volume Spike Filter Analysis")
    print(f"{'='*70}")
    print(f"Total closed trades to analyze: {len(closed)}")
    print(f"Fetching 1-min candle data from Binance for each trade...\n")

    results = []
    api_calls = 0

    for i, trade in enumerate(closed):
        title = trade.get("market_title", "")
        side = trade.get("side", "")
        pnl = trade.get("pnl", 0)
        entry_time = trade.get("entry_time", "")
        is_win = pnl > 0

        if "Bitcoin" in title:
            asset = "BTC"
        elif "Ethereum" in title:
            asset = "ETH"
        elif "Solana" in title:
            asset = "SOL"
        else:
            asset = "BTC"

        # Rate limit: ~10/sec for Binance
        if api_calls > 0 and api_calls % 15 == 0:
            time.sleep(1.5)

        candles = fetch_candles(asset, entry_time, count=60)
        api_calls += 1

        if len(candles) < 25:
            print(f"  [{i+1}/{len(closed)}] {asset} {side} -> insufficient data ({len(candles)} candles)")
            results.append({
                "trade": trade, "asset": asset,
                "cbc_pass": True, "bb_rsi_pass": True, "vol_pass": True,
                "all_pass": True, "data_ok": False,
            })
            continue

        closes = [c["close"] for c in candles]
        volumes = [c["volume"] for c in candles]

        # Compute filters
        cbc_flip = compute_cbc_flip(candles)
        cbc_pass = check_cbc_confirmation(side, cbc_flip)

        bb_upper, bb_mid, bb_lower, pct_b = compute_bollinger(closes, 20, 2.0)
        rsi = compute_rsi(closes, 14)
        bb_rsi_pass = True
        if pct_b is not None and rsi is not None:
            bb_rsi_pass = check_bb_rsi_confirmation(side, pct_b, rsi)

        vol_spike = compute_volume_spike(volumes, 20, 1.5)

        all_pass = cbc_pass and bb_rsi_pass and vol_spike

        result = {
            "trade": trade,
            "asset": asset,
            "cbc_flip": cbc_flip,
            "cbc_pass": cbc_pass,
            "pct_b": pct_b,
            "rsi": rsi,
            "bb_rsi_pass": bb_rsi_pass,
            "vol_spike": vol_spike,
            "vol_pass": vol_spike,
            "all_pass": all_pass,
            "data_ok": True,
        }
        results.append(result)

        status = "WIN" if is_win else "LOSS"
        filters = f"CBC={'OK' if cbc_pass else 'BLOCK'} BB={'OK' if bb_rsi_pass else 'BLOCK'} VOL={'OK' if vol_spike else 'BLOCK'}"
        blocked = "" if all_pass else " << FILTERED OUT"
        print(f"  [{i+1}/{len(closed)}] {asset} {side} {status} ${pnl:+.2f} | {filters}{blocked}")

    # ============================================================
    # ANALYSIS
    # ============================================================
    print(f"\n{'='*70}")
    print(f"RESULTS ANALYSIS")
    print(f"{'='*70}")

    valid = [r for r in results if r.get("data_ok", False)]
    total = len(valid)

    curr_wins = sum(1 for r in valid if r["trade"]["pnl"] > 0)
    curr_losses = total - curr_wins
    curr_pnl = sum(r["trade"]["pnl"] for r in valid)
    curr_wr = curr_wins / max(1, total) * 100

    passed = [r for r in valid if r["all_pass"]]
    blocked = [r for r in valid if not r["all_pass"]]
    new_wins = sum(1 for r in passed if r["trade"]["pnl"] > 0)
    new_losses = len(passed) - new_wins
    new_pnl = sum(r["trade"]["pnl"] for r in passed)
    new_wr = new_wins / max(1, len(passed)) * 100

    blocked_wins = sum(1 for r in blocked if r["trade"]["pnl"] > 0)
    blocked_losses = len(blocked) - blocked_wins
    blocked_pnl = sum(r["trade"]["pnl"] for r in blocked)

    print(f"\n--- CURRENT STRATEGY (no new filters) ---")
    print(f"  Trades: {total} | Wins: {curr_wins} | Losses: {curr_losses}")
    print(f"  Win Rate: {curr_wr:.1f}%")
    print(f"  Total PnL: ${curr_pnl:.2f}")
    print(f"  PnL per Trade: ${curr_pnl/max(1,total):.2f}")

    print(f"\n--- WITH ALL FILTERS (CBC + BB+RSI + Volume) ---")
    print(f"  Trades taken: {len(passed)} ({len(passed)/max(1,total)*100:.0f}% of original)")
    print(f"  Wins: {new_wins} | Losses: {new_losses}")
    print(f"  Win Rate: {new_wr:.1f}%")
    print(f"  Total PnL: ${new_pnl:.2f}")
    print(f"  PnL per Trade: ${new_pnl/max(1,len(passed)):.2f}")
    print(f"  PnL Change: ${new_pnl - curr_pnl:+.2f}")

    print(f"\n--- FILTERED OUT TRADES ---")
    print(f"  Total blocked: {len(blocked)}")
    blocked_win_pnl = sum(r['trade']['pnl'] for r in blocked if r['trade']['pnl'] > 0)
    blocked_loss_pnl = sum(r['trade']['pnl'] for r in blocked if r['trade']['pnl'] <= 0)
    print(f"  Blocked wins: {blocked_wins} (lost profit: ${blocked_win_pnl:.2f})")
    print(f"  Blocked losses: {blocked_losses} (saved: ${abs(blocked_loss_pnl):.2f})")
    print(f"  Net value of filter: ${-blocked_pnl:+.2f}")

    # Per-filter analysis
    print(f"\n--- INDIVIDUAL FILTER IMPACT ---")
    for fname, fkey in [("CBC Gate", "cbc_pass"), ("BB+RSI Combo", "bb_rsi_pass"), ("Volume Spike", "vol_pass")]:
        f_passed = [r for r in valid if r[fkey]]
        f_blocked = [r for r in valid if not r[fkey]]
        f_pass_wins = sum(1 for r in f_passed if r["trade"]["pnl"] > 0)
        f_pass_pnl = sum(r["trade"]["pnl"] for r in f_passed)
        f_block_wins = sum(1 for r in f_blocked if r["trade"]["pnl"] > 0)
        f_block_losses = len(f_blocked) - f_block_wins
        f_block_pnl = sum(r["trade"]["pnl"] for r in f_blocked)
        f_wr = f_pass_wins / max(1, len(f_passed)) * 100

        print(f"\n  {fname}:")
        print(f"    Passed: {len(f_passed)} trades, WR: {f_wr:.1f}%, PnL: ${f_pass_pnl:.2f}")
        print(f"    Blocked: {len(f_blocked)} trades ({f_block_wins}W/{f_block_losses}L)")
        print(f"    Blocked PnL: ${f_block_pnl:.2f} (savings: ${-f_block_pnl:+.2f})")
        if f_block_wins > 0:
            f_block_win_pnl = sum(r['trade']['pnl'] for r in f_blocked if r['trade']['pnl'] > 0)
            print(f"    [!] Would filter out {f_block_wins} winning trades (${f_block_win_pnl:.2f})")

    # Combo comparison
    combos = [
        ("CBC only", lambda r: r["cbc_pass"]),
        ("BB+RSI only", lambda r: r["bb_rsi_pass"]),
        ("Volume only", lambda r: r["vol_pass"]),
        ("CBC + Volume", lambda r: r["cbc_pass"] and r["vol_pass"]),
        ("CBC + BB+RSI", lambda r: r["cbc_pass"] and r["bb_rsi_pass"]),
        ("BB+RSI + Volume", lambda r: r["bb_rsi_pass"] and r["vol_pass"]),
        ("ALL THREE", lambda r: r["all_pass"]),
    ]

    print(f"\n--- FILTER COMBINATION COMPARISON ---")
    print(f"  {'Combo':<20} {'Trades':>7} {'WR':>7} {'PnL':>10} {'PnL/Trade':>10}")
    print(f"  {'-'*55}")
    print(f"  {'No filters':<20} {total:>7} {curr_wr:>6.1f}% ${curr_pnl:>9.2f} ${curr_pnl/max(1,total):>9.2f}")
    combo_data = []
    for name, fn in combos:
        cp = [r for r in valid if fn(r)]
        if not cp:
            print(f"  {name:<20} {'0':>7} {'N/A':>7} {'$0.00':>10} {'N/A':>10}")
            combo_data.append((name, 0, 0, 0, 0))
            continue
        cw = sum(1 for r in cp if r["trade"]["pnl"] > 0)
        cpnl = sum(r["trade"]["pnl"] for r in cp)
        cwr = cw / len(cp) * 100
        print(f"  {name:<20} {len(cp):>7} {cwr:>6.1f}% ${cpnl:>9.2f} ${cpnl/len(cp):>9.2f}")
        combo_data.append((name, len(cp), cwr, cpnl, cpnl / len(cp)))

    # Blocked trade details
    print(f"\n--- BLOCKED TRADES DETAIL ---")
    print(f"  {'#':>3} {'Asset':<5} {'Side':<5} {'Result':<5} {'PnL':>8} {'CBC':<8} {'BB%B':>6} {'RSI':>5} {'VolSpk':<6}")
    print(f"  {'-'*60}")
    for idx, r in enumerate(blocked):
        t = r["trade"]
        res = "WIN" if t["pnl"] > 0 else "LOSS"
        cbc = r.get("cbc_flip", "?")
        pb = f"{r.get('pct_b', 0) or 0:.2f}"
        rsi_v = f"{r.get('rsi', 0) or 0:.0f}"
        vs = "YES" if r.get("vol_spike") else "no"
        print(f"  {idx+1:>3} {r['asset']:<5} {t['side']:<5} {res:<5} ${t['pnl']:>+7.2f} {cbc:<8} {pb:>6} {rsi_v:>5} {vs:<6}")

    # ============================================================
    # CHARTS
    # ============================================================
    if HAS_MPL:
        print(f"\nGenerating charts...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Filter Backtest: CBC + BB+RSI + Volume Spike\n(101 Paper Trades)", fontsize=14, fontweight='bold')

        # Chart 1: Cumulative PnL comparison
        ax1 = axes[0][0]
        curr_cum = []
        filt_cum = []
        running_curr = 0
        running_filt = 0
        trade_nums_filt = []
        for i, r in enumerate(valid):
            pnl_val = r["trade"]["pnl"]
            running_curr += pnl_val
            curr_cum.append(running_curr)
            if r["all_pass"]:
                running_filt += pnl_val
                filt_cum.append(running_filt)
                trade_nums_filt.append(i + 1)

        ax1.plot(range(1, len(curr_cum) + 1), curr_cum, 'b-', linewidth=2, label=f'Current ({total} trades, ${curr_pnl:.0f})')
        ax1.plot(trade_nums_filt, filt_cum, 'g-', linewidth=2, label=f'With Filters ({len(passed)} trades, ${new_pnl:.0f})')
        for i, r in enumerate(valid):
            if not r["all_pass"]:
                color = 'red' if r["trade"]["pnl"] > 0 else 'lime'
                marker = 'x' if r["trade"]["pnl"] > 0 else 'o'
                label_text = 'Blocked win' if r["trade"]["pnl"] > 0 and i == next((j for j, rr in enumerate(valid) if not rr["all_pass"] and rr["trade"]["pnl"] > 0), -1) else None
                if label_text is None:
                    label_text = 'Blocked loss' if r["trade"]["pnl"] <= 0 and i == next((j for j, rr in enumerate(valid) if not rr["all_pass"] and rr["trade"]["pnl"] <= 0), -1) else None
                ax1.scatter(i + 1, curr_cum[i], color=color, marker=marker, s=40, zorder=5, alpha=0.8, label=label_text)

        ax1.set_xlabel('Trade #')
        ax1.set_ylabel('Cumulative PnL ($)')
        ax1.set_title('Cumulative PnL: Current vs Filtered')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)

        # Chart 2: Win rate by filter combo
        ax2 = axes[0][1]
        combo_names_short = ["None", "CBC", "BB+RSI", "Vol", "CBC+V", "CBC+BB", "BB+V", "ALL"]
        combo_wrs = [curr_wr] + [cd[2] for cd in combo_data]
        combo_counts = [total] + [cd[1] for cd in combo_data]
        colors = ['#4488cc'] + ['#66aa66'] * 7
        bars = ax2.bar(range(len(combo_names_short)), combo_wrs, color=colors, edgecolor='black', alpha=0.8)
        ax2.set_xticks(range(len(combo_names_short)))
        ax2.set_xticklabels(combo_names_short, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('Win Rate by Filter Combination')
        ax2.axhline(y=curr_wr, color='blue', linestyle='--', alpha=0.5, label=f'Current: {curr_wr:.1f}%')
        for bar, wr, cnt in zip(bars, combo_wrs, combo_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{wr:.0f}%\n({cnt})', ha='center', va='bottom', fontsize=8)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Chart 3: PnL per trade by combo
        ax3 = axes[1][0]
        combo_ppt = [curr_pnl / max(1, total)] + [cd[4] for cd in combo_data]
        bars3 = ax3.bar(range(len(combo_names_short)), combo_ppt,
                       color=['#cc4444' if p < combo_ppt[0] else '#44aa44' for p in combo_ppt],
                       edgecolor='black', alpha=0.8)
        ax3.set_xticks(range(len(combo_names_short)))
        ax3.set_xticklabels(combo_names_short, rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('PnL per Trade ($)')
        ax3.set_title('PnL per Trade by Filter Combination')
        ax3.axhline(y=combo_ppt[0], color='blue', linestyle='--', alpha=0.5, label=f'Current: ${combo_ppt[0]:.2f}')
        for bar, ppt in zip(bars3, combo_ppt):
            ax3.text(bar.get_x() + bar.get_width()/2,
                    max(bar.get_height(), 0) + 0.2,
                    f'${ppt:.2f}', ha='center', va='bottom', fontsize=8)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Chart 4: Pie chart of what got filtered
        ax4 = axes[1][1]
        if blocked:
            labels = [
                f'Blocked Wins ({blocked_wins})\n${blocked_win_pnl:.0f} lost',
                f'Blocked Losses ({blocked_losses})\n${abs(blocked_loss_pnl):.0f} saved',
                f'Passed Wins ({new_wins})',
                f'Passed Losses ({new_losses})',
            ]
            sizes = [max(blocked_wins, 0.1), max(blocked_losses, 0.1),
                    max(new_wins, 0.1), max(new_losses, 0.1)]
            colors4 = ['#ff6666', '#66ff66', '#4488cc', '#cc8844']
            explode = (0.05, 0.1, 0, 0)
            ax4.pie(sizes, explode=explode, labels=labels, colors=colors4,
                   autopct='%1.0f%%', shadow=True, startangle=90, textprops={'fontsize': 8})
        else:
            ax4.text(0.5, 0.5, 'No trades filtered', ha='center', va='center', fontsize=14)
        ax4.set_title('Filter Impact: What Gets Blocked')

        plt.tight_layout()
        chart_path = Path(__file__).parent / "filter_backtest_results.png"
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"\nCharts saved to: {chart_path}")
        plt.close()

    # Save detailed results JSON
    output = {
        "backtest_time": datetime.now(timezone.utc).isoformat(),
        "total_trades_analyzed": total,
        "current_strategy": {
            "trades": total, "wins": curr_wins, "losses": curr_losses,
            "wr": round(curr_wr, 1), "pnl": round(curr_pnl, 2),
            "pnl_per_trade": round(curr_pnl / max(1, total), 2),
        },
        "with_all_filters": {
            "trades": len(passed), "wins": new_wins, "losses": new_losses,
            "wr": round(new_wr, 1), "pnl": round(new_pnl, 2),
            "pnl_per_trade": round(new_pnl / max(1, len(passed)), 2),
        },
        "filtered_out": {
            "total": len(blocked), "wins_lost": blocked_wins, "losses_saved": blocked_losses,
            "pnl_of_blocked": round(blocked_pnl, 2),
            "net_filter_value": round(-blocked_pnl, 2),
        },
        "per_trade": [{
            "market": r["trade"].get("market_title", ""),
            "side": r["trade"].get("side", ""),
            "pnl": round(r["trade"]["pnl"], 2),
            "is_win": r["trade"]["pnl"] > 0,
            "cbc_flip": r.get("cbc_flip", ""),
            "cbc_pass": r["cbc_pass"],
            "pct_b": round(r.get("pct_b", 0) or 0, 3),
            "rsi": round(r.get("rsi", 0) or 0, 1),
            "bb_rsi_pass": r["bb_rsi_pass"],
            "vol_spike": r.get("vol_spike", False),
            "all_pass": r["all_pass"],
        } for r in valid],
    }

    detail_path = Path(__file__).parent / "filter_backtest_detail.json"
    with open(detail_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Detailed results saved to: {detail_path}")

    return output


if __name__ == "__main__":
    run_backtest()
