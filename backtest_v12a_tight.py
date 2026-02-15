"""
V12a Tightened: Sweep buy_ratio and vol_ratio thresholds to find
the sweet spot with 80%+ WR on trades that DON'T overlap with V3.0.
"""

import json
import math
import time
import requests
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from itertools import product


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
            "num_trades": int(k[8]),
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


# ============================================================
# Build V3.0 bar set (for overlap exclusion)
# ============================================================

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


# ============================================================
# Generate ALL V12a candidate trades (loose filters)
# ============================================================

def generate_all_flow_trades(bars_5m, bars_1m):
    """Generate all trades with their features, no filtering yet."""
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

        m0 = mins[0]
        m1 = mins[1]
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

        # Also compute: trade count ratio, delta magnitude, minute-1 vs minute-0 flow
        m0_buy_ratio = m0["taker_buy"] / max(1e-9, m0["volume"])
        m1_buy_ratio = m1["taker_buy"] / max(1e-9, m1["volume"])
        flow_consistent = (m0_buy_ratio > 0.5 and m1_buy_ratio > 0.5) or \
                          (m0_buy_ratio < 0.5 and m1_buy_ratio < 0.5)

        # Delta as % of volume (how lopsided)
        delta_pct = abs(buy_ratio - 0.5) * 2  # 0 = even, 1 = all one side

        # Momentum (for reference/filtering)
        mom_bps = (m1["close"] - m0["open"]) / m0["open"] * 10000

        # Settlement
        actual_up = bar["close"] > bar["open"]

        trades.append({
            "bar_idx": i,
            "buy_ratio": buy_ratio,
            "vol_ratio": vol_ratio,
            "delta_pct": delta_pct,
            "flow_consistent": flow_consistent,
            "m0_buy_ratio": m0_buy_ratio,
            "m1_buy_ratio": m1_buy_ratio,
            "mom_bps": mom_bps,
            "actual_up": actual_up,
        })

    return trades


# ============================================================
# PARAMETER SWEEP
# ============================================================

def sweep(all_trades, v30_bars):
    """
    Sweep buy_ratio threshold, vol_ratio threshold, and other filters.
    Report WR for NON-OVERLAPPING trades only.
    """
    BUY_PRICE = 0.51

    buy_thresholds = [0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
    vol_thresholds = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    consistency_opts = [False, True]  # require both minutes to agree

    results = []

    for buy_t, vol_t, consist in product(buy_thresholds, vol_thresholds, consistency_opts):
        sell_t = 1.0 - buy_t  # symmetric

        unique_wins = 0
        unique_total = 0
        all_wins = 0
        all_total = 0

        for t in all_trades:
            # Apply filters
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
            all_total += 1
            all_wins += 1 if win else 0

            # Non-overlapping only
            if t["bar_idx"] not in v30_bars:
                unique_total += 1
                unique_wins += 1 if win else 0

        if unique_total >= 5:  # need minimum trades
            unique_wr = unique_wins / unique_total * 100
            all_wr = all_wins / all_total * 100 if all_total > 0 else 0
            unique_pnl = unique_wins * (9 * BUY_PRICE) - (unique_total - unique_wins) * (9 * BUY_PRICE)
            # At $5 sizing: shares=9, win=$4.41, loss=-$4.59
            unique_pnl_real = unique_wins * 4.41 + (unique_total - unique_wins) * (-4.59)

            results.append({
                "buy_t": buy_t, "vol_t": vol_t, "consist": consist,
                "unique_trades": unique_total, "unique_wr": unique_wr,
                "unique_pnl": unique_pnl_real,
                "all_trades": all_total, "all_wr": all_wr,
                "unique_tpd": unique_total / 7.0,
            })

    # Sort by unique WR descending, then by unique trades descending
    results.sort(key=lambda r: (-r["unique_wr"], -r["unique_trades"]))
    return results


def main():
    print("Fetching 7 days of BTC 1-min data from Binance (with taker volume)...")
    bars_1m = fetch_binance_klines(days=7)
    print(f"  Got {len(bars_1m)} 1-min bars")
    bars_5m = aggregate_5m(bars_1m)
    print(f"  Aggregated to {len(bars_5m)} 5-min bars")

    date_start = datetime.fromtimestamp(bars_1m[0]["ts"] / 1000, tz=timezone.utc)
    date_end = datetime.fromtimestamp(bars_1m[-1]["ts"] / 1000, tz=timezone.utc)
    natural_up = sum(1 for b in bars_5m if b["close"] > b["open"]) / len(bars_5m) * 100
    print(f"  Range: {date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}")
    print(f"  Natural UP%: {natural_up:.1f}%")

    # V3.0 bars for overlap exclusion
    v30_bars = get_v30_bars(bars_5m)
    print(f"  V3.0 trades on {len(v30_bars)} bars")

    # Generate all candidate trades
    all_trades = generate_all_flow_trades(bars_5m, bars_1m)
    print(f"  Generated {len(all_trades)} candidate flow observations")

    # Sweep
    print("\n" + "=" * 90)
    print("  PARAMETER SWEEP — V12a NON-OVERLAPPING trades only (excluding V3.0 bars)")
    print("=" * 90)
    print(f"  {'BuyT':>6} {'VolT':>6} {'Consist':>8} | {'UniqueT':>8} {'UniqueWR':>9} "
          f"{'UniquePnL':>10} {'T/day':>6} | {'AllT':>6} {'AllWR':>7}")
    print(f"  {'-'*6} {'-'*6} {'-'*8} | {'-'*8} {'-'*9} {'-'*10} {'-'*6} | {'-'*6} {'-'*7}")

    results = sweep(all_trades, v30_bars)

    # Show top 30 results
    shown = 0
    for r in results:
        if shown >= 30:
            break
        flag = " ***" if r["unique_wr"] >= 80 else (" ** " if r["unique_wr"] >= 70 else "    ")
        print(f"  {r['buy_t']:>5.2f} {r['vol_t']:>6.1f} {str(r['consist']):>8} | "
              f"{r['unique_trades']:>8} {r['unique_wr']:>8.1f}% "
              f"${r['unique_pnl']:>+8.2f} {r['unique_tpd']:>5.1f} | "
              f"{r['all_trades']:>6} {r['all_wr']:>6.1f}%{flag}")
        shown += 1

    # Find best 80%+ WR config
    best_80 = [r for r in results if r["unique_wr"] >= 80 and r["unique_trades"] >= 5]
    best_70 = [r for r in results if r["unique_wr"] >= 70 and r["unique_trades"] >= 10]

    print("\n" + "=" * 90)
    print("  BEST CONFIGS")
    print("=" * 90)

    if best_80:
        # Sort by trades descending (most trades at 80%+)
        best_80.sort(key=lambda r: -r["unique_trades"])
        b = best_80[0]
        print(f"\n  BEST 80%+ WR (most trades):")
        print(f"    buy_ratio >= {b['buy_t']:.2f}, vol_ratio >= {b['vol_t']:.1f}, consistent={b['consist']}")
        print(f"    {b['unique_trades']} unique trades ({b['unique_tpd']:.1f}/day), "
              f"{b['unique_wr']:.1f}% WR, ${b['unique_pnl']:+.2f} PnL")
        print(f"    All trades (incl overlap): {b['all_trades']}T, {b['all_wr']:.1f}% WR")
    else:
        print(f"\n  No config achieved 80%+ WR with 5+ unique trades")

    if best_70:
        best_70.sort(key=lambda r: -r["unique_trades"])
        b = best_70[0]
        print(f"\n  BEST 70%+ WR (most trades):")
        print(f"    buy_ratio >= {b['buy_t']:.2f}, vol_ratio >= {b['vol_t']:.1f}, consistent={b['consist']}")
        print(f"    {b['unique_trades']} unique trades ({b['unique_tpd']:.1f}/day), "
              f"{b['unique_wr']:.1f}% WR, ${b['unique_pnl']:+.2f} PnL")

        # Also find best PnL at 70%+
        best_70_pnl = sorted(best_70, key=lambda r: -r["unique_pnl"])
        bp = best_70_pnl[0]
        print(f"\n  BEST 70%+ WR (most PnL):")
        print(f"    buy_ratio >= {bp['buy_t']:.2f}, vol_ratio >= {bp['vol_t']:.1f}, consistent={bp['consist']}")
        print(f"    {bp['unique_trades']} unique trades ({bp['unique_tpd']:.1f}/day), "
              f"{bp['unique_wr']:.1f}% WR, ${bp['unique_pnl']:+.2f} PnL")

    # ================================================================
    # Now run the BEST config as a proper backtest with full stats
    # ================================================================
    if best_80:
        winner = best_80[0]
    elif best_70:
        winner = best_70[0]
    else:
        print("\n  No viable config found.")
        return

    print("\n" + "=" * 90)
    print(f"  FULL BACKTEST — V12a* (buy>={winner['buy_t']:.2f}, vol>={winner['vol_t']:.1f}, "
          f"consist={winner['consist']})")
    print("=" * 90)

    BUY_PRICE = 0.51
    SIZE = 5.0
    buy_t = winner["buy_t"]
    sell_t = 1.0 - buy_t
    vol_t = winner["vol_t"]
    consist = winner["consist"]

    v30_trades = []
    v12_trades = []
    v12_unique_trades = []

    for t in all_trades:
        if t["vol_ratio"] < vol_t:
            continue
        if consist and not t["flow_consistent"]:
            continue
        if t["buy_ratio"] >= buy_t:
            side = "UP"
        elif t["buy_ratio"] <= sell_t:
            side = "DOWN"
        else:
            continue

        actual = "UP" if t["actual_up"] else "DOWN"
        win = side == actual
        sh = max(1, math.floor(SIZE / BUY_PRICE))
        cost = BUY_PRICE * sh
        pnl = (sh - cost) if win else -cost

        trade = {
            "bar_idx": t["bar_idx"], "direction": side, "actual": actual,
            "win": win, "buy_ratio": t["buy_ratio"], "vol_ratio": t["vol_ratio"],
            "flow_consistent": t["flow_consistent"], "mom_bps": t["mom_bps"],
            "pnl": pnl, "cost": cost,
        }

        v12_trades.append(trade)
        if t["bar_idx"] not in v30_bars:
            v12_unique_trades.append(trade)

    # V3.0 trades for comparison
    for i, bar in enumerate(bars_5m):
        mins = bar["minutes"]
        if len(mins) < 5:
            continue
        op = mins[0]["open"]
        p2 = mins[1]["close"]
        mom = (p2 - op) / op * 10000
        if abs(mom) < 15.0:
            continue
        d = "UP" if mom > 0 else "DOWN"
        actual = "UP" if bar["close"] > op else "DOWN"
        sh = max(1, math.floor(SIZE / BUY_PRICE))
        cost = BUY_PRICE * sh
        pnl = (sh - cost) if d == actual else -cost
        v30_trades.append({"direction": d, "actual": actual, "win": d == actual,
                           "pnl": pnl, "cost": cost})

    def report(trades, label):
        if not trades:
            print(f"  {label}: NO TRADES")
            return
        total = len(trades)
        wins = sum(1 for t in trades if t["win"])
        wr = wins / total * 100
        total_pnl = sum(t["pnl"] for t in trades)
        total_cost = sum(t["cost"] for t in trades)

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

        roi = total_pnl / total_cost * 100 if total_cost > 0 else 0
        print(f"\n  {label}")
        print(f"    Trades: {total} ({total/7:.1f}/day) | Wins: {wins} ({wr:.1f}%)")
        print(f"    PnL: ${total_pnl:+,.2f} | ROI: {roi:+.1f}% | MaxDD: ${max_dd:.2f}")
        print(f"    Win streak: {mws} | Loss streak: {mls}")

        ups = [t for t in trades if t["direction"] == "UP"]
        dns = [t for t in trades if t["direction"] == "DOWN"]
        if ups:
            uwr = sum(1 for t in ups if t["win"]) / len(ups) * 100
            print(f"    UP: {len(ups)}T ({uwr:.1f}% WR)")
        if dns:
            dwr = sum(1 for t in dns if t["win"]) / len(dns) * 100
            print(f"    DOWN: {len(dns)}T ({dwr:.1f}% WR)")

    report(v30_trades, "V3.0 - Simple Momentum")
    report(v12_trades, "V12a* - Tightened Flow (ALL trades)")
    report(v12_unique_trades, "V12a* - UNIQUE ONLY (non-overlapping with V3.0)")

    # Combined: V3.0 + V12a* unique
    combined = v30_trades + v12_unique_trades
    report(combined, "COMBINED: V3.0 + V12a* unique")

    # Added PnL from stacking
    v30_pnl = sum(t["pnl"] for t in v30_trades)
    unique_pnl = sum(t["pnl"] for t in v12_unique_trades)
    print(f"\n  STACKING VALUE:")
    print(f"    V3.0 alone:       ${v30_pnl:+,.2f}")
    print(f"    + V12a* unique:   ${unique_pnl:+,.2f}")
    print(f"    = Combined:       ${v30_pnl + unique_pnl:+,.2f} "
          f"(+{unique_pnl/max(1,v30_pnl)*100:.0f}% more)")

    # Save results
    results_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": f"{date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}",
        "bars_1m": len(bars_1m), "bars_5m": len(bars_5m),
        "natural_up_pct": round(natural_up, 2),
        "best_config": {
            "buy_ratio_threshold": winner["buy_t"],
            "vol_ratio_threshold": winner["vol_t"],
            "require_consistency": winner["consist"],
        },
        "v30_trades": len(v30_trades),
        "v30_wr": round(sum(1 for t in v30_trades if t["win"]) / len(v30_trades) * 100, 2) if v30_trades else 0,
        "v30_pnl": round(v30_pnl, 2),
        "v12a_all_trades": len(v12_trades),
        "v12a_all_wr": round(sum(1 for t in v12_trades if t["win"]) / len(v12_trades) * 100, 2) if v12_trades else 0,
        "v12a_unique_trades": len(v12_unique_trades),
        "v12a_unique_wr": round(sum(1 for t in v12_unique_trades if t["win"]) / len(v12_unique_trades) * 100, 2) if v12_unique_trades else 0,
        "v12a_unique_pnl": round(unique_pnl, 2),
        "combined_pnl": round(v30_pnl + unique_pnl, 2),
    }
    out = Path(__file__).parent / "backtest_v12a_tight_results.json"
    out.write_text(json.dumps(results_data, indent=2))
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
