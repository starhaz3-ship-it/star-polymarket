"""
Focused backtest: Late Entry strategy for 15-minute BTC Polymarket.

Key finding: Waiting 10 minutes into a 15m bar gives 90%+ WR.
This script does a full parameter sweep to find the optimal config.
"""

import json
import math
import time
import requests
import numpy as np
from datetime import datetime, timezone
from pathlib import Path


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


def aggregate_15m(bars_1m):
    bars = []
    for i in range(0, len(bars_1m) - 14, 15):
        chunk = bars_1m[i:i + 15]
        if len(chunk) < 15:
            break
        bars.append({
            "ts": chunk[0]["ts"],
            "open": chunk[0]["open"],
            "high": max(c["high"] for c in chunk),
            "low": min(c["low"] for c in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(c["volume"] for c in chunk),
            "taker_buy": sum(c["taker_buy"] for c in chunk),
            "minutes": chunk,
        })
    return bars


BUY_PRICE = 0.51
SIZE_USD = 5.0


def simulate_late_entry(bars_15m, wait_min=10, threshold_bps=15.0,
                        trend_filter=False, trend_lookback=4,
                        vol_filter=False, vol_min=1.5, bars_1m=None,
                        flow_filter=False, flow_thresh=0.52):
    """Late entry momentum on 15m bars with optional filters."""
    sh = max(1, math.floor(SIZE_USD / BUY_PRICE))
    cost = BUY_PRICE * sh
    win_pnl = sh - cost
    loss_pnl = -cost

    # Precompute volume SMA if needed
    vol_sma = None
    if vol_filter and bars_1m:
        vols = np.array([b["volume"] for b in bars_1m], dtype=float)
        vol_sma = np.full_like(vols, np.nan)
        for i in range(19, len(vols)):
            vol_sma[i] = np.mean(vols[i - 19:i + 1])

    trades = []
    start_idx = trend_lookback if trend_filter else 0

    for idx in range(start_idx, len(bars_15m)):
        bar = bars_15m[idx]
        mins = bar["minutes"]
        if len(mins) < wait_min + 1:
            continue

        op = mins[0]["open"]
        px = mins[wait_min - 1]["close"]  # price after wait_min minutes
        mom = (px - op) / op * 10000

        if abs(mom) < threshold_bps:
            continue

        signal = "UP" if mom > 0 else "DOWN"

        # Optional: trend alignment filter
        if trend_filter:
            prior_closes = [bars_15m[j]["close"] for j in range(idx - trend_lookback, idx)]
            trend_up = prior_closes[-1] > prior_closes[0]
            if signal == "UP" and not trend_up:
                continue
            if signal == "DOWN" and trend_up:
                continue

        # Optional: volume filter
        if vol_filter and vol_sma is not None:
            m_global = idx * 15 + wait_min - 1
            if m_global >= len(vol_sma) or not np.isfinite(vol_sma[m_global]) or vol_sma[m_global] <= 0:
                continue
            v_sum = sum(mins[j]["volume"] for j in range(wait_min))
            vr = v_sum / (vol_sma[m_global] * wait_min)
            if vr < vol_min:
                continue

        # Optional: order flow confirmation
        if flow_filter:
            total_buy = sum(mins[j]["taker_buy"] for j in range(wait_min))
            total_vol = sum(mins[j]["volume"] for j in range(wait_min))
            if total_vol > 0:
                buy_ratio = total_buy / total_vol
                if signal == "UP" and buy_ratio < flow_thresh:
                    continue
                if signal == "DOWN" and (1 - buy_ratio) < flow_thresh:
                    continue

        actual = "UP" if bar["close"] > op else "DOWN"
        win = signal == actual
        pnl = win_pnl if win else loss_pnl

        trades.append({
            "bar_idx": idx, "direction": signal, "actual": actual,
            "win": win, "pnl": pnl, "cost": cost, "momentum_bps": mom,
            "ts": bar["ts"],
        })

    return trades


def stats(trades, days=7):
    if not trades:
        return {"trades": 0, "wr": 0, "pnl": 0, "roi": 0, "dd": 0, "tpd": 0,
                "mws": 0, "mls": 0}
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

    return {"trades": total, "wr": round(wr, 1), "pnl": round(total_pnl, 2),
            "roi": round(roi, 1), "dd": round(max_dd, 2), "tpd": round(total / days, 1),
            "mws": mws, "mls": mls}


def main():
    DAYS = 7
    print("=" * 70)
    print("  LATE ENTRY 15m — FOCUSED PARAMETER SWEEP")
    print("=" * 70)

    print(f"\nFetching {DAYS} days of BTC 1m data...")
    bars_1m = fetch_binance_klines(days=DAYS)
    bars_15m = aggregate_15m(bars_1m)
    print(f"  {len(bars_1m)} 1m -> {len(bars_15m)} 15m bars")

    date_start = datetime.fromtimestamp(bars_1m[0]["ts"] / 1000, tz=timezone.utc)
    date_end = datetime.fromtimestamp(bars_1m[-1]["ts"] / 1000, tz=timezone.utc)
    nat = sum(1 for b in bars_15m if b["close"] > b["open"]) / len(bars_15m) * 100
    print(f"  Range: {date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}")
    print(f"  Natural UP%: {nat:.1f}%")

    # ============================================================
    # SWEEP 1: Wait time x Threshold (core parameters)
    # ============================================================
    print(f"\n{'=' * 70}")
    print("  SWEEP 1: Wait Time x Threshold")
    print(f"{'=' * 70}")
    print(f"  {'Wait':>6s} {'Thresh':>7s} {'Trades':>7s} {'T/d':>5s} {'WR%':>6s} "
          f"{'PnL':>10s} {'ROI%':>7s} {'DD':>8s} {'W/L':>6s}")
    print("  " + "-" * 65)

    best_pnl = -999
    best_config = {}
    sweep1 = []

    for wait in [7, 8, 9, 10, 11, 12]:
        for thresh in [5, 10, 15, 20, 25, 30]:
            t = simulate_late_entry(bars_15m, wait, thresh)
            s = stats(t, DAYS)
            sweep1.append({"wait": wait, "thresh": thresh, **s})
            marker = ""
            if s["pnl"] > best_pnl and s["trades"] >= 10:
                best_pnl = s["pnl"]
                best_config = {"wait": wait, "thresh": thresh}
                marker = " <-- BEST"
            print(f"  {wait:>4d}m {thresh:>5.0f}bp {s['trades']:>6d} {s['tpd']:>5.1f} "
                  f"{s['wr']:>5.1f}% ${s['pnl']:>+8.2f} {s['roi']:>+6.1f}% "
                  f"${s['dd']:>6.2f} {s['mws']}/{s['mls']}{marker}")

    print(f"\n  BEST: wait={best_config['wait']}min, thresh={best_config['thresh']}bp "
          f"-> ${best_pnl:+.2f}/week")

    # ============================================================
    # SWEEP 2: Best wait time + filters (trend, volume, flow)
    # ============================================================
    bw = best_config.get("wait", 10)
    bt = best_config.get("thresh", 15)

    print(f"\n{'=' * 70}")
    print(f"  SWEEP 2: Filters on wait={bw}min, thresh={bt}bp")
    print(f"{'=' * 70}")

    # Base (no filters)
    t_base = simulate_late_entry(bars_15m, bw, bt)
    s_base = stats(t_base, DAYS)
    print(f"\n  BASE (no filters):  {s_base['trades']}T  {s_base['wr']}% WR  "
          f"${s_base['pnl']:+.2f}  ROI={s_base['roi']}%  DD=${s_base['dd']}")

    # + Trend filter
    print(f"\n  + TREND FILTER:")
    for lb in [2, 4, 8]:
        t = simulate_late_entry(bars_15m, bw, bt, trend_filter=True, trend_lookback=lb)
        s = stats(t, DAYS)
        print(f"    LB={lb}: {s['trades']}T  {s['wr']}% WR  ${s['pnl']:+.2f}  "
              f"ROI={s['roi']}%  DD=${s['dd']}")

    # + Volume filter
    print(f"\n  + VOLUME FILTER:")
    for vm in [1.0, 1.3, 1.5, 2.0]:
        t = simulate_late_entry(bars_15m, bw, bt, vol_filter=True, vol_min=vm, bars_1m=bars_1m)
        s = stats(t, DAYS)
        print(f"    vol>={vm:.1f}x: {s['trades']}T  {s['wr']}% WR  ${s['pnl']:+.2f}  "
              f"ROI={s['roi']}%  DD=${s['dd']}")

    # + Flow filter
    print(f"\n  + FLOW CONFIRMATION:")
    for ft in [0.50, 0.52, 0.54, 0.56]:
        t = simulate_late_entry(bars_15m, bw, bt, flow_filter=True, flow_thresh=ft)
        s = stats(t, DAYS)
        print(f"    flow>={ft:.2f}: {s['trades']}T  {s['wr']}% WR  ${s['pnl']:+.2f}  "
              f"ROI={s['roi']}%  DD=${s['dd']}")

    # + Combined filters
    print(f"\n  + COMBINED FILTERS:")
    combos = [
        {"trend_filter": True, "trend_lookback": 4, "label": "trend4"},
        {"trend_filter": True, "trend_lookback": 4, "flow_filter": True, "flow_thresh": 0.52, "label": "trend4+flow52"},
        {"vol_filter": True, "vol_min": 1.3, "label": "vol1.3"},
        {"trend_filter": True, "trend_lookback": 4, "vol_filter": True, "vol_min": 1.3, "label": "trend4+vol1.3"},
        {"flow_filter": True, "flow_thresh": 0.52, "vol_filter": True, "vol_min": 1.3, "label": "flow52+vol1.3"},
    ]
    best_filtered = None
    for c in combos:
        label = c.pop("label")
        t = simulate_late_entry(bars_15m, bw, bt, bars_1m=bars_1m, **c)
        s = stats(t, DAYS)
        print(f"    {label:25s}: {s['trades']}T  {s['wr']}% WR  ${s['pnl']:+.2f}  "
              f"ROI={s['roi']}%  DD=${s['dd']}")
        if s["trades"] >= 5 and (best_filtered is None or s["wr"] > best_filtered["wr"]):
            best_filtered = {**s, "label": label}

    # ============================================================
    # SWEEP 3: Higher thresholds (for tighter WR)
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  SWEEP 3: High-Confidence Variants (wait={bw}min)")
    print(f"{'=' * 70}")

    for thresh in [15, 20, 25, 30, 40, 50]:
        t = simulate_late_entry(bars_15m, bw, thresh)
        s = stats(t, DAYS)
        # Also check with trend filter
        t2 = simulate_late_entry(bars_15m, bw, thresh, trend_filter=True, trend_lookback=4)
        s2 = stats(t2, DAYS)
        print(f"  {thresh:>3.0f}bp: {s['trades']:>4d}T {s['wr']:>5.1f}% ${s['pnl']:>+8.2f} "
              f"DD=${s['dd']:>5.2f}  |  +trend4: {s2['trades']:>4d}T {s2['wr']:>5.1f}% "
              f"${s2['pnl']:>+8.2f}")

    # ============================================================
    # HOUR-BY-HOUR BREAKDOWN
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  HOUR-BY-HOUR BREAKDOWN (wait={bw}min, thresh={bt}bp)")
    print(f"{'=' * 70}")

    t_all = simulate_late_entry(bars_15m, bw, bt)
    from collections import defaultdict
    hour_data = defaultdict(list)
    for trade in t_all:
        hr = datetime.fromtimestamp(trade["ts"] / 1000, tz=timezone.utc).hour
        hour_data[hr].append(trade)

    print(f"  {'Hour':>4s} {'Trades':>7s} {'WR%':>6s} {'PnL':>8s}")
    print("  " + "-" * 30)
    for hr in sorted(hour_data.keys()):
        trades_hr = hour_data[hr]
        wins = sum(1 for t in trades_hr if t["win"])
        wr = wins / len(trades_hr) * 100 if trades_hr else 0
        pnl = sum(t["pnl"] for t in trades_hr)
        flag = " ***" if wr >= 95 else (" !!" if wr < 70 else "")
        print(f"  {hr:>4d}  {len(trades_hr):>6d}  {wr:>5.1f}%  ${pnl:>+6.2f}{flag}")

    # ============================================================
    # DIRECTION BREAKDOWN
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  DIRECTION BREAKDOWN (wait={bw}min, thresh={bt}bp)")
    print(f"{'=' * 70}")
    ups = [t for t in t_all if t["direction"] == "UP"]
    dns = [t for t in t_all if t["direction"] == "DOWN"]
    if ups:
        up_wr = sum(1 for t in ups if t["win"]) / len(ups) * 100
        up_pnl = sum(t["pnl"] for t in ups)
        print(f"  UP:   {len(ups)}T  {up_wr:.1f}% WR  ${up_pnl:+.2f}")
    if dns:
        dn_wr = sum(1 for t in dns if t["win"]) / len(dns) * 100
        dn_pnl = sum(t["pnl"] for t in dns)
        print(f"  DOWN: {len(dns)}T  {dn_wr:.1f}% WR  ${dn_pnl:+.2f}")

    # ============================================================
    # MOMENTUM MAGNITUDE BREAKDOWN
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  MOMENTUM MAGNITUDE BREAKDOWN")
    print(f"{'=' * 70}")
    for lo, hi in [(15, 20), (20, 30), (30, 50), (50, 100), (100, 500)]:
        bucket = [t for t in t_all if lo <= abs(t["momentum_bps"]) < hi]
        if bucket:
            bwr = sum(1 for t in bucket if t["win"]) / len(bucket) * 100
            bpnl = sum(t["pnl"] for t in bucket)
            print(f"  {lo}-{hi}bp: {len(bucket)}T  {bwr:.1f}% WR  ${bpnl:+.2f}")

    # ============================================================
    # FINAL RECOMMENDATION
    # ============================================================
    print(f"\n{'=' * 70}")
    print("  RECOMMENDATION FOR LIVE DEPLOYMENT")
    print(f"{'=' * 70}")
    print(f"  Strategy: Late Entry Momentum on 15m BTC markets")
    print(f"  Wait: {bw} minutes after market opens")
    print(f"  Threshold: {bt}bp minimum momentum")
    print(f"  Max wait: {bw + 2} minutes (don't enter in last 3 min)")
    print(f"  Expected: {s_base['trades']}T/week, {s_base['wr']}% WR, "
          f"${s_base['pnl']:+.2f}/week, DD=${s_base['dd']}")

    # Save
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": f"{date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}",
        "bars_1m": len(bars_1m),
        "bars_15m": len(bars_15m),
        "natural_up_pct": round(nat, 2),
        "best_config": best_config,
        "base_stats": s_base,
        "sweep1": sweep1,
    }
    out = Path(__file__).parent / "backtest_15m_late_entry_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
