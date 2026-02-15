"""
Backtest V3.0 Momentum + V12a Order Flow on 15-MINUTE BTC bars.

Same signal logic (first 2 minutes), but the resolution window is 15 min instead of 5 min.
Question: Does the momentum continuation effect hold over a longer 15-min window?
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


def aggregate_Nm(bars_1m, N):
    """Aggregate 1m bars into N-minute bars."""
    bars_nm = []
    for i in range(0, len(bars_1m) - (N - 1), N):
        chunk = bars_1m[i:i + N]
        if len(chunk) < N:
            break
        bars_nm.append({
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
    return bars_nm


# ============================================================
# V3.0: Simple Momentum (first 2 min predicts close)
# ============================================================

def simulate_v30(bars_nm, size_usd=5.0, threshold_bps=15.0):
    BUY = 0.51
    trades = []
    for i, bar in enumerate(bars_nm):
        mins = bar["minutes"]
        if len(mins) < 3:
            continue
        op = mins[0]["open"]
        p2 = mins[1]["close"]  # price after 2 min
        mom = (p2 - op) / op * 10000
        if abs(mom) < threshold_bps:
            continue
        d = "UP" if mom > 0 else "DOWN"
        actual = "UP" if bar["close"] > op else "DOWN"
        sh = max(1, math.floor(size_usd / BUY))
        cost = BUY * sh
        pnl = (sh - cost) if d == actual else -cost
        trades.append({
            "bar_idx": i, "direction": d, "actual": actual, "win": d == actual,
            "momentum_bps": mom, "pnl": pnl, "cost": cost,
        })
    return trades


# ============================================================
# V12a: Taker Flow Imbalance (2x volume surge version = live config)
# ============================================================

def simulate_v12a(bars_nm, bars_1m, N, size_usd=5.0,
                  buy_ratio_thresh=0.56, vol_ratio_min=2.0):
    BUY = 0.51
    vols = np.array([b["volume"] for b in bars_1m], dtype=float)
    vol_sma = np.full_like(vols, np.nan)
    for i in range(19, len(vols)):
        vol_sma[i] = np.mean(vols[i - 19:i + 1])

    trades = []
    for i, bar in enumerate(bars_nm):
        mins = bar["minutes"]
        if len(mins) < 3:
            continue

        # First 2 minutes of order flow
        total_vol = mins[0]["volume"] + mins[1]["volume"]
        total_buy = mins[0]["taker_buy"] + mins[1]["taker_buy"]
        if total_vol < 1e-9:
            continue
        buy_ratio = total_buy / total_vol

        # Volume surge check
        m1_global = i * N + 1
        if m1_global >= len(vols) or not np.isfinite(vol_sma[m1_global]) or vol_sma[m1_global] <= 0:
            continue
        avg_2min_vol = vol_sma[m1_global] * 2
        vol_ratio = total_vol / max(1e-9, avg_2min_vol)
        if vol_ratio < vol_ratio_min:
            continue

        # Direction from flow
        if buy_ratio >= buy_ratio_thresh:
            side = "UP"
        elif (1.0 - buy_ratio) >= buy_ratio_thresh:
            side = "DOWN"
        else:
            continue

        # Settlement: 15m bar close vs open
        actual = "UP" if bar["close"] > bar["open"] else "DOWN"
        win = side == actual
        sh = max(1, math.floor(size_usd / BUY))
        cost = BUY * sh
        pnl = (sh - cost) if win else -cost

        trades.append({
            "bar_idx": i, "direction": side, "actual": actual, "win": win,
            "buy_ratio": buy_ratio, "vol_ratio": vol_ratio,
            "pnl": pnl, "cost": cost,
        })
    return trades


def analyze(trades, label, days=7):
    if not trades:
        print(f"\n  {label}: NO TRADES")
        return {"label": label, "trades": 0, "wins": 0, "win_rate": 0,
                "total_pnl": 0, "avg_pnl": 0, "roi_pct": 0,
                "max_drawdown": 0, "trades_per_day": 0,
                "max_win_streak": 0, "max_loss_streak": 0}

    total = len(trades)
    wins = sum(1 for t in trades if t["win"])
    wr = wins / total * 100
    total_pnl = sum(t["pnl"] for t in trades)
    total_cost = sum(t["cost"] for t in trades)
    avg_pnl = total_pnl / total

    wp = [t["pnl"] for t in trades if t["win"]]
    lp = [t["pnl"] for t in trades if not t["win"]]
    avg_win = sum(wp) / len(wp) if wp else 0
    avg_loss = sum(lp) / len(lp) if lp else 0

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
    tpd = total / days

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Trades:        {total}  ({tpd:.1f}/day)")
    print(f"  Wins:          {wins} ({wr:.1f}%)")
    print(f"  Total PnL:     ${total_pnl:+,.2f}")
    print(f"  Avg PnL:       ${avg_pnl:+.2f}/trade")
    print(f"  Avg Win:       ${avg_win:+.2f}  |  Avg Loss: ${avg_loss:+.2f}")
    if avg_loss != 0:
        print(f"  W:L Ratio:     {abs(avg_win / avg_loss):.2f}x")
    print(f"  ROI:           {roi:+.1f}%")
    print(f"  Max Drawdown:  ${max_dd:,.2f}")
    print(f"  Streaks:       W={mws} / L={mls}")

    ups = [t for t in trades if t["direction"] == "UP"]
    dns = [t for t in trades if t["direction"] == "DOWN"]
    if ups:
        up_wr = sum(1 for t in ups if t["win"]) / len(ups) * 100
        print(f"  UP:            {len(ups)}T ({up_wr:.1f}% WR)")
    if dns:
        dn_wr = sum(1 for t in dns if t["win"]) / len(dns) * 100
        print(f"  DOWN:          {len(dns)}T ({dn_wr:.1f}% WR)")

    return {
        "label": label, "trades": total, "wins": wins,
        "win_rate": round(wr, 2), "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 4), "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2), "roi_pct": round(roi, 2),
        "max_drawdown": round(max_dd, 2),
        "trades_per_day": round(tpd, 1),
        "max_win_streak": mws, "max_loss_streak": mls,
    }


def main():
    DAYS = 7

    print("=" * 70)
    print("  V3.0 + V12a BACKTEST ON 15-MINUTE BTC BARS")
    print("=" * 70)
    print(f"\nFetching {DAYS} days of BTC 1-min data from Binance...")
    bars_1m = fetch_binance_klines(days=DAYS)
    print(f"  Got {len(bars_1m)} 1-min bars")

    # Aggregate to both 5m and 15m
    bars_5m = aggregate_Nm(bars_1m, 5)
    bars_15m = aggregate_Nm(bars_1m, 15)
    print(f"  5m bars:  {len(bars_5m)}")
    print(f"  15m bars: {len(bars_15m)}")

    date_start = datetime.fromtimestamp(bars_1m[0]["ts"] / 1000, tz=timezone.utc)
    date_end = datetime.fromtimestamp(bars_1m[-1]["ts"] / 1000, tz=timezone.utc)
    print(f"  Range: {date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}")

    nat_5m = sum(1 for b in bars_5m if b["close"] > b["open"]) / len(bars_5m) * 100
    nat_15m = sum(1 for b in bars_15m if b["close"] > b["open"]) / len(bars_15m) * 100
    print(f"  Natural UP%: 5m={nat_5m:.1f}%  15m={nat_15m:.1f}%")

    SIZE = 5.0

    # ---- 5-MIN (baseline comparison) ----
    print("\n" + "#" * 70)
    print("  5-MINUTE BARS (baseline)")
    print("#" * 70)

    v30_5m = simulate_v30(bars_5m, SIZE)
    s30_5m = analyze(v30_5m, "V3.0 Momentum @ 5min", DAYS)

    v12a_5m = simulate_v12a(bars_5m, bars_1m, 5, SIZE, 0.56, 2.0)
    s12a_5m = analyze(v12a_5m, "V12a Flow (2x vol) @ 5min", DAYS)

    # V12a unique (non-overlapping with V3.0) on 5m
    v30_5m_bars = set(t["bar_idx"] for t in v30_5m)
    v12a_5m_unique = [t for t in v12a_5m if t["bar_idx"] not in v30_5m_bars]
    s12a_5m_u = analyze(v12a_5m_unique, "V12a UNIQUE (non-overlap V3.0) @ 5min", DAYS)

    # ---- 15-MIN ----
    print("\n" + "#" * 70)
    print("  15-MINUTE BARS")
    print("#" * 70)

    v30_15m = simulate_v30(bars_15m, SIZE)
    s30_15m = analyze(v30_15m, "V3.0 Momentum @ 15min", DAYS)

    # Test multiple momentum thresholds on 15m
    for thresh in [10.0, 15.0, 20.0, 30.0]:
        t = simulate_v30(bars_15m, SIZE, thresh)
        if t:
            wins = sum(1 for x in t if x["win"])
            wr = wins / len(t) * 100
            pnl = sum(x["pnl"] for x in t)
            print(f"    V3.0 @ {thresh:.0f}bp: {len(t)}T, {wr:.1f}% WR, ${pnl:+.2f}")

    v12a_15m = simulate_v12a(bars_15m, bars_1m, 15, SIZE, 0.56, 2.0)
    s12a_15m = analyze(v12a_15m, "V12a Flow (2x vol) @ 15min", DAYS)

    # Test multiple vol thresholds on 15m
    print(f"\n  V12a volume threshold sweep (15m):")
    for vt in [1.0, 1.5, 2.0, 2.5, 3.0]:
        t = simulate_v12a(bars_15m, bars_1m, 15, SIZE, 0.56, vt)
        if t:
            wins = sum(1 for x in t if x["win"])
            wr = wins / len(t) * 100
            pnl = sum(x["pnl"] for x in t)
            print(f"    vol>={vt:.1f}x: {len(t)}T, {wr:.1f}% WR, ${pnl:+.2f}")

    # V12a unique on 15m
    v30_15m_bars = set(t["bar_idx"] for t in v30_15m)
    v12a_15m_unique = [t for t in v12a_15m if t["bar_idx"] not in v30_15m_bars]
    s12a_15m_u = analyze(v12a_15m_unique, "V12a UNIQUE (non-overlap V3.0) @ 15min", DAYS)

    # ---- HEAD TO HEAD: 5m vs 15m ----
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD: 5min vs 15min")
    print("=" * 70)

    def row(label, *vals):
        parts = [f"  {label:22s}"]
        for v in vals:
            parts.append(f"{v:>14s}")
        print("".join(parts))

    row("", "V3.0 @5m", "V3.0 @15m", "V12a @5m", "V12a @15m")
    row("-" * 22, "-" * 14, "-" * 14, "-" * 14, "-" * 14)

    for label, key in [
        ("Trades", "trades"), ("Trades/day", "trades_per_day"),
        ("Win Rate %", "win_rate"), ("Total PnL", "total_pnl"),
        ("Avg PnL/trade", "avg_pnl"), ("ROI%", "roi_pct"),
        ("Max Drawdown", "max_drawdown"),
        ("Win Streak", "max_win_streak"), ("Loss Streak", "max_loss_streak"),
    ]:
        vals = []
        for s in [s30_5m, s30_15m, s12a_5m, s12a_15m]:
            v = s.get(key, 0)
            if key in ["total_pnl", "avg_pnl", "max_drawdown"]:
                vals.append(f"${v:+,.2f}")
            elif key in ["win_rate", "roi_pct", "trades_per_day"]:
                vals.append(f"{v:.1f}")
            else:
                vals.append(f"{v}")
        row(label, *vals)

    # ---- STACKING on 15m ----
    print("\n" + "=" * 70)
    print("  STACKING POTENTIAL (15min)")
    print("=" * 70)

    if v30_15m:
        pnl_30 = sum(t["pnl"] for t in v30_15m)
        print(f"  V3.0 alone:        {len(v30_15m)}T, {s30_15m['win_rate']:.1f}% WR, ${pnl_30:+.2f}")
    if v12a_15m_unique:
        pnl_u = sum(t["pnl"] for t in v12a_15m_unique)
        wr_u = sum(1 for t in v12a_15m_unique if t["win"]) / len(v12a_15m_unique) * 100
        print(f"  V12a unique:       {len(v12a_15m_unique)}T, {wr_u:.1f}% WR, ${pnl_u:+.2f}")
        combined_pnl = pnl_30 + pnl_u if v30_15m else pnl_u
        print(f"  Combined:          ${combined_pnl:+.2f}")
    else:
        print(f"  V12a unique:       0 trades (full overlap with V3.0)")

    # Save results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": f"{date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}",
        "bars_1m": len(bars_1m),
        "bars_5m": len(bars_5m),
        "bars_15m": len(bars_15m),
        "natural_up_5m": round(nat_5m, 2),
        "natural_up_15m": round(nat_15m, 2),
        "v30_5m": s30_5m,
        "v30_15m": s30_15m,
        "v12a_5m": s12a_5m,
        "v12a_15m": s12a_15m,
        "v12a_5m_unique": s12a_5m_u,
        "v12a_15m_unique": s12a_15m_u,
    }
    out = Path(__file__).parent / "backtest_15m_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
