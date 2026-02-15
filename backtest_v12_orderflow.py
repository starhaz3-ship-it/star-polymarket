"""
Head-to-head backtest: V3.0 vs V12a vs V12b (Order Flow strategies)

V12a - Taker Flow Imbalance:
  Uses REAL taker buy/sell volume from Binance (not a proxy).
  After 1-2 minutes: if buy_ratio >> 0.5, predict UP. No price momentum required.
  Should fire on DIFFERENT bars than V3.0 (volume signal, not price signal).

V12b - CVD Acceleration + Absorption:
  Cumulative Volume Delta (taker buy - taker sell) over rolling window.
  Detects "stealth accumulation" — CVD trending up while price is flat.
  Combined with absorption detection (high volume + tight range).

Both use Binance's taker_buy_base_volume (kline field index 9).
"""

import json
import math
import time
import requests
import numpy as np
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# DATA FETCHING — now includes taker buy volume
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
        taker_buy = float(k[9])  # Binance field 9: taker buy base asset volume
        bars.append({
            "ts": k[0], "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]),
            "volume": total_vol,
            "taker_buy": taker_buy,
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
            "num_trades": sum(c["num_trades"] for c in chunk),
            "minutes": chunk,
        })
    return bars_5m


# ============================================================
# V3.0: Simple Momentum (baseline)
# ============================================================

def simulate_v30(bars_5m, size_usd=5.0):
    BUY = 0.51
    trades = []
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
        sh = max(1, math.floor(size_usd / BUY))
        cost = BUY * sh
        pnl = (sh - cost) if d == actual else -cost
        trades.append({"bar_idx": i, "direction": d, "actual": actual, "win": d == actual,
                       "momentum_bps": mom, "pnl": pnl, "cost": cost,
                       "shares": sh, "buy_price": BUY})
    return trades


# ============================================================
# V12a: Taker Flow Imbalance
# ============================================================

def simulate_v12a(bars_5m, bars_1m, size_usd=5.0):
    """
    After first 1-2 minutes: compute taker buy ratio.
    If buy_ratio > threshold -> UP, if < (1 - threshold) -> DOWN.
    No price momentum required — purely volume-based signal.

    Also uses: volume surge (vol vs 20-bar avg) and delta acceleration.
    """
    BUY_PRICE = 0.51
    # Thresholds tuned for signal frequency
    BUY_RATIO_THRESHOLD = 0.56  # 56% buy = bullish
    SELL_RATIO_THRESHOLD = 0.44  # 44% buy (56% sell) = bearish
    MIN_VOL_RATIO = 1.2  # volume must be above average (confirms real flow)

    n1 = len(bars_1m)
    # Precompute 1m volume SMA for relative volume
    vols = np.array([b["volume"] for b in bars_1m], dtype=float)
    vol_sma = np.full_like(vols, np.nan)
    for i in range(19, len(vols)):
        vol_sma[i] = np.mean(vols[i - 19:i + 1])

    trades = []

    for i, bar in enumerate(bars_5m):
        mins = bar["minutes"]
        if len(mins) < 5:
            continue

        # Use first 2 minutes of order flow
        m0 = mins[0]
        m1 = mins[1]

        # Aggregate taker flow over first 2 minutes
        total_vol = m0["volume"] + m1["volume"]
        total_buy = m0["taker_buy"] + m1["taker_buy"]

        if total_vol < 1e-9:
            continue

        buy_ratio = total_buy / total_vol

        # Volume must be meaningful (above average)
        m1_global = i * 5 + 1
        if m1_global >= n1 or not np.isfinite(vol_sma[m1_global]) or vol_sma[m1_global] <= 0:
            continue

        # Combined 2-min volume vs average
        avg_2min_vol = vol_sma[m1_global] * 2  # 2 bars worth of average
        vol_ratio = total_vol / max(1e-9, avg_2min_vol)

        if vol_ratio < MIN_VOL_RATIO:
            continue  # low volume = unreliable signal

        # Decision
        if buy_ratio >= BUY_RATIO_THRESHOLD:
            side = "UP"
        elif buy_ratio <= SELL_RATIO_THRESHOLD:
            side = "DOWN"
        else:
            continue

        # Settlement
        actual = "UP" if bar["close"] > bar["open"] else "DOWN"
        win = side == actual
        sh = max(1, math.floor(size_usd / BUY_PRICE))
        cost = BUY_PRICE * sh
        pnl = (sh - cost) if win else -cost

        trades.append({
            "bar_idx": i, "direction": side, "actual": actual, "win": win,
            "buy_ratio": buy_ratio, "vol_ratio": vol_ratio,
            "momentum_bps": (m1["close"] - m0["open"]) / m0["open"] * 10000,
            "pnl": pnl, "cost": cost, "shares": sh, "buy_price": BUY_PRICE,
        })

    return trades


# ============================================================
# V12b: CVD Acceleration + Absorption
# ============================================================

def simulate_v12b(bars_5m, bars_1m, size_usd=5.0):
    """
    Cumulative Volume Delta over rolling window.
    Detect "stealth accumulation/distribution":
      - CVD trending strongly in one direction (buying/selling pressure)
      - Combined with absorption: high volume + tight price range
      - These predict breakout direction even when V3.0 sees no momentum

    Uses first 1-2 minutes of session + prior 20 bars of CVD context.
    """
    BUY_PRICE = 0.51
    CVD_LOOKBACK = 20      # bars of CVD history
    CVD_SLOPE_MIN = 0.60   # normalized CVD slope threshold
    ABSORPTION_VOL_RATIO = 1.3  # volume above avg
    ABSORPTION_RANGE_RATIO = 0.7  # range below avg (tight = absorption)

    n1 = len(bars_1m)

    # Precompute 1m volume delta (taker_buy - taker_sell)
    deltas = np.array([b["taker_buy"] - b["taker_sell"] for b in bars_1m], dtype=float)
    # Cumulative volume delta
    cvd = np.cumsum(deltas)

    # 1m volume and range for absorption detection
    vols = np.array([b["volume"] for b in bars_1m], dtype=float)
    ranges = np.array([b["high"] - b["low"] for b in bars_1m], dtype=float)

    vol_sma = np.full_like(vols, np.nan)
    range_sma = np.full_like(ranges, np.nan)
    for idx in range(19, len(vols)):
        vol_sma[idx] = np.mean(vols[idx - 19:idx + 1])
        range_sma[idx] = np.mean(ranges[idx - 19:idx + 1])

    trades = []

    for i, bar in enumerate(bars_5m):
        if i < 5:  # need CVD history
            continue
        mins = bar["minutes"]
        if len(mins) < 5:
            continue

        # Decision point: after first 1 minute
        m1_global = i * 5 + 1
        if m1_global >= n1 or m1_global < CVD_LOOKBACK:
            continue

        # CVD slope over lookback window (normalized by volume)
        cvd_now = cvd[m1_global]
        cvd_past = cvd[m1_global - CVD_LOOKBACK]
        cvd_change = cvd_now - cvd_past

        # Normalize by total volume in window for comparability
        total_vol_window = np.sum(vols[m1_global - CVD_LOOKBACK + 1:m1_global + 1])
        if total_vol_window < 1e-9:
            continue

        cvd_slope_norm = cvd_change / total_vol_window  # [-1, +1] range

        # Current minute's order flow
        m0 = mins[0]
        m1 = mins[1]
        cur_delta = (m0["taker_buy"] - m0["taker_sell"]) + (m1["taker_buy"] - m1["taker_sell"])
        cur_vol = m0["volume"] + m1["volume"]

        if not np.isfinite(vol_sma[m1_global]) or vol_sma[m1_global] <= 0:
            continue
        if not np.isfinite(range_sma[m1_global]) or range_sma[m1_global] <= 0:
            continue

        vol_ratio = cur_vol / (vol_sma[m1_global] * 2)
        range_ratio = (m1["high"] - m0["low"]) / max(1e-9, range_sma[m1_global] * 2)

        # Absorption detection: high volume + tight range
        is_absorption = (vol_ratio >= ABSORPTION_VOL_RATIO and range_ratio <= ABSORPTION_RANGE_RATIO)

        # CVD acceleration: current delta confirms CVD trend
        cvd_confirms = (cvd_slope_norm > 0 and cur_delta > 0) or (cvd_slope_norm < 0 and cur_delta < 0)

        # Need either strong CVD slope OR CVD + absorption
        if abs(cvd_slope_norm) >= CVD_SLOPE_MIN:
            # Strong CVD trend
            side = "UP" if cvd_slope_norm > 0 else "DOWN"
        elif abs(cvd_slope_norm) >= 0.35 and is_absorption and cvd_confirms:
            # Moderate CVD + absorption + confirmation
            side = "UP" if cvd_slope_norm > 0 else "DOWN"
        else:
            continue

        # Settlement
        actual = "UP" if bar["close"] > bar["open"] else "DOWN"
        win = side == actual
        sh = max(1, math.floor(size_usd / BUY_PRICE))
        cost = BUY_PRICE * sh
        pnl = (sh - cost) if win else -cost

        trades.append({
            "bar_idx": i, "direction": side, "actual": actual, "win": win,
            "cvd_slope": cvd_slope_norm, "vol_ratio": vol_ratio,
            "range_ratio": range_ratio, "is_absorption": is_absorption,
            "cur_delta_sign": 1 if cur_delta > 0 else -1,
            "momentum_bps": (m1["close"] - m0["open"]) / m0["open"] * 10000,
            "pnl": pnl, "cost": cost, "shares": sh, "buy_price": BUY_PRICE,
        })

    return trades


# ============================================================
# ANALYSIS
# ============================================================

def analyze(trades, label):
    if not trades:
        print(f"\n{'=' * 60}")
        print(f"  {label}: NO TRADES")
        print(f"{'=' * 60}")
        return {"label": label, "trades": 0, "wins": 0, "win_rate": 0,
                "total_pnl": 0, "avg_pnl": 0, "avg_win": 0, "avg_loss": 0,
                "max_drawdown": 0, "total_cost": 0, "roi_pct": 0,
                "max_win_streak": 0, "max_loss_streak": 0, "trades_per_day": 0}

    total = len(trades)
    wins = sum(1 for t in trades if t["win"])
    wr = wins / total * 100
    total_pnl = sum(t["pnl"] for t in trades)
    avg_pnl = total_pnl / total
    total_cost = sum(t["cost"] for t in trades)

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
    tpd = total / 7.0

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Trades:        {total}  ({tpd:.1f}/day)")
    print(f"  Wins:          {wins} ({wr:.1f}%)")
    print(f"  Losses:        {total - wins} ({100 - wr:.1f}%)")
    print(f"  Total PnL:     ${total_pnl:+,.2f}")
    print(f"  Avg PnL:       ${avg_pnl:+.2f}/trade")
    print(f"  Avg Win:       ${avg_win:+.2f}")
    print(f"  Avg Loss:      ${avg_loss:+.2f}")
    if avg_loss != 0:
        print(f"  W:L Ratio:     {abs(avg_win / avg_loss):.2f}x")
    print(f"  ROI:           {roi:+.2f}%")
    print(f"  Max Drawdown:  ${max_dd:,.2f}")
    print(f"  Win Streak:    {mws}")
    print(f"  Loss Streak:   {mls}")

    ups = [t for t in trades if t["direction"] == "UP"]
    dns = [t for t in trades if t["direction"] == "DOWN"]
    if ups:
        print(f"  UP trades:     {len(ups)} ({sum(1 for t in ups if t['win']) / len(ups) * 100:.1f}% WR)")
    if dns:
        print(f"  DOWN trades:   {len(dns)} ({sum(1 for t in dns if t['win']) / len(dns) * 100:.1f}% WR)")

    return {
        "label": label, "trades": total, "wins": wins,
        "win_rate": round(wr, 2), "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 4), "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2), "max_drawdown": round(max_dd, 2),
        "total_cost": round(total_cost, 2), "roi_pct": round(roi, 2),
        "max_win_streak": mws, "max_loss_streak": mls,
        "trades_per_day": round(tpd, 1),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("Fetching 7 days of BTC 1-min data from Binance (with taker volume)...")
    bars_1m = fetch_binance_klines(days=7)
    print(f"  Got {len(bars_1m)} 1-min bars")

    # Verify taker volume is present
    sample = bars_1m[100]
    print(f"  Sample bar: vol={sample['volume']:.1f}, buy={sample['taker_buy']:.1f}, "
          f"sell={sample['taker_sell']:.1f}, ratio={sample['taker_buy']/max(1,sample['volume']):.3f}")

    bars_5m = aggregate_5m(bars_1m)
    print(f"  Aggregated to {len(bars_5m)} 5-min bars")

    date_start = datetime.fromtimestamp(bars_1m[0]["ts"] / 1000, tz=timezone.utc)
    date_end = datetime.fromtimestamp(bars_1m[-1]["ts"] / 1000, tz=timezone.utc)
    natural_up = sum(1 for b in bars_5m if b["close"] > b["open"]) / len(bars_5m) * 100
    print(f"  Range: {date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}")
    print(f"  Natural UP%: {natural_up:.1f}%")

    SIZE = 5.0

    # V3.0 baseline
    v30 = simulate_v30(bars_5m, SIZE)
    s30 = analyze(v30, "V3.0 - Simple Momentum (2min, 15bp, $5)")

    # V12a: Taker Flow Imbalance
    v12a = simulate_v12a(bars_5m, bars_1m, SIZE)
    s12a = analyze(v12a, "V12a - Taker Flow Imbalance ($5)")

    # V12a feature breakdown
    if v12a:
        print(f"\n{'=' * 60}")
        print(f"  V12a BREAKDOWN")
        print(f"{'=' * 60}")
        # By buy ratio strength
        strong = [t for t in v12a if abs(t["buy_ratio"] - 0.5) >= 0.10]
        moderate = [t for t in v12a if 0.06 <= abs(t["buy_ratio"] - 0.5) < 0.10]
        if strong:
            wr = sum(1 for t in strong if t["win"]) / len(strong) * 100
            print(f"  Strong flow (>60%): {len(strong)}T, {wr:.1f}% WR")
        if moderate:
            wr = sum(1 for t in moderate if t["win"]) / len(moderate) * 100
            print(f"  Moderate (56-60%):  {len(moderate)}T, {wr:.1f}% WR")
        # By volume surge
        hi_vol = [t for t in v12a if t["vol_ratio"] >= 2.0]
        lo_vol = [t for t in v12a if t["vol_ratio"] < 2.0]
        if hi_vol:
            wr = sum(1 for t in hi_vol if t["win"]) / len(hi_vol) * 100
            print(f"  Vol surge >= 2x:    {len(hi_vol)}T, {wr:.1f}% WR")
        if lo_vol:
            wr = sum(1 for t in lo_vol if t["win"]) / len(lo_vol) * 100
            print(f"  Vol surge 1.2-2x:   {len(lo_vol)}T, {wr:.1f}% WR")

    # V12b: CVD Acceleration + Absorption
    v12b = simulate_v12b(bars_5m, bars_1m, SIZE)
    s12b = analyze(v12b, "V12b - CVD Acceleration + Absorption ($5)")

    # V12b feature breakdown
    if v12b:
        print(f"\n{'=' * 60}")
        print(f"  V12b BREAKDOWN")
        print(f"{'=' * 60}")
        # By CVD slope strength
        strong_cvd = [t for t in v12b if abs(t["cvd_slope"]) >= 0.60]
        mod_cvd = [t for t in v12b if abs(t["cvd_slope"]) < 0.60]
        if strong_cvd:
            wr = sum(1 for t in strong_cvd if t["win"]) / len(strong_cvd) * 100
            print(f"  Strong CVD (>0.6):  {len(strong_cvd)}T, {wr:.1f}% WR")
        if mod_cvd:
            wr = sum(1 for t in mod_cvd if t["win"]) / len(mod_cvd) * 100
            print(f"  Moderate CVD+Abs:   {len(mod_cvd)}T, {wr:.1f}% WR")
        # Absorption trades
        absorb = [t for t in v12b if t["is_absorption"]]
        no_absorb = [t for t in v12b if not t["is_absorption"]]
        if absorb:
            wr = sum(1 for t in absorb if t["win"]) / len(absorb) * 100
            print(f"  With absorption:    {len(absorb)}T, {wr:.1f}% WR")
        if no_absorb:
            wr = sum(1 for t in no_absorb if t["win"]) / len(no_absorb) * 100
            print(f"  No absorption:      {len(no_absorb)}T, {wr:.1f}% WR")

    # Head to head
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD (all $5 sizing)")
    print("=" * 70)

    def row(label, *vals):
        parts = [f"  {label:20s}"]
        for v in vals:
            parts.append(f"{v:>16s}")
        print("".join(parts))

    row("", "V3.0", "V12a Flow", "V12b CVD")
    row("-" * 20, "-" * 16, "-" * 16, "-" * 16)

    for label, key in [
        ("Trades", "trades"), ("Trades/day", "trades_per_day"),
        ("Win Rate %", "win_rate"), ("Total PnL", "total_pnl"),
        ("Avg PnL/trade", "avg_pnl"), ("Avg Win", "avg_win"),
        ("Avg Loss", "avg_loss"), ("Max Drawdown", "max_drawdown"),
        ("ROI%", "roi_pct"), ("Win Streak", "max_win_streak"),
        ("Loss Streak", "max_loss_streak"),
    ]:
        vals = []
        for s in [s30, s12a, s12b]:
            v = s.get(key, 0)
            if key in ["total_pnl", "avg_pnl", "avg_win", "avg_loss", "max_drawdown"]:
                vals.append(f"${v:+,.2f}")
            elif key in ["win_rate", "roi_pct", "trades_per_day"]:
                vals.append(f"{v:.1f}")
            else:
                vals.append(f"{v}")
        row(label, *vals)

    # Overlap analysis
    print("\n" + "=" * 70)
    print("  OVERLAP ANALYSIS")
    print("=" * 70)

    v30_bars = set(t["bar_idx"] for t in v30)
    v12a_bars = set(t["bar_idx"] for t in v12a)
    v12b_bars = set(t["bar_idx"] for t in v12b)

    def overlap_report(name_a, set_a, name_b, set_b):
        both = set_a & set_b
        a_only = set_a - set_b
        b_only = set_b - set_a
        print(f"  {name_a}: {len(set_a)} | {name_b}: {len(set_b)} | "
              f"Overlap: {len(both)} | {name_a}-only: {len(a_only)} | {name_b}-only: {len(b_only)}")
        if b_only:
            # WR of unique trades
            unique_trades = [t for t in (v12a if "12a" in name_b else v12b) if t["bar_idx"] in b_only]
            if unique_trades:
                wr = sum(1 for t in unique_trades if t["win"]) / len(unique_trades) * 100
                print(f"    {name_b} unique WR: {wr:.1f}%")

    overlap_report("V3.0", v30_bars, "V12a", v12a_bars)
    overlap_report("V3.0", v30_bars, "V12b", v12b_bars)
    overlap_report("V12a", v12a_bars, "V12b", v12b_bars)

    # Verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print(f"  V3.0:   {s30['trades']}T, {s30['win_rate']:.1f}% WR, ${s30['total_pnl']:+,.2f}, {s30['roi_pct']:+.1f}% ROI")
    print(f"  V12a:   {s12a['trades']}T, {s12a['win_rate']:.1f}% WR, ${s12a['total_pnl']:+,.2f}, {s12a['roi_pct']:+.1f}% ROI")
    print(f"  V12b:   {s12b['trades']}T, {s12b['win_rate']:.1f}% WR, ${s12b['total_pnl']:+,.2f}, {s12b['roi_pct']:+.1f}% ROI")

    # Stacking potential
    print(f"\n  STACKING POTENTIAL:")
    for name, bars, strat_trades in [("V12a", v12a_bars, v12a), ("V12b", v12b_bars, v12b)]:
        unique = bars - v30_bars
        if unique:
            unique_t = [t for t in strat_trades if t["bar_idx"] in unique]
            wr = sum(1 for t in unique_t if t["win"]) / len(unique_t) * 100 if unique_t else 0
            pnl = sum(t["pnl"] for t in unique_t)
            print(f"  {name} unique: {len(unique)}T, {wr:.1f}% WR, ${pnl:+,.2f}")
            if wr >= 70:
                print(f"    -> STACKABLE! {len(unique)} new trades at {wr:.1f}% WR")
            else:
                print(f"    -> Not worth stacking ({wr:.1f}% < 70% threshold)")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": f"{date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}",
        "bars_1m": len(bars_1m), "bars_5m": len(bars_5m),
        "natural_up_pct": round(natural_up, 2),
        "v30": s30, "v12a": s12a, "v12b": s12b,
        "overlap": {
            "v30": len(v30_bars), "v12a": len(v12a_bars), "v12b": len(v12b_bars),
            "v30_v12a_overlap": len(v30_bars & v12a_bars),
            "v30_v12b_overlap": len(v30_bars & v12b_bars),
            "v12a_unique": len(v12a_bars - v30_bars),
            "v12b_unique": len(v12b_bars - v30_bars),
        },
    }
    out = Path(__file__).parent / "backtest_v12_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
