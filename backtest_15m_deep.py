"""
Deep exploration of 15-minute BTC Polymarket edges.

The 2-min momentum continuation effect decays on 15m bars (89% -> 71% WR).
This script tests fundamentally different approaches:

1. MEAN REVERSION: First 2-5 min spike -> expect pullback over 15 min
2. TREND ALIGNMENT: Use higher timeframe (1h, 4h) trend to filter direction
3. VOLUME PROFILE: Accumulation/distribution patterns over longer windows
4. TIME-OF-DAY: Specific hours with directional bias (Asian/EU/US sessions)
5. MULTI-BAR CONTEXT: Use prior 15m bar(s) to predict current bar direction
6. LATE ENTRY: Wait longer (5-7 min) for clearer signal on 15m bars
7. REVERSAL DETECTION: When first 2-5 min move hard, predict reversal (anti-momentum)
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


BUY_PRICE = 0.51
SIZE_USD = 5.0


def pnl_calc(win):
    sh = max(1, math.floor(SIZE_USD / BUY_PRICE))
    cost = BUY_PRICE * sh
    return (sh - cost) if win else -cost, cost


def analyze(trades, label, days=7):
    if not trades:
        print(f"  {label}: NO TRADES")
        return {"label": label, "trades": 0, "win_rate": 0, "total_pnl": 0,
                "trades_per_day": 0, "max_drawdown": 0, "roi_pct": 0}
    total = len(trades)
    wins = sum(1 for t in trades if t["win"])
    wr = wins / total * 100
    total_pnl = sum(t["pnl"] for t in trades)
    total_cost = sum(t["cost"] for t in trades)
    roi = total_pnl / total_cost * 100 if total_cost > 0 else 0
    tpd = total / days

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

    print(f"  {label}")
    print(f"    {total}T ({tpd:.1f}/d) | {wr:.1f}% WR | ${total_pnl:+.2f} PnL | "
          f"{roi:+.1f}% ROI | DD=${max_dd:.2f} | W{mws}/L{mls}")
    return {"label": label, "trades": total, "win_rate": round(wr, 2),
            "total_pnl": round(total_pnl, 2), "trades_per_day": round(tpd, 1),
            "max_drawdown": round(max_dd, 2), "roi_pct": round(roi, 2),
            "max_win_streak": mws, "max_loss_streak": mls}


# ============================================================
# STRATEGY 1: MEAN REVERSION (anti-momentum on 15m)
# If first 2 min spike hard, predict REVERSAL over remaining 13 min
# ============================================================

def strat_mean_reversion(bars_15m, threshold_bps=15.0):
    trades = []
    for bar in bars_15m:
        mins = bar["minutes"]
        op = mins[0]["open"]
        p2 = mins[1]["close"]
        mom = (p2 - op) / op * 10000
        if abs(mom) < threshold_bps:
            continue
        # REVERSE the momentum signal
        d = "DOWN" if mom > 0 else "UP"
        actual = "UP" if bar["close"] > op else "DOWN"
        pnl, cost = pnl_calc(d == actual)
        trades.append({"win": d == actual, "pnl": pnl, "cost": cost, "mom": mom})
    return trades


# ============================================================
# STRATEGY 2: TREND ALIGNMENT (use prior bars as trend filter)
# Only trade in direction of prevailing trend from prior N bars
# ============================================================

def strat_trend_aligned(bars_15m, lookback=4, threshold_bps=10.0):
    trades = []
    for i in range(lookback, len(bars_15m)):
        bar = bars_15m[i]
        mins = bar["minutes"]
        op = mins[0]["open"]
        p2 = mins[1]["close"]
        mom = (p2 - op) / op * 10000
        if abs(mom) < threshold_bps:
            continue

        # Check prior bars for trend
        prior_closes = [bars_15m[j]["close"] for j in range(i - lookback, i)]
        trend_up = prior_closes[-1] > prior_closes[0]
        signal = "UP" if mom > 0 else "DOWN"

        # Only trade if momentum aligns with prior trend
        if signal == "UP" and not trend_up:
            continue
        if signal == "DOWN" and trend_up:
            continue

        actual = "UP" if bar["close"] > op else "DOWN"
        pnl, cost = pnl_calc(signal == actual)
        trades.append({"win": signal == actual, "pnl": pnl, "cost": cost})
    return trades


# ============================================================
# STRATEGY 3: TIME-OF-DAY (directional bias by session)
# ============================================================

def strat_time_of_day(bars_15m, bars_1m):
    """Analyze hour-by-hour directional bias, then trade only strong hours."""
    from collections import defaultdict
    # First pass: compute per-hour UP%
    hour_stats = defaultdict(lambda: {"up": 0, "total": 0})
    for bar in bars_15m:
        hr = datetime.fromtimestamp(bar["ts"] / 1000, tz=timezone.utc).hour
        actual_up = bar["close"] > bar["open"]
        hour_stats[hr]["total"] += 1
        if actual_up:
            hour_stats[hr]["up"] += 1

    # Find biased hours (>60% or <40% UP rate with enough samples)
    biased_hours = {}
    for hr, s in hour_stats.items():
        if s["total"] >= 5:
            up_pct = s["up"] / s["total"]
            if up_pct >= 0.60:
                biased_hours[hr] = "UP"
            elif up_pct <= 0.40:
                biased_hours[hr] = "DOWN"

    # Second pass: trade biased hours
    trades = []
    for bar in bars_15m:
        hr = datetime.fromtimestamp(bar["ts"] / 1000, tz=timezone.utc).hour
        if hr not in biased_hours:
            continue
        signal = biased_hours[hr]
        actual = "UP" if bar["close"] > bar["open"] else "DOWN"
        pnl, cost = pnl_calc(signal == actual)
        trades.append({"win": signal == actual, "pnl": pnl, "cost": cost, "hour": hr})

    return trades, biased_hours


# ============================================================
# STRATEGY 4: MULTI-BAR PATTERN (prior bar predicts next)
# ============================================================

def strat_prior_bar_continuation(bars_15m):
    """If prior bar closed UP, predict current bar UP (and vice versa)."""
    trades = []
    for i in range(1, len(bars_15m)):
        prev = bars_15m[i - 1]
        bar = bars_15m[i]
        prev_up = prev["close"] > prev["open"]
        signal = "UP" if prev_up else "DOWN"
        actual = "UP" if bar["close"] > bar["open"] else "DOWN"
        pnl, cost = pnl_calc(signal == actual)
        trades.append({"win": signal == actual, "pnl": pnl, "cost": cost})
    return trades


def strat_prior_bar_reversal(bars_15m):
    """If prior bar closed UP, predict current bar DOWN (mean reversion)."""
    trades = []
    for i in range(1, len(bars_15m)):
        prev = bars_15m[i - 1]
        bar = bars_15m[i]
        prev_up = prev["close"] > prev["open"]
        signal = "DOWN" if prev_up else "UP"  # reverse
        actual = "UP" if bar["close"] > bar["open"] else "DOWN"
        pnl, cost = pnl_calc(signal == actual)
        trades.append({"win": signal == actual, "pnl": pnl, "cost": cost})
    return trades


# ============================================================
# STRATEGY 5: LATE ENTRY (wait 5-7 min instead of 2 min)
# ============================================================

def strat_late_entry(bars_15m, wait_min=5, threshold_bps=10.0):
    trades = []
    for bar in bars_15m:
        mins = bar["minutes"]
        if len(mins) < wait_min + 1:
            continue
        op = mins[0]["open"]
        px = mins[wait_min - 1]["close"]
        mom = (px - op) / op * 10000
        if abs(mom) < threshold_bps:
            continue
        d = "UP" if mom > 0 else "DOWN"
        actual = "UP" if bar["close"] > op else "DOWN"
        pnl, cost = pnl_calc(d == actual)
        trades.append({"win": d == actual, "pnl": pnl, "cost": cost, "mom": mom})
    return trades


# ============================================================
# STRATEGY 6: VOLUME-WEIGHTED MOMENTUM (combine price + volume)
# Higher volume in first 2 min + momentum = stronger signal
# ============================================================

def strat_vol_weighted_momentum(bars_15m, bars_1m, threshold_bps=10.0, vol_min=1.5):
    vols = np.array([b["volume"] for b in bars_1m], dtype=float)
    vol_sma = np.full_like(vols, np.nan)
    for i in range(19, len(vols)):
        vol_sma[i] = np.mean(vols[i - 19:i + 1])

    trades = []
    for idx, bar in enumerate(bars_15m):
        mins = bar["minutes"]
        op = mins[0]["open"]
        p2 = mins[1]["close"]
        mom = (p2 - op) / op * 10000
        if abs(mom) < threshold_bps:
            continue

        # Volume check
        m1_global = idx * 15 + 1
        if m1_global >= len(vols) or not np.isfinite(vol_sma[m1_global]) or vol_sma[m1_global] <= 0:
            continue
        v2 = mins[0]["volume"] + mins[1]["volume"]
        vol_ratio = v2 / (vol_sma[m1_global] * 2)
        if vol_ratio < vol_min:
            continue

        d = "UP" if mom > 0 else "DOWN"
        actual = "UP" if bar["close"] > op else "DOWN"
        pnl, cost = pnl_calc(d == actual)
        trades.append({"win": d == actual, "pnl": pnl, "cost": cost,
                       "mom": mom, "vol_ratio": vol_ratio})
    return trades


# ============================================================
# STRATEGY 7: COMBINED — trend + momentum + volume (kitchen sink)
# ============================================================

def strat_combined(bars_15m, bars_1m, lookback=4, mom_thresh=10.0, vol_min=1.3):
    vols = np.array([b["volume"] for b in bars_1m], dtype=float)
    vol_sma = np.full_like(vols, np.nan)
    for i in range(19, len(vols)):
        vol_sma[i] = np.mean(vols[i - 19:i + 1])

    trades = []
    for idx in range(lookback, len(bars_15m)):
        bar = bars_15m[idx]
        mins = bar["minutes"]
        op = mins[0]["open"]
        p2 = mins[1]["close"]
        mom = (p2 - op) / op * 10000
        if abs(mom) < mom_thresh:
            continue

        # Trend filter
        prior_closes = [bars_15m[j]["close"] for j in range(idx - lookback, idx)]
        trend_up = prior_closes[-1] > prior_closes[0]
        signal = "UP" if mom > 0 else "DOWN"
        if signal == "UP" and not trend_up:
            continue
        if signal == "DOWN" and trend_up:
            continue

        # Volume filter
        m1_global = idx * 15 + 1
        if m1_global >= len(vols) or not np.isfinite(vol_sma[m1_global]) or vol_sma[m1_global] <= 0:
            continue
        v2 = mins[0]["volume"] + mins[1]["volume"]
        vol_ratio = v2 / (vol_sma[m1_global] * 2)
        if vol_ratio < vol_min:
            continue

        # Order flow confirmation
        total_buy = mins[0]["taker_buy"] + mins[1]["taker_buy"]
        total_vol_2m = mins[0]["volume"] + mins[1]["volume"]
        buy_ratio = total_buy / total_vol_2m if total_vol_2m > 0 else 0.5
        if signal == "UP" and buy_ratio < 0.52:
            continue  # momentum up but sellers dominate = skip
        if signal == "DOWN" and buy_ratio > 0.48:
            continue  # momentum down but buyers dominate = skip

        actual = "UP" if bar["close"] > op else "DOWN"
        pnl, cost = pnl_calc(signal == actual)
        trades.append({"win": signal == actual, "pnl": pnl, "cost": cost,
                       "mom": mom, "vol_ratio": vol_ratio, "buy_ratio": buy_ratio})
    return trades


# ============================================================
# STRATEGY 8: EMA CROSSOVER (uses 1m EMAs computed at 2-min mark)
# ============================================================

def strat_ema_cross(bars_15m, bars_1m, fast=9, slow=21):
    """At the 2-min mark of each 15m bar, check if EMA fast > EMA slow on 1m chart."""
    closes_1m = np.array([b["close"] for b in bars_1m], dtype=float)
    # Compute EMAs
    ema_fast = np.full_like(closes_1m, np.nan)
    ema_slow = np.full_like(closes_1m, np.nan)
    af = 2.0 / (fast + 1)
    als = 2.0 / (slow + 1)
    ema_fast[0] = closes_1m[0]
    ema_slow[0] = closes_1m[0]
    for i in range(1, len(closes_1m)):
        ema_fast[i] = af * closes_1m[i] + (1 - af) * ema_fast[i - 1]
        ema_slow[i] = als * closes_1m[i] + (1 - als) * ema_slow[i - 1]

    trades = []
    for idx, bar in enumerate(bars_15m):
        m1_global = idx * 15 + 1  # 2 min in
        if m1_global >= len(closes_1m) or m1_global < slow:
            continue
        if np.isnan(ema_fast[m1_global]) or np.isnan(ema_slow[m1_global]):
            continue

        # EMA cross direction at signal time
        fast_above = ema_fast[m1_global] > ema_slow[m1_global]
        # Also check that cross is fresh (wasn't the same on prior bar)
        m0_global = idx * 15
        prev_fast_above = ema_fast[m0_global] > ema_slow[m0_global] if m0_global >= slow else fast_above

        signal = "UP" if fast_above else "DOWN"
        actual = "UP" if bar["close"] > bar["open"] else "DOWN"
        pnl, cost = pnl_calc(signal == actual)
        trades.append({"win": signal == actual, "pnl": pnl, "cost": cost,
                       "fresh_cross": fast_above != prev_fast_above})
    return trades


# ============================================================
# STRATEGY 9: CANDLE PATTERN (engulfing, pin bars on prior 15m)
# ============================================================

def strat_candle_pattern(bars_15m):
    """Look for bullish/bearish engulfing on prior bar to predict current direction."""
    trades = []
    for i in range(2, len(bars_15m)):
        prev2 = bars_15m[i - 2]
        prev = bars_15m[i - 1]
        bar = bars_15m[i]

        prev2_body = prev2["close"] - prev2["open"]
        prev_body = prev["close"] - prev["open"]

        # Bullish engulfing: prev2 red, prev green and body covers prev2
        if prev2_body < 0 and prev_body > 0 and prev_body > abs(prev2_body):
            signal = "UP"
        # Bearish engulfing
        elif prev2_body > 0 and prev_body < 0 and abs(prev_body) > prev2_body:
            signal = "DOWN"
        else:
            continue

        actual = "UP" if bar["close"] > bar["open"] else "DOWN"
        pnl, cost = pnl_calc(signal == actual)
        trades.append({"win": signal == actual, "pnl": pnl, "cost": cost})
    return trades


# ============================================================
# STRATEGY 10: RSI EXTREME (2-period RSI on 15m bars, extremes predict reversal)
# ============================================================

def strat_rsi_extreme(bars_15m, period=2, os_thresh=15, ob_thresh=85):
    """2-period RSI on 15m closes. Extreme oversold -> UP, extreme overbought -> DOWN."""
    closes = [b["close"] for b in bars_15m]
    trades = []

    for i in range(period + 1, len(bars_15m)):
        # Compute RSI(2) using previous bars (known at bar open)
        gains = []
        losses = []
        for j in range(i - period, i):
            change = closes[j] - closes[j - 1]
            gains.append(max(0, change))
            losses.append(max(0, -change))

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        if avg_loss == 0:
            rsi = 100
        elif avg_gain == 0:
            rsi = 0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - 100 / (1 + rs)

        if rsi <= os_thresh:
            signal = "UP"  # oversold -> bounce
        elif rsi >= ob_thresh:
            signal = "DOWN"  # overbought -> pullback
        else:
            continue

        bar = bars_15m[i]
        actual = "UP" if bar["close"] > bar["open"] else "DOWN"
        pnl, cost = pnl_calc(signal == actual)
        trades.append({"win": signal == actual, "pnl": pnl, "cost": cost, "rsi": rsi})
    return trades


# ============================================================
# MAIN
# ============================================================

def main():
    DAYS = 7
    print("=" * 70)
    print("  DEEP 15-MINUTE BTC STRATEGY EXPLORATION")
    print("=" * 70)
    print(f"\nFetching {DAYS} days of BTC 1-min data...")
    bars_1m = fetch_binance_klines(days=DAYS)
    bars_15m = aggregate_Nm(bars_1m, 15)
    print(f"  {len(bars_1m)} 1m bars -> {len(bars_15m)} 15m bars")

    date_start = datetime.fromtimestamp(bars_1m[0]["ts"] / 1000, tz=timezone.utc)
    date_end = datetime.fromtimestamp(bars_1m[-1]["ts"] / 1000, tz=timezone.utc)
    nat_15m = sum(1 for b in bars_15m if b["close"] > b["open"]) / len(bars_15m) * 100
    print(f"  Range: {date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}")
    print(f"  Natural UP%: {nat_15m:.1f}%\n")

    results = {}

    # --- 1. MEAN REVERSION ---
    print("=" * 60)
    print("  1. MEAN REVERSION (anti-momentum)")
    print("=" * 60)
    for thresh in [10, 15, 20, 30]:
        t = strat_mean_reversion(bars_15m, thresh)
        s = analyze(t, f"MeanRev {thresh}bp", DAYS)
        if thresh == 15:
            results["mean_reversion"] = s

    # --- 2. TREND ALIGNED ---
    print(f"\n{'=' * 60}")
    print("  2. TREND ALIGNED (prior bars + momentum)")
    print("=" * 60)
    for lb in [2, 4, 8]:
        for thresh in [10, 15, 20]:
            t = strat_trend_aligned(bars_15m, lb, thresh)
            s = analyze(t, f"Trend LB={lb} {thresh}bp", DAYS)
            if lb == 4 and thresh == 15:
                results["trend_aligned"] = s

    # --- 3. TIME OF DAY ---
    print(f"\n{'=' * 60}")
    print("  3. TIME-OF-DAY BIAS")
    print("=" * 60)
    t, biased = strat_time_of_day(bars_15m, bars_1m)
    s = analyze(t, "Time-of-Day", DAYS)
    results["time_of_day"] = s
    if biased:
        print(f"    Biased hours: {biased}")

    # --- 4. PRIOR BAR PATTERNS ---
    print(f"\n{'=' * 60}")
    print("  4. PRIOR BAR PATTERNS")
    print("=" * 60)
    t = strat_prior_bar_continuation(bars_15m)
    results["prior_continuation"] = analyze(t, "Prior bar continuation", DAYS)
    t = strat_prior_bar_reversal(bars_15m)
    results["prior_reversal"] = analyze(t, "Prior bar reversal", DAYS)

    # --- 5. LATE ENTRY ---
    print(f"\n{'=' * 60}")
    print("  5. LATE ENTRY (wait longer)")
    print("=" * 60)
    for wait in [3, 5, 7, 10]:
        for thresh in [10, 15, 20]:
            t = strat_late_entry(bars_15m, wait, thresh)
            s = analyze(t, f"Late {wait}min {thresh}bp", DAYS)
            if wait == 5 and thresh == 15:
                results["late_entry"] = s

    # --- 6. VOLUME-WEIGHTED MOMENTUM ---
    print(f"\n{'=' * 60}")
    print("  6. VOLUME-WEIGHTED MOMENTUM")
    print("=" * 60)
    for thresh in [10, 15, 20]:
        for vmin in [1.0, 1.5, 2.0]:
            t = strat_vol_weighted_momentum(bars_15m, bars_1m, thresh, vmin)
            s = analyze(t, f"VolMom {thresh}bp vol>={vmin}x", DAYS)
            if thresh == 15 and vmin == 1.5:
                results["vol_momentum"] = s

    # --- 7. COMBINED (kitchen sink) ---
    print(f"\n{'=' * 60}")
    print("  7. COMBINED (trend + momentum + volume + flow)")
    print("=" * 60)
    for mom in [10, 15]:
        for vmin in [1.0, 1.3, 1.5]:
            t = strat_combined(bars_15m, bars_1m, 4, mom, vmin)
            s = analyze(t, f"Combined mom={mom} vol>={vmin}", DAYS)
            if mom == 10 and vmin == 1.3:
                results["combined"] = s

    # --- 8. EMA CROSS ---
    print(f"\n{'=' * 60}")
    print("  8. EMA CROSSOVER (1m EMAs at signal time)")
    print("=" * 60)
    for fast, slow in [(5, 13), (9, 21), (12, 26)]:
        t = strat_ema_cross(bars_15m, bars_1m, fast, slow)
        s = analyze(t, f"EMA {fast}/{slow}", DAYS)
        if fast == 9:
            results["ema_cross"] = s
            # Filter to fresh crosses only
            fresh = [x for x in t if x.get("fresh_cross")]
            if fresh:
                analyze(fresh, f"EMA {fast}/{slow} FRESH cross only", DAYS)

    # --- 9. CANDLE PATTERNS ---
    print(f"\n{'=' * 60}")
    print("  9. CANDLE PATTERNS (engulfing)")
    print("=" * 60)
    t = strat_candle_pattern(bars_15m)
    results["candle_pattern"] = analyze(t, "Engulfing pattern", DAYS)

    # --- 10. RSI EXTREME ---
    print(f"\n{'=' * 60}")
    print("  10. RSI EXTREME (2-period on 15m)")
    print("=" * 60)
    for os_t, ob_t in [(10, 90), (15, 85), (20, 80), (25, 75)]:
        t = strat_rsi_extreme(bars_15m, 2, os_t, ob_t)
        s = analyze(t, f"RSI2 OS<{os_t} OB>{ob_t}", DAYS)
        if os_t == 15:
            results["rsi_extreme"] = s

    # --- LEADERBOARD ---
    print("\n" + "=" * 70)
    print("  LEADERBOARD (sorted by PnL)")
    print("=" * 70)

    ranked = sorted(results.values(), key=lambda x: x.get("total_pnl", 0), reverse=True)
    for i, s in enumerate(ranked, 1):
        if s["trades"] > 0:
            print(f"  #{i:2d}  {s['label']:40s}  {s['trades']:4d}T  {s['win_rate']:5.1f}%  "
                  f"${s['total_pnl']:+8.2f}  DD=${s['max_drawdown']:.2f}")

    # Save
    out = Path(__file__).parent / "backtest_15m_deep_results.json"
    out.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bars_15m": len(bars_15m),
        "natural_up_pct": round(nat_15m, 2),
        "strategies": results,
    }, indent=2))
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
