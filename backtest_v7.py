"""
Head-to-head backtest: V3.0 vs V7.0
V7.0: 5m regime (HTF MA, Range Filter, RSI) + 1m micro-confirmation (intra-bar returns, 1m RSI)

Uses 7 days of BTC 1-minute data from Binance.
Computes all indicators from raw data (no TradingView CSV needed).
"""

import json
import math
import time
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


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
        bars.append({
            "ts": k[0], "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
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
            "minutes": chunk,
        })
    return bars_5m


# ============================================================
# INDICATORS
# ============================================================

def compute_rsi(closes, period=14):
    rsi = [50.0] * len(closes)
    if len(closes) < period + 1:
        return rsi
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(0, d))
        losses.append(max(0, -d))
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        if avg_l == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    if sum(losses[:period]) > 0:
        rs = sum(gains[:period]) / sum(losses[:period])
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_ema(values, period):
    if not values:
        return []
    a = 2.0 / (period + 1.0)
    out = [values[0]]
    for i in range(1, len(values)):
        out.append(a * values[i] + (1 - a) * out[i - 1])
    return out


def compute_range_filter(closes, highs, lows, period=14, mult=2.5):
    n = len(closes)
    if n < period + 1:
        return closes[:]
    ranges = [highs[0] - lows[0]]
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        ranges.append(tr)
    smooth = compute_ema(ranges, period)
    rf = [closes[0]] * n
    for i in range(1, n):
        fr = smooth[i] * mult
        if closes[i] > rf[i - 1]:
            rf[i] = max(rf[i - 1], closes[i] - fr)
        elif closes[i] < rf[i - 1]:
            rf[i] = min(rf[i - 1], closes[i] + fr)
        else:
            rf[i] = rf[i - 1]
    return rf


def compute_1m_rsi(bars_1m, period=6):
    """Compute RSI on 1-minute closes with a fast period."""
    closes = [b["close"] for b in bars_1m]
    return compute_rsi(closes, period)


# ============================================================
# V3.0: Simple Momentum (baseline)
# ============================================================

def simulate_v30(bars_5m, size_usd=5.0):
    BUY = 0.51
    trades = []
    for bar in bars_5m:
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
        trades.append({"direction": d, "actual": actual, "win": d == actual,
                       "momentum_bps": mom, "pnl": pnl, "cost": cost,
                       "shares": sh, "buy_price": BUY})
    return trades


# ============================================================
# V7.0: 5m Regime + 1m Micro-Confirmation
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))


def fair_prob_up(prev5_close, prev5_htf_ma, prev5_range_filter, prev5_rsi,
                 m1_open, m1_close, m1_rsi,
                 m2_open=None, m2_close=None, m2_rsi=None):
    """
    V7 probability model.
    Uses prior 5m bar regime + first 1-2 minutes of current bar for micro-confirmation.
    Returns p_up in [0.08, 0.92].
    """
    score = 0.0

    # --- 5m regime (previous CLOSED bar) ---
    # Trend via HTF MA
    score += 0.6 if prev5_close > prev5_htf_ma else -0.6
    # Range filter bias
    score += 0.3 if prev5_close > prev5_range_filter else -0.3
    # RSI bias
    score += (prev5_rsi - 50.0) / 25.0 * 0.4

    # --- 1m micro (minute 1) ---
    ret1 = (m1_close - m1_open) / max(1e-9, m1_open)
    score += clamp(ret1 / 0.0008, -1.2, 1.2) * 0.7  # 8bp scaled
    score += (m1_rsi - 50.0) / 20.0 * 0.5

    # --- minute 2 (optional confirmation) ---
    if m2_close is not None and m2_open is not None and m2_rsi is not None:
        ret2 = (m2_close - m2_open) / max(1e-9, m2_open)
        score += clamp(ret2 / 0.0008, -1.2, 1.2) * 0.35
        # RSI slope confirmation
        score += clamp((m2_rsi - m1_rsi) / 10.0, -0.4, 0.4)

    p = sigmoid(score)
    p = clamp(0.15 + 0.70 * p, 0.08, 0.92)  # compress to realistic band
    return p


def simulate_v7(bars_5m, bars_1m, size_usd=5.0):
    """
    V7.0:
    - Prior 5m bar: HTF MA (EMA21), Range Filter, RSI(14) -> regime
    - First 1-2 minutes of current bar: 1m returns, 1m RSI(6) -> micro-confirmation
    - Decision band: p_up >= 0.54 for UP, <= 0.46 for DOWN
    - Edge gate: fair_yes - 0.51 >= 0 (backtest mode, no min_edge since no real book)
    - Settles on current bar (close > open = UP wins)
    """
    DECISION_BAND = 0.04
    BUY_PRICE = 0.51

    # Pre-compute 5m indicators
    n5 = len(bars_5m)
    closes_5m = [b["close"] for b in bars_5m]
    highs_5m = [b["high"] for b in bars_5m]
    lows_5m = [b["low"] for b in bars_5m]
    rsi_5m = compute_rsi(closes_5m, 14)
    htf_ma = compute_ema(closes_5m, 21)
    range_filter = compute_range_filter(closes_5m, highs_5m, lows_5m, 14, 2.5)

    # Pre-compute 1m RSI(6) on all 1-min bars
    rsi_1m = compute_1m_rsi(bars_1m, 6)

    # Build a mapping: 5m bar index -> 1m bar indices
    # Each 5m bar has 5 consecutive 1-min bars
    # bars_5m[i]["minutes"] already contains them, but we need RSI from the global 1m series
    # Since bars are sequential, bar_5m[i] corresponds to 1m indices [i*5, i*5+4]

    trades = []

    for i in range(max(1, 22), n5):  # need EMA21 warmup + prior bar
        bar = bars_5m[i]
        mins = bar["minutes"]
        if len(mins) < 5:
            continue

        # Prior 5m bar regime
        prev_close = closes_5m[i - 1]
        prev_htf_ma = htf_ma[i - 1]
        prev_rf = range_filter[i - 1]
        prev_rsi = rsi_5m[i - 1]

        # Current bar's 1-min data
        # Global 1m index for this 5m bar
        m1_global_idx = i * 5 + 1  # minute 1 (second 1-min candle)
        m2_global_idx = i * 5 + 2  # minute 2 (third 1-min candle)

        if m1_global_idx >= len(bars_1m):
            continue

        m1 = bars_1m[m1_global_idx]
        m1_rsi_val = rsi_1m[m1_global_idx] if m1_global_idx < len(rsi_1m) else 50.0

        # Minute 2 (optional)
        m2_open, m2_close, m2_rsi_val = None, None, None
        if m2_global_idx < len(bars_1m):
            m2 = bars_1m[m2_global_idx]
            m2_open = m2["open"]
            m2_close = m2["close"]
            m2_rsi_val = rsi_1m[m2_global_idx] if m2_global_idx < len(rsi_1m) else 50.0

        # Compute fair prob
        p_up = fair_prob_up(
            prev5_close=prev_close,
            prev5_htf_ma=prev_htf_ma,
            prev5_range_filter=prev_rf,
            prev5_rsi=prev_rsi,
            m1_open=m1["open"],
            m1_close=m1["close"],
            m1_rsi=m1_rsi_val,
            m2_open=m2_open,
            m2_close=m2_close,
            m2_rsi=m2_rsi_val,
        )

        # Decision
        if p_up >= 0.5 + DECISION_BAND:
            side = "UP"
            fair_yes = p_up
        elif p_up <= 0.5 - DECISION_BAND:
            side = "DOWN"
            fair_yes = 1.0 - p_up
        else:
            continue

        # Edge gate (backtest: just require fair_yes > ask)
        edge = fair_yes - BUY_PRICE
        if edge < 0:
            continue

        # Execute
        shares = max(1, math.floor(size_usd / BUY_PRICE))
        cost = BUY_PRICE * shares

        # Settlement: current bar close > open = UP wins
        up_win = bar["close"] > bar["open"]
        win = (up_win and side == "UP") or (not up_win and side == "DOWN")
        pnl = (shares * 1.0 - cost) if win else -cost

        trades.append({
            "direction": side,
            "actual": "UP" if up_win else "DOWN",
            "win": win,
            "p_up": p_up,
            "fair_yes": fair_yes,
            "edge": edge,
            "momentum_bps": (m1["close"] - mins[0]["open"]) / mins[0]["open"] * 10000,
            "pnl": pnl,
            "cost": cost,
            "shares": shares,
            "buy_price": BUY_PRICE,
        })

    return trades


def simulate_v7_no_regime(bars_5m, bars_1m, size_usd=5.0):
    """
    V7.0 variant: ONLY micro-confirmation (1m returns + 1m RSI), NO 5m regime.
    Tests if the 1m micro adds anything to V3.0's raw momentum.
    """
    DECISION_BAND = 0.04
    BUY_PRICE = 0.51

    rsi_1m = compute_1m_rsi(bars_1m, 6)
    n5 = len(bars_5m)
    trades = []

    for i in range(1, n5):
        bar = bars_5m[i]
        mins = bar["minutes"]
        if len(mins) < 5:
            continue

        m1_idx = i * 5 + 1
        m2_idx = i * 5 + 2
        if m1_idx >= len(bars_1m):
            continue

        m1 = bars_1m[m1_idx]
        m1_rsi = rsi_1m[m1_idx] if m1_idx < len(rsi_1m) else 50.0

        m2_open, m2_close, m2_rsi = None, None, None
        if m2_idx < len(bars_1m):
            m2 = bars_1m[m2_idx]
            m2_open = m2["open"]
            m2_close = m2["close"]
            m2_rsi = rsi_1m[m2_idx] if m2_idx < len(rsi_1m) else 50.0

        # Score using ONLY micro (no 5m regime)
        score = 0.0
        ret1 = (m1["close"] - m1["open"]) / max(1e-9, m1["open"])
        score += clamp(ret1 / 0.0008, -1.2, 1.2) * 0.7
        score += (m1_rsi - 50.0) / 20.0 * 0.5

        if m2_close is not None:
            ret2 = (m2_close - m2_open) / max(1e-9, m2_open)
            score += clamp(ret2 / 0.0008, -1.2, 1.2) * 0.35
            score += clamp((m2_rsi - m1_rsi) / 10.0, -0.4, 0.4)

        p = sigmoid(score)
        p_up = clamp(0.15 + 0.70 * p, 0.08, 0.92)

        if p_up >= 0.5 + DECISION_BAND:
            side = "UP"
            fair_yes = p_up
        elif p_up <= 0.5 - DECISION_BAND:
            side = "DOWN"
            fair_yes = 1.0 - p_up
        else:
            continue

        edge = fair_yes - BUY_PRICE
        if edge < 0:
            continue

        shares = max(1, math.floor(size_usd / BUY_PRICE))
        cost = BUY_PRICE * shares
        up_win = bar["close"] > bar["open"]
        win = (up_win and side == "UP") or (not up_win and side == "DOWN")
        pnl = (shares - cost) if win else -cost

        trades.append({
            "direction": side, "actual": "UP" if up_win else "DOWN",
            "win": win, "p_up": p_up, "momentum_bps": ret1 * 10000,
            "pnl": pnl, "cost": cost, "shares": shares, "buy_price": BUY_PRICE,
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

    # p_up distribution
    pups = [t.get("p_up", 0) for t in trades if t.get("p_up")]
    if pups:
        wp_up = [t["p_up"] for t in trades if t["win"] and t.get("p_up")]
        lp_up = [t["p_up"] for t in trades if not t["win"] and t.get("p_up")]
        print(f"  Avg p_up:      {sum(pups) / len(pups):.3f}")
        if wp_up:
            print(f"  Avg p_up (W):  {sum(wp_up) / len(wp_up):.3f}")
        if lp_up:
            print(f"  Avg p_up (L):  {sum(lp_up) / len(lp_up):.3f}")

    # Direction
    ups = [t for t in trades if t["direction"] == "UP"]
    dns = [t for t in trades if t["direction"] == "DOWN"]
    if ups:
        print(f"  UP trades:     {len(ups)} ({sum(1 for t in ups if t['win']) / len(ups) * 100:.1f}% WR)")
    if dns:
        print(f"  DOWN trades:   {len(dns)} ({sum(1 for t in dns if t['win']) / len(dns) * 100:.1f}% WR)")

    return {
        "label": label, "trades": total, "wins": wins,
        "win_rate": round(wr, 2), "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 4), "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4), "max_drawdown": round(max_dd, 2),
        "total_cost": round(total_cost, 2), "roi_pct": round(roi, 2),
        "max_win_streak": mws, "max_loss_streak": mls,
        "trades_per_day": round(tpd, 1),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("Fetching 7 days of BTC 1-min data from Binance...")
    bars_1m = fetch_binance_klines(days=7)
    print(f"  Got {len(bars_1m)} 1-min bars")
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

    # V7.0 full (5m regime + 1m micro)
    v7 = simulate_v7(bars_5m, bars_1m, SIZE)
    s7 = analyze(v7, "V7.0 - 5m Regime + 1m Micro (full)")

    # V7.0 micro-only (no 5m regime, just 1m confirmation)
    v7m = simulate_v7_no_regime(bars_5m, bars_1m, SIZE)
    s7m = analyze(v7m, "V7.0m - 1m Micro Only (no regime)")

    # Head to head
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD (all $5 sizing)")
    print("=" * 70)

    def row(label, *vals):
        parts = [f"  {label:20s}"]
        for v in vals:
            parts.append(f"{v:>16s}")
        print("".join(parts))

    row("", "V3.0", "V7.0 (full)", "V7.0m (micro)")
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
        for s in [s30, s7, s7m]:
            v = s.get(key, 0)
            if key in ["total_pnl", "avg_pnl", "avg_win", "avg_loss", "max_drawdown"]:
                vals.append(f"${v:+,.2f}")
            elif key in ["win_rate", "roi_pct", "trades_per_day"]:
                vals.append(f"{v:.1f}")
            else:
                vals.append(f"{v}")
        row(label, *vals)

    # Overlap
    print("\n" + "=" * 70)
    print("  OVERLAP ANALYSIS")
    print("=" * 70)

    v30_bars = set()
    for i, bar in enumerate(bars_5m):
        mins = bar["minutes"]
        if len(mins) < 5:
            continue
        op = mins[0]["open"]
        p2 = mins[1]["close"]
        mom = (p2 - op) / op * 10000
        if abs(mom) >= 15.0:
            v30_bars.add(i)

    v7_bars = set()
    for t in v7:
        # Find the bar index by matching open price
        for i, bar in enumerate(bars_5m):
            if bar["minutes"][0]["open"] == t.get("bar_open", None):
                v7_bars.add(i)
                break

    # Simpler: just track bar indices directly
    v7_bars2 = set()
    v7m_bars = set()

    # Re-run tracking indices
    closes_5m = [b["close"] for b in bars_5m]
    highs_5m = [b["high"] for b in bars_5m]
    lows_5m = [b["low"] for b in bars_5m]
    rsi_5m = compute_rsi(closes_5m, 14)
    htf_ma = compute_ema(closes_5m, 21)
    rf = compute_range_filter(closes_5m, highs_5m, lows_5m, 14, 2.5)
    rsi_1m = compute_1m_rsi(bars_1m, 6)

    for i in range(22, len(bars_5m)):
        mins = bars_5m[i]["minutes"]
        if len(mins) < 5:
            continue
        m1_idx = i * 5 + 1
        m2_idx = i * 5 + 2
        if m1_idx >= len(bars_1m):
            continue
        m1 = bars_1m[m1_idx]
        m1r = rsi_1m[m1_idx] if m1_idx < len(rsi_1m) else 50
        m2o, m2c, m2r = None, None, None
        if m2_idx < len(bars_1m):
            m2o = bars_1m[m2_idx]["open"]
            m2c = bars_1m[m2_idx]["close"]
            m2r = rsi_1m[m2_idx] if m2_idx < len(rsi_1m) else 50

        p_up = fair_prob_up(closes_5m[i - 1], htf_ma[i - 1], rf[i - 1], rsi_5m[i - 1],
                            m1["open"], m1["close"], m1r, m2o, m2c, m2r)
        if p_up >= 0.54 or p_up <= 0.46:
            fair_yes = p_up if p_up >= 0.54 else 1.0 - p_up
            if fair_yes - 0.51 >= 0:
                v7_bars2.add(i)

    both = v30_bars & v7_bars2
    v7_only = v7_bars2 - v30_bars
    v30_only = v30_bars - v7_bars2

    print(f"  V3.0 trades on:   {len(v30_bars)} bars")
    print(f"  V7.0 trades on:   {len(v7_bars2)} bars")
    print(f"  OVERLAP:          {len(both)} bars")
    print(f"  V7.0 UNIQUE:      {len(v7_only)} bars (not in V3.0)")
    print(f"  V3.0 UNIQUE:      {len(v30_only)} bars (not in V7.0)")

    if v7_only:
        w = sum(1 for i in v7_only if bars_5m[i]["close"] > bars_5m[i]["open"])
        print(f"  V7 unique UP%:    {w / len(v7_only) * 100:.1f}%")

    # Verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print(f"  V3.0:  {s30['trades']}T, {s30['win_rate']:.1f}% WR, ${s30['total_pnl']:+,.2f}, {s30['roi_pct']:+.1f}% ROI")
    print(f"  V7.0:  {s7['trades']}T, {s7['win_rate']:.1f}% WR, ${s7['total_pnl']:+,.2f}, {s7['roi_pct']:+.1f}% ROI")
    print(f"  V7.0m: {s7m['trades']}T, {s7m['win_rate']:.1f}% WR, ${s7m['total_pnl']:+,.2f}, {s7m['roi_pct']:+.1f}% ROI")

    if len(v7_only) > 20:
        print(f"\n  V7 has {len(v7_only)} unique trades - potential to stack with V3.0")
    elif len(v7_only) > 0:
        print(f"\n  V7 has {len(v7_only)} unique trades - marginal value")
    else:
        print(f"\n  V7 is a subset of V3.0 - no unique value")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": f"{date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}",
        "bars_1m": len(bars_1m), "bars_5m": len(bars_5m),
        "natural_up_pct": round(natural_up, 2),
        "v30": s30, "v7": s7, "v7m": s7m,
        "overlap": {"v30": len(v30_bars), "v7": len(v7_bars2),
                    "both": len(both), "v7_unique": len(v7_only)},
    }
    out = Path(__file__).parent / "backtest_v7_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
