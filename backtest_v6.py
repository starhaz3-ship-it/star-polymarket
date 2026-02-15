"""
Head-to-head backtest: V3.0 vs V5 vs V6.0
V6.0: RSI pullback + HTF MA trend + Range Filter + fair prob edge gating

Uses 7 days of BTC 1-minute data from Binance, aggregated to 5-min.
Computes RSI, EMA (HTF Moving Average), and Range Filter from raw data.
"""

import json
import math
import time
import statistics as stats_mod
import requests
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import List


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
        chunk = bars_1m[i:i+5]
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
    """Standard RSI."""
    rsi = [50.0] * len(closes)
    if len(closes) < period + 1:
        return rsi
    gains = []
    losses = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i-1]
        gains.append(max(0, delta))
        losses.append(max(0, -delta))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    # Fill initial values
    if avg_loss > 0:
        rs = (sum(gains[:period]) / period) / (sum(losses[:period]) / period)
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_ema(values, period):
    """Exponential Moving Average."""
    if not values:
        return []
    a = 2.0 / (period + 1.0)
    out = [values[0]]
    for i in range(1, len(values)):
        out.append(a * values[i] + (1 - a) * out[i-1])
    return out


def compute_range_filter(closes, highs, lows, period=14, multiplier=2.5):
    """Range Filter indicator.

    Based on smoothed ATR. The filter acts as a dynamic support/resistance level.
    Price above filter = bullish, below = bearish.
    """
    n = len(closes)
    if n < period + 1:
        return closes[:]

    # Compute smoothed range (similar to ATR)
    ranges = [highs[0] - lows[0]]
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        ranges.append(tr)

    # EMA of ranges
    smooth_range = compute_ema(ranges, period)

    # Range filter calculation
    rf = [closes[0]] * n
    for i in range(1, n):
        filt_range = smooth_range[i] * multiplier

        if closes[i] > rf[i-1]:
            rf[i] = max(rf[i-1], closes[i] - filt_range)
        elif closes[i] < rf[i-1]:
            rf[i] = min(rf[i-1], closes[i] + filt_range)
        else:
            rf[i] = rf[i-1]

    return rf


# ============================================================
# V3.0: Simple Momentum (baseline)
# ============================================================

def simulate_v30(bars_5m, size_usd=5.0):
    THRESHOLD_BPS = 15.0
    BUY_PRICE = 0.51
    trades = []
    for bar in bars_5m:
        mins = bar["minutes"]
        if len(mins) < 5:
            continue
        open_price = mins[0]["open"]
        price_at_2m = mins[1]["close"]
        mom_bps = (price_at_2m - open_price) / open_price * 10000
        if abs(mom_bps) < THRESHOLD_BPS:
            continue
        direction = "UP" if mom_bps > 0 else "DOWN"
        bar_close = bar["close"]
        actual_dir = "UP" if bar_close > open_price else "DOWN"
        shares = max(1, math.floor(size_usd / BUY_PRICE))
        cost = BUY_PRICE * shares
        pnl = (shares * 1.0 - cost) if direction == actual_dir else -cost
        trades.append({
            "direction": direction, "actual": actual_dir,
            "win": direction == actual_dir,
            "momentum_bps": mom_bps, "pnl": pnl,
            "cost": cost, "shares": shares, "buy_price": BUY_PRICE,
        })
    return trades


# ============================================================
# V6.0: RSI Pullback + HTF MA + Range Filter + Fair Prob
# ============================================================

def fair_prob(rsi, mom, trend, strength):
    """V6 fair probability model."""
    score = 0.0
    score += 0.6 * trend
    score += 0.4 * mom
    score += 0.3 * strength
    score += (rsi - 50) / 25.0
    p = 1.0 / (1.0 + math.exp(-max(-20, min(20, score))))
    return max(0.08, min(0.92, p))


def simulate_v6(bars_5m, size_usd=5.0):
    """
    V6.0: RSI pullback strategy.
    - Trend: close > HTF MA (EMA 21)
    - Pullback: RSI 40-50 for long, 50-60 for short
    - Momentum confirmation: close > prev_close for long
    - Range Filter: close > RF for long
    - Fair prob edge gating
    - Settles on NEXT bar
    """
    MIN_EDGE = 0.012

    n = len(bars_5m)
    closes = [b["close"] for b in bars_5m]
    opens = [b["open"] for b in bars_5m]
    highs = [b["high"] for b in bars_5m]
    lows = [b["low"] for b in bars_5m]

    # Compute indicators
    rsi = compute_rsi(closes, 14)
    htf_ma = compute_ema(closes, 21)  # "HTF Moving Average"
    rf = compute_range_filter(closes, highs, lows, 14, 2.5)

    trades = []

    for i in range(5, n - 1):  # need i+1 for settlement
        close = closes[i]
        prev_close = closes[i-1]
        open_ = opens[i]

        r = rsi[i]
        ma = htf_ma[i]
        range_f = rf[i]

        # --- REGIME ---
        trend_up = close > ma
        trend_dn = close < ma

        # --- PULLBACK ---
        pullback_long = trend_up and (40 <= r <= 50)
        pullback_short = trend_dn and (50 <= r <= 60)

        # --- MOMENTUM ---
        mom = (close - prev_close) / prev_close if prev_close > 0 else 0

        long_sig = pullback_long and mom > 0 and close > range_f
        short_sig = pullback_short and mom < 0 and close < range_f

        if not (long_sig or short_sig):
            continue

        side = "UP" if long_sig else "DOWN"

        # --- FAIR PROB ---
        trend_val = 1 if side == "UP" else -1
        strength = abs(mom) * 100

        p_up = fair_prob(r, mom * 100, trend_val, strength)
        fair_yes = p_up if side == "UP" else (1 - p_up)

        # --- MARKET PRICE (deterministic for reproducible backtest) ---
        mid = 0.50
        ask = 0.51  # standard ask for 5m markets
        edge = fair_yes - ask

        if edge < MIN_EDGE:
            continue

        # --- SIZE ---
        shares = max(1, math.floor(size_usd / ask))
        cost = ask * shares

        # --- SETTLEMENT (next bar) ---
        settle_bar = bars_5m[i + 1]
        up_win = settle_bar["close"] > settle_bar["open"]

        win = (up_win and side == "UP") or (not up_win and side == "DOWN")

        if win:
            pnl = shares * 1.0 - cost
        else:
            pnl = -cost

        trades.append({
            "direction": side,
            "actual": "UP" if up_win else "DOWN",
            "win": win,
            "momentum_bps": mom * 10000,
            "rsi": r,
            "edge": edge,
            "fair_yes": fair_yes,
            "pnl": pnl,
            "cost": cost,
            "shares": shares,
            "buy_price": ask,
        })

    return trades


def simulate_v6_current_bar(bars_5m, size_usd=5.0):
    """
    V6.0 variant: Same RSI pullback logic but settles on CURRENT bar
    (like Polymarket 5m markets actually work — you bet on THIS bar's outcome).
    """
    MIN_EDGE = 0.012

    n = len(bars_5m)
    closes = [b["close"] for b in bars_5m]
    opens = [b["open"] for b in bars_5m]
    highs = [b["high"] for b in bars_5m]
    lows = [b["low"] for b in bars_5m]

    # Use PRIOR bar indicators (can't use current bar's RSI before it closes)
    rsi = compute_rsi(closes, 14)
    htf_ma = compute_ema(closes, 21)
    rf = compute_range_filter(closes, highs, lows, 14, 2.5)

    trades = []

    for i in range(5, n):
        # Use prior bar's indicators (current bar hasn't closed yet when we'd enter)
        close_prev = closes[i-1]
        prev_prev = closes[i-2]

        r = rsi[i-1]
        ma = htf_ma[i-1]
        range_f = rf[i-1]

        # --- REGIME (from prior bar) ---
        trend_up = close_prev > ma
        trend_dn = close_prev < ma

        # --- PULLBACK ---
        pullback_long = trend_up and (40 <= r <= 50)
        pullback_short = trend_dn and (50 <= r <= 60)

        # --- MOMENTUM ---
        mom = (close_prev - prev_prev) / prev_prev if prev_prev > 0 else 0

        long_sig = pullback_long and mom > 0 and close_prev > range_f
        short_sig = pullback_short and mom < 0 and close_prev < range_f

        if not (long_sig or short_sig):
            continue

        side = "UP" if long_sig else "DOWN"

        # --- FAIR PROB ---
        trend_val = 1 if side == "UP" else -1
        strength = abs(mom) * 100
        p_up = fair_prob(r, mom * 100, trend_val, strength)
        fair_yes = p_up if side == "UP" else (1 - p_up)

        ask = 0.51
        edge = fair_yes - ask
        if edge < MIN_EDGE:
            continue

        shares = max(1, math.floor(size_usd / ask))
        cost = ask * shares

        # --- SETTLEMENT (current bar) ---
        bar = bars_5m[i]
        up_win = bar["close"] > bar["open"]
        win = (up_win and side == "UP") or (not up_win and side == "DOWN")

        pnl = (shares * 1.0 - cost) if win else -cost

        trades.append({
            "direction": side,
            "actual": "UP" if up_win else "DOWN",
            "win": win,
            "momentum_bps": mom * 10000,
            "rsi": r,
            "edge": edge,
            "fair_yes": fair_yes,
            "pnl": pnl,
            "cost": cost,
            "shares": shares,
            "buy_price": ask,
        })

    return trades


# ============================================================
# ANALYSIS
# ============================================================

def analyze(trades, label):
    if not trades:
        print(f"\n{'='*60}")
        print(f"  {label}: NO TRADES")
        print(f"{'='*60}")
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

    win_pnls = [t["pnl"] for t in trades if t["win"]]
    loss_pnls = [t["pnl"] for t in trades if not t["win"]]
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0

    cum = peak = max_dd = 0
    for t in trades:
        cum += t["pnl"]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    max_ws = max_ls = cur_w = cur_l = 0
    for t in trades:
        if t["win"]:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_ws = max(max_ws, cur_w)
        max_ls = max(max_ls, cur_l)

    roi = total_pnl / total_cost * 100 if total_cost > 0 else 0
    tpd = total / 7.0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades:        {total}  ({tpd:.1f}/day)")
    print(f"  Wins:          {wins} ({wr:.1f}%)")
    print(f"  Losses:        {total - wins} ({100-wr:.1f}%)")
    print(f"  Total PnL:     ${total_pnl:+,.2f}")
    print(f"  Avg PnL:       ${avg_pnl:+.2f}/trade")
    print(f"  Avg Win:       ${avg_win:+.2f}")
    print(f"  Avg Loss:      ${avg_loss:+.2f}")
    if avg_loss != 0:
        print(f"  W:L Ratio:     {abs(avg_win/avg_loss):.2f}x")
    print(f"  Total Cost:    ${total_cost:,.2f}")
    print(f"  ROI:           {roi:+.2f}%")
    print(f"  Max Drawdown:  ${max_dd:,.2f}")
    print(f"  Win Streak:    {max_ws}")
    print(f"  Loss Streak:   {max_ls}")

    # V6-specific: RSI distribution
    rsis = [t.get("rsi", 0) for t in trades if t.get("rsi")]
    if rsis:
        win_rsi = [t["rsi"] for t in trades if t["win"] and t.get("rsi")]
        loss_rsi = [t["rsi"] for t in trades if not t["win"] and t.get("rsi")]
        print(f"  Avg RSI:       {sum(rsis)/len(rsis):.1f}")
        if win_rsi:
            print(f"  Avg RSI (W):   {sum(win_rsi)/len(win_rsi):.1f}")
        if loss_rsi:
            print(f"  Avg RSI (L):   {sum(loss_rsi)/len(loss_rsi):.1f}")

    # Direction breakdown
    ups = [t for t in trades if t["direction"] == "UP"]
    dns = [t for t in trades if t["direction"] == "DOWN"]
    if ups:
        up_wr = sum(1 for t in ups if t["win"]) / len(ups) * 100
        print(f"  UP trades:     {len(ups)} ({up_wr:.1f}% WR)")
    if dns:
        dn_wr = sum(1 for t in dns if t["win"]) / len(dns) * 100
        print(f"  DOWN trades:   {len(dns)} ({dn_wr:.1f}% WR)")

    return {
        "label": label, "trades": total, "wins": wins,
        "win_rate": round(wr, 2), "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 4), "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4), "max_drawdown": round(max_dd, 2),
        "total_cost": round(total_cost, 2), "roi_pct": round(roi, 2),
        "max_win_streak": max_ws, "max_loss_streak": max_ls,
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

    SIZE = 5.0  # Fair comparison

    # V3.0
    v30 = simulate_v30(bars_5m, SIZE)
    s30 = analyze(v30, "V3.0 — Simple Momentum (2min, 15bp, $5)")

    # V6.0 next-bar settlement
    v6 = simulate_v6(bars_5m, SIZE)
    s6 = analyze(v6, "V6.0 — RSI Pullback + MA + RF (next-bar settle)")

    # V6.0 current-bar settlement
    v6c = simulate_v6_current_bar(bars_5m, SIZE)
    s6c = analyze(v6c, "V6.0c — RSI Pullback + MA + RF (current-bar settle)")

    # Head to head
    print("\n" + "="*70)
    print("  HEAD-TO-HEAD COMPARISON (all $5 sizing)")
    print("="*70)

    def row(label, *vals):
        parts = [f"  {label:20s}"]
        for v in vals:
            parts.append(f"{v:>16s}")
        print("".join(parts))

    row("", "V3.0", "V6.0 (next)", "V6.0c (current)")
    row("-"*20, "-"*16, "-"*16, "-"*16)

    for label, key in [
        ("Trades", "trades"),
        ("Trades/day", "trades_per_day"),
        ("Win Rate %", "win_rate"),
        ("Total PnL", "total_pnl"),
        ("Avg PnL/trade", "avg_pnl"),
        ("Avg Win", "avg_win"),
        ("Avg Loss", "avg_loss"),
        ("Max Drawdown", "max_drawdown"),
        ("ROI%", "roi_pct"),
        ("Win Streak", "max_win_streak"),
        ("Loss Streak", "max_loss_streak"),
    ]:
        vals = []
        for s in [s30, s6, s6c]:
            v = s.get(key, 0)
            if key in ["total_pnl", "avg_pnl", "avg_win", "avg_loss", "max_drawdown"]:
                vals.append(f"${v:+,.2f}")
            elif key in ["win_rate", "roi_pct", "trades_per_day"]:
                vals.append(f"{v:.1f}")
            else:
                vals.append(f"{v}")
        row(label, *vals)

    # Check overlap with V3.0
    print("\n" + "="*70)
    print("  OVERLAP ANALYSIS")
    print("="*70)

    # Build bar index sets
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

    # V6 uses prior bar signals -> maps to bar i+1 for next-bar, bar i for current-bar
    closes = [b["close"] for b in bars_5m]
    opens = [b["open"] for b in bars_5m]
    highs = [b["high"] for b in bars_5m]
    lows = [b["low"] for b in bars_5m]
    rsi_vals = compute_rsi(closes, 14)
    ma_vals = compute_ema(closes, 21)
    rf_vals = compute_range_filter(closes, highs, lows, 14, 2.5)

    v6_next_bars = set()
    v6_curr_bars = set()
    for i in range(5, len(bars_5m) - 1):
        close = closes[i]
        prev_close = closes[i-1]
        r = rsi_vals[i]
        ma = ma_vals[i]
        range_f = rf_vals[i]
        trend_up = close > ma
        trend_dn = close < ma
        pullback_long = trend_up and (40 <= r <= 50)
        pullback_short = trend_dn and (50 <= r <= 60)
        mom = (close - prev_close) / prev_close if prev_close > 0 else 0
        long_sig = pullback_long and mom > 0 and close > range_f
        short_sig = pullback_short and mom < 0 and close < range_f
        if long_sig or short_sig:
            side = "UP" if long_sig else "DOWN"
            trend_val = 1 if side == "UP" else -1
            strength = abs(mom) * 100
            p_up = fair_prob(r, mom * 100, trend_val, strength)
            fair_yes = p_up if side == "UP" else (1 - p_up)
            edge = fair_yes - 0.51
            if edge >= 0.012:
                v6_next_bars.add(i + 1)  # settles on next bar

    # Current-bar variant uses prior bar signals
    for i in range(5, len(bars_5m)):
        close_prev = closes[i-1]
        prev_prev = closes[i-2]
        r = rsi_vals[i-1]
        ma = ma_vals[i-1]
        range_f = rf_vals[i-1]
        trend_up = close_prev > ma
        trend_dn = close_prev < ma
        pullback_long = trend_up and (40 <= r <= 50)
        pullback_short = trend_dn and (50 <= r <= 60)
        mom = (close_prev - prev_prev) / prev_prev if prev_prev > 0 else 0
        long_sig = pullback_long and mom > 0 and close_prev > range_f
        short_sig = pullback_short and mom < 0 and close_prev < range_f
        if long_sig or short_sig:
            side = "UP" if long_sig else "DOWN"
            trend_val = 1 if side == "UP" else -1
            strength = abs(mom) * 100
            p_up = fair_prob(r, mom * 100, trend_val, strength)
            fair_yes = p_up if side == "UP" else (1 - p_up)
            edge = fair_yes - 0.51
            if edge >= 0.012:
                v6_curr_bars.add(i)

    both_next = v30_bars & v6_next_bars
    v6_only_next = v6_next_bars - v30_bars
    both_curr = v30_bars & v6_curr_bars
    v6_only_curr = v6_curr_bars - v30_bars

    print(f"  V3.0 trades on:        {len(v30_bars)} bars")
    print(f"  V6 next-bar trades on: {len(v6_next_bars)} bars")
    print(f"  V6 curr-bar trades on: {len(v6_curr_bars)} bars")
    print(f"  V6-next UNIQUE:        {len(v6_only_next)} bars (not in V3.0)")
    print(f"  V6-curr UNIQUE:        {len(v6_only_curr)} bars (not in V3.0)")
    print(f"  V6-next OVERLAP:       {len(both_next)} bars")
    print(f"  V6-curr OVERLAP:       {len(both_curr)} bars")

    if v6_only_next:
        # Check WR of V6-only trades
        w = l = 0
        for i in v6_only_next:
            bar = bars_5m[i]
            up_win = bar["close"] > bar["open"]
            # Need to know what direction V6 predicted — approximate
            # For next-bar, the signal was at bar i-1
            if i > 0 and i < len(bars_5m):
                up_win_val = bar["close"] > bar["open"]
                # Just check raw bar direction
                w += 1 if up_win_val else 0
                l += 1 if not up_win_val else 0
        print(f"  V6-next unique UP%:    {w/(w+l)*100:.1f}% (natural direction)")

    if v6_only_curr:
        w = l = 0
        for i in v6_only_curr:
            bar = bars_5m[i]
            up_win = bar["close"] > bar["open"]
            w += 1 if up_win else 0
            l += 1 if not up_win else 0
        print(f"  V6-curr unique UP%:    {w/(w+l)*100:.1f}% (natural direction)")

    # Verdict
    print("\n" + "="*70)
    print("  VERDICT")
    print("="*70)
    print(f"  V3.0:  {s30['trades']} trades, {s30['win_rate']:.1f}% WR, ${s30['total_pnl']:+,.2f} PnL, {s30['roi_pct']:+.1f}% ROI")
    print(f"  V6.0:  {s6['trades']} trades, {s6['win_rate']:.1f}% WR, ${s6['total_pnl']:+,.2f} PnL, {s6['roi_pct']:+.1f}% ROI")
    print(f"  V6.0c: {s6c['trades']} trades, {s6c['win_rate']:.1f}% WR, ${s6c['total_pnl']:+,.2f} PnL, {s6c['roi_pct']:+.1f}% ROI")

    can_stack = len(v6_only_next) > 0 or len(v6_only_curr) > 0
    if can_stack:
        print(f"\n  V6 has {len(v6_only_next)} next-bar / {len(v6_only_curr)} curr-bar UNIQUE trades")
        print(f"  Could potentially run alongside V3.0 for extra entries")
    else:
        print(f"\n  V6 is a subset of V3.0 — no unique trades to add")

    # Save
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": f"{date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}",
        "bars_1m": len(bars_1m), "bars_5m": len(bars_5m),
        "natural_up_pct": round(natural_up, 2),
        "v30": s30, "v6_next": s6, "v6_current": s6c,
        "overlap": {
            "v30_bars": len(v30_bars),
            "v6_next_unique": len(v6_only_next),
            "v6_curr_unique": len(v6_only_curr),
        },
    }
    out = Path(__file__).parent / "backtest_v6_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
