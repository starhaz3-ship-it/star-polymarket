"""
Head-to-head backtest: V3.0 vs V11.0
V11.0: Vortex Indicator + Coppock Curve + KAMA on 5m regime
       + Fisher Transform + Aroon Oscillator + OBV slope on 1m micro
       + Sigmoid probability model with decision band

Uses 7 days of BTC 1-minute data from Binance.
Computes all indicators from raw data (no TradingView CSV needed).
"""

import json
import math
import time
import requests
import numpy as np
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
# HELPERS
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))


def np_sma(a, n):
    out = np.full_like(a, np.nan, dtype=float)
    if n <= 0 or len(a) < n:
        return out
    c = np.cumsum(a, dtype=float)
    c[n:] = c[n:] - c[:-n]
    out[n - 1:] = c[n - 1:] / n
    return out


def np_ema(a, n):
    out = np.empty_like(a, dtype=float)
    alpha = 2.0 / (n + 1.0)
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1.0 - alpha) * out[i - 1]
    return out


def np_wma(a, n):
    out = np.full_like(a, np.nan, dtype=float)
    if len(a) < n:
        return out
    w = np.arange(1, n + 1, dtype=float)
    ws = float(np.sum(w))
    for i in range(n - 1, len(a)):
        out[i] = float(np.sum(a[i - n + 1:i + 1] * w) / ws)
    return out


def rolling_max(a, n):
    out = np.full_like(a, np.nan, dtype=float)
    if len(a) < n:
        return out
    for i in range(n - 1, len(a)):
        out[i] = float(np.max(a[i - n + 1:i + 1]))
    return out


def rolling_min(a, n):
    out = np.full_like(a, np.nan, dtype=float)
    if len(a) < n:
        return out
    for i in range(n - 1, len(a)):
        out[i] = float(np.min(a[i - n + 1:i + 1]))
    return out


def true_range(h, l, c):
    tr = np.empty_like(c, dtype=float)
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return tr


# ============================================================
# V11 INDICATORS
# ============================================================

def compute_vortex(h, l, c, n=14):
    tr = true_range(h, l, c)
    vm_plus = np.zeros_like(c, dtype=float)
    vm_minus = np.zeros_like(c, dtype=float)
    for i in range(1, len(c)):
        vm_plus[i] = abs(h[i] - l[i - 1])
        vm_minus[i] = abs(l[i] - h[i - 1])
    trn = np_sma(tr, n) * n
    vp = np_sma(vm_plus, n) * n
    vm = np_sma(vm_minus, n) * n
    vi_plus = vp / np.where(trn == 0, np.nan, trn)
    vi_minus = vm / np.where(trn == 0, np.nan, trn)
    return vi_plus, vi_minus


def compute_coppock(c, r1=11, r2=14, wlen=10):
    out = np.full_like(c, np.nan, dtype=float)
    roc1 = np.full_like(c, np.nan, dtype=float)
    roc2 = np.full_like(c, np.nan, dtype=float)
    for i in range(max(r1, r2), len(c)):
        roc1[i] = (c[i] / c[i - r1] - 1.0) * 100.0
        roc2[i] = (c[i] / c[i - r2] - 1.0) * 100.0
    s = roc1 + roc2
    return np_wma(np.nan_to_num(s, nan=0.0), wlen)


def compute_kama(c, er_len=10, fast=2, slow=30):
    out = np.full_like(c, np.nan, dtype=float)
    if len(c) < er_len + 2:
        return out
    out[0] = c[0]
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    for i in range(1, len(c)):
        if i < er_len:
            out[i] = c[i]
            continue
        change = abs(c[i] - c[i - er_len])
        volatility = np.sum(np.abs(np.diff(c[i - er_len:i + 1])))
        er = 0.0 if volatility == 0 else (change / volatility)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out[i] = out[i - 1] + sc * (c[i] - out[i - 1])
    return out


def compute_fisher(c, n=10):
    out = np.full_like(c, np.nan, dtype=float)
    if len(c) < n:
        return out
    v = 0.0
    for i in range(n - 1, len(c)):
        w = c[i - n + 1:i + 1]
        mn = float(np.min(w))
        mx = float(np.max(w))
        x = 0.0 if mx == mn else (2.0 * (c[i] - mn) / (mx - mn) - 1.0)
        x = clamp(x, -0.999, 0.999)
        v = 0.33 * x + 0.67 * v
        v = clamp(v, -0.999, 0.999)
        out[i] = 0.5 * math.log((1 + v) / (1 - v))
    return out


def compute_aroon_osc(h, l, n=14):
    out = np.full_like(h, np.nan, dtype=float)
    if len(h) < n:
        return out
    for i in range(n - 1, len(h)):
        hh = h[i - n + 1:i + 1]
        ll = l[i - n + 1:i + 1]
        days_since_hh = n - 1 - int(np.argmax(hh))
        days_since_ll = n - 1 - int(np.argmin(ll))
        aroon_up = 100.0 * (n - 1 - days_since_hh) / (n - 1)
        aroon_dn = 100.0 * (n - 1 - days_since_ll) / (n - 1)
        out[i] = aroon_up - aroon_dn
    return out


def compute_obv(c, v):
    out = np.zeros_like(c, dtype=float)
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            out[i] = out[i - 1] + v[i]
        elif c[i] < c[i - 1]:
            out[i] = out[i - 1] - v[i]
        else:
            out[i] = out[i - 1]
    return out


# ============================================================
# V11 PROBABILITY MODEL
# ============================================================

def fair_prob_up_v11(vi_dir, copp, kama_slope, fisher, aroon, obv_slope, micro_ret):
    score = 0.0
    # 5m regime
    score += 0.75 * vi_dir
    score += clamp(copp / 30.0, -1.0, 1.0) * 0.35
    score += clamp(kama_slope / 0.0010, -1.0, 1.0) * 0.45
    # 1m triggers
    score += clamp(fisher / 2.5, -1.2, 1.2) * 0.35
    score += clamp(aroon / 100.0, -1.0, 1.0) * 0.40
    score += clamp(obv_slope / 1e7, -1.0, 1.0) * 0.25
    score += clamp(micro_ret / 0.0008, -1.2, 1.2) * 0.40
    p = sigmoid(score)
    return float(clamp(0.14 + 0.72 * p, 0.08, 0.92))


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
# V11.0: Vortex + Coppock + KAMA + Fisher + Aroon + OBV
# ============================================================

def simulate_v11(bars_5m, bars_1m, size_usd=5.0):
    P_BAND = 0.035
    BUY_PRICE = 0.51
    DECIDE_AFTER = 1
    OBV_SLOPE_LEN = 10

    n5 = len(bars_5m)
    h5 = np.array([b["high"] for b in bars_5m])
    l5 = np.array([b["low"] for b in bars_5m])
    c5 = np.array([b["close"] for b in bars_5m])
    o5 = np.array([b["open"] for b in bars_5m])

    # 5m indicators
    vi_plus, vi_minus = compute_vortex(h5, l5, c5, 14)
    copp5 = compute_coppock(c5, 11, 14, 10)
    kama5 = compute_kama(c5, 10, 2, 30)

    # 1m arrays
    n1 = len(bars_1m)
    h1 = np.array([b["high"] for b in bars_1m])
    l1 = np.array([b["low"] for b in bars_1m])
    c1 = np.array([b["close"] for b in bars_1m])
    o1 = np.array([b["open"] for b in bars_1m])
    v1 = np.array([b["volume"] for b in bars_1m])

    # 1m indicators
    fish1 = compute_fisher(c1, 10)
    aroon1 = compute_aroon_osc(h1, l1, 14)
    obv1 = compute_obv(c1, v1)

    trades = []
    start = 200

    for i in range(start, n5):
        bar = bars_5m[i]
        mins = bar["minutes"]
        if len(mins) < 5:
            continue

        j = i - 1
        if j <= 1:
            continue

        if not (np.isfinite(vi_plus[j]) and np.isfinite(vi_minus[j]) and
                np.isfinite(copp5[j]) and np.isfinite(kama5[j]) and np.isfinite(kama5[j - 1])):
            continue

        vi_dir = 1.0 if vi_plus[j] > vi_minus[j] else -1.0
        kama_slope = float((kama5[j] - kama5[j - 1]) / max(1e-9, c5[j - 1]))

        # 1m decision
        m_idx = i * 5 + DECIDE_AFTER
        if m_idx >= n1 or m_idx <= 0:
            continue

        if not (np.isfinite(fish1[m_idx]) and np.isfinite(aroon1[m_idx]) and
                np.isfinite(obv1[m_idx]) and np.isfinite(obv1[max(0, m_idx - OBV_SLOPE_LEN)])):
            continue

        obv_slope = float(obv1[m_idx] - obv1[max(0, m_idx - OBV_SLOPE_LEN)])

        # Micro return (minute 1 + half of minute 2)
        micro_ret = float((c1[m_idx] - o1[m_idx]) / max(1e-9, o1[m_idx]))
        m2_idx = m_idx + 1
        if m2_idx < n1 and np.isfinite(c1[m2_idx]) and np.isfinite(o1[m2_idx]):
            micro_ret += 0.5 * float((c1[m2_idx] - o1[m2_idx]) / max(1e-9, o1[m2_idx]))

        p_up = fair_prob_up_v11(
            vi_dir=vi_dir, copp=float(copp5[j]), kama_slope=kama_slope,
            fisher=float(fish1[m_idx]), aroon=float(aroon1[m_idx]),
            obv_slope=obv_slope, micro_ret=micro_ret,
        )

        if p_up >= 0.5 + P_BAND:
            side = "UP"
            fair_yes = p_up
        elif p_up <= 0.5 - P_BAND:
            side = "DOWN"
            fair_yes = 1.0 - p_up
        else:
            continue

        if fair_yes - BUY_PRICE < 0:
            continue

        shares = max(1, math.floor(size_usd / BUY_PRICE))
        cost = BUY_PRICE * shares

        up_win = bar["close"] > bar["open"]
        win = (up_win and side == "UP") or (not up_win and side == "DOWN")
        pnl = (shares * 1.0 - cost) if win else -cost

        trades.append({
            "direction": side, "actual": "UP" if up_win else "DOWN",
            "win": win, "p_up": p_up, "fair_yes": fair_yes,
            "vi_dir": vi_dir, "coppock": float(copp5[j]),
            "kama_slope": kama_slope, "fisher": float(fish1[m_idx]),
            "aroon": float(aroon1[m_idx]), "obv_slope": obv_slope,
            "momentum_bps": micro_ret * 10000,
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

    pups = [t.get("p_up", 0) for t in trades if t.get("p_up")]
    if pups:
        wp_up = [t["p_up"] for t in trades if t["win"] and t.get("p_up")]
        lp_up = [t["p_up"] for t in trades if not t["win"] and t.get("p_up")]
        print(f"  Avg p_up:      {sum(pups) / len(pups):.3f}")
        if wp_up:
            print(f"  Avg p_up (W):  {sum(wp_up) / len(wp_up):.3f}")
        if lp_up:
            print(f"  Avg p_up (L):  {sum(lp_up) / len(lp_up):.3f}")

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

    v30 = simulate_v30(bars_5m, SIZE)
    s30 = analyze(v30, "V3.0 - Simple Momentum (2min, 15bp, $5)")

    v11 = simulate_v11(bars_5m, bars_1m, SIZE)
    s11 = analyze(v11, "V11.0 - Vortex+Coppock+KAMA+Fisher+Aroon+OBV ($5)")

    # Feature breakdown
    if v11:
        print(f"\n{'=' * 60}")
        print(f"  V11.0 FEATURE BREAKDOWN")
        print(f"{'=' * 60}")

        # Vortex direction
        vi_bull = [t for t in v11 if t.get("vi_dir") == 1.0]
        vi_bear = [t for t in v11 if t.get("vi_dir") == -1.0]
        if vi_bull:
            wr = sum(1 for t in vi_bull if t["win"]) / len(vi_bull) * 100
            print(f"  Vortex Bull:     {len(vi_bull)}T, {wr:.1f}% WR")
        if vi_bear:
            wr = sum(1 for t in vi_bear if t["win"]) / len(vi_bear) * 100
            print(f"  Vortex Bear:     {len(vi_bear)}T, {wr:.1f}% WR")

        # Coppock sign
        cop_pos = [t for t in v11 if t.get("coppock", 0) > 0]
        cop_neg = [t for t in v11 if t.get("coppock", 0) <= 0]
        if cop_pos:
            wr = sum(1 for t in cop_pos if t["win"]) / len(cop_pos) * 100
            print(f"  Coppock > 0:     {len(cop_pos)}T, {wr:.1f}% WR")
        if cop_neg:
            wr = sum(1 for t in cop_neg if t["win"]) / len(cop_neg) * 100
            print(f"  Coppock <= 0:    {len(cop_neg)}T, {wr:.1f}% WR")

        # Fisher sign
        fish_pos = [t for t in v11 if t.get("fisher", 0) > 0]
        fish_neg = [t for t in v11 if t.get("fisher", 0) <= 0]
        if fish_pos:
            wr = sum(1 for t in fish_pos if t["win"]) / len(fish_pos) * 100
            print(f"  Fisher > 0:      {len(fish_pos)}T, {wr:.1f}% WR")
        if fish_neg:
            wr = sum(1 for t in fish_neg if t["win"]) / len(fish_neg) * 100
            print(f"  Fisher <= 0:     {len(fish_neg)}T, {wr:.1f}% WR")

        # Aroon
        ar_pos = [t for t in v11 if t.get("aroon", 0) > 0]
        ar_neg = [t for t in v11 if t.get("aroon", 0) <= 0]
        if ar_pos:
            wr = sum(1 for t in ar_pos if t["win"]) / len(ar_pos) * 100
            print(f"  Aroon > 0:       {len(ar_pos)}T, {wr:.1f}% WR")
        if ar_neg:
            wr = sum(1 for t in ar_neg if t["win"]) / len(ar_neg) * 100
            print(f"  Aroon <= 0:      {len(ar_neg)}T, {wr:.1f}% WR")

        # KAMA slope
        km_pos = [t for t in v11 if t.get("kama_slope", 0) > 0]
        km_neg = [t for t in v11 if t.get("kama_slope", 0) <= 0]
        if km_pos:
            wr = sum(1 for t in km_pos if t["win"]) / len(km_pos) * 100
            print(f"  KAMA slope UP:   {len(km_pos)}T, {wr:.1f}% WR")
        if km_neg:
            wr = sum(1 for t in km_neg if t["win"]) / len(km_neg) * 100
            print(f"  KAMA slope DN:   {len(km_neg)}T, {wr:.1f}% WR")

    # Head to head
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD (all $5 sizing)")
    print("=" * 70)

    def row(label, *vals):
        parts = [f"  {label:20s}"]
        for v in vals:
            parts.append(f"{v:>16s}")
        print("".join(parts))

    row("", "V3.0", "V11.0")
    row("-" * 20, "-" * 16, "-" * 16)

    for label, key in [
        ("Trades", "trades"), ("Trades/day", "trades_per_day"),
        ("Win Rate %", "win_rate"), ("Total PnL", "total_pnl"),
        ("Avg PnL/trade", "avg_pnl"), ("Avg Win", "avg_win"),
        ("Avg Loss", "avg_loss"), ("Max Drawdown", "max_drawdown"),
        ("ROI%", "roi_pct"), ("Win Streak", "max_win_streak"),
        ("Loss Streak", "max_loss_streak"),
    ]:
        vals = []
        for s in [s30, s11]:
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

    # Track V11 bar indices
    v11_bars = set()
    h5 = np.array([b["high"] for b in bars_5m])
    l5 = np.array([b["low"] for b in bars_5m])
    c5 = np.array([b["close"] for b in bars_5m])
    h1 = np.array([b["high"] for b in bars_1m])
    l1 = np.array([b["low"] for b in bars_1m])
    c1 = np.array([b["close"] for b in bars_1m])
    o1 = np.array([b["open"] for b in bars_1m])
    v1a = np.array([b["volume"] for b in bars_1m])

    vi_p, vi_m = compute_vortex(h5, l5, c5, 14)
    cop5 = compute_coppock(c5, 11, 14, 10)
    km5 = compute_kama(c5, 10, 2, 30)
    f1 = compute_fisher(c1, 10)
    ar1 = compute_aroon_osc(h1, l1, 14)
    ob1 = compute_obv(c1, v1a)

    for i in range(200, len(bars_5m)):
        j = i - 1
        if j <= 1:
            continue
        if not (np.isfinite(vi_p[j]) and np.isfinite(vi_m[j]) and
                np.isfinite(cop5[j]) and np.isfinite(km5[j]) and np.isfinite(km5[j - 1])):
            continue
        vi_dir = 1.0 if vi_p[j] > vi_m[j] else -1.0
        kama_slope = float((km5[j] - km5[j - 1]) / max(1e-9, c5[j - 1]))
        m_idx = i * 5 + 1
        if m_idx >= len(bars_1m) or m_idx <= 0:
            continue
        if not (np.isfinite(f1[m_idx]) and np.isfinite(ar1[m_idx]) and
                np.isfinite(ob1[m_idx]) and np.isfinite(ob1[max(0, m_idx - 10)])):
            continue
        obv_slope = float(ob1[m_idx] - ob1[max(0, m_idx - 10)])
        micro_ret = float((c1[m_idx] - o1[m_idx]) / max(1e-9, o1[m_idx]))
        m2_idx = m_idx + 1
        if m2_idx < len(bars_1m) and np.isfinite(c1[m2_idx]):
            micro_ret += 0.5 * float((c1[m2_idx] - o1[m2_idx]) / max(1e-9, o1[m2_idx]))
        p_up = fair_prob_up_v11(vi_dir, float(cop5[j]), kama_slope,
                                float(f1[m_idx]), float(ar1[m_idx]), obv_slope, micro_ret)
        if p_up >= 0.535 or p_up <= 0.465:
            fair_yes = p_up if p_up >= 0.535 else 1.0 - p_up
            if fair_yes > 0.51:
                v11_bars.add(i)

    both = v30_bars & v11_bars
    v11_only = v11_bars - v30_bars
    v30_only = v30_bars - v11_bars

    print(f"  V3.0 trades on:   {len(v30_bars)} bars")
    print(f"  V11.0 trades on:  {len(v11_bars)} bars")
    print(f"  OVERLAP:          {len(both)} bars")
    print(f"  V11.0 UNIQUE:     {len(v11_only)} bars (not in V3.0)")
    print(f"  V3.0 UNIQUE:      {len(v30_only)} bars (not in V11.0)")

    if v11_only:
        w = sum(1 for idx in v11_only if bars_5m[idx]["close"] > bars_5m[idx]["open"])
        print(f"  V11 unique UP%:   {w / len(v11_only) * 100:.1f}%")

    # Verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print(f"  V3.0:   {s30['trades']}T, {s30['win_rate']:.1f}% WR, ${s30['total_pnl']:+,.2f}, {s30['roi_pct']:+.1f}% ROI")
    print(f"  V11.0:  {s11['trades']}T, {s11['win_rate']:.1f}% WR, ${s11['total_pnl']:+,.2f}, {s11['roi_pct']:+.1f}% ROI")

    if len(v11_only) > 20:
        print(f"\n  V11 has {len(v11_only)} unique trades - potential to stack with V3.0")
    elif len(v11_only) > 0:
        print(f"\n  V11 has {len(v11_only)} unique trades - marginal value")
    else:
        print(f"\n  V11 is a subset of V3.0 - no unique value")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": f"{date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}",
        "bars_1m": len(bars_1m), "bars_5m": len(bars_5m),
        "natural_up_pct": round(natural_up, 2),
        "v30": s30, "v11": s11,
        "overlap": {"v30": len(v30_bars), "v11": len(v11_bars),
                    "both": len(both), "v11_unique": len(v11_only),
                    "v30_unique": len(v30_only)},
    }
    out = Path(__file__).parent / "backtest_v11_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
