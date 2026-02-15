"""
Head-to-head backtest: V3.0 vs V10.0
V10.0: PSAR + Hull MA + Ulcer Index on 5m regime
       + Williams %R + Klinger Volume Osc + ATR impulse on 1m micro
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


def np_atr(h, l, c, n=14):
    return np_sma(true_range(h, l, c), n)


# ============================================================
# V10 INDICATORS
# ============================================================

def compute_hma(c, n=55):
    n2 = max(1, n // 2)
    ns = max(1, int(math.sqrt(n)))
    w1 = np_wma(c, n2)
    w2 = np_wma(c, n)
    diff = 2.0 * w1 - w2
    return np_wma(diff, ns)


def compute_psar(h, l, step=0.02, max_step=0.2):
    sar = np.full_like(h, np.nan, dtype=float)
    bull = True
    af = step
    ep = h[0]
    sar[0] = l[0]

    for i in range(1, len(h)):
        prev_sar = sar[i - 1]
        if not np.isfinite(prev_sar):
            continue
        sar[i] = prev_sar + af * (ep - prev_sar)
        if bull:
            sar[i] = min(sar[i], l[i - 1], l[i])
        else:
            sar[i] = max(sar[i], h[i - 1], h[i])
        if bull:
            if l[i] < sar[i]:
                bull = False
                sar[i] = ep
                ep = l[i]
                af = step
            else:
                if h[i] > ep:
                    ep = h[i]
                    af = min(max_step, af + step)
        else:
            if h[i] > sar[i]:
                bull = True
                sar[i] = ep
                ep = h[i]
                af = step
            else:
                if l[i] < ep:
                    ep = l[i]
                    af = min(max_step, af + step)
    return sar


def compute_ulcer_index(c, n=28):
    out = np.full_like(c, np.nan, dtype=float)
    if len(c) < n:
        return out
    for i in range(n - 1, len(c)):
        w = c[i - n + 1:i + 1]
        mx = np.max(w)
        if mx <= 0:
            continue
        dd = (w - mx) / mx * 100.0
        out[i] = float(math.sqrt(np.mean(dd * dd)))
    return out


def compute_williams_r(h, l, c, n=14):
    out = np.full_like(c, np.nan, dtype=float)
    hh = rolling_max(h, n)
    ll = rolling_min(l, n)
    for i in range(n - 1, len(c)):
        if not np.isfinite(hh[i]) or not np.isfinite(ll[i]) or (hh[i] - ll[i]) == 0:
            continue
        out[i] = -100.0 * (hh[i] - c[i]) / (hh[i] - ll[i])
    return out


def compute_klinger(h, l, c, v, fast=34, slow=55, signal=13):
    hlc = h + l + c
    vf = np.zeros_like(c, dtype=float)
    for i in range(1, len(c)):
        trend = 1.0 if hlc[i] > hlc[i - 1] else -1.0
        rng = max(1e-12, h[i] - l[i])
        vf[i] = trend * v[i] * abs(hlc[i] - hlc[i - 1]) / rng
    kvo = np_ema(vf, fast) - np_ema(vf, slow)
    sig = np_ema(kvo, signal)
    return kvo, sig


# ============================================================
# V10 PROBABILITY MODEL
# ============================================================

def fair_prob_up_v10(psar_bull, hma_slope, ulcer, wr_cross, kvo_hist, atr_impulse, micro_ret):
    score = 0.0
    # 5m regime
    score += 0.75 * psar_bull
    score += clamp(hma_slope / 0.0010, -1.0, 1.0) * 0.45
    score += clamp((1.5 - ulcer) / 1.5, -0.8, 0.8) * 0.35
    # 1m triggers
    score += 0.50 * wr_cross
    score += clamp(kvo_hist / 50.0, -1.0, 1.0) * 0.35
    score += clamp((atr_impulse - 1.0) / 0.25, -0.5, 1.0) * 0.25
    score += clamp(micro_ret / 0.0008, -1.2, 1.2) * 0.35
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
# V10.0: PSAR + HMA + Ulcer + Williams%R + Klinger + ATR impulse
# ============================================================

def simulate_v10(bars_5m, bars_1m, size_usd=5.0):
    P_BAND = 0.035
    BUY_PRICE = 0.51
    DECIDE_AFTER = 1
    UI_MAX = 1.9
    WR_CROSS_LEVEL = -50.0
    ATR_IMPULSE_RATIO = 1.05

    n5 = len(bars_5m)
    h5 = np.array([b["high"] for b in bars_5m])
    l5 = np.array([b["low"] for b in bars_5m])
    c5 = np.array([b["close"] for b in bars_5m])
    o5 = np.array([b["open"] for b in bars_5m])

    # 5m indicators
    sar5 = compute_psar(h5, l5, 0.02, 0.2)
    hma5 = compute_hma(c5, 55)
    ui5 = compute_ulcer_index(c5, 28)

    # 1m arrays
    n1 = len(bars_1m)
    h1 = np.array([b["high"] for b in bars_1m])
    l1 = np.array([b["low"] for b in bars_1m])
    c1 = np.array([b["close"] for b in bars_1m])
    o1 = np.array([b["open"] for b in bars_1m])
    v1 = np.array([b["volume"] for b in bars_1m])

    # 1m indicators
    wr1 = compute_williams_r(h1, l1, c1, 14)
    kvo1, ksig1 = compute_klinger(h1, l1, c1, v1, 34, 55, 13)
    kvo_hist1 = kvo1 - ksig1
    atr1 = np_atr(h1, l1, c1, 14)
    atr1_ma = np_sma(atr1, 20)

    trades = []
    start = max(200, 60)  # HMA(55) + Ulcer(28) warmup

    for i in range(start, n5):
        bar = bars_5m[i]
        mins = bar["minutes"]
        if len(mins) < 5:
            continue

        j = i - 1
        if j <= 1:
            continue

        if not (np.isfinite(sar5[j]) and np.isfinite(hma5[j]) and
                np.isfinite(hma5[j - 1]) and np.isfinite(ui5[j])):
            continue

        # 5m regime
        psar_bull = 1.0 if c5[j] > sar5[j] else -1.0
        hma_slope = float((hma5[j] - hma5[j - 1]) / max(1e-9, c5[j - 1]))
        ulcer = float(ui5[j])

        if ulcer > UI_MAX:
            continue

        # 1m decision
        m_idx = i * 5 + DECIDE_AFTER
        if m_idx >= n1 or m_idx <= 0:
            continue

        prev_m = m_idx - 1
        if prev_m < 0:
            continue

        if not (np.isfinite(wr1[m_idx]) and np.isfinite(wr1[prev_m]) and
                np.isfinite(kvo_hist1[m_idx]) and np.isfinite(atr1[m_idx]) and
                np.isfinite(atr1_ma[m_idx])):
            continue

        # Williams %R cross
        wr_cross = 0.0
        if wr1[prev_m] <= WR_CROSS_LEVEL and wr1[m_idx] > WR_CROSS_LEVEL:
            wr_cross = 1.0
        elif wr1[prev_m] >= WR_CROSS_LEVEL and wr1[m_idx] < WR_CROSS_LEVEL:
            wr_cross = -1.0

        kvh = float(kvo_hist1[m_idx])
        atr_imp = float(atr1[m_idx] / max(1e-9, atr1_ma[m_idx]))

        # Micro return (minute 1 + half of minute 2)
        micro_ret = float((c1[m_idx] - o1[m_idx]) / max(1e-9, o1[m_idx]))
        m2_idx = m_idx + 1
        if m2_idx < n1 and np.isfinite(c1[m2_idx]) and np.isfinite(o1[m2_idx]):
            micro_ret += 0.5 * float((c1[m2_idx] - o1[m2_idx]) / max(1e-9, o1[m2_idx]))

        # Frequency gate: need WR cross OR (ATR impulse + Klinger alignment)
        if wr_cross == 0.0 and not (atr_imp >= ATR_IMPULSE_RATIO and abs(kvh) > 5.0):
            continue

        p_up = fair_prob_up_v10(psar_bull, hma_slope, ulcer, wr_cross, kvh, atr_imp, micro_ret)

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
            "psar_bull": psar_bull, "hma_slope": hma_slope, "ulcer": ulcer,
            "wr_cross": wr_cross, "kvo_hist": kvh, "atr_impulse": atr_imp,
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

    v10 = simulate_v10(bars_5m, bars_1m, SIZE)
    s10 = analyze(v10, "V10.0 - PSAR+HMA+Ulcer+WR+Klinger+ATR ($5)")

    # Feature breakdown
    if v10:
        print(f"\n{'=' * 60}")
        print(f"  V10.0 FEATURE BREAKDOWN")
        print(f"{'=' * 60}")

        # PSAR direction
        psar_up = [t for t in v10 if t.get("psar_bull") == 1.0]
        psar_dn = [t for t in v10 if t.get("psar_bull") == -1.0]
        if psar_up:
            wr = sum(1 for t in psar_up if t["win"]) / len(psar_up) * 100
            print(f"  PSAR Bullish:    {len(psar_up)}T, {wr:.1f}% WR")
        if psar_dn:
            wr = sum(1 for t in psar_dn if t["win"]) / len(psar_dn) * 100
            print(f"  PSAR Bearish:    {len(psar_dn)}T, {wr:.1f}% WR")

        # Williams %R cross
        wr_yes = [t for t in v10 if t.get("wr_cross") != 0.0]
        wr_no = [t for t in v10 if t.get("wr_cross") == 0.0]
        if wr_yes:
            wr = sum(1 for t in wr_yes if t["win"]) / len(wr_yes) * 100
            print(f"  WR Cross:        {len(wr_yes)}T, {wr:.1f}% WR")
        if wr_no:
            wr = sum(1 for t in wr_no if t["win"]) / len(wr_no) * 100
            print(f"  No WR Cross:     {len(wr_no)}T, {wr:.1f}% WR (ATR+KVO gate)")

        # Ulcer Index ranges
        ui_lo = [t for t in v10 if t.get("ulcer", 2) < 0.8]
        ui_md = [t for t in v10 if 0.8 <= t.get("ulcer", 2) < 1.4]
        ui_hi = [t for t in v10 if 1.4 <= t.get("ulcer", 2) <= 1.9]
        if ui_lo:
            wr = sum(1 for t in ui_lo if t["win"]) / len(ui_lo) * 100
            print(f"  Ulcer < 0.8:     {len(ui_lo)}T, {wr:.1f}% WR (smooth)")
        if ui_md:
            wr = sum(1 for t in ui_md if t["win"]) / len(ui_md) * 100
            print(f"  Ulcer 0.8-1.4:   {len(ui_md)}T, {wr:.1f}% WR (moderate)")
        if ui_hi:
            wr = sum(1 for t in ui_hi if t["win"]) / len(ui_hi) * 100
            print(f"  Ulcer 1.4-1.9:   {len(ui_hi)}T, {wr:.1f}% WR (volatile)")

        # HMA slope
        hma_pos = [t for t in v10 if t.get("hma_slope", 0) > 0]
        hma_neg = [t for t in v10 if t.get("hma_slope", 0) <= 0]
        if hma_pos:
            wr = sum(1 for t in hma_pos if t["win"]) / len(hma_pos) * 100
            print(f"  HMA slope UP:    {len(hma_pos)}T, {wr:.1f}% WR")
        if hma_neg:
            wr = sum(1 for t in hma_neg if t["win"]) / len(hma_neg) * 100
            print(f"  HMA slope DOWN:  {len(hma_neg)}T, {wr:.1f}% WR")

    # Head to head
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD (all $5 sizing)")
    print("=" * 70)

    def row(label, *vals):
        parts = [f"  {label:20s}"]
        for v in vals:
            parts.append(f"{v:>16s}")
        print("".join(parts))

    row("", "V3.0", "V10.0")
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
        for s in [s30, s10]:
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

    # Track V10 bar indices — re-derive
    v10_bars = set()
    h5 = np.array([b["high"] for b in bars_5m])
    l5 = np.array([b["low"] for b in bars_5m])
    c5 = np.array([b["close"] for b in bars_5m])
    h1 = np.array([b["high"] for b in bars_1m])
    l1 = np.array([b["low"] for b in bars_1m])
    c1 = np.array([b["close"] for b in bars_1m])
    o1 = np.array([b["open"] for b in bars_1m])
    v1a = np.array([b["volume"] for b in bars_1m])

    sar5 = compute_psar(h5, l5, 0.02, 0.2)
    hma5 = compute_hma(c5, 55)
    ui5 = compute_ulcer_index(c5, 28)
    wr1 = compute_williams_r(h1, l1, c1, 14)
    kvo1, ksig1 = compute_klinger(h1, l1, c1, v1a, 34, 55, 13)
    kvo_hist1 = kvo1 - ksig1
    atr1 = np_atr(h1, l1, c1, 14)
    atr1_ma = np_sma(atr1, 20)

    for i in range(200, len(bars_5m)):
        j = i - 1
        if j <= 1:
            continue
        if not (np.isfinite(sar5[j]) and np.isfinite(hma5[j]) and
                np.isfinite(hma5[j - 1]) and np.isfinite(ui5[j])):
            continue
        if ui5[j] > 1.9:
            continue
        psar_bull = 1.0 if c5[j] > sar5[j] else -1.0
        hma_slope = float((hma5[j] - hma5[j - 1]) / max(1e-9, c5[j - 1]))
        ulcer = float(ui5[j])
        m_idx = i * 5 + 1
        if m_idx >= len(bars_1m) or m_idx <= 0:
            continue
        prev_m = m_idx - 1
        if not (np.isfinite(wr1[m_idx]) and np.isfinite(wr1[prev_m]) and
                np.isfinite(kvo_hist1[m_idx]) and np.isfinite(atr1[m_idx]) and
                np.isfinite(atr1_ma[m_idx])):
            continue
        wr_cross = 0.0
        if wr1[prev_m] <= -50.0 and wr1[m_idx] > -50.0:
            wr_cross = 1.0
        elif wr1[prev_m] >= -50.0 and wr1[m_idx] < -50.0:
            wr_cross = -1.0
        kvh = float(kvo_hist1[m_idx])
        atr_imp = float(atr1[m_idx] / max(1e-9, atr1_ma[m_idx]))
        if wr_cross == 0.0 and not (atr_imp >= 1.05 and abs(kvh) > 5.0):
            continue
        micro_ret = float((c1[m_idx] - o1[m_idx]) / max(1e-9, o1[m_idx]))
        m2_idx = m_idx + 1
        if m2_idx < len(bars_1m) and np.isfinite(c1[m2_idx]):
            micro_ret += 0.5 * float((c1[m2_idx] - o1[m2_idx]) / max(1e-9, o1[m2_idx]))
        p_up = fair_prob_up_v10(psar_bull, hma_slope, ulcer, wr_cross, kvh, atr_imp, micro_ret)
        if p_up >= 0.535 or p_up <= 0.465:
            fair_yes = p_up if p_up >= 0.535 else 1.0 - p_up
            if fair_yes > 0.51:
                v10_bars.add(i)

    both = v30_bars & v10_bars
    v10_only = v10_bars - v30_bars
    v30_only = v30_bars - v10_bars

    print(f"  V3.0 trades on:   {len(v30_bars)} bars")
    print(f"  V10.0 trades on:  {len(v10_bars)} bars")
    print(f"  OVERLAP:          {len(both)} bars")
    print(f"  V10.0 UNIQUE:     {len(v10_only)} bars (not in V3.0)")
    print(f"  V3.0 UNIQUE:      {len(v30_only)} bars (not in V10.0)")

    if v10_only:
        w = sum(1 for idx in v10_only if bars_5m[idx]["close"] > bars_5m[idx]["open"])
        print(f"  V10 unique UP%:   {w / len(v10_only) * 100:.1f}%")

    # Verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print(f"  V3.0:   {s30['trades']}T, {s30['win_rate']:.1f}% WR, ${s30['total_pnl']:+,.2f}, {s30['roi_pct']:+.1f}% ROI")
    print(f"  V10.0:  {s10['trades']}T, {s10['win_rate']:.1f}% WR, ${s10['total_pnl']:+,.2f}, {s10['roi_pct']:+.1f}% ROI")

    if len(v10_only) > 20:
        print(f"\n  V10 has {len(v10_only)} unique trades - potential to stack with V3.0")
    elif len(v10_only) > 0:
        print(f"\n  V10 has {len(v10_only)} unique trades - marginal value")
    else:
        print(f"\n  V10 is a subset of V3.0 - no unique value")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": f"{date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}",
        "bars_1m": len(bars_1m), "bars_5m": len(bars_5m),
        "natural_up_pct": round(natural_up, 2),
        "v30": s30, "v10": s10,
        "overlap": {"v30": len(v30_bars), "v10": len(v10_bars),
                    "both": len(both), "v10_unique": len(v10_only),
                    "v30_unique": len(v30_only)},
    }
    out = Path(__file__).parent / "backtest_v10_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
