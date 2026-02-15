"""
Head-to-head backtest: V3.0 vs V8.0
V8.0: Supertrend + ADX + Squeeze/Expansion (BB vs KC) on 5m regime
      + VWAP reclaim + Donchian breakout + Volume impulse on 1m micro
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
# DATA FETCHING (same as all previous backtests)
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
# V8 INDICATORS (ported from user's V8 code)
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


def true_range(h, l, c):
    tr = np.empty_like(c, dtype=float)
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return tr


def np_atr(h, l, c, n=14):
    return np_sma(true_range(h, l, c), n)


def wilder_smooth(x, n):
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) < n + 1:
        return out
    out[n] = np.nanmean(x[1:n + 1])
    for i in range(n + 1, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def compute_adx(h, l, c, n=14):
    up = np.zeros_like(c, dtype=float)
    dn = np.zeros_like(c, dtype=float)
    for i in range(1, len(c)):
        up_move = h[i] - h[i - 1]
        dn_move = l[i - 1] - l[i]
        up[i] = up_move if (up_move > dn_move and up_move > 0) else 0.0
        dn[i] = dn_move if (dn_move > up_move and dn_move > 0) else 0.0
    tr = true_range(h, l, c)
    atr_w = wilder_smooth(tr, n)
    plus_dm = wilder_smooth(up, n)
    minus_dm = wilder_smooth(dn, n)
    plus_di = 100.0 * (plus_dm / np.where(atr_w == 0, np.nan, atr_w))
    minus_di = 100.0 * (minus_dm / np.where(atr_w == 0, np.nan, atr_w))
    dx = 100.0 * (np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, (plus_di + minus_di)))
    return wilder_smooth(dx, n)


def bollinger(close, n=20, k=2.0):
    mid = np_sma(close, n)
    sd = np.full_like(close, np.nan, dtype=float)
    if len(close) >= n:
        for i in range(n - 1, len(close)):
            sd[i] = float(np.std(close[i - n + 1:i + 1], ddof=0))
    up = mid + k * sd
    dn = mid - k * sd
    bw = (up - dn) / np.where(mid == 0, np.nan, mid)
    return mid, up, dn, bw


def keltner(h, l, c, ema_len=20, atr_len=10, mult=1.5):
    mid = np_ema(c, ema_len)
    a = np_atr(h, l, c, atr_len)
    up = mid + mult * a
    dn = mid - mult * a
    width = (up - dn) / np.where(mid == 0, np.nan, mid)
    return mid, up, dn, width


def supertrend(h, l, c, atr_len=10, mult=2.5):
    a = np_atr(h, l, c, atr_len)
    hl2 = (h + l) / 2.0
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    st = np.full_like(c, np.nan, dtype=float)
    dir_ = np.full_like(c, 0, dtype=int)
    st[0] = upper[0]
    dir_[0] = 1
    for i in range(1, len(c)):
        if not (np.isfinite(upper[i]) and np.isfinite(lower[i])):
            continue
        if np.isfinite(upper[i - 1]) and upper[i] > upper[i - 1] and c[i - 1] <= upper[i - 1]:
            upper[i] = upper[i - 1]
        if np.isfinite(lower[i - 1]) and lower[i] < lower[i - 1] and c[i - 1] >= lower[i - 1]:
            lower[i] = lower[i - 1]
        if dir_[i - 1] == 1:
            if c[i] < lower[i]:
                dir_[i] = -1
                st[i] = upper[i]
            else:
                dir_[i] = 1
                st[i] = lower[i]
        else:
            if c[i] > upper[i]:
                dir_[i] = 1
                st[i] = lower[i]
            else:
                dir_[i] = -1
                st[i] = upper[i]
    return dir_, st


def donchian(h, l, n=20):
    up = np.full_like(h, np.nan, dtype=float)
    dn = np.full_like(l, np.nan, dtype=float)
    if len(h) >= n:
        for i in range(n - 1, len(h)):
            up[i] = float(np.max(h[i - n + 1:i + 1]))
            dn[i] = float(np.min(l[i - n + 1:i + 1]))
    return up, dn


def vwap_rolling(h, l, c, v, n=30):
    tp = (h + l + c) / 3.0
    out = np.full_like(c, np.nan, dtype=float)
    if len(c) < n:
        return out
    for i in range(n - 1, len(c)):
        w_tp = tp[i - n + 1:i + 1]
        w_v = v[i - n + 1:i + 1]
        denom = float(np.sum(w_v))
        if denom <= 0:
            continue
        out[i] = float(np.sum(w_tp * w_v) / denom)
    return out


# ============================================================
# V8 PROBABILITY MODEL
# ============================================================

def fair_prob_up_v8(st_dir, adx_val, exp_ratio,
                    micro_vwap_reclaim, micro_breakout,
                    relvol, micro_ret):
    score = 0.0
    # 5m regime
    score += 0.75 * float(st_dir)
    score += clamp((adx_val - 12.0) / 10.0, -0.4, 1.0) * 0.45
    score += clamp((exp_ratio - 1.0) / 0.10, -0.6, 1.2) * 0.35
    # 1m confirmation
    score += 0.45 * micro_vwap_reclaim
    score += 0.35 * micro_breakout
    score += clamp((relvol - 1.0) / 0.30, -0.4, 1.2) * 0.25
    score += clamp(micro_ret / 0.0008, -1.2, 1.2) * 0.40
    p = sigmoid(score)
    p = clamp(0.14 + 0.72 * p, 0.08, 0.92)
    return float(p)


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
# V8.0: Supertrend + ADX + Squeeze + VWAP + Donchian + Volume
# ============================================================

def simulate_v8(bars_5m, bars_1m, size_usd=5.0):
    """
    V8.0 strategy — adapted from user's code to use Binance bars.

    5m regime (previous closed bar): Supertrend direction, ADX >= 14, BB/KC expansion ratio
    1m micro (first 1-2 minutes): VWAP reclaim/loss, Donchian breakout, rel volume, return
    Probability model: sigmoid scoring -> decision band >= 0.535 or <= 0.465
    Edge gate: fair_yes > 0.51
    """
    # Config (same as user's CFG defaults)
    ADX_MIN = 14.0
    EXP_RATIO_MIN = 1.05
    RELVOL_MIN = 1.05
    P_BAND = 0.035
    BUY_PRICE = 0.51
    DECIDE_AFTER = 1  # after first 1m candle

    # Build numpy arrays for 5m bars
    n5 = len(bars_5m)
    h5 = np.array([b["high"] for b in bars_5m])
    l5 = np.array([b["low"] for b in bars_5m])
    c5 = np.array([b["close"] for b in bars_5m])

    # 5m indicators
    st_dir, _ = supertrend(h5, l5, c5, atr_len=10, mult=2.5)
    adx5 = compute_adx(h5, l5, c5, n=14)
    _, _, _, bbw5 = bollinger(c5, n=20, k=2.0)
    _, _, _, kcw5 = keltner(h5, l5, c5, ema_len=20, atr_len=10, mult=1.5)
    exp_ratio = bbw5 / np.where(kcw5 == 0, np.nan, kcw5)

    # Build numpy arrays for 1m bars
    n1 = len(bars_1m)
    h1 = np.array([b["high"] for b in bars_1m])
    l1 = np.array([b["low"] for b in bars_1m])
    c1 = np.array([b["close"] for b in bars_1m])
    o1 = np.array([b["open"] for b in bars_1m])
    v1 = np.array([b["volume"] for b in bars_1m])

    # 1m indicators
    vwap1 = vwap_rolling(h1, l1, c1, v1, n=30)
    dcu1, dcd1 = donchian(h1, l1, n=20)
    vol_ma1 = np_sma(v1, 20)

    trades = []
    start = max(200, 25)  # warmup for BB(20), ADX(14), etc.

    for i in range(start, n5):
        bar = bars_5m[i]
        mins = bar["minutes"]
        if len(mins) < 5:
            continue

        # Previous closed 5m bar for regime (no lookahead)
        j = i - 1
        if j <= 0:
            continue

        if not np.isfinite(adx5[j]) or not np.isfinite(exp_ratio[j]):
            continue

        # ADX filter
        if adx5[j] < ADX_MIN:
            continue

        stj = int(st_dir[j])
        expj = float(exp_ratio[j])

        # Expansion filter (with relaxed skip)
        if expj < EXP_RATIO_MIN:
            if expj < 0.95:
                continue

        # 1m micro at decision minute
        # Global 1m index: bar i corresponds to 1m bars [i*5, i*5+4]
        # After DECIDE_AFTER minutes = index i*5 + DECIDE_AFTER
        m_idx = i * 5 + DECIDE_AFTER
        if m_idx >= n1:
            continue

        if not (np.isfinite(vwap1[m_idx]) and np.isfinite(dcu1[m_idx]) and
                np.isfinite(dcd1[m_idx]) and np.isfinite(vol_ma1[m_idx])):
            continue

        # VWAP reclaim/loss
        prev_m = m_idx - 1
        micro_vwap = 0.0
        if prev_m >= i * 5:  # stay within this 5m bar's minute range
            if np.isfinite(vwap1[prev_m]):
                if (c1[m_idx] > vwap1[m_idx]) and (c1[prev_m] <= vwap1[prev_m]):
                    micro_vwap = 1.0
                elif (c1[m_idx] < vwap1[m_idx]) and (c1[prev_m] >= vwap1[prev_m]):
                    micro_vwap = -1.0

        # Donchian micro-breakout
        micro_bo = 0.0
        if c1[m_idx] >= dcu1[m_idx]:
            micro_bo = 1.0
        elif c1[m_idx] <= dcd1[m_idx]:
            micro_bo = -1.0

        # Relative volume
        relv = float(v1[m_idx] / max(1e-9, vol_ma1[m_idx]))

        # Early minute return
        micro_ret = float((c1[m_idx] - o1[m_idx]) / max(1e-9, o1[m_idx]))

        # Frequency gate: need a trigger OR volume impulse
        if (abs(micro_vwap) + abs(micro_bo)) == 0 and relv < RELVOL_MIN:
            continue

        p_up = fair_prob_up_v8(
            st_dir=stj, adx_val=float(adx5[j]), exp_ratio=expj,
            micro_vwap_reclaim=micro_vwap, micro_breakout=micro_bo,
            relvol=relv, micro_ret=micro_ret,
        )

        # Decision band
        if p_up >= 0.5 + P_BAND:
            side = "UP"
            fair_yes = p_up
        elif p_up <= 0.5 - P_BAND:
            side = "DOWN"
            fair_yes = 1.0 - p_up
        else:
            continue

        # Edge gate
        if fair_yes - BUY_PRICE < 0:
            continue

        # Execute at $5 fixed sizing
        shares = max(1, math.floor(size_usd / BUY_PRICE))
        cost = BUY_PRICE * shares

        # Settlement: close > open = UP wins
        up_win = bar["close"] > bar["open"]
        win = (up_win and side == "UP") or (not up_win and side == "DOWN")
        pnl = (shares * 1.0 - cost) if win else -cost

        trades.append({
            "direction": side, "actual": "UP" if up_win else "DOWN",
            "win": win, "p_up": p_up, "fair_yes": fair_yes,
            "adx": float(adx5[j]), "exp_ratio": expj, "st_dir": stj,
            "micro_vwap": micro_vwap, "micro_bo": micro_bo, "relvol": relv,
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

    # V3.0 baseline
    v30 = simulate_v30(bars_5m, SIZE)
    s30 = analyze(v30, "V3.0 - Simple Momentum (2min, 15bp, $5)")

    # V8.0
    v8 = simulate_v8(bars_5m, bars_1m, SIZE)
    s8 = analyze(v8, "V8.0 - Supertrend+ADX+Squeeze+VWAP+Donchian+Vol ($5)")

    # V8 feature breakdown (if enough trades)
    if v8:
        print(f"\n{'=' * 60}")
        print(f"  V8.0 FEATURE BREAKDOWN")
        print(f"{'=' * 60}")

        # By supertrend direction
        st_up = [t for t in v8 if t.get("st_dir") == 1]
        st_dn = [t for t in v8 if t.get("st_dir") == -1]
        if st_up:
            wr_up = sum(1 for t in st_up if t["win"]) / len(st_up) * 100
            print(f"  Supertrend UP:    {len(st_up)}T, {wr_up:.1f}% WR")
        if st_dn:
            wr_dn = sum(1 for t in st_dn if t["win"]) / len(st_dn) * 100
            print(f"  Supertrend DOWN:  {len(st_dn)}T, {wr_dn:.1f}% WR")

        # By VWAP signal
        vwap_r = [t for t in v8 if t.get("micro_vwap") != 0]
        vwap_n = [t for t in v8 if t.get("micro_vwap") == 0]
        if vwap_r:
            wr_vr = sum(1 for t in vwap_r if t["win"]) / len(vwap_r) * 100
            print(f"  VWAP signal:      {len(vwap_r)}T, {wr_vr:.1f}% WR")
        if vwap_n:
            wr_vn = sum(1 for t in vwap_n if t["win"]) / len(vwap_n) * 100
            print(f"  No VWAP signal:   {len(vwap_n)}T, {wr_vn:.1f}% WR")

        # By Donchian breakout
        bo_r = [t for t in v8 if t.get("micro_bo") != 0]
        bo_n = [t for t in v8 if t.get("micro_bo") == 0]
        if bo_r:
            wr_br = sum(1 for t in bo_r if t["win"]) / len(bo_r) * 100
            print(f"  Donchian BO:      {len(bo_r)}T, {wr_br:.1f}% WR")
        if bo_n:
            wr_bn = sum(1 for t in bo_n if t["win"]) / len(bo_n) * 100
            print(f"  No Donchian BO:   {len(bo_n)}T, {wr_bn:.1f}% WR")

        # By ADX strength
        adx_hi = [t for t in v8 if t.get("adx", 0) >= 25]
        adx_lo = [t for t in v8 if 14 <= t.get("adx", 0) < 25]
        if adx_hi:
            wr_ah = sum(1 for t in adx_hi if t["win"]) / len(adx_hi) * 100
            print(f"  ADX >= 25:        {len(adx_hi)}T, {wr_ah:.1f}% WR")
        if adx_lo:
            wr_al = sum(1 for t in adx_lo if t["win"]) / len(adx_lo) * 100
            print(f"  ADX 14-25:        {len(adx_lo)}T, {wr_al:.1f}% WR")

    # Head to head
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD (all $5 sizing)")
    print("=" * 70)

    def row(label, *vals):
        parts = [f"  {label:20s}"]
        for v in vals:
            parts.append(f"{v:>16s}")
        print("".join(parts))

    row("", "V3.0", "V8.0")
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
        for s in [s30, s8]:
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

    # Track V8 bar indices
    v8_bars = set()
    # Re-derive which bars V8 trades on (same logic, just collecting indices)
    n5 = len(bars_5m)
    h5 = np.array([b["high"] for b in bars_5m])
    l5 = np.array([b["low"] for b in bars_5m])
    c5 = np.array([b["close"] for b in bars_5m])
    h1 = np.array([b["high"] for b in bars_1m])
    l1 = np.array([b["low"] for b in bars_1m])
    c1 = np.array([b["close"] for b in bars_1m])
    o1 = np.array([b["open"] for b in bars_1m])
    v1_arr = np.array([b["volume"] for b in bars_1m])

    st_dir_arr, _ = supertrend(h5, l5, c5, 10, 2.5)
    adx5_arr = compute_adx(h5, l5, c5, 14)
    _, _, _, bbw5_arr = bollinger(c5, 20, 2.0)
    _, _, _, kcw5_arr = keltner(h5, l5, c5, 20, 10, 1.5)
    exp_arr = bbw5_arr / np.where(kcw5_arr == 0, np.nan, kcw5_arr)
    vwap1_arr = vwap_rolling(h1, l1, c1, v1_arr, 30)
    dcu1_arr, dcd1_arr = donchian(h1, l1, 20)
    vol_ma1_arr = np_sma(v1_arr, 20)

    for i in range(200, n5):
        j = i - 1
        if j <= 0:
            continue
        if not np.isfinite(adx5_arr[j]) or not np.isfinite(exp_arr[j]):
            continue
        if adx5_arr[j] < 14.0:
            continue
        expj = float(exp_arr[j])
        if expj < 0.95:
            continue
        m_idx = i * 5 + 1
        if m_idx >= len(bars_1m):
            continue
        if not (np.isfinite(vwap1_arr[m_idx]) and np.isfinite(dcu1_arr[m_idx]) and
                np.isfinite(dcd1_arr[m_idx]) and np.isfinite(vol_ma1_arr[m_idx])):
            continue
        prev_m = m_idx - 1
        micro_vwap = 0.0
        if prev_m >= i * 5 and np.isfinite(vwap1_arr[prev_m]):
            if (c1[m_idx] > vwap1_arr[m_idx]) and (c1[prev_m] <= vwap1_arr[prev_m]):
                micro_vwap = 1.0
            elif (c1[m_idx] < vwap1_arr[m_idx]) and (c1[prev_m] >= vwap1_arr[prev_m]):
                micro_vwap = -1.0
        micro_bo = 0.0
        if c1[m_idx] >= dcu1_arr[m_idx]:
            micro_bo = 1.0
        elif c1[m_idx] <= dcd1_arr[m_idx]:
            micro_bo = -1.0
        relv = float(v1_arr[m_idx] / max(1e-9, vol_ma1_arr[m_idx]))
        if (abs(micro_vwap) + abs(micro_bo)) == 0 and relv < 1.05:
            continue
        micro_ret = float((c1[m_idx] - o1[m_idx]) / max(1e-9, o1[m_idx]))
        p_up = fair_prob_up_v8(int(st_dir_arr[j]), float(adx5_arr[j]), expj,
                               micro_vwap, micro_bo, relv, micro_ret)
        if p_up >= 0.535 or p_up <= 0.465:
            fair_yes = p_up if p_up >= 0.535 else 1.0 - p_up
            if fair_yes > 0.51:
                v8_bars.add(i)

    both = v30_bars & v8_bars
    v8_only = v8_bars - v30_bars
    v30_only = v30_bars - v8_bars

    print(f"  V3.0 trades on:   {len(v30_bars)} bars")
    print(f"  V8.0 trades on:   {len(v8_bars)} bars")
    print(f"  OVERLAP:          {len(both)} bars")
    print(f"  V8.0 UNIQUE:      {len(v8_only)} bars (not in V3.0)")
    print(f"  V3.0 UNIQUE:      {len(v30_only)} bars (not in V8.0)")

    if v8_only:
        w = sum(1 for idx in v8_only if bars_5m[idx]["close"] > bars_5m[idx]["open"])
        print(f"  V8 unique UP%:    {w / len(v8_only) * 100:.1f}%")

    # Verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print(f"  V3.0:  {s30['trades']}T, {s30['win_rate']:.1f}% WR, ${s30['total_pnl']:+,.2f}, {s30['roi_pct']:+.1f}% ROI")
    print(f"  V8.0:  {s8['trades']}T, {s8['win_rate']:.1f}% WR, ${s8['total_pnl']:+,.2f}, {s8['roi_pct']:+.1f}% ROI")

    if len(v8_only) > 20:
        print(f"\n  V8 has {len(v8_only)} unique trades - potential to stack with V3.0")
    elif len(v8_only) > 0:
        print(f"\n  V8 has {len(v8_only)} unique trades - marginal value")
    else:
        print(f"\n  V8 is a subset of V3.0 - no unique value")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": f"{date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}",
        "bars_1m": len(bars_1m), "bars_5m": len(bars_5m),
        "natural_up_pct": round(natural_up, 2),
        "v30": s30, "v8": s8,
        "overlap": {"v30": len(v30_bars), "v8": len(v8_bars),
                    "both": len(both), "v8_unique": len(v8_only),
                    "v30_unique": len(v30_only)},
    }
    out = Path(__file__).parent / "backtest_v8_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
