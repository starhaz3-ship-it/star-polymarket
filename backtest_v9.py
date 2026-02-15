"""
Head-to-head backtest: V3.0 vs V9.0
V9.0: Ichimoku Cloud + Choppiness Index + RSI-of-Returns on 5m regime
      + StochRSI + CCI + Heikin-Ashi micro-trend on 1m micro
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
# V9 INDICATORS (from user's code)
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


def compute_rsi(series, n=14):
    out = np.full_like(series, np.nan, dtype=float)
    if len(series) < n + 1:
        return out
    gains = np.zeros_like(series, dtype=float)
    losses = np.zeros_like(series, dtype=float)
    for i in range(1, len(series)):
        ch = series[i] - series[i - 1]
        gains[i] = max(0.0, ch)
        losses[i] = max(0.0, -ch)
    avg_g = np.full_like(series, np.nan, dtype=float)
    avg_l = np.full_like(series, np.nan, dtype=float)
    avg_g[n] = np.mean(gains[1:n + 1])
    avg_l[n] = np.mean(losses[1:n + 1])
    for i in range(n + 1, len(series)):
        avg_g[i] = (avg_g[i - 1] * (n - 1) + gains[i]) / n
        avg_l[i] = (avg_l[i - 1] * (n - 1) + losses[i]) / n
    for i in range(n, len(series)):
        if not np.isfinite(avg_g[i]) or not np.isfinite(avg_l[i]) or avg_l[i] == 0:
            out[i] = 100.0 if np.isfinite(avg_g[i]) else np.nan
        else:
            rs = avg_g[i] / avg_l[i]
            out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def compute_cci(h, l, c, n=20):
    tp = (h + l + c) / 3.0
    ma = np_sma(tp, n)
    out = np.full_like(c, np.nan, dtype=float)
    if len(c) < n:
        return out
    for i in range(n - 1, len(c)):
        w = tp[i - n + 1:i + 1]
        md = np.mean(np.abs(w - np.mean(w)))
        if md == 0:
            out[i] = 0.0
        else:
            out[i] = (tp[i] - ma[i]) / (0.015 * md)
    return out


def heikin_ashi(o, h, l, c):
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(c, dtype=float)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(c)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    ha_h = np.maximum.reduce([h, ha_o, ha_c])
    ha_l = np.minimum.reduce([l, ha_o, ha_c])
    return ha_o, ha_h, ha_l, ha_c


def compute_stoch_rsi(rsi_vals, n=14):
    out = np.full_like(rsi_vals, np.nan, dtype=float)
    if len(rsi_vals) < n:
        return out
    rmin = rolling_min(rsi_vals, n)
    rmax = rolling_max(rsi_vals, n)
    denom = rmax - rmin
    for i in range(n - 1, len(rsi_vals)):
        if not np.isfinite(rmin[i]) or not np.isfinite(rmax[i]) or denom[i] == 0:
            out[i] = np.nan
        else:
            out[i] = (rsi_vals[i] - rmin[i]) / denom[i]
    return out


def choppiness_index(h, l, c, n=14):
    tr = true_range(h, l, c)
    tr_sum = np_sma(tr, n) * n  # approximate sum
    hh = rolling_max(h, n)
    ll = rolling_min(l, n)
    out = np.full_like(c, np.nan, dtype=float)
    for i in range(n - 1, len(c)):
        if not np.isfinite(tr_sum[i]) or not np.isfinite(hh[i]) or not np.isfinite(ll[i]) or (hh[i] - ll[i]) == 0:
            continue
        out[i] = 100.0 * math.log10(tr_sum[i] / (hh[i] - ll[i])) / math.log10(n)
    return out


def ichimoku(h, l, c, tenkan_n=9, kijun_n=26, senkou_b_n=52):
    tenkan_sen = (rolling_max(h, tenkan_n) + rolling_min(l, tenkan_n)) / 2.0
    kijun_sen = (rolling_max(h, kijun_n) + rolling_min(l, kijun_n)) / 2.0
    senkou_a = (tenkan_sen + kijun_sen) / 2.0
    senkou_b = (rolling_max(h, senkou_b_n) + rolling_min(l, senkou_b_n)) / 2.0
    return tenkan_sen, kijun_sen, senkou_a, senkou_b


# ============================================================
# V9 PROBABILITY MODEL
# ============================================================

def fair_prob_up_v9(
    ichi_bias,        # +1 above cloud, -1 below, 0 in cloud
    tk_cross,         # tenkan > kijun = +1, else -1
    chop_val,         # choppiness index (high = choppy, low = trending)
    rsi_ret_val,      # RSI of 5m returns
    stoch_rsi_val,    # 1m StochRSI [0-1]
    cci_val,          # 1m CCI
    ha_trend,         # +1 if HA bullish, -1 if HA bearish, 0 if doji
    micro_ret,        # 1m return
):
    score = 0.0

    # 5m regime
    score += 0.65 * float(ichi_bias)                    # cloud direction is backbone
    score += 0.40 * float(tk_cross)                     # tenkan/kijun cross
    # Choppiness: low = trending (good), high = choppy (bad)
    # CHOP < 38.2 = trending, > 61.8 = choppy (standard thresholds)
    if np.isfinite(chop_val):
        score += clamp((50.0 - chop_val) / 15.0, -0.8, 0.8) * 0.35
    # RSI of returns: momentum quality
    if np.isfinite(rsi_ret_val):
        score += clamp((rsi_ret_val - 50.0) / 20.0, -1.0, 1.0) * 0.45

    # 1m micro triggers
    if np.isfinite(stoch_rsi_val):
        # StochRSI > 0.8 = overbought (bullish momentum), < 0.2 = oversold (bearish)
        score += clamp((stoch_rsi_val - 0.5) / 0.25, -1.0, 1.0) * 0.40
    if np.isfinite(cci_val):
        # CCI > 100 = strong bullish impulse, < -100 = bearish
        score += clamp(cci_val / 120.0, -1.2, 1.2) * 0.30
    score += 0.35 * float(ha_trend)                     # HA micro-trend confirmation
    score += clamp(micro_ret / 0.0008, -1.2, 1.2) * 0.40  # raw return

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
# V9.0: Ichimoku + CHOP + RSI-Returns + StochRSI + CCI + HA
# ============================================================

def simulate_v9(bars_5m, bars_1m, size_usd=5.0):
    """
    V9.0 strategy.

    5m regime (previous closed bar):
      - Ichimoku: cloud bias (price vs senkou A/B) + tenkan/kijun cross
      - Choppiness Index: filter choppy markets
      - RSI of 5m returns: momentum quality

    1m micro (first 1 minute of current bar):
      - StochRSI(14): fast turn detection
      - CCI(20): impulse/mean reversion
      - Heikin-Ashi trend: noise-reduced micro direction
      - Raw 1m return
    """
    P_BAND = 0.035
    BUY_PRICE = 0.51
    DECIDE_AFTER = 1  # after first 1m candle
    CHOP_MAX = 65.0   # skip if choppiness > this (too choppy)

    # Build 5m numpy arrays
    n5 = len(bars_5m)
    h5 = np.array([b["high"] for b in bars_5m])
    l5 = np.array([b["low"] for b in bars_5m])
    c5 = np.array([b["close"] for b in bars_5m])
    o5 = np.array([b["open"] for b in bars_5m])

    # 5m indicators
    tenkan, kijun, senkou_a, senkou_b = ichimoku(h5, l5, c5, 9, 26, 52)
    chop5 = choppiness_index(h5, l5, c5, 14)

    # RSI of 5m returns (not RSI of price — RSI of bar returns)
    rets5 = np.zeros_like(c5)
    rets5[1:] = (c5[1:] - c5[:-1]) / c5[:-1] * 10000  # in bps
    rsi_ret5 = compute_rsi(rets5, 14)

    # Build 1m numpy arrays
    n1 = len(bars_1m)
    h1 = np.array([b["high"] for b in bars_1m])
    l1 = np.array([b["low"] for b in bars_1m])
    c1 = np.array([b["close"] for b in bars_1m])
    o1 = np.array([b["open"] for b in bars_1m])

    # 1m indicators
    rsi_1m = compute_rsi(c1, 14)
    stoch_rsi_1m = compute_stoch_rsi(rsi_1m, 14)
    cci_1m = compute_cci(h1, l1, c1, 20)
    ha_o1, ha_h1, ha_l1, ha_c1 = heikin_ashi(o1, h1, l1, c1)

    trades = []
    # Ichimoku needs senkou_b(52) warmup
    start = max(55, 30)

    for i in range(start, n5):
        bar = bars_5m[i]
        mins = bar["minutes"]
        if len(mins) < 5:
            continue

        # Previous closed 5m bar (no lookahead)
        j = i - 1
        if j <= 0:
            continue

        # --- 5m regime ---

        # Ichimoku cloud bias
        if not (np.isfinite(senkou_a[j]) and np.isfinite(senkou_b[j]) and
                np.isfinite(tenkan[j]) and np.isfinite(kijun[j])):
            continue

        cloud_top = max(senkou_a[j], senkou_b[j])
        cloud_bot = min(senkou_a[j], senkou_b[j])

        if c5[j] > cloud_top:
            ichi_bias = 1   # above cloud = bullish
        elif c5[j] < cloud_bot:
            ichi_bias = -1  # below cloud = bearish
        else:
            ichi_bias = 0   # inside cloud = neutral

        tk_cross = 1 if tenkan[j] > kijun[j] else -1

        # Choppiness filter
        if not np.isfinite(chop5[j]):
            continue
        if chop5[j] > CHOP_MAX:
            continue

        # RSI of returns
        rsi_ret_val = rsi_ret5[j] if np.isfinite(rsi_ret5[j]) else 50.0

        # --- 1m micro at decision minute ---
        m_idx = i * 5 + DECIDE_AFTER
        if m_idx >= n1:
            continue

        stoch_val = stoch_rsi_1m[m_idx] if m_idx < len(stoch_rsi_1m) and np.isfinite(stoch_rsi_1m[m_idx]) else 0.5
        cci_val = cci_1m[m_idx] if m_idx < len(cci_1m) and np.isfinite(cci_1m[m_idx]) else 0.0

        # Heikin-Ashi micro-trend: bullish if ha_close > ha_open
        if m_idx < len(ha_c1):
            if ha_c1[m_idx] > ha_o1[m_idx]:
                ha_trend = 1
            elif ha_c1[m_idx] < ha_o1[m_idx]:
                ha_trend = -1
            else:
                ha_trend = 0
        else:
            ha_trend = 0

        # Raw 1m return
        micro_ret = float((c1[m_idx] - o1[m_idx]) / max(1e-9, o1[m_idx]))

        p_up = fair_prob_up_v9(
            ichi_bias=ichi_bias,
            tk_cross=tk_cross,
            chop_val=chop5[j],
            rsi_ret_val=rsi_ret_val,
            stoch_rsi_val=stoch_val,
            cci_val=cci_val,
            ha_trend=ha_trend,
            micro_ret=micro_ret,
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

        # Execute
        shares = max(1, math.floor(size_usd / BUY_PRICE))
        cost = BUY_PRICE * shares

        up_win = bar["close"] > bar["open"]
        win = (up_win and side == "UP") or (not up_win and side == "DOWN")
        pnl = (shares * 1.0 - cost) if win else -cost

        trades.append({
            "direction": side, "actual": "UP" if up_win else "DOWN",
            "win": win, "p_up": p_up, "fair_yes": fair_yes,
            "ichi_bias": ichi_bias, "tk_cross": tk_cross,
            "chop": float(chop5[j]), "rsi_ret": rsi_ret_val,
            "stoch_rsi": stoch_val, "cci": cci_val,
            "ha_trend": ha_trend,
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

    # V3.0 baseline
    v30 = simulate_v30(bars_5m, SIZE)
    s30 = analyze(v30, "V3.0 - Simple Momentum (2min, 15bp, $5)")

    # V9.0
    v9 = simulate_v9(bars_5m, bars_1m, SIZE)
    s9 = analyze(v9, "V9.0 - Ichimoku+CHOP+RSI-Ret+StochRSI+CCI+HA ($5)")

    # V9 feature breakdown
    if v9:
        print(f"\n{'=' * 60}")
        print(f"  V9.0 FEATURE BREAKDOWN")
        print(f"{'=' * 60}")

        # By Ichimoku bias
        ichi_bull = [t for t in v9 if t.get("ichi_bias") == 1]
        ichi_bear = [t for t in v9 if t.get("ichi_bias") == -1]
        ichi_neut = [t for t in v9 if t.get("ichi_bias") == 0]
        if ichi_bull:
            wr = sum(1 for t in ichi_bull if t["win"]) / len(ichi_bull) * 100
            print(f"  Above Cloud:     {len(ichi_bull)}T, {wr:.1f}% WR")
        if ichi_bear:
            wr = sum(1 for t in ichi_bear if t["win"]) / len(ichi_bear) * 100
            print(f"  Below Cloud:     {len(ichi_bear)}T, {wr:.1f}% WR")
        if ichi_neut:
            wr = sum(1 for t in ichi_neut if t["win"]) / len(ichi_neut) * 100
            print(f"  In Cloud:        {len(ichi_neut)}T, {wr:.1f}% WR")

        # By TK cross
        tk_bull = [t for t in v9 if t.get("tk_cross") == 1]
        tk_bear = [t for t in v9 if t.get("tk_cross") == -1]
        if tk_bull:
            wr = sum(1 for t in tk_bull if t["win"]) / len(tk_bull) * 100
            print(f"  TK Bull:         {len(tk_bull)}T, {wr:.1f}% WR")
        if tk_bear:
            wr = sum(1 for t in tk_bear if t["win"]) / len(tk_bear) * 100
            print(f"  TK Bear:         {len(tk_bear)}T, {wr:.1f}% WR")

        # By choppiness
        chop_lo = [t for t in v9 if t.get("chop", 50) < 45]
        chop_hi = [t for t in v9 if 45 <= t.get("chop", 50) <= 65]
        if chop_lo:
            wr = sum(1 for t in chop_lo if t["win"]) / len(chop_lo) * 100
            print(f"  CHOP < 45:       {len(chop_lo)}T, {wr:.1f}% WR (trending)")
        if chop_hi:
            wr = sum(1 for t in chop_hi if t["win"]) / len(chop_hi) * 100
            print(f"  CHOP 45-65:      {len(chop_hi)}T, {wr:.1f}% WR (moderate)")

        # By HA trend
        ha_bull = [t for t in v9 if t.get("ha_trend") == 1]
        ha_bear = [t for t in v9 if t.get("ha_trend") == -1]
        if ha_bull:
            wr = sum(1 for t in ha_bull if t["win"]) / len(ha_bull) * 100
            print(f"  HA Bullish:      {len(ha_bull)}T, {wr:.1f}% WR")
        if ha_bear:
            wr = sum(1 for t in ha_bear if t["win"]) / len(ha_bear) * 100
            print(f"  HA Bearish:      {len(ha_bear)}T, {wr:.1f}% WR")

    # Head to head
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD (all $5 sizing)")
    print("=" * 70)

    def row(label, *vals):
        parts = [f"  {label:20s}"]
        for v in vals:
            parts.append(f"{v:>16s}")
        print("".join(parts))

    row("", "V3.0", "V9.0")
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
        for s in [s30, s9]:
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

    # Track V9 bar indices by re-deriving which bars V9 trades on
    v9_bars = set()
    n5 = len(bars_5m)
    h5 = np.array([b["high"] for b in bars_5m])
    l5 = np.array([b["low"] for b in bars_5m])
    c5 = np.array([b["close"] for b in bars_5m])
    o5 = np.array([b["open"] for b in bars_5m])
    h1 = np.array([b["high"] for b in bars_1m])
    l1 = np.array([b["low"] for b in bars_1m])
    c1 = np.array([b["close"] for b in bars_1m])
    o1 = np.array([b["open"] for b in bars_1m])

    tenkan, kijun, senkou_a, senkou_b = ichimoku(h5, l5, c5, 9, 26, 52)
    chop5 = choppiness_index(h5, l5, c5, 14)
    rets5 = np.zeros_like(c5)
    rets5[1:] = (c5[1:] - c5[:-1]) / c5[:-1] * 10000
    rsi_ret5 = compute_rsi(rets5, 14)
    rsi_1m = compute_rsi(c1, 14)
    stoch_rsi_1m = compute_stoch_rsi(rsi_1m, 14)
    cci_1m = compute_cci(h1, l1, c1, 20)
    ha_o1, ha_h1, ha_l1, ha_c1 = heikin_ashi(o1, h1, l1, c1)

    for i in range(55, n5):
        j = i - 1
        if j <= 0:
            continue
        if not (np.isfinite(senkou_a[j]) and np.isfinite(senkou_b[j]) and
                np.isfinite(tenkan[j]) and np.isfinite(kijun[j])):
            continue
        if not np.isfinite(chop5[j]) or chop5[j] > 65.0:
            continue
        cloud_top = max(senkou_a[j], senkou_b[j])
        cloud_bot = min(senkou_a[j], senkou_b[j])
        if c5[j] > cloud_top:
            ichi_bias = 1
        elif c5[j] < cloud_bot:
            ichi_bias = -1
        else:
            ichi_bias = 0
        tk_cross = 1 if tenkan[j] > kijun[j] else -1
        rsi_ret_val = rsi_ret5[j] if np.isfinite(rsi_ret5[j]) else 50.0

        m_idx = i * 5 + 1
        if m_idx >= len(bars_1m):
            continue
        stoch_val = stoch_rsi_1m[m_idx] if m_idx < len(stoch_rsi_1m) and np.isfinite(stoch_rsi_1m[m_idx]) else 0.5
        cci_val = cci_1m[m_idx] if m_idx < len(cci_1m) and np.isfinite(cci_1m[m_idx]) else 0.0
        if m_idx < len(ha_c1):
            ha_trend = 1 if ha_c1[m_idx] > ha_o1[m_idx] else (-1 if ha_c1[m_idx] < ha_o1[m_idx] else 0)
        else:
            ha_trend = 0
        micro_ret = float((c1[m_idx] - o1[m_idx]) / max(1e-9, o1[m_idx]))

        p_up = fair_prob_up_v9(ichi_bias, tk_cross, chop5[j], rsi_ret_val,
                               stoch_val, cci_val, ha_trend, micro_ret)
        if p_up >= 0.535 or p_up <= 0.465:
            fair_yes = p_up if p_up >= 0.535 else 1.0 - p_up
            if fair_yes > 0.51:
                v9_bars.add(i)

    both = v30_bars & v9_bars
    v9_only = v9_bars - v30_bars
    v30_only = v30_bars - v9_bars

    print(f"  V3.0 trades on:   {len(v30_bars)} bars")
    print(f"  V9.0 trades on:   {len(v9_bars)} bars")
    print(f"  OVERLAP:          {len(both)} bars")
    print(f"  V9.0 UNIQUE:      {len(v9_only)} bars (not in V3.0)")
    print(f"  V3.0 UNIQUE:      {len(v30_only)} bars (not in V9.0)")

    if v9_only:
        w = sum(1 for idx in v9_only if bars_5m[idx]["close"] > bars_5m[idx]["open"])
        print(f"  V9 unique UP%:    {w / len(v9_only) * 100:.1f}%")

    # Verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    print(f"  V3.0:  {s30['trades']}T, {s30['win_rate']:.1f}% WR, ${s30['total_pnl']:+,.2f}, {s30['roi_pct']:+.1f}% ROI")
    print(f"  V9.0:  {s9['trades']}T, {s9['win_rate']:.1f}% WR, ${s9['total_pnl']:+,.2f}, {s9['roi_pct']:+.1f}% ROI")

    if len(v9_only) > 20:
        print(f"\n  V9 has {len(v9_only)} unique trades - potential to stack with V3.0")
    elif len(v9_only) > 0:
        print(f"\n  V9 has {len(v9_only)} unique trades - marginal value")
    else:
        print(f"\n  V9 is a subset of V3.0 - no unique value")

    # All-versions summary
    print("\n" + "=" * 70)
    print("  ALL VERSIONS TESTED (7-day backtest, $5 sizing)")
    print("=" * 70)
    print(f"  {'Version':<12} {'Trades':>7} {'WR%':>7} {'PnL':>10} {'ROI%':>8} {'MaxDD':>8}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*10} {'-'*8} {'-'*8}")
    for name, wr, pnl, roi, dd, tr in [
        ("V3.0", s30['win_rate'], s30['total_pnl'], s30['roi_pct'], s30['max_drawdown'], s30['trades']),
        ("V9.0", s9['win_rate'], s9['total_pnl'], s9['roi_pct'], s9['max_drawdown'], s9['trades']),
    ]:
        print(f"  {name:<12} {tr:>7} {wr:>6.1f}% ${pnl:>+8.2f} {roi:>+7.1f}% ${dd:>7.2f}")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": f"{date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}",
        "bars_1m": len(bars_1m), "bars_5m": len(bars_5m),
        "natural_up_pct": round(natural_up, 2),
        "v30": s30, "v9": s9,
        "overlap": {"v30": len(v30_bars), "v9": len(v9_bars),
                    "both": len(both), "v9_unique": len(v9_only),
                    "v30_unique": len(v30_only)},
    }
    out = Path(__file__).parent / "backtest_v9_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
