"""
Unified Multi-TF BTC Strategy Paper Trader V2.0

Runs 6 independent BTC directional strategies in ONE process:
  1. EMA_STOCHRSI      - EMA(12/26/55) + VWAP + StochRSI pullback      [15M]
  2. FRAMA_CMO         - FRAMA(16) + CMO(14) + Donchian(20) pullback    [15M]
  3. ICHI_ADXR_SMI     - Ichimoku(9/26/52) + ADXR(14) + SMI pullback   [15M]
  4. VIDYA_AROON       - VIDYA(14) + Aroon(25) + Chaikin Osc cross      [15M]
  5. VORTEX_RVI        - Vortex(14) + RVI(10) + OBV slope filter        [15M]
  6. FIB_EMA_CONFIRM   - 1M Fib Confluence + 5M EMA trend confirm       [5M]

Shared: candle fetch (multi-TF), market discovery, resolution, stats, ML optimizer.
Each strategy generates signals independently per cycle.
ML auto-optimizer retrains after 30 trades to gate entries and scale sizing.

Usage:
  python run_15m_strategies.py          # Paper mode (default)
  python run_15m_strategies.py --live   # Live mode
"""

import sys
import math
import json
import time
import os
import argparse
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from functools import partial as fn_partial
from typing import Optional, Tuple, List, Dict

import httpx

print = fn_partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# SHARED CONFIG
# ============================================================================
SCAN_INTERVAL = 30
MAX_CONCURRENT_PER_STRAT = 2
TIME_WINDOW_15M = (2.0, 12.0)
TIME_WINDOW_5M = (1.5, 7.0)
SPREAD_OFFSET = 0.03
MIN_ENTRY_PRICE = 0.10
MAX_ENTRY_PRICE = 0.60
RESOLVE_AGE_MIN = 16.0
RESULTS_FILE = Path(__file__).parent / "fifteen_min_strategies_results.json"
ML_FEATURES_FILE = Path(__file__).parent / "fifteen_min_ml_features.json"

# FIB strategy config (5M markets, separate from 15M STRATEGIES registry)
FIB_5M_CONFIG = {"min_edge": 0.010, "tp": 0.020, "sl": 0.015, "base": 3.0, "max": 8.0}

# ============================================================================
# SHARED INDICATORS
# ============================================================================

def calc_ema(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(arr, dtype=float)
    a = 2.0 / (n + 1.0)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out


def calc_sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if n <= 0 or len(arr) < n:
        return out
    c = np.cumsum(arr, dtype=float)
    c[n:] = c[n:] - c[:-n]
    out[n - 1:] = c[n - 1:] / n
    return out


def calc_stddev(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        out[i] = float(np.std(arr[i - n + 1:i + 1], ddof=0))
    return out


def calc_rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=float)
    if len(close) < n + 1:
        return out
    delta = np.diff(close, prepend=close[0])
    up = np.clip(delta, 0, None)
    dn = np.clip(-delta, 0, None)
    au = calc_sma(up, n)
    ad = calc_sma(dn, n)
    rs = au / np.where(ad == 0, np.nan, ad)
    out[:] = 100.0 - (100.0 / (1.0 + rs))
    return out


def calc_stochrsi(close: np.ndarray, rsi_len: int = 14, stoch_len: int = 14, smooth: int = 3):
    r = calc_rsi(close, rsi_len)
    k = np.full_like(close, np.nan, dtype=float)
    for i in range(stoch_len - 1, len(close)):
        w = r[i - stoch_len + 1:i + 1]
        lo, hi = np.nanmin(w), np.nanmax(w)
        if np.isfinite(lo) and np.isfinite(hi) and hi != lo:
            k[i] = (r[i] - lo) / (hi - lo)
    k_s = calc_sma(np.where(np.isnan(k), 0.0, k), smooth)
    d_s = calc_sma(np.where(np.isnan(k_s), 0.0, k_s), smooth)
    return k_s, d_s


def calc_rolling_vwap(high, low, close, vol, n=96):
    tp = (high + low + close) / 3.0
    pv = tp * vol
    out = np.full_like(close, np.nan, dtype=float)
    for i in range(n - 1, len(close)):
        v = float(np.sum(vol[i - n + 1:i + 1]))
        if v > 0:
            out[i] = float(np.sum(pv[i - n + 1:i + 1])) / v
    return out


def calc_bb_width(close, n=20, mult=2.0):
    mid = calc_sma(close, n)
    sd = calc_stddev(close, n)
    return (mid + mult * sd - (mid - mult * sd)) / np.where(mid == 0, np.nan, mid)


def calc_true_range(high, low, close):
    out = np.empty_like(close, dtype=float)
    out[0] = high[0] - low[0]
    for i in range(1, len(close)):
        out[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    return out


def calc_atr(high, low, close, n=14):
    return calc_sma(calc_true_range(high, low, close), n)


def rolling_hhll(high, low, n):
    hh = np.full_like(high, np.nan, dtype=float)
    ll = np.full_like(low, np.nan, dtype=float)
    for i in range(n - 1, len(high)):
        hh[i] = float(np.max(high[i - n + 1:i + 1]))
        ll[i] = float(np.min(low[i - n + 1:i + 1]))
    return hh, ll


# --- FRAMA ---
def calc_frama(close, n=16, fast=4.0, slow=300.0):
    out = np.full_like(close, np.nan, dtype=float)
    if len(close) < n + 2 or n < 4:
        return out
    n2 = n // 2
    w = float(n)
    af = 2.0 / (fast + 1.0)
    asl = 2.0 / (slow + 1.0)
    out[n - 1] = float(np.mean(close[:n]))
    for i in range(n, len(close)):
        w0 = close[i - n + 1:i + 1]
        w1 = close[i - n + 1:i - n2 + 1]
        w2 = close[i - n2 + 1:i + 1]
        hi0, lo0 = float(np.max(w0)), float(np.min(w0))
        hi1, lo1 = float(np.max(w1)), float(np.min(w1))
        hi2, lo2 = float(np.max(w2)), float(np.min(w2))
        n0 = (hi0 - lo0) / w
        n1 = (hi1 - lo1) / (w / 2.0)
        n2v = (hi2 - lo2) / (w / 2.0)
        if n0 > 0 and n1 > 0 and n2v > 0:
            dim = (math.log(n1 + n2v) - math.log(n0)) / math.log(2.0)
            dim = float(np.clip(dim, 1.0, 2.0))
            alpha = math.exp(-4.6 * (dim - 1.0))
            alpha = float(np.clip(alpha, asl, af))
        else:
            alpha = asl
        prev = out[i - 1] if np.isfinite(out[i - 1]) else close[i - 1]
        out[i] = alpha * close[i] + (1.0 - alpha) * prev
    return out


# --- CMO ---
def calc_cmo(close, n=14):
    out = np.full_like(close, np.nan, dtype=float)
    if len(close) < n + 1:
        return out
    d = np.diff(close, prepend=close[0])
    up = np.clip(d, 0, None)
    dn = np.clip(-d, 0, None)
    for i in range(n - 1, len(close)):
        su = float(np.sum(up[i - n + 1:i + 1]))
        sd = float(np.sum(dn[i - n + 1:i + 1]))
        denom = su + sd
        if denom > 0:
            out[i] = 100.0 * (su - sd) / denom
    return out


# --- Donchian ---
def calc_donchian(high, low, n):
    up, dn, mid = np.full_like(high, np.nan, dtype=float), np.full_like(low, np.nan, dtype=float), np.full_like(high, np.nan, dtype=float)
    for i in range(n - 1, len(high)):
        hh = float(np.max(high[i - n + 1:i + 1]))
        ll = float(np.min(low[i - n + 1:i + 1]))
        up[i], dn[i], mid[i] = hh, ll, (hh + ll) / 2.0
    return up, dn, mid


# --- Ichimoku ---
def calc_ichimoku(high, low, tenkan=9, kijun=26, senkou_b=52):
    hh_t, ll_t = rolling_hhll(high, low, tenkan)
    hh_k, ll_k = rolling_hhll(high, low, kijun)
    hh_b, ll_b = rolling_hhll(high, low, senkou_b)
    return (hh_t + ll_t) / 2.0, (hh_k + ll_k) / 2.0, ((hh_t + ll_t) / 2.0 + (hh_k + ll_k) / 2.0) / 2.0, (hh_b + ll_b) / 2.0


# --- DMI/ADX/ADXR ---
def calc_dmi_adx(high, low, close, n=14):
    up_move = np.diff(high, prepend=high[0])
    dn_move = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr = calc_true_range(high, low, close)
    atrn = calc_sma(tr, n)
    safe = np.where(atrn == 0, np.nan, atrn)
    pdi = 100.0 * calc_sma(plus_dm, n) / safe
    mdi = 100.0 * calc_sma(minus_dm, n) / safe
    dx = 100.0 * np.abs(pdi - mdi) / np.where((pdi + mdi) == 0, np.nan, (pdi + mdi))
    adx = calc_sma(np.nan_to_num(dx, nan=0.0), n)
    return pdi, mdi, adx


def calc_adxr(adx, n=14):
    out = np.full_like(adx, np.nan, dtype=float)
    for i in range(len(adx)):
        if i - n >= 0 and np.isfinite(adx[i]) and np.isfinite(adx[i - n]):
            out[i] = 0.5 * (adx[i] + adx[i - n])
    return out


# --- SMI ---
def calc_smi(high, low, close, length=14, s1=3, s2=3):
    hh, ll = rolling_hhll(high, low, length)
    m = (hh + ll) / 2.0
    d2 = calc_ema(calc_ema(np.nan_to_num(close - m, nan=0.0), s1), s2)
    r2 = calc_ema(calc_ema(np.nan_to_num(hh - ll, nan=0.0), s1), s2)
    denom = 0.5 * r2
    smi_val = 100.0 * d2 / np.where(denom == 0, np.nan, denom)
    sig = calc_ema(np.nan_to_num(smi_val, nan=0.0), s2)
    return smi_val, sig


# --- VIDYA ---
def calc_vidya(close, er_len=14, alpha_min=0.05, alpha_max=0.50):
    out = np.full_like(close, np.nan, dtype=float)
    if len(close) < er_len + 2:
        return out
    out[er_len] = float(np.mean(close[:er_len + 1]))
    for i in range(er_len + 1, len(close)):
        change = abs(close[i] - close[i - er_len])
        vol = float(np.sum(np.abs(np.diff(close[i - er_len:i + 1]))))
        er = (change / vol) if vol > 0 else 0.0
        a = alpha_min + er * (alpha_max - alpha_min)
        prev = out[i - 1] if np.isfinite(out[i - 1]) else close[i - 1]
        out[i] = prev + a * (close[i] - prev)
    return out


# --- Aroon ---
def calc_aroon(high, low, n=25):
    up = np.full_like(high, np.nan, dtype=float)
    dn = np.full_like(low, np.nan, dtype=float)
    for i in range(n - 1, len(high)):
        hh_idx = int(np.argmax(high[i - n + 1:i + 1]))
        ll_idx = int(np.argmin(low[i - n + 1:i + 1]))
        up[i] = 100.0 * (n - ((n - 1) - hh_idx)) / n
        dn[i] = 100.0 * (n - ((n - 1) - ll_idx)) / n
    return up, dn


# --- ADL / Chaikin ---
def calc_adl(high, low, close, vol):
    out = np.full_like(close, np.nan, dtype=float)
    acc = 0.0
    for i in range(len(close)):
        rng = high[i] - low[i]
        mfm = ((close[i] - low[i]) - (high[i] - close[i])) / rng if rng > 0 else 0.0
        acc += mfm * vol[i]
        out[i] = acc
    return out


def calc_chaikin(high, low, close, vol, fast=3, slow=10):
    a = calc_adl(high, low, close, vol)
    return calc_ema(np.nan_to_num(a, nan=0.0), fast) - calc_ema(np.nan_to_num(a, nan=0.0), slow)


# --- Vortex ---
def calc_vortex(high, low, close, n=14):
    vm_plus = np.full_like(close, np.nan, dtype=float)
    vm_minus = np.full_like(close, np.nan, dtype=float)
    tr = calc_true_range(high, low, close)
    for i in range(1, len(close)):
        vm_plus[i] = abs(high[i] - low[i - 1])
        vm_minus[i] = abs(low[i] - high[i - 1])
    vi_plus = np.full_like(close, np.nan, dtype=float)
    vi_minus = np.full_like(close, np.nan, dtype=float)
    for i in range(n - 1, len(close)):
        tr_sum = float(np.sum(tr[i - n + 1:i + 1]))
        if tr_sum <= 0:
            continue
        vi_plus[i] = float(np.sum(vm_plus[i - n + 1:i + 1])) / tr_sum
        vi_minus[i] = float(np.sum(vm_minus[i - n + 1:i + 1])) / tr_sum
    return vi_plus, vi_minus


# --- RVI ---
def calc_rvi(open_, high, low, close, n=10):
    co = close - open_
    hl = high - low
    num = calc_sma(co, 4)
    den = calc_sma(hl, 4)
    ratio = num / np.where(den == 0, np.nan, den)
    r = calc_sma(np.nan_to_num(ratio, nan=0.0), n)
    sig = calc_sma(np.nan_to_num(r, nan=0.0), 4)
    return r, sig


# --- OBV ---
def calc_obv(close, vol):
    out = np.full_like(close, np.nan, dtype=float)
    acc = 0.0
    out[0] = 0.0
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            acc += vol[i]
        elif close[i] < close[i - 1]:
            acc -= vol[i]
        out[i] = acc
    return out


# --- Linear slope ---
def calc_lin_slope(x, n=30):
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) < n:
        return out
    t = np.arange(n, dtype=float)
    t_mean = t.mean()
    denom = np.sum((t - t_mean) ** 2)
    for i in range(n - 1, len(x)):
        y = x[i - n + 1:i + 1].astype(float)
        num = np.sum((t - t_mean) * (y - y.mean()))
        out[i] = num / denom if denom != 0 else 0.0
    return out


# ============================================================================
# STRATEGY 1: EMA_STOCHRSI
# ============================================================================

def signal_ema_stochrsi(close, high, low, vol, open_=None) -> Optional[dict]:
    if len(close) < 60:
        return None
    ef = calc_ema(close, 12)
    em = calc_ema(close, 26)
    es = calc_ema(close, 55)
    vw = calc_rolling_vwap(high, low, close, vol, 96)
    bw = calc_bb_width(close, 20, 2.0)
    r = calc_rsi(close, 14)
    k, d = calc_stochrsi(close, 14, 14, 3)

    i = len(close) - 1
    c = close[i]
    vals = [c, ef[i], em[i], es[i], vw[i], bw[i], r[i], k[i], k[i-1]]
    if any(not np.isfinite(x) for x in vals):
        return None
    if bw[i] < 0.010:
        return None

    trend_up = (ef[i] > em[i] > es[i]) and (c >= vw[i])
    trend_dn = (ef[i] < em[i] < es[i]) and (c <= vw[i])
    long_pull = trend_up and (r[i] >= 52.0) and (k[i-1] <= 0.35) and (k[i] > 0.35)
    short_pull = trend_dn and (r[i] <= 48.0) and (k[i-1] >= 0.65) and (k[i] < 0.65)

    # Fair prob
    p = 0.5
    if trend_up: p += 0.10
    elif trend_dn: p -= 0.10
    p += (r[i] - 50.0) / 400.0
    slope = (ef[i] - es[i]) / max(1e-9, c)
    p += float(np.clip(slope * 30.0, -0.06, 0.06))
    p += float(np.clip((bw[i] - 0.012) * 2.0, -0.04, 0.04))
    fair = float(np.clip(p, 0.05, 0.95))

    sep = abs(ef[i] - es[i]) / max(1e-9, c)
    strength = float(np.clip((sep * 40.0) + ((bw[i] - 0.010) * 10.0), 0.0, 1.0))

    side = None
    reason = "no_entry"
    if long_pull:
        side = "UP"; reason = "ema_up_stochrsi_pullback"
    elif short_pull:
        side = "DOWN"; reason = "ema_dn_stochrsi_pullback"

    return {"side": side, "strength": strength, "fair_up": fair, "reason": reason,
            "price": c, "detail": f"ef={ef[i]:.0f} em={em[i]:.0f} es={es[i]:.0f} rsi={r[i]:.0f} stK={k[i]:.2f} bbw={bw[i]:.3f}"}


# ============================================================================
# STRATEGY 2: FRAMA_CMO
# ============================================================================

def signal_frama_cmo(close, high, low, vol, open_=None) -> Optional[dict]:
    if len(close) < 30:
        return None
    f = calc_frama(close, 16)
    m = calc_cmo(close, 14)
    dc_up, dc_dn, dc_mid = calc_donchian(high, low, 20)

    i = len(close) - 1
    c, fv, mv, mid = close[i], f[i], m[i], dc_mid[i]
    if any(not np.isfinite(x) for x in [c, fv, mv, mid]):
        return None

    trend_up = (c > fv) and (mv >= 10.0)
    trend_dn = (c < fv) and (mv <= -10.0)
    near_frama = abs(c - fv) / max(1e-9, c) <= 0.0012
    near_mid = abs(c - mid) / max(1e-9, c) <= 0.0012
    prev_c = close[i - 1]
    resume_long = (c > prev_c) and (c > mid)
    resume_short = (c < prev_c) and (c < mid)

    p = 0.50
    p += float(np.clip((c - fv) / max(1e-9, c) * 8.0, -0.12, 0.12))
    p += float(np.clip(mv / 200.0, -0.10, 0.10))
    p += float(np.clip((c - mid) / max(1e-9, c) * 6.0, -0.08, 0.08))
    fair = float(np.clip(p, 0.05, 0.95))

    dist = abs(c - fv) / max(1e-9, c)
    strength = float(np.clip((abs(mv) / 80.0) * 0.6 + np.clip(dist / 0.0013, 0, 2) * 0.2 + 0.2, 0.0, 1.0))

    side = None
    reason = "no_entry"
    if trend_up and (near_frama or near_mid) and resume_long:
        side = "UP"; reason = "frama_up_pullback"
    elif trend_dn and (near_frama or near_mid) and resume_short:
        side = "DOWN"; reason = "frama_dn_pullback"

    return {"side": side, "strength": strength, "fair_up": fair, "reason": reason,
            "price": c, "detail": f"frama={fv:.0f} cmo={mv:.1f} dcmid={mid:.0f}"}


# ============================================================================
# STRATEGY 3: ICHI_ADXR_SMI
# ============================================================================

def signal_ichi_adxr_smi(close, high, low, vol, open_=None) -> Optional[dict]:
    min_len = max(52, 14 * 3, 14 + 6, 14) + 10
    if len(close) < min_len:
        return None

    ten, kij, spa, spb = calc_ichimoku(high, low, 9, 26, 52)
    pdi, mdi, adx = calc_dmi_adx(high, low, close, 14)
    axr = calc_adxr(adx, 14)
    sm, sm_sig = calc_smi(high, low, close, 14, 3, 3)
    a = calc_atr(high, low, close, 14)

    i = len(close) - 1
    c = close[i]
    vals = [c, ten[i], kij[i], spa[i], spb[i], pdi[i], mdi[i], axr[i], sm[i], sm[i-1], sm_sig[i], a[i]]
    if any(not np.isfinite(x) for x in vals):
        return None

    atrp = a[i] / max(1e-9, c)
    if atrp < 0.0009 or atrp > 0.04:
        return None
    if axr[i] < 18.0:
        return None

    ct = max(spa[i], spb[i])
    cb = min(spa[i], spb[i])
    up_regime = (c > ct) and (ten[i] >= kij[i]) and ((pdi[i] - mdi[i]) >= 2.0)
    dn_regime = (c < cb) and (ten[i] <= kij[i]) and ((mdi[i] - pdi[i]) >= 2.0)

    long_pull = up_regime and sm[i-1] <= -20.0 and sm[i] > -17.0 and sm[i] >= sm_sig[i]
    short_pull = dn_regime and sm[i-1] >= 20.0 and sm[i] < 17.0 and sm[i] <= sm_sig[i]

    cloud_thick = (ct - cb) / max(1e-9, c)
    vol_pen = float(np.clip((atrp - 0.02) * 1.2, 0.0, 0.08))
    p = 0.50
    if c > ct and ten[i] > kij[i]: p += 0.10
    elif c < cb and ten[i] < kij[i]: p -= 0.10
    p += float(np.clip(((pdi[i] - mdi[i]) / 100.0) * 0.10, -0.10, 0.10))
    p += float(np.clip((axr[i] - 18.0) * 0.003, -0.03, 0.06))
    p += float(np.clip(sm[i] / 300.0, -0.10, 0.10))
    p -= float(np.clip(cloud_thick * 3.0, 0.0, 0.05))
    p -= vol_pen
    fair = float(np.clip(p, 0.05, 0.95))

    strength = min(1.0, abs(pdi[i] - mdi[i]) / 20.0) * 0.45 + min(1.0, max(0, (axr[i] - 18) / 15)) * 0.25 + min(1.0, abs(sm[i]) / 60.0) * 0.30
    strength = float(np.clip(strength, 0.0, 1.0))

    side = None
    reason = "no_entry"
    if long_pull:
        side = "UP"; reason = "ichi_up_smi_pullback"
    elif short_pull:
        side = "DOWN"; reason = "ichi_dn_smi_pullback"

    return {"side": side, "strength": strength, "fair_up": fair, "reason": reason,
            "price": c, "detail": f"ten={ten[i]:.0f} kij={kij[i]:.0f} adxr={axr[i]:.0f} smi={sm[i]:.0f} +di={pdi[i]:.0f} -di={mdi[i]:.0f}"}


# ============================================================================
# STRATEGY 4: VIDYA_AROON_CHAIKIN
# ============================================================================

def signal_vidya_aroon(close, high, low, vol, open_=None) -> Optional[dict]:
    if len(close) < 40:
        return None

    v = calc_vidya(close, 14, 0.06, 0.45)
    au, ad = calc_aroon(high, low, 25)
    ch = calc_chaikin(high, low, close, vol, 3, 10)
    a = calc_atr(high, low, close, 14)

    i = len(close) - 1
    c = close[i]
    vals = [c, v[i], v[i-1], au[i], ad[i], ch[i], ch[i-1], a[i]]
    if any(not np.isfinite(x) for x in vals):
        return None

    atrp = a[i] / max(1e-9, c)
    if atrp < 0.0009 or atrp > 0.04:
        return None

    v_slope = (v[i] - v[i-1]) / max(1e-9, c)
    up_regime = (c >= v[i]) and (v_slope > 0) and ((au[i] - ad[i]) >= 12.0)
    dn_regime = (c <= v[i]) and (v_slope < 0) and ((ad[i] - au[i]) >= 12.0)

    long_pull = up_regime and (ch[i-1] <= 0.0) and (ch[i] > 0.0)
    short_pull = dn_regime and (ch[i-1] >= 0.0) and (ch[i] < 0.0)

    vol_pen = float(np.clip((atrp - 0.02) * 1.2, 0.0, 0.08))
    p = 0.50
    p += float(np.clip((c - v[i]) / max(1e-9, c) * 10.0, -0.14, 0.14))
    p += float(np.clip(((au[i] - ad[i]) / 100.0) * 0.10, -0.10, 0.10))
    ch_std = float(np.nanstd(ch[max(0, i-200):i+1])) + 1e-9
    p += float(np.clip(ch[i] / ch_std * 0.03, -0.06, 0.06))
    p -= vol_pen
    fair = float(np.clip(p, 0.05, 0.95))

    strength = min(1.0, abs(au[i] - ad[i]) / 30.0) * 0.45 + min(1.0, abs(v_slope) / 0.00008) * 0.25 + min(1.0, abs(ch[i]) / ch_std) * 0.30
    strength = float(np.clip(strength, 0.0, 1.0))

    side = None
    reason = "no_entry"
    if long_pull:
        side = "UP"; reason = "vidya_up_chaikin_cross"
    elif short_pull:
        side = "DOWN"; reason = "vidya_dn_chaikin_cross"

    return {"side": side, "strength": strength, "fair_up": fair, "reason": reason,
            "price": c, "detail": f"vidya={v[i]:.0f} arU={au[i]:.0f} arD={ad[i]:.0f} chaik={ch[i]:.0f}"}


# ============================================================================
# STRATEGY 5: VORTEX_RVI_OBV
# ============================================================================

def signal_vortex_rvi_obv(close, high, low, vol, open_=None) -> Optional[dict]:
    if len(close) < 45:
        return None

    if open_ is None:
        open_ = np.empty_like(close)
        open_[0] = close[0]
        open_[1:] = close[:-1]

    vip, vim = calc_vortex(high, low, close, 14)
    rv, rvs = calc_rvi(open_, high, low, close, 10)
    a = calc_atr(high, low, close, 14)
    ob = calc_obv(close, vol)
    obs = calc_lin_slope(np.nan_to_num(ob, nan=0.0), 30)

    i = len(close) - 1
    c = close[i]
    vals = [c, vip[i], vim[i], vip[i-1], vim[i-1], rv[i], rvs[i], obs[i], a[i]]
    if any(not np.isfinite(x) for x in vals):
        return None

    atrp = a[i] / max(1e-9, c)
    if atrp < 0.0009 or atrp > 0.04:
        return None

    # Vortex cross
    cross_up = (vip[i-1] <= vim[i-1]) and (vip[i] > vim[i])
    cross_dn = (vip[i-1] >= vim[i-1]) and (vip[i] < vim[i])

    sep = abs(vip[i] - vim[i])
    if sep < 0.03:
        return None

    # RVI momentum confirm
    mom_up = rv[i] >= rvs[i]
    mom_dn = rv[i] <= rvs[i]

    # OBV slope participation
    obv_up = obs[i] > 0
    obv_dn = obs[i] < 0

    # Fair prob
    vol_pen = float(np.clip((atrp - 0.02) * 1.2, 0.0, 0.08))
    p = 0.50
    p += float(np.clip((vip[i] - vim[i]) * 0.10, -0.14, 0.14))
    p += float(np.clip((rv[i] - rvs[i]) * 0.30, -0.10, 0.10))
    p += float(np.clip(obs[i] / (abs(obs[i]) + 1e-9) * 0.03, -0.03, 0.03))
    p -= vol_pen
    fair = float(np.clip(p, 0.05, 0.95))

    strength = min(1.0, sep / 0.10) * 0.55 + min(1.0, abs(rv[i] - rvs[i]) / 0.15) * 0.30 + 0.15
    strength = float(np.clip(strength, 0.0, 1.0))

    side = None
    reason = "no_entry"
    if cross_up and mom_up and obv_up:
        side = "UP"; reason = "vortex_cross_up_rvi_obv"
    elif cross_dn and mom_dn and obv_dn:
        side = "DOWN"; reason = "vortex_cross_dn_rvi_obv"

    return {"side": side, "strength": strength, "fair_up": fair, "reason": reason,
            "price": c, "detail": f"vi+={vip[i]:.3f} vi-={vim[i]:.3f} rvi={rv[i]:.3f} obvSlope={obs[i]:.0f}"}


# ============================================================================
# STRATEGY 6: FIB_CONFLUENCE_BOUNCE (core signal, used by FIB_EMA_CONFIRM)
# ============================================================================

def signal_fib_confluence(close, high, low, open_, end_idx):
    """0.786 touch -> 0.618 higher-low pivot -> bullish/bearish confirm.
    Returns (side, conf) where side is 'LONG'/'SHORT' or None."""
    if end_idx < 55:
        return None, 0
    c, h, l, o = close[:end_idx], high[:end_idx], low[:end_idx], open_[:end_idx]
    n = len(c)
    fc_pass, fc_side, strong_confirm = False, None, False
    bat_val, ba_val, sr_val, sh_val = 0, 0, 0.0, 0.0

    # --- LONG ---
    for swing_lb in [30, 50]:
        if fc_pass or n < swing_lb + 5:
            break
        wh, wl = h[-(swing_lb+1):-1], l[-(swing_lb+1):-1]
        sh, sl_v = float(np.max(wh)), float(np.min(wl))
        sr = sh - sl_v
        if sr <= 0 or sh <= 0 or (sr / sh) < 0.0015:
            continue
        f786, f618, f500 = sh - 0.786*sr, sh - 0.618*sr, sh - 0.500*sr
        tol = sh * 0.0015
        touched, tk = False, -1
        for k in range(min(n-2, n-2), max(0, n-16)-1, -1):
            if l[k] <= f786+tol and l[k] >= f786-tol*2 and c[k] > f786:
                touched, tk = True, k; break
        if not touched:
            for k in range(min(n-2, n-2), max(0, n-16)-1, -1):
                if l[k] <= f786+tol and c[k] > f618:
                    touched, tk = True, k; break
        if not touched:
            continue
        bat = n - 1 - tk
        if bat < 2:
            continue
        rl = l[tk+1:]
        lowest = float(np.min(rl)) if len(rl) > 0 else 0
        if not (lowest >= f618-tol or (lowest >= f786-tol and lowest <= f618+tol)):
            continue
        if (c[-1] > o[-1]) and (c[-1] > f618):
            fc_side, fc_pass = 'LONG', True
            strong_confirm = c[-1] > f500
            bat_val, sr_val, sh_val = bat, sr, sh
            break

    # --- SHORT ---
    if not fc_pass:
        for swing_lb in [30, 50]:
            if fc_pass or n < swing_lb + 5:
                break
            wh, wl = h[-(swing_lb+1):-1], l[-(swing_lb+1):-1]
            sh, sl_v = float(np.max(wh)), float(np.min(wl))
            sr = sh - sl_v
            if sr <= 0 or sl_v <= 0 or (sr / sl_v) < 0.0015:
                continue
            f786, f618 = sl_v + 0.786*sr, sl_v + 0.618*sr
            tol = sl_v * 0.0015
            touched, tk = False, -1
            for k in range(min(n-2, n-2), max(0, n-16)-1, -1):
                if h[k] >= f786-tol and h[k] <= f786+tol*2 and c[k] < f786:
                    touched, tk = True, k; break
            if not touched:
                for k in range(min(n-2, n-2), max(0, n-16)-1, -1):
                    if h[k] >= f786-tol and c[k] < f618:
                        touched, tk = True, k; break
            if not touched:
                continue
            ba = n - 1 - tk
            if ba < 2:
                continue
            rh = h[tk+1:]
            highest = float(np.max(rh)) if len(rh) > 0 else float('inf')
            if not (highest <= f618+tol or (highest <= f786+tol and highest >= f618-tol)):
                continue
            if (c[-1] < o[-1]) and (c[-1] < f618):
                fc_side, fc_pass = 'SHORT', True
                ba_val, sr_val, sh_val = ba, sr, sh
                break

    if fc_pass:
        atr_arr = calc_atr(h, l, c, 14)
        atr_pct = (atr_arr[-1] / c[-1] * 100) if c[-1] > 0 and np.isfinite(atr_arr[-1]) else 0
        if atr_pct < 0.15 or atr_pct > 8.0:
            fc_pass = False
    if fc_pass and fc_side:
        conf = 0.75
        if fc_side == 'LONG' and strong_confirm: conf += 0.05
        if sh_val > 0 and sr_val / sh_val <= 0.005: conf += 0.03
        if fc_side == 'LONG' and bat_val >= 4: conf += 0.04
        elif fc_side == 'SHORT' and ba_val >= 4: conf += 0.04
        return fc_side, min(0.92, conf)
    return None, 0


# ============================================================================
# ML OPTIMIZER (auto-retrain after 30+ trades)
# ============================================================================

class MLOptimizer:
    """Lightweight ML gate: records features, trains LogisticRegression, gates entries."""

    def __init__(self, features_file: Path):
        self.features_file = features_file
        self.samples = []  # list of {features: {...}, won: bool}
        self.model = None
        self.feature_keys = ["conf", "ema_gap", "rsi", "atr_pct", "hour_utc", "edge", "strength"]
        self.min_samples = 30
        self._load()

    def _load(self):
        if self.features_file.exists():
            try:
                data = json.loads(self.features_file.read_text())
                self.samples = data.get("samples", [])
                if len(self.samples) >= self.min_samples:
                    self._train()
            except Exception:
                pass

    def _save(self):
        try:
            self.features_file.write_text(json.dumps({"samples": self.samples[-500:]}, indent=1))
        except Exception:
            pass

    def record(self, features: dict, won: bool):
        self.samples.append({"features": features, "won": won})
        if len(self.samples) % 20 == 0 and len(self.samples) >= self.min_samples:
            self._train()
        self._save()

    def _train(self):
        try:
            from sklearn.linear_model import LogisticRegression
            X, y = [], []
            for s in self.samples:
                row = [float(s["features"].get(k, 0.0)) for k in self.feature_keys]
                if any(not np.isfinite(v) for v in row):
                    continue
                X.append(row)
                y.append(1 if s["won"] else 0)
            if len(X) < self.min_samples or len(set(y)) < 2:
                return
            self.model = LogisticRegression(max_iter=500, C=0.5)
            self.model.fit(X, y)
            acc = self.model.score(X, y)
            print(f"[ML] Retrained on {len(X)} samples | train acc: {acc:.1%}")
        except ImportError:
            pass
        except Exception as e:
            print(f"[ML] Train error: {e}")

    def predict_win_prob(self, features: dict) -> float:
        if self.model is None:
            return 0.5
        try:
            row = [[float(features.get(k, 0.0)) for k in self.feature_keys]]
            return float(self.model.predict_proba(row)[0][1])
        except Exception:
            return 0.5

    def should_enter(self, features: dict, min_prob: float = 0.48) -> Tuple[bool, float]:
        p = self.predict_win_prob(features)
        return p >= min_prob, p

    def size_multiplier(self, features: dict) -> float:
        """Scale sizing 0.5x-1.5x based on ML confidence."""
        p = self.predict_win_prob(features)
        if p <= 0.45:
            return 0.5
        elif p >= 0.60:
            return 1.5
        else:
            return 0.5 + (p - 0.45) / 0.15 * 1.0


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

STRATEGIES = {
    "EMA_STOCHRSI":  {"fn": signal_ema_stochrsi, "min_edge": 0.015, "tp": 0.020, "sl": 0.015, "base": 3.0, "max": 5.0},
    "FRAMA_CMO":     {"fn": signal_frama_cmo,     "min_edge": 0.010, "tp": 0.020, "sl": 0.015, "base": 3.0, "max": 8.0},
    "ICHI_ADXR_SMI": {"fn": signal_ichi_adxr_smi, "min_edge": 0.010, "tp": 0.018, "sl": 0.014, "base": 3.0, "max": 8.0},
    "VIDYA_AROON":   {"fn": signal_vidya_aroon,    "min_edge": 0.010, "tp": 0.018, "sl": 0.014, "base": 3.0, "max": 8.0},
    "VORTEX_RVI":    {"fn": signal_vortex_rvi_obv, "min_edge": 0.010, "tp": 0.018, "sl": 0.014, "base": 3.0, "max": 8.0},
}


# ============================================================================
# UNIFIED TRADER
# ============================================================================

class UnifiedMultiTFTrader:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.client = None
        self.trades = {}       # key -> trade dict
        self.resolved = []
        self.stats = {}        # per-strategy stats
        for name in STRATEGIES:
            self.stats[name] = {"wins": 0, "losses": 0, "pnl": 0.0}
        self.stats["FIB_EMA_CONFIRM"] = {"wins": 0, "losses": 0, "pnl": 0.0}
        self.stats["_total"] = {"wins": 0, "losses": 0, "pnl": 0.0}
        self.ml = MLOptimizer(ML_FEATURES_FILE)

        if not paper:
            try:
                from py_clob_client.client import ClobClient
                host = "https://clob.polymarket.com"
                key = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
                funder = os.environ.get("POLYMARKET_PROXY_ADDRESS", "")
                self.client = ClobClient(host, key=key, chain_id=137,
                                         signature_type=1, funder=funder)
                print("[LIVE] CLOB client initialized")
            except Exception as e:
                print(f"[WARN] CLOB init failed: {e} -- falling back to paper")
                self.paper = True

        self._load()

    def _load(self):
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.resolved = data.get("resolved", [])
                self.trades = data.get("active", {})
                saved_stats = data.get("stats", {})
                for name in list(STRATEGIES.keys()) + ["FIB_EMA_CONFIRM"]:
                    if name in saved_stats:
                        self.stats[name] = saved_stats[name]
                if "_total" in saved_stats:
                    self.stats["_total"] = saved_stats["_total"]
                t = self.stats["_total"]
                total = t["wins"] + t["losses"]
                wr = t["wins"] / total * 100 if total > 0 else 0
                print(f"[LOAD] {total} resolved ({t['wins']}W/{t['losses']}L {wr:.0f}%WR) | "
                      f"PnL: ${t['pnl']:+.2f} | {len(self.trades)} active")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        data = {
            "stats": self.stats,
            "resolved": self.resolved,
            "active": self.trades,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            RESULTS_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    # ========================================================================
    # DATA FETCHING
    # ========================================================================

    def fetch_candles(self) -> list:
        url = "https://api.binance.com/api/v3/klines"
        for attempt in range(3):
            try:
                r = httpx.get(url, params={"symbol": "BTCUSDT", "interval": "15m", "limit": 200}, timeout=15)
                r.raise_for_status()
                return [{"time": int(b[0]), "open": float(b[1]), "high": float(b[2]),
                         "low": float(b[3]), "close": float(b[4]), "volume": float(b[5])} for b in r.json()]
            except Exception as e:
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                print(f"[BINANCE] Error: {e}")
                return []

    def discover_15m_markets(self) -> list:
        markets = []
        try:
            r = httpx.get("https://gamma-api.polymarket.com/events",
                          params={"tag_slug": "15M", "active": "true", "closed": "false", "limit": 50},
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if r.status_code != 200:
                return markets
            now = datetime.now(timezone.utc)
            for event in r.json():
                title = event.get("title", "").lower()
                if "bitcoin" not in title and "btc" not in title:
                    continue
                for m in event.get("markets", []):
                    if m.get("closed", True):
                        continue
                    end_date = m.get("endDate", "")
                    if end_date:
                        try:
                            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                            if end_dt < now:
                                continue
                            m["_time_left"] = (end_dt - now).total_seconds() / 60
                            m["_tte_seconds"] = int((end_dt - now).total_seconds())
                        except Exception:
                            continue
                    if not m.get("question"):
                        m["question"] = event.get("title", "")
                    markets.append(m)
        except Exception as e:
            print(f"[API] Error: {e}")
        return markets

    def _fetch_binance(self, interval: str, limit: int = 200) -> list:
        url = "https://api.binance.com/api/v3/klines"
        for attempt in range(3):
            try:
                r = httpx.get(url, params={"symbol": "BTCUSDT", "interval": interval, "limit": limit}, timeout=15)
                r.raise_for_status()
                return [{"time": int(b[0]), "open": float(b[1]), "high": float(b[2]),
                         "low": float(b[3]), "close": float(b[4]), "volume": float(b[5])} for b in r.json()]
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                return []

    def fetch_1m_candles(self) -> list:
        return self._fetch_binance("1m", 200)

    def fetch_5m_candles(self) -> list:
        return self._fetch_binance("5m", 200)

    def discover_5m_markets(self) -> list:
        markets = []
        try:
            r = httpx.get("https://gamma-api.polymarket.com/events",
                          params={"tag_slug": "5M", "active": "true", "closed": "false", "limit": 50},
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if r.status_code != 200:
                return markets
            now = datetime.now(timezone.utc)
            for event in r.json():
                title = event.get("title", "").lower()
                if "bitcoin" not in title and "btc" not in title:
                    continue
                for m in event.get("markets", []):
                    if m.get("closed", True):
                        continue
                    end_date = m.get("endDate", "")
                    if end_date:
                        try:
                            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                            if end_dt < now:
                                continue
                            m["_time_left"] = (end_dt - now).total_seconds() / 60
                            m["_tte_seconds"] = int((end_dt - now).total_seconds())
                        except Exception:
                            continue
                    if not m.get("question"):
                        m["question"] = event.get("title", "")
                    markets.append(m)
        except Exception as e:
            if "429" not in str(e):
                print(f"[API-5M] Error: {e}")
        return markets

    # ========================================================================
    # HELPERS
    # ========================================================================

    def get_prices(self, market: dict):
        outcomes = market.get("outcomes", [])
        prices = market.get("outcomePrices", [])
        if isinstance(outcomes, str): outcomes = json.loads(outcomes)
        if isinstance(prices, str): prices = json.loads(prices)
        up_price, down_price = None, None
        for i, o in enumerate(outcomes):
            if i < len(prices):
                p = float(prices[i])
                if str(o).lower() == "up": up_price = p
                elif str(o).lower() == "down": down_price = p
        return up_price, down_price

    def compute_edge(self, sig: dict, market: dict, cfg: dict):
        up_p, down_p = self.get_prices(market)
        if up_p is None or down_p is None:
            return None, None, 0.0, 0.0, 0
        tte = market.get("_tte_seconds", 0)
        if tte < 60:
            return None, None, 0.0, 0.0, 0

        fair_up = sig["fair_up"]
        side = sig["side"]
        min_edge = cfg["min_edge"]

        if side == "UP":
            entry = up_p + SPREAD_OFFSET
            if entry < MIN_ENTRY_PRICE or entry > MAX_ENTRY_PRICE:
                return None, None, 0.0, 0.0, 0
            edge = fair_up - entry
            return ("UP", entry, fair_up, edge, tte) if edge >= min_edge else (None, None, fair_up, edge, tte)
        elif side == "DOWN":
            entry = down_p + SPREAD_OFFSET
            if entry < MIN_ENTRY_PRICE or entry > MAX_ENTRY_PRICE:
                return None, None, 0.0, 0.0, 0
            edge = (1.0 - fair_up) - entry
            return ("DOWN", entry, 1.0 - fair_up, edge, tte) if edge >= min_edge else (None, None, 1.0 - fair_up, edge, tte)
        return None, None, 0.0, 0.0, 0

    # ========================================================================
    # RESOLVE
    # ========================================================================

    def resolve_trades(self):
        now = datetime.now(timezone.utc)
        for tid, trade in list(self.trades.items()):
            if trade.get("status") != "open":
                continue
            try:
                entry_dt = datetime.fromisoformat(trade["entry_time"])
                age_min = (now - entry_dt).total_seconds() / 60
            except Exception:
                age_min = 999
            if age_min < RESOLVE_AGE_MIN:
                continue

            nid = trade.get("market_numeric_id")
            if nid:
                try:
                    r = httpx.get(f"https://gamma-api.polymarket.com/markets/{nid}",
                                  headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                    if r.status_code == 200:
                        rm = r.json()
                        up_p, down_p = self.get_prices(rm)
                        if up_p is not None:
                            price = up_p if trade["side"] == "UP" else down_p
                            if price >= 0.95:
                                exit_val = trade["size_usd"] / trade["entry_price"]
                            elif price <= 0.05:
                                exit_val = 0
                            else:
                                exit_val = (trade["size_usd"] / trade["entry_price"]) * price

                            trade["exit_price"] = price
                            trade["exit_time"] = now.isoformat()
                            trade["pnl"] = round(exit_val - trade["size_usd"], 2)
                            trade["status"] = "closed"

                            won = trade["pnl"] > 0
                            strat = trade.get("strategy", "_total")
                            if strat in self.stats:
                                self.stats[strat]["wins" if won else "losses"] += 1
                                self.stats[strat]["pnl"] += trade["pnl"]
                            self.stats["_total"]["wins" if won else "losses"] += 1
                            self.stats["_total"]["pnl"] += trade["pnl"]

                            # ML: record features for learning
                            ml_feats = trade.get("ml_features", {})
                            if ml_feats:
                                self.ml.record(ml_feats, won)

                            self.resolved.append(trade)
                            del self.trades[tid]

                            t = self.stats["_total"]
                            total = t["wins"] + t["losses"]
                            wr = t["wins"] / total * 100 if total > 0 else 0
                            tag = "WIN" if won else "LOSS"
                            s = self.stats.get(strat, t)
                            sw = s["wins"] + s["losses"]
                            swr = s["wins"] / sw * 100 if sw > 0 else 0

                            print(f"[{tag}] {strat} {trade['side']} ${trade['pnl']:+.2f} | "
                                  f"entry=${trade['entry_price']:.2f} exit=${price:.2f} | "
                                  f"edge={trade.get('edge', 0):.1%} str={trade.get('strength', 0):.2f} | "
                                  f"{trade.get('reason', '?')} | "
                                  f"Strat: {s['wins']}W/{s['losses']}L {swr:.0f}%WR ${s['pnl']:+.2f} | "
                                  f"Total: {t['wins']}W/{t['losses']}L {wr:.0f}%WR ${t['pnl']:+.2f}")
                except Exception as e:
                    if "429" not in str(e):
                        print(f"[RESOLVE] API error: {e}")
            elif age_min > 25:
                strat = trade.get("strategy", "_total")
                trade["status"] = "closed"
                trade["pnl"] = -trade["size_usd"]
                if strat in self.stats:
                    self.stats[strat]["losses"] += 1
                    self.stats[strat]["pnl"] += trade["pnl"]
                self.stats["_total"]["losses"] += 1
                self.stats["_total"]["pnl"] += trade["pnl"]
                self.resolved.append(trade)
                del self.trades[tid]
                print(f"[LOSS] {strat} {trade['side']} ${trade['pnl']:+.2f} | aged out")

    # ========================================================================
    # ENTRY (runs all strategies)
    # ========================================================================

    def find_entries(self, candles: list, markets: list):
        now = datetime.now(timezone.utc)
        close = np.array([c["close"] for c in candles], dtype=float)
        open_ = np.array([c["open"] for c in candles], dtype=float)
        high = np.array([c["high"] for c in candles], dtype=float)
        low = np.array([c["low"] for c in candles], dtype=float)
        vol = np.array([c.get("volume", 0.0) for c in candles], dtype=float)

        for strat_name, cfg in STRATEGIES.items():
            # Count open for this strategy
            open_count = sum(1 for t in self.trades.values()
                             if t.get("status") == "open" and t.get("strategy") == strat_name)
            if open_count >= MAX_CONCURRENT_PER_STRAT:
                continue

            sig = cfg["fn"](close, high, low, vol, open_)
            if not sig or sig["side"] is None:
                continue

            for market in markets:
                if open_count >= MAX_CONCURRENT_PER_STRAT:
                    break
                time_left = market.get("_time_left", 99)
                if time_left < TIME_WINDOW_15M[0] or time_left > TIME_WINDOW_15M[1]:
                    continue

                cid = market.get("conditionId", "")
                question = market.get("question", "")
                nid = market.get("id")
                trade_key = f"{strat_name}_{cid}_{sig['side']}"
                if trade_key in self.trades:
                    continue

                side, entry_price, fair_px, edge, tte = self.compute_edge(sig, market, cfg)
                if not side:
                    continue

                # ML features for gating and recording
                hour_utc = datetime.now(timezone.utc).hour
                ml_feats = {
                    "conf": sig.get("strength", 0.5),
                    "ema_gap": 0.0,
                    "rsi": 50.0,
                    "atr_pct": 0.0,
                    "hour_utc": float(hour_utc),
                    "edge": float(edge),
                    "strength": sig.get("strength", 0.5),
                }
                # ML gate: skip if model says low probability
                should, ml_prob = self.ml.should_enter(ml_feats)
                if not should:
                    continue
                ml_mult = self.ml.size_multiplier(ml_feats)

                trade_size = cfg["base"] + sig["strength"] * (cfg["max"] - cfg["base"])
                trade_size = round(max(cfg["base"], min(cfg["max"], trade_size * ml_mult)), 2)

                self.trades[trade_key] = {
                    "side": side,
                    "entry_price": entry_price,
                    "size_usd": trade_size,
                    "entry_time": now.isoformat(),
                    "condition_id": cid,
                    "market_numeric_id": nid,
                    "title": question,
                    "strategy": strat_name,
                    "edge": round(edge, 4),
                    "fair_px": round(fair_px, 4),
                    "strength": round(sig["strength"], 3),
                    "reason": sig["reason"],
                    "btc_price": sig["price"],
                    "detail": sig.get("detail", ""),
                    "tte_seconds": tte,
                    "ml_features": ml_feats,
                    "ml_prob": round(ml_prob, 3),
                    "status": "open",
                    "pnl": 0.0,
                }
                open_count += 1

                mode = "LIVE" if not self.paper else "PAPER"
                print(f"[ENTRY] {strat_name} {side} ${entry_price:.2f} ${trade_size:.0f} | "
                      f"edge={edge:.1%} fair={fair_px:.2f} str={sig['strength']:.2f} | "
                      f"{sig['reason']} | {sig.get('detail', '')} | "
                      f"tte={tte}s | BTC=${sig['price']:,.0f} | [{mode}]")

    # ========================================================================
    # FIB_EMA_CONFIRM ENTRIES (5M markets, 1M signal + 5M EMA confirm)
    # ========================================================================

    def find_fib_entries(self, candles_1m: list, candles_5m: list, markets_5m: list):
        if not candles_1m or len(candles_1m) < 60 or not candles_5m or len(candles_5m) < 25:
            return
        if not markets_5m:
            return

        strat_name = "FIB_EMA_CONFIRM"
        cfg = FIB_5M_CONFIG
        open_count = sum(1 for t in self.trades.values()
                         if t.get("status") == "open" and t.get("strategy") == strat_name)
        if open_count >= MAX_CONCURRENT_PER_STRAT:
            return

        # Run fib signal on 1M candles
        c1m = np.array([c["close"] for c in candles_1m], dtype=float)
        h1m = np.array([c["high"] for c in candles_1m], dtype=float)
        l1m = np.array([c["low"] for c in candles_1m], dtype=float)
        o1m = np.array([c["open"] for c in candles_1m], dtype=float)

        fib_side, fib_conf = signal_fib_confluence(c1m, h1m, l1m, o1m, len(c1m))
        if fib_side is None:
            return

        # 5M EMA trend confirmation
        c5m = np.array([c["close"] for c in candles_5m], dtype=float)
        ema9 = calc_ema(c5m, 9)
        ema21 = calc_ema(c5m, 21)
        ema_gap = (float(ema9[-1]) - float(ema21[-1])) / max(1e-9, float(c5m[-1]))

        if fib_side == 'LONG' and ema9[-1] <= ema21[-1]:
            return
        if fib_side == 'SHORT' and ema9[-1] >= ema21[-1]:
            return

        # Build signal dict
        poly_side = "UP" if fib_side == "LONG" else "DOWN"
        rsi14 = calc_rsi(c1m, 14)
        rsi_val = float(rsi14[-1]) if np.isfinite(rsi14[-1]) else 50.0
        atr14 = calc_atr(h1m, l1m, c1m, 14)
        atr_pct = float(atr14[-1] / max(1e-9, c1m[-1])) if np.isfinite(atr14[-1]) else 0.0

        # Fair prob from EMA+RSI
        p = 0.50
        p += float(np.clip(ema_gap * 30, -0.12, 0.12))
        p += float(np.clip((rsi_val - 50.0) / 400.0, -0.06, 0.06))
        fair_up = float(np.clip(p, 0.05, 0.95))
        strength = float(np.clip((fib_conf - 0.70) / 0.22, 0.0, 1.0))

        sig = {
            "side": poly_side,
            "strength": strength,
            "fair_up": fair_up,
            "reason": f"fib_ema_{fib_side.lower()}",
            "price": float(c1m[-1]),
            "detail": f"conf={fib_conf:.2f} ema={'bull' if ema9[-1]>ema21[-1] else 'bear'} rsi={rsi_val:.0f}",
        }

        now = datetime.now(timezone.utc)
        hour_utc = now.hour
        for market in markets_5m:
            if open_count >= MAX_CONCURRENT_PER_STRAT:
                break
            time_left = market.get("_time_left", 99)
            if time_left < TIME_WINDOW_5M[0] or time_left > TIME_WINDOW_5M[1]:
                continue

            cid = market.get("conditionId", "")
            nid = market.get("id")
            trade_key = f"{strat_name}_{cid}_{poly_side}"
            if trade_key in self.trades:
                continue

            side_out, entry_price, fair_px, edge, tte = self.compute_edge(sig, market, cfg)
            if not side_out:
                continue

            ml_feats = {
                "conf": fib_conf,
                "ema_gap": ema_gap,
                "rsi": rsi_val,
                "atr_pct": atr_pct,
                "hour_utc": float(hour_utc),
                "edge": float(edge),
                "strength": strength,
            }
            should, ml_prob = self.ml.should_enter(ml_feats)
            if not should:
                continue
            ml_mult = self.ml.size_multiplier(ml_feats)

            trade_size = cfg["base"] + strength * (cfg["max"] - cfg["base"])
            trade_size = round(max(cfg["base"], min(cfg["max"], trade_size * ml_mult)), 2)

            self.trades[trade_key] = {
                "side": side_out,
                "entry_price": entry_price,
                "size_usd": trade_size,
                "entry_time": now.isoformat(),
                "condition_id": cid,
                "market_numeric_id": nid,
                "title": market.get("question", ""),
                "strategy": strat_name,
                "edge": round(edge, 4),
                "fair_px": round(fair_px, 4),
                "strength": round(strength, 3),
                "reason": sig["reason"],
                "btc_price": sig["price"],
                "detail": sig["detail"],
                "tte_seconds": tte,
                "ml_features": ml_feats,
                "ml_prob": round(ml_prob, 3),
                "status": "open",
                "pnl": 0.0,
            }
            open_count += 1
            mode = "LIVE" if not self.paper else "PAPER"
            print(f"[ENTRY] {strat_name} {side_out} ${entry_price:.2f} ${trade_size:.0f} | "
                  f"edge={edge:.1%} fair={fair_px:.2f} conf={fib_conf:.2f} | "
                  f"{sig['reason']} | {sig['detail']} | "
                  f"tte={tte}s | BTC=${sig['price']:,.0f} | ml={ml_prob:.2f} | [{mode}]")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    def run(self):
        mode = "PAPER" if self.paper else "LIVE"
        print("=" * 76)
        all_strats = list(STRATEGIES.keys()) + ["FIB_EMA_CONFIRM"]
        print(f"  UNIFIED MULTI-TF BTC STRATEGY TRADER V2.0 - {mode} MODE")
        print(f"  Strategies: {', '.join(all_strats)}")
        print(f"  Max {MAX_CONCURRENT_PER_STRAT}/strategy concurrent")
        print(f"  15M window: {TIME_WINDOW_15M[0]}-{TIME_WINDOW_15M[1]}min | 5M window: {TIME_WINDOW_5M[0]}-{TIME_WINDOW_5M[1]}min")
        print(f"  Spread offset: ${SPREAD_OFFSET} | Price range: ${MIN_ENTRY_PRICE}-${MAX_ENTRY_PRICE}")
        for name, cfg in STRATEGIES.items():
            print(f"    {name}: edge>={cfg['min_edge']:.1%} size=${cfg['base']}-${cfg['max']} [15M]")
        print(f"    FIB_EMA_CONFIRM: edge>={FIB_5M_CONFIG['min_edge']:.1%} size=${FIB_5M_CONFIG['base']}-${FIB_5M_CONFIG['max']} [5M]")
        print(f"  ML: {len(self.ml.samples)} samples | model={'active' if self.ml.model else 'warming up'}")
        print("=" * 76)

        if self.resolved:
            t = self.stats["_total"]
            total = t["wins"] + t["losses"]
            wr = t["wins"] / total * 100 if total > 0 else 0
            print(f"[RESUME] {t['wins']}W/{t['losses']}L {wr:.0f}%WR | PnL: ${t['pnl']:+.2f}")

        cycle = 0
        while True:
            try:
                cycle += 1
                # Multi-TF candle fetch
                candles_15m = self.fetch_candles()
                candles_1m = self.fetch_1m_candles()
                candles_5m = self.fetch_5m_candles()
                markets_15m = self.discover_15m_markets()
                markets_5m = self.discover_5m_markets()

                if not candles_15m:
                    print("[WARN] No 15M candles")
                    time.sleep(10)
                    continue

                self.resolve_trades()
                self.find_entries(candles_15m, markets_15m)
                self.find_fib_entries(candles_1m, candles_5m, markets_5m)

                if cycle % 10 == 0:
                    btc = candles_15m[-1]["close"]
                    t = self.stats["_total"]
                    total = t["wins"] + t["losses"]
                    wr = t["wins"] / total * 100 if total > 0 else 0

                    active_by_strat = {}
                    for tr in self.trades.values():
                        s = tr.get("strategy", "?")
                        active_by_strat[s] = active_by_strat.get(s, 0) + 1

                    strat_summary = []
                    for name in list(STRATEGIES.keys()) + ["FIB_EMA_CONFIRM"]:
                        s = self.stats.get(name, {"wins": 0, "losses": 0, "pnl": 0.0})
                        st = s["wins"] + s["losses"]
                        swr = s["wins"] / st * 100 if st > 0 else 0
                        act = active_by_strat.get(name, 0)
                        strat_summary.append(f"{name}:{s['wins']}W/{s['losses']}L/{swr:.0f}%/{act}a")

                    ml_tag = f"ml={len(self.ml.samples)}s" if self.ml.model else f"ml=warmup({len(self.ml.samples)})"
                    print(f"\n--- Cycle {cycle} | {mode} | BTC ${btc:,.0f} | "
                          f"Total: {t['wins']}W/{t['losses']}L {wr:.0f}%WR ${t['pnl']:+.2f} | "
                          f"Active: {len(self.trades)} | 15M:{len(markets_15m)} 5M:{len(markets_5m)} | {ml_tag} ---")
                    print(f"    {' | '.join(strat_summary)}")

                self._save()
                time.sleep(SCAN_INTERVAL)

            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Saving...")
                self._save()
                break
            except Exception as e:
                print(f"[ERROR] Cycle {cycle}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(15)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--paper", action="store_true")
    args = parser.parse_args()

    lock = acquire_pid_lock("15m_strategies")
    try:
        trader = UnifiedMultiTFTrader(paper=not args.live)
        trader.run()
    finally:
        release_pid_lock("15m_strategies")
