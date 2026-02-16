"""V2 Direction Prediction — Exhaustive Feature Search + ML.
50K+ markets from PolyData. 40+ features. XGBoost + walk-forward validation.
Goal: find ANY combination giving 55%+ directional accuracy.
"""
import pandas as pd
import numpy as np
import json, re, time, os, glob, sys, pickle, math
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import httpx
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────
# 1. LOAD MARKETS + PARSE (reuse from v1)
# ──────────────────────────────────────────────────────────
POLYDATA = r"C:\Users\Star\Documents\polymarket-assistant-main\PolyData"
CACHE_DIR = r"C:\Users\Star\.local\bin\star-polymarket\kline_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

ET = timezone(timedelta(hours=-5))

def parse_question(q):
    asset = "BTC" if "Bitcoin" in q else "ETH" if "Ethereum" in q else None
    if not asset:
        return None
    m = re.search(r'(\w+)\s+(\d{1,2}),\s*(\d{1,2}):(\d{2})(AM|PM)-(\d{1,2}):(\d{2})(AM|PM)\s+ET', q)
    if not m:
        return None
    month_name, day = m.group(1), int(m.group(2))
    h1, m1, ap1 = int(m.group(3)), int(m.group(4)), m.group(5)
    h2, m2, ap2 = int(m.group(6)), int(m.group(7)), m.group(8)
    if ap1 == 'PM' and h1 != 12: h1 += 12
    elif ap1 == 'AM' and h1 == 12: h1 = 0
    if ap2 == 'PM' and h2 != 12: h2 += 12
    elif ap2 == 'AM' and h2 == 12: h2 = 0
    dur = (h2 * 60 + m2) - (h1 * 60 + m1)
    if dur < 0: dur += 1440
    if dur not in (5, 15): return None
    months = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
              "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
    mon = months.get(month_name)
    if not mon: return None
    year = 2025 if mon >= 9 else 2026
    try:
        et_dt = datetime(year, mon, day, h1, m1, tzinfo=ET)
        utc_dt = et_dt.astimezone(timezone.utc).replace(tzinfo=None)
    except: return None
    return {"asset": asset, "candle_start_utc": utc_dt, "duration_min": dur}

def parse_outcome(row):
    prices = row.get('outcome_prices', '')
    outcomes = row.get('outcomes', '')
    if isinstance(prices, str):
        try: prices = json.loads(prices)
        except: return None
    if isinstance(outcomes, str):
        try: outcomes = json.loads(outcomes)
        except: return None
    if not prices or not outcomes or len(prices) < 2: return None
    try:
        up_idx = next(i for i, o in enumerate(outcomes) if o.lower() == 'up')
    except: return None
    if float(prices[up_idx]) > 0.9: return "UP"
    elif float(prices[1 - up_idx]) > 0.9: return "DOWN"
    return None

print("Loading markets...")
market_dir = os.path.join(POLYDATA, "data", "polymarket", "markets")
all_markets = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(market_dir, "*.parquet"))], ignore_index=True)
updown = all_markets[
    all_markets['question'].str.contains('Up or Down', case=False, na=False) &
    all_markets['question'].str.contains('Bitcoin|Ethereum', case=False, na=False) &
    all_markets['closed'].fillna(False)
].copy()

parsed = []
for _, row in updown.iterrows():
    info = parse_question(row['question'])
    if not info: continue
    outcome = parse_outcome(row)
    if not outcome: continue
    info['outcome'] = outcome
    info['question'] = row['question']
    parsed.append(info)

parsed.sort(key=lambda x: x['candle_start_utc'])
print(f"  {len(parsed)} markets parsed ({parsed[0]['candle_start_utc']} to {parsed[-1]['candle_start_utc']})")

# ──────────────────────────────────────────────────────────
# 2. DOWNLOAD + CACHE KLINES
# ──────────────────────────────────────────────────────────
BINANCE_URL = "https://api.binance.com/api/v3/klines"

def download_and_cache(symbol, start_dt, end_dt):
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_1m.pkl")
    if os.path.exists(cache_file):
        print(f"  Loading {symbol} from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"  Downloading {symbol} klines...")
    cache = {}
    cursor = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    total = 0
    with httpx.Client(timeout=15) as client:
        while cursor < end_ms:
            try:
                r = client.get(BINANCE_URL, params={"symbol": symbol, "interval": "1m", "startTime": cursor, "limit": 1000})
                if r.status_code == 429:
                    time.sleep(60); continue
                if r.status_code != 200: break
                data = r.json()
                if not data: break
                for k in data:
                    ts = int(k[0]) // 60000
                    cache[ts] = [float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[9])]
                total += len(data)
                cursor = int(data[-1][0]) + 60000
                if total % 20000 == 0: print(f"    {total} klines...")
                time.sleep(0.08)
            except Exception as e:
                print(f"    Error: {e}"); time.sleep(2)
    print(f"  {symbol}: {total} klines cached")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    return cache

# Download both assets
btc_mkts = [m for m in parsed if m['asset'] == 'BTC']
eth_mkts = [m for m in parsed if m['asset'] == 'ETH']
pad = timedelta(hours=6)  # Extra lookback for longer indicators
btc_cache = download_and_cache("BTCUSDT", btc_mkts[0]['candle_start_utc'] - pad, btc_mkts[-1]['candle_start_utc'] + timedelta(hours=1))
eth_cache = download_and_cache("ETHUSDT", eth_mkts[0]['candle_start_utc'] - pad, eth_mkts[-1]['candle_start_utc'] + timedelta(hours=1))
KLINES = {"BTC": btc_cache, "ETH": eth_cache}

def get_klines(asset, dt, lookback=100):
    """Get `lookback` 1m OHLCV bars ending just before dt."""
    cache = KLINES.get(asset, {})
    end_ts = int(dt.timestamp()) // 60
    result = []
    for i in range(lookback, 0, -1):
        ts = end_ts - i
        if ts in cache:
            result.append(cache[ts])  # [O, H, L, C, V, taker_buy_vol]
        else:
            result.append(None)
    # Filter None gaps (small gaps OK, big gaps = skip)
    clean = [r for r in result if r is not None]
    return clean


# ──────────────────────────────────────────────────────────
# 3. MASSIVE FEATURE ENGINE (40+ features)
# ──────────────────────────────────────────────────────────
def ema(values, period):
    if len(values) < period: return None
    alpha = 2.0 / (period + 1)
    r = values[0]
    for v in values[1:]:
        r = alpha * v + (1 - alpha) * r
    return r

def sma(values, period):
    if len(values) < period: return None
    return sum(values[-period:]) / period

def compute_rsi(closes, period=14):
    if len(closes) < period + 1: return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(0, d))
        losses.append(max(0, -d))
    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period
    if al < 1e-10: return 100.0
    return 100.0 - (100.0 / (1.0 + ag / al))

def stochastic(highs, lows, closes, k_period=14, d_period=3):
    if len(closes) < k_period: return None, None
    hh = max(highs[-k_period:])
    ll = min(lows[-k_period:])
    if hh == ll: return 50.0, 50.0
    k = (closes[-1] - ll) / (hh - ll) * 100
    return k, None  # D would need history

def williams_r(highs, lows, closes, period=14):
    if len(closes) < period: return None
    hh = max(highs[-period:])
    ll = min(lows[-period:])
    if hh == ll: return -50.0
    return (hh - closes[-1]) / (hh - ll) * -100

def cci(highs, lows, closes, period=20):
    if len(closes) < period: return None
    tp = [(highs[i]+lows[i]+closes[i])/3 for i in range(len(closes))]
    tp_sma = sum(tp[-period:]) / period
    md = sum(abs(tp[-period+i] - tp_sma) for i in range(period)) / period
    if md < 1e-10: return 0.0
    return (tp[-1] - tp_sma) / (0.015 * md)

def obv_slope(closes, volumes, period=10):
    if len(closes) < period + 1: return None
    obv = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]: obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]: obv.append(obv[-1] - volumes[i])
        else: obv.append(obv[-1])
    # Slope of last `period` OBV values
    recent = obv[-period:]
    x = list(range(period))
    mx, my = sum(x)/period, sum(recent)/period
    num = sum((x[i]-mx)*(recent[i]-my) for i in range(period))
    den = sum((x[i]-mx)**2 for i in range(period))
    if abs(den) < 1e-10: return 0.0
    slope = num / den
    # Normalize by average volume
    avg_vol = sum(volumes[-period:]) / period
    if avg_vol < 1e-10: return 0.0
    return slope / avg_vol

def cmf(highs, lows, closes, volumes, period=20):
    if len(closes) < period: return None
    mfv_sum = 0.0
    vol_sum = 0.0
    for i in range(-period, 0):
        hl = highs[i] - lows[i]
        if hl < 1e-10: continue
        mfm = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl
        mfv_sum += mfm * volumes[i]
        vol_sum += volumes[i]
    if vol_sum < 1e-10: return 0.0
    return mfv_sum / vol_sum

def linear_reg_slope(values, period=20):
    if len(values) < period: return None
    y = values[-period:]
    x = list(range(period))
    mx, my = sum(x)/period, sum(y)/period
    num = sum((x[i]-mx)*(y[i]-my) for i in range(period))
    den = sum((x[i]-mx)**2 for i in range(period))
    if abs(den) < 1e-10: return 0.0
    return num / den / my if my != 0 else 0.0  # Normalized

def donchian_position(highs, lows, closes, period=20):
    if len(closes) < period: return None
    hh = max(highs[-period:])
    ll = min(lows[-period:])
    if hh == ll: return 0.0
    return (closes[-1] - ll) / (hh - ll) * 2 - 1  # [-1, 1]

def keltner_position(closes, highs, lows, ema_period=20, atr_mult=2.0, atr_period=10):
    if len(closes) < max(ema_period, atr_period + 1): return None
    mid = ema(closes, ema_period)
    if mid is None: return None
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
           for i in range(1, len(closes))]
    if len(trs) < atr_period: return None
    atr = sum(trs[-atr_period:]) / atr_period
    if atr < 1e-10: return 0.0
    return (closes[-1] - mid) / (atr_mult * atr)

def ttm_squeeze(closes, highs, lows, bb_period=20, bb_mult=2.0, kc_period=20, kc_mult=1.5):
    """Returns (squeeze_on, momentum) where squeeze_on means BB inside KC."""
    if len(closes) < max(bb_period, kc_period + 1): return None, None
    # BB
    w = closes[-bb_period:]
    bb_mid = sum(w) / bb_period
    bb_std = (sum((x - bb_mid)**2 for x in w) / bb_period) ** 0.5
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std
    # KC
    kc_mid = ema(closes, kc_period)
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
           for i in range(1, len(closes))]
    atr = sum(trs[-kc_period:]) / kc_period if len(trs) >= kc_period else None
    if kc_mid is None or atr is None: return None, None
    kc_upper = kc_mid + kc_mult * atr
    kc_lower = kc_mid - kc_mult * atr
    squeeze = 1.0 if bb_lower > kc_lower and bb_upper < kc_upper else 0.0
    # Momentum = close - midline of (donchian_mid + ema)/2
    hh = max(highs[-kc_period:])
    ll = min(lows[-kc_period:])
    momentum = closes[-1] - ((hh + ll)/2 + kc_mid) / 2
    if kc_mid > 0:
        momentum /= kc_mid  # Normalize
    return squeeze, momentum

def supertrend_signal(closes, highs, lows, period=10, mult=3.0):
    if len(closes) < period + 1: return None
    trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
           for i in range(1, len(closes))]
    if len(trs) < period: return None
    atr = sum(trs[-period:]) / period
    hl2 = (highs[-1] + lows[-1]) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    if closes[-1] > upper: return 1.0
    elif closes[-1] < lower: return -1.0
    return 0.0

def hurst_exponent(closes, max_lag=20):
    """Estimate Hurst exponent. H > 0.5 = trending, H < 0.5 = mean reverting."""
    if len(closes) < max_lag + 2: return None
    lags = range(2, max_lag + 1)
    tau = []
    for lag in lags:
        diffs = [closes[i] - closes[i - lag] for i in range(lag, len(closes))]
        if not diffs: return None
        tau.append(np.std(diffs))
    if any(t <= 0 for t in tau): return None
    try:
        log_lags = [math.log(l) for l in lags]
        log_tau = [math.log(t) for t in tau]
        n = len(log_lags)
        mx = sum(log_lags) / n
        my = sum(log_tau) / n
        num = sum((log_lags[i]-mx)*(log_tau[i]-my) for i in range(n))
        den = sum((log_lags[i]-mx)**2 for i in range(n))
        if abs(den) < 1e-10: return 0.5
        return num / den
    except:
        return 0.5

def autocorrelation(closes, lag=1):
    if len(closes) < lag + 5: return None
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes)) if closes[i-1] > 0]
    if len(returns) < lag + 5: return None
    n = len(returns)
    mean = sum(returns) / n
    var = sum((r - mean)**2 for r in returns) / n
    if var < 1e-15: return 0.0
    cov = sum((returns[i] - mean) * (returns[i - lag] - mean) for i in range(lag, n)) / n
    return cov / var

def price_vs_round(price):
    """Distance from nearest round number (psychological level)."""
    # For BTC: round to nearest 1000, for ETH: nearest 100
    if price > 1000:
        round_level = round(price / 1000) * 1000
        return (price - round_level) / 1000
    else:
        round_level = round(price / 100) * 100
        return (price - round_level) / 100

def volume_surge(volumes, period=20):
    if len(volumes) < period + 1: return None
    avg = sum(volumes[-period-1:-1]) / period
    if avg < 1e-10: return 0.0
    return volumes[-1] / avg - 1.0  # How much above/below average

def candle_body_ratio(opens, highs, lows, closes):
    """Ratio of body to total range for last candle. Positive = bullish."""
    if not opens or not closes: return None
    body = closes[-1] - opens[-1]
    total = highs[-1] - lows[-1]
    if total < 1e-10: return 0.0
    return body / total

def consecutive_direction(closes, max_look=10):
    """Count consecutive up or down closes. Positive = consecutive ups."""
    if len(closes) < 3: return 0
    count = 0
    for i in range(len(closes)-1, 0, -1):
        if closes[i] > closes[i-1]:
            if count >= 0: count += 1
            else: break
        elif closes[i] < closes[i-1]:
            if count <= 0: count -= 1
            else: break
        if abs(count) >= max_look: break
    return count

def range_compression(highs, lows, short=5, long=20):
    """ATR ratio short/long. < 1 = compression (squeeze), > 1 = expansion."""
    if len(highs) < long + 1: return None
    trs_short = [highs[i]-lows[i] for i in range(-short, 0)]
    trs_long = [highs[i]-lows[i] for i in range(-long, 0)]
    avg_short = sum(trs_short) / short
    avg_long = sum(trs_long) / long
    if avg_long < 1e-10: return 1.0
    return avg_short / avg_long


def compute_all_features(bars, cross_bars=None):
    """Compute 40+ features from OHLCV bars.
    bars: list of [O, H, L, C, V, taker_buy_vol]
    cross_bars: bars from the other asset (for cross-asset signals)
    """
    if len(bars) < 30:
        return {}

    opens = [b[0] for b in bars]
    highs = [b[1] for b in bars]
    lows = [b[2] for b in bars]
    closes = [b[3] for b in bars]
    volumes = [b[4] for b in bars]
    taker_buy = [b[5] for b in bars]
    n = len(closes)
    f = {}

    # === TREND INDICATORS ===
    # EMA crosses (multiple timeframes)
    for fast, slow in [(5, 13), (9, 21), (12, 26)]:
        ef = ema(closes, fast)
        es = ema(closes, slow)
        if ef and es and closes[-1] > 0:
            f[f"ema_{fast}_{slow}"] = max(-1, min(1, (ef - es) / closes[-1] * 500))

    # SMA crosses
    for fast, slow in [(5, 20), (10, 30)]:
        sf = sma(closes, fast)
        ss = sma(closes, slow)
        if sf and ss and closes[-1] > 0:
            f[f"sma_{fast}_{slow}"] = max(-1, min(1, (sf - ss) / closes[-1] * 500))

    # Linear regression slope (multiple periods)
    for p in [10, 20, 50]:
        s = linear_reg_slope(closes, p)
        if s is not None:
            f[f"linreg_slope_{p}"] = max(-1, min(1, s * 1000))

    # Supertrend
    st = supertrend_signal(closes, highs, lows)
    if st is not None:
        f["supertrend"] = st

    # === MOMENTUM INDICATORS ===
    # RSI (multiple periods)
    for p in [7, 14, 21]:
        r = compute_rsi(closes, p)
        if r is not None:
            f[f"rsi_{p}"] = max(-1, min(1, (r - 50) / 50))

    # Stochastic
    k, _ = stochastic(highs, lows, closes, 14)
    if k is not None:
        f["stoch_k"] = max(-1, min(1, (k - 50) / 50))

    # Williams %R
    wr = williams_r(highs, lows, closes, 14)
    if wr is not None:
        f["williams_r"] = max(-1, min(1, (wr + 50) / 50))

    # CCI
    c = cci(highs, lows, closes, 20)
    if c is not None:
        f["cci_20"] = max(-1, min(1, c / 200))

    # Rate of change (multiple periods)
    for p in [5, 10, 20]:
        if n > p and closes[-p-1] > 0:
            roc = (closes[-1] - closes[-p-1]) / closes[-p-1]
            f[f"roc_{p}"] = max(-1, min(1, roc * 100))

    # Momentum z-score (multiple lookbacks)
    for lookback in [5, 10, 20]:
        if n >= lookback + 5:
            rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(max(1,n-lookback*2), n) if closes[i-1] > 0]
            if len(rets) >= 3:
                vol = np.std(rets)
                if vol > 1e-8 and closes[-lookback] > 0:
                    mom = (closes[-1] - closes[-lookback]) / closes[-lookback]
                    f[f"mom_z_{lookback}"] = max(-1, min(1, mom / vol / 3))

    # MACD histogram
    if n >= 26:
        fe = ema(closes, 12)
        se = ema(closes, 26)
        if fe and se and closes[-1] > 0:
            macd_line = fe - se
            f["macd_hist"] = max(-1, min(1, macd_line / closes[-1] * 5000))

    # === VOLATILITY INDICATORS ===
    # Bollinger Band position
    for p in [10, 20]:
        if n >= p:
            w = closes[-p:]
            mid = sum(w) / p
            std = (sum((x - mid)**2 for x in w) / p) ** 0.5
            if std > 0:
                f[f"bb_pos_{p}"] = max(-1, min(1, (closes[-1] - mid) / (2 * std)))

    # Keltner position
    kp = keltner_position(closes, highs, lows)
    if kp is not None:
        f["keltner_pos"] = max(-1, min(1, kp))

    # Donchian position
    for p in [10, 20]:
        dp = donchian_position(highs, lows, closes, p)
        if dp is not None:
            f[f"donchian_{p}"] = dp

    # TTM Squeeze
    sq, sq_mom = ttm_squeeze(closes, highs, lows)
    if sq is not None:
        f["squeeze_on"] = sq
        f["squeeze_mom"] = max(-1, min(1, sq_mom * 500)) if sq_mom else 0.0

    # Range compression
    rc = range_compression(highs, lows)
    if rc is not None:
        f["range_compress"] = max(-1, min(1, rc - 1.0))

    # ATR percent
    if n >= 15:
        trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, n)]
        if len(trs) >= 14:
            atr = sum(trs[-14:]) / 14
            f["atr_pct"] = atr / closes[-1] * 100 if closes[-1] > 0 else 0

    # === VOLUME INDICATORS ===
    # VWAP deviation
    pv = sum((highs[i]+lows[i]+closes[i])/3 * volumes[i] for i in range(n))
    vs = sum(volumes)
    if vs > 0:
        vwap = pv / vs
        if vwap > 0:
            f["vwap_dev"] = max(-1, min(1, (closes[-1] - vwap) / vwap * 200))

    # Taker flow (multiple lookbacks)
    for lb in [3, 5, 10, 20]:
        rv = sum(volumes[-lb:])
        rb = sum(taker_buy[-lb:])
        if rv > 0:
            f[f"taker_flow_{lb}"] = max(-1, min(1, (rb / rv - 0.5) * 2))

    # OBV slope
    obvs = obv_slope(closes, volumes, 10)
    if obvs is not None:
        f["obv_slope"] = max(-1, min(1, obvs * 10))

    # CMF
    cmf_val = cmf(highs, lows, closes, volumes, 20)
    if cmf_val is not None:
        f["cmf_20"] = max(-1, min(1, cmf_val))

    # Volume surge
    vsurge = volume_surge(volumes)
    if vsurge is not None:
        f["vol_surge"] = max(-1, min(1, vsurge / 3))

    # === ADX / TREND STRENGTH ===
    if n >= 15:
        pdm, mdm, trl = [], [], []
        for i in range(1, n):
            um = highs[i] - highs[i-1]
            dm = lows[i-1] - lows[i]
            pdm.append(um if um > dm and um > 0 else 0)
            mdm.append(dm if dm > um and dm > 0 else 0)
            trl.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
        if len(trl) >= 14:
            p = 14
            atr14 = sum(trl[:p])/p
            pds = sum(pdm[:p])/p
            mds = sum(mdm[:p])/p
            for i in range(p, len(trl)):
                atr14 = atr14 - atr14/p + trl[i]
                pds = pds - pds/p + pdm[i]
                mds = mds - mds/p + mdm[i]
            pdi = (pds/atr14*100) if atr14 > 0 else 0
            mdi = (mds/atr14*100) if atr14 > 0 else 0
            f["di_diff"] = max(-1, min(1, (pdi - mdi) / 50))
            dx = abs(pdi-mdi)/(pdi+mdi)*100 if (pdi+mdi) > 0 else 0
            f["adx"] = dx / 100  # 0-1 scale
            f["adx_direction"] = (1.0 if pdi > mdi else -1.0) * max(0, min(1, (dx - 15) / 30))

    # === STATISTICAL FEATURES ===
    # Hurst exponent
    h = hurst_exponent(closes)
    if h is not None:
        f["hurst"] = max(0, min(1, h))

    # Autocorrelation (multiple lags)
    for lag in [1, 2, 5]:
        ac = autocorrelation(closes, lag)
        if ac is not None:
            f[f"autocorr_{lag}"] = max(-1, min(1, ac))

    # === PRICE ACTION ===
    # Candle body ratio
    cbr = candle_body_ratio(opens, highs, lows, closes)
    if cbr is not None:
        f["candle_body"] = max(-1, min(1, cbr))

    # Consecutive direction
    cd = consecutive_direction(closes)
    f["consec_dir"] = max(-1, min(1, cd / 5))

    # Price vs round number
    pvr = price_vs_round(closes[-1])
    f["round_dist"] = max(-1, min(1, pvr))

    # Higher high / lower low
    if n >= 10:
        recent_hh = max(highs[-5:])
        prior_hh = max(highs[-10:-5])
        recent_ll = min(lows[-5:])
        prior_ll = min(lows[-10:-5])
        if prior_hh > 0:
            f["hh_ll"] = 1.0 if (recent_hh > prior_hh and recent_ll > prior_ll) else \
                        -1.0 if (recent_hh < prior_hh and recent_ll < prior_ll) else 0.0

    # === TIME FEATURES ===
    # These come from the market data, not klines

    # === CROSS-ASSET SIGNALS ===
    if cross_bars and len(cross_bars) >= 20:
        cross_closes = [b[3] for b in cross_bars]
        # Cross-asset momentum
        if cross_closes[-1] > 0 and cross_closes[-5] > 0:
            cross_mom = (cross_closes[-1] - cross_closes[-5]) / cross_closes[-5]
            f["cross_mom_5"] = max(-1, min(1, cross_mom * 100))
        if cross_closes[-1] > 0 and cross_closes[-10] > 0:
            cross_mom10 = (cross_closes[-1] - cross_closes[-10]) / cross_closes[-10]
            f["cross_mom_10"] = max(-1, min(1, cross_mom10 * 50))
        # Cross-asset RSI
        cross_rsi = compute_rsi(cross_closes, 14)
        if cross_rsi is not None:
            f["cross_rsi"] = max(-1, min(1, (cross_rsi - 50) / 50))
        # Cross taker flow
        cross_taker = [b[5] for b in cross_bars]
        cross_vols = [b[4] for b in cross_bars]
        rv = sum(cross_vols[-5:])
        rb = sum(cross_taker[-5:])
        if rv > 0:
            f["cross_taker_flow"] = max(-1, min(1, (rb / rv - 0.5) * 2))

    return f


# ──────────────────────────────────────────────────────────
# 4. COMPUTE FEATURES FOR ALL MARKETS
# ──────────────────────────────────────────────────────────
print(f"\nComputing features for {len(parsed)} markets...")
all_features = []
all_labels = []
all_meta = []

for i, mkt in enumerate(parsed):
    asset = mkt['asset']
    cross_asset = "ETH" if asset == "BTC" else "BTC"
    dt = mkt['candle_start_utc']

    bars = get_klines(asset, dt, lookback=100)
    cross_bars = get_klines(cross_asset, dt, lookback=30)

    if len(bars) < 30:
        continue

    feats = compute_all_features(bars, cross_bars)

    # Add time features
    et_hour = (dt.hour - 5) % 24  # Approximate ET hour
    feats["hour_sin"] = math.sin(2 * math.pi * dt.hour / 24)
    feats["hour_cos"] = math.cos(2 * math.pi * dt.hour / 24)
    feats["is_us_open"] = 1.0 if 14 <= dt.hour <= 20 else 0.0  # US market hours in UTC
    feats["is_asia"] = 1.0 if 0 <= dt.hour <= 8 else 0.0
    feats["is_europe"] = 1.0 if 7 <= dt.hour <= 15 else 0.0
    feats["day_of_week"] = dt.weekday() / 6.0  # 0=Mon, 1=Sun

    label = 1 if mkt['outcome'] == "UP" else 0

    all_features.append(feats)
    all_labels.append(label)
    all_meta.append({
        "asset": asset,
        "duration": mkt['duration_min'],
        "candle_start": dt.isoformat(),
        "outcome": mkt['outcome'],
    })

    if (i + 1) % 5000 == 0:
        print(f"  {i+1}/{len(parsed)} computed...")

print(f"  {len(all_features)} markets with features")

# ──────────────────────────────────────────────────────────
# 5. BUILD FEATURE MATRIX
# ──────────────────────────────────────────────────────────
# Get all feature names
all_feat_names = set()
for f in all_features:
    all_feat_names.update(f.keys())
feat_names = sorted(all_feat_names)
print(f"  {len(feat_names)} unique features")

# Build matrix
X = np.zeros((len(all_features), len(feat_names)))
for i, f in enumerate(all_features):
    for j, name in enumerate(feat_names):
        X[i, j] = f.get(name, 0.0)

y = np.array(all_labels)
print(f"  X shape: {X.shape}, y balance: {y.mean():.4f} (0.5 = balanced)")

# ──────────────────────────────────────────────────────────
# 6. INDIVIDUAL FEATURE ACCURACY
# ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"INDIVIDUAL FEATURE ACCURACY (sorted by accuracy)")
print(f"{'='*70}")

feat_results = []
for j, name in enumerate(feat_names):
    col = X[:, j]
    nonzero = np.abs(col) > 0.01
    if nonzero.sum() < 100:
        continue
    preds = (col > 0).astype(int)
    correct = (preds[nonzero] == y[nonzero]).sum()
    total = nonzero.sum()
    acc = correct / total * 100
    # Correlation
    corr = np.corrcoef(col, y.astype(float))[0, 1] if col.std() > 0 else 0
    feat_results.append((name, acc, total, corr))

feat_results.sort(key=lambda x: -x[1])
print(f"\nTop 20 features:")
for name, acc, cnt, corr in feat_results[:20]:
    marker = " ***" if acc > 52 else ""
    print(f"  {name:>20}: {acc:.2f}% ({cnt:,} trades) corr={corr:+.4f}{marker}")

print(f"\nBottom 10 features (contrarian signal?):")
for name, acc, cnt, corr in feat_results[-10:]:
    # If a feature is consistently WRONG, inverting it is an edge
    marker = " *** INVERT?" if acc < 48 else ""
    print(f"  {name:>20}: {acc:.2f}% ({cnt:,} trades) corr={corr:+.4f}{marker}")

# ──────────────────────────────────────────────────────────
# 7. WALK-FORWARD ML (the real test)
# ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"WALK-FORWARD ML BACKTEST")
print(f"{'='*70}")
print("Training on past data, predicting future data (no look-ahead bias)")

# Sort by time (already sorted)
# Walk-forward: train on first N, predict next batch, slide forward
TRAIN_SIZE = 10000
TEST_SIZE = 2000
STEP_SIZE = 2000

# Replace NaN/inf
X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

models = {
    "XGBoost": lambda: xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
        eval_metric='logloss', verbosity=0
    ),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    "LogReg": lambda: LogisticRegression(max_iter=1000, C=0.1),
    "GBM": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05),
}

for model_name, model_fn in models.items():
    all_preds = []
    all_true = []
    all_probs = []
    n_windows = 0

    start = 0
    while start + TRAIN_SIZE + TEST_SIZE <= len(X):
        train_end = start + TRAIN_SIZE
        test_end = train_end + TEST_SIZE

        X_train = X[start:train_end]
        y_train = y[start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_fn()
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        probs = model.predict_proba(X_test_s)[:, 1] if hasattr(model, 'predict_proba') else preds.astype(float)

        all_preds.extend(preds)
        all_true.extend(y_test)
        all_probs.extend(probs)
        n_windows += 1

        start += STEP_SIZE

    if all_preds:
        acc = accuracy_score(all_true, all_preds) * 100
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)
        all_probs = np.array(all_probs)

        # Accuracy at different confidence thresholds
        print(f"\n  {model_name}: {acc:.2f}% overall ({len(all_preds):,} predictions, {n_windows} windows)")

        # High-confidence subset
        for thresh in [0.52, 0.55, 0.58, 0.60, 0.65]:
            high_conf = (all_probs > thresh) | (all_probs < (1 - thresh))
            if high_conf.sum() > 50:
                hc_acc = accuracy_score(all_true[high_conf], all_preds[high_conf]) * 100
                print(f"    conf>{thresh:.0%}: {hc_acc:.2f}% ({high_conf.sum():,} trades, {high_conf.sum()/len(all_preds)*100:.0f}% of total)")

# ──────────────────────────────────────────────────────────
# 8. XGBOOST FEATURE IMPORTANCE
# ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"XGBOOST FEATURE IMPORTANCE (last training window)")
print(f"{'='*70}")

# Train on last window for feature importance
last_start = max(0, len(X) - TRAIN_SIZE - TEST_SIZE)
X_train_last = X[last_start:last_start + TRAIN_SIZE]
y_train_last = y[last_start:last_start + TRAIN_SIZE]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_last)

xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
    eval_metric='logloss', verbosity=0
)
xgb_model.fit(X_train_s, y_train_last)

importances = xgb_model.feature_importances_
imp_sorted = sorted(zip(feat_names, importances), key=lambda x: -x[1])
print("\nTop 20 most important features (XGBoost gain):")
for name, imp in imp_sorted[:20]:
    bar = "#" * int(imp * 200)
    print(f"  {name:>20}: {imp:.4f} {bar}")

# ──────────────────────────────────────────────────────────
# 9. SUBSET ANALYSIS (best conditions)
# ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"CONDITIONAL ACCURACY (searching for profitable subsets)")
print(f"{'='*70}")

# Does accuracy vary by volatility regime?
atr_idx = feat_names.index("atr_pct") if "atr_pct" in feat_names else None
if atr_idx is not None:
    for label, lo, hi in [("Low vol", 0, 0.1), ("Med vol", 0.1, 0.3), ("High vol", 0.3, 10)]:
        mask = (X[:, atr_idx] >= lo) & (X[:, atr_idx] < hi)
        if mask.sum() > 100:
            subset_acc = y[mask].mean()  # UP ratio
            print(f"  {label}: {mask.sum():,} trades, UP ratio={subset_acc:.4f}")

# Squeeze conditions
sq_idx = feat_names.index("squeeze_on") if "squeeze_on" in feat_names else None
if sq_idx is not None:
    for val, label in [(1.0, "Squeeze ON"), (0.0, "Squeeze OFF")]:
        mask = X[:, sq_idx] == val
        if mask.sum() > 100:
            # When squeeze is on, does momentum direction predict?
            sm_idx = feat_names.index("squeeze_mom") if "squeeze_mom" in feat_names else None
            if sm_idx is not None:
                col = X[mask, sm_idx]
                preds = (col > 0).astype(int)
                nonzero = np.abs(col) > 0.01
                if nonzero.sum() > 50:
                    acc = (preds[nonzero] == y[mask][nonzero]).mean() * 100
                    print(f"  {label} + momentum direction: {acc:.2f}% ({nonzero.sum():,} trades)")

# Hurst exponent regimes
hurst_idx = feat_names.index("hurst") if "hurst" in feat_names else None
if hurst_idx is not None:
    for label, lo, hi in [("Mean-revert (H<0.4)", 0, 0.4), ("Random (0.4-0.6)", 0.4, 0.6), ("Trending (H>0.6)", 0.6, 1.1)]:
        mask = (X[:, hurst_idx] >= lo) & (X[:, hurst_idx] < hi)
        if mask.sum() > 100:
            # In trending regime, does momentum predict?
            mom_idx = feat_names.index("mom_z_5") if "mom_z_5" in feat_names else None
            if mom_idx is not None:
                col = X[mask, mom_idx]
                preds = (col > 0).astype(int)
                nonzero = np.abs(col) > 0.01
                if nonzero.sum() > 50:
                    acc = (preds[nonzero] == y[mask][nonzero]).mean() * 100
                    print(f"  {label} + momentum: {acc:.2f}% ({nonzero.sum():,} trades)")

# Autocorrelation-conditional
ac_idx = feat_names.index("autocorr_1") if "autocorr_1" in feat_names else None
if ac_idx is not None:
    for label, lo, hi in [("Neg autocorr (mean-rev)", -1, -0.1), ("No autocorr", -0.1, 0.1), ("Pos autocorr (trend)", 0.1, 1)]:
        mask = (X[:, ac_idx] >= lo) & (X[:, ac_idx] < hi)
        if mask.sum() > 100:
            mom_idx = feat_names.index("mom_z_5") if "mom_z_5" in feat_names else None
            if mom_idx is not None:
                col = X[mask, mom_idx]
                preds = (col > 0).astype(int)
                nonzero = np.abs(col) > 0.01
                if nonzero.sum() > 50:
                    acc = (preds[nonzero] == y[mask][nonzero]).mean() * 100
                    print(f"  {label} + momentum: {acc:.2f}% ({nonzero.sum():,} trades)")

# Cross-asset lead-lag
cross_mom_idx = feat_names.index("cross_mom_5") if "cross_mom_5" in feat_names else None
if cross_mom_idx is not None:
    col = X[:, cross_mom_idx]
    preds = (col > 0).astype(int)
    nonzero = np.abs(col) > 0.01
    if nonzero.sum() > 100:
        acc = (preds[nonzero] == y[nonzero]).mean() * 100
        print(f"  Cross-asset momentum (5m): {acc:.2f}% ({nonzero.sum():,} trades)")

# High-confidence ensemble: top 3 features must agree
print(f"\nMulti-feature agreement (top features must agree):")
# Get top 5 features by accuracy
top5_names = [f[0] for f in feat_results[:5]]
top5_indices = [feat_names.index(n) for n in top5_names if n in feat_names]

for min_agree in [5, 4, 3, 2]:
    correct = 0
    total = 0
    for i in range(len(X)):
        votes_up = sum(1 for idx in top5_indices if X[i, idx] > 0.05)
        votes_dn = sum(1 for idx in top5_indices if X[i, idx] < -0.05)
        if max(votes_up, votes_dn) >= min_agree:
            pred = 1 if votes_up > votes_dn else 0
            if pred == y[i]:
                correct += 1
            total += 1
    if total > 50:
        print(f"  Top-5 agree {min_agree}+: {correct}/{total} = {correct/total*100:.2f}%")

# ──────────────────────────────────────────────────────────
# 10. FINAL SUMMARY
# ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"FINAL SUMMARY")
print(f"{'='*70}")
print(f"Markets tested: {len(all_features):,}")
print(f"Features computed: {len(feat_names)}")
print(f"Date range: {parsed[0]['candle_start_utc']} to {parsed[-1]['candle_start_utc']}")
print(f"\nBest individual features (>51% = potentially interesting):")
for name, acc, cnt, corr in feat_results:
    if acc > 51:
        print(f"  {name}: {acc:.2f}%")
print(f"\nDone.")
