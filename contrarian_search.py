"""Contrarian Signal Search — find features so BAD at predicting that inverting them gives an edge."""
import pickle, json, re, os, math, glob
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

# Load cached klines
CACHE_DIR = r"C:\Users\Star\.local\bin\star-polymarket\kline_cache"
with open(os.path.join(CACHE_DIR, "BTCUSDT_1m.pkl"), 'rb') as f:
    btc_cache = pickle.load(f)
with open(os.path.join(CACHE_DIR, "ETHUSDT_1m.pkl"), 'rb') as f:
    eth_cache = pickle.load(f)
KLINES = {"BTC": btc_cache, "ETH": eth_cache}

def get_klines(asset, dt, lookback=100):
    cache = KLINES.get(asset, {})
    end_ts = int(dt.timestamp()) // 60
    return [cache[end_ts - i] for i in range(lookback, 0, -1) if (end_ts - i) in cache]

# Load markets
import pandas as pd
POLYDATA = r"C:\Users\Star\Documents\polymarket-assistant-main\PolyData"
ET = timezone(timedelta(hours=-5))

def parse_question(q):
    asset = "BTC" if "Bitcoin" in q else "ETH" if "Ethereum" in q else None
    if not asset: return None
    m = re.search(r'(\w+)\s+(\d{1,2}),\s*(\d{1,2}):(\d{2})(AM|PM)-(\d{1,2}):(\d{2})(AM|PM)\s+ET', q)
    if not m: return None
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
    prices, outcomes = row.get('outcome_prices', ''), row.get('outcomes', '')
    if isinstance(prices, str):
        try: prices = json.loads(prices)
        except: return None
    if isinstance(outcomes, str):
        try: outcomes = json.loads(outcomes)
        except: return None
    if not prices or not outcomes or len(prices) < 2: return None
    try: up_idx = next(i for i, o in enumerate(outcomes) if o.lower() == 'up')
    except: return None
    if float(prices[up_idx]) > 0.9: return "UP"
    elif float(prices[1 - up_idx]) > 0.9: return "DOWN"
    return None

print("Loading markets...")
market_dir = os.path.join(POLYDATA, "data", "polymarket", "markets")
all_markets = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(market_dir, "*.parquet"))], ignore_index=True)
updown = all_markets[all_markets['question'].str.contains('Up or Down', case=False, na=False) &
                     all_markets['question'].str.contains('Bitcoin|Ethereum', case=False, na=False) &
                     all_markets['closed'].fillna(False)].copy()

parsed = []
for _, row in updown.iterrows():
    info = parse_question(row['question'])
    if not info: continue
    outcome = parse_outcome(row)
    if not outcome: continue
    info['outcome'] = outcome
    parsed.append(info)
parsed.sort(key=lambda x: x['candle_start_utc'])
print(f"  {len(parsed)} markets loaded")

# Helper
def ema(values, period):
    if len(values) < period: return None
    alpha = 2.0 / (period + 1)
    r = values[0]
    for v in values[1:]: r = alpha * v + (1 - alpha) * r
    return r

def compute_rsi(closes, period=14):
    if len(closes) < period + 1: return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(0, d)); losses.append(max(0, -d))
    ag = sum(gains[-period:]) / period
    al = sum(losses[-period:]) / period
    if al < 1e-10: return 100.0
    return 100.0 - (100.0 / (1.0 + ag / al))

# ── Compute features for every market ──
print("Computing features...")
records = []  # list of (features_dict, is_up, asset, duration, hour_utc)

for i, mkt in enumerate(parsed):
    bars = get_klines(mkt['asset'], mkt['candle_start_utc'], 100)
    if len(bars) < 30: continue

    closes = [b[3] for b in bars]
    highs = [b[1] for b in bars]
    lows = [b[2] for b in bars]
    volumes = [b[4] for b in bars]
    taker_buy = [b[5] for b in bars]
    n = len(closes)
    is_up = 1 if mkt['outcome'] == 'UP' else 0
    f = {}

    # ── TREND (the worst predictors from V2) ──
    for fast, slow, scale in [(5, 13, 500), (9, 21, 500), (12, 26, 500)]:
        if n >= slow:
            ef = ema(closes, fast); es = ema(closes, slow)
            if ef and es and closes[-1] > 0:
                f[f"ema_{fast}_{slow}"] = np.clip((ef - es) / closes[-1] * scale, -1, 1)

    if n >= 26:
        e12 = ema(closes, 12); e26 = ema(closes, 26)
        if e12 and e26 and closes[-1] > 0:
            f["macd_hist"] = np.clip((e12 - e26) / closes[-1] * 5000, -1, 1)

    if n >= 20:
        s5, s20 = sum(closes[-5:])/5, sum(closes[-20:])/20
        if closes[-1] > 0:
            f["sma_5_20"] = np.clip((s5 - s20) / closes[-1] * 500, -1, 1)
        hh, ll = max(highs[-20:]), min(lows[-20:])
        if hh != ll:
            f["donchian_20"] = (closes[-1] - ll) / (hh - ll) * 2 - 1
        # BB
        w = closes[-20:]
        mid = sum(w) / 20
        std = (sum((x - mid)**2 for x in w) / 20) ** 0.5
        if std > 0:
            f["bb_pos_20"] = np.clip((closes[-1] - mid) / (2 * std), -1, 1)

    # ── MOMENTUM ──
    for p in [5, 10, 20]:
        if n > p and closes[-p-1] > 0:
            f[f"roc_{p}"] = np.clip((closes[-1] - closes[-p-1]) / closes[-p-1] * 100, -1, 1)

    for p in [7, 14, 21]:
        rsi = compute_rsi(closes, p)
        if rsi is not None:
            f[f"rsi_{p}"] = np.clip((rsi - 50) / 50, -1, 1)

    # Stochastic
    if n >= 14:
        hh14, ll14 = max(highs[-14:]), min(lows[-14:])
        if hh14 != ll14:
            f["stoch_k"] = np.clip(((closes[-1] - ll14) / (hh14 - ll14) - 0.5) * 2, -1, 1)

    # Williams %R
    if n >= 14:
        hh14, ll14 = max(highs[-14:]), min(lows[-14:])
        if hh14 != ll14:
            f["williams_r"] = np.clip(((hh14 - closes[-1]) / (hh14 - ll14) - 0.5) * -2, -1, 1)

    # CCI
    if n >= 20:
        tp = [(highs[j]+lows[j]+closes[j])/3 for j in range(n)]
        tp_sma = sum(tp[-20:])/20
        md = sum(abs(tp[-20+j] - tp_sma) for j in range(20)) / 20
        if md > 1e-10:
            f["cci_20"] = np.clip((tp[-1] - tp_sma) / (0.015 * md) / 200, -1, 1)

    # ── ADX ──
    if n >= 15:
        pdm, mdm, trl = [], [], []
        for j in range(1, n):
            um = highs[j] - highs[j-1]; dm = lows[j-1] - lows[j]
            pdm.append(um if um > dm and um > 0 else 0)
            mdm.append(dm if dm > um and dm > 0 else 0)
            trl.append(max(highs[j]-lows[j], abs(highs[j]-closes[j-1]), abs(lows[j]-closes[j-1])))
        if len(trl) >= 14:
            p = 14
            atr14 = sum(trl[:p])/p; pds = sum(pdm[:p])/p; mds = sum(mdm[:p])/p
            for j in range(p, len(trl)):
                atr14 = atr14 - atr14/p + trl[j]
                pds = pds - pds/p + pdm[j]
                mds = mds - mds/p + mdm[j]
            pdi = (pds/atr14*100) if atr14 > 0 else 0
            mdi = (mds/atr14*100) if atr14 > 0 else 0
            dx = abs(pdi-mdi)/(pdi+mdi)*100 if (pdi+mdi) > 0 else 0
            di_dir = 1.0 if pdi > mdi else -1.0
            f["adx_direction"] = di_dir * max(0, min(1, (dx - 15) / 30))
            f["adx_value"] = dx / 100

    # ── VOLUME ──
    pv = sum((highs[j]+lows[j]+closes[j])/3 * volumes[j] for j in range(n))
    vs = sum(volumes)
    if vs > 0:
        vwap = pv / vs
        if vwap > 0:
            f["vwap_dev"] = np.clip((closes[-1] - vwap) / vwap * 200, -1, 1)

    for lb in [3, 5, 10, 20]:
        rv = sum(volumes[-lb:]); rb = sum(taker_buy[-lb:])
        if rv > 0:
            f[f"taker_flow_{lb}"] = np.clip((rb / rv - 0.5) * 2, -1, 1)

    # OBV slope
    if n >= 11:
        obv = [0.0]
        for j in range(1, n):
            if closes[j] > closes[j-1]: obv.append(obv[-1] + volumes[j])
            elif closes[j] < closes[j-1]: obv.append(obv[-1] - volumes[j])
            else: obv.append(obv[-1])
        recent = obv[-10:]
        x = list(range(10)); mx = 4.5; my = sum(recent) / 10
        num = sum((x[j]-mx)*(recent[j]-my) for j in range(10))
        den = sum((x[j]-mx)**2 for j in range(10))
        if abs(den) > 1e-10:
            slope = num / den
            avg_vol = sum(volumes[-10:]) / 10
            if avg_vol > 1e-10:
                f["obv_slope"] = np.clip(slope / avg_vol * 10, -1, 1)

    # Linreg slopes
    for period in [10, 20, 50]:
        if n >= period:
            y = closes[-period:]
            x = list(range(period)); mx = sum(x)/period; my = sum(y)/period
            num = sum((x[j]-mx)*(y[j]-my) for j in range(period))
            den = sum((x[j]-mx)**2 for j in range(period))
            if abs(den) > 1e-10 and my != 0:
                f[f"linreg_{period}"] = np.clip((num/den/my) * 1000, -1, 1)

    # CMF
    if n >= 20:
        mfv_sum, vol_sum = 0.0, 0.0
        for j in range(-20, 0):
            hl = highs[j] - lows[j]
            if hl < 1e-10: continue
            mfm = ((closes[j] - lows[j]) - (highs[j] - closes[j])) / hl
            mfv_sum += mfm * volumes[j]
            vol_sum += volumes[j]
        if vol_sum > 1e-10:
            f["cmf_20"] = np.clip(mfv_sum / vol_sum, -1, 1)

    # Autocorrelation
    rets = [(closes[j]-closes[j-1])/closes[j-1] for j in range(1, n) if closes[j-1] > 0]
    if len(rets) >= 10:
        mean_r = sum(rets) / len(rets)
        var_r = sum((r - mean_r)**2 for r in rets) / len(rets)
        if var_r > 1e-15:
            for lag in [1, 2, 5]:
                if len(rets) > lag + 5:
                    cov = sum((rets[j]-mean_r)*(rets[j-lag]-mean_r) for j in range(lag, len(rets))) / len(rets)
                    f[f"autocorr_{lag}"] = np.clip(cov / var_r, -1, 1)

    # Consecutive direction
    count = 0
    for j in range(n-1, 0, -1):
        if closes[j] > closes[j-1]:
            if count >= 0: count += 1
            else: break
        elif closes[j] < closes[j-1]:
            if count <= 0: count -= 1
            else: break
        if abs(count) >= 10: break
    f["consec_dir"] = np.clip(count / 5, -1, 1)

    # Volume surge
    if n >= 21:
        avg_v = sum(volumes[-21:-1]) / 20
        if avg_v > 1e-10:
            f["vol_surge"] = np.clip(volumes[-1] / avg_v - 1.0, -1, 1)

    records.append((f, is_up, mkt['asset'], mkt['duration_min'], mkt['candle_start_utc'].hour))
    if (i + 1) % 10000 == 0:
        print(f"  {i+1}/{len(parsed)}...")

print(f"  {len(records)} records computed")

# ── Get all feature names ──
all_fnames = set()
for f, _, _, _, _ in records:
    all_fnames.update(f.keys())
all_fnames = sorted(all_fnames)
print(f"  {len(all_fnames)} features")

# ══════════════════════════════════════════════════════════
# CONTRARIAN ANALYSIS
# ══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"CONTRARIAN SIGNAL SEARCH")
print(f"{'='*70}")

results_table = []

for fname in all_fnames:
    vals = []
    labels = []
    for f, is_up, _, _, _ in records:
        if fname in f:
            vals.append(f[fname])
            labels.append(is_up)

    vals = np.array(vals)
    labels = np.array(labels)

    if len(vals) < 1000: continue

    # Normal: predict UP when feature > 0
    nonzero = np.abs(vals) > 0.01
    if nonzero.sum() < 100: continue
    normal_acc = ((vals[nonzero] > 0).astype(int) == labels[nonzero]).mean() * 100
    inverted_acc = ((vals[nonzero] < 0).astype(int) == labels[nonzero]).mean() * 100

    # EXTREME: |val| > 0.5
    extreme = np.abs(vals) > 0.5
    ext_inv_acc = 0
    n_ext = extreme.sum()
    if n_ext >= 50:
        ext_inv_acc = ((vals[extreme] < 0).astype(int) == labels[extreme]).mean() * 100

    # VERY EXTREME: |val| > 0.8
    vext = np.abs(vals) > 0.8
    vext_inv_acc = 0
    n_vext = int(vext.sum())
    if n_vext >= 30:
        vext_inv_acc = ((vals[vext] < 0).astype(int) == labels[vext]).mean() * 100

    # TOP QUARTILE: only strongest 25%
    abs_vals = np.abs(vals)
    q75 = np.percentile(abs_vals, 75)
    top_q = abs_vals >= q75
    tq_inv_acc = 0
    n_tq = int(top_q.sum())
    if n_tq >= 100:
        tq_inv_acc = ((vals[top_q] < 0).astype(int) == labels[top_q]).mean() * 100

    results_table.append({
        'name': fname,
        'normal': normal_acc,
        'inverted': inverted_acc,
        'n': int(nonzero.sum()),
        'ext_inv': ext_inv_acc,
        'n_ext': int(n_ext),
        'vext_inv': vext_inv_acc,
        'n_vext': n_vext,
        'tq_inv': tq_inv_acc,
        'n_tq': n_tq,
    })

# Sort by worst normal accuracy (= best inverted)
results_table.sort(key=lambda x: x['normal'])

print(f"\n{'Feature':>20} | {'Normal':>7} | {'INVERT':>7} | {'N':>7} | {'ExtINV':>7} | {'Nex':>6} | {'VExtINV':>8} | {'Nvx':>5} | {'TopQ INV':>9} | {'Ntq':>6}")
print("-" * 120)
for r in results_table:
    markers = ""
    if r['ext_inv'] > 52: markers += " *EXT*"
    if r['vext_inv'] > 54: markers += " **VEXT**"
    if r['tq_inv'] > 52: markers += " *TQ*"
    print(f"  {r['name']:>18} | {r['normal']:>6.2f}% | {r['inverted']:>6.2f}% | {r['n']:>6,} | "
          f"{r['ext_inv']:>6.2f}% | {r['n_ext']:>5,} | {r['vext_inv']:>7.2f}% | {r['n_vext']:>4,} | "
          f"{r['tq_inv']:>8.2f}% | {r['n_tq']:>5,}{markers}")

# ── STATISTICAL SIGNIFICANCE ──
print(f"\n{'='*70}")
print(f"SIGNIFICANCE TESTS (only features with inverted > 50.5%)")
print(f"{'='*70}")
for r in results_table:
    for mode, acc_key, n_key in [("all-inverted", "inverted", "n"),
                                   ("extreme-inv", "ext_inv", "n_ext"),
                                   ("vextreme-inv", "vext_inv", "n_vext"),
                                   ("topQ-inv", "tq_inv", "n_tq")]:
        acc = r[acc_key]
        n = r[n_key]
        if acc > 50.5 and n >= 50:
            p = acc / 100
            se = (p * (1-p) / n) ** 0.5
            ci_low = (p - 1.96 * se) * 100
            ci_high = (p + 1.96 * se) * 100
            sig = "SIGNIFICANT" if ci_low > 50 else "not sig"
            print(f"  {r['name']:>18} [{mode:>14}]: {acc:.2f}% on {n:,} trades | CI: [{ci_low:.1f}%, {ci_high:.1f}%] | {sig}")

# ── CONTRARIAN COMBOS ──
print(f"\n{'='*70}")
print(f"CONTRARIAN COMBOS (worst 6 features inverted, voting)")
print(f"{'='*70}")

worst6 = [r['name'] for r in results_table[:6]]
print(f"Using INVERTED: {worst6}")

for min_agree in [6, 5, 4, 3, 2]:
    for min_strength in [0.01, 0.1, 0.3, 0.5]:
        correct = 0
        total = 0
        for f, is_up, _, _, _ in records:
            inv_up = sum(1 for fn in worst6 if fn in f and f[fn] < -min_strength)
            inv_dn = sum(1 for fn in worst6 if fn in f and f[fn] > min_strength)
            if max(inv_up, inv_dn) >= min_agree:
                pred = 1 if inv_up > inv_dn else 0
                if pred == is_up: correct += 1
                total += 1
        if total >= 30:
            acc = correct / total * 100
            se = ((acc/100)*(1-acc/100)/total)**0.5
            ci_low = (acc/100 - 1.96*se) * 100
            sig = "SIG" if ci_low > 50 else ""
            print(f"  agree={min_agree} strength>{min_strength:.2f}: {correct}/{total} = {acc:.2f}% +/-{1.96*se*100:.1f}% {sig}")

# ── EXTREME CONTRARIAN ──
print(f"\n{'='*70}")
print(f"EXTREME CONTRARIAN: when ALL trend indicators strongly agree, bet OPPOSITE")
print(f"{'='*70}")

trend_feats = ["ema_5_13", "ema_9_21", "ema_12_26", "sma_5_20", "macd_hist", "adx_direction"]

for n_agree in [6, 5, 4]:
    for min_str in [0.1, 0.3, 0.5, 0.7]:
        correct = 0
        total = 0
        for f, is_up, _, _, _ in records:
            strong_up = sum(1 for fn in trend_feats if fn in f and f[fn] > min_str)
            strong_dn = sum(1 for fn in trend_feats if fn in f and f[fn] < -min_str)
            if strong_up >= n_agree:
                if is_up == 0: correct += 1  # Bet DOWN
                total += 1
            elif strong_dn >= n_agree:
                if is_up == 1: correct += 1  # Bet UP
                total += 1
        if total >= 20:
            acc = correct / total * 100
            se = ((acc/100)*(1-acc/100)/total)**0.5
            ci_low = (acc/100 - 1.96*se) * 100
            sig = "*** SIGNIFICANT ***" if ci_low > 50 else ""
            print(f"  {n_agree}+ trend agree, strength>{min_str}: {correct}/{total} = {acc:.2f}% {sig}")

# ── PER-ASSET CONTRARIAN ──
print(f"\n{'='*70}")
print(f"PER-ASSET: Do contrarian signals work better for BTC vs ETH?")
print(f"{'='*70}")
for asset in ["BTC", "ETH"]:
    print(f"\n  --- {asset} ---")
    for r in results_table[:10]:
        fname = r['name']
        vals = []; labels = []
        for f, is_up, a, _, _ in records:
            if a == asset and fname in f:
                vals.append(f[fname]); labels.append(is_up)
        vals = np.array(vals); labels = np.array(labels)
        nonzero = np.abs(vals) > 0.01
        if nonzero.sum() < 100: continue
        inv_acc = ((vals[nonzero] < 0).astype(int) == labels[nonzero]).mean() * 100
        extreme = np.abs(vals) > 0.5
        ext_inv = 0
        if extreme.sum() >= 30:
            ext_inv = ((vals[extreme] < 0).astype(int) == labels[extreme]).mean() * 100
        print(f"    {fname:>18}: inv={inv_acc:.2f}% ({nonzero.sum():,}T) | ext_inv={ext_inv:.2f}% ({extreme.sum():,}T)")

# ── PER-DURATION CONTRARIAN ──
print(f"\n{'='*70}")
print(f"PER-DURATION: 5min vs 15min contrarian")
print(f"{'='*70}")
for dur in [5, 15]:
    print(f"\n  --- {dur}min ---")
    for r in results_table[:10]:
        fname = r['name']
        vals = []; labels = []
        for f, is_up, _, d, _ in records:
            if d == dur and fname in f:
                vals.append(f[fname]); labels.append(is_up)
        vals = np.array(vals); labels = np.array(labels)
        nonzero = np.abs(vals) > 0.01
        if nonzero.sum() < 100: continue
        inv_acc = ((vals[nonzero] < 0).astype(int) == labels[nonzero]).mean() * 100
        extreme = np.abs(vals) > 0.5
        ext_inv = 0
        if extreme.sum() >= 30:
            ext_inv = ((vals[extreme] < 0).astype(int) == labels[extreme]).mean() * 100
        print(f"    {fname:>18}: inv={inv_acc:.2f}% ({nonzero.sum():,}T) | ext_inv={ext_inv:.2f}% ({extreme.sum():,}T)")

# ── HOURLY CONTRARIAN (do some hours show stronger mean reversion?) ──
print(f"\n{'='*70}")
print(f"PER-HOUR: Contrarian signal by UTC hour (top 3 worst features)")
print(f"{'='*70}")
worst3 = [r['name'] for r in results_table[:3]]
for hour in range(24):
    for fname in worst3:
        vals = []; labels = []
        for f, is_up, _, _, h in records:
            if h == hour and fname in f:
                vals.append(f[fname]); labels.append(is_up)
        vals = np.array(vals); labels = np.array(labels)
        nonzero = np.abs(vals) > 0.01
        if nonzero.sum() < 50: continue
        inv_acc = ((vals[nonzero] < 0).astype(int) == labels[nonzero]).mean() * 100
        if inv_acc > 53 or inv_acc < 47:
            se = ((inv_acc/100)*(1-inv_acc/100)/nonzero.sum())**0.5
            print(f"  UTC {hour:02d} {fname:>18}: inv={inv_acc:.2f}% +/-{1.96*se*100:.1f}% ({nonzero.sum()} trades)")

print(f"\n{'='*70}")
print(f"FINAL VERDICT")
print(f"{'='*70}")
