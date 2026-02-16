"""V4.8 Large-Scale Direction Skew Backtest using PolyData (50K+ markets).

Uses resolved BTC/ETH 5min/15min Up/Down markets from the PolyData parquet dataset
and Binance 1m klines to test the DirectionPredictor on thousands of candle outcomes.
"""
import pandas as pd
import numpy as np
import json, re, time, statistics, os, glob, sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import httpx

# ──────────────────────────────────────────────────────────
# 1. LOAD POLYDATA MARKETS
# ──────────────────────────────────────────────────────────
POLYDATA = r"C:\Users\Star\Documents\polymarket-assistant-main\PolyData"
MARKET_DIR = os.path.join(POLYDATA, "data", "polymarket", "markets")

print("Loading market parquet files...")
market_files = glob.glob(os.path.join(MARKET_DIR, "*.parquet"))
all_markets = pd.concat([pd.read_parquet(f) for f in market_files], ignore_index=True)
print(f"  Total markets: {len(all_markets)}")

# Filter for BTC/ETH Up or Down with 5min or 15min candles
updown = all_markets[
    all_markets['question'].str.contains('Up or Down', case=False, na=False) &
    all_markets['question'].str.contains('Bitcoin|Ethereum', case=False, na=False) &
    all_markets['closed'].fillna(False)
].copy()
print(f"  BTC/ETH Up or Down (closed): {len(updown)}")

# ──────────────────────────────────────────────────────────
# 2. PARSE CANDLE TIME + OUTCOME
# ──────────────────────────────────────────────────────────
ET = timezone(timedelta(hours=-5))

def parse_question(q: str) -> Optional[dict]:
    """Parse market question into asset, candle_start (UTC), duration_min, outcome."""
    asset = "BTC" if "Bitcoin" in q else "ETH" if "Ethereum" in q else None
    if not asset:
        return None

    # Try format: "Bitcoin Up or Down - February 16, 1:35PM-1:40PM ET"
    m = re.search(
        r'(\w+)\s+(\d{1,2}),\s*(\d{1,2}):(\d{2})(AM|PM)-(\d{1,2}):(\d{2})(AM|PM)\s+ET',
        q
    )
    if m:
        month_name = m.group(1)
        day = int(m.group(2))
        h1, m1, ap1 = int(m.group(3)), int(m.group(4)), m.group(5)
        h2, m2, ap2 = int(m.group(6)), int(m.group(7)), m.group(8)

        if ap1 == 'PM' and h1 != 12: h1 += 12
        elif ap1 == 'AM' and h1 == 12: h1 = 0
        if ap2 == 'PM' and h2 != 12: h2 += 12
        elif ap2 == 'AM' and h2 == 12: h2 = 0

        dur = (h2 * 60 + m2) - (h1 * 60 + m1)
        if dur < 0:
            dur += 1440  # cross midnight

        # Only 5min and 15min candles
        if dur not in (5, 15):
            return None

        # Determine year from month
        months = {"January": 1, "February": 2, "March": 3, "April": 4,
                  "May": 5, "June": 6, "July": 7, "August": 8,
                  "September": 9, "October": 10, "November": 11, "December": 12}
        mon = months.get(month_name)
        if not mon:
            return None

        # Polymarket crypto candles started ~Sept 2025
        year = 2025 if mon >= 9 else 2026

        try:
            et_dt = datetime(year, mon, day, h1, m1, tzinfo=ET)
            utc_dt = et_dt.astimezone(timezone.utc).replace(tzinfo=None)
        except (ValueError, OverflowError):
            return None

        return {"asset": asset, "candle_start_utc": utc_dt, "duration_min": dur}

    return None


def parse_outcome(row) -> Optional[str]:
    """Determine if UP or DOWN won from outcome_prices."""
    prices = row.get('outcome_prices', '')
    outcomes = row.get('outcomes', '')

    # Handle string format
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except:
            return None
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except:
            return None

    if not prices or not outcomes or len(prices) < 2 or len(outcomes) < 2:
        return None

    try:
        p0 = float(prices[0])
        p1 = float(prices[1])
    except:
        return None

    # Find which outcome is "Up"
    up_idx = None
    for i, o in enumerate(outcomes):
        if o.lower() == 'up':
            up_idx = i
            break

    if up_idx is None:
        return None

    down_idx = 1 - up_idx
    up_price = float(prices[up_idx])
    down_price = float(prices[down_idx])

    # Resolved: winning side has price ~1.0, losing side ~0.0
    if up_price > 0.9:
        return "UP"
    elif down_price > 0.9:
        return "DOWN"
    return None


# Parse all markets
parsed_markets = []
parse_fails = 0
outcome_fails = 0

for _, row in updown.iterrows():
    info = parse_question(row['question'])
    if not info:
        parse_fails += 1
        continue

    outcome = parse_outcome(row)
    if not outcome:
        outcome_fails += 1
        continue

    info['outcome'] = outcome
    info['question'] = row['question']
    info['market_id'] = row['id']
    parsed_markets.append(info)

print(f"  Parsed: {len(parsed_markets)} markets (parse_fail={parse_fails}, outcome_fail={outcome_fails})")

# Sort by candle_start
parsed_markets.sort(key=lambda x: x['candle_start_utc'])

# Show date range
if parsed_markets:
    print(f"  Date range: {parsed_markets[0]['candle_start_utc']} to {parsed_markets[-1]['candle_start_utc']}")

# Breakdown
btc5 = [m for m in parsed_markets if m['asset'] == 'BTC' and m['duration_min'] == 5]
btc15 = [m for m in parsed_markets if m['asset'] == 'BTC' and m['duration_min'] == 15]
eth5 = [m for m in parsed_markets if m['asset'] == 'ETH' and m['duration_min'] == 5]
eth15 = [m for m in parsed_markets if m['asset'] == 'ETH' and m['duration_min'] == 15]
print(f"  BTC 5min: {len(btc5)} | BTC 15min: {len(btc15)} | ETH 5min: {len(eth5)} | ETH 15min: {len(eth15)}")

# UP/DOWN balance
up_count = sum(1 for m in parsed_markets if m['outcome'] == 'UP')
print(f"  UP: {up_count} ({up_count/len(parsed_markets)*100:.1f}%) | DOWN: {len(parsed_markets)-up_count} ({(len(parsed_markets)-up_count)/len(parsed_markets)*100:.1f}%)")

# ──────────────────────────────────────────────────────────
# 3. DOWNLOAD BINANCE KLINES IN BULK
# ──────────────────────────────────────────────────────────
BINANCE_URL = "https://api.binance.com/api/v3/klines"
KLINE_CACHE = {}  # {symbol: {ts_minute: [O,H,L,C,V,taker_buy_vol]}}

def download_klines_bulk(symbol: str, start_dt: datetime, end_dt: datetime):
    """Download all 1m klines between start and end, store in KLINE_CACHE."""
    if symbol not in KLINE_CACHE:
        KLINE_CACHE[symbol] = {}

    cache = KLINE_CACHE[symbol]
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    cursor = start_ms
    total_fetched = 0

    with httpx.Client(timeout=15) as client:
        while cursor < end_ms:
            try:
                r = client.get(BINANCE_URL, params={
                    "symbol": symbol, "interval": "1m",
                    "startTime": cursor, "limit": 1000
                })
                if r.status_code == 429:
                    print("  Rate limited, sleeping 60s...")
                    time.sleep(60)
                    continue
                if r.status_code != 200:
                    print(f"  Binance error {r.status_code}: {r.text[:100]}")
                    break

                data = r.json()
                if not data:
                    break

                for k in data:
                    ts = int(k[0]) // 60000  # Minute-level key
                    cache[ts] = [
                        float(k[1]),  # open
                        float(k[2]),  # high
                        float(k[3]),  # low
                        float(k[4]),  # close
                        float(k[5]),  # volume
                        float(k[9]),  # taker buy volume
                    ]
                total_fetched += len(data)
                cursor = int(data[-1][0]) + 60000  # Next minute

                if total_fetched % 10000 == 0:
                    print(f"    {symbol}: {total_fetched} klines downloaded...")

                time.sleep(0.08)  # Rate limit (12 req/s)

            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")
                time.sleep(2)
                continue

    print(f"  {symbol}: {total_fetched} total klines cached ({len(cache)} unique minutes)")


def get_klines_window(symbol: str, candle_start_utc: datetime, lookback: int = 30) -> List[list]:
    """Get `lookback` 1m klines BEFORE candle_start from cache."""
    cache = KLINE_CACHE.get(symbol, {})
    if not cache:
        return []

    end_ts = int(candle_start_utc.timestamp()) // 60
    result = []
    for i in range(lookback, 0, -1):
        ts = end_ts - i
        if ts in cache:
            k = cache[ts]
            # Format: [open_time_ms, open, high, low, close, volume, ..., taker_buy_vol, ...]
            # We need to match the Binance kline format that compute_features expects
            # klines[i][4] = close, [2] = high, [3] = low, [5] = volume, [9] = taker_buy_vol
            open_ms = ts * 60000
            result.append([
                open_ms,     # [0] open time
                str(k[0]),   # [1] open
                str(k[1]),   # [2] high
                str(k[2]),   # [3] low
                str(k[3]),   # [4] close
                str(k[4]),   # [5] volume
                0, 0, 0,     # [6-8] unused
                str(k[5]),   # [9] taker buy volume
            ])
    return result


# Determine date ranges needed
btc_markets = [m for m in parsed_markets if m['asset'] == 'BTC']
eth_markets = [m for m in parsed_markets if m['asset'] == 'ETH']

if btc_markets:
    btc_start = btc_markets[0]['candle_start_utc'] - timedelta(hours=1)
    btc_end = btc_markets[-1]['candle_start_utc'] + timedelta(hours=1)
    print(f"\nDownloading BTCUSDT klines: {btc_start} to {btc_end}...")
    download_klines_bulk("BTCUSDT", btc_start, btc_end)

if eth_markets:
    eth_start = eth_markets[0]['candle_start_utc'] - timedelta(hours=1)
    eth_end = eth_markets[-1]['candle_start_utc'] + timedelta(hours=1)
    print(f"\nDownloading ETHUSDT klines: {eth_start} to {eth_end}...")
    download_klines_bulk("ETHUSDT", eth_start, eth_end)


# ──────────────────────────────────────────────────────────
# 4. DIRECTION PREDICTOR (same as run_maker.py V4.8)
# ──────────────────────────────────────────────────────────
class DirectionPredictor:
    def __init__(self, weights=None):
        self.weights = weights or {
            "ema_cross": 1.0, "rsi_zone": 1.0, "bb_position": 1.0,
            "momentum_z": 1.0, "vwap_dev": 1.0, "taker_flow": 1.0,
            "macd_hist": 1.0, "adx_trend": 1.0,
        }

    @staticmethod
    def _ema(values, period):
        if not values or len(values) < period:
            return None
        alpha = 2.0 / (period + 1)
        result = values[0]
        for v in values[1:]:
            result = alpha * v + (1 - alpha) * result
        return result

    @staticmethod
    def _compute_rsi(closes, period=14):
        if len(closes) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i - 1]
            gains.append(max(0, d))
            losses.append(max(0, -d))
        if len(gains) < period:
            return None
        ag = sum(gains[-period:]) / period
        al = sum(losses[-period:]) / period
        if al < 1e-10:
            return 100.0
        return 100.0 - (100.0 / (1.0 + ag / al))

    def compute_features(self, klines):
        if not klines or len(klines) < 15:
            return {}
        bars = klines[:-1]
        closes = [float(k[4]) for k in bars]
        highs = [float(k[2]) for k in bars]
        lows = [float(k[3]) for k in bars]
        volumes = [float(k[5]) for k in bars]
        taker_buy = [float(k[9]) for k in bars]
        n = len(closes)
        f = {}

        ema9 = self._ema(closes, 9)
        ema21 = self._ema(closes, 21)
        if ema9 and ema21 and closes[-1] > 0:
            f["ema_cross"] = max(-1, min(1, (ema9 - ema21) / closes[-1] * 500))

        rsi = self._compute_rsi(closes, 14)
        if rsi is not None:
            f["rsi_zone"] = max(-1, min(1, (rsi - 50) / 50))

        if n >= 20:
            w = closes[-20:]
            mid = sum(w) / 20
            std = (sum((x - mid) ** 2 for x in w) / 20) ** 0.5
            if std > 0:
                f["bb_position"] = max(-1, min(1, (closes[-1] - mid) / (2 * std)))

        if n >= 12:
            rets = [(closes[i] - closes[i - 1]) / closes[i - 1]
                    for i in range(max(1, n - 12), n) if closes[i - 1] > 0]
            if len(rets) >= 2:
                vol = statistics.stdev(rets)
                if vol > 1e-8 and n >= 6 and closes[-6] > 0:
                    mom = (closes[-1] - closes[-6]) / closes[-6]
                    f["momentum_z"] = max(-1, min(1, mom / vol / 3))

        pv = sum((highs[i] + lows[i] + closes[i]) / 3 * volumes[i] for i in range(n))
        vs = sum(volumes)
        if vs > 0:
            vwap = pv / vs
            if vwap > 0:
                f["vwap_dev"] = max(-1, min(1, (closes[-1] - vwap) / vwap * 200))

        rv = sum(volumes[-5:])
        rb = sum(taker_buy[-5:])
        if rv > 0:
            f["taker_flow"] = max(-1, min(1, (rb / rv - 0.5) * 2))

        if n >= 26:
            fe = self._ema(closes, 12)
            se = self._ema(closes, 26)
            if fe and se and closes[-1] > 0:
                f["macd_hist"] = max(-1, min(1, (fe - se) / closes[-1] * 5000))

        if n >= 15:
            pdm, mdm, trl = [], [], []
            for i in range(1, n):
                um = highs[i] - highs[i - 1]
                dm = lows[i - 1] - lows[i]
                pdm.append(um if um > dm and um > 0 else 0)
                mdm.append(dm if dm > um and dm > 0 else 0)
                trl.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                               abs(lows[i] - closes[i - 1])))
            if len(trl) >= 14:
                p = 14
                atr14 = sum(trl[:p]) / p
                pds = sum(pdm[:p]) / p
                mds = sum(mdm[:p]) / p
                for i in range(p, len(trl)):
                    atr14 = atr14 - atr14 / p + trl[i]
                    pds = pds - pds / p + pdm[i]
                    mds = mds - mds / p + mdm[i]
                pdi = (pds / atr14 * 100) if atr14 > 0 else 0
                mdi = (mds / atr14 * 100) if atr14 > 0 else 0
                dx = abs(pdi - mdi) / (pdi + mdi) * 100 if (pdi + mdi) > 0 else 0
                di_dir = 1.0 if pdi > mdi else -1.0
                adx_mult = max(0, min(1, (dx - 15) / 30))
                f["adx_trend"] = di_dir * adx_mult
        return f

    def predict_direction(self, features):
        if not features:
            return (0.0, {})
        contribs = {}
        raw = 0.0
        wsum = 0.0
        for name, w in self.weights.items():
            v = features.get(name, 0.0)
            c = v * w
            contribs[name] = round(c, 4)
            raw += c
            wsum += abs(w)
        norm = raw / wsum if wsum > 0 else 0.0
        adx_v = features.get("adx_trend", 0.0)
        strength = min(1.0, abs(adx_v) + 0.3)
        conf = max(-1, min(1, norm * strength))
        return (conf, contribs)

    @staticmethod
    def confidence_to_skew(conf, max_skew=0.60, min_side=0.35):
        hr = max_skew - 0.5
        up = 0.5 + conf * hr
        up = max(min_side, min(max_skew, up))
        return (up, 1.0 - up)


# ──────────────────────────────────────────────────────────
# 5. RUN BACKTEST
# ──────────────────────────────────────────────────────────
predictor = DirectionPredictor()
results = []
feature_outcomes = defaultdict(list)
skipped_no_klines = 0

print(f"\nRunning backtest on {len(parsed_markets)} markets...")
for i, mkt in enumerate(parsed_markets):
    symbol = "BTCUSDT" if mkt['asset'] == 'BTC' else "ETHUSDT"
    klines = get_klines_window(symbol, mkt['candle_start_utc'], lookback=30)

    if len(klines) < 15:
        skipped_no_klines += 1
        continue

    features = predictor.compute_features(klines)
    confidence, contribs = predictor.predict_direction(features)

    outcome = mkt['outcome']
    predicted_dir = "UP" if confidence > 0 else "DOWN" if confidence < 0 else "NEUTRAL"
    correct = (confidence > 0 and outcome == "UP") or (confidence < 0 and outcome == "DOWN")

    for fname, fval in features.items():
        correct_sign = 1.0 if outcome == "UP" else -1.0
        feature_outcomes[fname].append((fval, correct_sign))

    # Compute shadow PnL at various skew levels
    # Assume typical prices: UP=$0.48, DOWN=$0.47, 10 shares each side
    up_price = 0.48
    down_price = 0.47
    total_shares = 10
    shadow_pnl = {}
    for skew in [0.50, 0.55, 0.60, 0.65, 0.70]:
        up_r, dn_r = predictor.confidence_to_skew(confidence, max_skew=skew)
        up_sh = total_shares * up_r
        dn_sh = total_shares * dn_r
        if outcome == "UP":
            pnl_hyp = up_sh * (1.0 - up_price) - dn_sh * down_price
        else:
            pnl_hyp = dn_sh * (1.0 - down_price) - up_sh * up_price
        shadow_pnl[f"skew_{int(skew * 100)}"] = round(pnl_hyp, 6)

    results.append({
        "asset": mkt['asset'],
        "duration": mkt['duration_min'],
        "outcome": outcome,
        "confidence": confidence,
        "predicted_dir": predicted_dir,
        "correct": correct,
        "shadow_pnl": shadow_pnl,
        "features": features,
        "candle_start": mkt['candle_start_utc'].isoformat(),
    })

    if (i + 1) % 5000 == 0:
        correct_so_far = sum(1 for r in results if r['correct'])
        print(f"  Progress: {i+1}/{len(parsed_markets)} | {len(results)} analyzed | "
              f"accuracy: {correct_so_far/len(results)*100:.1f}%")

print(f"\nSkipped (no klines): {skipped_no_klines}")

# ──────────────────────────────────────────────────────────
# 6. ANALYSIS
# ──────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"POLYDATA BACKTEST RESULTS: {len(results)} markets analyzed")
print(f"{'=' * 70}")

# Overall accuracy
non_neutral = [r for r in results if r['confidence'] != 0]
correct_cnt = sum(1 for r in non_neutral if r['correct'])
total_nn = len(non_neutral)
if total_nn:
    print(f"\nOverall Direction Accuracy: {correct_cnt}/{total_nn} = {correct_cnt / total_nn * 100:.2f}%")
    print(f"  (50% = random, need >55% for edge)")

# By asset
print(f"\nBy Asset:")
for asset in ['BTC', 'ETH']:
    subset = [r for r in non_neutral if r['asset'] == asset]
    if subset:
        c = sum(1 for r in subset if r['correct'])
        print(f"  {asset}: {c}/{len(subset)} = {c / len(subset) * 100:.2f}%")

# By duration
print(f"\nBy Duration:")
for dur in [5, 15]:
    subset = [r for r in non_neutral if r['duration'] == dur]
    if subset:
        c = sum(1 for r in subset if r['correct'])
        print(f"  {dur}min: {c}/{len(subset)} = {c / len(subset) * 100:.2f}%")

# By asset + duration
print(f"\nBy Asset + Duration:")
for asset in ['BTC', 'ETH']:
    for dur in [5, 15]:
        subset = [r for r in non_neutral if r['asset'] == asset and r['duration'] == dur]
        if subset:
            c = sum(1 for r in subset if r['correct'])
            print(f"  {asset} {dur}min: {c}/{len(subset)} = {c / len(subset) * 100:.2f}%")

# By confidence bucket
print(f"\nAccuracy by confidence level:")
for lo, hi, label in [(0, 0.05, "< 5%"), (0.05, 0.10, "5-10%"),
                       (0.10, 0.20, "10-20%"), (0.20, 0.35, "20-35%"),
                       (0.35, 0.50, "35-50%"), (0.50, 0.75, "50-75%"),
                       (0.75, 1.01, "75%+")]:
    subset = [r for r in results if lo <= abs(r['confidence']) < hi]
    if subset:
        c = sum(1 for r in subset if r['correct'])
        print(f"  |conf| {label:>8}: {c}/{len(subset)} = {c / len(subset) * 100:.1f}%  ({len(subset)} trades)")

# Shadow PnL comparison
print(f"\n{'=' * 70}")
print(f"SHADOW PnL COMPARISON (per 10 shares, $0.48/$0.47)")
print(f"{'=' * 70}")
for key in ["skew_50", "skew_55", "skew_60", "skew_65", "skew_70"]:
    total_shadow = sum(r['shadow_pnl'].get(key, 0) for r in results)
    wins = sum(1 for r in results if r['shadow_pnl'].get(key, 0) > 0)
    avg = total_shadow / len(results) if results else 0
    wr = wins / len(results) * 100 if results else 0
    base = sum(r['shadow_pnl'].get("skew_50", 0) for r in results)
    delta = total_shadow - base
    delta_str = f"  delta: ${delta:+.2f}" if key != "skew_50" else " (baseline 50/50)"
    print(f"  {key:>8}: {wr:.1f}%WR, ${total_shadow:+.2f} total (${avg:+.4f}/T){delta_str}")

# Per-feature correlation analysis
print(f"\n{'=' * 70}")
print(f"FEATURE ANALYSIS")
print(f"{'=' * 70}")
feat_scores = []
for fname in sorted(feature_outcomes.keys()):
    pairs = feature_outcomes[fname]
    if len(pairs) < 100:
        continue
    corrs = [val * sign for val, sign in pairs]
    avg_corr = sum(corrs) / len(corrs)
    correct_when_used = sum(1 for v, s in pairs if (v > 0 and s > 0) or (v < 0 and s < 0))
    total_nonzero = sum(1 for v, _ in pairs if abs(v) > 0.01)
    acc = correct_when_used / total_nonzero * 100 if total_nonzero > 0 else 0
    feat_scores.append((fname, avg_corr, acc, total_nonzero))
    print(f"  {fname:>15}: corr={avg_corr:+.6f} | accuracy={acc:.2f}% ({total_nonzero:,}T)")

# Feature agreement analysis
print(f"\nFeature agreement (how many agree on direction):")
for thresh in [7, 6, 5, 4, 3]:
    subset_correct = 0
    subset_total = 0
    for r in results:
        f = r['features']
        up_votes = sum(1 for v in f.values() if v > 0.05)
        dn_votes = sum(1 for v in f.values() if v < -0.05)
        agreement = max(up_votes, dn_votes)
        if agreement >= thresh:
            majority = "UP" if up_votes > dn_votes else "DOWN"
            if majority == r['outcome']:
                subset_correct += 1
            subset_total += 1
    if subset_total > 0:
        print(f"  {thresh}+ features agree: {subset_correct}/{subset_total} = "
              f"{subset_correct / subset_total * 100:.2f}% ({subset_total:,} trades)")

# Best single features
print(f"\nBest individual features (accuracy on non-zero predictions):")
for fname, corr, acc, cnt in sorted(feat_scores, key=lambda x: -x[2]):
    direction_str = "bullish bias" if corr > 0 else "bearish bias" if corr < 0 else "neutral"
    print(f"  {fname:>15}: {acc:.2f}% accuracy ({cnt:,}T) | corr={corr:+.6f} ({direction_str})")

# Time-of-day analysis
print(f"\nAccuracy by UTC hour:")
hour_stats = defaultdict(lambda: [0, 0])
for r in non_neutral:
    dt = datetime.fromisoformat(r['candle_start'])
    h = dt.hour
    hour_stats[h][0] += 1 if r['correct'] else 0
    hour_stats[h][1] += 1
for h in sorted(hour_stats.keys()):
    c, t = hour_stats[h]
    pct = c / t * 100 if t > 0 else 0
    bar = "#" * int(pct / 2)
    print(f"  UTC {h:02d}: {c}/{t} = {pct:.1f}% {bar}")

# Month analysis
print(f"\nAccuracy by month:")
month_stats = defaultdict(lambda: [0, 0])
for r in non_neutral:
    dt = datetime.fromisoformat(r['candle_start'])
    key = f"{dt.year}-{dt.month:02d}"
    month_stats[key][0] += 1 if r['correct'] else 0
    month_stats[key][1] += 1
for m in sorted(month_stats.keys()):
    c, t = month_stats[m]
    pct = c / t * 100 if t > 0 else 0
    print(f"  {m}: {c}/{t} = {pct:.1f}% ({t} trades)")

# ──────────────────────────────────────────────────────────
# 7. WEIGHT OPTIMIZATION (grid search)
# ──────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"WEIGHT OPTIMIZATION")
print(f"{'=' * 70}")

# Try different weight combinations
best_acc = 0
best_weights = None
best_pnl = -999

# Start with correlation-derived weights
base_weights = {}
for fname, corr, acc, cnt in feat_scores:
    # Weight based on correlation direction and magnitude
    base_weights[fname] = corr * 100  # Scale up for search

# Test: what if we just use each feature as the sole predictor?
print(f"\nSingle-feature predictor accuracy:")
for fname in sorted(feature_outcomes.keys()):
    pairs = feature_outcomes[fname]
    if len(pairs) < 100:
        continue
    correct = sum(1 for v, s in pairs if (v > 0 and s > 0) or (v < 0 and s < 0))
    wrong = sum(1 for v, s in pairs if (v > 0 and s < 0) or (v < 0 and s > 0))
    neutral = sum(1 for v, _ in pairs if abs(v) <= 0.01)
    total_pred = correct + wrong
    acc_solo = correct / total_pred * 100 if total_pred > 0 else 0
    print(f"  {fname:>15}: {acc_solo:.2f}% ({total_pred:,} predictions, {neutral:,} neutral)")

# Test: inverted momentum_z (from earlier backtest finding)
print(f"\nInverted momentum_z test:")
inv_predictor = DirectionPredictor(weights={
    "ema_cross": 1.0, "rsi_zone": 1.0, "bb_position": 1.0,
    "momentum_z": -1.0, "vwap_dev": 1.0, "taker_flow": 1.0,
    "macd_hist": 1.0, "adx_trend": 1.0,
})
inv_correct = 0
inv_total = 0
for r in results:
    conf_inv, _ = inv_predictor.predict_direction(r['features'])
    if conf_inv != 0:
        is_correct = (conf_inv > 0 and r['outcome'] == "UP") or (conf_inv < 0 and r['outcome'] == "DOWN")
        if is_correct:
            inv_correct += 1
        inv_total += 1
if inv_total:
    print(f"  Inverted momentum_z: {inv_correct}/{inv_total} = {inv_correct / inv_total * 100:.2f}%")

# Test: correlation-weighted
print(f"\nCorrelation-weighted predictor:")
corr_weights = {}
for fname, corr, acc, cnt in feat_scores:
    # Use sign and magnitude of correlation as weight
    corr_weights[fname] = max(0.01, corr * 10) if corr > 0 else min(-0.01, corr * 10)
corr_predictor = DirectionPredictor(weights=corr_weights)
corr_correct = 0
corr_total = 0
for r in results:
    conf_c, _ = corr_predictor.predict_direction(r['features'])
    if conf_c != 0:
        is_correct = (conf_c > 0 and r['outcome'] == "UP") or (conf_c < 0 and r['outcome'] == "DOWN")
        if is_correct:
            corr_correct += 1
        corr_total += 1
if corr_total:
    print(f"  Correlation-weighted: {corr_correct}/{corr_total} = {corr_correct / corr_total * 100:.2f}%")
    print(f"  Weights: {json.dumps({k: round(v, 3) for k, v in corr_weights.items()})}")

# Test: RSI-only (best single feature from prior backtest)
print(f"\nRSI-only predictor:")
rsi_correct = 0
rsi_total = 0
for r in results:
    rsi_val = r['features'].get('rsi_zone', 0)
    if abs(rsi_val) > 0.01:
        is_correct = (rsi_val > 0 and r['outcome'] == "UP") or (rsi_val < 0 and r['outcome'] == "DOWN")
        if is_correct:
            rsi_correct += 1
        rsi_total += 1
if rsi_total:
    print(f"  RSI-only: {rsi_correct}/{rsi_total} = {rsi_correct / rsi_total * 100:.2f}%")

# Test: Taker flow only
print(f"\nTaker flow only predictor:")
tf_correct = 0
tf_total = 0
for r in results:
    tf_val = r['features'].get('taker_flow', 0)
    if abs(tf_val) > 0.01:
        is_correct = (tf_val > 0 and r['outcome'] == "UP") or (tf_val < 0 and r['outcome'] == "DOWN")
        if is_correct:
            tf_correct += 1
        tf_total += 1
if tf_total:
    print(f"  Taker-flow-only: {tf_correct}/{tf_total} = {tf_correct / tf_total * 100:.2f}%")

# ──────────────────────────────────────────────────────────
# 8. FINAL VERDICT
# ──────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"VERDICT")
print(f"{'=' * 70}")
overall_acc = correct_cnt / total_nn * 100 if total_nn else 0
if overall_acc > 55:
    print(f"  EDGE DETECTED: {overall_acc:.2f}% accuracy over {total_nn:,} trades")
    print(f"  Recommended: Enable skew in Phase 2 (paper) with conservative 55/45 allocation")
elif overall_acc > 52:
    print(f"  MARGINAL SIGNAL: {overall_acc:.2f}% accuracy over {total_nn:,} trades")
    print(f"  Recommended: Continue shadow-tracking, may have weak edge with filtered confidence")
else:
    print(f"  NO EDGE: {overall_acc:.2f}% accuracy over {total_nn:,} trades")
    print(f"  The 8-feature direction predictor does NOT predict 5min/15min candle outcomes")
    print(f"  These markets are efficient — Binance 1m TA cannot reliably predict them")
    print(f"  Recommended: Stay with 50/50 paired trades (guaranteed 5.3% per trade)")

# Save results summary
summary = {
    "total_markets": len(results),
    "overall_accuracy": round(overall_acc, 4),
    "by_asset": {},
    "by_duration": {},
    "by_confidence": {},
    "feature_correlations": {f[0]: round(f[1], 6) for f in feat_scores},
    "feature_accuracy": {f[0]: round(f[2], 4) for f in feat_scores},
}
for asset in ['BTC', 'ETH']:
    subset = [r for r in non_neutral if r['asset'] == asset]
    if subset:
        c = sum(1 for r in subset if r['correct'])
        summary["by_asset"][asset] = round(c / len(subset) * 100, 4)
for dur in [5, 15]:
    subset = [r for r in non_neutral if r['duration'] == dur]
    if subset:
        c = sum(1 for r in subset if r['correct'])
        summary["by_duration"][str(dur)] = round(c / len(subset) * 100, 4)

with open("backtest_skew_polydata_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to backtest_skew_polydata_results.json")
print("Done.")
