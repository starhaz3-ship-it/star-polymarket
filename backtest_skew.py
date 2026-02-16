"""V4.8 Direction Skew Backtest — test predictor against all historical trades."""
import json, re, time, statistics, httpx, asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# --- Load resolved trades ---
data = json.loads(open('maker_results.json').read())
resolved = data.get('resolved', [])
trades_with_outcome = [r for r in resolved if r.get('outcome') in ('UP', 'DOWN')]
print(f"Loaded {len(trades_with_outcome)} trades with outcomes")

# --- Parse candle start time from question ---
ET_OFFSET = timedelta(hours=-5)  # ET = UTC-5

def parse_candle_start(question: str) -> Optional[datetime]:
    m = re.search(r'(\w+ \d+),\s*(\d{1,2}):(\d{2})(AM|PM)-', question)
    if not m:
        return None
    hour = int(m.group(2))
    minute = int(m.group(3))
    ampm = m.group(4)
    if ampm == 'PM' and hour != 12:
        hour += 12
    elif ampm == 'AM' and hour == 12:
        hour = 0
    try:
        et_time = datetime(2026, 2, 16, hour, minute, tzinfo=timezone(ET_OFFSET))
        utc_time = et_time.astimezone(timezone.utc).replace(tzinfo=None)
        return utc_time
    except:
        return None

test = parse_candle_start("Bitcoin Up or Down - February 16, 1:35AM-1:40AM ET")
print(f"Parse test: '1:35AM ET' -> {test} UTC")

# --- Fetch historical klines from Binance ---
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"

async def fetch_klines_at(symbol: str, dt: datetime, limit: int = 30) -> List[list]:
    end_ms = int(dt.timestamp() * 1000)
    start_ms = end_ms - (limit * 60 * 1000)
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(BINANCE_KLINES, params={
            "symbol": symbol, "interval": "1m",
            "startTime": start_ms, "endTime": end_ms, "limit": limit
        })
        if r.status_code == 200:
            return r.json()
    return []

# --- DirectionPredictor (same as run_maker.py) ---
class DirectionPredictor:
    def __init__(self):
        self.weights = {
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
            d = closes[i] - closes[i-1]
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
            std = (sum((x - mid)**2 for x in w) / 20) ** 0.5
            if std > 0:
                f["bb_position"] = max(-1, min(1, (closes[-1] - mid) / (2 * std)))

        if n >= 12:
            rets = [(closes[i] - closes[i-1]) / closes[i-1]
                    for i in range(max(1, n-12), n) if closes[i-1] > 0]
            if len(rets) >= 2:
                vol = statistics.stdev(rets)
                if vol > 1e-8 and n >= 6 and closes[-6] > 0:
                    mom = (closes[-1] - closes[-6]) / closes[-6]
                    f["momentum_z"] = max(-1, min(1, mom / vol / 3))

        pv = sum((highs[i]+lows[i]+closes[i])/3 * volumes[i] for i in range(n))
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
                dx = abs(pdi-mdi)/(pdi+mdi)*100 if (pdi+mdi) > 0 else 0
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


# --- Run backtest ---
async def run_backtest():
    predictor = DirectionPredictor()

    trade_groups = defaultdict(list)
    skipped = 0
    for t in trades_with_outcome:
        start = parse_candle_start(t.get('question', ''))
        if not start:
            skipped += 1
            continue
        asset = t.get('asset', 'BTC')
        symbol = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"}.get(asset, "BTCUSDT")
        key = f"{symbol}_{int(start.timestamp())}"
        trade_groups[key].append((t, start, symbol))

    print(f"Trade groups: {len(trade_groups)} unique (candle, asset) pairs, {skipped} skipped")

    results = []
    feature_outcomes = defaultdict(list)

    total = len(trade_groups)
    done = 0
    kline_cache = {}

    for key, group in trade_groups.items():
        t0, start, symbol = group[0]

        cache_key = f"{symbol}_{int(start.timestamp())}"
        if cache_key in kline_cache:
            klines = kline_cache[cache_key]
        else:
            klines = await fetch_klines_at(symbol, start, limit=30)
            kline_cache[cache_key] = klines
            time.sleep(0.05)  # Rate limit
        done += 1

        if not klines or len(klines) < 15:
            continue

        if done % 30 == 0:
            print(f"  Progress: {done}/{total} kline fetches...")

        features = predictor.compute_features(klines)
        confidence, contribs = predictor.predict_direction(features)

        for t, _, _ in group:
            outcome = t['outcome']
            correct_sign = 1.0 if outcome == "UP" else -1.0
            predicted_dir = "UP" if confidence > 0 else "DOWN" if confidence < 0 else "NEUTRAL"
            correct = (confidence > 0 and outcome == "UP") or (confidence < 0 and outcome == "DOWN")

            for fname, fval in features.items():
                feature_outcomes[fname].append((fval, correct_sign))

            up_price = t.get('up_price', 0.48) or 0.48
            down_price = t.get('down_price', 0.47) or 0.47
            total_shares = 12

            actual_pnl = t.get('pnl', 0)

            shadow_pnl = {}
            for skew in [0.50, 0.55, 0.60, 0.65, 0.70]:
                up_r, dn_r = predictor.confidence_to_skew(confidence, max_skew=skew)
                up_sh = total_shares * up_r
                dn_sh = total_shares * dn_r
                if outcome == "UP":
                    pnl_hyp = up_sh * (1.0 - up_price) - dn_sh * down_price
                else:
                    pnl_hyp = dn_sh * (1.0 - down_price) - up_sh * up_price
                shadow_pnl[f"skew_{int(skew*100)}"] = round(pnl_hyp, 6)

            results.append({
                "asset": t.get('asset'),
                "outcome": outcome,
                "confidence": confidence,
                "predicted_dir": predicted_dir,
                "correct": correct,
                "actual_pnl": actual_pnl,
                "entry_tier": t.get('entry_tier', 'deep'),
                "shadow_pnl": shadow_pnl,
                "features": features,
            })

    return results, feature_outcomes


results, feature_outcomes = asyncio.run(run_backtest())

# --- Analysis ---
print(f"\n{'='*70}")
print(f"BACKTEST RESULTS: {len(results)} trades analyzed")
print(f"{'='*70}")

# Direction accuracy
non_neutral = [r for r in results if r['confidence'] != 0]
correct_cnt = sum(1 for r in non_neutral if r['correct'])
total_nn = len(non_neutral)
print(f"\nDirection Accuracy: {correct_cnt}/{total_nn} = {correct_cnt/total_nn*100:.1f}%" if total_nn else "\nNo predictions")

# By asset
for asset in ['BTC', 'ETH', 'SOL', 'XRP']:
    subset = [r for r in non_neutral if r['asset'] == asset]
    if subset:
        c = sum(1 for r in subset if r['correct'])
        print(f"  {asset}: {c}/{len(subset)} = {c/len(subset)*100:.1f}%")

# By confidence bucket
print(f"\nAccuracy by confidence level:")
for lo, hi, label in [(0, 0.05, "|conf| 0-5%"), (0.05, 0.10, "|conf| 5-10%"),
                       (0.10, 0.20, "|conf| 10-20%"), (0.20, 0.50, "|conf| 20-50%"),
                       (0.50, 1.0, "|conf| 50%+")]:
    subset = [r for r in results if lo <= abs(r['confidence']) < hi]
    if subset:
        c = sum(1 for r in subset if r['correct'])
        print(f"  {label:>15}: {c}/{len(subset)} = {c/len(subset)*100:.1f}%")

# Shadow PnL comparison
print(f"\nPnL Comparison (shadow vs actual equal-shares):")
actual_total = sum(r['actual_pnl'] for r in results)
for key in ["skew_50", "skew_55", "skew_60", "skew_65", "skew_70"]:
    total_shadow = sum(r['shadow_pnl'].get(key, 0) for r in results)
    wins = sum(1 for r in results if r['shadow_pnl'].get(key, 0) > 0)
    avg = total_shadow / len(results) if results else 0
    wr = wins / len(results) * 100 if results else 0
    delta = total_shadow - actual_total if key != "skew_50" else 0
    delta_str = f"  ({'+' if delta >= 0 else ''}{delta:.2f} vs equal)" if key != "skew_50" else " (baseline)"
    print(f"  {key:>8}: {len(results)}T, {wr:.0f}%WR, ${total_shadow:+.2f} total (${avg:+.4f}/T){delta_str}")
print(f"  {'actual':>8}: {len(results)}T, ${actual_total:+.2f} total (${actual_total/len(results):+.4f}/T)")

# Per-feature correlation
print(f"\nFeature-Direction Correlation (higher = better predictor):")
feat_scores = []
for fname in sorted(feature_outcomes.keys()):
    pairs = feature_outcomes[fname]
    if len(pairs) < 10:
        continue
    corrs = [val * sign for val, sign in pairs]
    avg_corr = sum(corrs) / len(corrs)
    correct_when_used = sum(1 for v, s in pairs if (v > 0 and s > 0) or (v < 0 and s < 0))
    total_nonzero = sum(1 for v, _ in pairs if abs(v) > 0.01)
    acc = correct_when_used / total_nonzero * 100 if total_nonzero > 0 else 0
    feat_scores.append((fname, avg_corr, acc, total_nonzero))
    print(f"  {fname:>15}: corr={avg_corr:+.4f} | accuracy={acc:.1f}% ({total_nonzero}T)")

# Feature agreement analysis
print(f"\nFeature agreement (how many agree on direction):")
for thresh in [6, 5, 4, 3]:
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
        print(f"  {thresh}+ features agree: {subset_correct}/{subset_total} = {subset_correct/subset_total*100:.1f}% correct")

# Best individual and pair combos
print(f"\nBest feature combos:")
best_results = []

# Singles
for fname, corr, acc, cnt in feat_scores:
    if cnt >= 20:
        best_results.append((acc, f"{fname} alone", cnt))

# Pairs
feature_names = [f[0] for f in feat_scores if f[3] >= 20]
for i, f1 in enumerate(feature_names):
    for f2 in feature_names[i+1:]:
        correct = 0
        total = 0
        for r in results:
            v1 = r['features'].get(f1, 0)
            v2 = r['features'].get(f2, 0)
            if abs(v1) < 0.01 and abs(v2) < 0.01:
                continue
            pred = v1 + v2
            s = 1.0 if r['outcome'] == 'UP' else -1.0
            if (pred > 0 and s > 0) or (pred < 0 and s < 0):
                correct += 1
            total += 1
        if total >= 20:
            best_results.append((correct / total * 100, f"{f1}+{f2}", total))

best_results.sort(key=lambda x: -x[0])
for acc, combo, cnt in best_results[:10]:
    print(f"  {acc:.1f}% | {combo} ({cnt}T)")

# Optimal weight recommendation
print(f"\n{'='*70}")
print(f"RECOMMENDED WEIGHTS (from correlation analysis):")
print(f"{'='*70}")
for fname, corr, acc, cnt in sorted(feat_scores, key=lambda x: -abs(x[1])):
    # Weight proportional to correlation strength
    new_w = max(0.1, min(3.0, 1.0 + corr * 5))
    print(f"  {fname:>15}: weight={new_w:.2f} (corr={corr:+.4f}, acc={acc:.1f}%)")

print(f"\nDone.")
