"""
FIB_FIB_CONFIRM Paper Trader V1.0

1M Fib Confluence signal + 5M Fib Confluence confirmation.
Backtest: 389 trades/14d, 59.6% WR, $61.42/day, $1,843/mo.

Separate from unified trader for independent evaluation.
Includes ML auto-optimizer that retrains after 30 trades.
"""
import sys
import json
import time
import os
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from functools import partial as fn_partial
from typing import Optional, Tuple

import httpx

print = fn_partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()
from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# CONFIG
# ============================================================================
SCAN_INTERVAL = 30
MAX_CONCURRENT = 3
TIME_WINDOW = (1.5, 7.0)
SPREAD_OFFSET = 0.03
MIN_ENTRY_PRICE = 0.10
MAX_ENTRY_PRICE = 0.60
RESOLVE_AGE_MIN = 8.0
BASE_SIZE = 3.0
MAX_SIZE = 8.0
MIN_EDGE = 0.010
RESULTS_FILE = Path(__file__).parent / "fib_fib_results.json"
ML_FILE = Path(__file__).parent / "fib_fib_ml.json"


# ============================================================================
# INDICATORS
# ============================================================================

def calc_ema(arr, n):
    out = np.empty(len(arr), dtype=float)
    a = 2.0 / (n + 1.0)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out

def calc_sma(arr, n):
    out = np.full(len(arr), np.nan, dtype=float)
    if n <= 0 or len(arr) < n:
        return out
    c = np.cumsum(arr, dtype=float)
    c[n:] = c[n:] - c[:-n]
    out[n - 1:] = c[n - 1:] / n
    return out

def calc_rsi(close, n=14):
    out = np.full(len(close), np.nan, dtype=float)
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

def calc_true_range(high, low, close):
    out = np.empty(len(close), dtype=float)
    out[0] = high[0] - low[0]
    for i in range(1, len(close)):
        out[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    return out

def calc_atr(high, low, close, n=14):
    return calc_sma(calc_true_range(high, low, close), n)


# ============================================================================
# FIB_CONFLUENCE_BOUNCE SIGNAL
# ============================================================================

def signal_fib_confluence(close, high, low, open_, end_idx):
    """0.786 touch -> 0.618 higher-low pivot -> entry. Returns (side, conf)."""
    if end_idx < 55:
        return None, 0
    c, h, l, o = close[:end_idx], high[:end_idx], low[:end_idx], open_[:end_idx]
    n = len(c)
    fc_pass, fc_side, strong_confirm = False, None, False
    bat_val, ba_val, sr_val, sh_val = 0, 0, 0.0, 0.0

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
# ML OPTIMIZER
# ============================================================================

class MLOptimizer:
    def __init__(self, features_file):
        self.features_file = features_file
        self.samples = []
        self.model = None
        self.feature_keys = ["conf", "fib_conf_5m", "ema_gap", "rsi", "atr_pct", "hour_utc", "edge", "strength"]
        self._load()

    def _load(self):
        if self.features_file.exists():
            try:
                self.samples = json.loads(self.features_file.read_text()).get("samples", [])
                if len(self.samples) >= 30:
                    self._train()
            except Exception:
                pass

    def _save(self):
        try:
            self.features_file.write_text(json.dumps({"samples": self.samples[-500:]}, indent=1))
        except Exception:
            pass

    def record(self, features, won):
        self.samples.append({"features": features, "won": won})
        if len(self.samples) % 20 == 0 and len(self.samples) >= 30:
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
            if len(X) < 30 or len(set(y)) < 2:
                return
            self.model = LogisticRegression(max_iter=500, C=0.5)
            self.model.fit(X, y)
            acc = self.model.score(X, y)
            print(f"[ML] Retrained on {len(X)} samples | acc: {acc:.1%}")
        except ImportError:
            pass
        except Exception as e:
            print(f"[ML] Error: {e}")

    def predict(self, features):
        if self.model is None:
            return 0.5
        try:
            row = [[float(features.get(k, 0.0)) for k in self.feature_keys]]
            return float(self.model.predict_proba(row)[0][1])
        except Exception:
            return 0.5

    def should_enter(self, features, min_prob=0.48):
        p = self.predict(features)
        return p >= min_prob, p

    def size_mult(self, features):
        p = self.predict(features)
        if p <= 0.45: return 0.5
        elif p >= 0.60: return 1.5
        else: return 0.5 + (p - 0.45) / 0.15


# ============================================================================
# PAPER TRADER
# ============================================================================

class FibFibPaperTrader:
    def __init__(self):
        self.trades = {}
        self.resolved = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0}
        self.ml = MLOptimizer(ML_FILE)
        self._load()

    def _load(self):
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.resolved = data.get("resolved", [])
                self.trades = data.get("active", {})
                self.stats = data.get("stats", self.stats)
                total = self.stats["wins"] + self.stats["losses"]
                wr = self.stats["wins"] / total * 100 if total > 0 else 0
                print(f"[LOAD] {total} resolved ({self.stats['wins']}W/{self.stats['losses']}L {wr:.0f}%WR) | "
                      f"PnL: ${self.stats['pnl']:+.2f} | {len(self.trades)} active")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        try:
            RESULTS_FILE.write_text(json.dumps({
                "stats": self.stats, "resolved": self.resolved, "active": self.trades,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }, indent=2))
        except Exception:
            pass

    def fetch_candles(self, interval, limit=200):
        for attempt in range(3):
            try:
                r = httpx.get("https://api.binance.com/api/v3/klines",
                              params={"symbol": "BTCUSDT", "interval": interval, "limit": limit}, timeout=15)
                r.raise_for_status()
                return [{"time": int(b[0]), "open": float(b[1]), "high": float(b[2]),
                         "low": float(b[3]), "close": float(b[4]), "volume": float(b[5])} for b in r.json()]
            except Exception:
                if attempt < 2: time.sleep(2 * (attempt + 1))
        return []

    def discover_5m_markets(self):
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
                print(f"[API] Error: {e}")
        return markets

    def get_prices(self, market):
        outcomes = market.get("outcomes", [])
        prices = market.get("outcomePrices", [])
        if isinstance(outcomes, str): outcomes = json.loads(outcomes)
        if isinstance(prices, str): prices = json.loads(prices)
        up_p, dn_p = None, None
        for i, o in enumerate(outcomes):
            if i < len(prices):
                p = float(prices[i])
                if str(o).lower() == "up": up_p = p
                elif str(o).lower() == "down": dn_p = p
        return up_p, dn_p

    def resolve_trades(self):
        now = datetime.now(timezone.utc)
        for tid, trade in list(self.trades.items()):
            if trade.get("status") != "open":
                continue
            try:
                age_min = (now - datetime.fromisoformat(trade["entry_time"])).total_seconds() / 60
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
                        up_p, dn_p = self.get_prices(rm)
                        if up_p is not None:
                            price = up_p if trade["side"] == "UP" else dn_p
                            if price >= 0.95:
                                exit_val = trade["size_usd"] / trade["entry_price"]
                            elif price <= 0.05:
                                exit_val = 0
                            else:
                                exit_val = (trade["size_usd"] / trade["entry_price"]) * price
                            trade["pnl"] = round(exit_val - trade["size_usd"], 2)
                            trade["exit_price"] = price
                            trade["exit_time"] = now.isoformat()
                            trade["status"] = "closed"
                            won = trade["pnl"] > 0
                            self.stats["wins" if won else "losses"] += 1
                            self.stats["pnl"] += trade["pnl"]
                            ml_feats = trade.get("ml_features", {})
                            if ml_feats:
                                self.ml.record(ml_feats, won)
                            self.resolved.append(trade)
                            del self.trades[tid]
                            t = self.stats
                            total = t["wins"] + t["losses"]
                            wr = t["wins"] / total * 100 if total > 0 else 0
                            tag = "WIN" if won else "LOSS"
                            print(f"[{tag}] FIB_FIB {trade['side']} ${trade['pnl']:+.2f} | "
                                  f"entry=${trade['entry_price']:.2f} exit=${price:.2f} | "
                                  f"{trade.get('reason', '')} | "
                                  f"Total: {t['wins']}W/{t['losses']}L {wr:.0f}%WR ${t['pnl']:+.2f}")
                except Exception as e:
                    if "429" not in str(e):
                        print(f"[RESOLVE] Error: {e}")
            elif age_min > 15:
                trade["status"] = "closed"
                trade["pnl"] = -trade["size_usd"]
                self.stats["losses"] += 1
                self.stats["pnl"] += trade["pnl"]
                self.resolved.append(trade)
                del self.trades[tid]

    def find_entries(self, candles_1m, candles_5m, markets_5m):
        if not candles_1m or len(candles_1m) < 60:
            return
        if not candles_5m or len(candles_5m) < 60:
            return
        if not markets_5m:
            return

        open_count = sum(1 for t in self.trades.values() if t.get("status") == "open")
        if open_count >= MAX_CONCURRENT:
            return

        # 1M fib signal
        c1m = np.array([c["close"] for c in candles_1m], dtype=float)
        h1m = np.array([c["high"] for c in candles_1m], dtype=float)
        l1m = np.array([c["low"] for c in candles_1m], dtype=float)
        o1m = np.array([c["open"] for c in candles_1m], dtype=float)

        side_1m, conf_1m = signal_fib_confluence(c1m, h1m, l1m, o1m, len(c1m))
        if side_1m is None:
            return

        # 5M fib confirmation (must agree)
        c5m = np.array([c["close"] for c in candles_5m], dtype=float)
        h5m = np.array([c["high"] for c in candles_5m], dtype=float)
        l5m = np.array([c["low"] for c in candles_5m], dtype=float)
        o5m = np.array([c["open"] for c in candles_5m], dtype=float)

        side_5m, conf_5m = signal_fib_confluence(c5m, h5m, l5m, o5m, len(c5m))
        if side_5m != side_1m:
            return  # No dual-TF confirmation

        # Both timeframes agree - strong signal
        poly_side = "UP" if side_1m == "LONG" else "DOWN"

        # Compute fair prob + indicators
        rsi14 = calc_rsi(c1m, 14)
        rsi_val = float(rsi14[-1]) if np.isfinite(rsi14[-1]) else 50.0
        atr14 = calc_atr(h1m, l1m, c1m, 14)
        atr_pct = float(atr14[-1] / max(1e-9, c1m[-1])) if np.isfinite(atr14[-1]) else 0.0
        ema9 = calc_ema(c5m, 9)
        ema21 = calc_ema(c5m, 21)
        ema_gap = (float(ema9[-1]) - float(ema21[-1])) / max(1e-9, float(c5m[-1]))

        p = 0.50
        p += float(np.clip(ema_gap * 30, -0.12, 0.12))
        p += float(np.clip((rsi_val - 50.0) / 400.0, -0.06, 0.06))
        fair_up = float(np.clip(p, 0.05, 0.95))
        strength = float(np.clip(((conf_1m + conf_5m) / 2.0 - 0.70) / 0.22, 0.0, 1.0))

        now = datetime.now(timezone.utc)
        hour_utc = now.hour

        for market in markets_5m:
            if open_count >= MAX_CONCURRENT:
                break
            time_left = market.get("_time_left", 99)
            if time_left < TIME_WINDOW[0] or time_left > TIME_WINDOW[1]:
                continue

            cid = market.get("conditionId", "")
            nid = market.get("id")
            trade_key = f"FIB_FIB_{cid}_{poly_side}"
            if trade_key in self.trades:
                continue

            up_p, dn_p = self.get_prices(market)
            if up_p is None or dn_p is None:
                continue
            tte = market.get("_tte_seconds", 0)
            if tte < 60:
                continue

            if poly_side == "UP":
                entry = up_p + SPREAD_OFFSET
                edge = fair_up - entry
            else:
                entry = dn_p + SPREAD_OFFSET
                edge = (1.0 - fair_up) - entry

            if entry < MIN_ENTRY_PRICE or entry > MAX_ENTRY_PRICE:
                continue
            if edge < MIN_EDGE:
                continue

            ml_feats = {
                "conf": conf_1m, "fib_conf_5m": conf_5m,
                "ema_gap": ema_gap, "rsi": rsi_val, "atr_pct": atr_pct,
                "hour_utc": float(hour_utc), "edge": float(edge), "strength": strength,
            }
            should, ml_prob = self.ml.should_enter(ml_feats)
            if not should:
                continue
            ml_mult = self.ml.size_mult(ml_feats)

            trade_size = BASE_SIZE + strength * (MAX_SIZE - BASE_SIZE)
            trade_size = round(max(BASE_SIZE, min(MAX_SIZE, trade_size * ml_mult)), 2)

            fair_px = fair_up if poly_side == "UP" else 1.0 - fair_up
            self.trades[trade_key] = {
                "side": poly_side, "entry_price": entry, "size_usd": trade_size,
                "entry_time": now.isoformat(), "condition_id": cid,
                "market_numeric_id": nid, "title": market.get("question", ""),
                "strategy": "FIB_FIB_CONFIRM", "edge": round(edge, 4),
                "fair_px": round(fair_px, 4), "strength": round(strength, 3),
                "reason": f"fib_fib_{side_1m.lower()}", "btc_price": float(c1m[-1]),
                "detail": f"1m_conf={conf_1m:.2f} 5m_conf={conf_5m:.2f} rsi={rsi_val:.0f}",
                "tte_seconds": tte, "ml_features": ml_feats, "ml_prob": round(ml_prob, 3),
                "status": "open", "pnl": 0.0,
            }
            open_count += 1
            print(f"[ENTRY] FIB_FIB {poly_side} ${entry:.2f} ${trade_size:.0f} | "
                  f"edge={edge:.1%} 1m={conf_1m:.2f} 5m={conf_5m:.2f} | "
                  f"fib_fib_{side_1m.lower()} | rsi={rsi_val:.0f} | "
                  f"tte={tte}s | BTC=${c1m[-1]:,.0f} | ml={ml_prob:.2f}")

    def run(self):
        print("=" * 70)
        print("  FIB_FIB_CONFIRM Paper Trader V1.0")
        print("  Signal: 1M Fib Confluence + 5M Fib Confluence confirmation")
        print(f"  Max concurrent: {MAX_CONCURRENT} | Window: {TIME_WINDOW[0]}-{TIME_WINDOW[1]}min")
        print(f"  Size: ${BASE_SIZE}-${MAX_SIZE} | Edge >= {MIN_EDGE:.1%}")
        print(f"  ML: {len(self.ml.samples)} samples | model={'active' if self.ml.model else 'warming up'}")
        print("=" * 70)

        if self.resolved:
            t = self.stats
            total = t["wins"] + t["losses"]
            wr = t["wins"] / total * 100 if total > 0 else 0
            print(f"[RESUME] {t['wins']}W/{t['losses']}L {wr:.0f}%WR | PnL: ${t['pnl']:+.2f}")

        cycle = 0
        while True:
            try:
                cycle += 1
                candles_1m = self.fetch_candles("1m", 200)
                candles_5m = self.fetch_candles("5m", 200)
                markets = self.discover_5m_markets()

                if not candles_1m:
                    time.sleep(10)
                    continue

                self.resolve_trades()
                self.find_entries(candles_1m, candles_5m, markets)

                if cycle % 10 == 0:
                    btc = candles_1m[-1]["close"] if candles_1m else 0
                    t = self.stats
                    total = t["wins"] + t["losses"]
                    wr = t["wins"] / total * 100 if total > 0 else 0
                    ml_tag = f"ml={len(self.ml.samples)}s" if self.ml.model else f"warmup({len(self.ml.samples)})"
                    print(f"\n--- Cycle {cycle} | BTC ${btc:,.0f} | "
                          f"{t['wins']}W/{t['losses']}L {wr:.0f}%WR ${t['pnl']:+.2f} | "
                          f"Active: {len(self.trades)} | 5M mkt: {len(markets)} | {ml_tag} ---")

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


if __name__ == "__main__":
    lock = acquire_pid_lock("fib_fib_paper")
    try:
        FibFibPaperTrader().run()
    finally:
        release_pid_lock("fib_fib_paper")
