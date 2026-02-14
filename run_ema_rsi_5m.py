"""
ML-Optimized EMA/RSI 5M BTC Directional Paper Trader V1.0

Combines:
  1. EMA(9)/EMA(21) crossover + RSI(14) + ATR impulse gate
  2. Polymarket 5M BTC Up/Down market matching
  3. Edge-based entry vs market probability
  4. Gradient Boosting ML filter (auto-trains after 30 trades)

Signal logic (from user's spec):
  LONG:  EMA(9) > EMA(21) AND RSI > 56 AND impulse > 0.6 ATR AND EMA gap > 6bp
  SHORT: EMA(9) < EMA(21) AND RSI < 44 AND impulse > 0.6 ATR AND EMA gap > 6bp

Edge gate:
  Estimate directional probability from TA features, compare to Polymarket price.
  Require >= 3% edge (6% for late entries < 2min to expiry).

ML layer:
  Trains GradientBoostingClassifier on trade features after 30+ resolved trades.
  Features: impulse, ema_gap_bp, rsi, entry_price, edge, tte_seconds, hour_utc,
            btc_volatility, momentum_5m, momentum_10m
  Retrains every 20 new trades. Filters out trades with < 45% ML win prob.

Usage:
  python run_ema_rsi_5m.py          # Paper mode (default)
  python run_ema_rsi_5m.py --live   # Live mode (real CLOB orders)
"""

import sys
import math
import json
import time
import os
import asyncio
import argparse
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from functools import partial as fn_partial
from typing import Optional

import httpx

print = fn_partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# STRATEGY PARAMS (from user's spec)
# ============================================================================
EDGE_MIN = 0.03
EDGE_STRONG = 0.06
MIN_TTE_SECONDS = 60
LATE_TTE_SECONDS = 120
EMA_GAP_MIN_BP = 6
IMPULSE_MIN = 0.6
IMPULSE_STRONG = 1.2
RSI_LONG = 56
RSI_SHORT = 44
BASE_BET = 3.0
MIN_BET = 3.0
MAX_BET = 5.0
EDGE_SIZE_K = 35.0

# ============================================================================
# PAPER TRADING CONFIG
# ============================================================================
SCAN_INTERVAL = 25              # seconds between cycles
MAX_CONCURRENT = 3              # max open paper positions
TIME_WINDOW = (0.5, 4.8)        # minutes before expiry to enter
SPREAD_OFFSET = 0.02            # paper fill spread simulation
RESOLVE_AGE_MIN = 6.0           # minutes before resolving via API
MIN_ENTRY_PRICE = 0.10          # avoid noise at extremes
MAX_ENTRY_PRICE = 0.62          # avoid overpaying

# ============================================================================
# ML CONFIG
# ============================================================================
ML_MIN_TRADES = 30              # minimum trades before ML activates
ML_RETRAIN_EVERY = 20           # retrain after N new trades
ML_MIN_CONFIDENCE = 0.45        # minimum ML win probability to take trade
ML_BOOST_THRESHOLD = 0.65       # ML confidence above this = +$1 sizing boost

# ============================================================================
# PATHS
# ============================================================================
RESULTS_FILE = Path(__file__).parent / "ema_rsi_5m_results.json"
ML_MODEL_FILE = Path(__file__).parent / "ema_rsi_5m_model.joblib"

# ============================================================================
# INDICATOR FUNCTIONS
# ============================================================================
def calc_ema(series: np.ndarray, length: int) -> np.ndarray:
    result = np.zeros_like(series, dtype=float)
    alpha = 2.0 / (length + 1)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def calc_rsi(closes: np.ndarray, length: int) -> np.ndarray:
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    rsi_vals = np.full(len(closes), np.nan)
    for i in range(length, len(deltas)):
        avg_gain = np.mean(gains[i - length + 1:i + 1])
        avg_loss = np.mean(losses[i - length + 1:i + 1])
        if avg_loss == 0:
            rsi_vals[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_vals[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return rsi_vals


def calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, length: int) -> np.ndarray:
    tr = np.zeros(len(closes))
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
    atr_vals = np.full(len(closes), np.nan)
    for i in range(length, len(closes)):
        atr_vals[i] = np.mean(tr[i - length + 1:i + 1])
    return atr_vals


# ============================================================================
# SIGNAL HELPERS (from user's spec)
# ============================================================================
def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def prob_to_logit(p: float) -> float:
    p = _clamp(p, 1e-6, 1 - 1e-6)
    return math.log(p / (1 - p))


def logit_to_prob(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def get_signal(candles: list) -> Optional[dict]:
    """EMA crossover + RSI + ATR impulse gate. Returns signal dict or None."""
    if len(candles) < 12:
        return None

    last = candles[-1]
    price = float(last["close"])
    ef = float(last["ema_fast"])
    es = float(last["ema_slow"])
    r = float(last["rsi"])
    atr_val = float(last.get("atr", 0.0) or 0.0)

    if np.isnan(r) or np.isnan(atr_val) or atr_val <= 0:
        return None

    # Chop filter: EMA gap in basis points
    ema_gap_bp = abs(ef - es) / price * 10_000.0
    if ema_gap_bp < EMA_GAP_MIN_BP:
        return None

    # ATR-normalized 10m impulse (2 bars back on 5m)
    if len(candles) < 3:
        return None
    close_10m = float(candles[-3]["close"])
    impulse = abs(price - close_10m) / atr_val

    if impulse < IMPULSE_MIN:
        return None

    # Momentum for ML features
    close_5m = float(candles[-2]["close"])
    momentum_5m = (price - close_5m) / close_5m if close_5m > 0 else 0
    momentum_10m = (price - close_10m) / close_10m if close_10m > 0 else 0

    # Volatility (stddev of last 20 closes / mean)
    recent = np.array([float(c["close"]) for c in candles[-20:]])
    volatility = np.std(recent) / np.mean(recent) if len(recent) >= 5 else 0

    if ef > es and r >= RSI_LONG:
        return {
            "dir": "long", "impulse": impulse, "ema_gap_bp": ema_gap_bp,
            "rsi": r, "atr": atr_val, "price": price,
            "momentum_5m": momentum_5m, "momentum_10m": momentum_10m,
            "volatility": volatility,
        }
    if ef < es and r <= RSI_SHORT:
        return {
            "dir": "short", "impulse": impulse, "ema_gap_bp": ema_gap_bp,
            "rsi": r, "atr": atr_val, "price": price,
            "momentum_5m": momentum_5m, "momentum_10m": momentum_10m,
            "volatility": volatility,
        }
    return None


def estimate_prob_from_ta(sig: dict) -> float:
    """Convert TA features to directional probability estimate."""
    impulse = float(sig["impulse"])
    ema_gap_bp = float(sig["ema_gap_bp"])
    rsi = float(sig["rsi"])
    direction = sig["dir"]

    imp_n = _clamp((impulse - IMPULSE_MIN) / (IMPULSE_STRONG - IMPULSE_MIN), 0.0, 1.5)
    gap_n = _clamp((ema_gap_bp - EMA_GAP_MIN_BP) / (20.0 - EMA_GAP_MIN_BP), 0.0, 1.5)

    if direction == "long":
        rsi_n = _clamp((rsi - RSI_LONG) / (70.0 - RSI_LONG), 0.0, 1.5)
    else:
        rsi_n = _clamp((RSI_SHORT - rsi) / (RSI_SHORT - 30.0), 0.0, 1.5)

    z = -0.2 + 0.9 * imp_n + 0.6 * gap_n + 0.5 * rsi_n
    p = logit_to_prob(z)
    return _clamp(p, 0.05, 0.95)


def position_size_from_edge(edge: float, ml_boost: bool = False) -> float:
    extra = EDGE_SIZE_K * max(0.0, edge - EDGE_MIN)
    size = BASE_BET + extra
    if ml_boost:
        size += 1.0  # ML confidence bonus
    return _clamp(size, MIN_BET, MAX_BET)


# ============================================================================
# ML LAYER
# ============================================================================
class MLFilter:
    """Gradient Boosting classifier trained on trade outcomes."""

    FEATURE_NAMES = [
        "impulse", "ema_gap_bp", "rsi_norm", "entry_price", "edge",
        "tte_seconds", "hour_utc", "volatility", "momentum_5m", "momentum_10m",
    ]

    def __init__(self):
        self.model = None
        self.trades_at_last_train = 0
        self._load_model()

    def _load_model(self):
        if ML_MODEL_FILE.exists():
            try:
                import joblib
                self.model = joblib.load(ML_MODEL_FILE)
                print(f"[ML] Loaded model from disk")
            except Exception as e:
                print(f"[ML] Load failed: {e}")
                self.model = None

    def _save_model(self):
        if self.model is not None:
            try:
                import joblib
                joblib.dump(self.model, ML_MODEL_FILE)
            except Exception as e:
                print(f"[ML] Save failed: {e}")

    def extract_features(self, sig: dict, entry_price: float, edge: float,
                         tte_seconds: int, hour_utc: int) -> list:
        """Extract feature vector for ML prediction."""
        direction = sig["dir"]
        # Normalize RSI relative to threshold
        if direction == "long":
            rsi_norm = (sig["rsi"] - RSI_LONG) / (70.0 - RSI_LONG)
        else:
            rsi_norm = (RSI_SHORT - sig["rsi"]) / (RSI_SHORT - 30.0)
        rsi_norm = _clamp(rsi_norm, -1.0, 2.0)

        return [
            sig["impulse"],
            sig["ema_gap_bp"],
            rsi_norm,
            entry_price,
            edge,
            tte_seconds / 300.0,  # normalize to 5min
            hour_utc / 24.0,      # normalize to day
            sig.get("volatility", 0),
            sig.get("momentum_5m", 0) * 10000,   # scale to bps
            sig.get("momentum_10m", 0) * 10000,
        ]

    def should_take_trade(self, features: list) -> tuple:
        """Returns (should_take, ml_prob, ml_boost)."""
        if self.model is None:
            return True, 0.5, False  # No model yet, take all trades

        try:
            X = np.array([features])
            prob = self.model.predict_proba(X)[0][1]  # P(win)
            take = prob >= ML_MIN_CONFIDENCE
            boost = prob >= ML_BOOST_THRESHOLD
            return take, prob, boost
        except Exception as e:
            print(f"[ML] Predict error: {e}")
            return True, 0.5, False

    def train(self, resolved_trades: list):
        """Train/retrain on resolved trades."""
        n = len(resolved_trades)
        if n < ML_MIN_TRADES:
            return
        if n - self.trades_at_last_train < ML_RETRAIN_EVERY and self.model is not None:
            return

        # Build training set
        X, y = [], []
        for t in resolved_trades:
            feats = t.get("ml_features")
            if feats is None or len(feats) != len(self.FEATURE_NAMES):
                continue
            won = 1 if t.get("pnl", 0) > 0 else 0
            X.append(feats)
            y.append(won)

        if len(X) < ML_MIN_TRADES:
            return

        X = np.array(X)
        y = np.array(y)

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import cross_val_score

            # Train with conservative hyperparameters to avoid overfitting
            model = GradientBoostingClassifier(
                n_estimators=min(50, max(20, len(X) // 3)),
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=max(3, len(X) // 15),
                subsample=0.8,
                random_state=42,
            )
            model.fit(X, y)

            # Cross-validation accuracy
            if len(X) >= 20:
                cv_scores = cross_val_score(model, X, y, cv=min(5, len(X) // 5), scoring='accuracy')
                cv_acc = cv_scores.mean()
            else:
                cv_acc = 0.0

            # Feature importance
            importances = dict(zip(self.FEATURE_NAMES, model.feature_importances_))
            top3 = sorted(importances.items(), key=lambda x: -x[1])[:3]
            top3_str = ", ".join(f"{k}={v:.2f}" for k, v in top3)

            train_acc = model.score(X, y)
            win_rate = np.mean(y)

            self.model = model
            self.trades_at_last_train = n
            self._save_model()

            print(f"[ML] Retrained on {len(X)} trades | "
                  f"Train acc: {train_acc:.1%} | CV acc: {cv_acc:.1%} | "
                  f"Base WR: {win_rate:.1%} | Top: {top3_str}")

        except Exception as e:
            print(f"[ML] Train error: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# TRADER
# ============================================================================
class EmaRsi5MTrader:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.client = None
        self.trades: dict = {}       # active: {trade_key: trade_dict}
        self.resolved: list = []     # closed trades
        self.stats = {
            "wins": 0, "losses": 0, "pnl": 0.0,
            "ml_filtered": 0, "ml_boosted": 0,
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        self.ml = MLFilter()
        self._load()

        if not self.paper:
            self._init_clob()

    def _init_clob(self):
        try:
            from py_clob_client.client import ClobClient
            from crypto_utils import decrypt_key

            password = os.getenv("POLYMARKET_PASSWORD", "")
            pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
            if pk.startswith("ENC:"):
                pk = decrypt_key(pk[4:], os.getenv("POLYMARKET_KEY_SALT", ""), password)
            proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "")

            self.client = ClobClient(
                host="https://clob.polymarket.com", key=pk, chain_id=137,
                signature_type=1, funder=proxy if proxy else None,
            )
            creds = self.client.derive_api_key()
            self.client.set_api_creds(creds)
            print(f"[CLOB] Client initialized - LIVE MODE")
        except Exception as e:
            print(f"[CLOB] Init failed: {e}")
            self.paper = True

    def _load(self):
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.trades = data.get("active", {})
                self.resolved = data.get("resolved", [])
                self.stats = data.get("stats", self.stats)
                # Restore ML training state
                self.ml.trades_at_last_train = data.get("ml_trades_at_last_train", 0)
                w, l = self.stats["wins"], self.stats["losses"]
                total = w + l
                wr = w / total * 100 if total > 0 else 0
                print(f"[LOAD] {total} resolved ({w}W/{l}L {wr:.0f}%WR) | "
                      f"PnL: ${self.stats['pnl']:+.2f} | {len(self.trades)} active | "
                      f"ML filtered: {self.stats.get('ml_filtered', 0)}")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        data = {
            "active": self.trades,
            "resolved": self.resolved[-500:],  # keep last 500 for ML training
            "stats": self.stats,
            "ml_trades_at_last_train": self.ml.trades_at_last_train,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            RESULTS_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    # ========================================================================
    # DATA FETCHING
    # ========================================================================

    async def fetch_candles(self) -> list:
        """Fetch 5m BTC/USDT candles from Binance and apply indicators."""
        url = "https://api.binance.com/api/v3/klines"
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    r = await client.get(url, params={
                        "symbol": "BTCUSDT", "interval": "5m", "limit": 100
                    })
                    r.raise_for_status()
                    raw = r.json()

                candles = []
                for bar in raw:
                    candles.append({
                        "time": int(bar[0]),
                        "open": float(bar[1]),
                        "high": float(bar[2]),
                        "low": float(bar[3]),
                        "close": float(bar[4]),
                        "volume": float(bar[5]),
                    })

                # Apply indicators
                closes = np.array([c["close"] for c in candles])
                highs = np.array([c["high"] for c in candles])
                lows = np.array([c["low"] for c in candles])

                ema_fast = calc_ema(closes, 9)
                ema_slow = calc_ema(closes, 21)
                rsi_vals = calc_rsi(closes, 14)
                atr_vals = calc_atr(highs, lows, closes, 14)

                for i, c in enumerate(candles):
                    c["ema_fast"] = ema_fast[i]
                    c["ema_slow"] = ema_slow[i]
                    c["rsi"] = rsi_vals[i]
                    c["atr"] = atr_vals[i]

                return candles

            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                print(f"[BINANCE] Error: {e}")
                return []

    async def discover_5m_markets(self) -> list:
        """Find active Polymarket 5M BTC Up/Down markets."""
        markets = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    "https://gamma-api.polymarket.com/events",
                    params={"tag_slug": "5M", "active": "true", "closed": "false", "limit": 200},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
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
                                time_left_min = (end_dt - now).total_seconds() / 60
                                tte_seconds = int((end_dt - now).total_seconds())
                                m["_time_left"] = time_left_min
                                m["_tte_seconds"] = tte_seconds
                            except Exception:
                                continue
                        if not m.get("question"):
                            m["question"] = event.get("title", "")
                        markets.append(m)
        except Exception as e:
            print(f"[API] Error discovering markets: {e}")
        return markets

    # ========================================================================
    # MARKET HELPERS
    # ========================================================================

    def get_prices(self, market: dict) -> tuple:
        outcomes = market.get("outcomes", [])
        prices = market.get("outcomePrices", [])
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(prices, str):
            prices = json.loads(prices)
        up_price, down_price = None, None
        for i, o in enumerate(outcomes):
            if i < len(prices):
                p = float(prices[i])
                if str(o).lower() == "up":
                    up_price = p
                elif str(o).lower() == "down":
                    down_price = p
        return up_price, down_price

    def get_token_ids(self, market: dict) -> tuple:
        outcomes = market.get("outcomes", [])
        token_ids = market.get("clobTokenIds", [])
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        up_tid, down_tid = None, None
        for i, o in enumerate(outcomes):
            if i < len(token_ids):
                if str(o).lower() == "up":
                    up_tid = token_ids[i]
                elif str(o).lower() == "down":
                    down_tid = token_ids[i]
        return up_tid, down_tid

    # ========================================================================
    # EDGE GATE
    # ========================================================================

    def compute_edge(self, sig: dict, market: dict) -> tuple:
        """Returns (side, entry_price, p_hat, edge, tte_seconds) or Nones."""
        up_p, down_p = self.get_prices(market)
        if up_p is None or down_p is None:
            return None, None, 0.0, 0.0, 0

        tte = market.get("_tte_seconds", 0)
        if tte < MIN_TTE_SECONDS:
            return None, None, 0.0, 0.0, 0

        side = "UP" if sig["dir"] == "long" else "DOWN"
        p_mkt = float(up_p if side == "UP" else down_p)
        p_hat = estimate_prob_from_ta(sig)
        edge = p_hat - p_mkt

        # Entry price with spread offset
        entry_price = round(p_mkt + SPREAD_OFFSET, 2)

        # Price range filter
        if entry_price < MIN_ENTRY_PRICE or entry_price > MAX_ENTRY_PRICE:
            return None, None, p_hat, edge, tte

        # Late regime requires bigger edge
        if tte < LATE_TTE_SECONDS:
            if edge < EDGE_STRONG:
                return None, None, p_hat, edge, tte

        if edge < EDGE_MIN:
            return None, None, p_hat, edge, tte

        return side, entry_price, p_hat, edge, tte

    # ========================================================================
    # RESOLVE
    # ========================================================================

    async def resolve_trades(self):
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
                    async with httpx.AsyncClient(timeout=8) as cl:
                        r = await cl.get(
                            f"https://gamma-api.polymarket.com/markets/{nid}",
                            headers={"User-Agent": "Mozilla/5.0"},
                        )
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
                                self.stats["wins" if won else "losses"] += 1
                                self.stats["pnl"] += trade["pnl"]
                                self.resolved.append(trade)
                                del self.trades[tid]

                                w, l = self.stats["wins"], self.stats["losses"]
                                wr = w / (w + l) * 100 if (w + l) > 0 else 0
                                tag = "WIN" if won else "LOSS"

                                ml_str = ""
                                if trade.get("ml_prob"):
                                    ml_str = f" ml={trade['ml_prob']:.0%}"

                                print(f"[{tag}] {trade['side']} ${trade['pnl']:+.2f} | "
                                      f"entry=${trade['entry_price']:.2f} exit=${price:.2f} | "
                                      f"edge={trade.get('edge', 0):.1%}{ml_str} | "
                                      f"{w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f}")

                                # Retrain ML after new resolved trades
                                self.ml.train(self.resolved)

                except Exception as e:
                    if "429" not in str(e):
                        print(f"[RESOLVE] API error: {e}")
            elif age_min > 15:
                trade["status"] = "closed"
                trade["pnl"] = -trade["size_usd"]
                self.stats["losses"] += 1
                self.stats["pnl"] += trade["pnl"]
                self.resolved.append(trade)
                del self.trades[tid]
                print(f"[LOSS] {trade['side']} ${trade['pnl']:+.2f} | aged out")

    # ========================================================================
    # ENTRY
    # ========================================================================

    async def find_entries(self, candles: list, markets: list):
        now = datetime.now(timezone.utc)
        hour_utc = now.hour
        open_count = sum(1 for t in self.trades.values() if t.get("status") == "open")
        if open_count >= MAX_CONCURRENT:
            return

        # Get TA signal
        sig = get_signal(candles)
        if not sig:
            return

        btc_price = sig["price"]
        entered = False

        for market in markets:
            if open_count >= MAX_CONCURRENT:
                break

            time_left = market.get("_time_left", 99)
            if time_left < TIME_WINDOW[0] or time_left > TIME_WINDOW[1]:
                continue

            cid = market.get("conditionId", "")
            question = market.get("question", "")
            nid = market.get("id")

            if f"ema5m_{cid}_UP" in self.trades or f"ema5m_{cid}_DOWN" in self.trades:
                continue

            # Compute edge
            side, entry_price, p_hat, edge, tte = self.compute_edge(sig, market)
            if not side:
                continue

            # ML gate
            features = self.ml.extract_features(
                sig, entry_price, edge, tte, hour_utc
            )
            take, ml_prob, ml_boost = self.ml.should_take_trade(features)

            if not take:
                self.stats["ml_filtered"] = self.stats.get("ml_filtered", 0) + 1
                if not entered:  # only log once per cycle
                    print(f"[ML-SKIP] {sig['dir'].upper()} | "
                          f"edge={edge:.1%} ml={ml_prob:.0%} < {ML_MIN_CONFIDENCE:.0%} | "
                          f"imp={sig['impulse']:.2f} gap={sig['ema_gap_bp']:.0f}bp rsi={sig['rsi']:.0f}")
                continue

            if ml_boost:
                self.stats["ml_boosted"] = self.stats.get("ml_boosted", 0) + 1

            # Size based on edge + ML boost
            trade_size = position_size_from_edge(edge, ml_boost=ml_boost)

            # Record trade
            trade_key = f"ema5m_{cid}_{side}"
            order_id = None

            if not self.paper and self.client:
                try:
                    from py_clob_client.clob_types import OrderArgs, OrderType
                    from py_clob_client.order_builder.constants import BUY

                    up_tid, down_tid = self.get_token_ids(market)
                    token_id = up_tid if side == "UP" else down_tid
                    if not token_id:
                        continue

                    shares = round(trade_size / entry_price, 2)
                    order_args = OrderArgs(
                        price=round(entry_price, 2),
                        size=shares,
                        side=BUY,
                        token_id=token_id,
                    )
                    signed = self.client.create_order(order_args)
                    resp = self.client.post_order(signed, OrderType.GTC)
                    if resp.get("success"):
                        order_id = resp.get("orderID", "?")
                    else:
                        print(f"[LIVE] Order failed: {resp.get('errorMsg', '?')}")
                        continue
                except Exception as e:
                    print(f"[LIVE] Order error: {e}")
                    continue

            self.trades[trade_key] = {
                "side": side,
                "entry_price": entry_price,
                "size_usd": trade_size,
                "entry_time": now.isoformat(),
                "condition_id": cid,
                "market_numeric_id": nid,
                "title": question,
                "strategy": "ema_rsi_5m",
                "edge": round(edge, 4),
                "p_hat": round(p_hat, 4),
                "impulse": round(sig["impulse"], 3),
                "ema_gap_bp": round(sig["ema_gap_bp"], 1),
                "rsi": round(sig["rsi"], 1),
                "btc_price": btc_price,
                "tte_seconds": tte,
                "ml_prob": round(ml_prob, 3),
                "ml_features": features,
                "order_id": order_id,
                "status": "open",
                "pnl": 0.0,
            }
            open_count += 1
            entered = True

            mode = "LIVE" if not self.paper else "PAPER"
            ml_tag = f" ML={ml_prob:.0%}" if self.ml.model else ""
            boost_tag = " BOOST" if ml_boost else ""
            print(f"[ENTRY] {side} ${entry_price:.2f} ${trade_size:.0f} | "
                  f"edge={edge:.1%} p_hat={p_hat:.2f}{ml_tag}{boost_tag} | "
                  f"imp={sig['impulse']:.2f} gap={sig['ema_gap_bp']:.0f}bp rsi={sig['rsi']:.0f} | "
                  f"tte={tte}s | BTC=${btc_price:,.0f} | [{mode}]")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        mode = "PAPER" if self.paper else "LIVE"
        print("=" * 72)
        print(f"  EMA/RSI 5M DIRECTIONAL TRADER V1.0 - {mode} MODE")
        print(f"  Strategy: EMA(9/21) + RSI(14) + ATR impulse -> Polymarket 5M")
        print(f"  Edge gate: {EDGE_MIN:.0%} min / {EDGE_STRONG:.0%} late")
        print(f"  ML: GradientBoosting filter (activates after {ML_MIN_TRADES} trades)")
        print(f"  Size: ${MIN_BET}-${MAX_BET} (edge-scaled) | Max concurrent: {MAX_CONCURRENT}")
        print(f"  Impulse: {IMPULSE_MIN}-{IMPULSE_STRONG} ATR | EMA gap: {EMA_GAP_MIN_BP}bp min")
        print(f"  RSI: >{RSI_LONG} long / <{RSI_SHORT} short")
        print(f"  ML model: {'LOADED' if self.ml.model else 'COLLECTING DATA'}")
        print("=" * 72)

        if self.resolved:
            w, l = self.stats["wins"], self.stats["losses"]
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            print(f"[RESUME] {w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
                  f"ML filtered: {self.stats.get('ml_filtered', 0)}")

        cycle = 0
        while True:
            try:
                cycle += 1

                candles, markets = await asyncio.gather(
                    self.fetch_candles(),
                    self.discover_5m_markets(),
                )

                if not candles:
                    print(f"[WARN] No candles")
                    await asyncio.sleep(10)
                    continue

                # Resolve expired trades
                await self.resolve_trades()

                # Find entries
                await self.find_entries(candles, markets)

                # Status every 10 cycles (~4 min)
                if cycle % 10 == 0:
                    w, l = self.stats["wins"], self.stats["losses"]
                    total = w + l
                    wr = w / total * 100 if total > 0 else 0
                    btc = candles[-1]["close"]
                    last = candles[-1]
                    sig_state = "NONE"
                    ef, es, r = last["ema_fast"], last["ema_slow"], last["rsi"]
                    if not np.isnan(r):
                        if ef > es and r >= RSI_LONG:
                            sig_state = "LONG"
                        elif ef < es and r <= RSI_SHORT:
                            sig_state = "SHORT"

                    ml_status = "OFF"
                    if self.ml.model:
                        ml_status = f"ON ({self.ml.trades_at_last_train}T)"

                    print(f"\n--- Cycle {cycle} | {mode} | "
                          f"Active: {len(self.trades)} | "
                          f"{w}W/{l}L {wr:.0f}%WR | "
                          f"PnL: ${self.stats['pnl']:+.2f} | "
                          f"BTC: ${btc:,.0f} | sig={sig_state} | "
                          f"ML: {ml_status} | "
                          f"mkt: {len(markets)} ---")

                self._save()
                await asyncio.sleep(SCAN_INTERVAL)

            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Saving...")
                self._save()
                break
            except Exception as e:
                print(f"[ERROR] Cycle {cycle}: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(15)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Live mode (real orders)")
    parser.add_argument("--paper", action="store_true", help="Paper mode (default)")
    args = parser.parse_args()

    lock = acquire_pid_lock("ema_rsi_5m")
    try:
        trader = EmaRsi5MTrader(paper=not args.live)
        asyncio.run(trader.run())
    finally:
        release_pid_lock("ema_rsi_5m")
