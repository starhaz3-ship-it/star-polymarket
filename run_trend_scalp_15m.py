"""
15M BTC Trend Scalper Paper Trader V1.0

Strategy: EMA stack (12/26/55) + VWAP trend regime -> StochRSI pullback entries.
High trade count from frequent pullback triggers within established trends.
High WR bias from:
  (1) EMA stack + VWAP trend confirmation,
  (2) StochRSI pullback-and-resume triggers,
  (3) BB width vol filter (avoid dead chop),
  (4) Edge gate vs Polymarket market price.

Usage:
  python run_trend_scalp_15m.py          # Paper mode (default)
  python run_trend_scalp_15m.py --live   # Live mode (real CLOB orders)
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
from typing import Optional, Tuple, List, Dict

import httpx

print = fn_partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# STRATEGY PARAMS
# ============================================================================
# Trend regime
EMA_FAST = 12
EMA_MID = 26
EMA_SLOW = 55
VWAP_LEN = 96            # ~1 day on 15m
MIN_BB_WIDTH = 0.010      # avoid dead chop

# Pullback trigger
RSI_LEN = 14
RSI_PULL_LONG = 52.0
RSI_PULL_SHORT = 48.0
STOCH_K_LOW = 0.35
STOCH_K_HIGH = 0.65

# Edge / execution
MIN_EDGE = 0.015          # 1.5% min edge

# Sizing
BASE_BET = 3.0
MAX_BET = 5.0
RISK_SCALE = 1.0

# ============================================================================
# PAPER TRADING CONFIG
# ============================================================================
SCAN_INTERVAL = 30
MAX_CONCURRENT = 3
TIME_WINDOW = (2.0, 12.0)    # minutes before expiry to enter
SPREAD_OFFSET = 0.03         # realistic fill spread
MIN_ENTRY_PRICE = 0.10
MAX_ENTRY_PRICE = 0.60
RESOLVE_AGE_MIN = 16.0       # 15m market + 1min grace
RESULTS_FILE = Path(__file__).parent / "trend_scalp_15m_results.json"

# ============================================================================
# INDICATORS
# ============================================================================
def calc_ema(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(arr, dtype=float)
    alpha = 2.0 / (n + 1.0)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def calc_sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if n <= 0:
        return out
    c = np.cumsum(arr, dtype=float)
    c[n:] = c[n:] - c[:-n]
    out[n - 1:] = c[n - 1:] / n
    return out


def calc_stddev(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        w = arr[i - n + 1:i + 1]
        out[i] = float(np.std(w, ddof=0))
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


def calc_stochrsi(close: np.ndarray, rsi_len: int = 14, stoch_len: int = 14, smooth: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    r = calc_rsi(close, rsi_len)
    k = np.full_like(close, np.nan, dtype=float)
    for i in range(stoch_len - 1, len(close)):
        w = r[i - stoch_len + 1:i + 1]
        lo = np.nanmin(w)
        hi = np.nanmax(w)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            continue
        k[i] = (r[i] - lo) / (hi - lo)
    k_clean = np.where(np.isnan(k), 0.0, k)
    k_s = calc_sma(k_clean, smooth)
    d_s = calc_sma(np.where(np.isnan(k_s), 0.0, k_s), smooth)
    return k_s, d_s


def calc_rolling_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, vol: np.ndarray, n: int = 96) -> np.ndarray:
    tp = (high + low + close) / 3.0
    pv = tp * vol
    out = np.full_like(close, np.nan, dtype=float)
    for i in range(n - 1, len(close)):
        v = float(np.sum(vol[i - n + 1:i + 1]))
        if v <= 0:
            continue
        out[i] = float(np.sum(pv[i - n + 1:i + 1])) / v
    return out


def calc_bb_width(close: np.ndarray, n: int = 20, mult: float = 2.0) -> np.ndarray:
    mid = calc_sma(close, n)
    sd = calc_stddev(close, n)
    up = mid + mult * sd
    dn = mid - mult * sd
    w = (up - dn) / np.where(mid == 0, np.nan, mid)
    return w


# ============================================================================
# SIGNAL LOGIC
# ============================================================================
def compute_features(candles: list) -> Optional[dict]:
    """Compute all indicator features from 15m candles."""
    if len(candles) < 60:
        return None

    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)
    vol = np.array([c.get("volume", 0.0) for c in candles], dtype=float)

    ef = calc_ema(close, EMA_FAST)
    em = calc_ema(close, EMA_MID)
    es = calc_ema(close, EMA_SLOW)
    vw = calc_rolling_vwap(high, low, close, vol, VWAP_LEN)
    bw = calc_bb_width(close, 20, 2.0)
    r = calc_rsi(close, RSI_LEN)
    k, d = calc_stochrsi(close, 14, 14, 3)

    return {
        "close": close, "high": high, "low": low, "vol": vol,
        "ema_fast": ef, "ema_mid": em, "ema_slow": es,
        "vwap": vw, "bb_width": bw,
        "rsi": r, "stoch_k": k, "stoch_d": d,
    }


def model_fair_prob_up(feats: dict) -> Optional[float]:
    """Convert technical state to fair probability for UP."""
    i = len(feats["close"]) - 1
    if i < 60:
        return None

    c = feats["close"][i]
    ef = feats["ema_fast"][i]
    em = feats["ema_mid"][i]
    es = feats["ema_slow"][i]
    vw = feats["vwap"][i]
    bw = feats["bb_width"][i]
    r = feats["rsi"][i]
    k = feats["stoch_k"][i]

    if any(not np.isfinite(x) for x in [c, ef, em, es, vw, bw, r, k]):
        return None

    trend_up = (ef > em > es) and (c >= vw)
    trend_dn = (ef < em < es) and (c <= vw)

    slope = (ef - es) / max(1e-9, c)

    p = 0.5
    if trend_up:
        p += 0.10
    elif trend_dn:
        p -= 0.10

    p += (r - 50.0) / 400.0
    p += np.clip(slope * 30.0, -0.06, 0.06)
    p += np.clip((bw - 0.012) * 2.0, -0.04, 0.04)

    return float(np.clip(p, 0.05, 0.95))


def get_signal(candles: list) -> Optional[dict]:
    """Generate trend scalp signal from 15m candles."""
    feats = compute_features(candles)
    if feats is None:
        return None

    i = len(feats["close"]) - 1
    c = feats["close"][i]
    ef = feats["ema_fast"][i]
    em = feats["ema_mid"][i]
    es = feats["ema_slow"][i]
    vw = feats["vwap"][i]
    bw = feats["bb_width"][i]
    r = feats["rsi"][i]
    k = feats["stoch_k"][i]
    k_prev = feats["stoch_k"][i - 1]

    if any(not np.isfinite(x) for x in [c, ef, em, es, vw, bw, r, k, k_prev]):
        return None

    # Vol regime filter
    if bw < MIN_BB_WIDTH:
        return None

    # Trend regime
    trend_up = (ef > em > es) and (c >= vw)
    trend_dn = (ef < em < es) and (c <= vw)

    # Pullback-and-resume triggers
    long_pull = trend_up and (r >= RSI_PULL_LONG) and (k_prev <= STOCH_K_LOW) and (k > STOCH_K_LOW)
    short_pull = trend_dn and (r <= RSI_PULL_SHORT) and (k_prev >= STOCH_K_HIGH) and (k < STOCH_K_HIGH)

    fair = model_fair_prob_up(feats)
    if fair is None:
        return None

    # Strength from EMA separation + BB width
    sep = abs(ef - es) / max(1e-9, c)
    strength = float(np.clip((sep * 40.0) + ((bw - MIN_BB_WIDTH) * 10.0), 0.0, 1.0))

    if long_pull:
        return {
            "dir": "long", "fair_up": fair, "strength": strength,
            "reason": "trend_up_pullback", "price": c,
            "ema_fast": ef, "ema_mid": em, "ema_slow": es,
            "vwap": vw, "bb_width": bw, "rsi": r, "stoch_k": k,
        }
    if short_pull:
        return {
            "dir": "short", "fair_up": fair, "strength": strength,
            "reason": "trend_dn_pullback", "price": c,
            "ema_fast": ef, "ema_mid": em, "ema_slow": es,
            "vwap": vw, "bb_width": bw, "rsi": r, "stoch_k": k,
        }
    return None


# ============================================================================
# TRADER
# ============================================================================
class TrendScalp15MTrader:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.client = None
        self.trades: dict = {}
        self.resolved: list = []
        self.stats = {
            "wins": 0, "losses": 0, "pnl": 0.0,
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
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
                w, l = self.stats["wins"], self.stats["losses"]
                total = w + l
                wr = w / total * 100 if total > 0 else 0
                print(f"[LOAD] {total} resolved ({w}W/{l}L {wr:.0f}%WR) | "
                      f"PnL: ${self.stats['pnl']:+.2f} | {len(self.trades)} active")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        data = {
            "active": self.trades,
            "resolved": self.resolved[-500:],
            "stats": self.stats,
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
        """Fetch 15m BTC/USDT candles from Binance."""
        url = "https://api.binance.com/api/v3/klines"
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    r = await client.get(url, params={
                        "symbol": "BTCUSDT", "interval": "15m", "limit": 200
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
                return candles

            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                print(f"[BINANCE] Error: {e}")
                return []

    async def discover_15m_markets(self) -> list:
        """Find active Polymarket 15M BTC Up/Down markets."""
        markets = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    "https://gamma-api.polymarket.com/events",
                    params={"tag_slug": "15M", "active": "true", "closed": "false", "limit": 50},
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
            print(f"[API] Error: {e}")
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
        """Returns (side, entry_price, fair_px, edge, tte_seconds) or Nones."""
        up_p, down_p = self.get_prices(market)
        if up_p is None or down_p is None:
            return None, None, 0.0, 0.0, 0

        tte = market.get("_tte_seconds", 0)
        if tte < 60:
            return None, None, 0.0, 0.0, 0

        fair_up = sig["fair_up"]

        if sig["dir"] == "long":
            side = "UP"
            market_px = float(up_p)
            fair_px = fair_up
        else:
            side = "DOWN"
            market_px = float(down_p)
            fair_px = 1.0 - fair_up  # fair DOWN = 1 - P(UP)

        edge = fair_px - market_px
        entry_price = round(market_px + SPREAD_OFFSET, 2)

        if entry_price < MIN_ENTRY_PRICE or entry_price > MAX_ENTRY_PRICE:
            return None, None, fair_px, edge, tte

        if edge < MIN_EDGE:
            return None, None, fair_px, edge, tte

        return side, entry_price, fair_px, edge, tte

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

                                print(f"[{tag}] {trade['side']} ${trade['pnl']:+.2f} | "
                                      f"entry=${trade['entry_price']:.2f} exit=${price:.2f} | "
                                      f"edge={trade.get('edge', 0):.1%} str={trade.get('strength', 0):.2f} | "
                                      f"reason={trade.get('reason', '?')} | "
                                      f"{w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f}")
                except Exception as e:
                    if "429" not in str(e):
                        print(f"[RESOLVE] API error: {e}")
            elif age_min > 25:
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
        open_count = sum(1 for t in self.trades.values() if t.get("status") == "open")
        if open_count >= MAX_CONCURRENT:
            return

        sig = get_signal(candles)
        if not sig:
            return

        btc_price = sig["price"]

        for market in markets:
            if open_count >= MAX_CONCURRENT:
                break

            time_left = market.get("_time_left", 99)
            if time_left < TIME_WINDOW[0] or time_left > TIME_WINDOW[1]:
                continue

            cid = market.get("conditionId", "")
            question = market.get("question", "")
            nid = market.get("id")

            if f"ts15_{cid}_UP" in self.trades or f"ts15_{cid}_DOWN" in self.trades:
                continue

            side, entry_price, fair_px, edge, tte = self.compute_edge(sig, market)
            if not side:
                continue

            # Sizing: base + strength scaling
            trade_size = BASE_BET + RISK_SCALE * sig["strength"] * (MAX_BET - BASE_BET)
            trade_size = max(BASE_BET, min(MAX_BET, trade_size))
            trade_size = round(trade_size, 2)

            trade_key = f"ts15_{cid}_{side}"
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
                "strategy": "trend_scalp_15m",
                "edge": round(edge, 4),
                "fair_px": round(fair_px, 4),
                "strength": round(sig["strength"], 3),
                "reason": sig["reason"],
                "rsi": round(sig["rsi"], 1),
                "stoch_k": round(sig["stoch_k"], 3),
                "bb_width": round(sig["bb_width"], 4),
                "btc_price": btc_price,
                "tte_seconds": tte,
                "order_id": order_id,
                "status": "open",
                "pnl": 0.0,
            }
            open_count += 1

            mode = "LIVE" if not self.paper else "PAPER"
            print(f"[ENTRY] {side} ${entry_price:.2f} ${trade_size:.0f} | "
                  f"edge={edge:.1%} fair={fair_px:.2f} str={sig['strength']:.2f} | "
                  f"{sig['reason']} | rsi={sig['rsi']:.0f} stK={sig['stoch_k']:.2f} bbw={sig['bb_width']:.3f} | "
                  f"tte={tte}s | BTC=${btc_price:,.0f} | [{mode}]")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        mode = "PAPER" if self.paper else "LIVE"
        print("=" * 72)
        print(f"  15M BTC TREND SCALPER V1.0 - {mode} MODE")
        print(f"  Strategy: EMA({EMA_FAST}/{EMA_MID}/{EMA_SLOW}) + VWAP trend -> StochRSI pullback")
        print(f"  Edge gate: {MIN_EDGE:.1%} min | BB width: {MIN_BB_WIDTH}")
        print(f"  RSI: >{RSI_PULL_LONG} long / <{RSI_PULL_SHORT} short")
        print(f"  StochRSI: K cross {STOCH_K_LOW} up (long) / {STOCH_K_HIGH} down (short)")
        print(f"  Size: ${BASE_BET}-${MAX_BET} (strength-scaled) | Max concurrent: {MAX_CONCURRENT}")
        print(f"  Entry window: {TIME_WINDOW[0]}-{TIME_WINDOW[1]} min before close")
        print("=" * 72)

        if self.resolved:
            w, l = self.stats["wins"], self.stats["losses"]
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            print(f"[RESUME] {w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f}")

        cycle = 0
        while True:
            try:
                cycle += 1

                candles, markets = await asyncio.gather(
                    self.fetch_candles(),
                    self.discover_15m_markets(),
                )

                if not candles:
                    print(f"[WARN] No candles")
                    await asyncio.sleep(10)
                    continue

                await self.resolve_trades()
                await self.find_entries(candles, markets)

                # Status every 10 cycles (~5 min)
                if cycle % 10 == 0:
                    w, l = self.stats["wins"], self.stats["losses"]
                    total = w + l
                    wr = w / total * 100 if total > 0 else 0
                    btc = candles[-1]["close"]
                    last = candles[-1]

                    # Quick trend state
                    close_arr = np.array([c["close"] for c in candles[-60:]], dtype=float)
                    ef = calc_ema(close_arr, EMA_FAST)[-1]
                    em = calc_ema(close_arr, EMA_MID)[-1]
                    es = calc_ema(close_arr, EMA_SLOW)[-1]
                    trend = "UP" if ef > em > es else ("DOWN" if ef < em < es else "CHOP")

                    print(f"\n--- Cycle {cycle} | {mode} | "
                          f"Active: {len(self.trades)} | "
                          f"{w}W/{l}L {wr:.0f}%WR | "
                          f"PnL: ${self.stats['pnl']:+.2f} | "
                          f"BTC: ${btc:,.0f} | trend={trend} | "
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

    lock = acquire_pid_lock("trend_scalp_15m")
    try:
        trader = TrendScalp15MTrader(paper=not args.live)
        asyncio.run(trader.run())
    finally:
        release_pid_lock("trend_scalp_15m")
