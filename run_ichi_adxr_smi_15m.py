"""
15M BTC Ichimoku+ADXR+SMI Paper Trader V1.0

Strategy: Ichimoku (Tenkan/Kijun + Cloud) trend regime + ADXR trend persistence
+ SMI pullback/resume timing.

Usage:
  python run_ichi_adxr_smi_15m.py          # Paper mode (default)
  python run_ichi_adxr_smi_15m.py --live   # Live mode (real CLOB orders)
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
# STRATEGY PARAMS
# ============================================================================
# Ichimoku
TENKAN_LEN = 9
KIJUN_LEN = 26
SENKOU_B_LEN = 52

# ADX / ADXR
DMI_LEN = 14
ADXR_MIN = 18.0
DI_MARGIN = 2.0

# SMI timing
SMI_LEN = 14
SMI_S1 = 3
SMI_S2 = 3
SMI_PULL_LONG = -20.0
SMI_PULL_SHORT = 20.0
SMI_CROSS_BUFFER = 3.0

# Vol filter
ATR_LEN = 14
ATRP_MIN = 0.0009
ATRP_MAX = 0.0400

# Edge & exits
MIN_EDGE = 0.010
TP_EDGE = 0.018
SL_EDGE = 0.014
MAX_HOLD_BARS = 7

# Sizing
BASE_BET = 3.0
MAX_BET = 8.0
RISK_SCALE = 1.0

# ============================================================================
# PAPER TRADING CONFIG
# ============================================================================
SCAN_INTERVAL = 30
MAX_CONCURRENT = 3
TIME_WINDOW = (2.0, 12.0)
SPREAD_OFFSET = 0.03
MIN_ENTRY_PRICE = 0.10
MAX_ENTRY_PRICE = 0.60
RESOLVE_AGE_MIN = 16.0
RESULTS_FILE = Path(__file__).parent / "ichi_adxr_smi_15m_results.json"

# ============================================================================
# INDICATORS
# ============================================================================

def calc_sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    if n <= 0 or len(x) < n:
        return out
    c = np.cumsum(x, dtype=float)
    c[n:] = c[n:] - c[:-n]
    out[n - 1:] = c[n - 1:] / n
    return out


def calc_ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    a = 2.0 / (n + 1.0)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
    return out


def calc_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    out = np.empty_like(close, dtype=float)
    out[0] = high[0] - low[0]
    for i in range(1, len(close)):
        out[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return out


def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    return calc_sma(calc_true_range(high, low, close), n)


def rolling_hhll(high: np.ndarray, low: np.ndarray, n: int):
    hh = np.full_like(high, np.nan, dtype=float)
    ll = np.full_like(low, np.nan, dtype=float)
    for i in range(n - 1, len(high)):
        hh[i] = float(np.max(high[i - n + 1:i + 1]))
        ll[i] = float(np.min(low[i - n + 1:i + 1]))
    return hh, ll


def calc_ichimoku(high: np.ndarray, low: np.ndarray, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
    hh_t, ll_t = rolling_hhll(high, low, tenkan)
    hh_k, ll_k = rolling_hhll(high, low, kijun)
    hh_b, ll_b = rolling_hhll(high, low, senkou_b)
    ten = (hh_t + ll_t) / 2.0
    kij = (hh_k + ll_k) / 2.0
    spa = (ten + kij) / 2.0
    spb = (hh_b + ll_b) / 2.0
    return ten, kij, spa, spb


def calc_dmi_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14):
    up_move = np.diff(high, prepend=high[0])
    dn_move = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    tr = calc_true_range(high, low, close)
    atrn = calc_sma(tr, n)
    safe_atr = np.where(atrn == 0, np.nan, atrn)
    plus_di = 100.0 * calc_sma(plus_dm, n) / safe_atr
    minus_di = 100.0 * calc_sma(minus_dm, n) / safe_atr
    dx = 100.0 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, (plus_di + minus_di))
    adx = calc_sma(np.nan_to_num(dx, nan=0.0), n)
    return plus_di, minus_di, adx


def calc_adxr(adx: np.ndarray, n: int = 14) -> np.ndarray:
    out = np.full_like(adx, np.nan, dtype=float)
    for i in range(len(adx)):
        if i - n >= 0 and np.isfinite(adx[i]) and np.isfinite(adx[i - n]):
            out[i] = 0.5 * (adx[i] + adx[i - n])
    return out


def calc_smi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
             length: int = 14, smooth1: int = 3, smooth2: int = 3):
    hh, ll = rolling_hhll(high, low, length)
    m = (hh + ll) / 2.0
    d = close - m
    r = hh - ll
    d1 = calc_ema(np.nan_to_num(d, nan=0.0), smooth1)
    d2 = calc_ema(d1, smooth2)
    r1 = calc_ema(np.nan_to_num(r, nan=0.0), smooth1)
    r2 = calc_ema(r1, smooth2)
    denom = 0.5 * r2
    smi_val = 100.0 * d2 / np.where(denom == 0, np.nan, denom)
    sig = calc_ema(np.nan_to_num(smi_val, nan=0.0), smooth2)
    return smi_val, sig


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def get_signal(candles: list) -> Optional[dict]:
    """Compute Ichimoku+ADXR+SMI signal."""
    min_len = max(SENKOU_B_LEN, DMI_LEN * 3, SMI_LEN + SMI_S1 + SMI_S2, ATR_LEN) + 10
    if len(candles) < min_len:
        return None

    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)

    ten, kij, spa, spb = calc_ichimoku(high, low, TENKAN_LEN, KIJUN_LEN, SENKOU_B_LEN)
    pdi, mdi, adx = calc_dmi_adx(high, low, close, DMI_LEN)
    axr = calc_adxr(adx, DMI_LEN)
    sm, sm_sig = calc_smi(high, low, close, SMI_LEN, SMI_S1, SMI_S2)
    a = calc_atr(high, low, close, ATR_LEN)

    i = len(close) - 1
    c = close[i]
    vals = [c, ten[i], kij[i], spa[i], spb[i], pdi[i], mdi[i], axr[i], sm[i], sm[i-1], sm_sig[i], a[i]]
    if any(not np.isfinite(x) for x in vals):
        return None

    atrp = a[i] / max(1e-9, c)
    if atrp < ATRP_MIN or atrp > ATRP_MAX:
        return None

    if axr[i] < ADXR_MIN:
        return None

    cloud_top = max(spa[i], spb[i])
    cloud_bot = min(spa[i], spb[i])

    # Trend regimes
    up_regime = (c > cloud_top) and (ten[i] >= kij[i]) and ((pdi[i] - mdi[i]) >= DI_MARGIN)
    dn_regime = (c < cloud_bot) and (ten[i] <= kij[i]) and ((mdi[i] - pdi[i]) >= DI_MARGIN)

    # SMI pullback triggers
    long_pull = (up_regime
                 and sm[i-1] <= SMI_PULL_LONG
                 and sm[i] > (SMI_PULL_LONG + SMI_CROSS_BUFFER)
                 and sm[i] >= sm_sig[i])
    short_pull = (dn_regime
                  and sm[i-1] >= SMI_PULL_SHORT
                  and sm[i] < (SMI_PULL_SHORT - SMI_CROSS_BUFFER)
                  and sm[i] <= sm_sig[i])

    # Fair probability model
    cloud_thick = (cloud_top - cloud_bot) / max(1e-9, c)
    vol_pen = float(np.clip((atrp - 0.02) * 1.2, 0.0, 0.08))
    p = 0.50
    if c > cloud_top and ten[i] > kij[i]:
        p += 0.10
    elif c < cloud_bot and ten[i] < kij[i]:
        p -= 0.10
    p += float(np.clip(((pdi[i] - mdi[i]) / 100.0) * 0.10, -0.10, 0.10))
    p += float(np.clip((axr[i] - ADXR_MIN) * 0.003, -0.03, 0.06))
    p += float(np.clip(sm[i] / 300.0, -0.10, 0.10))
    p -= float(np.clip(cloud_thick * 3.0, 0.0, 0.05))
    p -= vol_pen
    fair_up = float(np.clip(p, 0.05, 0.95))

    # Strength
    strength = 0.0
    strength += min(1.0, abs(pdi[i] - mdi[i]) / 20.0) * 0.45
    strength += min(1.0, max(0.0, (axr[i] - ADXR_MIN) / 15.0)) * 0.25
    strength += min(1.0, abs(sm[i]) / 60.0) * 0.30
    strength = float(np.clip(strength, 0.0, 1.0))

    side = None
    reason = "no_entry"
    if long_pull:
        side = "UP"
        reason = "ichi_up_adxr_ok_smi_pullback"
    elif short_pull:
        side = "DOWN"
        reason = "ichi_dn_adxr_ok_smi_pullback"

    return {
        "side": side,
        "strength": strength,
        "fair_up": fair_up,
        "reason": reason,
        "price": c,
        "tenkan": ten[i],
        "kijun": kij[i],
        "cloud_top": cloud_top,
        "cloud_bot": cloud_bot,
        "pdi": pdi[i],
        "mdi": mdi[i],
        "adxr": axr[i],
        "smi": sm[i],
        "smi_sig": sm_sig[i],
        "atrp": atrp,
    }


# ============================================================================
# TRADER
# ============================================================================

class IchiAdxrSmi15MTrader:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.client = None
        self.trades = {}
        self.resolved = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0}

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
    # DATA FETCHING (sync)
    # ========================================================================

    def fetch_candles(self) -> list:
        url = "https://api.binance.com/api/v3/klines"
        for attempt in range(3):
            try:
                r = httpx.get(url, params={
                    "symbol": "BTCUSDT", "interval": "15m", "limit": 200
                }, timeout=15)
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
                    time.sleep(3 * (attempt + 1))
                    continue
                print(f"[BINANCE] Error: {e}")
                return []

    def discover_15m_markets(self) -> list:
        markets = []
        try:
            r = httpx.get(
                "https://gamma-api.polymarket.com/events",
                params={"tag_slug": "15M", "active": "true", "closed": "false", "limit": 50},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
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

    def compute_edge(self, sig: dict, market: dict):
        up_p, down_p = self.get_prices(market)
        if up_p is None or down_p is None:
            return None, None, 0.0, 0.0, 0

        tte = market.get("_tte_seconds", 0)
        if tte < 60:
            return None, None, 0.0, 0.0, 0

        fair_up = sig["fair_up"]
        fair_down = 1.0 - fair_up
        side = sig["side"]

        if side == "UP":
            entry_price = up_p + SPREAD_OFFSET
            if entry_price < MIN_ENTRY_PRICE or entry_price > MAX_ENTRY_PRICE:
                return None, None, 0.0, 0.0, 0
            edge = fair_up - entry_price
            if edge < MIN_EDGE:
                return None, None, fair_up, edge, tte
            return "UP", entry_price, fair_up, edge, tte

        elif side == "DOWN":
            entry_price = down_p + SPREAD_OFFSET
            if entry_price < MIN_ENTRY_PRICE or entry_price > MAX_ENTRY_PRICE:
                return None, None, 0.0, 0.0, 0
            edge = fair_down - entry_price
            if edge < MIN_EDGE:
                return None, None, fair_down, edge, tte
            return "DOWN", entry_price, fair_down, edge, tte

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
                    r = httpx.get(
                        f"https://gamma-api.polymarket.com/markets/{nid}",
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=8,
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

    def find_entries(self, candles: list, markets: list):
        now = datetime.now(timezone.utc)
        open_count = sum(1 for t in self.trades.values() if t.get("status") == "open")
        if open_count >= MAX_CONCURRENT:
            return

        sig = get_signal(candles)
        if not sig or sig["side"] is None:
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

            if f"ias15_{cid}_UP" in self.trades or f"ias15_{cid}_DOWN" in self.trades:
                continue

            side, entry_price, fair_px, edge, tte = self.compute_edge(sig, market)
            if not side:
                continue

            trade_size = BASE_BET + RISK_SCALE * sig["strength"] * (MAX_BET - BASE_BET)
            trade_size = max(BASE_BET, min(MAX_BET, trade_size))
            trade_size = round(trade_size, 2)

            trade_key = f"ias15_{cid}_{side}"
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
                    order_args = OrderArgs(price=round(entry_price, 2), size=shares,
                                           side=BUY, token_id=token_id)
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
                "strategy": "ichi_adxr_smi_15m",
                "edge": round(edge, 4),
                "fair_px": round(fair_px, 4),
                "strength": round(sig["strength"], 3),
                "reason": sig["reason"],
                "tenkan": round(sig["tenkan"], 2),
                "kijun": round(sig["kijun"], 2),
                "adxr": round(sig["adxr"], 1),
                "smi": round(sig["smi"], 1),
                "pdi": round(sig["pdi"], 1),
                "mdi": round(sig["mdi"], 1),
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
                  f"{sig['reason']} | ten={sig['tenkan']:.0f} kij={sig['kijun']:.0f} "
                  f"adxr={sig['adxr']:.0f} smi={sig['smi']:.0f} +di={sig['pdi']:.0f} -di={sig['mdi']:.0f} | "
                  f"tte={tte}s | BTC=${btc_price:,.0f} | [{mode}]")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    def run(self):
        mode = "PAPER" if self.paper else "LIVE"
        print("=" * 72)
        print(f"  15M BTC ICHIMOKU+ADXR+SMI V1.0 - {mode} MODE")
        print(f"  Strategy: Ichimoku({TENKAN_LEN}/{KIJUN_LEN}/{SENKOU_B_LEN}) + ADXR({DMI_LEN}) + SMI({SMI_LEN})")
        print(f"  Edge gate: {MIN_EDGE:.1%} min | TP: +{TP_EDGE:.1%} | SL: -{SL_EDGE:.1%}")
        print(f"  ADXR min: {ADXR_MIN} | DI margin: {DI_MARGIN}")
        print(f"  SMI pullback: long<{SMI_PULL_LONG} short>{SMI_PULL_SHORT} buf={SMI_CROSS_BUFFER}")
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
                candles = self.fetch_candles()
                markets = self.discover_15m_markets()

                if not candles:
                    print("[WARN] No candles")
                    time.sleep(10)
                    continue

                self.resolve_trades()
                self.find_entries(candles, markets)

                if cycle % 10 == 0:
                    w, l = self.stats["wins"], self.stats["losses"]
                    total = w + l
                    wr = w / total * 100 if total > 0 else 0
                    btc = candles[-1]["close"]

                    close_arr = np.array([c["close"] for c in candles[-80:]], dtype=float)
                    high_arr = np.array([c["high"] for c in candles[-80:]], dtype=float)
                    low_arr = np.array([c["low"] for c in candles[-80:]], dtype=float)
                    _, _, adx_arr = calc_dmi_adx(high_arr, low_arr, close_arr, DMI_LEN)
                    axr_arr = calc_adxr(adx_arr, DMI_LEN)
                    sm_arr, _ = calc_smi(high_arr, low_arr, close_arr, SMI_LEN, SMI_S1, SMI_S2)
                    axr_v = axr_arr[-1] if np.isfinite(axr_arr[-1]) else 0
                    sm_v = sm_arr[-1] if np.isfinite(sm_arr[-1]) else 0
                    trend = "TREND" if axr_v >= ADXR_MIN else "CHOP"

                    print(f"\n--- Cycle {cycle} | {mode} | "
                          f"Active: {len(self.trades)} | "
                          f"{w}W/{l}L {wr:.0f}%WR | "
                          f"PnL: ${self.stats['pnl']:+.2f} | "
                          f"BTC: ${btc:,.0f} adxr={axr_v:.0f} smi={sm_v:.0f} | "
                          f"{trend} | mkt: {len(markets)} ---")

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
    parser.add_argument("--live", action="store_true", help="Live mode (real orders)")
    parser.add_argument("--paper", action="store_true", help="Paper mode (default)")
    args = parser.parse_args()

    lock = acquire_pid_lock("ichi_adxr_smi_15m")
    try:
        trader = IchiAdxrSmi15MTrader(paper=not args.live)
        trader.run()
    finally:
        release_pid_lock("ichi_adxr_smi_15m")
