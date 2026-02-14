"""
15M BTC FRAMA+CMO+Donchian Paper Trader V1.0

Strategy: FRAMA adaptive trend + CMO momentum + Donchian pullback entries.
Different indicator set from trend_scalp_15m (EMA stack + StochRSI).

- FRAMA: fractal adaptive moving average (trend baseline)
- CMO: Chande Momentum Oscillator (momentum confirmation)
- Donchian Channel: structure + pullback timing

Usage:
  python run_frama_cmo_15m.py          # Paper mode (default)
  python run_frama_cmo_15m.py --live   # Live mode (real CLOB orders)
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
FRAMA_LEN = 16
CMO_LEN = 14
DONCH_LEN = 20

# Trend / momentum thresholds
CMO_LONG_MIN = 10.0
CMO_SHORT_MAX = -10.0
PULLBACK_EPS = 0.0012       # ~0.12% proximity to FRAMA/mid for pullback

# Edge & exits
MIN_EDGE = 0.010             # 1.0% min edge
TP_EDGE = 0.020              # +2.0% contract move
SL_EDGE = 0.015              # -1.5% contract move
MAX_HOLD_BARS = 7            # ~1h45m

# Sizing
BASE_BET = 3.0
MAX_BET = 8.0
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
RESULTS_FILE = Path(__file__).parent / "frama_cmo_15m_results.json"

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


def calc_donchian(high: np.ndarray, low: np.ndarray, n: int):
    up = np.full_like(high, np.nan, dtype=float)
    dn = np.full_like(low, np.nan, dtype=float)
    mid = np.full_like(high, np.nan, dtype=float)
    for i in range(n - 1, len(high)):
        hh = float(np.max(high[i - n + 1:i + 1]))
        ll = float(np.min(low[i - n + 1:i + 1]))
        up[i] = hh
        dn[i] = ll
        mid[i] = (hh + ll) / 2.0
    return up, dn, mid


def calc_cmo(close: np.ndarray, n: int = 14) -> np.ndarray:
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


def calc_frama(close: np.ndarray, n: int = 16, fast: float = 4.0, slow: float = 300.0) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=float)
    if len(close) < n + 2 or n < 4:
        return out

    n2 = n // 2
    w = float(n)
    alpha_fast = 2.0 / (fast + 1.0)
    alpha_slow = 2.0 / (slow + 1.0)

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
            alpha = float(np.clip(alpha, alpha_slow, alpha_fast))
        else:
            alpha = alpha_slow

        prev = out[i - 1] if np.isfinite(out[i - 1]) else close[i - 1]
        out[i] = alpha * close[i] + (1.0 - alpha) * prev

    return out


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def get_signal(candles: list) -> Optional[dict]:
    """Compute FRAMA+CMO+Donchian signal from candles."""
    if len(candles) < max(FRAMA_LEN, CMO_LEN, DONCH_LEN) + 10:
        return None

    close = np.array([c["close"] for c in candles], dtype=float)
    high = np.array([c["high"] for c in candles], dtype=float)
    low = np.array([c["low"] for c in candles], dtype=float)

    f = calc_frama(close, FRAMA_LEN)
    m = calc_cmo(close, CMO_LEN)
    dc_up, dc_dn, dc_mid = calc_donchian(high, low, DONCH_LEN)

    i = len(close) - 1
    c = close[i]
    fv = f[i]
    mv = m[i]
    mid = dc_mid[i]

    if any(not np.isfinite(x) for x in [c, fv, mv, mid]):
        return None

    # Trend regime
    trend_up = (c > fv) and (mv >= CMO_LONG_MIN)
    trend_dn = (c < fv) and (mv <= CMO_SHORT_MAX)

    # Pullback: price near FRAMA or Donchian mid
    near_frama = abs(c - fv) / max(1e-9, c) <= PULLBACK_EPS
    near_mid = abs(c - mid) / max(1e-9, c) <= PULLBACK_EPS

    # Resume hint: close pushing away from midline in trend direction
    prev_c = close[i - 1]
    resume_long = (c > prev_c) and (c > mid)
    resume_short = (c < prev_c) and (c < mid)

    # Fair probability model
    p = 0.50
    p += float(np.clip((c - fv) / max(1e-9, c) * 8.0, -0.12, 0.12))
    p += float(np.clip(mv / 200.0, -0.10, 0.10))
    p += float(np.clip((c - mid) / max(1e-9, c) * 6.0, -0.08, 0.08))
    fair_up = float(np.clip(p, 0.05, 0.95))

    # Strength: momentum magnitude + distance from FRAMA
    dist = abs(c - fv) / max(1e-9, c)
    strength = float(np.clip(
        (abs(mv) / 80.0) * 0.6
        + np.clip(dist / (PULLBACK_EPS + 1e-9), 0, 2) * 0.2
        + 0.2,
        0.0, 1.0
    ))

    side = None
    reason = "no_entry"

    if trend_up and (near_frama or near_mid) and resume_long:
        side = "UP"
        reason = "frama_up_pullback_resume"
    elif trend_dn and (near_frama or near_mid) and resume_short:
        side = "DOWN"
        reason = "frama_dn_pullback_resume"

    return {
        "side": side,
        "strength": strength,
        "fair_up": fair_up,
        "reason": reason,
        "price": c,
        "frama": fv,
        "cmo": mv,
        "dc_mid": mid,
        "dc_up": dc_up[i],
        "dc_dn": dc_dn[i],
    }


# ============================================================================
# TRADER
# ============================================================================

class FramaCmo15MTrader:
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
                print(f"[WARN] CLOB init failed: {e} — falling back to paper")
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
    # DATA FETCHING (sync — Windows asyncio hangs with httpx.AsyncClient)
    # ========================================================================

    def fetch_candles(self) -> list:
        """Fetch 15m BTC/USDT candles from Binance."""
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
        """Find active Polymarket 15M BTC Up/Down markets."""
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

    def compute_edge(self, sig: dict, market: dict):
        """Returns (side, entry_price, fair_px, edge, tte_seconds) or Nones."""
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

            if f"fc15_{cid}_UP" in self.trades or f"fc15_{cid}_DOWN" in self.trades:
                continue

            side, entry_price, fair_px, edge, tte = self.compute_edge(sig, market)
            if not side:
                continue

            # Sizing: base + strength scaling
            trade_size = BASE_BET + RISK_SCALE * sig["strength"] * (MAX_BET - BASE_BET)
            trade_size = max(BASE_BET, min(MAX_BET, trade_size))
            trade_size = round(trade_size, 2)

            trade_key = f"fc15_{cid}_{side}"
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
                "strategy": "frama_cmo_15m",
                "edge": round(edge, 4),
                "fair_px": round(fair_px, 4),
                "strength": round(sig["strength"], 3),
                "reason": sig["reason"],
                "frama": round(sig["frama"], 2),
                "cmo": round(sig["cmo"], 1),
                "dc_mid": round(sig["dc_mid"], 2),
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
                  f"{sig['reason']} | frama={sig['frama']:.0f} cmo={sig['cmo']:.1f} dcmid={sig['dc_mid']:.0f} | "
                  f"tte={tte}s | BTC=${btc_price:,.0f} | [{mode}]")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    def run(self):
        mode = "PAPER" if self.paper else "LIVE"
        print("=" * 72)
        print(f"  15M BTC FRAMA+CMO+DONCHIAN V1.0 - {mode} MODE")
        print(f"  Strategy: FRAMA({FRAMA_LEN}) + CMO({CMO_LEN}) + Donchian({DONCH_LEN})")
        print(f"  Edge gate: {MIN_EDGE:.1%} min | TP: +{TP_EDGE:.1%} | SL: -{SL_EDGE:.1%}")
        print(f"  CMO: >{CMO_LONG_MIN} long / <{CMO_SHORT_MAX} short")
        print(f"  Pullback eps: {PULLBACK_EPS:.4f} ({PULLBACK_EPS*100:.2f}%)")
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
                    print(f"[WARN] No candles")
                    time.sleep(10)
                    continue

                self.resolve_trades()
                self.find_entries(candles, markets)

                # Status every 10 cycles (~5 min)
                if cycle % 10 == 0:
                    w, l = self.stats["wins"], self.stats["losses"]
                    total = w + l
                    wr = w / total * 100 if total > 0 else 0
                    btc = candles[-1]["close"]

                    # Quick FRAMA/CMO state
                    close_arr = np.array([c["close"] for c in candles[-60:]], dtype=float)
                    fv = calc_frama(close_arr, FRAMA_LEN)
                    mv = calc_cmo(close_arr, CMO_LEN)
                    fval = fv[-1] if np.isfinite(fv[-1]) else 0
                    mval = mv[-1] if np.isfinite(mv[-1]) else 0
                    trend = "UP" if btc > fval and mval > CMO_LONG_MIN else (
                        "DOWN" if btc < fval and mval < CMO_SHORT_MAX else "CHOP")

                    print(f"\n--- Cycle {cycle} | {mode} | "
                          f"Active: {len(self.trades)} | "
                          f"{w}W/{l}L {wr:.0f}%WR | "
                          f"PnL: ${self.stats['pnl']:+.2f} | "
                          f"BTC: ${btc:,.0f} frama=${fval:,.0f} cmo={mval:.0f} | "
                          f"trend={trend} | mkt: {len(markets)} ---")

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

    lock = acquire_pid_lock("frama_cmo_15m")
    try:
        trader = FramaCmo15MTrader(paper=not args.live)
        trader.run()
    finally:
        release_pid_lock("frama_cmo_15m")
