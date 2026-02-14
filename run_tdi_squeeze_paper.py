"""
TDI_SQUEEZE_15m Paper Trader V1.0

Single-strategy paper trader for TDI_SQUEEZE on 15-minute Polymarket BTC markets.
Uses 1-minute BTC candles for signal generation (TDI extreme + BB/KC squeeze release),
then trades 15M binary outcome markets.

Backtest results (14-day):
  - TDI_SQUEEZE_15m: 52.6% WR, 173 trades (best of 10 strategy instances)
  - 0.9 confidence signals: 67.5% WR on 40 trades
  - Best hours: 8 (83%), 10 (75%), 15 (100%), 22 (86%)

Usage:
    python run_tdi_squeeze_paper.py
"""

import sys
import json
import time
import math
import os
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from functools import partial as fn_partial
from typing import Optional, Tuple

import httpx

print = fn_partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# CONFIG
# ============================================================================
SCAN_INTERVAL = 25                # seconds between cycles
MAX_CONCURRENT = 3                # max open positions
TIME_WINDOW = (2.0, 12.0)        # minutes before expiry to enter
SPREAD_OFFSET = 0.03             # simulated spread
MIN_ENTRY_PRICE = 0.10
MAX_ENTRY_PRICE = 0.60
MIN_CONFIDENCE = 0.72            # minimum strategy confidence to trade
MIN_EDGE = 0.008                 # minimum edge over market price
RESOLVE_AGE_MIN = 16.0           # minutes before checking resolution
BASE_SIZE = 3.0
MAX_SIZE = 8.0

# Hours to skip (UTC) — worst hours from 14-day backtest
SKIP_HOURS_UTC = {2, 3, 4, 5, 6, 18, 23}

RESULTS_FILE = Path(__file__).parent / "tdi_squeeze_paper_results.json"

# ============================================================================
# TDI INDICATOR (numpy, stateless)
# ============================================================================

def _rsi_wilder(close: np.ndarray, period: int = 13) -> np.ndarray:
    """Wilder-smoothed RSI."""
    out = np.full(len(close), 50.0, dtype=float)
    if len(close) < period + 1:
        return out
    delta = np.diff(close, prepend=close[0])
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)

    avg_g = np.mean(gain[1:period + 1])
    avg_l = np.mean(loss[1:period + 1])

    for i in range(period + 1, len(close)):
        avg_g = (avg_g * (period - 1) + gain[i]) / period
        avg_l = (avg_l * (period - 1) + loss[i]) / period
        rs = avg_g / max(avg_l, 1e-10)
        out[i] = 100.0 - (100.0 / (1.0 + rs))

    # Fill initial RSI value
    if avg_g + avg_l > 0:
        rs0 = np.mean(gain[1:period + 1]) / max(np.mean(loss[1:period + 1]), 1e-10)
        out[period] = 100.0 - (100.0 / (1.0 + rs0))

    return out


def _sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < n:
        return out
    c = np.cumsum(arr, dtype=float)
    c[n:] = c[n:] - c[:-n]
    out[n - 1:] = c[n - 1:] / n
    return out


def _ema(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(arr, dtype=float)
    a = 2.0 / (n + 1.0)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out


def _stddev(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n - 1, len(arr)):
        out[i] = float(np.std(arr[i - n + 1:i + 1], ddof=0))
    return out


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    out = np.empty_like(close, dtype=float)
    out[0] = high[0] - low[0]
    for i in range(1, len(close)):
        out[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return out


# ============================================================================
# TDI_SQUEEZE SIGNAL (stateless, numpy)
# ============================================================================

def signal_tdi_squeeze(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                       vol: np.ndarray) -> Optional[dict]:
    """TDI Extreme + BB/KC Squeeze Release signal.

    Returns dict with side, strength, fair_up, confidence, reason, detail
    or None if no signal.
    """
    n = len(close)
    if n < 55:
        return None

    # ---- TDI: RSI → BB on RSI → EMA on RSI ----
    rsi = _rsi_wilder(close, 13)
    rsi_bb_period = 34
    rsi_bb_mult = 1.6185

    # BB on RSI
    rsi_sma = _sma(rsi, rsi_bb_period)
    rsi_std = _stddev(rsi, rsi_bb_period)
    tdi_upper = rsi_sma + rsi_bb_mult * rsi_std
    tdi_lower = rsi_sma - rsi_bb_mult * rsi_std

    # ---- BB(20,2) on price ----
    bb_mid = _sma(close, 20)
    bb_std = _stddev(close, 20)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std

    # ---- Keltner Channel(20, 1.5) ----
    tr = _true_range(high, low, close)
    kc_mid = _sma(close, 20)
    atr = _sma(tr, 20)
    kc_upper = kc_mid + 1.5 * atr
    kc_lower = kc_mid - 1.5 * atr

    # ---- EMA(9) for direction ----
    ema9 = _ema(close, 9)

    # Check last few bars
    i = n - 1

    # Validate all indicators are finite
    vals = [rsi[i], tdi_upper[i], tdi_lower[i], bb_upper[i], bb_lower[i],
            kc_upper[i], kc_lower[i], bb_upper[i-1], bb_lower[i-1],
            kc_upper[i-1], kc_lower[i-1], ema9[i]]
    if any(not np.isfinite(v) for v in vals):
        return None

    # ---- Squeeze detection ----
    # Current bar: squeeze ON if BB inside KC
    squeeze_now = (bb_lower[i] > kc_lower[i]) and (bb_upper[i] < kc_upper[i])
    # Previous bar: was squeeze on?
    squeeze_prev = (bb_lower[i-1] > kc_lower[i-1]) and (bb_upper[i-1] < kc_upper[i-1])

    # Count squeeze duration (look back)
    squeeze_bars = 0
    for j in range(i - 1, max(i - 30, 20), -1):
        if (np.isfinite(bb_lower[j]) and np.isfinite(kc_lower[j]) and
            np.isfinite(bb_upper[j]) and np.isfinite(kc_upper[j])):
            if bb_lower[j] > kc_lower[j] and bb_upper[j] < kc_upper[j]:
                squeeze_bars += 1
            else:
                break
        else:
            break

    # Squeeze just released OR squeeze is on with TDI extreme
    squeeze_releasing = squeeze_prev and not squeeze_now
    squeeze_active = squeeze_now and squeeze_bars >= 3

    # ---- TDI extreme ----
    tdi_oversold = rsi[i] < tdi_lower[i]
    tdi_overbought = rsi[i] > tdi_upper[i]

    if not tdi_oversold and not tdi_overbought:
        return None

    # Need either squeeze releasing or active squeeze
    if not squeeze_releasing and not squeeze_active:
        return None

    # ---- Direction ----
    if tdi_oversold:
        side = "UP"
    else:
        side = "DOWN"

    # ---- Confidence ----
    # Base: 0.68
    # Extremity bonus: how far RSI is outside TDI bands
    if tdi_oversold:
        extremity = max(0, tdi_lower[i] - rsi[i]) / 20.0
    else:
        extremity = max(0, rsi[i] - tdi_upper[i]) / 20.0

    # Squeeze bonus (longer squeeze = more stored energy)
    squeeze_bonus = min(0.08, squeeze_bars * 0.005)

    # Release bonus (squeeze releasing is stronger than just active)
    release_bonus = 0.04 if squeeze_releasing else 0.0

    confidence = 0.68 + min(0.10, extremity) + squeeze_bonus + release_bonus
    confidence = min(confidence, 0.92)

    # ---- Fair probability ----
    p = 0.50
    # RSI contribution
    p += (rsi[i] - 50.0) / 500.0
    # EMA direction
    ema_slope = (ema9[i] - ema9[max(0, i - 5)]) / max(1e-9, close[i])
    p += float(np.clip(ema_slope * 40.0, -0.08, 0.08))
    # Squeeze implies mean reversion
    if side == "UP":
        p = max(p, 0.52)  # slight upward bias for oversold reversal
    else:
        p = min(p, 0.48)  # slight downward bias for overbought reversal

    fair_up = float(np.clip(p, 0.05, 0.95))

    # ---- Strength (for sizing) ----
    strength = float(np.clip(
        min(1.0, extremity / 0.08) * 0.4 +
        min(1.0, squeeze_bars / 15.0) * 0.3 +
        (0.3 if squeeze_releasing else 0.1),
        0.0, 1.0
    ))

    detail = (f"rsi={rsi[i]:.1f} tdi_u={tdi_upper[i]:.1f} tdi_l={tdi_lower[i]:.1f} "
              f"sq={squeeze_bars}bars {'RELEASE' if squeeze_releasing else 'ACTIVE'} "
              f"ema9={ema9[i]:.0f}")

    reason = "tdi_squeeze_release" if squeeze_releasing else "tdi_squeeze_active"

    return {
        "side": side,
        "strength": strength,
        "fair_up": fair_up,
        "confidence": confidence,
        "reason": reason,
        "price": float(close[i]),
        "detail": detail,
    }


# ============================================================================
# PAPER TRADER
# ============================================================================

class TdiSqueezePaperTrader:
    def __init__(self):
        self.trades = {}
        self.resolved = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0}
        self._load()

    def _load(self):
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.resolved = data.get("resolved", [])
                self.trades = data.get("active", {})
                self.stats = data.get("stats", {"wins": 0, "losses": 0, "pnl": 0.0})
                total = self.stats["wins"] + self.stats["losses"]
                wr = self.stats["wins"] / total * 100 if total > 0 else 0
                print(f"[LOAD] {total} resolved ({self.stats['wins']}W/{self.stats['losses']}L "
                      f"{wr:.0f}%WR) | PnL: ${self.stats['pnl']:+.2f} | {len(self.trades)} active")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        data = {
            "stats": self.stats,
            "resolved": self.resolved[-200:],  # Keep last 200
            "active": self.trades,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            RESULTS_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    # ========================================================================
    # DATA
    # ========================================================================

    def fetch_1m_candles(self) -> list:
        """Fetch 200 BTC/USDT 1-min candles from Binance."""
        url = "https://api.binance.com/api/v3/klines"
        for attempt in range(3):
            try:
                r = httpx.get(url, params={"symbol": "BTCUSDT", "interval": "1m", "limit": 200},
                              timeout=15)
                r.raise_for_status()
                return [{"time": int(b[0]), "open": float(b[1]), "high": float(b[2]),
                         "low": float(b[3]), "close": float(b[4]), "volume": float(b[5])}
                        for b in r.json()]
            except Exception as e:
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                print(f"[BINANCE] Error: {e}")
                return []

    def discover_markets(self) -> list:
        """Discover BTC 15M markets from Gamma API."""
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
            if "429" not in str(e):
                print(f"[API] Error: {e}")
        return markets

    # ========================================================================
    # HELPERS
    # ========================================================================

    def get_prices(self, market: dict):
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

    def compute_edge(self, sig: dict, market: dict):
        up_p, down_p = self.get_prices(market)
        if up_p is None or down_p is None:
            return None, None, 0.0, 0.0, 0
        tte = market.get("_tte_seconds", 0)
        if tte < 60:
            return None, None, 0.0, 0.0, 0

        fair_up = sig["fair_up"]
        side = sig["side"]

        if side == "UP":
            entry = up_p + SPREAD_OFFSET
            if entry < MIN_ENTRY_PRICE or entry > MAX_ENTRY_PRICE:
                return None, None, 0.0, 0.0, 0
            edge = fair_up - entry
            return ("UP", entry, fair_up, edge, tte) if edge >= MIN_EDGE else (None, None, fair_up, edge, tte)
        elif side == "DOWN":
            entry = down_p + SPREAD_OFFSET
            if entry < MIN_ENTRY_PRICE or entry > MAX_ENTRY_PRICE:
                return None, None, 0.0, 0.0, 0
            edge = (1.0 - fair_up) - entry
            return ("DOWN", entry, 1.0 - fair_up, edge, tte) if edge >= MIN_EDGE else (None, None, 1.0 - fair_up, edge, tte)
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
                                shares = trade["size_usd"] / trade["entry_price"]
                                trade["pnl"] = round(shares - trade["size_usd"], 2)
                                trade["exit_price"] = 1.0
                            elif price <= 0.05:
                                trade["pnl"] = round(-trade["size_usd"], 2)
                                trade["exit_price"] = 0.0
                            else:
                                continue  # Not resolved yet

                            trade["exit_time"] = now.isoformat()
                            trade["status"] = "closed"

                            won = trade["pnl"] > 0
                            self.stats["wins" if won else "losses"] += 1
                            self.stats["pnl"] += trade["pnl"]
                            self.resolved.append(trade)
                            del self.trades[tid]

                            total = self.stats["wins"] + self.stats["losses"]
                            wr = self.stats["wins"] / total * 100 if total > 0 else 0
                            tag = "WIN" if won else "LOSS"
                            print(f"[{tag}] TDI_SQUEEZE {trade['side']} ${trade['pnl']:+.2f} | "
                                  f"entry=${trade['entry_price']:.2f} exit=${trade.get('exit_price', 0):.2f} | "
                                  f"conf={trade.get('confidence', 0):.2f} edge={trade.get('edge', 0):.1%} | "
                                  f"{trade.get('reason', '?')} | "
                                  f"Total: {self.stats['wins']}W/{self.stats['losses']}L {wr:.0f}%WR "
                                  f"${self.stats['pnl']:+.2f}")
                except Exception as e:
                    if "429" not in str(e):
                        print(f"[RESOLVE] API error: {e}")
            elif age_min > 25:
                trade["status"] = "closed"
                trade["pnl"] = round(-trade["size_usd"], 2)
                self.stats["losses"] += 1
                self.stats["pnl"] += trade["pnl"]
                self.resolved.append(trade)
                del self.trades[tid]
                print(f"[LOSS] TDI_SQUEEZE {trade['side']} ${trade['pnl']:+.2f} | aged out")

    # ========================================================================
    # ENTRY
    # ========================================================================

    def find_entries(self, candles: list, markets: list):
        now = datetime.now(timezone.utc)
        hour_utc = now.hour

        # Hour filter
        if hour_utc in SKIP_HOURS_UTC:
            return

        # Position limit
        open_count = sum(1 for t in self.trades.values() if t.get("status") == "open")
        if open_count >= MAX_CONCURRENT:
            return

        # Build numpy arrays from 1m candles
        close = np.array([c["close"] for c in candles], dtype=float)
        high = np.array([c["high"] for c in candles], dtype=float)
        low = np.array([c["low"] for c in candles], dtype=float)
        vol = np.array([c.get("volume", 0.0) for c in candles], dtype=float)

        # Generate signal
        sig = signal_tdi_squeeze(close, high, low, vol)
        if not sig or sig["side"] is None:
            return

        # Confidence filter
        if sig["confidence"] < MIN_CONFIDENCE:
            return

        # Try each market in the time window
        for market in markets:
            if open_count >= MAX_CONCURRENT:
                break
            time_left = market.get("_time_left", 99)
            if time_left < TIME_WINDOW[0] or time_left > TIME_WINDOW[1]:
                continue

            cid = market.get("conditionId", "")
            nid = market.get("id")
            trade_key = f"TDI_SQUEEZE_{cid}_{sig['side']}"
            if trade_key in self.trades:
                continue

            side, entry_price, fair_px, edge, tte = self.compute_edge(sig, market)
            if not side:
                continue

            # Size based on confidence and strength
            trade_size = BASE_SIZE + sig["strength"] * (MAX_SIZE - BASE_SIZE)
            # Confidence boost: high confidence = bigger size
            if sig["confidence"] >= 0.85:
                trade_size *= 1.3
            trade_size = round(max(BASE_SIZE, min(MAX_SIZE, trade_size)), 2)

            self.trades[trade_key] = {
                "side": side,
                "entry_price": entry_price,
                "size_usd": trade_size,
                "entry_time": now.isoformat(),
                "condition_id": cid,
                "market_numeric_id": nid,
                "title": market.get("question", ""),
                "strategy": "TDI_SQUEEZE_15m",
                "confidence": round(sig["confidence"], 3),
                "edge": round(edge, 4),
                "fair_px": round(fair_px, 4),
                "strength": round(sig["strength"], 3),
                "reason": sig["reason"],
                "btc_price": sig["price"],
                "detail": sig.get("detail", ""),
                "tte_seconds": tte,
                "hour_utc": hour_utc,
                "status": "open",
                "pnl": 0.0,
            }
            open_count += 1

            print(f"[ENTRY] TDI_SQUEEZE {side} ${entry_price:.2f} ${trade_size:.0f} | "
                  f"conf={sig['confidence']:.2f} edge={edge:.1%} fair={fair_px:.2f} "
                  f"str={sig['strength']:.2f} | "
                  f"{sig['reason']} | {sig.get('detail', '')} | "
                  f"tte={tte}s | BTC=${sig['price']:,.0f} | [PAPER]")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    def run(self):
        print("=" * 76)
        print("  TDI_SQUEEZE_15m PAPER TRADER V1.0")
        print(f"  Max {MAX_CONCURRENT} concurrent | Window: {TIME_WINDOW[0]}-{TIME_WINDOW[1]}min")
        print(f"  Min confidence: {MIN_CONFIDENCE} | Min edge: {MIN_EDGE:.1%}")
        print(f"  Skip hours (UTC): {sorted(SKIP_HOURS_UTC)}")
        print(f"  Spread: ${SPREAD_OFFSET} | Size: ${BASE_SIZE}-${MAX_SIZE}")
        print("=" * 76)

        if self.resolved:
            total = self.stats["wins"] + self.stats["losses"]
            wr = self.stats["wins"] / total * 100 if total > 0 else 0
            print(f"[RESUME] {self.stats['wins']}W/{self.stats['losses']}L "
                  f"{wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f}")

        cycle = 0
        while True:
            try:
                cycle += 1

                candles = self.fetch_1m_candles()
                markets = self.discover_markets()

                if not candles:
                    print("[WARN] No 1m candles")
                    time.sleep(10)
                    continue

                self.resolve_trades()
                self.find_entries(candles, markets)

                if cycle % 12 == 0:  # Every ~5 minutes
                    btc = candles[-1]["close"]
                    total = self.stats["wins"] + self.stats["losses"]
                    wr = self.stats["wins"] / total * 100 if total > 0 else 0
                    in_window = sum(1 for m in markets
                                    if TIME_WINDOW[0] <= m.get("_time_left", 99) <= TIME_WINDOW[1])
                    hour = datetime.now(timezone.utc).hour
                    skip_tag = " [SKIP HOUR]" if hour in SKIP_HOURS_UTC else ""
                    print(f"\n--- Cycle {cycle} | BTC ${btc:,.0f} | "
                          f"{self.stats['wins']}W/{self.stats['losses']}L {wr:.0f}%WR "
                          f"${self.stats['pnl']:+.2f} | "
                          f"Active: {len(self.trades)} | Markets: {in_window}/{len(markets)}"
                          f"{skip_tag} ---")

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
    lock = acquire_pid_lock("tdi_squeeze_paper")
    try:
        trader = TdiSqueezePaperTrader()
        trader.run()
    finally:
        release_pid_lock("tdi_squeeze_paper")
