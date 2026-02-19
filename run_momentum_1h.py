"""
Hourly Momentum Paper Trader V1.0

Paper-only trader for Polymarket 1H crypto Up/Down markets.
Uses RSI + 30-minute price change momentum on Binance 5-minute candles
to predict direction before hourly market close.

Strategy:
  1. Fetch BTC/ETH/SOL/XRP 5-minute candles from Binance (last 12 = 1 hour)
  2. Compute RSI-14 on 5m candles + 30-minute price change %
  3. Find active Polymarket 1H Up/Down markets
  4. UP signal: RSI > 55 AND 30min change > +0.05%
  5. DOWN signal: RSI < 45 AND 30min change < -0.05%
  6. SKIP if neither condition met (flat/uncertain)
  7. Wait for market resolution via Binance 1H candle

Entry rules:
  - Time window: 10-30 minutes before market close
  - Entry price range: $0.35-$0.65 (mid-range)
  - $2.50/trade (paper)
  - Max 4 concurrent positions (one per asset)

Resolution:
  - Binance 1H candle covering the market window
  - close > open = UP wins, close < open = DOWN wins

Usage:
  python -u run_momentum_1h.py
"""

import sys
import json
import time
import os
import asyncio
import functools
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

import httpx
from dotenv import load_dotenv

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# CONFIG
# ============================================================================
SCAN_INTERVAL = 30          # seconds between scan cycles
LOG_INTERVAL = 60           # print status every 60 seconds
MAX_CONCURRENT = 4          # max open positions (one per asset)
TRADE_SIZE = 2.50           # USD per trade (paper)
MIN_ENTRY = 0.35            # minimum entry price
MAX_ENTRY = 0.65            # maximum entry price
TIME_WINDOW_MIN = 10.0      # earliest entry: 30 min before close
TIME_WINDOW_MAX = 30.0      # latest entry: 10 min before close
RESOLVE_DELAY = 30          # seconds after market close before resolving
DAILY_LOSS_LIMIT = 15.0     # stop trading if session losses exceed $15

# RSI + momentum thresholds
RSI_UP_THRESHOLD = 55       # RSI must be above this for UP signal
RSI_DOWN_THRESHOLD = 45     # RSI must be below this for DOWN signal
MOMENTUM_UP_THRESHOLD = 0.0005   # 30min change > +0.05%
MOMENTUM_DOWN_THRESHOLD = -0.0005  # 30min change < -0.05%

# Skip hours (UTC) — dead hours with low edge
SKIP_HOURS = {7, 8, 11, 12, 13, 14, 15}

RESULTS_FILE = Path(__file__).parent / "momentum_1h_results.json"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
GAMMA_API_URL = "https://gamma-api.polymarket.com/events"

# Multi-asset config
ASSET_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "XRP": ["xrp", "ripple"],
}
BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}


# ============================================================================
# RSI CALCULATION (inline, no numpy needed)
# ============================================================================
def compute_rsi(closes, period=14):
    """Compute RSI from a list of close prices. Returns 50.0 if insufficient data."""
    if len(closes) < period + 1:
        return 50.0  # neutral
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(0, delta))
        losses.append(max(0, -delta))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ============================================================================
# HOURLY MOMENTUM PAPER TRADER
# ============================================================================
class Momentum1HTrader:
    def __init__(self):
        self.active: Dict[str, dict] = {}       # {trade_key: trade_dict}
        self.resolved: list = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0}
        self.session_pnl = 0.0
        self.attempted_cids: set = set()         # prevent re-entering same market
        self.running = True
        self._last_log_time = 0.0
        self._load()

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _load(self):
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.active = data.get("active", {})
                self.resolved = data.get("resolved", [])
                self.stats = {**self.stats, **data.get("stats", {})}
                # Rebuild attempted CIDs from active + resolved
                for key in self.active:
                    cid = self.active[key].get("condition_id", "")
                    if cid:
                        self.attempted_cids.add(cid)
                for trade in self.resolved:
                    cid = trade.get("condition_id", "")
                    if cid:
                        self.attempted_cids.add(cid)
                w, l = self.stats["wins"], self.stats["losses"]
                total = w + l
                wr = w / total * 100 if total > 0 else 0
                print(f"[LOAD] {len(self.resolved)} resolved, {len(self.active)} active | "
                      f"{w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f}")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        data = {
            "active": self.active,
            "resolved": self.resolved[-500:],  # keep last 500
            "stats": self.stats,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        tmp = RESULTS_FILE.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(RESULTS_FILE)
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    # ========================================================================
    # MARKET DISCOVERY
    # ========================================================================

    async def discover_1h_markets(self) -> list:
        """Find active Polymarket 1H Up/Down markets for BTC, ETH, SOL, XRP."""
        markets = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    GAMMA_API_URL,
                    params={"tag_slug": "1H", "active": "true",
                            "closed": "false", "limit": 200},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if r.status_code != 200:
                    print(f"[API] Gamma error: {r.status_code}")
                    return markets

                now = datetime.now(timezone.utc)
                for event in r.json():
                    title = event.get("title", "").lower()

                    # Detect which asset this market is for
                    asset = None
                    for name, keywords in ASSET_KEYWORDS.items():
                        if any(kw in title for kw in keywords):
                            asset = name
                            break
                    if not asset:
                        continue

                    for m in event.get("markets", []):
                        if m.get("closed", True):
                            continue
                        end_date = m.get("endDate", "")
                        if not end_date:
                            continue
                        try:
                            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                            if end_dt < now:
                                continue
                            time_left_min = (end_dt - now).total_seconds() / 60
                            m["_time_left_min"] = time_left_min
                            m["_end_dt"] = end_dt.isoformat()
                            m["_end_dt_parsed"] = end_dt
                            m["_asset"] = asset
                        except Exception:
                            continue

                        if not m.get("question"):
                            m["question"] = event.get("title", "")
                        markets.append(m)
        except Exception as e:
            print(f"[API] Discovery error: {e}")
        return markets

    # ========================================================================
    # MARKET HELPERS
    # ========================================================================

    def get_prices(self, market: dict) -> tuple:
        """Extract UP/DOWN prices from market data."""
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

    # ========================================================================
    # BINANCE DATA
    # ========================================================================

    async def fetch_binance_5m_candles(self, symbol: str, limit: int = 15) -> list:
        """Fetch 5-minute candles from Binance for signal generation.

        Default limit=15 gives ~75 minutes of data (enough for RSI-14 + 30min change).
        Returns list of dicts with 'time', 'open', 'close' keys.
        """
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    r = await client.get(BINANCE_REST_URL, params={
                        "symbol": symbol, "interval": "5m", "limit": limit,
                    })
                    r.raise_for_status()
                    return [
                        {
                            "time": int(k[0]),
                            "open": float(k[1]),
                            "close": float(k[4]),
                        }
                        for k in r.json()
                    ]
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                print(f"[BINANCE] Error fetching 5m candles ({symbol}): {e}")
                return []

    async def resolve_via_binance(self, end_dt: datetime, asset: str) -> Optional[str]:
        """Determine if asset went UP or DOWN during the 1-hour window.

        The 1H market covers the hour ending at end_dt.
        Fetch the Binance 1h candle and compare close vs open.

        Returns 'UP', 'DOWN', or None if data unavailable.
        """
        binance_symbol = BINANCE_SYMBOLS.get(asset, "BTCUSDT")
        try:
            # The 1H market covers the hour ending at end_dt
            start_ms = int((end_dt - timedelta(hours=1)).timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient(timeout=15) as client:
                # Fetch the 1h candle from Binance
                r = await client.get(
                    BINANCE_REST_URL,
                    params={
                        "symbol": binance_symbol,
                        "interval": "1h",
                        "startTime": start_ms,
                        "endTime": end_ms,
                        "limit": 5,
                    },
                )
                r.raise_for_status()
                klines = r.json()

                if not klines:
                    # Fallback: use 5-minute candles to span the hour
                    r2 = await client.get(
                        BINANCE_REST_URL,
                        params={
                            "symbol": binance_symbol,
                            "interval": "5m",
                            "startTime": start_ms,
                            "endTime": end_ms,
                            "limit": 15,
                        },
                    )
                    r2.raise_for_status()
                    klines_5m = r2.json()
                    if not klines_5m:
                        return None
                    open_price = float(klines_5m[0][1])    # first candle open
                    close_price = float(klines_5m[-1][4])  # last candle close
                else:
                    # Find the candle whose open time is closest to start of window
                    best_candle = None
                    best_diff = float("inf")
                    for k in klines:
                        candle_open_ms = int(k[0])
                        diff = abs(candle_open_ms - start_ms)
                        if diff < best_diff:
                            best_diff = diff
                            best_candle = k
                    if best_candle is None:
                        return None
                    open_price = float(best_candle[1])
                    close_price = float(best_candle[4])

                if close_price > open_price:
                    return "UP"
                elif close_price < open_price:
                    return "DOWN"
                else:
                    # Exact tie — treat as DOWN (convention)
                    return "DOWN"

        except Exception as e:
            print(f"[BINANCE] Resolution error ({asset}): {e}")
            return None

    # ========================================================================
    # SIGNAL GENERATION
    # ========================================================================

    def compute_signal(self, candles: list) -> dict:
        """Compute RSI + 30min momentum signal from 5-minute candles.

        Uses last 12 candles (1 hour of 5m data).
        Returns dict with 'side' ('UP', 'DOWN', or None), 'rsi', 'change_30m', 'price'.
        """
        if len(candles) < 7:
            # Need at least 7 candles for 30min change + minimal RSI
            return {"side": None, "rsi": 50.0, "change_30m": 0.0, "price": 0.0}

        closes = [c["close"] for c in candles]
        current_price = closes[-1]

        # RSI-14 on 5m closes
        rsi = compute_rsi(closes, period=14)

        # Price change over last 30 minutes (6 x 5m candles)
        lookback = min(6, len(closes) - 1)
        price_30m_ago = closes[-(lookback + 1)]
        if price_30m_ago == 0:
            return {"side": None, "rsi": rsi, "change_30m": 0.0, "price": current_price}

        change_30m = (current_price - price_30m_ago) / price_30m_ago

        # Signal logic
        side = None
        if rsi > RSI_UP_THRESHOLD and change_30m > MOMENTUM_UP_THRESHOLD:
            side = "UP"
        elif rsi < RSI_DOWN_THRESHOLD and change_30m < MOMENTUM_DOWN_THRESHOLD:
            side = "DOWN"

        return {
            "side": side,
            "rsi": round(rsi, 2),
            "change_30m": round(change_30m, 6),
            "price": current_price,
        }

    # ========================================================================
    # ENTRY
    # ========================================================================

    async def find_entries(self, markets: list):
        """Find momentum-based entries on 1H markets across all assets."""
        now = datetime.now(timezone.utc)

        # Skip dead hours
        if now.hour in SKIP_HOURS:
            return

        open_count = len(self.active)
        if open_count >= MAX_CONCURRENT:
            return

        # Daily loss limit check
        if self.session_pnl <= -DAILY_LOSS_LIMIT:
            return

        # Track which assets already have an active position
        active_assets = set()
        for trade in self.active.values():
            active_assets.add(trade.get("_asset", ""))

        # Group markets by asset, pick the one closest to closing within window
        best_per_asset: Dict[str, dict] = {}
        for market in markets:
            asset = market.get("_asset", "")
            time_left = market.get("_time_left_min", 99)

            # Must be in entry window: 10-30 min before close
            if time_left < TIME_WINDOW_MIN or time_left > TIME_WINDOW_MAX:
                continue

            # Skip if already have a position in this asset
            if asset in active_assets:
                continue

            cid = market.get("conditionId", "")
            if not cid or cid in self.attempted_cids:
                continue

            # Pick the market closest to closing
            if asset not in best_per_asset or time_left < best_per_asset[asset].get("_time_left_min", 99):
                best_per_asset[asset] = market

        if not best_per_asset:
            return

        # Fetch Binance 5m candles for each asset that has a candidate market
        candle_tasks = {}
        for asset in best_per_asset:
            symbol = BINANCE_SYMBOLS.get(asset)
            if symbol:
                candle_tasks[asset] = self.fetch_binance_5m_candles(symbol, limit=15)

        if not candle_tasks:
            return

        results = await asyncio.gather(*candle_tasks.values())
        all_candles = dict(zip(candle_tasks.keys(), results))

        # Evaluate signals and enter trades
        for asset, market in best_per_asset.items():
            if open_count >= MAX_CONCURRENT:
                break

            candles = all_candles.get(asset, [])
            if len(candles) < 7:
                continue

            # Compute signal
            signal = self.compute_signal(candles)
            side = signal["side"]
            rsi = signal["rsi"]
            change_30m = signal["change_30m"]
            asset_price = signal["price"]

            if side is None:
                # No clear signal — skip
                continue

            # Price filter
            up_price, down_price = self.get_prices(market)
            if up_price is None or down_price is None:
                continue

            entry_price = up_price if side == "UP" else down_price
            if entry_price < MIN_ENTRY or entry_price > MAX_ENTRY:
                continue

            # Paper spread simulation
            entry_price = round(entry_price + 0.02, 2)
            if entry_price > MAX_ENTRY:
                continue

            cid = market.get("conditionId", "")
            question = market.get("question", "?")
            nid = market.get("id")
            time_left = market.get("_time_left_min", 0)

            # Calculate shares and cost
            shares = TRADE_SIZE / entry_price
            cost = round(shares * entry_price, 2)

            # Mark as attempted
            self.attempted_cids.add(cid)

            # Build trade record
            trade_key = f"1h_{cid}_{side}"
            trade = {
                "condition_id": cid,
                "market_numeric_id": nid,
                "question": question,
                "side": side,
                "entry_price": round(entry_price, 4),
                "shares": round(shares, 4),
                "size_usd": cost,
                "rsi": rsi,
                "change_30m": change_30m,
                "asset_price": asset_price,
                "_asset": asset,
                "time_left_min": round(time_left, 1),
                "entry_time": now.isoformat(),
                "end_dt": market.get("_end_dt", ""),
                "status": "open",
                "pnl": 0.0,
                "result": None,
            }

            self.active[trade_key] = trade
            open_count += 1

            print(f"\n[ENTRY] {asset}:{side} @ ${entry_price:.2f} ${cost:.2f} ({shares:.1f}sh) | "
                  f"RSI={rsi:.1f} chg30m={change_30m:+.4%} | "
                  f"{asset}=${asset_price:,.2f} | "
                  f"{time_left:.0f}min left | "
                  f"{question[:55]}")

        self._save()

    # ========================================================================
    # RESOLUTION
    # ========================================================================

    async def resolve_trades(self):
        """Resolve trades whose 1H windows have closed."""
        now = datetime.now(timezone.utc)
        to_remove = []

        for key, trade in list(self.active.items()):
            if trade.get("status") != "open":
                continue

            end_dt_str = trade.get("end_dt", "")
            if not end_dt_str:
                continue

            try:
                end_dt = datetime.fromisoformat(end_dt_str)
            except Exception:
                continue

            # Wait RESOLVE_DELAY seconds after market close
            seconds_past_close = (now - end_dt).total_seconds()
            if seconds_past_close < RESOLVE_DELAY:
                continue

            # Resolve via Binance
            asset = trade.get("_asset", "BTC")
            outcome = await self.resolve_via_binance(end_dt, asset)

            if outcome is None:
                # Retry on next cycle if data not available yet
                if seconds_past_close > 300:
                    # Stale — mark as loss
                    trade["status"] = "closed"
                    trade["result"] = "unknown"
                    trade["pnl"] = round(-trade["size_usd"], 2)
                    trade["resolve_time"] = now.isoformat()
                    self.stats["losses"] += 1
                    self.stats["pnl"] += trade["pnl"]
                    self.session_pnl += trade["pnl"]
                    self.resolved.append(trade)
                    to_remove.append(key)
                    print(f"[LOSS] {asset}:{trade['side']} ${trade['pnl']:+.2f} | "
                          f"Could not resolve (no Binance data) | {trade['question'][:45]}")
                continue

            # Determine win/loss
            won = (trade["side"] == outcome)
            if won:
                # Winner: shares pay $1.00 each
                pnl = round(trade["shares"] * 1.0 - trade["size_usd"], 2)
                trade["result"] = "WIN"
            else:
                # Loser: shares worth $0
                pnl = round(-trade["size_usd"], 2)
                trade["result"] = "LOSS"

            trade["status"] = "closed"
            trade["pnl"] = pnl
            trade["market_outcome"] = outcome
            trade["resolve_time"] = now.isoformat()

            self.stats["wins" if won else "losses"] += 1
            self.stats["pnl"] += pnl
            self.session_pnl += pnl

            self.resolved.append(trade)
            to_remove.append(key)

            w, l = self.stats["wins"], self.stats["losses"]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            tag = "WIN" if won else "LOSS"

            print(f"[{tag}] {asset}:{trade['side']} ${pnl:+.2f} | "
                  f"entry=${trade['entry_price']:.2f} | "
                  f"RSI={trade.get('rsi', '?')} chg30m={trade.get('change_30m', 0):+.4%} | "
                  f"market={outcome} | "
                  f"{w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f} | "
                  f"session=${self.session_pnl:+.2f} | "
                  f"{trade['question'][:45]}")

        for key in to_remove:
            del self.active[key]

        if to_remove:
            self._save()

    # ========================================================================
    # LOGGING
    # ========================================================================

    def print_status(self, markets: list):
        """Print cycle status every LOG_INTERVAL seconds."""
        now = time.time()
        if now - self._last_log_time < LOG_INTERVAL:
            return
        self._last_log_time = now

        w, l = self.stats["wins"], self.stats["losses"]
        total = w + l
        wr = w / total * 100 if total > 0 else 0

        now_dt = datetime.now(timezone.utc)
        mst_offset = timedelta(hours=-7)
        now_mst = now_dt + mst_offset

        active_count = len(self.active)

        print(f"\n--- {now_mst.strftime('%H:%M:%S')} MST | 1H MOMENTUM | "
              f"Active: {active_count} | "
              f"{w}W/{l}L {wr:.0f}%WR | "
              f"PnL: ${self.stats['pnl']:+.2f} | "
              f"Session: ${self.session_pnl:+.2f} ---")

        # Per-asset market status
        asset_status = {}
        for m in markets:
            asset = m.get("_asset", "?")
            tl = m.get("_time_left_min", 9999)
            # Pick the closest market per asset
            if asset not in asset_status or tl < asset_status[asset]:
                asset_status[asset] = tl

        parts = []
        for asset in ["BTC", "ETH", "SOL", "XRP"]:
            if asset in asset_status:
                tl = asset_status[asset]
                in_window = TIME_WINDOW_MIN <= tl <= TIME_WINDOW_MAX
                tag = " [IN WINDOW]" if in_window else ""
                parts.append(f"{asset}: {tl:.0f}min left{tag}")
            else:
                parts.append(f"{asset}: no market")
        print(f"    {' | '.join(parts)}")

        if now_dt.hour in SKIP_HOURS:
            print(f"    *** SKIP HOUR (UTC {now_dt.hour}) — not trading ***")

        if self.session_pnl <= -DAILY_LOSS_LIMIT:
            print(f"    *** DAILY LOSS LIMIT HIT: ${self.session_pnl:+.2f} ***")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        print("=" * 70)
        print("  HOURLY MOMENTUM PAPER TRADER V1.0")
        print("=" * 70)
        print(f"  Strategy: RSI + price change momentum on 1H crypto Up/Down markets")
        print(f"  Assets: BTC, ETH, SOL, XRP")
        print(f"  Entry: {TIME_WINDOW_MIN:.0f}-{TIME_WINDOW_MAX:.0f} min before close | "
              f"Price: ${MIN_ENTRY:.2f}-${MAX_ENTRY:.2f}")
        print(f"  UP: RSI >{RSI_UP_THRESHOLD} + 30min change >+{MOMENTUM_UP_THRESHOLD*100:.2f}%")
        print(f"  DOWN: RSI <{RSI_DOWN_THRESHOLD} + 30min change <{MOMENTUM_DOWN_THRESHOLD*100:+.2f}%")
        print(f"  Size: ${TRADE_SIZE:.2f}/trade | Max concurrent: {MAX_CONCURRENT}")
        print(f"  Resolution: Binance 1H candle (close vs open)")
        print(f"  Skip hours (UTC): {sorted(SKIP_HOURS)}")
        print(f"  Daily loss limit: ${DAILY_LOSS_LIMIT:.2f}")
        print(f"  Scan: every {SCAN_INTERVAL}s | Log: every {LOG_INTERVAL}s")
        print(f"  Results: {RESULTS_FILE.name}")
        print("=" * 70)

        if self.resolved:
            w, l = self.stats["wins"], self.stats["losses"]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            print(f"[RESUME] {w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
                  f"{len(self.resolved)} resolved, {len(self.active)} active")

        cycle = 0
        while self.running:
            try:
                cycle += 1

                # Daily loss limit — slow down when halted
                if self.session_pnl <= -DAILY_LOSS_LIMIT:
                    self.print_status([])
                    await asyncio.sleep(60)
                    continue

                # Discover 1H markets
                markets = await self.discover_1h_markets()

                # Resolve expired trades
                await self.resolve_trades()

                # Find new entries
                await self.find_entries(markets)

                # Periodic status log
                self.print_status(markets)

                # Save periodically (every 4 cycles = ~120s)
                if cycle % 4 == 0:
                    self._save()

                await asyncio.sleep(SCAN_INTERVAL)

            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Saving state...")
                self._save()
                break
            except Exception as e:
                print(f"[ERROR] Cycle {cycle}: {e}")
                traceback.print_exc()
                await asyncio.sleep(10)

        # Final save
        self._save()
        w, l = self.stats["wins"], self.stats["losses"]
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        print(f"\n[FINAL] {w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
              f"Session: ${self.session_pnl:+.2f}")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    lock = acquire_pid_lock("momentum_1h")
    try:
        trader = Momentum1HTrader()
        asyncio.run(trader.run())
    finally:
        release_pid_lock("momentum_1h")
