"""
Cross-Timeframe Arbitrage V1.0 — PAPER MODE

Structural arbitrage between Polymarket 15-minute and 5-minute BTC Up/Down markets.

When a 15M market has ~5 minutes left, a 5M market opens covering the SAME final period.
Both share the same close time but different open prices:
  - 15M: BTC close vs BTC open 15 min ago
  - 5M:  BTC close vs BTC open 5 min ago (≈current price)

If BTC is near the 15M open price (±$2-$20), outcomes are highly correlated:
  - Both UP or both DOWN in >90% of cases
  - Buy opposing sides for combined < $1.00 → one always wins
  - The most likely divergence scenario gives a DOUBLE WIN (asymmetric upside)

Side selection:
  - current < 15M open → buy 15M DOWN + 5M UP (divergence = jackpot)
  - current > 15M open → buy 15M UP + 5M DOWN (divergence = jackpot)

Usage:
  python -u run_cross_arb.py                    # Paper mode (default)
  python -u run_cross_arb.py --live             # Live mode (future)
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
from typing import Dict, Optional, List, Tuple

import httpx
import websockets
from dotenv import load_dotenv

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# V1.1: POLYMARKET WEBSOCKET FEED — real-time market detection + prices
# ============================================================================
POLY_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class PolyWSLite:
    """Lightweight Polymarket WebSocket for cross-arb.
    Tracks: best_bid_ask, new_market, market_resolved."""

    def __init__(self):
        self._ws = None
        self._connected = False
        self._running = False
        self._subscribed: set = set()
        self.prices: dict = {}  # {token_id: {"best_bid": float, "best_ask": float}}
        self.resolved_tokens: dict = {}  # {token_id: "UP"/"DOWN"}
        self.new_markets: list = []
        self.last_update = 0

    async def start(self):
        self._running = True
        asyncio.create_task(self._loop())

    async def _loop(self):
        while self._running:
            try:
                async with websockets.connect(POLY_WS_URL, ping_interval=30, ping_timeout=10) as ws:
                    self._ws = ws
                    self._connected = True
                    print("[POLY-WS] Connected")
                    if self._subscribed:
                        await self._send_sub(list(self._subscribed))
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                            self._handle(msg)
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                self._connected = False
                if self._running:
                    await asyncio.sleep(5)
        self._connected = False

    async def _send_sub(self, ids):
        if not self._ws or not self._connected:
            return
        try:
            await self._ws.send(json.dumps({
                "assets_ids": ids, "type": "market", "custom_feature_enabled": True
            }))
            self._subscribed.update(ids)
        except Exception:
            pass

    async def subscribe(self, token_ids: list):
        new = [t for t in token_ids if t and t not in self._subscribed]
        if new:
            await self._send_sub(new)

    def _handle(self, msg):
        if isinstance(msg, list):
            for m in msg:
                self._handle_one(m)
        else:
            self._handle_one(msg)

    def _handle_one(self, msg):
        et = msg.get("event_type", "")
        self.last_update = time.time()
        if et == "best_bid_ask":
            aid = msg.get("asset_id", "")
            if aid:
                self.prices[aid] = {
                    "best_bid": float(msg.get("best_bid", 0)),
                    "best_ask": float(msg.get("best_ask", 0)),
                }
        elif et == "price_change":
            for pc in msg.get("price_changes", []):
                aid = pc.get("asset_id", "")
                bb, ba = pc.get("best_bid"), pc.get("best_ask")
                if aid and bb is not None and ba is not None:
                    self.prices[aid] = {"best_bid": float(bb), "best_ask": float(ba)}
        elif et == "market_resolved":
            winning = msg.get("winning_outcome", "").upper()
            for aid in (msg.get("assets_ids", []) or []):
                self.resolved_tokens[aid] = winning
            if winning:
                print(f"[POLY-WS] Resolved: {winning} | {msg.get('question', '')[:50]}")
        elif et == "new_market":
            self.new_markets.append(msg)

    def get_ask(self, token_id: str) -> float:
        return self.prices.get(token_id, {}).get("best_ask", 0)

    def get_bid(self, token_id: str) -> float:
        return self.prices.get(token_id, {}).get("best_bid", 0)

    def pop_new(self) -> list:
        out = list(self.new_markets)
        self.new_markets.clear()
        return out

    def is_connected(self) -> bool:
        return self._connected

    def stop(self):
        self._running = False

# ============================================================================
# CONFIG
# ============================================================================
PROXIMITY_MAX = 50.0        # max $ distance from 15M open (widened for paper data collection)
PROXIMITY_MIN = 2.0         # skip if too close (fees eat the edge)
MAX_COMBINED_COST = 1.50    # uncapped for paper — collect correlation data, filter later
TRADE_SIZE = 5.00           # USD per leg
ENTRY_WINDOW_15M = (4.0, 6.0)  # minutes before 15M close to look for match
SCAN_INTERVAL = 10          # seconds between scans
MAX_CONCURRENT = 2          # max open arb pairs at once
RESOLVE_DELAY = 20          # seconds after close before checking Binance
DAILY_LOSS_LIMIT = 30.0     # stop if losses exceed this
PAIR_MATCH_TOLERANCE = 60   # seconds tolerance for matching close times

RESULTS_FILE = Path(__file__).parent / "cross_arb_results.json"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
GAMMA_API_URL = "https://gamma-api.polymarket.com/events"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"


# ============================================================================
# CROSS-TIMEFRAME ARBITRAGE BOT
# ============================================================================
class CrossTimeframeArb:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.active: Dict[str, dict] = {}
        self.resolved: list = []
        self.stats = {
            "wins": 0, "losses": 0, "double_wins": 0, "double_losses": 0,
            "correlated": 0, "pnl": 0.0, "pairs_entered": 0, "pairs_skipped": 0,
        }
        self.session_pnl = 0.0
        self.attempted_pairs: set = set()  # close_time keys already entered
        self.running = True
        self._last_log_time = 0.0
        self._last_price_cache = (0.0, 0.0)  # (timestamp, price)
        self.client = None
        self.poly_ws = PolyWSLite()  # V1.1: Polymarket WebSocket
        self._load()

        if not self.paper:
            self._init_clob()

    # ========================================================================
    # CLOB CLIENT (for future live mode)
    # ========================================================================

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
            traceback.print_exc()
            print(f"[CLOB] Falling back to PAPER mode")
            self.paper = True

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
                for key in self.active:
                    self.attempted_pairs.add(key)
                for trade in self.resolved:
                    key = trade.get("pair_key", "")
                    if key:
                        self.attempted_pairs.add(key)
                w = self.stats["wins"]
                l = self.stats["losses"]
                dw = self.stats["double_wins"]
                dl = self.stats["double_losses"]
                total = w + l + dw + dl
                wr = (w + dw) / total * 100 if total > 0 else 0
                print(f"[LOAD] {len(self.resolved)} resolved, {len(self.active)} active | "
                      f"{w+dw}W/{l+dl}L {wr:.0f}%WR (dbl:{dw}W/{dl}L) | "
                      f"PnL: ${self.stats['pnl']:+.2f}")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        data = {
            "active": self.active,
            "resolved": self.resolved[-500:],
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

    async def _fetch_markets(self, tag_slug: str) -> list:
        """Fetch BTC Up/Down markets for a given timeframe."""
        markets = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    GAMMA_API_URL,
                    params={"tag_slug": tag_slug, "active": "true",
                            "closed": "false", "limit": 200},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if r.status_code != 200:
                    print(f"[API] Gamma {tag_slug} error: {r.status_code}")
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
                        if not end_date:
                            continue
                        try:
                            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                            if end_dt < now - timedelta(seconds=RESOLVE_DELAY + 60):
                                continue
                            time_left_min = (end_dt - now).total_seconds() / 60.0
                            m["_time_left_min"] = time_left_min
                            m["_end_dt"] = end_dt.isoformat()
                            m["_end_dt_parsed"] = end_dt
                        except Exception:
                            continue
                        if not m.get("question"):
                            m["question"] = event.get("title", "")
                        markets.append(m)

            # V1.1: Subscribe all discovered markets to WS for real-time prices
            if self.poly_ws.is_connected():
                all_tids = []
                for m in markets:
                    tids = m.get("clobTokenIds", [])
                    if isinstance(tids, str):
                        tids = json.loads(tids)
                    all_tids.extend([t for t in tids if t])
                if all_tids:
                    await self.poly_ws.subscribe(all_tids)

        except Exception as e:
            print(f"[API] {tag_slug} discovery error: {e}")
        return markets

    async def discover_pairs(self) -> List[Tuple[dict, dict]]:
        """Find matched 15M+5M pairs with same close time."""
        markets_15m, markets_5m = await asyncio.gather(
            self._fetch_markets("15M"),
            self._fetch_markets("5M"),
        )

        # Filter 15M: within entry window (4-6 min before close)
        candidates_15m = [
            m for m in markets_15m
            if ENTRY_WINDOW_15M[0] <= m["_time_left_min"] <= ENTRY_WINDOW_15M[1]
        ]

        if not candidates_15m:
            return []

        # Match by close time
        pairs = []
        for m15 in candidates_15m:
            end_15m = m15["_end_dt_parsed"]
            best_5m = None
            best_diff = float("inf")
            for m5 in markets_5m:
                end_5m = m5["_end_dt_parsed"]
                diff = abs((end_15m - end_5m).total_seconds())
                if diff < best_diff and diff <= PAIR_MATCH_TOLERANCE:
                    best_diff = diff
                    best_5m = m5
            if best_5m:
                pairs.append((m15, best_5m))

        return pairs

    # ========================================================================
    # MARKET HELPERS
    # ========================================================================

    def get_token_ids(self, market: dict) -> tuple:
        """Extract UP/DOWN token IDs from market data."""
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

    def get_prices(self, market: dict) -> tuple:
        """Extract UP/DOWN prices. Prefers WS real-time ask, falls back to Gamma."""
        outcomes = market.get("outcomes", [])
        prices = market.get("outcomePrices", [])
        token_ids = market.get("clobTokenIds", [])
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(prices, str):
            prices = json.loads(prices)
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)

        up_price, down_price = None, None
        for i, o in enumerate(outcomes):
            # V1.1: Try WS real-time ask first
            ws_price = 0
            if i < len(token_ids) and self.poly_ws.is_connected():
                ws_price = self.poly_ws.get_ask(token_ids[i])

            p = ws_price if ws_price > 0 else (float(prices[i]) if i < len(prices) else 0)
            if p > 0:
                if str(o).lower() == "up":
                    up_price = p
                elif str(o).lower() == "down":
                    down_price = p
        return up_price, down_price

    async def get_best_ask(self, token_id: str) -> Optional[float]:
        """Get best ask price. V1.1: WS first, then CLOB REST fallback."""
        # Try WS price first (instant, no network call)
        if self.poly_ws.is_connected():
            ws_ask = self.poly_ws.get_ask(token_id)
            if ws_ask > 0:
                return ws_ask

        # Fallback to REST
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(CLOB_BOOK_URL, params={"token_id": token_id})
                if r.status_code != 200:
                    return None
                book = r.json()
                asks = book.get("asks", [])
                if not asks:
                    return None
                # CLOB asks are sorted descending — use min()
                return min(float(a["price"]) for a in asks)
        except Exception:
            return None

    # ========================================================================
    # BINANCE PRICE DATA
    # ========================================================================

    async def get_current_price(self) -> Optional[float]:
        """Get current BTC price from Binance (cached 5s)."""
        now = time.time()
        if now - self._last_price_cache[0] < 5 and self._last_price_cache[1] > 0:
            return self._last_price_cache[1]
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(BINANCE_REST_URL, params={
                    "symbol": "BTCUSDT", "interval": "1m", "limit": 1,
                })
                r.raise_for_status()
                klines = r.json()
                if klines:
                    price = float(klines[-1][4])  # latest 1m close
                    self._last_price_cache = (now, price)
                    return price
        except Exception as e:
            print(f"[BINANCE] Current price error: {e}")
        return None

    async def get_15m_open_price(self, end_dt: datetime) -> Optional[float]:
        """Get the opening price of the 15M candle from Binance."""
        try:
            start_dt = end_dt - timedelta(minutes=15)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(BINANCE_REST_URL, params={
                    "symbol": "BTCUSDT", "interval": "15m",
                    "startTime": start_ms, "endTime": end_ms, "limit": 2,
                })
                r.raise_for_status()
                klines = r.json()
                if klines:
                    # Find candle closest to start time
                    best = min(klines, key=lambda k: abs(int(k[0]) - start_ms))
                    return float(best[1])  # open price

                # Fallback: use 1m candles
                r2 = await client.get(BINANCE_REST_URL, params={
                    "symbol": "BTCUSDT", "interval": "1m",
                    "startTime": start_ms, "endTime": start_ms + 60000,
                    "limit": 2,
                })
                r2.raise_for_status()
                klines_1m = r2.json()
                if klines_1m:
                    return float(klines_1m[0][1])  # first 1m candle open
        except Exception as e:
            print(f"[BINANCE] 15M open price error: {e}")
        return None

    async def resolve_binance(self, end_dt: datetime, interval: str, minutes: int) -> Optional[str]:
        """Resolve a market outcome via Binance candle data."""
        try:
            start_dt = end_dt - timedelta(minutes=minutes)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(BINANCE_REST_URL, params={
                    "symbol": "BTCUSDT", "interval": interval,
                    "startTime": start_ms, "endTime": end_ms, "limit": 5,
                })
                r.raise_for_status()
                klines = r.json()

                if not klines:
                    # Fallback to 1m candles
                    r2 = await client.get(BINANCE_REST_URL, params={
                        "symbol": "BTCUSDT", "interval": "1m",
                        "startTime": start_ms, "endTime": end_ms, "limit": 20,
                    })
                    r2.raise_for_status()
                    klines_1m = r2.json()
                    if not klines_1m:
                        return None
                    open_price = float(klines_1m[0][1])
                    close_price = float(klines_1m[-1][4])
                else:
                    best = min(klines, key=lambda k: abs(int(k[0]) - start_ms))
                    open_price = float(best[1])
                    close_price = float(best[4])

                return "UP" if close_price >= open_price else "DOWN"
        except Exception as e:
            print(f"[BINANCE] Resolution error ({interval}): {e}")
            return None

    # ========================================================================
    # CORE LOGIC
    # ========================================================================

    async def evaluate_pair(self, m15: dict, m5: dict) -> Optional[dict]:
        """Evaluate a 15M+5M pair for arb opportunity."""
        end_dt = m15["_end_dt_parsed"]
        pair_key = end_dt.strftime("%Y%m%d_%H%M")

        if pair_key in self.attempted_pairs:
            return None

        if len(self.active) >= MAX_CONCURRENT:
            return None

        # Get 15M open price and current price
        open_15m, current = await asyncio.gather(
            self.get_15m_open_price(end_dt),
            self.get_current_price(),
        )

        if open_15m is None or current is None:
            return None

        gap = current - open_15m
        abs_gap = abs(gap)

        if abs_gap < PROXIMITY_MIN:
            print(f"[PROX] gap=${gap:+.2f} TOO CLOSE (min ${PROXIMITY_MIN}) | "
                  f"BTC=${current:,.2f} vs 15M open=${open_15m:,.2f}")
            self.attempted_pairs.add(pair_key)
            self.stats["pairs_skipped"] += 1
            return None
        if abs_gap > PROXIMITY_MAX:
            print(f"[PROX] gap=${gap:+.2f} TOO FAR (max ${PROXIMITY_MAX}) | "
                  f"BTC=${current:,.2f} vs 15M open=${open_15m:,.2f}")
            self.attempted_pairs.add(pair_key)
            self.stats["pairs_skipped"] += 1
            return None

        # Get prices for both markets
        up_15m, down_15m = self.get_prices(m15)
        up_5m, down_5m = self.get_prices(m5)

        if None in (up_15m, down_15m, up_5m, down_5m):
            print(f"[SKIP] Missing prices: 15M={up_15m}/{down_15m} 5M={up_5m}/{down_5m}")
            return None

        # Also try live CLOB book for more accurate prices
        up_tid_15m, down_tid_15m = self.get_token_ids(m15)
        up_tid_5m, down_tid_5m = self.get_token_ids(m5)

        if not all([up_tid_15m, down_tid_15m, up_tid_5m, down_tid_5m]):
            print(f"[SKIP] Missing token IDs")
            return None

        # Fetch live asks for the sides we want to buy
        # Optimal side selection: buy the side whose divergence = jackpot
        if gap < 0:
            # current < 15M open → buy 15M DOWN + 5M UP
            side_15m, side_5m = "DOWN", "UP"
            tid_15m, tid_5m = down_tid_15m, up_tid_5m
            # Use Gamma prices as baseline, try CLOB for accuracy
            price_15m_gamma = down_15m
            price_5m_gamma = up_5m
        else:
            # current > 15M open → buy 15M UP + 5M DOWN
            side_15m, side_5m = "UP", "DOWN"
            tid_15m, tid_5m = up_tid_15m, down_tid_5m
            price_15m_gamma = up_15m
            price_5m_gamma = down_5m

        # Try live CLOB asks (more accurate than Gamma cached prices)
        ask_15m, ask_5m = await asyncio.gather(
            self.get_best_ask(tid_15m),
            self.get_best_ask(tid_5m),
        )

        # Use CLOB if available, else fallback to Gamma
        price_15m = ask_15m if ask_15m is not None else price_15m_gamma
        price_5m = ask_5m if ask_5m is not None else price_5m_gamma

        combined = price_15m + price_5m

        # Log the evaluation
        q15 = m15.get("question", "")[:40]
        q5 = m5.get("question", "")[:40]
        print(f"[EVAL] {q15}")
        print(f"       BTC=${current:,.2f} | 15M open=${open_15m:,.2f} | gap=${gap:+.2f}")
        print(f"       15M {side_15m} @ ${price_15m:.2f} + 5M {side_5m} @ ${price_5m:.2f} = ${combined:.2f}")

        if combined > MAX_COMBINED_COST:
            print(f"       SKIP: combined ${combined:.2f} > max ${MAX_COMBINED_COST}")
            self.stats["pairs_skipped"] += 1
            return None

        profit_pct = (1.0 - combined) / combined * 100
        print(f"       ENTRY! profit potential: {profit_pct:.1f}% | gap: ${gap:+.2f}")

        return {
            "pair_key": pair_key,
            "end_dt": end_dt.isoformat(),
            "end_dt_parsed": end_dt,
            "m15_question": m15.get("question", ""),
            "m5_question": m5.get("question", ""),
            "m15_cid": m15.get("conditionId", ""),
            "m5_cid": m5.get("conditionId", ""),
            "m15_market_id": m15.get("id", ""),
            "m5_market_id": m5.get("id", ""),
            "side_15m": side_15m,
            "side_5m": side_5m,
            "tid_15m": tid_15m,
            "tid_5m": tid_5m,
            "price_15m": price_15m,
            "price_5m": price_5m,
            "combined": combined,
            "gap": gap,
            "open_15m": open_15m,
            "current_at_entry": current,
        }

    def enter_pair(self, entry: dict):
        """Record a paper arb entry."""
        pair_key = entry["pair_key"]
        self.attempted_pairs.add(pair_key)

        shares_15m = max(5, int(TRADE_SIZE / entry["price_15m"]))
        shares_5m = max(5, int(TRADE_SIZE / entry["price_5m"]))
        cost_15m = round(shares_15m * entry["price_15m"], 2)
        cost_5m = round(shares_5m * entry["price_5m"], 2)

        trade = {
            "pair_key": pair_key,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "end_dt": entry["end_dt"],
            "leg_15m": {
                "cid": entry["m15_cid"],
                "market_id": entry["m15_market_id"],
                "question": entry["m15_question"],
                "side": entry["side_15m"],
                "token_id": entry["tid_15m"],
                "entry_price": entry["price_15m"],
                "shares": shares_15m,
                "cost": cost_15m,
            },
            "leg_5m": {
                "cid": entry["m5_cid"],
                "market_id": entry["m5_market_id"],
                "question": entry["m5_question"],
                "side": entry["side_5m"],
                "token_id": entry["tid_5m"],
                "entry_price": entry["price_5m"],
                "shares": shares_5m,
                "cost": cost_5m,
            },
            "combined_cost": round(entry["price_15m"] + entry["price_5m"], 4),
            "gap_at_entry": round(entry["gap"], 2),
            "open_15m": entry["open_15m"],
            "current_at_entry": entry["current_at_entry"],
            "status": "open",
            "paper": self.paper,
        }

        self.active[pair_key] = trade
        self.stats["pairs_entered"] += 1
        self._save()

        s15 = entry["side_15m"]
        s5 = entry["side_5m"]
        p15 = entry["price_15m"]
        p5 = entry["price_5m"]
        comb = entry["combined"]
        gap = entry["gap"]
        mode = "PAPER" if self.paper else "LIVE"
        print(f"\n{'='*60}")
        print(f"[ENTER] {mode} CROSS-ARB")
        print(f"  15M {s15} @ ${p15:.2f} ({shares_15m} sh, ${cost_15m:.2f})")
        print(f"  5M  {s5} @ ${p5:.2f} ({shares_5m} sh, ${cost_5m:.2f})")
        print(f"  Combined: ${comb:.2f} | Gap: ${gap:+.2f}")
        print(f"  Profit if one wins: ${1.0 - comb:.2f}/share")
        print(f"{'='*60}\n")

    # ========================================================================
    # RESOLUTION
    # ========================================================================

    async def resolve_pairs(self):
        """Check and resolve completed pairs."""
        now = datetime.now(timezone.utc)
        to_resolve = []

        for key, trade in list(self.active.items()):
            end_dt = datetime.fromisoformat(trade["end_dt"])
            secs_past = (now - end_dt).total_seconds()
            if secs_past >= RESOLVE_DELAY:
                to_resolve.append((key, trade, end_dt))

        for key, trade, end_dt in to_resolve:
            result_15m, result_5m = await asyncio.gather(
                self.resolve_binance(end_dt, "15m", 15),
                self.resolve_binance(end_dt, "5m", 5),
            )

            if result_15m is None or result_5m is None:
                secs_past = (now - end_dt).total_seconds()
                if secs_past > 300:  # 5 min past close, give up
                    print(f"[RESOLVE] {key}: Binance data unavailable, marking as unknown")
                    self._finalize_pair(key, trade, "UNKNOWN", "UNKNOWN")
                continue

            self._finalize_pair(key, trade, result_15m, result_5m)

    def _finalize_pair(self, key: str, trade: dict, result_15m: str, result_5m: str):
        """Calculate PnL and record final result."""
        leg_15m = trade["leg_15m"]
        leg_5m = trade["leg_5m"]

        # 15M leg result
        won_15m = (leg_15m["side"] == result_15m)
        pnl_15m = (1.0 - leg_15m["entry_price"]) * leg_15m["shares"] if won_15m \
            else -leg_15m["entry_price"] * leg_15m["shares"]
        pnl_15m = round(pnl_15m, 2)

        # 5M leg result
        won_5m = (leg_5m["side"] == result_5m)
        pnl_5m = (1.0 - leg_5m["entry_price"]) * leg_5m["shares"] if won_5m \
            else -leg_5m["entry_price"] * leg_5m["shares"]
        pnl_5m = round(pnl_5m, 2)

        total_pnl = round(pnl_15m + pnl_5m, 2)

        # Classify outcome
        if won_15m and won_5m:
            correlation = "diverged_win"  # Both won = jackpot (outcomes diverged in our favor)
            self.stats["double_wins"] += 1
        elif not won_15m and not won_5m:
            correlation = "diverged_loss"  # Both lost = worst case
            self.stats["double_losses"] += 1
        elif won_15m or won_5m:
            correlation = "correlated"  # One won = expected (outcomes were same direction)
            self.stats["correlated"] += 1
            if total_pnl >= 0:
                self.stats["wins"] += 1
            else:
                self.stats["losses"] += 1

        self.stats["pnl"] = round(self.stats["pnl"] + total_pnl, 2)
        self.session_pnl = round(self.session_pnl + total_pnl, 2)

        # Build resolved record
        resolved = {
            **trade,
            "status": "resolved",
            "result_15m": result_15m,
            "result_5m": result_5m,
            "won_15m": won_15m,
            "won_5m": won_5m,
            "pnl_15m": pnl_15m,
            "pnl_5m": pnl_5m,
            "total_pnl": total_pnl,
            "correlation": correlation,
            "resolve_time": datetime.now(timezone.utc).isoformat(),
        }

        self.resolved.append(resolved)
        del self.active[key]
        self._save()

        # Log result
        w15 = "WIN" if won_15m else "LOSS"
        w5 = "WIN" if won_5m else "LOSS"
        emoji_map = {
            "diverged_win": "DOUBLE WIN",
            "diverged_loss": "DOUBLE LOSS",
            "correlated": "ONE WIN",
        }
        tag = emoji_map.get(correlation, correlation)

        w = self.stats["wins"] + self.stats["double_wins"]
        l = self.stats["losses"] + self.stats["double_losses"]
        total = w + l
        wr = w / total * 100 if total > 0 else 0

        print(f"\n{'='*60}")
        print(f"[RESOLVE] {tag} | {correlation}")
        print(f"  15M: {result_15m} → {leg_15m['side']} = {w15} (${pnl_15m:+.2f})")
        print(f"  5M:  {result_5m} → {leg_5m['side']} = {w5} (${pnl_5m:+.2f})")
        print(f"  Total: ${total_pnl:+.2f} | Session: ${self.session_pnl:+.2f}")
        print(f"  Record: {w}W/{l}L {wr:.0f}%WR | All-time: ${self.stats['pnl']:+.2f}")
        print(f"{'='*60}\n")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        mode = "PAPER" if self.paper else "LIVE"
        print(f"\n{'='*60}")
        print(f"  CROSS-TIMEFRAME ARB V1.1 -- {mode} MODE")
        print(f"  15M + 5M BTC Up/Down Structural Arbitrage")
        print(f"  Buy opposing sides when BTC near 15M open")
        print(f"  Max combined: ${MAX_COMBINED_COST} | Proximity: ${PROXIMITY_MIN}-${PROXIMITY_MAX}")
        print(f"  Trade size: ${TRADE_SIZE}/leg | Max concurrent: {MAX_CONCURRENT}")
        print(f"  V1.1: +Polymarket WS (real-time prices, instant market detection)")
        print(f"{'='*60}\n")

        # V1.1: Start Polymarket WebSocket
        await self.poly_ws.start()

        cycle = 0
        while self.running:
            try:
                cycle += 1
                now = datetime.now(timezone.utc)

                # Daily loss limit
                if self.session_pnl < -DAILY_LOSS_LIMIT:
                    print(f"[HALT] Daily loss limit hit: ${self.session_pnl:.2f}")
                    await asyncio.sleep(300)
                    continue

                # Resolve completed pairs
                await self.resolve_pairs()

                # Discover and evaluate new pairs
                pairs = await self.discover_pairs()

                for m15, m5 in pairs:
                    entry = await self.evaluate_pair(m15, m5)
                    if entry:
                        self.enter_pair(entry)

                # Periodic status
                if time.time() - self._last_log_time > 60:
                    self._last_log_time = time.time()
                    w = self.stats["wins"] + self.stats["double_wins"]
                    l = self.stats["losses"] + self.stats["double_losses"]
                    total = w + l
                    wr = w / total * 100 if total > 0 else 0
                    active_keys = list(self.active.keys())
                    price = self._last_price_cache[1]
                    ts = now.strftime("%H:%M:%S")
                    print(f"[{ts}] C{cycle} | {w}W/{l}L {wr:.0f}%WR | "
                          f"PnL: ${self.stats['pnl']:+.2f} | "
                          f"Active: {len(self.active)} | "
                          f"Entered: {self.stats['pairs_entered']} | "
                          f"Skipped: {self.stats['pairs_skipped']} | "
                          f"BTC: ${price:,.0f}" if price > 0 else
                          f"[{ts}] C{cycle} | Waiting for pairs...")

                await asyncio.sleep(SCAN_INTERVAL)

            except KeyboardInterrupt:
                print("\n[EXIT] Shutting down...")
                self.running = False
                break
            except Exception as e:
                print(f"[ERROR] Cycle {cycle}: {e}")
                traceback.print_exc()
                await asyncio.sleep(30)


# ============================================================================
# MAIN
# ============================================================================
def main():
    paper = "--live" not in sys.argv
    pid_file = acquire_pid_lock("cross_arb")

    try:
        bot = CrossTimeframeArb(paper=paper)
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted")
    finally:
        release_pid_lock("cross_arb")


if __name__ == "__main__":
    main()
