"""
5M Pre-Placement Sniper V1.0 — PAPER MODE

Places GTC orders at $0.45 on freshly-created 5-minute BTC markets BEFORE
market makers set prices. The edge: be first in the order book.

How it works:
  1. Monitor BTC momentum (Binance 1m candles) to predict direction
  2. Predict when 5M markets open (every 15 min at :10, :25, :40, :55 UTC)
  3. Rapid-poll Gamma API starting 30s before expected open
  4. The moment a new 5M market appears, place GTC buy at $0.45 on predicted side
  5. Cancel after 30s if not filled
  6. Wait for resolution at market close

Economics:
  - Fill at $0.45 → win pays $0.55/share (122% ROI), lose costs $0.45/share
  - Break-even WR = 45%. Momentum signal should deliver 55-65%.
  - If filled early and price moves to $0.50+, we pocket the spread even on loss.

Usage:
  python -u run_preplace_5m.py              # Paper mode (default)
  python -u run_preplace_5m.py --live       # Live mode (real CLOB orders)
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
from typing import Dict, Optional, Tuple

import httpx
from dotenv import load_dotenv

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# CONFIG
# ============================================================================
TARGET_PRICE = 0.50             # GTC limit at fair value — edge from momentum accuracy
MAX_ENTRY_PRICE = 0.58          # abort if best ask already above this
TRADE_SIZE_USD = 5.00           # USD per trade
MIN_SHARES = 5                  # CLOB minimum
GTC_TIMEOUT = 30                # seconds to wait for fill before cancelling
PRE_OPEN_LEAD = 45              # seconds before predicted open to start rapid polling
RAPID_POLL_INTERVAL = 2         # seconds between rapid polls during pre-open window
SCAN_INTERVAL = 15              # seconds between idle scans
MOMENTUM_LOOKBACK_10M = 10      # bars for 10m momentum (1m candles)
MOMENTUM_LOOKBACK_5M = 5        # bars for 5m momentum
MIN_MOMENTUM = 0.0001           # 0.01% minimum — lowered for testing (was 0.03%)
MAX_CONCURRENT = 2              # max open pre-placed trades
DAILY_LOSS_LIMIT = 20.0         # stop if session losses exceed this
RESOLVE_DELAY = 20              # seconds after close before resolving
FOK_SLIPPAGE = 0.02             # additional slippage buffer for live GTC

RESULTS_FILE = Path(__file__).parent / "preplace_5m_results.json"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
GAMMA_API_URL = "https://gamma-api.polymarket.com/events"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"

# 5M markets open at these minutes past the hour (10 min into each 15M window)
# 15M windows: :00-:15, :15-:30, :30-:45, :45-:00
# 5M markets covering last 5 min open at: :10, :25, :40, :55
OPEN_MINUTES = [10, 25, 40, 55]


# ============================================================================
# PRE-PLACEMENT SNIPER
# ============================================================================
class PreplacementSniper:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.client = None
        self.trades: Dict[str, dict] = {}       # active trades
        self.resolved: list = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0,
                      "fills": 0, "misses": 0, "cancelled": 0}
        self.session_pnl = 0.0
        self.attempted_cids: set = set()
        self.known_market_ids: set = set()       # track discovered 5M market IDs
        self.running = True
        self.max_trades = 0                      # 0 = unlimited
        self._last_log_time = 0.0
        self._momentum_cache = (0.0, 0.0, 0.0)  # (timestamp, mom_10m, mom_5m)
        self._load()

        if not self.paper:
            self._init_clob()

    # ========================================================================
    # CLOB CLIENT
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
                self.trades = data.get("active", {})
                self.resolved = data.get("resolved", [])
                self.stats = {**self.stats, **data.get("stats", {})}
                for cid in self.trades:
                    self.attempted_cids.add(cid)
                for t in self.resolved:
                    cid = t.get("condition_id", "")
                    if cid:
                        self.attempted_cids.add(cid)
                w, l = self.stats["wins"], self.stats["losses"]
                total = w + l
                wr = w / total * 100 if total > 0 else 0
                print(f"[LOAD] {len(self.resolved)} resolved, {len(self.trades)} active | "
                      f"{w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
                      f"Fills: {self.stats['fills']} Misses: {self.stats['misses']}")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        data = {
            "active": self.trades,
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
    # MOMENTUM SIGNAL (from Binance 1m candles)
    # ========================================================================

    async def get_momentum(self) -> Tuple[float, float]:
        """Get 10m and 5m momentum from Binance 1m candles. Cached 10s."""
        now = time.time()
        if now - self._momentum_cache[0] < 10:
            return self._momentum_cache[1], self._momentum_cache[2]
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(BINANCE_REST_URL, params={
                    "symbol": "BTCUSDT", "interval": "1m", "limit": 15,
                })
                r.raise_for_status()
                klines = r.json()
                if len(klines) < 12:
                    return 0.0, 0.0

                closes = [float(k[4]) for k in klines]
                current = closes[-1]

                # 10m momentum: current vs 10 bars ago
                mom_10m = (current - closes[-11]) / closes[-11] if closes[-11] > 0 else 0
                # 5m momentum: current vs 5 bars ago
                mom_5m = (current - closes[-6]) / closes[-6] if closes[-6] > 0 else 0

                self._momentum_cache = (now, mom_10m, mom_5m)
                return mom_10m, mom_5m
        except Exception as e:
            print(f"[BINANCE] Momentum error: {e}")
            return 0.0, 0.0

    def predict_direction(self, mom_10m: float, mom_5m: float) -> Optional[str]:
        """Predict UP or DOWN from momentum. Returns None if unclear."""
        # Both must agree and exceed minimum threshold
        if mom_10m > MIN_MOMENTUM and mom_5m > MIN_MOMENTUM:
            return "UP"
        elif mom_10m < -MIN_MOMENTUM and mom_5m < -MIN_MOMENTUM:
            return "DOWN"
        return None

    # ========================================================================
    # TIMING: PREDICT NEXT 5M MARKET OPEN
    # ========================================================================

    def next_open_time(self) -> datetime:
        """Calculate when the next 5M market should open (UTC).
        5M markets open at :10, :25, :40, :55 past each hour."""
        now = datetime.now(timezone.utc)
        current_minute = now.minute
        current_second = now.second

        for om in OPEN_MINUTES:
            if om > current_minute or (om == current_minute and current_second < 30):
                return now.replace(minute=om, second=0, microsecond=0)

        # Next hour, first slot
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return next_hour.replace(minute=OPEN_MINUTES[0])

    def seconds_to_next_open(self) -> float:
        """Seconds until next predicted 5M market open."""
        return (self.next_open_time() - datetime.now(timezone.utc)).total_seconds()

    # ========================================================================
    # 5M MARKET DISCOVERY
    # ========================================================================

    async def discover_5m_markets(self) -> list:
        """Fetch active 5M BTC markets from Gamma API."""
        markets = []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    GAMMA_API_URL,
                    params={"tag_slug": "5M", "active": "true",
                            "closed": "false", "limit": 200},
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
                        if not end_date:
                            continue
                        try:
                            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                            if end_dt < now:
                                continue
                            time_left_sec = (end_dt - now).total_seconds()
                            m["_time_left_sec"] = time_left_sec
                            m["_end_dt_parsed"] = end_dt
                        except Exception:
                            continue
                        if not m.get("question"):
                            m["question"] = event.get("title", "")
                        markets.append(m)
        except Exception as e:
            print(f"[API] 5M discovery error: {e}")
        return markets

    def find_new_markets(self, markets: list) -> list:
        """Find markets we haven't seen before."""
        new = []
        for m in markets:
            mid = m.get("id") or m.get("conditionId", "")
            if mid and mid not in self.known_market_ids:
                new.append(m)
                self.known_market_ids.add(mid)
        return new

    # ========================================================================
    # MARKET HELPERS
    # ========================================================================

    def get_token_ids(self, market: dict) -> Tuple[Optional[str], Optional[str]]:
        """Extract (UP_token_id, DOWN_token_id)."""
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

    async def get_best_ask(self, token_id: str) -> Optional[float]:
        """Get best ask from CLOB book."""
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(CLOB_BOOK_URL, params={"token_id": token_id})
                if r.status_code != 200:
                    return None
                book = r.json()
                asks = book.get("asks", [])
                if not asks:
                    return None
                return min(float(a["price"]) for a in asks)
        except Exception:
            return None

    # ========================================================================
    # BINANCE RESOLUTION
    # ========================================================================

    async def resolve_binance(self, end_dt: datetime) -> Optional[str]:
        """Resolve 5M market via Binance data."""
        try:
            start_dt = end_dt - timedelta(minutes=5)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(BINANCE_REST_URL, params={
                    "symbol": "BTCUSDT", "interval": "5m",
                    "startTime": start_ms, "endTime": end_ms, "limit": 3,
                })
                r.raise_for_status()
                klines = r.json()
                if not klines:
                    # Fallback: 1m candles
                    r2 = await client.get(BINANCE_REST_URL, params={
                        "symbol": "BTCUSDT", "interval": "1m",
                        "startTime": start_ms, "endTime": end_ms, "limit": 10,
                    })
                    r2.raise_for_status()
                    klines_1m = r2.json()
                    if not klines_1m:
                        return None
                    open_p = float(klines_1m[0][1])
                    close_p = float(klines_1m[-1][4])
                else:
                    best = min(klines, key=lambda k: abs(int(k[0]) - start_ms))
                    open_p = float(best[1])
                    close_p = float(best[4])
                return "UP" if close_p >= open_p else "DOWN"
        except Exception as e:
            print(f"[BINANCE] Resolution error: {e}")
            return None

    # ========================================================================
    # FILL VERIFICATION
    # ========================================================================

    def _verify_fill(self, order_id: str, entry_price: float,
                     shares: float, cost: float) -> tuple:
        """Check if a GTC order has been filled. Returns (filled, price, shares, cost)."""
        try:
            order_info = self.client.get_order(order_id)
            if isinstance(order_info, str):
                try:
                    order_info = json.loads(order_info)
                except json.JSONDecodeError:
                    return (False, entry_price, shares, cost)
            if not isinstance(order_info, dict):
                return (False, entry_price, shares, cost)

            size_matched = float(order_info.get("size_matched", 0) or 0)
            if size_matched >= MIN_SHARES:
                assoc = order_info.get("associate_trades", [])
                if assoc and assoc[0].get("price"):
                    entry_price = float(assoc[0]["price"])
                    cost = round(entry_price * size_matched, 2)
                    shares = size_matched
                return (True, entry_price, shares, cost)
            return (False, entry_price, shares, cost)
        except Exception:
            return (False, entry_price, shares, cost)

    # ========================================================================
    # CORE: PRE-PLACEMENT EXECUTION
    # ========================================================================

    async def attempt_preplace(self, market: dict, predicted_side: str,
                               mom_10m: float, mom_5m: float) -> bool:
        """Attempt to pre-place a GTC order on a newly discovered 5M market."""
        cid = market.get("conditionId", "")
        mid = market.get("id", "")
        question = market.get("question", "?")[:60]
        end_dt = market.get("_end_dt_parsed")

        if cid in self.attempted_cids:
            return False
        if len(self.trades) >= MAX_CONCURRENT:
            return False

        up_tid, down_tid = self.get_token_ids(market)
        if not up_tid or not down_tid:
            print(f"[SKIP] No token IDs: {question}")
            return False

        token_id = up_tid if predicted_side == "UP" else down_tid

        # Check if orderbook already has asks below our target (MM beat us)
        best_ask = await self.get_best_ask(token_id)
        book_status = "EMPTY"
        if best_ask is not None:
            book_status = f"${best_ask:.2f}"
            if best_ask <= TARGET_PRICE:
                # Someone already has asks at or below our price — we'd be buying from them, not sniping
                print(f"[SKIP] Book already has ask @ ${best_ask:.2f} <= ${TARGET_PRICE} | {question}")
                self.attempted_cids.add(cid)
                return False
            if best_ask > MAX_ENTRY_PRICE:
                # Book is priced too high — we'd never fill at $0.45
                print(f"[SKIP] Ask ${best_ask:.2f} > max ${MAX_ENTRY_PRICE} — MM priced out | {question}")
                self.attempted_cids.add(cid)
                return False

        self.attempted_cids.add(cid)

        # Calculate shares
        shares = max(MIN_SHARES, int(TRADE_SIZE_USD / TARGET_PRICE))
        cost = round(shares * TARGET_PRICE, 2)

        # === PLACE ORDER ===
        order_id = None
        fill_confirmed = False
        fill_price = TARGET_PRICE

        if not self.paper and self.client:
            try:
                from py_clob_client.clob_types import OrderArgs, OrderType
                from py_clob_client.order_builder.constants import BUY
                import time as _t

                gtc_price = TARGET_PRICE

                order_args = OrderArgs(
                    price=gtc_price,
                    size=shares,
                    side=BUY,
                    token_id=token_id,
                )
                print(f"[LIVE] GTC {predicted_side} {shares}sh @ ${gtc_price:.2f} | {question}")
                signed = self.client.create_order(order_args)
                resp = self.client.post_order(signed, OrderType.GTC)

                if not resp.get("success"):
                    print(f"[LIVE] GTC failed: {resp.get('errorMsg', '?')}")
                    return False

                order_id = resp.get("orderID", "?")
                status = resp.get("status", "")
                print(f"[LIVE] GTC posted: {order_id[:20]}... status={status}")

                if status == "matched":
                    # Verify fill details
                    _t.sleep(2)
                    filled, fp, fs, fc = self._verify_fill(order_id, gtc_price, shares, cost)
                    if filled:
                        fill_confirmed = True
                        fill_price, shares, cost = fp, fs, fc
                        self.stats["fills"] += 1
                    else:
                        # Trust the match even if verify fails
                        fill_confirmed = True
                        fill_price = gtc_price
                        self.stats["fills"] += 1
                    print(f"[LIVE] INSTANT FILL! {shares}sh @ ${fill_price:.2f}")
                else:
                    # Poll for fill
                    print(f"[LIVE] Polling for fill ({GTC_TIMEOUT}s max)...")
                    for wait_round in range(GTC_TIMEOUT // 5):
                        _t.sleep(5)
                        filled, fp, fs, fc = self._verify_fill(
                            order_id, gtc_price, shares, cost)
                        if filled:
                            fill_confirmed = True
                            fill_price, shares, cost = fp, fs, fc
                            self.stats["fills"] += 1
                            print(f"[LIVE] FILLED after {(wait_round+1)*5}s: "
                                  f"{shares}sh @ ${fill_price:.2f}")
                            break

                    if not fill_confirmed:
                        try:
                            self.client.cancel(order_id)
                            print(f"[LIVE] Unfilled after {GTC_TIMEOUT}s — cancelled")
                        except Exception:
                            pass
                        self.stats["misses"] += 1
                        return False

            except Exception as e:
                print(f"[LIVE] Order error: {e}")
                traceback.print_exc()
                return False
        else:
            # PAPER MODE: simulate fill based on whether book was empty or had favorable asks
            if book_status == "EMPTY":
                # Empty book = we'd be first! Simulate fill at target price
                fill_confirmed = True
                fill_price = TARGET_PRICE
                self.stats["fills"] += 1
                print(f"[PAPER] FILL (empty book) {predicted_side} {shares}sh @ ${fill_price:.2f} | {question}")
            elif best_ask is not None and best_ask > TARGET_PRICE:
                # Book exists but ask is above our target — we'd be best bid, simulating fill
                # In reality, someone might sell into our bid if momentum shifts
                # Conservative: 50% fill probability when book exists but spread is wide
                import random
                if best_ask - TARGET_PRICE > 0.10:
                    # Wide spread — likely to get filled as book builds
                    fill_confirmed = True
                    fill_price = TARGET_PRICE
                    self.stats["fills"] += 1
                    print(f"[PAPER] FILL (wide spread ${best_ask:.2f}) {predicted_side} "
                          f"{shares}sh @ ${fill_price:.2f} | {question}")
                elif random.random() < 0.3:
                    fill_confirmed = True
                    fill_price = TARGET_PRICE
                    self.stats["fills"] += 1
                    print(f"[PAPER] FILL (narrow spread ${best_ask:.2f}, 30% sim) {predicted_side} "
                          f"{shares}sh @ ${fill_price:.2f} | {question}")
                else:
                    self.stats["misses"] += 1
                    print(f"[PAPER] MISS (narrow spread ${best_ask:.2f}) | {question}")
                    return False
            else:
                self.stats["misses"] += 1
                print(f"[PAPER] MISS (no fill condition) | {question}")
                return False

        if not fill_confirmed:
            return False

        # Record trade
        trade_key = f"pre5m_{cid}"
        self.trades[trade_key] = {
            "side": predicted_side,
            "entry_price": fill_price,
            "size_usd": cost,
            "_shares": shares,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "condition_id": cid,
            "market_id": mid,
            "question": market.get("question", ""),
            "end_dt": end_dt.isoformat() if end_dt else "",
            "strategy": "preplace_5m",
            "momentum_10m": round(mom_10m, 6),
            "momentum_5m": round(mom_5m, 6),
            "order_id": order_id,
            "book_at_entry": book_status,
            "paper": self.paper,
            "_token_ids": [up_tid, down_tid],
        }
        self._save()

        mode = "PAPER" if self.paper else "LIVE"
        print(f"\n{'='*60}")
        print(f"  [{mode}] PRE-PLACED 5M SNIPE")
        print(f"  {predicted_side} {shares}sh @ ${fill_price:.2f} (${cost:.2f})")
        print(f"  Mom: 10m={mom_10m:+.4%} 5m={mom_5m:+.4%}")
        print(f"  Book: {book_status} | {question}")
        print(f"  Win payout: ${shares * (1.0 - fill_price):.2f} | "
              f"Loss: -${cost:.2f}")
        print(f"{'='*60}\n")
        return True

    # ========================================================================
    # RESOLVE TRADES
    # ========================================================================

    async def resolve_trades(self):
        """Resolve expired pre-placed trades."""
        now = datetime.now(timezone.utc)
        to_resolve = []

        for key, trade in list(self.trades.items()):
            end_str = trade.get("end_dt", "")
            if not end_str:
                continue
            try:
                end_dt = datetime.fromisoformat(end_str)
            except Exception:
                continue
            secs_past = (now - end_dt).total_seconds()
            if secs_past >= RESOLVE_DELAY:
                to_resolve.append((key, trade, end_dt))

        for key, trade, end_dt in to_resolve:
            result = await self.resolve_binance(end_dt)
            if result is None:
                secs_past = (now - end_dt).total_seconds()
                if secs_past > 300:
                    print(f"[RESOLVE] {key}: Binance data unavailable, marking unknown")
                    self._finalize(key, trade, "UNKNOWN")
                continue
            self._finalize(key, trade, result)

    def _finalize(self, key: str, trade: dict, result: str):
        """Calculate PnL and record final result."""
        side = trade["side"]
        entry_price = trade["entry_price"]
        shares = trade["_shares"]

        won = (side == result) if result != "UNKNOWN" else False
        if won:
            pnl = round((1.0 - entry_price) * shares, 2)
            self.stats["wins"] += 1
        else:
            pnl = round(-entry_price * shares, 2)
            self.stats["losses"] += 1

        self.stats["pnl"] = round(self.stats["pnl"] + pnl, 2)
        self.session_pnl = round(self.session_pnl + pnl, 2)

        resolved = {
            **trade,
            "status": "resolved",
            "result": result,
            "won": won,
            "pnl": pnl,
            "resolve_time": datetime.now(timezone.utc).isoformat(),
        }
        self.resolved.append(resolved)
        del self.trades[key]
        self._save()

        w = self.stats["wins"]
        l = self.stats["losses"]
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        tag = "WIN" if won else "LOSS"

        print(f"\n{'='*60}")
        print(f"  [RESOLVE] {tag} | {side} vs actual {result}")
        print(f"  PnL: ${pnl:+.2f} | Session: ${self.session_pnl:+.2f}")
        print(f"  Record: {w}W/{l}L {wr:.0f}%WR | All-time: ${self.stats['pnl']:+.2f}")
        print(f"  Fills: {self.stats['fills']} | Misses: {self.stats['misses']}")
        print(f"{'='*60}\n")

    # ========================================================================
    # WAIT FOR ALL TRADES TO RESOLVE (for --max-trades mode)
    # ========================================================================

    async def _wait_and_resolve(self):
        """Wait for all active trades to resolve before exiting."""
        while self.trades:
            print(f"[WAIT] {len(self.trades)} trade(s) pending resolution...")
            await self.resolve_trades()
            if self.trades:
                await asyncio.sleep(10)
        print(f"[DONE] All trades resolved. Final: {self.stats['wins']}W/{self.stats['losses']}L | "
              f"PnL: ${self.stats['pnl']:+.2f}")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        mode = "PAPER" if self.paper else "LIVE"
        print(f"\n{'='*70}")
        print(f"  5M PRE-PLACEMENT SNIPER V1.0 — {mode} MODE")
        print(f"  Strategy: Place GTC @ ${TARGET_PRICE} on new 5M markets before MMs")
        print(f"  Momentum: 10m+5m aligned from Binance 1m candles")
        print(f"  Markets open at: :10, :25, :40, :55 UTC (10 min into each 15M window)")
        print(f"  Target: ${TARGET_PRICE} | Max: ${MAX_ENTRY_PRICE} | Size: ${TRADE_SIZE_USD}")
        print(f"  GTC timeout: {GTC_TIMEOUT}s | Daily loss limit: ${DAILY_LOSS_LIMIT}")
        if self.max_trades > 0:
            print(f"  MAX TRADES: {self.max_trades} — will exit after fill + resolution")
        print(f"{'='*70}\n")

        # Initial scan to populate known market IDs (don't treat existing as "new")
        initial = await self.discover_5m_markets()
        for m in initial:
            mid = m.get("id") or m.get("conditionId", "")
            if mid:
                self.known_market_ids.add(mid)
        print(f"[INIT] Populated {len(self.known_market_ids)} existing 5M market IDs")

        cycle = 0
        while self.running:
            try:
                cycle += 1
                now = datetime.now(timezone.utc)

                # Daily loss limit
                if self.session_pnl < -DAILY_LOSS_LIMIT:
                    if time.time() - self._last_log_time > 60:
                        print(f"[HALT] Daily loss limit hit: ${self.session_pnl:.2f}")
                        self._last_log_time = time.time()
                    await asyncio.sleep(60)
                    continue

                # Resolve completed trades
                await self.resolve_trades()

                # Calculate time to next predicted 5M market open
                secs_to_open = self.seconds_to_next_open()
                next_open = self.next_open_time()

                # Check momentum
                mom_10m, mom_5m = await self.get_momentum()
                predicted = self.predict_direction(mom_10m, mom_5m)

                # Periodic status log
                if time.time() - self._last_log_time > 30:
                    self._last_log_time = time.time()
                    mst = (now - timedelta(hours=7)).strftime("%H:%M:%S")
                    w, l = self.stats["wins"], self.stats["losses"]
                    total = w + l
                    wr = w / total * 100 if total > 0 else 0
                    dir_str = predicted or "UNCLEAR"
                    print(f"[{mst} MST] C{cycle} | {w}W/{l}L {wr:.0f}%WR | "
                          f"PnL: ${self.stats['pnl']:+.2f} | "
                          f"Active: {len(self.trades)} | "
                          f"Fills: {self.stats['fills']}/{self.stats['fills']+self.stats['misses']} | "
                          f"Next 5M: {secs_to_open:.0f}s | "
                          f"Mom: {dir_str} (10m={mom_10m:+.3%} 5m={mom_5m:+.3%})")

                # === PRE-OPEN WINDOW: rapid poll when close to predicted open ===
                if secs_to_open <= PRE_OPEN_LEAD and predicted:
                    print(f"\n[SNIPE] Entering pre-open window! "
                          f"Next 5M in {secs_to_open:.0f}s | Signal: {predicted}")

                    # Rapid poll until we find new market or window expires
                    deadline = time.time() + PRE_OPEN_LEAD + 60  # extra 60s past predicted open
                    found = False
                    poll_count = 0

                    while time.time() < deadline and not found and self.running:
                        poll_count += 1
                        markets = await self.discover_5m_markets()
                        new_markets = self.find_new_markets(markets)

                        if new_markets:
                            # Found new market(s)! Try to pre-place on each
                            for nm in new_markets:
                                q = nm.get("question", "")[:50]
                                tl = nm.get("_time_left_sec", 0)
                                print(f"[NEW] Discovered: {q} | {tl:.0f}s left")

                                # Refresh momentum right before placing
                                mom_10m, mom_5m = await self.get_momentum()
                                predicted = self.predict_direction(mom_10m, mom_5m)
                                if not predicted:
                                    print(f"[SKIP] Momentum unclear at entry time")
                                    continue

                                success = await self.attempt_preplace(
                                    nm, predicted, mom_10m, mom_5m)
                                if success:
                                    found = True
                                    # Check max_trades limit
                                    if self.max_trades > 0 and self.stats["fills"] >= self.max_trades:
                                        print(f"\n[MAX] Reached {self.max_trades} fill(s) — "
                                              f"waiting for resolution then exiting...")
                                        await self._wait_and_resolve()
                                        self.running = False
                                    break
                            if found:
                                break

                        secs_remain = deadline - time.time()
                        if poll_count % 5 == 0:
                            print(f"  [POLL] #{poll_count} | {len(markets)} markets | "
                                  f"{secs_remain:.0f}s remaining")
                        await asyncio.sleep(RAPID_POLL_INTERVAL)

                    if not found:
                        print(f"[SNIPE] Window closed — no new market found after {poll_count} polls")

                    # After snipe attempt, sleep until next window
                    await asyncio.sleep(SCAN_INTERVAL)
                else:
                    # Idle phase — wait for next window
                    sleep_time = min(SCAN_INTERVAL, max(1, secs_to_open - PRE_OPEN_LEAD))
                    await asyncio.sleep(sleep_time)

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-trades", type=int, default=0,
                        help="Stop after N filled trades (0=unlimited)")
    args = parser.parse_args()

    paper = not args.live
    pid_file = acquire_pid_lock("preplace_5m")

    try:
        bot = PreplacementSniper(paper=paper)
        bot.max_trades = args.max_trades
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted")
    finally:
        release_pid_lock("preplace_5m")


if __name__ == "__main__":
    main()
