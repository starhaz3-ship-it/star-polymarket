"""
Sniper 5M LIVE Trader V1.0

Front-runs Chainlink oracle delay on 5-minute BTC Up/Down markets.
Enters at 62-61 seconds before market close when direction is clear from CLOB orderbook.

Auto-scaling:
  - $5/trade for first 20 trades
  - $10/trade after 20 trades with >80% WR
  - $20/trade after 100 trades with >80% WR

Usage:
  python -u run_sniper_5m_live.py --live
"""

import sys
import json
import time
import os
import math
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
SNIPER_WINDOW = 62          # seconds before close to start looking (front-run whales)
MIN_CONFIDENCE = 0.70       # minimum ask price to trigger entry
BASE_TRADE_SIZE = 5.00      # USD per trade (starting size)
SCAN_INTERVAL = 5           # seconds between scans
MAX_CONCURRENT = 1          # only 1 active trade at a time
DAILY_LOSS_LIMIT = 30.0     # stop trading if daily losses exceed this
LOG_INTERVAL = 30           # print status every N seconds
RESOLVE_DELAY = 15          # seconds after market close before resolving
MIN_SHARES = 3              # minimum shares for fill confirmation

RESULTS_FILE = Path(__file__).parent / "sniper_5m_live_results.json"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
GAMMA_API_URL = "https://gamma-api.polymarket.com/events"


def get_trade_size(wins: int, losses: int) -> float:
    """Auto-scale trade size based on performance."""
    total = wins + losses
    if total == 0:
        return BASE_TRADE_SIZE
    wr = wins / total
    if total >= 100 and wr > 0.80:
        return 20.00
    if total >= 20 and wr > 0.80:
        return 10.00
    return BASE_TRADE_SIZE


# ============================================================================
# SNIPER 5M LIVE TRADER
# ============================================================================
class Sniper5MLive:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.active: Dict[str, dict] = {}
        self.resolved: list = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0, "skipped": 0}
        self.session_pnl = 0.0
        self.attempted_cids: set = set()
        self.running = True
        self._last_log_time = 0.0
        self.client = None
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
                self.active = data.get("active", {})
                self.resolved = data.get("resolved", [])
                self.stats = {**self.stats, **data.get("stats", {})}
                for cid in self.active:
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

    async def discover_5m_markets(self) -> list:
        markets = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    GAMMA_API_URL,
                    params={"tag_slug": "5M", "active": "true",
                            "closed": "false", "limit": 200},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if r.status_code != 200:
                    print(f"[API] Gamma error: {r.status_code}")
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
                            m["_end_dt"] = end_dt.isoformat()
                            m["_end_dt_parsed"] = end_dt
                        except Exception:
                            continue
                        if not m.get("question"):
                            m["question"] = event.get("title", "")
                        markets.append(m)
        except Exception as e:
            print(f"[API] Discovery error: {e}")
        return markets

    # ========================================================================
    # ORDERBOOK READING
    # ========================================================================

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

    async def get_best_ask(self, token_id: str) -> Optional[float]:
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(CLOB_BOOK_URL, params={"token_id": token_id})
                if r.status_code != 200:
                    return None
                book = r.json()
                asks = book.get("asks", [])
                if not asks:
                    return None
                # CLOB asks are sorted descending — find the true best ask
                return min(float(a["price"]) for a in asks)
        except Exception:
            return None

    async def get_orderbook_prices(self, market: dict) -> tuple:
        up_tid, down_tid = self.get_token_ids(market)
        if not up_tid or not down_tid:
            return None, None
        up_ask, down_ask = await asyncio.gather(
            self.get_best_ask(up_tid), self.get_best_ask(down_tid)
        )
        return up_ask, down_ask

    # ========================================================================
    # BINANCE
    # ========================================================================

    async def resolve_via_binance(self, end_dt: datetime) -> Optional[str]:
        try:
            start_dt = end_dt - timedelta(minutes=5)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(BINANCE_REST_URL, params={
                    "symbol": "BTCUSDT", "interval": "5m",
                    "startTime": start_ms, "endTime": end_ms, "limit": 5,
                })
                r.raise_for_status()
                klines = r.json()

                if not klines:
                    r2 = await client.get(BINANCE_REST_URL, params={
                        "symbol": "BTCUSDT", "interval": "1m",
                        "startTime": start_ms, "endTime": end_ms, "limit": 10,
                    })
                    r2.raise_for_status()
                    klines_1m = r2.json()
                    if not klines_1m:
                        return None
                    open_price = float(klines_1m[0][1])
                    close_price = float(klines_1m[-1][4])
                else:
                    best_candle = None
                    best_diff = float("inf")
                    for k in klines:
                        diff = abs(int(k[0]) - start_ms)
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
                    return "DOWN"
        except Exception as e:
            print(f"[BINANCE] Resolution error: {e}")
            return None

    async def _check_binance_direction(self, end_dt: datetime) -> Optional[str]:
        try:
            start_dt = end_dt - timedelta(minutes=5)
            start_ms = int(start_dt.timestamp() * 1000)
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(BINANCE_REST_URL, params={
                    "symbol": "BTCUSDT", "interval": "1m",
                    "startTime": start_ms, "endTime": now_ms, "limit": 10,
                })
                if r.status_code != 200:
                    return None
                klines = r.json()
                if not klines:
                    return None
                open_price = float(klines[0][1])
                close_price = float(klines[-1][4])
                pct = (close_price - open_price) / open_price * 100
                if pct > 0.03:
                    return "UP"
                elif pct < -0.03:
                    return "DOWN"
                return None
        except Exception:
            return None

    # ========================================================================
    # LIVE ORDER EXECUTION
    # ========================================================================

    async def place_live_order(self, market: dict, side: str, entry_price: float,
                                shares: float, trade_size: float) -> Optional[dict]:
        """Place a FOK buy order on the CLOB. Returns fill info or None."""
        if self.paper or not self.client:
            return {"filled": True, "shares": shares, "price": entry_price, "cost": trade_size}

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            up_tid, down_tid = self.get_token_ids(market)
            token_id = up_tid if side == "UP" else down_tid
            if not token_id:
                print(f"[LIVE] No token ID for BTC:{side}")
                return None

            order_args = OrderArgs(
                price=round(entry_price, 2),
                size=shares,
                side=BUY,
                token_id=token_id,
            )
            signed = self.client.create_order(order_args)
            resp = self.client.post_order(signed, OrderType.FOK)

            if not resp.get("success"):
                print(f"[LIVE] FOK failed: {resp.get('errorMsg', '?')}")
                return None

            order_id = resp.get("orderID", "?")
            print(f"[LIVE] FOK order {order_id[:20]}...")

            # Confirm fill
            if order_id and order_id != "?":
                await asyncio.sleep(1)
                try:
                    order_info = self.client.get_order(order_id)
                    # get_order may return string or dict
                    if isinstance(order_info, str):
                        order_info = json.loads(order_info)
                    size_matched = float(order_info.get("size_matched", 0) or 0)
                    if size_matched >= MIN_SHARES:
                        actual_price = entry_price
                        assoc = order_info.get("associate_trades", [])
                        if assoc and isinstance(assoc[0], dict) and assoc[0].get("price"):
                            actual_price = float(assoc[0]["price"])
                        actual_cost = round(actual_price * size_matched, 2)
                        print(f"[LIVE] FILLED: {size_matched:.1f}sh @ ${actual_price:.2f} = ${actual_cost:.2f}")
                        return {
                            "filled": True,
                            "shares": size_matched,
                            "price": actual_price,
                            "cost": actual_cost,
                            "order_id": order_id,
                        }
                    else:
                        status = order_info.get("status", "?")
                        print(f"[LIVE] NOT FILLED: status={status} matched={size_matched}")
                        return None
                except json.JSONDecodeError:
                    # Can't parse — assume filled since FOK succeeded
                    print(f"[LIVE] Fill check parse error — assuming filled (FOK success)")
                    return {
                        "filled": True,
                        "shares": shares,
                        "price": entry_price,
                        "cost": trade_size,
                        "order_id": order_id,
                    }
                except Exception as e:
                    print(f"[LIVE] Fill check error: {e} — assuming filled (FOK success)")
                    return {
                        "filled": True,
                        "shares": shares,
                        "price": entry_price,
                        "cost": trade_size,
                        "order_id": order_id,
                    }
            return None
        except Exception as e:
            print(f"[LIVE] Order error: {e}")
            traceback.print_exc()
            return None

    # ========================================================================
    # SIGNAL + ENTRY
    # ========================================================================

    async def check_sniper_entry(self, markets: list):
        if len(self.active) >= MAX_CONCURRENT:
            return
        if self.session_pnl <= -DAILY_LOSS_LIMIT:
            return

        for market in markets:
            if len(self.active) >= MAX_CONCURRENT:
                break

            cid = market.get("conditionId", "")
            if not cid or cid in self.attempted_cids:
                continue

            time_left_sec = market.get("_time_left_sec", 9999)

            # Only act in sniper window
            if time_left_sec > SNIPER_WINDOW or time_left_sec <= 0:
                continue

            # Read orderbook
            up_ask, down_ask = await self.get_orderbook_prices(market)
            question = market.get("question", "?")

            # Binance fallback for empty books
            if up_ask is None or down_ask is None:
                end_dt = market.get("_end_dt_parsed")
                if end_dt:
                    btc_dir = await self._check_binance_direction(end_dt)
                    if btc_dir:
                        est_price = 0.75 if time_left_sec > 30 else 0.85
                        if btc_dir == "UP":
                            up_ask, down_ask = est_price, 1.0 - est_price
                        else:
                            up_ask, down_ask = 1.0 - est_price, est_price
                        print(f"[BOOK EMPTY] Binance fallback: BTC {btc_dir} | "
                              f"est UP=${up_ask:.2f} DN=${down_ask:.2f} | {time_left_sec:.0f}s left")
                    else:
                        continue
                else:
                    continue

            # Signal logic
            side = None
            confidence = 0.0
            entry_price = 0.0

            if up_ask >= MIN_CONFIDENCE and down_ask >= MIN_CONFIDENCE:
                if up_ask > down_ask:
                    side, confidence, entry_price = "UP", up_ask, up_ask
                else:
                    side, confidence, entry_price = "DOWN", down_ask, down_ask
            elif up_ask >= MIN_CONFIDENCE:
                side, confidence, entry_price = "UP", up_ask, up_ask
            elif down_ask >= MIN_CONFIDENCE:
                side, confidence, entry_price = "DOWN", down_ask, down_ask
            else:
                self.stats["skipped"] += 1
                self.attempted_cids.add(cid)
                print(f"[SKIP] Direction unclear | UP=${up_ask:.2f} DOWN=${down_ask:.2f} "
                      f"| {time_left_sec:.0f}s left | {question[:50]}")
                continue

            # Auto-scale trade size
            trade_size = get_trade_size(self.stats["wins"], self.stats["losses"])
            shares = math.floor(trade_size / entry_price)  # CLOB needs integer shares
            if shares < 1:
                shares = 1

            # Mark as attempted
            self.attempted_cids.add(cid)

            # Place order (live or paper)
            fill = await self.place_live_order(market, side, entry_price, shares, trade_size)
            if fill is None:
                print(f"[SKIP] Order not filled | BTC {side} @ ${entry_price:.2f} | {question[:50]}")
                continue

            # Use actual fill data
            shares = fill["shares"]
            entry_price = fill["price"]
            cost = fill["cost"]

            now = datetime.now(timezone.utc)
            trade = {
                "condition_id": cid,
                "question": question,
                "side": side,
                "entry_price": round(entry_price, 4),
                "confidence": round(confidence, 4),
                "up_ask": round(up_ask, 4),
                "down_ask": round(down_ask, 4),
                "shares": round(shares, 4),
                "cost": cost,
                "trade_size_tier": trade_size,
                "time_remaining_sec": round(time_left_sec, 1),
                "entry_time": now.isoformat(),
                "end_dt": market.get("_end_dt", ""),
                "status": "open",
                "pnl": 0.0,
                "result": None,
                "order_id": fill.get("order_id", ""),
                "live": not self.paper,
            }

            self.active[cid] = trade
            self._save()

            mode = "LIVE" if not self.paper else "PAPER"
            print(f"\n[{mode}] BTC {side} @ ${entry_price:.2f} (conf={confidence:.2f}) | "
                  f"${cost:.2f} ({shares:.1f}sh) | "
                  f"{time_left_sec:.0f}s left | tier=${trade_size:.0f} | "
                  f"UP=${up_ask:.2f} DN=${down_ask:.2f} | "
                  f"{question[:50]}")

    # ========================================================================
    # RESOLUTION
    # ========================================================================

    async def resolve_trades(self):
        now = datetime.now(timezone.utc)
        to_remove = []

        for cid, trade in list(self.active.items()):
            if trade.get("status") != "open":
                continue

            end_dt_str = trade.get("end_dt", "")
            if not end_dt_str:
                continue

            try:
                end_dt = datetime.fromisoformat(end_dt_str)
            except Exception:
                continue

            seconds_past_close = (now - end_dt).total_seconds()
            if seconds_past_close < RESOLVE_DELAY:
                continue

            outcome = await self.resolve_via_binance(end_dt)
            if outcome is None:
                if seconds_past_close > 120:
                    trade["status"] = "closed"
                    trade["result"] = "unknown"
                    trade["pnl"] = round(-trade["cost"], 2)
                    trade["resolve_time"] = now.isoformat()
                    self.stats["losses"] += 1
                    self.stats["pnl"] += trade["pnl"]
                    self.session_pnl += trade["pnl"]
                    self.resolved.append(trade)
                    to_remove.append(cid)
                    print(f"[LOSS] BTC:{trade['side']} ${trade['pnl']:+.2f} | "
                          f"No Binance data | {trade['question'][:45]}")
                continue

            won = (trade["side"] == outcome)
            if won:
                pnl = round(trade["shares"] * 1.0 - trade["cost"], 2)
                trade["result"] = "WIN"
            else:
                pnl = round(-trade["cost"], 2)
                trade["result"] = "LOSS"

            trade["status"] = "closed"
            trade["pnl"] = pnl
            trade["market_outcome"] = outcome
            trade["resolve_time"] = now.isoformat()

            self.stats["wins" if won else "losses"] += 1
            self.stats["pnl"] += pnl
            self.session_pnl += pnl

            self.resolved.append(trade)
            to_remove.append(cid)

            w, l = self.stats["wins"], self.stats["losses"]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            tag = "WIN" if won else "LOSS"
            tier = trade.get("trade_size_tier", BASE_TRADE_SIZE)

            print(f"[{tag}] BTC:{trade['side']} ${pnl:+.2f} | "
                  f"entry=${trade['entry_price']:.2f} tier=${tier:.0f} | "
                  f"market={outcome} | "
                  f"{w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f} | "
                  f"session=${self.session_pnl:+.2f} | "
                  f"{trade['question'][:45]}")

        for cid in to_remove:
            del self.active[cid]
        if to_remove:
            self._save()

    # ========================================================================
    # LOGGING
    # ========================================================================

    def print_status(self, markets: list):
        now = time.time()
        if now - self._last_log_time < LOG_INTERVAL:
            return
        self._last_log_time = now

        w, l = self.stats["wins"], self.stats["losses"]
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        tier = get_trade_size(w, l)

        now_dt = datetime.now(timezone.utc)
        now_mst = now_dt + timedelta(hours=-7)

        closest = None
        closest_time = 9999
        for m in markets:
            tl = m.get("_time_left_sec", 9999)
            if 0 < tl < closest_time:
                closest_time = tl
                closest = m

        market_str = "No active 5M BTC markets"
        sniper_str = "NO"
        if closest:
            q = closest.get("question", "?")[:50]
            tl = closest.get("_time_left_sec", 0)
            sniper_str = "YES" if tl <= SNIPER_WINDOW else "NO"
            market_str = f"{q} | {tl:.0f}s left"

        mode = "LIVE" if not self.paper else "PAPER"
        print(f"\n--- {now_mst.strftime('%H:%M:%S MST')} | "
              f"SNIPER 5M {mode} | "
              f"Active: {len(self.active)} | Resolved: {len(self.resolved)} | "
              f"{w}W/{l}L {wr:.0f}%WR | "
              f"PnL: ${self.stats['pnl']:+.2f} | "
              f"Session: ${self.session_pnl:+.2f} | "
              f"Tier: ${tier:.0f} ---")
        print(f"    Market: {market_str}")
        print(f"    Sniper window: {sniper_str} (< {SNIPER_WINDOW}s)")

        if self.session_pnl <= -DAILY_LOSS_LIMIT:
            print(f"    *** DAILY LOSS LIMIT HIT: ${self.session_pnl:+.2f} ***")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        mode = "LIVE" if not self.paper else "PAPER"
        tier = get_trade_size(self.stats["wins"], self.stats["losses"])
        print("=" * 70)
        print(f"  SNIPER 5M {mode} TRADER V1.0")
        print("=" * 70)
        print(f"  Strategy: Chainlink oracle front-run (62s window)")
        print(f"    - Enter at {SNIPER_WINDOW}s before 5M BTC market close")
        print(f"    - If dominant side ask > ${MIN_CONFIDENCE:.2f}, buy via FOK")
        print(f"    - Binance fallback when CLOB book is empty")
        print(f"    - Resolve via Binance 5M candle")
        print(f"  Size: ${tier:.2f}/trade (auto-scale: $5->$10->$20)")
        print(f"    - $5 base | $10 after 20 trades >80%WR | $20 after 100 trades >80%WR")
        print(f"  Scan: every {SCAN_INTERVAL}s | Daily loss limit: ${DAILY_LOSS_LIMIT:.2f}")
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

                if self.session_pnl <= -DAILY_LOSS_LIMIT:
                    self.print_status([])
                    await asyncio.sleep(60)
                    continue

                markets = await self.discover_5m_markets()
                await self.resolve_trades()
                await self.check_sniper_entry(markets)
                self.print_status(markets)

                if cycle % 12 == 0:
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
    paper = "--live" not in sys.argv
    if paper:
        print("[MODE] PAPER — pass --live for real trading")
    else:
        print("[MODE] *** LIVE TRADING — REAL MONEY ***")

    lock = acquire_pid_lock("sniper_5m_live")
    try:
        trader = Sniper5MLive(paper=paper)
        asyncio.run(trader.run())
    finally:
        release_pid_lock("sniper_5m_live")
