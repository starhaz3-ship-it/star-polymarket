"""
Sniper 5M Paper Trader V1.0

Reverse-engineered from bratanbratishka's strategy:
  - Wait until < 60 seconds before a 5-minute BTC Up/Down market closes
  - Check if odds are already > $0.70 on one side (direction is clear)
  - Paper-bet on the dominant side
  - The edge: when the market is already strongly directional near close,
    it's priced in. bratanbratishka's losses came from unclear markets.

Resolution:
  - Uses Binance REST API to determine if BTC price went up or down
    during the 5-minute window (close > open = UP wins, else DOWN wins)

Usage:
  python -u run_sniper_5m_paper.py
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
SNIPER_WINDOW = 60          # seconds before close to start looking
MIN_CONFIDENCE = 0.70       # minimum ask price to trigger entry
TRADE_SIZE = 2.50           # USD per trade (paper)
SCAN_INTERVAL = 5           # seconds between scans
MAX_CONCURRENT = 1          # only 1 active trade at a time (5M markets are sequential)
DAILY_LOSS_LIMIT = 15.0     # stop trading if daily losses exceed this
LOG_INTERVAL = 30           # print status every N seconds
RESOLVE_DELAY = 15          # seconds after market close before resolving (let candle finish)

RESULTS_FILE = Path(__file__).parent / "sniper_5m_results.json"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
GAMMA_API_URL = "https://gamma-api.polymarket.com/events"


# ============================================================================
# SNIPER 5M PAPER TRADER
# ============================================================================
class Sniper5MTrader:
    def __init__(self):
        self.active: Dict[str, dict] = {}       # {condition_id: trade_dict}
        self.resolved: list = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0, "skipped": 0}
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
                      f"{w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
                      f"Skipped: {self.stats['skipped']}")
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

    async def discover_5m_markets(self) -> list:
        """Find active BTC 5-minute Up/Down markets on Polymarket."""
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
                    # Only BTC markets
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

    async def get_best_ask(self, token_id: str) -> Optional[float]:
        """Get the best (lowest) ask price from the CLOB orderbook for a token."""
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(
                    CLOB_BOOK_URL,
                    params={"token_id": token_id},
                )
                if r.status_code != 200:
                    return None
                book = r.json()
                asks = book.get("asks", [])
                if not asks:
                    return None
                # CLOB asks are sorted descending (highest first) — find the true best ask
                return min(float(a["price"]) for a in asks)
        except Exception as e:
            return None

    async def get_orderbook_prices(self, market: dict) -> tuple:
        """Get best ask prices for UP and DOWN sides.
        Returns (up_ask, down_ask) or (None, None) on error."""
        up_tid, down_tid = self.get_token_ids(market)
        if not up_tid or not down_tid:
            return None, None

        # Fetch both orderbooks in parallel
        up_ask_task = self.get_best_ask(up_tid)
        down_ask_task = self.get_best_ask(down_tid)
        up_ask, down_ask = await asyncio.gather(up_ask_task, down_ask_task)
        return up_ask, down_ask

    # ========================================================================
    # BINANCE RESOLUTION
    # ========================================================================

    async def resolve_via_binance(self, end_dt: datetime) -> Optional[str]:
        """Determine if BTC went UP or DOWN during the 5-minute window.

        The market covers a 5-minute window ending at end_dt.
        We fetch the Binance 5m candle that covers this window and check
        if close > open (UP) or close < open (DOWN).

        Returns 'UP', 'DOWN', or None if data unavailable.
        """
        try:
            # The 5-minute window starts 5 minutes before end_dt
            start_dt = end_dt - timedelta(minutes=5)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient(timeout=15) as client:
                # Fetch the 5-minute candle from Binance
                r = await client.get(
                    BINANCE_REST_URL,
                    params={
                        "symbol": "BTCUSDT",
                        "interval": "5m",
                        "startTime": start_ms,
                        "endTime": end_ms,
                        "limit": 5,
                    },
                )
                r.raise_for_status()
                klines = r.json()

                if not klines:
                    # Fallback: use 1-minute candles
                    r2 = await client.get(
                        BINANCE_REST_URL,
                        params={
                            "symbol": "BTCUSDT",
                            "interval": "1m",
                            "startTime": start_ms,
                            "endTime": end_ms,
                            "limit": 10,
                        },
                    )
                    r2.raise_for_status()
                    klines_1m = r2.json()
                    if not klines_1m:
                        return None
                    open_price = float(klines_1m[0][1])   # first candle open
                    close_price = float(klines_1m[-1][4])  # last candle close
                else:
                    # Find the candle whose open time is closest to start_dt
                    # Binance 5m candles align to :00, :05, :10, etc.
                    # The market window might not align exactly, so we pick
                    # the best matching candle
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
                    # Exact tie — extremely rare, count as DOWN (bratanbratishka convention)
                    return "DOWN"

        except Exception as e:
            print(f"[BINANCE] Resolution error: {e}")
            return None

    async def _check_binance_direction(self, end_dt: datetime) -> Optional[str]:
        """Check real-time BTC price movement during this 5M window.
        Returns 'UP' if price has risen, 'DOWN' if fallen, None if flat/error."""
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
                open_price = float(klines[0][1])       # open of first 1m candle
                close_price = float(klines[-1][4])      # close of latest 1m candle
                pct = (close_price - open_price) / open_price * 100

                if pct > 0.03:  # BTC up by > 0.03%
                    return "UP"
                elif pct < -0.03:
                    return "DOWN"
                return None  # Too flat to call
        except Exception:
            return None

    # ========================================================================
    # SIGNAL + ENTRY
    # ========================================================================

    async def check_sniper_entry(self, markets: list):
        """Check if any 5M market is in the sniper window and has a clear signal."""
        if len(self.active) >= MAX_CONCURRENT:
            return

        # Daily loss check
        if self.session_pnl <= -DAILY_LOSS_LIMIT:
            return

        for market in markets:
            if len(self.active) >= MAX_CONCURRENT:
                break

            cid = market.get("conditionId", "")
            if not cid or cid in self.attempted_cids:
                continue

            time_left_sec = market.get("_time_left_sec", 9999)

            # Only act in sniper window: < SNIPER_WINDOW seconds before close
            if time_left_sec > SNIPER_WINDOW or time_left_sec <= 0:
                continue

            # Read orderbook
            up_ask, down_ask = await self.get_orderbook_prices(market)
            question = market.get("question", "?")

            # If book is empty near close, use Binance price as signal
            if up_ask is None or down_ask is None:
                # Fallback: check BTC price movement via Binance
                end_dt = market.get("_end_dt_parsed")
                if end_dt:
                    btc_dir = await self._check_binance_direction(end_dt)
                    if btc_dir:
                        # Estimate entry price based on time remaining
                        # Closer to close = higher confidence = higher price
                        est_price = 0.75 if time_left_sec > 30 else 0.85
                        if btc_dir == "UP":
                            up_ask, down_ask = est_price, 1.0 - est_price
                        else:
                            up_ask, down_ask = 1.0 - est_price, est_price
                        print(f"[BOOK EMPTY] Using Binance fallback: BTC {btc_dir} | "
                              f"est UP=${up_ask:.2f} DN=${down_ask:.2f} | {time_left_sec:.0f}s left")
                    else:
                        print(f"[BOOK EMPTY] No Binance signal | {time_left_sec:.0f}s left | {question[:40]}")
                        continue
                else:
                    continue

            # Signal logic: dominant side must have ask > MIN_CONFIDENCE
            side = None
            confidence = 0.0
            entry_price = 0.0

            if up_ask >= MIN_CONFIDENCE and down_ask >= MIN_CONFIDENCE:
                # Both sides high — unusual, pick the higher one
                if up_ask > down_ask:
                    side = "UP"
                    confidence = up_ask
                    entry_price = up_ask
                else:
                    side = "DOWN"
                    confidence = down_ask
                    entry_price = down_ask
            elif up_ask >= MIN_CONFIDENCE:
                side = "UP"
                confidence = up_ask
                entry_price = up_ask
            elif down_ask >= MIN_CONFIDENCE:
                side = "DOWN"
                confidence = down_ask
                entry_price = down_ask
            else:
                # Direction unclear — SKIP
                self.stats["skipped"] += 1
                self.attempted_cids.add(cid)
                print(f"[SKIP] Direction unclear | UP ask=${up_ask:.2f} DOWN ask=${down_ask:.2f} "
                      f"| {time_left_sec:.0f}s left | {question[:50]}")
                continue

            # Calculate shares and cost
            shares = TRADE_SIZE / entry_price
            cost = round(shares * entry_price, 2)

            # Mark as attempted
            self.attempted_cids.add(cid)

            # Build trade record
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
                "time_remaining_sec": round(time_left_sec, 1),
                "entry_time": now.isoformat(),
                "end_dt": market.get("_end_dt", ""),
                "status": "open",
                "pnl": 0.0,
                "result": None,
            }

            self.active[cid] = trade
            self._save()

            print(f"\n[ENTRY] BTC {side} @ ${entry_price:.2f} (confidence={confidence:.2f}) | "
                  f"${cost:.2f} ({shares:.2f}sh) | "
                  f"{time_left_sec:.0f}s left | "
                  f"UP=${up_ask:.2f} DOWN=${down_ask:.2f} | "
                  f"{question[:55]}")

    # ========================================================================
    # RESOLUTION
    # ========================================================================

    async def resolve_trades(self):
        """Resolve trades whose 5M windows have closed."""
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

            # Wait RESOLVE_DELAY seconds after market close before resolving
            seconds_past_close = (now - end_dt).total_seconds()
            if seconds_past_close < RESOLVE_DELAY:
                continue

            # Resolve via Binance
            outcome = await self.resolve_via_binance(end_dt)
            if outcome is None:
                # Retry on next cycle if data not available yet
                if seconds_past_close > 120:
                    # Stale — mark as loss (data never available)
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
                          f"Could not resolve (no Binance data) | {trade['question'][:45]}")
                continue

            # Determine win/loss
            won = (trade["side"] == outcome)
            if won:
                # Winner: shares pay $1.00 each
                pnl = round(trade["shares"] * 1.0 - trade["cost"], 2)
                trade["result"] = "WIN"
            else:
                # Loser: shares worth $0
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

            print(f"[{tag}] BTC:{trade['side']} ${pnl:+.2f} | "
                  f"entry=${trade['entry_price']:.2f} conf={trade['confidence']:.2f} | "
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

        # Find the closest 5M market
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
            in_window = tl <= SNIPER_WINDOW
            sniper_str = "YES" if in_window else "NO"
            market_str = f"{q} | {tl:.0f}s left"

        active_count = len(self.active)
        resolved_count = len(self.resolved)

        print(f"\n--- {now_mst.strftime('%H:%M:%S MST')} | "
              f"SNIPER 5M | "
              f"Active: {active_count} | Resolved: {resolved_count} | "
              f"{w}W/{l}L {wr:.0f}%WR | "
              f"PnL: ${self.stats['pnl']:+.2f} | "
              f"Session: ${self.session_pnl:+.2f} | "
              f"Skipped: {self.stats['skipped']} ---")
        print(f"    Market: {market_str}")
        print(f"    Sniper window: {sniper_str} (< {SNIPER_WINDOW}s)")

        if self.session_pnl <= -DAILY_LOSS_LIMIT:
            print(f"    *** DAILY LOSS LIMIT HIT: ${self.session_pnl:+.2f} ***")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        print("=" * 70)
        print("  SNIPER 5M PAPER TRADER V1.0")
        print("=" * 70)
        print(f"  Strategy: bratanbratishka reverse-engineer")
        print(f"    - Wait until < {SNIPER_WINDOW}s before 5M BTC market closes")
        print(f"    - If one side ask > ${MIN_CONFIDENCE:.2f}, bet on dominant side")
        print(f"    - If neither side > ${MIN_CONFIDENCE:.2f}, skip (unclear direction)")
        print(f"    - Resolve via Binance 5M candle (close > open = UP)")
        print(f"  Size: ${TRADE_SIZE:.2f}/trade | Max concurrent: {MAX_CONCURRENT}")
        print(f"  Scan: every {SCAN_INTERVAL}s | Log: every {LOG_INTERVAL}s")
        print(f"  Daily loss limit: ${DAILY_LOSS_LIMIT:.2f}")
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

                # Daily loss limit
                if self.session_pnl <= -DAILY_LOSS_LIMIT:
                    self.print_status([])
                    await asyncio.sleep(60)  # slow down when halted
                    continue

                # Discover 5M BTC markets
                markets = await self.discover_5m_markets()

                # Resolve expired trades
                await self.resolve_trades()

                # Check for sniper entries
                await self.check_sniper_entry(markets)

                # Periodic status log
                self.print_status(markets)

                # Save periodically (every 12 cycles = ~60s)
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

        # Final save
        self._save()
        w, l = self.stats["wins"], self.stats["losses"]
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        print(f"\n[FINAL] {w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
              f"Skipped: {self.stats['skipped']} | Session: ${self.session_pnl:+.2f}")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    lock = acquire_pid_lock("sniper_5m")
    try:
        trader = Sniper5MTrader()
        asyncio.run(trader.run())
    finally:
        release_pid_lock("sniper_5m")
