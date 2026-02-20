#!/usr/bin/env python3
"""
Volatility Split 15M Paper Trader v1.0
=======================================
Strategy: Buy BOTH UP and DOWN shares at market open, sell into FOMO at $0.65+.

The idea: in a binary 15M market, prices start near $0.50/$0.50. As the candle
develops, one side rallies while the other drops. If either side hits our sell
target, we lock in profit on that side. At resolution, the unsold winner pays
$1.00 and the unsold loser pays $0.00.

Best case: sell one side at $0.65 + winner resolves at $1.00 -> big profit.
Worst case: neither side hits target, loser resolves $0 -> net loss = cost of loser side.

Entry rules:
  1. Find BTC 15M markets within first 60 seconds of opening
  2. Fetch CLOB ask prices for both UP and DOWN tokens
  3. If combined cost <= $1.10 per share pair, enter both sides
  4. Paper-simulate buy at CLOB ask price

Exit rules:
  1. Poll CLOB every scan cycle. If bid price >= sell target -> mark sold
  2. At resolution: unsold winning shares = $1.00, unsold losing = $0.00

PnL = revenue from sells + resolution revenue - total entry cost

Usage:
  python run_volsplit_15m_paper.py
"""

import asyncio
import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from functools import partial as fn_partial

import httpx

print = fn_partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))

from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# CONFIG
# ============================================================================
VERSION = "1.0"
TRADE_SIZE = 3.00               # total per market ($1.50 per side)
SELL_TARGET = 0.65              # sell when a side's bid reaches this price
MAX_COMBINED_COST = 1.10        # skip if both sides cost > $1.10 per share pair
LIMIT_ENTRY_PRICE = 0.45        # place limit buys at $0.45 per side (vs market ask)
USE_LIMIT_ENTRY = True          # True = limit at $0.45, False = market buy at ask
SCAN_INTERVAL = 15              # seconds between cycles
MAX_CONCURRENT = 3              # max open split positions
ENTRY_WINDOW = (10, 180)        # enter 10-180 seconds after market open (wider for 15M markets)
RESOLVE_DELAY = 30              # seconds after market close before resolving

GAMMA_URL = "https://gamma-api.polymarket.com/events"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
RESULTS_FILE = Path(__file__).parent / "volsplit_15m_results.json"
LOG_EVERY_N = 4                 # print summary every N cycles

# ============================================================================
# HELPERS
# ============================================================================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ts_str() -> str:
    return now_utc().strftime("%H:%M:%S")


# ============================================================================
# VOLSPLIT PAPER TRADER
# ============================================================================

class VolSplitPaperTrader:
    """Paper trader for the volatility split strategy on 15M BTC markets."""

    def __init__(self):
        self._running = True
        self.cycle = 0
        self.session_pnl = 0.0

        # Active positions: condition_id -> position dict
        self.active = {}
        # Resolved positions
        self.resolved = []
        # Stats
        self.stats = {
            "entered": 0,
            "sells": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "pnl": 0.0,
        }
        # Track condition IDs we already attempted (avoid re-entering same market)
        self.attempted_cids = set()

    # ========================================================================
    # STATE PERSISTENCE
    # ========================================================================

    def load(self):
        """Load state from JSON results file."""
        if not RESULTS_FILE.exists():
            return
        try:
            data = json.loads(RESULTS_FILE.read_text())
            self.active = data.get("active", {})
            self.resolved = data.get("resolved", [])
            self.stats = data.get("stats", self.stats)

            # Rebuild attempted CIDs from active + resolved
            for cid in self.active:
                self.attempted_cids.add(cid)
            for trade in self.resolved:
                cid = trade.get("cid", "")
                if cid:
                    self.attempted_cids.add(cid)

            w = self.stats["wins"]
            l = self.stats["losses"]
            total = w + l + self.stats["breakeven"]
            wr = 100 * w / total if total else 0
            print(f"[LOAD] {total} resolved, {len(self.active)} active | "
                  f"PnL: ${self.stats['pnl']:+.2f} | {w}W/{l}L {wr:.0f}%WR | "
                  f"Sells: {self.stats['sells']}")
        except Exception as e:
            print(f"[LOAD] Error: {e}")

    def save(self):
        """Atomic save to JSON (write .tmp then rename)."""
        data = {
            "active": self.active,
            "resolved": self.resolved[-500:],
            "stats": self.stats,
            "last_updated": now_utc().isoformat(),
        }
        try:
            tmp = RESULTS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(RESULTS_FILE)
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    # ========================================================================
    # MARKET DISCOVERY
    # ========================================================================

    async def discover_markets(self) -> list:
        """Find active 15M BTC markets that just opened (within entry window)."""
        markets = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    GAMMA_URL,
                    params={
                        "tag_slug": "15M",
                        "active": "true",
                        "closed": "false",
                        "limit": 200,
                    },
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if r.status_code != 200:
                    return markets

                now = now_utc()

                for event in r.json():
                    title = event.get("title", "").lower()

                    # BTC only
                    if "bitcoin" not in title and "btc" not in title:
                        continue

                    for m in event.get("markets", []):
                        if m.get("closed", True):
                            continue

                        cid = m.get("conditionId", "")
                        if not cid or cid in self.attempted_cids:
                            continue

                        # Parse start/end times
                        end_date = m.get("endDate", "")
                        start_date = m.get("startDate", "")
                        if not end_date:
                            continue

                        try:
                            end_dt = datetime.fromisoformat(
                                end_date.replace("Z", "+00:00")
                            )
                            if end_dt < now:
                                continue
                        except Exception:
                            continue

                        # Compute time since market opened
                        # NOTE: Gamma startDate = creation timestamp (hours/days ago),
                        # NOT the 15-minute trading window start. Always compute
                        # open time as endDate - 15 minutes.
                        start_dt = end_dt - timedelta(minutes=15)
                        elapsed_sec = (now - start_dt).total_seconds()

                        # Only enter within the entry window
                        if elapsed_sec < ENTRY_WINDOW[0] or elapsed_sec > ENTRY_WINDOW[1]:
                            continue

                        # Extract UP and DOWN token IDs
                        outcomes = m.get("outcomes", [])
                        token_ids = m.get("clobTokenIds", [])
                        if isinstance(outcomes, str):
                            outcomes = json.loads(outcomes)
                        if isinstance(token_ids, str):
                            token_ids = json.loads(token_ids)

                        up_token = down_token = None
                        for i, o in enumerate(outcomes):
                            if i < len(token_ids):
                                if str(o).lower() == "up":
                                    up_token = token_ids[i]
                                elif str(o).lower() == "down":
                                    down_token = token_ids[i]

                        if up_token and down_token:
                            markets.append({
                                "cid": cid,
                                "title": m.get("question", event.get("title", "")),
                                "start_dt": start_dt,
                                "end_dt": end_dt,
                                "elapsed_sec": elapsed_sec,
                                "up_token": up_token,
                                "down_token": down_token,
                            })

        except Exception as e:
            if "429" not in str(e):
                print(f"[API] Discovery error: {e}")

        return markets

    # ========================================================================
    # CLOB PRICE FETCHING
    # ========================================================================

    async def get_clob_ask(self, token_id: str) -> float:
        """Get best ask price from CLOB order book for a token."""
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(
                    CLOB_BOOK_URL,
                    params={"token_id": token_id},
                )
                book = r.json()
                asks = book.get("asks", [])
                if asks:
                    return min(float(a["price"]) for a in asks)
        except Exception:
            pass
        return 0.50  # default at open

    async def get_clob_bid(self, token_id: str) -> float:
        """Get best bid price from CLOB order book for a token."""
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(
                    CLOB_BOOK_URL,
                    params={"token_id": token_id},
                )
                book = r.json()
                bids = book.get("bids", [])
                if bids:
                    return max(float(b["price"]) for b in bids)
        except Exception:
            pass
        return 0.0

    # ========================================================================
    # ENTRY: BUY BOTH SIDES
    # ========================================================================

    async def enter_split(self, market: dict):
        """Enter a volatility split position: buy both UP and DOWN at CLOB ask."""
        cid = market["cid"]
        self.attempted_cids.add(cid)

        # Check concurrent limit
        open_count = sum(
            1 for p in self.active.values() if p.get("status") in ("open", "monitoring")
        )
        if open_count >= MAX_CONCURRENT:
            return

        # Fetch both CLOB ask prices
        up_ask = await self.get_clob_ask(market["up_token"])
        down_ask = await self.get_clob_ask(market["down_token"])

        if USE_LIMIT_ENTRY:
            # Limit order mode: place at fixed price for better entry
            # If ask is already below our limit, use ask (it's better)
            up_entry = min(up_ask, LIMIT_ENTRY_PRICE)
            down_entry = min(down_ask, LIMIT_ENTRY_PRICE)
        else:
            up_entry = up_ask
            down_entry = down_ask

        combined_cost_per_pair = up_entry + down_entry

        # Skip if too expensive
        if combined_cost_per_pair > MAX_COMBINED_COST:
            print(f"  [SKIP] Combined cost ${combined_cost_per_pair:.2f}/pair > "
                  f"${MAX_COMBINED_COST:.2f} | UP@${up_entry:.2f} + DOWN@${down_entry:.2f}")
            return

        if up_entry <= 0.02 or down_entry <= 0.02:
            print(f"  [SKIP] Price too low: UP@${up_entry:.2f} DOWN@${down_entry:.2f}")
            return

        # Calculate shares: split trade size evenly per side
        side_budget = TRADE_SIZE / 2.0
        # Use the cheaper side to determine share count (equal shares both sides)
        shares = max(5, math.floor(side_budget / max(up_entry, down_entry)))

        up_cost = round(shares * up_entry, 4)
        down_cost = round(shares * down_entry, 4)
        total_cost = round(up_cost + down_cost, 4)

        # Build position record
        position = {
            "cid": cid,
            "title": market["title"][:80],
            "start_dt": market["start_dt"].isoformat(),
            "end_dt": market["end_dt"].isoformat(),
            "up_token": market["up_token"],
            "down_token": market["down_token"],
            "up_entry_price": round(up_entry, 4),
            "down_entry_price": round(down_entry, 4),
            "shares": shares,
            "up_cost": up_cost,
            "down_cost": down_cost,
            "total_cost": total_cost,
            "up_sold": False,
            "up_sell_price": None,
            "up_sell_revenue": 0.0,
            "down_sold": False,
            "down_sell_price": None,
            "down_sell_revenue": 0.0,
            "status": "open",
            "entry_time": now_utc().isoformat(),
        }

        self.active[cid] = position
        self.stats["entered"] += 1

        limit_tag = " LIMIT" if USE_LIMIT_ENTRY else ""
        print(f"[SPLIT] Entered{limit_tag} BTC 15M | UP@${up_entry:.2f} + DOWN@${down_entry:.2f} = "
              f"${combined_cost_per_pair:.2f}/pair ({shares} pairs/${total_cost:.2f}) | "
              f"{market['title'][:50]}")

    # ========================================================================
    # MONITOR: CHECK FOR SELL TARGET HITS
    # ========================================================================

    async def monitor_positions(self):
        """Poll CLOB prices for active positions; sell when target is hit."""
        for cid, pos in list(self.active.items()):
            if pos["status"] not in ("open", "monitoring"):
                continue

            # Mark as monitoring after initial entry
            if pos["status"] == "open":
                pos["status"] = "monitoring"

            # Check if market has ended (skip monitoring, wait for resolution)
            try:
                end_dt = datetime.fromisoformat(pos["end_dt"])
                if now_utc() >= end_dt:
                    continue
            except Exception:
                pass

            # Check UP side
            if not pos["up_sold"]:
                up_bid = await self.get_clob_bid(pos["up_token"])
                if up_bid >= SELL_TARGET:
                    revenue = round(pos["shares"] * up_bid, 4)
                    pos["up_sold"] = True
                    pos["up_sell_price"] = round(up_bid, 4)
                    pos["up_sell_revenue"] = revenue
                    self.stats["sells"] += 1

                    holding = "DOWN" if not pos["down_sold"] else "NONE"
                    print(f"[SELL] UP hit ${up_bid:.2f} -> sold {pos['shares']}sh "
                          f"for ${revenue:.2f} | Still holding {holding} | "
                          f"{pos['title'][:40]}")

            # Check DOWN side
            if not pos["down_sold"]:
                down_bid = await self.get_clob_bid(pos["down_token"])
                if down_bid >= SELL_TARGET:
                    revenue = round(pos["shares"] * down_bid, 4)
                    pos["down_sold"] = True
                    pos["down_sell_price"] = round(down_bid, 4)
                    pos["down_sell_revenue"] = revenue
                    self.stats["sells"] += 1

                    holding = "UP" if not pos["up_sold"] else "NONE"
                    print(f"[SELL] DOWN hit ${down_bid:.2f} -> sold {pos['shares']}sh "
                          f"for ${revenue:.2f} | Still holding {holding} | "
                          f"{pos['title'][:40]}")

    # ========================================================================
    # RESOLUTION: SETTLE AT MARKET CLOSE
    # ========================================================================

    async def resolve_positions(self):
        """Resolve positions whose market has closed."""
        now = now_utc()
        to_resolve = []

        for cid, pos in list(self.active.items()):
            if pos["status"] not in ("open", "monitoring"):
                continue

            try:
                end_dt = datetime.fromisoformat(pos["end_dt"])
                elapsed_past_close = (now - end_dt).total_seconds()
            except Exception:
                continue

            if elapsed_past_close < RESOLVE_DELAY:
                continue

            # Resolve via Binance REST kline
            outcome = await self._resolve_via_binance(pos)
            if outcome:
                to_resolve.append((cid, outcome))
            elif elapsed_past_close > 300:
                # Timeout after 5 minutes — mark as loss on both unsold sides
                to_resolve.append((cid, "TIMEOUT"))

        for cid, outcome in to_resolve:
            pos = self.active.pop(cid)
            self._settle_position(pos, outcome)

        if to_resolve:
            self.save()

    async def _resolve_via_binance(self, pos: dict) -> str:
        """Determine market outcome via Binance 15M kline (open vs close)."""
        try:
            start_dt = datetime.fromisoformat(pos["start_dt"])
            end_dt = datetime.fromisoformat(pos["end_dt"])
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    BINANCE_REST_URL,
                    params={
                        "symbol": "BTCUSDT",
                        "interval": "15m",
                        "startTime": start_ms,
                        "endTime": end_ms,
                        "limit": 5,
                    },
                )
                klines = r.json()

            if not klines:
                return None

            # Find the candle matching our market window
            best = None
            for k in klines:
                k_open_ts = k[0]
                if abs(k_open_ts - start_ms) < 60000:
                    best = k
                    break
            if not best:
                best = klines[0]

            open_price = float(best[1])
            close_price = float(best[4])

            if close_price > open_price:
                return "UP"
            elif close_price < open_price:
                return "DOWN"
            else:
                return "DOWN"  # flat = DOWN by convention

        except Exception as e:
            print(f"[RESOLVE] Binance error: {e}")
            return None

    def _settle_position(self, pos: dict, outcome: str):
        """Calculate final PnL for a resolved split position."""
        shares = pos["shares"]
        total_cost = pos["total_cost"]

        # Revenue from pre-resolution sells
        sell_revenue = pos.get("up_sell_revenue", 0.0) + pos.get("down_sell_revenue", 0.0)

        # Revenue from resolution of unsold shares
        resolution_revenue = 0.0

        if outcome == "TIMEOUT":
            # Timeout: assume worst case, unsold shares worth $0
            pass
        elif outcome == "UP":
            # UP won: unsold UP shares -> $1.00 each, unsold DOWN -> $0.00
            if not pos["up_sold"]:
                resolution_revenue += shares * 1.00
            # DOWN unsold -> $0.00 (no revenue)
        elif outcome == "DOWN":
            # DOWN won: unsold DOWN shares -> $1.00 each, unsold UP -> $0.00
            if not pos["down_sold"]:
                resolution_revenue += shares * 1.00
            # UP unsold -> $0.00 (no revenue)

        total_revenue = round(sell_revenue + resolution_revenue, 4)
        pnl = round(total_revenue - total_cost, 4)

        # Classify outcome
        if pnl > 0.01:
            self.stats["wins"] += 1
            tag = "WIN"
        elif pnl < -0.01:
            self.stats["losses"] += 1
            tag = "LOSS"
        else:
            self.stats["breakeven"] += 1
            tag = "BREAK"

        self.stats["pnl"] = round(self.stats["pnl"] + pnl, 4)
        self.session_pnl = round(self.session_pnl + pnl, 4)

        # Build resolution detail
        up_detail = f"sold@${pos['up_sell_price']:.2f}" if pos["up_sold"] else (
            "$1.00" if outcome == "UP" else "$0.00"
        )
        down_detail = f"sold@${pos['down_sell_price']:.2f}" if pos["down_sold"] else (
            "$1.00" if outcome == "DOWN" else "$0.00"
        )

        # Log resolution
        w, l = self.stats["wins"], self.stats["losses"]
        total = w + l + self.stats["breakeven"]
        wr = 100 * w / total if total else 0

        print(f"[RESOLVE] [{tag}] Market={outcome} | "
              f"UP:{up_detail} DOWN:{down_detail} | "
              f"Revenue: ${total_revenue:.2f} (sells=${sell_revenue:.2f} + "
              f"resolve=${resolution_revenue:.2f}) on ${total_cost:.2f} cost = "
              f"${pnl:+.2f} | "
              f"{w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f} | "
              f"{pos['title'][:40]}")

        # Save to resolved list
        pos["status"] = "closed"
        pos["pnl"] = pnl
        pos["total_revenue"] = total_revenue
        pos["sell_revenue"] = sell_revenue
        pos["resolution_revenue"] = resolution_revenue
        pos["market_outcome"] = outcome
        pos["resolve_time"] = now_utc().isoformat()
        pos["result"] = tag
        self.resolved.append(pos)

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        """Main event loop."""
        w, l = self.stats["wins"], self.stats["losses"]
        total = w + l + self.stats["breakeven"]
        wr = 100 * w / total if total else 0

        print(f"""
======================================================================
VOLATILITY SPLIT 15M PAPER TRADER v{VERSION}
Strategy: Buy BOTH sides at open, sell into FOMO at ${SELL_TARGET}+
Signal: None needed -- pure volatility farming
Sell target: ${SELL_TARGET} per side
Max combined cost: ${MAX_COMBINED_COST}/pair
Trade size: ${TRADE_SIZE:.2f} per market (${TRADE_SIZE / 2:.2f}/side)
Max concurrent: {MAX_CONCURRENT}
Scan interval: {SCAN_INTERVAL}s
Entry window: {ENTRY_WINDOW[0]}-{ENTRY_WINDOW[1]}s after market open
Entry mode: {'Limit @ $' + str(LIMIT_ENTRY_PRICE) if USE_LIMIT_ENTRY else 'Market (CLOB ask)'}
======================================================================
[RESUME] {w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
         {total} resolved, {len(self.active)} active
======================================================================
""")

        while self._running:
            try:
                self.cycle += 1

                # 1. Discover new 15M BTC markets within entry window
                markets = await self.discover_markets()

                # 2. Enter splits on new markets
                for market in markets:
                    await self.enter_split(market)

                # 3. Monitor active positions for sell target hits
                await self.monitor_positions()

                # 4. Resolve closed markets
                await self.resolve_positions()

                # 5. Periodic status log
                if self.cycle % LOG_EVERY_N == 0:
                    w, l = self.stats["wins"], self.stats["losses"]
                    total = w + l + self.stats["breakeven"]
                    wr = 100 * w / total if total else 0
                    open_count = sum(
                        1 for p in self.active.values()
                        if p.get("status") in ("open", "monitoring")
                    )

                    # Count active sells
                    active_sells = sum(
                        (1 if p.get("up_sold") else 0) + (1 if p.get("down_sold") else 0)
                        for p in self.active.values()
                    )

                    print(f"--- Cycle {self.cycle} | PAPER | "
                          f"Active: {open_count} (sells:{active_sells}) | "
                          f"{w}W/{l}L {wr:.0f}%WR | "
                          f"PnL: ${self.stats['pnl']:+.2f} | "
                          f"Session: ${self.session_pnl:+.2f} | "
                          f"Entered: {self.stats['entered']} ---")

                # 6. Save state
                self.save()

                await asyncio.sleep(SCAN_INTERVAL)

            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Saving state...")
                self.save()
                break
            except Exception as e:
                print(f"[ERROR] Cycle {self.cycle}: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(10)

    def stop(self):
        """Signal the main loop to stop."""
        self._running = False


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    lock = acquire_pid_lock("volsplit_15m")

    trader = VolSplitPaperTrader()
    trader.load()

    # Graceful shutdown
    def handle_signal(signum, frame):
        print(f"\n[SHUTDOWN] Signal {signum} received, saving state...")
        trader.stop()
        trader.save()
        release_pid_lock("volsplit_15m")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] KeyboardInterrupt -- saving...")
    finally:
        trader.save()
        release_pid_lock("volsplit_15m")


if __name__ == "__main__":
    main()
