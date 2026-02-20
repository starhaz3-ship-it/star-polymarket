"""
Merge Arbitrage Paper Trader V1.0

Inspired by 7thStaircase's $44,510 profit strategy:
  - Buy BOTH Up and Down shares on BTC 5M/15M markets
  - When combined ask < $1.00 (after fees), merge pairs for guaranteed $1.00 payout
  - Leftover imbalanced shares are free directional exposure

Paper-only: simulates fills by walking the CLOB orderbook.

Usage:
  python -u run_merge_arb_paper.py
"""

import sys
import json
import time
import math
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
SCAN_INTERVAL = 10              # seconds between scans
TRADE_SIZE = 25.0               # $ budget per side (total ~$50 per arb)
MAX_COMBINED_COST = 0.97        # max combined ask (before fees) to enter
FEE_RATE = 0.02                 # 2% Polymarket taker fee
MIN_SHARES = 5                  # minimum shares available on each side
DAILY_LOSS_LIMIT = 50.0         # stop if session PnL < -$LIMIT
LOG_INTERVAL = 30               # print status every N seconds
RESOLVE_DELAY = 15              # seconds after close before resolving
MIN_TIME_LEFT = 30              # don't enter markets closing in < 30s

RESULTS_FILE = Path(__file__).parent / "merge_arb_paper_results.json"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
GAMMA_API_URL = "https://gamma-api.polymarket.com/events"
TAG_SLUGS = ["5M", "15M"]      # scan both timeframes


# ============================================================================
# MERGE ARB PAPER TRADER
# ============================================================================
class MergeArbPaperTrader:
    def __init__(self):
        self.active: Dict[str, dict] = {}
        self.resolved: list = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0, "total_merge_profit": 0.0,
                      "total_directional_pnl": 0.0, "opportunities_seen": 0}
        self.session_pnl = 0.0
        self.attempted_cids: set = set()
        self.running = True
        self._last_log_time = 0
        self._load()

    def _load(self):
        """Load saved state."""
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.active = data.get("active", {})
                self.resolved = data.get("resolved", [])
                self.stats = data.get("stats", self.stats)
                self.attempted_cids = set(data.get("attempted_cids", []))
                print(f"[LOAD] Restored {len(self.active)} active, {len(self.resolved)} resolved, "
                      f"PnL=${self.stats['pnl']:+.2f}")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        """Atomic save to JSON."""
        data = {
            "active": self.active,
            "resolved": self.resolved[-500:],
            "stats": self.stats,
            "attempted_cids": list(self.attempted_cids)[-2000:],
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

    async def discover_markets(self) -> list:
        """Find active BTC 5M and 15M Up/Down markets."""
        all_markets = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                for tag in TAG_SLUGS:
                    r = await client.get(
                        GAMMA_API_URL,
                        params={"tag_slug": tag, "active": "true",
                                "closed": "false", "limit": 200},
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    if r.status_code != 200:
                        print(f"[API] Gamma error for {tag}: {r.status_code}")
                        continue

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
                                time_left = (end_dt - now).total_seconds()
                                m["_time_left_sec"] = time_left
                                m["_end_dt"] = end_dt.isoformat()
                                m["_end_dt_parsed"] = end_dt
                                m["_tag"] = tag
                            except Exception:
                                continue
                            if not m.get("question"):
                                m["question"] = event.get("title", "")
                            all_markets.append(m)
        except Exception as e:
            print(f"[API] Discovery error: {e}")
        return all_markets

    # ========================================================================
    # ORDERBOOK
    # ========================================================================

    def get_token_ids(self, market: dict) -> Tuple[Optional[str], Optional[str]]:
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

    async def walk_book(self, token_id: str, budget: float) -> Tuple[float, int, float]:
        """Walk the ask side of the orderbook up to $budget.

        Returns (avg_fill_price, shares_filled, total_cost).
        CLOB asks are sorted DESCENDING — we sort ascending to walk cheapest first.
        """
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(CLOB_BOOK_URL, params={"token_id": token_id})
                if r.status_code != 200:
                    return 1.0, 0, 0.0
                book = r.json()
                asks = book.get("asks", [])
                if not asks:
                    return 1.0, 0, 0.0

                # Sort ascending by price (cheapest first)
                asks_sorted = sorted(asks, key=lambda a: float(a["price"]))

                total_cost = 0.0
                total_shares = 0
                for ask in asks_sorted:
                    price = float(ask["price"])
                    size = float(ask["size"])
                    remaining_budget = budget - total_cost
                    if remaining_budget <= 0:
                        break
                    can_buy = min(size, remaining_budget / price)
                    shares = math.floor(can_buy)
                    if shares < 1:
                        break
                    total_cost += shares * price
                    total_shares += shares

                if total_shares == 0:
                    return 1.0, 0, 0.0
                avg_fill = total_cost / total_shares
                return avg_fill, total_shares, total_cost
        except Exception:
            return 1.0, 0, 0.0

    # ========================================================================
    # OPPORTUNITY SCANNING
    # ========================================================================

    async def scan_opportunities(self, markets: list):
        """Scan markets for merge arb opportunities."""
        for market in markets:
            cid = market.get("conditionId", "")
            if not cid or cid in self.active or cid in self.attempted_cids:
                continue

            time_left = market.get("_time_left_sec", 0)
            if time_left < MIN_TIME_LEFT:
                continue

            up_tid, down_tid = self.get_token_ids(market)
            if not up_tid or not down_tid:
                continue

            # Fetch both orderbooks in parallel
            up_result, down_result = await asyncio.gather(
                self.walk_book(up_tid, TRADE_SIZE),
                self.walk_book(down_tid, TRADE_SIZE),
            )
            up_avg, up_shares, up_cost = up_result
            down_avg, down_shares, down_cost = down_result

            if up_shares < MIN_SHARES or down_shares < MIN_SHARES:
                continue

            # Calculate combined cost and fees
            combined_ask = up_avg + down_avg
            total_cost = up_cost + down_cost
            fee_cost = up_cost * FEE_RATE + down_cost * FEE_RATE
            merge_pairs = min(up_shares, down_shares)
            merge_payout = merge_pairs * 1.00
            # Cost of the paired shares only
            paired_up_cost = merge_pairs * up_avg
            paired_down_cost = merge_pairs * down_avg
            paired_total = paired_up_cost + paired_down_cost
            paired_fees = paired_total * FEE_RATE
            merge_profit = merge_payout - paired_total - paired_fees

            # Net cost per unit (including fees)
            net_combined = combined_ask * (1 + FEE_RATE)

            self.stats["opportunities_seen"] += 1

            # Log every opportunity for analysis
            tag = market.get("_tag", "?")
            q = market.get("question", "")[:50]

            if net_combined >= 1.00:
                # Not profitable — skip silently (log occasionally)
                if self.stats["opportunities_seen"] % 20 == 0:
                    print(f"  [SCAN] {tag} {q} | Up@{up_avg:.3f}+Down@{down_avg:.3f}="
                          f"{combined_ask:.4f} (net {net_combined:.4f}) | NO EDGE")
                continue

            if combined_ask > MAX_COMBINED_COST:
                continue

            # ENTRY — merge arb opportunity found
            self.attempted_cids.add(cid)

            # Leftover directional shares
            leftover_shares = abs(up_shares - down_shares)
            if up_shares > down_shares:
                leftover_side = "UP"
                leftover_cost = leftover_shares * up_avg
            elif down_shares > up_shares:
                leftover_side = "DOWN"
                leftover_cost = leftover_shares * down_avg
            else:
                leftover_side = "NONE"
                leftover_cost = 0.0

            trade = {
                "condition_id": cid,
                "question": market.get("question", ""),
                "tag": tag,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "end_dt": market.get("_end_dt", ""),
                "time_remaining_sec": round(time_left),
                "up_tid": up_tid, "down_tid": down_tid,
                "up_avg_fill": round(up_avg, 4),
                "down_avg_fill": round(down_avg, 4),
                "combined_ask": round(combined_ask, 4),
                "net_combined": round(net_combined, 4),
                "up_shares": up_shares,
                "down_shares": down_shares,
                "up_cost": round(up_cost, 2),
                "down_cost": round(down_cost, 2),
                "total_cost": round(total_cost, 2),
                "fee_cost": round(fee_cost, 2),
                "merge_pairs": merge_pairs,
                "merge_payout": round(merge_payout, 2),
                "merge_profit": round(merge_profit, 2),
                "leftover_side": leftover_side,
                "leftover_shares": leftover_shares,
                "leftover_cost": round(leftover_cost, 2),
                "status": "open",
                "directional_pnl": 0.0,
                "total_pnl": 0.0,
            }

            self.active[cid] = trade
            edge_pct = (1.0 - net_combined) * 100

            print(f"[ENTER] {tag} {trade['question'][:50]}")
            print(f"  Up@${up_avg:.3f} x{up_shares} + Down@${down_avg:.3f} x{down_shares} = "
                  f"${combined_ask:.4f} (net ${net_combined:.4f})")
            print(f"  Merge {merge_pairs} pairs -> ${merge_payout:.2f} | "
                  f"profit=${merge_profit:+.2f} ({edge_pct:.1f}% edge) | "
                  f"leftover: {leftover_shares} {leftover_side}")

    # ========================================================================
    # RESOLUTION
    # ========================================================================

    async def resolve_via_binance(self, end_dt: datetime, tag: str) -> Optional[str]:
        """Determine if BTC went UP or DOWN during the market window."""
        try:
            minutes = 5 if tag == "5M" else 15
            start_dt = end_dt - timedelta(minutes=minutes)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient(timeout=15) as client:
                interval = "5m" if tag == "5M" else "15m"
                r = await client.get(
                    BINANCE_REST_URL,
                    params={
                        "symbol": "BTCUSDT",
                        "interval": interval,
                        "startTime": start_ms,
                        "endTime": end_ms,
                        "limit": 5,
                    },
                )
                r.raise_for_status()
                klines = r.json()

                if not klines:
                    # Fallback to 1m candles
                    r2 = await client.get(
                        BINANCE_REST_URL,
                        params={"symbol": "BTCUSDT", "interval": "1m",
                                "startTime": start_ms, "endTime": end_ms, "limit": 20},
                    )
                    r2.raise_for_status()
                    klines_1m = r2.json()
                    if not klines_1m:
                        return None
                    open_price = float(klines_1m[0][1])
                    close_price = float(klines_1m[-1][4])
                else:
                    best_candle = min(klines, key=lambda k: abs(int(k[0]) - start_ms))
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

    async def resolve_expired(self):
        """Resolve trades whose windows have closed."""
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

            seconds_past = (now - end_dt).total_seconds()
            if seconds_past < RESOLVE_DELAY:
                continue

            tag = trade.get("tag", "5M")
            outcome = await self.resolve_via_binance(end_dt, tag)

            if outcome is None:
                if seconds_past > 120:
                    # Stale — count merge profit only, leftover as loss
                    trade["status"] = "closed"
                    trade["result"] = "unknown"
                    trade["total_pnl"] = round(trade["merge_profit"] - trade["leftover_cost"], 2)
                    self._record_result(trade, to_remove, cid)
                continue

            # Merge profit is guaranteed (already calculated at entry)
            merge_profit = trade["merge_profit"]

            # Leftover directional exposure
            leftover_pnl = 0.0
            if trade["leftover_shares"] > 0:
                if trade["leftover_side"] == outcome:
                    # Leftover wins — shares pay $1.00
                    leftover_pnl = trade["leftover_shares"] * 1.00 - trade["leftover_cost"]
                else:
                    # Leftover loses — shares worth $0
                    leftover_pnl = -trade["leftover_cost"]
                # Apply fee on leftover cost too (already in fee_cost, but leftover has
                # additional fee on the winning payout side)
                if trade["leftover_side"] == outcome:
                    leftover_pnl -= trade["leftover_shares"] * 1.00 * FEE_RATE

            trade["directional_pnl"] = round(leftover_pnl, 2)
            trade["total_pnl"] = round(merge_profit + leftover_pnl, 2)
            trade["market_outcome"] = outcome
            trade["status"] = "closed"
            trade["result"] = "WIN" if trade["total_pnl"] > 0 else "LOSS"
            trade["resolve_time"] = now.isoformat()

            self._record_result(trade, to_remove, cid)

            w, l = self.stats["wins"], self.stats["losses"]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            tag_str = f"{'WIN' if trade['total_pnl'] > 0 else 'LOSS'}"

            print(f"[{tag_str}] {trade['tag']} {trade['question'][:45]}")
            print(f"  merge=${merge_profit:+.2f} + leftover({trade['leftover_side']})=${leftover_pnl:+.2f}"
                  f" = ${trade['total_pnl']:+.2f} | outcome={outcome}")
            print(f"  {w}W/{l}L {wr:.0f}%WR | session=${self.session_pnl:+.2f} | "
                  f"all-time=${self.stats['pnl']:+.2f}")

        for cid in to_remove:
            del self.active[cid]
        if to_remove:
            self._save()

    def _record_result(self, trade, to_remove, cid):
        """Record a resolved trade in stats."""
        pnl = trade["total_pnl"]
        if pnl > 0:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1
        self.stats["pnl"] += pnl
        self.stats["total_merge_profit"] += trade.get("merge_profit", 0)
        self.stats["total_directional_pnl"] += trade.get("directional_pnl", 0)
        self.session_pnl += pnl
        self.resolved.append(trade)
        to_remove.append(cid)

    # ========================================================================
    # STATUS DISPLAY
    # ========================================================================

    def print_status(self, markets: list):
        """Print dashboard every LOG_INTERVAL seconds."""
        now = time.time()
        if now - self._last_log_time < LOG_INTERVAL:
            return
        self._last_log_time = now

        w, l = self.stats["wins"], self.stats["losses"]
        total = w + l
        wr = w / total * 100 if total > 0 else 0

        now_dt = datetime.now(timezone.utc)
        # MST = UTC-7
        mst = now_dt - timedelta(hours=7)
        mst_str = mst.strftime("%Y-%m-%d %H:%M:%S MST")

        m5 = sum(1 for m in markets if m.get("_tag") == "5M")
        m15 = sum(1 for m in markets if m.get("_tag") == "15M")

        print(f"\n[MERGE ARB] {mst_str}")
        print(f"  Markets: {len(markets)} active ({m5}x 5M, {m15}x 15M) | "
              f"Opps seen: {self.stats['opportunities_seen']}")
        print(f"  Open: {len(self.active)} | Resolved: {total} ({w}W/{l}L {wr:.0f}%WR)")
        print(f"  Session: ${self.session_pnl:+.2f} | All-time: ${self.stats['pnl']:+.2f}")
        print(f"  Merge profit: ${self.stats['total_merge_profit']:+.2f} | "
              f"Directional: ${self.stats['total_directional_pnl']:+.2f}")

        for cid, t in self.active.items():
            q = t.get("question", "")[:40]
            print(f"  | {t['tag']} {q} | Up@{t['up_avg_fill']:.3f}+Down@{t['down_avg_fill']:.3f}"
                  f"={t['combined_ask']:.4f} | merge=${t['merge_profit']:+.2f} | "
                  f"left: {t['leftover_shares']} {t['leftover_side']}")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        """Main scanning loop."""
        print("=" * 70)
        print("  MERGE ARBITRAGE PAPER TRADER V1.0")
        print("  Strategy: Buy Up + Down when combined < $0.97, merge for $1.00")
        print(f"  Trade size: ${TRADE_SIZE}/side | Fee: {FEE_RATE*100:.0f}% | "
              f"Threshold: ${MAX_COMBINED_COST}")
        print(f"  Scanning: {', '.join(TAG_SLUGS)} BTC markets")
        print("=" * 70)

        cycle = 0
        while self.running:
            try:
                cycle += 1

                if self.session_pnl <= -DAILY_LOSS_LIMIT:
                    if cycle % 60 == 0:
                        print(f"[HALT] Session loss limit hit: ${self.session_pnl:+.2f}")
                    await asyncio.sleep(60)
                    continue

                markets = await self.discover_markets()
                await self.resolve_expired()
                await self.scan_opportunities(markets)
                self.print_status(markets)

                if cycle % 6 == 0:
                    self._save()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[ERROR] Cycle {cycle}: {e}")
                traceback.print_exc()

            await asyncio.sleep(SCAN_INTERVAL)

        self._save()
        print(f"\n[EXIT] Final: {self.stats['wins']}W/{self.stats['losses']}L | "
              f"PnL=${self.stats['pnl']:+.2f}")


# ============================================================================
# ENTRY POINT
# ============================================================================
def main():
    lock = acquire_pid_lock("merge_arb_paper")
    try:
        trader = MergeArbPaperTrader()
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user")
    finally:
        release_pid_lock("merge_arb_paper")


if __name__ == "__main__":
    main()
