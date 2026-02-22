"""
Fade-Impulse 5M Paper Trader V1.0

Inspired by polyrec repo — fades sharp price spikes near 5M close.
Complementary to the sniper (which follows momentum).

Strategy:
  - Track mid-price of BTC 5M Up/Down markets
  - If one side spikes >15 cents in <90 seconds before close, FADE it
  - Rationale: Late impulses often overshoot, revert toward fair value
  - Only enter if the spiked side is > $0.80 (clearly overextended)
    AND the fade side is < $0.35 (cheap entry = good risk/reward)

Risk management:
  - Paper only
  - $2.50/trade
  - Daily loss limit $10
  - Skip if both sides are unclear (neither >$0.70)

Usage:
  python -u run_fade_impulse_5m.py
"""

import sys
import json
import time
import asyncio
import functools
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import httpx

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))

from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# CONFIG
# ============================================================================
OBSERVATION_WINDOW = 120    # seconds: start tracking mid-prices this far before close
ENTRY_WINDOW = 60           # seconds: only enter within this window before close
MIN_SPIKE = 0.12            # minimum price spike to trigger fade ($0.12 = 12 cents)
MAX_FADE_ENTRY = 0.35       # max entry price for the FADE side (cheap entries only)
MIN_SPIKE_SIDE = 0.78       # the spiked side must be >= this (overextended)
TRADE_SIZE = 2.50           # USD per paper trade
SCAN_INTERVAL = 5           # seconds between scans
DAILY_LOSS_LIMIT = 10.0     # stop trading if daily losses exceed this
RESOLVE_DELAY = 15          # seconds after close before resolving
LOG_INTERVAL = 30           # print status every N seconds

RESULTS_FILE = Path(__file__).parent / "fade_impulse_5m_results.json"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
GAMMA_API_URL = "https://gamma-api.polymarket.com/events"


# ============================================================================
# PRICE TRACKER
# ============================================================================
@dataclass
class PriceSnapshot:
    timestamp: float
    up_ask: float
    down_ask: float


class FadeImpulse5MTrader:
    def __init__(self):
        self.active: Dict[str, dict] = {}
        self.resolved: list = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0, "skipped": 0, "faded": 0}
        self.session_pnl = 0.0
        self.attempted_cids: set = set()
        self.price_history: Dict[str, list] = {}  # {condition_id: [PriceSnapshot, ...]}
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
                      f"Faded: {self.stats['faded']}")
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
            tmp.write_text(json.dumps(data, indent=2, default=str))
            tmp.replace(RESULTS_FILE)
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    # ========================================================================
    # MARKET DISCOVERY
    # ========================================================================
    async def discover_5m_markets(self) -> list:
        """Find active BTC 5-minute Up/Down markets."""
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
    def get_token_ids(self, market: dict) -> Tuple[Optional[str], Optional[str]]:
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
                # CLOB asks are sorted DESCENDING — use min()
                return min(float(a["price"]) for a in asks)
        except Exception:
            return None

    # ========================================================================
    # SPIKE DETECTION
    # ========================================================================
    def detect_spike(self, condition_id: str) -> Optional[dict]:
        """Check if there's been a price spike worth fading.

        Returns dict with fade direction and entry price, or None.
        """
        history = self.price_history.get(condition_id, [])
        if len(history) < 3:
            return None

        latest = history[-1]
        now = latest.timestamp

        # Find the price from OBSERVATION_WINDOW seconds ago (or earliest available)
        baseline = None
        for snap in history:
            if now - snap.timestamp >= OBSERVATION_WINDOW * 0.6:  # at least 60% of window
                baseline = snap
                break
        if baseline is None:
            baseline = history[0]

        if now - baseline.timestamp < 15:  # need at least 15s of history
            return None

        # Calculate price movement
        up_delta = latest.up_ask - baseline.up_ask if latest.up_ask and baseline.up_ask else 0
        down_delta = latest.down_ask - baseline.down_ask if latest.down_ask and baseline.down_ask else 0

        # UP side spiked — fade by buying DOWN
        if up_delta >= MIN_SPIKE and latest.up_ask and latest.up_ask >= MIN_SPIKE_SIDE:
            if latest.down_ask and latest.down_ask <= MAX_FADE_ENTRY:
                return {
                    "fade_side": "DOWN",
                    "spiked_side": "UP",
                    "spike_delta": up_delta,
                    "entry_price": latest.down_ask,
                    "spiked_price": latest.up_ask,
                    "baseline_up": baseline.up_ask,
                    "elapsed_sec": now - baseline.timestamp,
                }

        # DOWN side spiked — fade by buying UP
        if down_delta >= MIN_SPIKE and latest.down_ask and latest.down_ask >= MIN_SPIKE_SIDE:
            if latest.up_ask and latest.up_ask <= MAX_FADE_ENTRY:
                return {
                    "fade_side": "UP",
                    "spiked_side": "DOWN",
                    "spike_delta": down_delta,
                    "entry_price": latest.up_ask,
                    "spiked_price": latest.down_ask,
                    "baseline_down": baseline.down_ask,
                    "elapsed_sec": now - baseline.timestamp,
                }

        return None

    # ========================================================================
    # RESOLUTION
    # ========================================================================
    async def resolve_trade(self, condition_id: str, trade: dict):
        """Resolve a fade trade using Binance candle data."""
        try:
            end_dt = datetime.fromisoformat(trade["end_dt"])
            start_ms = int((end_dt - timedelta(minutes=5)).timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    BINANCE_REST_URL,
                    params={"symbol": "BTCUSDT", "interval": "5m",
                            "startTime": start_ms, "endTime": end_ms, "limit": 1},
                )
                if r.status_code != 200:
                    return
                candles = r.json()
                if not candles:
                    return

                open_price = float(candles[0][1])
                close_price = float(candles[0][4])
                actual_winner = "UP" if close_price > open_price else "DOWN"

                fade_side = trade["fade_side"]
                entry_price = trade["entry_price"]
                shares = trade.get("shares", 0)

                if actual_winner == fade_side:
                    # WIN: fade worked, spiked side reverted
                    payout = shares * 1.0  # $1 per share
                    cost = shares * entry_price
                    pnl = payout - cost
                    self.stats["wins"] += 1
                    outcome = "WIN"
                else:
                    # LOSS: spike held, fade failed
                    pnl = -(shares * entry_price)
                    outcome = "LOSS"

                pnl = round(pnl, 4)
                self.stats["pnl"] += pnl
                self.session_pnl += pnl

                trade["outcome"] = outcome
                trade["pnl"] = pnl
                trade["actual_winner"] = actual_winner
                trade["resolved_at"] = datetime.now(timezone.utc).isoformat()
                trade["btc_open"] = open_price
                trade["btc_close"] = close_price

                self.resolved.append(trade)
                del self.active[condition_id]

                w, l = self.stats["wins"], self.stats["losses"]
                total = w + l
                wr = w / total * 100 if total > 0 else 0
                print(f"[{outcome}] Fade {fade_side} | spike={trade['spike_delta']:.2f} "
                      f"entry=${entry_price:.2f} | PnL: ${pnl:+.2f} | "
                      f"Total: {w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f}")

                self._save()
        except Exception as e:
            print(f"[RESOLVE] Error: {e}")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================
    async def run(self):
        print(f"[START] Fade-Impulse 5M Paper Trader V1.0")
        print(f"  Spike threshold: ${MIN_SPIKE:.2f} | Max fade entry: ${MAX_FADE_ENTRY:.2f}")
        print(f"  Min spiked side: ${MIN_SPIKE_SIDE:.2f} | Observation: {OBSERVATION_WINDOW}s")
        print(f"  Trade size: ${TRADE_SIZE:.2f} | Daily loss limit: ${DAILY_LOSS_LIMIT:.2f}")

        while self.running:
            try:
                now = time.time()

                # Daily loss check
                if self.session_pnl <= -DAILY_LOSS_LIMIT:
                    if now - self._last_log_time > LOG_INTERVAL:
                        print(f"[HALT] Daily loss limit hit: ${self.session_pnl:+.2f}")
                        self._last_log_time = now
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue

                # Resolve completed trades
                for cid, trade in list(self.active.items()):
                    end_dt = datetime.fromisoformat(trade["end_dt"])
                    elapsed = (datetime.now(timezone.utc) - end_dt).total_seconds()
                    if elapsed >= RESOLVE_DELAY:
                        await self.resolve_trade(cid, trade)

                # Discover markets
                markets = await self.discover_5m_markets()

                for m in markets:
                    cid = m.get("conditionId", "")
                    time_left = m.get("_time_left_sec", 9999)

                    # Skip if already traded or active
                    if cid in self.attempted_cids or cid in self.active:
                        # But still track prices for observation
                        pass

                    # Track prices within observation window
                    if time_left <= OBSERVATION_WINDOW:
                        up_tid, down_tid = self.get_token_ids(m)
                        if up_tid and down_tid:
                            up_ask = await self.get_best_ask(up_tid)
                            down_ask = await self.get_best_ask(down_tid)

                            if up_ask is not None and down_ask is not None:
                                snap = PriceSnapshot(
                                    timestamp=time.time(),
                                    up_ask=up_ask,
                                    down_ask=down_ask,
                                )
                                if cid not in self.price_history:
                                    self.price_history[cid] = []
                                self.price_history[cid].append(snap)

                                # Only look for entry within entry window
                                if time_left <= ENTRY_WINDOW and cid not in self.attempted_cids:
                                    spike = self.detect_spike(cid)
                                    if spike:
                                        # FADE the spike
                                        import math
                                        shares = math.floor(TRADE_SIZE / spike["entry_price"])
                                        if shares >= 1:
                                            self.attempted_cids.add(cid)
                                            self.stats["faded"] += 1

                                            trade = {
                                                "condition_id": cid,
                                                "question": m.get("question", ""),
                                                "fade_side": spike["fade_side"],
                                                "spiked_side": spike["spiked_side"],
                                                "spike_delta": round(spike["spike_delta"], 4),
                                                "entry_price": spike["entry_price"],
                                                "spiked_price": spike["spiked_price"],
                                                "shares": shares,
                                                "cost": round(shares * spike["entry_price"], 4),
                                                "end_dt": m["_end_dt"],
                                                "entered_at": datetime.now(timezone.utc).isoformat(),
                                                "time_left_sec": round(time_left, 1),
                                                "elapsed_sec": round(spike["elapsed_sec"], 1),
                                            }
                                            self.active[cid] = trade
                                            print(f"[FADE] {spike['fade_side']} | "
                                                  f"Spike {spike['spiked_side']} +${spike['spike_delta']:.2f} "
                                                  f"in {spike['elapsed_sec']:.0f}s | "
                                                  f"Entry ${spike['entry_price']:.2f} x{shares}sh | "
                                                  f"Time left: {time_left:.0f}s")
                                            self._save()

                    # Clean old price history (markets that already closed)
                    if time_left < -60:
                        self.price_history.pop(cid, None)

                # Periodic status log
                if now - self._last_log_time > LOG_INTERVAL:
                    w, l = self.stats["wins"], self.stats["losses"]
                    total = w + l
                    wr = w / total * 100 if total > 0 else 0
                    tracking = len(self.price_history)
                    print(f"[STATUS] {w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
                          f"Session: ${self.session_pnl:+.2f} | Active: {len(self.active)} | "
                          f"Tracking: {tracking} | Faded: {self.stats['faded']}")
                    self._last_log_time = now

                await asyncio.sleep(SCAN_INTERVAL)

            except KeyboardInterrupt:
                print("[STOP] Shutting down...")
                self.running = False
                break
            except Exception as e:
                print(f"[ERROR] {e}\n{traceback.format_exc()}")
                await asyncio.sleep(10)


# ============================================================================
# MAIN
# ============================================================================
async def main():
    pid_file = acquire_pid_lock("fade_impulse_5m")
    if not pid_file:
        print("[ERROR] Another fade-impulse instance is running. Exiting.")
        return

    trader = FadeImpulse5MTrader()
    try:
        await trader.run()
    finally:
        release_pid_lock(pid_file)
        trader._save()
        print("[EXIT] Fade-impulse trader stopped.")


if __name__ == "__main__":
    asyncio.run(main())
