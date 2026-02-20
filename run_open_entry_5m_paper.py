#!/usr/bin/env python3
"""
Open-Entry 5M BTC Paper Trader v1.1 + ML Tuner
================================================
Inspired by 7thStaircase strategy ($44.5K all-time, 2,485 predictions):
- Enter EVERY 5M BTC market at open (~30s after open)
- Quick directional signal from Binance real-time price
- Paper trade at CLOB ask price (realistic fill simulation)
- Hold to resolution, track W/L and PnL
- ML auto-tunes signal threshold + entry timing via Thompson sampling

Key difference from our other bots: ENTER AT OPEN, not near close.
At open, prices are ~$0.50 = 100% ROI if correct vs 33-67% at $0.60-0.75.
Target: 78%+ market coverage (59/76 possible per 6-hour session).
"""

import asyncio
import json
import math
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np

from pid_lock import acquire_pid_lock, release_pid_lock
from adaptive_tuner import ParameterTuner

# ── Config ──────────────────────────────────────────────────────────────────
VERSION = "1.0"
TRADE_SIZE = 3.00               # $ per paper trade
MIN_SHARES = 5                  # CLOB minimum
SCAN_INTERVAL = 15              # seconds between Gamma API polls
ENTRY_WINDOW = (15, 90)         # enter 15-90 sec after market open
RESOLVE_DELAY = 20              # seconds after close to check resolution
SIGNAL_THRESHOLD_BP = 1.0       # min 1bp move to pick a direction (else skip)
MAX_CONCURRENT = 5              # max open positions at once
ASSET = "BTC"
GAMMA_URL = "https://gamma-api.polymarket.com/events"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
RESULTS_FILE = Path(__file__).parent / "open_entry_5m_results.json"
LOG_EVERY_N = 5                 # print summary every N cycles

# ── ML Tuner Config ─────────────────────────────────────────────────────────
OPEN_ENTRY_TUNER_CONFIG = {
    "signal_threshold_bp": {
        # How many basis points of BTC move needed before we pick a direction
        # Lower = more trades (enter on tiny moves), Higher = fewer but more confident
        "bins": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
        "labels": ["0.5", "1.0", "2.0", "3.0", "5.0", "8.0"],
        "default_idx": 1,  # start at 1.0bp
        "floor_idx": 0,
        "ceil_idx": 5,
    },
    "entry_delay_sec": {
        # How many seconds after market open to enter
        # Earlier = cheaper price (~$0.50), Later = more signal data
        "bins": [15, 20, 30, 45, 60],
        "labels": ["15", "20", "30", "45", "60"],
        "default_idx": 2,  # start at 30s
        "floor_idx": 0,
        "ceil_idx": 4,
    },
}
TUNER_STATE_FILE = str(Path(__file__).parent / "open_entry_tuner_state.json")

# ── Binance Live Feed ───────────────────────────────────────────────────────
class BinanceFeed:
    """Lightweight Binance BTC price feed via WebSocket."""

    def __init__(self):
        self.current_price = 0.0
        self.last_update = 0.0
        self._task = None
        self._prices_1m = []    # (timestamp, price) tuples for last 5 min
        self._candle_opens = {} # {candle_start_ts: open_price} for 5M candles

    def start(self):
        self._task = asyncio.create_task(self._ws_loop())

    async def seed(self):
        """Seed current price from REST."""
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{BINANCE_REST_URL}?symbol=BTCUSDT&interval=1m&limit=5",
                timeout=10,
            )
            klines = r.json()
            if klines:
                self.current_price = float(klines[-1][4])
                self.last_update = time.time()
                for k in klines:
                    self._prices_1m.append((k[0] / 1000, float(k[4])))

    async def _ws_loop(self):
        import websockets
        while True:
            try:
                async with websockets.connect(BINANCE_WS_URL, ping_interval=20) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        price = float(data["p"])
                        now = time.time()
                        self.current_price = price
                        self.last_update = now
                        self._prices_1m.append((now, price))
                        # Trim to last 5 minutes
                        cutoff = now - 300
                        while self._prices_1m and self._prices_1m[0][0] < cutoff:
                            self._prices_1m.pop(0)
            except Exception as e:
                print(f"[BINANCE-WS] Error: {e}. Reconnecting in 3s...")
                await asyncio.sleep(3)

    def get_5m_candle_open(self, market_start_ts: float) -> float:
        """Get the BTC price at the start of a 5M candle."""
        # Find the first price after market_start_ts
        for ts, price in self._prices_1m:
            if ts >= market_start_ts:
                return price
        # Fallback: use REST
        return 0.0

    def get_direction_signal(self, market_start_ts: float):
        """
        Quick directional signal: compare current price vs candle open.
        Returns (side, strength_bp, detail).
        """
        now = time.time()
        if now - self.last_update > 10:
            return None, 0, "stale"

        open_price = self.get_5m_candle_open(market_start_ts)
        if open_price <= 0:
            return None, 0, "no_open"

        change_bp = (self.current_price - open_price) / open_price * 10000

        if abs(change_bp) < SIGNAL_THRESHOLD_BP:
            return None, change_bp, f"flat ({change_bp:+.1f}bp)"

        side = "UP" if change_bp > 0 else "DOWN"
        return side, change_bp, f"{side} {change_bp:+.1f}bp (open=${open_price:.2f} now=${self.current_price:.2f})"

    def is_stale(self, max_age=10):
        return time.time() - self.last_update > max_age


# ── Paper Trader ────────────────────────────────────────────────────────────
class OpenEntry5MPaper:

    def __init__(self):
        self.active = {}        # cid -> trade dict
        self.resolved = []      # list of resolved trades
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0, "skipped": 0, "entered": 0}
        self.attempted_cids = set()
        self.binance = BinanceFeed()
        self.tuner = ParameterTuner(OPEN_ENTRY_TUNER_CONFIG, TUNER_STATE_FILE)
        self.cycle = 0
        self.session_pnl = 0.0
        self._running = True

    def load(self):
        """Load prior state from JSON."""
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.resolved = data.get("resolved", [])
                self.active = data.get("active", {})
                self.stats = data.get("stats", self.stats)
                # Rebuild attempted CIDs from history
                for t in self.resolved:
                    self.attempted_cids.add(t.get("condition_id", ""))
                for cid in self.active:
                    self.attempted_cids.add(cid)
                w, l = self.stats["wins"], self.stats["losses"]
                total = w + l
                wr = 100 * w / total if total else 0
                print(f"[LOAD] {total} resolved, {len(self.active)} active | "
                      f"PnL: ${self.stats['pnl']:+.2f} | {w}W/{l}L {wr:.0f}%WR")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def save(self):
        """Atomic save to JSON."""
        data = {
            "active": self.active,
            "resolved": self.resolved[-500:],
            "stats": self.stats,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        tmp = RESULTS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(RESULTS_FILE)

    async def discover_markets(self):
        """Find 5M BTC markets that just opened (within entry window).

        Uses slug-based discovery: constructs the expected slug from the current
        5-minute timestamp slot and fetches directly. This is reliable because
        Polymarket 5M markets follow a deterministic naming pattern:
        btc-updown-5m-{unix_timestamp_of_candle_start}
        """
        now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        markets = []

        # Check current and next 5M slots
        slot_len = 300  # 5 minutes
        current_slot = (now_ts // slot_len) * slot_len
        slots_to_check = [current_slot, current_slot + slot_len]

        for slot_ts in slots_to_check:
            slug = f"btc-updown-5m-{slot_ts}"
            start_dt = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
            end_dt = start_dt + timedelta(minutes=5)
            elapsed_sec = (now - start_dt).total_seconds()

            # ML-tuned entry delay: enter when elapsed >= tuned delay
            ml_delay = self.tuner.get_active_value("entry_delay_sec")
            if not (ml_delay <= elapsed_sec <= ENTRY_WINDOW[1]):
                continue

            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(
                        f"{GAMMA_URL}?slug={slug}",
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=10,
                    )
                    events = r.json()
            except Exception as e:
                print(f"[SCAN] Gamma API error for {slug}: {e}")
                continue

            if not events:
                continue

            event = events[0]
            for m in event.get("markets", []):
                if m.get("closed", True):
                    continue

                cid = m.get("conditionId", "")
                if cid in self.attempted_cids:
                    continue

                # Extract token IDs
                outcomes = m.get("outcomes", [])
                token_ids = m.get("clobTokenIds", [])
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                if isinstance(token_ids, str):
                    token_ids = json.loads(token_ids)

                up_token = down_token = None
                for i, o in enumerate(outcomes):
                    if str(o).lower() == "up":
                        up_token = token_ids[i]
                    elif str(o).lower() == "down":
                        down_token = token_ids[i]

                if up_token and down_token:
                    markets.append({
                        "cid": cid,
                        "title": m.get("question", event.get("title", "")),
                        "end_dt": end_dt,
                        "start_dt": start_dt,
                        "elapsed_sec": elapsed_sec,
                        "up_token": up_token,
                        "down_token": down_token,
                        "slug": slug,
                    })

        return markets

    async def get_clob_price(self, token_id: str) -> float:
        """Get best ask from CLOB order book."""
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    CLOB_BOOK_URL,
                    params={"token_id": token_id},
                    timeout=5,
                )
                book = r.json()
                asks = book.get("asks", [])
                if asks:
                    return min(float(a["price"]) for a in asks)
        except:
            pass
        return 0.50  # default to $0.50 at open

    async def enter_trade(self, market: dict):
        """Enter a paper trade on a newly opened market."""
        cid = market["cid"]
        self.attempted_cids.add(cid)

        if len(self.active) >= MAX_CONCURRENT:
            self.stats["skipped"] += 1
            return

        # Get directional signal from Binance (ML-tuned threshold)
        start_ts = market["start_dt"].timestamp()
        ml_threshold = self.tuner.get_active_value("signal_threshold_bp")
        side, strength_bp, detail = self.binance.get_direction_signal(start_ts)

        # Shadow-track ALL markets (even if we skip due to threshold)
        # so the ML tuner can learn from skipped opportunities
        shadow_side = "UP" if strength_bp > 0 else "DOWN" if strength_bp < 0 else None

        if side is None or abs(strength_bp) < ml_threshold:
            # Signal below ML threshold — shadow-track and skip
            if shadow_side:
                self.tuner.record_shadow(
                    market_id=cid,
                    end_time_iso=market["end_dt"].isoformat(),
                    side=shadow_side,
                    entry_price=0.50,  # approximate open price
                    clob_dominant=abs(strength_bp),
                    extra={"elapsed_sec": market["elapsed_sec"], "strength_bp": strength_bp},
                )
            self.stats["skipped"] += 1
            return

        # Shadow-track this trade too (for tuner learning on all bins)
        self.tuner.record_shadow(
            market_id=cid,
            end_time_iso=market["end_dt"].isoformat(),
            side=side,
            entry_price=0.50,
            clob_dominant=abs(strength_bp),
            extra={"elapsed_sec": market["elapsed_sec"], "strength_bp": strength_bp},
        )

        # Get realistic entry price from CLOB
        token_id = market["up_token"] if side == "UP" else market["down_token"]
        entry_price = await self.get_clob_price(token_id)

        if entry_price <= 0.05 or entry_price >= 0.95:
            self.stats["skipped"] += 1
            return

        shares = max(MIN_SHARES, math.floor(TRADE_SIZE / entry_price))
        cost = round(shares * entry_price, 4)

        trade = {
            "condition_id": cid,
            "question": market["title"][:80],
            "side": side,
            "entry_price": round(entry_price, 4),
            "shares": shares,
            "cost": round(cost, 4),
            "trade_size": TRADE_SIZE,
            "signal_strength_bp": round(strength_bp, 2),
            "signal_detail": detail,
            "elapsed_at_entry_sec": round(market["elapsed_sec"], 1),
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "end_dt": market["end_dt"].isoformat(),
            "start_dt": market["start_dt"].isoformat(),
            "btc_price_at_entry": round(self.binance.current_price, 2),
            "status": "open",
            "pnl": 0.0,
            "result": None,
            "strategy": "open_entry_momentum",
            "paper": True,
        }

        self.active[cid] = trade
        self.stats["entered"] += 1
        self.save()

        print(f"  [ENTER] {side} @ ${entry_price:.2f} ({shares}sh/${cost:.2f}) | "
              f"Signal: {detail} | {market['title'][:55]}")

    async def resolve_trades(self):
        """Check if any active trades have resolved."""
        now = datetime.now(timezone.utc)
        to_resolve = []

        for cid, trade in list(self.active.items()):
            end_dt = datetime.fromisoformat(trade["end_dt"])
            elapsed_past_close = (now - end_dt).total_seconds()

            if elapsed_past_close < RESOLVE_DELAY:
                continue

            # Resolve via Binance REST
            outcome = await self._resolve_via_binance(trade)
            if outcome:
                to_resolve.append((cid, outcome))
            elif elapsed_past_close > 180:
                # Timeout — mark as loss
                to_resolve.append((cid, "TIMEOUT"))

        for cid, outcome in to_resolve:
            trade = self.active.pop(cid)
            won = (trade["side"] == outcome) if outcome != "TIMEOUT" else False

            if won:
                pnl = round(trade["shares"] * 1.0 - trade["cost"], 4)
                trade["result"] = "WIN"
                self.stats["wins"] += 1
            else:
                pnl = round(-trade["cost"], 4)
                trade["result"] = "LOSS"
                self.stats["losses"] += 1

            trade["pnl"] = pnl
            trade["market_outcome"] = outcome
            trade["resolve_time"] = now.isoformat()
            trade["status"] = "closed"

            self.stats["pnl"] = round(self.stats["pnl"] + pnl, 4)
            self.session_pnl = round(self.session_pnl + pnl, 4)
            self.resolved.append(trade)

            w, l = self.stats["wins"], self.stats["losses"]
            wr = 100 * w / (w + l) if (w + l) else 0
            tag = "WIN" if won else "LOSS"
            print(f"  [{tag}] {trade['side']} | PnL: ${pnl:+.2f} | "
                  f"Entry: ${trade['entry_price']:.2f} | Outcome: {outcome} | "
                  f"Running: {w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f}")

            # Resolve shadow entries in tuner
            self.tuner.resolve_shadows({cid: outcome})

        if to_resolve:
            self.save()

    async def _resolve_shadow_entries(self):
        """Resolve shadow-tracked markets via Binance for ML tuner learning."""
        shadows = self.tuner.state.get("shadow", [])
        if not shadows:
            return

        now = datetime.now(timezone.utc)
        resolutions = {}

        for entry in shadows:
            end_time = entry.get("end_time", "")
            mid = entry.get("market_id", "")
            if not end_time or not mid:
                continue

            try:
                end_dt = datetime.fromisoformat(end_time)
            except:
                continue

            # Only resolve if market is past close + delay
            if (now - end_dt).total_seconds() < RESOLVE_DELAY:
                continue

            # Already resolved?
            if mid in resolutions:
                continue

            # Resolve via Binance
            start_dt = end_dt - timedelta(minutes=5)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(
                        BINANCE_REST_URL,
                        params={
                            "symbol": "BTCUSDT",
                            "interval": "5m",
                            "startTime": start_ms,
                            "endTime": end_ms,
                            "limit": 5,
                        },
                        timeout=10,
                    )
                    klines = r.json()

                if klines:
                    best = klines[0]
                    for k in klines:
                        if abs(k[0] - start_ms) < 60000:
                            best = k
                            break
                    o, c = float(best[1]), float(best[4])
                    resolutions[mid] = "UP" if c > o else "DOWN"
            except:
                pass

        if resolutions:
            n = self.tuner.resolve_shadows(resolutions)
            if n > 0:
                total = self.tuner.state.get("total_resolved", 0)
                if total % 10 == 0:
                    print(f"[ML TUNER] {total} shadows resolved. Report:")
                    print(self.tuner.get_report())

    async def _resolve_via_binance(self, trade: dict) -> str:
        """Resolve a 5M market using Binance klines."""
        try:
            start_dt = datetime.fromisoformat(trade["start_dt"])
            end_dt = datetime.fromisoformat(trade["end_dt"])
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient() as client:
                r = await client.get(
                    BINANCE_REST_URL,
                    params={
                        "symbol": "BTCUSDT",
                        "interval": "5m",
                        "startTime": start_ms,
                        "endTime": end_ms,
                        "limit": 5,
                    },
                    timeout=10,
                )
                klines = r.json()

            if not klines:
                return None

            # Find the candle that best matches our window
            best = None
            for k in klines:
                k_open_ts = k[0]
                if abs(k_open_ts - start_ms) < 60000:  # within 1 min
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
            print(f"  [RESOLVE] Binance error: {e}")
            return None

    async def run(self):
        """Main event loop."""
        print(f"[PID] {Path('open_entry_5m').name} running as PID {__import__('os').getpid()}")

        # Seed Binance feed
        await self.binance.seed()
        self.binance.start()
        print(f"[BINANCE] WebSocket started | BTC=${self.binance.current_price:,.2f}")

        # Wait for WS to get fresh data
        await asyncio.sleep(2)

        w, l = self.stats["wins"], self.stats["losses"]
        total = w + l
        wr = 100 * w / total if total else 0
        print(f"[RESUME] {w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
              f"{total} resolved, {len(self.active)} active")

        ml_thresh = self.tuner.get_active_value("signal_threshold_bp")
        ml_delay = self.tuner.get_active_value("entry_delay_sec")
        print(f"""
======================================================================
OPEN ENTRY 5M PAPER TRADER v{VERSION} + ML TUNER
Strategy: Enter EVERY 5M BTC market at open (~30s after open)
Signal: Binance real-time momentum (price vs candle open)
ML threshold: {ml_thresh}bp (auto-tuned via Thompson sampling)
ML entry delay: {ml_delay}s after open (auto-tuned)
Entry window: {ml_delay}-{ENTRY_WINDOW[1]}s after market open
Trade size: ${TRADE_SIZE:.2f} paper
Max concurrent: {MAX_CONCURRENT}
Scan interval: {SCAN_INTERVAL}s
Inspired by: 7thStaircase (+$44.5K, 78% market coverage)
======================================================================
""")

        while self._running:
            self.cycle += 1

            # 1. Discover new markets
            markets = await self.discover_markets()

            # 2. Enter new trades
            for market in markets:
                await self.enter_trade(market)

            # 3. Resolve completed trades
            await self.resolve_trades()

            # 4. Resolve shadow entries for ML tuner (every 4 cycles = ~60s)
            if self.cycle % 4 == 0:
                await self._resolve_shadow_entries()

            # 5. Status update
            if self.cycle % LOG_EVERY_N == 0:
                w, l = self.stats["wins"], self.stats["losses"]
                total = w + l
                wr = 100 * w / total if total else 0
                stale = " STALE!" if self.binance.is_stale() else ""
                btc_str = f"BTC=${self.binance.current_price:,.2f}" if self.binance.current_price else "BTC=?"
                ml_thresh = self.tuner.get_active_value("signal_threshold_bp")
                ml_delay = self.tuner.get_active_value("entry_delay_sec")
                print(f"--- Cycle {self.cycle} | PAPER | Active: {len(self.active)} | "
                      f"{w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
                      f"Session: ${self.session_pnl:+.2f} | "
                      f"Entered: {self.stats['entered']} Skip: {self.stats['skipped']} | "
                      f"ML: thresh={ml_thresh}bp delay={ml_delay}s | "
                      f"{btc_str}{stale} ---")

            await asyncio.sleep(SCAN_INTERVAL)

    def stop(self):
        self._running = False


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    lock = acquire_pid_lock("open_entry_5m")

    trader = OpenEntry5MPaper()
    trader.load()

    # Graceful shutdown
    def handle_signal(signum, frame):
        print("\n[SHUTDOWN] Saving state...")
        trader.stop()
        trader.save()
        release_pid_lock("open_entry_5m")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] KeyboardInterrupt — saving...")
    finally:
        trader.save()
        release_pid_lock("open_entry_5m")


if __name__ == "__main__":
    main()
