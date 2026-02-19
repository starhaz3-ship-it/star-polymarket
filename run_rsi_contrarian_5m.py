"""
RSI Contrarian 5M Paper Trader V1.0

Contrarian/mean-reversion strategy on Polymarket BTC 5-minute Up/Down markets.
Uses RSI-14 computed from Binance 5-minute candles to detect oversold/overbought
conditions, then bets on a reversal at cheap entry prices.

Strategy:
  - RSI < 30 (oversold) + high volume  -> bet UP (expect bounce)
  - RSI > 70 (overbought) + high volume -> bet DOWN (expect pullback)
  - Volume filter: previous completed 5m candle > 1.5x avg of prior 10
  - Entry price: $0.25-$0.55 (cheap side only, for 2x+ payout on win)
  - Entry window: 1-4 minutes before 5M market close (late entry)
  - Resolution: Binance 5M candle (close > open = UP wins, else DOWN)

NOTE: Contrarian was ~8% WR in 15M backtests. Testing on 5M with strict
      volume filter + cheap entry prices to see if edge exists.

Usage:
  python -u run_rsi_contrarian_5m.py
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
TRADE_SIZE = 2.50           # USD per trade (paper)
MAX_CONCURRENT = 1          # one trade at a time (sequential 5M markets)
MIN_ENTRY = 0.25            # minimum entry price (cheap = high payout)
MAX_ENTRY = 0.55            # maximum entry price (above this, payout too low)
SCAN_INTERVAL = 5           # seconds between scans
LOG_INTERVAL = 30           # print status every N seconds
DAILY_LOSS_LIMIT = 15.0     # stop trading if session losses exceed this
RESOLVE_DELAY = 15          # seconds after market close before resolving
RSI_PERIOD = 14             # RSI lookback
RSI_OVERSOLD = 30           # RSI below this = oversold -> bet UP
RSI_OVERBOUGHT = 70         # RSI above this = overbought -> bet DOWN
VOLUME_MULTIPLIER = 1.5     # last completed candle volume must exceed avg * this
ENTRY_WINDOW_MIN = 60       # earliest entry: 4 min before close (240s)
ENTRY_WINDOW_MAX = 240      # latest entry: 1 min before close (60s)
RSI_BUCKET_SIZE = 5         # RSI bucket width for stats tracking

RESULTS_FILE = Path(__file__).parent / "rsi_contrarian_5m_results.json"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
GAMMA_API_URL = "https://gamma-api.polymarket.com/events"


# ============================================================================
# RSI CALCULATION
# ============================================================================
def compute_rsi(closes: list, period: int = 14) -> float:
    """Compute RSI-14 from a list of close prices.

    Uses the standard SMA-based RSI calculation (not exponential smoothing).
    Returns 50.0 if insufficient data.
    """
    if len(closes) < period + 1:
        return 50.0

    gains = []
    losses = []
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
# RSI CONTRARIAN 5M PAPER TRADER
# ============================================================================
class RSIContrarian5MTrader:
    def __init__(self):
        self.active: Dict[str, dict] = {}       # {condition_id: trade_dict}
        self.resolved: list = []
        self.stats = {
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
            "skipped": 0,
            "by_rsi_bucket": {},
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        self.session_pnl = 0.0
        self.attempted_cids: set = set()
        self.running = True
        self._last_log_time = 0.0
        self._last_rsi = 50.0
        self._last_vol_ok = False
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
                saved_stats = data.get("stats", {})
                # Merge saved stats into defaults
                for k, v in saved_stats.items():
                    self.stats[k] = v
                # Rebuild attempted CIDs
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
                    params={
                        "tag_slug": "5M",
                        "active": "true",
                        "closed": "false",
                        "limit": 200,
                    },
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
    # BINANCE DATA
    # ========================================================================

    async def fetch_5m_candles(self, limit: int = 20) -> list:
        """Fetch 5-minute BTC/USDT candles from Binance."""
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    r = await client.get(
                        BINANCE_REST_URL,
                        params={
                            "symbol": "BTCUSDT",
                            "interval": "5m",
                            "limit": limit,
                        },
                    )
                    r.raise_for_status()
                    candles = []
                    for k in r.json():
                        candles.append({
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                            "time": int(k[0]),
                        })
                    return candles
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                print(f"[BINANCE] Candle fetch error: {e}")
                return []

    # ========================================================================
    # SIGNAL GENERATION
    # ========================================================================

    async def get_signal(self):
        """Compute RSI signal from Binance 5m candles.

        Returns:
            (signal, rsi, volume_ok)
            signal: "UP", "DOWN", or None
            rsi: current RSI value
            volume_ok: whether volume filter passed
        """
        candles = await self.fetch_5m_candles(20)
        if len(candles) < 15:
            return None, 50.0, False

        closes = [c["close"] for c in candles]
        rsi = compute_rsi(closes, RSI_PERIOD)

        # Volume filter: last completed candle volume > 1.5x avg of previous 10
        # candles[-1] is still forming, candles[-2] is last completed
        volumes = [c["volume"] for c in candles]
        high_volume = False
        if len(volumes) >= 12:
            last_vol = volumes[-2]      # last completed candle
            avg_vol = sum(volumes[-12:-2]) / 10
            high_volume = avg_vol > 0 and last_vol > avg_vol * VOLUME_MULTIPLIER

        # Cache for status display
        self._last_rsi = rsi
        self._last_vol_ok = high_volume

        if rsi < RSI_OVERSOLD and high_volume:
            return "UP", rsi, True      # Oversold + volume -> expect bounce
        elif rsi > RSI_OVERBOUGHT and high_volume:
            return "DOWN", rsi, True    # Overbought + volume -> expect pullback
        else:
            return None, rsi, high_volume

    # ========================================================================
    # PRICE EXTRACTION
    # ========================================================================

    def get_prices(self, market: dict) -> tuple:
        """Extract UP/DOWN outcome prices from market data.

        Returns (up_price, down_price) or (None, None).
        """
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
    # BINANCE RESOLUTION
    # ========================================================================

    async def resolve_via_binance(self, end_dt: datetime) -> Optional[str]:
        """Determine if BTC went UP or DOWN during the 5-minute window.

        Fetches the Binance 5m candle covering the market window.
        close > open = UP, close < open = DOWN, tie = DOWN.

        Returns 'UP', 'DOWN', or None if data unavailable.
        """
        try:
            start_dt = end_dt - timedelta(minutes=5)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient(timeout=15) as client:
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
                    open_price = float(klines_1m[0][1])
                    close_price = float(klines_1m[-1][4])
                else:
                    # Find the candle whose open time is closest to start_dt
                    best_candle = min(klines, key=lambda k: abs(int(k[0]) - start_ms))
                    open_price = float(best_candle[1])
                    close_price = float(best_candle[4])

                if close_price > open_price:
                    return "UP"
                elif close_price < open_price:
                    return "DOWN"
                else:
                    return "DOWN"  # Tie counts as DOWN

        except Exception as e:
            print(f"[BINANCE] Resolution error: {e}")
            return None

    # ========================================================================
    # ENTRY LOGIC
    # ========================================================================

    async def check_entry(self, markets: list):
        """Check for RSI contrarian entries on 5M BTC markets."""
        if len(self.active) >= MAX_CONCURRENT:
            return

        # Daily loss check
        if self.session_pnl <= -DAILY_LOSS_LIMIT:
            return

        signal, rsi, vol_ok = await self.get_signal()
        if signal is None:
            return

        for market in markets:
            if len(self.active) >= MAX_CONCURRENT:
                break

            cid = market.get("conditionId", "")
            if not cid or cid in self.attempted_cids:
                continue

            time_left = market.get("_time_left_sec", 9999)
            # Entry window: 1-4 minutes before close (60-240 seconds)
            if time_left > ENTRY_WINDOW_MAX or time_left <= ENTRY_WINDOW_MIN:
                continue

            up_price, down_price = self.get_prices(market)
            if up_price is None or down_price is None:
                continue

            # Contrarian logic:
            #   RSI < 30 -> oversold, price falling, UP side should be cheap
            #   RSI > 70 -> overbought, price rising, DOWN side should be cheap
            entry_price = None
            if signal == "UP" and MIN_ENTRY <= up_price <= MAX_ENTRY:
                entry_price = up_price
            elif signal == "DOWN" and MIN_ENTRY <= down_price <= MAX_ENTRY:
                entry_price = down_price
            else:
                continue  # Price not in the cheap sweet spot

            # Calculate position
            shares = TRADE_SIZE / entry_price
            cost = round(shares * entry_price, 2)

            # Mark as attempted
            self.attempted_cids.add(cid)

            # RSI bucket for tracking (e.g., "20-25", "25-30", "70-75", "75-80")
            bucket_start = int(rsi // RSI_BUCKET_SIZE) * RSI_BUCKET_SIZE
            rsi_bucket = f"{bucket_start}-{bucket_start + RSI_BUCKET_SIZE}"

            # Build trade record
            now = datetime.now(timezone.utc)
            question = market.get("question", "?")
            trade = {
                "condition_id": cid,
                "question": question,
                "side": signal,
                "entry_price": round(entry_price, 4),
                "rsi": round(rsi, 1),
                "rsi_bucket": rsi_bucket,
                "volume_ok": vol_ok,
                "shares": round(shares, 4),
                "cost": cost,
                "time_remaining_sec": round(time_left, 1),
                "entry_time": now.isoformat(),
                "end_dt": market.get("_end_dt", ""),
                "status": "open",
                "pnl": 0.0,
                "result": None,
            }

            self.active[cid] = trade
            self._save()

            payout = round(shares * 1.0 - cost, 2)
            print(f"\n[ENTRY] RSI={rsi:.1f} -> BTC {signal} @ ${entry_price:.2f} | "
                  f"${cost:.2f} ({shares:.1f}sh) | potential +${payout:.2f} | "
                  f"{time_left:.0f}s left | bucket={rsi_bucket} | "
                  f"{question[:50]}")
            break  # Only one entry per scan cycle

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

            # Wait RESOLVE_DELAY seconds after market close
            seconds_past_close = (now - end_dt).total_seconds()
            if seconds_past_close < RESOLVE_DELAY:
                continue

            # Resolve via Binance
            outcome = await self.resolve_via_binance(end_dt)
            if outcome is None:
                # Retry on next cycle; mark as loss after 2 minutes
                if seconds_past_close > 120:
                    trade["status"] = "closed"
                    trade["result"] = "unknown"
                    trade["pnl"] = round(-trade["cost"], 2)
                    trade["resolve_time"] = now.isoformat()
                    self._update_stats(trade, won=False)
                    self.resolved.append(trade)
                    to_remove.append(cid)
                    print(f"[LOSS] BTC:{trade['side']} ${trade['pnl']:+.2f} | "
                          f"No Binance data (aged out) | {trade['question'][:45]}")
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

            self._update_stats(trade, won)
            self.resolved.append(trade)
            to_remove.append(cid)

            w, l = self.stats["wins"], self.stats["losses"]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            tag = "WIN" if won else "LOSS"

            print(f"[{tag}] BTC:{trade['side']} ${pnl:+.2f} | "
                  f"RSI={trade['rsi']:.1f} bucket={trade['rsi_bucket']} | "
                  f"entry=${trade['entry_price']:.2f} | market={outcome} | "
                  f"{w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f} | "
                  f"session=${self.session_pnl:+.2f} | "
                  f"{trade['question'][:45]}")

        for cid in to_remove:
            del self.active[cid]

        if to_remove:
            self._save()

    def _update_stats(self, trade: dict, won: bool):
        """Update global and per-bucket stats after trade resolution."""
        pnl = trade["pnl"]

        # Global stats
        self.stats["wins" if won else "losses"] += 1
        self.stats["pnl"] += pnl
        self.session_pnl += pnl

        # Per-RSI-bucket stats
        bucket = trade.get("rsi_bucket", "unknown")
        by_bucket = self.stats.setdefault("by_rsi_bucket", {})
        if bucket not in by_bucket:
            by_bucket[bucket] = {"wins": 0, "losses": 0, "pnl": 0.0}
        by_bucket[bucket]["wins" if won else "losses"] += 1
        by_bucket[bucket]["pnl"] = round(by_bucket[bucket]["pnl"] + pnl, 2)

    # ========================================================================
    # STATUS DISPLAY
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
        window_str = ""
        if closest:
            q = closest.get("question", "?")[:55]
            tl = closest.get("_time_left_sec", 0)
            in_window = ENTRY_WINDOW_MIN < tl <= ENTRY_WINDOW_MAX
            window_str = " [IN WINDOW]" if in_window else ""
            market_str = f"{q} | {tl:.0f}s left{window_str}"

        active_count = len(self.active)
        rsi = self._last_rsi
        vol_str = "VOL OK" if self._last_vol_ok else "vol low"

        print(f"\n--- {now_mst.strftime('%H:%M:%S')} MST | RSI CONTRARIAN | "
              f"Active: {active_count} | "
              f"{w}W/{l}L {wr:.0f}%WR | "
              f"PnL: ${self.stats['pnl']:+.2f} | "
              f"RSI={rsi:.1f} ({vol_str}) ---")
        print(f"    Next 5M: {market_str}")

        # RSI bucket stats
        by_bucket = self.stats.get("by_rsi_bucket", {})
        if by_bucket:
            bucket_parts = []
            for bucket in sorted(by_bucket.keys()):
                bdata = by_bucket[bucket]
                bw, bl = bdata["wins"], bdata["losses"]
                bpnl = bdata["pnl"]
                bucket_parts.append(f"{bucket}: {bw}W/{bl}L ${bpnl:+.2f}")
            print(f"    RSI buckets: {' | '.join(bucket_parts)}")

        if self.session_pnl <= -DAILY_LOSS_LIMIT:
            print(f"    *** DAILY LOSS LIMIT HIT: ${self.session_pnl:+.2f} ***")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        print("=" * 70)
        print("  RSI CONTRARIAN 5M PAPER TRADER V1.0")
        print("=" * 70)
        print("  Strategy: Mean reversion on BTC 5-minute Up/Down markets")
        print(f"  Signal: RSI < {RSI_OVERSOLD} -> bet UP (oversold) | "
              f"RSI > {RSI_OVERBOUGHT} -> bet DOWN (overbought)")
        print(f"  Volume: Previous candle must show {VOLUME_MULTIPLIER}x average volume")
        print(f"  Entry: ${MIN_ENTRY:.2f}-${MAX_ENTRY:.2f} (cheap side for 2x+ payout)")
        print(f"  Entry window: {ENTRY_WINDOW_MIN//60}-{ENTRY_WINDOW_MAX//60} min before close")
        print(f"  Size: ${TRADE_SIZE:.2f}/trade | Max concurrent: {MAX_CONCURRENT}")
        print("  Resolution: Binance 5M candle (close vs open)")
        print("  Note: Contrarian was 8% WR in 15M backtest -- testing on 5M at cheap entries")
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

                # Check for entries
                await self.check_entry(markets)

                # Periodic status log
                self.print_status(markets)

                # Save periodically (every 12 cycles ~= 60s)
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
              f"Session: ${self.session_pnl:+.2f}")

        # Print final RSI bucket breakdown
        by_bucket = self.stats.get("by_rsi_bucket", {})
        if by_bucket:
            print("\n[BUCKET BREAKDOWN]")
            for bucket in sorted(by_bucket.keys()):
                bdata = by_bucket[bucket]
                bw, bl = bdata["wins"], bdata["losses"]
                bt = bw + bl
                bwr = bw / bt * 100 if bt > 0 else 0
                bpnl = bdata["pnl"]
                print(f"  RSI {bucket}: {bw}W/{bl}L {bwr:.0f}%WR | PnL: ${bpnl:+.2f}")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    lock = acquire_pid_lock("rsi_contrarian_5m")
    try:
        trader = RSIContrarian5MTrader()
        asyncio.run(trader.run())
    finally:
        release_pid_lock("rsi_contrarian_5m")
