"""
0x732F1 Shadow Trader V1.0 — PAPER MODE ONLY

Follows whale 0x732F1 (Antique-Twig) directional bias on Polymarket crypto
Up/Down markets. When the whale's UP spend is 1.5x+ their DOWN spend (or vice
versa), we paper-trade the dominant direction.

Target wallet: 0x732f189193d7a8c8bc8d8eb91f501a22736af081
Portfolio: $25K+, ~93% visible WR on crypto Up/Down markets.

Strategy:
  1. Poll whale positions every 30 seconds
  2. Group by conditionId, sum UP vs DOWN cost (size * avgPrice)
  3. If ratio >= 1.5x, follow the dominant direction
  4. Paper trade $2.50 on the bias direction
  5. Resolve via Binance candle (close > open = UP, else DOWN)

Usage:
  python run_shadow_0x732f1.py
"""

import sys
import os
import json
import time
import re
import asyncio
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from functools import partial as fn_partial
from typing import Dict, Optional, Tuple

import httpx
from dotenv import load_dotenv

print = fn_partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# CONFIG
# ============================================================================
TARGET_WALLET = "0x732f189193d7a8c8bc8d8eb91f501a22736af081"
TARGET_NAME = "0x732F1"
TARGET_ALIAS = "Antique-Twig"

POLL_INTERVAL = 30          # seconds between whale position polls
LOG_INTERVAL = 60           # seconds between status line prints
TRADE_SIZE = 2.50           # paper trade size ($)
MAX_CONCURRENT = 5          # max simultaneous paper positions
BIAS_RATIO_MIN = 1.5        # minimum UP:DOWN or DOWN:UP spend ratio to trigger
MIN_WHALE_SPEND = 10.0      # minimum total spend ($) per market to consider
RESOLVE_DELAY = 30          # seconds after market end to wait before resolving

BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
POSITIONS_URL = "https://data-api.polymarket.com/positions"
ACTIVITY_URL = "https://data-api.polymarket.com/activity"

RESULTS_FILE = Path(__file__).parent / "shadow_0x732f1_results.json"

# Crypto keywords for filtering crypto Up/Down markets
CRYPTO_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "XRP": ["xrp", "ripple"],
}

# Binance symbol mapping
BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}

# MST offset (UTC - 7)
MST_OFFSET = timedelta(hours=-7)


# ============================================================================
# UTILITIES
# ============================================================================

def mst_now() -> datetime:
    """Current time in MST."""
    return datetime.now(timezone.utc) + MST_OFFSET


def mst_str() -> str:
    """Current time formatted as HH:MM:SS MST."""
    return mst_now().strftime("%H:%M:%S")


def atomic_save(filepath: Path, data: dict):
    """Write JSON atomically: write to .tmp then rename."""
    tmp = filepath.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2))
        if filepath.exists():
            filepath.unlink()
        tmp.rename(filepath)
    except Exception as e:
        print(f"[SAVE] Error: {e}")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def detect_asset(title: str) -> Optional[str]:
    """Detect crypto asset from market title."""
    title_lower = title.lower()
    for asset, keywords in CRYPTO_KEYWORDS.items():
        if any(kw in title_lower for kw in keywords):
            return asset
    return None


def detect_duration(title: str) -> int:
    """Detect market duration in minutes from title time range pattern.

    Examples:
      '3:00PM-3:15PM' = 15min
      '3:10PM-3:15PM' = 5min
      '3PM ET' = 60min (standalone = hourly)
      '8:00PM-12:00AM' = 240min (4hr)
    """
    range_match = re.search(
        r'(\d{1,2}):(\d{2})\s*(AM|PM)\s*-\s*(\d{1,2}):(\d{2})\s*(AM|PM)',
        title, re.IGNORECASE
    )
    if range_match:
        h1 = int(range_match.group(1))
        m1 = int(range_match.group(2))
        ap1 = range_match.group(3).upper()
        h2 = int(range_match.group(4))
        m2 = int(range_match.group(5))
        ap2 = range_match.group(6).upper()

        t1 = (h1 % 12 + (12 if ap1 == "PM" else 0)) * 60 + m1
        t2 = (h2 % 12 + (12 if ap2 == "PM" else 0)) * 60 + m2
        if t2 < t1:
            t2 += 24 * 60  # crosses midnight

        diff = t2 - t1
        if diff <= 6:
            return 5
        elif diff <= 16:
            return 15
        elif diff <= 61:
            return 60
        else:
            return 240
    else:
        # Standalone time reference (e.g. "11 PM ET") = hourly
        return 60


def binance_interval(duration_min: int) -> str:
    """Map duration in minutes to Binance kline interval string."""
    if duration_min <= 5:
        return "5m"
    elif duration_min <= 15:
        return "15m"
    elif duration_min <= 60:
        return "1h"
    else:
        return "4h"


# ============================================================================
# SHADOW TRADER
# ============================================================================

class Shadow0x732F1:
    def __init__(self):
        self.active_trades: Dict[str, dict] = {}   # {condition_id: trade}
        self.resolved: list = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0,
                      "start_time": datetime.now(timezone.utc).isoformat()}
        self.attempted_cids: set = set()  # avoid re-entering the same market
        self.running = True
        self._last_log_time = 0.0
        self._last_whale_summary: Dict[str, dict] = {}  # for status display
        self._load()

    # ====================================================================
    # PERSISTENCE
    # ====================================================================

    def _load(self):
        """Load saved state from results file."""
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.active_trades = data.get("active", {})
                self.resolved = data.get("resolved", [])
                self.stats = data.get("stats", self.stats)
                self.attempted_cids = set(data.get("attempted_cids", []))
                w, l = self.stats["wins"], self.stats["losses"]
                total = w + l
                wr = w / total * 100 if total > 0 else 0
                print(f"[LOAD] {len(self.resolved)} resolved, "
                      f"{len(self.active_trades)} active | "
                      f"{w}W/{l}L {wr:.0f}%WR | "
                      f"PnL: ${self.stats['pnl']:+.2f}")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        """Save state atomically."""
        data = {
            "active": self.active_trades,
            "resolved": self.resolved[-500:],  # keep last 500
            "stats": self.stats,
            "attempted_cids": list(self.attempted_cids)[-2000:],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        atomic_save(RESULTS_FILE, data)

    # ====================================================================
    # WHALE POSITION DETECTION
    # ====================================================================

    async def detect_whale_bias(self) -> Dict[str, dict]:
        """Poll whale positions and detect directional bias per market.

        Returns: {condition_id: {direction, ratio, up_cost, down_cost,
                                 title, end_date, cid, asset, duration_min}}
        """
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    POSITIONS_URL,
                    params={"user": TARGET_WALLET, "sizeThreshold": "0"},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if r.status_code != 200:
                    print(f"[WHALE] Positions API error: {r.status_code}")
                    return {}
                positions = r.json()
        except Exception as e:
            print(f"[WHALE] Positions fetch error: {e}")
            return {}

        # Group by conditionId
        market_positions: Dict[str, dict] = {}
        for p in positions:
            title = p.get("title", "")
            # Only crypto Up/Down markets
            if "Up or Down" not in title:
                continue

            asset = detect_asset(title)
            if asset is None:
                continue

            cid = p.get("conditionId", "")
            if not cid:
                continue

            outcome = (p.get("outcome", "") or "").upper()
            size = float(p.get("size", 0) or 0)
            avg_price = float(p.get("avgPrice", 0) or 0)
            cost = size * avg_price
            end_date = p.get("endDate", "")

            if cid not in market_positions:
                market_positions[cid] = {
                    "up_cost": 0.0,
                    "down_cost": 0.0,
                    "title": title,
                    "end_date": end_date,
                    "cid": cid,
                    "asset": asset,
                    "duration_min": detect_duration(title),
                }

            if outcome == "UP":
                market_positions[cid]["up_cost"] += cost
            elif outcome == "DOWN":
                market_positions[cid]["down_cost"] += cost

        # Calculate bias
        signals: Dict[str, dict] = {}
        for cid, data in market_positions.items():
            up = data["up_cost"]
            down = data["down_cost"]
            total = up + down
            if total < MIN_WHALE_SPEND:
                continue

            ratio = up / max(down, 0.01)
            inv_ratio = down / max(up, 0.01)

            if ratio >= BIAS_RATIO_MIN:
                signals[cid] = {
                    "direction": "UP",
                    "ratio": round(ratio, 1),
                    **data,
                }
            elif inv_ratio >= BIAS_RATIO_MIN:
                signals[cid] = {
                    "direction": "DOWN",
                    "ratio": round(inv_ratio, 1),
                    **data,
                }
            # else: no clear bias, skip

        self._last_whale_summary = signals
        return signals

    # ====================================================================
    # MARKET PRICE FETCH (for paper entry price)
    # ====================================================================

    async def fetch_market_prices(self, condition_id: str) -> Optional[Tuple[float, float]]:
        """Fetch current UP/DOWN outcome prices from Gamma API.

        Returns (up_price, down_price) or None on failure.
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={"condition_id": condition_id},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if r.status_code != 200:
                    return None
                markets = r.json()
                if not markets:
                    return None

                market = markets[0] if isinstance(markets, list) else markets
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
                return (up_price, down_price) if up_price is not None and down_price is not None else None
        except Exception as e:
            print(f"[GAMMA] Price fetch error for {condition_id[:12]}...: {e}")
            return None

    # ====================================================================
    # ENTRY LOGIC
    # ====================================================================

    async def check_entries(self, signals: Dict[str, dict]):
        """Enter paper trades based on whale bias signals."""
        now = datetime.now(timezone.utc)
        open_count = len(self.active_trades)

        for cid, sig in signals.items():
            if open_count >= MAX_CONCURRENT:
                break

            # Skip if already traded or attempted this market
            if cid in self.active_trades or cid in self.attempted_cids:
                continue

            # Check if market hasn't expired
            end_date_str = sig.get("end_date", "")
            if end_date_str:
                try:
                    end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                    if end_dt <= now:
                        self.attempted_cids.add(cid)
                        continue
                    # Skip if less than 30 seconds to expiry (too late)
                    if (end_dt - now).total_seconds() < 30:
                        continue
                except Exception:
                    continue
            else:
                continue  # No end date = can't resolve, skip

            # Fetch current prices from Gamma API
            prices = await self.fetch_market_prices(cid)
            if prices is None:
                continue
            up_price, down_price = prices

            direction = sig["direction"]
            entry_price = up_price if direction == "UP" else down_price

            # Skip extreme prices (near 0 or 1 = no edge)
            if entry_price < 0.05 or entry_price > 0.95:
                self.attempted_cids.add(cid)
                continue

            # Paper-enter
            shares = round(TRADE_SIZE / entry_price, 2)
            cost = round(shares * entry_price, 2)

            trade = {
                "condition_id": cid,
                "direction": direction,
                "entry_price": round(entry_price, 4),
                "shares": shares,
                "cost": cost,
                "whale_ratio": sig["ratio"],
                "whale_up_cost": round(sig["up_cost"], 2),
                "whale_down_cost": round(sig["down_cost"], 2),
                "title": sig["title"],
                "asset": sig["asset"],
                "duration_min": sig["duration_min"],
                "end_date": end_date_str,
                "entry_time": now.isoformat(),
                "status": "open",
                "pnl": 0.0,
            }

            self.active_trades[cid] = trade
            self.attempted_cids.add(cid)
            open_count += 1

            print(f"[ENTRY] {sig['asset']}:{direction} ${entry_price:.2f} "
                  f"${cost:.2f} ({shares:.1f}sh) | "
                  f"whale {sig['ratio']:.1f}x {direction} bias "
                  f"(UP:${sig['up_cost']:.0f} DN:${sig['down_cost']:.0f}) | "
                  f"{sig['duration_min']}min | {sig['title'][:60]}")

    # ====================================================================
    # RESOLUTION VIA BINANCE CANDLE
    # ====================================================================

    async def resolve_via_binance(self, trade: dict) -> Optional[str]:
        """Resolve a trade by fetching the Binance candle covering the market window.

        Returns 'UP', 'DOWN', or None if data unavailable yet.
        """
        asset = trade.get("asset", "BTC")
        symbol = BINANCE_SYMBOLS.get(asset, "BTCUSDT")
        duration_min = trade.get("duration_min", 60)
        end_date_str = trade.get("end_date", "")

        try:
            end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        except Exception:
            return None

        start_dt = end_dt - timedelta(minutes=duration_min)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        interval = binance_interval(duration_min)

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    BINANCE_REST_URL,
                    params={
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": start_ms,
                        "endTime": end_ms,
                        "limit": 10,
                    },
                )
                r.raise_for_status()
                klines = r.json()

                if not klines:
                    # Fallback: 1-minute candles
                    r2 = await client.get(
                        BINANCE_REST_URL,
                        params={
                            "symbol": symbol,
                            "interval": "1m",
                            "startTime": start_ms,
                            "endTime": end_ms,
                            "limit": 300,
                        },
                    )
                    r2.raise_for_status()
                    klines = r2.json()
                    if not klines:
                        return None
                    open_price = float(klines[0][1])
                    close_price = float(klines[-1][4])
                else:
                    # For exact-match intervals (e.g. 1h market -> 1h candle),
                    # pick the candle closest to our start time
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
                    return "DOWN"  # exact tie convention

        except Exception as e:
            print(f"[BINANCE] Resolution error ({asset} {interval}): {e}")
            return None

    # ====================================================================
    # RESOLVE EXPIRED TRADES
    # ====================================================================

    async def resolve_trades(self):
        """Check expired trades and resolve via Binance candle."""
        now = datetime.now(timezone.utc)

        for cid, trade in list(self.active_trades.items()):
            if trade.get("status") != "open":
                continue

            end_date_str = trade.get("end_date", "")
            if not end_date_str:
                continue

            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            except Exception:
                continue

            # Wait for market to expire + delay for candle to close
            seconds_past = (now - end_dt).total_seconds()
            if seconds_past < RESOLVE_DELAY:
                continue

            # Resolve
            outcome = await self.resolve_via_binance(trade)
            if outcome is None:
                # If it's been more than 10 minutes past end and still can't resolve,
                # log a warning but keep waiting
                if seconds_past > 600:
                    print(f"[WARN] Cannot resolve {trade['asset']}:{trade['direction']} "
                          f"after {seconds_past:.0f}s | {trade['title'][:40]}")
                continue

            direction = trade["direction"]
            won = (outcome == direction)

            if won:
                # Shares pay $1 each on win
                pnl = round(trade["shares"] * 1.0 - trade["cost"], 2)
            else:
                # Shares worth $0 on loss
                pnl = round(-trade["cost"], 2)

            trade["status"] = "closed"
            trade["pnl"] = pnl
            trade["market_outcome"] = outcome
            trade["exit_time"] = now.isoformat()
            trade["exit_price"] = 1.0 if won else 0.0

            if won:
                self.stats["wins"] += 1
            else:
                self.stats["losses"] += 1
            self.stats["pnl"] += pnl
            self.stats["pnl"] = round(self.stats["pnl"], 2)

            self.resolved.append(trade)
            del self.active_trades[cid]

            w, l = self.stats["wins"], self.stats["losses"]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            tag = "WIN" if won else "LOSS"
            print(f"[{tag}] {trade['asset']}:{direction} "
                  f"${pnl:+.2f} | market={outcome} | "
                  f"entry=${trade['entry_price']:.2f} | "
                  f"whale {trade['whale_ratio']:.1f}x bias | "
                  f"{w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f} | "
                  f"{trade['title'][:50]}")

    # ====================================================================
    # STATUS DISPLAY
    # ====================================================================

    def print_status(self):
        """Print periodic status line with whale positions and tracker state."""
        now_ts = time.time()
        if now_ts - self._last_log_time < LOG_INTERVAL:
            return
        self._last_log_time = now_ts

        w, l = self.stats["wins"], self.stats["losses"]
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        active = len(self.active_trades)
        wr_str = f"{w}/{l} {wr:.0f}%WR" if total > 0 else "0/0"

        print(f"\n--- {mst_str()} MST | SHADOW {TARGET_NAME} | "
              f"Active: {active} | {wr_str} | "
              f"PnL: ${self.stats['pnl']:+.2f} ---")

        # Whale positions summary
        if self._last_whale_summary:
            parts = []
            for cid, sig in self._last_whale_summary.items():
                asset = sig["asset"]
                # Extract short time reference from title
                title = sig.get("title", "")
                time_ref = ""
                tm = re.search(r'(\d{1,2}(?::\d{2})?\s*(?:AM|PM))', title, re.IGNORECASE)
                if tm:
                    time_ref = f" {tm.group(1)}"

                up_cost = sig["up_cost"]
                down_cost = sig["down_cost"]
                ratio = sig["ratio"]
                direction = sig["direction"]

                if direction == "UP":
                    parts.append(f"{asset}{time_ref} UP:${up_cost:.0f} DN:${down_cost:.0f} "
                                 f"({ratio:.1f}x UP bias)")
                else:
                    parts.append(f"{asset}{time_ref} UP:${up_cost:.0f} DN:${down_cost:.0f} "
                                 f"({ratio:.1f}x DN bias)")

            if parts:
                print(f"    Whale positions: {' | '.join(parts)}")

        # Following / skipping summary
        following = []
        skipping_assets = set()
        for cid, sig in self._last_whale_summary.items():
            following.append(f"{sig['asset']} {sig['direction']}")

        # Also show active trades
        active_parts = []
        for cid, trade in self.active_trades.items():
            active_parts.append(f"{trade['asset']}:{trade['direction']}")

        if following:
            follow_str = ", ".join(following)
            print(f"    Bias signals: {follow_str}", end="")
            if active_parts:
                print(f" | Active: {', '.join(active_parts)}", end="")
            print()

    # ====================================================================
    # MAIN LOOP
    # ====================================================================

    async def run(self):
        """Main trading loop."""
        print("=" * 70)
        print(f"  {TARGET_NAME} SHADOW TRADER V1.0 -- PAPER MODE")
        print("=" * 70)
        print(f"  Target: {TARGET_NAME} ({TARGET_ALIAS}) | $25K+ portfolio, ~93% visible WR")
        print(f"  Strategy: Follow whale's directional bias (UP:DOWN ratio > {BIAS_RATIO_MIN:.1f}x)")
        print(f"  Markets: BTC, ETH, SOL, XRP hourly + 15M Up/Down")
        print(f"  Size: ${TRADE_SIZE:.2f}/trade | Max concurrent: {MAX_CONCURRENT}")
        print(f"  Poll: {POLL_INTERVAL}s | Resolution: Binance candle")
        print("=" * 70)

        if self.resolved:
            w, l = self.stats["wins"], self.stats["losses"]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            print(f"[RESUME] {w}W/{l}L {wr:.0f}%WR | "
                  f"PnL: ${self.stats['pnl']:+.2f} | "
                  f"{len(self.resolved)} resolved, "
                  f"{len(self.active_trades)} active")

        cycle = 0
        while self.running:
            try:
                cycle += 1

                # 1. Detect whale bias
                signals = await self.detect_whale_bias()

                # 2. Resolve expired trades
                await self.resolve_trades()

                # 3. Enter new trades based on bias signals
                await self.check_entries(signals)

                # 4. Status display
                self.print_status()

                # 5. Save state
                self._save()

                await asyncio.sleep(POLL_INTERVAL)

            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Saving state...")
                self._save()
                break
            except Exception as e:
                print(f"[ERROR] Cycle {cycle}: {e}")
                traceback.print_exc()
                await asyncio.sleep(10)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    lock = acquire_pid_lock("shadow_0x732f1")
    try:
        trader = Shadow0x732F1()
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted.")
    finally:
        release_pid_lock("shadow_0x732f1")
