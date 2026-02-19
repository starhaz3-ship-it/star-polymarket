"""
Whale Consensus 15M Paper Trader V1.0

Paper-only strategy: Enter 15M BTC Up/Down markets when 3+ tracked whales
independently agree on a direction (all positioned the same way).

Key Insight:
  When multiple profitable whales ($97K-$885K PnL each) independently take the
  same directional position on a 15M market, it's a high-confidence signal.
  We monitor their live positions via the Polymarket data API and paper-trade
  when consensus forms.

Rules:
  - 10 tracked whales (top performers from our research)
  - Consensus: 3+ whales agree on same direction (UP or DOWN)
  - Each whale must have >$5 cost basis on the market to count
  - Entry window: 3-12 minutes before market close
  - Paper trade size: $2.50/trade
  - Max 3 concurrent positions
  - Poll every 20 seconds
  - Resolution via Binance 15m candle (close > open = UP, else DOWN)

Usage:
  python run_whale_consensus_15m.py          # Paper mode (only mode)
"""

import sys
import os
import json
import time
import asyncio
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from functools import partial as fn_partial
from typing import Dict, List, Optional, Tuple

import httpx

print = fn_partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# CONFIG
# ============================================================================
POLL_INTERVAL = 20              # seconds between cycles (fast — consensus can form quickly)
MAX_CONCURRENT = 3              # max open paper positions
CONSENSUS_MIN = 3               # minimum whales agreeing for consensus
MIN_WHALE_COST = 5.0            # minimum cost basis ($) for a whale vote to count
TRADE_SIZE = 2.50               # paper trade size per entry
MIN_ENTRY_PRICE = 0.10          # minimum entry price (avoid extremes)
MAX_ENTRY_PRICE = 0.65          # maximum entry price
TIME_WINDOW_MIN = 3.0           # earliest entry (minutes before close)
TIME_WINDOW_MAX = 12.0          # latest entry (minutes before close)
RESOLVE_DELAY_SEC = 20          # seconds after endDate to wait before resolving
SPREAD_OFFSET = 0.02            # simulated spread for paper fills

RESULTS_FILE = Path(__file__).parent / "whale_consensus_results.json"

# ============================================================================
# TRACKED WHALES — Top performers from our research
# ============================================================================
CONSENSUS_WHALES = {
    "0x732F1":          "0x732f189193d7a8c8bc8d8eb91f501a22736af081",      # $25K+, multi-asset, ~93% visible WR
    "vidarx":           "0x2d8b401d2f0e6937afebf18e19e11ca568a5260a",      # $154K PnL, $18.75M vol
    "Square-Guy":       "0x1979ae6b7e6534de9c4539d0c205e582ca637c9d",      # $500K PnL, crypto Up/Down
    "0x8dxd":           "0x63ce342161250d705dc0b16df89036c8e5f9ba9a",      # $885K PnL, early entries
    "Canine-Commandment": "0x1d0034134e339a309700ff2d34e99fa2d48b0313",    # $270K PnL, pairs arb
    "k9Q2mX4L8A7ZP3R": "0xd0d6053c3c37e727402d84c14069780d360993aa",      # $712K PnL, multi-timeframe
    "BoneReader":       "0xd84c2b6d65dc596f49c7b6aadd6d74ca91e407b9",      # High-freq BTC/ETH
    "vague-sourdough":  "0x70ec235a31eb35f243e2618d6ea3b5b8962bbb5d",      # $48K PnL, 5m scalper
    "gabagool22":       "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d",      # $849K PnL, high volume
    "bratanbratishka":  "0xcbb1a3174d9ac5a0f57f5b86808204b9382e7afb",      # $97K PnL, 87% WR sniper
}

# ============================================================================
# HELPERS
# ============================================================================

def mst_now() -> str:
    """Current time formatted in MST (UTC-7)."""
    utc = datetime.now(timezone.utc)
    mst = utc - timedelta(hours=7)
    return mst.strftime("%I:%M:%S %p MST")


def mst_format(dt: datetime) -> str:
    """Format a UTC datetime as MST time string."""
    mst = dt - timedelta(hours=7)
    return mst.strftime("%I:%M %p")


def atomic_save(filepath: Path, data: dict):
    """Write JSON atomically — write to .tmp then rename."""
    tmp = filepath.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2))
        # On Windows, need to remove target first if it exists
        if filepath.exists():
            filepath.unlink()
        tmp.rename(filepath)
    except Exception as e:
        print(f"[SAVE] Error: {e}")
        # Fallback: direct write
        try:
            filepath.write_text(json.dumps(data, indent=2))
        except Exception as e2:
            print(f"[SAVE] Fallback also failed: {e2}")


# ============================================================================
# WHALE CONSENSUS 15M PAPER TRADER
# ============================================================================
class WhaleConsensus15MTrader:
    def __init__(self):
        self.active: Dict[str, dict] = {}       # {trade_key: trade_dict}
        self.resolved: List[dict] = []
        self.attempted_cids: set = set()         # condition IDs already attempted (skip duplicates)
        self.stats = {
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        self.last_scan_result: Optional[str] = None   # status line from last scan
        self.last_scan_time: Optional[str] = None
        self._load()

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _load(self):
        """Load state from results file."""
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.active = data.get("active", {})
                self.resolved = data.get("resolved", [])
                self.stats = data.get("stats", self.stats)
                # Rebuild attempted CIDs from resolved + active
                for trade in self.resolved:
                    cid = trade.get("condition_id", "")
                    if cid:
                        self.attempted_cids.add(cid)
                for trade in self.active.values():
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
        """Save state atomically."""
        data = {
            "active": self.active,
            "resolved": self.resolved[-500:],    # keep last 500 resolved
            "stats": self.stats,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        atomic_save(RESULTS_FILE, data)

    # ========================================================================
    # MARKET DISCOVERY (15M BTC only)
    # ========================================================================

    async def discover_15m_markets(self) -> List[dict]:
        """Find active Polymarket 15M BTC Up/Down markets."""
        markets = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    "https://gamma-api.polymarket.com/events",
                    params={
                        "tag_slug": "15M",
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
                    # BTC only
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
                        except Exception:
                            continue
                        if end_dt < now:
                            continue

                        time_left_min = (end_dt - now).total_seconds() / 60
                        m["_time_left_min"] = time_left_min
                        m["_end_dt"] = end_dt.isoformat()
                        m["_end_dt_parsed"] = end_dt
                        m["_event_title"] = event.get("title", "")
                        if not m.get("question"):
                            m["question"] = event.get("title", "")
                        markets.append(m)
        except Exception as e:
            print(f"[API] Discovery error: {e}")
        return markets

    # ========================================================================
    # MARKET HELPERS
    # ========================================================================

    def get_prices(self, market: dict) -> Tuple[Optional[float], Optional[float]]:
        """Extract UP/DOWN prices from market data."""
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

    def _format_time_window(self, market: dict) -> str:
        """Format market time window as readable MST string."""
        try:
            end_dt = market["_end_dt_parsed"]
            start_dt = end_dt - timedelta(minutes=15)
            return f"{mst_format(start_dt)}-{mst_format(end_dt)}"
        except Exception:
            return "?"

    # ========================================================================
    # WHALE POSITION CHECKING
    # ========================================================================

    async def _check_whale_position(
        self,
        client: httpx.AsyncClient,
        name: str,
        address: str,
        condition_id: str,
    ) -> Optional[Tuple[str, str, float]]:
        """
        Check if a whale has a position on a specific market.

        Returns:
            (whale_name, direction, cost) or None if no qualifying position.
        """
        try:
            r = await client.get(
                "https://data-api.polymarket.com/positions",
                params={"user": address, "sizeThreshold": "0"},
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if r.status_code != 200:
                return None

            up_cost, down_cost = 0.0, 0.0
            for p in r.json():
                if p.get("conditionId") != condition_id:
                    continue
                outcome = (p.get("outcome") or "").upper()
                size = float(p.get("size", 0) or 0)
                avg_price = float(p.get("avgPrice", 0) or 0)
                cost = size * avg_price

                if outcome == "UP":
                    up_cost += cost
                elif outcome == "DOWN":
                    down_cost += cost

            if up_cost > down_cost and up_cost >= MIN_WHALE_COST:
                return (name, "UP", round(up_cost, 2))
            elif down_cost > up_cost and down_cost >= MIN_WHALE_COST:
                return (name, "DOWN", round(down_cost, 2))
            return None
        except Exception:
            return None

    # ========================================================================
    # CONSENSUS SCAN
    # ========================================================================

    async def scan_whale_consensus(self):
        """Check all whales' positions on active 15M BTC markets."""
        # 1. Discover active 15M BTC markets
        markets = await self.discover_15m_markets()
        target_markets = [
            m for m in markets
            if TIME_WINDOW_MIN <= m.get("_time_left_min", 999) <= TIME_WINDOW_MAX
        ]

        if not target_markets:
            return

        # 2. For each target market, poll all whales' positions
        for market in target_markets:
            cid = market.get("conditionId", "")
            if not cid:
                continue
            if cid in self.attempted_cids:
                continue
            if len(self.active) >= MAX_CONCURRENT:
                break

            # Poll all whale positions in parallel
            whale_votes: Dict[str, List[Tuple[str, float]]] = {"UP": [], "DOWN": []}

            async with httpx.AsyncClient(timeout=10) as client:
                tasks = []
                for name, address in CONSENSUS_WHALES.items():
                    tasks.append(self._check_whale_position(client, name, address, cid))
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception) or result is None:
                        continue
                    name, direction, cost = result
                    if direction and cost >= MIN_WHALE_COST:
                        whale_votes[direction].append((name, cost))

            # 3. Check consensus
            up_count = len(whale_votes["UP"])
            down_count = len(whale_votes["DOWN"])

            time_left = market.get("_time_left_min", 0)
            time_win = self._format_time_window(market)

            if up_count >= CONSENSUS_MIN and up_count > down_count:
                # UP consensus
                whale_names = [w[0] for w in whale_votes["UP"]]
                total_cost = sum(w[1] for w in whale_votes["UP"])
                self.last_scan_result = (
                    f"UP:{up_count} DOWN:{down_count} -> CONSENSUS UP "
                    f"({', '.join(whale_names)}) total=${total_cost:.0f}"
                )
                self.last_scan_time = mst_now()
                print(f"[CONSENSUS] UP x{up_count} ({', '.join(whale_names)}) | "
                      f"DOWN x{down_count} | {time_win} | {time_left:.1f}m left")
                await self._enter_consensus_trade(market, "UP", whale_votes["UP"])

            elif down_count >= CONSENSUS_MIN and down_count > up_count:
                # DOWN consensus
                whale_names = [w[0] for w in whale_votes["DOWN"]]
                total_cost = sum(w[1] for w in whale_votes["DOWN"])
                self.last_scan_result = (
                    f"UP:{up_count} DOWN:{down_count} -> CONSENSUS DOWN "
                    f"({', '.join(whale_names)}) total=${total_cost:.0f}"
                )
                self.last_scan_time = mst_now()
                print(f"[CONSENSUS] DOWN x{down_count} ({', '.join(whale_names)}) | "
                      f"UP x{up_count} | {time_win} | {time_left:.1f}m left")
                await self._enter_consensus_trade(market, "DOWN", whale_votes["DOWN"])

            else:
                if up_count > 0 or down_count > 0:
                    self.last_scan_result = (
                        f"UP:{up_count} DOWN:{down_count} | No consensus"
                    )
                    self.last_scan_time = mst_now()
                    print(f"[SPLIT] {cid[:12]} | UP:{up_count} DOWN:{down_count} | "
                          f"No consensus | {time_win} | {time_left:.1f}m left")

    # ========================================================================
    # ENTRY
    # ========================================================================

    async def _enter_consensus_trade(
        self,
        market: dict,
        direction: str,
        voters: List[Tuple[str, float]],
    ):
        """Enter a paper trade based on whale consensus."""
        cid = market.get("conditionId", "")
        if not cid:
            return

        # Mark as attempted so we don't re-enter
        self.attempted_cids.add(cid)

        # Get entry price
        up_price, down_price = self.get_prices(market)
        if up_price is None or down_price is None:
            print(f"[SKIP] No prices for {cid[:12]}")
            return

        entry_price = up_price if direction == "UP" else down_price

        # Apply spread offset for paper fill
        entry_price = round(entry_price + SPREAD_OFFSET, 4)

        # Price bounds check
        if entry_price < MIN_ENTRY_PRICE or entry_price > MAX_ENTRY_PRICE:
            print(f"[SKIP] Price ${entry_price:.2f} out of range "
                  f"[${MIN_ENTRY_PRICE}-${MAX_ENTRY_PRICE}]")
            return

        # Calculate shares
        shares = TRADE_SIZE / entry_price
        actual_cost = round(shares * entry_price, 2)

        # Build trade record
        now = datetime.now(timezone.utc)
        trade_key = f"wc15m_{cid}_{direction}"
        whale_names = [w[0] for w in voters]
        whale_costs = {w[0]: w[1] for w in voters}
        total_whale_cost = sum(w[1] for w in voters)
        time_left = market.get("_time_left_min", 0)

        self.active[trade_key] = {
            "side": direction,
            "entry_price": entry_price,
            "size_usd": actual_cost,
            "shares": round(shares, 4),
            "entry_time": now.isoformat(),
            "condition_id": cid,
            "market_numeric_id": market.get("id"),
            "question": market.get("question", ""),
            "event_title": market.get("_event_title", ""),
            "end_dt": market.get("_end_dt", ""),
            "strategy": "whale_consensus",
            "consensus_count": len(voters),
            "consensus_whales": whale_names,
            "consensus_whale_costs": whale_costs,
            "total_whale_cost": round(total_whale_cost, 2),
            "time_left_min": round(time_left, 1),
            "status": "open",
            "pnl": 0.0,
        }

        print(f"[ENTRY] {direction} ${entry_price:.2f} ${actual_cost:.2f} "
              f"({shares:.1f}sh) | {len(voters)} whales: "
              f"{', '.join(whale_names)} | whale_cost=${total_whale_cost:.0f} | "
              f"time={time_left:.1f}m | {market.get('question', '')[:60]}")

    # ========================================================================
    # RESOLUTION (via Binance 15m candle)
    # ========================================================================

    async def resolve_via_binance(self, end_dt: datetime) -> Optional[str]:
        """
        Resolve a 15M market by checking the Binance 15m candle.
        close > open = UP, else DOWN.
        """
        start_dt = end_dt - timedelta(minutes=15)
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    "https://api.binance.com/api/v3/klines",
                    params={
                        "symbol": "BTCUSDT",
                        "interval": "15m",
                        "startTime": start_ms,
                        "endTime": end_ms,
                        "limit": 5,
                    },
                )
                r.raise_for_status()
                klines = r.json()
                if not klines:
                    return None

                # Find best matching candle (closest open time to start_ms)
                best = min(klines, key=lambda k: abs(int(k[0]) - start_ms))
                open_p = float(best[1])
                close_p = float(best[4])

                if close_p > open_p:
                    return "UP"
                elif close_p < open_p:
                    return "DOWN"
                else:
                    return "DOWN"  # tie = DOWN
        except Exception as e:
            print(f"[RESOLVE] Binance error: {e}")
            return None

    async def resolve_trades(self):
        """Resolve expired trades using Binance 15m candles."""
        now = datetime.now(timezone.utc)

        for tid, trade in list(self.active.items()):
            if trade.get("status") != "open":
                continue

            end_dt_str = trade.get("end_dt", "")
            if not end_dt_str:
                continue

            try:
                end_dt = datetime.fromisoformat(end_dt_str.replace("Z", "+00:00"))
            except Exception:
                continue

            # Wait until endDate + delay
            elapsed_after_end = (now - end_dt).total_seconds()
            if elapsed_after_end < RESOLVE_DELAY_SEC:
                continue

            # Try Binance resolution
            outcome = await self.resolve_via_binance(end_dt)
            if outcome is None:
                # If more than 5 minutes past end and still can't resolve, try again next cycle
                if elapsed_after_end > 300:
                    print(f"[RESOLVE] Can't resolve {tid} after 5m — marking LOSS")
                    outcome = "DOWN" if trade["side"] == "UP" else "UP"  # assume loss
                else:
                    continue

            # Determine win/loss
            won = trade["side"] == outcome
            if won:
                # Shares pay $1 each
                pnl = round(trade["shares"] - trade["size_usd"], 2)
            else:
                # Shares worth $0
                pnl = round(-trade["size_usd"], 2)

            trade["status"] = "resolved"
            trade["pnl"] = pnl
            trade["outcome"] = outcome
            trade["resolve_time"] = now.isoformat()
            trade["exit_price"] = 1.0 if won else 0.0

            # Update stats
            if won:
                self.stats["wins"] += 1
            else:
                self.stats["losses"] += 1
            self.stats["pnl"] = round(self.stats["pnl"] + pnl, 2)

            # Move to resolved
            self.resolved.append(trade)
            del self.active[tid]

            # Log
            w, l = self.stats["wins"], self.stats["losses"]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            tag = "WIN" if won else "LOSS"
            whales_str = ", ".join(trade.get("consensus_whales", []))

            print(f"[{tag}] {trade['side']} ${pnl:+.2f} | "
                  f"entry=${trade['entry_price']:.2f} | outcome={outcome} | "
                  f"{trade['consensus_count']} whales ({whales_str}) | "
                  f"{w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f} | "
                  f"{trade.get('question', '')[:50]}")

    # ========================================================================
    # CLEANUP — Prune old attempted CIDs to prevent memory bloat
    # ========================================================================

    def _prune_attempted_cids(self):
        """Keep attempted_cids from growing forever. Keep last 500."""
        if len(self.attempted_cids) > 1000:
            # We can't easily prune a set by time, so just keep the most recent
            # by rebuilding from resolved (newest first) + active
            recent_cids = set()
            for trade in self.active.values():
                cid = trade.get("condition_id", "")
                if cid:
                    recent_cids.add(cid)
            for trade in self.resolved[-500:]:
                cid = trade.get("condition_id", "")
                if cid:
                    recent_cids.add(cid)
            self.attempted_cids = recent_cids

    # ========================================================================
    # STATUS DISPLAY
    # ========================================================================

    def print_status(self, cycle: int, markets: List[dict]):
        """Print status line."""
        now_mst = mst_now()
        w, l = self.stats["wins"], self.stats["losses"]
        total = w + l
        wr = w / total * 100 if total > 0 else 0
        wr_str = f"{w}W/{l}L {wr:.0f}%" if total > 0 else "0W/0L"

        active_count = len(self.active)
        pnl = self.stats["pnl"]

        print(f"\n--- {now_mst} | WHALE CONSENSUS | "
              f"Active: {active_count} | {wr_str} | PnL: ${pnl:+.2f} ---")

        # Show next 15M market in window
        in_window = [
            m for m in markets
            if TIME_WINDOW_MIN <= m.get("_time_left_min", 999) <= TIME_WINDOW_MAX
        ]
        upcoming = [
            m for m in markets
            if m.get("_time_left_min", 999) > TIME_WINDOW_MAX
        ]

        if in_window:
            m = in_window[0]
            time_win = self._format_time_window(m)
            tl = m.get("_time_left_min", 0)
            print(f"    Next 15M: {m.get('question', m.get('_event_title', ''))[:60]} "
                  f"| {time_win} | {tl:.1f} min left [IN WINDOW]")
        elif upcoming:
            m = min(upcoming, key=lambda x: x.get("_time_left_min", 999))
            time_win = self._format_time_window(m)
            tl = m.get("_time_left_min", 0)
            print(f"    Next 15M: {m.get('question', m.get('_event_title', ''))[:60]} "
                  f"| {time_win} | {tl:.1f} min left")
        else:
            print(f"    No active 15M BTC markets found")

        if self.last_scan_result:
            print(f"    Last scan: {self.last_scan_result}")

        # Show active trades
        if self.active:
            for tid, trade in self.active.items():
                whales = ", ".join(trade.get("consensus_whales", [])[:4])
                print(f"    OPEN: {trade['side']} ${trade['entry_price']:.2f} "
                      f"(x{trade['consensus_count']} {whales}) | "
                      f"{trade.get('question', '')[:40]}")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        """Main trading loop."""
        # Banner
        print("=" * 70)
        print("  WHALE CONSENSUS 15M PAPER TRADER V1.0")
        print("=" * 70)
        print(f"  Strategy: Follow 3+ whale agreement on BTC 15M Up/Down direction")
        print(f"  Whales tracked: {len(CONSENSUS_WHALES)} top performers ($97K-$885K PnL each)")
        print(f"  Consensus: {CONSENSUS_MIN}+ whales agree = ENTER | Split = SKIP")
        print(f"  Entry window: {TIME_WINDOW_MIN:.0f}-{TIME_WINDOW_MAX:.0f} min before close")
        print(f"  Size: ${TRADE_SIZE:.2f}/trade | Max concurrent: {MAX_CONCURRENT}")
        print(f"  Poll: {POLL_INTERVAL}s | Resolution: Binance 15M candle")
        print("=" * 70)

        # Show whale roster
        print(f"\n  Tracked whales:")
        for name, addr in CONSENSUS_WHALES.items():
            print(f"    {name:<22s} {addr[:10]}...{addr[-4:]}")
        print()

        # Show resume stats if applicable
        if self.resolved:
            w, l = self.stats["wins"], self.stats["losses"]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            print(f"[RESUME] {w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
                  f"{len(self.resolved)} resolved, {len(self.active)} active")

        cycle = 0
        while True:
            try:
                cycle += 1

                # 1. Discover markets (needed for both resolve + scan)
                markets = await self.discover_15m_markets()

                # 2. Resolve expired trades
                await self.resolve_trades()

                # 3. Scan for whale consensus on in-window markets
                await self.scan_whale_consensus()

                # 4. Status display every 3 cycles (~60 seconds)
                if cycle % 3 == 0:
                    self.print_status(cycle, markets)

                # 5. Prune old CIDs periodically
                if cycle % 100 == 0:
                    self._prune_attempted_cids()

                # 6. Save state
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
    lock = acquire_pid_lock("whale_consensus_15m")
    try:
        trader = WhaleConsensus15MTrader()
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        print("\n[EXIT] Whale Consensus 15M stopped.")
    finally:
        release_pid_lock("whale_consensus_15m")
