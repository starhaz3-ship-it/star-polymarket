"""
Whale Consensus 15M Trader V3.2

Enter 15M BTC Up/Down markets when 3+ tracked whales independently agree
on a direction (all positioned the same way).

Key Insight:
  When multiple profitable whales ($97K-$885K PnL each) independently take the
  same directional position on a 15M market, it's a high-confidence signal.
  We monitor their live positions via the Polymarket data API and trade
  when consensus forms.

Paper results: 136 trades, 67.6% WR, +$97.61
  - 3-whale consensus: 74% WR
  - 5-whale consensus: 72% WR

Rules:
  - 10 tracked whales (top performers from our research)
  - Consensus: 3+ whales agree on same direction (UP or DOWN)
  - Each whale must have >$5 cost basis on the market to count
  - Entry window: 3-12 minutes before market close
  - PROBATION: $2.50/trade, promotes to $5/trade after 20 trades at 60%+ WR
  - Max 3 concurrent positions
  - Poll every 20 seconds
  - Resolution via Binance 15m candle (close > open = UP, else DOWN)

Usage:
  python run_whale_consensus_15m.py              # Paper mode
  python run_whale_consensus_15m.py --live       # Live trading (REAL MONEY)
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

import math
import httpx

print = fn_partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# CLOB CONFIG (for live trading)
# ============================================================================
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
FOK_SLIPPAGE = 0.03         # cents above best ask to sweep multiple levels
MIN_SHARES = 3              # minimum shares for fill confirmation

# ============================================================================
# CONFIG
# ============================================================================
POLL_INTERVAL = 20              # seconds between cycles (fast — consensus can form quickly)
MAX_CONCURRENT = 3              # max open paper positions
CONSENSUS_MIN = 3               # minimum whales agreeing for consensus
MIN_WHALE_COST = 5.0            # minimum cost basis ($) for a whale vote to count
TRADE_SIZE = 2.50               # base trade size (PROBATION)
MIN_ENTRY_PRICE = 0.50          # V3.2: Raised from $0.10 — entries <$0.50 are 55% WR coin flips
MAX_ENTRY_PRICE = 0.65          # maximum entry price
UP_SIZE_MULT = 0.50             # V2.1: Half size on UP trades (CSV: DOWN 12W/0L 100%, UP 3W/3L 50%)
MIN_ENTRY_UP = 0.50             # V2.1: UP trades need $0.50+ entry (cheap UP = coin flip)
SKIP_HOURS_UTC = {4, 8, 15, 18}  # V3.2: Dead hours — UTC 4(40%WR), 8(43%), 15(38%), 18(33%)

# V3.0: Circuit Breaker + Kelly Sizing + SynthData
CIRCUIT_BREAKER_LOSSES = 3      # consecutive losses → halt
CIRCUIT_BREAKER_COOLDOWN = 1800  # 30 min cooldown after halt
CIRCUIT_BREAKER_HALFSIZE = 2    # consecutive losses → half-size mode
CIRCUIT_BREAKER_RECOVERY = 2    # trades at half-size before resuming full
WARMUP_SECONDS = 30             # seconds of stable data before trading
API_BACKOFF_BASE = 2.0          # exponential backoff base
API_BACKOFF_MAX = 30.0          # max backoff delay
KELLY_CAP = 0.05                # 5% max Kelly fraction
KELLY_MIN_TRADES = 10           # min trades before Kelly activates
BANKROLL_ESTIMATE = 100.0       # conservative bankroll estimate
SYNTH_API_KEY = os.environ.get("SYNTHDATA_API_KEY", "")
SYNTH_EDGE_THRESHOLD = 0.05    # 5pp minimum edge
SYNTH_SIZE_BOOST = 1.5         # 1.5x when Synth agrees
SYNTH_SIZE_REDUCE = 0.5        # 0.5x when Synth strongly disagrees


def get_trade_size(wins: int, losses: int, live_trades: int = 0,
                    entry_price: float = 0.50, cumulative_pnl: float = 0.0) -> float:
    """V3.0: Kelly criterion sizing with minimum data fallback."""
    total = wins + losses
    if total < KELLY_MIN_TRADES:
        return TRADE_SIZE  # $2.50 PROBATION until enough data
    bankroll = BANKROLL_ESTIMATE + cumulative_pnl
    if bankroll <= 10:
        return 1.0
    p_win = wins / total
    p_mkt = max(min(entry_price, 0.95), 0.05)
    b = (1 - p_mkt) / p_mkt
    f_star = (p_win * (b + 1) - 1) / b
    f_eff = min(max(f_star, 0), KELLY_CAP)
    if f_eff <= 0:
        return 0  # no edge
    size = round(bankroll * f_eff, 2)
    return max(min(size, 10.0), 1.0)

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
    # "BoneReader" REMOVED V3.2: 57% WR with him vs 79% without. Active value destroyer.
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
# V3.0: SYNTHDATA ORACLE
# ============================================================================
_synth_cache: dict = {"data": None, "ts": 0}

async def query_synthdata(asset: str = "BTC"):
    """Query SynthData for hourly probability forecast. Cached 60s."""
    if not SYNTH_API_KEY:
        return None
    now_ts = time.time()
    if _synth_cache["data"] and now_ts - _synth_cache["ts"] < 60:
        cached = _synth_cache["data"]
        if cached.get("_asset") == asset:
            return cached
    try:
        url = "https://api.synthdata.co/insights/polymarket/up-down/hourly"
        headers = {"Authorization": f"Apikey {SYNTH_API_KEY}"}
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(url, headers=headers, params={"asset": asset})
            if r.status_code != 200:
                return None
            data = r.json()
        synth_up = float(data.get("synth_probability_up", 0.5))
        poly_up = float(data.get("polymarket_probability_up", 0.5))
        result = {
            "_asset": asset,
            "synth_up": synth_up,
            "synth_down": 1 - synth_up,
            "poly_up": poly_up,
            "edge_up": synth_up - poly_up,
            "edge_down": -(synth_up - poly_up),
        }
        _synth_cache["data"] = result
        _synth_cache["ts"] = now_ts
        return result
    except Exception:
        return None


# ============================================================================
# WHALE CONSENSUS 15M PAPER TRADER
# ============================================================================
class WhaleConsensus15MTrader:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.client = None
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

        # V3.0: Circuit breaker
        self._circuit_halt_until = 0
        self._consecutive_losses = 0
        for trade in reversed(self.resolved):
            if trade.get("pnl", 0) >= 0:
                break
            self._consecutive_losses += 1

        self._recovery_trades = 0  # V3.1: trades at half-size after halt
        self._warmup_start = time.time()
        self._warmup_ready = False
        self._api_consecutive_errors = 0

        if not self.paper:
            self._init_clob()

    # ========================================================================
    # CLOB CLIENT (live trading)
    # ========================================================================

    def _init_clob(self):
        """Initialize CLOB client for live order execution."""
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
            self._cleanup_stale_orders()
        except Exception as e:
            print(f"[CLOB] Init failed: {e}")
            traceback.print_exc()
            print(f"[CLOB] Falling back to PAPER mode")
            self.paper = True

    def _cleanup_stale_orders(self):
        if not self.client:
            return
        try:
            resp = self.client.cancel_all()
            if resp:
                print(f"[CLEANUP] Cancelled stale orders from previous session")
        except Exception as e:
            print(f"[CLEANUP] No stale orders or error: {e}")

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

    async def get_best_ask(self, token_id: str) -> Optional[float]:
        """Get best ask price from CLOB orderbook."""
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

    async def get_orderbook_depth(self, token_id: str, max_price: float) -> Tuple[Optional[float], float, list]:
        """Get total available shares up to max_price."""
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(CLOB_BOOK_URL, params={"token_id": token_id})
                if r.status_code != 200:
                    return None, 0, []
                book = r.json()
                asks = book.get("asks", [])
                if not asks:
                    return None, 0, []
                sorted_asks = sorted(asks, key=lambda a: float(a["price"]))
                best_ask = float(sorted_asks[0]["price"])
                total_shares = 0
                levels = []
                for a in sorted_asks:
                    price = float(a["price"])
                    size = float(a["size"])
                    if price <= max_price:
                        total_shares += size
                        levels.append((price, size))
                return best_ask, total_shares, levels
        except Exception:
            return None, 0, []

    async def place_live_order(self, market: dict, side: str, entry_price: float,
                                shares: float, trade_size: float) -> Optional[dict]:
        """Place a FOK buy order on the CLOB. Returns fill info or None."""
        if self.paper or not self.client:
            return {"filled": True, "shares": shares, "price": entry_price,
                    "cost": trade_size, "order_id": "paper"}

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            up_tid, down_tid = self.get_token_ids(market)
            token_id = up_tid if side == "UP" else down_tid
            if not token_id:
                print(f"[LIVE] No token ID for BTC:{side}")
                return None

            fok_price = min(round(entry_price + FOK_SLIPPAGE, 2), 0.99)
            order_args = OrderArgs(
                price=fok_price, size=shares, side=BUY, token_id=token_id,
            )
            print(f"[LIVE] FOK {side} {shares}sh @ limit ${fok_price:.2f} "
                  f"(ask=${entry_price:.2f} +{FOK_SLIPPAGE:.0%} slip)")
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
                        self._api_consecutive_errors = 0
                        return {
                            "filled": True, "shares": size_matched,
                            "price": actual_price, "cost": actual_cost,
                            "order_id": order_id,
                        }
                    else:
                        print(f"[LIVE] NOT FILLED: matched={size_matched}")
                        return None
                except Exception as e:
                    print(f"[LIVE] Fill check error: {e} — assuming filled (FOK success)")
                    return {
                        "filled": True, "shares": shares, "price": entry_price,
                        "cost": trade_size, "order_id": order_id,
                    }
            return None
        except Exception as e:
            self._api_consecutive_errors += 1
            backoff = min(API_BACKOFF_BASE ** self._api_consecutive_errors, API_BACKOFF_MAX)
            print(f"[BACKOFF] API error #{self._api_consecutive_errors}: {e} — {backoff:.0f}s")
            await asyncio.sleep(backoff)
            return None

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
        # V3.0: Circuit breaker
        if time.time() < self._circuit_halt_until:
            return

        if not self._warmup_ready:
            elapsed = time.time() - self._warmup_start
            if elapsed < WARMUP_SECONDS:
                return
            self._warmup_ready = True
            print(f"[WARMUP] Ready after {elapsed:.0f}s")

        # V3.2: Skip dead hours
        utc_hour = datetime.now(timezone.utc).hour
        if utc_hour in SKIP_HOURS_UTC:
            return

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
        """Enter a trade based on whale consensus (paper or live)."""
        cid = market.get("conditionId", "")
        if not cid:
            return

        # Mark as attempted so we don't re-enter
        self.attempted_cids.add(cid)

        # V3.0: Get price estimate early for Kelly sizing
        live_count = sum(1 for t in self.resolved if t.get("live"))
        up_price, down_price = self.get_prices(market)
        est_price = (up_price if direction == "UP" else down_price) if up_price else 0.50
        trade_size = get_trade_size(self.stats["wins"], self.stats["losses"], live_count,
                                     est_price, self.stats["pnl"])
        if trade_size <= 0 and self.stats["wins"] + self.stats["losses"] >= KELLY_MIN_TRADES:
            print(f"[KELLY-SKIP] No edge — Kelly says don't trade")
            return

        # V3.0: SynthData consensus
        synth = await query_synthdata("BTC")
        if synth:
            s_edge = synth["edge_up"] if direction == "UP" else synth["edge_down"]
            if s_edge >= SYNTH_EDGE_THRESHOLD:
                trade_size = round(trade_size * SYNTH_SIZE_BOOST, 2)
                print(f"  [SYNTH] Agrees: edge={s_edge:+.1%} → ${trade_size:.2f}")
            elif s_edge <= -SYNTH_EDGE_THRESHOLD:
                trade_size = round(trade_size * SYNTH_SIZE_REDUCE, 2)
                print(f"  [SYNTH] Disagrees: edge={s_edge:+.1%} → ${trade_size:.2f}")

        # V3.1: Graduated circuit breaker — half-size
        if (self._consecutive_losses >= CIRCUIT_BREAKER_HALFSIZE or
                self._recovery_trades < CIRCUIT_BREAKER_RECOVERY):
            trade_size = round(trade_size * 0.5, 2)
            print(f"  [HALFSIZE] ${trade_size:.2f}")

        # V2.1: DOWN-bias — UP trades get half size (CSV: DOWN 12/12 100%, UP 3/6 50%)
        if direction == "UP":
            trade_size = round(trade_size * UP_SIZE_MULT, 2)
            print(f"  [UP-BIAS] Reduced size: ${trade_size:.2f} (50% for UP trades)")

        if self.paper:
            # PAPER MODE: use gamma API prices + spread offset
            up_price, down_price = self.get_prices(market)
            if up_price is None or down_price is None:
                print(f"[SKIP] No prices for {cid[:12]}")
                return
            entry_price = up_price if direction == "UP" else down_price
            entry_price = round(entry_price + SPREAD_OFFSET, 4)

            if entry_price < MIN_ENTRY_PRICE or entry_price > MAX_ENTRY_PRICE:
                print(f"[SKIP] Price ${entry_price:.2f} out of range "
                      f"[${MIN_ENTRY_PRICE}-${MAX_ENTRY_PRICE}]")
                return

            # V2.1: UP trades need higher minimum entry
            if direction == "UP" and entry_price < MIN_ENTRY_UP:
                print(f"[SKIP] UP too cheap | ${entry_price:.2f} < ${MIN_ENTRY_UP} min for UP")
                return

            shares = trade_size / entry_price
            actual_cost = round(shares * entry_price, 2)
            order_id = "paper"
        else:
            # LIVE MODE: use CLOB orderbook for real pricing + FOK execution
            up_tid, down_tid = self.get_token_ids(market)
            target_tid = up_tid if direction == "UP" else down_tid
            if not target_tid:
                print(f"[SKIP] No token ID for {direction}")
                return

            sweep_limit = min(MAX_ENTRY_PRICE + FOK_SLIPPAGE, 0.99)
            best_ask, avail_shares, levels = await self.get_orderbook_depth(target_tid, sweep_limit)

            if best_ask is None:
                print(f"[SKIP] No CLOB asks for {direction}")
                return

            entry_price = best_ask
            if entry_price < MIN_ENTRY_PRICE or entry_price > MAX_ENTRY_PRICE:
                print(f"[SKIP] CLOB price ${entry_price:.2f} out of range "
                      f"[${MIN_ENTRY_PRICE}-${MAX_ENTRY_PRICE}]")
                return

            # V2.1: UP trades need higher minimum entry
            if direction == "UP" and entry_price < MIN_ENTRY_UP:
                print(f"[SKIP] UP too cheap | ${entry_price:.2f} < ${MIN_ENTRY_UP} min for UP")
                return

            desired_shares = math.floor(trade_size / entry_price)
            if desired_shares < 1:
                desired_shares = 1
            if int(avail_shares) < MIN_SHARES:
                print(f"[SKIP] Too thin | only {int(avail_shares)}sh available (need {MIN_SHARES})")
                return

            shares = min(desired_shares, int(avail_shares))
            actual_cost = round(shares * entry_price, 2)

            depth_str = " | ".join(f"${p:.2f}x{int(s)}" for p, s in levels[:4])
            print(f"[CLOB] {direction} | entry=${entry_price:.2f} | "
                  f"depth={int(avail_shares)}sh [{depth_str}]")

            fill = await self.place_live_order(market, direction, entry_price, shares, actual_cost)
            if fill is None:
                print(f"[SKIP] Order not filled | {direction} @ ${entry_price:.2f}")
                return

            shares = fill["shares"]
            entry_price = fill["price"]
            actual_cost = fill["cost"]
            order_id = fill.get("order_id", "")

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
            "order_id": order_id,
            "live": not self.paper,
        }

        mode = "LIVE" if not self.paper else "PAPER"
        print(f"[{mode}|ENTRY] {direction} ${entry_price:.2f} ${actual_cost:.2f} "
              f"({shares:.1f}sh) | {len(voters)} whales: "
              f"{', '.join(whale_names)} | whale_cost=${total_whale_cost:.0f} | "
              f"time={time_left:.1f}m | tier=${trade_size:.2f} | "
              f"{market.get('question', '')[:60]}")

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

            if won:
                if self._consecutive_losses >= CIRCUIT_BREAKER_HALFSIZE:
                    self._recovery_trades = 1
                    print(f"[RECOVERY] Win after {self._consecutive_losses} losses → half-size for {CIRCUIT_BREAKER_RECOVERY} trades")
                else:
                    self._recovery_trades = CIRCUIT_BREAKER_RECOVERY
                self._consecutive_losses = 0
            else:
                self._consecutive_losses += 1
                self._recovery_trades = 0
                if self._consecutive_losses >= CIRCUIT_BREAKER_LOSSES:
                    self._circuit_halt_until = time.time() + CIRCUIT_BREAKER_COOLDOWN
                    print(f"[CIRCUIT BREAKER] {self._consecutive_losses} losses → HALTED {CIRCUIT_BREAKER_COOLDOWN // 60}m")
                elif self._consecutive_losses >= CIRCUIT_BREAKER_HALFSIZE:
                    print(f"[HALFSIZE] {self._consecutive_losses} consecutive losses → half-size mode")

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
        mode = "LIVE" if not self.paper else "PAPER"
        live_count = sum(1 for t in self.resolved if t.get("live"))
        tier = get_trade_size(self.stats["wins"], self.stats["losses"], live_count)
        # Banner
        print("=" * 70)
        print(f"  WHALE CONSENSUS 15M {mode} TRADER V3.2 — DOWN-BIAS")
        print("=" * 70)
        print(f"  V3.2: BoneReader removed (57%WR->79% without), min entry $0.50, skip dead hours")
        print(f"    - UP size: {UP_SIZE_MULT:.0%} (${tier*UP_SIZE_MULT:.2f}) | DOWN: full (${tier:.2f})")
        print(f"    - Entry range: ${MIN_ENTRY_PRICE:.2f}-${MAX_ENTRY_PRICE:.2f} | Skip UTC: {SKIP_HOURS_UTC}")
        print(f"  Strategy: Follow 3+ whale agreement on BTC 15M Up/Down direction")
        print(f"  Whales tracked: {len(CONSENSUS_WHALES)} top performers ($97K-$885K PnL each)")
        print(f"  Consensus: {CONSENSUS_MIN}+ whales agree = ENTER | Split = SKIP")
        print(f"  Entry window: {TIME_WINDOW_MIN:.0f}-{TIME_WINDOW_MAX:.0f} min before close")
        print(f"  Size: ${tier:.2f}/trade (PROBATION $2.50 -> PROMOTED $5.00)")
        print(f"  Max concurrent: {MAX_CONCURRENT} | Poll: {POLL_INTERVAL}s")
        print(f"  {'FOK orders on CLOB' if not self.paper else 'Paper fills (spread offset)'}")
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
    paper = "--live" not in sys.argv
    if paper:
        print("[MODE] PAPER — pass --live for real trading")
    else:
        print("[MODE] *** LIVE TRADING — REAL MONEY ***")

    lock = acquire_pid_lock("whale_consensus_15m")
    try:
        trader = WhaleConsensus15MTrader(paper=paper)
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        print("\n[EXIT] Whale Consensus 15M stopped.")
    finally:
        release_pid_lock("whale_consensus_15m")
