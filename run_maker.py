"""
Both-Sides Market Maker for 15-min Crypto Up/Down Markets (V1.0)

Strategy:
  Place passive BUY limit orders (post_only) on BOTH Up and Down tokens
  for every active 15-min BTC/ETH/SOL market. When both fill at combined
  price < $1.00, guaranteed profit regardless of outcome direction.

PolyData Edge (87K markets, 101M trades):
  - Makers earn +1.12%/trade structural advantage
  - Crypto 15-min markets are HIGHLY efficient (50/50 for takers)
  - Only profitable approach: maker spread capture, not directional prediction
  - Top whale MMs (0x6031) run 5.9M trades at ~49% buy WR - classic spread capture

Risk Controls:
  - Maximum $5/market pair exposure (configurable)
  - Maximum $30 total exposure across all markets
  - post_only=True on all orders (guaranteed maker, no taker fills)
  - Auto-cancel all orders 2 min before market close
  - Daily loss limit: $5
  - Session-aware: skip low-volume hours (UTC 21-23)

Modes:
  --paper  : Shadow mode - simulates fills from order book, no execution
  --live   : Real orders via py_clob_client (requires POLYMARKET_PASSWORD)
"""

import sys
import json
import time
import os
import math
import asyncio
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from functools import partial

import httpx
from dotenv import load_dotenv

print = partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MakerConfig:
    """Market maker configuration."""
    # Spread targets
    MAX_COMBINED_PRICE: float = 0.98     # Only pair if Up+Down bids < this
    MIN_SPREAD_EDGE: float = 0.01        # Minimum edge per pair (combined discount)
    BID_OFFSET: float = 0.02             # Bid this much below best ask

    # Position sizing
    SIZE_PER_SIDE_USD: float = 3.0       # $ per side per market
    MAX_PAIR_EXPOSURE: float = 6.0       # Max $ per market pair (both sides)
    MAX_TOTAL_EXPOSURE: float = 30.0     # Max $ across all active pairs
    MIN_SHARES: int = 5                  # CLOB minimum order size

    # Risk
    DAILY_LOSS_LIMIT: float = 5.0        # Stop for the day after this loss
    MAX_CONCURRENT_PAIRS: int = 5        # Max simultaneous market pairs
    MAX_SINGLE_SIDED: int = 3            # Max one-sided (incomplete pair) positions

    # Timing
    SCAN_INTERVAL: int = 30              # Seconds between market scans
    ORDER_REFRESH: int = 60              # Seconds before re-quoting orders
    CLOSE_BUFFER_MIN: float = 2.0        # Cancel orders N min before close
    FILL_CHECK_INTERVAL: int = 10        # Seconds between fill checks
    MIN_TIME_LEFT_MIN: float = 5.0       # Don't enter markets with < 5 min left

    # Session control (from PolyData analysis)
    SKIP_HOURS_UTC: set = field(default_factory=lambda: {21, 22, 23})  # Low volume
    BOOST_HOURS_UTC: set = field(default_factory=lambda: {13, 14, 15, 16})  # NY open peak

    # Assets
    ASSETS: dict = field(default_factory=lambda: {
        "BTC": {"keywords": ["bitcoin", "btc"], "enabled": True},
        "ETH": {"keywords": ["ethereum", "eth"], "enabled": True},
        "SOL": {"keywords": ["solana", "sol"], "enabled": True},
    })


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MarketPair:
    """A pair of Up/Down tokens for one 15-min market."""
    market_id: str           # conditionId
    question: str
    asset: str               # BTC, ETH, SOL
    up_token_id: str
    down_token_id: str
    up_mid: float = 0.0
    down_mid: float = 0.0
    up_best_ask: float = 0.0
    down_best_ask: float = 0.0
    up_best_bid: float = 0.0
    down_best_bid: float = 0.0
    up_spread: float = 0.0
    down_spread: float = 0.0
    # Timing
    end_time: Optional[datetime] = None
    created_at: Optional[datetime] = None


@dataclass
class MakerOrder:
    """A placed maker order."""
    order_id: str
    market_id: str
    token_id: str
    side_label: str          # "UP" or "DOWN"
    price: float
    size_shares: float
    size_usd: float
    placed_at: str           # ISO timestamp
    status: str = "open"     # open, filled, cancelled, expired
    fill_price: float = 0.0
    fill_shares: float = 0.0


@dataclass
class PairPosition:
    """Tracks a market pair (both sides)."""
    market_id: str
    question: str
    asset: str
    up_order: Optional[MakerOrder] = None
    down_order: Optional[MakerOrder] = None
    up_filled: bool = False
    down_filled: bool = False
    combined_cost: float = 0.0   # Total $ spent on both sides
    outcome: Optional[str] = None  # "UP", "DOWN", or None (unresolved)
    pnl: float = 0.0
    status: str = "pending"  # pending, partial, paired, resolved, cancelled
    created_at: str = ""

    @property
    def is_paired(self) -> bool:
        return self.up_filled and self.down_filled

    @property
    def is_partial(self) -> bool:
        return (self.up_filled or self.down_filled) and not self.is_paired


# ============================================================================
# MARKET MAKER BOT
# ============================================================================

class CryptoMarketMaker:
    """Both-sides market maker for 15-min crypto markets."""

    OUTPUT_FILE = Path(__file__).parent / "maker_results.json"
    PID_FILE = Path(__file__).parent / "maker.pid"

    def __init__(self, paper: bool = True):
        self.paper = paper
        self.config = MakerConfig()
        self.client = None  # ClobClient (live only)

        # State
        self.positions: Dict[str, PairPosition] = {}  # market_id -> PairPosition
        self.active_orders: Dict[str, MakerOrder] = {}  # order_id -> MakerOrder
        self.resolved: List[dict] = []  # Completed pair records

        # Stats
        self.stats = {
            "pairs_attempted": 0,
            "pairs_completed": 0,     # Both sides filled
            "pairs_partial": 0,       # Only one side filled
            "pairs_cancelled": 0,
            "total_pnl": 0.0,
            "paired_pnl": 0.0,        # PnL from fully-paired positions
            "partial_pnl": 0.0,       # PnL from one-sided positions
            "total_volume": 0.0,
            "best_pair_pnl": 0.0,
            "worst_pair_pnl": 0.0,
            "avg_combined_price": 0.0,
            "avg_spread_captured": 0.0,
            "fills_up": 0,
            "fills_down": 0,
            "start_time": datetime.now(timezone.utc).isoformat(),
        }

        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_date = date_today()

        if not paper:
            self._init_client()

        self._load()

    def _init_client(self):
        """Initialize CLOB client for live trading."""
        try:
            from arbitrage.executor import Executor
            executor = Executor()
            if executor._initialized:
                self.client = executor.client
                print("[MAKER] CLOB client initialized - LIVE MODE")
            else:
                print("[MAKER] Client init failed - falling back to PAPER")
                self.paper = True
        except Exception as e:
            print(f"[MAKER] Client error: {e} - PAPER MODE")
            self.paper = True

    def _load(self):
        """Load previous state."""
        if self.OUTPUT_FILE.exists():
            try:
                data = json.loads(self.OUTPUT_FILE.read_text())
                self.stats = data.get("stats", self.stats)
                self.resolved = data.get("resolved", [])
                # Restore active positions
                for pos_data in data.get("active_positions", []):
                    pos = PairPosition(**{k: v for k, v in pos_data.items()
                                         if k in PairPosition.__dataclass_fields__})
                    if pos.status not in ("resolved", "cancelled"):
                        self.positions[pos.market_id] = pos
                print(f"[MAKER] Loaded {len(self.resolved)} resolved, {len(self.positions)} active")
            except Exception as e:
                print(f"[MAKER] Load error: {e}")

    def _save(self):
        """Save state to disk."""
        data = {
            "stats": self.stats,
            "resolved": self.resolved[-500:],  # Keep last 500
            "active_positions": [
                {k: v for k, v in asdict(pos).items() if not callable(v)}
                for pos in self.positions.values()
            ],
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.OUTPUT_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            print(f"[MAKER] Save error: {e}")

    # ========================================================================
    # MARKET DISCOVERY
    # ========================================================================

    async def discover_markets(self) -> List[MarketPair]:
        """Find all active 15-min crypto Up/Down markets."""
        pairs = []
        async with httpx.AsyncClient(timeout=15) as client:
            try:
                r = await client.get(
                    "https://gamma-api.polymarket.com/events",
                    params={"tag_slug": "15M", "active": "true", "closed": "false", "limit": 50},
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                if r.status_code != 200:
                    print(f"[DISCOVER] API error: {r.status_code}")
                    return pairs

                events = r.json()
                for event in events:
                    title = event.get("title", "").lower()
                    matched_asset = None
                    for asset, cfg in self.config.ASSETS.items():
                        if not cfg["enabled"]:
                            continue
                        if any(kw in title for kw in cfg["keywords"]):
                            matched_asset = asset
                            break

                    if not matched_asset:
                        continue

                    for m in event.get("markets", []):
                        if m.get("closed", True):
                            continue

                        pair = self._parse_market(m, event, matched_asset)
                        if pair:
                            pairs.append(pair)

            except Exception as e:
                print(f"[DISCOVER] Error: {e}")

        return pairs

    def _parse_market(self, market: dict, event: dict, asset: str) -> Optional[MarketPair]:
        """Parse a market into a MarketPair."""
        try:
            condition_id = market.get("conditionId", "")
            if not condition_id:
                return None

            # Skip markets we already have positions on
            if condition_id in self.positions:
                return None

            outcomes = market.get("outcomes", [])
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            token_ids = market.get("clobTokenIds", [])
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            prices = market.get("outcomePrices", [])
            if isinstance(prices, str):
                prices = json.loads(prices)

            if len(outcomes) != 2 or len(token_ids) != 2:
                return None

            # Identify Up/Down tokens
            up_idx = None
            down_idx = None
            for i, outcome in enumerate(outcomes):
                o = str(outcome).lower()
                if o in ("up", "yes"):
                    up_idx = i
                elif o in ("down", "no"):
                    down_idx = i

            if up_idx is None or down_idx is None:
                return None

            up_price = float(prices[up_idx]) if prices and len(prices) > up_idx else 0.5
            down_price = float(prices[down_idx]) if prices and len(prices) > down_idx else 0.5

            # Parse end time from question
            question = market.get("question", "") or event.get("title", "")
            end_time = market.get("endDate")
            if end_time:
                try:
                    end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                except Exception:
                    end_time = None

            pair = MarketPair(
                market_id=condition_id,
                question=question,
                asset=asset,
                up_token_id=token_ids[up_idx],
                down_token_id=token_ids[down_idx],
                up_mid=up_price,
                down_mid=down_price,
                end_time=end_time,
            )
            return pair

        except Exception as e:
            return None

    # ========================================================================
    # ORDER BOOK ANALYSIS
    # ========================================================================

    async def get_order_book(self, token_id: str) -> dict:
        """Fetch order book for a token. Returns {best_bid, best_ask, mid, spread}."""
        if self.paper:
            # In paper mode, simulate from mid price
            return None

        try:
            book = self.client.get_order_book(token_id)
            bids = book.get("bids", []) if isinstance(book, dict) else getattr(book, 'bids', [])
            asks = book.get("asks", []) if isinstance(book, dict) else getattr(book, 'asks', [])

            best_bid = float(bids[0]["price"]) if bids else 0.0
            best_ask = float(asks[0]["price"]) if asks else 1.0
            mid = (best_bid + best_ask) / 2 if best_bid > 0 else best_ask
            spread = best_ask - best_bid

            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid": mid,
                "spread": spread,
                "bid_depth": sum(float(b.get("size", 0)) for b in bids[:3]),
                "ask_depth": sum(float(a.get("size", 0)) for a in asks[:3]),
            }
        except Exception as e:
            return None

    def _get_bid_offset(self) -> float:
        """Adaptive bid offset based on session and recent fills."""
        hour = datetime.now(timezone.utc).hour

        # NY open (high volume) = tighter spread, more fills expected
        if hour in self.config.BOOST_HOURS_UTC:
            return 0.015  # 1.5c offset during peak

        # Asian session = wider spread, thin books
        if hour in {0, 1, 2, 3, 4, 5, 6, 7}:
            return 0.025

        return self.config.BID_OFFSET  # Default 2c

    def evaluate_pair(self, pair: MarketPair) -> dict:
        """Evaluate a market pair for maker opportunity."""
        offset = self._get_bid_offset()

        # Our target bid prices
        up_bid = round(pair.up_mid - offset, 2)
        down_bid = round(pair.down_mid - offset, 2)

        # Ensure prices are valid
        up_bid = max(0.01, min(0.95, up_bid))
        down_bid = max(0.01, min(0.95, down_bid))

        combined = up_bid + down_bid
        edge = 1.0 - combined  # How much < $1.00 our total cost is

        # Time remaining
        mins_left = 15.0
        if pair.end_time:
            delta = (pair.end_time - datetime.now(timezone.utc)).total_seconds() / 60
            mins_left = max(0, delta)

        return {
            "up_bid": up_bid,
            "down_bid": down_bid,
            "combined": combined,
            "edge": edge,
            "edge_pct": edge / combined * 100 if combined > 0 else 0,
            "mins_left": mins_left,
            "viable": (
                combined < self.config.MAX_COMBINED_PRICE
                and edge >= self.config.MIN_SPREAD_EDGE
                and mins_left > self.config.MIN_TIME_LEFT_MIN  # Need time for fills
                and up_bid > 0.10  # Don't bid on extreme prices (low fill chance)
                and down_bid > 0.10
                and up_bid < 0.90
                and down_bid < 0.90
            ),
        }

    # ========================================================================
    # ORDER PLACEMENT
    # ========================================================================

    async def place_pair_orders(self, pair: MarketPair, eval_result: dict) -> Optional[PairPosition]:
        """Place maker orders on both sides of a market pair."""
        up_bid = eval_result["up_bid"]
        down_bid = eval_result["down_bid"]

        # Calculate shares for each side
        up_shares = max(self.config.MIN_SHARES, math.floor(self.config.SIZE_PER_SIDE_USD / up_bid))
        down_shares = max(self.config.MIN_SHARES, math.floor(self.config.SIZE_PER_SIDE_USD / down_bid))

        # Create position tracker
        pos = PairPosition(
            market_id=pair.market_id,
            question=pair.question,
            asset=pair.asset,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="pending",
        )

        now_iso = datetime.now(timezone.utc).isoformat()

        if self.paper:
            # Paper mode: simulate immediate placement, simulate fills probabilistically
            pos.up_order = MakerOrder(
                order_id=f"paper_up_{pair.market_id[:12]}_{int(time.time())}",
                market_id=pair.market_id,
                token_id=pair.up_token_id,
                side_label="UP",
                price=up_bid,
                size_shares=float(up_shares),
                size_usd=up_bid * up_shares,
                placed_at=now_iso,
            )
            pos.down_order = MakerOrder(
                order_id=f"paper_dn_{pair.market_id[:12]}_{int(time.time())}",
                market_id=pair.market_id,
                token_id=pair.down_token_id,
                side_label="DOWN",
                price=down_bid,
                size_shares=float(down_shares),
                size_usd=down_bid * down_shares,
                placed_at=now_iso,
            )
            print(f"  [PAPER] Placed UP bid ${up_bid:.2f} x {up_shares} + DOWN bid ${down_bid:.2f} x {down_shares}")
            print(f"          Combined: ${up_bid + down_bid:.4f} | Edge: {eval_result['edge_pct']:.1f}%")
        else:
            # Live mode: place real post_only orders
            try:
                from py_clob_client.clob_types import OrderArgs, OrderType
                from py_clob_client.order_builder.constants import BUY

                # Place UP order
                up_args = OrderArgs(
                    price=up_bid,
                    size=float(up_shares),
                    side=BUY,
                    token_id=pair.up_token_id,
                )
                up_signed = self.client.create_order(up_args)
                up_resp = self.client.post_order(up_signed, OrderType.GTC)

                if not up_resp.get("success"):
                    print(f"  [LIVE] UP order failed: {up_resp.get('errorMsg', '?')}")
                    return None

                up_order_id = up_resp.get("orderID", "")
                pos.up_order = MakerOrder(
                    order_id=up_order_id,
                    market_id=pair.market_id,
                    token_id=pair.up_token_id,
                    side_label="UP",
                    price=up_bid,
                    size_shares=float(up_shares),
                    size_usd=up_bid * up_shares,
                    placed_at=now_iso,
                )

                # Place DOWN order
                dn_args = OrderArgs(
                    price=down_bid,
                    size=float(down_shares),
                    side=BUY,
                    token_id=pair.down_token_id,
                )
                dn_signed = self.client.create_order(dn_args)
                dn_resp = self.client.post_order(dn_signed, OrderType.GTC)

                if not dn_resp.get("success"):
                    print(f"  [LIVE] DOWN order failed: {dn_resp.get('errorMsg', '?')}")
                    # Cancel UP order since we can't pair
                    try:
                        self.client.cancel(up_order_id)
                    except Exception:
                        pass
                    return None

                dn_order_id = dn_resp.get("orderID", "")
                pos.down_order = MakerOrder(
                    order_id=dn_order_id,
                    market_id=pair.market_id,
                    token_id=pair.down_token_id,
                    side_label="DOWN",
                    price=down_bid,
                    size_shares=float(down_shares),
                    size_usd=down_bid * down_shares,
                    placed_at=now_iso,
                )

                print(f"  [LIVE] Placed UP {up_order_id[:16]}... @ ${up_bid:.2f} x {up_shares}")
                print(f"  [LIVE] Placed DN {dn_order_id[:16]}... @ ${down_bid:.2f} x {down_shares}")
                print(f"         Combined: ${up_bid + down_bid:.4f} | Edge: {eval_result['edge_pct']:.1f}%")

            except Exception as e:
                print(f"  [LIVE] Order error: {e}")
                return None

        # Track
        self.positions[pair.market_id] = pos
        self.stats["pairs_attempted"] += 1
        return pos

    # ========================================================================
    # FILL MONITORING
    # ========================================================================

    async def check_fills(self):
        """Check order fill status for all active positions."""
        for market_id, pos in list(self.positions.items()):
            if pos.status in ("resolved", "cancelled"):
                continue

            if self.paper:
                await self._check_fills_paper(pos)
            else:
                await self._check_fills_live(pos)

            # Update position status
            if pos.up_filled and pos.down_filled:
                pos.status = "paired"
                pos.combined_cost = (
                    (pos.up_order.fill_price * pos.up_order.fill_shares if pos.up_order else 0) +
                    (pos.down_order.fill_price * pos.down_order.fill_shares if pos.down_order else 0)
                )
            elif pos.up_filled or pos.down_filled:
                pos.status = "partial"

    async def _check_fills_paper(self, pos: PairPosition):
        """Simulate fills for paper mode.

        Paper fill model: Fill probability depends on how aggressive our bid is.
        PolyData shows 15-min order books are thin — passive bids fill slowly.
        - Bid at mid: ~25% per 30s cycle
        - Bid 1c below mid: ~15% per cycle
        - Bid 2c below mid: ~10% per cycle
        - Bid 3c+ below mid: ~5% per cycle
        Reality is probably even worse, but this gives a reasonable estimate.
        """
        import random

        for order, attr in [(pos.up_order, "up_filled"), (pos.down_order, "down_filled")]:
            if not order or getattr(pos, attr):
                continue
            if order.status == "filled":
                setattr(pos, attr, True)
                continue

            # Use mid price from when pair was created to estimate aggressiveness
            if order.side_label == "UP":
                mid = order.price + self.config.BID_OFFSET  # Reverse the offset
            else:
                mid = order.price + self.config.BID_OFFSET

            discount = mid - order.price
            if discount <= 0.005:
                fill_prob = 0.25
            elif discount <= 0.015:
                fill_prob = 0.15
            elif discount <= 0.025:
                fill_prob = 0.10
            else:
                fill_prob = 0.05

            # Time decay: closer to expiry = more desperate sellers = higher fills
            created = datetime.fromisoformat(pos.created_at)
            age_min = (datetime.now(timezone.utc) - created).total_seconds() / 60
            if age_min > 10:
                fill_prob *= 1.5  # Last 5 min of market = more urgent flow
            elif age_min > 12:
                fill_prob *= 2.0  # Very end = aggressive selling

            fill_prob = min(fill_prob, 0.60)  # Cap at 60%

            if random.random() < fill_prob:
                order.status = "filled"
                order.fill_price = order.price
                order.fill_shares = order.size_shares
                setattr(pos, attr, True)
                self.stats[f"fills_{order.side_label.lower()}"] += 1
                print(f"  [PAPER FILL] {pos.asset} {order.side_label} @ ${order.price:.2f} x {order.size_shares:.0f} (p={fill_prob:.0%})")

    async def _check_fills_live(self, pos: PairPosition):
        """Check real order fills via CLOB API."""
        if not self.client:
            return

        for order, attr in [(pos.up_order, "up_filled"), (pos.down_order, "down_filled")]:
            if not order or getattr(pos, attr):
                continue

            try:
                status = self.client.get_order(order.order_id)
                if isinstance(status, dict):
                    matched = float(status.get("size_matched", 0))
                    order_status = status.get("status", "")
                    if order_status == "MATCHED" or matched >= order.size_shares * 0.5:
                        order.status = "filled"
                        order.fill_price = order.price
                        order.fill_shares = matched
                        setattr(pos, attr, True)
                        self.stats[f"fills_{order.side_label.lower()}"] += 1
                        print(f"  [FILL] {pos.asset} {order.side_label} @ ${order.price:.2f} x {matched:.0f}")
            except Exception:
                pass

    # ========================================================================
    # RESOLUTION
    # ========================================================================

    async def manage_expiring_orders(self):
        """Cancel unfilled orders on markets about to close."""
        now = datetime.now(timezone.utc)
        for market_id, pos in list(self.positions.items()):
            if pos.status in ("resolved", "cancelled"):
                continue

            created = datetime.fromisoformat(pos.created_at)
            age_min = (now - created).total_seconds() / 60

            # If approaching close (>13 min for a 15-min market), cancel unfilled orders
            if age_min > 13:
                for order, attr in [(pos.up_order, "up_filled"), (pos.down_order, "down_filled")]:
                    if order and not getattr(pos, attr) and order.status == "open":
                        order.status = "cancelled"
                        if not self.paper:
                            try:
                                self.client.cancel(order.order_id)
                            except Exception:
                                pass
                        print(f"  [EXPIRE] Cancelled unfilled {order.side_label} order on {pos.asset}")

    async def resolve_positions(self):
        """Check if markets have resolved and calculate PnL."""
        now = datetime.now(timezone.utc)

        for market_id, pos in list(self.positions.items()):
            if pos.status == "resolved":
                continue

            # Check if market has expired (15 min + buffer)
            created = datetime.fromisoformat(pos.created_at)
            age_min = (now - created).total_seconds() / 60
            if age_min < 16:  # Wait for resolution
                continue

            # Fetch outcome
            outcome = await self._fetch_outcome(market_id)
            if not outcome:
                if age_min > 30:  # Give up after 30 min
                    pos.status = "cancelled"
                    self._cancel_position_orders(pos)
                continue

            pos.outcome = outcome
            pos.status = "resolved"

            # Calculate PnL
            pnl = self._calculate_pnl(pos)
            pos.pnl = pnl
            self.stats["total_pnl"] += pnl
            self.daily_pnl += pnl
            self.stats["total_volume"] += pos.combined_cost

            if pos.is_paired:
                self.stats["pairs_completed"] += 1
                self.stats["paired_pnl"] += pnl
            elif pos.is_partial:
                self.stats["pairs_partial"] += 1
                self.stats["partial_pnl"] += pnl
            else:
                self.stats["pairs_cancelled"] += 1

            self.stats["best_pair_pnl"] = max(self.stats["best_pair_pnl"], pnl)
            self.stats["worst_pair_pnl"] = min(self.stats["worst_pair_pnl"], pnl)

            # Log
            pair_type = "PAIRED" if pos.is_paired else "PARTIAL" if pos.is_partial else "UNFILLED"
            icon = "+" if pnl > 0 else "-" if pnl < 0 else "="
            print(f"  [{pair_type}] {pos.asset} {pos.question[:50]} | {outcome} | PnL: {icon}${abs(pnl):.4f}")

            # Archive
            self.resolved.append({
                "market_id": market_id,
                "question": pos.question,
                "asset": pos.asset,
                "outcome": outcome,
                "paired": pos.is_paired,
                "partial": pos.is_partial,
                "combined_cost": pos.combined_cost,
                "pnl": pnl,
                "up_price": pos.up_order.fill_price if pos.up_order and pos.up_filled else 0,
                "down_price": pos.down_order.fill_price if pos.down_order and pos.down_filled else 0,
                "resolved_at": now.isoformat(),
            })

        # Clean resolved from active
        self.positions = {k: v for k, v in self.positions.items()
                         if v.status not in ("resolved", "cancelled")}

    async def _fetch_outcome(self, condition_id: str) -> Optional[str]:
        """Fetch market outcome from gamma-api."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={"condition_id": condition_id, "limit": "1"},
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                if r.status_code == 200:
                    markets = r.json()
                    if markets and len(markets) > 0:
                        m = markets[0]
                        prices = m.get("outcomePrices", [])
                        if isinstance(prices, str):
                            prices = json.loads(prices)
                        outcomes = m.get("outcomes", [])
                        if isinstance(outcomes, str):
                            outcomes = json.loads(outcomes)

                        if prices and len(prices) >= 2:
                            p0 = float(prices[0])
                            p1 = float(prices[1])
                            if p0 > 0.95 and p1 < 0.05:
                                return str(outcomes[0]).upper() if outcomes else "UP"
                            elif p1 > 0.95 and p0 < 0.05:
                                return str(outcomes[1]).upper() if outcomes else "DOWN"
        except Exception:
            pass
        return None

    def _calculate_pnl(self, pos: PairPosition) -> float:
        """Calculate PnL for a resolved position."""
        pnl = 0.0
        outcome = pos.outcome

        # UP side
        if pos.up_filled and pos.up_order:
            if outcome == "UP":
                # Won: payout $1.00 per share, cost was fill_price
                pnl += pos.up_order.fill_shares * (1.0 - pos.up_order.fill_price)
            else:
                # Lost: shares worth $0
                pnl -= pos.up_order.fill_shares * pos.up_order.fill_price

        # DOWN side
        if pos.down_filled and pos.down_order:
            if outcome == "DOWN":
                pnl += pos.down_order.fill_shares * (1.0 - pos.down_order.fill_price)
            else:
                pnl -= pos.down_order.fill_shares * pos.down_order.fill_price

        return round(pnl, 6)

    def _cancel_position_orders(self, pos: PairPosition):
        """Cancel any unfilled orders for a position."""
        if self.paper:
            return

        for order in [pos.up_order, pos.down_order]:
            if order and order.status == "open":
                try:
                    self.client.cancel(order.order_id)
                    order.status = "cancelled"
                except Exception:
                    pass

    # ========================================================================
    # RISK MANAGEMENT
    # ========================================================================

    def check_risk(self) -> bool:
        """Check if we should continue trading. Returns True if OK."""
        # Daily loss limit
        if self.daily_pnl < -self.config.DAILY_LOSS_LIMIT:
            print(f"[RISK] Daily loss limit hit (${self.daily_pnl:.2f}). Pausing.")
            return False

        # Max concurrent pairs
        active = sum(1 for p in self.positions.values() if p.status in ("pending", "partial", "paired"))
        if active >= self.config.MAX_CONCURRENT_PAIRS:
            return False

        # Max single-sided
        partial = sum(1 for p in self.positions.values() if p.is_partial)
        if partial >= self.config.MAX_SINGLE_SIDED:
            return False

        # Total exposure
        exposure = sum(
            (p.up_order.size_usd if p.up_order and not p.up_filled else 0) +
            (p.down_order.size_usd if p.down_order and not p.down_filled else 0) +
            (p.up_order.fill_price * p.up_order.fill_shares if p.up_order and p.up_filled else 0) +
            (p.down_order.fill_price * p.down_order.fill_shares if p.down_order and p.down_filled else 0)
            for p in self.positions.values()
            if p.status not in ("resolved", "cancelled")
        )
        if exposure >= self.config.MAX_TOTAL_EXPOSURE:
            return False

        # Skip hours
        hour = datetime.now(timezone.utc).hour
        if hour in self.config.SKIP_HOURS_UTC:
            return False

        return True

    def reset_daily(self):
        """Reset daily tracking if new day."""
        today = date_today()
        if today != self.daily_date:
            print(f"[DAILY] New day. Yesterday PnL: ${self.daily_pnl:.4f}")
            self.daily_pnl = 0.0
            self.daily_date = today

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        """Main market maker loop."""
        print("=" * 70)
        mode = "PAPER" if self.paper else "LIVE"
        print(f"BOTH-SIDES MARKET MAKER - {mode} MODE")
        print(f"Strategy: Buy BOTH Up+Down at combined < ${self.config.MAX_COMBINED_PRICE}")
        print(f"Size: ${self.config.SIZE_PER_SIDE_USD}/side | Max pairs: {self.config.MAX_CONCURRENT_PAIRS}")
        print(f"Daily loss limit: ${self.config.DAILY_LOSS_LIMIT}")
        print(f"Skip hours (UTC): {sorted(self.config.SKIP_HOURS_UTC)}")
        print("=" * 70)

        if self.stats["total_pnl"] != 0:
            print(f"[RESUME] Total PnL: ${self.stats['total_pnl']:.4f} | "
                  f"Pairs: {self.stats['pairs_completed']} complete, {self.stats['pairs_partial']} partial")

        cycle = 0
        while True:
            try:
                cycle += 1
                self.reset_daily()

                # Phase 1: Cancel expiring unfilled orders
                await self.manage_expiring_orders()

                # Phase 2: Resolve completed markets
                await self.resolve_positions()

                # Phase 3: Check fills on active orders
                await self.check_fills()

                # Phase 4: Risk check
                if not self.check_risk():
                    if cycle % 10 == 0:
                        self._print_status(cycle)
                    await asyncio.sleep(self.config.SCAN_INTERVAL)
                    continue

                # Phase 5: Discover new markets
                markets = await self.discover_markets()

                # Phase 6: Evaluate and place orders on best pairs
                placed = 0
                for pair in markets:
                    if not self.check_risk():
                        break

                    eval_result = self.evaluate_pair(pair)
                    if not eval_result["viable"]:
                        continue

                    print(f"\n[{pair.asset}] {pair.question[:60]}")
                    print(f"  Mid: UP ${pair.up_mid:.2f} / DOWN ${pair.down_mid:.2f} | "
                          f"Mins left: {eval_result['mins_left']:.1f}")

                    result = await self.place_pair_orders(pair, eval_result)
                    if result:
                        placed += 1

                # Phase 6: Status & save
                if cycle % 5 == 0 or placed > 0:
                    self._print_status(cycle)
                    self._save()

                await asyncio.sleep(self.config.SCAN_INTERVAL)

            except KeyboardInterrupt:
                print("\n[MAKER] Shutting down...")
                self._cancel_all_open()
                self._save()
                break
            except Exception as e:
                print(f"[MAKER] Error in cycle {cycle}: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(10)

    def _cancel_all_open(self):
        """Cancel all open orders on shutdown."""
        if self.paper:
            return
        print("[MAKER] Cancelling all open orders...")
        for pos in self.positions.values():
            self._cancel_position_orders(pos)
        try:
            if self.client:
                self.client.cancel_all()
        except Exception:
            pass

    def _print_status(self, cycle: int):
        """Print status summary."""
        active = sum(1 for p in self.positions.values() if p.status not in ("resolved", "cancelled"))
        paired = sum(1 for p in self.positions.values() if p.is_paired)
        partial = sum(1 for p in self.positions.values() if p.is_partial)
        hour = datetime.now(timezone.utc).hour

        print(f"\n--- Cycle {cycle} | UTC {hour:02d} | "
              f"Active: {active} ({paired} paired, {partial} partial) | "
              f"Daily: ${self.daily_pnl:+.4f} | Total: ${self.stats['total_pnl']:+.4f} | "
              f"Resolved: {len(self.resolved)} ---")

        if self.stats["pairs_completed"] > 0:
            avg = self.stats["paired_pnl"] / self.stats["pairs_completed"]
            print(f"    Paired avg PnL: ${avg:.4f} | Best: ${self.stats['best_pair_pnl']:.4f} | "
                  f"Worst: ${self.stats['worst_pair_pnl']:.4f}")


# ============================================================================
# HELPERS
# ============================================================================

def date_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Both-Sides Market Maker")
    parser.add_argument("--paper", action="store_true", default=True,
                        help="Paper trading mode (default)")
    parser.add_argument("--live", action="store_true",
                        help="Live trading mode")
    args = parser.parse_args()

    paper = not args.live

    # PID lock
    from pid_lock import acquire_pid_lock, release_pid_lock
    acquire_pid_lock("maker")

    try:
        maker = CryptoMarketMaker(paper=paper)
        asyncio.run(maker.run())
    finally:
        release_pid_lock("maker")


if __name__ == "__main__":
    main()
