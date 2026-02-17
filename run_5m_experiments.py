"""
Whale Copy + Flow Experiments V2.0 — PAPER ONLY

Upgraded from V1.0:
  - 15 directional whales (tiered: T1 proven, T2 strong, T3 unproven)
  - 5M + 15M markets, BTC + ETH + SOL + XRP
  - Consensus filter: T1 solo = enter, 2+ agree = enter, solo = shadow
  - Per-whale win rate tracking (auto promote/demote by WR)
  - Concurrent whale polling (5x parallel)
  - Market makers EXCLUDED (gabagool22, likebot, Canine-Commandment, vague-sourdough)

Experiments:
  1. ORDER FLOW (5M only): Track mid-price drift, enter on momentum
  2. WHALE CONSENSUS: Copy when T1 solo or 2+ whales agree on direction
  3. WHALE SOLO (shadow): Track solo non-T1 signals for data collection

Usage:
  python run_5m_experiments.py
"""

import sys
import json
import time
import asyncio
import re
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from functools import partial as fn_partial
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import httpx

print = fn_partial(print, flush=True)
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# PID LOCK
# ============================================================================
from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# CONFIG
# ============================================================================
SCAN_INTERVAL = 15          # seconds between cycles
PAPER_SIZE = 3.0            # $3/trade
MIN_SHARES = 5              # CLOB minimum

# --- Experiment 1: Order Flow (5M BTC only) ---
FLOW_OBSERVATION_SEC = 75
FLOW_MIN_DRIFT = 0.06
FLOW_MAX_ENTRY = 0.60
FLOW_MIN_ENTRY = 0.35

# --- Experiment 2+3: Whale Copy (5M + 15M, multi-asset) ---
WHALE_POLL_SEC = 30
WHALE_MAX_ENTRY = 0.75
WHALE_MIN_ENTRY = 0.35
WHALE_POLL_CONCURRENCY = 5  # Max concurrent API calls

# Consensus rules
# - 1 Tier-1 whale = ENTER (proven track record: $100K+/mo)
# - 2+ whales agree (any tier) = ENTER
# - Solo Tier-2/3 = SHADOW only (log for data, don't paper-enter)
CONSENSUS_MIN_WHALES = 2

# --- Whale Registry (directional only, market makers excluded) ---
# Tier 1: Proven crypto candle kings ($100K+/mo, directional)
# Tier 2: Strong directional (solid PnL, active)
# Tier 3: Unproven/intermittent
# EXCLUDED: gabagool22 (50/50 MM), likebot (balanced MM),
#           Canine-Commandment (56% balanced MM), vague-sourdough (both-sides MM)
WHALES = {
    # --- TIER 1: Proven ($100K+/mo crypto PnL) ---
    "0x8dxd":            {"addr": "0x63ce342161250d705dc0b16df89036c8e5f9ba9a", "tier": 1},  # $627K/mo, Up-heavy
    "BoneReader":        {"addr": "0xd84c2b6d65dc596f49c7b6aadd6d74ca91e407b9", "tier": 1},  # $163K/mo, $2K+/trade, multi-asset
    "bratanbratishka":   {"addr": "0xcbb1a3174d9ac5a0f57f5b86808204b9382e7afb", "tier": 1},  # +$57K today, BTC 5m
    # --- TIER 2: Strong directional ---
    "k9Q2mX4L8A7ZP3R":  {"addr": "0xd0d6053c3c37e727402d84c14069780d360993aa", "tier": 2},  # $416K/mo, 78% Down, multi-asset
    "Bidou28old":        {"addr": "0x4460bf2c0aa59db412a6493c2c08970797b62970", "tier": 2},  # $88.5K, Down-heavy sniper
    "RetamzZ":           {"addr": "0x19f19dd8ee1f7e5f6ec666987e2963a65971a9c6", "tier": 2},  # $38K, 100% WR, massive bets
    "MuseumOfBees":      {"addr": "0x61276aba49117fd9299707d5d573652949d5c977", "tier": 2},  # $13.6K, 67% Up, BTC 5m
    "distinct-baguette": {"addr": "0xe00740bce98a594e26861838885ab310ec3b548c", "tier": 2},  # $132K/mo, XRP focused
    "Radiant-Spy":       {"addr": "0xe89ea370d21cfe36a27c4236f348a884c36b0600", "tier": 2},  # Active, pure Up, BTC hourly
    # --- TIER 3: Unproven/intermittent ---
    "Internal-Collection": {"addr": "0x1eda9a5cdaddd0a996c6f804c4697b1d2bdc9c47", "tier": 3},  # Tiny, pure Down
    "kindlydelta":       {"addr": "0x7780e9f103f370a7a0351217501d99da178503ed", "tier": 3},  # $120K, 137% edge, 9d inactive
    "Giving-Chorus":     {"addr": "0xd0bde12c58772999c61c2b8e0d31ba608c52a5d6", "tier": 3},  # $91K, selective high-conviction
    "late-to-tomorrow":  {"addr": "0x5924ca480d8b08cd5f3e5811fa378c4082475af6", "tier": 3},  # $25.7K grinder
    "qwqbw":             {"addr": "0x799c2267096b7523577d1d30564f914e649637f7", "tier": 3},  # $111K sniper, sports too
}

# Assets to discover
ASSETS_MAP = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "xrp": "XRP"}
TAG_SLUGS = ["5M", "15M"]

RESULTS_FILE = Path(__file__).parent / "5m_experiments.json"

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class MarketCandle:
    """A candle market (5M or 15M, any crypto asset)."""
    condition_id: str
    question: str
    end_time: Optional[datetime]
    up_token: str
    down_token: str
    market_num_id: str = ""
    duration_min: int = 5        # 5 or 15
    asset: str = "BTC"           # BTC, ETH, SOL, XRP
    # Book tracking
    first_seen: float = 0.0
    up_mids: List[float] = field(default_factory=list)
    down_mids: List[float] = field(default_factory=list)
    up_mid_initial: float = 0.0
    down_mid_initial: float = 0.0
    up_mid_latest: float = 0.0
    down_mid_latest: float = 0.0
    # Flow signal (5M BTC only)
    flow_direction: str = ""
    flow_drift: float = 0.0
    flow_entered: bool = False
    # Whale signals (V2.0: multi-whale, consensus-based)
    whale_signals: Dict[str, str] = field(default_factory=dict)  # whale_name -> "UP"/"DOWN"
    whale_entered: bool = False      # Consensus entry done
    whale_solo_entered: bool = False  # Shadow solo entry done
    # Resolution
    resolved: bool = False
    outcome: str = ""


@dataclass
class PaperTrade:
    """A paper trade for one experiment."""
    experiment: str             # "flow", "whale", "whale_solo"
    market_id: str
    question: str
    direction: str              # "UP" or "DOWN"
    entry_price: float
    shares: float
    cost: float
    entry_time: str
    signal_detail: str
    # V2.0: whale metadata
    whale_names: str = ""       # comma-separated whale names that confirmed
    whale_tiers: str = ""       # comma-separated tiers
    consensus_type: str = ""    # "t1_solo", "consensus", "solo_shadow"
    duration_min: int = 5
    asset: str = "BTC"
    # Resolution
    resolved: bool = False
    outcome: str = ""
    pnl: float = 0.0
    resolve_time: str = ""


# ============================================================================
# MAIN BOT
# ============================================================================
class WhaleCopyExperiments:
    def __init__(self):
        self.markets: Dict[str, MarketCandle] = {}
        self.trades: List[PaperTrade] = []
        self.resolved_trades: List[dict] = []
        self.entered_markets: set = set()       # condition_ids with consensus entries
        self.entered_solo: set = set()           # condition_ids with solo shadow entries
        self._whale_last_poll: float = 0.0
        self._token_to_market: Dict[str, Tuple[str, str]] = {}
        self._cycle = 0

        self.stats = {
            "flow": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            "whale": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            "whale_solo": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
        }
        # Per-whale win rate tracking
        self.whale_stats: Dict[str, dict] = {}  # whale_name -> {trades, wins, losses, pnl}

        self._load()

    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    def _load(self):
        if RESULTS_FILE.exists():
            try:
                d = json.loads(RESULTS_FILE.read_text())
                self.resolved_trades = d.get("resolved", [])
                self.stats = {**self.stats, **d.get("stats", {})}
                self.whale_stats = d.get("whale_stats", {})
                for r in self.resolved_trades[-500:]:
                    mid = r.get("market_id", "")
                    if r.get("experiment") == "whale_solo":
                        self.entered_solo.add(mid)
                    else:
                        self.entered_markets.add(mid)
                total_resolved = len(self.resolved_trades)
                w_s = self.stats["whale"]
                ws_s = self.stats["whale_solo"]
                f_s = self.stats["flow"]
                print(f"[LOAD] {total_resolved} resolved | "
                      f"Flow: {f_s['wins']}W/{f_s['losses']}L ${f_s['pnl']:+.2f} | "
                      f"Whale: {w_s['wins']}W/{w_s['losses']}L ${w_s['pnl']:+.2f} | "
                      f"Solo: {ws_s['wins']}W/{ws_s['losses']}L ${ws_s['pnl']:+.2f}")
                if self.whale_stats:
                    print(f"[LOAD] Tracking {len(self.whale_stats)} whale WRs")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        try:
            RESULTS_FILE.write_text(json.dumps({
                "resolved": self.resolved_trades[-1000:],
                "stats": self.stats,
                "whale_stats": self.whale_stats,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }, indent=2))
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    # ========================================================================
    # ASSET PARSER
    # ========================================================================
    @staticmethod
    def _parse_asset(title: str) -> str:
        t = title.lower()
        for keyword, asset in ASSETS_MAP.items():
            if keyword in t:
                return asset
        return ""

    # ========================================================================
    # MARKET DISCOVERY — 5M + 15M, multi-asset
    # ========================================================================
    async def discover_markets(self) -> List[MarketCandle]:
        """Find active 5M and 15M candle markets for BTC, ETH, SOL, XRP."""
        new_markets = []
        now = datetime.now(timezone.utc)

        async with httpx.AsyncClient(timeout=15) as client:
            for tag_slug in TAG_SLUGS:
                expected_dur = 5 if tag_slug == "5M" else 15
                max_secs = 1800 if expected_dur == 5 else 3600  # 30min/60min lookahead

                try:
                    r = await client.get(
                        "https://gamma-api.polymarket.com/events",
                        params={"tag_slug": tag_slug, "active": "true", "closed": "false",
                                "limit": 200, "order": "endDate", "ascending": "true"},
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    if r.status_code != 200:
                        continue

                    for event in r.json():
                        title = event.get("title", "")
                        asset = self._parse_asset(title)
                        if not asset:
                            continue

                        for m in event.get("markets", []):
                            if m.get("closed", True):
                                continue
                            cid = m.get("conditionId", "")
                            if not cid or cid in self.markets:
                                continue

                            # Parse end time
                            end_str = m.get("endDate", "")
                            if not end_str:
                                continue
                            try:
                                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                                secs_to_end = (end_dt - now).total_seconds()
                                if secs_to_end < 0 or secs_to_end > max_secs:
                                    continue
                            except Exception:
                                continue

                            # Parse duration from title
                            q = m.get("question", "") or title
                            time_match = re.search(r"(\d{1,2}):(\d{2})(AM|PM)-(\d{1,2}):(\d{2})(AM|PM)", q)
                            if time_match:
                                h1, m1, p1 = int(time_match.group(1)), int(time_match.group(2)), time_match.group(3)
                                h2, m2, p2 = int(time_match.group(4)), int(time_match.group(5)), time_match.group(6)
                                t1 = (h1 % 12 + (12 if p1 == "PM" else 0)) * 60 + m1
                                t2 = (h2 % 12 + (12 if p2 == "PM" else 0)) * 60 + m2
                                dur = t2 - t1 if t2 > t1 else t2 + 1440 - t1
                                if dur != expected_dur:
                                    continue
                            else:
                                continue

                            # Parse UP/DOWN tokens
                            tokens = m.get("clobTokenIds", "")
                            if isinstance(tokens, str):
                                tokens = json.loads(tokens) if tokens.startswith("[") else tokens.split(",")
                            outcomes = m.get("outcomes", "")
                            if isinstance(outcomes, str):
                                outcomes = json.loads(outcomes) if outcomes.startswith("[") else outcomes.split(",")

                            up_token = down_token = ""
                            for tok, out in zip(tokens, outcomes):
                                out_lower = out.strip().lower().strip('"')
                                if out_lower in ("up", "yes"):
                                    up_token = tok.strip().strip('"')
                                elif out_lower in ("down", "no"):
                                    down_token = tok.strip().strip('"')

                            if not up_token or not down_token:
                                continue

                            mkt = MarketCandle(
                                condition_id=cid,
                                question=q,
                                end_time=end_dt,
                                up_token=up_token,
                                down_token=down_token,
                                market_num_id=str(m.get("id", "")),
                                duration_min=expected_dur,
                                asset=asset,
                                first_seen=time.time(),
                            )
                            new_markets.append(mkt)
                            self.markets[cid] = mkt
                            self._token_to_market[up_token] = (cid, "UP")
                            self._token_to_market[down_token] = (cid, "DOWN")

                except Exception as e:
                    print(f"[DISCOVER] Error ({tag_slug}): {e}")

        return new_markets

    # ========================================================================
    # ORDER BOOK — get mid prices via REST
    # ========================================================================
    async def get_book(self, token_id: str) -> Optional[dict]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://clob.polymarket.com/book",
                    params={"token_id": token_id},
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                if r.status_code != 200:
                    return None
                data = r.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                best_bid = max((float(b["price"]) for b in bids), default=0)
                best_ask = min((float(a["price"]) for a in asks), default=1.0)
                mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask < 1.0 else 0.5
                return {"best_bid": best_bid, "best_ask": best_ask, "mid": mid,
                        "bid_depth": sum(float(b.get("size", 0)) for b in bids[:5]),
                        "ask_depth": sum(float(a.get("size", 0)) for a in asks[:5])}
        except Exception:
            return None

    async def update_market_books(self):
        """Fetch order books for all tracked markets concurrently."""
        active = [(cid, mkt) for cid, mkt in self.markets.items() if not mkt.resolved]
        if not active:
            return

        async def fetch_pair(cid, mkt):
            up_book, down_book = await asyncio.gather(
                self.get_book(mkt.up_token),
                self.get_book(mkt.down_token),
            )
            if up_book:
                mkt.up_mid_latest = up_book["mid"]
                mkt.up_mids.append(up_book["mid"])
                if mkt.up_mid_initial == 0:
                    mkt.up_mid_initial = up_book["mid"]
            if down_book:
                mkt.down_mid_latest = down_book["mid"]
                mkt.down_mids.append(down_book["mid"])
                if mkt.down_mid_initial == 0:
                    mkt.down_mid_initial = down_book["mid"]

        await asyncio.gather(*[fetch_pair(cid, mkt) for cid, mkt in active])

    # ========================================================================
    # EXPERIMENT 1: ORDER FLOW — track mid-price drift (5M BTC only)
    # ========================================================================
    def evaluate_flow_signal(self, mkt: MarketCandle) -> Optional[Tuple[str, float, str]]:
        """Check if order flow drift exceeds threshold. 5M BTC only."""
        if mkt.flow_entered or mkt.resolved:
            return None
        if mkt.duration_min != 5 or mkt.asset != "BTC":
            return None  # Flow only for 5M BTC
        if mkt.up_mid_initial == 0 or mkt.down_mid_initial == 0:
            return None

        age_sec = time.time() - mkt.first_seen
        if age_sec < FLOW_OBSERVATION_SEC:
            return None

        up_drift = mkt.up_mid_latest - mkt.up_mid_initial
        down_drift = mkt.down_mid_latest - mkt.down_mid_initial
        net_drift = up_drift - down_drift
        abs_drift = abs(net_drift)
        if abs_drift < FLOW_MIN_DRIFT:
            return None

        direction = "UP" if net_drift > 0 else "DOWN"
        entry_price = mkt.up_mid_latest if direction == "UP" else mkt.down_mid_latest

        if entry_price > FLOW_MAX_ENTRY or entry_price < FLOW_MIN_ENTRY:
            return None

        if mkt.end_time:
            secs_left = (mkt.end_time - datetime.now(timezone.utc)).total_seconds()
            if secs_left < 120 or secs_left > 450:
                return None

        detail = (f"drift={net_drift:+.3f} (UP:{up_drift:+.3f}, DN:{down_drift:+.3f}) | "
                  f"UP:{mkt.up_mid_initial:.2f}->{mkt.up_mid_latest:.2f} | "
                  f"DN:{mkt.down_mid_initial:.2f}->{mkt.down_mid_latest:.2f} | "
                  f"age={age_sec:.0f}s | snapshots={len(mkt.up_mids)}")
        return direction, entry_price, detail

    # ========================================================================
    # EXPERIMENT 2+3: WHALE COPY — concurrent polling + consensus filter
    # ========================================================================
    async def poll_whale_activity(self):
        """Poll all directional whales concurrently for recent trades."""
        now = time.time()
        if now - self._whale_last_poll < WHALE_POLL_SEC:
            return
        self._whale_last_poll = now

        active_cids = {cid for cid, mkt in self.markets.items() if not mkt.resolved}
        if not active_cids:
            return

        sem = asyncio.Semaphore(WHALE_POLL_CONCURRENCY)

        async def fetch_whale(whale_name, whale_info):
            async with sem:
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        r = await client.get(
                            "https://data-api.polymarket.com/activity",
                            params={"user": whale_info["addr"], "limit": 10},
                            headers={"User-Agent": "Mozilla/5.0"}
                        )
                        if r.status_code != 200:
                            return

                        trades = r.json()
                        if not isinstance(trades, list):
                            return

                        for trade in trades:
                            asset_id = trade.get("asset", "") or trade.get("maker_asset_id", "")
                            if asset_id in self._token_to_market:
                                cid, side = self._token_to_market[asset_id]
                                if cid in active_cids:
                                    mkt = self.markets.get(cid)
                                    if mkt and whale_name not in mkt.whale_signals:
                                        action = trade.get("type", "").upper()
                                        if action == "BUY" or trade.get("side", "").upper() == "BUY":
                                            mkt.whale_signals[whale_name] = side
                                            tier = whale_info["tier"]
                                            print(f"  [WHALE T{tier}] {whale_name} BOUGHT {side} on "
                                                  f"{mkt.asset} {mkt.duration_min}M: {mkt.question[:50]}")

                except Exception as e:
                    if "429" not in str(e):
                        pass  # Silently skip errors (too spammy with 15 whales)

        await asyncio.gather(*[fetch_whale(n, info) for n, info in WHALES.items()])

    def evaluate_whale_consensus(self, mkt: MarketCandle) -> Tuple[Optional[tuple], Optional[tuple]]:
        """Evaluate whale consensus for a market.

        Returns (consensus_entry, solo_shadow):
            consensus_entry: (direction, names, tiers, consensus_type) or None
            solo_shadow:     (direction, names, tiers, "solo_shadow") or None
        """
        if not mkt.whale_signals or mkt.resolved:
            return None, None

        # Count directions
        up_whales = [(n, WHALES[n]["tier"]) for n, d in mkt.whale_signals.items() if d == "UP" and n in WHALES]
        down_whales = [(n, WHALES[n]["tier"]) for n, d in mkt.whale_signals.items() if d == "DOWN" and n in WHALES]

        consensus_result = None
        solo_result = None

        if not mkt.whale_entered:
            # --- T1 solo entry ---
            t1_up = [(n, t) for n, t in up_whales if t == 1]
            t1_down = [(n, t) for n, t in down_whales if t == 1]

            if t1_up and not t1_down:
                names = [n for n, _ in up_whales]  # Include all UP whales
                tiers = [str(WHALES[n]["tier"]) for n in names]
                consensus_result = ("UP", names, tiers, "t1_solo")
            elif t1_down and not t1_up:
                names = [n for n, _ in down_whales]
                tiers = [str(WHALES[n]["tier"]) for n in names]
                consensus_result = ("DOWN", names, tiers, "t1_solo")
            elif t1_up and t1_down:
                pass  # T1 whales disagree — skip entirely
            else:
                # --- No T1, check consensus (2+ whales agree) ---
                if len(up_whales) >= CONSENSUS_MIN_WHALES:
                    names = [n for n, _ in up_whales]
                    tiers = [str(WHALES[n]["tier"]) for n in names]
                    consensus_result = ("UP", names, tiers, "consensus")
                elif len(down_whales) >= CONSENSUS_MIN_WHALES:
                    names = [n for n, _ in down_whales]
                    tiers = [str(WHALES[n]["tier"]) for n in names]
                    consensus_result = ("DOWN", names, tiers, "consensus")

        # --- Solo shadow (for data collection) ---
        if not mkt.whale_solo_entered and consensus_result is None:
            # Log the majority direction as shadow
            if len(up_whales) == 1 and len(down_whales) == 0:
                n, t = up_whales[0]
                solo_result = ("UP", [n], [str(t)], "solo_shadow")
            elif len(down_whales) == 1 and len(up_whales) == 0:
                n, t = down_whales[0]
                solo_result = ("DOWN", [n], [str(t)], "solo_shadow")

        return consensus_result, solo_result

    # ========================================================================
    # PAPER TRADE EXECUTION
    # ========================================================================
    def paper_enter(self, mkt: MarketCandle, experiment: str, direction: str,
                    entry_price: float, detail: str,
                    whale_names: str = "", whale_tiers: str = "",
                    consensus_type: str = "") -> Optional[PaperTrade]:
        """Create a paper trade."""
        # Validate entry price
        if entry_price <= 0:
            return None
        max_price = FLOW_MAX_ENTRY if experiment == "flow" else WHALE_MAX_ENTRY
        min_price = FLOW_MIN_ENTRY if experiment == "flow" else WHALE_MIN_ENTRY
        if entry_price > max_price or entry_price < min_price:
            return None

        # Check time remaining
        if mkt.end_time:
            secs_left = (mkt.end_time - datetime.now(timezone.utc)).total_seconds()
            min_secs = 60
            max_secs = 450 if mkt.duration_min == 5 else 840  # 7.5min for 5M, 14min for 15M
            if secs_left < min_secs or secs_left > max_secs:
                return None

        shares = max(MIN_SHARES, math.floor(PAPER_SIZE / entry_price))
        cost = round(entry_price * shares, 4)

        trade = PaperTrade(
            experiment=experiment,
            market_id=mkt.condition_id,
            question=mkt.question[:80],
            direction=direction,
            entry_price=entry_price,
            shares=shares,
            cost=cost,
            entry_time=datetime.now(timezone.utc).isoformat(),
            signal_detail=detail,
            whale_names=whale_names,
            whale_tiers=whale_tiers,
            consensus_type=consensus_type,
            duration_min=mkt.duration_min,
            asset=mkt.asset,
        )
        self.trades.append(trade)

        if experiment == "flow":
            mkt.flow_entered = True
            mkt.flow_direction = direction
            mkt.flow_drift = abs(float(detail.split("drift=")[1].split(" ")[0])) if "drift=" in detail else 0
            self.entered_markets.add(mkt.condition_id)
        elif experiment == "whale":
            mkt.whale_entered = True
            self.entered_markets.add(mkt.condition_id)
        elif experiment == "whale_solo":
            mkt.whale_solo_entered = True
            self.entered_solo.add(mkt.condition_id)

        tag = {"flow": "FLOW", "whale": "WHALE-CONSENSUS", "whale_solo": "WHALE-SOLO"}[experiment]
        tier_tag = f" [{consensus_type}]" if consensus_type else ""
        print(f"  [{tag}] {mkt.asset} {mkt.duration_min}M {direction} {shares}sh @ ${entry_price:.2f} = ${cost:.2f}{tier_tag}")
        if whale_names:
            print(f"    Whales: {whale_names} (tiers: {whale_tiers})")
        elif detail:
            print(f"    Signal: {detail}")

        return trade

    # ========================================================================
    # RESOLUTION — check if markets expired, update per-whale stats
    # ========================================================================
    async def resolve_trades(self):
        now = datetime.now(timezone.utc)

        for trade in self.trades:
            if trade.resolved:
                continue

            mkt = self.markets.get(trade.market_id)
            if not mkt:
                continue

            if mkt.end_time and now > mkt.end_time + timedelta(minutes=2):
                outcome = await self._get_market_outcome(mkt)
                if outcome:
                    mkt.resolved = True
                    mkt.outcome = outcome
                    trade.resolved = True
                    trade.outcome = outcome
                    trade.resolve_time = now.isoformat()

                    if trade.direction == outcome:
                        revenue = trade.shares * 1.0
                        trade.pnl = round(revenue - trade.cost, 4)
                        self.stats[trade.experiment]["wins"] += 1
                    else:
                        trade.pnl = round(-trade.cost, 4)
                        self.stats[trade.experiment]["losses"] += 1

                    self.stats[trade.experiment]["trades"] += 1
                    self.stats[trade.experiment]["pnl"] = round(
                        self.stats[trade.experiment]["pnl"] + trade.pnl, 4)

                    # Update per-whale stats for all whales that signaled this market
                    if trade.whale_names:
                        for whale_name in trade.whale_names.split(","):
                            whale_name = whale_name.strip()
                            if not whale_name:
                                continue
                            if whale_name not in self.whale_stats:
                                self.whale_stats[whale_name] = {
                                    "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0
                                }
                            ws = self.whale_stats[whale_name]
                            ws["trades"] += 1
                            if trade.direction == outcome:
                                ws["wins"] += 1
                            else:
                                ws["losses"] += 1
                            ws["pnl"] = round(ws["pnl"] + trade.pnl, 4)

                    # Also update per-whale stats for ALL whales that signaled this market
                    # (even if they weren't in the consensus — for tracking individual WR)
                    if mkt.whale_signals:
                        for wn, wd in mkt.whale_signals.items():
                            shadow_key = f"_all_{wn}"
                            if shadow_key not in self.whale_stats:
                                self.whale_stats[shadow_key] = {
                                    "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0
                                }
                            ws = self.whale_stats[shadow_key]
                            ws["trades"] += 1
                            if wd == outcome:
                                ws["wins"] += 1
                            else:
                                ws["losses"] += 1
                            # Hypothetical PnL at $3/trade
                            hyp_shares = max(5, math.floor(3.0 / 0.50))
                            hyp_pnl = round(hyp_shares * 1.0 - hyp_shares * 0.50, 4) if wd == outcome else round(-hyp_shares * 0.50, 4)
                            ws["pnl"] = round(ws["pnl"] + hyp_pnl, 4)

                    tag = {"flow": "FLOW", "whale": "WHALE", "whale_solo": "SOLO"}[trade.experiment]
                    s = self.stats[trade.experiment]
                    wr = s["wins"] / max(1, s["trades"]) * 100
                    result = "WIN" if trade.direction == outcome else "LOSS"
                    print(f"  [{tag}] {result} {trade.asset} {trade.duration_min}M {trade.direction} | "
                          f"PnL: ${trade.pnl:+.2f} | "
                          f"Total: {s['wins']}W/{s['losses']}L ({wr:.0f}%) ${s['pnl']:+.2f}")
                    if trade.whale_names:
                        print(f"    Whales: {trade.whale_names} ({trade.consensus_type})")

                    self.resolved_trades.append(asdict(trade))
                    self._save()

        self.trades = [t for t in self.trades if not t.resolved]

    async def _get_market_outcome(self, mkt: MarketCandle) -> Optional[str]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                if mkt.market_num_id:
                    r = await client.get(
                        f"https://gamma-api.polymarket.com/markets/{mkt.market_num_id}",
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                else:
                    r = await client.get(
                        "https://gamma-api.polymarket.com/markets",
                        params={"condition_id": mkt.condition_id},
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                if r.status_code != 200:
                    return None

                data = r.json()
                if isinstance(data, list):
                    data = data[0] if data else None
                if not data:
                    return None

                prices = data.get("outcomePrices", "")
                if isinstance(prices, str):
                    try:
                        prices = json.loads(prices)
                    except Exception:
                        return None

                if not prices or len(prices) < 2:
                    return None

                up_price = float(prices[0])
                down_price = float(prices[1])

                if up_price > 0.95:
                    return "UP"
                elif down_price > 0.95:
                    return "DOWN"
                return None
        except Exception:
            return None

    # ========================================================================
    # CLEANUP
    # ========================================================================
    def cleanup_old_markets(self):
        now = datetime.now(timezone.utc)
        to_remove = []
        for cid, mkt in self.markets.items():
            buffer = 15 if mkt.duration_min == 5 else 20
            if mkt.end_time and now > mkt.end_time + timedelta(minutes=buffer):
                to_remove.append(cid)
        for cid in to_remove:
            mkt = self.markets.pop(cid, None)
            if mkt:
                for tok in [mkt.up_token, mkt.down_token]:
                    self._token_to_market.pop(tok, None)

    # ========================================================================
    # WHALE STATS REPORT — every 100 cycles
    # ========================================================================
    def print_whale_report(self):
        """Print per-whale win rate leaderboard."""
        if not self.whale_stats:
            return

        print("\n[WHALE LEADERBOARD] Per-whale directional accuracy:")
        # Show _all_ stats (tracks every individual signal regardless of consensus)
        rows = []
        for key, ws in sorted(self.whale_stats.items()):
            if not key.startswith("_all_"):
                continue
            name = key[5:]  # Strip "_all_" prefix
            tier = WHALES.get(name, {}).get("tier", "?")
            trades = ws["trades"]
            if trades == 0:
                continue
            wr = ws["wins"] / trades * 100
            rows.append((wr, name, tier, trades, ws["wins"], ws["losses"], ws["pnl"]))

        rows.sort(key=lambda x: (-x[0], -x[3]))  # Sort by WR desc, then trades desc
        for wr, name, tier, trades, wins, losses, pnl in rows:
            marker = " ***" if wr >= 60 and trades >= 5 else ""
            print(f"    T{tier} {name:25s} | {wins}W/{losses}L ({wr:5.1f}%) | ${pnl:+.2f}{marker}")

        # Show consensus vs solo comparison
        w_s = self.stats["whale"]
        ws_s = self.stats["whale_solo"]
        if w_s["trades"] > 0 or ws_s["trades"] > 0:
            w_wr = w_s["wins"] / max(1, w_s["trades"]) * 100
            ws_wr = ws_s["wins"] / max(1, ws_s["trades"]) * 100
            print(f"\n    CONSENSUS: {w_s['wins']}W/{w_s['losses']}L ({w_wr:.0f}%) ${w_s['pnl']:+.2f}")
            print(f"    SOLO:      {ws_s['wins']}W/{ws_s['losses']}L ({ws_wr:.0f}%) ${ws_s['pnl']:+.2f}")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================
    async def run(self):
        t1_names = [n for n, info in WHALES.items() if info["tier"] == 1]
        t2_names = [n for n, info in WHALES.items() if info["tier"] == 2]
        t3_names = [n for n, info in WHALES.items() if info["tier"] == 3]

        print("=" * 70)
        print("WHALE COPY + FLOW EXPERIMENTS V2.0 — PAPER ONLY")
        print("=" * 70)
        print(f"Exp 1: ORDER FLOW — 5M BTC, observe {FLOW_OBSERVATION_SEC}s, min drift {FLOW_MIN_DRIFT}")
        print(f"Exp 2: WHALE CONSENSUS — T1 solo OR {CONSENSUS_MIN_WHALES}+ agree = enter")
        print(f"Exp 3: WHALE SOLO — shadow-only (data collection)")
        print(f"Markets: 5M + 15M | Assets: BTC, ETH, SOL, XRP")
        print(f"Whales: {len(WHALES)} directional ({len(t1_names)} T1, {len(t2_names)} T2, {len(t3_names)} T3)")
        print(f"  T1 (solo enter): {', '.join(t1_names)}")
        print(f"  T2 (consensus):  {', '.join(t2_names)}")
        print(f"  T3 (consensus):  {', '.join(t3_names)}")
        print(f"Paper size: ${PAPER_SIZE}/trade | Whale poll: {WHALE_POLL_SEC}s | Scan: {SCAN_INTERVAL}s")
        print("=" * 70)

        while True:
            try:
                self._cycle += 1

                # 1. Discover new markets (5M + 15M, all assets)
                new_markets = await self.discover_markets()
                if new_markets:
                    by_tag = {}
                    for m in new_markets:
                        key = f"{m.asset} {m.duration_min}M"
                        by_tag[key] = by_tag.get(key, 0) + 1
                    tag_str = ", ".join(f"{v} {k}" for k, v in sorted(by_tag.items()))
                    print(f"\n[DISCOVER] Found {len(new_markets)} new markets: {tag_str}")

                # 2. Update order books
                await self.update_market_books()

                # 3. Poll whale activity (concurrent)
                await self.poll_whale_activity()

                # 4. Evaluate signals
                for cid, mkt in list(self.markets.items()):
                    if mkt.resolved:
                        continue

                    # Experiment 1: Flow (5M BTC only)
                    if cid not in self.entered_markets:
                        flow_sig = self.evaluate_flow_signal(mkt)
                        if flow_sig:
                            direction, entry_price, detail = flow_sig
                            print(f"\n[FLOW SIGNAL] {mkt.asset} {mkt.duration_min}M: {mkt.question[:55]}")
                            self.paper_enter(mkt, "flow", direction, entry_price, detail)

                    # Experiment 2+3: Whale consensus + solo shadow
                    if not mkt.flow_entered:
                        consensus, solo = self.evaluate_whale_consensus(mkt)

                        if consensus and cid not in self.entered_markets:
                            direction, names, tiers, ctype = consensus
                            entry_price = mkt.up_mid_latest if direction == "UP" else mkt.down_mid_latest
                            detail = f"{ctype} | whales={','.join(names)} | tiers={','.join(tiers)}"
                            print(f"\n[WHALE CONSENSUS] {mkt.asset} {mkt.duration_min}M: {mkt.question[:55]}")
                            self.paper_enter(mkt, "whale", direction, entry_price, detail,
                                             whale_names=",".join(names),
                                             whale_tiers=",".join(tiers),
                                             consensus_type=ctype)

                        if solo and cid not in self.entered_solo:
                            direction, names, tiers, ctype = solo
                            entry_price = mkt.up_mid_latest if direction == "UP" else mkt.down_mid_latest
                            detail = f"solo | whale={names[0]} | tier={tiers[0]}"
                            self.paper_enter(mkt, "whale_solo", direction, entry_price, detail,
                                             whale_names=",".join(names),
                                             whale_tiers=",".join(tiers),
                                             consensus_type=ctype)

                # 5. Resolve expired trades
                await self.resolve_trades()

                # 6. Cleanup
                self.cleanup_old_markets()

                # 7. Status every 10 cycles
                if self._cycle % 10 == 0:
                    active_mkts = sum(1 for m in self.markets.values() if not m.resolved)
                    active_5m = sum(1 for m in self.markets.values() if not m.resolved and m.duration_min == 5)
                    active_15m = sum(1 for m in self.markets.values() if not m.resolved and m.duration_min == 15)
                    active_trades = len(self.trades)

                    f_s = self.stats["flow"]
                    w_s = self.stats["whale"]
                    ws_s = self.stats["whale_solo"]
                    f_wr = f_s["wins"] / max(1, f_s["trades"]) * 100
                    w_wr = w_s["wins"] / max(1, w_s["trades"]) * 100
                    ws_wr = ws_s["wins"] / max(1, ws_s["trades"]) * 100

                    print(f"\n--- Cycle {self._cycle} | Mkts: {active_mkts} "
                          f"(5M:{active_5m} 15M:{active_15m}) | Active: {active_trades} ---")
                    print(f"    FLOW:      {f_s['wins']}W/{f_s['losses']}L ({f_wr:.0f}%) ${f_s['pnl']:+.2f}")
                    print(f"    CONSENSUS: {w_s['wins']}W/{w_s['losses']}L ({w_wr:.0f}%) ${w_s['pnl']:+.2f}")
                    print(f"    SOLO:      {ws_s['wins']}W/{ws_s['losses']}L ({ws_wr:.0f}%) ${ws_s['pnl']:+.2f}")

                # 8. Whale leaderboard every 100 cycles
                if self._cycle % 100 == 0:
                    self.print_whale_report()

                await asyncio.sleep(SCAN_INTERVAL)

            except KeyboardInterrupt:
                print("\n[STOP] Shutting down...")
                self._save()
                break
            except Exception as e:
                print(f"[ERROR] Main loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(30)


# ============================================================================
# ENTRY POINT
# ============================================================================
async def main():
    pid = acquire_pid_lock("5m_experiments")
    if not pid:
        print("[ERROR] Another instance is already running!")
        sys.exit(1)

    bot = WhaleCopyExperiments()
    try:
        await bot.run()
    finally:
        release_pid_lock("5m_experiments")

if __name__ == "__main__":
    asyncio.run(main())
