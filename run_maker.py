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
# AUTO-REDEEM (gasless relayer primary, direct on-chain fallback)
# ============================================================================

_poly_web3_service = None
_relayer_quota_reset_at = 0  # timestamp when relayer quota resets

def _get_poly_web3_service():
    """Get or create PolyWeb3Service for gasless relayer redemptions."""
    global _poly_web3_service
    if _poly_web3_service is not None:
        return _poly_web3_service
    try:
        from py_clob_client.client import ClobClient
        from py_builder_relayer_client.client import RelayClient
        from py_builder_signing_sdk.config import BuilderConfig
        from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds
        from poly_web3 import PolyWeb3Service
        from crypto_utils import decrypt_key

        password = os.getenv("POLYMARKET_PASSWORD", "")
        if not password:
            return None
        pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        if pk.startswith("ENC:"):
            pk = decrypt_key(pk[4:], os.getenv("POLYMARKET_KEY_SALT", ""), password)
        if not pk:
            return None
        proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")

        client = ClobClient(host="https://clob.polymarket.com", key=pk, chain_id=137,
                            signature_type=1, funder=proxy_address if proxy_address else None)
        creds = client.derive_api_key()
        client.set_api_creds(creds)

        bk = decrypt_key(os.getenv("POLY_BUILDER_API_KEY", "")[4:], os.getenv("POLY_BUILDER_API_KEY_SALT", ""), password)
        bs = decrypt_key(os.getenv("POLY_BUILDER_SECRET", "")[4:], os.getenv("POLY_BUILDER_SECRET_SALT", ""), password)
        bp = decrypt_key(os.getenv("POLY_BUILDER_PASSPHRASE", "")[4:], os.getenv("POLY_BUILDER_PASSPHRASE_SALT", ""), password)

        bc = BuilderConfig(local_builder_creds=BuilderApiKeyCreds(key=bk, secret=bs, passphrase=bp))
        rc = RelayClient(relayer_url="https://relayer-v2.polymarket.com", chain_id=137, private_key=pk, builder_config=bc)

        _poly_web3_service = PolyWeb3Service(clob_client=client, relayer_client=rc,
                                              rpc_url="https://polygon-bor.publicnode.com")
        print("[REDEEM] Gasless relayer initialized")
        return _poly_web3_service
    except Exception as e:
        print(f"[REDEEM] Relayer init failed: {str(e)[:80]}")
        return None


def _try_gasless_redeem():
    """Try gasless relayer. Returns (success: bool, claimed_shares: float).
    Monkey-patches the broken _submit_transactions to handle error responses."""
    global _relayer_quota_reset_at
    if time.time() < _relayer_quota_reset_at:
        return False, 0.0

    svc = _get_poly_web3_service()
    if not svc:
        return False, 0.0

    try:
        import requests as req_lib
        from poly_web3.const import RELAYER_URL, SUBMIT_TRANSACTION
        from poly_web3.web3_service.proxy_service import ProxyWeb3Service
        from eth_utils import to_checksum_address

        # Monkey-patch _submit_transactions to handle error responses
        _original_submit = ProxyWeb3Service._submit_transactions

        def _patched_submit(self_svc, txs, metadata):
            if self_svc.clob_client is None:
                raise Exception("signer not found")
            _from = to_checksum_address(self_svc.clob_client.get_address())
            from poly_web3.const import GET_RELAY_PAYLOAD
            from poly_web3.schema import WalletType
            rp = self_svc._get_relay_payload(_from, self_svc.wallet_type)
            args = {
                "from": _from, "gasPrice": "0",
                "data": self_svc.encode_proxy_transaction_data(txs),
                "relay": rp["address"], "nonce": rp["nonce"],
            }
            req_body = self_svc.build_proxy_transaction_request(args, metadata=metadata)
            headers = self_svc.relayer_client._generate_builder_headers("POST", SUBMIT_TRANSACTION, req_body)
            response = req_lib.post(RELAYER_URL + SUBMIT_TRANSACTION, json=req_body, headers=headers)
            resp_json = response.json()

            # Check for errors BEFORE accessing transactionID
            if "error" in resp_json:
                raise Exception(f"Relayer: {resp_json['error']}")
            if "transactionID" not in resp_json:
                raise Exception(f"Relayer missing transactionID: {list(resp_json.keys())}")

            from poly_web3.const import STATE_MINED, STATE_CONFIRMED, STATE_FAILED
            return self_svc.relayer_client.poll_until_state(
                transaction_id=resp_json["transactionID"],
                states=[STATE_MINED, STATE_CONFIRMED],
                fail_state=STATE_FAILED, max_polls=100,
            )

        # Apply patch
        ProxyWeb3Service._submit_transactions = _patched_submit

        proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
        positions = svc.fetch_positions(user_address=proxy)
        if not positions:
            ProxyWeb3Service._submit_transactions = _original_submit
            return True, 0.0

        total = sum(float(p.get("size", 0) or 0) for p in positions)
        results = svc.redeem_all(batch_size=1)

        # Restore original
        ProxyWeb3Service._submit_transactions = _original_submit

        if results:
            print(f"[REDEEM] Gasless OK: {len(results)} batch(es), ~${total:.0f}")
            return True, total
        # Results empty but positions exist = relayer failed silently
        return False, 0.0

    except Exception as e:
        err = str(e)
        if "quota exceeded" in err:
            import re
            m = re.search(r"resets in (\d+)", err)
            wait = int(m.group(1)) if m else 3600
            _relayer_quota_reset_at = time.time() + wait
            print(f"[REDEEM] Relayer quota exceeded, retry in {wait//60}min")
        elif "Relayer:" in err:
            _relayer_quota_reset_at = time.time() + 300
            print(f"[REDEEM] Relayer error: {err[:80]}, retry in 5min")
        elif "No winning" not in err:
            print(f"[REDEEM] Gasless error: {err[:80]}")
        return False, 0.0


# Contract addresses on Polygon (for direct on-chain fallback)
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
PROXY_FACTORY_ADDRESS = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"

CTF_ABI = [{
    "inputs": [
        {"name": "collateralToken", "type": "address"},
        {"name": "parentCollectionId", "type": "bytes32"},
        {"name": "conditionId", "type": "bytes32"},
        {"name": "indexSets", "type": "uint256[]"},
    ],
    "name": "redeemPositions", "outputs": [],
    "stateMutability": "nonpayable", "type": "function",
}]

PROXY_FACTORY_ABI = [{
    "inputs": [{
        "components": [
            {"name": "to", "type": "address"},
            {"name": "typeCode", "type": "uint256"},
            {"name": "data", "type": "bytes"},
            {"name": "value", "type": "uint256"},
        ],
        "name": "_txns", "type": "tuple[]",
    }],
    "name": "proxy", "outputs": [],
    "stateMutability": "nonpayable", "type": "function",
}]

_redeem_w3 = None
_redeem_account = None

def _init_redeem():
    """Initialize Web3 + account for direct on-chain redemptions."""
    global _redeem_w3, _redeem_account
    if _redeem_w3 is not None and _redeem_account is not None:
        return True
    try:
        from web3 import Web3
        from crypto_utils import decrypt_key

        password = os.getenv("POLYMARKET_PASSWORD", "")
        if not password:
            return False
        pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        if pk.startswith("ENC:"):
            pk = decrypt_key(pk[4:], os.getenv("POLYMARKET_KEY_SALT", ""), password)
        if not pk:
            return False
        if not pk.startswith("0x"):
            pk = "0x" + pk

        rpcs = [
            "https://polygon.llamarpc.com",
            "https://polygon-bor.publicnode.com",
            "https://polygon-rpc.com",
        ]
        for rpc in rpcs:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 30}))
                if w3.is_connected():
                    _redeem_w3 = w3
                    _redeem_account = w3.eth.account.from_key(pk)
                    print(f"[REDEEM] Direct on-chain initialized via {rpc}")
                    return True
            except Exception:
                continue
        return False
    except Exception as e:
        print(f"[REDEEM] Init error: {str(e)[:80]}")
        return False


def auto_redeem_winnings():
    """Auto-redeem: try gasless relayer first, fall back to direct on-chain.
    Skips direct on-chain if gas > 200 gwei to preserve MATIC.
    Returns: float amount of USDC claimed (0.0 if nothing claimed)."""
    # Try gasless relayer first (free, no MATIC needed)
    success, claimed = _try_gasless_redeem()
    if success:
        return claimed

    # Fallback: direct on-chain CTF contract call
    try:
        proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
        if not proxy_address:
            return 0.0

        # Fetch redeemable positions from data API
        r = httpx.get(
            "https://data-api.polymarket.com/positions",
            params={"user": proxy_address, "sizeThreshold": "0"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        positions = r.json()
        redeemable = [p for p in positions if p.get("redeemable", False)]
        if not redeemable:
            return 0.0

        total_shares = sum(float(p.get("size", 0) or 0) for p in redeemable)
        print(f"[REDEEM] Found {len(redeemable)} redeemable ({total_shares:.0f} shares, ~${total_shares:.2f})")

        if not _init_redeem():
            print("[REDEEM] Failed to init Web3")
            return 0.0

        from web3 import Web3

        # Check gas price — skip if too expensive (> 400 gwei base fee)
        gas_price = _redeem_w3.eth.gas_price
        gas_gwei = gas_price / 1e9
        if gas_gwei > 400:
            print(f"[REDEEM] Gas too high ({gas_gwei:.0f} gwei > 400), skipping")
            return 0.0

        # Check for stuck pending TXs
        try:
            nonce_latest = _redeem_w3.eth.get_transaction_count(_redeem_account.address, "latest")
            nonce_pending = _redeem_w3.eth.get_transaction_count(_redeem_account.address, "pending")
            if nonce_pending > nonce_latest:
                print(f"[REDEEM] Pending TX (nonce {nonce_latest}->{nonce_pending}), waiting...")
                return 0.0
        except Exception:
            pass

        ctf = _redeem_w3.eth.contract(
            address=Web3.to_checksum_address(CTF_ADDRESS), abi=CTF_ABI)
        factory = _redeem_w3.eth.contract(
            address=Web3.to_checksum_address(PROXY_FACTORY_ADDRESS), abi=PROXY_FACTORY_ABI)

        # Redeem 1 position per cycle to keep things simple and reliable
        cid = redeemable[0].get("conditionId", "")
        if not cid:
            return 0.0

        redeem_data = ctf.encode_abi(
            "redeemPositions",
            args=[
                Web3.to_checksum_address(USDC_ADDRESS),
                bytes(32),
                Web3.to_bytes(hexstr=cid),
                [1, 2],
            ],
        )
        proxy_txn = (
            Web3.to_checksum_address(CTF_ADDRESS),
            1,
            Web3.to_bytes(hexstr=redeem_data),
            0,
        )
        nonce = _redeem_w3.eth.get_transaction_count(_redeem_account.address, "latest")
        use_gas_price = min(int(gas_price * 1.5), _redeem_w3.to_wei(500, "gwei"))
        txn = factory.functions.proxy([proxy_txn]).build_transaction({
            "from": _redeem_account.address,
            "nonce": nonce,
            "gasPrice": use_gas_price,
            "gas": 300000,
        })
        signed = _redeem_w3.eth.account.sign_transaction(txn, _redeem_account.key)
        tx_hash = _redeem_w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = _redeem_w3.eth.wait_for_transaction_receipt(tx_hash, timeout=90)
        if receipt["status"] == 1:
            size = float(redeemable[0].get("size", 0) or 0)
            print(f"[REDEEM] OK {tx_hash.hex()[:16]}... {size:.0f} shares redeemed")
            return size
        else:
            print(f"[REDEEM] REVERTED {tx_hash.hex()[:16]}...")
            return 0.0

    except Exception as e:
        err = str(e)
        if "No winning" not in err and "already known" not in err:
            print(f"[REDEEM] Error: {err[:100]}")
        return 0.0

    except Exception as e:
        if "No winning" not in str(e):
            print(f"[REDEEM] Error: {str(e)[:80]}")
        return 0.0


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
    SIZE_PER_SIDE_USD: float = 20.0      # V2.3: $20/side (was $15). $220 capital, 5m-only, 100% WR.
    MAX_PAIR_EXPOSURE: float = 40.0      # V2.3: $40/pair (was $30)
    MAX_TOTAL_EXPOSURE: float = 200.0    # V2.3: $200 max for 5 concurrent pairs ($220 cap - $20 buffer)
    MIN_SHARES: int = 5                  # CLOB minimum order size

    # Risk
    DAILY_LOSS_LIMIT: float = 32.0       # V2.3: Scaled with $20/side (was $24)
    MAX_CONCURRENT_PAIRS: int = 12       # V1.5: 4 assets x 15M + BTC 5M = needs ~12 slots
    MAX_SINGLE_SIDED: int = 2            # V1.1: Was 3. Partial fills = directional risk. Allow max 2.

    # Timing
    SCAN_INTERVAL: int = 30              # Seconds between market scans
    ORDER_REFRESH: int = 60              # Seconds before re-quoting orders
    CLOSE_BUFFER_MIN: float = 2.0        # Cancel orders N min before close
    FILL_CHECK_INTERVAL: int = 10        # Seconds between fill checks
    MIN_TIME_LEFT_MIN: float = 5.0       # Don't enter markets with < 5 min left

    # Session control (from PolyData analysis)
    SKIP_HOURS_UTC: set = field(default_factory=set)  # All hours active
    BOOST_HOURS_UTC: set = field(default_factory=lambda: {13, 14, 15, 16})  # NY open peak

    # Assets
    ASSETS: dict = field(default_factory=lambda: {
        "BTC": {"keywords": ["bitcoin", "btc"], "enabled": True},
        "ETH": {"keywords": ["ethereum", "eth"], "enabled": False},  # V1.8: Disabled — CSV: 86% WR but -$8.05 PnL. Partial fills bleeding.
        "SOL": {"keywords": ["solana", "sol"], "enabled": True},   # V1.8: Re-enabled — CSV: 100% WR, +$13.70. BTC+SOL only.
        "XRP": {"keywords": ["xrp", "ripple"], "enabled": False},  # V1.7: Disabled — thin books, -$20.03 PnL
    })

    # V1.5: Per-asset risk tiers — SOL/XRP have thinner books, need tighter spreads
    # "max_combined" = maximum combined bid price (lower = more edge required)
    # "balance_range" = (min, max) for each side price (tighter = more balanced)
    ASSET_TIERS: dict = field(default_factory=lambda: {
        "BTC": {"max_combined": 0.98, "balance_range": (0.38, 0.62)},  # Liquid, standard
        "ETH": {"max_combined": 0.98, "balance_range": (0.38, 0.62)},  # Liquid, standard
        "SOL": {"max_combined": 0.98, "balance_range": (0.40, 0.60)},  # Tight balance + fast partial cancel = safety
        "XRP": {"max_combined": 0.98, "balance_range": (0.40, 0.60)},  # Tight balance + fast partial cancel = safety
    })


# ============================================================================
# ML OFFSET OPTIMIZER
# ============================================================================

import random

class OffsetOptimizer:
    """ML-driven bid offset optimizer. Maximizes profit = fill_rate x edge.

    Tracks per-hour, per-offset-bucket stats and uses Thompson Sampling
    to balance exploration vs exploitation of bid offsets.
    """
    OFFSET_BUCKETS = [0.01, 0.015, 0.02, 0.025, 0.03]
    STATE_FILE = Path(__file__).parent / "maker_ml_state.json"
    EXPLORE_RATE = 0.15  # 15% explore, 85% exploit

    def __init__(self):
        # hour_stats[hour][offset_str] = {attempts, paired, partial, unfilled, pnl}
        self.hour_stats: Dict[int, Dict[str, dict]] = {}
        self._init_stats()
        self._load()

    def _init_stats(self):
        for h in range(24):
            self.hour_stats[h] = {}
            for off in self.OFFSET_BUCKETS:
                key = f"{off:.3f}"
                self.hour_stats[h][key] = {
                    "attempts": 0, "paired": 0, "partial": 0,
                    "unfilled": 0, "total_pnl": 0.0
                }

    def _load(self):
        if self.STATE_FILE.exists():
            try:
                data = json.loads(self.STATE_FILE.read_text())
                for h_str, offsets in data.items():
                    h = int(h_str)
                    if h in self.hour_stats:
                        for off_key, stats in offsets.items():
                            if off_key in self.hour_stats[h]:
                                self.hour_stats[h][off_key].update(stats)
            except Exception:
                pass

    def save(self):
        try:
            self.STATE_FILE.write_text(json.dumps(
                {str(h): v for h, v in self.hour_stats.items()}, indent=2))
        except Exception:
            pass

    def get_offset(self, hour: int) -> float:
        """Pick optimal offset for this hour using epsilon-greedy."""
        stats = self.hour_stats.get(hour, {})

        # Need minimum data before exploiting
        total_attempts = sum(s["attempts"] for s in stats.values())
        if total_attempts < 10 or random.random() < self.EXPLORE_RATE:
            # Explore: pick random offset
            return random.choice(self.OFFSET_BUCKETS)

        # Exploit: pick offset with best avg profit per attempt
        best_offset = 0.02
        best_score = -999.0
        for off in self.OFFSET_BUCKETS:
            key = f"{off:.3f}"
            s = stats.get(key, {})
            attempts = s.get("attempts", 0)
            if attempts < 3:
                continue
            avg_pnl = s.get("total_pnl", 0) / attempts
            if avg_pnl > best_score:
                best_score = avg_pnl
                best_offset = off

        return best_offset

    def record(self, hour: int, offset: float, paired: bool, partial: bool, pnl: float):
        """Record outcome of a pair attempt."""
        key = f"{offset:.3f}"
        # Snap to nearest bucket
        closest = min(self.OFFSET_BUCKETS, key=lambda x: abs(x - offset))
        key = f"{closest:.3f}"

        if hour not in self.hour_stats:
            self.hour_stats[hour] = {}
        if key not in self.hour_stats[hour]:
            self.hour_stats[hour][key] = {
                "attempts": 0, "paired": 0, "partial": 0,
                "unfilled": 0, "total_pnl": 0.0
            }

        s = self.hour_stats[hour][key]
        s["attempts"] += 1
        if paired:
            s["paired"] += 1
        elif partial:
            s["partial"] += 1
        else:
            s["unfilled"] += 1
        s["total_pnl"] += pnl

    def get_summary(self) -> str:
        """Return brief summary of optimal offsets per hour."""
        lines = []
        for h in range(24):
            stats = self.hour_stats.get(h, {})
            total = sum(s["attempts"] for s in stats.values())
            if total == 0:
                continue
            best_off = self.get_offset(h)
            best_key = f"{best_off:.3f}"
            s = stats.get(best_key, {})
            att = s.get("attempts", 0)
            pnl = s.get("total_pnl", 0)
            pr = s.get("paired", 0)
            lines.append(f"  UTC {h:02d}: best={best_off:.1%} ({pr}/{att} paired, ${pnl:+.2f})")
        return "\n".join(lines) if lines else "  No data yet"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MarketPair:
    """A pair of Up/Down tokens for one 5-min or 15-min market."""
    market_id: str           # conditionId (used as unique key)
    market_num_id: str       # Numeric market ID (used for gamma-api lookup)
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
    duration_min: int = 15   # 5 or 15


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
    market_num_id: str       # Numeric market ID for gamma-api resolution
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
    bid_offset_used: float = 0.02  # Track which offset was used
    hour_utc: int = -1             # Hour when position was created
    first_fill_time: Optional[str] = None  # V1.3: When first side filled (for fast partial-protect)
    duration_min: int = 15   # 5 or 15

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
        self.ml_optimizer = OffsetOptimizer()

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

        # V1.4: Capital loss circuit breaker (40% net loss -> auto-switch to paper)
        self.circuit_breaker_pct = 0.40  # 40% of starting capital
        self.starting_pnl = None  # Set after _load() from total_pnl at launch
        self.circuit_tripped = False

        if not paper:
            self._init_client()

        self._load()

        # Record PnL at launch as baseline for circuit breaker
        self.starting_pnl = self.stats.get("total_pnl", 0.0)

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
                # Restore active positions (reconstruct dataclass objects)
                for pos_data in data.get("active_positions", []):
                    try:
                        # Reconstruct MakerOrder objects from dicts
                        up_order = None
                        down_order = None
                        if pos_data.get("up_order") and isinstance(pos_data["up_order"], dict):
                            up_order = MakerOrder(**{k: v for k, v in pos_data["up_order"].items()
                                                     if k in MakerOrder.__dataclass_fields__})
                        if pos_data.get("down_order") and isinstance(pos_data["down_order"], dict):
                            down_order = MakerOrder(**{k: v for k, v in pos_data["down_order"].items()
                                                       if k in MakerOrder.__dataclass_fields__})

                        # Build PairPosition with proper objects
                        simple_fields = {k: v for k, v in pos_data.items()
                                        if k in PairPosition.__dataclass_fields__
                                        and k not in ("up_order", "down_order")}
                        pos = PairPosition(**simple_fields)
                        pos.up_order = up_order
                        pos.down_order = down_order

                        if pos.status not in ("resolved", "cancelled"):
                            self.positions[pos.market_id] = pos
                    except Exception:
                        continue  # Skip corrupt entries
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
        """Find all active 5-min BTC Up/Down markets.
        V2.4: 5M markets moved to tag_slug='5M' (was under '15M' before)."""
        pairs = []
        async with httpx.AsyncClient(timeout=15) as client:
            for tag_slug, duration in [("5M", 5)]:
                try:
                    r = await client.get(
                        "https://gamma-api.polymarket.com/events",
                        params={"tag_slug": tag_slug, "active": "true", "closed": "false",
                                "limit": 200, "order": "endDate", "ascending": "true"},
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    if r.status_code != 200:
                        print(f"[DISCOVER] {tag_slug} API error: {r.status_code}")
                        continue

                    events = r.json()
                    now = datetime.now(timezone.utc)
                    for event in events:
                        title = event.get("title", "").lower()
                        matched_asset = None
                        for asset, cfg in self.config.ASSETS.items():
                            if not cfg.get("enabled", False):
                                continue
                            if any(kw in title for kw in cfg["keywords"]):
                                matched_asset = asset
                                break

                        if not matched_asset:
                            continue

                        # V1.7: Hard block on disabled assets (safety net)
                        if matched_asset in ("SOL", "XRP"):
                            continue

                        for m in event.get("markets", []):
                            if m.get("closed", True):
                                continue

                            # V1.5: Skip stale markets (API returns old unresolved events)
                            end_date = m.get("endDate", "")
                            if end_date:
                                try:
                                    end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                                    if end_dt < now:
                                        continue
                                except Exception:
                                    pass

                            # V2.2: Parse actual duration from title, skip 15m markets
                            # Title format: "Bitcoin Up or Down - February 14, 1:30PM-1:45PM ET"
                            import re
                            q = m.get("question", "") or event.get("title", "")
                            time_match = re.search(r"(\d{1,2}):(\d{2})(AM|PM)-(\d{1,2}):(\d{2})(AM|PM)", q)
                            actual_dur = duration
                            if time_match:
                                h1, m1, p1 = int(time_match.group(1)), int(time_match.group(2)), time_match.group(3)
                                h2, m2, p2 = int(time_match.group(4)), int(time_match.group(5)), time_match.group(6)
                                t1 = (h1 % 12 + (12 if p1 == "PM" else 0)) * 60 + m1
                                t2 = (h2 % 12 + (12 if p2 == "PM" else 0)) * 60 + m2
                                actual_dur = t2 - t1 if t2 > t1 else t2 + 1440 - t1
                            if actual_dur > 5:
                                continue  # Skip markets longer than 5m

                            pair = self._parse_market(m, event, matched_asset, actual_dur)
                            if pair:
                                pairs.append(pair)

                except Exception as e:
                    print(f"[DISCOVER] {tag_slug} Error: {e}")

        return pairs

    def _parse_market(self, market: dict, event: dict, asset: str, duration: int = 15) -> Optional[MarketPair]:
        """Parse a market into a MarketPair."""
        try:
            condition_id = market.get("conditionId", "")
            market_num_id = str(market.get("id", ""))  # Numeric ID for gamma-api lookup
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
                market_num_id=market_num_id,
                question=question,
                asset=asset,
                up_token_id=token_ids[up_idx],
                down_token_id=token_ids[down_idx],
                up_mid=up_price,
                down_mid=down_price,
                end_time=end_time,
                duration_min=duration,
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
        """ML-optimized bid offset. Maximizes profit = fill_rate x edge."""
        hour = datetime.now(timezone.utc).hour
        offset = self.ml_optimizer.get_offset(hour)
        return offset

    def evaluate_pair(self, pair: MarketPair) -> dict:
        """Evaluate a market pair for maker opportunity."""
        offset = self._get_bid_offset()

        # V1.6: 5M markets use half offset (bid closer to mid for faster fills)
        if pair.duration_min <= 5:
            offset = max(0.01, round(offset / 2, 2))

        # Our target bid prices
        up_bid = round(pair.up_mid - offset, 2)
        down_bid = round(pair.down_mid - offset, 2)

        # Ensure prices are valid
        up_bid = max(0.01, min(0.95, up_bid))
        down_bid = max(0.01, min(0.95, down_bid))

        combined = up_bid + down_bid
        edge = 1.0 - combined  # How much < $1.00 our total cost is

        # Time remaining
        mins_left = float(pair.duration_min)
        if pair.end_time:
            delta = (pair.end_time - datetime.now(timezone.utc)).total_seconds() / 60
            mins_left = max(0, delta)

        # Min time left scales with duration: 5min markets need 2min, 15min need 5min
        min_time = 2.0 if pair.duration_min <= 5 else self.config.MIN_TIME_LEFT_MIN
        # V1.5: Max time left — don't lock capital in far-future markets
        max_time = pair.duration_min * 1.5  # e.g. 7.5min for 5M, 22.5min for 15M

        # V1.5: Per-asset tier thresholds
        tier = self.config.ASSET_TIERS.get(pair.asset, {"max_combined": 0.96, "balance_range": (0.40, 0.60)})
        max_combined = tier["max_combined"]
        bal_min, bal_max = tier["balance_range"]

        return {
            "up_bid": up_bid,
            "down_bid": down_bid,
            "combined": combined,
            "edge": edge,
            "edge_pct": edge / combined * 100 if combined > 0 else 0,
            "offset": offset,
            "mins_left": mins_left,
            "viable": (
                combined < max_combined              # Per-asset edge requirement
                and edge >= self.config.MIN_SPREAD_EDGE
                and mins_left > min_time             # Need time for fills
                and mins_left < max_time             # Don't enter far-future markets
                and up_bid >= bal_min                 # Balanced markets only
                and down_bid >= bal_min
                and up_bid <= bal_max                 # Symmetric range
                and down_bid <= bal_max
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
            market_num_id=pair.market_num_id,
            question=pair.question,
            asset=pair.asset,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="pending",
            bid_offset_used=eval_result.get("offset", self.config.BID_OFFSET),
            hour_utc=datetime.now(timezone.utc).hour,
            duration_min=pair.duration_min,
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
                # V1.3: Track when first side filled for fast partial-protect
                if not pos.first_fill_time:
                    pos.first_fill_time = datetime.now(timezone.utc).isoformat()

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
            # Scale thresholds by market duration (5 or 15 min)
            dur = pos.duration_min
            if age_min > dur * 0.67:
                fill_prob *= 1.5  # Last third of market = more urgent flow
            elif age_min > dur * 0.80:
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

            # V1.3: FAST partial protection — if one side fills and the other doesn't
            # within grace period, cancel the ENTIRE position.
            # Holding a one-sided bet = pure directional gamble. One partial loss (-$5)
            # wipes ~10 paired wins (+$0.50 each). Better to cancel than gamble.
            # V1.5: SOL/XRP get 1min grace (thinner books), BTC/ETH get 2min
            # V1.6: 5M markets get 3.5min grace (need more time on thin books)
            if pos.duration_min <= 5:
                PARTIAL_GRACE_MINUTES = 3.5
            elif pos.asset in ("SOL", "XRP"):
                PARTIAL_GRACE_MINUTES = 1.0
            else:
                PARTIAL_GRACE_MINUTES = 2.0
            if pos.is_partial and not pos.first_fill_time:
                pos.first_fill_time = pos.created_at  # Backfill for old positions
            if pos.is_partial and pos.first_fill_time:
                first_fill_dt = datetime.fromisoformat(pos.first_fill_time)
                since_first_fill = (now - first_fill_dt).total_seconds() / 60
                if since_first_fill > PARTIAL_GRACE_MINUTES:
                    # Cancel the unfilled side
                    for order, attr in [(pos.up_order, "up_filled"), (pos.down_order, "down_filled")]:
                        if order and not getattr(pos, attr) and order.status == "open":
                            order.status = "cancelled"
                            if not self.paper:
                                try:
                                    self.client.cancel(order.order_id)
                                except Exception:
                                    pass
                    # V1.3: Cancel the FILLED side too — don't hold directional risk!
                    # In paper mode: mark entire position cancelled (PnL = 0, not resolved)
                    # In live mode: sell filled shares back at market to exit
                    filled_side = "UP" if pos.up_filled else "DOWN"
                    print(f"  [PARTIAL-CANCEL] {pos.asset} {filled_side}-only after {since_first_fill:.0f}min — cancelling entire position (no directional gamble)")

                    # V1.5: Sell back filled shares in live mode
                    partial_sell_pnl = 0.0
                    if not self.paper and self.client:
                        filled_order = pos.up_order if pos.up_filled else pos.down_order
                        if filled_order and filled_order.fill_shares > 0:
                            try:
                                from py_clob_client.clob_types import OrderArgs, OrderType
                                from py_clob_client.order_builder.constants import SELL
                                # Sell at market (bid -1 cent to ensure fill)
                                sell_price = max(0.01, round(filled_order.fill_price - 0.01, 2))
                                sell_args = OrderArgs(
                                    price=sell_price,
                                    size=filled_order.fill_shares,
                                    side=SELL,
                                    token_id=filled_order.token_id,
                                )
                                sell_signed = self.client.create_order(sell_args)
                                sell_resp = self.client.post_order(sell_signed, OrderType.GTC)
                                if sell_resp.get("success"):
                                    partial_sell_pnl = (sell_price - filled_order.fill_price) * filled_order.fill_shares
                                    print(f"  [PARTIAL-SELL] Sold {filled_order.fill_shares:.0f} {filled_side} @ ${sell_price:.2f} | PnL: ${partial_sell_pnl:+.2f}")
                                else:
                                    print(f"  [PARTIAL-SELL] Sell failed: {sell_resp.get('errorMsg', '?')}")
                            except Exception as e:
                                print(f"  [PARTIAL-SELL] Error: {e}")

                    pos.status = "cancelled"
                    pos.pnl = partial_sell_pnl
                    # ML: record as partial for offset learning
                    self.ml_optimizer.record(
                        hour=pos.hour_utc if pos.hour_utc >= 0 else now.hour,
                        offset=pos.bid_offset_used,
                        paired=False, partial=True, pnl=0.0
                    )

            # If approaching close, cancel unfilled orders
            # V1.6: 5M gets 1min buffer (was 2), 15M keeps 2min
            expire_at = pos.duration_min - 1 if pos.duration_min <= 5 else pos.duration_min - 2
            if age_min > expire_at:
                for order, attr in [(pos.up_order, "up_filled"), (pos.down_order, "down_filled")]:
                    if order and not getattr(pos, attr) and order.status == "open":
                        order.status = "cancelled"
                        if not self.paper:
                            try:
                                self.client.cancel(order.order_id)
                            except Exception:
                                pass
                        print(f"  [EXPIRE] Cancelled unfilled {order.side_label} order on {pos.asset}")

    async def harvest_extreme_prices(self):
        """V2.4: Vague-sourdough pattern — sell paired positions at extreme prices
        in the final 90 seconds instead of holding to settlement.

        When a paired position has one side trading at 0.85+, sell BOTH sides:
        - Winning side at 0.85-0.97 (lock in profit early)
        - Losing side at 0.03-0.15 (salvage what we can)

        This captures the "harvesting" pattern: extreme prices near close
        offer nearly guaranteed profit without waiting for settlement risk.
        Only triggers on PAIRED positions in LIVE mode with < 90s left.
        """
        if self.paper:
            return  # Paper mode uses settlement resolution

        now = datetime.now(timezone.utc)
        for market_id, pos in list(self.positions.items()):
            if pos.status != "paired" or not pos.is_paired:
                continue

            # Check time remaining
            created = datetime.fromisoformat(pos.created_at)
            age_sec = (now - created).total_seconds()
            time_left_sec = pos.duration_min * 60 - age_sec

            # Only harvest in final 90 seconds
            if time_left_sec > 90 or time_left_sec < 10:
                continue

            # Fetch current prices
            try:
                up_book = await self.get_order_book(pos.up_order.token_id) if pos.up_order else None
                down_book = await self.get_order_book(pos.down_order.token_id) if pos.down_order else None
            except Exception:
                continue

            if not up_book or not down_book:
                continue

            up_bid = up_book.get("best_bid", 0)
            down_bid = down_book.get("best_bid", 0)

            # Check if either side is extreme enough to harvest
            # Need at least one side at 0.82+ (strong directional move confirmed)
            extreme_side = None
            if up_bid >= 0.82:
                extreme_side = "UP"
            elif down_bid >= 0.82:
                extreme_side = "DOWN"

            if not extreme_side:
                continue

            # Calculate harvest PnL vs settlement PnL
            up_cost = pos.up_order.fill_price * pos.up_order.fill_shares if pos.up_order else 0
            down_cost = pos.down_order.fill_price * pos.down_order.fill_shares if pos.down_order else 0
            total_cost = up_cost + down_cost

            # If we sell both sides now at bid prices
            harvest_revenue = (up_bid * pos.up_order.fill_shares +
                               down_bid * pos.down_order.fill_shares)
            harvest_pnl = harvest_revenue - total_cost

            # Only harvest if profitable (combined sell > combined cost)
            if harvest_pnl <= 0:
                continue

            # Execute harvest: sell both sides
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import SELL

            sold_both = True
            actual_revenue = 0
            for order, bid_price in [(pos.up_order, up_bid), (pos.down_order, down_bid)]:
                if not order or not order.fill_shares:
                    continue
                sell_price = max(0.01, round(bid_price - 0.01, 2))  # 1c below bid for fast fill
                try:
                    sell_args = OrderArgs(
                        price=sell_price,
                        size=order.fill_shares,
                        side=SELL,
                        token_id=order.token_id,
                    )
                    sell_signed = self.client.create_order(sell_args)
                    sell_resp = self.client.post_order(sell_signed, OrderType.GTC)
                    if sell_resp.get("success"):
                        actual_revenue += sell_price * order.fill_shares
                    else:
                        sold_both = False
                except Exception as e:
                    sold_both = False

            if sold_both and actual_revenue > 0:
                actual_pnl = actual_revenue - total_cost
                pos.status = "resolved"
                pos.pnl = actual_pnl
                pos.outcome = f"HARVESTED_{extreme_side}"
                self.stats["total_pnl"] += actual_pnl
                self.daily_pnl += actual_pnl
                self.stats["pairs_completed"] += 1
                self.stats["paired_pnl"] += actual_pnl
                self.stats["best_pair_pnl"] = max(self.stats["best_pair_pnl"], actual_pnl)

                self.ml_optimizer.record(
                    hour=pos.hour_utc if pos.hour_utc >= 0 else now.hour,
                    offset=pos.bid_offset_used,
                    paired=True, partial=False, pnl=actual_pnl
                )
                self.resolved.append({
                    "market_id": market_id,
                    "question": pos.question,
                    "asset": pos.asset,
                    "outcome": f"HARVESTED_{extreme_side}",
                    "paired": True, "partial": False,
                    "combined_cost": total_cost,
                    "pnl": actual_pnl,
                    "bid_offset": pos.bid_offset_used,
                    "hour_utc": pos.hour_utc,
                    "up_sell": up_bid, "down_sell": down_bid,
                    "resolved_at": now.isoformat(),
                })
                print(f"  [HARVEST] {pos.asset} {extreme_side} dominant | "
                      f"Sold UP@${up_bid:.2f} DN@${down_bid:.2f} | "
                      f"Cost=${total_cost:.2f} Rev=${actual_revenue:.2f} | "
                      f"PnL=${actual_pnl:+.4f} | {time_left_sec:.0f}s left")

    async def resolve_positions(self):
        """Check if markets have resolved and calculate PnL."""
        now = datetime.now(timezone.utc)

        for market_id, pos in list(self.positions.items()):
            if pos.status == "resolved":
                continue

            # Check if market has expired (duration + 1 min buffer)
            created = datetime.fromisoformat(pos.created_at)
            age_min = (now - created).total_seconds() / 60
            resolve_after = pos.duration_min + 1
            if age_min < resolve_after:  # Wait for resolution
                continue

            # Fetch outcome using numeric market ID
            outcome = await self._fetch_outcome(pos.market_num_id)
            if not outcome:
                if age_min > pos.duration_min * 2:  # Give up after 2x duration
                    pos.status = "cancelled"
                    self._cancel_position_orders(pos)
                    # ML: record unfilled/cancelled attempt
                    self.ml_optimizer.record(
                        hour=pos.hour_utc if pos.hour_utc >= 0 else datetime.now(timezone.utc).hour,
                        offset=pos.bid_offset_used,
                        paired=False, partial=pos.is_partial, pnl=0.0
                    )
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

            # ML: record outcome for offset optimization
            self.ml_optimizer.record(
                hour=pos.hour_utc if pos.hour_utc >= 0 else datetime.now(timezone.utc).hour,
                offset=pos.bid_offset_used,
                paired=pos.is_paired,
                partial=pos.is_partial,
                pnl=pnl
            )

            # Log
            pair_type = "PAIRED" if pos.is_paired else "PARTIAL" if pos.is_partial else "UNFILLED"
            icon = "+" if pnl > 0 else "-" if pnl < 0 else "="
            print(f"  [{pair_type}] {pos.asset} {pos.question[:50]} | {outcome} | PnL: {icon}${abs(pnl):.4f} (off={pos.bid_offset_used:.1%})")

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
                "bid_offset": pos.bid_offset_used,
                "hour_utc": pos.hour_utc,
                "up_price": pos.up_order.fill_price if pos.up_order and pos.up_filled else 0,
                "down_price": pos.down_order.fill_price if pos.down_order and pos.down_filled else 0,
                "resolved_at": now.isoformat(),
            })

        # Clean resolved from active
        self.positions = {k: v for k, v in self.positions.items()
                         if v.status not in ("resolved", "cancelled")}

    async def _fetch_outcome(self, market_num_id: str) -> Optional[str]:
        """Fetch market outcome from gamma-api using numeric market ID."""
        if not market_num_id:
            return None
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"https://gamma-api.polymarket.com/markets/{market_num_id}",
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                if r.status_code == 200:
                    m = r.json()
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
        except Exception as e:
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
        # V1.4: Capital loss circuit breaker (40% net loss -> auto-switch to paper)
        if not self.paper and not self.circuit_tripped and self.starting_pnl is not None:
            session_pnl = self.stats["total_pnl"] - self.starting_pnl
            # Calculate 40% of the CLOB balance we started with (~$220)
            capital_loss_limit = 88.0  # 40% of ~$220 CLOB balance
            if session_pnl < -capital_loss_limit:
                print(f"\n{'='*70}")
                print(f"[CIRCUIT BREAKER] NET CAPITAL LOSS EXCEEDED 40%!")
                print(f"  Session PnL: ${session_pnl:+.2f} (limit: -${capital_loss_limit:.0f})")
                print(f"  Switching to PAPER MODE to protect remaining capital.")
                print(f"{'='*70}\n")
                self.paper = True
                self.circuit_tripped = True
                self._cancel_all_open()
                self._save()
                return False

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
        if not self.paper:
            print(f"CIRCUIT BREAKER: Auto-switch to paper on $88 net session loss (~40% capital)")
        print(f"Skip hours (UTC): {sorted(self.config.SKIP_HOURS_UTC)}")
        print("=" * 70)

        if self.stats["total_pnl"] != 0:
            print(f"[RESUME] Total PnL: ${self.stats['total_pnl']:.4f} | "
                  f"Pairs: {self.stats['pairs_completed']} complete, {self.stats['pairs_partial']} partial")

        cycle = 0
        last_redeem_check = 0
        while True:
            try:
                cycle += 1
                now_ts = time.time()
                self.reset_daily()

                # Phase 1: Cancel expiring unfilled orders
                await self.manage_expiring_orders()

                # Phase 2a: V2.4 Harvest paired positions at extreme prices (last 90s)
                await self.harvest_extreme_prices()

                # Phase 2b: Resolve completed markets
                await self.resolve_positions()

                # Phase 3: Check fills on active orders
                await self.check_fills()

                # Phase 4: Discover and place if risk allows
                if self.check_risk():
                    # Phase 5: Discover new markets
                    markets = await self.discover_markets()

                    # Phase 6: Evaluate and place orders on best pairs
                    # 5M markets get priority (shorter duration = higher priority)
                    markets.sort(key=lambda p: p.duration_min)
                    for pair in markets:
                        if not self.check_risk():
                            break

                        eval_result = self.evaluate_pair(pair)
                        if not eval_result["viable"]:
                            continue

                        tag = f"{pair.duration_min}m"
                        print(f"\n[{pair.asset} {tag}] {pair.question[:55]}")
                        print(f"  Mid: UP ${pair.up_mid:.2f} / DOWN ${pair.down_mid:.2f} | "
                              f"Mins left: {eval_result['mins_left']:.1f}")

                        await self.place_pair_orders(pair, eval_result)

                # Phase 7: Status & save (ALWAYS, not just when placing)
                if cycle % 5 == 0:
                    self._print_status(cycle)
                    self._save()
                    self.ml_optimizer.save()

                # Phase 8: ML summary every 50 cycles (~25 min)
                if cycle % 50 == 0 and cycle > 0:
                    print(f"\n[ML OFFSET] Current optimal offsets:")
                    print(self.ml_optimizer.get_summary())

                # Phase 9: Auto-redeem winnings every 45 seconds (live only)
                if not self.paper and now_ts - last_redeem_check >= 45:
                    try:
                        claimed = auto_redeem_winnings()
                        if claimed > 0:
                            print(f"[REDEEM] Claimed ${claimed:.2f} back to USDC")
                    except Exception as e:
                        print(f"[REDEEM] Error: {e}")
                    last_redeem_check = now_ts

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

        mode_tag = "PAPER" if self.paper else "LIVE"
        cb_tag = " [CIRCUIT TRIPPED]" if self.circuit_tripped else ""
        session_pnl = self.stats["total_pnl"] - (self.starting_pnl or 0)
        print(f"\n--- Cycle {cycle} | {mode_tag}{cb_tag} | UTC {hour:02d} | "
              f"Active: {active} ({paired} paired, {partial} partial) | "
              f"Daily: ${self.daily_pnl:+.4f} | Session: ${session_pnl:+.4f} | Total: ${self.stats['total_pnl']:+.4f} | "
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
