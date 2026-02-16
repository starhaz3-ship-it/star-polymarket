"""
Both-Sides Accumulation Maker for BTC 5-min Up/Down Markets (V4.1)

Strategy:
  Bid deep on BOTH Up and Down tokens ($0.47/$0.46 = combined $0.93).
  Natural pairs: 7.5% guaranteed edge ($0.21/pair on 6 shares).
  Partials: sell back at market after 120s timeout (small spread loss).
  No hedge. No rides. Cut partials fast.

  V4.1: Late-candle entry (MuseumOfBees strategy):
    At T+150s, direction is 70-80% clear. Buy the cheap LOSER side.
    Completes partials into cheap pairs ($0.67 combined vs $0.93 normal).
    Standalone loser bets for reversal upside ($0.20 risk, $0.80 reward).

  Inspired by MuseumOfBees + k9Q2 whale analysis:
    - Both-sides arb with directional tilt (75% correct side)
    - Late-candle entries when direction is 70-80% clear
    - Limit order accumulation (many small orders, not one big hit)
    - Merge recycling to free capital mid-market

  V4.1 vs V4: + late-candle loser buys, ML monitoring per entry type.
  V4 vs V3: deeper bids (7.5% vs 2% edge), BTC 5M only, sell partials.

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
import statistics
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
        # Limit to 5 conditions per attempt to conserve relayer quota (~100 units / 4h)
        results = svc.redeem_all(batch_size=5)

        # Restore original
        ProxyWeb3Service._submit_transactions = _original_submit

        if results:
            print(f"[REDEEM] Gasless OK: {len(results)} batch(es), ~${total:.0f}")
            # Cooldown 10 min between successful batches to spread quota over 4h window
            _relayer_quota_reset_at = time.time() + 600
            return True, total
        # Results empty but positions exist = relayer failed silently (poly_web3 swallowed exception)
        # Relayer quota resets every ~4h (14400s) — long cooldown to avoid wasting quota
        _relayer_quota_reset_at = time.time() + 14400
        print(f"[REDEEM] Gasless failed (quota?), backing off 4h")
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


_onchain_redeem_backoff_until = 0.0  # Cooldown for on-chain fallback

def auto_redeem_winnings():
    """Auto-redeem: try gasless relayer first, fall back to direct on-chain.
    Skips direct on-chain if gas > 200 gwei to preserve MATIC.
    Returns: float amount of USDC claimed (0.0 if nothing claimed)."""
    global _onchain_redeem_backoff_until
    # Try gasless relayer first (free, no MATIC needed)
    success, claimed = _try_gasless_redeem()
    if success:
        return claimed

    # Fallback: direct on-chain CTF contract call (with backoff to avoid log spam)
    if time.time() < _onchain_redeem_backoff_until:
        return 0.0

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

        # Check gas price — skip if too expensive (> 700 gwei base fee)
        gas_price = _redeem_w3.eth.gas_price
        gas_gwei = gas_price / 1e9
        if gas_gwei > 700:
            print(f"[REDEEM] Gas too high ({gas_gwei:.0f} gwei > 700), skipping")
            return 0.0

        # Check for stuck pending TXs
        try:
            nonce_latest = _redeem_w3.eth.get_transaction_count(_redeem_account.address, "latest")
            nonce_pending = _redeem_w3.eth.get_transaction_count(_redeem_account.address, "pending")
            if nonce_pending > nonce_latest:
                print(f"[REDEEM] Pending TX (nonce {nonce_latest}->{nonce_pending}), backing off 5min")
                _onchain_redeem_backoff_until = time.time() + 1800
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
        use_gas_price = min(int(gas_price * 1.5), _redeem_w3.to_wei(900, "gwei"))
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
        _onchain_redeem_backoff_until = time.time() + 1800  # Back off on errors too
        return 0.0


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MakerConfig:
    """Market maker configuration — V4.4 Avellaneda + Hummingbot patterns."""
    # Spread targets — V4.4: Avellaneda time-decay spreads
    MAX_COMBINED_PRICE: float = 0.92     # V4.3: 8% minimum edge (balanced pair rate vs edge)
    MIN_SPREAD_EDGE: float = 0.08        # V4.3: 8% minimum edge
    BID_OFFSET: float = 0.06            # V4.4: Fallback only — Avellaneda computes dynamic offset
    MIN_BID_OFFSET: float = 0.04         # V4.4: Avellaneda floor at 4c
    # V4.4: Avellaneda time-decay spread parameters
    BASE_SPREAD: float = 0.02           # Minimum 2c spread even at expiry
    DEFAULT_GAMMA: float = 0.25          # Risk aversion (ML-tuned). Higher = wider spreads

    # V4.3: Taker hedge — on first fill, FOK buy other side to guarantee pair
    TAKER_HEDGE_ENABLED: bool = True     # V4.3: Instant FOK hedge on first fill
    TAKER_HEDGE_MAX_COMBINED: float = 1.08  # V4.3: Allow 8c/share overpay — beats riding EV (-$0.88 at 37% WR)

    # Position sizing
    SIZE_PER_SIDE_USD: float = 5.0       # V4.3: $5/side (proven profitable, scaling up)
    MAX_PAIR_EXPOSURE: float = 20.0      # V4: $20/pair (scaled for $3/side)
    MAX_TOTAL_EXPOSURE: float = 90.0     # V4.3: $90 max (scaled for $111 account)
    MIN_SHARES: int = 5                  # CLOB minimum order size

    # Risk
    DAILY_LOSS_LIMIT: float = 15.0       # V4.2: $15 daily loss limit (4 assets × $5/side)
    MAX_CONCURRENT_PAIRS: int = 16       # V4.2: 4 assets × ~4 active markets
    MAX_SINGLE_SIDED: int = 8            # V4.2: 4 assets × 2 timeframes, allow partials to ride
    RIDE_AFTER_SEC: float = 120.0        # V4: 2min timeout on partials (was 45s)

    # V4: Partial fill handling
    PARTIAL_TIMEOUT_ACTION: str = "sell"  # V4: "sell" = dump filled side at market, "ride" = hold to resolution

    # Timing
    SCAN_INTERVAL: int = 10              # Seconds between market scans
    ORDER_REFRESH: int = 60              # V4.4: Seconds before checking stale orders for refresh
    ORDER_REFRESH_TOLERANCE: float = 0.02  # V4.4: Only re-quote if optimal price moved by $0.02+
    CLOSE_BUFFER_MIN: float = 2.0        # Cancel orders N min before close
    FILL_CHECK_INTERVAL: int = 10        # Seconds between fill checks
    MIN_TIME_LEFT_MIN: float = 5.0       # Don't enter markets with < 5 min left
    FILLED_ORDER_DELAY: float = 5.0      # V4.4: Pause new orders for 5s after any fill (adverse selection cooldown)
    RIDING_CUT_THRESHOLD: float = 0.20   # V4.4: Sell riding positions if token dropped 20c+ from entry

    # Session control (from PolyData analysis)
    SKIP_HOURS_UTC: set = field(default_factory=set)  # All hours active
    BOOST_HOURS_UTC: set = field(default_factory=lambda: {13, 14, 15, 16})  # NY open peak

    # Assets
    ASSETS: dict = field(default_factory=lambda: {
        "BTC": {"keywords": ["bitcoin", "btc"], "enabled": True},
        "ETH": {"keywords": ["ethereum", "eth"], "enabled": True},   # V4.2: Re-enabled for multi-asset deep bids
        "SOL": {"keywords": ["solana", "sol"], "enabled": True},    # V4.2: Re-enabled for multi-asset deep bids
        "XRP": {"keywords": ["xrp", "ripple"], "enabled": True},   # V4.2: Re-enabled for multi-asset deep bids
    })

    # V1.5: Per-asset risk tiers — SOL/XRP have thinner books, need tighter spreads
    # "max_combined" = maximum combined bid price (lower = more edge required)
    # "balance_range" = (min, max) for each side price (tighter = more balanced)
    ASSET_TIERS: dict = field(default_factory=lambda: {
        "BTC": {"max_combined": 0.92, "balance_range": (0.35, 0.65)},  # V4.3: Moderate offset
        "ETH": {"max_combined": 0.92, "balance_range": (0.35, 0.65)},  # V4.3: Moderate offset
        "SOL": {"max_combined": 0.92, "balance_range": (0.35, 0.65)},  # V4.3: Moderate offset
        "XRP": {"max_combined": 0.92, "balance_range": (0.35, 0.65)},  # V4.3: Moderate offset
    })

    # V3.0: Momentum directional trading (@vague-sourdough reverse-engineered strategy)
    # Watch Binance BTC for first 2-3 min of each 5-min round. If momentum > threshold,
    # buy the predicted direction. Backtested: 72-90% accuracy depending on threshold.
    MOMENTUM_ENABLED: bool = False    # V3.1: disabled — 25% WR, -$11.05 live
    MOMENTUM_SIZE_USD: float = 5.0       # $ per directional trade (base)
    MOMENTUM_CONFIRMED_SIZE_USD: float = 10.0  # $ when V5 regime confirms (2x)
    MOMENTUM_THRESHOLD_BPS: float = 15.0 # Minimum momentum in basis points to trigger
    MOMENTUM_WAIT_SECONDS: float = 120.0 # Wait this long after market opens (2 min)
    MOMENTUM_MAX_WAIT_SECONDS: float = 210.0  # Don't enter after 3.5 min (too late)
    MOMENTUM_MAX_CONCURRENT: int = 3     # Max simultaneous momentum positions
    MOMENTUM_DAILY_LOSS_LIMIT: float = 10.0  # Separate daily loss limit for momentum

    # V12a: Order Flow (Taker Flow Imbalance) — non-overlapping with V3.0 momentum
    # Fires when taker buy volume is heavily skewed AND volume surges >= 2x average.
    # Only fires on bars where V3.0 momentum did NOT fire (different signal source).
    ORDERFLOW_ENABLED: bool = False   # V3.1: disabled with momentum
    ORDERFLOW_SIZE_USD: float = 5.0          # $ per order flow trade
    ORDERFLOW_BUY_RATIO_THRESHOLD: float = 0.56  # Min taker_buy / total_volume ratio
    ORDERFLOW_VOL_RATIO_MIN: float = 2.0     # Volume must be >= 2x rolling average
    ORDERFLOW_VOL_SMA_PERIOD: int = 20       # Rolling average period (1m bars)

    # Late Entry: 15m-only momentum strategy (LEGACY — disabled)
    LATE_ENTRY_ENABLED: bool = False   # V3.1: disabled — back to paper
    LATE_ENTRY_SIZE_USD: float = 5.0
    LATE_ENTRY_WAIT_SECONDS: float = 600.0    # 10 minutes
    LATE_ENTRY_MAX_WAIT_SECONDS: float = 750.0  # Don't enter after 12.5 min
    LATE_ENTRY_THRESHOLD_BPS: float = 15.0    # Minimum momentum in basis points

    # V4.1: Late-Candle Entry (MuseumOfBees strategy) — 5M markets
    # At T+150s, direction is 70-80% clear. Buy the CHEAP LOSER side.
    # Key insight: loser token drops to $0.10-$0.25. Buy it to:
    #   1. Complete partials into cheap pairs ($0.67 combined vs $0.93 normal)
    #   2. Standalone reversal bets (small loss if wrong, huge win if BTC reverses)
    LATE_CANDLE_ENABLED: bool = True
    LATE_CANDLE_WAIT_SEC: float = 90.0        # V4.2: 1.5 min into 5-min market (was 150s)
    LATE_CANDLE_MAX_WAIT_SEC: float = 240.0   # Stop at 4 min (60s before close)
    LATE_CANDLE_SIZE_USD: float = 5.0         # Same as normal maker
    LATE_CANDLE_MAX_LOSER_PRICE: float = 0.30 # Only buy loser if <= $0.30
    LATE_CANDLE_MIN_MOMENTUM_BPS: float = 15.0  # Direction must be this clear

    # V4.2: Late-winner buying (vague-sourdough strategy)
    # In the last 30-60s, buy the WINNER token at $0.85-$0.92.
    # Near-guaranteed $0.08-$0.15 profit per share. 90%+ WR.
    LATE_WINNER_ENABLED: bool = True
    LATE_WINNER_WINDOW_SEC: float = 120.0     # Start buying 120s before market close (12 chances at 10s cycles)
    LATE_WINNER_MIN_SEC: float = 10.0         # Stop buying 10s before close (FOK is instant)
    LATE_WINNER_MAX_BUY_PRICE: float = 0.95   # Don't pay more than $0.95 (min $0.05 edge)
    LATE_WINNER_MIN_BUY_PRICE: float = 0.75   # Token must be at least $0.75 (75%+ implied prob)
    LATE_WINNER_SIZE_USD: float = 5.0         # Same as normal maker


# ============================================================================
# ML OFFSET OPTIMIZER
# ============================================================================

import random

class OffsetOptimizer:
    """V4.4: ML-driven gamma optimizer for Avellaneda time-decay spreads.

    Instead of choosing raw offsets, the ML tunes the gamma (risk aversion)
    parameter of the Avellaneda-Stoikov formula. Higher gamma = wider spreads.
    Uses epsilon-greedy to explore gamma values and exploit the best per hour.
    """
    GAMMA_BUCKETS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]  # V4.4: gamma risk aversion values
    STATE_FILE = Path(__file__).parent / "maker_ml_state.json"
    EXPLORE_RATE = 0.15  # 15% explore, 85% exploit

    def __init__(self):
        # hour_stats[hour][gamma_str] = {attempts, paired, partial, unfilled, pnl}
        self.hour_stats: Dict[int, Dict[str, dict]] = {}
        self._init_stats()
        self._load()

    def _init_stats(self):
        for h in range(24):
            self.hour_stats[h] = {}
            for g in self.GAMMA_BUCKETS:
                key = f"{g:.3f}"
                self.hour_stats[h][key] = {
                    "attempts": 0, "paired": 0, "partial": 0,
                    "unfilled": 0, "total_pnl": 0.0
                }

    def _load(self):
        if self.STATE_FILE.exists():
            try:
                data = json.loads(self.STATE_FILE.read_text())
                for h_str, gammas in data.items():
                    h = int(h_str)
                    if h in self.hour_stats:
                        for g_key, stats in gammas.items():
                            if g_key in self.hour_stats[h]:
                                self.hour_stats[h][g_key].update(stats)
            except Exception:
                pass

    def save(self):
        try:
            self.STATE_FILE.write_text(json.dumps(
                {str(h): v for h, v in self.hour_stats.items()}, indent=2))
        except Exception:
            pass

    def get_gamma(self, hour: int) -> float:
        """Pick optimal gamma for this hour using epsilon-greedy."""
        stats = self.hour_stats.get(hour, {})

        # Need minimum data before exploiting
        total_attempts = sum(s["attempts"] for s in stats.values())
        if total_attempts < 10 or random.random() < self.EXPLORE_RATE:
            return random.choice(self.GAMMA_BUCKETS)

        # Exploit: pick gamma with best avg profit per attempt
        best_gamma = 0.25  # V4.4: default gamma
        best_score = -999.0
        for g in self.GAMMA_BUCKETS:
            key = f"{g:.3f}"
            s = stats.get(key, {})
            attempts = s.get("attempts", 0)
            if attempts < 3:
                continue
            avg_pnl = s.get("total_pnl", 0) / attempts
            if avg_pnl > best_score:
                best_score = avg_pnl
                best_gamma = g

        return best_gamma

    def record(self, hour: int, gamma: float, paired: bool, partial: bool, pnl: float):
        """Record outcome of a pair attempt."""
        # Snap to nearest gamma bucket
        closest = min(self.GAMMA_BUCKETS, key=lambda x: abs(x - gamma))
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
        """Return brief summary of optimal gamma per hour."""
        lines = []
        for h in range(24):
            stats = self.hour_stats.get(h, {})
            total = sum(s["attempts"] for s in stats.values())
            if total == 0:
                continue
            best_g = self.get_gamma(h)
            best_key = f"{best_g:.3f}"
            s = stats.get(best_key, {})
            att = s.get("attempts", 0)
            pnl = s.get("total_pnl", 0)
            pr = s.get("paired", 0)
            lines.append(f"  UTC {h:02d}: gamma={best_g:.2f} ({pr}/{att} paired, ${pnl:+.2f})")
        return "\n".join(lines) if lines else "  No data yet"


# ============================================================================
# BINANCE MOMENTUM TRACKER (V3.0 — @vague-sourdough strategy)
# ============================================================================

class BinanceMomentum:
    """Tracks Binance BTC price and generates momentum signals for 5-min Polymarket rounds.

    Strategy (reverse-engineered from @vague-sourdough, $38K profit in 3 days):
    1. When a 5-min Polymarket round opens, record Binance BTC price
    2. After 2 minutes, check current Binance BTC price
    3. If momentum > threshold (15-30bp), buy the direction on Polymarket
    4. The first 2-3 minutes of a 5-min bar predict the close 72-90% of the time

    This is a MOMENTUM CONTINUATION effect — once BTC starts moving in a direction
    within the first 2 minutes, it tends to continue for the full 5 minutes.
    """

    BINANCE_URL = "https://api.binance.com/api/v3/ticker/price"
    BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
    RESULTS_FILE = Path(__file__).parent / "momentum_results.json"

    def __init__(self):
        # market_id -> {"open_price": float, "open_time": datetime, "traded": bool}
        self.tracked_markets: Dict[str, dict] = {}
        self.last_btc_price: float = 0.0
        self.last_price_fetch: float = 0.0
        self.momentum_trades: List[dict] = []
        self.momentum_daily_pnl: float = 0.0
        self.momentum_daily_date: str = date_today()
        # V5 regime confirmation: rolling 5m candle history
        self.btc_5m_closes: List[float] = []
        self.btc_5m_highs: List[float] = []
        self.btc_5m_lows: List[float] = []
        self.last_5m_bar_ts: int = 0
        # V12a order flow: rolling 1m volume + taker buy tracking
        self.vol_1m_history: List[float] = []  # total volume per 1m bar
        self.last_kline_fetch: float = 0.0
        self.last_kline_data: List[list] = []  # cached klines
        self._load()

    def _load(self):
        if self.RESULTS_FILE.exists():
            try:
                data = json.loads(self.RESULTS_FILE.read_text())
                self.momentum_trades = data.get("trades", [])
                self.momentum_daily_pnl = data.get("daily_pnl", 0.0)
                self.momentum_daily_date = data.get("daily_date", date_today())
            except Exception:
                pass

    def save(self):
        try:
            # Calculate stats
            trades = self.momentum_trades
            wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
            total = len(trades)
            total_pnl = sum((t.get("pnl") or 0) for t in trades)

            data = {
                "trades": trades[-500:],
                "daily_pnl": self.momentum_daily_pnl,
                "daily_date": self.momentum_daily_date,
                "stats": {
                    "total_trades": total,
                    "wins": wins,
                    "win_rate": wins / total * 100 if total > 0 else 0,
                    "total_pnl": total_pnl,
                    "avg_pnl": total_pnl / total if total > 0 else 0,
                },
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            self.RESULTS_FILE.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            print(f"[MOM] Save error: {e}")

    def reset_daily(self):
        today = date_today()
        if today != self.momentum_daily_date:
            print(f"[MOM] New day. Yesterday momentum PnL: ${self.momentum_daily_pnl:.2f}")
            self.momentum_daily_pnl = 0.0
            self.momentum_daily_date = today

    async def fetch_btc_price(self) -> float:
        """Fetch current BTC price from Binance. Cached for 5 seconds."""
        now = time.time()
        if now - self.last_price_fetch < 5 and self.last_btc_price > 0:
            return self.last_btc_price

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(self.BINANCE_URL, params={"symbol": "BTCUSDT"})
                if r.status_code == 200:
                    self.last_btc_price = float(r.json()["price"])
                    self.last_price_fetch = now
        except Exception as e:
            if self.last_btc_price == 0:
                print(f"[MOM] Binance price fetch error: {e}")
        return self.last_btc_price

    def track_market_open(self, market_id: str, btc_price: float, created_at: datetime,
                           market_num_id: str = "", duration_min: int = 5):
        """Record the Binance BTC price when a Polymarket market is first seen."""
        if market_id not in self.tracked_markets:
            self.tracked_markets[market_id] = {
                "open_price": btc_price,
                "open_time": created_at,
                "traded": False,
                "signal": None,
                "market_num_id": market_num_id,
                "duration_min": duration_min,
            }

    def get_momentum_signal(self, market_id: str, current_price: float,
                             config: 'MakerConfig') -> Optional[str]:
        """Check if momentum signal fires for a tracked market.

        Returns: "UP", "DOWN", or None (no signal / already traded / too early/late)
        """
        tracking = self.tracked_markets.get(market_id)
        if not tracking or tracking["traded"]:
            return None

        open_price = tracking["open_price"]
        open_time = tracking["open_time"]
        now = datetime.now(timezone.utc)

        # How long since market opened
        elapsed_sec = (now - open_time).total_seconds()

        # Too early — need to wait for momentum to develop
        if elapsed_sec < config.MOMENTUM_WAIT_SECONDS:
            return None

        # Too late — don't enter in the last 90 seconds
        if elapsed_sec > config.MOMENTUM_MAX_WAIT_SECONDS:
            return None

        # Calculate momentum in basis points
        if open_price <= 0:
            return None
        momentum_bps = (current_price - open_price) / open_price * 10000

        # Check threshold
        if abs(momentum_bps) < config.MOMENTUM_THRESHOLD_BPS:
            return None

        signal = "UP" if momentum_bps > 0 else "DOWN"
        tracking["signal"] = signal
        tracking["momentum_bps"] = momentum_bps
        tracking["signal_time"] = now.isoformat()
        tracking["signal_price"] = current_price
        return signal

    def mark_traded(self, market_id: str):
        """Mark a market as traded (prevents re-entry)."""
        if market_id in self.tracked_markets:
            self.tracked_markets[market_id]["traded"] = True

    def record_trade(self, market_id: str, direction: str, buy_price: float,
                     shares: float, cost_usd: float, momentum_bps: float,
                     question: str, market_num_id: str = ""):
        """Record a momentum trade for tracking."""
        now = datetime.now(timezone.utc)
        tracking = self.tracked_markets.get(market_id, {})
        self.momentum_trades.append({
            "market_id": market_id,
            "market_num_id": market_num_id,
            "direction": direction,
            "buy_price": buy_price,
            "shares": shares,
            "cost_usd": cost_usd,
            "momentum_bps": momentum_bps,
            "question": question,
            "placed_at": now.isoformat(),
            "hour_utc": now.hour,
            "minute_utc": now.minute,
            "btc_open_price": tracking.get("open_price", 0),
            "btc_signal_price": tracking.get("signal_price", 0),
            "elapsed_sec": (now - tracking["open_time"]).total_seconds() if "open_time" in tracking else 0,
            "outcome": None,
            "pnl": None,
        })

    def record_outcome(self, market_id: str, outcome: str, pnl: float):
        """Record the outcome of a momentum trade."""
        for trade in reversed(self.momentum_trades):
            if trade["market_id"] == market_id and trade["outcome"] is None:
                trade["outcome"] = outcome
                trade["pnl"] = pnl
                trade["resolved_at"] = datetime.now(timezone.utc).isoformat()
                self.momentum_daily_pnl += pnl
                break

    def update_5m_candle(self, price: float):
        """Append a 5m candle snapshot. Called every ~30s, we bucket by 5-min window."""
        now_ts = int(time.time())
        bar_ts = now_ts - (now_ts % 300)  # floor to 5-min boundary
        if bar_ts != self.last_5m_bar_ts:
            # New 5m bar — push previous bar's data
            if self.last_5m_bar_ts > 0 and price > 0:
                self.btc_5m_closes.append(price)
                self.btc_5m_highs.append(price)
                self.btc_5m_lows.append(price)
            self.last_5m_bar_ts = bar_ts
            # Keep last 50 bars
            self.btc_5m_closes = self.btc_5m_closes[-50:]
            self.btc_5m_highs = self.btc_5m_highs[-50:]
            self.btc_5m_lows = self.btc_5m_lows[-50:]
        else:
            # Update current bar's high/low
            if self.btc_5m_highs and price > 0:
                self.btc_5m_highs[-1] = max(self.btc_5m_highs[-1], price)
                self.btc_5m_lows[-1] = min(self.btc_5m_lows[-1], price)

    def v5_confirms(self, direction: str) -> bool:
        """V5 regime confirmation: ATR%, EMA trend, BB bandwidth, z-score.

        When this returns True, the momentum signal is higher confidence (90.6% vs 88.6% WR).
        Used to double position size.
        """
        closes = self.btc_5m_closes
        highs = self.btc_5m_highs
        lows = self.btc_5m_lows
        n = len(closes)
        if n < 25:  # need enough history for EMA21 + ATR14
            return False

        # ATR% (14-period)
        trs = [highs[0] - lows[0]]
        for i in range(1, n):
            tr = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
            trs.append(tr)
        # Simple EMA of TR
        a = 2.0 / 15.0
        atr_val = trs[0]
        for i in range(1, len(trs)):
            atr_val = a * trs[i] + (1 - a) * atr_val
        atrp = atr_val / closes[-1] if closes[-1] > 0 else 0

        if atrp < 0.0008 or atrp > 0.012:
            return False

        # EMA 9/21 trend
        a9 = 2.0 / 10.0
        a21 = 2.0 / 22.0
        ema9 = closes[0]
        ema21 = closes[0]
        for c in closes[1:]:
            ema9 = a9 * c + (1 - a9) * ema9
            ema21 = a21 * c + (1 - a21) * ema21

        if direction == "UP" and ema9 <= ema21:
            return False
        if direction == "DOWN" and ema9 >= ema21:
            return False

        # BB bandwidth
        if n >= 20:
            window = closes[-20:]
            mid = sum(window) / 20
            std = (sum((x - mid)**2 for x in window) / 20) ** 0.5
            bb_bw = (4.0 * std) / mid if mid > 0 else 0
            if bb_bw < 0.005 or bb_bw > 0.06:
                return False

        # Z-score of 10m momentum
        if n >= 12:
            rets = [(closes[i] - closes[i-1]) / closes[i-1]
                    for i in range(max(1, n-12), n) if closes[i-1] > 0]
            if len(rets) >= 2:
                vol = statistics.stdev(rets)
                if vol > 1e-8:
                    mom_10m = (closes[-1] - closes[-3]) / closes[-3] if n >= 3 and closes[-3] > 0 else 0
                    mom_z = mom_10m / vol
                    if abs(mom_z) < 0.8:
                        return False

        return True

    def cleanup_old(self):
        """Remove tracking data for markets > 30 minutes old."""
        now = datetime.now(timezone.utc)
        expired = [mid for mid, data in self.tracked_markets.items()
                    if (now - data["open_time"]).total_seconds() > 1800]
        for mid in expired:
            del self.tracked_markets[mid]

    def get_active_momentum_count(self) -> int:
        """Count how many momentum positions are currently open (unresolved)."""
        return sum(1 for t in self.momentum_trades
                   if t.get("outcome") is None
                   and t.get("placed_at", "")
                   and (datetime.now(timezone.utc) -
                        datetime.fromisoformat(t["placed_at"])).total_seconds() < 600)

    def get_stats_summary(self) -> str:
        """Return brief stats string."""
        trades = self.momentum_trades
        total = len(trades)
        resolved = [t for t in trades if t.get("outcome")]
        wins = sum(1 for t in resolved if (t.get("pnl") or 0) > 0)
        total_pnl = sum(t.get("pnl", 0) for t in resolved)
        wr = wins / len(resolved) * 100 if resolved else 0
        return (f"MOM: {len(resolved)}/{total} resolved | "
                f"WR: {wr:.0f}% | PnL: ${total_pnl:+.2f} | "
                f"Today: ${self.momentum_daily_pnl:+.2f}")

    # --- V12a Order Flow ---

    async def fetch_btc_klines(self, limit: int = 25) -> List[list]:
        """Fetch recent 1-minute BTCUSDT klines from Binance. Cached 10s."""
        now = time.time()
        if now - self.last_kline_fetch < 10 and self.last_kline_data:
            return self.last_kline_data
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(self.BINANCE_KLINES_URL, params={
                    "symbol": "BTCUSDT", "interval": "1m", "limit": limit
                })
                if r.status_code == 200:
                    self.last_kline_data = r.json()
                    self.last_kline_fetch = now
                    # Update rolling volume history from completed bars (exclude last = current)
                    if len(self.last_kline_data) > 1:
                        self.vol_1m_history = [float(k[5]) for k in self.last_kline_data[:-1]]
        except Exception as e:
            print(f"[FLOW] Kline fetch error: {e}")
        return self.last_kline_data

    def get_orderflow_signal(self, market_id: str, config: 'MakerConfig') -> Optional[str]:
        """V12a: Check taker flow imbalance on the current 5-min window.

        Uses the last 2 completed 1m bars (matching 2-min wait) to compute:
        - buy_ratio = sum(taker_buy_volume) / sum(total_volume)
        - vol_ratio = sum(total_volume of 2 bars) / avg(1m volume over last 20 bars)

        Returns "UP" if buy_ratio >= threshold and vol surge, "DOWN" if inverse, None otherwise.
        Only fires if V3.0 momentum did NOT fire on this market.
        """
        tracking = self.tracked_markets.get(market_id)
        if not tracking or tracking["traded"]:
            return None

        open_time = tracking["open_time"]
        now = datetime.now(timezone.utc)
        elapsed_sec = (now - open_time).total_seconds()

        # Same 2-min wait window as V3.0
        if elapsed_sec < config.MOMENTUM_WAIT_SECONDS:
            return None
        if elapsed_sec > config.MOMENTUM_MAX_WAIT_SECONDS:
            return None

        # Skip if V3.0 already fired on this bar (non-overlapping)
        if tracking.get("signal") is not None:
            return None

        klines = self.last_kline_data
        if not klines or len(klines) < 3:
            return None

        # Use last 2 completed bars (index -3 and -2, since -1 is current)
        recent_bars = klines[-3:-1]
        total_vol = sum(float(k[5]) for k in recent_bars)
        taker_buy_vol = sum(float(k[9]) for k in recent_bars)

        if total_vol <= 0:
            return None

        buy_ratio = taker_buy_vol / total_vol

        # Volume surge check: compare 2-bar volume to rolling avg
        vol_sma_period = config.ORDERFLOW_VOL_SMA_PERIOD
        if len(self.vol_1m_history) < vol_sma_period:
            return None
        avg_vol = sum(self.vol_1m_history[-vol_sma_period:]) / vol_sma_period
        if avg_vol <= 0:
            return None
        # Per-bar volume ratio (divide by 2 since we summed 2 bars)
        vol_ratio = (total_vol / 2.0) / avg_vol

        if vol_ratio < config.ORDERFLOW_VOL_RATIO_MIN:
            return None

        # Determine direction from flow imbalance
        if buy_ratio >= config.ORDERFLOW_BUY_RATIO_THRESHOLD:
            direction = "UP"
        elif (1.0 - buy_ratio) >= config.ORDERFLOW_BUY_RATIO_THRESHOLD:
            direction = "DOWN"
        else:
            return None

        tracking["flow_signal"] = direction
        tracking["flow_buy_ratio"] = buy_ratio
        tracking["flow_vol_ratio"] = vol_ratio
        tracking["flow_signal_time"] = now.isoformat()
        return direction

    def get_late_entry_signal(self, market_id: str, current_price: float,
                               config: 'MakerConfig') -> Optional[str]:
        """Late Entry: momentum signal for 15m markets.

        Waits 10 minutes into a 15m bar, then checks if BTC moved > threshold.
        By minute 10, the direction is locked in 93%+ of the time.
        Only fires on 15m markets (checked by caller).
        """
        tracking = self.tracked_markets.get(market_id)
        if not tracking or tracking["traded"]:
            return None

        open_price = tracking["open_price"]
        open_time = tracking["open_time"]
        now = datetime.now(timezone.utc)
        elapsed_sec = (now - open_time).total_seconds()

        # Must wait 10 minutes
        if elapsed_sec < config.LATE_ENTRY_WAIT_SECONDS:
            return None
        # Don't enter too late (last 2.5 min)
        if elapsed_sec > config.LATE_ENTRY_MAX_WAIT_SECONDS:
            return None

        if open_price <= 0:
            return None
        momentum_bps = (current_price - open_price) / open_price * 10000

        if abs(momentum_bps) < config.LATE_ENTRY_THRESHOLD_BPS:
            return None

        signal = "UP" if momentum_bps > 0 else "DOWN"
        tracking["signal"] = signal
        tracking["momentum_bps"] = momentum_bps
        tracking["signal_time"] = now.isoformat()
        tracking["signal_price"] = current_price
        tracking["signal_type"] = "LATE"
        return signal

    def get_late_candle_signal(self, market_id: str, current_price: float,
                                config: 'MakerConfig') -> Optional[dict]:
        """V4.1 Late-Candle: detect clear direction for 5M markets at T+150s.

        Unlike old late entry (buys winner), this identifies the LOSER side
        for cheap buying. MuseumOfBees strategy: buy dirt cheap loser token.

        Returns dict: {winner, loser, momentum_bps, elapsed_sec} or None.
        """
        tracking = self.tracked_markets.get(market_id)
        if not tracking:
            return None

        open_price = tracking["open_price"]
        open_time = tracking["open_time"]
        now = datetime.now(timezone.utc)
        elapsed_sec = (now - open_time).total_seconds()

        if elapsed_sec < config.LATE_CANDLE_WAIT_SEC:
            return None
        if elapsed_sec > config.LATE_CANDLE_MAX_WAIT_SEC:
            return None
        if open_price <= 0:
            return None

        momentum_bps = (current_price - open_price) / open_price * 10000

        if abs(momentum_bps) < config.LATE_CANDLE_MIN_MOMENTUM_BPS:
            return None

        winner = "UP" if momentum_bps > 0 else "DOWN"
        loser = "DOWN" if momentum_bps > 0 else "UP"

        return {
            "winner": winner,
            "loser": loser,
            "momentum_bps": momentum_bps,
            "elapsed_sec": elapsed_sec,
        }


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
    placed_at_ts: float = 0.0  # V4.4: Unix timestamp for order age checks
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
    # V4.1: Late-candle tracking
    entry_type: str = "maker"    # "maker", "late_candle_pair", "late_candle_standalone"
    late_candle_pending: bool = False  # True when late-candle order placed, awaiting fill
    loser_buy_price: float = 0.0      # Price paid for loser side (for ML)
    sell_attempts: int = 0             # V4.1: Track sell-back attempts (ride after 3 failures)
    last_ride_check_ts: float = 0.0    # V4.4: Rate limit ride monitoring API calls
    gamma_used: float = 0.25           # V4.4: Track which gamma was used (for ML)

    @property
    def is_paired(self) -> bool:
        return self.up_filled and self.down_filled

    @property
    def is_partial(self) -> bool:
        return (self.up_filled or self.down_filled) and not self.is_paired


# ============================================================================
# SHADOW ORDER TRACKER (V4.4: duplicate fill prevention)
# ============================================================================

class ShadowTracker:
    """Cache recently-processed fills to detect duplicates on restart.
    Prevents double-entry when on-chain sync misses in-transit fills."""

    def __init__(self, ttl_sec: int = 180):
        self._fills: Dict[str, float] = {}  # order_id -> timestamp
        self.ttl = ttl_sec

    def is_duplicate(self, order_id: str) -> bool:
        return order_id in self._fills

    def record(self, order_id: str):
        self._fills[order_id] = time.time()

    def cleanup(self):
        cutoff = time.time() - self.ttl
        self._fills = {k: v for k, v in self._fills.items() if v > cutoff}


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
        self.momentum = BinanceMomentum() if (self.config.MOMENTUM_ENABLED or self.config.LATE_CANDLE_ENABLED) else None

        # State
        self.positions: Dict[str, PairPosition] = {}  # market_id -> PairPosition
        self.active_orders: Dict[str, MakerOrder] = {}  # order_id -> MakerOrder
        self.resolved: List[dict] = []  # Completed pair records
        self._entered_markets: set = set()  # All condition_ids entered this session (never re-enter)
        self._entered_markets_file = Path(__file__).parent / "maker_entered_markets.json"

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
            # V4.1: Late-candle stats
            "late_candle_attempts": 0,
            "late_candle_paired": 0,       # Partials completed via late candle
            "late_candle_standalone": 0,    # Standalone loser-side bets
            "late_candle_pnl": 0.0,
            # V4.2: Late-winner stats
            "late_winner_attempts": 0,
            "late_winner_wins": 0,
            "late_winner_pnl": 0.0,
            # V4.3: Taker hedge stats
            "hedge_attempts": 0,
            "hedge_successes": 0,
            "hedge_failures": 0,
            "version": "V4.4",
        }

        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_date = date_today()

        # V1.4: Capital loss circuit breaker (40% net loss -> auto-switch to paper)
        self.circuit_breaker_pct = 0.40  # 40% of starting capital
        self.starting_pnl = None  # Set after _load() from total_pnl at launch
        self.circuit_tripped = False

        # V4.4: New state
        self.shadow = ShadowTracker()      # Duplicate fill prevention
        self._last_fill_time: float = 0.0  # Filled order delay

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
                # V2.6c: Cancel any orphaned orders from previous session
                try:
                    # First check what open orders exist (to track their markets)
                    try:
                        open_orders = self.client.get_orders() or []
                        live_orders = [o for o in open_orders
                                       if isinstance(o, dict) and o.get("status") in ("live", "open")]
                        for o in live_orders:
                            asset_id = o.get("asset_id", "")
                            # We can't easily map token_id back to condition_id here,
                            # but cancel_all will remove them. The on-chain sync handles blocking.
                        if live_orders:
                            print(f"[MAKER] Found {len(live_orders)} orphan orders to cancel")
                    except Exception:
                        pass
                    self.client.cancel_all()
                    print("[MAKER] Cancelled all orphaned orders from previous session")
                except Exception as e:
                    print(f"[MAKER] Cancel orphans warning: {e}")
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
                        # Always track in entered set (prevents re-entry even after cleanup)
                        self._entered_markets.add(pos.market_id)
                    except Exception:
                        continue  # Skip corrupt entries
                # Also track recently resolved markets from history
                for rec in self.resolved[-100:]:
                    mid = rec.get("market_id", "")
                    if mid:
                        self._entered_markets.add(mid)
                print(f"[MAKER] Loaded {len(self.resolved)} resolved, {len(self.positions)} active")
            except Exception as e:
                print(f"[MAKER] Load error: {e}")

        # CRITICAL: Query on-chain positions to prevent stacking after restart
        self._sync_onchain_positions()

        # V4.2: Load persisted entered-markets (survives restarts)
        self._load_entered_markets()

    def _load_entered_markets(self):
        """V4.2: Load persisted entered-markets from disk. Prevents restart duplicates."""
        try:
            if self._entered_markets_file.exists():
                data = json.loads(self._entered_markets_file.read_text())
                # Only keep entries from the last 24h (markets expire)
                now_ts = datetime.now(timezone.utc).timestamp()
                fresh = {cid for cid, ts in data.items() if now_ts - ts < 86400}
                before = len(self._entered_markets)
                self._entered_markets.update(fresh)
                added = len(self._entered_markets) - before
                if added > 0:
                    print(f"[MAKER] Loaded {added} persisted entered-markets from disk")
        except Exception as e:
            print(f"[MAKER] entered-markets load error: {e}")

    def _persist_entered_market(self, market_id: str):
        """V4.2: Immediately persist a market entry to disk. Crash-safe."""
        self._entered_markets.add(market_id)
        try:
            # Load existing, merge, write back
            data = {}
            if self._entered_markets_file.exists():
                data = json.loads(self._entered_markets_file.read_text())
            data[market_id] = datetime.now(timezone.utc).timestamp()
            # Prune entries older than 24h
            now_ts = datetime.now(timezone.utc).timestamp()
            data = {k: v for k, v in data.items() if now_ts - v < 86400}
            self._entered_markets_file.write_text(json.dumps(data))
        except Exception:
            pass  # Non-critical, on-chain sync is the fallback

    def _sync_onchain_positions(self):
        """Query Polymarket data-api for ALL held positions and block those markets.
        This prevents stacking after restarts — even if maker_results.json is wiped,
        we discover what we already hold on-chain and refuse to re-enter."""
        proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
        if not proxy:
            return
        try:
            r = httpx.get("https://data-api.polymarket.com/positions",
                          params={"user": proxy, "sizeThreshold": 0}, timeout=15)
            if r.status_code != 200:
                print(f"[MAKER] On-chain sync failed: HTTP {r.status_code}")
                return
            positions = r.json()
            blocked = 0
            for p in positions:
                size = float(p.get("size", 0))
                price = float(p.get("curPrice", p.get("price", 0)))
                cid = p.get("conditionId", "")
                # Block ANY market where we hold shares (live or dead)
                # Dead positions (price=0) are resolved but still in wallet
                if size > 0 and cid and cid not in self._entered_markets:
                    self._entered_markets.add(cid)
                    blocked += 1
            if blocked > 0:
                print(f"[MAKER] On-chain sync: blocked {blocked} already-held markets "
                      f"(total blocked: {len(self._entered_markets)})")
            else:
                print(f"[MAKER] On-chain sync: no new markets to block "
                      f"(total blocked: {len(self._entered_markets)})")
        except Exception as e:
            print(f"[MAKER] On-chain sync error: {e}")

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
            for tag_slug, duration in [("5M", 5), ("15M", 15)]:  # V4.2: 5M BTC + 15M all assets
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

                        # V4.2: All assets enabled (was V1.7 hard block on SOL/XRP)

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

                            # V2.2/V3.1: Parse actual duration from title, match to expected tag duration
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
                            if actual_dur != duration:
                                continue  # Skip markets that don't match expected duration

                            pair = self._parse_market(m, event, matched_asset, actual_dur)
                            if pair:
                                pairs.append(pair)

                except Exception as e:
                    print(f"[DISCOVER] {tag_slug} Error: {e}")

        return pairs

    def _parse_market(self, market: dict, event: dict, asset: str, duration: int = 15,
                      skip_entered_check: bool = False) -> Optional[MarketPair]:
        """Parse a market into a MarketPair."""
        try:
            condition_id = market.get("conditionId", "")
            market_num_id = str(market.get("id", ""))  # Numeric ID for gamma-api lookup
            if not condition_id:
                return None

            # Skip markets we already entered this session (prevents re-entry after sell-back cleanup)
            if not skip_entered_check:
                if condition_id in self._entered_markets or condition_id in self.positions:
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

        def _entry_price(entry) -> float:
            """Extract price from order book entry (dict or OrderSummary object)."""
            if isinstance(entry, dict):
                return float(entry.get("price", 0))
            return float(getattr(entry, 'price', 0))

        def _entry_size(entry) -> float:
            """Extract size from order book entry (dict or OrderSummary object)."""
            if isinstance(entry, dict):
                return float(entry.get("size", 0))
            return float(getattr(entry, 'size', 0))

        try:
            book = self.client.get_order_book(token_id)
            bids = book.get("bids", []) if isinstance(book, dict) else getattr(book, 'bids', [])
            asks = book.get("asks", []) if isinstance(book, dict) else getattr(book, 'asks', [])

            best_bid = _entry_price(bids[0]) if bids else 0.0
            best_ask = _entry_price(asks[0]) if asks else 1.0
            mid = (best_bid + best_ask) / 2 if best_bid > 0 else best_ask
            spread = best_ask - best_bid

            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid": mid,
                "spread": spread,
                "bid_depth": sum(_entry_size(b) for b in bids[:3]),
                "ask_depth": sum(_entry_size(a) for a in asks[:3]),
            }
        except Exception as e:
            print(f"  [BOOK-ERR] get_order_book failed: {str(e)[:80]}")
            return None

    def _get_bid_offset(self, mid_price: float = 0.5, secs_left: float = 300, total_secs: float = 900) -> Tuple[float, float]:
        """V4.4: Avellaneda time-decay offset. Returns (offset, gamma_used).

        Formula: offset = gamma * sqrt(mid*(1-mid)) * (secs_left/total_secs) + BASE_SPREAD
        - Binary vol = sqrt(p*(1-p)): widest at 50/50, tightest at extremes
        - Time fraction decays toward expiry, naturally tightening spreads
        - ML tunes gamma (risk aversion) via epsilon-greedy
        """
        hour = datetime.now(timezone.utc).hour
        gamma = self.ml_optimizer.get_gamma(hour)

        # Binary volatility — widest at 0.50 (vol=0.50), tightest near 0/1
        mid_clamped = max(0.05, min(0.95, mid_price))
        vol = math.sqrt(mid_clamped * (1.0 - mid_clamped))

        # Time decay — wider early, tighter near expiry
        time_frac = max(0.0, min(1.0, secs_left / max(1, total_secs)))

        # Avellaneda spread
        offset = gamma * vol * time_frac + self.config.BASE_SPREAD

        # Clamp to [MIN_BID_OFFSET, 0.15]
        offset = max(self.config.MIN_BID_OFFSET, min(0.15, offset))
        return (offset, gamma)

    def evaluate_pair(self, pair: MarketPair) -> dict:
        """Evaluate a market pair for maker opportunity. V4.4: Avellaneda time-decay offsets."""
        # Time remaining (seconds and minutes)
        mins_left = float(pair.duration_min)
        secs_left = pair.duration_min * 60.0
        if pair.end_time:
            delta = (pair.end_time - datetime.now(timezone.utc)).total_seconds()
            secs_left = max(0, delta)
            mins_left = secs_left / 60.0

        total_secs = pair.duration_min * 60.0

        # V4.4: Avellaneda time-decay offset — adapts to mid price, time remaining, and ML gamma
        avg_mid = (pair.up_mid + pair.down_mid) / 2.0  # Should be ~0.50 for balanced markets
        offset, gamma_used = self._get_bid_offset(avg_mid, secs_left, total_secs)

        # Our target bid prices — symmetric offsets on both sides for max edge
        up_bid = round(pair.up_mid - offset, 2)
        down_bid = round(pair.down_mid - offset, 2)

        # Ensure prices are valid
        up_bid = max(0.01, min(0.95, up_bid))
        down_bid = max(0.01, min(0.95, down_bid))

        combined = up_bid + down_bid
        edge = 1.0 - combined

        # Min time left scales with duration: 5min markets need 2min, 15min need 5min
        min_time = 2.0 if pair.duration_min <= 5 else self.config.MIN_TIME_LEFT_MIN
        # V1.5: Max time left — don't lock capital in far-future markets
        max_time = pair.duration_min * 1.5

        # V1.5: Per-asset tier thresholds
        tier = self.config.ASSET_TIERS.get(pair.asset, {"max_combined": 0.80, "balance_range": (0.25, 0.75)})
        max_combined = tier["max_combined"]
        bal_min, bal_max = tier["balance_range"]

        return {
            "up_bid": up_bid,
            "down_bid": down_bid,
            "combined": combined,
            "edge": edge,
            "edge_pct": edge / combined * 100 if combined > 0 else 0,
            "offset": offset,
            "gamma": gamma_used,
            "mins_left": mins_left,
            "secs_left": secs_left,
            "total_secs": total_secs,
            "viable": (
                combined < max_combined
                and edge >= self.config.MIN_SPREAD_EDGE
                and mins_left > min_time
                and mins_left < max_time
                and up_bid >= bal_min
                and down_bid >= bal_min
                and up_bid <= bal_max
                and down_bid <= bal_max
            ),
        }

    # ========================================================================
    # V4.4: INVENTORY SKEW
    # ========================================================================

    def _get_inventory_skew(self) -> Tuple[float, float]:
        """V4.4: Adjust per-side sizes based on current YES/NO inventory imbalance.
        Returns (up_mult, down_mult) multipliers for SIZE_PER_SIDE_USD.
        If overweight UP: reduce UP bids, increase DOWN bids (and vice versa)."""
        total_up = 0.0
        total_down = 0.0
        for pos in self.positions.values():
            if pos.status in ("resolved", "cancelled"):
                continue
            if pos.up_filled and pos.up_order:
                total_up += pos.up_order.fill_shares
            if pos.down_filled and pos.down_order:
                total_down += pos.down_order.fill_shares
        total = total_up + total_down
        if total == 0:
            return (1.0, 1.0)
        up_ratio = total_up / total
        # Linear interpolation: at 50/50 both 1.0x, at 75/25 UP=0.5x DOWN=1.5x
        up_mult = max(0.5, min(1.5, 2.0 - 2.0 * up_ratio))
        down_mult = max(0.5, min(1.5, 2.0 * up_ratio))
        return (up_mult, down_mult)

    # ========================================================================
    # ORDER PLACEMENT
    # ========================================================================

    async def place_pair_orders(self, pair: MarketPair, eval_result: dict) -> Optional[PairPosition]:
        """Place maker orders on both sides of a market pair."""

        # HARD SAFETY: Double-check we haven't already entered this market
        if pair.market_id in self._entered_markets or pair.market_id in self.positions:
            print(f"  [SAFETY] BLOCKED duplicate entry on {pair.question[:40]} — already in _entered_markets")
            return None

        up_bid = eval_result["up_bid"]
        down_bid = eval_result["down_bid"]

        # V4.4: Inventory skew — adjust sizes based on directional imbalance
        up_skew, down_skew = self._get_inventory_skew()
        up_size = self.config.SIZE_PER_SIDE_USD * up_skew
        down_size = self.config.SIZE_PER_SIDE_USD * down_skew

        # Calculate shares for each side (with skew-adjusted sizes)
        up_shares = max(self.config.MIN_SHARES, math.floor(up_size / up_bid))
        down_shares = max(self.config.MIN_SHARES, math.floor(down_size / down_bid))

        # Cap shares to skew-adjusted size
        if up_bid * up_shares > up_size:
            up_shares = max(self.config.MIN_SHARES, math.floor(up_size / up_bid))
        if down_bid * down_shares > down_size:
            down_shares = max(self.config.MIN_SHARES, math.floor(down_size / down_bid))

        # Create position tracker
        pos = PairPosition(
            market_id=pair.market_id,
            market_num_id=pair.market_num_id,
            question=pair.question,
            asset=pair.asset,
            created_at=datetime.now(timezone.utc).isoformat(),
            status="pending",
            bid_offset_used=eval_result.get("offset", self.config.BID_OFFSET),
            gamma_used=eval_result.get("gamma", self.config.DEFAULT_GAMMA),
            hour_utc=datetime.now(timezone.utc).hour,
            duration_min=pair.duration_min,
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        now_ts = time.time()

        # V4.4: Log inventory skew if non-trivial
        if abs(up_skew - 1.0) > 0.05 or abs(down_skew - 1.0) > 0.05:
            print(f"  [SKEW] UP {up_skew:.1f}x (${up_size:.1f}) / DOWN {down_skew:.1f}x (${down_size:.1f})")

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
                placed_at_ts=now_ts,
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
                placed_at_ts=now_ts,
            )
            print(f"  [PAPER] Placed UP bid ${up_bid:.2f} x {up_shares} + DOWN bid ${down_bid:.2f} x {down_shares}")
            print(f"          Combined: ${up_bid + down_bid:.4f} | Edge: {eval_result['edge_pct']:.1f}% | γ={eval_result.get('gamma', 0):.2f}")
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
                    placed_at_ts=now_ts,
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
                    placed_at_ts=now_ts,
                )

                print(f"  [LIVE] Placed UP {up_order_id[:16]}... @ ${up_bid:.2f} x {up_shares}")
                print(f"  [LIVE] Placed DN {dn_order_id[:16]}... @ ${down_bid:.2f} x {down_shares}")
                print(f"         Combined: ${up_bid + down_bid:.4f} | Edge: {eval_result['edge_pct']:.1f}% | γ={eval_result.get('gamma', 0):.2f}")

            except Exception as e:
                print(f"  [LIVE] Order error: {e}")
                return None

        # Track (add to permanent session set so resolved cleanup can't cause re-entry)
        self.positions[pair.market_id] = pos
        self._persist_entered_market(pair.market_id)
        self.stats["pairs_attempted"] += 1
        return pos

    # ========================================================================
    # FILL MONITORING
    # ========================================================================

    async def check_fills(self):
        """Check order fill status for all active positions."""
        for market_id, pos in list(self.positions.items()):
            if pos.status in ("resolved", "cancelled", "riding"):
                continue

            prev_up = pos.up_filled
            prev_down = pos.down_filled

            if self.paper:
                await self._check_fills_paper(pos)
            else:
                await self._check_fills_live(pos)

            # V4.4: Track last fill time for filled order delay
            if (pos.up_filled and not prev_up) or (pos.down_filled and not prev_down):
                self._last_fill_time = time.time()

            # Update position status
            if pos.up_filled and pos.down_filled:
                pos.status = "paired"
                pos.combined_cost = (
                    (pos.up_order.fill_price * pos.up_order.fill_shares if pos.up_order else 0) +
                    (pos.down_order.fill_price * pos.down_order.fill_shares if pos.down_order else 0)
                )
            elif pos.up_filled or pos.down_filled:
                was_pending = pos.status == "pending"
                pos.status = "partial"
                # V1.3: Track when first side filled for fast partial-protect
                if not pos.first_fill_time:
                    pos.first_fill_time = datetime.now(timezone.utc).isoformat()
                # V4.3: Instant taker hedge — FOK buy other side on first detection
                if was_pending and self.config.TAKER_HEDGE_ENABLED and not self.paper:
                    await self._taker_hedge(pos)

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
        """Check real order fills via CLOB API. V4.4: Shadow duplicate detection."""
        if not self.client:
            return

        for order, attr in [(pos.up_order, "up_filled"), (pos.down_order, "down_filled")]:
            if not order or getattr(pos, attr):
                continue

            # V4.4: Skip if already processed (shadow cache)
            if self.shadow.is_duplicate(order.order_id):
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
                        self.shadow.record(order.order_id)  # V4.4: prevent duplicate processing
                        print(f"  [FILL] {pos.asset} {order.side_label} @ ${order.price:.2f} x {matched:.0f}")
            except Exception:
                pass

    # ========================================================================
    # V4.3: TAKER HEDGE
    # ========================================================================

    async def _taker_hedge(self, pos: PairPosition):
        """V4.3: When one side fills, immediately FOK-buy the other side to guarantee a pair.

        This eliminates partial-ride coin flips. Instead of losing ~$3-5 when the wrong
        side fills, we lock in a guaranteed pair with known edge.

        Flow:
        1. Cancel the unfilled GTC order
        2. Get order book for the unfilled side's token
        3. FOK buy at best ask (up to max_combined limit)
        4. If FOK fills -> paired, guaranteed profit
        5. If FOK fails -> ride as partial (no worse than before)
        """
        if not self.client:
            return

        # V4.3: Only hedge if enough time remains — late fills mean the market has moved
        created = datetime.fromisoformat(pos.created_at)
        age_min = (datetime.now(timezone.utc) - created).total_seconds() / 60
        time_left = pos.duration_min - age_min
        if time_left < 3.0:
            print(f"  [HEDGE-SKIP] {pos.asset} — only {time_left:.1f}min left, too late to hedge")
            return

        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        # Determine which side filled and which needs hedging
        if pos.up_filled and not pos.down_filled:
            filled_order = pos.up_order
            hedge_side = "DOWN"
            hedge_token_id = pos.down_order.token_id if pos.down_order else None
            unfilled_order = pos.down_order
        elif pos.down_filled and not pos.up_filled:
            filled_order = pos.down_order
            hedge_side = "UP"
            hedge_token_id = pos.up_order.token_id if pos.up_order else None
            unfilled_order = pos.up_order
        else:
            return  # Both filled or neither — shouldn't happen

        if not hedge_token_id or not filled_order:
            return

        self.stats["hedge_attempts"] = self.stats.get("hedge_attempts", 0) + 1

        # Step 1: Cancel the unfilled GTC order
        if unfilled_order and unfilled_order.status == "open":
            try:
                self.client.cancel(unfilled_order.order_id)
                unfilled_order.status = "cancelled"
            except Exception:
                pass  # May already be cancelled/expired

        # Step 2: Get order book for hedge side
        book = await self.get_order_book(hedge_token_id)
        if not book or book.get("best_ask", 1.0) >= 1.0:
            print(f"  [HEDGE-SKIP] {pos.asset} {hedge_side} — no asks available, will ride")
            self.stats["hedge_failures"] = self.stats.get("hedge_failures", 0) + 1
            return

        best_ask = book["best_ask"]
        maker_fill_price = filled_order.fill_price

        # Step 3: Check if combined cost is acceptable
        combined = maker_fill_price + best_ask
        overpay = max(0, combined - 1.0)
        overpay_total = overpay * filled_order.fill_shares
        if combined > self.config.TAKER_HEDGE_MAX_COMBINED:
            print(f"  [HEDGE-SKIP] {pos.asset} {hedge_side} — combined ${combined:.2f} "
                  f"(overpay ${overpay_total:.2f}), limit ${self.config.TAKER_HEDGE_MAX_COMBINED}, will ride")
            self.stats["hedge_failures"] = self.stats.get("hedge_failures", 0) + 1
            return

        # Hedging at a small loss is better than riding at 37% partial WR
        if combined > 1.0:
            print(f"  [HEDGE-OVERPAY] {pos.asset} {hedge_side} — combined ${combined:.2f}, "
                  f"overpay ${overpay_total:.2f} (beats riding EV -${0.88 * filled_order.fill_shares / 11:.2f})")

        edge_pct = (1.0 - combined) * 100
        # Match shares to filled side for balanced pair
        hedge_shares = filled_order.fill_shares
        hedge_price = round(best_ask, 2)

        # Step 4: Place FOK buy order
        try:
            args = OrderArgs(
                price=hedge_price,
                size=float(hedge_shares),
                side=BUY,
                token_id=hedge_token_id,
            )
            signed = self.client.create_order(args)
            resp = self.client.post_order(signed, OrderType.FOK)

            if resp.get("success"):
                order_id = resp.get("orderID", "")
                now_iso = datetime.now(timezone.utc).isoformat()
                new_order = MakerOrder(
                    order_id=order_id,
                    market_id=pos.market_id,
                    token_id=hedge_token_id,
                    side_label=hedge_side,
                    price=hedge_price,
                    size_shares=float(hedge_shares),
                    size_usd=hedge_price * hedge_shares,
                    placed_at=now_iso,
                    status="filled",
                    fill_price=hedge_price,
                    fill_shares=float(hedge_shares),
                )
                if hedge_side == "DOWN":
                    pos.down_order = new_order
                    pos.down_filled = True
                else:
                    pos.up_order = new_order
                    pos.up_filled = True

                pos.combined_cost = (maker_fill_price * filled_order.fill_shares) + (hedge_price * hedge_shares)
                pos.status = "paired"
                pos.entry_type = "taker_hedge"

                self.stats["hedge_successes"] = self.stats.get("hedge_successes", 0) + 1
                self.stats["fills_" + hedge_side.lower()] += 1

                print(f"  [HEDGE-OK] {pos.asset} FOK {hedge_side} @ ${hedge_price:.2f} x {hedge_shares:.0f} | "
                      f"Combined: ${combined:.2f} | Edge: {edge_pct:.1f}% | oid={order_id[:16]}...")
            else:
                err = resp.get("errorMsg", "?")
                print(f"  [HEDGE-FAIL] {pos.asset} {hedge_side} FOK rejected ({err}), will ride")
                self.stats["hedge_failures"] = self.stats.get("hedge_failures", 0) + 1
        except Exception as e:
            print(f"  [HEDGE-ERR] {pos.asset} {hedge_side}: {str(e)[:60]}, will ride")
            self.stats["hedge_failures"] = self.stats.get("hedge_failures", 0) + 1

    # ========================================================================
    # RESOLUTION
    # ========================================================================

    async def manage_expiring_orders(self):
        """Cancel unfilled orders on markets about to close."""
        now = datetime.now(timezone.utc)
        for market_id, pos in list(self.positions.items()):
            if pos.status in ("resolved", "cancelled", "riding"):
                continue

            created = datetime.fromisoformat(pos.created_at)
            age_min = (now - created).total_seconds() / 60

            # V3.3: NO HEDGE — ride unpaired fills to resolution.
            # Data showed hedging was #1 PnL destroyer: -$0.60/hedge guaranteed loss.
            # Rides at $0.48 are +EV at 50% WR (+$0.40 EV). Accept variance, kill bleed.
            if pos.is_partial and not pos.first_fill_time:
                pos.first_fill_time = pos.created_at  # Backfill for old positions
            if pos.is_partial and pos.first_fill_time:
                first_fill_dt = datetime.fromisoformat(pos.first_fill_time)
                since_first_fill_sec = (now - first_fill_dt).total_seconds()
                # V4.1: Duration-aware sell-back — 5M markets need faster action
                # 120s on a 5M market = token already near-worthless
                sell_timeout = 45.0 if pos.duration_min <= 5 else self.config.RIDE_AFTER_SEC
                if since_first_fill_sec > sell_timeout:
                    # Cancel the unfilled passive order
                    for order, attr in [(pos.up_order, "up_filled"), (pos.down_order, "down_filled")]:
                        if order and not getattr(pos, attr) and order.status == "open":
                            order.status = "cancelled"
                            if not self.paper:
                                try:
                                    self.client.cancel(order.order_id)
                                except Exception:
                                    pass

                    # Re-check fills after cancel — the passive order may have filled
                    # between our last fill check and the cancel (race condition).
                    if not self.paper:
                        await self._check_fills_live(pos)
                        if pos.up_filled and pos.down_filled:
                            pos.combined_cost = (
                                (pos.up_order.fill_price * pos.up_order.fill_shares if pos.up_order else 0) +
                                (pos.down_order.fill_price * pos.down_order.fill_shares if pos.down_order else 0)
                            )
                            pos.status = "paired"
                            print(f"  [NATURAL-PAIR] {pos.asset} — both sides filled! No ride needed")
                            continue

                    filled_side = "UP" if pos.up_filled else "DOWN"
                    filled_order = pos.up_order if pos.up_filled else pos.down_order
                    shares = filled_order.fill_shares if filled_order else 0

                    # V4.1: Skip sell-back if late-candle order is pending
                    if pos.late_candle_pending:
                        continue

                    # V4: Sell-on-timeout — dump filled side at market to cut losses.
                    # Small guaranteed loss (spread) is better than 50/50 coin flip on entire stake.
                    if self.config.PARTIAL_TIMEOUT_ACTION == "sell" and not self.paper and filled_order:
                        sell_pnl = await self._sell_partial_at_market(pos, filled_order)
                        if sell_pnl is not None:
                            pos.status = "resolved"
                            pos.pnl = sell_pnl
                            pos.outcome = f"SOLD_{filled_side}"
                            self.stats["total_pnl"] += sell_pnl
                            self.daily_pnl += sell_pnl
                            self.stats["pairs_partial"] += 1
                            self.stats["partial_pnl"] += sell_pnl
                            self.ml_optimizer.record(
                                hour=pos.hour_utc if pos.hour_utc >= 0 else now.hour,
                                gamma=pos.gamma_used,
                                paired=False, partial=True, pnl=sell_pnl
                            )
                            print(f"  [SELL-CUT] {pos.asset} {filled_side} {shares:.0f}sh — "
                                  f"sold at market | PnL: ${sell_pnl:+.4f}")
                            continue
                        else:
                            # V4.1: Retry sell-back across cycles — don't permanently ride on transient failure
                            pos.sell_attempts += 1
                            if pos.sell_attempts < 3:
                                print(f"  [SELL-FAIL] {pos.asset} {filled_side} — "
                                      f"sell failed (attempt {pos.sell_attempts}/3), will retry next cycle")
                                continue  # Stay as partial, try again next cycle
                            else:
                                print(f"  [SELL-FAIL] {pos.asset} {filled_side} — "
                                      f"sell failed 3x, falling back to ride")

                    # Fallback: ride to resolution (V3.3 behavior)
                    print(f"  [RIDE] {pos.asset} {filled_side} {shares:.0f}sh @ ${filled_order.fill_price:.2f} — "
                          f"riding to resolution (win=${(1.0 - filled_order.fill_price) * shares:.2f}, "
                          f"lose=${filled_order.fill_price * shares:.2f})")
                    pos.status = "riding"
                    self.stats["pairs_partial"] += 1
                    self.ml_optimizer.record(
                        hour=pos.hour_utc if pos.hour_utc >= 0 else now.hour,
                        gamma=pos.gamma_used,
                        paired=False, partial=True, pnl=0.0
                    )

            # V4.4: Order refresh with tolerance — re-quote stale pending orders
            # Only for fully-unfilled positions (both sides still open)
            if (pos.status == "pending" and not pos.up_filled and not pos.down_filled
                    and pos.up_order and pos.down_order
                    and pos.up_order.placed_at_ts > 0):
                order_age = (now - datetime.fromisoformat(pos.created_at)).total_seconds()
                if order_age > self.config.ORDER_REFRESH:
                    # Recalculate optimal offset with current Avellaneda params
                    total_secs = pos.duration_min * 60.0
                    secs_left = max(0, total_secs - order_age)
                    avg_mid = (pos.up_order.price + pos.bid_offset_used + pos.down_order.price + pos.bid_offset_used) / 2.0
                    new_offset, _ = self._get_bid_offset(avg_mid, secs_left, total_secs)
                    if abs(new_offset - pos.bid_offset_used) > self.config.ORDER_REFRESH_TOLERANCE:
                        # Cancel both orders and allow re-entry
                        for order in [pos.up_order, pos.down_order]:
                            if order and order.status == "open":
                                order.status = "cancelled"
                                if not self.paper:
                                    try:
                                        self.client.cancel(order.order_id)
                                    except Exception:
                                        pass
                        print(f"  [REFRESH] {pos.asset} — offset shifted {pos.bid_offset_used:.2f}→{new_offset:.2f}, re-quoting")
                        self._entered_markets.discard(market_id)
                        del self.positions[market_id]
                        continue  # Skip expiry check for this (now deleted) position

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

    async def manage_riding_positions(self):
        """V4.4: Monitor riding positions and sell early if price moved too far against us.
        Frees locked capital for new profitable pairs."""
        now = datetime.now(timezone.utc)
        for market_id, pos in list(self.positions.items()):
            if pos.status != "riding":
                continue

            # Rate limit: max 1 check per 30s per position
            if time.time() - pos.last_ride_check_ts < 30:
                continue

            # Determine which side is filled
            filled_order = pos.up_order if pos.up_filled else pos.down_order
            if not filled_order or not filled_order.fill_price:
                continue

            # Check time remaining — don't sell if near resolution
            created = datetime.fromisoformat(pos.created_at)
            age_sec = (now - created).total_seconds()
            secs_left = pos.duration_min * 60.0 - age_sec
            if secs_left < 180:  # < 3 min left, just ride it out
                continue

            pos.last_ride_check_ts = time.time()

            # Get current best bid for our filled token
            try:
                book = await self.get_order_book(filled_order.token_id)
                if not book or not book.get("best_bid"):
                    continue
                best_bid = book["best_bid"]
            except Exception:
                continue

            # If token dropped 20c+ from our entry, sell to free capital
            if best_bid < filled_order.fill_price - self.config.RIDING_CUT_THRESHOLD:
                if best_bid < 0.10:
                    continue  # Too cheap to sell, just ride

                sell_pnl = await self._sell_partial_at_market(pos, filled_order)
                if sell_pnl is not None:
                    pos.status = "resolved"
                    pos.pnl = sell_pnl
                    pos.outcome = f"RIDE_CUT_{filled_order.side_label}"
                    self.stats["total_pnl"] += sell_pnl
                    self.daily_pnl += sell_pnl
                    self.stats["partial_pnl"] += sell_pnl
                    print(f"  [RIDE-CUT] {pos.asset} {filled_order.side_label} — "
                          f"dropped {filled_order.fill_price:.2f}→{best_bid:.2f}, "
                          f"sold to free capital | PnL: ${sell_pnl:+.4f}")

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
            from py_clob_client.clob_types import OrderArgs, OrderType, BalanceAllowanceParams, AssetType
            from py_clob_client.order_builder.constants import SELL

            sold_both = True
            actual_revenue = 0
            for order, bid_price in [(pos.up_order, up_bid), (pos.down_order, down_bid)]:
                if not order or not order.fill_shares:
                    continue
                sell_price = max(0.01, round(bid_price - 0.01, 2))  # 1c below bid for fast fill
                try:
                    # Approve conditional token for selling
                    try:
                        self.client.update_balance_allowance(
                            BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=order.token_id))
                    except Exception:
                        pass
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
                    gamma=pos.gamma_used,
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

    # ========================================================================
    # MOMENTUM DIRECTIONAL TRADING (V3.0 — @vague-sourdough)
    # ========================================================================

    async def run_momentum_scanner(self, markets: List[MarketPair]):
        """Scan for momentum signals on active 5-min BTC markets.

        Flow:
        1. For each active 5-min BTC market, record Binance price at market open
        2. After MOMENTUM_WAIT_SECONDS, check if BTC moved > MOMENTUM_THRESHOLD_BPS
        3. If yes, place directional BUY order on predicted winning side
        """
        if not self.momentum or not self.config.MOMENTUM_ENABLED:
            return

        # Lifetime circuit breaker: pause if <30% WR AND -$20 running PnL
        resolved = [t for t in self.momentum.momentum_trades if t.get("outcome") is not None]
        if len(resolved) >= 5:  # need minimum sample
            wins = sum(1 for t in resolved if t.get("pnl", 0) > 0)
            wr = wins / len(resolved)
            total_pnl = sum(t.get("pnl", 0) for t in resolved)
            if wr < 0.30 and total_pnl < -20.0:
                print(f"  [MOM PAUSED] WR {wr:.0%} < 30% and PnL ${total_pnl:.2f} < -$20 — momentum disabled")
                self.config.MOMENTUM_ENABLED = False
                return

        # Fetch current BTC price + klines (for order flow)
        btc_price = await self.momentum.fetch_btc_price()
        if btc_price <= 0:
            return

        # V12a: fetch 1m klines for taker volume data
        if self.config.ORDERFLOW_ENABLED:
            await self.momentum.fetch_btc_klines(limit=self.config.ORDERFLOW_VOL_SMA_PERIOD + 5)

        self.momentum.reset_daily()
        self.momentum.update_5m_candle(btc_price)  # V5 regime tracking

        # Check daily loss limit for momentum
        if self.momentum.momentum_daily_pnl < -self.config.MOMENTUM_DAILY_LOSS_LIMIT:
            return

        # Track newly discovered markets (5m for V3.0/V12a, 15m for Late Entry)
        now = datetime.now(timezone.utc)
        allowed_durations = {5}
        if self.config.LATE_ENTRY_ENABLED:
            allowed_durations.add(15)

        for pair in markets:
            if pair.asset != "BTC" or pair.duration_min not in allowed_durations:
                continue

            # Compute approximate start time from end_time
            if pair.end_time:
                start_time = pair.end_time - timedelta(minutes=pair.duration_min)
            else:
                start_time = now  # Fallback: treat as just opened

            self.momentum.track_market_open(pair.market_id, btc_price, start_time,
                                              market_num_id=pair.market_num_id,
                                              duration_min=pair.duration_min)

        # Also track markets we already have positions on (from maker)
        for mid, pos in self.positions.items():
            if pos.asset == "BTC" and pos.duration_min in allowed_durations:
                created = datetime.fromisoformat(pos.created_at)
                self.momentum.track_market_open(mid, btc_price, created,
                                                market_num_id=pos.market_num_id,
                                                duration_min=pos.duration_min)

        # Check for signals on tracked markets
        active_mom = self.momentum.get_active_momentum_count()
        if active_mom >= self.config.MOMENTUM_MAX_CONCURRENT:
            return

        for market_id, tracking in list(self.momentum.tracked_markets.items()):
            if tracking["traded"]:
                continue
            if tracking.get("duration_min", 5) != 5:
                continue  # V3.0 momentum is 5m only
            if active_mom >= self.config.MOMENTUM_MAX_CONCURRENT:
                break

            signal = self.momentum.get_momentum_signal(market_id, btc_price, self.config)
            if not signal:
                continue

            # Find the matching MarketPair for token IDs
            target_pair = None
            for pair in markets:
                if pair.market_id == market_id:
                    target_pair = pair
                    break

            if not target_pair:
                # Market might have been discovered earlier but not in current scan
                continue

            # V5 regime confirmation -> double size
            confirmed = self.momentum.v5_confirms(signal)
            mom_bps = tracking.get("momentum_bps", 0)
            await self._place_momentum_order(target_pair, signal, mom_bps, confirmed)
            active_mom += 1

        # V12a: Order flow pass — fires on 5m markets where V3.0 did NOT fire
        if self.config.ORDERFLOW_ENABLED:
            for market_id, tracking in list(self.momentum.tracked_markets.items()):
                if tracking["traded"]:
                    continue
                if tracking.get("duration_min", 5) != 5:
                    continue  # V12a is 5m only
                if active_mom >= self.config.MOMENTUM_MAX_CONCURRENT:
                    break

                flow_signal = self.momentum.get_orderflow_signal(market_id, self.config)
                if not flow_signal:
                    continue

                # Find matching MarketPair
                target_pair = None
                for pair in markets:
                    if pair.market_id == market_id:
                        target_pair = pair
                        break
                if not target_pair:
                    continue

                buy_ratio = tracking.get("flow_buy_ratio", 0)
                vol_ratio = tracking.get("flow_vol_ratio", 0)
                print(f"  [FLOW] BTC {flow_signal} | buy_ratio={buy_ratio:.3f} vol={vol_ratio:.1f}x")
                await self._place_orderflow_order(target_pair, flow_signal, buy_ratio, vol_ratio)
                active_mom += 1

        # Late Entry: 15m markets only — wait 10 min then check momentum
        if self.config.LATE_ENTRY_ENABLED:
            for market_id, tracking in list(self.momentum.tracked_markets.items()):
                if tracking["traded"]:
                    continue
                if tracking.get("duration_min", 5) != 15:
                    continue  # Late entry is 15m only
                if active_mom >= self.config.MOMENTUM_MAX_CONCURRENT:
                    break

                late_signal = self.momentum.get_late_entry_signal(
                    market_id, btc_price, self.config)
                if not late_signal:
                    continue

                target_pair = None
                for pair in markets:
                    if pair.market_id == market_id:
                        target_pair = pair
                        break
                if not target_pair:
                    continue

                mom_bps = tracking.get("momentum_bps", 0)
                print(f"  [LATE] BTC 15m {late_signal} | {mom_bps:+.1f}bp @ 10min")
                await self._place_late_entry_order(target_pair, late_signal, mom_bps)
                active_mom += 1

        # Cleanup old tracking data
        self.momentum.cleanup_old()

    async def _place_late_entry_order(self, pair: MarketPair, direction: str,
                                      momentum_bps: float):
        """Place a directional BUY on 15m market based on late entry signal (10-min momentum)."""
        token_id = pair.up_token_id if direction == "UP" else pair.down_token_id
        mid_price = pair.up_mid if direction == "UP" else pair.down_mid

        buy_price = round(min(0.95, mid_price + 0.01), 2)
        buy_price = max(0.05, buy_price)

        size_usd = self.config.LATE_ENTRY_SIZE_USD
        shares = max(self.config.MIN_SHARES, math.floor(size_usd / buy_price))
        cost_usd = buy_price * shares

        if self.paper:
            order_id = f"late_{direction.lower()}_{pair.market_id[:12]}_{int(time.time())}"
            print(f"  [LATE PAPER] {pair.asset} 15m {direction} @ ${buy_price:.2f} x {shares} "
                  f"(${cost_usd:.2f}) | momentum: {momentum_bps:+.1f}bp")
        else:
            try:
                from py_clob_client.clob_types import OrderArgs, OrderType
                from py_clob_client.order_builder.constants import BUY

                args = OrderArgs(
                    price=buy_price,
                    size=float(shares),
                    side=BUY,
                    token_id=token_id,
                )
                signed = self.client.create_order(args)
                resp = self.client.post_order(signed, OrderType.GTC)

                if not resp.get("success"):
                    print(f"  [LATE LIVE] Order failed: {resp.get('errorMsg', '?')}")
                    return

                order_id = resp.get("orderID", "")
                print(f"  [LATE LIVE] {pair.asset} 15m {direction} @ ${buy_price:.2f} x {shares} "
                      f"(${cost_usd:.2f}) | momentum: {momentum_bps:+.1f}bp | oid={order_id[:16]}...")
            except Exception as e:
                print(f"  [LATE LIVE] Error: {e}")
                return

        self.momentum.mark_traded(pair.market_id)
        self.momentum.record_trade(
            market_id=pair.market_id,
            direction=direction,
            buy_price=buy_price,
            shares=shares,
            cost_usd=cost_usd,
            momentum_bps=momentum_bps,
            question=pair.question,
            market_num_id=pair.market_num_id,
        )
        if self.momentum.momentum_trades:
            self.momentum.momentum_trades[-1]["signal_type"] = "LATE_15m"

        self._persist_entered_market(pair.market_id)

    # ========================================================================
    # V4.1: LATE-CANDLE ENTRY (MuseumOfBees strategy)
    # ========================================================================

    async def run_late_candle_scanner(self, markets: List[MarketPair]):
        """V4.1: Scan for late-candle opportunities on 5M BTC markets.

        At T+150s, direction is 70-80% clear. The LOSER token is dirt cheap.
        Two modes:
          A) Complete existing partials — if we have winner side filled, buy cheap loser
          B) Standalone — buy loser side on markets we haven't entered
        """
        if not self.momentum:
            return

        btc_price = await self.momentum.fetch_btc_price()
        if btc_price <= 0:
            return

        # Auto-disable: if late candle WR < 30% after 20+ resolved trades
        lc_resolved = [r for r in self.resolved if r.get("entry_type", "").startswith("late_candle")]
        if len(lc_resolved) >= 20:
            lc_wins = sum(1 for r in lc_resolved if r.get("pnl", 0) > 0)
            lc_wr = lc_wins / len(lc_resolved)
            if lc_wr < 0.30:
                lc_pnl = sum(r.get("pnl", 0) for r in lc_resolved)
                print(f"  [LC AUTO-OFF] Late candle WR {lc_wr:.0%} < 30% over {len(lc_resolved)} trades "
                      f"(PnL: ${lc_pnl:+.2f}) — disabling")
                self.config.LATE_CANDLE_ENABLED = False
                return

        now = datetime.now(timezone.utc)

        # Ensure markets are tracked in momentum tracker
        for pair in markets:
            if pair.asset == "BTC" and pair.duration_min == 5:
                start_time = pair.end_time - timedelta(minutes=5) if pair.end_time else now
                self.momentum.track_market_open(pair.market_id, btc_price, start_time,
                                                market_num_id=pair.market_num_id, duration_min=5)

        # Also track markets we have positions on
        for mid, pos in self.positions.items():
            if pos.asset == "BTC" and pos.duration_min == 5:
                created = datetime.fromisoformat(pos.created_at)
                self.momentum.track_market_open(mid, btc_price, created,
                                                market_num_id=pos.market_num_id, duration_min=5)

        # --- Case A: Complete existing partial positions ---
        for market_id, pos in list(self.positions.items()):
            if pos.late_candle_pending or pos.entry_type != "maker":
                continue
            if not pos.is_partial or pos.status in ("resolved", "cancelled"):
                continue
            if pos.asset != "BTC" or pos.duration_min != 5:
                continue

            signal = self.momentum.get_late_candle_signal(market_id, btc_price, self.config)
            if not signal:
                continue

            # Check: is our filled side the WINNER side?
            filled_side = "UP" if pos.up_filled else "DOWN"
            if filled_side != signal["winner"]:
                continue  # We hold the loser — can't pair cheaply

            loser_side = signal["loser"]
            unfilled_order = pos.down_order if loser_side == "DOWN" else pos.up_order
            if not unfilled_order:
                continue
            loser_token_id = unfilled_order.token_id

            # Get loser price
            if not self.paper:
                book = await self.get_order_book(loser_token_id)
                if not book:
                    continue
                loser_price = book.get("best_ask", 0)
                if loser_price <= 0:
                    loser_price = book.get("mid", 0)
            else:
                # Paper: estimate from momentum. Stronger momentum = cheaper loser.
                loser_price = round(max(0.05, 0.50 - abs(signal["momentum_bps"]) / 100), 2)

            if loser_price <= 0 or loser_price > self.config.LATE_CANDLE_MAX_LOSER_PRICE:
                continue

            # Place the order to complete the pair
            success = await self._place_late_candle_order(
                pos, loser_side, loser_token_id, loser_price, signal, is_pair_completion=True)
            if success:
                self.stats["late_candle_attempts"] = self.stats.get("late_candle_attempts", 0) + 1
                self.stats["late_candle_paired"] = self.stats.get("late_candle_paired", 0) + 1

        # --- Case B: Standalone loser-side bets on unentered markets ---
        for pair in markets:
            if pair.market_id in self._entered_markets or pair.market_id in self.positions:
                continue
            if pair.asset != "BTC" or pair.duration_min != 5:
                continue

            signal = self.momentum.get_late_candle_signal(pair.market_id, btc_price, self.config)
            if not signal:
                continue

            loser_side = signal["loser"]
            loser_token_id = pair.down_token_id if loser_side == "DOWN" else pair.up_token_id
            loser_mid = pair.down_mid if loser_side == "DOWN" else pair.up_mid

            if loser_mid > self.config.LATE_CANDLE_MAX_LOSER_PRICE:
                continue

            # Create a standalone position for the loser side
            success = await self._place_standalone_late_candle(
                pair, loser_side, loser_token_id, loser_mid, signal)
            if success:
                self.stats["late_candle_attempts"] = self.stats.get("late_candle_attempts", 0) + 1
                self.stats["late_candle_standalone"] = self.stats.get("late_candle_standalone", 0) + 1

        self.momentum.cleanup_old()

    async def _place_late_candle_order(self, pos: PairPosition, loser_side: str,
                                        loser_token_id: str, loser_price: float,
                                        signal: dict, is_pair_completion: bool = True) -> bool:
        """V4.1: Buy the cheap loser side to complete a partial into a pair.

        Returns True if order placed successfully.
        """
        buy_price = round(min(0.95, loser_price + 0.01), 2)  # 1c above ask for fast fill
        buy_price = max(0.01, buy_price)

        shares = max(self.config.MIN_SHARES, math.floor(self.config.LATE_CANDLE_SIZE_USD / buy_price))
        cost_usd = buy_price * shares
        mom_bps = signal["momentum_bps"]

        # Match shares to existing filled side for clean pairing
        filled_order = pos.up_order if pos.up_filled else pos.down_order
        if filled_order and filled_order.fill_shares > 0:
            shares = int(filled_order.fill_shares)

        now_iso = datetime.now(timezone.utc).isoformat()

        if self.paper:
            order_id = f"lc_{loser_side.lower()}_{pos.market_id[:12]}_{int(time.time())}"
            new_order = MakerOrder(
                order_id=order_id,
                market_id=pos.market_id,
                token_id=loser_token_id,
                side_label=loser_side,
                price=buy_price,
                size_shares=float(shares),
                size_usd=buy_price * shares,
                placed_at=now_iso,
                status="filled",  # Paper: instant fill
                fill_price=buy_price,
                fill_shares=float(shares),
            )
            # Update position
            if loser_side == "DOWN":
                pos.down_order = new_order
                pos.down_filled = True
            else:
                pos.up_order = new_order
                pos.up_filled = True

            filled_cost = (filled_order.fill_price * filled_order.fill_shares) if filled_order else 0
            pos.combined_cost = filled_cost + (buy_price * shares)
            pos.status = "paired"
            pos.entry_type = "late_candle_pair"
            pos.late_candle_pending = False
            pos.loser_buy_price = buy_price

            print(f"  [LC PAIR] {pos.asset} bought {loser_side} @ ${buy_price:.2f} x {shares} | "
                  f"Combined: ${pos.combined_cost:.2f} | Edge: {1.0 - pos.combined_cost / shares:.1%} | "
                  f"Mom: {mom_bps:+.1f}bp")
            return True
        else:
            # Live: place FOK buy order
            try:
                from py_clob_client.clob_types import OrderArgs, OrderType
                from py_clob_client.order_builder.constants import BUY

                args = OrderArgs(
                    price=buy_price,
                    size=float(shares),
                    side=BUY,
                    token_id=loser_token_id,
                )
                signed = self.client.create_order(args)
                resp = self.client.post_order(signed, OrderType.FOK)

                if resp.get("success"):
                    order_id = resp.get("orderID", "")
                    new_order = MakerOrder(
                        order_id=order_id,
                        market_id=pos.market_id,
                        token_id=loser_token_id,
                        side_label=loser_side,
                        price=buy_price,
                        size_shares=float(shares),
                        size_usd=buy_price * shares,
                        placed_at=now_iso,
                        status="filled",
                        fill_price=buy_price,
                        fill_shares=float(shares),
                    )
                    if loser_side == "DOWN":
                        pos.down_order = new_order
                        pos.down_filled = True
                    else:
                        pos.up_order = new_order
                        pos.up_filled = True

                    filled_cost = (filled_order.fill_price * filled_order.fill_shares) if filled_order else 0
                    pos.combined_cost = filled_cost + (buy_price * shares)
                    pos.status = "paired"
                    pos.entry_type = "late_candle_pair"
                    pos.late_candle_pending = False
                    pos.loser_buy_price = buy_price

                    print(f"  [LC PAIR LIVE] {pos.asset} bought {loser_side} @ ${buy_price:.2f} x {shares} | "
                          f"Combined: ${pos.combined_cost:.2f} | Mom: {mom_bps:+.1f}bp | oid={order_id[:16]}...")
                    return True
                else:
                    # FOK failed — no fill available at this price. Place GTC and mark pending.
                    args2 = OrderArgs(price=buy_price, size=float(shares), side=BUY, token_id=loser_token_id)
                    signed2 = self.client.create_order(args2)
                    resp2 = self.client.post_order(signed2, OrderType.GTC)
                    if resp2.get("success"):
                        order_id = resp2.get("orderID", "")
                        new_order = MakerOrder(
                            order_id=order_id, market_id=pos.market_id,
                            token_id=loser_token_id, side_label=loser_side,
                            price=buy_price, size_shares=float(shares),
                            size_usd=buy_price * shares, placed_at=now_iso,
                        )
                        if loser_side == "DOWN":
                            pos.down_order = new_order
                        else:
                            pos.up_order = new_order
                        pos.late_candle_pending = True
                        pos.entry_type = "late_candle_pair"
                        pos.loser_buy_price = buy_price
                        print(f"  [LC PENDING] {pos.asset} GTC {loser_side} @ ${buy_price:.2f} x {shares} | "
                              f"Mom: {mom_bps:+.1f}bp — awaiting fill")
                        return True
                    return False
            except Exception as e:
                print(f"  [LC ERROR] {e}")
                return False

    async def _place_standalone_late_candle(self, pair: MarketPair, loser_side: str,
                                             loser_token_id: str, loser_mid: float,
                                             signal: dict) -> bool:
        """V4.1: Standalone loser-side bet — buy cheap loser token for reversal upside."""
        buy_price = round(min(0.95, loser_mid + 0.01), 2)
        buy_price = max(0.01, buy_price)

        if not self.paper:
            book = await self.get_order_book(loser_token_id)
            if book and book.get("best_ask", 0) > 0:
                buy_price = round(min(0.95, book["best_ask"]), 2)
            if buy_price > self.config.LATE_CANDLE_MAX_LOSER_PRICE:
                return False

        shares = max(self.config.MIN_SHARES, math.floor(self.config.LATE_CANDLE_SIZE_USD / buy_price))
        cost_usd = buy_price * shares
        mom_bps = signal["momentum_bps"]
        now_iso = datetime.now(timezone.utc).isoformat()

        # Create position (one-sided: loser only)
        pos = PairPosition(
            market_id=pair.market_id,
            market_num_id=pair.market_num_id,
            question=pair.question,
            asset=pair.asset,
            created_at=now_iso,
            status="partial",
            bid_offset_used=0.0,
            hour_utc=datetime.now(timezone.utc).hour,
            duration_min=pair.duration_min,
            entry_type="late_candle_standalone",
            loser_buy_price=buy_price,
        )

        if self.paper:
            order_id = f"lcs_{loser_side.lower()}_{pair.market_id[:12]}_{int(time.time())}"
            order = MakerOrder(
                order_id=order_id, market_id=pair.market_id,
                token_id=loser_token_id, side_label=loser_side,
                price=buy_price, size_shares=float(shares),
                size_usd=cost_usd, placed_at=now_iso,
                status="filled", fill_price=buy_price, fill_shares=float(shares),
            )
            if loser_side == "DOWN":
                pos.down_order = order
                pos.down_filled = True
            else:
                pos.up_order = order
                pos.up_filled = True

            print(f"  [LC STANDALONE] {pair.asset} {loser_side} @ ${buy_price:.2f} x {shares} "
                  f"(${cost_usd:.2f}) | Mom: {mom_bps:+.1f}bp — reversal bet")
        else:
            try:
                from py_clob_client.clob_types import OrderArgs, OrderType
                from py_clob_client.order_builder.constants import BUY

                args = OrderArgs(price=buy_price, size=float(shares), side=BUY, token_id=loser_token_id)
                signed = self.client.create_order(args)
                resp = self.client.post_order(signed, OrderType.FOK)

                if resp.get("success"):
                    oid = resp.get("orderID", "")
                    order = MakerOrder(
                        order_id=oid, market_id=pair.market_id,
                        token_id=loser_token_id, side_label=loser_side,
                        price=buy_price, size_shares=float(shares),
                        size_usd=cost_usd, placed_at=now_iso,
                        status="filled", fill_price=buy_price, fill_shares=float(shares),
                    )
                    if loser_side == "DOWN":
                        pos.down_order = order
                        pos.down_filled = True
                    else:
                        pos.up_order = order
                        pos.up_filled = True

                    print(f"  [LC STANDALONE LIVE] {pair.asset} {loser_side} @ ${buy_price:.2f} x {shares} "
                          f"(${cost_usd:.2f}) | Mom: {mom_bps:+.1f}bp | oid={oid[:16]}...")
                else:
                    return False
            except Exception as e:
                print(f"  [LC STANDALONE ERROR] {e}")
                return False

        self.positions[pair.market_id] = pos
        self._persist_entered_market(pair.market_id)
        return True

    # ========================================================================
    # V4.2: LATE-WINNER SCANNER (vague-sourdough strategy)
    # ========================================================================

    async def run_late_winner_scanner(self, markets: List[MarketPair]):
        """V4.2: Buy the WINNER token in the last 60s before market close.

        Scans TWO sources:
        1. Active positions — we already hold one side, buy the OTHER (winner) side
           to complete the pair at a near-guaranteed profit
        2. New markets from discover (not yet entered)

        When direction is 90%+ clear (winner token at $0.80-$0.92), buy it.
        Profit = $1.00 - buy_price per share. Near-guaranteed ~$0.08-$0.15/share.
        """
        if not self.config.LATE_WINNER_ENABLED:
            return

        now = datetime.now(timezone.utc)

        # --- Source 1: Unfilled maker positions near expiry ---
        # When our maker orders haven't filled (pending) and the market is about to close,
        # cancel them and buy the winner token instead. Pure standalone profit.
        for market_id, pos in list(self.positions.items()):
            if pos.status not in ("pending",):
                continue  # Only fully unfilled positions (both orders open)
            if pos.entry_type == "late_winner":
                continue

            # Estimate end time from created_at + duration
            try:
                created = datetime.fromisoformat(pos.created_at)
            except Exception:
                continue
            end_time = created + timedelta(minutes=pos.duration_min)
            secs_left = (end_time - now).total_seconds()

            if secs_left > self.config.LATE_WINNER_WINDOW_SEC:
                continue
            if secs_left < self.config.LATE_WINNER_MIN_SEC:
                continue

            # Check both sides to find the winner
            if not pos.up_order or not pos.down_order:
                continue
            up_book = await self.get_order_book(pos.up_order.token_id) if not self.paper else None
            if not up_book:
                continue
            up_ask = up_book.get("best_ask", 0)
            down_book = await self.get_order_book(pos.down_order.token_id) if not self.paper else None
            if not down_book:
                continue
            down_ask = down_book.get("best_ask", 0)

            # Pick the winner side
            if up_ask >= self.config.LATE_WINNER_MIN_BUY_PRICE and up_ask <= self.config.LATE_WINNER_MAX_BUY_PRICE:
                winner_side = "UP"
                buy_price = up_ask
                winner_token_id = pos.up_order.token_id
            elif down_ask >= self.config.LATE_WINNER_MIN_BUY_PRICE and down_ask <= self.config.LATE_WINNER_MAX_BUY_PRICE:
                winner_side = "DOWN"
                buy_price = down_ask
                winner_token_id = pos.down_order.token_id
            else:
                continue  # Neither side is clearly winning at $0.80-$0.92

            # Cancel the unfilled maker orders first
            if not self.paper:
                try:
                    self.client.cancel(pos.up_order.order_id)
                except Exception:
                    pass
                try:
                    self.client.cancel(pos.down_order.order_id)
                except Exception:
                    pass

            edge = 1.0 - buy_price
            shares = max(self.config.MIN_SHARES, math.floor(self.config.LATE_WINNER_SIZE_USD / buy_price))
            cost_usd = buy_price * shares
            profit_if_win = edge * shares

            # Place FOK buy on winner
            try:
                from py_clob_client.clob_types import OrderArgs, OrderType
                from py_clob_client.order_builder.constants import BUY

                args = OrderArgs(price=round(buy_price, 2), size=float(shares), side=BUY, token_id=winner_token_id)
                signed = self.client.create_order(args)
                resp = self.client.post_order(signed, OrderType.FOK)

                if resp.get("success"):
                    oid = resp.get("orderID", "")
                    order = MakerOrder(
                        order_id=oid, market_id=pos.market_id,
                        token_id=winner_token_id, side_label=winner_side,
                        price=buy_price, size_shares=float(shares),
                        size_usd=cost_usd, placed_at=now.isoformat(),
                        status="filled", fill_price=buy_price, fill_shares=float(shares),
                    )
                    if winner_side == "UP":
                        pos.up_order = order
                        pos.up_filled = True
                    else:
                        pos.down_order = order
                        pos.down_filled = True
                    pos.status = "partial"
                    pos.entry_type = "late_winner"
                    pos.first_fill_time = now.isoformat()
                    self.stats["late_winner_attempts"] = self.stats.get("late_winner_attempts", 0) + 1
                    print(f"  [LW-CONVERT] {pos.asset} {winner_side} @ ${buy_price:.2f} x {shares} "
                          f"(${cost_usd:.2f}) | Edge: ${profit_if_win:.2f} | {secs_left:.0f}s left | oid={oid[:16]}...")
                else:
                    print(f"  [LW-FAIL] {pos.asset} {winner_side} @ ${buy_price:.2f} — FOK rejected")
            except Exception as e:
                print(f"  [LW-ERROR] {pos.asset}: {e}")

        # --- Source 2: Scan ALL 5M/15M markets via API for near-expiry opportunities ---
        candidates = []

        # Markets from discover that we haven't entered (usually filtered by min_time)
        for pair in markets:
            if pair.market_id in self.positions:
                continue
            if not pair.end_time:
                continue
            secs_left = (pair.end_time - now).total_seconds()
            if self.config.LATE_WINNER_MIN_SEC <= secs_left <= self.config.LATE_WINNER_WINDOW_SEC:
                candidates.append(pair)

        # Gamma-api: fetch near-expiry markets for late-winner opportunities
        # NOTE: These markets may already be in _entered_markets from maker orders —
        # that's OK, late_winner is a separate strategy. Only skip if we have an
        # ACTIVE position (to avoid doubling up).
        lw_api_found = 0
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                for tag_slug, duration in [("5M", 5), ("15M", 15)]:
                    r = await client.get(
                        "https://gamma-api.polymarket.com/events",
                        params={"tag_slug": tag_slug, "active": "true", "closed": "false",
                                "limit": 200},
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    if r.status_code != 200:
                        continue
                    for event in r.json():
                        for m in event.get("markets", []):
                            cid = m.get("conditionId", "")
                            if not cid:
                                continue
                            # Skip if we already have a late_winner position on this market
                            lw_key = f"lw_{cid}"
                            if lw_key in self.positions:
                                continue
                            end_date = m.get("endDate", "")
                            if not end_date:
                                continue
                            try:
                                end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                            except Exception:
                                continue
                            secs_left = (end_dt - now).total_seconds()
                            if not (self.config.LATE_WINNER_MIN_SEC <= secs_left <= self.config.LATE_WINNER_WINDOW_SEC):
                                continue
                            # Check asset match
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
                            lw_api_found += 1
                            # Parse into MarketPair (skip entered check — late winner is standalone)
                            pair = self._parse_market(m, event, matched_asset, duration,
                                                       skip_entered_check=True)
                            if pair:
                                pair.end_time = end_dt
                                candidates.append(pair)
        except Exception as e:
            print(f"  [LW-API] Gamma fetch error: {e}")

        # Always log late winner scan results for debugging
        pending_count = sum(1 for _, p in self.positions.items() if p.status == "pending")
        if candidates or lw_api_found or pending_count:
            print(f"  [LW-SCAN] candidates={len(candidates)} | api_found={lw_api_found} | pending_pos={pending_count}")

        # Process candidates (skip risk check — late winner is near-guaranteed $0.05-$0.20/share)
        for pair in candidates:

            secs_left = (pair.end_time - now).total_seconds() if pair.end_time else 0
            print(f"  [LW-EVAL] {pair.asset} {pair.question[:45]} | {secs_left:.0f}s left | "
                  f"UP ${pair.up_mid:.2f} / DOWN ${pair.down_mid:.2f}")

            # Determine which side is winning by looking at mid prices
            if pair.up_mid >= self.config.LATE_WINNER_MIN_BUY_PRICE:
                winner_side = "UP"
                winner_mid = pair.up_mid
                winner_token_id = pair.up_token_id
            elif pair.down_mid >= self.config.LATE_WINNER_MIN_BUY_PRICE:
                winner_side = "DOWN"
                winner_mid = pair.down_mid
                winner_token_id = pair.down_token_id
            else:
                print(f"  [LW-SKIP] {pair.asset} {pair.question[:40]} — no clear winner "
                      f"(UP ${pair.up_mid:.2f} / DOWN ${pair.down_mid:.2f})")
                continue

            # Get actual best_ask from order book
            if not self.paper:
                book = await self.get_order_book(winner_token_id)
                if not book:
                    print(f"  [LW-SKIP] {pair.asset} {winner_side} — no order book")
                    continue
                buy_price = book.get("best_ask", 0)
                if buy_price <= 0:
                    buy_price = winner_mid
            else:
                buy_price = winner_mid

            # Price guards
            if buy_price < self.config.LATE_WINNER_MIN_BUY_PRICE:
                print(f"  [LW-SKIP] {pair.asset} {winner_side} @ ${buy_price:.2f} — below min ${self.config.LATE_WINNER_MIN_BUY_PRICE}")
                continue
            if buy_price > self.config.LATE_WINNER_MAX_BUY_PRICE:
                print(f"  [LW-SKIP] {pair.asset} {winner_side} @ ${buy_price:.2f} — above max ${self.config.LATE_WINNER_MAX_BUY_PRICE}")
                continue

            edge = 1.0 - buy_price
            shares = max(self.config.MIN_SHARES, math.floor(self.config.LATE_WINNER_SIZE_USD / buy_price))
            cost_usd = buy_price * shares
            profit_if_win = edge * shares
            now_iso = now.isoformat()

            # Create position
            pos = PairPosition(
                market_id=pair.market_id,
                market_num_id=pair.market_num_id,
                question=pair.question,
                asset=pair.asset,
                created_at=now_iso,
                status="partial",
                bid_offset_used=0.0,
                hour_utc=now.hour,
                duration_min=pair.duration_min,
                entry_type="late_winner",
                first_fill_time=now_iso,
            )

            if self.paper:
                order_id = f"lw_{winner_side.lower()}_{pair.market_id[:12]}_{int(time.time())}"
                order = MakerOrder(
                    order_id=order_id, market_id=pair.market_id,
                    token_id=winner_token_id, side_label=winner_side,
                    price=buy_price, size_shares=float(shares),
                    size_usd=cost_usd, placed_at=now_iso,
                    status="filled", fill_price=buy_price, fill_shares=float(shares),
                )
                if winner_side == "UP":
                    pos.up_order = order
                    pos.up_filled = True
                else:
                    pos.down_order = order
                    pos.down_filled = True

                print(f"  [LW PAPER] {pair.asset} {winner_side} @ ${buy_price:.2f} x {shares} "
                      f"(${cost_usd:.2f}) | Edge: ${profit_if_win:.2f} | {secs_left:.0f}s left")
            else:
                try:
                    from py_clob_client.clob_types import OrderArgs, OrderType
                    from py_clob_client.order_builder.constants import BUY

                    args = OrderArgs(price=buy_price, size=float(shares), side=BUY, token_id=winner_token_id)
                    signed = self.client.create_order(args)
                    resp = self.client.post_order(signed, OrderType.FOK)

                    if not resp.get("success"):
                        continue

                    oid = resp.get("orderID", "")
                    order = MakerOrder(
                        order_id=oid, market_id=pair.market_id,
                        token_id=winner_token_id, side_label=winner_side,
                        price=buy_price, size_shares=float(shares),
                        size_usd=cost_usd, placed_at=now_iso,
                        status="filled", fill_price=buy_price, fill_shares=float(shares),
                    )
                    if winner_side == "UP":
                        pos.up_order = order
                        pos.up_filled = True
                    else:
                        pos.down_order = order
                        pos.down_filled = True

                    print(f"  [LW LIVE] {pair.asset} {winner_side} @ ${buy_price:.2f} x {shares} "
                          f"(${cost_usd:.2f}) | Edge: ${profit_if_win:.2f} | {secs_left:.0f}s left | oid={oid[:16]}...")
                except Exception as e:
                    print(f"  [LW ERROR] {pair.asset}: {e}")
                    continue

            # Use a unique key so late_winner doesn't overwrite maker positions
            lw_key = f"lw_{pair.market_id}"
            self.positions[lw_key] = pos
            self.stats["late_winner_attempts"] = self.stats.get("late_winner_attempts", 0) + 1

    async def _place_momentum_order(self, pair: MarketPair, direction: str,
                                    momentum_bps: float, confirmed: bool = False):
        """Place a directional BUY order based on momentum signal.

        If confirmed=True (V5 regime alignment), uses MOMENTUM_CONFIRMED_SIZE_USD (2x).
        """
        token_id = pair.up_token_id if direction == "UP" else pair.down_token_id
        mid_price = pair.up_mid if direction == "UP" else pair.down_mid

        # Buy at mid price or slightly above for faster fill
        buy_price = round(min(0.95, mid_price + 0.01), 2)
        buy_price = max(0.05, buy_price)

        # V5 tiered sizing: double down when regime confirms
        size_usd = self.config.MOMENTUM_CONFIRMED_SIZE_USD if confirmed else self.config.MOMENTUM_SIZE_USD
        tag = "2x" if confirmed else "1x"

        shares = max(self.config.MIN_SHARES, math.floor(size_usd / buy_price))
        cost_usd = buy_price * shares

        if self.paper:
            order_id = f"mom_{direction.lower()}_{pair.market_id[:12]}_{int(time.time())}"
            print(f"  [MOM PAPER {tag}] {pair.asset} {direction} @ ${buy_price:.2f} x {shares} "
                  f"(${cost_usd:.2f}) | momentum: {momentum_bps:+.1f}bp")
        else:
            try:
                from py_clob_client.clob_types import OrderArgs, OrderType
                from py_clob_client.order_builder.constants import BUY

                args = OrderArgs(
                    price=buy_price,
                    size=float(shares),
                    side=BUY,
                    token_id=token_id,
                )
                signed = self.client.create_order(args)
                resp = self.client.post_order(signed, OrderType.GTC)

                if not resp.get("success"):
                    print(f"  [MOM LIVE] Order failed: {resp.get('errorMsg', '?')}")
                    return

                order_id = resp.get("orderID", "")
                print(f"  [MOM LIVE {tag}] {pair.asset} {direction} @ ${buy_price:.2f} x {shares} "
                      f"(${cost_usd:.2f}) | momentum: {momentum_bps:+.1f}bp | oid={order_id[:16]}...")
            except Exception as e:
                print(f"  [MOM LIVE] Error: {e}")
                return

        # Record and mark as traded
        self.momentum.mark_traded(pair.market_id)
        self.momentum.record_trade(
            market_id=pair.market_id,
            direction=direction,
            buy_price=buy_price,
            shares=shares,
            cost_usd=cost_usd,
            momentum_bps=momentum_bps,
            question=pair.question,
            market_num_id=pair.market_num_id,
        )

        # Also add to _entered_markets so maker doesn't double-enter
        self._persist_entered_market(pair.market_id)

    async def _place_orderflow_order(self, pair: MarketPair, direction: str,
                                     buy_ratio: float, vol_ratio: float):
        """Place a directional BUY order based on V12a order flow signal."""
        token_id = pair.up_token_id if direction == "UP" else pair.down_token_id
        mid_price = pair.up_mid if direction == "UP" else pair.down_mid

        buy_price = round(min(0.95, mid_price + 0.01), 2)
        buy_price = max(0.05, buy_price)

        size_usd = self.config.ORDERFLOW_SIZE_USD
        shares = max(self.config.MIN_SHARES, math.floor(size_usd / buy_price))
        cost_usd = buy_price * shares

        if self.paper:
            order_id = f"flow_{direction.lower()}_{pair.market_id[:12]}_{int(time.time())}"
            print(f"  [FLOW PAPER] {pair.asset} {direction} @ ${buy_price:.2f} x {shares} "
                  f"(${cost_usd:.2f}) | ratio={buy_ratio:.3f} vol={vol_ratio:.1f}x")
        else:
            try:
                from py_clob_client.clob_types import OrderArgs, OrderType
                from py_clob_client.order_builder.constants import BUY

                args = OrderArgs(
                    price=buy_price,
                    size=float(shares),
                    side=BUY,
                    token_id=token_id,
                )
                signed = self.client.create_order(args)
                resp = self.client.post_order(signed, OrderType.GTC)

                if not resp.get("success"):
                    print(f"  [FLOW LIVE] Order failed: {resp.get('errorMsg', '?')}")
                    return

                order_id = resp.get("orderID", "")
                print(f"  [FLOW LIVE] {pair.asset} {direction} @ ${buy_price:.2f} x {shares} "
                      f"(${cost_usd:.2f}) | ratio={buy_ratio:.3f} vol={vol_ratio:.1f}x | oid={order_id[:16]}...")
            except Exception as e:
                print(f"  [FLOW LIVE] Error: {e}")
                return

        # Record and mark as traded (reuse momentum tracking — same resolution flow)
        self.momentum.mark_traded(pair.market_id)
        self.momentum.record_trade(
            market_id=pair.market_id,
            direction=direction,
            buy_price=buy_price,
            shares=shares,
            cost_usd=cost_usd,
            momentum_bps=0.0,  # No momentum — this is order flow
            question=pair.question,
            market_num_id=pair.market_num_id,
        )
        # Tag as flow trade for separate tracking
        if self.momentum.momentum_trades:
            self.momentum.momentum_trades[-1]["signal_type"] = "FLOW"
            self.momentum.momentum_trades[-1]["flow_buy_ratio"] = buy_ratio
            self.momentum.momentum_trades[-1]["flow_vol_ratio"] = vol_ratio

        self._persist_entered_market(pair.market_id)

    async def resolve_momentum_trades(self):
        """Resolve momentum trades that have settled."""
        if not self.momentum:
            return

        for trade in self.momentum.momentum_trades:
            if trade.get("outcome") is not None:
                continue
            if not trade.get("placed_at"):
                continue

            placed = datetime.fromisoformat(trade["placed_at"])
            age_min = (datetime.now(timezone.utc) - placed).total_seconds() / 60

            # Wait at least 6 minutes for resolution
            if age_min < 6:
                continue

            # Fetch outcome using gamma-api
            market_id = trade["market_id"]

            # Use stored numeric ID first, fall back to lookups
            num_id = trade.get("market_num_id", "")
            if not num_id:
                for pos in self.positions.values():
                    if pos.market_id == market_id:
                        num_id = pos.market_num_id
                        break
            if not num_id:
                tracking = self.momentum.tracked_markets.get(market_id, {})
                num_id = tracking.get("market_num_id", "")

            # Try to resolve via condition_id market search
            if not num_id:
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        r = await client.get(
                            "https://gamma-api.polymarket.com/markets",
                            params={"condition_id": market_id, "limit": 1},
                            headers={"User-Agent": "Mozilla/5.0"})
                        if r.status_code == 200:
                            results = r.json()
                            if results and isinstance(results, list) and len(results) > 0:
                                num_id = str(results[0].get("id", ""))
                except Exception:
                    pass

            if not num_id:
                # Give up after 30 minutes
                if age_min > 30:
                    trade["outcome"] = "UNKNOWN"
                    trade["pnl"] = -trade.get("cost_usd", 0)
                    self.momentum.momentum_daily_pnl += trade["pnl"]
                    print(f"  [MOM] TIMEOUT {trade['direction']} — marking as loss ${trade['pnl']:.2f}")
                continue

            # Fetch outcome
            outcome = await self._fetch_outcome(num_id)
            if not outcome:
                if age_min > 20:
                    trade["outcome"] = "UNKNOWN"
                    trade["pnl"] = -trade.get("cost_usd", 0)
                    self.momentum.momentum_daily_pnl += trade["pnl"]
                continue

            # Calculate PnL
            direction = trade["direction"]
            shares = trade.get("shares", 0)
            cost = trade.get("cost_usd", 0)

            if outcome == direction:
                # Win: shares * $1.00 - cost
                pnl = shares * 1.0 - cost
            else:
                # Lose: shares worth $0
                pnl = -cost

            trade["outcome"] = outcome
            trade["pnl"] = round(pnl, 4)
            self.momentum.momentum_daily_pnl += pnl

            icon = "WIN" if pnl > 0 else "LOSS"
            print(f"  [MOM {icon}] {trade['direction']} | outcome={outcome} | "
                  f"PnL=${pnl:+.2f} | mom={trade.get('momentum_bps', 0):+.1f}bp")

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
                        gamma=pos.gamma_used,
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

            # V4.1: Track late-candle stats separately
            if pos.entry_type.startswith("late_candle"):
                self.stats["late_candle_pnl"] = self.stats.get("late_candle_pnl", 0) + pnl

            # ML: record outcome for gamma optimization
            self.ml_optimizer.record(
                hour=pos.hour_utc if pos.hour_utc >= 0 else datetime.now(timezone.utc).hour,
                gamma=pos.gamma_used,
                paired=pos.is_paired,
                partial=pos.is_partial,
                pnl=pnl
            )

            # Log
            pair_type = "PAIRED" if pos.is_paired else "PARTIAL" if pos.is_partial else "UNFILLED"
            lc_tag = f" [LC:{pos.entry_type}]" if pos.entry_type != "maker" else ""
            icon = "+" if pnl > 0 else "-" if pnl < 0 else "="
            print(f"  [{pair_type}]{lc_tag} {pos.asset} {pos.question[:50]} | {outcome} | PnL: {icon}${abs(pnl):.4f} (off={pos.bid_offset_used:.1%})")

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
                # V4.1: ML tracking fields
                "entry_type": pos.entry_type,
                "loser_buy_price": pos.loser_buy_price,
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

    async def _sell_partial_at_market(self, pos: PairPosition, filled_order: MakerOrder) -> Optional[float]:
        """V4: Sell a partially-filled position at market to cut losses.
        Retries up to 3 times on transient failures.
        Returns PnL if sold successfully, None if all attempts failed."""
        if not self.client or not filled_order:
            return None

        from py_clob_client.clob_types import OrderArgs, OrderType, BalanceAllowanceParams, AssetType
        from py_clob_client.order_builder.constants import SELL

        for attempt in range(3):
            try:
                # Get current best bid for the filled token
                book = await self.get_order_book(filled_order.token_id)
                if not book or book.get("best_bid", 0) <= 0:
                    if attempt < 2:
                        print(f"  [SELL-RETRY] No bids on attempt {attempt+1}/3, retrying in 2s...")
                        await asyncio.sleep(2)
                        continue
                    print(f"  [SELL-CUT] No bids after 3 attempts (book empty)")
                    return None

                best_bid = book["best_bid"]

                # V4.1: Don't sell for pennies — if token is nearly worthless (<$0.10),
                # ride instead. Selling at $0.02 saves ~$0.10 vs riding, but kills reversal upside.
                if best_bid < 0.10:
                    print(f"  [SELL-SKIP] best_bid=${best_bid:.2f} < $0.10 — too cheap, ride instead")
                    return None

                sell_price = max(0.01, round(best_bid - 0.01, 2))  # 1c below bid for fast fill

                # Approve conditional token for selling
                try:
                    self.client.update_balance_allowance(
                        BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=filled_order.token_id))
                except Exception:
                    pass

                sell_args = OrderArgs(
                    price=sell_price,
                    size=filled_order.fill_shares,
                    side=SELL,
                    token_id=filled_order.token_id,
                )
                sell_signed = self.client.create_order(sell_args)
                sell_resp = self.client.post_order(sell_signed, OrderType.GTC)

                if sell_resp.get("success"):
                    # PnL = sell revenue - buy cost
                    sell_revenue = sell_price * filled_order.fill_shares
                    buy_cost = filled_order.fill_price * filled_order.fill_shares
                    pnl = round(sell_revenue - buy_cost, 6)
                    return pnl
                else:
                    err_msg = sell_resp.get('errorMsg', '?')
                    if attempt < 2:
                        print(f"  [SELL-RETRY] Order rejected ({err_msg}), attempt {attempt+1}/3, retrying in 2s...")
                        await asyncio.sleep(2)
                        continue
                    print(f"  [SELL-CUT] Order failed after 3 attempts: {err_msg}")
                    return None

            except Exception as e:
                if attempt < 2:
                    print(f"  [SELL-RETRY] Error on attempt {attempt+1}/3: {str(e)[:60]}, retrying in 2s...")
                    await asyncio.sleep(2)
                    continue
                print(f"  [SELL-CUT] Error after 3 attempts: {e}")
                return None

        return None

    # ========================================================================
    # RISK MANAGEMENT
    # ========================================================================

    def check_risk(self) -> bool:
        """Check if we should continue trading. Returns True if OK."""
        # V1.4: Capital loss circuit breaker (40% net loss -> auto-switch to paper)
        if not self.paper and not self.circuit_tripped and self.starting_pnl is not None:
            session_pnl = self.stats["total_pnl"] - self.starting_pnl
            # Calculate 40% of the CLOB cash balance (~$76)
            capital_loss_limit = 42.0  # 40% of $106 CLOB cash
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
        print(f"AVELLANEDA MAKER V4.4 - {mode} MODE")
        print(f"Strategy: Avellaneda time-decay spreads + inventory skew + FOK hedge + order refresh")
        print(f"Size: ${self.config.SIZE_PER_SIDE_USD}/side | Gamma: {self.config.DEFAULT_GAMMA} (ML-tuned) | "
              f"Base spread: {self.config.BASE_SPREAD*100:.0f}c")
        print(f"Max combined: ${self.config.MAX_COMBINED_PRICE} | Min edge: {self.config.MIN_SPREAD_EDGE*100:.0f}% | "
              f"Hedge max: ${self.config.TAKER_HEDGE_MAX_COMBINED}")
        print(f"Order refresh: {self.config.ORDER_REFRESH}s (tolerance: ${self.config.ORDER_REFRESH_TOLERANCE}) | "
              f"Fill delay: {self.config.FILLED_ORDER_DELAY}s | Ride cut: {self.config.RIDING_CUT_THRESHOLD*100:.0f}c")
        print(f"Daily loss limit: ${self.config.DAILY_LOSS_LIMIT}")
        if not self.paper:
            print(f"CIRCUIT BREAKER: Auto-switch to paper on $42 net session loss (~40% of $106 capital)")
        if self.config.TAKER_HEDGE_ENABLED:
            print(f"TAKER HEDGE: ON | Max combined: ${self.config.TAKER_HEDGE_MAX_COMBINED} | "
                  f"Fallback: ride to resolution")
        print(f"Skip hours (UTC): {sorted(self.config.SKIP_HOURS_UTC)}")
        if self.config.MOMENTUM_ENABLED:
            print(f"MOMENTUM 5m: ${self.config.MOMENTUM_SIZE_USD}/trade | "
                  f"Threshold: {self.config.MOMENTUM_THRESHOLD_BPS}bp | "
                  f"Wait: {self.config.MOMENTUM_WAIT_SECONDS}s | "
                  f"Max: {self.config.MOMENTUM_MAX_CONCURRENT} concurrent")
        if self.config.LATE_ENTRY_ENABLED:
            print(f"LATE ENTRY 15m: ${self.config.LATE_ENTRY_SIZE_USD}/trade | "
                  f"Threshold: {self.config.LATE_ENTRY_THRESHOLD_BPS}bp | "
                  f"Wait: {self.config.LATE_ENTRY_WAIT_SECONDS}s")
        if self.config.LATE_CANDLE_ENABLED:
            print(f"LATE CANDLE 5m: ${self.config.LATE_CANDLE_SIZE_USD}/trade | "
                  f"Buy loser @ <${self.config.LATE_CANDLE_MAX_LOSER_PRICE} | "
                  f"Window: {self.config.LATE_CANDLE_WAIT_SEC}-{self.config.LATE_CANDLE_MAX_WAIT_SEC}s | "
                  f"Min momentum: {self.config.LATE_CANDLE_MIN_MOMENTUM_BPS}bp")
        if self.config.LATE_WINNER_ENABLED:
            print(f"LATE WINNER: ${self.config.LATE_WINNER_SIZE_USD}/trade | "
                  f"Buy winner @ ${self.config.LATE_WINNER_MIN_BUY_PRICE}-${self.config.LATE_WINNER_MAX_BUY_PRICE} | "
                  f"Last {self.config.LATE_WINNER_WINDOW_SEC:.0f}s before close")
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

                # Phase 1: Cancel expiring unfilled orders + V4.4 order refresh
                await self.manage_expiring_orders()

                # Phase 1.5: V4.4 Monitor riding positions (sell if price moved too far against)
                await self.manage_riding_positions()

                # Phase 2a: V2.4 Harvest paired positions at extreme prices (last 90s)
                await self.harvest_extreme_prices()

                # Phase 2b: Resolve completed markets
                await self.resolve_positions()

                # Phase 3: Check fills on active orders
                await self.check_fills()

                # V4.4: Shadow tracker cleanup
                self.shadow.cleanup()

                # Phase 4: Discover new markets (always, even if risk limits hit for maker)
                markets = await self.discover_markets()

                # Phase 4a: Momentum directional trading (V3.0)
                if self.momentum and self.config.MOMENTUM_ENABLED:
                    await self.run_momentum_scanner(markets)
                    await self.resolve_momentum_trades()

                # Phase 4b: V4.1 Late-candle scanner (buy cheap loser side)
                if self.momentum and self.config.LATE_CANDLE_ENABLED:
                    await self.run_late_candle_scanner(markets)

                # Phase 4c: V4.2 Late-winner scanner (buy winner at $0.85-$0.92)
                if self.config.LATE_WINNER_ENABLED:
                    await self.run_late_winner_scanner(markets)

                # Phase 5: Place maker orders if risk allows
                # V4.4: Filled order delay — skip new placements for 5s after any fill
                fill_cooldown = time.time() - self._last_fill_time < self.config.FILLED_ORDER_DELAY
                if self.check_risk() and not fill_cooldown:
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
                    if self.momentum:
                        self.momentum.save()

                # Phase 8: ML summary every 50 cycles (~25 min)
                if cycle % 50 == 0 and cycle > 0:
                    print(f"\n[ML GAMMA] Current optimal gammas (Avellaneda):")
                    print(self.ml_optimizer.get_summary())

                # Phase 9: Auto-redeem winnings every 45 seconds (live only)
                if not self.paper and now_ts - last_redeem_check >= 300:
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
        h_ok = self.stats.get("hedge_successes", 0)
        h_fail = self.stats.get("hedge_failures", 0)
        h_att = self.stats.get("hedge_attempts", 0)
        hedge_tag = f" | Hedge: {h_ok}/{h_att}" if h_att > 0 else ""
        print(f"\n--- Cycle {cycle} | {mode_tag}{cb_tag} | UTC {hour:02d} | "
              f"Active: {active} ({paired} paired, {partial} partial){hedge_tag} | "
              f"Daily: ${self.daily_pnl:+.4f} | Session: ${session_pnl:+.4f} | Total: ${self.stats['total_pnl']:+.4f} | "
              f"Resolved: {len(self.resolved)} ---")

        if self.stats["pairs_completed"] > 0:
            avg = self.stats["paired_pnl"] / self.stats["pairs_completed"]
            print(f"    Paired avg PnL: ${avg:.4f} | Best: ${self.stats['best_pair_pnl']:.4f} | "
                  f"Worst: ${self.stats['worst_pair_pnl']:.4f}")

        # V4.1: Late-candle stats
        lc_attempts = self.stats.get("late_candle_attempts", 0)
        if lc_attempts > 0:
            lc_paired = self.stats.get("late_candle_paired", 0)
            lc_standalone = self.stats.get("late_candle_standalone", 0)
            lc_pnl = self.stats.get("late_candle_pnl", 0)
            lc_resolved = [r for r in self.resolved if r.get("entry_type", "").startswith("late_candle")]
            lc_wins = sum(1 for r in lc_resolved if r.get("pnl", 0) > 0)
            lc_wr = lc_wins / len(lc_resolved) * 100 if lc_resolved else 0
            print(f"    Late candle: {lc_attempts} placed ({lc_paired} pairs, {lc_standalone} standalone) | "
                  f"Resolved: {len(lc_resolved)} ({lc_wr:.0f}% WR) | PnL: ${lc_pnl:+.4f}")

        if self.momentum and self.config.MOMENTUM_ENABLED:
            print(f"    {self.momentum.get_stats_summary()}")


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
