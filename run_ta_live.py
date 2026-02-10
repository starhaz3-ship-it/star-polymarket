"""
TA Live Trading with ML Optimization

Live trades BTC 15m markets using TA+Bregman signals.
Includes ML feature tracking to optimize win rate.
"""
import sys
import asyncio
import json
import time
import os
import math
import numpy as np
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from functools import partial

import httpx
from dotenv import load_dotenv

# Force unbuffered output
print = partial(print, flush=True)

# Import from arbitrage package
sys.path.insert(0, str(Path(__file__).parent))

from arbitrage.ta_signals import TASignalGenerator, Candle, SignalStrength, TASignal, MarketRegime
from arbitrage.bregman_optimizer import BregmanOptimizer, BregmanSignal
from arbitrage.nyu_volatility import NYUVolatilityModel

# Hydra strategy import for auto-promoted strategies
try:
    from hydra_strategies import scan_strategies, StrategySignal as HydraSignal
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

# Loss forensics engine for automated post-loss analysis
try:
    from loss_forensics import ForensicsEngine
    FORENSICS_AVAILABLE = True
except ImportError:
    FORENSICS_AVAILABLE = False

load_dotenv()


@dataclass
class MLFeatures:
    """Features for ML optimization."""
    rsi: float = 0.0
    rsi_slope: float = 0.0
    vwap_distance: float = 0.0  # % distance from VWAP
    vwap_slope: float = 0.0
    heiken_count: int = 0
    heiken_bullish: bool = False
    macd_histogram: float = 0.0
    macd_expanding: bool = False
    time_remaining: float = 0.0
    model_confidence: float = 0.0  # How far from 50%
    kl_divergence: float = 0.0
    kelly_fraction: float = 0.0
    market_edge: float = 0.0
    btc_1h_change: float = 0.0  # 1-hour BTC change
    btc_volatility: float = 0.0  # Recent volatility
    # V3.17: GARCH + CEEMD features
    garch_vol_forecast: float = 0.0   # GARCH(1,1) conditional vol forecast
    garch_vol_regime: float = 0.5     # 0=LOW, 0.5=NORMAL, 1.0=HIGH
    ceemd_trend_slope: float = 0.0    # CEEMD lowest-freq IMF trend slope
    ceemd_noise_ratio: float = 0.5    # 0=pure trend, 1=pure noise


@dataclass
class LiveTrade:
    """A live trade record."""
    trade_id: str
    market_id: str
    market_title: str
    side: str  # "UP" or "DOWN"
    entry_price: float
    entry_time: str
    size_usd: float
    # Signal data
    signal_strength: str
    edge_at_entry: float
    kl_divergence: float
    kelly_fraction: float
    # ML features at entry
    features: Dict = field(default_factory=dict)
    # Results
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: float = 0.0
    status: str = "open"
    # Execution
    order_id: Optional[str] = None
    execution_error: Optional[str] = None


class MLOptimizer:
    """Simple ML optimizer for trade signals."""

    def __init__(self):
        self.feature_weights = {
            'rsi_oversold': 1.0,  # RSI < 30 for DOWN, > 70 for UP
            'vwap_distance': 1.0,  # Distance from VWAP
            'heiken_streak': 1.0,  # Consecutive candles
            'kl_divergence': 1.5,  # Higher weight for info edge
            'model_confidence': 1.2,
            'time_remaining': 0.8,
        }
        self.win_features: List[Dict] = []
        self.loss_features: List[Dict] = []
        self.min_trades_for_ml = 10  # Start adjusting after 10 trades

    def extract_features(self, signal: TASignal, bregman: BregmanSignal, candles: List[Candle]) -> MLFeatures:
        """Extract ML features from signal."""
        features = MLFeatures()

        # RSI features
        features.rsi = signal.rsi or 50.0
        features.rsi_slope = signal.rsi_slope or 0.0

        # VWAP features
        if signal.vwap and signal.current_price:
            features.vwap_distance = (signal.current_price - signal.vwap) / signal.vwap * 100
        features.vwap_slope = signal.vwap_slope or 0.0

        # Heiken Ashi
        features.heiken_count = signal.heiken_count or 0
        features.heiken_bullish = signal.heiken_color == "green"

        # MACD
        if signal.macd:
            features.macd_histogram = signal.macd.histogram or 0.0
            features.macd_expanding = (signal.macd.hist_delta or 0) * (signal.macd.histogram or 0) > 0

        # Time and confidence
        features.time_remaining = signal.time_remaining_min or 15.0
        features.model_confidence = abs(signal.model_up - 0.5) * 2 if signal.model_up else 0.0

        # Bregman metrics
        features.kl_divergence = bregman.kl_divergence
        features.kelly_fraction = bregman.kelly_fraction
        features.market_edge = bregman.optimal_edge

        # BTC price action
        if len(candles) >= 60:
            hour_ago_price = candles[-60].close
            current_price = candles[-1].close
            features.btc_1h_change = (current_price - hour_ago_price) / hour_ago_price * 100

            # Volatility (std of returns)
            returns = [(candles[i].close - candles[i-1].close) / candles[i-1].close
                      for i in range(-30, 0)]
            features.btc_volatility = np.std(returns) * 100

        return features

    def enrich_with_quant_models(self, features: MLFeatures, garch: dict, ceemd: dict) -> MLFeatures:
        """V3.17: Add GARCH + CEEMD features to MLFeatures for scoring."""
        features.garch_vol_forecast = garch.get("forecast", 0.0)
        regime_map = {"LOW": 0.0, "NORMAL": 0.5, "HIGH": 1.0, "UNKNOWN": 0.5}
        features.garch_vol_regime = regime_map.get(garch.get("regime", "UNKNOWN"), 0.5)
        features.ceemd_trend_slope = ceemd.get("trend_slope", 0.0)
        features.ceemd_noise_ratio = ceemd.get("noise_ratio", 0.5)
        return features

    def score_trade(self, features: MLFeatures, side: str) -> float:
        """Score a potential trade based on features and learned weights."""
        score = 0.0

        # RSI scoring
        if side == "DOWN":
            if features.rsi < 30:
                score -= 0.5  # Oversold, might bounce
            elif features.rsi > 50:
                score += self.feature_weights['rsi_oversold']
        else:  # UP
            if features.rsi > 70:
                score -= 0.5  # Overbought, might drop
            elif features.rsi < 50:
                score += self.feature_weights['rsi_oversold']

        # VWAP scoring
        if side == "DOWN" and features.vwap_distance < 0:
            score += self.feature_weights['vwap_distance'] * abs(features.vwap_distance) / 2
        elif side == "UP" and features.vwap_distance > 0:
            score += self.feature_weights['vwap_distance'] * features.vwap_distance / 2

        # Heiken streak
        if features.heiken_count >= 3:
            if (side == "DOWN" and not features.heiken_bullish) or \
               (side == "UP" and features.heiken_bullish):
                score += self.feature_weights['heiken_streak'] * min(features.heiken_count / 10, 1.0)

        # KL divergence (information edge)
        score += self.feature_weights['kl_divergence'] * features.kl_divergence * 5

        # Model confidence
        score += self.feature_weights['model_confidence'] * features.model_confidence

        # Time remaining penalty (less confident with more time)
        if features.time_remaining > 10:
            score -= 0.2

        return score

    def update_weights(self, features: Dict, won: bool):
        """Update feature weights based on trade outcome."""
        if won:
            self.win_features.append(features)
        else:
            self.loss_features.append(features)

        total_trades = len(self.win_features) + len(self.loss_features)
        if total_trades < self.min_trades_for_ml:
            return

        # Simple weight adjustment based on win/loss correlation
        # This is a basic approach - could be enhanced with proper ML
        if len(self.win_features) >= 5 and len(self.loss_features) >= 5:
            # Compare average features between wins and losses
            win_avg = {k: np.mean([f.get(k, 0) for f in self.win_features[-20:]])
                      for k in self.win_features[0].keys() if isinstance(self.win_features[0].get(k), (int, float))}
            loss_avg = {k: np.mean([f.get(k, 0) for f in self.loss_features[-20:]])
                       for k in self.loss_features[0].keys() if isinstance(self.loss_features[0].get(k), (int, float))}

            # Adjust weights based on discriminative power
            for key in ['kl_divergence', 'model_confidence']:
                if key in win_avg and key in loss_avg:
                    diff = win_avg[key] - loss_avg[key]
                    if diff > 0:
                        # This feature is higher in wins - increase weight
                        self.feature_weights[key] = min(2.0, self.feature_weights.get(key, 1.0) * 1.05)

    def get_min_score_threshold(self) -> float:
        """Get minimum score threshold for trading."""
        total_trades = len(self.win_features) + len(self.loss_features)
        if total_trades < self.min_trades_for_ml:
            return 0.5  # Default threshold (lowered from 1.0)

        # Adjust threshold based on win rate
        win_rate = len(self.win_features) / total_trades
        if win_rate > 0.6:
            return 0.3  # Lower threshold if winning
        elif win_rate < 0.4:
            return 1.0  # Higher threshold if losing
        return 0.5


_poly_web3_service = None  # Cached PolyWeb3Service instance


def _get_poly_web3_service():
    """Get or create a cached PolyWeb3Service instance for gasless relayer redemptions."""
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

        # Decrypt private key
        pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        if pk.startswith("ENC:"):
            pk = decrypt_key(pk[4:], os.getenv("POLYMARKET_KEY_SALT", ""), password)
        if not pk:
            return None

        proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")

        # Init ClobClient
        client = ClobClient(
            host="https://clob.polymarket.com",
            key=pk,
            chain_id=137,
            signature_type=1,
            funder=proxy_address if proxy_address else None,
        )
        creds = client.derive_api_key()
        client.set_api_creds(creds)

        # Decrypt Builder credentials for relayer
        builder_key = decrypt_key(
            os.getenv("POLY_BUILDER_API_KEY", "")[4:],
            os.getenv("POLY_BUILDER_API_KEY_SALT", ""), password)
        builder_secret = decrypt_key(
            os.getenv("POLY_BUILDER_SECRET", "")[4:],
            os.getenv("POLY_BUILDER_SECRET_SALT", ""), password)
        builder_passphrase = decrypt_key(
            os.getenv("POLY_BUILDER_PASSPHRASE", "")[4:],
            os.getenv("POLY_BUILDER_PASSPHRASE_SALT", ""), password)

        builder_config = BuilderConfig(
            local_builder_creds=BuilderApiKeyCreds(
                key=builder_key, secret=builder_secret, passphrase=builder_passphrase))

        relay_client = RelayClient(
            relayer_url="https://relayer-v2.polymarket.com",
            chain_id=137,
            private_key=pk,
            builder_config=builder_config,
        )

        _poly_web3_service = PolyWeb3Service(
            clob_client=client,
            relayer_client=relay_client,
            rpc_url="https://polygon-bor.publicnode.com",
        )
        print(f"[REDEEM] PolyWeb3Service initialized (gasless relayer)")
        return _poly_web3_service
    except Exception as e:
        print(f"[REDEEM] Failed to init PolyWeb3Service: {str(e)[:80]}")
        return None


def auto_redeem_winnings():
    """Auto-redeem resolved WINNING positions to USDC via Polymarket's gasless relayer.
    Uses poly-web3 package with Builder API credentials.
    Returns: float amount of USDC claimed (0.0 if nothing claimed)."""
    try:
        proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
        if not proxy_address:
            return 0.0

        service = _get_poly_web3_service()
        if not service:
            return 0.0

        # Check for redeemable positions
        positions = service.fetch_positions(user_address=proxy_address)
        if not positions:
            return 0.0

        total_shares = sum(float(p.get("size", 0) or 0) for p in positions)
        print(f"[REDEEM] Found {len(positions)} winning positions ({total_shares:.0f} shares, ~${total_shares:.2f} USDC)")

        # Redeem all via relayer (gasless!)
        results = service.redeem_all(batch_size=10)
        if results:
            print(f"[REDEEM] Claimed {len(results)} batch(es) (~${total_shares:.2f} USDC)")
            for r in results:
                if r and isinstance(r, dict):
                    tx_hash = r.get("transactionHash", "?")[:20]
                    state = r.get("state", "?")
                    print(f"[REDEEM]   tx={tx_hash}... state={state}")
            return total_shares
        else:
            print(f"[REDEEM] No claims succeeded this cycle")
            return 0.0

    except Exception as e:
        if "No winning" not in str(e):
            print(f"[REDEEM] Error: {str(e)[:80]}")
        return 0.0


class TALiveTrader:
    """Live trades based on TA + Bregman signals with ML optimization."""

    OUTPUT_FILE = Path(__file__).parent / "ta_live_results.json"
    PID_FILE = Path(__file__).parent / "ta_live.pid"

    # === HARD BET LIMITS (V3.12c: $3 FLAT — ULTRA CONSERVATIVE RESTART) ===
    # Restarting live after liquidity shutdown. Minimal size, prove fills work.
    HARD_MAX_BET = 3.0         # ABSOLUTE CEILING — $3 flat, no exceptions
    BASE_POSITION_SIZE = 3.0   # Standard bet
    MIN_POSITION_SIZE = 3.0    # Floor $3
    MAX_POSITION_SIZE = 3.0    # Hard cap $3 — same as floor. Every trade = $3.

    def __init__(self, dry_run: bool = False, bankroll: float = 73.14):
        self.dry_run = dry_run
        self.generator = TASignalGenerator()
        self.bregman = BregmanOptimizer(bankroll=bankroll)
        self.ml = MLOptimizer()
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.clob_balance = None  # Actual CLOB USDC balance (synced periodically)
        self.asset_consecutive_losses: Dict[str, int] = {}  # Per-asset loss tracker
        self.ASSET_MAX_CONSECUTIVE_LOSSES = 3  # Stop trading asset after N consecutive losses
        self.asset_cooldown_start: Dict[str, str] = {}  # ISO timestamp when cooldown began
        self.ASSET_COOLDOWN_MINUTES = 30  # Time-based expiry to prevent permanent lockout

        self.trades: Dict[str, LiveTrade] = {}
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.signals_count = 0
        self.ml_rejections = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.start_time = datetime.now(timezone.utc).isoformat()

        # Low-edge circuit breaker: 3 consecutive low-edge losses → revert to 0.30 for 2 hours
        self.low_edge_consecutive_losses = 0
        self.low_edge_lockout_until = None  # datetime when lockout expires

        # V3.5 MOMENTUM PAUSE: After 2 consecutive losses, pause 30 min
        # Data: After 2 losses WR=29.7% vs After 2 wins WR=63.8% (279 trades)
        self.momentum_pause_until = None  # datetime when pause expires

        # Duplicate order protection - track markets we've traded this cycle
        self.recently_traded_markets: set = set()

        # NYU Two-Parameter Volatility Model (V3.3 port)
        # V3.15: DISABLED — shadow data showed NYU blocked 43W/19L = +$88.30 in missed winners
        self.nyu_model = NYUVolatilityModel()
        self.use_nyu_model = False  # V3.15: Disabled, was costing +$88 in missed winners

        # === HOURLY ML POSITION SIZING ===
        # Bayesian approach: start at 1.0x, reduce for low WR hours, boost high WR after proof
        # Prior: 5 virtual trades at 50% WR per hour (regularization)
        # Phase 1 (first day): only REDUCE (0.5x-1.0x multiplier)
        # Phase 2 (after 24h + 10 trades): allow BOOST (0.5x-1.5x multiplier)
        self.hourly_stats: Dict[int, dict] = {
            h: {"wins": 0, "losses": 0, "pnl": 0.0} for h in range(24)
        }

        # === IH2P ADAPTATIONS STATE ===
        self.pending_topups: Dict[str, dict] = {}  # DCA top-up orders waiting for next scan

        # === SHADOW TRADE TRACKER ===
        # Records trades blocked by ATR filter + counterfactual trend bias sizing
        # Used to ML-evaluate whether these filters should stay or be removed
        self.shadow_trades: Dict[str, dict] = {}  # {trade_key: shadow_trade_dict}
        # V3.17: GARCH + CEEMD caches
        self._garch_cache = {"timestamp": 0, "result": None}
        self._ceemd_cache = {"timestamp": 0, "result": None}
        self._last_garch = {"forecast": 0.0, "regime": "UNKNOWN", "ratio": 1.0}
        self._last_ceemd = {"trend_slope": 0.0, "noise_ratio": 0.5, "n_imfs": 0}

        self.shadow_stats = {
            "atr_blocked": 0, "atr_blocked_wins": 0, "atr_blocked_losses": 0, "atr_blocked_pnl": 0.0,
            "trend_bias_trades": 0, "trend_bias_actual_pnl": 0.0, "trend_bias_full_pnl": 0.0,
            # IH2P feature tracking
            "bond_trades": 0, "bond_wins": 0, "bond_losses": 0, "bond_pnl": 0.0,
            "dca_topups": 0, "dca_topup_pnl": 0.0,
            "hedge_trades": 0, "hedge_wins": 0, "hedge_losses": 0, "hedge_pnl": 0.0,
            # V3.17: GARCH regime tracking
            "garch_trades_low": 0, "garch_wins_low": 0,
            "garch_trades_normal": 0, "garch_wins_normal": 0,
            "garch_trades_high": 0, "garch_wins_high": 0,
            # V3.17: CEEMD alignment tracking
            "ceemd_aligned_trades": 0, "ceemd_aligned_wins": 0,
            "ceemd_opposed_trades": 0, "ceemd_opposed_wins": 0,
            "ceemd_noisy_blocked": 0, "ceemd_noisy_shadow_wins": 0,
        }
        # === FILTER REJECTION TRACKING (V3.9) ===
        # Every filter rejection creates a shadow trade to measure filter effectiveness
        self.filter_stats: Dict[str, dict] = {}  # {filter_name: {blocked, wins, losses, pnl}}

        # === V3.18: ML-TUNABLE FILTER THRESHOLDS ===
        # Auto-evolve adjusts these at runtime based on shadow trade data
        self.ml_vwap_abs_cap = 0.80    # V3.18: Was 0.50 hardcoded
        self.ml_vwap_dir_cap = 0.40    # V3.18: Was 0.25 hardcoded
        self.ml_kl_floor = 0.03        # V3.18: Was 0.10 hardcoded

        # === V3.15: AUTO-REVERT FILTER MODE ===
        # Track post-reduction WR. If < 50% after 15 trades, restore strict filters.
        self.filter_mode = "LOOSE"  # "LOOSE" = reduced filters, "STRICT" = original filters
        self.filter_mode_trades = 0  # Trades since filter mode changed
        self.filter_mode_wins = 0
        self.filter_mode_losses = 0
        self.FILTER_REVERT_THRESHOLD = 15  # Check after this many trades
        self.FILTER_REVERT_MIN_WR = 0.50  # Revert if WR below this

        # === V3.15b: HYDRA LIVE PROBATION ===
        # Promoted from quarantine based on paper performance.
        # Auto-demote back to quarantine if WR < 40% after 8+ trades.
        self.hydra_promoted = ["MULTI_SMA_TREND", "TRENDLINE_BREAK", "ALIGNMENT"]  # V3.16: ALIGNMENT promoted (88.9% WR, +$73 paper)
        self.hydra_boost = 0.10  # Confidence boost when hydra agrees
        self.HYDRA_DEMOTE_MIN_TRADES = 8   # Min trades before demotion check
        self.HYDRA_DEMOTE_MIN_WR = 0.40    # Demote if WR below this
        # Per-strategy live tracking
        self.hydra_live_stats: Dict[str, dict] = {
            name: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "status": "PROBATION"}
            for name in self.hydra_promoted
        }
        self.hydra_last_signals: Dict[str, dict] = {}  # {asset: {strategy: direction}}

        # V3.15b: Loss forensics engine (@sharbel's "why did this lose?" loop)
        self.forensics = ForensicsEngine() if FORENSICS_AVAILABLE else None

        # Executor for live trading
        self.executor = None
        if not dry_run:
            self._init_executor()

        self._load()
        self._resolve_stale_trades()

    def _init_executor(self):
        """Initialize the trade executor."""
        try:
            from arbitrage.executor import Executor
            self.executor = Executor()
            if self.executor._initialized:
                print("[Live] Executor initialized - LIVE TRADING ENABLED")
            else:
                print("[Live] Executor failed to initialize - DRY RUN MODE")
                self.dry_run = True
        except Exception as e:
            print(f"[Live] Executor error: {e} - DRY RUN MODE")
            self.dry_run = True

    def _sync_clob_balance(self) -> float:
        """Query actual USDC balance from Polymarket CLOB. Returns balance or -1 on error."""
        if self.dry_run or not self.executor or not self.executor._initialized:
            return -1
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            balance_data = self.executor.client.get_balance_allowance(params)
            if balance_data and isinstance(balance_data, dict):
                raw = float(balance_data.get("balance", 0))
                # CLOB returns balance in USDC atomic units (6 decimals) or as decimal
                usdc = raw / 1e6 if raw > 1000 else raw
                self.clob_balance = usdc
                return usdc
        except Exception as e:
            print(f"[BALANCE] Error querying CLOB: {e}")
        return -1

    def _cancel_all_pending_orders(self):
        """Cancel all pending GTC orders on startup to prevent ghost order stacking."""
        if self.dry_run or not self.executor or not self.executor._initialized:
            return
        try:
            orders = self.executor.get_open_orders()
            if orders:
                count = len(orders) if isinstance(orders, list) else 0
                print(f"[STARTUP] Found {count} pending GTC order(s) — cancelling ALL to prevent stacking")
                self.executor.client.cancel_all()
                print(f"[STARTUP] All pending orders cancelled successfully")
            else:
                print(f"[STARTUP] No pending GTC orders found")
        except Exception as e:
            print(f"[STARTUP] Error cancelling orders: {e}")

    def _load(self):
        """Load previous state."""
        if self.OUTPUT_FILE.exists():
            try:
                data = json.load(open(self.OUTPUT_FILE))
                for tid, t in data.get("trades", {}).items():
                    # Handle features dict
                    features = t.pop("features", {})
                    self.trades[tid] = LiveTrade(**t, features=features)
                self.total_pnl = data.get("total_pnl", 0)
                self.wins = data.get("wins", 0)
                self.losses = data.get("losses", 0)
                self.signals_count = data.get("signals_count", 0)
                self.start_time = data.get("start_time", self.start_time)
                self.consecutive_wins = data.get("consecutive_wins", 0)
                self.consecutive_losses = data.get("consecutive_losses", 0)
                self.bankroll = data.get("bankroll", self.bankroll)

                # Load ML state
                self.ml.win_features = data.get("ml_win_features", [])
                self.ml.loss_features = data.get("ml_loss_features", [])

                # Load shadow trade state
                self.shadow_trades = data.get("shadow_trades", {})
                saved_stats = data.get("shadow_stats", {})
                # Merge saved stats into defaults (preserves new keys)
                for k, v in saved_stats.items():
                    self.shadow_stats[k] = v

                # Load filter rejection stats (V3.9)
                self.filter_stats = data.get("filter_stats", {})
                # V3.18: Reset stale shadow stats for loosened/disabled filters (clean slate)
                for fname in ["edge_floor", "kl_divergence", "nyu_vol", "v38_vwap", "v39_eth_price", "v314_vwap_dist"]:
                    if fname in self.filter_stats:
                        self.filter_stats[fname] = {"blocked": 0, "wins": 0, "losses": 0, "pnl": 0.0}

                # V3.18: Restore ML-tuned filter thresholds
                ml_t = data.get("ml_thresholds", {})
                if ml_t:
                    self.ml_vwap_abs_cap = ml_t.get("vwap_abs_cap", self.ml_vwap_abs_cap)
                    self.ml_vwap_dir_cap = ml_t.get("vwap_dir_cap", self.ml_vwap_dir_cap)
                    self.ml_kl_floor = ml_t.get("kl_floor", self.ml_kl_floor)
                    self.ETH_MAX_PRICE = ml_t.get("eth_max_price", self.ETH_MAX_PRICE)
                    print(f"[ML] Restored thresholds: VWAP_abs={self.ml_vwap_abs_cap:.2f}, VWAP_dir={self.ml_vwap_dir_cap:.2f}, KL={self.ml_kl_floor:.3f}, ETH_max={self.ETH_MAX_PRICE:.2f}")

                # Restore IH2P feature flags from saved state
                # V3.15c: Only restore ML-revoked flags, don't override class-level re-enable
                ih2p = data.get("ih2p_features", {})
                if "bond_enabled" in ih2p:
                    if not ih2p["bond_enabled"] and not self.BOND_MODE_ENABLED:
                        self.BOND_MODE_ENABLED = False  # ML-revoked and class agrees
                    elif not ih2p["bond_enabled"] and self.BOND_MODE_ENABLED:
                        print("[LOAD] Bond mode was ML-revoked but class re-enabled it — keeping ENABLED")
                if "dca_enabled" in ih2p:
                    if not ih2p["dca_enabled"] and not self.DCA_ENABLED:
                        self.DCA_ENABLED = False
                if "hedge_enabled" in ih2p:
                    if not ih2p["hedge_enabled"] and not self.HEDGE_ENABLED:
                        self.HEDGE_ENABLED = False

                # Load hourly stats
                saved_hourly = data.get("hourly_stats", {})
                for h_str, stats in saved_hourly.items():
                    h = int(h_str)
                    if h in self.hourly_stats:
                        self.hourly_stats[h] = stats

                # V3.11: Restore per-asset consecutive loss tracker
                self.asset_consecutive_losses = data.get("asset_consecutive_losses", {})
                self.asset_cooldown_start = data.get("asset_cooldown_start", {})

                # V3.13: Restore DOWN expensive ML tracking
                down_ml = data.get("down_expensive_ml", {})
                self.DOWN_EXPENSIVE_TRADES = down_ml.get("trades", 0)
                self.DOWN_EXPENSIVE_WINS = down_ml.get("wins", 0)
                self.DOWN_EXPENSIVE_LOSSES = down_ml.get("losses", 0)
                self.DOWN_EXPENSIVE_PNL = down_ml.get("pnl", 0.0)

                # V3.15: Restore filter mode + hydra promotion state
                fm = data.get("filter_mode", {})
                if fm.get("mode"):
                    self.filter_mode = fm["mode"]
                    self.filter_mode_trades = fm.get("trades", 0)
                    self.filter_mode_wins = fm.get("wins", 0)
                    self.filter_mode_losses = fm.get("losses", 0)
                    # If mode was reverted to STRICT, re-apply strict settings
                    if self.filter_mode == "STRICT":
                        self.MIN_EDGE = 0.25
                        self.LOW_EDGE_THRESHOLD = 0.25
                        self.MAX_ENTRY_PRICE = 0.45
                        self.MIN_ENTRY_PRICE = 0.25
                        self.ETH_MIN_CONFIDENCE = 0.70
                        self.SOL_DOWN_MIN_EDGE = 0.35
                        self.SKIP_HOURS_UTC = {5, 6, 8, 9, 10, 12, 14, 16}
                        self.MIN_TIME_REMAINING = 5.0
                        self.MAX_TIME_REMAINING = 9.0
                        self.use_nyu_model = True
                hl = data.get("hydra_live", {})
                saved_promoted = hl.get("promoted", [])
                if hl.get("per_strategy"):
                    self.hydra_live_stats = hl["per_strategy"]
                # Merge: keep saved list but add any new default promotions not yet demoted
                demoted = {n for n, s in self.hydra_live_stats.items() if s.get("status") == "DEMOTED"}
                for name in self.hydra_promoted:
                    if name not in saved_promoted and name not in demoted:
                        saved_promoted.append(name)
                        print(f"[HYDRA] New promotion: {name} added to live probation")
                self.hydra_promoted = saved_promoted
                # Init stats for any newly promoted strategies
                for name in self.hydra_promoted:
                    if name not in self.hydra_live_stats:
                        self.hydra_live_stats[name] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "status": "PROBATION"}

                shadow_open = sum(1 for s in self.shadow_trades.values() if s.get("status") == "open")
                print(f"[Live] Loaded {len(self.trades)} trades + {len(self.shadow_trades)} shadow ({shadow_open} open)")
            except Exception as e:
                print(f"[Live] Error loading state: {e}")

    def _resolve_stale_trades(self):
        """Resolve any open trades that are past expiry on startup.

        Prevents orphaned trades when the process dies mid-trade and restarts.
        Uses the same gamma-api resolution logic as the main scan loop.
        """
        now = datetime.now(timezone.utc)
        stale = [(tid, t) for tid, t in self.trades.items()
                 if t.status == "open" and
                 (now - datetime.fromisoformat(t.entry_time)).total_seconds() / 60 > 16]
        if not stale:
            return

        print(f"[Startup] Resolving {len(stale)} stale open trade(s)...")
        import httpx
        for tid, trade in stale:
            try:
                r = httpx.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={"condition_id": trade.market_id, "limit": "1"},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=10
                )
                if r.status_code == 200 and r.json():
                    mkt = r.json()[0]
                    outcomes = mkt.get("outcomes", [])
                    prices = mkt.get("outcomePrices", [])
                    if isinstance(outcomes, str):
                        outcomes = json.loads(outcomes)
                    if isinstance(prices, str):
                        prices = json.loads(prices)

                    res_price = None
                    for i, outcome in enumerate(outcomes):
                        if str(outcome).lower() == trade.side.lower() and i < len(prices):
                            res_price = float(prices[i])
                            break

                    if res_price is not None:
                        if res_price >= 0.95:
                            exit_val = trade.size_usd / trade.entry_price
                        elif res_price <= 0.05:
                            exit_val = 0
                        else:
                            exit_val = (trade.size_usd / trade.entry_price) * res_price

                        trade.exit_price = res_price
                        trade.exit_time = now.isoformat()
                        trade.pnl = exit_val - trade.size_usd
                        trade.status = "closed"

                        self.total_pnl += trade.pnl
                        won = trade.pnl > 0
                        if won:
                            self.wins += 1
                            self.consecutive_wins += 1
                            self.consecutive_losses = 0
                        else:
                            self.losses += 1
                            self.consecutive_losses += 1
                            self.consecutive_wins = 0
                            if self.consecutive_losses >= 2:
                                self.momentum_pause_until = datetime.now(timezone.utc) + timedelta(minutes=30)
                        # V3.12c: Sync bankroll from CLOB (fixes double-counting)
                        clob_startup = self._sync_clob_balance()
                        if clob_startup >= 0:
                            self.bankroll = clob_startup
                        else:
                            self.bankroll = max(self.initial_bankroll * 0.5, self.bankroll + trade.pnl)
                        self.ml.update_weights(trade.features, won)

                        # V3.11: Per-asset consecutive loss tracking on startup resolution
                        if trade.market_title.startswith("["):
                            ta = trade.market_title[1:trade.market_title.index("]")]
                            if won:
                                self.asset_consecutive_losses[ta] = 0
                                if ta in self.asset_cooldown_start:
                                    del self.asset_cooldown_start[ta]
                            else:
                                self.asset_consecutive_losses[ta] = self.asset_consecutive_losses.get(ta, 0) + 1
                                acl_s = self.asset_consecutive_losses[ta]
                                if acl_s >= self.ASSET_MAX_CONSECUTIVE_LOSSES and ta not in self.asset_cooldown_start:
                                    self.asset_cooldown_start[ta] = datetime.now(timezone.utc).isoformat()

                        result = "WIN" if won else "LOSS"
                        print(f"  [{result}] {trade.side} ${trade.pnl:+.2f} | {trade.market_title[:50]}")
                        # V3.13: ML auto-tighten tracking for expensive DOWN trades
                        if trade.side == "DOWN" and trade.entry_price > 0.45:
                            self.DOWN_EXPENSIVE_TRADES += 1
                            self.DOWN_EXPENSIVE_PNL += trade.pnl
                            if won:
                                self.DOWN_EXPENSIVE_WINS += 1
                            else:
                                self.DOWN_EXPENSIVE_LOSSES += 1
                            print(f"  [ML-TRACK] DOWN>${0.45}: {self.DOWN_EXPENSIVE_WINS}W/{self.DOWN_EXPENSIVE_LOSSES}L PnL=${self.DOWN_EXPENSIVE_PNL:+.2f}")
                    else:
                        trade.status = "closed"
                        trade.pnl = 0.0
                        trade.exit_price = trade.entry_price
                        trade.exit_time = now.isoformat()
                        print(f"  [EXPIRED] {trade.side} | No resolved price | {trade.market_title[:50]}")
                else:
                    trade.status = "closed"
                    trade.pnl = 0.0
                    trade.exit_price = trade.entry_price
                    trade.exit_time = now.isoformat()
                    print(f"  [EXPIRED] {trade.side} | Market not found | {trade.market_title[:50]}")
            except Exception as e:
                print(f"  [ERROR] {tid[:20]}: {e}")

        self._save()
        print(f"[Startup] Resolved {len(stale)} stale trade(s), saved state")

    ARCHIVE_FILE = Path(__file__).parent / "ta_live_archive.json"

    def _save(self):
        """Save current state to working file AND permanent archive."""
        # V3.15c FIX: Recalculate total_pnl from actual trades every save (prevents accumulator drift)
        self.total_pnl = round(sum(t.pnl for t in self.trades.values() if t.status == "closed"), 4)
        data = {
            "start_time": self.start_time,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "dry_run": self.dry_run,
            "trades": {tid: asdict(t) for tid, t in self.trades.items()},
            "total_pnl": self.total_pnl,
            "wins": self.wins,
            "losses": self.losses,
            "signals_count": self.signals_count,
            "ml_rejections": self.ml_rejections,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "bankroll": self.bankroll,
            "ml_win_features": self.ml.win_features[-100:],  # Keep last 100
            "ml_loss_features": self.ml.loss_features[-100:],
            "ml_weights": self.ml.feature_weights,
            "shadow_trades": self.shadow_trades,
            "shadow_stats": self.shadow_stats,
            "filter_stats": self.filter_stats,
            "hourly_stats": self.hourly_stats,
            "ih2p_features": {
                "bond_enabled": self.BOND_MODE_ENABLED,
                "dca_enabled": self.DCA_ENABLED,
                "hedge_enabled": self.HEDGE_ENABLED,
            },
            "asset_consecutive_losses": self.asset_consecutive_losses,
            "asset_cooldown_start": self.asset_cooldown_start,
            "down_expensive_ml": {
                "trades": self.DOWN_EXPENSIVE_TRADES,
                "wins": self.DOWN_EXPENSIVE_WINS,
                "losses": self.DOWN_EXPENSIVE_LOSSES,
                "pnl": round(self.DOWN_EXPENSIVE_PNL, 2),
            },
            # V3.15: Filter mode + hydra promotion persistence
            "filter_mode": {
                "mode": self.filter_mode,
                "trades": self.filter_mode_trades,
                "wins": self.filter_mode_wins,
                "losses": self.filter_mode_losses,
            },
            "hydra_live": {
                "promoted": self.hydra_promoted,
                "per_strategy": self.hydra_live_stats,
            },
            # V3.18: ML-tunable filter thresholds (auto-evolve adjusts these)
            "ml_thresholds": {
                "vwap_abs_cap": self.ml_vwap_abs_cap,
                "vwap_dir_cap": self.ml_vwap_dir_cap,
                "kl_floor": self.ml_kl_floor,
                "eth_max_price": self.ETH_MAX_PRICE,
            },
        }
        # Working file (can be reset)
        with open(self.OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

        # Permanent archive (NEVER WIPE) - append new closed trades
        self._append_to_archive()

    def _append_to_archive(self):
        """Append closed trades to permanent archive (NEVER WIPE THIS FILE)."""
        try:
            # Load existing archive
            if self.ARCHIVE_FILE.exists():
                with open(self.ARCHIVE_FILE, 'r') as f:
                    archive = json.load(f)
                # Normalize: if archive is a plain list, convert to dict format
                if isinstance(archive, list):
                    archive = {'trades': archive, 'daily_snapshots': []}
            else:
                archive = {'trades': [], 'daily_snapshots': []}

            # Get existing trade IDs to avoid duplicates
            existing_ids = {t.get('trade_id') for t in archive.get('trades', [])}

            # Add new closed trades
            for tid, trade in self.trades.items():
                if trade.status == 'closed' and tid not in existing_ids:
                    trade_data = asdict(trade)
                    trade_data['archived_at'] = datetime.now(timezone.utc).isoformat()
                    archive['trades'].append(trade_data)

            # Save archive
            with open(self.ARCHIVE_FILE, 'w') as f:
                json.dump(archive, f, indent=2)
        except Exception as e:
            print(f"[Archive] Error: {e}")

    # Multi-asset configuration
    # V3.9: ETH promoted to live (UP only). Paper: 70% WR on ETH UP, +$158 over 20 trades.
    # ETH DOWN stays blocked (47.1% WR in paper — not worth the risk during capital rebuild).
    ASSETS = {
        "BTC": {
            "symbol": "BTCUSDT",
            "keywords": ["bitcoin", "btc"],
        },
        "SOL": {
            "symbol": "SOLUSDT",
            "keywords": ["solana", "sol"],
        },
        "ETH": {
            "symbol": "ETHUSDT",
            "keywords": ["ethereum", "eth"],
        },
    }
    # Shadow-only assets: fetched + signals generated + logged, but NEVER executed
    SHADOW_ASSETS = {}
    # ETH-specific constraints (V3.9)
    ETH_UP_ONLY = True  # Only allow UP trades on ETH (70% WR vs 47% DOWN)
    ETH_MAX_PRICE = 0.70  # V3.18: Was 0.55, shadow 23W/10L 70% WR blocked (+$24.97 missed)
    ETH_MIN_CONFIDENCE = 0.70  # V3.16: Restored to 0.70. ETH is -$3.87 live (50% WR, only loser asset).
    # SOL DOWN constraint (V3.10): Paper data shows SOL_DOWN = 50% WR (coin flip)
    # Require higher edge for SOL DOWN trades (edge >= 0.35 lifts to ~71% WR per paper analysis)
    SOL_DOWN_MIN_EDGE = 0.20  # V3.15: Was 0.35, blocked +$5.50 in winners. Loosened.

    # Directional bias based on 200 EMA trend
    # Below 200 EMA = bearish: 70% capital on DOWN, 30% on UP
    # Above 200 EMA = bullish: 70% capital on UP, 30% on DOWN
    # TODO: ML should tune these ratios over time
    TREND_BIAS_STRONG = 0.70   # Capital % for with-trend trades
    TREND_BIAS_WEAK = 0.30     # Capital % for counter-trend trades (30% less)

    # Trading hours filter (UTC) - skip low-WR hours
    # V3.5: Expanded from {0,1,8,22,23} based on overnight massacre:
    #   08h: 0W/4L (-$15.38), 10h: 0W/2L (-$13.10), 11h: 0W/2L (-$9.44)
    #   12h: 0W/7L (-$51.94), 13h: 0W/1L (-$6.72), 15h: 0W/2L (-$14.12)
    #   20h: 0W/2L (-$8.78)
    # Shadow-tracked on paper account for re-evaluation
    # V3.6: Reopened UTC 10-13 (73-80% WR in paper, +$222 from 34 trades)
    # Only skip: UTC 1 (no data), UTC 8 (20% WR), UTC 14 (33% WR), UTC 15 (no data)
    # V3.15: Reduced from 8 to 4 skip hours to increase trade flow.
    # Keep only the absolute worst: 8 (20% WR), 12 (-$52 PnL), 5 (0 data), 16 (worst).
    # Auto-revert will catch if reopened hours lose money.
    SKIP_HOURS_UTC = {5, 8, 12, 16}

    def _ema(self, candles, period: int) -> float:
        """Calculate EMA from candle close prices."""
        closes = [c.close for c in candles]
        if len(closes) < period:
            return closes[-1] if closes else 0
        mult = 2 / (period + 1)
        ema = sum(closes[:period]) / period
        for price in closes[period:]:
            ema = (price - ema) * mult + ema
        return ema

    def _get_trend_bias(self, candles, price: float) -> str:
        """V3.13: Dual-timeframe trend — 200 EMA for macro + 30-min slope for active moves.
        Old: 200 EMA only → missed BTC -1% dump, said BULLISH while falling knife.
        Fix: If price dropped >0.3% in last 30 candles, override to BEARISH."""
        ema200 = self._ema(candles, 200)
        macro_bull = price >= ema200

        # Short-term momentum: 30-candle (30 min) price change
        if len(candles) >= 30:
            price_30m_ago = candles[-30].close
            short_change = (price - price_30m_ago) / price_30m_ago if price_30m_ago else 0
        else:
            short_change = 0

        # Override: active selloff beats lagging EMA
        if macro_bull and short_change < -0.003:
            return "FALLING"  # Was BULLISH on EMA but price actively dropping >0.3%
        elif not macro_bull and short_change > 0.003:
            return "RISING"   # Was BEARISH on EMA but price actively rising >0.3%
        elif price < ema200:
            return "BEARISH"
        else:
            return "BULLISH"

    def _check_mtf_confirmation(self, candles, signal_side: str) -> bool:
        """V3.13 ML: Multi-timeframe confirmation. Paper: 0W/3L when TFs disagree.
        Returns True if short and long TF agree on direction."""
        if len(candles) < 60:
            return True  # Not enough data, assume agree
        short_candles = candles[-30:]
        long_candles = candles[-120:] if len(candles) >= 120 else candles

        short_closes = [c.close for c in short_candles]
        short_slope = (short_closes[-1] - short_closes[0]) / short_closes[0] if short_closes[0] else 0

        long_closes = [c.close for c in long_candles]
        long_slope = (long_closes[-1] - long_closes[0]) / long_closes[0] if long_closes[0] else 0

        if signal_side == "UP":
            return not (short_slope < -0.001 and long_slope < -0.001)
        else:
            return not (short_slope > 0.001 and long_slope > 0.001)

    def _get_price_momentum(self, candles: List, lookback: int = 5) -> float:
        """Calculate recent price momentum (% change over lookback candles)."""
        if len(candles) < lookback:
            return 0.0
        old_price = candles[-lookback].close
        new_price = candles[-1].close
        return (new_price - old_price) / old_price

    def _compute_atr(self, candles, period: int = 14) -> float:
        """Compute Average True Range from 1-min candles."""
        if len(candles) < period + 1:
            return 0.0
        trs = []
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        if len(trs) < period:
            return sum(trs) / len(trs) if trs else 0.0
        # Smoothed ATR (Wilder's method)
        atr = sum(trs[:period]) / period
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr

    # === V3.17: GARCH(1,1) VOLATILITY FORECAST ===
    def _compute_garch_vol(self, candles, lookback: int = 100) -> dict:
        """GARCH(1,1) conditional variance forecast for vol regime detection.
        Returns: {forecast: float, regime: str, ratio: float}
        """
        closes = [c.close for c in candles[-lookback:]]
        if len(closes) < 50:
            return {"forecast": 0.0, "regime": "UNKNOWN", "ratio": 1.0}

        returns = np.diff(np.log(closes)) * 100  # log returns in %

        try:
            from arch import arch_model
            model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', rescale=False)
            result = model.fit(disp='off', show_warning=False)
            forecast = result.forecast(horizon=1)
            cond_vol = float(np.sqrt(forecast.variance.values[-1, 0]))
        except Exception:
            cond_vol = float(np.std(returns[-20:]))

        hist_vol = float(np.std(returns[-60:])) if len(returns) >= 60 else cond_vol
        ratio = cond_vol / hist_vol if hist_vol > 0 else 1.0

        if ratio < 0.7:
            regime = "LOW"
        elif ratio > 1.4:
            regime = "HIGH"
        else:
            regime = "NORMAL"

        return {"forecast": round(cond_vol, 4), "regime": regime, "ratio": round(ratio, 4)}

    def _get_garch(self, candles):
        """Cached GARCH computation (25s TTL)."""
        now = time.time()
        if now - self._garch_cache["timestamp"] < 25 and self._garch_cache["result"]:
            return self._garch_cache["result"]
        result = self._compute_garch_vol(candles)
        self._garch_cache = {"timestamp": now, "result": result}
        return result

    # === V3.17: CEEMD TREND DECOMPOSITION ===
    def _compute_ceemd_trend(self, candles, lookback: int = 100) -> dict:
        """CEEMD trend extraction via lowest-frequency IMF.
        Returns: {trend_slope: float, noise_ratio: float, n_imfs: int}
        """
        closes = np.array([c.close for c in candles[-lookback:]])
        if len(closes) < 50:
            return {"trend_slope": 0.0, "noise_ratio": 0.5, "n_imfs": 0}

        try:
            from PyEMD import CEEMDAN
            ceemdan = CEEMDAN(trials=50, epsilon=0.05, ext_EMD=None)
            ceemdan.noise_seed(42)
            imfs = ceemdan.ceemdan(closes)

            if len(imfs) < 2:
                return {"trend_slope": 0.0, "noise_ratio": 0.5, "n_imfs": len(imfs)}

            trend = imfs[-1]
            trend_slope = (trend[-1] - trend[-10]) / closes[-1] if len(trend) >= 10 else 0.0

            noise_energy = sum(np.var(imf) for imf in imfs[:-1])
            total_energy = sum(np.var(imf) for imf in imfs)
            noise_ratio = noise_energy / total_energy if total_energy > 0 else 0.5

            return {
                "trend_slope": round(float(trend_slope), 6),
                "noise_ratio": round(float(noise_ratio), 4),
                "n_imfs": len(imfs)
            }
        except Exception:
            return {"trend_slope": 0.0, "noise_ratio": 0.5, "n_imfs": 0}

    def _get_ceemd(self, candles):
        """Cached CEEMD computation (25s TTL)."""
        now = time.time()
        if now - self._ceemd_cache["timestamp"] < 25 and self._ceemd_cache["result"]:
            return self._ceemd_cache["result"]
        result = self._compute_ceemd_trend(candles)
        self._ceemd_cache = {"timestamp": now, "result": result}
        return result

    def _is_volatile_atr(self, candles, period: int = 14, multiplier: float = 1.5) -> bool:
        """Check if recent price action is too volatile vs ATR.
        Returns True if avg range of last 3 bars > multiplier * ATR (skip trade).
        """
        if len(candles) < period + 4:
            return False  # Not enough data, allow trade
        atr = self._compute_atr(candles[:-3], period)
        if atr <= 0:
            return False
        # Average range of last 3 bars
        recent_ranges = []
        for c in candles[-3:]:
            recent_ranges.append(c.high - c.low)
        avg_recent = sum(recent_ranges) / len(recent_ranges)
        return avg_recent > multiplier * atr

    def _compute_adx(self, candles, period: int = 14):
        """Compute Average Directional Index.
        Returns dict with 'adx', 'plus_di', 'minus_di' or None.
        ADX > 30 = strong trend, < 20 = ranging/coiled.
        """
        if len(candles) < period * 2 + 1:
            return None
        plus_dm = []
        minus_dm = []
        tr_list = []
        for i in range(1, len(candles)):
            high_diff = candles[i].high - candles[i - 1].high
            low_diff = candles[i - 1].low - candles[i].low
            pdm = high_diff if high_diff > low_diff and high_diff > 0 else 0
            mdm = low_diff if low_diff > high_diff and low_diff > 0 else 0
            plus_dm.append(pdm)
            minus_dm.append(mdm)
            tr = max(
                candles[i].high - candles[i].low,
                abs(candles[i].high - candles[i - 1].close),
                abs(candles[i].low - candles[i - 1].close)
            )
            tr_list.append(tr)
        if len(tr_list) < period:
            return None
        atr = sum(tr_list[:period]) / period
        plus_di_smooth = sum(plus_dm[:period]) / period
        minus_di_smooth = sum(minus_dm[:period]) / period
        for i in range(period, len(tr_list)):
            atr = (atr * (period - 1) + tr_list[i]) / period
            plus_di_smooth = (plus_di_smooth * (period - 1) + plus_dm[i]) / period
            minus_di_smooth = (minus_di_smooth * (period - 1) + minus_dm[i]) / period
        if atr == 0:
            return None
        plus_di = (plus_di_smooth / atr) * 100
        minus_di = (minus_di_smooth / atr) * 100
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return None
        dx_values = []
        temp_atr = sum(tr_list[:period]) / period
        temp_pdm = sum(plus_dm[:period]) / period
        temp_mdm = sum(minus_dm[:period]) / period
        for i in range(period, len(tr_list)):
            temp_atr = (temp_atr * (period - 1) + tr_list[i]) / period
            temp_pdm = (temp_pdm * (period - 1) + plus_dm[i]) / period
            temp_mdm = (temp_mdm * (period - 1) + minus_dm[i]) / period
            if temp_atr == 0:
                continue
            pdi = (temp_pdm / temp_atr) * 100
            mdi = (temp_mdm / temp_atr) * 100
            di_s = pdi + mdi
            if di_s > 0:
                dx_values.append(abs(pdi - mdi) / di_s * 100)
        if len(dx_values) < period:
            return None
        adx = sum(dx_values[:period]) / period
        for dx in dx_values[period:]:
            adx = (adx * (period - 1) + dx) / period
        return {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}

    def _compute_macd_v(self, candles, fast: int = 12, slow: int = 26) -> Optional[float]:
        """MACD-V: Volatility Normalized Momentum (Spiroglou 2022).
        Formula: (EMA_fast - EMA_slow) / ATR(slow) * 100
        Returns momentum as % of ATR. Neutral zone: -50 to +50.
        Positive = bullish momentum, Negative = bearish momentum.
        """
        if len(candles) < slow + 1:
            return None
        ema_fast = self._ema(candles, fast)
        ema_slow = self._ema(candles, slow)
        atr = self._compute_atr(candles, slow)
        if atr <= 0:
            return None
        return ((ema_fast - ema_slow) / atr) * 100

    def _is_atr_compressing(self, candles, short: int = 20, long: int = 30) -> bool:
        """ATR Compression: short-term ATR < long-term ATR = coiled volatility.
        From BTA breakout model: ATR(20) < ATR(30) means volatility is
        compressing -> breakout imminent -> higher confidence entry.
        """
        if len(candles) < long + 1:
            return False
        atr_short = self._compute_atr(candles, short)
        atr_long = self._compute_atr(candles, long)
        if atr_long <= 0:
            return False
        return atr_short < atr_long

    async def fetch_data(self):
        """Fetch candles and markets for all assets (including shadow assets)."""
        asset_data = {}  # {asset: (candles, price, markets)}

        # Merge live + shadow assets for data fetching
        all_assets = {**self.ASSETS, **self.SHADOW_ASSETS}

        async with httpx.AsyncClient(timeout=15) as client:
            # Fetch candles + prices for all assets from Binance
            for asset, cfg in all_assets.items():
                try:
                    r = await client.get(
                        "https://api.binance.com/api/v3/klines",
                        params={"symbol": cfg["symbol"], "interval": "1m", "limit": 240}
                    )
                    klines = r.json()

                    pr = await client.get(
                        "https://api.binance.com/api/v3/ticker/price",
                        params={"symbol": cfg["symbol"]}
                    )
                    price = float(pr.json()["price"])

                    candles = [
                        Candle(k[0]/1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
                        for k in klines
                    ]

                    asset_data[asset] = {"candles": candles, "price": price, "markets": []}
                    await asyncio.sleep(0.3)  # Binance rate limit
                except Exception as e:
                    print(f"[API] Error fetching {asset}: {e}")

            # Fetch all 15m markets from Polymarket
            await asyncio.sleep(1.0)
            try:
                mr = await client.get(
                    "https://gamma-api.polymarket.com/events",
                    params={"tag_slug": "15M", "active": "true", "closed": "false", "limit": 50},
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                if mr.status_code == 200:
                    events = mr.json()
                    for event in events:
                        title = event.get("title", "").lower()
                        # Match event to asset
                        matched_asset = None
                        for asset, cfg in all_assets.items():
                            if any(kw in title for kw in cfg["keywords"]):
                                matched_asset = asset
                                break
                        if not matched_asset or matched_asset not in asset_data:
                            continue
                        for m in event.get("markets", []):
                            if not m.get("closed", True):
                                if not m.get("question"):
                                    m["question"] = event.get("title", "")
                                m["_asset"] = matched_asset  # Tag with asset name
                                asset_data[matched_asset]["markets"].append(m)
                    self._market_cache_all = asset_data
                else:
                    asset_data = getattr(self, '_market_cache_all', asset_data)
            except Exception as e:
                print(f"[API] Error fetching markets: {e}")
                asset_data = getattr(self, '_market_cache_all', asset_data)

        return asset_data

    def get_market_prices(self, market: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Extract UP/DOWN prices."""
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

    def get_time_remaining(self, market: Dict) -> float:
        """Get minutes remaining."""
        end = market.get("endDate")
        if not end:
            return 15.0
        try:
            end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return max(0, (end_dt - now).total_seconds() / 60)
        except:
            return 15.0

    async def execute_trade(self, market: Dict, side: str, size: float, price: float) -> Tuple[bool, str]:
        """Execute a live trade."""
        if self.dry_run:
            return True, "dry_run"

        if not self.executor or not self.executor._initialized:
            return False, "executor_not_ready"

        # === HARD_MAX_BET FINAL SAFETY NET ===
        # No matter what upstream code calculates, this is the absolute ceiling.
        if size > self.HARD_MAX_BET:
            print(f"[SAFETY] Size ${size:.2f} exceeds HARD_MAX_BET ${self.HARD_MAX_BET}. Clamped.")
            size = self.HARD_MAX_BET

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            # Get token ID
            token_ids = market.get("clobTokenIds", "[]")
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)

            outcomes = market.get("outcomes", [])
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)

            # Find the right token
            token_id = None
            for i, outcome in enumerate(outcomes):
                if str(outcome).lower() == side.lower() and i < len(token_ids):
                    token_id = token_ids[i]
                    break

            if not token_id:
                return False, "token_not_found"

            # === ACCOUNT-LEVEL POSITION CHECK ===
            # Query existing positions on this market to prevent overlap from ghost instances.
            condition_id = market.get("conditionId", "")
            if condition_id and hasattr(self.executor, 'client'):
                try:
                    existing = self.executor.client.get_positions(
                        params={"condition_id": condition_id}
                    ) if hasattr(self.executor.client, 'get_positions') else []
                    if existing:
                        for pos in existing:
                            pos_size = float(pos.get("size", 0))
                            if pos_size > 0:
                                print(f"[OVERLAP] Already have position on this market ({pos_size} shares). SKIPPING.")
                                return False, "position_already_exists"
                except Exception as e:
                    # Non-fatal: proceed if check fails, but log it
                    if "404" not in str(e) and "not found" not in str(e).lower():
                        print(f"[OVERLAP-CHECK] Warning: {e}")

            # 15-min markets have thin order books (asks at $0.97-$0.99 only).
            # Place limit orders at the outcomePrices mid price and let them fill.
            # MAX_ENTRY_PRICE check is already applied in conviction filters.
            print(f"[ORDER] Placing limit {side} @ ${price:.4f} (mid price)")

            # Calculate shares - CLOB requires minimum 5 shares
            shares = math.floor(size / price)
            if shares < 5:
                shares = 5  # Force minimum 5 shares (CLOB requirement)

            # FINAL cost check: shares * price must not exceed HARD_MAX_BET
            actual_cost = shares * price
            if actual_cost > self.HARD_MAX_BET + 0.50:  # Small tolerance for rounding
                shares = math.floor(self.HARD_MAX_BET / price)
                actual_cost = shares * price
                print(f"[SAFETY] Reduced to {shares} shares (${actual_cost:.2f}) to stay under ${self.HARD_MAX_BET}")

            order_args = OrderArgs(
                price=price,
                size=float(shares),
                side=BUY,
                token_id=token_id,
            )

            print(f"[LIVE] Placing {side} order: {shares} shares @ ${price} = ${actual_cost:.2f}")

            signed_order = self.executor.client.create_order(order_args)
            response = self.executor.client.post_order(signed_order, OrderType.GTC)

            success = response.get("success", False)
            order_id = response.get("orderID", "")

            if not success:
                return False, response.get("errorMsg", "unknown_error")

            # GTC placed — wait up to 90s for fill, then cancel if unfilled
            # V3.12c: Extended from 60s (10+20+30) to 90s (15+30+45) for thin books
            print(f"[LIVE] GTC order placed: {order_id[:20]}... waiting for fill...")
            import time as _time
            for wait_sec in [15, 30, 45]:
                _time.sleep(wait_sec)
                try:
                    order_status = self.executor.client.get_order(order_id)
                    status = order_status.get("status", "") if isinstance(order_status, dict) else ""
                    size_matched = float(order_status.get("size_matched", 0)) if isinstance(order_status, dict) else 0
                    if status == "MATCHED" or size_matched >= shares * 0.5:
                        print(f"[LIVE] Order FILLED ({size_matched:.0f}/{shares} shares)")
                        return True, order_id
                except Exception:
                    pass
            # Not filled after 90s — cancel to prevent hanging
            try:
                self.executor.client.cancel(order_id)
                print(f"[LIVE] Order CANCELLED after 90s (no fill at ${price})")
            except Exception:
                pass
            return False, "timeout_no_fill"

        except Exception as e:
            return False, str(e)

    async def sell_position(self, market: Dict, side: str, shares: int, price: float) -> Tuple[bool, str]:
        """Sell shares to exit a position early (take profit)."""
        if self.dry_run:
            return True, "dry_run"
        if not self.executor or not self.executor._initialized:
            return False, "executor_not_ready"
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType, BalanceAllowanceParams, AssetType
            from py_clob_client.order_builder.constants import SELL

            token_ids = market.get("clobTokenIds", "[]")
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            outcomes = market.get("outcomes", [])
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)

            token_id = None
            for i, outcome in enumerate(outcomes):
                if str(outcome).lower() == side.lower() and i < len(token_ids):
                    token_id = token_ids[i]
                    break
            if not token_id:
                return False, "token_not_found"

            # Approve conditional token for selling (required by CLOB)
            try:
                self.executor.client.update_balance_allowance(
                    BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=token_id)
                )
            except Exception as e:
                print(f"[SELL] Allowance update warning: {e}")

            order_args = OrderArgs(
                price=price,
                size=float(shares),
                side=SELL,
                token_id=token_id,
            )
            print(f"[SELL] Placing {side} sell: {shares} shares @ ${price:.4f}")
            signed_order = self.executor.client.create_order(order_args)
            response = self.executor.client.post_order(signed_order, OrderType.GTC)
            success = response.get("success", False)
            order_id = response.get("orderID", "")
            if not success:
                return False, response.get("errorMsg", "unknown_error")
            return True, order_id
        except Exception as e:
            return False, str(e)

    async def check_early_exit(self, markets: list):
        """Check open trades for early profit-taking opportunity.

        Sells when current price implies >= 85% of max possible profit AND
        there are signs of momentum reversal (price moving against us).

        For a DOWN trade bought at $0.33:
          Max profit = ($1.00 - $0.33) / $0.33 = 203%
          85% of max = price at $0.90 → sell

        ML-adaptive: starts at 85% threshold, tracks outcomes to optimize.
        """
        # V3.17: GARCH regime-adaptive take-profit
        garch = self._last_garch
        if garch["regime"] == "LOW":
            PROFIT_TAKE_PCT = 0.80   # Take faster in low vol (smaller moves available)
        elif garch["regime"] == "HIGH":
            PROFIT_TAKE_PCT = 0.90   # Let winners run in high vol (bigger swings)
        else:
            PROFIT_TAKE_PCT = 0.85

        for tid, trade in list(self.trades.items()):
            if trade.status != "open":
                continue

            # Find this trade's market in current market data
            mkt_match = None
            for mkt in markets:
                mkt_id = mkt.get("conditionId", "")
                if mkt_id in tid:
                    mkt_match = mkt
                    break
            if not mkt_match:
                continue

            # Get current price for our side
            up_p, down_p = self.get_market_prices(mkt_match)
            if up_p is None or down_p is None:
                continue
            current_price = up_p if trade.side == "UP" else down_p

            # Calculate unrealized profit percentage
            # We bought shares at entry_price. Each share pays $1 if we win.
            # Max profit per share = (1.0 - entry_price)
            # Current unrealized per share = (current_price - entry_price)
            max_profit_per_share = 1.0 - trade.entry_price
            unrealized_per_share = current_price - trade.entry_price

            if max_profit_per_share <= 0 or unrealized_per_share <= 0:
                continue  # Not in profit

            profit_ratio = unrealized_per_share / max_profit_per_share

            if profit_ratio >= PROFIT_TAKE_PCT:
                # We're at 85%+ of max profit — sell to lock it in
                shares = math.floor(trade.size_usd / trade.entry_price)
                if shares < 1:
                    continue

                # Sell slightly below current price for faster fill
                sell_price = round(current_price - 0.01, 4)
                sell_price = max(sell_price, trade.entry_price + 0.01)  # Never sell below entry

                print(f"[TAKE PROFIT] {trade.side} @ ${trade.entry_price:.3f} -> ${current_price:.3f} ({profit_ratio:.0%} of max) | Selling {shares} shares @ ${sell_price:.3f}")

                success, result = await self.sell_position(mkt_match, trade.side, shares, sell_price)

                if success:
                    # Calculate actual PnL from selling
                    sell_value = shares * sell_price
                    cost = shares * trade.entry_price
                    pnl = sell_value - cost

                    trade.exit_price = sell_price
                    trade.exit_time = datetime.now(timezone.utc).isoformat()
                    trade.pnl = pnl
                    trade.status = "closed"

                    self.total_pnl += pnl
                    self.wins += 1
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                    # V3.12c: Sync bankroll from CLOB (fixes double-counting)
                    clob_early = self._sync_clob_balance()
                    if clob_early >= 0:
                        self.bankroll = clob_early
                    else:
                        self.bankroll = max(self.initial_bankroll * 0.5, self.bankroll + pnl)
                    self.ml.update_weights(trade.features, True)

                    # Update hourly stats
                    try:
                        entry_hour = datetime.fromisoformat(trade.entry_time).hour
                        if entry_hour not in self.hourly_stats:
                            self.hourly_stats[entry_hour] = {"wins": 0, "losses": 0, "pnl": 0.0}
                        self.hourly_stats[entry_hour]["wins"] += 1
                        self.hourly_stats[entry_hour]["pnl"] += pnl
                    except Exception:
                        pass

                    print(f"[EARLY WIN] {trade.side} PnL: ${pnl:+.2f} ({profit_ratio:.0%} of max) | Bankroll: ${self.bankroll:.2f} | {trade.market_title[:40]}...")
                    # V3.13: ML auto-tighten tracking for expensive DOWN trades
                    if trade.side == "DOWN" and trade.entry_price > 0.45:
                        self.DOWN_EXPENSIVE_TRADES += 1
                        self.DOWN_EXPENSIVE_PNL += pnl
                        self.DOWN_EXPENSIVE_WINS += 1
                        print(f"  [ML-TRACK] DOWN>${0.45}: {self.DOWN_EXPENSIVE_WINS}W/{self.DOWN_EXPENSIVE_LOSSES}L PnL=${self.DOWN_EXPENSIVE_PNL:+.2f}")

                    # Auto-redeem
                    if not self.dry_run:
                        try:
                            auto_redeem_winnings()
                        except Exception:
                            pass

                    self._save()
                else:
                    print(f"[TAKE PROFIT] Sell failed: {result} | Will retry next cycle")

    async def verify_fill(self, token_id: str, expected_shares: float) -> bool:
        """Verify a position exists on Polymarket after placing an order."""
        try:
            import httpx
            proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
            if not proxy:
                return False

            r = httpx.get(
                "https://data-api.polymarket.com/positions",
                params={"user": proxy, "sizeThreshold": "0.1"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            if r.status_code != 200:
                print(f"[VERIFY] API returned {r.status_code}")
                return False

            positions = r.json()
            for p in positions:
                p_tokens = p.get("clobTokenIds", [])
                if isinstance(p_tokens, str):
                    p_tokens = json.loads(p_tokens)
                p_size = float(p.get("size", 0))
                # Check if this position matches our token
                if token_id in str(p.get("asset", "")) or token_id in str(p_tokens):
                    if p_size > 0:
                        print(f"[VERIFY] Found position: {p_size} shares")
                        return True

            # Also check recent activity as fallback
            r2 = httpx.get(
                "https://data-api.polymarket.com/activity",
                params={"user": proxy, "limit": "5"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            if r2.status_code == 200:
                acts = r2.json()
                for a in acts:
                    if a.get("type") == "TRADE" and a.get("side") == "BUY":
                        # Check if this trade happened in the last 30 seconds
                        ts = a.get("timestamp", "")
                        usd = float(a.get("usdcSize", 0))
                        if usd > 0 and abs(usd - (expected_shares * 0.55)) < 5:
                            print(f"[VERIFY] Found matching recent trade: ${usd:.2f}")
                            return True

            print("[VERIFY] No matching position or recent trade found")
            return False
        except Exception as e:
            print(f"[VERIFY] Error checking fill: {e}")
            return False

    # Conviction thresholds - V3.15: LOOSENED based on filter shadow data
    # Shadow stats showed most filters blocking more winners than losers:
    # NYU vol: +$88 missed, v33_conviction: +$47, edge_floor: +$27, KL: +$8
    MIN_MODEL_CONFIDENCE = 0.55  # Keep: basic sanity
    MIN_EDGE = 0.12              # V3.15: Was 0.25, blocked +$26.88 in winners. Loosened.
    LOW_EDGE_THRESHOLD = 0.18    # V3.15: Was 0.25, match new floor
    MAX_ENTRY_PRICE = 0.52       # V3.15: Was 0.45, blocked 87 winners. Widened.
    DOWN_MAX_PRICE = 0.65        # V3.13: Dynamic DOWN cap
    DOWN_EXPENSIVE_TRADES = 0    # ML auto-tighten: count trades where DOWN > $0.45
    DOWN_EXPENSIVE_WINS = 0      # ML auto-tighten: wins where DOWN > $0.45
    DOWN_EXPENSIVE_LOSSES = 0    # ML auto-tighten: losses where DOWN > $0.45
    DOWN_EXPENSIVE_PNL = 0.0     # ML auto-tighten: cumulative PnL for DOWN > $0.45
    MIN_ENTRY_PRICE = 0.38       # V3.16: Was 0.20. Live data: $0.20-0.40 = 33% WR, -$18. Cheap entries are losers.
    MIN_KL_DIVERGENCE = 0.08     # V3.12: Shadow 7W/1L 88%WR blocked at 0.15. Loosened.

    # Paper trades both sides successfully (UP 81% WR in paper)
    DOWN_ONLY_MODE = False       # Disabled - match paper

    # Criteria to re-enable UP trades (kept but not needed with DOWN_ONLY_MODE=False)
    UP_ENABLE_MIN_WINS = 5
    UP_ENABLE_MIN_WR = 0.65
    UP_ENABLE_MIN_PNL = 10.0

    # UP trade settings - V3.11: MATCHED TO PAPER (UP_MIN_CONFIDENCE=0.65 in paper)
    UP_MIN_CONFIDENCE = 0.68     # V3.18 CSV: 10/11 losses are UP. Raise from 0.65 to reduce UP losses.
    UP_MIN_EDGE = 0.25           # V3.12: Match new MIN_EDGE
    UP_RSI_MIN = 45              # Relaxed to match paper (was 60)
    CANDLE_LOOKBACK = 120        # V3.6b: Was 15 — killed MACD(35), TTM(25), EMA Cross(20), RSI slope. Now all 9 indicators active.
    DOWN_MIN_MOMENTUM_DROP = 0.001  # Match paper tuner (was -0.002, more permissive)

    # Risk management - matched to paper volumes
    MAX_DAILY_LOSS = 30.0        # Same as paper
    MAX_TOTAL_EXPOSURE = 50.0    # V3.5: Tightened (was 100)
    MAX_SLIPPAGE = 0.03          # Keep tight for live execution
    MAX_CONCURRENT_POSITIONS = 1 # V3.5: Max 1 — single best signal only (multi-asset was -$207)

    # === IH2P ADAPTATIONS (ALL DISABLED — capital rebuild mode) ===
    # DCA caused double-betting ($6.80 + $3.78 topup = $10.58 per market).
    # With 7 ghost processes, each one doing DCA = $21+ per market. NEVER AGAIN.
    BOND_MODE_ENABLED = True      # V3.15c: Enabled for directional markets ($0.85+ sides)
    BOND_MIN_CONFIDENCE = 0.40    # V3.15c: Lowered — market price ($0.85+) IS the confidence signal, model just can't disagree
    BOND_MIN_PRICE = 0.85         # Entry price must be $0.85+
    DCA_ENABLED = False           # DISABLED: No DCA stacking during capital rebuild
    DCA_INITIAL_RATIO = 0.60      # First entry: 60% of position
    DCA_TOPUP_RATIO = 0.40        # Second entry: 40% (next scan, same or better price)
    HEDGE_ENABLED = False         # DISABLED: No hedges during capital rebuild
    HEDGE_MAX_PRICE = 0.15        # Only hedge if opposite side <= $0.15
    HEDGE_SIZE = 1.50             # $1.50 hedge bet

    # Entry window - V3.15: widened for more trade flow
    MIN_TIME_REMAINING = 4.0     # V3.15: Was 5.0, loosened slightly
    MAX_TIME_REMAINING = 12.0    # V3.15: Was 9.0, widened for more opportunities

    # Kelly position sizing - CONSERVATIVE: slow churn, protect capital
    KELLY_FRACTION = 0.25        # Quarter-Kelly for safety

    def _get_hour_multiplier(self, hour: int) -> float:
        """
        Bayesian hourly position multiplier. Learns win rates per hour and adjusts.

        Phase 1 (< 10 total hourly trades): only REDUCE for bad hours (0.5x-1.0x)
        Phase 2 (>= 10 total hourly trades): allow BOOST for proven hours (0.5x-1.5x)

        Uses Beta distribution prior: 5 virtual wins + 5 virtual losses (50% WR prior)
        so new hours start neutral and need real data to move the needle.
        """
        stats = self.hourly_stats.get(hour, {"wins": 0, "losses": 0, "pnl": 0.0})
        real_trades = stats["wins"] + stats["losses"]

        # Bayesian posterior: add prior of 5W/5L to smooth early estimates
        prior_wins, prior_losses = 5, 5
        post_wins = stats["wins"] + prior_wins
        post_losses = stats["losses"] + prior_losses
        post_wr = post_wins / (post_wins + post_losses)  # Posterior mean

        # Neutral WR = 0.50 -> multiplier = 1.0
        # WR < 0.50 -> reduce toward 0.5x
        # WR > 0.50 -> boost toward 1.5x (Phase 2 only)
        total_all_hours = sum(s["wins"] + s["losses"] for s in self.hourly_stats.values())
        phase2 = total_all_hours >= 10

        if post_wr < 0.50:
            # Scale down: 0.50 WR -> 1.0x, 0.30 WR -> 0.5x
            mult = max(0.5, 0.5 + (post_wr / 0.50) * 0.5)
        elif phase2 and post_wr > 0.50:
            # Scale up: 0.50 WR -> 1.0x, 0.70 WR -> 1.5x
            mult = min(1.5, 1.0 + ((post_wr - 0.50) / 0.20) * 0.5)
        else:
            mult = 1.0

        return round(mult, 2)

    def calculate_position_size(self, edge: float, kelly_fraction: float,
                                  btc_1h_change: float, volatility: float) -> float:
        """
        Conservative position sizing with hourly ML multiplier.
        Base $3, scale with conviction, then adjust by time-of-day win rate.
        """
        # Start at base
        size = self.BASE_POSITION_SIZE

        # ML conviction scaling: only increase toward max with strong edge+kelly
        conviction = min(1.0, kelly_fraction * edge / 0.10)  # 0-1 score
        size += conviction * (self.MAX_POSITION_SIZE - self.BASE_POSITION_SIZE)

        # V3.17: GARCH-based vol regime damping (replaces static threshold)
        garch = self._last_garch  # Use cached from latest cycle
        if garch["regime"] == "HIGH":
            size *= 0.70   # Reduce in expanding vol
        elif garch["regime"] == "LOW":
            size *= 1.15   # Boost in contracting vol (trend continuation)

        # Losing streak protection: drop to minimum
        if self.consecutive_losses >= 2:
            size = self.MIN_POSITION_SIZE

        # === HOURLY ML MULTIPLIER ===
        hour = datetime.now(timezone.utc).hour
        hour_mult = self._get_hour_multiplier(hour)
        if hour_mult != 1.0:
            size *= hour_mult

        # === V3.16: DAY-OF-WEEK MULTIPLIER ===
        # CSV data: Monday = 70% WR ($0.76/trade), Wednesday = 93.4% WR ($6.20/trade)
        dow = datetime.now(timezone.utc).weekday()  # 0=Mon, 6=Sun
        if dow == 0:  # Monday
            size *= 0.80
        elif dow == 5 or dow == 6:  # Saturday/Sunday — CSV shows strong weekends
            size *= 1.10

        # === V3.18: SIDE MULTIPLIER (CSV: DOWN=99% WR, UP=82% WR) ===
        # 10/11 losses are UP trades. DOWN is near-perfect. Adjust sizing accordingly.
        if hasattr(self, '_current_trade_side'):
            if self._current_trade_side == "DOWN":
                size *= 1.15  # Boost DOWN — 99% WR in CSV
            elif self._current_trade_side == "UP":
                size *= 0.85  # Reduce UP — 82% WR, source of 10/11 losses

        # Hard clamp — HARD_MAX_BET is the absolute ceiling
        size = max(self.MIN_POSITION_SIZE, min(self.HARD_MAX_BET, size))
        return round(size, 2)

    async def run_cycle(self):
        """Run one trading cycle across all assets."""
        # Skip low win-rate hours — still resolve open trades, just don't place new ones
        current_hour = datetime.now(timezone.utc).hour
        is_skip_hour = current_hour in self.SKIP_HOURS_UTC
        if is_skip_hour:
            open_count = sum(1 for t in self.trades.values() if t.status == "open")
            if open_count > 0:
                print(f"[SKIP] Hour {current_hour:02d} UTC - resolving {open_count} open trades...")
            else:
                print(f"[SKIP] Hour {current_hour:02d} UTC is in skip list - resting")
                return None

        asset_data = await self.fetch_data()

        if not asset_data:
            return None

        # Process pending DCA top-ups from previous cycle
        await self._process_dca_topups(asset_data)

        # Summary of markets found
        total_markets = sum(len(d["markets"]) for d in asset_data.values())
        asset_counts = {a: len(d["markets"]) for a, d in asset_data.items() if d["markets"]}
        if asset_counts:
            counts_str = " | ".join(f"{a}:{c}" for a, c in asset_counts.items())
            print(f"[Markets] {total_markets} total ({counts_str})")

        self.signals_count += 1
        main_signal = None

        # === V3.15b: HYDRA LIVE PROBATION SCAN ===
        # Run promoted strategies on each asset's candle data for directional confirmation
        if HYDRA_AVAILABLE and self.hydra_promoted:
            try:
                candles_for_hydra = {
                    asset: data["candles"] for asset, data in asset_data.items()
                    if data.get("candles")
                }
                hydra_signals = scan_strategies(candles_for_hydra)
                self.hydra_last_signals = {}
                for hsig in hydra_signals:
                    if hsig.name in self.hydra_promoted and hsig.asset:
                        # Track per-asset, per-strategy
                        if hsig.asset not in self.hydra_last_signals:
                            self.hydra_last_signals[hsig.asset] = {}
                        self.hydra_last_signals[hsig.asset][hsig.name] = hsig.direction
                        if self.signals_count % 10 == 0:
                            print(f"  [HYDRA] {hsig.name} -> {hsig.asset} {hsig.direction} (conf={hsig.confidence:.0%})")
            except Exception as e:
                if self.signals_count % 50 == 0:
                    print(f"  [HYDRA] Error: {e}")

        # Process each asset (live + shadow)
        for asset, data in asset_data.items():
            is_shadow_asset = asset in self.SHADOW_ASSETS
            candles = data["candles"]
            price = data["price"]
            markets = data["markets"]

            if not candles or not markets:
                continue

            # === EARLY PROFIT-TAKING: Check open trades for this asset ===
            await self.check_early_exit(markets)

            recent_candles = candles[-self.CANDLE_LOOKBACK:]

            # V3.17: Compute GARCH + CEEMD for this asset's candles (cached 25s)
            if asset == "BTC":  # Primary asset drives regime
                self._last_garch = self._get_garch(candles)
                self._last_ceemd = self._get_ceemd(candles)

            # Generate main signal for display (use BTC as primary)
            if asset == "BTC":
                main_signal = self.generator.generate_signal(
                    market_id="btc_main",
                    candles=recent_candles,
                    current_price=price,
                    market_yes_price=0.5,
                    market_no_price=0.5,
                    time_remaining_min=10.0
                )

            # Find nearest eligible market for this asset
            eligible_markets = []
            for market in markets:
                up_price, down_price = self.get_market_prices(market)
                time_left = self.get_time_remaining(market)
                if up_price is None or down_price is None:
                    continue
                if time_left > self.MAX_TIME_REMAINING or time_left < self.MIN_TIME_REMAINING:
                    continue
                eligible_markets.append((time_left, market, up_price, down_price))

            eligible_markets.sort(key=lambda x: x[0])

            if not eligible_markets:
                continue

            print(f"[{asset}] ${price:,.2f} | nearest market: {eligible_markets[0][0]:.1f}min left")

            # Try all eligible markets (nearest first) — near-expiry markets may already be heavily priced
            for time_left, market, up_price, down_price in eligible_markets:
                if is_skip_hour:
                    break
                market_id = market.get("conditionId", "")
                question = market.get("question", "")

                signal = self.generator.generate_signal(
                    market_id=market_id,
                    candles=recent_candles,
                    current_price=price,
                    market_yes_price=up_price,
                    market_no_price=down_price,
                    time_remaining_min=time_left
                )

                trade_key = f"{market_id}_{signal.side}" if signal.side else None

                # === AUTO-ENABLE UP TRADES when criteria met ===
                if self.DOWN_ONLY_MODE:
                    total_trades = self.wins + self.losses
                    win_rate = self.wins / total_trades if total_trades > 0 else 0
                    if (self.wins >= self.UP_ENABLE_MIN_WINS and
                        win_rate >= self.UP_ENABLE_MIN_WR and
                        self.total_pnl >= self.UP_ENABLE_MIN_PNL):
                        self.DOWN_ONLY_MODE = False
                        print(f"[MODE] UP TRADES RE-ENABLED! ({self.wins}W/{self.losses}L = {win_rate:.0%} WR, ${self.total_pnl:.2f} PnL)")

                # === DOWN ONLY MODE - skip UP trades ===
                if self.DOWN_ONLY_MODE and signal.side == "UP":
                    continue  # Silent skip - don't spam logs

                # === V3.9: ETH UP-ONLY + BTC DOWN CONTRARIAN BLOCK ===
                # ETH DOWN = 47.1% WR in paper. Only allow UP (70% WR).
                if asset == "ETH" and self.ETH_UP_ONLY and signal.side == "DOWN":
                    continue  # Silent skip
                # ETH max price (best in $0.15-$0.40 range, $0.40-$0.50 = 50% WR)
                if asset == "ETH" and signal.side == "UP" and up_price > self.ETH_MAX_PRICE:
                    print(f"  [{asset}] V3.9: ETH UP price ${up_price:.2f} > ${self.ETH_MAX_PRICE} cap")
                    self._record_filter_shadow(asset, signal.side, up_price, market_id, question, "v39_eth_price")
                    continue
                # V3.14: ETH = worst asset (29.4% WR in 326 CSV trades). Require higher confidence.
                if asset == "ETH":
                    eth_conf = signal.model_up if signal.side == "UP" else signal.model_down
                    if eth_conf < self.ETH_MIN_CONFIDENCE:
                        print(f"  [{asset}] V3.14: ETH needs {self.ETH_MIN_CONFIDENCE:.0%} conf, got {eth_conf:.0%} (29.4% WR overall)")
                        self._record_filter_shadow(asset, signal.side, up_price if signal.side == "UP" else down_price, market_id, question, "v314_eth_conf")
                        continue
                # BTC DOWN contrarian: both live BTC DOWN losses were against uptrend.
                # Require with-trend OR entry > $0.50 for BTC DOWN.
                if asset == "BTC" and signal.side == "DOWN":
                    btc_trend = self._get_trend_bias(candles, price)
                    if btc_trend in ("BULLISH", "RISING") and down_price < 0.50:
                        print(f"  [{asset}] V3.9: BTC DOWN contrarian (trend=BULLISH, price=${down_price:.2f}<$0.50)")
                        self._record_filter_shadow(asset, signal.side, down_price, market_id, question, "v39_btc_down_contrarian")
                        continue
                # V3.10: SOL DOWN = 50% WR in paper (coin flip). Require higher edge.
                if asset == "SOL" and signal.side == "DOWN":
                    sol_edge = signal.edge_down if signal.edge_down else 0
                    if sol_edge < self.SOL_DOWN_MIN_EDGE:
                        print(f"  [{asset}] V3.10: SOL DOWN edge {sol_edge:.1%} < {self.SOL_DOWN_MIN_EDGE:.0%} (paper: 50% WR, need >=35%)")
                        self._record_filter_shadow(asset, signal.side, down_price, market_id, question, "v310_sol_down_edge")
                        continue

                # === BOND MODE CHECK (before conviction filters) ===
                # V3.15c: Bond mode can force-enter when signal says NO_TRADE.
                # Market price $0.85+ IS the conviction. Model just can't strongly disagree.
                is_bond_trade = False
                bond_override_side = None
                if self.BOND_MODE_ENABLED:
                    if up_price >= self.BOND_MIN_PRICE and signal.model_up >= self.BOND_MIN_CONFIDENCE:
                        is_bond_trade = True
                        if signal.side != "UP":
                            bond_override_side = "UP"
                            print(f"  [{asset}] BOND OVERRIDE: Market UP@${up_price:.2f} (model_up={signal.model_up:.0%}) — overriding {signal.action}/{signal.side}")
                            signal.side = "UP"
                            signal.action = "ENTER"
                    elif down_price >= self.BOND_MIN_PRICE and signal.model_down >= self.BOND_MIN_CONFIDENCE:
                        is_bond_trade = True
                        if signal.side != "DOWN":
                            bond_override_side = "DOWN"
                            print(f"  [{asset}] BOND OVERRIDE: Market DOWN@${down_price:.2f} (model_down={signal.model_down:.0%}) — overriding {signal.action}/{signal.side}")
                            signal.side = "DOWN"
                            signal.action = "ENTER"

                    if is_bond_trade:
                        print(f"  [{asset}] BOND MODE: {signal.side} @ ${up_price if signal.side == 'UP' else down_price:.2f} | Model: {signal.model_up if signal.side == 'UP' else signal.model_down:.0%}")
                    else:
                        print(f"  [{asset}] Bond check: UP=${up_price:.2f} DOWN=${down_price:.2f} model_up={signal.model_up:.0%} model_down={signal.model_down:.0%} (need>=${self.BOND_MIN_PRICE}+>={self.BOND_MIN_CONFIDENCE:.0%})")

                # === CONVICTION FILTERS (V3.3 PORT) ===
                if not signal.side:
                    print(f"  [{asset}] No signal side (action={signal.action}) | mkt UP=${up_price:.2f} DOWN=${down_price:.2f} | {time_left:.1f}min | reason={signal.reason}")
                    continue
                if not is_bond_trade and signal.side == "UP" and signal.model_up < self.MIN_MODEL_CONFIDENCE:
                    print(f"  [{asset}] UP confidence too low: {signal.model_up:.1%} < {self.MIN_MODEL_CONFIDENCE:.0%}")
                    continue
                if not is_bond_trade and signal.side == "DOWN" and signal.model_down < self.MIN_MODEL_CONFIDENCE:
                    print(f"  [{asset}] DOWN confidence too low: {signal.model_down:.1%} < {self.MIN_MODEL_CONFIDENCE:.0%}")
                    continue

                if signal.action != "ENTER":
                    print(f"  [{asset}] Action is {signal.action}, not ENTER")
                    continue

                # === V3.3: SIDE-SPECIFIC FILTERS WITH TREND + DEATH ZONE + BREAK-EVEN AWARE ===
                skip_reason = None
                momentum = self._get_price_momentum(candles, lookback=5)

                if signal.side == "UP":
                    # V3.5b: Block ALL UP < $0.20 — paper showed -$20 on SOL UP@$0.21
                    if up_price < 0.20:
                        skip_reason = f"UP_ultra_cheap_{up_price:.2f}_(V3.5b_block)"
                    # TREND FILTER: Don't take ultra-cheap UP (<$0.15) during downtrends
                    elif up_price < 0.15 and hasattr(signal, 'regime') and signal.regime.value == 'trend_down':
                        skip_reason = f"UP_cheap_contrarian_{up_price:.2f}_in_downtrend"
                    else:
                        # Scaled confidence for cheap UP prices (V3.3)
                        up_conf_req = self.UP_MIN_CONFIDENCE
                        if up_price < 0.25:
                            up_conf_req = max(0.45, up_conf_req - 0.15)
                        elif up_price < 0.35:
                            up_conf_req = max(0.50, up_conf_req - 0.10)

                        # Dynamic UP max price: confidence unlocks higher prices (V3.3, softened V3.9)
                        # V3.18 CSV: 10/11 losses are UP. UP>$0.50 has 3 losses ($0.51/$0.53/$0.57).
                        # Tighten: cap at $0.50 even with high confidence. DOWN=99% WR, UP=82%.
                        effective_up_max = self.MAX_ENTRY_PRICE  # $0.52 base
                        if signal.model_up >= 0.85:
                            effective_up_max = max(effective_up_max, 0.52)  # V3.18: Was 0.58
                        elif signal.model_up >= 0.75:
                            effective_up_max = max(effective_up_max, 0.50)  # V3.18: Was 0.52
                        elif signal.model_up >= 0.65:
                            effective_up_max = max(effective_up_max, 0.46)  # V3.18: Was 0.48

                        if signal.model_up < up_conf_req:
                            skip_reason = f"UP_conf_{signal.model_up:.0%}<{up_conf_req:.0%}"
                        elif up_price > effective_up_max:
                            skip_reason = f"UP_price_{up_price:.2f}>{effective_up_max:.2f}"
                        # V3.15: UP momentum check REMOVED — was blocking winners in v33_conviction

                elif signal.side == "DOWN":
                    # DEATH ZONE: $0.40-0.45 = 14% WR, -$321 PnL. NEVER trade here. (V3.3)
                    if 0.40 <= down_price < 0.45:
                        skip_reason = f"DOWN_DEATH_ZONE_{down_price:.2f}_(14%WR)"
                    # V3.15: Loosened from $0.35 to $0.25 (was blocking winners)
                    elif down_price < 0.25:
                        skip_reason = f"DOWN_cheap_{down_price:.2f}_(V3.15_block)"
                    else:
                        # Break-even aware confidence for DOWN (V3.3)
                        down_conf_req = self.MIN_MODEL_CONFIDENCE
                        if down_price < 0.10:
                            down_conf_req = max(down_price * 2.5, 0.08)  # 10:1+ payoff
                        elif down_price < 0.20:
                            down_conf_req = max(down_price * 2.0, 0.15)  # 5:1+ payoff
                        elif down_price < 0.30:
                            down_conf_req = max(0.40, down_conf_req - 0.15)  # 3.3:1 payoff
                        elif down_price < 0.40:
                            down_conf_req = max(0.45, down_conf_req - 0.10)  # 2.5:1 payoff

                        if signal.model_down < down_conf_req:
                            skip_reason = f"DOWN_conf_{signal.model_down:.0%}<{down_conf_req:.0%}"
                        else:
                            # V3.13: Dynamic DOWN max price — confidence unlocks higher prices
                            # Same pattern as UP dynamic pricing. ML auto-tightens if expensive DOWNs lose.
                            effective_down_max = self.MAX_ENTRY_PRICE  # $0.45 base
                            if self.DOWN_EXPENSIVE_TRADES >= 5 and self.DOWN_EXPENSIVE_PNL < -3.0:
                                # ML AUTO-TIGHTEN: expensive DOWN trades are losing, revert to $0.45
                                effective_down_max = 0.45
                                print(f"  [{asset}] ML-TIGHTEN: DOWN>${0.45} has {self.DOWN_EXPENSIVE_WINS}W/{self.DOWN_EXPENSIVE_LOSSES}L, PnL=${self.DOWN_EXPENSIVE_PNL:+.2f} — capped at $0.45")
                            elif signal.model_down >= 0.85:
                                effective_down_max = max(effective_down_max, self.DOWN_MAX_PRICE)  # $0.65
                            elif signal.model_down >= 0.75:
                                effective_down_max = max(effective_down_max, 0.58)
                            elif signal.model_down >= 0.65:
                                effective_down_max = max(effective_down_max, 0.52)

                            if down_price > effective_down_max:
                                skip_reason = f"DOWN_price_{down_price:.2f}>{effective_down_max:.2f}"
                        # V3.15: DOWN momentum check REMOVED — was part of v33_conviction (+$47 missed)
                        # Momentum confirmation disabled to increase trade flow
                        pass

                if skip_reason and not is_bond_trade:
                    print(f"  [{asset}] V3.3 filter: {skip_reason}")
                    ep = up_price if signal.side == "UP" else down_price
                    self._record_filter_shadow(asset, signal.side, ep, market_id, question, "v33_conviction")
                    continue

                # === EDGE FILTER (with low-edge circuit breaker) ===
                # Bond mode skips edge filter (high-price entries have small edge by design)
                best_edge = signal.edge_up if signal.side == "UP" else signal.edge_down
                # If 3 low-edge trades failed in a row, lock out low-edge for 2 hours
                low_edge_locked = (self.low_edge_lockout_until and
                                   datetime.now(timezone.utc) < self.low_edge_lockout_until)
                if low_edge_locked:
                    edge_floor = self.LOW_EDGE_THRESHOLD  # 0.30
                else:
                    edge_floor = self.MIN_EDGE  # 0.30 (V3.5: raised from 0.25)
                if best_edge is not None and best_edge < edge_floor and not is_bond_trade:
                    lock_tag = " [LOCKOUT]" if low_edge_locked else ""
                    print(f"  [{asset}] Edge low (tag-only): {best_edge:.1%} < {edge_floor:.0%}{lock_tag}")
                    # V3.18: Converted to tag-only — shadow showed 7/7 blocked = winners (+$22.29)
                    ep = up_price if signal.side == "UP" else down_price
                    self._record_filter_shadow(asset, signal.side, ep, market_id, question, "edge_floor")
                    # continue  # REMOVED V3.18 — no longer blocks
                is_low_edge = best_edge is not None and best_edge < self.LOW_EDGE_THRESHOLD

                # === ATR VOLATILITY TAG (filter REMOVED - was blocking 100% winners) ===
                # Shadow data showed 7/7 blocked trades were winners (+$29.41 missed)
                # Now tags trades with atr_ratio for ongoing backtesting
                atr_high = False
                atr_ratio = 0.0
                atr_val = self._compute_atr(candles[:-3], 14)
                if atr_val > 0:
                    recent_ranges = [c.high - c.low for c in candles[-3:]]
                    avg_recent = sum(recent_ranges) / len(recent_ranges) if recent_ranges else 0
                    atr_ratio = round(avg_recent / atr_val, 2)
                    atr_high = atr_ratio > 1.5
                    if atr_high:
                        print(f"  [{asset}] ATR HIGH (ratio={atr_ratio:.1f}x) - proceeding (filter disabled, tagging for backtest)")

                # === NYU TWO-PARAMETER VOLATILITY FILTER (V3.3, adaptive V3.6) ===
                if self.use_nyu_model:
                    entry_price_nyu = up_price if signal.side == "UP" else down_price
                    nyu_result = self.nyu_model.calculate_volatility(entry_price_nyu, time_left)
                    # Adaptive threshold: high vol = more opportunity, relax gate
                    # V3.10: Shadow data showed 7/7 blocked = winners (+$21.99 missed). Loosened all tiers.
                    nyu_threshold = {"low": 0.05, "medium": 0.02, "high": 0.01}.get(nyu_result.volatility_regime, 0.05)
                    if nyu_result.edge_score < nyu_threshold:
                        print(f"  [{asset}] NYU filter: edge={nyu_result.edge_score:.2f}<{nyu_threshold} (vol={nyu_result.volatility_regime})")
                        self._record_filter_shadow(asset, signal.side, entry_price_nyu, market_id, question, "nyu_vol")
                        continue
                    if nyu_result.recommended_action == "TRADE":
                        print(f"  [{asset}] NYU TRADE: edge={nyu_result.edge_score:.2f}, vol={nyu_result.instantaneous_volatility:.3f}")

                if signal.action == "ENTER" and signal.side and trade_key:
                    if trade_key not in self.trades:
                        # SHADOW ASSET: log signal but never execute (V3.5)
                        if is_shadow_asset:
                            entry_price = up_price if signal.side == "UP" else down_price
                            edge = signal.edge_up if signal.side == "UP" else signal.edge_down
                            print(f"  [{asset}] SHADOW: {signal.side} @ ${entry_price:.3f} edge={edge:.1%} (paper-only, not executed)")
                            shadow_key = f"shadow_{market_id}_{signal.side}"
                            self.shadow_trades[shadow_key] = {
                                "asset": asset, "side": signal.side, "entry_price": entry_price,
                                "entry_time": datetime.now(timezone.utc).isoformat(),
                                "market_title": f"[{asset}] {question[:75]}",
                                "edge": edge, "status": "open", "reason": "shadow_asset",
                            }
                            continue
                        # DUPLICATE PROTECTION - skip if we already traded this market this cycle
                        if market_id in self.recently_traded_markets:
                            continue
                        # Don't place opposite-side trade on same market (self-hedging)
                        opposite_key = f"{market_id}_{'DOWN' if signal.side == 'UP' else 'UP'}"
                        if opposite_key in self.trades and self.trades[opposite_key].status == "open":
                            continue
                        # === V3.11: PER-ASSET CONSECUTIVE LOSS BLOCK ===
                        # Stop trading an asset after 3 consecutive losses (prevents SOL DOWN repeat bleed)
                        # Bug fix: Added time-based expiry to prevent permanent deadlock
                        asset_losses = self.asset_consecutive_losses.get(asset, 0)
                        if asset_losses >= self.ASSET_MAX_CONSECUTIVE_LOSSES:
                            # Check if cooldown has expired (30 min)
                            cooldown_start = self.asset_cooldown_start.get(asset)
                            expired = False
                            if cooldown_start:
                                try:
                                    start_dt = datetime.fromisoformat(cooldown_start)
                                    age_min = (datetime.now(timezone.utc) - start_dt).total_seconds() / 60
                                    if age_min >= self.ASSET_COOLDOWN_MINUTES:
                                        expired = True
                                        self.asset_consecutive_losses[asset] = 0
                                        del self.asset_cooldown_start[asset]
                                        print(f"  [{asset}] COOLDOWN EXPIRED after {age_min:.0f}min — allowing trades again")
                                except Exception:
                                    expired = True  # Can't parse = assume expired
                                    self.asset_consecutive_losses[asset] = 0
                            else:
                                # First time hitting cooldown — record start time
                                self.asset_cooldown_start[asset] = datetime.now(timezone.utc).isoformat()
                            if not expired:
                                print(f"  [{asset}] ASSET COOLDOWN: {asset_losses} consecutive losses — blocking new trades")
                                break  # Skip all remaining markets for this asset

                        # === MOMENTUM PAUSE (V3.5) ===
                        # After 2 consecutive losses, WR drops to 29.7%. Pause 30 min.
                        if self.momentum_pause_until and datetime.now(timezone.utc) < self.momentum_pause_until:
                            remaining = (self.momentum_pause_until - datetime.now(timezone.utc)).total_seconds() / 60
                            print(f"  [{asset}] MOMENTUM PAUSE: {remaining:.0f}min left (2 consecutive losses)")
                            break  # Break out of this asset's markets entirely

                        # === V3.13 ML: MULTI-TIMEFRAME BLOCK (paper: 0W/3L when disagree) ===
                        if not self._check_mtf_confirmation(candles, signal.side):
                            print(f"  [{asset}] MTF BLOCK: short+long TFs disagree with {signal.side} - SKIP")
                            continue

                        # === RISK CHECKS ===
                        if self.total_pnl <= -self.MAX_DAILY_LOSS:
                            print(f"[RISK] Daily loss limit hit: ${self.total_pnl:.2f} - stopping")
                            continue

                        open_exposure = sum(t.size_usd for t in self.trades.values() if t.status == "open")
                        if open_exposure >= self.MAX_TOTAL_EXPOSURE:
                            print(f"[RISK] Exposure limit: ${open_exposure:.2f} >= ${self.MAX_TOTAL_EXPOSURE}")
                            continue

                        open_count = sum(1 for t in self.trades.values() if t.status == "open" and not t.features.get("_is_hedge"))
                        if open_count >= self.MAX_CONCURRENT_POSITIONS:
                            continue

                        # V3.13: CLOB position check — catch ghost positions from dead processes
                        # The internal trade dict only knows about THIS instance's trades.
                        # Dead bot instances may have filled orders we don't know about.
                        try:
                            clob_positions = self.executor.get_positions()
                            if clob_positions:
                                active_positions = [p for p in clob_positions if float(p.get("size", 0)) > 0]
                                if active_positions:
                                    total_clob_value = sum(float(p.get("size", 0)) * float(p.get("avgPrice", 0)) for p in active_positions)
                                    if total_clob_value > self.HARD_MAX_BET * 1.5:
                                        print(f"  [GHOST CHECK] CLOB has ${total_clob_value:.2f} in positions (>{self.HARD_MAX_BET * 1.5:.2f}) — blocking new trades until resolved")
                                        break
                        except Exception:
                            pass  # Don't block trading on API errors

                        mid_price = up_price if signal.side == "UP" else down_price
                        # V3.15: Reduced spread offset from $0.05 to $0.03 (more competitive fills)
                        entry_price = round(mid_price + 0.03, 2)

                        # === V3.15: Raised MAX_ACTUAL_ENTRY from $0.52 to $0.58 ===
                        MAX_ACTUAL_ENTRY = 0.58  # V3.15: Was $0.52, widened for more fills
                        if entry_price > MAX_ACTUAL_ENTRY and not is_bond_trade:
                            print(f"  [{asset}] Entry too expensive after spread: ${entry_price:.2f} > ${MAX_ACTUAL_ENTRY} (mid=${mid_price:.2f}+$0.05)")
                            continue

                        # === V3.6: Hard floor — entries <$0.15 are 16.7% WR losers ===
                        if entry_price < self.MIN_ENTRY_PRICE and not is_bond_trade:
                            print(f"  [{asset}] Entry too cheap: ${entry_price:.3f} < ${self.MIN_ENTRY_PRICE}")
                            continue

                        edge = signal.edge_up if signal.side == "UP" else signal.edge_down

                        bregman_signal = self.bregman.calculate_optimal_trade(
                            model_prob=signal.model_up,
                            market_yes_price=up_price,
                            market_no_price=down_price
                        )

                        # === KL DIVERGENCE FILTER (V3.15: loosened from 0.20 to 0.10) ===
                        # Shadow data: blocked 9W/6L = +$7.99 in missed winners at 0.20 floor
                        # V3.15c: Bond trades skip KL (price near certainty = naturally low KL)
                        if bregman_signal.kl_divergence < self.ml_kl_floor and not is_bond_trade:  # V3.18: ML-tunable
                            print(f"  [{asset}] KL too low: {bregman_signal.kl_divergence:.3f} < {self.ml_kl_floor}")
                            self._record_filter_shadow(asset, signal.side, entry_price, market_id, question, "kl_divergence")
                            continue

                        features = self.ml.extract_features(signal, bregman_signal, candles)
                        features = self.ml.enrich_with_quant_models(features, self._last_garch, self._last_ceemd)

                        ml_score = self.ml.score_trade(features, signal.side)
                        min_score = self.ml.get_min_score_threshold()

                        if ml_score < min_score and not is_bond_trade:
                            self.ml_rejections += 1
                            print(f"[ML] {asset} Rejected: {signal.side} score={ml_score:.2f} < {min_score:.2f}")
                            self._record_filter_shadow(asset, signal.side, entry_price, market_id, question, "ml_score")
                            continue

                        # === V3.8 DATA-DRIVEN FILTERS (103 paper + 6 live trades) ===

                        # 1. MODEL CONFIDENCE HARD FLOOR
                        # The 0.128 confidence trade lost $6.54. Without it, session goes from -$3.94 to +$2.60.
                        # Paper: edge 20-30% = 26.7% WR. Edge 30-50% = 73-79% WR.
                        raw_conf = signal.model_up if signal.side == "UP" else signal.model_down
                        if raw_conf < 0.55 and not is_bond_trade:
                            print(f"  [{asset}] V3.8: Raw confidence too low: {raw_conf:.1%} < 55%")
                            self._record_filter_shadow(asset, signal.side, entry_price, market_id, question, "v38_confidence")
                            continue

                        # 2. VWAP ALIGNMENT + DISTANCE FILTER
                        # V3.14: CSV 29 bot trades: VWAP dist<=0.30 = 65% WR, >0.30 = 50% WR
                        # Hard cap on absolute VWAP distance + directional alignment check
                        vwap_dist = features.vwap_distance
                        if not is_bond_trade:  # V3.15c: Bond trades skip VWAP filters (price-is-confidence)
                            if abs(vwap_dist) > self.ml_vwap_abs_cap:  # V3.18: ML-tunable (was 0.50)
                                print(f"  [{asset}] V3.18: VWAP too far: {vwap_dist:+.3f} (max {self.ml_vwap_abs_cap:.2f})")
                                self._record_filter_shadow(asset, signal.side, entry_price, market_id, question, "v314_vwap_dist")
                                continue
                            if signal.side == "DOWN" and vwap_dist > self.ml_vwap_dir_cap:  # V3.18: ML-tunable
                                print(f"  [{asset}] V3.18: VWAP misalign DOWN (vwap_dist={vwap_dist:+.3f} > +{self.ml_vwap_dir_cap:.2f})")
                                self._record_filter_shadow(asset, signal.side, entry_price, market_id, question, "v38_vwap")
                                continue
                            if signal.side == "UP" and vwap_dist < -self.ml_vwap_dir_cap:  # V3.18: ML-tunable
                                print(f"  [{asset}] V3.18: VWAP misalign UP (vwap_dist={vwap_dist:+.3f} < -{self.ml_vwap_dir_cap:.2f})")
                                self._record_filter_shadow(asset, signal.side, entry_price, market_id, question, "v38_vwap")
                                continue

                        # 3. HEIKEN ASHI CONTRADICTION VETO
                        # Trade 6 bet UP with 3 bearish Heiken candles → lost.
                        # If Heiken strongly contradicts signal direction (3+ candles), skip.
                        ha_bullish = features.heiken_bullish
                        ha_count = features.heiken_count
                        if not is_bond_trade:  # V3.15c: Bond trades skip Heiken check
                            if signal.side == "UP" and not ha_bullish and ha_count >= 3:
                                print(f"  [{asset}] V3.8: Heiken contradiction (UP vs {ha_count} bearish candles)")
                                self._record_filter_shadow(asset, signal.side, entry_price, market_id, question, "v38_heiken")
                                continue
                            if signal.side == "DOWN" and ha_bullish and ha_count >= 3:
                                print(f"  [{asset}] V3.8: Heiken contradiction (DOWN vs {ha_count} bullish candles)")
                                self._record_filter_shadow(asset, signal.side, entry_price, market_id, question, "v38_heiken")
                                continue

                        # 4. GARCH VOLATILITY GATE (V3.17: replaces ATR ratio)
                        # GARCH regime-aware: HIGH vol + low confidence = skip
                        garch = self._last_garch
                        if garch["regime"] == "HIGH" and raw_conf < 0.65 and not is_bond_trade:
                            print(f"  [{asset}] V3.17: GARCH HIGH vol (ratio={garch['ratio']:.2f}) needs conf>65%, got {raw_conf:.1%}")
                            self._record_filter_shadow(asset, signal.side, entry_price, market_id, question, "v317_garch_vol")
                            continue

                        # 5. CEEMD NOISE FILTER (V3.17)
                        # Block trades when market is too noisy (high-freq IMFs dominate)
                        ceemd = self._last_ceemd
                        ceemd_aligned = False
                        if ceemd["n_imfs"] > 0:
                            if signal.side == "UP" and ceemd["trend_slope"] > 0.001:
                                ceemd_aligned = True
                            elif signal.side == "DOWN" and ceemd["trend_slope"] < -0.001:
                                ceemd_aligned = True

                            if ceemd["noise_ratio"] > 0.85 and not is_bond_trade:
                                print(f"  [{asset}] V3.17: CEEMD noisy market (noise={ceemd['noise_ratio']:.2f}, {ceemd['n_imfs']} IMFs)")
                                self.shadow_stats["ceemd_noisy_blocked"] += 1
                                self._record_filter_shadow(asset, signal.side, entry_price, market_id, question, "v317_ceemd_noise")
                                continue

                        # === V3.15b: HYDRA PROBATION BOOST ===
                        # Check how many promoted strategies agree with this direction
                        asset_hydra = self.hydra_last_signals.get(asset, {})
                        hydra_agreeing = [s for s, d in asset_hydra.items() if d == signal.side]
                        hydra_agrees = len(hydra_agreeing) > 0
                        hydra_tag = ""
                        if hydra_agreeing:
                            hydra_tag = f" [HYDRA+{'+'.join(hydra_agreeing)}]"
                            print(f"  [{asset}] HYDRA BOOST: {', '.join(hydra_agreeing)} agree with {signal.side}")

                        self._current_trade_side = signal.side  # V3.18: For side multiplier in sizing
                        position_size = self.calculate_position_size(
                            edge=edge if edge else 0.10,
                            kelly_fraction=bregman_signal.kelly_fraction,
                            btc_1h_change=features.btc_1h_change,
                            volatility=features.btc_volatility,
                        )

                        # === V3.16: PRE-MARKET OPEN BOOST (9:00-9:15 EST = UTC 14) ===
                        # Research: BTC LONG at US pre-market open shows +700%/mo compounding.
                        # Our data: UTC 14 = 100% WR live, 71.4% WR paper. Boost BTC UP here.
                        utc_hour = datetime.now(timezone.utc).hour
                        if utc_hour == 14 and asset == "BTC" and signal.side == "UP":
                            premarket_boost = 1.30
                            position_size = round(position_size * premarket_boost, 2)
                            print(f"  [{asset}] PRE-MARKET BOOST: BTC UP at US open -> 1.3x size (${position_size:.2f})")

                        # === TREND BIAS (soft, with counterfactual tracking) ===
                        # 200 EMA trend: counter-trend trades get 85% size (softer than old 30%)
                        # Track what full-size would have earned for ML evaluation
                        trend = self._get_trend_bias(candles, price)
                        # V3.13: FALLING/RISING = active momentum override, HARD BLOCK counter-trend
                        # Old bug: 200 EMA said BULLISH while BTC dumped -1% → tried to go UP
                        if trend == "FALLING" and signal.side == "UP":
                            print(f"  [{asset}] FALLING KNIFE BLOCK: price dropping >0.3% in 30min, refusing UP")
                            continue
                        if trend == "RISING" and signal.side == "DOWN":
                            print(f"  [{asset}] RISING BLOCK: price rising >0.3% in 30min, refusing DOWN")
                            continue

                        with_trend = (trend in ("BEARISH", "FALLING") and signal.side == "DOWN") or \
                                     (trend in ("BULLISH", "RISING") and signal.side == "UP")
                        full_size = position_size  # What we'd bet without trend bias
                        if not with_trend:
                            position_size = round(position_size * 0.85, 2)  # Softer: 85% (was 43%)
                            position_size = max(self.MIN_POSITION_SIZE, position_size)
                            trend_tag = f"COUNTER({trend}) 85%"
                        else:
                            trend_tag = f"WITH({trend})"

                        # === SWEET SPOT BOOST DISABLED — capital rebuild mode ===
                        # Was: 1.3x for $0.15-$0.35 entries. Disabled to enforce $5-$8 hard cap.

                        hour_mult = self._get_hour_multiplier(datetime.now(timezone.utc).hour)
                        hour_tag = f", Hour={hour_mult}x" if hour_mult != 1.0 else ""

                        # Bond mode: force minimum size (high entry price = small profit per share)
                        if is_bond_trade:
                            position_size = self.MIN_POSITION_SIZE
                            hour_tag += ", BOND=$" + str(self.MIN_POSITION_SIZE)

                        # Low-edge trades: force minimum size ($5)
                        if is_low_edge:
                            position_size = self.MIN_POSITION_SIZE
                            hour_tag += ", LOW-EDGE=$5"

                        # === HARD_MAX_BET ENFORCEMENT — ABSOLUTE CEILING ===
                        # Nothing can exceed $8. Not DCA, not boost, not anything.
                        position_size = min(position_size, self.HARD_MAX_BET)
                        position_size = max(self.MIN_POSITION_SIZE, position_size)

                        # === DCA SPLIT: Execute 60% now, queue 40% for next scan ===
                        if self.DCA_ENABLED and not is_bond_trade and position_size > self.MIN_POSITION_SIZE:
                            initial_size = round(position_size * self.DCA_INITIAL_RATIO, 2)
                            topup_size = round(position_size * self.DCA_TOPUP_RATIO, 2)
                            initial_size = max(self.MIN_POSITION_SIZE, initial_size)
                            # Even with DCA, total cannot exceed HARD_MAX_BET
                            if initial_size + topup_size > self.HARD_MAX_BET:
                                topup_size = max(0, round(self.HARD_MAX_BET - initial_size, 2))
                        else:
                            initial_size = position_size
                            topup_size = 0

                        # FINAL SAFETY: initial_size alone must not exceed HARD_MAX_BET
                        initial_size = min(initial_size, self.HARD_MAX_BET)

                        # === V3.11: BANKROLL GUARD — don't trade if we can't afford it ===
                        if self.clob_balance is not None and self.clob_balance < initial_size:
                            print(f"[GUARD] Insufficient CLOB balance: ${self.clob_balance:.2f} < ${initial_size:.2f} — skipping")
                            continue
                        if self.bankroll < initial_size:
                            print(f"[GUARD] Insufficient bankroll: ${self.bankroll:.2f} < ${initial_size:.2f} — skipping")
                            continue

                        print(f"[SIZE] {asset} ${initial_size:.2f}{f' (+${topup_size:.2f} DCA pending)' if topup_size > 0 else ''} (Kelly={bregman_signal.kelly_fraction:.0%}, Edge={edge:.1%}, {trend_tag}{hour_tag}{hydra_tag})")

                        success, order_result = await self.execute_trade(
                            market, signal.side, initial_size, entry_price
                        )

                        if not success:
                            print(f"[LIVE] {asset} {signal.side} @ ${entry_price:.4f} FAILED: {order_result} - NOT recorded")
                            continue

                        # Build feature dict with IH2P tags + hydra agreement
                        trade_features = {**asdict(features), "_full_size": full_size, "_with_trend": with_trend, "_atr_ratio": atr_ratio, "_atr_high": atr_high, "_hydra_agrees": hydra_agrees, "_hydra_strategies": hydra_agreeing}
                        if is_bond_trade:
                            trade_features["_bond_mode"] = True
                            if bond_override_side:
                                trade_features["_bond_override"] = bond_override_side
                        if topup_size > 0:
                            trade_features["_dca_initial"] = initial_size
                            trade_features["_dca_pending"] = topup_size
                        # V3.17: Tag with GARCH regime + CEEMD alignment
                        trade_features["_garch_regime"] = garch["regime"]
                        trade_features["_garch_ratio"] = garch["ratio"]
                        trade_features["_ceemd_aligned"] = ceemd_aligned
                        trade_features["_ceemd_trend_slope"] = ceemd["trend_slope"]
                        trade_features["_ceemd_noise_ratio"] = ceemd["noise_ratio"]

                        new_trade = LiveTrade(
                            trade_id=trade_key,
                            market_id=market_id,
                            market_title=f"[{asset}] {question[:75]}",
                            side=signal.side,
                            entry_price=entry_price,
                            entry_time=datetime.now(timezone.utc).isoformat(),
                            size_usd=initial_size,
                            signal_strength=signal.strength.value,
                            edge_at_entry=edge if edge else 0,
                            kl_divergence=bregman_signal.kl_divergence,
                            kelly_fraction=bregman_signal.kelly_fraction,
                            features=trade_features,
                            order_id=order_result,
                            execution_error=None,
                            status="open",
                        )
                        self.trades[trade_key] = new_trade
                        self.recently_traded_markets.add(market_id)  # Prevent duplicate orders

                        # V3.15c FIX: Query actual fill details to correct entry_price and size_usd
                        # The requested price/size may differ from what actually filled on-chain
                        try:
                            order_details = self.executor.client.get_order(order_result)
                            if isinstance(order_details, dict):
                                size_matched = float(order_details.get("size_matched", 0))
                                price_avg = float(order_details.get("associate_trades", [{}])[0].get("price", 0)) if order_details.get("associate_trades") else 0
                                # size_matched = shares filled, actual cost = size_matched * avg_price
                                if size_matched > 0 and price_avg > 0:
                                    actual_cost = round(size_matched * price_avg, 4)
                                    if abs(actual_cost - new_trade.size_usd) > 0.01 or abs(price_avg - new_trade.entry_price) > 0.005:
                                        print(f"[FILL FIX] {asset}: requested ${new_trade.size_usd:.2f}@${new_trade.entry_price:.3f} -> actual ${actual_cost:.2f}@${price_avg:.3f} ({size_matched:.0f} shares)")
                                        new_trade.entry_price = round(price_avg, 4)
                                        new_trade.size_usd = actual_cost
                                elif size_matched > 0:
                                    # No price in associate_trades, use matched size * requested price as approximation
                                    actual_cost = round(size_matched * entry_price, 4)
                                    if abs(actual_cost - new_trade.size_usd) > 0.10:
                                        print(f"[FILL FIX] {asset}: partial fill {size_matched:.0f}/{shares} shares -> size ${actual_cost:.2f} (was ${new_trade.size_usd:.2f})")
                                        new_trade.size_usd = actual_cost
                        except Exception as e:
                            print(f"[FILL FIX] Could not query fill details: {e}")

                        # === DCA: Queue top-up for next scan ===
                        if topup_size > 0:
                            self.pending_topups[trade_key] = {
                                "market": market, "market_id": market_id, "asset": asset,
                                "side": signal.side, "size": topup_size,
                                "max_price": entry_price,
                                "created": datetime.now(timezone.utc).isoformat(),
                            }
                            print(f"[DCA] Queued ${topup_size:.2f} topup for {asset} {signal.side} (max ${entry_price:.3f})")

                        # Track trend bias counterfactual
                        if not with_trend:
                            self.shadow_stats["trend_bias_trades"] += 1

                        # === HEDGE: Place small opposite-side bet if cheap ===
                        if self.HEDGE_ENABLED and not is_bond_trade:
                            opp_price = down_price if signal.side == "UP" else up_price
                            if opp_price is not None and 0.01 < opp_price <= self.HEDGE_MAX_PRICE:
                                opp_side = "DOWN" if signal.side == "UP" else "UP"
                                hedge_success, hedge_order = await self.execute_trade(
                                    market, opp_side, self.HEDGE_SIZE, opp_price
                                )
                                if hedge_success:
                                    hedge_key = f"{market_id}_{opp_side}_hedge"
                                    hedge_trade = LiveTrade(
                                        trade_id=hedge_key,
                                        market_id=market_id,
                                        market_title=f"[{asset}] HEDGE {opp_side} {question[:60]}",
                                        side=opp_side,
                                        entry_price=opp_price,
                                        entry_time=datetime.now(timezone.utc).isoformat(),
                                        size_usd=self.HEDGE_SIZE,
                                        signal_strength="HEDGE",
                                        edge_at_entry=0,
                                        kl_divergence=0,
                                        kelly_fraction=0,
                                        features={"_is_hedge": True, "_primary_trade": trade_key},
                                        order_id=hedge_order,
                                        execution_error=None,
                                        status="open",
                                    )
                                    self.trades[hedge_key] = hedge_trade
                                    self.shadow_stats["hedge_trades"] += 1
                                    print(f"[HEDGE] {opp_side} @ ${opp_price:.3f} | ${self.HEDGE_SIZE:.2f} | Opposite-side insurance")
                                else:
                                    print(f"[HEDGE] Failed: {hedge_order}")

                        print(f"[LIVE] [{asset}] {signal.side} @ ${entry_price:.4f} | Size: ${initial_size:.2f} | Edge: {edge:.1%} | Model: {signal.model_up:.0%} UP | ML: {ml_score:.2f} | FILLED{' [BOND]' if is_bond_trade else ''}")
                    break  # One trade per asset per cycle

            # Check ALL markets for resolution (not just eligible ones)
            for mkt in markets:
                mkt_time = self.get_time_remaining(mkt)
                if mkt_time < 0.5:
                    mkt_id = mkt.get("conditionId", "")
                    mkt_up, mkt_down = self.get_market_prices(mkt)
                    if mkt_up is None or mkt_down is None:
                        continue
                    for tid, open_trade in list(self.trades.items()):
                        if open_trade.status == "open" and mkt_id in tid:
                            res_price = mkt_up if open_trade.side == "UP" else mkt_down

                            # Determine outcome using actual position size
                            if res_price >= 0.95:
                                exit_val = open_trade.size_usd / open_trade.entry_price
                            elif res_price <= 0.05:
                                exit_val = 0
                            else:
                                exit_val = (open_trade.size_usd / open_trade.entry_price) * res_price

                            open_trade.exit_price = res_price
                            open_trade.exit_time = datetime.now(timezone.utc).isoformat()
                            open_trade.pnl = exit_val - open_trade.size_usd
                            open_trade.status = "closed"

                            self.total_pnl += open_trade.pnl
                            won = open_trade.pnl > 0
                            is_hedge = open_trade.features.get("_is_hedge", False)
                            was_low_edge = open_trade.edge_at_entry < self.LOW_EDGE_THRESHOLD

                            # Hedges don't affect win/loss streaks or momentum pause
                            if not is_hedge:
                                if won:
                                    self.wins += 1
                                    self.consecutive_wins += 1
                                    self.consecutive_losses = 0
                                    self.momentum_pause_until = None  # Win clears pause
                                    if was_low_edge:
                                        self.low_edge_consecutive_losses = 0
                                else:
                                    self.losses += 1
                                    self.consecutive_losses += 1
                                    self.consecutive_wins = 0
                                    # MOMENTUM PAUSE: 2 consecutive losses → pause 30 min
                                    if self.consecutive_losses >= 2:
                                        self.momentum_pause_until = datetime.now(timezone.utc) + timedelta(minutes=30)
                                        print(f"[MOMENTUM PAUSE] {self.consecutive_losses} consecutive losses — pausing until {self.momentum_pause_until.strftime('%H:%M UTC')}")
                                    if was_low_edge:
                                        self.low_edge_consecutive_losses += 1
                                        if self.low_edge_consecutive_losses >= 3:
                                            self.low_edge_lockout_until = datetime.now(timezone.utc) + timedelta(hours=2)
                                            print(f"[CIRCUIT BREAKER] 3 low-edge losses — edge floor back to 30% until {self.low_edge_lockout_until.strftime('%H:%M UTC')}")
                                            self.low_edge_consecutive_losses = 0

                            # === V3.11: PER-ASSET CONSECUTIVE LOSS TRACKER ===
                            # Extract asset from market_title (format: "[BTC] ..." or "[SOL] ...")
                            trade_asset = None
                            if open_trade.market_title.startswith("["):
                                trade_asset = open_trade.market_title[1:open_trade.market_title.index("]")]
                            if trade_asset and not is_hedge:
                                if won:
                                    self.asset_consecutive_losses[trade_asset] = 0
                                    # Clear cooldown timer on win
                                    if trade_asset in self.asset_cooldown_start:
                                        del self.asset_cooldown_start[trade_asset]
                                else:
                                    self.asset_consecutive_losses[trade_asset] = self.asset_consecutive_losses.get(trade_asset, 0) + 1
                                    acl = self.asset_consecutive_losses[trade_asset]
                                    if acl >= self.ASSET_MAX_CONSECUTIVE_LOSSES:
                                        # Record cooldown start time for time-based expiry
                                        if trade_asset not in self.asset_cooldown_start:
                                            self.asset_cooldown_start[trade_asset] = datetime.now(timezone.utc).isoformat()
                                        print(f"[ASSET COOLDOWN] {trade_asset}: {acl} consecutive losses — blocking new trades (30min expiry)")

                            # === UPDATE HOURLY STATS ===
                            try:
                                entry_hour = datetime.fromisoformat(open_trade.entry_time).hour
                                if entry_hour not in self.hourly_stats:
                                    self.hourly_stats[entry_hour] = {"wins": 0, "losses": 0, "pnl": 0.0}
                                self.hourly_stats[entry_hour]["wins" if won else "losses"] += 1
                                self.hourly_stats[entry_hour]["pnl"] += open_trade.pnl
                            except Exception:
                                pass

                            # V3.12c: Sync bankroll from CLOB after resolution (fixes double-counting)
                            # Old: bankroll += pnl, but CLOB sync ALSO adjusts -> double-counted
                            clob_after = self._sync_clob_balance()
                            if clob_after >= 0:
                                self.bankroll = clob_after
                            else:
                                # Fallback if CLOB query fails
                                self.bankroll = max(self.initial_bankroll * 0.5, self.bankroll + open_trade.pnl)

                            # Update ML
                            self.ml.update_weights(open_trade.features, won)

                            hour_mult = self._get_hour_multiplier(datetime.now(timezone.utc).hour)
                            result = "WIN" if won else "LOSS"
                            print(f"[{result}] {open_trade.side} PnL: ${open_trade.pnl:+.2f} | Size: ${open_trade.size_usd:.2f} | Bankroll: ${self.bankroll:.2f} | HourMult: {hour_mult}x | {open_trade.market_title[:40]}...")
                            # V3.13: ML auto-tighten tracking for expensive DOWN trades
                            if open_trade.side == "DOWN" and open_trade.entry_price > 0.45:
                                self.DOWN_EXPENSIVE_TRADES += 1
                                self.DOWN_EXPENSIVE_PNL += open_trade.pnl
                                if won:
                                    self.DOWN_EXPENSIVE_WINS += 1
                                else:
                                    self.DOWN_EXPENSIVE_LOSSES += 1
                                print(f"  [ML-TRACK] DOWN>${0.45}: {self.DOWN_EXPENSIVE_WINS}W/{self.DOWN_EXPENSIVE_LOSSES}L PnL=${self.DOWN_EXPENSIVE_PNL:+.2f}")

                            # V3.15b: Auto-revert filter check + per-strategy hydra tracking
                            self._check_filter_auto_revert(won)
                            self._track_hydra_result(open_trade, won)

                            # V3.15b: Post-loss forensics (@sharbel's "why did this lose?" loop)
                            if not won and self.forensics:
                                try:
                                    self.forensics.analyze_loss(asdict(open_trade), {k: asdict(t) for k, t in self.trades.items() if t.status == "closed"})
                                except Exception as e:
                                    print(f"  [FORENSICS] Error: {e}")

                            # Auto-redeem winnings after each win
                            if won and not self.dry_run:
                                try:
                                    auto_redeem_winnings()
                                except Exception:
                                    pass

                            # Track trend bias counterfactual PnL
                            full_size = open_trade.features.get("_full_size", open_trade.size_usd)
                            if full_size != open_trade.size_usd:
                                if res_price >= 0.95:
                                    full_exit = full_size / open_trade.entry_price
                                elif res_price <= 0.05:
                                    full_exit = 0
                                else:
                                    full_exit = (full_size / open_trade.entry_price) * res_price
                                full_pnl = full_exit - full_size
                                self.shadow_stats["trend_bias_actual_pnl"] += open_trade.pnl
                                self.shadow_stats["trend_bias_full_pnl"] += full_pnl

                            # === IH2P FEATURE PNL TRACKING ===
                            ss = self.shadow_stats
                            if open_trade.features.get("_bond_mode"):
                                ss["bond_trades"] += 1
                                ss["bond_pnl"] += open_trade.pnl
                                if won:
                                    ss["bond_wins"] += 1
                                else:
                                    ss["bond_losses"] += 1
                                print(f"[BOND] {'WIN' if won else 'LOSS'} ${open_trade.pnl:+.2f} | Total: {ss['bond_trades']} trades, ${ss['bond_pnl']:+.2f}")
                            # V3.17: GARCH regime tracking
                            g_regime = open_trade.features.get("_garch_regime", "UNKNOWN")
                            if g_regime in ("LOW", "NORMAL", "HIGH"):
                                ss[f"garch_trades_{g_regime.lower()}"] += 1
                                if won:
                                    ss[f"garch_wins_{g_regime.lower()}"] += 1
                            # V3.17: CEEMD alignment tracking
                            if open_trade.features.get("_ceemd_aligned"):
                                ss["ceemd_aligned_trades"] += 1
                                if won:
                                    ss["ceemd_aligned_wins"] += 1
                            elif open_trade.features.get("_ceemd_aligned") is False:
                                ss["ceemd_opposed_trades"] += 1
                                if won:
                                    ss["ceemd_opposed_wins"] += 1

                            if open_trade.features.get("_is_hedge"):
                                ss["hedge_pnl"] += open_trade.pnl
                                if won:
                                    ss["hedge_wins"] += 1
                                else:
                                    ss["hedge_losses"] += 1
                                print(f"[HEDGE] {'WIN' if won else 'LOSS'} ${open_trade.pnl:+.2f} | Total: {ss['hedge_trades']} trades, ${ss['hedge_pnl']:+.2f}")
                            if open_trade.features.get("_dca_topup"):
                                # Attribute PnL of the DCA portion only
                                topup_size = open_trade.features["_dca_topup"]
                                topup_price = open_trade.features.get("_dca_topup_price", open_trade.entry_price)
                                if res_price >= 0.95:
                                    topup_exit = topup_size / topup_price
                                elif res_price <= 0.05:
                                    topup_exit = 0
                                else:
                                    topup_exit = (topup_size / topup_price) * res_price
                                dca_pnl = topup_exit - topup_size
                                ss["dca_topup_pnl"] += dca_pnl
                                print(f"[DCA] Topup PnL: ${dca_pnl:+.2f} | Total DCA PnL: ${ss['dca_topup_pnl']:+.2f}")

                            # ML auto-revoke check
                            self._check_feature_revoke()

                    # === RESOLVE SHADOW TRADES on this market ===
                    for skey, shadow in list(self.shadow_trades.items()):
                        if shadow.get("status") == "open" and shadow.get("market_id") == mkt_id:
                            s_price = mkt_up if shadow["side"] == "UP" else mkt_down
                            if s_price >= 0.95:
                                s_exit = shadow["size_usd"] / shadow["entry_price"]
                            elif s_price <= 0.05:
                                s_exit = 0
                            else:
                                s_exit = (shadow["size_usd"] / shadow["entry_price"]) * s_price
                            shadow["exit_price"] = s_price
                            shadow["pnl"] = s_exit - shadow["size_usd"]
                            shadow["status"] = "closed"
                            shadow["exit_time"] = datetime.now(timezone.utc).isoformat()
                            won = shadow["pnl"] > 0
                            reason = shadow.get("filter_reason", shadow.get("reason", "unknown"))
                            if reason == "atr_volatility":
                                if won:
                                    self.shadow_stats["atr_blocked_wins"] += 1
                                else:
                                    self.shadow_stats["atr_blocked_losses"] += 1
                                self.shadow_stats["atr_blocked_pnl"] += shadow["pnl"]
                            # Track filter rejection stats (V3.9)
                            fname = shadow.get("filter_name")
                            if fname and fname in self.filter_stats:
                                if won:
                                    self.filter_stats[fname]["wins"] += 1
                                else:
                                    self.filter_stats[fname]["losses"] += 1
                                self.filter_stats[fname]["pnl"] += shadow["pnl"]
                            result = "WIN" if won else "LOSS"
                            print(f"[SHADOW {result}] {shadow['side']} ${shadow['pnl']:+.2f} ({reason}) | {shadow['market_title'][:35]}...")

        # Check for expired trades (markets disappear from active query after close)
        now = datetime.now(timezone.utc)
        for tid, open_trade in list(self.trades.items()):
            if open_trade.status != "open":
                continue
            entry_dt = datetime.fromisoformat(open_trade.entry_time)
            age_minutes = (now - entry_dt).total_seconds() / 60
            if age_minutes > 16:  # 15-min market + 1 min buffer
                # Fetch resolved market data
                try:
                    import httpx
                    r = httpx.get(
                        "https://gamma-api.polymarket.com/markets",
                        params={"condition_id": open_trade.market_id, "limit": "1"},
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=10
                    )
                    if r.status_code == 200 and r.json():
                        mkt = r.json()[0]
                        outcomes = mkt.get("outcomes", [])
                        prices = mkt.get("outcomePrices", [])
                        if isinstance(outcomes, str):
                            outcomes = json.loads(outcomes)
                        if isinstance(prices, str):
                            prices = json.loads(prices)

                        # Find resolved price for our side
                        res_price = None
                        for i, outcome in enumerate(outcomes):
                            if str(outcome).lower() == open_trade.side.lower() and i < len(prices):
                                res_price = float(prices[i])
                                break

                        if res_price is not None:
                            if res_price >= 0.95:
                                exit_val = open_trade.size_usd / open_trade.entry_price
                            elif res_price <= 0.05:
                                exit_val = 0
                            else:
                                exit_val = (open_trade.size_usd / open_trade.entry_price) * res_price

                            open_trade.exit_price = res_price
                            open_trade.exit_time = now.isoformat()
                            open_trade.pnl = exit_val - open_trade.size_usd
                            open_trade.status = "closed"

                            self.total_pnl += open_trade.pnl
                            won = open_trade.pnl > 0
                            was_low_edge = open_trade.edge_at_entry < self.LOW_EDGE_THRESHOLD
                            if won:
                                self.wins += 1
                                self.consecutive_wins += 1
                                self.consecutive_losses = 0
                                if was_low_edge:
                                    self.low_edge_consecutive_losses = 0
                            else:
                                self.losses += 1
                                self.consecutive_losses += 1
                                self.consecutive_wins = 0
                                if self.consecutive_losses >= 2:
                                    self.momentum_pause_until = datetime.now(timezone.utc) + timedelta(minutes=30)
                                    print(f"[MOMENTUM PAUSE] {self.consecutive_losses} consecutive losses — pausing until {self.momentum_pause_until.strftime('%H:%M UTC')}")
                                if was_low_edge:
                                    self.low_edge_consecutive_losses += 1
                                    if self.low_edge_consecutive_losses >= 3:
                                        self.low_edge_lockout_until = datetime.now(timezone.utc) + timedelta(hours=2)
                                        print(f"[CIRCUIT BREAKER] 3 low-edge losses — edge floor back to 30% until {self.low_edge_lockout_until.strftime('%H:%M UTC')}")
                                        self.low_edge_consecutive_losses = 0

                            # V3.12c: Sync bankroll from CLOB (fixes double-counting)
                            clob_after2 = self._sync_clob_balance()
                            if clob_after2 >= 0:
                                self.bankroll = clob_after2
                            else:
                                self.bankroll = max(self.initial_bankroll * 0.5, self.bankroll + open_trade.pnl)
                            self.ml.update_weights(open_trade.features, won)

                            # V3.11: Per-asset consecutive loss tracking
                            if open_trade.market_title.startswith("["):
                                ta2 = open_trade.market_title[1:open_trade.market_title.index("]")]
                                if not is_hedge:
                                    if won:
                                        self.asset_consecutive_losses[ta2] = 0
                                        if ta2 in self.asset_cooldown_start:
                                            del self.asset_cooldown_start[ta2]
                                    else:
                                        self.asset_consecutive_losses[ta2] = self.asset_consecutive_losses.get(ta2, 0) + 1
                                        acl2 = self.asset_consecutive_losses[ta2]
                                        if acl2 >= self.ASSET_MAX_CONSECUTIVE_LOSSES:
                                            if ta2 not in self.asset_cooldown_start:
                                                self.asset_cooldown_start[ta2] = datetime.now(timezone.utc).isoformat()
                                            print(f"[ASSET COOLDOWN] {ta2}: {acl2} consecutive losses — blocking new trades (30min expiry)")

                            result = "WIN" if won else "LOSS"
                            print(f"[{result}] {open_trade.side} PnL: ${open_trade.pnl:+.2f} | Size: ${open_trade.size_usd:.2f} | Bankroll: ${self.bankroll:.2f} | {open_trade.market_title[:40]}...")
                            # V3.13: ML auto-tighten tracking for expensive DOWN trades
                            if open_trade.side == "DOWN" and open_trade.entry_price > 0.45:
                                self.DOWN_EXPENSIVE_TRADES += 1
                                self.DOWN_EXPENSIVE_PNL += open_trade.pnl
                                if won:
                                    self.DOWN_EXPENSIVE_WINS += 1
                                else:
                                    self.DOWN_EXPENSIVE_LOSSES += 1
                                print(f"  [ML-TRACK] DOWN>${0.45}: {self.DOWN_EXPENSIVE_WINS}W/{self.DOWN_EXPENSIVE_LOSSES}L PnL=${self.DOWN_EXPENSIVE_PNL:+.2f}")

                            # V3.15b: Auto-revert filter check + per-strategy hydra tracking
                            self._check_filter_auto_revert(won)
                            self._track_hydra_result(open_trade, won)

                            # V3.15b: Post-loss forensics
                            if not won and self.forensics:
                                try:
                                    self.forensics.analyze_loss(asdict(open_trade), {k: asdict(t) for k, t in self.trades.items() if t.status == "closed"})
                                except Exception as e:
                                    print(f"  [FORENSICS] Error: {e}")

                            # Auto-redeem winnings after each win
                            if won and not self.dry_run:
                                try:
                                    auto_redeem_winnings()
                                except Exception:
                                    pass
                        else:
                            # Can't determine resolution, close at entry (break even)
                            open_trade.status = "closed"
                            open_trade.pnl = 0.0
                            open_trade.exit_price = open_trade.entry_price
                            open_trade.exit_time = now.isoformat()
                            print(f"[EXPIRED] {open_trade.side} | No resolved price | {open_trade.market_title[:40]}...")
                    else:
                        # Market not found, close at break even
                        open_trade.status = "closed"
                        open_trade.pnl = 0.0
                        open_trade.exit_price = open_trade.entry_price
                        open_trade.exit_time = now.isoformat()
                        print(f"[EXPIRED] {open_trade.side} | Market not found | {open_trade.market_title[:40]}...")
                except Exception as e:
                    print(f"[EXPIRE CHECK] Error for {tid[:20]}: {e}")

        # === RESOLVE EXPIRED SHADOW TRADES via gamma-api ===
        # Shadow trades that are > 16 min old: fetch actual market outcome instead of defaulting to pnl=0
        for skey, shadow in list(self.shadow_trades.items()):
            if shadow.get("status") != "open":
                continue
            try:
                s_entry = datetime.fromisoformat(shadow["entry_time"])
                s_age = (now - s_entry).total_seconds() / 60
                if s_age > 16:
                    s_market_id = shadow.get("market_id", "")
                    s_size = shadow.get("size_usd", 5.0)
                    s_resolved = False
                    if s_market_id:
                        try:
                            import httpx
                            r = httpx.get(
                                "https://gamma-api.polymarket.com/markets",
                                params={"condition_id": s_market_id, "limit": "1"},
                                headers={"User-Agent": "Mozilla/5.0"},
                                timeout=10
                            )
                            if r.status_code == 200 and r.json():
                                mkt = r.json()[0]
                                outcomes = mkt.get("outcomes", [])
                                prices = mkt.get("outcomePrices", [])
                                if isinstance(outcomes, str):
                                    outcomes = json.loads(outcomes)
                                if isinstance(prices, str):
                                    prices = json.loads(prices)
                                for i, outcome in enumerate(outcomes):
                                    if str(outcome).lower() == shadow["side"].lower() and i < len(prices):
                                        res_price = float(prices[i])
                                        if res_price >= 0.95:
                                            s_exit = s_size / shadow["entry_price"]
                                        elif res_price <= 0.05:
                                            s_exit = 0
                                        else:
                                            s_exit = (s_size / shadow["entry_price"]) * res_price
                                        shadow["exit_price"] = res_price
                                        shadow["pnl"] = round(s_exit - s_size, 2)
                                        shadow["status"] = "closed"
                                        shadow["exit_time"] = now.isoformat()
                                        won = shadow["pnl"] > 0
                                        # Track filter stats
                                        fname = shadow.get("filter_name")
                                        if fname and fname in self.filter_stats:
                                            if won:
                                                self.filter_stats[fname]["wins"] += 1
                                            else:
                                                self.filter_stats[fname]["losses"] += 1
                                            self.filter_stats[fname]["pnl"] += shadow["pnl"]
                                        reason = shadow.get("reason", "shadow")
                                        result = "WIN" if won else "LOSS"
                                        print(f"[SHADOW {result}] {shadow['side']} ${shadow['pnl']:+.2f} ({reason}) | {shadow.get('market_title', skey)[:35]}...")
                                        s_resolved = True
                                        break
                        except Exception:
                            pass
                    if not s_resolved:
                        shadow["status"] = "closed"
                        shadow["pnl"] = 0.0
                        shadow["exit_price"] = shadow["entry_price"]
                        shadow["exit_time"] = now.isoformat()
            except Exception:
                pass

        # Keep only last 200 shadow trades to prevent file bloat
        if len(self.shadow_trades) > 200:
            closed_shadows = [(k, v) for k, v in self.shadow_trades.items() if v.get("status") == "closed"]
            closed_shadows.sort(key=lambda x: x[1].get("entry_time", ""))
            for k, _ in closed_shadows[:len(closed_shadows) - 150]:
                del self.shadow_trades[k]

        self._save()
        return main_signal

    def _check_feature_revoke(self):
        """ML auto-revoke: disable IH2P features if they're net-negative after enough trades."""
        ss = self.shadow_stats

        # Bond mode: after 10 trades, disable if WR < 60% or net PnL < 0
        if self.BOND_MODE_ENABLED and ss["bond_trades"] >= 10:
            bond_wr = ss["bond_wins"] / max(1, ss["bond_trades"]) * 100
            if bond_wr < 60 or ss["bond_pnl"] < 0:
                self.BOND_MODE_ENABLED = False
                print(f"[ML-REVOKE] Bond mode DISABLED (WR={bond_wr:.0f}%, PnL=${ss['bond_pnl']:+.2f} after {ss['bond_trades']} trades)")

        # DCA: after 10 topups, disable if topup portion PnL < 0
        if self.DCA_ENABLED and ss["dca_topups"] >= 10:
            if ss["dca_topup_pnl"] < 0:
                self.DCA_ENABLED = False
                print(f"[ML-REVOKE] DCA topups DISABLED (topup PnL=${ss['dca_topup_pnl']:+.2f} after {ss['dca_topups']} topups)")

        # Hedge: after 8 trades, disable if PnL < -$3
        if self.HEDGE_ENABLED and ss["hedge_trades"] >= 8:
            if ss["hedge_pnl"] < -3.0:
                self.HEDGE_ENABLED = False
                print(f"[ML-REVOKE] Hedges DISABLED (PnL=${ss['hedge_pnl']:+.2f} after {ss['hedge_trades']} trades)")

    def _track_hydra_result(self, trade, won: bool):
        """V3.15b: Track per-strategy hydra results and check for demotion."""
        strategies = trade.features.get("_hydra_strategies", [])
        if not strategies:
            return
        for strat_name in strategies:
            if strat_name not in self.hydra_live_stats:
                self.hydra_live_stats[strat_name] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0, "status": "PROBATION"}
            s = self.hydra_live_stats[strat_name]
            s["trades"] += 1
            s["pnl"] += trade.pnl
            if won:
                s["wins"] += 1
            else:
                s["losses"] += 1
            wr = s["wins"] / max(1, s["trades"]) * 100
            print(f"  [HYDRA-TRACK] {strat_name}: {s['trades']}T {wr:.0f}%WR ${s['pnl']:+.2f}")

            # Auto-demotion check: WR < 40% after 8+ trades -> back to quarantine
            if s["trades"] >= self.HYDRA_DEMOTE_MIN_TRADES:
                if s["wins"] / s["trades"] < self.HYDRA_DEMOTE_MIN_WR:
                    s["status"] = "DEMOTED"
                    if strat_name in self.hydra_promoted:
                        self.hydra_promoted.remove(strat_name)
                    print(f"  [HYDRA-DEMOTE] {strat_name} demoted! WR={wr:.0f}% < 40% after {s['trades']} trades -> back to quarantine")

    def _check_filter_auto_revert(self, won: bool):
        """V3.15: Track post-reduction WR and auto-revert to strict filters if losing.

        Called after each trade resolves. If WR drops below 50% after 15 trades
        in LOOSE mode, reverts to original strict filter settings.
        """
        if self.filter_mode != "LOOSE":
            return  # Already reverted or in strict mode

        if won:
            self.filter_mode_wins += 1
        else:
            self.filter_mode_losses += 1
        self.filter_mode_trades += 1

        if self.filter_mode_trades >= self.FILTER_REVERT_THRESHOLD:
            wr = self.filter_mode_wins / self.filter_mode_trades
            if wr < self.FILTER_REVERT_MIN_WR:
                # REVERT TO STRICT MODE
                print(f"[AUTO-REVERT] WR={wr:.0%} ({self.filter_mode_wins}W/{self.filter_mode_losses}L) < 50% after {self.filter_mode_trades} trades")
                print(f"[AUTO-REVERT] Reverting to STRICT filter mode!")
                self.filter_mode = "STRICT"
                # Restore strict filter values
                self.MIN_EDGE = 0.25
                self.LOW_EDGE_THRESHOLD = 0.25
                self.MAX_ENTRY_PRICE = 0.45
                self.MIN_ENTRY_PRICE = 0.25
                self.ETH_MIN_CONFIDENCE = 0.70
                self.SOL_DOWN_MIN_EDGE = 0.35
                self.SKIP_HOURS_UTC = {5, 6, 8, 9, 10, 12, 14, 16}
                self.MIN_TIME_REMAINING = 5.0
                self.MAX_TIME_REMAINING = 9.0
                self.use_nyu_model = True  # Re-enable NYU filter
                # Also auto-promote top hydra strategy from paper quarantine
                self._auto_promote_top_hydra()
            else:
                print(f"[FILTER-MODE] LOOSE: {wr:.0%} WR ({self.filter_mode_wins}W/{self.filter_mode_losses}L) after {self.filter_mode_trades} trades - keeping loose filters")

    def _auto_promote_top_hydra(self):
        """V3.15: Read paper quarantine data and promote top performer to live."""
        try:
            paper_file = Path(__file__).parent / "ta_paper_results.json"
            if not paper_file.exists():
                return
            paper_data = json.load(open(paper_file))
            hq = paper_data.get("hydra_quarantine", {})
            best_name = None
            best_score = 0
            for name, q in hq.items():
                n = q.get("predictions", 0)
                if n >= 10:
                    wr = q.get("correct", 0) / n
                    pnl = q.get("pnl", 0)
                    if wr >= 0.60 and pnl > 0:
                        score = wr * pnl  # Combined score
                        if score > best_score:
                            best_score = score
                            best_name = name
            if best_name and best_name not in self.hydra_promoted:
                self.hydra_promoted.append(best_name)
                print(f"[AUTO-PROMOTE] {best_name} added to live hydra strategies (score={best_score:.1f})")
        except Exception as e:
            print(f"[AUTO-PROMOTE] Error reading paper data: {e}")

    def _record_filter_shadow(self, asset: str, side: str, entry_price: float,
                              market_id: str, market_title: str, filter_name: str,
                              size_usd: float = 5.0):
        """Record a shadow trade for a filter rejection (V3.9).

        Tracks what would have happened if we took the trade the filter blocked.
        Used to evaluate whether each filter is helping or hurting.
        """
        shadow_key = f"filter_{market_id}_{side}_{filter_name}"
        if shadow_key in self.shadow_trades:
            return  # Already tracking this one
        self.shadow_trades[shadow_key] = {
            "asset": asset, "side": side, "entry_price": entry_price,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "market_id": market_id,
            "market_title": f"[{asset}] {market_title[:75]}",
            "size_usd": size_usd,
            "status": "open", "reason": f"filter_{filter_name}",
            "filter_name": filter_name,
        }
        # Init filter stats if new
        if filter_name not in self.filter_stats:
            self.filter_stats[filter_name] = {"blocked": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        self.filter_stats[filter_name]["blocked"] += 1

    def _auto_evolve(self):
        """Auto-evolution engine (V3.9): adjusts filters based on shadow trade data.

        Runs every 30 minutes. Analyzes filter_stats to determine which filters
        are helping (blocking losers) vs hurting (blocking winners). Auto-adjusts
        parameters when data is conclusive (10+ resolved shadows per filter).
        """
        changes = []
        total_trades = self.wins + self.losses

        # 1. FILTER EFFECTIVENESS: Auto-loosen filters blocking >60% winners
        for fname, fs in self.filter_stats.items():
            resolved = fs["wins"] + fs["losses"]
            if resolved < 10:
                continue  # Not enough data
            fwr = fs["wins"] / resolved * 100

            if fwr > 65 and fs["pnl"] > 5.0:
                # V3.18: ML AUTO-ADJUST — actually loosen hurting filters
                adjusted = False
                if fname == "v314_vwap_dist":
                    old = self.ml_vwap_abs_cap
                    self.ml_vwap_abs_cap = min(old + 0.10, 1.50)
                    changes.append(f"[ML-EVOLVE] {fname}: vwap_abs_cap {old:.2f} -> {self.ml_vwap_abs_cap:.2f}")
                    adjusted = True
                elif fname == "v38_vwap":
                    old = self.ml_vwap_dir_cap
                    self.ml_vwap_dir_cap = min(old + 0.05, 0.80)
                    changes.append(f"[ML-EVOLVE] {fname}: vwap_dir_cap {old:.2f} -> {self.ml_vwap_dir_cap:.2f}")
                    adjusted = True
                elif fname == "kl_divergence":
                    old = self.ml_kl_floor
                    self.ml_kl_floor = max(old - 0.01, 0.01)
                    changes.append(f"[ML-EVOLVE] {fname}: kl_floor {old:.3f} -> {self.ml_kl_floor:.3f}")
                    adjusted = True
                elif fname == "v39_eth_price":
                    old = self.ETH_MAX_PRICE
                    self.ETH_MAX_PRICE = min(old + 0.05, 0.85)
                    changes.append(f"[ML-EVOLVE] {fname}: ETH_MAX_PRICE {old:.2f} -> {self.ETH_MAX_PRICE:.2f}")
                    adjusted = True
                elif fname == "edge_floor":
                    changes.append(f"[ML-EVOLVE] {fname}: Already tag-only (V3.18)")
                else:
                    changes.append(f"[EVOLVE] WARNING: {fname} blocking {fwr:.0f}% winners (${fs['pnl']:+.2f} missed). No auto-fix mapped.")
                # Reset stats after adjustment for clean re-evaluation
                if adjusted:
                    self.filter_stats[fname] = {"blocked": 0, "wins": 0, "losses": 0, "pnl": 0.0}

            elif fwr < 30:
                # V3.18: ML AUTO-TIGHTEN — filter is working great, tighten if very loose
                tightened = False
                if fname == "v314_vwap_dist" and self.ml_vwap_abs_cap > 0.50:
                    old = self.ml_vwap_abs_cap
                    self.ml_vwap_abs_cap = max(old - 0.10, 0.40)
                    changes.append(f"[ML-EVOLVE] {fname}: TIGHTENED vwap_abs_cap {old:.2f} -> {self.ml_vwap_abs_cap:.2f}")
                    tightened = True
                elif fname == "v38_vwap" and self.ml_vwap_dir_cap > 0.30:
                    old = self.ml_vwap_dir_cap
                    self.ml_vwap_dir_cap = max(old - 0.05, 0.20)
                    changes.append(f"[ML-EVOLVE] {fname}: TIGHTENED vwap_dir_cap {old:.2f} -> {self.ml_vwap_dir_cap:.2f}")
                    tightened = True
                elif fname == "kl_divergence" and self.ml_kl_floor < 0.10:
                    old = self.ml_kl_floor
                    self.ml_kl_floor = min(old + 0.01, 0.15)
                    changes.append(f"[ML-EVOLVE] {fname}: TIGHTENED kl_floor {old:.3f} -> {self.ml_kl_floor:.3f}")
                    tightened = True
                else:
                    changes.append(f"[EVOLVE] {fname} EFFECTIVE: blocking {100-fwr:.0f}% losers. KEEP.")
                if tightened:
                    self.filter_stats[fname] = {"blocked": 0, "wins": 0, "losses": 0, "pnl": 0.0}

        # 2. HOURLY PERFORMANCE: Auto-add/remove skip hours
        for h in range(24):
            s = self.hourly_stats[h]
            trades_h = s["wins"] + s["losses"]
            if trades_h < 5:
                continue

            wr = s["wins"] / trades_h * 100
            # If an active hour has <25% WR over 5+ trades, add to skip hours
            if h not in self.SKIP_HOURS_UTC and wr < 25 and trades_h >= 5:
                self.SKIP_HOURS_UTC.add(h)
                changes.append(f"[EVOLVE] UTC {h} BLOCKED: {wr:.0f}% WR over {trades_h} trades — auto-added to skip hours")

            # If a skipped hour has >65% WR over 8+ trades in shadow, remove from skip
            # (Would need shadow data for skip hours — future enhancement)

        # 3. ETH PROMOTION CHECK: If ETH shadows show >60% WR over 15+ resolved
        eth_shadows = [s for s in self.shadow_trades.values()
                       if s.get("asset") == "ETH" and s.get("status") == "closed"
                       and s.get("reason") == "shadow_asset"]
        eth_wins = sum(1 for s in eth_shadows if s.get("pnl", 0) > 0)
        eth_total = len(eth_shadows)
        if eth_total >= 15:
            eth_wr = eth_wins / eth_total * 100
            if eth_wr >= 60:
                changes.append(f"[EVOLVE] ETH PROMOTION READY: {eth_wr:.0f}% WR over {eth_total} shadow trades. Consider promoting to live.")
            elif eth_wr < 40:
                changes.append(f"[EVOLVE] ETH STRUGGLING: {eth_wr:.0f}% WR over {eth_total} shadows. Keep shadow-only.")

        # 4. OVERALL HEALTH: Warn if WR is declining
        if total_trades >= 10:
            overall_wr = self.wins / total_trades * 100
            if overall_wr < 40:
                changes.append(f"[EVOLVE] WARNING: Overall WR {overall_wr:.0f}% is below 40%. Filters may be too loose or market regime shifted.")
            elif overall_wr > 65:
                changes.append(f"[EVOLVE] STRONG: Overall WR {overall_wr:.0f}%. Current filters are working well.")

        if changes:
            print(f"\n{'='*50}")
            print(f"AUTO-EVOLUTION CHECK ({datetime.now(timezone.utc).strftime('%H:%M UTC')})")
            for c in changes:
                print(f"  {c}")
            print(f"{'='*50}")

    async def _process_dca_topups(self, asset_data: Dict):
        """Execute pending DCA top-up orders if price is same or better."""
        if not self.DCA_ENABLED or not self.pending_topups:
            return

        now = datetime.now(timezone.utc)
        for key, topup in list(self.pending_topups.items()):
            # Expire old topups (> 2 min)
            created = datetime.fromisoformat(topup["created"])
            if (now - created).total_seconds() > 120:
                print(f"[DCA] Topup expired for {key[:30]} (>2min)")
                del self.pending_topups[key]
                continue

            # Check if the primary trade is still open
            if key not in self.trades or self.trades[key].status != "open":
                print(f"[DCA] Trade already closed, discarding topup for {key[:30]}")
                del self.pending_topups[key]
                continue

            # Get current price for this market
            asset = topup["asset"]
            if asset not in asset_data:
                continue

            current_price = None
            for mkt in asset_data[asset].get("markets", []):
                mkt_id = mkt.get("conditionId", mkt.get("condition_id", ""))
                if mkt_id == topup["market_id"]:
                    outcomes = mkt.get("outcomes", [])
                    prices = mkt.get("outcomePrices", [])
                    if isinstance(outcomes, str):
                        outcomes = json.loads(outcomes)
                    if isinstance(prices, str):
                        prices = json.loads(prices)
                    for i, outcome in enumerate(outcomes):
                        if str(outcome).upper() == topup["side"] and i < len(prices):
                            current_price = float(prices[i])
                            break
                    break

            if current_price is None:
                continue

            # Only top up if price is same or better (cheaper)
            if current_price > topup["max_price"]:
                print(f"[DCA] Price worse ({current_price:.3f} > {topup['max_price']:.3f}), skipping topup")
                continue

            # Execute the top-up
            topup_size = topup["size"]
            success, order_result = await self.execute_trade(
                topup["market"], topup["side"], topup_size, current_price
            )
            if success:
                # Update the primary trade's size
                self.trades[key].size_usd += topup_size
                self.trades[key].features["_dca_topup"] = topup_size
                self.trades[key].features["_dca_topup_price"] = current_price
                self.shadow_stats["dca_topups"] += 1
                print(f"[DCA] Topup +${topup_size:.2f} @ ${current_price:.3f} for {key[:30]} | Total: ${self.trades[key].size_usd:.2f}")
            else:
                print(f"[DCA] Topup failed for {key[:30]}: {order_result}")

            del self.pending_topups[key]

    def print_update(self, signal):
        """Print status update."""
        now = datetime.now(timezone.utc)
        open_trades = [t for t in self.trades.values() if t.status == "open"]
        closed_trades = [t for t in self.trades.values() if t.status == "closed"]

        print()
        print("=" * 70)
        mode = "LIVE" if not self.dry_run else "DRY RUN"
        print(f"TA {mode} TRADING UPDATE - {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 70)

        if signal:
            print(f"\nCURRENT SIGNAL:")
            print(f"  BTC: ${signal.current_price:,.2f}")
            print(f"  Model: {signal.model_up:.1%} UP / {signal.model_down:.1%} DOWN")

        print(f"\nTRADING STATS:")
        print(f"  Base Size: ${self.BASE_POSITION_SIZE} (Kelly-scaled ${self.MIN_POSITION_SIZE}-${self.MAX_POSITION_SIZE})")
        print(f"  Bankroll: ${self.bankroll:.2f} ({(self.bankroll/self.initial_bankroll - 1)*100:+.1f}%)")
        print(f"  Total trades: {len(closed_trades)}")
        print(f"  Win/Loss: {self.wins}/{self.losses}")
        win_rate = (self.wins / max(1, self.wins + self.losses)) * 100
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Total PnL: ${self.total_pnl:+.2f}")
        print(f"  Streak: {self.consecutive_wins}W / {self.consecutive_losses}L")

        print(f"\nML OPTIMIZATION:")
        print(f"  ML Rejections: {self.ml_rejections}")
        print(f"  Score Threshold: {self.ml.get_min_score_threshold():.2f}")
        print(f"  Feature Weights: KL={self.ml.feature_weights['kl_divergence']:.2f}")

        # Shadow trade analysis - ML evaluation of ATR filter + trend bias
        ss = self.shadow_stats
        atr_total = ss["atr_blocked_wins"] + ss["atr_blocked_losses"]
        print(f"\nSHADOW TRACKING (filter evaluation):")
        print(f"  ATR Filter:")
        print(f"    Blocked: {ss['atr_blocked']} trades | Resolved: {atr_total}")
        if atr_total > 0:
            atr_wr = ss["atr_blocked_wins"] / atr_total * 100
            print(f"    Shadow WR: {atr_wr:.0f}% ({ss['atr_blocked_wins']}W/{ss['atr_blocked_losses']}L)")
            print(f"    Shadow PnL: ${ss['atr_blocked_pnl']:+.2f}")
            # ML verdict: if shadow trades would have won, filter is HURTING us
            if atr_wr > win_rate and ss["atr_blocked_pnl"] > 0:
                print(f"    VERDICT: ATR filter HURTING (+${ss['atr_blocked_pnl']:.2f} missed) - CONSIDER REMOVING")
            elif atr_wr < 40:
                print(f"    VERDICT: ATR filter HELPING (blocked {atr_wr:.0f}% WR losers)")
            else:
                print(f"    VERDICT: INCONCLUSIVE (need more data)")
        else:
            print(f"    No resolved shadow trades yet")

        print(f"  Trend Bias (85% counter-trend):")
        print(f"    Counter-trend trades: {ss['trend_bias_trades']}")
        if ss["trend_bias_trades"] > 0:
            actual = ss["trend_bias_actual_pnl"]
            full = ss["trend_bias_full_pnl"]
            saved = actual - full  # Positive = bias saved money (lost less)
            print(f"    Actual PnL: ${actual:+.2f} | Full-size PnL: ${full:+.2f}")
            if saved > 0:
                print(f"    VERDICT: Trend bias SAVED ${saved:.2f} - KEEP")
            elif saved < -5:
                print(f"    VERDICT: Trend bias COST ${-saved:.2f} - CONSIDER REMOVING")
            else:
                print(f"    VERDICT: INCONCLUSIVE (diff ${saved:+.2f})")

        # Filter rejection tracking (V3.9)
        if self.filter_stats:
            print(f"\nFILTER EFFECTIVENESS (V3.9 shadow tracking):")
            for fname, fs in sorted(self.filter_stats.items(), key=lambda x: x[1]["blocked"], reverse=True):
                resolved = fs["wins"] + fs["losses"]
                if resolved > 0:
                    fwr = fs["wins"] / resolved * 100
                    verdict = "HELPING" if fwr < 40 else ("HURTING" if fwr > 60 else "NEUTRAL")
                    print(f"  {fname}: {fs['blocked']} blocked | {resolved} resolved ({fs['wins']}W/{fs['losses']}L {fwr:.0f}%WR) ${fs['pnl']:+.2f} | {verdict}")
                else:
                    print(f"  {fname}: {fs['blocked']} blocked | 0 resolved (awaiting data)")

        # Hourly ML sizing display
        total_hourly_trades = sum(s["wins"] + s["losses"] for s in self.hourly_stats.values())
        phase = "BOOST+REDUCE" if total_hourly_trades >= 10 else "REDUCE-ONLY"
        print(f"\nHOURLY ML SIZING ({phase}, {total_hourly_trades} total trades):")
        active_hours = []
        for h in range(24):
            s = self.hourly_stats[h]
            trades_h = s["wins"] + s["losses"]
            if trades_h > 0:
                wr = s["wins"] / trades_h * 100
                mult = self._get_hour_multiplier(h)
                mst = (h - 7) % 24
                tag = ""
                if mult < 1.0:
                    tag = " REDUCED"
                elif mult > 1.0:
                    tag = " BOOSTED"
                active_hours.append(f"    UTC {h:>2} (MST {mst:>2}h): {trades_h}T {wr:.0f}% WR ${s['pnl']:+.1f} -> {mult}x{tag}")
        if active_hours:
            for line in active_hours:
                print(line)
        else:
            print("    No hourly data yet - all hours at 1.0x (collecting...)")
        # Show current hour multiplier
        cur_h = now.hour
        cur_mult = self._get_hour_multiplier(cur_h)
        print(f"  Current: UTC {cur_h} -> {cur_mult}x multiplier")

        # V3.18 ML-Tunable Thresholds
        print(f"\nML FILTER THRESHOLDS (V3.18 auto-evolve):")
        print(f"  VWAP abs_cap: {self.ml_vwap_abs_cap:.2f} | VWAP dir_cap: {self.ml_vwap_dir_cap:.2f} | KL floor: {self.ml_kl_floor:.3f} | ETH max: {self.ETH_MAX_PRICE:.2f}")
        print(f"  edge_floor: TAG-ONLY (disabled V3.18)")

        # V3.15 Filter Mode + Hydra Status
        fm_wr = self.filter_mode_wins / max(1, self.filter_mode_trades) * 100
        print(f"\nV3.15 FILTER MODE: {self.filter_mode}")
        print(f"  Post-reduction: {self.filter_mode_trades}T ({self.filter_mode_wins}W/{self.filter_mode_losses}L, {fm_wr:.0f}%WR)")
        if self.filter_mode == "LOOSE":
            print(f"  Auto-revert: at {self.FILTER_REVERT_THRESHOLD} trades if WR < {self.FILTER_REVERT_MIN_WR:.0%}")
        else:
            print(f"  Reverted to STRICT mode (filters restored)")
        print(f"  Hydra promoted: {', '.join(self.hydra_promoted) if self.hydra_promoted else 'none'}")
        print(f"  Hydra auto-demote: WR < {self.HYDRA_DEMOTE_MIN_WR:.0%} after {self.HYDRA_DEMOTE_MIN_TRADES}+ trades")
        for sname, hs in self.hydra_live_stats.items():
            if sname.startswith("_"):
                continue
            h_wr = hs["wins"] / max(1, hs["trades"]) * 100
            print(f"    {sname}: {hs.get('status','?')} | {hs['trades']}T ({hs['wins']}W/{hs['losses']}L, {h_wr:.0f}%WR, ${hs['pnl']:+.2f})")

        # IH2P Adaptations status
        ss = self.shadow_stats
        print(f"\nIH2P ADAPTATIONS:")
        bond_status = "ENABLED" if self.BOND_MODE_ENABLED else "DISABLED (ML-revoked)"
        bond_wr = ss["bond_wins"] / max(1, ss["bond_trades"]) * 100 if ss["bond_trades"] > 0 else 0
        print(f"  Bond Mode: {bond_status} | {ss['bond_trades']} trades ({ss['bond_wins']}W/{ss['bond_losses']}L, {bond_wr:.0f}% WR, ${ss['bond_pnl']:+.2f})")
        dca_status = "ENABLED" if self.DCA_ENABLED else "DISABLED (ML-revoked)"
        print(f"  DCA Topup: {dca_status} | {ss['dca_topups']} topups (${ss['dca_topup_pnl']:+.2f})")
        hedge_status = "ENABLED" if self.HEDGE_ENABLED else "DISABLED (ML-revoked)"
        hedge_wr = ss["hedge_wins"] / max(1, ss["hedge_trades"]) * 100 if ss["hedge_trades"] > 0 else 0
        print(f"  Hedge:     {hedge_status} | {ss['hedge_trades']} trades ({ss['hedge_wins']}W/{ss['hedge_losses']}L, {hedge_wr:.0f}% WR, ${ss['hedge_pnl']:+.2f})")

        # V3.17: GARCH + CEEMD quant models status
        g = self._last_garch
        c = self._last_ceemd
        garch_damper = {
            "HIGH": "0.70x", "LOW": "1.15x", "NORMAL": "1.00x"
        }.get(g["regime"], "?")
        garch_tp = {"LOW": "0.80", "HIGH": "0.90", "NORMAL": "0.85"}.get(g["regime"], "0.85")
        trend_dir = "UP" if c["trend_slope"] > 0 else "DOWN" if c["trend_slope"] < 0 else "FLAT"
        print(f"\nQUANT MODELS (V3.17):")
        print(f"  GARCH: {g['regime']} vol (ratio={g['ratio']:.2f}, forecast={g['forecast']:.4f}%) | Damper: {garch_damper} | TP: {garch_tp}")
        print(f"  CEEMD: slope={c['trend_slope']:+.6f} ({trend_dir}) | noise={c['noise_ratio']:.2f} | {c['n_imfs']} IMFs")
        # GARCH per-regime WR
        for regime in ("low", "normal", "high"):
            rt = ss.get(f"garch_trades_{regime}", 0)
            rw = ss.get(f"garch_wins_{regime}", 0)
            if rt > 0:
                print(f"    {regime.upper()}: {rt}T ({rw}W, {rw/rt*100:.0f}% WR)")
        # CEEMD alignment WR
        ca = ss.get("ceemd_aligned_trades", 0)
        caw = ss.get("ceemd_aligned_wins", 0)
        co = ss.get("ceemd_opposed_trades", 0)
        cow = ss.get("ceemd_opposed_wins", 0)
        cb = ss.get("ceemd_noisy_blocked", 0)
        if ca + co > 0:
            print(f"    Aligned: {ca}T ({caw}W, {caw/max(1,ca)*100:.0f}% WR) | Opposed: {co}T ({cow}W, {cow/max(1,co)*100:.0f}% WR) | Noisy blocked: {cb}")

        print(f"\nOPEN TRADES ({len(open_trades)}):")
        for t in open_trades[:5]:
            print(f"  {t.side} @ ${t.entry_price:.4f} | {t.market_title[:35]}...")

        print("=" * 70)

    async def run(self):
        """Main loop."""
        print("=" * 70)
        mode = "LIVE" if not self.dry_run else "DRY RUN"
        print(f"TA + BREGMAN + ML {mode} TRADER - Multi-Asset 15-Minute Markets")
        print("=" * 70)
        print(f"HARD BET LIMITS: ${self.MIN_POSITION_SIZE}-${self.HARD_MAX_BET} (ABSOLUTE CEILING, NO EXCEPTIONS)")
        print(f"PID LOCK: ENABLED — prevents ghost double-instances")
        print(f"OVERLAP CHECK: ENABLED — queries account before every trade")
        print(f"IH2P: ALL DISABLED (Bond/DCA/Hedge off during capital rebuild)")
        print(f"ML Optimization: ENABLED")
        print(f"Live Assets: {', '.join(self.ASSETS.keys())} | Shadow: {', '.join(self.SHADOW_ASSETS.keys())}")
        print(f"Max {self.MAX_CONCURRENT_POSITIONS} concurrent | Edge>={self.MIN_EDGE:.0%}")
        print(f"HOURLY ML SIZING: Bayesian WR per hour -> reduce bad hours, boost good (after 10 trades)")
        print(f"Filters: Edge>={self.MIN_EDGE:.0%} | Conf>={self.MIN_MODEL_CONFIDENCE:.0%} | KL>={self.MIN_KL_DIVERGENCE} | ATR(14)x1.5 | NYU>0.10")
        print(f"UP: Dynamic max price (70%->$0.42, 80%->$0.48, 85%->$0.55) | Scaled conf for cheap entries")
        print(f"DOWN: Death zone $0.40-0.45=SKIP | Break-even conf for cheap | Momentum confirm >{self.DOWN_MIN_MOMENTUM_DROP}")
        print(f"NYU model: edge_score>0.15 (avoid 50% zone)")
        print(f"Skip Hours (UTC): {sorted(self.SKIP_HOURS_UTC)}")
        print(f"Entry Window: {self.MIN_TIME_REMAINING}-{self.MAX_TIME_REMAINING} min")
        print(f"Scan interval: 30 seconds (+ auto-redeem every 30s)")
        print("=" * 70)

        # === V3.11: GHOST ORDER CLEANUP + BALANCE SYNC ===
        # Cancel ALL pending GTC orders from previous sessions to prevent stacking
        self._cancel_all_pending_orders()

        # Sync bankroll with actual CLOB balance — ALWAYS trust CLOB
        clob_bal = self._sync_clob_balance()
        if clob_bal >= 0:
            drift = abs(clob_bal - self.bankroll)
            if drift > 1:  # V3.12c: Tightened from $5 to $1 — CLOB is truth
                print(f"[BALANCE] CLOB ${clob_bal:.2f} vs tracked ${self.bankroll:.2f} (drift ${drift:.2f})")
                print(f"[BALANCE] Syncing bankroll to CLOB reality: ${self.bankroll:.2f} -> ${clob_bal:.2f}")
                self.bankroll = clob_bal
                self._save()
            else:
                print(f"[BALANCE] CLOB ${clob_bal:.2f} matches tracked ${self.bankroll:.2f} (drift ${drift:.2f})")

        last_update = 0
        last_redeem_check = 0
        last_evolve_check = 0
        last_balance_sync = time.time()  # V3.11: periodic balance sync
        cycle = 0

        while True:
            try:
                cycle += 1
                now = time.time()

                # Auto-redeem check every 30 seconds
                if not self.dry_run and now - last_redeem_check >= 30:
                    try:
                        claimed = auto_redeem_winnings()
                        if claimed and claimed > 0:
                            # V3.10: Do NOT add to bankroll directly — PnL already counted on resolution
                            # V3.15c FIX: But DO sync CLOB balance so bankroll reflects redeemed funds
                            print(f"[REDEEM] Claimed ${claimed:.2f} — syncing CLOB balance...")
                            clob_bal = self._sync_clob_balance()
                            if clob_bal >= 0:
                                self.bankroll = clob_bal
                                print(f"[REDEEM] Bankroll updated to ${self.bankroll:.2f} (was stale before redeem)")
                            self._save()
                    except Exception as e:
                        if "No winning" not in str(e):
                            print(f"[REDEEM] Error: {e}")
                    last_redeem_check = now

                # V3.12c: CLOB balance sync every 2 min (was 5 min), $1 threshold (was $10)
                if not self.dry_run and now - last_balance_sync >= 120:
                    clob_bal = self._sync_clob_balance()
                    if clob_bal >= 0 and abs(clob_bal - self.bankroll) > 1:
                        print(f"[BALANCE] Drift detected: CLOB ${clob_bal:.2f} vs tracked ${self.bankroll:.2f}")
                        self.bankroll = clob_bal
                        self._save()
                    last_balance_sync = now

                signal = await self.run_cycle()

                if signal:
                    print(f"[Scan {cycle}] BTC ${signal.current_price:,.2f} | {signal.regime.value} | Model: {signal.model_up:.0%} UP")

                # 10-minute update
                if now - last_update >= 600:
                    self.print_update(signal)
                    last_update = now

                # 30-minute auto-evolution check
                if now - last_evolve_check >= 1800:
                    self._auto_evolve()
                    last_evolve_check = now

                await asyncio.sleep(30)  # 30s - faster scanning + redeem checks

            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(60)

        self.print_update(None)
        print(f"Results saved to: {self.OUTPUT_FILE}")


def _is_pid_alive(pid: int) -> bool:
    """Check if a process is alive (Windows-compatible)."""
    import subprocess
    try:
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
            capture_output=True, text=True, timeout=5
        )
        # tasklist output contains the PID number if process exists
        return str(pid) in result.stdout
    except Exception:
        # Fallback to os.kill
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def acquire_pid_lock():
    """Prevent multiple live trader instances. Dies if another is running."""
    pid_file = Path(__file__).parent / "ta_live.pid"
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            if old_pid == os.getpid():
                pass  # It's us, re-acquiring
            elif _is_pid_alive(old_pid):
                print(f"[FATAL] Another live trader is already running (PID {old_pid})")
                print(f"[FATAL] Kill it first: taskkill /F /PID {old_pid}")
                print(f"[FATAL] Or delete {pid_file} if the process is dead")
                sys.exit(1)
            else:
                print(f"[PID] Stale lock (PID {old_pid} dead) — taking over")
        except (ValueError, IOError):
            pass  # Corrupt file, overwrite
    # Write our PID
    pid_file.write_text(str(os.getpid()))
    print(f"[PID] Lock acquired (PID {os.getpid()}) — {pid_file}")
    return pid_file


def release_pid_lock():
    """Release PID lock on exit."""
    pid_file = Path(__file__).parent / "ta_live.pid"
    try:
        if pid_file.exists():
            stored_pid = int(pid_file.read_text().strip())
            if stored_pid == os.getpid():
                pid_file.unlink()
                print(f"[PID] Lock released")
    except Exception:
        pass


if __name__ == "__main__":
    import argparse
    import atexit
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Paper trade only")
    args = parser.parse_args()

    # === PID LOCK: Prevent ghost double-instances ===
    pid_file = acquire_pid_lock()
    atexit.register(release_pid_lock)

    trader = TALiveTrader(dry_run=args.dry_run)
    asyncio.run(trader.run())
