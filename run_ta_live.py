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
from datetime import datetime, timezone
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

from arbitrage.ta_signals import TASignalGenerator, Candle, SignalStrength, TASignal
from arbitrage.bregman_optimizer import BregmanOptimizer, BregmanSignal

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


def auto_redeem_winnings():
    """Auto-redeem any resolved winning positions to USDC. Runs silently."""
    try:
        from web3 import Web3
        from dotenv import load_dotenv
        load_dotenv()

        proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
        if not proxy_address:
            return

        import httpx
        r = httpx.get(
            "https://data-api.polymarket.com/positions",
            params={"user": proxy_address, "sizeThreshold": "0"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        redeemable = [p for p in r.json() if p.get("redeemable", False)]
        if not redeemable:
            return

        total_val = sum(float(p.get("currentValue", 0) or 0) for p in redeemable)
        if total_val <= 0:
            return  # Nothing worth claiming

        rpcs = ["https://polygon-rpc.com", "https://rpc.ankr.com/polygon", "https://polygon.llamarpc.com"]
        w3 = None
        for rpc in rpcs:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 30}))
                if w3.is_connected():
                    break
            except Exception:
                continue
        if not w3 or not w3.is_connected():
            return

        key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        if key.startswith("ENC:"):
            sys.path.insert(0, str(Path(__file__).parent))
            from crypto_utils import decrypt_key
            salt = os.getenv("POLYMARKET_KEY_SALT", "")
            password = os.getenv("POLYMARKET_PASSWORD", "")
            if not password:
                return
            key = decrypt_key(key[4:], salt, password)
        if not key:
            return
        if not key.startswith("0x"):
            key = "0x" + key

        account = w3.eth.account.from_key(key)
        CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
        USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        FACTORY = "0xaB45c5A4B0c941a2F231C04C3f49182e1A254052"

        ctf_abi = [{"inputs":[{"name":"collateralToken","type":"address"},{"name":"parentCollectionId","type":"bytes32"},{"name":"conditionId","type":"bytes32"},{"name":"indexSets","type":"uint256[]"}],"name":"redeemPositions","outputs":[],"stateMutability":"nonpayable","type":"function"}]
        factory_abi = [{"inputs":[{"components":[{"name":"to","type":"address"},{"name":"typeCode","type":"uint256"},{"name":"data","type":"bytes"},{"name":"value","type":"uint256"}],"name":"_txns","type":"tuple[]"}],"name":"proxy","outputs":[],"stateMutability":"nonpayable","type":"function"}]

        ctf = w3.eth.contract(address=Web3.to_checksum_address(CTF), abi=ctf_abi)
        factory = w3.eth.contract(address=Web3.to_checksum_address(FACTORY), abi=factory_abi)

        claimed = 0
        done = set()
        for p in redeemable:
            cid = p.get("conditionId", "")
            val = float(p.get("currentValue", 0) or 0)
            if not cid or cid in done or val <= 0:
                continue
            done.add(cid)
            try:
                redeem_data = ctf.encode_abi("redeemPositions", args=[
                    Web3.to_checksum_address(USDC), bytes(32),
                    Web3.to_bytes(hexstr=cid), [1, 2],
                ])
                proxy_txn = (Web3.to_checksum_address(CTF), 1, Web3.to_bytes(hexstr=redeem_data), 0)
                nonce = w3.eth.get_transaction_count(account.address)
                gas_price = w3.eth.gas_price
                txn = factory.functions.proxy([proxy_txn]).build_transaction({
                    "from": account.address, "nonce": nonce,
                    "gasPrice": min(gas_price * 2, w3.to_wei(100, "gwei")), "gas": 300000,
                })
                signed = w3.eth.account.sign_transaction(txn, key)
                tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                if receipt["status"] == 1:
                    claimed += 1
                    print(f"[REDEEM] Claimed ${val:.2f} | {p.get('title','')[:40]}")
                time.sleep(15)  # Rate limit
            except Exception:
                time.sleep(15)
                continue

        if claimed:
            print(f"[REDEEM] Auto-claimed {claimed} winning positions")

        # Check remaining POL balance for gas
        try:
            signer = account.address
            bal_wei = w3.eth.get_balance(signer)
            bal_pol = float(w3.from_wei(bal_wei, "ether"))
            gas_price_gwei = float(w3.from_wei(w3.eth.gas_price, "gwei"))
            cost_per = 300000 * gas_price_gwei / 1e9
            remaining = int(bal_pol / cost_per) if cost_per > 0 else 0
            if remaining <= 100:
                print(f"[WARNING] LOW POL BALANCE: {bal_pol:.2f} POL - only ~{remaining} redemptions left!")
                print(f"[WARNING] Top up signer wallet: {signer}")
        except Exception:
            pass
    except Exception as e:
        pass  # Silently fail - don't disrupt trading


class TALiveTrader:
    """Live trades based on TA + Bregman signals with ML optimization."""

    OUTPUT_FILE = Path(__file__).parent / "ta_live_results.json"
    BASE_POSITION_SIZE = 5.0   # Base $5 per trade - conservative
    MIN_POSITION_SIZE = 5.0    # Minimum position
    MAX_POSITION_SIZE = 10.0   # Hard cap $10 - never risk more

    def __init__(self, dry_run: bool = False, bankroll: float = 500.0):
        self.dry_run = dry_run
        self.generator = TASignalGenerator()
        self.bregman = BregmanOptimizer(bankroll=bankroll)
        self.ml = MLOptimizer()
        self.initial_bankroll = bankroll
        self.bankroll = bankroll

        self.trades: Dict[str, LiveTrade] = {}
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.signals_count = 0
        self.ml_rejections = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.start_time = datetime.now(timezone.utc).isoformat()

        # Executor for live trading
        self.executor = None
        if not dry_run:
            self._init_executor()

        self._load()

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

                print(f"[Live] Loaded {len(self.trades)} trades from previous session")
            except Exception as e:
                print(f"[Live] Error loading state: {e}")

    def _save(self):
        """Save current state."""
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
        }
        with open(self.OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    # Multi-asset configuration
    ASSETS = {
        "BTC": {
            "symbol": "BTCUSDT",
            "keywords": ["bitcoin", "btc"],
        },
        "ETH": {
            "symbol": "ETHUSDT",
            "keywords": ["ethereum", "eth"],
        },
        # XRP removed: 46.9% WR, -$4.69 PnL - worst performing asset
        "SOL": {
            "symbol": "SOLUSDT",
            "keywords": ["solana", "sol"],
        },
    }

    # Directional bias based on 200 EMA trend
    # Below 200 EMA = bearish: 70% capital on DOWN, 30% on UP
    # Above 200 EMA = bullish: 70% capital on UP, 30% on DOWN
    # TODO: ML should tune these ratios over time
    TREND_BIAS_STRONG = 0.70   # Capital % for with-trend trades
    TREND_BIAS_WEAK = 0.30     # Capital % for counter-trend trades (30% less)

    # Trading hours filter (UTC) - skip low-WR hours
    # 06:00-08:00 UTC = 40-44% WR, -$27 PnL - avoid
    SKIP_HOURS_UTC = {6, 7, 8}

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
        """Determine trend direction using 200 EMA on 1-min candles (=200 min lookback)."""
        ema200 = self._ema(candles, 200)
        if price < ema200:
            return "BEARISH"  # Below 200 EMA -> favor DOWN
        else:
            return "BULLISH"  # Above 200 EMA -> favor UP

    async def fetch_data(self):
        """Fetch candles and markets for all assets."""
        asset_data = {}  # {asset: (candles, price, markets)}

        async with httpx.AsyncClient(timeout=15) as client:
            # Fetch candles + prices for all assets from Binance
            for asset, cfg in self.ASSETS.items():
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
                        for asset, cfg in self.ASSETS.items():
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

            # 15-min markets have thin order books (asks at $0.97-$0.99 only).
            # Place limit orders at the outcomePrices mid price and let them fill.
            # MAX_ENTRY_PRICE check is already applied in conviction filters.
            print(f"[ORDER] Placing limit {side} @ ${price:.4f} (mid price)")

            # Calculate shares - use whole number to avoid CLOB decimal precision errors
            shares = math.floor(size / price)
            if shares < 1:
                return False, "shares_too_small"

            order_args = OrderArgs(
                price=price,
                size=float(shares),
                side=BUY,
                token_id=token_id,
            )

            actual_cost = shares * price
            print(f"[LIVE] Placing {side} order: {shares} shares @ ${price} = ${actual_cost:.2f}")

            signed_order = self.executor.client.create_order(order_args)
            response = self.executor.client.post_order(signed_order, OrderType.GTC)

            success = response.get("success", False)
            order_id = response.get("orderID", "")

            if not success:
                return False, response.get("errorMsg", "unknown_error")

            # GTC order placed successfully — it will fill when matched
            print(f"[LIVE] Order placed: {order_id[:20]}...")
            return True, order_id

        except Exception as e:
            return False, str(e)

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

    # Conviction thresholds - prevent flip-flopping
    MIN_MODEL_CONFIDENCE = 0.58  # Model must be at least 58% confident in direction
    MIN_EDGE = 0.08              # At least 8% edge vs market price
    MAX_ENTRY_PRICE = 0.55       # Don't buy at prices above 55¢ (no edge zone)
    CANDLE_LOOKBACK = 15         # Only use last 15 minutes of price action

    # Risk management (inspired by crypmancer/polymarket-arbitrage-copy-bot)
    MAX_DAILY_LOSS = 50.0        # Stop trading after $50 daily loss
    MAX_TOTAL_EXPOSURE = 100.0   # Max $100 total open positions
    MAX_SLIPPAGE = 0.05          # Max 5% slippage from mid price to ask
    MAX_CONCURRENT_POSITIONS = 6 # Max 6 open positions (across 4 assets)

    # Entry window - trade EARLY when market is still ~50/50 but model has conviction
    MIN_TIME_REMAINING = 3.0     # Don't enter with < 3 min (market already priced in)
    MAX_TIME_REMAINING = 14.0    # Enter up to 14 min out (catch fresh markets at ~50/50)

    # Kelly position sizing - CONSERVATIVE: slow churn, protect capital
    KELLY_FRACTION = 0.25        # Quarter-Kelly for safety

    def calculate_position_size(self, edge: float, kelly_fraction: float,
                                  btc_1h_change: float, volatility: float) -> float:
        """
        Conservative position sizing: $5 base, scale to $10 max only with
        high ML conviction. Protect capital, churn profits slowly.
        """
        # Start at base ($5)
        size = self.BASE_POSITION_SIZE

        # ML conviction scaling: only increase toward $10 with strong edge+kelly
        conviction = min(1.0, kelly_fraction * edge / 0.10)  # 0-1 score
        size += conviction * (self.MAX_POSITION_SIZE - self.BASE_POSITION_SIZE)

        # Volatility damping: reduce in high vol
        if volatility > 0.15:
            vol_damper = max(0.7, 1.0 - (volatility - 0.15) * 2)
            size *= vol_damper

        # Losing streak protection: drop to minimum
        if self.consecutive_losses >= 2:
            size = self.MIN_POSITION_SIZE

        # Hard clamp $5-$10 always
        size = max(self.MIN_POSITION_SIZE, min(self.MAX_POSITION_SIZE, size))
        return round(size, 2)

    async def run_cycle(self):
        """Run one trading cycle across all assets."""
        # Skip low win-rate hours (06:00-08:00 UTC = 42% WR, -$27 PnL)
        current_hour = datetime.now(timezone.utc).hour
        if current_hour in self.SKIP_HOURS_UTC:
            print(f"[SKIP] Hour {current_hour:02d} UTC is in skip list - resting")
            return None

        asset_data = await self.fetch_data()

        if not asset_data:
            return None

        # Summary of markets found
        total_markets = sum(len(d["markets"]) for d in asset_data.values())
        asset_counts = {a: len(d["markets"]) for a, d in asset_data.items() if d["markets"]}
        if asset_counts:
            counts_str = " | ".join(f"{a}:{c}" for a, c in asset_counts.items())
            print(f"[Markets] {total_markets} total ({counts_str})")

        self.signals_count += 1
        main_signal = None

        # Process each asset
        for asset, data in asset_data.items():
            candles = data["candles"]
            price = data["price"]
            markets = data["markets"]

            if not candles or not markets:
                continue

            recent_candles = candles[-self.CANDLE_LOOKBACK:]

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

                # === CONVICTION FILTERS (with debug) ===
                if not signal.side:
                    print(f"  [{asset}] No signal side (action={signal.action}) | mkt UP=${up_price:.2f} DOWN=${down_price:.2f} | {time_left:.1f}min | reason={signal.reason}")
                    continue
                if signal.side == "UP" and signal.model_up < self.MIN_MODEL_CONFIDENCE:
                    print(f"  [{asset}] UP confidence too low: {signal.model_up:.1%} < {self.MIN_MODEL_CONFIDENCE:.0%}")
                    continue
                if signal.side == "DOWN" and signal.model_down < self.MIN_MODEL_CONFIDENCE:
                    print(f"  [{asset}] DOWN confidence too low: {signal.model_down:.1%} < {self.MIN_MODEL_CONFIDENCE:.0%}")
                    continue

                best_edge = signal.edge_up if signal.side == "UP" else signal.edge_down
                if best_edge is not None and best_edge < self.MIN_EDGE:
                    print(f"  [{asset}] Edge too small: {best_edge:.1%} < {self.MIN_EDGE:.0%}")
                    continue

                entry_price_check = up_price if signal.side == "UP" else down_price
                if entry_price_check is not None and entry_price_check > self.MAX_ENTRY_PRICE:
                    print(f"  [{asset}] Entry price too high: ${entry_price_check:.2f} > ${self.MAX_ENTRY_PRICE}")
                    continue

                if signal.action != "ENTER":
                    print(f"  [{asset}] Action is {signal.action}, not ENTER")
                    continue

                if signal.action == "ENTER" and signal.side and trade_key:
                    if trade_key not in self.trades:
                        # Don't place opposite-side trade on same market (self-hedging)
                        opposite_key = f"{market_id}_{'DOWN' if signal.side == 'UP' else 'UP'}"
                        if opposite_key in self.trades and self.trades[opposite_key].status == "open":
                            continue
                        # === RISK CHECKS ===
                        if self.total_pnl <= -self.MAX_DAILY_LOSS:
                            print(f"[RISK] Daily loss limit hit: ${self.total_pnl:.2f} - stopping")
                            continue

                        open_exposure = sum(t.size_usd for t in self.trades.values() if t.status == "open")
                        if open_exposure >= self.MAX_TOTAL_EXPOSURE:
                            print(f"[RISK] Exposure limit: ${open_exposure:.2f} >= ${self.MAX_TOTAL_EXPOSURE}")
                            continue

                        open_count = sum(1 for t in self.trades.values() if t.status == "open")
                        if open_count >= self.MAX_CONCURRENT_POSITIONS:
                            continue

                        entry_price = up_price if signal.side == "UP" else down_price
                        edge = signal.edge_up if signal.side == "UP" else signal.edge_down

                        bregman_signal = self.bregman.calculate_optimal_trade(
                            model_prob=signal.model_up,
                            market_yes_price=up_price,
                            market_no_price=down_price
                        )

                        features = self.ml.extract_features(signal, bregman_signal, candles)

                        ml_score = self.ml.score_trade(features, signal.side)
                        min_score = self.ml.get_min_score_threshold()

                        if ml_score < min_score:
                            self.ml_rejections += 1
                            print(f"[ML] {asset} Rejected: {signal.side} score={ml_score:.2f} < {min_score:.2f}")
                            continue

                        position_size = self.calculate_position_size(
                            edge=edge if edge else 0.10,
                            kelly_fraction=bregman_signal.kelly_fraction,
                            btc_1h_change=features.btc_1h_change,
                            volatility=features.btc_volatility,
                        )

                        # Directional bias: 200 EMA trend filter
                        # With-trend = full size, counter-trend = 30% less capital
                        trend = self._get_trend_bias(candles, price)
                        with_trend = (trend == "BEARISH" and signal.side == "DOWN") or \
                                     (trend == "BULLISH" and signal.side == "UP")
                        if not with_trend:
                            position_size = round(position_size * self.TREND_BIAS_WEAK / self.TREND_BIAS_STRONG, 2)
                            position_size = max(self.MIN_POSITION_SIZE, position_size)
                            trend_tag = f"COUNTER-TREND({trend})"
                        else:
                            trend_tag = f"WITH-TREND({trend})"

                        # "Strong" signals (edge>20%) underperform "good" (10-20%) in backtest
                        # Reduce overconfident strong signals slightly
                        if signal.strength.value == "strong" and edge and edge > 0.25:
                            position_size = round(position_size * 0.8, 2)
                            position_size = max(self.MIN_POSITION_SIZE, position_size)

                        print(f"[SIZE] {asset} ${position_size:.2f} (Kelly={bregman_signal.kelly_fraction:.0%}, Edge={edge:.1%}, {trend_tag}, Streak={self.consecutive_wins}W/{self.consecutive_losses}L)")

                        success, order_result = await self.execute_trade(
                            market, signal.side, position_size, entry_price
                        )

                        if not success:
                            print(f"[LIVE] {asset} {signal.side} @ ${entry_price:.4f} FAILED: {order_result} - NOT recorded")
                            continue

                        new_trade = LiveTrade(
                            trade_id=trade_key,
                            market_id=market_id,
                            market_title=f"[{asset}] {question[:75]}",
                            side=signal.side,
                            entry_price=entry_price,
                            entry_time=datetime.now(timezone.utc).isoformat(),
                            size_usd=position_size,
                            signal_strength=signal.strength.value,
                            edge_at_entry=edge if edge else 0,
                            kl_divergence=bregman_signal.kl_divergence,
                            kelly_fraction=bregman_signal.kelly_fraction,
                            features=asdict(features),
                            order_id=order_result,
                            execution_error=None,
                            status="open",
                        )
                        self.trades[trade_key] = new_trade
                        print(f"[LIVE] [{asset}] {signal.side} @ ${entry_price:.4f} | Size: ${position_size:.2f} | Edge: {edge:.1%} | Model: {signal.model_up:.0%} UP | ML: {ml_score:.2f} | FILLED")
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
                            if won:
                                self.wins += 1
                                self.consecutive_wins += 1
                                self.consecutive_losses = 0
                            else:
                                self.losses += 1
                                self.consecutive_losses += 1
                                self.consecutive_wins = 0

                            # Compound: update bankroll with PnL
                            self.bankroll = max(self.initial_bankroll * 0.5, self.bankroll + open_trade.pnl)

                            # Update ML
                            self.ml.update_weights(open_trade.features, won)

                            result = "WIN" if won else "LOSS"
                            print(f"[{result}] {open_trade.side} PnL: ${open_trade.pnl:+.2f} | Size: ${open_trade.size_usd:.2f} | Bankroll: ${self.bankroll:.2f} | {open_trade.market_title[:40]}...")

                            # Auto-redeem winnings after each win
                            if won and not self.dry_run:
                                try:
                                    auto_redeem_winnings()
                                except Exception:
                                    pass

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
                            if won:
                                self.wins += 1
                                self.consecutive_wins += 1
                                self.consecutive_losses = 0
                            else:
                                self.losses += 1
                                self.consecutive_losses += 1
                                self.consecutive_wins = 0

                            self.bankroll = max(self.initial_bankroll * 0.5, self.bankroll + open_trade.pnl)
                            self.ml.update_weights(open_trade.features, won)

                            result = "WIN" if won else "LOSS"
                            print(f"[{result}] {open_trade.side} PnL: ${open_trade.pnl:+.2f} | Size: ${open_trade.size_usd:.2f} | Bankroll: ${self.bankroll:.2f} | {open_trade.market_title[:40]}...")

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

        self._save()
        return main_signal

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
        print(f"Position Size: ${self.BASE_POSITION_SIZE} base (${self.MIN_POSITION_SIZE}-${self.MAX_POSITION_SIZE} Kelly-scaled)")
        print(f"ML Optimization: ENABLED")
        print(f"Assets: {', '.join(self.ASSETS.keys())} (XRP dropped - underperforming)")
        print(f"Trend Bias: 200 EMA | With-trend {self.TREND_BIAS_STRONG:.0%} / Counter-trend {self.TREND_BIAS_WEAK:.0%}")
        print(f"Skip Hours (UTC): {sorted(self.SKIP_HOURS_UTC)}")
        print(f"Strong signal dampening: -20% size when edge > 25%")
        print(f"Entry Window: {self.MIN_TIME_REMAINING}-{self.MAX_TIME_REMAINING} min")
        print(f"Scan interval: 60 seconds")
        print("=" * 70)

        last_update = 0
        cycle = 0

        while True:
            try:
                cycle += 1
                now = time.time()

                signal = await self.run_cycle()

                if signal:
                    print(f"[Scan {cycle}] BTC ${signal.current_price:,.2f} | {signal.regime.value} | Model: {signal.model_up:.0%} UP")

                # 10-minute update
                if now - last_update >= 600:
                    self.print_update(signal)
                    last_update = now

                await asyncio.sleep(60)  # 1 minute - faster scanning for optimal entries

            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(60)

        self.print_update(None)
        print(f"Results saved to: {self.OUTPUT_FILE}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Paper trade only")
    args = parser.parse_args()

    trader = TALiveTrader(dry_run=args.dry_run)
    asyncio.run(trader.run())
