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
            return 1.0  # Default threshold

        # Adjust threshold based on win rate
        win_rate = len(self.win_features) / total_trades
        if win_rate > 0.6:
            return 0.8  # Lower threshold if winning
        elif win_rate < 0.4:
            return 1.5  # Higher threshold if losing
        return 1.0


class TALiveTrader:
    """Live trades based on TA + Bregman signals with ML optimization."""

    OUTPUT_FILE = Path(__file__).parent / "ta_live_results.json"
    POSITION_SIZE = 10.0  # $10 per trade as requested

    def __init__(self, dry_run: bool = False, bankroll: float = 500.0):
        self.dry_run = dry_run
        self.generator = TASignalGenerator()
        self.bregman = BregmanOptimizer(bankroll=bankroll)
        self.ml = MLOptimizer()
        self.bankroll = bankroll

        self.trades: Dict[str, LiveTrade] = {}
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.signals_count = 0
        self.ml_rejections = 0
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
            "ml_win_features": self.ml.win_features[-100:],  # Keep last 100
            "ml_loss_features": self.ml.loss_features[-100:],
            "ml_weights": self.ml.feature_weights,
        }
        with open(self.OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    async def fetch_data(self):
        """Fetch BTC candles and markets."""
        async with httpx.AsyncClient(timeout=15) as client:
            # Candles from Binance
            r = await client.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "1m", "limit": 240}
            )
            klines = r.json()

            # Current price
            pr = await client.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": "BTCUSDT"}
            )
            btc_price = float(pr.json()["price"])

            # Rate limit
            await asyncio.sleep(1.5)

            # BTC 15m markets
            markets = []
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
                        if "bitcoin" not in title and "btc" not in title:
                            continue
                        for m in event.get("markets", []):
                            if not m.get("closed", True):
                                if not m.get("question"):
                                    m["question"] = event.get("title", "")
                                markets.append(m)
                    if markets:
                        self._market_cache = markets
                else:
                    markets = getattr(self, '_market_cache', [])
            except Exception as e:
                print(f"[API] Error: {e}")
                markets = getattr(self, '_market_cache', [])

        candles = [
            Candle(k[0]/1000, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]))
            for k in klines
        ]

        return candles, btc_price, markets

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

            # Calculate shares
            shares = size / price

            order_args = OrderArgs(
                price=price,
                size=shares,
                side=BUY,
                token_id=token_id,
            )

            print(f"[LIVE] Placing {side} order: {shares:.2f} shares @ ${price:.4f}")

            signed_order = self.executor.client.create_order(order_args)
            response = self.executor.client.post_order(signed_order, OrderType.GTC)

            success = response.get("success", False)
            order_id = response.get("orderID", "")

            if success:
                return True, order_id
            else:
                return False, response.get("errorMsg", "unknown_error")

        except Exception as e:
            return False, str(e)

    # Conviction thresholds - prevent flip-flopping
    MIN_MODEL_CONFIDENCE = 0.65  # Model must be at least 65% confident in direction
    MIN_EDGE = 0.10              # At least 10% edge vs market price
    MAX_ENTRY_PRICE = 0.55       # Don't buy at prices above 55¢ (no edge zone)
    CANDLE_LOOKBACK = 15         # Only use last 15 minutes of price action

    async def run_cycle(self):
        """Run one trading cycle."""
        candles, btc_price, markets = await self.fetch_data()

        if not candles:
            return None

        # IMPORTANT: Only use the most recent 15 minutes of price action
        # Not 4 hours - we're trading 15-minute binary options
        recent_candles = candles[-self.CANDLE_LOOKBACK:]

        # Generate main signal from recent price action only
        main_signal = self.generator.generate_signal(
            market_id="btc_main",
            candles=recent_candles,
            current_price=btc_price,
            market_yes_price=0.5,
            market_no_price=0.5,
            time_remaining_min=10.0
        )

        self.signals_count += 1

        if markets:
            print(f"[Markets] Found {len(markets)} BTC 15m market(s)")

        # Sort markets by time remaining - only trade the NEAREST expiring one
        eligible_markets = []
        for market in markets:
            up_price, down_price = self.get_market_prices(market)
            time_left = self.get_time_remaining(market)
            if up_price is None or down_price is None:
                continue
            if time_left > 15 or time_left < 2:
                continue
            eligible_markets.append((time_left, market, up_price, down_price))

        # Sort by time remaining (ascending) - nearest expiring first
        eligible_markets.sort(key=lambda x: x[0])

        if eligible_markets:
            print(f"[Filter] {len(eligible_markets)} markets in window, trading nearest ({eligible_markets[0][0]:.1f}min left)")

        # Only trade the nearest expiring market for high turnover
        for time_left, market, up_price, down_price in eligible_markets[:1]:
            market_id = market.get("conditionId", "")
            question = market.get("question", "")

            # Generate signal from recent 15-min price action only
            signal = self.generator.generate_signal(
                market_id=market_id,
                candles=recent_candles,
                current_price=btc_price,
                market_yes_price=up_price,
                market_no_price=down_price,
                time_remaining_min=time_left
            )

            trade_key = f"{market_id}_{signal.side}" if signal.side else None

            # === CONVICTION FILTERS ===
            # 1. Model must be confident enough (not flip-flopping near 50%)
            if signal.side == "UP" and signal.model_up < self.MIN_MODEL_CONFIDENCE:
                continue  # Not confident enough for UP
            if signal.side == "DOWN" and signal.model_down < self.MIN_MODEL_CONFIDENCE:
                continue  # Not confident enough for DOWN

            # 2. Minimum edge requirement - don't trade without real edge
            best_edge = signal.edge_up if signal.side == "UP" else signal.edge_down
            if best_edge is not None and best_edge < self.MIN_EDGE:
                continue  # Edge too small

            # 3. Don't buy at prices near 50¢ (no edge zone)
            entry_price_check = up_price if signal.side == "UP" else down_price
            if entry_price_check is not None and entry_price_check > self.MAX_ENTRY_PRICE:
                continue  # Entry price too high, no real edge

            # Enter trade if signal passes all filters
            if signal.action == "ENTER" and signal.side and trade_key:
                if trade_key not in self.trades:
                    entry_price = up_price if signal.side == "UP" else down_price
                    edge = signal.edge_up if signal.side == "UP" else signal.edge_down

                    # Bregman optimization
                    bregman_signal = self.bregman.calculate_optimal_trade(
                        model_prob=signal.model_up,
                        market_yes_price=up_price,
                        market_no_price=down_price
                    )

                    # Extract ML features (use full candles for feature extraction)
                    features = self.ml.extract_features(signal, bregman_signal, candles)

                    # ML scoring
                    ml_score = self.ml.score_trade(features, signal.side)
                    min_score = self.ml.get_min_score_threshold()

                    if ml_score < min_score:
                        self.ml_rejections += 1
                        print(f"[ML] Rejected: {signal.side} score={ml_score:.2f} < {min_score:.2f}")
                        continue

                    # Execute trade
                    success, order_result = await self.execute_trade(
                        market, signal.side, self.POSITION_SIZE, entry_price
                    )

                    trade = LiveTrade(
                        trade_id=trade_key,
                        market_id=market_id,
                        market_title=question[:80],
                        side=signal.side,
                        entry_price=entry_price,
                        entry_time=datetime.now(timezone.utc).isoformat(),
                        size_usd=self.POSITION_SIZE,
                        signal_strength=signal.strength.value,
                        edge_at_entry=edge if edge else 0,
                        kl_divergence=bregman_signal.kl_divergence,
                        kelly_fraction=bregman_signal.kelly_fraction,
                        features=asdict(features),
                        order_id=order_result if success else None,
                        execution_error=None if success else order_result,
                        status="open" if success else "failed",
                    )
                    self.trades[trade_key] = trade

                    mode = "LIVE" if not self.dry_run else "DRY"
                    status = "OK" if success else f"FAILED: {order_result}"
                    print(f"[{mode}] {signal.side} @ ${entry_price:.4f} | Edge: {edge:.1%} | Model: {signal.model_up:.0%} UP | ML: {ml_score:.2f} | {status}")

            # Check for resolved markets
            if time_left < 0.5:
                for tid, trade in list(self.trades.items()):
                    if trade.status == "open" and market_id in tid:
                        price = up_price if trade.side == "UP" else down_price

                        # Determine outcome
                        if price >= 0.95:
                            exit_val = self.POSITION_SIZE / trade.entry_price
                        elif price <= 0.05:
                            exit_val = 0
                        else:
                            exit_val = (self.POSITION_SIZE / trade.entry_price) * price

                        trade.exit_price = price
                        trade.exit_time = datetime.now(timezone.utc).isoformat()
                        trade.pnl = exit_val - self.POSITION_SIZE
                        trade.status = "closed"

                        self.total_pnl += trade.pnl
                        won = trade.pnl > 0
                        if won:
                            self.wins += 1
                        else:
                            self.losses += 1

                        # Update ML
                        self.ml.update_weights(trade.features, won)

                        result = "WIN" if won else "LOSS"
                        print(f"[{result}] {trade.side} PnL: ${trade.pnl:+.2f} | {trade.market_title[:40]}...")

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
        print(f"  Position Size: ${self.POSITION_SIZE}")
        print(f"  Total trades: {len(closed_trades)}")
        print(f"  Win/Loss: {self.wins}/{self.losses}")
        win_rate = (self.wins / max(1, self.wins + self.losses)) * 100
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Total PnL: ${self.total_pnl:+.2f}")

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
        print(f"TA + BREGMAN + ML {mode} TRADER - BTC 15-Minute Markets")
        print("=" * 70)
        print(f"Position Size: ${self.POSITION_SIZE}")
        print(f"ML Optimization: ENABLED")
        print(f"Scan interval: 2 minutes")
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

                await asyncio.sleep(120)  # 2 minutes

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
