"""
ML Trader V2 - Integrated Trading System

Combines all advanced components:
- Enhanced feature engineering
- XGBoost/LightGBM ensemble ML
- Advanced Kelly position sizing
- Arbitrage detection
- Whale copy trading
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional
import httpx

from .config import config
from .features import EnhancedFeatures, feature_extractor, extract_enhanced_features
from .ml_engine_v2 import ml_engine_v2, MLPrediction
from .kelly_v2 import kelly, multi_kelly, calculate_position_size, KellyResult
from .arb_detector import arb_scanner, ArbitrageOpportunity
from .whale_tracker import whale_tracker, WhaleSignal


@dataclass
class TradeV2:
    """An ML-driven trade with enhanced features."""
    id: str
    timestamp: str
    source: str  # "ml", "arbitrage", "whale_copy"
    market: str
    condition_id: str
    token_id: str
    side: str
    entry_price: float
    shares: float
    cost: float
    status: str  # OPEN, WON, LOST
    ml_win_prob: float
    ml_confidence: float
    kelly_fraction: float
    edge: float
    features_snapshot: Dict
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
    closed_at: Optional[str] = None


class MLTraderV2:
    """
    Advanced ML-powered trading system.

    Integrates:
    1. Enhanced feature engineering (31 features)
    2. Ensemble ML model (XGBoost + LightGBM + Online)
    3. Probability calibration
    4. Confidence-adjusted Kelly sizing
    5. Arbitrage detection
    6. Whale copy trading
    """

    def __init__(
        self,
        state_file: str = "ml_trader_v2_state.json",
        bankroll: float = 100.0,
    ):
        self.state_file = state_file
        self.bankroll = bankroll
        self.trades: Dict[str, TradeV2] = {}

        # Components
        self.feature_extractor = feature_extractor
        self.ml_engine = ml_engine_v2
        self.kelly = kelly
        self.multi_kelly = multi_kelly
        self.arb_scanner = arb_scanner
        self.whale_tracker = whale_tracker

        # Settings
        self.min_edge = 0.02
        self.min_confidence = 0.3
        self.max_position_pct = 0.15
        self.max_open_positions = 10

        self._load_state()

    def _load_state(self):
        """Load previous state."""
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                self.bankroll = data.get("bankroll", self.bankroll)
                for trade_data in data.get("trades", []):
                    trade = TradeV2(**trade_data)
                    self.trades[trade.id] = trade
                print(f"[MLTraderV2] Loaded {len(self.trades)} trades, bankroll: ${self.bankroll:.2f}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[MLTraderV2] Error loading state: {e}")

    def _save_state(self):
        """Save current state."""
        try:
            data = {
                "bankroll": self.bankroll,
                "trades": [asdict(t) for t in self.trades.values()],
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[MLTraderV2] Error saving state: {e}")

    async def scan_ml_opportunities(self) -> List[Dict]:
        """Scan for ML-approved opportunities."""
        opportunities = []

        async with httpx.AsyncClient(timeout=30) as client:
            # Get whale positions for copy trading
            whale_signals = await self.whale_tracker.get_copy_signals(
                min_signal_strength=0.4,
                min_whale_pnl=0,
            )

            for signal in whale_signals:
                # Skip if already have position
                if signal.market_id in [t.condition_id for t in self.trades.values() if t.status == "OPEN"]:
                    continue

                try:
                    # Get market details
                    r = await client.get(
                        f"https://gamma-api.polymarket.com/markets/{signal.market_id}"
                    )
                    market_data = r.json()
                except:
                    continue

                # Build position data for feature extraction
                position_data = {
                    "conditionId": signal.market_id,
                    "title": signal.market_title,
                    "outcome": signal.direction,
                    "avgPrice": signal.entry_price,
                    "curPrice": signal.entry_price,
                    "cashPnl": signal.whale_pnl,
                    "endDate": market_data.get("endDate", ""),
                }

                # Extract enhanced features
                features = await self.feature_extractor.extract_features(position_data)

                # Get ML prediction
                prediction = self.ml_engine.predict(features, signal.entry_price)

                # Check if ML approves
                should_trade, reason = self.ml_engine.should_trade(features, signal.entry_price)

                if not should_trade:
                    continue

                # Calculate Kelly position size
                kelly_result = calculate_position_size(
                    win_prob=prediction.calibrated_probability,
                    market_price=signal.entry_price,
                    confidence=prediction.confidence,
                    bankroll=self.bankroll,
                )

                if kelly_result.recommended_size < config.MIN_POSITION_SIZE:
                    continue

                opportunities.append({
                    "source": "ml",
                    "market": signal.market_title,
                    "condition_id": signal.market_id,
                    "token_id": market_data.get("clobTokenIds", [""])[0] if signal.direction == "Yes" else market_data.get("clobTokenIds", ["", ""])[1],
                    "outcome": signal.direction,
                    "entry_price": signal.entry_price,
                    "features": features,
                    "prediction": prediction,
                    "kelly": kelly_result,
                    "whale_signal": signal,
                    "reason": reason,
                })

        # Sort by expected return
        opportunities.sort(
            key=lambda x: x["prediction"].edge * x["kelly"].recommended_size,
            reverse=True
        )

        return opportunities

    async def scan_arbitrage_opportunities(self) -> List[Dict]:
        """Scan for arbitrage opportunities."""
        opportunities = []

        arb_opps = await self.arb_scanner.scan_all()

        for arb in arb_opps[:5]:  # Top 5
            # Skip low confidence
            if arb.confidence < 0.5:
                continue

            # Skip if already have position
            if arb.market_id in [t.condition_id for t in self.trades.values() if t.status == "OPEN"]:
                continue

            opportunities.append({
                "source": "arbitrage",
                "arb_type": arb.arb_type,
                "market": arb.market_title,
                "condition_id": arb.market_id,
                "profit_margin": arb.profit_margin,
                "confidence": arb.confidence,
                "action": arb.action,
                "details": arb.details,
            })

        return opportunities

    async def scan_all_opportunities(self) -> Dict[str, List]:
        """Scan for all types of opportunities."""
        ml_opps = await self.scan_ml_opportunities()
        arb_opps = await self.scan_arbitrage_opportunities()

        return {
            "ml": ml_opps,
            "arbitrage": arb_opps,
        }

    def open_trade(self, opp: Dict) -> TradeV2:
        """Open a new trade."""
        source = opp.get("source", "ml")

        if source == "ml":
            prediction = opp["prediction"]
            features = opp["features"]
            kelly_result = opp["kelly"]

            size_usd = min(
                kelly_result.recommended_size,
                self.bankroll * self.max_position_pct,
                config.MAX_POSITION_SIZE
            )
            size_usd = max(size_usd, config.MIN_POSITION_SIZE)

            entry_price = opp["entry_price"]
            shares = size_usd / entry_price if entry_price > 0 else 0

            trade = TradeV2(
                id=f"MLv2_{opp['condition_id'][:8]}_{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                source=source,
                market=opp["market"],
                condition_id=opp["condition_id"],
                token_id=opp["token_id"],
                side=opp["outcome"],
                entry_price=entry_price,
                shares=shares,
                cost=size_usd,
                status="OPEN",
                ml_win_prob=prediction.calibrated_probability,
                ml_confidence=prediction.confidence,
                kelly_fraction=kelly_result.fractional_kelly,
                edge=prediction.edge,
                features_snapshot=asdict(features) if hasattr(features, '__dict__') else {},
                current_price=entry_price,
                unrealized_pnl=0.0,
            )

        elif source == "arbitrage":
            # Arbitrage trade
            profit_margin = opp["profit_margin"]
            size_usd = min(self.bankroll * 0.05, 50)  # Conservative for arb

            trade = TradeV2(
                id=f"ARB_{opp['condition_id'][:8]}_{int(time.time())}",
                timestamp=datetime.now().isoformat(),
                source=source,
                market=opp["market"],
                condition_id=opp["condition_id"],
                token_id="",
                side=opp.get("action", ""),
                entry_price=0,
                shares=0,
                cost=size_usd,
                status="OPEN",
                ml_win_prob=opp["confidence"],
                ml_confidence=opp["confidence"],
                kelly_fraction=0.05,
                edge=profit_margin,
                features_snapshot=opp.get("details", {}),
                current_price=0,
                unrealized_pnl=0.0,
            )

        else:
            raise ValueError(f"Unknown source: {source}")

        self.trades[trade.id] = trade
        self.bankroll -= trade.cost
        self._save_state()

        print(f"\n[MLTraderV2] OPENED: {trade.side} @ ${trade.entry_price:.3f}")
        print(f"  Market: {trade.market}")
        print(f"  Source: {trade.source}")
        print(f"  ML Win Prob: {trade.ml_win_prob:.0%}")
        print(f"  Confidence: {trade.ml_confidence:.0%}")
        print(f"  Kelly Size: ${trade.cost:.2f} ({trade.kelly_fraction:.0%})")
        print(f"  Edge: {trade.edge:.1%}")

        return trade

    async def update_positions(self):
        """Update all open positions."""
        async with httpx.AsyncClient(timeout=30) as client:
            for trade_id, trade in list(self.trades.items()):
                if trade.status != "OPEN":
                    continue

                if trade.source == "arbitrage":
                    # Arbitrage trades settle differently
                    continue

                try:
                    r = await client.get(
                        f"https://clob.polymarket.com/price",
                        params={"token_id": trade.token_id}
                    )
                    data = r.json()
                    current_price = float(data.get("price", trade.entry_price))

                    trade.current_price = current_price
                    trade.unrealized_pnl = (current_price - trade.entry_price) * trade.shares

                    # Check for settlement
                    if current_price <= 0.001:
                        self._close_trade(trade, won=False)
                    elif current_price >= 0.999:
                        self._close_trade(trade, won=True)

                except:
                    pass

        self._save_state()

    def _close_trade(self, trade: TradeV2, won: bool):
        """Close a trade and update ML engine."""
        if won:
            trade.status = "WON"
            trade.realized_pnl = trade.shares - trade.cost
            self.bankroll += trade.shares
        else:
            trade.status = "LOST"
            trade.realized_pnl = -trade.cost

        trade.closed_at = datetime.now().isoformat()

        # Update ML engine with outcome
        if trade.features_snapshot:
            try:
                features = EnhancedFeatures()
                for key, value in trade.features_snapshot.items():
                    if hasattr(features, key):
                        setattr(features, key, value)
                self.ml_engine.record_outcome(features, won)
            except:
                pass

        # Update whale tracker
        if trade.source == "ml":
            # Would need to track which whale we copied
            pass

        print(f"\n[MLTraderV2] {'WON' if won else 'LOST'}: {trade.market[:40]}")
        print(f"  P&L: ${trade.realized_pnl:+.2f}")
        print(f"  New Bankroll: ${self.bankroll:.2f}")

        self._save_state()

    def get_summary(self) -> Dict:
        """Get trading summary."""
        open_trades = [t for t in self.trades.values() if t.status == "OPEN"]
        won_trades = [t for t in self.trades.values() if t.status == "WON"]
        lost_trades = [t for t in self.trades.values() if t.status == "LOST"]

        total_invested = sum(t.cost for t in open_trades)
        unrealized = sum(t.unrealized_pnl or 0 for t in open_trades)
        realized = sum(t.realized_pnl or 0 for t in won_trades + lost_trades)

        total_closed = len(won_trades) + len(lost_trades)
        win_rate = len(won_trades) / total_closed if total_closed > 0 else 0

        # By source
        by_source = {}
        for source in ["ml", "arbitrage", "whale_copy"]:
            source_trades = [t for t in self.trades.values() if t.source == source]
            source_won = [t for t in source_trades if t.status == "WON"]
            source_lost = [t for t in source_trades if t.status == "LOST"]
            source_closed = len(source_won) + len(source_lost)
            by_source[source] = {
                "total": len(source_trades),
                "open": len([t for t in source_trades if t.status == "OPEN"]),
                "won": len(source_won),
                "lost": len(source_lost),
                "win_rate": len(source_won) / source_closed if source_closed > 0 else 0,
                "pnl": sum(t.realized_pnl or 0 for t in source_won + source_lost),
            }

        return {
            "bankroll": self.bankroll,
            "open_positions": len(open_trades),
            "total_invested": total_invested,
            "won": len(won_trades),
            "lost": len(lost_trades),
            "win_rate": win_rate,
            "unrealized_pnl": unrealized,
            "realized_pnl": realized,
            "total_pnl": unrealized + realized,
            "by_source": by_source,
        }

    def print_status(self):
        """Print current status."""
        summary = self.get_summary()
        ml_stats = self.ml_engine.get_stats()

        print("\n" + "=" * 70)
        print("ML TRADER V2 - ADVANCED SYSTEM")
        print("=" * 70)
        print(f"Bankroll: ${summary['bankroll']:.2f}")
        print(f"Open: {summary['open_positions']} | Won: {summary['won']} | Lost: {summary['lost']}")
        print(f"Win Rate: {summary['win_rate']:.0%}")
        print(f"Invested: ${summary['total_invested']:.2f}")
        print(f"Unrealized P&L: ${summary['unrealized_pnl']:+.2f}")
        print(f"Realized P&L: ${summary['realized_pnl']:+.2f}")
        print(f"TOTAL P&L: ${summary['total_pnl']:+.2f}")

        print("\n--- BY SOURCE ---")
        for source, data in summary["by_source"].items():
            if data["total"] > 0:
                print(f"  {source.upper()}: {data['total']} trades | {data['win_rate']:.0%} win | ${data['pnl']:+.2f}")

        print("\n--- ML ENGINE ---")
        print(f"  Training Samples: {ml_stats.get('training_samples', 0)}")
        print(f"  XGBoost: {'Active' if ml_stats.get('has_xgboost') else 'N/A'}")
        print(f"  LightGBM: {'Active' if ml_stats.get('has_lightgbm') else 'N/A'}")
        if 'brier_score' in ml_stats:
            print(f"  Brier Score: {ml_stats['brier_score']:.4f}")
            print(f"  High Conf Accuracy: {ml_stats.get('high_conf_accuracy', 0):.0%}")

        print("-" * 70)

        if summary['open_positions'] > 0:
            print("\nOPEN POSITIONS:")
            for t in self.trades.values():
                if t.status == "OPEN":
                    pnl = f"${t.unrealized_pnl:+.2f}" if t.unrealized_pnl else "N/A"
                    print(f"  [{t.source}] {t.side} @ ${t.entry_price:.2f} | ML:{t.ml_win_prob:.0%} | {pnl} | {t.market[:30]}")

        print("=" * 70)


async def run_ml_trading_v2(
    bankroll: float = 100.0,
    max_positions: int = 10,
    scan_interval: int = 120,
):
    """Run the advanced ML trading system."""
    trader = MLTraderV2(bankroll=bankroll)

    print("\n" + "=" * 70)
    print("ML TRADER V2 - STARTING")
    print("=" * 70)
    print(f"Bankroll: ${bankroll}")
    print(f"Max Positions: {max_positions}")
    print(f"Scan Interval: {scan_interval}s")
    print("Features: Enhanced (31 features)")
    print("Models: XGBoost + LightGBM + Online Ensemble")
    print("Sizing: Confidence-Adjusted Kelly")
    print("=" * 70 + "\n")

    # Initial training from historical data
    print("[MLTraderV2] Loading historical data for training...")
    try:
        with open('C:/Users/Star/.local/bin/star-polymarket/data/btc_backtest_compact.json', 'r') as f:
            data = json.load(f)

        for trade_data in data.get('whale_trades', []):
            # Create minimal features from historical data
            features = EnhancedFeatures()
            features.entry_price = float(trade_data.get('entry_price', 0) or 0)
            features.current_price = float(trade_data.get('current_price', 0) or 0)
            features.outcome_is_yes = trade_data.get('outcome', '').lower() == 'yes'

            pnl = float(trade_data.get('pnl', 0) or 0)
            trader.ml_engine.record_outcome(features, pnl > 0)

        print(f"[MLTraderV2] Trained on {len(data.get('whale_trades', []))} historical trades")
    except Exception as e:
        print(f"[MLTraderV2] Could not load training data: {e}")

    while True:
        try:
            # Update existing positions
            await trader.update_positions()

            # Check if we can open more positions
            summary = trader.get_summary()
            if summary["open_positions"] < max_positions:
                # Scan for opportunities
                all_opps = await trader.scan_all_opportunities()

                ml_opps = all_opps.get("ml", [])
                arb_opps = all_opps.get("arbitrage", [])

                if ml_opps or arb_opps:
                    print(f"\n[MLTraderV2] Found {len(ml_opps)} ML + {len(arb_opps)} arb opportunities")

                # Prioritize arbitrage (risk-free profit)
                for opp in arb_opps[:1]:
                    if summary["open_positions"] >= max_positions:
                        break
                    if summary["bankroll"] < config.MIN_POSITION_SIZE:
                        break
                    trader.open_trade(opp)
                    summary["open_positions"] += 1

                # Then ML opportunities
                for opp in ml_opps[:2]:
                    if summary["open_positions"] >= max_positions:
                        break
                    if summary["bankroll"] < config.MIN_POSITION_SIZE:
                        break
                    trader.open_trade(opp)
                    summary["open_positions"] += 1

            # Print status
            trader.print_status()

        except Exception as e:
            print(f"[MLTraderV2] Error: {e}")
            import traceback
            traceback.print_exc()

        await asyncio.sleep(scan_interval)


if __name__ == "__main__":
    asyncio.run(run_ml_trading_v2(bankroll=100.0, max_positions=10, scan_interval=120))
