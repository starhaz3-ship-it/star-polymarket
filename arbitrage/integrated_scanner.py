"""Integrated scanner combining all detection methods."""
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import logging

from .kalshi_arb import CrossPlatformArbitrage, ArbOpportunity
from .combinatorial_arb import CombinatorialArbDetector, CombinatorialOpportunity
from .brier_tracker import BrierTracker
from .news_analyzer import NewsAnalyzer, MarketNewsAnalysis
from .whale_tracker_v2 import EnhancedWhaleTracker, ConsensusSignal
from .cross_market import CrossMarketScanner, CrossMarketOpportunity

logger = logging.getLogger(__name__)


@dataclass
class IntegratedSignal:
    """Combined signal from multiple detection methods."""
    market_id: str
    market_question: str
    recommended_side: str
    confidence: float
    expected_edge: float
    sources: list[str]  # Which methods detected this
    details: dict
    kelly_fraction: float
    suggested_size_usd: float
    timestamp: datetime


class IntegratedScanner:
    """Combine all detection methods for optimal signal generation."""

    def __init__(
        self,
        kalshi_key: Optional[str] = None,
        news_api_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        polygonscan_key: Optional[str] = None,
        bankroll: float = 100.0,
    ):
        self.cross_platform = CrossPlatformArbitrage(kalshi_key)
        self.combinatorial = CombinatorialArbDetector()
        self.brier = BrierTracker()
        self.news = NewsAnalyzer(news_api_key, anthropic_key)
        self.whales = EnhancedWhaleTracker(polygonscan_key)
        self.cross_market = CrossMarketScanner(anthropic_api_key=anthropic_key)
        self.bankroll = bankroll
        self.signals: list[IntegratedSignal] = []

    async def full_scan(self) -> list[IntegratedSignal]:
        """Run all detection methods and combine results."""
        self.signals = []
        market_signals = {}  # market_id -> list of signals

        print("=" * 70)
        print("INTEGRATED MARKET SCANNER")
        print("=" * 70)
        print()

        # 1. Cross-Platform Arbitrage
        print("1. Scanning cross-platform arbitrage...")
        try:
            cross_opps = await self.cross_platform.scan_for_arbitrage(min_spread=0.02)
            for opp in cross_opps:
                key = opp.poly_token_id
                if key not in market_signals:
                    market_signals[key] = {
                        "question": opp.market_name,
                        "signals": [],
                    }
                market_signals[key]["signals"].append({
                    "source": "cross_platform",
                    "side": "YES" if "yes" in opp.strategy else "NO",
                    "edge": opp.expected_profit_pct / 100,
                    "confidence": 0.95,  # Arbitrage is high confidence
                    "details": {"strategy": opp.strategy, "spread": opp.spread},
                })
            print(f"   Found {len(cross_opps)} cross-platform opportunities")
        except Exception as e:
            logger.warning(f"Cross-platform scan error: {e}")
            print(f"   Error: {e}")

        # 2. Combinatorial Arbitrage
        print("2. Scanning combinatorial arbitrage...")
        try:
            combo_opps = await self.combinatorial.scan_for_arbitrage()
            for opp in combo_opps:
                for market in opp.markets[:1]:  # Take first market as representative
                    tokens = market.get("clobTokenIds", "[]")
                    if isinstance(tokens, str):
                        tokens = json.loads(tokens)
                    key = tokens[0] if tokens else ""

                    if key and key not in market_signals:
                        market_signals[key] = {
                            "question": market.get("question", ""),
                            "signals": [],
                        }

                    if key:
                        market_signals[key]["signals"].append({
                            "source": "combinatorial",
                            "side": "YES" if opp.arbitrage_type == "under_100" else "NO",
                            "edge": opp.profit_potential,
                            "confidence": 0.90,
                            "details": {
                                "type": opp.arbitrage_type,
                                "category": opp.category,
                                "total_prob": opp.total_probability,
                            },
                        })
            print(f"   Found {len(combo_opps)} combinatorial opportunities")
        except Exception as e:
            logger.warning(f"Combinatorial scan error: {e}")
            print(f"   Error: {e}")

        # 3. Whale Consensus
        print("3. Scanning whale consensus...")
        try:
            await self.whales.scan_all_whales()
            consensus = self.whales.detect_consensus(min_whales=2, min_total_size=500)
            for signal in consensus:
                key = signal.market_id
                if key and key not in market_signals:
                    market_signals[key] = {
                        "question": signal.market_question,
                        "signals": [],
                    }

                if key:
                    market_signals[key]["signals"].append({
                        "source": "whale_consensus",
                        "side": signal.side,
                        "edge": 0.05,  # Estimated edge from whale following
                        "confidence": signal.weighted_confidence,
                        "details": {
                            "whale_count": signal.whale_count,
                            "total_size": signal.total_size,
                            "whales": signal.whales,
                        },
                    })
            print(f"   Found {len(consensus)} whale consensus signals")
        except Exception as e:
            logger.warning(f"Whale scan error: {e}")
            print(f"   Error: {e}")

        # 4. Cross-Market Dependency Arbitrage (AFT 2025 paper)
        print("4. Scanning cross-market dependencies...")
        try:
            cross_market_opps = await self.cross_market.scan()
            for opp in cross_market_opps:
                # Add signals for both markets in the relationship
                for market_id, side, price in [
                    (opp.relationship.market_a_id, "YES" if opp.recommended_trades and opp.recommended_trades[0].side == "YES" else "NO", opp.market_a_price),
                    (opp.relationship.market_b_id, "YES" if len(opp.recommended_trades) > 1 and opp.recommended_trades[1].side == "YES" else "NO", opp.market_b_price),
                ]:
                    if market_id and market_id not in market_signals:
                        market_signals[market_id] = {
                            "question": opp.relationship.market_a_question if market_id == opp.relationship.market_a_id else opp.relationship.market_b_question,
                            "signals": [],
                        }
                    if market_id and opp.recommended_trades:
                        trade = next((t for t in opp.recommended_trades if t.market_id == market_id), None)
                        if trade:
                            market_signals[market_id]["signals"].append({
                                "source": "cross_market",
                                "side": trade.side,
                                "edge": opp.expected_profit_pct / 100,
                                "confidence": opp.relationship.confidence * (1 - opp.execution_risk_score),
                                "details": {
                                    "dependency_type": opp.relationship.dependency_type.value,
                                    "constraint": opp.relationship.constraint,
                                    "violation_amount": opp.violation_amount,
                                    "execution_risk": opp.execution_risk_score,
                                    "rationale": trade.rationale,
                                },
                            })
            print(f"   Found {len(cross_market_opps)} cross-market opportunities")
        except Exception as e:
            logger.warning(f"Cross-market scan error: {e}")
            print(f"   Error: {e}")

        # 5. Combine signals and generate recommendations
        print("\n5. Combining signals...")

        for market_id, data in market_signals.items():
            if not data["signals"]:
                continue

            # Aggregate signals
            yes_signals = [s for s in data["signals"] if s["side"] == "YES"]
            no_signals = [s for s in data["signals"] if s["side"] == "NO"]

            # Determine dominant side
            yes_strength = sum(s["edge"] * s["confidence"] for s in yes_signals)
            no_strength = sum(s["edge"] * s["confidence"] for s in no_signals)

            if yes_strength > no_strength:
                side = "YES"
                signals = yes_signals
                strength = yes_strength
            else:
                side = "NO"
                signals = no_signals
                strength = no_strength

            if not signals:
                continue

            # Calculate combined metrics
            avg_edge = sum(s["edge"] for s in signals) / len(signals)
            avg_confidence = sum(s["confidence"] for s in signals) / len(signals)
            sources = list(set(s["source"] for s in signals))

            # Apply Brier-based Kelly adjustment
            kelly_adjustment = self.brier.get_kelly_adjustment()

            # Kelly fraction: edge / odds
            # For binary market: f = (p*b - q) / b where b = 1/price - 1
            # Simplified: f = edge * confidence * adjustment
            kelly_fraction = avg_edge * avg_confidence * kelly_adjustment * 0.5  # Half-Kelly

            suggested_size = self.bankroll * kelly_fraction

            signal = IntegratedSignal(
                market_id=market_id,
                market_question=data["question"],
                recommended_side=side,
                confidence=avg_confidence,
                expected_edge=avg_edge,
                sources=sources,
                details={s["source"]: s["details"] for s in signals},
                kelly_fraction=kelly_fraction,
                suggested_size_usd=min(suggested_size, self.bankroll * 0.1),  # Max 10% per trade
                timestamp=datetime.now(timezone.utc),
            )
            self.signals.append(signal)

        # Sort by expected edge * confidence
        self.signals.sort(key=lambda x: -(x.expected_edge * x.confidence))

        print(f"\nGenerated {len(self.signals)} integrated signals")

        return self.signals

    def print_signals(self, top_n: int = 10):
        """Print top signals in a readable format."""
        print()
        print("=" * 70)
        print("TOP TRADING SIGNALS")
        print("=" * 70)
        print()

        for i, sig in enumerate(self.signals[:top_n], 1):
            print(f"{i}. {sig.market_question[:55]}")
            print(f"   Side: {sig.recommended_side}")
            print(f"   Edge: {sig.expected_edge*100:.1f}% | Confidence: {sig.confidence*100:.0f}%")
            print(f"   Sources: {', '.join(sig.sources)}")
            print(f"   Kelly: {sig.kelly_fraction*100:.1f}% | Suggested: ${sig.suggested_size_usd:.2f}")
            print()

    async def record_trade(self, signal: IntegratedSignal, actual_entry_price: float):
        """Record a trade for Brier tracking."""
        self.brier.record_prediction(
            market_id=signal.market_id,
            market_question=signal.market_question,
            predicted_side=signal.recommended_side,
            confidence=signal.confidence,
            entry_price=actual_entry_price,
        )

    def get_calibration_report(self):
        """Print Brier calibration report."""
        self.brier.print_report()


async def main():
    """Run integrated scanner."""
    scanner = IntegratedScanner(bankroll=100)

    # Run full scan
    signals = await scanner.full_scan()

    # Print results
    scanner.print_signals()

    # Show calibration
    print()
    scanner.get_calibration_report()


if __name__ == "__main__":
    asyncio.run(main())
