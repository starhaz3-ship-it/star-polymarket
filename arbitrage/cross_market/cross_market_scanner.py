"""
Cross-Market Dependency Scanner

Main orchestrator for detecting and exploiting cross-market arbitrage opportunities.
Based on 2025 AFT paper: $94K+ extracted from logical dependencies.
"""

import asyncio
import time
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import httpx

from .dependency_types import (
    DependencyType,
    MarketRelationship,
    CrossMarketOpportunity,
    TradeRecommendation,
    MarketInfo,
    ConstraintOperator,
)
from .relationship_detector import RelationshipDetector
from .vwap_calculator import VWAPManager
from .dependency_cache import DependencyCache


@dataclass
class ScanConfig:
    """Configuration for cross-market scanning."""
    min_profit_threshold: float = 0.05      # 5% minimum profit (from paper)
    max_execution_risk: float = 0.3         # Max acceptable risk score
    min_liquidity_usd: float = 1000         # Min liquidity per market
    min_confidence: float = 0.7             # Min relationship confidence
    vwap_window_seconds: int = 3600         # 1-hour VWAP window
    use_llm_detection: bool = True          # Use LLM for novel relationships
    max_llm_calls_per_scan: int = 10        # Limit LLM API costs
    scan_interval_seconds: int = 60         # Time between scans
    cache_ttl_hours: int = 168              # 1 week relationship cache


class CrossMarketScanner:
    """
    Scanner for cross-market arbitrage opportunities.

    Workflow:
    1. Fetch active markets from Polymarket API
    2. Check cached relationships
    3. Detect new relationships (rule-based, then LLM)
    4. Calculate VWAPs for related markets
    5. Check constraint violations
    6. Return opportunities above threshold
    """

    GAMMA_API_URL = "https://gamma-api.polymarket.com"
    CLOB_API_URL = "https://clob.polymarket.com"

    def __init__(
        self,
        config: Optional[ScanConfig] = None,
        db_path: str = "market_dependencies.db",
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.config = config or ScanConfig()
        self.cache = DependencyCache(db_path)
        self.detector = RelationshipDetector(
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key,
        )
        self.vwap_manager = VWAPManager(self.config.vwap_window_seconds)
        self._markets_cache: Dict[str, MarketInfo] = {}
        self._last_market_fetch: float = 0
        self._market_fetch_interval: int = 300  # 5 minutes

    async def scan(self) -> List[CrossMarketOpportunity]:
        """
        Perform a full scan for cross-market opportunities.

        Returns:
            List of profitable opportunities sorted by expected profit
        """
        opportunities = []

        # 1. Fetch active markets
        markets = await self._fetch_active_markets()
        if len(markets) < 2:
            return []

        # 2. Get all cached relationships
        cached_relationships = self.cache.get_all_relationships(
            min_confidence=self.config.min_confidence
        )

        # 3. Check cached relationships for violations
        for rel in cached_relationships:
            market_a = self._markets_cache.get(rel.market_a_id)
            market_b = self._markets_cache.get(rel.market_b_id)

            if not market_a or not market_b:
                continue

            opp = await self._check_opportunity(rel, market_a, market_b)
            if opp and opp.is_profitable:
                opportunities.append(opp)

        # 4. Detect new relationships for unscanned pairs
        market_ids = list(self._markets_cache.keys())
        unscanned_pairs = self.cache.get_unscanned_pairs(
            market_ids,
            max_age_hours=self.config.cache_ttl_hours
        )

        # Limit pairs to scan (prioritize high-volume markets)
        sorted_markets = sorted(
            markets,
            key=lambda m: m.volume_24h,
            reverse=True
        )
        high_volume_ids = {m.condition_id for m in sorted_markets[:100]}

        priority_pairs = [
            (a, b) for a, b in unscanned_pairs
            if a in high_volume_ids or b in high_volume_ids
        ][:50]  # Limit to 50 pairs per scan

        # 5. Detect relationships for priority pairs
        llm_calls_remaining = self.config.max_llm_calls_per_scan
        new_relationships = []

        for market_a_id, market_b_id in priority_pairs:
            market_a = self._markets_cache.get(market_a_id)
            market_b = self._markets_cache.get(market_b_id)

            if not market_a or not market_b:
                self.cache.mark_pair_scanned(market_a_id, market_b_id, False)
                continue

            # Try rule-based detection first
            rel = self.detector.detect_relationship(market_a, market_b)

            # If no rule-based match and LLM enabled, try LLM
            if not rel and self.config.use_llm_detection and llm_calls_remaining > 0:
                rel = await self.detector.detect_with_llm(market_a, market_b)
                llm_calls_remaining -= 1

            if rel:
                new_relationships.append(rel)
                self.cache.save_relationship(rel)
                self.cache.mark_pair_scanned(market_a_id, market_b_id, True)

                # Check for opportunity
                opp = await self._check_opportunity(rel, market_a, market_b)
                if opp and opp.is_profitable:
                    opportunities.append(opp)
            else:
                self.cache.mark_pair_scanned(market_a_id, market_b_id, False)

        # 6. Sort by expected profit
        opportunities.sort(key=lambda o: o.expected_profit_pct, reverse=True)

        return opportunities

    async def scan_specific_markets(
        self,
        market_ids: List[str]
    ) -> List[CrossMarketOpportunity]:
        """
        Scan specific markets for cross-market opportunities.

        Useful for targeted scanning after market discovery.
        """
        opportunities = []

        # Fetch market info for specified IDs
        markets = await self._fetch_markets_by_ids(market_ids)
        if len(markets) < 2:
            return []

        # Check all pairs
        for i, market_a in enumerate(markets):
            for market_b in markets[i+1:]:
                # Check cache first
                cached_rel = self.cache.get_relationship(
                    market_a.condition_id,
                    market_b.condition_id
                )

                if cached_rel:
                    rel = cached_rel
                else:
                    # Detect relationship
                    rel = self.detector.detect_relationship(market_a, market_b)
                    if not rel and self.config.use_llm_detection:
                        rel = await self.detector.detect_with_llm(market_a, market_b)

                    if rel:
                        self.cache.save_relationship(rel)

                if rel:
                    opp = await self._check_opportunity(rel, market_a, market_b)
                    if opp and opp.is_profitable:
                        opportunities.append(opp)

        opportunities.sort(key=lambda o: o.expected_profit_pct, reverse=True)
        return opportunities

    async def _check_opportunity(
        self,
        relationship: MarketRelationship,
        market_a: MarketInfo,
        market_b: MarketInfo,
    ) -> Optional[CrossMarketOpportunity]:
        """Check if a relationship has a profitable violation."""

        # Get VWAPs (with fallback to spot)
        await self.vwap_manager.ensure_initialized([
            market_a.yes_token_id,
            market_b.yes_token_id,
        ])

        vwap_a, _ = self.vwap_manager.get_market_vwaps(
            market_a.yes_token_id,
            market_a.no_token_id,
            market_a.yes_price,
            market_a.no_price,
        )
        vwap_b, _ = self.vwap_manager.get_market_vwaps(
            market_b.yes_token_id,
            market_b.no_token_id,
            market_b.yes_price,
            market_b.no_price,
        )

        # Check constraint violation
        is_violated, violation_amount = relationship.check_violation(vwap_a, vwap_b)

        if not is_violated or violation_amount < 0.02:  # Minimum 2% violation
            return None

        # Calculate expected profit (after ~2% fees per leg)
        fee_rate = 0.02
        expected_profit_pct = (violation_amount - 2 * fee_rate) * 100

        if expected_profit_pct < self.config.min_profit_threshold * 100:
            return None

        # Calculate execution risk
        min_liquidity = min(market_a.liquidity, market_b.liquidity)
        execution_risk = self._calculate_execution_risk(
            violation_amount,
            min_liquidity,
            relationship.confidence,
        )

        # Generate trade recommendations
        trades = self._generate_trade_recommendations(
            relationship,
            market_a,
            market_b,
            vwap_a,
            vwap_b,
            violation_amount,
        )

        return CrossMarketOpportunity(
            relationship=relationship,
            market_a_price=market_a.yes_price,
            market_b_price=market_b.yes_price,
            market_a_vwap=vwap_a,
            market_b_vwap=vwap_b,
            violation_amount=violation_amount,
            expected_profit_pct=expected_profit_pct,
            recommended_trades=trades,
            execution_risk_score=execution_risk,
            market_a_liquidity=market_a.liquidity,
            market_b_liquidity=market_b.liquidity,
        )

    def _calculate_execution_risk(
        self,
        violation_amount: float,
        min_liquidity: float,
        confidence: float,
    ) -> float:
        """
        Calculate execution risk score (0-1, higher = riskier).

        Factors:
        - Small violations more likely to disappear during execution
        - Low liquidity increases slippage risk
        - Lower confidence relationships may be incorrect
        """
        # Violation risk: smaller violations are riskier
        violation_risk = max(0, 0.5 - violation_amount * 2)

        # Liquidity risk
        if min_liquidity < 500:
            liquidity_risk = 0.5
        elif min_liquidity < 2000:
            liquidity_risk = 0.3
        elif min_liquidity < 10000:
            liquidity_risk = 0.1
        else:
            liquidity_risk = 0.0

        # Confidence risk
        confidence_risk = (1 - confidence) * 0.5

        # Combined risk (weighted average)
        total_risk = (
            violation_risk * 0.4 +
            liquidity_risk * 0.4 +
            confidence_risk * 0.2
        )

        return min(1.0, total_risk)

    def _generate_trade_recommendations(
        self,
        relationship: MarketRelationship,
        market_a: MarketInfo,
        market_b: MarketInfo,
        vwap_a: float,
        vwap_b: float,
        violation_amount: float,
    ) -> List[TradeRecommendation]:
        """Generate recommended trades to exploit the violation."""
        trades = []

        # Base trade size (scale with liquidity)
        min_liquidity = min(market_a.liquidity, market_b.liquidity)
        base_size = min(100, min_liquidity * 0.1)  # Max 10% of liquidity

        if relationship.constraint_operator == ConstraintOperator.LESS_THAN_OR_EQUAL:
            # P(A) should be <= P(B), but A > B
            # Strategy: Sell A (overpriced), Buy B (underpriced)
            if vwap_a > vwap_b:
                trades.append(TradeRecommendation(
                    market_id=market_a.condition_id,
                    token_id=market_a.no_token_id,
                    side="NO",
                    size_usd=base_size,
                    target_price=1 - vwap_a,
                    rationale=f"Sell YES (buy NO) on overpriced market A",
                ))
                trades.append(TradeRecommendation(
                    market_id=market_b.condition_id,
                    token_id=market_b.yes_token_id,
                    side="YES",
                    size_usd=base_size,
                    target_price=vwap_b,
                    rationale=f"Buy YES on underpriced market B",
                ))

        elif relationship.constraint_operator == ConstraintOperator.GREATER_THAN_OR_EQUAL:
            # P(A) should be >= P(B), but A < B
            # Strategy: Buy A (underpriced), Sell B (overpriced)
            if vwap_a < vwap_b:
                trades.append(TradeRecommendation(
                    market_id=market_a.condition_id,
                    token_id=market_a.yes_token_id,
                    side="YES",
                    size_usd=base_size,
                    target_price=vwap_a,
                    rationale=f"Buy YES on underpriced market A",
                ))
                trades.append(TradeRecommendation(
                    market_id=market_b.condition_id,
                    token_id=market_b.no_token_id,
                    side="NO",
                    size_usd=base_size,
                    target_price=1 - vwap_b,
                    rationale=f"Sell YES (buy NO) on overpriced market B",
                ))

        elif relationship.constraint_operator == ConstraintOperator.SUM_LESS_THAN_OR_EQUAL:
            # P(A) + P(B) should be <= 1
            # Both are overpriced - sell both
            if vwap_a + vwap_b > 1.0:
                trades.append(TradeRecommendation(
                    market_id=market_a.condition_id,
                    token_id=market_a.no_token_id,
                    side="NO",
                    size_usd=base_size / 2,
                    target_price=1 - vwap_a,
                    rationale=f"Sell YES on mutually exclusive market A",
                ))
                trades.append(TradeRecommendation(
                    market_id=market_b.condition_id,
                    token_id=market_b.no_token_id,
                    side="NO",
                    size_usd=base_size / 2,
                    target_price=1 - vwap_b,
                    rationale=f"Sell YES on mutually exclusive market B",
                ))

        elif relationship.constraint_operator == ConstraintOperator.SUM_EQUALS:
            # P(A) + P(B) should = 1 (complementary)
            total = vwap_a + vwap_b
            if total > 1.02:  # Both overpriced
                trades.append(TradeRecommendation(
                    market_id=market_a.condition_id,
                    token_id=market_a.no_token_id,
                    side="NO",
                    size_usd=base_size / 2,
                    target_price=1 - vwap_a,
                    rationale=f"Sell YES on overpriced complementary market A",
                ))
                trades.append(TradeRecommendation(
                    market_id=market_b.condition_id,
                    token_id=market_b.no_token_id,
                    side="NO",
                    size_usd=base_size / 2,
                    target_price=1 - vwap_b,
                    rationale=f"Sell YES on overpriced complementary market B",
                ))
            elif total < 0.98:  # Both underpriced
                trades.append(TradeRecommendation(
                    market_id=market_a.condition_id,
                    token_id=market_a.yes_token_id,
                    side="YES",
                    size_usd=base_size / 2,
                    target_price=vwap_a,
                    rationale=f"Buy YES on underpriced complementary market A",
                ))
                trades.append(TradeRecommendation(
                    market_id=market_b.condition_id,
                    token_id=market_b.yes_token_id,
                    side="YES",
                    size_usd=base_size / 2,
                    target_price=vwap_b,
                    rationale=f"Buy YES on underpriced complementary market B",
                ))

        # Order by liquidity (execute most liquid first)
        if len(trades) == 2:
            if market_b.liquidity > market_a.liquidity:
                trades = [trades[1], trades[0]]

        return trades

    async def _fetch_active_markets(self) -> List[MarketInfo]:
        """Fetch active markets from Polymarket API."""
        now = time.time()

        # Use cache if recent
        if (
            self._markets_cache
            and now - self._last_market_fetch < self._market_fetch_interval
        ):
            return list(self._markets_cache.values())

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                # Fetch from gamma-api
                response = await client.get(
                    f"{self.GAMMA_API_URL}/markets",
                    params={
                        "closed": "false",
                        "limit": 500,
                    }
                )

                if response.status_code != 200:
                    return list(self._markets_cache.values())

                markets_data = response.json()
                markets = []

                for m in markets_data:
                    try:
                        # Extract token info
                        tokens = m.get("tokens", [])
                        yes_token = next(
                            (t for t in tokens if t.get("outcome") == "Yes"),
                            None
                        )
                        no_token = next(
                            (t for t in tokens if t.get("outcome") == "No"),
                            None
                        )

                        if not yes_token or not no_token:
                            continue

                        market_info = MarketInfo(
                            condition_id=m.get("conditionId", ""),
                            question=m.get("question", ""),
                            slug=m.get("slug", ""),
                            yes_price=float(yes_token.get("price", 0.5)),
                            no_price=float(no_token.get("price", 0.5)),
                            yes_token_id=yes_token.get("token_id", ""),
                            no_token_id=no_token.get("token_id", ""),
                            volume_24h=float(m.get("volume24hr", 0)),
                            liquidity=float(m.get("liquidity", 0)),
                            end_date=m.get("endDate"),
                        )

                        if market_info.condition_id:
                            markets.append(market_info)
                            self._markets_cache[market_info.condition_id] = market_info

                    except (ValueError, TypeError, KeyError):
                        continue

                self._last_market_fetch = now
                return markets

            except Exception as e:
                print(f"[CrossMarketScanner] Error fetching markets: {e}")
                return list(self._markets_cache.values())

    async def _fetch_markets_by_ids(
        self,
        market_ids: List[str]
    ) -> List[MarketInfo]:
        """Fetch specific markets by condition ID."""
        # Check cache first
        cached = [
            self._markets_cache[mid]
            for mid in market_ids
            if mid in self._markets_cache
        ]

        missing_ids = [
            mid for mid in market_ids
            if mid not in self._markets_cache
        ]

        if not missing_ids:
            return cached

        # Fetch missing markets
        async with httpx.AsyncClient(timeout=30) as client:
            markets = list(cached)

            for market_id in missing_ids:
                try:
                    response = await client.get(
                        f"{self.GAMMA_API_URL}/markets/{market_id}"
                    )

                    if response.status_code != 200:
                        continue

                    m = response.json()
                    tokens = m.get("tokens", [])
                    yes_token = next(
                        (t for t in tokens if t.get("outcome") == "Yes"),
                        None
                    )
                    no_token = next(
                        (t for t in tokens if t.get("outcome") == "No"),
                        None
                    )

                    if not yes_token or not no_token:
                        continue

                    market_info = MarketInfo(
                        condition_id=m.get("conditionId", ""),
                        question=m.get("question", ""),
                        slug=m.get("slug", ""),
                        yes_price=float(yes_token.get("price", 0.5)),
                        no_price=float(no_token.get("price", 0.5)),
                        yes_token_id=yes_token.get("token_id", ""),
                        no_token_id=no_token.get("token_id", ""),
                        volume_24h=float(m.get("volume24hr", 0)),
                        liquidity=float(m.get("liquidity", 0)),
                        end_date=m.get("endDate"),
                    )

                    if market_info.condition_id:
                        markets.append(market_info)
                        self._markets_cache[market_info.condition_id] = market_info

                except Exception:
                    continue

            return markets

    def get_statistics(self) -> dict:
        """Get scanner statistics."""
        cache_stats = self.cache.get_statistics()
        vwap_stats = self.vwap_manager.calculator.get_statistics()

        return {
            "cache": cache_stats,
            "vwap": vwap_stats,
            "markets_cached": len(self._markets_cache),
            "config": {
                "min_profit_threshold": self.config.min_profit_threshold,
                "max_execution_risk": self.config.max_execution_risk,
                "use_llm_detection": self.config.use_llm_detection,
            },
        }

    async def run_continuous(
        self,
        callback=None,
        max_iterations: Optional[int] = None
    ):
        """
        Run continuous scanning loop.

        Args:
            callback: Function to call with opportunities
            max_iterations: Max iterations (None for infinite)
        """
        iterations = 0

        while max_iterations is None or iterations < max_iterations:
            try:
                opportunities = await self.scan()

                if opportunities:
                    print(f"[CrossMarketScanner] Found {len(opportunities)} opportunities")
                    for opp in opportunities:
                        print(f"  - {opp.relationship.dependency_type.value}: "
                              f"{opp.expected_profit_pct:.1f}% profit, "
                              f"risk={opp.execution_risk_score:.2f}")

                    if callback:
                        await callback(opportunities)

                # Prune old VWAP data periodically
                self.vwap_manager.prune()

            except Exception as e:
                print(f"[CrossMarketScanner] Scan error: {e}")

            iterations += 1
            await asyncio.sleep(self.config.scan_interval_seconds)
