"""
Market Relationship Detector

Hybrid approach:
1. Rule-based detection for known patterns (fast, free)
2. LLM-based detection for novel relationships (API-based, no local GPU)
"""

import re
import time
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import httpx

from .dependency_types import (
    DependencyType,
    ConstraintOperator,
    MarketRelationship,
    MarketInfo,
    TEMPORAL_MONTHS,
    TEMPORAL_QUARTERS,
    THRESHOLD_PATTERNS,
    WINNER_KEYWORDS,
    EXCLUSIVE_CATEGORIES,
)


@dataclass
class DetectionConfig:
    """Configuration for relationship detection."""
    use_llm: bool = True
    llm_provider: str = "anthropic"         # "anthropic" or "openai"
    llm_model: str = "claude-3-haiku-20240307"  # Fast and cheap
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    max_pairs_per_batch: int = 10           # For LLM batching
    min_confidence: float = 0.7             # Threshold for accepting LLM results
    timeout_seconds: int = 30


class RelationshipDetector:
    """
    Detect logical relationships between markets.

    Uses a hybrid approach:
    1. Rule-based detection for known patterns (fast, free)
    2. LLM-based detection for novel relationships (slower, costs API)
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for rule-based detection."""
        # Temporal month pattern
        months_pattern = "|".join(TEMPORAL_MONTHS)
        self.month_pattern = re.compile(
            rf"\b(?:by|before|in|during)\s+({months_pattern})\b",
            re.IGNORECASE
        )

        # Year pattern
        self.year_pattern = re.compile(r"\b(202[4-9]|203[0-9])\b")

        # Threshold patterns
        self.threshold_patterns = [
            re.compile(p, re.IGNORECASE) for p in THRESHOLD_PATTERNS
        ]

        # State/region patterns for parent-child
        self.state_pattern = re.compile(
            r"\b(win|wins|winning|carry|carries)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
        )

        # National election pattern
        self.national_pattern = re.compile(
            r"\b(win|wins|winning|elected|become)\s+(?:the\s+)?(?:president|election|presidency)\b",
            re.IGNORECASE
        )

    # =========================================================================
    # RULE-BASED DETECTION (Fast, Free)
    # =========================================================================

    def detect_all_rules(
        self,
        market_a: MarketInfo,
        market_b: MarketInfo
    ) -> Optional[MarketRelationship]:
        """
        Try all rule-based detection methods.

        Returns the first relationship found, or None.
        """
        q1 = market_a.question
        q2 = market_b.question

        # Try temporal implication
        rel = self.detect_temporal_implication(market_a, market_b)
        if rel:
            return rel

        # Try margin market detection
        rel = self.detect_margin_market(market_a, market_b)
        if rel:
            return rel

        # Try parent-child (state vs national)
        rel = self.detect_parent_child(market_a, market_b)
        if rel:
            return rel

        # Try complementary detection
        rel = self.detect_complementary(market_a, market_b)
        if rel:
            return rel

        return None

    def detect_temporal_implication(
        self,
        market_a: MarketInfo,
        market_b: MarketInfo
    ) -> Optional[MarketRelationship]:
        """
        Detect temporal dependencies like "by March" implies "by December".

        If event happens "by March", it must also happen "by December".
        Therefore: P(by_march) <= P(by_december)
        """
        q1 = market_a.question.lower()
        q2 = market_b.question.lower()

        # Extract months
        match1 = self.month_pattern.search(q1)
        match2 = self.month_pattern.search(q2)

        if not (match1 and match2):
            return None

        month1 = match1.group(1).lower()
        month2 = match2.group(1).lower()

        if month1 == month2:
            return None

        # Get month indices
        try:
            idx1 = TEMPORAL_MONTHS.index(month1)
            idx2 = TEMPORAL_MONTHS.index(month2)
        except ValueError:
            return None

        # Check if questions are about the same event (similar text)
        # Remove temporal parts and compare
        q1_base = self.month_pattern.sub("", q1).strip()
        q2_base = self.month_pattern.sub("", q2).strip()

        # Simple similarity check - at least 50% word overlap
        words1 = set(q1_base.split())
        words2 = set(q2_base.split())
        overlap = len(words1 & words2) / max(len(words1), len(words2), 1)

        if overlap < 0.5:
            return None

        # Earlier month implies later month
        if idx1 < idx2:
            # P(market_a) <= P(market_b)
            return MarketRelationship(
                market_a_id=market_a.condition_id,
                market_b_id=market_b.condition_id,
                market_a_question=market_a.question,
                market_b_question=market_b.question,
                dependency_type=DependencyType.TEMPORAL_IMPLICATION,
                constraint=f"P(by {month1}) <= P(by {month2})",
                constraint_operator=ConstraintOperator.LESS_THAN_OR_EQUAL,
                confidence=1.0,
                detection_method="rule_temporal",
            )
        else:
            # P(market_b) <= P(market_a)
            return MarketRelationship(
                market_a_id=market_b.condition_id,
                market_b_id=market_a.condition_id,
                market_a_question=market_b.question,
                market_b_question=market_a.question,
                dependency_type=DependencyType.TEMPORAL_IMPLICATION,
                constraint=f"P(by {month2}) <= P(by {month1})",
                constraint_operator=ConstraintOperator.LESS_THAN_OR_EQUAL,
                confidence=1.0,
                detection_method="rule_temporal",
            )

    def detect_margin_market(
        self,
        market_a: MarketInfo,
        market_b: MarketInfo
    ) -> Optional[MarketRelationship]:
        """
        Detect margin markets (same event, different thresholds).

        E.g., "BTC above $90k" and "BTC above $95k"
        If BTC > $95k, then BTC > $90k.
        Therefore: P($95k) <= P($90k)
        """
        q1 = market_a.question
        q2 = market_b.question

        # Extract thresholds
        threshold1 = self._extract_threshold(q1)
        threshold2 = self._extract_threshold(q2)

        if threshold1 is None or threshold2 is None:
            return None

        if threshold1 == threshold2:
            return None

        # Check if same asset/event
        q1_clean = re.sub(r"[\d,.$]+", "", q1.lower())
        q2_clean = re.sub(r"[\d,.$]+", "", q2.lower())

        words1 = set(q1_clean.split())
        words2 = set(q2_clean.split())
        overlap = len(words1 & words2) / max(len(words1), len(words2), 1)

        if overlap < 0.6:
            return None

        # Check direction (above vs below)
        is_above1 = bool(re.search(r"above|over|exceed|reach|hit", q1, re.IGNORECASE))
        is_above2 = bool(re.search(r"above|over|exceed|reach|hit", q2, re.IGNORECASE))
        is_below1 = bool(re.search(r"below|under", q1, re.IGNORECASE))
        is_below2 = bool(re.search(r"below|under", q2, re.IGNORECASE))

        if is_above1 and is_above2:
            # Higher threshold implies lower threshold
            # P(higher) <= P(lower)
            if threshold1 > threshold2:
                return MarketRelationship(
                    market_a_id=market_a.condition_id,
                    market_b_id=market_b.condition_id,
                    market_a_question=market_a.question,
                    market_b_question=market_b.question,
                    dependency_type=DependencyType.MARGIN_MARKET,
                    constraint=f"P(>{threshold1}) <= P(>{threshold2})",
                    constraint_operator=ConstraintOperator.LESS_THAN_OR_EQUAL,
                    confidence=1.0,
                    detection_method="rule_margin",
                )
            else:
                return MarketRelationship(
                    market_a_id=market_b.condition_id,
                    market_b_id=market_a.condition_id,
                    market_a_question=market_b.question,
                    market_b_question=market_a.question,
                    dependency_type=DependencyType.MARGIN_MARKET,
                    constraint=f"P(>{threshold2}) <= P(>{threshold1})",
                    constraint_operator=ConstraintOperator.LESS_THAN_OR_EQUAL,
                    confidence=1.0,
                    detection_method="rule_margin",
                )

        elif is_below1 and is_below2:
            # Lower threshold implies higher threshold for "below"
            # P(below lower) <= P(below higher)
            if threshold1 < threshold2:
                return MarketRelationship(
                    market_a_id=market_a.condition_id,
                    market_b_id=market_b.condition_id,
                    market_a_question=market_a.question,
                    market_b_question=market_b.question,
                    dependency_type=DependencyType.MARGIN_MARKET,
                    constraint=f"P(<{threshold1}) <= P(<{threshold2})",
                    constraint_operator=ConstraintOperator.LESS_THAN_OR_EQUAL,
                    confidence=1.0,
                    detection_method="rule_margin",
                )
            else:
                return MarketRelationship(
                    market_a_id=market_b.condition_id,
                    market_b_id=market_a.condition_id,
                    market_a_question=market_b.question,
                    market_b_question=market_a.question,
                    dependency_type=DependencyType.MARGIN_MARKET,
                    constraint=f"P(<{threshold2}) <= P(<{threshold1})",
                    constraint_operator=ConstraintOperator.LESS_THAN_OR_EQUAL,
                    confidence=1.0,
                    detection_method="rule_margin",
                )

        return None

    def detect_parent_child(
        self,
        market_a: MarketInfo,
        market_b: MarketInfo
    ) -> Optional[MarketRelationship]:
        """
        Detect parent-child relationships.

        E.g., "X wins Pennsylvania" vs "X wins the election"
        Winning a state doesn't guarantee winning nationally,
        but winning nationally requires winning states.

        This is complex - we look for:
        - Same candidate/subject
        - One is state-level, one is national
        """
        q1 = market_a.question
        q2 = market_b.question

        # Check for state-level match
        state_match1 = self.state_pattern.search(q1)
        state_match2 = self.state_pattern.search(q2)

        # Check for national-level match
        national_match1 = self.national_pattern.search(q1)
        national_match2 = self.national_pattern.search(q2)

        # One should be state, one should be national
        if state_match1 and national_match2:
            # market_a is state, market_b is national
            # P(state) alone doesn't constrain P(national)
            # But we can note the relationship for monitoring
            return None  # Complex relationship - defer to LLM

        if state_match2 and national_match1:
            # market_b is state, market_a is national
            return None  # Complex relationship - defer to LLM

        return None

    def detect_complementary(
        self,
        market_a: MarketInfo,
        market_b: MarketInfo
    ) -> Optional[MarketRelationship]:
        """
        Detect complementary markets (A and NOT A).

        E.g., "Will X happen?" and "Will X NOT happen?"
        P(A) + P(NOT A) = 1
        """
        q1 = market_a.question.lower()
        q2 = market_b.question.lower()

        # Check for negation patterns
        negation_patterns = [
            (r"will (.+) happen", r"will \1 not happen"),
            (r"will (.+)", r"will \1 not"),
            (r"(.+) to win", r"\1 to not win"),
            (r"(.+) yes", r"\1 no"),
        ]

        for pos_pattern, neg_pattern in negation_patterns:
            pos_match = re.search(pos_pattern, q1)
            neg_match = re.search(neg_pattern, q2)

            if pos_match and neg_match:
                return MarketRelationship(
                    market_a_id=market_a.condition_id,
                    market_b_id=market_b.condition_id,
                    market_a_question=market_a.question,
                    market_b_question=market_b.question,
                    dependency_type=DependencyType.COMPLEMENTARY,
                    constraint="P(A) + P(B) = 1",
                    constraint_operator=ConstraintOperator.SUM_EQUALS,
                    confidence=0.9,
                    detection_method="rule_complementary",
                )

        return None

    def detect_mutually_exclusive_group(
        self,
        markets: List[MarketInfo]
    ) -> List[MarketRelationship]:
        """
        Detect mutually exclusive groups (e.g., multiple winner markets).

        Only one can be true, so sum of probabilities <= 1.
        """
        relationships = []

        # Group by category
        for category, config in EXCLUSIVE_CATEGORIES.items():
            group_markets = []

            for market in markets:
                q = market.question.lower()

                # Check keywords
                has_keyword = any(kw in q for kw in config["keywords"])
                has_required = all(req in q for req in config.get("must_contain", []))

                if has_keyword and has_required:
                    group_markets.append(market)

            # Create pairwise relationships for the group
            if len(group_markets) >= 2:
                for i, m1 in enumerate(group_markets):
                    for m2 in group_markets[i+1:]:
                        relationships.append(MarketRelationship(
                            market_a_id=m1.condition_id,
                            market_b_id=m2.condition_id,
                            market_a_question=m1.question,
                            market_b_question=m2.question,
                            dependency_type=DependencyType.MUTUALLY_EXCLUSIVE,
                            constraint=f"P(A) + P(B) <= 1 [{category}]",
                            constraint_operator=ConstraintOperator.SUM_LESS_THAN_OR_EQUAL,
                            confidence=0.95,
                            detection_method=f"rule_exclusive_{category}",
                        ))

        return relationships

    def _extract_threshold(self, text: str) -> Optional[float]:
        """Extract numeric threshold from text."""
        for pattern in self.threshold_patterns:
            match = pattern.search(text)
            if match:
                try:
                    # Remove commas and parse
                    num_str = match.group(1).replace(",", "")
                    return float(num_str)
                except (ValueError, IndexError):
                    continue
        return None

    # =========================================================================
    # LLM-BASED DETECTION (API-based, no local GPU)
    # =========================================================================

    async def detect_with_llm(
        self,
        market_pairs: List[Tuple[MarketInfo, MarketInfo]]
    ) -> List[MarketRelationship]:
        """
        Use LLM to detect relationships between market pairs.

        Uses Claude Haiku for fast, cheap inference.
        """
        if not self.config.use_llm:
            return []

        if not market_pairs:
            return []

        relationships = []

        # Process in batches
        for i in range(0, len(market_pairs), self.config.max_pairs_per_batch):
            batch = market_pairs[i:i + self.config.max_pairs_per_batch]
            batch_results = await self._process_llm_batch(batch)
            relationships.extend(batch_results)

        return relationships

    async def _process_llm_batch(
        self,
        pairs: List[Tuple[MarketInfo, MarketInfo]]
    ) -> List[MarketRelationship]:
        """Process a batch of market pairs with LLM."""
        if self.config.llm_provider == "anthropic":
            return await self._call_anthropic(pairs)
        elif self.config.llm_provider == "openai":
            return await self._call_openai(pairs)
        else:
            return []

    async def _call_anthropic(
        self,
        pairs: List[Tuple[MarketInfo, MarketInfo]]
    ) -> List[MarketRelationship]:
        """Call Anthropic Claude API for relationship detection."""
        if not self.config.anthropic_api_key:
            return []

        relationships = []

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            for market_a, market_b in pairs:
                prompt = self._build_llm_prompt(market_a, market_b)

                try:
                    response = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": self.config.anthropic_api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": self.config.llm_model,
                            "max_tokens": 256,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    )

                    if response.status_code == 200:
                        result = response.json()
                        text = result["content"][0]["text"]
                        rel = self._parse_llm_response(text, market_a, market_b)
                        if rel and rel.confidence >= self.config.min_confidence:
                            relationships.append(rel)

                except Exception as e:
                    print(f"[LLM] Error calling Anthropic: {e}")
                    continue

        return relationships

    async def _call_openai(
        self,
        pairs: List[Tuple[MarketInfo, MarketInfo]]
    ) -> List[MarketRelationship]:
        """Call OpenAI API for relationship detection."""
        if not self.config.openai_api_key:
            return []

        relationships = []

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            for market_a, market_b in pairs:
                prompt = self._build_llm_prompt(market_a, market_b)

                try:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.config.openai_api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "gpt-3.5-turbo",
                            "max_tokens": 256,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    )

                    if response.status_code == 200:
                        result = response.json()
                        text = result["choices"][0]["message"]["content"]
                        rel = self._parse_llm_response(text, market_a, market_b)
                        if rel and rel.confidence >= self.config.min_confidence:
                            relationships.append(rel)

                except Exception as e:
                    print(f"[LLM] Error calling OpenAI: {e}")
                    continue

        return relationships

    def _build_llm_prompt(self, market_a: MarketInfo, market_b: MarketInfo) -> str:
        """Build prompt for LLM relationship detection."""
        return f"""Analyze these two prediction markets for logical dependencies:

Market A: "{market_a.question}"
Market B: "{market_b.question}"

Is there a logical constraint between their probabilities?

Types:
1. PARENT_CHILD: P(A) >= P(B) or P(A) <= P(B) always (one implies the other)
2. MUTUALLY_EXCLUSIVE: P(A) + P(B) <= 1 (both cannot be true)
3. MARGIN: Same event with thresholds (higher implies lower or vice versa)
4. TEMPORAL: Time-based implication (earlier deadline implies later)
5. COMPLEMENTARY: P(A) + P(B) = 1 (one is negation of other)
6. NONE: No logical constraint

Respond in exactly this JSON format:
{{"type": "TYPE_NAME", "constraint": "mathematical constraint", "confidence": 0.0-1.0, "reason": "brief explanation"}}

If no relationship exists, respond:
{{"type": "NONE", "constraint": "", "confidence": 1.0, "reason": "no logical dependency"}}"""

    def _parse_llm_response(
        self,
        text: str,
        market_a: MarketInfo,
        market_b: MarketInfo
    ) -> Optional[MarketRelationship]:
        """Parse LLM response into MarketRelationship."""
        import json

        try:
            # Extract JSON from response
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            data = json.loads(text)

            type_str = data.get("type", "NONE").upper()
            if type_str == "NONE":
                return None

            # Map type string to enum
            type_map = {
                "PARENT_CHILD": DependencyType.PARENT_CHILD,
                "MUTUALLY_EXCLUSIVE": DependencyType.MUTUALLY_EXCLUSIVE,
                "MARGIN": DependencyType.MARGIN_MARKET,
                "MARGIN_MARKET": DependencyType.MARGIN_MARKET,
                "TEMPORAL": DependencyType.TEMPORAL_IMPLICATION,
                "TEMPORAL_IMPLICATION": DependencyType.TEMPORAL_IMPLICATION,
                "COMPLEMENTARY": DependencyType.COMPLEMENTARY,
            }

            dep_type = type_map.get(type_str)
            if not dep_type:
                return None

            # Determine constraint operator from constraint text
            constraint = data.get("constraint", "")
            if "<=" in constraint:
                if "+" in constraint:
                    op = ConstraintOperator.SUM_LESS_THAN_OR_EQUAL
                else:
                    op = ConstraintOperator.LESS_THAN_OR_EQUAL
            elif ">=" in constraint:
                op = ConstraintOperator.GREATER_THAN_OR_EQUAL
            elif "=" in constraint:
                if "+" in constraint:
                    op = ConstraintOperator.SUM_EQUALS
                else:
                    op = ConstraintOperator.EQUALS
            else:
                op = ConstraintOperator.LESS_THAN_OR_EQUAL  # Default

            return MarketRelationship(
                market_a_id=market_a.condition_id,
                market_b_id=market_b.condition_id,
                market_a_question=market_a.question,
                market_b_question=market_b.question,
                dependency_type=dep_type,
                constraint=constraint,
                constraint_operator=op,
                confidence=float(data.get("confidence", 0.5)),
                detection_method="llm",
                notes=data.get("reason", ""),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[LLM] Error parsing response: {e}")
            return None
