"""
Dependency Types and Data Structures for Cross-Market Arbitrage

Based on 2025 AFT research paper identifying 5 profitable dependency types.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import time


class DependencyType(Enum):
    """Types of cross-market dependencies (from AFT paper)."""
    PARENT_CHILD = "parent_child"           # P(X wins state) <= P(X wins national)
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"  # Only one can be true, sum <= 1
    MARGIN_MARKET = "margin_market"         # Same event, different thresholds
    TEMPORAL_IMPLICATION = "temporal"       # "By March" implies "By December"
    COMPLEMENTARY = "complementary"         # A and NOT A, sum = 1
    UNKNOWN = "unknown"                     # Detected by LLM, type unclear


class ConstraintOperator(Enum):
    """Mathematical constraint operators."""
    LESS_THAN_OR_EQUAL = "<="
    GREATER_THAN_OR_EQUAL = ">="
    EQUALS = "="
    SUM_LESS_THAN_OR_EQUAL = "sum<="
    SUM_EQUALS = "sum="


@dataclass
class MarketInfo:
    """Basic market information for dependency checking."""
    condition_id: str
    question: str
    slug: str
    yes_price: float
    no_price: float
    yes_token_id: str
    no_token_id: str
    volume_24h: float = 0.0
    liquidity: float = 0.0
    end_date: Optional[str] = None

    @property
    def midpoint(self) -> float:
        """Get midpoint price."""
        return (self.yes_price + self.no_price) / 2 if self.no_price else self.yes_price


@dataclass
class MarketRelationship:
    """Detected relationship between two markets."""
    market_a_id: str
    market_b_id: str
    market_a_question: str
    market_b_question: str
    dependency_type: DependencyType
    constraint: str                         # e.g., "P(A) >= P(B)" or "P(A) + P(B) <= 1"
    constraint_operator: ConstraintOperator
    confidence: float                       # 0-1 from LLM or 1.0 for rule-based
    detection_method: str                   # "llm", "rule_temporal", "rule_margin", etc.
    created_at: float = field(default_factory=time.time)
    validated: bool = False                 # Has been manually verified
    notes: str = ""

    def check_violation(self, price_a: float, price_b: float) -> tuple[bool, float]:
        """
        Check if the constraint is violated.

        Returns: (is_violated, violation_amount)
        """
        if self.constraint_operator == ConstraintOperator.LESS_THAN_OR_EQUAL:
            # P(A) <= P(B)
            if price_a > price_b:
                return True, price_a - price_b
        elif self.constraint_operator == ConstraintOperator.GREATER_THAN_OR_EQUAL:
            # P(A) >= P(B)
            if price_a < price_b:
                return True, price_b - price_a
        elif self.constraint_operator == ConstraintOperator.SUM_LESS_THAN_OR_EQUAL:
            # P(A) + P(B) <= 1
            total = price_a + price_b
            if total > 1.0:
                return True, total - 1.0
        elif self.constraint_operator == ConstraintOperator.SUM_EQUALS:
            # P(A) + P(B) = 1
            total = price_a + price_b
            if abs(total - 1.0) > 0.02:  # 2% tolerance
                return True, abs(total - 1.0)

        return False, 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "market_a_id": self.market_a_id,
            "market_b_id": self.market_b_id,
            "market_a_question": self.market_a_question,
            "market_b_question": self.market_b_question,
            "dependency_type": self.dependency_type.value,
            "constraint": self.constraint,
            "constraint_operator": self.constraint_operator.value,
            "confidence": self.confidence,
            "detection_method": self.detection_method,
            "created_at": self.created_at,
            "validated": self.validated,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MarketRelationship":
        """Create from dictionary."""
        return cls(
            market_a_id=data["market_a_id"],
            market_b_id=data["market_b_id"],
            market_a_question=data.get("market_a_question", ""),
            market_b_question=data.get("market_b_question", ""),
            dependency_type=DependencyType(data["dependency_type"]),
            constraint=data["constraint"],
            constraint_operator=ConstraintOperator(data["constraint_operator"]),
            confidence=data["confidence"],
            detection_method=data["detection_method"],
            created_at=data.get("created_at", time.time()),
            validated=data.get("validated", False),
            notes=data.get("notes", ""),
        )


@dataclass
class TradeRecommendation:
    """Recommended trade for a cross-market opportunity."""
    market_id: str
    token_id: str
    side: str           # "YES" or "NO"
    size_usd: float
    target_price: float
    rationale: str


@dataclass
class CrossMarketOpportunity:
    """Arbitrage opportunity from market dependency violation."""
    relationship: MarketRelationship
    market_a_price: float
    market_b_price: float
    market_a_vwap: float
    market_b_vwap: float
    violation_amount: float                 # How much constraint is violated
    expected_profit_pct: float              # After fees (should be >= 5%)
    recommended_trades: List[TradeRecommendation]
    execution_risk_score: float             # 0-1 (higher = riskier)
    market_a_liquidity: float
    market_b_liquidity: float
    timestamp: float = field(default_factory=time.time)

    @property
    def is_profitable(self) -> bool:
        """Check if opportunity meets minimum profit threshold (5%)."""
        return self.expected_profit_pct >= 5.0

    @property
    def is_executable(self) -> bool:
        """Check if execution risk is acceptable."""
        return self.execution_risk_score <= 0.3

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "relationship": self.relationship.to_dict(),
            "market_a_price": self.market_a_price,
            "market_b_price": self.market_b_price,
            "market_a_vwap": self.market_a_vwap,
            "market_b_vwap": self.market_b_vwap,
            "violation_amount": self.violation_amount,
            "expected_profit_pct": self.expected_profit_pct,
            "recommended_trades": [
                {
                    "market_id": t.market_id,
                    "token_id": t.token_id,
                    "side": t.side,
                    "size_usd": t.size_usd,
                    "target_price": t.target_price,
                    "rationale": t.rationale,
                }
                for t in self.recommended_trades
            ],
            "execution_risk_score": self.execution_risk_score,
            "market_a_liquidity": self.market_a_liquidity,
            "market_b_liquidity": self.market_b_liquidity,
            "timestamp": self.timestamp,
        }


# Pattern definitions for rule-based detection
TEMPORAL_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]

TEMPORAL_QUARTERS = ["q1", "q2", "q3", "q4"]

THRESHOLD_PATTERNS = [
    r"(?:above|over|exceed|reach|hit)\s*\$?([\d,]+(?:\.\d+)?)",
    r"(?:below|under)\s*\$?([\d,]+(?:\.\d+)?)",
    r"\$?([\d,]+(?:\.\d+)?)\s*(?:or more|or higher|\+)",
]

WINNER_KEYWORDS = ["win", "winner", "champion", "victory", "elected", "nomination"]

EXCLUSIVE_CATEGORIES = {
    "us_president": {
        "keywords": ["president", "presidential", "white house"],
        "must_contain": ["win", "elected", "2024", "2028"],
    },
    "super_bowl": {
        "keywords": ["super bowl", "superbowl"],
        "must_contain": ["win", "champion"],
    },
    "world_cup": {
        "keywords": ["world cup", "fifa"],
        "must_contain": ["win", "champion"],
    },
}
