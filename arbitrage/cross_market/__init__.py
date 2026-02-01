"""
Cross-Market Dependency Detector Module

Based on 2025 AFT research paper findings:
- $39.6M extracted via arbitrage (Apr 2024 - Apr 2025)
- $94K+ from cross-market logical dependencies
- 5 dependency types: parent-child, mutually exclusive, margin, temporal, complementary
"""

from .dependency_types import (
    DependencyType,
    MarketRelationship,
    CrossMarketOpportunity,
)
from .relationship_detector import RelationshipDetector
from .vwap_calculator import VWAPCalculator
from .dependency_cache import DependencyCache
from .cross_market_scanner import CrossMarketScanner

__all__ = [
    "DependencyType",
    "MarketRelationship",
    "CrossMarketOpportunity",
    "RelationshipDetector",
    "VWAPCalculator",
    "DependencyCache",
    "CrossMarketScanner",
]
