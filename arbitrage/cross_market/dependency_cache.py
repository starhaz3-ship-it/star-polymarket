"""
SQLite Cache for Market Dependency Relationships

Minimizes redundant LLM calls by caching detected relationships.
"""

import sqlite3
import json
import time
from typing import List, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime, timedelta

from .dependency_types import MarketRelationship, DependencyType, ConstraintOperator


class DependencyCache:
    """
    SQLite cache for detected market relationships.

    Features:
    - Persistent storage across runs
    - TTL-based invalidation
    - Efficient lookup by market ID
    - Batch operations for performance
    """

    def __init__(self, db_path: str = "market_dependencies.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Create tables if not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_a_id TEXT NOT NULL,
                    market_b_id TEXT NOT NULL,
                    market_a_question TEXT,
                    market_b_question TEXT,
                    dependency_type TEXT NOT NULL,
                    constraint_text TEXT NOT NULL,
                    constraint_operator TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    detection_method TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    validated INTEGER DEFAULT 0,
                    notes TEXT,
                    UNIQUE(market_a_id, market_b_id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_a
                ON relationships(market_a_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_b
                ON relationships(market_b_id)
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS scanned_pairs (
                    market_a_id TEXT NOT NULL,
                    market_b_id TEXT NOT NULL,
                    scanned_at REAL NOT NULL,
                    has_relationship INTEGER DEFAULT 0,
                    PRIMARY KEY(market_a_id, market_b_id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_scan_status (
                    market_id TEXT PRIMARY KEY,
                    last_scanned REAL NOT NULL,
                    relationships_found INTEGER DEFAULT 0
                )
            """)

            conn.commit()

    def get_relationship(
        self,
        market_a_id: str,
        market_b_id: str
    ) -> Optional[MarketRelationship]:
        """Get cached relationship between two markets (order-independent)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM relationships
                WHERE (market_a_id = ? AND market_b_id = ?)
                   OR (market_a_id = ? AND market_b_id = ?)
            """, (market_a_id, market_b_id, market_b_id, market_a_id))

            row = cursor.fetchone()
            if row:
                return self._row_to_relationship(row)
        return None

    def save_relationship(self, relationship: MarketRelationship):
        """Save a detected relationship."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO relationships
                (market_a_id, market_b_id, market_a_question, market_b_question,
                 dependency_type, constraint_text, constraint_operator,
                 confidence, detection_method, created_at, validated, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                relationship.market_a_id,
                relationship.market_b_id,
                relationship.market_a_question,
                relationship.market_b_question,
                relationship.dependency_type.value,
                relationship.constraint,
                relationship.constraint_operator.value,
                relationship.confidence,
                relationship.detection_method,
                relationship.created_at,
                1 if relationship.validated else 0,
                relationship.notes,
            ))
            conn.commit()

    def save_relationships_batch(self, relationships: List[MarketRelationship]):
        """Save multiple relationships in a single transaction."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO relationships
                (market_a_id, market_b_id, market_a_question, market_b_question,
                 dependency_type, constraint_text, constraint_operator,
                 confidence, detection_method, created_at, validated, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    r.market_a_id, r.market_b_id,
                    r.market_a_question, r.market_b_question,
                    r.dependency_type.value, r.constraint,
                    r.constraint_operator.value, r.confidence,
                    r.detection_method, r.created_at,
                    1 if r.validated else 0, r.notes,
                )
                for r in relationships
            ])
            conn.commit()

    def get_all_relationships_for_market(
        self,
        market_id: str
    ) -> List[MarketRelationship]:
        """Get all relationships involving a market."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM relationships
                WHERE market_a_id = ? OR market_b_id = ?
            """, (market_id, market_id))

            return [self._row_to_relationship(row) for row in cursor.fetchall()]

    def get_all_relationships(
        self,
        min_confidence: float = 0.0,
        dependency_type: Optional[DependencyType] = None
    ) -> List[MarketRelationship]:
        """Get all cached relationships with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM relationships WHERE confidence >= ?"
            params = [min_confidence]

            if dependency_type:
                query += " AND dependency_type = ?"
                params.append(dependency_type.value)

            cursor = conn.execute(query, params)
            return [self._row_to_relationship(row) for row in cursor.fetchall()]

    def mark_pair_scanned(
        self,
        market_a_id: str,
        market_b_id: str,
        has_relationship: bool = False
    ):
        """Mark a pair as scanned (even if no relationship found)."""
        # Ensure consistent ordering
        if market_a_id > market_b_id:
            market_a_id, market_b_id = market_b_id, market_a_id

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO scanned_pairs
                (market_a_id, market_b_id, scanned_at, has_relationship)
                VALUES (?, ?, ?, ?)
            """, (market_a_id, market_b_id, time.time(), 1 if has_relationship else 0))
            conn.commit()

    def is_pair_scanned(
        self,
        market_a_id: str,
        market_b_id: str,
        max_age_hours: int = 168  # 1 week
    ) -> bool:
        """Check if a pair has been scanned recently."""
        # Ensure consistent ordering
        if market_a_id > market_b_id:
            market_a_id, market_b_id = market_b_id, market_a_id

        min_time = time.time() - (max_age_hours * 3600)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 1 FROM scanned_pairs
                WHERE market_a_id = ? AND market_b_id = ? AND scanned_at > ?
            """, (market_a_id, market_b_id, min_time))

            return cursor.fetchone() is not None

    def get_unscanned_pairs(
        self,
        market_ids: List[str],
        max_age_hours: int = 168
    ) -> List[Tuple[str, str]]:
        """Get pairs that haven't been scanned recently."""
        min_time = time.time() - (max_age_hours * 3600)

        # Generate all possible pairs
        all_pairs = set()
        for i, m1 in enumerate(market_ids):
            for m2 in market_ids[i+1:]:
                # Consistent ordering
                if m1 > m2:
                    m1, m2 = m2, m1
                all_pairs.add((m1, m2))

        # Get already scanned pairs
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT market_a_id, market_b_id FROM scanned_pairs
                WHERE scanned_at > ?
            """, (min_time,))

            scanned = set((row[0], row[1]) for row in cursor.fetchall())

        return list(all_pairs - scanned)

    def update_market_scan_status(self, market_id: str, relationships_found: int):
        """Update the scan status for a market."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO market_scan_status
                (market_id, last_scanned, relationships_found)
                VALUES (?, ?, ?)
            """, (market_id, time.time(), relationships_found))
            conn.commit()

    def get_markets_needing_scan(
        self,
        all_market_ids: List[str],
        max_age_hours: int = 168,
        limit: int = 100
    ) -> List[str]:
        """Get market IDs that haven't been scanned recently."""
        min_time = time.time() - (max_age_hours * 3600)

        with sqlite3.connect(self.db_path) as conn:
            # Get recently scanned markets
            cursor = conn.execute("""
                SELECT market_id FROM market_scan_status
                WHERE last_scanned > ?
            """, (min_time,))

            recently_scanned = set(row[0] for row in cursor.fetchall())

        # Return markets not recently scanned
        needs_scan = [m for m in all_market_ids if m not in recently_scanned]
        return needs_scan[:limit]

    def invalidate_old_relationships(self, max_age_days: int = 30):
        """Delete relationships older than max_age_days."""
        min_time = time.time() - (max_age_days * 24 * 3600)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                DELETE FROM relationships WHERE created_at < ?
            """, (min_time,))

            conn.execute("""
                DELETE FROM scanned_pairs WHERE scanned_at < ?
            """, (min_time,))

            conn.commit()

    def get_statistics(self) -> dict:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            relationships_count = conn.execute(
                "SELECT COUNT(*) FROM relationships"
            ).fetchone()[0]

            scanned_pairs_count = conn.execute(
                "SELECT COUNT(*) FROM scanned_pairs"
            ).fetchone()[0]

            by_type = {}
            cursor = conn.execute("""
                SELECT dependency_type, COUNT(*) as cnt
                FROM relationships
                GROUP BY dependency_type
            """)
            for row in cursor.fetchall():
                by_type[row[0]] = row[1]

            return {
                "total_relationships": relationships_count,
                "total_scanned_pairs": scanned_pairs_count,
                "by_type": by_type,
            }

    def _row_to_relationship(self, row: sqlite3.Row) -> MarketRelationship:
        """Convert a database row to MarketRelationship."""
        return MarketRelationship(
            market_a_id=row["market_a_id"],
            market_b_id=row["market_b_id"],
            market_a_question=row["market_a_question"] or "",
            market_b_question=row["market_b_question"] or "",
            dependency_type=DependencyType(row["dependency_type"]),
            constraint=row["constraint_text"],
            constraint_operator=ConstraintOperator(row["constraint_operator"]),
            confidence=row["confidence"],
            detection_method=row["detection_method"],
            created_at=row["created_at"],
            validated=bool(row["validated"]),
            notes=row["notes"] or "",
        )
