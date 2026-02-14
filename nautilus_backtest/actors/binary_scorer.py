"""BinaryOutcomeScorer - tracks strategy signals and resolves against future price data.

Not a nautilus Actor. Simple class that registers directional signals (UP/DOWN)
and resolves them after N bars to compute binary-option-style PnL.

Simulates Polymarket binary outcomes: pay entry_cost + spread to enter,
receive $1.00 if correct, $0.00 if wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
from datetime import datetime, timezone


@dataclass
class Signal:
    """A single directional signal registered by a strategy."""
    strategy: str
    bar_index: int
    direction: str          # "UP" or "DOWN"
    confidence: float       # 0.0 - 1.0
    entry_price: float      # BTC price at signal bar
    horizon_bars: int       # Bars to wait before resolving
    open_time_ms: int = 0   # UTC timestamp of the bar (for hour-of-day analysis)
    resolved: bool = False
    correct: Optional[bool] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0


class BinaryOutcomeScorer:
    """Tracks pending signals and resolves them against future price data.

    Models Polymarket binary option PnL:
        WIN  = +($1.00 - entry_cost - spread)
        LOSS = -(entry_cost + spread)

    With defaults (entry=0.50, spread=0.03):
        WIN  = +$0.47
        LOSS = -$0.53
    """

    def __init__(self, entry_cost: float = 0.50, spread: float = 0.03):
        self.entry_cost = entry_cost
        self.spread = spread
        self.pending: list[Signal] = []
        self.resolved: list[Signal] = []

    def add_signal(
        self,
        strategy: str,
        bar_index: int,
        direction: str,
        confidence: float,
        entry_price: float,
        horizon_bars: int,
        open_time_ms: int = 0,
    ) -> None:
        """Register a new directional signal."""
        sig = Signal(
            strategy=strategy,
            bar_index=bar_index,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            horizon_bars=horizon_bars,
            open_time_ms=open_time_ms,
        )
        self.pending.append(sig)

    def check_resolution(self, bar_index: int, close_price: float) -> list[Signal]:
        """Check if any pending signals should be resolved.

        Returns list of signals that were resolved on this bar.
        """
        just_resolved = []

        for sig in self.pending:
            if bar_index >= sig.bar_index + sig.horizon_bars:
                sig.exit_price = close_price
                sig.correct = (
                    (sig.direction == "UP" and close_price > sig.entry_price)
                    or (sig.direction == "DOWN" and close_price < sig.entry_price)
                )
                if sig.correct:
                    sig.pnl = 1.0 - self.entry_cost - self.spread
                else:
                    sig.pnl = -(self.entry_cost + self.spread)
                sig.resolved = True
                self.resolved.append(sig)
                just_resolved.append(sig)

        self.pending = [s for s in self.pending if not s.resolved]
        return just_resolved

    def force_resolve_pending(self, close_price: float) -> None:
        """Force-resolve any remaining pending signals at end of data."""
        for sig in self.pending:
            sig.exit_price = close_price
            sig.correct = (
                (sig.direction == "UP" and close_price > sig.entry_price)
                or (sig.direction == "DOWN" and close_price < sig.entry_price)
            )
            if sig.correct:
                sig.pnl = 1.0 - self.entry_cost - self.spread
            else:
                sig.pnl = -(self.entry_cost + self.spread)
            sig.resolved = True
            self.resolved.append(sig)
        self.pending.clear()

    # ── Summary / Analytics ──────────────────────────────────────────────

    def summary(self) -> dict[str, dict]:
        """Per-strategy summary stats.

        Returns dict keyed by strategy name with:
            trades, wins, losses, wr, pnl, avg_confidence, avg_pnl
        """
        by_strat: dict[str, list[Signal]] = defaultdict(list)
        for sig in self.resolved:
            by_strat[sig.strategy].append(sig)

        results = {}
        for name, signals in sorted(by_strat.items()):
            wins = sum(1 for s in signals if s.correct)
            losses = len(signals) - wins
            total_pnl = sum(s.pnl for s in signals)
            avg_conf = sum(s.confidence for s in signals) / len(signals) if signals else 0
            avg_pnl = total_pnl / len(signals) if signals else 0

            results[name] = {
                "trades": len(signals),
                "wins": wins,
                "losses": losses,
                "wr": (wins / len(signals) * 100) if signals else 0,
                "pnl": total_pnl,
                "avg_confidence": avg_conf,
                "avg_pnl": avg_pnl,
            }

        return results

    def overall_summary(self) -> dict:
        """Aggregate stats across all strategies."""
        total = len(self.resolved)
        wins = sum(1 for s in self.resolved if s.correct)
        total_pnl = sum(s.pnl for s in self.resolved)
        return {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "wr": (wins / total * 100) if total else 0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / total if total else 0,
            "pending": len(self.pending),
        }

    def hourly_breakdown(self) -> dict[int, dict]:
        """Win rate breakdown by UTC hour of signal entry.

        Returns dict keyed by hour (0-23) with trades, wins, wr, pnl.
        """
        by_hour: dict[int, list[Signal]] = defaultdict(list)
        for sig in self.resolved:
            if sig.open_time_ms > 0:
                utc_hour = datetime.fromtimestamp(
                    sig.open_time_ms / 1000.0, tz=timezone.utc
                ).hour
            else:
                utc_hour = -1  # Unknown
            by_hour[utc_hour].append(sig)

        results = {}
        for hour in sorted(by_hour.keys()):
            signals = by_hour[hour]
            wins = sum(1 for s in signals if s.correct)
            results[hour] = {
                "trades": len(signals),
                "wins": wins,
                "wr": (wins / len(signals) * 100) if signals else 0,
                "pnl": sum(s.pnl for s in signals),
            }
        return results

    def consensus_analysis(self) -> dict:
        """When 2+ strategies agree on direction at the same bar, what is WR?

        Groups resolved signals by bar_index, checks where multiple strategies
        agree on direction, and computes WR for consensus vs solo signals.
        """
        # Group signals by bar_index
        by_bar: dict[int, list[Signal]] = defaultdict(list)
        for sig in self.resolved:
            by_bar[sig.bar_index].append(sig)

        consensus_wins = 0
        consensus_total = 0
        consensus_pnl = 0.0
        solo_wins = 0
        solo_total = 0
        solo_pnl = 0.0

        for bar_idx, signals in by_bar.items():
            if len(signals) < 2:
                # Solo signal
                for s in signals:
                    solo_total += 1
                    if s.correct:
                        solo_wins += 1
                    solo_pnl += s.pnl
                continue

            # Check for directional agreement
            up_sigs = [s for s in signals if s.direction == "UP"]
            down_sigs = [s for s in signals if s.direction == "DOWN"]

            # Consensus = majority direction with 2+ agreeing
            for group in [up_sigs, down_sigs]:
                if len(group) >= 2:
                    for s in group:
                        consensus_total += 1
                        if s.correct:
                            consensus_wins += 1
                        consensus_pnl += s.pnl
                elif len(group) == 1:
                    # Dissenting signal in a multi-signal bar
                    for s in group:
                        solo_total += 1
                        if s.correct:
                            solo_wins += 1
                        solo_pnl += s.pnl

        return {
            "consensus": {
                "trades": consensus_total,
                "wins": consensus_wins,
                "wr": (consensus_wins / consensus_total * 100) if consensus_total else 0,
                "pnl": consensus_pnl,
            },
            "solo": {
                "trades": solo_total,
                "wins": solo_wins,
                "wr": (solo_wins / solo_total * 100) if solo_total else 0,
                "pnl": solo_pnl,
            },
        }

    def cumulative_pnl_by_strategy(self) -> dict[str, list[float]]:
        """Returns cumulative PnL series per strategy, ordered by bar_index.

        Used for ASCII charting.
        """
        by_strat: dict[str, list[Signal]] = defaultdict(list)
        for sig in self.resolved:
            by_strat[sig.strategy].append(sig)

        result = {}
        for name, signals in sorted(by_strat.items()):
            signals.sort(key=lambda s: s.bar_index)
            cum = []
            running = 0.0
            for s in signals:
                running += s.pnl
                cum.append(running)
            result[name] = cum

        return result
