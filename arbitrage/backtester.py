"""
Backtesting Engine for Polymarket Strategies

Analyzes historical data to find optimal strategy parameters.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple
import statistics


@dataclass
class TradeResult:
    """Result of a simulated trade."""
    market: str
    outcome: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    won: bool
    strategy: str


class Backtester:
    """
    Backtests trading strategies on historical data.
    """

    def __init__(self, data_file: str = None):
        self.data_file = data_file or "C:/Users/Star/.local/bin/star-polymarket/data/btc_backtest_compact.json"
        self.trades: List[dict] = []
        self.results: List[TradeResult] = []
        self._load_data()

    def _load_data(self):
        """Load historical trade data."""
        try:
            with open(self.data_file, "r") as f:
                data = json.load(f)
                self.trades = data.get("whale_trades", [])
                print(f"[Backtester] Loaded {len(self.trades)} historical trades")
        except Exception as e:
            print(f"[Backtester] Error loading data: {e}")

    def analyze_by_entry_price(self) -> Dict:
        """
        Analyze performance by entry price bucket.
        Find optimal entry price ranges.
        """
        buckets = {
            "0-10%": {"trades": [], "pnl": 0, "wins": 0, "losses": 0},
            "10-30%": {"trades": [], "pnl": 0, "wins": 0, "losses": 0},
            "30-50%": {"trades": [], "pnl": 0, "wins": 0, "losses": 0},
            "50-70%": {"trades": [], "pnl": 0, "wins": 0, "losses": 0},
            "70-90%": {"trades": [], "pnl": 0, "wins": 0, "losses": 0},
            "90-100%": {"trades": [], "pnl": 0, "wins": 0, "losses": 0},
        }

        for t in self.trades:
            entry = float(t.get("entry_price", 0) or 0)
            pnl = float(t.get("pnl", 0) or 0)

            # Determine bucket
            if entry < 0.10:
                bucket = "0-10%"
            elif entry < 0.30:
                bucket = "10-30%"
            elif entry < 0.50:
                bucket = "30-50%"
            elif entry < 0.70:
                bucket = "50-70%"
            elif entry < 0.90:
                bucket = "70-90%"
            else:
                bucket = "90-100%"

            buckets[bucket]["trades"].append(t)
            buckets[bucket]["pnl"] += pnl
            if pnl > 0:
                buckets[bucket]["wins"] += 1
            else:
                buckets[bucket]["losses"] += 1

        # Calculate stats
        for bucket, data in buckets.items():
            total = data["wins"] + data["losses"]
            data["win_rate"] = data["wins"] / total if total > 0 else 0
            data["avg_pnl"] = data["pnl"] / total if total > 0 else 0
            data["trade_count"] = total

        return buckets

    def analyze_by_outcome(self) -> Dict:
        """Analyze performance by YES vs NO."""
        results = {"Yes": {"pnl": 0, "wins": 0, "losses": 0, "count": 0},
                   "No": {"pnl": 0, "wins": 0, "losses": 0, "count": 0}}

        for t in self.trades:
            outcome = t.get("outcome", "Yes")
            pnl = float(t.get("pnl", 0) or 0)

            if outcome in results:
                results[outcome]["pnl"] += pnl
                results[outcome]["count"] += 1
                if pnl > 0:
                    results[outcome]["wins"] += 1
                else:
                    results[outcome]["losses"] += 1

        for outcome, data in results.items():
            total = data["wins"] + data["losses"]
            data["win_rate"] = data["wins"] / total if total > 0 else 0

        return results

    def find_winning_patterns(self) -> List[Dict]:
        """
        Identify patterns that lead to winning trades.
        """
        winners = [t for t in self.trades if float(t.get("pnl", 0) or 0) > 0]
        losers = [t for t in self.trades if float(t.get("pnl", 0) or 0) <= 0]

        if not winners:
            return []

        # Analyze winner characteristics
        winner_entries = [float(t.get("entry_price", 0) or 0) for t in winners]
        loser_entries = [float(t.get("entry_price", 0) or 0) for t in losers] if losers else [0]

        patterns = []

        # Pattern 1: High probability entries
        avg_winner_entry = statistics.mean(winner_entries) if winner_entries else 0
        avg_loser_entry = statistics.mean(loser_entries) if loser_entries else 0

        patterns.append({
            "pattern": "Entry Price",
            "winner_avg": avg_winner_entry,
            "loser_avg": avg_loser_entry,
            "insight": f"Winners enter at {avg_winner_entry:.0%} avg vs losers at {avg_loser_entry:.0%}",
            "recommendation": f"Target entries above {max(0.70, avg_winner_entry - 0.1):.0%} probability"
        })

        # Pattern 2: YES vs NO performance
        yes_winners = len([t for t in winners if t.get("outcome") == "Yes"])
        no_winners = len([t for t in winners if t.get("outcome") == "No"])
        yes_losers = len([t for t in losers if t.get("outcome") == "Yes"])
        no_losers = len([t for t in losers if t.get("outcome") == "No"])

        yes_rate = yes_winners / (yes_winners + yes_losers) if (yes_winners + yes_losers) > 0 else 0
        no_rate = no_winners / (no_winners + no_losers) if (no_winners + no_losers) > 0 else 0

        patterns.append({
            "pattern": "YES vs NO",
            "yes_win_rate": yes_rate,
            "no_win_rate": no_rate,
            "insight": f"YES wins {yes_rate:.0%}, NO wins {no_rate:.0%}",
            "recommendation": "YES" if yes_rate > no_rate else "NO" + " positions perform better"
        })

        # Pattern 3: Current price vs entry (momentum)
        momentum_wins = []
        for t in winners:
            entry = float(t.get("entry_price", 0) or 0)
            current = float(t.get("current_price", 0) or 0)
            if entry > 0:
                momentum = (current - entry) / entry
                momentum_wins.append(momentum)

        if momentum_wins:
            avg_momentum = statistics.mean(momentum_wins)
            patterns.append({
                "pattern": "Price Momentum",
                "avg_winner_momentum": avg_momentum,
                "insight": f"Winners have {avg_momentum:+.0%} price movement on average",
                "recommendation": "Look for positions with positive momentum"
            })

        return patterns

    def simulate_strategy(
        self,
        min_entry_price: float = 0.70,
        max_entry_price: float = 0.95,
        preferred_outcome: str = None,
        position_size: float = 100.0,
    ) -> Dict:
        """
        Simulate a strategy on historical data.

        Args:
            min_entry_price: Minimum entry probability
            max_entry_price: Maximum entry probability
            preferred_outcome: "Yes", "No", or None for both
            position_size: USD per trade
        """
        simulated_trades = []
        total_pnl = 0
        wins = 0
        losses = 0

        for t in self.trades:
            entry = float(t.get("entry_price", 0) or 0)
            outcome = t.get("outcome", "")
            pnl_pct = float(t.get("pnl_pct", 0) or 0)

            # Apply filters
            if entry < min_entry_price or entry > max_entry_price:
                continue
            if preferred_outcome and outcome != preferred_outcome:
                continue

            # Simulate trade
            sim_pnl = position_size * (pnl_pct / 100)
            total_pnl += sim_pnl

            if sim_pnl > 0:
                wins += 1
            else:
                losses += 1

            simulated_trades.append({
                "market": t.get("market", "")[:40],
                "outcome": outcome,
                "entry": entry,
                "pnl": sim_pnl,
            })

        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        return {
            "strategy": f"Entry {min_entry_price:.0%}-{max_entry_price:.0%}, {preferred_outcome or 'Both'}",
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl_per_trade": avg_pnl,
            "trades": simulated_trades,
        }

    def optimize_parameters(self) -> Dict:
        """
        Find optimal strategy parameters through grid search.
        """
        best_strategy = None
        best_pnl = float("-inf")

        results = []

        # Grid search over parameters
        for min_entry in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
            for max_entry in [0.85, 0.90, 0.95, 0.99]:
                if min_entry >= max_entry:
                    continue

                for outcome in [None, "Yes", "No"]:
                    result = self.simulate_strategy(
                        min_entry_price=min_entry,
                        max_entry_price=max_entry,
                        preferred_outcome=outcome,
                        position_size=100.0
                    )

                    if result["total_trades"] >= 3:  # Minimum sample size
                        results.append(result)

                        if result["total_pnl"] > best_pnl:
                            best_pnl = result["total_pnl"]
                            best_strategy = result

        # Sort by total P&L
        results.sort(key=lambda x: x["total_pnl"], reverse=True)

        return {
            "best_strategy": best_strategy,
            "top_5_strategies": results[:5],
            "optimization_complete": True,
        }

    def print_analysis(self):
        """Print full analysis report."""
        print("\n" + "=" * 70)
        print("BACKTEST ANALYSIS REPORT")
        print("=" * 70)

        # Entry price analysis
        print("\n1. PERFORMANCE BY ENTRY PRICE")
        print("-" * 50)
        buckets = self.analyze_by_entry_price()
        for bucket, data in buckets.items():
            if data["trade_count"] > 0:
                print(f"  {bucket:12s}: {data['trade_count']:3d} trades | "
                      f"Win Rate: {data['win_rate']:5.0%} | "
                      f"P&L: ${data['pnl']:>+10,.0f} | "
                      f"Avg: ${data['avg_pnl']:>+8,.0f}")

        # YES vs NO analysis
        print("\n2. PERFORMANCE BY OUTCOME (YES vs NO)")
        print("-" * 50)
        outcome_data = self.analyze_by_outcome()
        for outcome, data in outcome_data.items():
            print(f"  {outcome:4s}: {data['count']:3d} trades | "
                  f"Win Rate: {data['win_rate']:5.0%} | "
                  f"P&L: ${data['pnl']:>+10,.0f}")

        # Winning patterns
        print("\n3. WINNING PATTERNS IDENTIFIED")
        print("-" * 50)
        patterns = self.find_winning_patterns()
        for p in patterns:
            print(f"  [{p['pattern']}]")
            print(f"    {p['insight']}")
            print(f"    >> {p['recommendation']}")

        # Optimization
        print("\n4. OPTIMIZED STRATEGY PARAMETERS")
        print("-" * 50)
        opt = self.optimize_parameters()

        if opt["best_strategy"]:
            best = opt["best_strategy"]
            print(f"  BEST STRATEGY: {best['strategy']}")
            print(f"    Trades: {best['total_trades']} | Win Rate: {best['win_rate']:.0%}")
            print(f"    Total P&L: ${best['total_pnl']:+,.0f}")
            print(f"    Avg P&L/Trade: ${best['avg_pnl_per_trade']:+,.0f}")

        print("\n  TOP 5 STRATEGIES:")
        for i, s in enumerate(opt["top_5_strategies"], 1):
            print(f"    {i}. {s['strategy']}")
            print(f"       {s['total_trades']} trades | {s['win_rate']:.0%} win | ${s['total_pnl']:+,.0f}")

        print("\n" + "=" * 70)
        print("RECOMMENDATIONS FOR MAXIMUM PROFIT")
        print("=" * 70)

        # Generate recommendations
        if opt["best_strategy"]:
            best = opt["best_strategy"]
            print(f"""
1. ENTRY CRITERIA:
   - Only enter at {best['strategy'].split(',')[0].replace('Entry ', '')} probability
   - Focus on {best['strategy'].split(',')[1].strip() if ',' in best['strategy'] else 'both YES and NO'}

2. POSITION SIZING:
   - Use Kelly Criterion: edge% / odds
   - Cap at 10% of bankroll per trade
   - Scale up on higher confidence signals

3. RISK MANAGEMENT:
   - Stop if daily loss exceeds 5% of bankroll
   - Max 5 concurrent positions
   - Avoid low probability (<50%) speculative bets

4. TIMING:
   - Enter positions with positive momentum
   - Prefer markets expiring within 1-7 days
   - Avoid markets expiring within 1 hour (high volatility)

5. COPY TRADING FILTER:
   - Only copy trades matching optimal parameters
   - Scale to 1-5% of whale size
   - Exit if whale exits
""")

        return opt


def run_backtest():
    """Run full backtest analysis."""
    bt = Backtester()
    return bt.print_analysis()


if __name__ == "__main__":
    run_backtest()
