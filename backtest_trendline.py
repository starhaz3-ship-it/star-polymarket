"""
TRENDLINE_BREAK Backtest for Polymarket 15-Minute Up/Down Markets

Strategy:
    Detect ascending trendlines formed by pivot lows on 1-minute candles.
    When price breaks below the trendline, signal SHORT (= bet DOWN on Polymarket).

Adapted for Polymarket:
    - Uses 1-minute candles (100 bar lookback) to detect trendline on the micro level
    - Signal = "bet DOWN on the current 15-min window"
    - Win = price at window close < price at window open
    - Entry at market price for DOWN side ($0.45-$0.55 range as realistic)
    - V3.14 filters: skip hours {5,6,8,9,10,12,14,16}, entry window 5-9 min before expiry

Trendline Break Detection:
    - Pivot detection (left=5, right=3)
    - Find ascending trendline from 2+ pivot lows
    - Price breaks 0.05% below trendline = signal
    - Filters: RSI < 50 (required), vol > 1.3x (bonus), below EMA50 (bonus)

Author: Claude Code (research backtest)
Date: 2026-02-09
"""

import asyncio
import json
import math
import random
import time
import statistics
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from functools import partial
from collections import defaultdict

import httpx

# Force unbuffered output
print = partial(print, flush=True)

# Import TA tools from the existing codebase
import sys
sys.path.insert(0, str(Path(__file__).parent))
from arbitrage.ta_signals import TASignalGenerator, Candle


# ============================================================================
# Trendline Break Detection Engine
# ============================================================================

@dataclass
class PivotLow:
    """A confirmed pivot low point."""
    index: int        # Index in the candle array
    price: float      # Low price at this pivot
    timestamp: float   # Epoch timestamp


@dataclass
class Trendline:
    """An ascending trendline connecting two pivot lows."""
    p1: PivotLow
    p2: PivotLow
    slope: float          # Price change per bar
    touches: int = 2      # Number of times price touched the trendline
    strength: float = 0.0 # Composite strength score


@dataclass
class TrendlineBreakSignal:
    """A trendline break signal."""
    break_index: int
    break_price: float
    trendline: Trendline
    break_pct: float         # How far below the trendline (%)
    rsi: Optional[float]
    vol_ratio: float         # Volume ratio vs average
    below_ema50: bool
    # Composite score
    score: float = 0.0
    filters_passed: bool = False


def find_pivot_lows(candles: List[Candle], left: int = 5, right: int = 3) -> List[PivotLow]:
    """
    Find pivot lows in candle data.

    A pivot low at index i means:
        candles[i].low <= all candles in [i-left, i+right]

    Args:
        candles: OHLCV candle data
        left: Number of bars to the left that must be higher
        right: Number of bars to the right that must be higher

    Returns:
        List of PivotLow objects
    """
    pivots = []

    for i in range(left, len(candles) - right):
        low = candles[i].low
        is_pivot = True

        # Check left side
        for j in range(i - left, i):
            if candles[j].low < low:
                is_pivot = False
                break

        if not is_pivot:
            continue

        # Check right side
        for j in range(i + 1, i + right + 1):
            if candles[j].low < low:
                is_pivot = False
                break

        if is_pivot:
            pivots.append(PivotLow(
                index=i,
                price=low,
                timestamp=candles[i].timestamp
            ))

    return pivots


def build_ascending_trendlines(
    pivots: List[PivotLow],
    candles: List[Candle],
    min_bars_apart: int = 8,
    max_bars_apart: int = 60,
    touch_tolerance_pct: float = 0.05
) -> List[Trendline]:
    """
    Build ascending trendlines from pivot lows.

    An ascending trendline connects two pivot lows where:
        1. The second pivot is higher than the first (ascending)
        2. They are at least min_bars_apart bars apart
        3. Price does not convincingly break below the trendline between them

    Args:
        pivots: List of pivot lows
        candles: Original candle data for validation
        min_bars_apart: Minimum distance between pivots
        max_bars_apart: Maximum distance between pivots
        touch_tolerance_pct: Percentage tolerance for trendline touches

    Returns:
        List of valid ascending Trendline objects
    """
    trendlines = []

    for i in range(len(pivots)):
        for j in range(i + 1, len(pivots)):
            p1 = pivots[i]
            p2 = pivots[j]

            bars_apart = p2.index - p1.index

            # Distance filter
            if bars_apart < min_bars_apart or bars_apart > max_bars_apart:
                continue

            # Must be ascending (p2 higher than p1)
            if p2.price <= p1.price:
                continue

            # Calculate slope
            slope = (p2.price - p1.price) / bars_apart

            # Validate: no candle closes convincingly below the trendline between p1 and p2
            valid = True
            touches = 2  # Start with the two anchor points

            for k in range(p1.index + 1, p2.index):
                tl_price_at_k = p1.price + slope * (k - p1.index)
                tolerance = tl_price_at_k * touch_tolerance_pct / 100

                # If a candle close breaks below trendline by more than tolerance, invalid
                if candles[k].close < tl_price_at_k - tolerance * 2:
                    valid = False
                    break

                # Count touches (low comes within tolerance of trendline)
                if abs(candles[k].low - tl_price_at_k) <= tolerance:
                    touches += 1

            if not valid:
                continue

            # Strength score: more touches + steeper slope = stronger
            # Normalize slope relative to price level
            slope_pct = slope / p1.price * 100 if p1.price > 0 else 0
            strength = touches * 1.0 + slope_pct * 10 + bars_apart * 0.05

            trendlines.append(Trendline(
                p1=p1,
                p2=p2,
                slope=slope,
                touches=touches,
                strength=strength
            ))

    # Sort by strength descending
    trendlines.sort(key=lambda t: t.strength, reverse=True)

    return trendlines


def detect_trendline_break(
    candles: List[Candle],
    break_threshold_pct: float = 0.05,
    pivot_left: int = 5,
    pivot_right: int = 3,
    lookback: int = 100
) -> Optional[TrendlineBreakSignal]:
    """
    Detect if price has broken below an ascending trendline.

    This is the main detection function. It:
    1. Finds pivot lows in the lookback window
    2. Builds ascending trendlines
    3. Checks if the current price breaks below any trendline
    4. Applies RSI/volume/EMA filters

    Args:
        candles: Full candle history (at least lookback bars)
        break_threshold_pct: Minimum break below trendline (%) to trigger signal
        pivot_left: Left lookback for pivot detection
        pivot_right: Right lookback for pivot detection
        lookback: Number of bars to look back for trendline construction

    Returns:
        TrendlineBreakSignal if break detected, None otherwise
    """
    if len(candles) < lookback:
        return None

    window = candles[-lookback:]
    current_bar = window[-1]

    # Step 1: Find pivot lows
    pivots = find_pivot_lows(window, left=pivot_left, right=pivot_right)

    if len(pivots) < 2:
        return None

    # Step 2: Build ascending trendlines
    trendlines = build_ascending_trendlines(pivots, window)

    if not trendlines:
        return None

    # Step 3: Check for break on the strongest trendline(s)
    best_break = None
    current_idx = len(window) - 1

    for tl in trendlines[:5]:  # Check top 5 strongest trendlines
        # Project trendline to current bar
        bars_from_p1 = current_idx - tl.p1.index
        tl_price_at_current = tl.p1.price + tl.slope * bars_from_p1

        # Skip if trendline is above the price range (nonsensical projection)
        if tl_price_at_current <= 0:
            continue

        # Calculate break percentage
        break_pct = (tl_price_at_current - current_bar.close) / tl_price_at_current * 100

        # Is this a break? (price below trendline by threshold)
        if break_pct >= break_threshold_pct:
            if best_break is None or break_pct > best_break.break_pct:
                best_break = TrendlineBreakSignal(
                    break_index=current_idx,
                    break_price=current_bar.close,
                    trendline=tl,
                    break_pct=break_pct,
                    rsi=None,
                    vol_ratio=1.0,
                    below_ema50=False,
                    score=0.0,
                    filters_passed=False
                )

    if best_break is None:
        return None

    # Step 4: Calculate filters
    ta = TASignalGenerator()
    closes = [c.close for c in window]

    # RSI
    rsi = ta.compute_rsi(closes, period=14)
    best_break.rsi = rsi

    # Volume ratio
    recent_vol = sum(c.volume for c in window[-5:]) / 5 if len(window) >= 5 else 0
    avg_vol = sum(c.volume for c in window[-30:]) / 30 if len(window) >= 30 else 1
    vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
    best_break.vol_ratio = vol_ratio

    # EMA50
    ema50 = ta._ema(closes, 50)
    best_break.below_ema50 = (current_bar.close < ema50) if ema50 else False

    # Filter check: RSI < 50 is REQUIRED
    if rsi is not None and rsi < 50:
        best_break.filters_passed = True
    else:
        best_break.filters_passed = False

    # Composite score
    score = best_break.break_pct * 10  # Base: break magnitude
    if rsi is not None:
        score += max(0, (50 - rsi)) * 0.5  # Lower RSI = stronger
    if vol_ratio > 1.3:
        score += 5  # Volume confirmation bonus
    if best_break.below_ema50:
        score += 3  # Below EMA50 bonus
    score += best_break.trendline.touches * 2  # More touches = stronger

    best_break.score = score

    return best_break


# ============================================================================
# Backtest Engine for Polymarket 15-Minute Markets
# ============================================================================

@dataclass
class BacktestResult:
    """Result for a single simulated trade."""
    window_start_time: datetime
    window_end_time: datetime
    side: str                    # "DOWN" for trendline break
    entry_price_asset: float     # BTC price at entry
    exit_price_asset: float      # BTC price at window close
    window_open_price: float     # BTC price at window open (15m boundary)
    price_change_pct: float      # % change over window
    won: bool                    # Did DOWN win?
    pnl: float                   # Simulated PnL
    poly_entry_price: float      # Simulated Polymarket DOWN price
    signal_score: float
    rsi: Optional[float]
    vol_ratio: float
    below_ema50: bool
    trendline_touches: int
    break_pct: float
    hour_utc: int


@dataclass
class BacktestSummary:
    """Overall backtest summary."""
    total_windows: int = 0
    total_signals: int = 0
    signals_passed_filter: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    max_drawdown: float = 0.0
    # Baseline comparison
    baseline_signals: int = 0
    baseline_wins: int = 0
    baseline_wr: float = 0.0
    baseline_pnl: float = 0.0
    # By filter
    filter_breakdown: Dict = field(default_factory=dict)
    hourly_breakdown: Dict = field(default_factory=dict)


class TrendlineBacktester:
    """Backtest trendline break strategy on Polymarket 15m windows."""

    # V3.14 Settings
    SKIP_HOURS_UTC = {5, 6, 8, 9, 10, 12, 14, 16}
    ENTRY_WINDOW_MIN = 5   # Minutes before expiry: earliest entry
    ENTRY_WINDOW_MAX = 9   # Minutes before expiry: latest entry
    BET_SIZE = 3.0          # $3 flat bet

    # Trendline params
    PIVOT_LEFT = 5
    PIVOT_RIGHT = 3
    CANDLE_LOOKBACK = 100
    BREAK_THRESHOLD_PCT = 0.05

    # Polymarket pricing simulation
    # DOWN price is roughly (1 - implied_prob_up)
    # For random windows: ~$0.45-$0.55 for DOWN
    DOWN_PRICE_LOW = 0.45
    DOWN_PRICE_HIGH = 0.55

    def __init__(self):
        self.trades: List[BacktestResult] = []
        self.all_windows_checked = 0

    async def fetch_historical_candles(
        self,
        symbol: str = "BTCUSDT",
        days: int = 7
    ) -> List[Candle]:
        """
        Fetch historical 1-minute candles from Binance.
        Uses the same API as run_ta_paper.py.
        """
        print(f"[{symbol}] Fetching {days} days of 1m candles from Binance...")

        candles = []
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        async with httpx.AsyncClient(timeout=30) as client:
            current = start_time
            batch = 0

            while current < end_time:
                start_ms = int(current.timestamp() * 1000)

                try:
                    r = await client.get(
                        "https://api.binance.com/api/v3/klines",
                        params={
                            "symbol": symbol,
                            "interval": "1m",
                            "startTime": start_ms,
                            "limit": 1000
                        }
                    )

                    if r.status_code == 200:
                        data = r.json()
                        for k in data:
                            candles.append(Candle(
                                timestamp=k[0] / 1000,
                                open=float(k[1]),
                                high=float(k[2]),
                                low=float(k[3]),
                                close=float(k[4]),
                                volume=float(k[5])
                            ))

                        if data:
                            last_ts = data[-1][0] / 1000
                            current = datetime.fromtimestamp(last_ts, tz=timezone.utc) + timedelta(minutes=1)
                        else:
                            break
                    elif r.status_code == 429:
                        print(f"  Rate limited, waiting 10s...")
                        await asyncio.sleep(10)
                        continue
                    else:
                        print(f"  API error: {r.status_code}")
                        break

                except Exception as e:
                    print(f"  Fetch error: {e}")
                    await asyncio.sleep(2)
                    continue

                batch += 1
                if batch % 5 == 0:
                    fetched_days = len(candles) / 1440
                    print(f"  ... {len(candles)} candles ({fetched_days:.1f} days)")

                await asyncio.sleep(0.15)  # Rate limit protection

        # Deduplicate by timestamp
        seen = set()
        unique = []
        for c in candles:
            if c.timestamp not in seen:
                seen.add(c.timestamp)
                unique.append(c)
        candles = sorted(unique, key=lambda c: c.timestamp)

        print(f"[{symbol}] Total: {len(candles)} candles ({len(candles)/1440:.1f} days)")
        return candles

    def _get_15m_window_boundaries(self, candles: List[Candle]) -> List[Tuple[int, int]]:
        """
        Find 15-minute window boundaries aligned to :00, :15, :30, :45.

        Returns list of (window_start_idx, window_end_idx) pairs.
        Each window is 15 candles (15 minutes of 1m data).
        """
        if not candles:
            return []

        windows = []
        i = 0

        while i < len(candles) - 15:
            ts = candles[i].timestamp
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)

            # Align to 15-minute boundary
            if dt.minute % 15 == 0:
                # This is a window start
                window_end = i + 15
                if window_end < len(candles):
                    windows.append((i, window_end))
                i += 15
            else:
                i += 1

        return windows

    def _simulate_poly_down_price(self, candles_before: List[Candle]) -> float:
        """
        Simulate a realistic Polymarket DOWN price.

        In real trading, DOWN price depends on market sentiment.
        We use recent momentum to estimate: if price trending up, DOWN is cheap.
        """
        if len(candles_before) < 10:
            return 0.50  # Default 50/50

        # Recent momentum (last 10 candles)
        momentum = (candles_before[-1].close - candles_before[-10].close) / candles_before[-10].close

        # Map momentum to DOWN price:
        # Strong up momentum -> DOWN is cheap (0.35-0.45)
        # Flat -> DOWN is ~0.50
        # Strong down momentum -> DOWN is expensive (0.55-0.65)
        base = 0.50
        adjustment = -momentum * 50  # Invert: up momentum makes DOWN cheaper

        poly_price = max(0.30, min(0.70, base + adjustment))

        # Add noise
        noise = random.gauss(0, 0.03)
        poly_price = max(0.30, min(0.70, poly_price + noise))

        return round(poly_price, 2)

    def run_backtest(self, candles: List[Candle]) -> BacktestSummary:
        """
        Run the trendline break backtest on historical candle data.

        For each 15-minute window:
        1. Check if it's in the entry window (5-9 min before expiry = minutes 6-10 of the window)
        2. Look at the 1m candles in that entry window for trendline break signals
        3. If signal detected + filters pass -> simulate a DOWN bet
        4. Determine if DOWN won (price at window end < price at window start)
        """
        print("\n" + "=" * 70)
        print("TRENDLINE BREAK BACKTEST - Polymarket 15-Min DOWN Markets")
        print("=" * 70)

        windows = self._get_15m_window_boundaries(candles)
        print(f"Total 15-minute windows found: {len(windows)}")

        summary = BacktestSummary()
        summary.total_windows = len(windows)

        # Baseline tracking (random DOWN bets for comparison)
        baseline_trades = []

        # Hourly stats
        hourly = defaultdict(lambda: {"signals": 0, "wins": 0, "pnl": 0.0})

        # Score bucket stats
        score_buckets = defaultdict(lambda: {"signals": 0, "wins": 0, "pnl": 0.0})

        signals_found = 0
        signals_filtered = 0

        for wi, (w_start, w_end) in enumerate(windows):
            w_start_dt = datetime.fromtimestamp(candles[w_start].timestamp, tz=timezone.utc)
            w_end_dt = datetime.fromtimestamp(candles[w_end].timestamp, tz=timezone.utc)
            hour = w_start_dt.hour

            # Skip hours filter (V3.14)
            if hour in self.SKIP_HOURS_UTC:
                continue

            self.all_windows_checked += 1

            # Window prices
            window_open_price = candles[w_start].open
            window_close_price = candles[w_end - 1].close  # Last candle's close
            actual_down = window_close_price < window_open_price
            price_change = (window_close_price - window_open_price) / window_open_price * 100

            # Baseline: bet DOWN on every non-skipped window
            poly_price_base = self._simulate_poly_down_price(
                candles[max(0, w_start - 30):w_start]
            )
            base_pnl = (1.0 - poly_price_base) * self.BET_SIZE if actual_down else -poly_price_base * self.BET_SIZE
            baseline_trades.append({"won": actual_down, "pnl": base_pnl})

            # Entry window: 5-9 minutes before expiry
            # In a 15-min window (indices 0-14), "5-9 min before expiry" means
            # minutes 6-10 of the window (indices 6, 7, 8, 9, 10)
            entry_start_idx = w_start + 6   # 6 minutes into window = 9 min before expiry
            entry_end_idx = w_start + 10     # 10 minutes into window = 5 min before expiry

            # Make sure we have enough lookback for trendline detection
            if entry_start_idx < self.CANDLE_LOOKBACK:
                continue

            # Scan each minute in the entry window for trendline break
            best_signal = None

            for scan_idx in range(entry_start_idx, min(entry_end_idx + 1, len(candles))):
                # Use lookback candles ending at scan_idx
                scan_candles = candles[scan_idx - self.CANDLE_LOOKBACK + 1:scan_idx + 1]

                signal = detect_trendline_break(
                    scan_candles,
                    break_threshold_pct=self.BREAK_THRESHOLD_PCT,
                    pivot_left=self.PIVOT_LEFT,
                    pivot_right=self.PIVOT_RIGHT,
                    lookback=self.CANDLE_LOOKBACK
                )

                if signal is not None:
                    if best_signal is None or signal.score > best_signal.score:
                        best_signal = signal

            if best_signal is None:
                continue

            signals_found += 1

            # Apply RSI filter (required: RSI < 50)
            if not best_signal.filters_passed:
                signals_filtered += 1
                continue

            # Signal passed all filters -> simulate DOWN bet
            summary.signals_passed_filter += 1

            # Simulate Polymarket DOWN price
            poly_down_price = self._simulate_poly_down_price(
                candles[max(0, w_start - 30):entry_start_idx]
            )

            # Clamp to realistic range
            poly_down_price = max(self.DOWN_PRICE_LOW, min(self.DOWN_PRICE_HIGH, poly_down_price))

            # PnL calculation: binary outcome
            # WIN: payout = $BET * (1 - entry_price) / entry_price ... no, simpler:
            # If DOWN wins: payout = BET_SIZE * (1 / poly_down_price - 1)
            #   e.g. buy DOWN @ $0.50 for $3 -> get 6 shares -> win $6 - $3 = $3
            # If DOWN loses: lose $BET_SIZE
            shares = self.BET_SIZE / poly_down_price
            if actual_down:
                pnl = shares * 1.0 - self.BET_SIZE  # Each share pays $1 if DOWN wins
            else:
                pnl = -self.BET_SIZE

            trade = BacktestResult(
                window_start_time=w_start_dt,
                window_end_time=w_end_dt,
                side="DOWN",
                entry_price_asset=candles[entry_start_idx].close,
                exit_price_asset=window_close_price,
                window_open_price=window_open_price,
                price_change_pct=price_change,
                won=actual_down,
                pnl=pnl,
                poly_entry_price=poly_down_price,
                signal_score=best_signal.score,
                rsi=best_signal.rsi,
                vol_ratio=best_signal.vol_ratio,
                below_ema50=best_signal.below_ema50,
                trendline_touches=best_signal.trendline.touches,
                break_pct=best_signal.break_pct,
                hour_utc=hour
            )

            self.trades.append(trade)

            # Track hourly
            hourly[hour]["signals"] += 1
            hourly[hour]["wins"] += 1 if actual_down else 0
            hourly[hour]["pnl"] += pnl

            # Track score buckets
            bucket = f"{int(best_signal.score // 5) * 5}-{int(best_signal.score // 5) * 5 + 5}"
            score_buckets[bucket]["signals"] += 1
            score_buckets[bucket]["wins"] += 1 if actual_down else 0
            score_buckets[bucket]["pnl"] += pnl

        # Compile summary
        summary.total_signals = signals_found
        wins = sum(1 for t in self.trades if t.won)
        losses = len(self.trades) - wins
        summary.wins = wins
        summary.losses = losses
        summary.win_rate = wins / len(self.trades) * 100 if self.trades else 0
        summary.total_pnl = sum(t.pnl for t in self.trades)
        summary.avg_pnl_per_trade = summary.total_pnl / len(self.trades) if self.trades else 0

        # Max drawdown
        cumulative = 0
        peak = 0
        max_dd = 0
        for t in self.trades:
            cumulative += t.pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        summary.max_drawdown = max_dd

        # Baseline
        summary.baseline_signals = len(baseline_trades)
        summary.baseline_wins = sum(1 for b in baseline_trades if b["won"])
        summary.baseline_wr = summary.baseline_wins / summary.baseline_signals * 100 if baseline_trades else 0
        summary.baseline_pnl = sum(b["pnl"] for b in baseline_trades)

        summary.hourly_breakdown = dict(hourly)
        summary.filter_breakdown = {
            "raw_signals": signals_found,
            "filtered_by_rsi": signals_filtered,
            "passed_all_filters": summary.signals_passed_filter,
            "score_buckets": dict(score_buckets)
        }

        return summary

    def print_report(self, summary: BacktestSummary):
        """Print detailed backtest report."""
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS - TRENDLINE_BREAK Strategy")
        print("=" * 70)

        print(f"\n--- Overview ---")
        print(f"Total 15-min windows scanned:    {summary.total_windows}")
        print(f"Windows after skip-hour filter:  {self.all_windows_checked}")
        print(f"Raw trendline break signals:     {summary.total_signals}")
        print(f"Signals filtered by RSI<50:      {summary.filter_breakdown.get('filtered_by_rsi', 0)}")
        print(f"Signals passing all filters:     {summary.signals_passed_filter}")
        print(f"Signal rate:                     {summary.signals_passed_filter / self.all_windows_checked * 100:.1f}%" if self.all_windows_checked > 0 else "  N/A")

        print(f"\n--- Performance ---")
        print(f"Trades taken:    {len(self.trades)}")
        print(f"Wins:            {summary.wins}")
        print(f"Losses:          {summary.losses}")
        print(f"Win Rate:        {summary.win_rate:.1f}%")
        print(f"Total PnL:       ${summary.total_pnl:+.2f}")
        print(f"Avg PnL/trade:   ${summary.avg_pnl_per_trade:+.2f}")
        print(f"Max Drawdown:    ${summary.max_drawdown:.2f}")

        print(f"\n--- Baseline (bet DOWN every window) ---")
        print(f"Total bets:      {summary.baseline_signals}")
        print(f"Wins:            {summary.baseline_wins}")
        print(f"Win Rate:        {summary.baseline_wr:.1f}%")
        print(f"Total PnL:       ${summary.baseline_pnl:+.2f}")

        # Edge vs baseline
        if summary.baseline_wr > 0 and summary.win_rate > 0:
            edge = summary.win_rate - summary.baseline_wr
            print(f"\n--- Edge vs Baseline ---")
            print(f"WR improvement:  {edge:+.1f}pp")
            print(f"PnL improvement: ${summary.total_pnl - (summary.baseline_pnl * len(self.trades) / max(1, summary.baseline_signals)):+.2f} (normalized)")

        # Hourly breakdown
        if summary.hourly_breakdown:
            print(f"\n--- Hourly Breakdown ---")
            print(f"{'Hour':>6} | {'Signals':>8} | {'Wins':>5} | {'WR':>7} | {'PnL':>10}")
            print("-" * 50)
            for h in sorted(summary.hourly_breakdown.keys()):
                data = summary.hourly_breakdown[h]
                wr = data['wins'] / data['signals'] * 100 if data['signals'] > 0 else 0
                print(f"{h:>4} UTC | {data['signals']:>8} | {data['wins']:>5} | {wr:>6.1f}% | ${data['pnl']:>+9.2f}")

        # Score bucket breakdown
        if summary.filter_breakdown.get("score_buckets"):
            print(f"\n--- Signal Score Breakdown ---")
            print(f"{'Score':>10} | {'Signals':>8} | {'Wins':>5} | {'WR':>7} | {'PnL':>10}")
            print("-" * 55)
            for bucket in sorted(summary.filter_breakdown["score_buckets"].keys()):
                data = summary.filter_breakdown["score_buckets"][bucket]
                wr = data['wins'] / data['signals'] * 100 if data['signals'] > 0 else 0
                print(f"{bucket:>10} | {data['signals']:>8} | {data['wins']:>5} | {wr:>6.1f}% | ${data['pnl']:>+9.2f}")

        # Filter analysis
        if self.trades:
            print(f"\n--- Filter Analysis ---")

            # RSI distribution
            rsi_values = [t.rsi for t in self.trades if t.rsi is not None]
            if rsi_values:
                print(f"RSI (at signal):  avg={statistics.mean(rsi_values):.1f}, "
                      f"median={statistics.median(rsi_values):.1f}, "
                      f"range=[{min(rsi_values):.1f}, {max(rsi_values):.1f}]")

                # RSI < 40 vs RSI 40-50
                rsi_low = [t for t in self.trades if t.rsi is not None and t.rsi < 40]
                rsi_mid = [t for t in self.trades if t.rsi is not None and t.rsi >= 40]
                if rsi_low:
                    wr_low = sum(1 for t in rsi_low if t.won) / len(rsi_low) * 100
                    print(f"  RSI < 40:       {len(rsi_low)} trades, {wr_low:.1f}% WR, "
                          f"${sum(t.pnl for t in rsi_low):+.2f}")
                if rsi_mid:
                    wr_mid = sum(1 for t in rsi_mid if t.won) / len(rsi_mid) * 100
                    print(f"  RSI 40-50:      {len(rsi_mid)} trades, {wr_mid:.1f}% WR, "
                          f"${sum(t.pnl for t in rsi_mid):+.2f}")

            # Volume ratio
            high_vol = [t for t in self.trades if t.vol_ratio > 1.3]
            low_vol = [t for t in self.trades if t.vol_ratio <= 1.3]
            if high_vol:
                wr_hv = sum(1 for t in high_vol if t.won) / len(high_vol) * 100
                print(f"  High vol (>1.3x): {len(high_vol)} trades, {wr_hv:.1f}% WR, "
                      f"${sum(t.pnl for t in high_vol):+.2f}")
            if low_vol:
                wr_lv = sum(1 for t in low_vol if t.won) / len(low_vol) * 100
                print(f"  Low vol (<=1.3x): {len(low_vol)} trades, {wr_lv:.1f}% WR, "
                      f"${sum(t.pnl for t in low_vol):+.2f}")

            # Below EMA50
            ema_below = [t for t in self.trades if t.below_ema50]
            ema_above = [t for t in self.trades if not t.below_ema50]
            if ema_below:
                wr_eb = sum(1 for t in ema_below if t.won) / len(ema_below) * 100
                print(f"  Below EMA50:      {len(ema_below)} trades, {wr_eb:.1f}% WR, "
                      f"${sum(t.pnl for t in ema_below):+.2f}")
            if ema_above:
                wr_ea = sum(1 for t in ema_above if t.won) / len(ema_above) * 100
                print(f"  Above EMA50:      {len(ema_above)} trades, {wr_ea:.1f}% WR, "
                      f"${sum(t.pnl for t in ema_above):+.2f}")

            # Trendline touches
            multi_touch = [t for t in self.trades if t.trendline_touches >= 3]
            dual_touch = [t for t in self.trades if t.trendline_touches == 2]
            if multi_touch:
                wr_mt = sum(1 for t in multi_touch if t.won) / len(multi_touch) * 100
                print(f"  3+ TL touches:    {len(multi_touch)} trades, {wr_mt:.1f}% WR, "
                      f"${sum(t.pnl for t in multi_touch):+.2f}")
            if dual_touch:
                wr_dt = sum(1 for t in dual_touch if t.won) / len(dual_touch) * 100
                print(f"  2 TL touches:     {len(dual_touch)} trades, {wr_dt:.1f}% WR, "
                      f"${sum(t.pnl for t in dual_touch):+.2f}")

        # Top 5 trades
        if self.trades:
            print(f"\n--- Top 5 Winning Trades ---")
            winners = sorted([t for t in self.trades if t.won], key=lambda t: t.pnl, reverse=True)[:5]
            for i, t in enumerate(winners, 1):
                print(f"  {i}. {t.window_start_time.strftime('%m/%d %H:%M')} UTC | "
                      f"PnL: ${t.pnl:+.2f} | "
                      f"Price chg: {t.price_change_pct:+.3f}% | "
                      f"Score: {t.signal_score:.1f} | "
                      f"RSI: {t.rsi:.1f}" if t.rsi else f"RSI: N/A")

            print(f"\n--- Top 5 Losing Trades ---")
            losers = sorted([t for t in self.trades if not t.won], key=lambda t: t.pnl)[:5]
            for i, t in enumerate(losers, 1):
                print(f"  {i}. {t.window_start_time.strftime('%m/%d %H:%M')} UTC | "
                      f"PnL: ${t.pnl:+.2f} | "
                      f"Price chg: {t.price_change_pct:+.3f}% | "
                      f"Score: {t.signal_score:.1f} | "
                      f"RSI: {t.rsi:.1f}" if t.rsi else f"RSI: N/A")

        # Verdict
        print(f"\n" + "=" * 70)
        print(f"VERDICT:")
        if summary.win_rate >= 60:
            print(f"  STRONG - {summary.win_rate:.1f}% WR with ${summary.total_pnl:+.2f} PnL")
            print(f"  Recommendation: Integrate into V3.15 as a signal booster for DOWN bets")
        elif summary.win_rate >= 55:
            print(f"  PROMISING - {summary.win_rate:.1f}% WR with ${summary.total_pnl:+.2f} PnL")
            print(f"  Recommendation: Add as +2 DOWN scoring boost in ta_signals.py")
        elif summary.win_rate >= 50:
            print(f"  MARGINAL - {summary.win_rate:.1f}% WR, barely above coin flip")
            print(f"  Recommendation: Only use with additional confirmation (vol + EMA50)")
        else:
            print(f"  WEAK - {summary.win_rate:.1f}% WR, below coin flip")
            print(f"  Recommendation: Do NOT integrate. Strategy needs rework.")
        print("=" * 70)

    def save_results(self, summary: BacktestSummary, filepath: str = None):
        """Save detailed results to JSON."""
        if filepath is None:
            filepath = str(Path(__file__).parent / "backtest_trendline_results.json")

        output = {
            "strategy": "TRENDLINE_BREAK",
            "run_time": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_windows": summary.total_windows,
                "windows_checked": self.all_windows_checked,
                "total_signals": summary.total_signals,
                "signals_passed": summary.signals_passed_filter,
                "wins": summary.wins,
                "losses": summary.losses,
                "win_rate": round(summary.win_rate, 2),
                "total_pnl": round(summary.total_pnl, 2),
                "avg_pnl": round(summary.avg_pnl_per_trade, 2),
                "max_drawdown": round(summary.max_drawdown, 2),
                "baseline_wr": round(summary.baseline_wr, 2),
                "baseline_pnl": round(summary.baseline_pnl, 2),
            },
            "settings": {
                "skip_hours": list(self.SKIP_HOURS_UTC),
                "entry_window": f"{self.ENTRY_WINDOW_MIN}-{self.ENTRY_WINDOW_MAX} min",
                "bet_size": self.BET_SIZE,
                "break_threshold_pct": self.BREAK_THRESHOLD_PCT,
                "pivot_left": self.PIVOT_LEFT,
                "pivot_right": self.PIVOT_RIGHT,
                "candle_lookback": self.CANDLE_LOOKBACK,
            },
            "hourly": {str(k): v for k, v in summary.hourly_breakdown.items()},
            "filter_breakdown": summary.filter_breakdown,
            "trades": [
                {
                    "time": t.window_start_time.isoformat(),
                    "won": t.won,
                    "pnl": round(t.pnl, 2),
                    "price_change_pct": round(t.price_change_pct, 4),
                    "poly_entry_price": t.poly_entry_price,
                    "score": round(t.signal_score, 2),
                    "rsi": round(t.rsi, 1) if t.rsi else None,
                    "vol_ratio": round(t.vol_ratio, 2),
                    "below_ema50": t.below_ema50,
                    "tl_touches": t.trendline_touches,
                    "break_pct": round(t.break_pct, 4),
                    "hour": t.hour_utc,
                }
                for t in self.trades
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {filepath}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run the trendline break backtest."""
    print("=" * 70)
    print("TRENDLINE BREAK STRATEGY BACKTEST")
    print("Target: Polymarket 15-Minute BTC Up/Down Markets")
    print("Signal: Ascending trendline break -> bet DOWN")
    print("=" * 70)

    bt = TrendlineBacktester()

    # Fetch 7 days of 1-minute BTC candles
    # 7 days = ~10,080 candles = ~672 fifteen-minute windows
    days = 7
    print(f"\nFetching {days} days of BTC 1m candles...")
    candles = await bt.fetch_historical_candles("BTCUSDT", days=days)

    if len(candles) < 500:
        print(f"ERROR: Only got {len(candles)} candles, need at least 500. Aborting.")
        return

    print(f"\nCandle range: {datetime.fromtimestamp(candles[0].timestamp, tz=timezone.utc)} "
          f"to {datetime.fromtimestamp(candles[-1].timestamp, tz=timezone.utc)}")
    print(f"Price range: ${min(c.low for c in candles):,.2f} - ${max(c.high for c in candles):,.2f}")

    # Run backtest
    print("\nRunning backtest...")
    t0 = time.time()
    summary = bt.run_backtest(candles)
    elapsed = time.time() - t0
    print(f"Backtest completed in {elapsed:.1f}s")

    # Print report
    bt.print_report(summary)

    # Save results
    bt.save_results(summary)

    return summary


if __name__ == "__main__":
    asyncio.run(main())
