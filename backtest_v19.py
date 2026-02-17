"""V1.9 Strategy Backtest - Contrarian + Moderate Down + UP signals.

Tests the new V1.9 strategy against existing V1.7 and UP-only baselines.

V1.9 Strategy Rules:
1. UP (primary): Bullish momentum (10m > 0.08%, 5m > 0.03%) AND RSI > 65 → bet UP
2. CONTRARIAN: Bearish momentum (10m < -0.08%, 5m < -0.03%) AND RSI 25-35 → bet UP (mean reversion)
3. MODERATE DOWN: Bearish momentum AND RSI 35-50 → bet DOWN
4. Skip hours (UTC): {7, 11, 12, 13, 14, 15} — losing hours
5. Entry price: $0.35-$0.60
6. Entry window: 2-5 min before market close
7. BTC only (no ETH, no SOL)
8. $5 per trade

Comparison:
- V1.7 (old): RSI >65 UP, RSI <35 DOWN, skip hours {7,13}
- UP-only: Just the UP signal with RSI >65
- V1.9 full: All three signal types with new skip hours
"""
import pandas as pd
import numpy as np
import json
import re
import os
import glob
import sys
import time
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Optional

# ──────────────────────────────────────────────────────────
# 1. LOAD POLYDATA MARKETS
# ──────────────────────────────────────────────────────────
POLYDATA = r"C:\Users\Star\Documents\polymarket-assistant-main\PolyData"
MARKET_DIR = os.path.join(POLYDATA, "data", "polymarket", "markets")
KLINE_CACHE = r"C:/Users/Star/.local/bin/star-polymarket/kline_cache"

print("=" * 80)
print("V1.9 STRATEGY BACKTEST — BTC 15M Markets")
print("=" * 80)

print("\nLoading market parquet files...")
market_files = glob.glob(os.path.join(MARKET_DIR, "*.parquet"))
print(f"  Found {len(market_files)} parquet files")
all_markets = pd.concat([pd.read_parquet(f) for f in market_files], ignore_index=True)
print(f"  Total markets in dataset: {len(all_markets):,}")

# Filter for BTC Up or Down with closed status
updown = all_markets[
    all_markets['question'].str.contains('Up or Down', case=False, na=False) &
    all_markets['question'].str.contains('Bitcoin', case=False, na=False) &
    all_markets['closed'].fillna(False)
].copy()
print(f"  BTC Up or Down (closed): {len(updown):,}")

# ──────────────────────────────────────────────────────────
# 2. PARSE CANDLE TIME + OUTCOME
# ──────────────────────────────────────────────────────────
ET = timezone(timedelta(hours=-5))

def parse_question(q: str) -> Optional[dict]:
    """Parse market question into asset, candle_start (UTC), duration_min, outcome."""
    asset = "BTC" if "Bitcoin" in q else None
    if not asset:
        return None

    # Format: "Bitcoin Up or Down - February 16, 1:35PM-1:40PM ET"
    m = re.search(
        r'(\w+)\s+(\d{1,2}),\s*(\d{1,2}):(\d{2})(AM|PM)-(\d{1,2}):(\d{2})(AM|PM)\s+ET',
        q
    )
    if m:
        month_name = m.group(1)
        day = int(m.group(2))
        h1, m1, ap1 = int(m.group(3)), int(m.group(4)), m.group(5)
        h2, m2, ap2 = int(m.group(6)), int(m.group(7)), m.group(8)

        if ap1 == 'PM' and h1 != 12: h1 += 12
        elif ap1 == 'AM' and h1 == 12: h1 = 0
        if ap2 == 'PM' and h2 != 12: h2 += 12
        elif ap2 == 'AM' and h2 == 12: h2 = 0

        dur = (h2 * 60 + m2) - (h1 * 60 + m1)
        if dur < 0:
            dur += 1440  # cross midnight

        # Only 15min candles
        if dur != 15:
            return None

        # Determine year from month
        months = {"January": 1, "February": 2, "March": 3, "April": 4,
                  "May": 5, "June": 6, "July": 7, "August": 8,
                  "September": 9, "October": 10, "November": 11, "December": 12}
        mon = months.get(month_name)
        if not mon:
            return None

        # Polymarket crypto candles started ~Sept 2025
        year = 2025 if mon >= 9 else 2026

        try:
            et_dt = datetime(year, mon, day, h1, m1, tzinfo=ET)
            utc_dt = et_dt.astimezone(timezone.utc).replace(tzinfo=None)
            candle_close_utc = utc_dt + timedelta(minutes=dur)
        except (ValueError, OverflowError):
            return None

        return {"asset": asset, "candle_start_utc": utc_dt, "candle_close_utc": candle_close_utc, "duration_min": dur}

    return None


def parse_outcome(row) -> Optional[str]:
    """Determine if UP or DOWN won from outcome_prices."""
    prices = row.get('outcome_prices', '')
    outcomes = row.get('outcomes', '')

    # Handle string format
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except:
            return None
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except:
            return None

    if not prices or not outcomes or len(prices) < 2 or len(outcomes) < 2:
        return None

    try:
        p0 = float(prices[0])
        p1 = float(prices[1])
    except:
        return None

    # Find which outcome is "Up"
    up_idx = None
    for i, o in enumerate(outcomes):
        if o.lower() == 'up':
            up_idx = i
            break

    if up_idx is None:
        return None

    down_idx = 1 - up_idx
    up_price = float(prices[up_idx])
    down_price = float(prices[down_idx])

    # Resolved: winning side has price ~1.0, losing side ~0.0
    if up_price > 0.9:
        return "UP"
    elif down_price > 0.9:
        return "DOWN"
    return None


# ──────────────────────────────────────────────────────────
# 3. PARSE ALL MARKETS
# ──────────────────────────────────────────────────────────
print("\nParsing BTC 15M markets...")
parsed_markets = []
parse_fails = 0
outcome_fails = 0

for _, row in updown.iterrows():
    info = parse_question(row['question'])
    if not info:
        parse_fails += 1
        continue

    outcome = parse_outcome(row)
    if not outcome:
        outcome_fails += 1
        continue

    info['outcome'] = outcome
    info['question'] = row['question']
    parsed_markets.append(info)

print(f"  Parsed: {len(parsed_markets):,} BTC 15M markets (parse_fail={parse_fails:,}, outcome_fail={outcome_fails:,})")

# Sort by candle_start_utc
parsed_markets.sort(key=lambda x: x['candle_start_utc'])

if parsed_markets:
    print(f"  Full date range: {parsed_markets[0]['candle_start_utc']} to {parsed_markets[-1]['candle_start_utc']}")

# Use last 21 days for faster testing
cutoff_date = datetime(2026, 1, 14, 0, 0, 0)
parsed_markets = [m for m in parsed_markets if m['candle_start_utc'] >= cutoff_date]
print(f"  Using last 21 days: {len(parsed_markets):,} markets ({parsed_markets[0]['candle_start_utc']} to {parsed_markets[-1]['candle_start_utc']})")

# UP/DOWN balance
up_count = sum(1 for m in parsed_markets if m['outcome'] == 'UP')
dn_count = len(parsed_markets) - up_count
print(f"  Overall: UP {up_count:,} ({up_count/len(parsed_markets)*100:.1f}%) | DOWN {dn_count:,} ({dn_count/len(parsed_markets)*100:.1f}%)")


# ──────────────────────────────────────────────────────────
# 4. LOAD BINANCE KLINES (fetch from API)
# ──────────────────────────────────────────────────────────
print("\nFetching Binance 1-minute klines from API...")

earliest = parsed_markets[0]['candle_start_utc']
latest = parsed_markets[-1]['candle_close_utc']

print(f"  Date range needed: {earliest} to {latest}")

def fetch_binance_klines(symbol, start_time, end_time, interval='1m'):
    """Fetch klines from Binance API in chunks."""
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []

    current_start = int(start_time.timestamp() * 1000)  # ms
    end_ms = int(end_time.timestamp() * 1000)

    while current_start < end_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': 1000,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_klines.extend(data)

            # Next batch starts at the last timestamp + 1
            current_start = data[-1][0] + 60000  # +1 minute in ms

            print(f"  Fetched {len(data)} candles (total: {len(all_klines):,})", end='\r')
            time.sleep(0.1)  # Rate limit

        except Exception as e:
            print(f"\n  Error fetching klines: {e}")
            break

    print()  # newline
    return all_klines

# Fetch BTC klines
print("  Fetching BTCUSDT 1m klines...")
raw_klines = fetch_binance_klines('BTCUSDT', earliest, latest + timedelta(hours=1))
print(f"  Fetched {len(raw_klines):,} candles")

# Convert to DataFrame
btc_klines = pd.DataFrame(raw_klines, columns=[
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
    'taker_buy_quote', 'ignore'
])

# Convert timestamps to datetime
btc_klines['open_time'] = pd.to_datetime(btc_klines['open_time'], unit='ms', utc=True)
btc_klines['close_time'] = pd.to_datetime(btc_klines['close_time'], unit='ms', utc=True)

# Convert to timezone-naive UTC
btc_klines['open_time'] = btc_klines['open_time'].dt.tz_localize(None)
btc_klines['close_time'] = btc_klines['close_time'].dt.tz_localize(None)

# Set index to open_time
btc_klines.set_index('open_time', inplace=True)

# Convert price/volume columns to float
for col in ['open', 'high', 'low', 'close', 'volume']:
    btc_klines[col] = btc_klines[col].astype(float)

print(f"  Date range: {btc_klines.index[0]} to {btc_klines.index[-1]}")
print(f"  Total candles: {len(btc_klines):,}")


# ──────────────────────────────────────────────────────────
# 5. MOMENTUM CALCULATION FUNCTIONS
# ──────────────────────────────────────────────────────────
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    deltas = prices.diff()
    gain = deltas.where(deltas > 0, 0.0)
    loss = -deltas.where(deltas < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_momentum_features(klines, entry_time):
    """Calculate momentum features at entry_time.

    Args:
        klines: DataFrame with datetime index and 'close' column
        entry_time: datetime object (timezone-naive UTC)

    Returns:
        dict with momentum features or None if insufficient data
    """
    # Get all data up to (but not including) entry_time
    historical = klines[klines.index < entry_time]

    if len(historical) < 21:
        return None

    # Get the last close price at entry time
    closes = historical['close']
    current_price = closes.iloc[-1]

    # 10-minute momentum (10 candles back)
    if len(closes) >= 11:
        price_10m_ago = closes.iloc[-11]
        mom_10m = (current_price - price_10m_ago) / price_10m_ago * 100  # percent
    else:
        return None

    # 5-minute momentum (5 candles back)
    if len(closes) >= 6:
        price_5m_ago = closes.iloc[-6]
        mom_5m = (current_price - price_5m_ago) / price_5m_ago * 100  # percent
    else:
        return None

    # RSI(14)
    rsi_series = calculate_rsi(closes, period=14)
    if pd.isna(rsi_series.iloc[-1]):
        return None
    rsi = rsi_series.iloc[-1]

    return {
        'mom_10m': mom_10m,
        'mom_5m': mom_5m,
        'rsi': rsi,
        'current_price': current_price,
    }


# ──────────────────────────────────────────────────────────
# 6. DEFINE STRATEGY VARIATIONS
# ──────────────────────────────────────────────────────────

def backtest_strategy(name, signal_func, skip_hours, entry_min_range=(2, 5)):
    """Run backtest for a strategy.

    Args:
        name: Strategy name
        signal_func: Function that takes features and returns ('UP', 'DOWN', or None)
        skip_hours: Set of UTC hours to skip
        entry_min_range: Tuple of (min_minutes, max_minutes) before close

    Returns:
        dict with results + breakdown by signal type
    """
    print(f"\nTesting: {name}...")

    signals = 0
    correct = 0
    pnl = 0.0

    # Track by signal type (for V1.9)
    signal_breakdown = defaultdict(lambda: {'total': 0, 'wins': 0, 'pnl': 0.0})

    for market in parsed_markets:
        # Entry window: 2-5 min before close
        # Try entry times from 5 min down to 2 min before close
        entry_min = None
        features = None

        for mins in range(entry_min_range[1], entry_min_range[0] - 1, -1):
            entry_time = market['candle_close_utc'] - timedelta(minutes=mins)
            features = get_momentum_features(btc_klines, entry_time)
            if features is not None:
                entry_min = mins
                break

        if features is None:
            continue

        # Skip hours filter
        if market['candle_start_utc'].hour in skip_hours:
            continue

        # Generate signal
        signal_result = signal_func(features)

        if signal_result is None:
            continue

        # Extract signal and signal_type
        if isinstance(signal_result, tuple):
            signal, signal_type = signal_result
        else:
            signal = signal_result
            signal_type = 'default'

        # Record signal
        signals += 1
        signal_breakdown[signal_type]['total'] += 1

        # Check if correct
        actual = market['outcome']
        if signal == actual:
            correct += 1
            # Win: $5 bet at avg 0.475 entry -> profit = 5/0.475 - 5 = ~$5.26
            # Simplified: +$5 profit
            profit = 5.0
            pnl += profit
            signal_breakdown[signal_type]['wins'] += 1
            signal_breakdown[signal_type]['pnl'] += profit
        else:
            # Loss: -$5
            loss = -5.0
            pnl += loss
            signal_breakdown[signal_type]['pnl'] += loss

    # Calculate stats
    wr = correct / signals * 100 if signals > 0 else 0.0

    # Calculate breakdown stats
    breakdown_stats = {}
    for sig_type, stats in signal_breakdown.items():
        wr_type = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0.0
        breakdown_stats[sig_type] = {
            'total': stats['total'],
            'wins': stats['wins'],
            'losses': stats['total'] - stats['wins'],
            'win_rate': wr_type,
            'pnl': stats['pnl'],
            'pnl_per_trade': stats['pnl'] / stats['total'] if stats['total'] > 0 else 0.0,
        }

    result = {
        'signals': signals,
        'correct': correct,
        'win_rate': wr,
        'pnl': pnl,
        'pnl_per_trade': pnl / signals if signals > 0 else 0.0,
        'breakdown': breakdown_stats,
    }

    print(f"  Total: {signals:,} trades | {correct:,} wins | WR: {wr:.2f}% | PnL: ${pnl:+.2f}")

    # Print breakdown
    if len(breakdown_stats) > 1:
        for sig_type, stats in breakdown_stats.items():
            print(f"    {sig_type}: {stats['total']:,} trades | {stats['wins']:,} wins | "
                  f"WR: {stats['win_rate']:.2f}% | PnL: ${stats['pnl']:+.2f}")

    return result


# ──────────────────────────────────────────────────────────
# 7. DEFINE SIGNAL FUNCTIONS
# ──────────────────────────────────────────────────────────

def v17_signal(features):
    """V1.7 strategy: RSI >65 UP, RSI <35 DOWN."""
    if features['rsi'] > 65:
        return 'UP', 'up_rsi65'
    elif features['rsi'] < 35:
        return 'DOWN', 'down_rsi35'
    return None


def up_only_signal(features):
    """UP-only: Bullish momentum + RSI >65."""
    if (features['mom_10m'] > 0.08 and
        features['mom_5m'] > 0.03 and
        features['rsi'] > 65):
        return 'UP', 'up_primary'
    return None


def v19_signal(features):
    """V1.9 full strategy: UP primary + CONTRARIAN + MODERATE DOWN."""
    # 1. UP signal (primary): Bullish momentum + RSI > 65
    if (features['mom_10m'] > 0.08 and
        features['mom_5m'] > 0.03 and
        features['rsi'] > 65):
        return 'UP', 'up_primary'

    # 2. CONTRARIAN: Bearish momentum + RSI 25-35 → bet UP (mean reversion)
    if (features['mom_10m'] < -0.08 and
        features['mom_5m'] < -0.03 and
        25 <= features['rsi'] <= 35):
        return 'UP', 'contrarian'

    # 3. MODERATE DOWN: Bearish momentum + RSI 35-50 → bet DOWN
    if (features['mom_10m'] < -0.08 and
        features['mom_5m'] < -0.03 and
        35 < features['rsi'] <= 50):
        return 'DOWN', 'moderate_down'

    return None


# ──────────────────────────────────────────────────────────
# 8. RUN ALL BACKTESTS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RUNNING BACKTESTS")
print("=" * 80)

results = {}

# V1.7 (old strategy): skip hours {7, 13}
results['v1.7'] = backtest_strategy(
    'V1.7 (RSI >65 UP, RSI <35 DOWN)',
    v17_signal,
    skip_hours={7, 13},
    entry_min_range=(2, 5)
)

# UP-only: skip hours {7, 11, 12, 13, 14, 15}
results['up_only'] = backtest_strategy(
    'UP-only (no contrarian, no DOWN)',
    up_only_signal,
    skip_hours={7, 11, 12, 13, 14, 15},
    entry_min_range=(2, 5)
)

# V1.9 full: skip hours {7, 11, 12, 13, 14, 15}
results['v1.9_full'] = backtest_strategy(
    'V1.9 Full (UP + Contrarian + Moderate DOWN)',
    v19_signal,
    skip_hours={7, 11, 12, 13, 14, 15},
    entry_min_range=(2, 5)
)

# V1.9 without contrarian (UP primary + Moderate DOWN only)
def v19_no_contrarian_signal(features):
    """V1.9 without contrarian: UP primary + MODERATE DOWN only."""
    # 1. UP signal (primary): Bullish momentum + RSI > 65
    if (features['mom_10m'] > 0.08 and
        features['mom_5m'] > 0.03 and
        features['rsi'] > 65):
        return 'UP', 'up_primary'

    # 3. MODERATE DOWN: Bearish momentum + RSI 35-50 → bet DOWN
    if (features['mom_10m'] < -0.08 and
        features['mom_5m'] < -0.03 and
        35 < features['rsi'] <= 50):
        return 'DOWN', 'moderate_down'

    return None

results['v1.9_no_contrarian'] = backtest_strategy(
    'V1.9 No Contrarian (UP + Moderate DOWN)',
    v19_no_contrarian_signal,
    skip_hours={7, 11, 12, 13, 14, 15},
    entry_min_range=(2, 5)
)


# ──────────────────────────────────────────────────────────
# 9. PRINT SUMMARY
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n{'Strategy':<40} {'Trades':>8} {'Wins':>8} {'WR%':>8} {'PnL':>10} {'$/Trade':>10}")
print("-" * 90)

for name, res in results.items():
    print(f"{name:<40} {res['signals']:>8,} {res['correct']:>8,} {res['win_rate']:>7.2f}% "
          f"${res['pnl']:>+9.2f} ${res['pnl_per_trade']:>+9.4f}")

print("-" * 90)


# ──────────────────────────────────────────────────────────
# 10. DETAILED BREAKDOWN
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("V1.9 SIGNAL BREAKDOWN")
print("=" * 80)

v19_breakdown = results['v1.9_full']['breakdown']
print(f"\n{'Signal Type':<20} {'Trades':>8} {'Wins':>8} {'Losses':>8} {'WR%':>8} {'PnL':>10} {'$/Trade':>10}")
print("-" * 90)

for sig_type, stats in v19_breakdown.items():
    print(f"{sig_type:<20} {stats['total']:>8,} {stats['wins']:>8,} {stats['losses']:>8,} "
          f"{stats['win_rate']:>7.2f}% ${stats['pnl']:>+9.2f} ${stats['pnl_per_trade']:>+9.4f}")

print("-" * 90)


# ──────────────────────────────────────────────────────────
# 11. SAVE RESULTS
# ──────────────────────────────────────────────────────────
output_path = r"C:/Users/Star/.local/bin/star-polymarket/backtest_v19_results.json"

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_path}")


# ──────────────────────────────────────────────────────────
# 12. RECOMMENDATIONS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

# Find best strategy
best = max(results.items(), key=lambda x: x[1]['win_rate'])
print(f"\nBest Win Rate: {best[0]}")
print(f"  WR: {best[1]['win_rate']:.2f}% ({best[1]['correct']:,}/{best[1]['signals']:,})")
print(f"  PnL: ${best[1]['pnl']:+.2f} (${best[1]['pnl_per_trade']:+.4f}/trade)")

# Compare V1.9 to baselines
v19 = results['v1.9_full']
v17 = results['v1.7']
up_only = results['up_only']

print(f"\nV1.9 vs V1.7:")
print(f"  WR improvement: {v19['win_rate'] - v17['win_rate']:+.2f}%")
print(f"  PnL improvement: ${v19['pnl'] - v17['pnl']:+.2f}")
print(f"  Trade volume: {v19['signals']} vs {v17['signals']} ({v19['signals'] - v17['signals']:+,})")

print(f"\nV1.9 vs UP-only:")
print(f"  WR improvement: {v19['win_rate'] - up_only['win_rate']:+.2f}%")
print(f"  PnL improvement: ${v19['pnl'] - up_only['pnl']:+.2f}")
print(f"  Trade volume: {v19['signals']} vs {up_only['signals']} ({v19['signals'] - up_only['signals']:+,})")

# V1.9 no contrarian vs V1.9 full
if 'v1.9_no_contrarian' in results:
    v19_nc = results['v1.9_no_contrarian']
    print(f"\nV1.9 No Contrarian vs V1.9 Full:")
    print(f"  WR improvement: {v19_nc['win_rate'] - v19['win_rate']:+.2f}%")
    print(f"  PnL improvement: ${v19_nc['pnl'] - v19['pnl']:+.2f}")
    print(f"  Trade volume: {v19_nc['signals']} vs {v19['signals']} ({v19_nc['signals'] - v19['signals']:+,})")

# Check contrarian and moderate_down performance
if 'contrarian' in v19_breakdown:
    contr = v19_breakdown['contrarian']
    print(f"\nContrarian Signal Performance:")
    print(f"  Trades: {contr['total']:,}")
    print(f"  WR: {contr['win_rate']:.2f}%")
    print(f"  PnL: ${contr['pnl']:+.2f} (${contr['pnl_per_trade']:+.4f}/trade)")

    if contr['win_rate'] < 50:
        print(f"  WARNING: Contrarian has <50% WR - consider removing")
    elif contr['win_rate'] > 60:
        print(f"  STRONG: Contrarian adds value!")

if 'moderate_down' in v19_breakdown:
    mod_down = v19_breakdown['moderate_down']
    print(f"\nModerate DOWN Signal Performance:")
    print(f"  Trades: {mod_down['total']:,}")
    print(f"  WR: {mod_down['win_rate']:.2f}%")
    print(f"  PnL: ${mod_down['pnl']:+.2f} (${mod_down['pnl_per_trade']:+.4f}/trade)")

    if mod_down['win_rate'] < 50:
        print(f"  WARNING: Moderate DOWN has <50% WR - consider removing")
    elif mod_down['win_rate'] > 60:
        print(f"  STRONG: Moderate DOWN adds value!")

print("\n" + "=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)
