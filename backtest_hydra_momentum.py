"""
Hydra + Momentum Strategy Backtest - BTC 15-Minute Markets

Compares:
1. V1.9 Momentum Baseline (momentum + RSI only)
2. Momentum + Hydra Confirmation (1+ Hydra strategies agree)
3. Momentum + Hydra Consensus (2+ Hydra strategies agree)
4. Hydra-Only (3+ Hydra strategies agree, no momentum required)
5. Momentum OR Hydra (either momentum triggers OR 3+ Hydra strategies agree)

Uses same infrastructure as backtest_v19.py + Hydra scan_strategies().
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
from typing import Optional, List
from dataclasses import dataclass

# Import Hydra strategies
sys.path.insert(0, os.path.dirname(__file__))
from hydra_strategies import scan_strategies, StrategySignal

# Import Candle class from hydra_strategies
@dataclass
class Candle:
    """1-minute candle for Hydra strategies."""
    timestamp: int  # Unix timestamp in milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD POLYDATA MARKETS
# ──────────────────────────────────────────────────────────────────────────────
POLYDATA = r"C:\Users\Star\Documents\polymarket-assistant-main\PolyData"
MARKET_DIR = os.path.join(POLYDATA, "data", "polymarket", "markets")

print("=" * 80)
print("HYDRA + MOMENTUM STRATEGY BACKTEST — BTC 15M Markets")
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

# ──────────────────────────────────────────────────────────────────────────────
# 2. PARSE CANDLE TIME + OUTCOME
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# 3. PARSE ALL MARKETS
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# 4. LOAD BINANCE KLINES
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# 5. MOMENTUM CALCULATION FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
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

    if len(historical) < 100:  # Need 100+ candles for Hydra strategies
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

    # Convert to Hydra Candle format (last 100 candles for strategy evaluation)
    hydra_candles = []
    for idx in range(max(0, len(historical) - 100), len(historical)):
        row = historical.iloc[idx]
        candle = Candle(
            timestamp=int(historical.index[idx].timestamp() * 1000),
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        hydra_candles.append(candle)

    return {
        'mom_10m': mom_10m,
        'mom_5m': mom_5m,
        'rsi': rsi,
        'current_price': current_price,
        'hydra_candles': hydra_candles,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. DEFINE STRATEGY VARIATIONS
# ──────────────────────────────────────────────────────────────────────────────

# Hydra strategies to use for confluence
HYDRA_STRATEGIES = [
    "TRENDLINE_BREAK", "FVG_RETEST", "DC03_KALMAN_ADX", "BB_BOUNCE",
    "RUBBER_BAND", "MULTI_SMA_TREND", "EXTREME_MOMENTUM", "ALIGNMENT"
]

# V1.9 momentum thresholds
MIN_MOMENTUM_10M = 0.0008  # 0.08%
MIN_MOMENTUM_5M = 0.0003   # 0.03%
RSI_CONFIRM_UP = 60
RSI_MODERATE_DOWN_LO = 30
RSI_MODERATE_DOWN_HI = 50

# Entry price range
MIN_ENTRY = 0.35
MAX_ENTRY = 0.60

# Skip hours (UTC)
SKIP_HOURS = {7, 11, 12, 13, 14, 15}


def v19_momentum_signal(features) -> tuple:
    """V1.9 momentum baseline: momentum + RSI only.
    Returns (signal, signal_type) or None."""
    mom_10m = features['mom_10m'] / 100  # convert to fraction
    mom_5m = features['mom_5m'] / 100
    rsi = features['rsi']

    # UP signal: Bullish momentum + RSI > 60
    if mom_10m > MIN_MOMENTUM_10M and mom_5m > MIN_MOMENTUM_5M and rsi > RSI_CONFIRM_UP:
        return ('UP', 'momentum_up')

    # Moderate DOWN: Bearish momentum + RSI 30-50
    if mom_10m < -MIN_MOMENTUM_10M and mom_5m < -MIN_MOMENTUM_5M:
        if RSI_MODERATE_DOWN_LO < rsi <= RSI_MODERATE_DOWN_HI:
            return ('DOWN', 'momentum_down')

    return None


def get_hydra_signals(hydra_candles: List[Candle]) -> List[StrategySignal]:
    """Get Hydra strategy signals for BTC."""
    candles_dict = {"BTC": hydra_candles}
    all_signals = scan_strategies(candles_dict)

    # Filter to only use specified strategies and unfiltered signals
    filtered_signals = [
        s for s in all_signals
        if s.name in HYDRA_STRATEGIES and not s.filtered and s.asset == "BTC"
    ]
    return filtered_signals


def backtest_variant(name: str, signal_func, entry_min_range=(2, 5)):
    """Run backtest for a strategy variant.

    Args:
        name: Variant name
        signal_func: Function that takes features and returns ('UP'/'DOWN', signal_type) or None
        entry_min_range: Tuple of (min_minutes, max_minutes) before close

    Returns:
        dict with results
    """
    print(f"\nTesting: {name}...")

    signals = 0
    correct = 0
    pnl = 0.0
    trades_log = []

    for market in parsed_markets:
        # Entry window: 2-5 min before close
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
        if market['candle_start_utc'].hour in SKIP_HOURS:
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

        # Check if correct
        actual = market['outcome']
        won = (signal == actual)

        if won:
            correct += 1
            # Win: $5 bet -> ~$5 profit
            profit = 5.0
            pnl += profit
        else:
            # Loss: -$5
            loss = -5.0
            pnl += loss

        trades_log.append({
            'time': market['candle_start_utc'].isoformat(),
            'signal': signal,
            'actual': actual,
            'won': won,
            'pnl': profit if won else loss,
            'signal_type': signal_type,
        })

    # Calculate stats
    wr = correct / signals * 100 if signals > 0 else 0.0
    pnl_per_trade = pnl / signals if signals > 0 else 0.0

    result = {
        'trades': signals,
        'wins': correct,
        'losses': signals - correct,
        'win_rate': wr,
        'pnl': pnl,
        'pnl_per_trade': pnl_per_trade,
        'trades_log': trades_log,
    }

    print(f"  {signals:,} trades | {correct:,} wins | WR: {wr:.2f}% | PnL: ${pnl:+.2f} | $/trade: ${pnl_per_trade:+.2f}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 7. DEFINE VARIANT SIGNAL FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def variant1_momentum_only(features):
    """Variant 1: Momentum + RSI only (V1.9 baseline)."""
    return v19_momentum_signal(features)


def variant2_momentum_hydra_confirm(features):
    """Variant 2: Momentum + Hydra confirmation (1+ Hydra strategies agree)."""
    # First check momentum
    mom_signal = v19_momentum_signal(features)
    if mom_signal is None:
        return None

    direction, signal_type = mom_signal

    # Check Hydra confirmation
    hydra_signals = get_hydra_signals(features['hydra_candles'])
    hydra_agrees = [s for s in hydra_signals if s.direction == direction]

    if len(hydra_agrees) >= 1:
        return (direction, f'{signal_type}+hydra_confirm')

    return None


def variant3_momentum_hydra_consensus(features):
    """Variant 3: Momentum + Hydra consensus (2+ Hydra strategies agree)."""
    # First check momentum
    mom_signal = v19_momentum_signal(features)
    if mom_signal is None:
        return None

    direction, signal_type = mom_signal

    # Check Hydra consensus
    hydra_signals = get_hydra_signals(features['hydra_candles'])
    hydra_agrees = [s for s in hydra_signals if s.direction == direction]

    if len(hydra_agrees) >= 2:
        return (direction, f'{signal_type}+hydra_consensus')

    return None


def variant4_hydra_only(features):
    """Variant 4: Hydra-only signals (3+ Hydra strategies agree, no momentum required)."""
    # Get Hydra signals
    hydra_signals = get_hydra_signals(features['hydra_candles'])

    # Count UP and DOWN signals
    up_signals = [s for s in hydra_signals if s.direction == "UP"]
    down_signals = [s for s in hydra_signals if s.direction == "DOWN"]

    if len(up_signals) >= 3:
        return ('UP', 'hydra_only_up')
    elif len(down_signals) >= 3:
        return ('DOWN', 'hydra_only_down')

    return None


def variant5_momentum_or_hydra(features):
    """Variant 5: Momentum OR Hydra (either momentum triggers OR 3+ Hydra strategies agree)."""
    # Check momentum first
    mom_signal = v19_momentum_signal(features)

    # Check Hydra consensus
    hydra_signals = get_hydra_signals(features['hydra_candles'])
    up_signals = [s for s in hydra_signals if s.direction == "UP"]
    down_signals = [s for s in hydra_signals if s.direction == "DOWN"]

    # Momentum triggers
    if mom_signal is not None:
        direction, signal_type = mom_signal
        return (direction, signal_type)

    # Hydra consensus triggers
    if len(up_signals) >= 3:
        return ('UP', 'hydra_only_up')
    elif len(down_signals) >= 3:
        return ('DOWN', 'hydra_only_down')

    return None


# ──────────────────────────────────────────────────────────────────────────────
# 8. RUN ALL BACKTESTS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RUNNING BACKTESTS")
print("=" * 80)

results = {}

# Variant 1: Momentum alone (V1.9 baseline)
results['variant1_momentum_only'] = backtest_variant(
    'Variant 1: Momentum Alone (V1.9 Baseline)',
    variant1_momentum_only,
    entry_min_range=(2, 5)
)

# Variant 2: Momentum + Hydra confirmation
results['variant2_momentum_hydra_confirm'] = backtest_variant(
    'Variant 2: Momentum + Hydra Confirmation (1+ agree)',
    variant2_momentum_hydra_confirm,
    entry_min_range=(2, 5)
)

# Variant 3: Momentum + Hydra consensus
results['variant3_momentum_hydra_consensus'] = backtest_variant(
    'Variant 3: Momentum + Hydra Consensus (2+ agree)',
    variant3_momentum_hydra_consensus,
    entry_min_range=(2, 5)
)

# Variant 4: Hydra-only
results['variant4_hydra_only'] = backtest_variant(
    'Variant 4: Hydra-Only (3+ agree, no momentum)',
    variant4_hydra_only,
    entry_min_range=(2, 5)
)

# Variant 5: Momentum OR Hydra
results['variant5_momentum_or_hydra'] = backtest_variant(
    'Variant 5: Momentum OR Hydra (either triggers)',
    variant5_momentum_or_hydra,
    entry_min_range=(2, 5)
)


# ──────────────────────────────────────────────────────────────────────────────
# 9. PRINT SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n{'Variant':<50} {'Trades':>8} {'Wins':>8} {'Losses':>8} {'WR%':>8} {'PnL':>10} {'$/Trade':>10}")
print("-" * 110)

for variant_key in ['variant1_momentum_only', 'variant2_momentum_hydra_confirm',
                     'variant3_momentum_hydra_consensus', 'variant4_hydra_only',
                     'variant5_momentum_or_hydra']:
    res = results[variant_key]
    variant_name = variant_key.replace('variant', 'V').replace('_', ' ').title()
    print(f"{variant_name:<50} {res['trades']:>8,} {res['wins']:>8,} {res['losses']:>8,} "
          f"{res['win_rate']:>7.2f}% ${res['pnl']:>+9.2f} ${res['pnl_per_trade']:>+9.4f}")

print("-" * 110)


# ──────────────────────────────────────────────────────────────────────────────
# 10. SAVE RESULTS
# ──────────────────────────────────────────────────────────────────────────────
output_path = r"C:/Users/Star/.local/bin/star-polymarket/backtest_hydra_momentum_results.json"

# Remove trades_log from results before saving (too large)
results_summary = {}
for key, res in results.items():
    results_summary[key] = {
        'trades': res['trades'],
        'wins': res['wins'],
        'losses': res['losses'],
        'win_rate': res['win_rate'],
        'pnl': res['pnl'],
        'pnl_per_trade': res['pnl_per_trade'],
    }

with open(output_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResults saved to: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 11. RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

# Find best variant
best = max(results.items(), key=lambda x: x[1]['win_rate'])
print(f"\nBest Win Rate: {best[0]}")
print(f"  WR: {best[1]['win_rate']:.2f}% ({best[1]['wins']:,}/{best[1]['trades']:,})")
print(f"  PnL: ${best[1]['pnl']:+.2f} (${best[1]['pnl_per_trade']:+.4f}/trade)")

# Best profitability
best_pnl = max(results.items(), key=lambda x: x[1]['pnl'])
print(f"\nBest Profitability: {best_pnl[0]}")
print(f"  PnL: ${best_pnl[1]['pnl']:+.2f} (${best_pnl[1]['pnl_per_trade']:+.4f}/trade)")
print(f"  WR: {best_pnl[1]['win_rate']:.2f}% ({best_pnl[1]['wins']:,}/{best_pnl[1]['trades']:,})")

# Most volume
most_volume = max(results.items(), key=lambda x: x[1]['trades'])
print(f"\nMost Volume: {most_volume[0]}")
print(f"  Trades: {most_volume[1]['trades']:,}")
print(f"  WR: {most_volume[1]['win_rate']:.2f}%")
print(f"  PnL: ${most_volume[1]['pnl']:+.2f}")

# Compare to baseline
baseline = results['variant1_momentum_only']
print(f"\nBaseline (V1.9 Momentum): {baseline['trades']:,} trades, {baseline['win_rate']:.2f}% WR, ${baseline['pnl']:+.2f}")

for variant_key in ['variant2_momentum_hydra_confirm', 'variant3_momentum_hydra_consensus',
                     'variant4_hydra_only', 'variant5_momentum_or_hydra']:
    res = results[variant_key]
    wr_delta = res['win_rate'] - baseline['win_rate']
    pnl_delta = res['pnl'] - baseline['pnl']
    trade_delta = res['trades'] - baseline['trades']

    print(f"\n{variant_key}:")
    print(f"  WR change: {wr_delta:+.2f}% | PnL change: ${pnl_delta:+.2f} | Trade change: {trade_delta:+,}")

    if res['win_rate'] > baseline['win_rate'] and res['pnl'] > baseline['pnl']:
        print(f"  [+] IMPROVEMENT: Better WR and PnL than baseline")
    elif res['win_rate'] > baseline['win_rate']:
        print(f"  [~] MIXED: Better WR but worse/same PnL")
    elif res['pnl'] > baseline['pnl']:
        print(f"  [~] MIXED: Better PnL but worse/same WR")
    else:
        print(f"  [-] WORSE: Lower WR and PnL than baseline")

print("\n" + "=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)
