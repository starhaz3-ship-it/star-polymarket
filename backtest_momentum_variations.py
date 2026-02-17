"""Comprehensive Momentum Backtest using PolyData + Binance klines.

Tests the live momentum_strong strategy with variations on:
- Momentum thresholds (10m/5m)
- RSI filters (none, current, extreme, very extreme)
- EMA gap filters (none, current, tight, wide)
- Entry timing (2, 5, 7, 10 min before close)
- Hour filters (all, skip {7,13}, skip {7,8,13}, etc.)

Focus: BTC 15M markets (the live strategy)
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
print("MOMENTUM VARIATIONS BACKTEST — BTC 15M Markets")
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

# OPTIMIZATION: Only test last 21 days (faster, still ~4000+ markets)
cutoff_date = datetime(2026, 1, 14, 0, 0, 0)  # ~3 weeks of data
parsed_markets = [m for m in parsed_markets if m['candle_start_utc'] >= cutoff_date]
print(f"  Using last 21 days: {len(parsed_markets):,} markets ({parsed_markets[0]['candle_start_utc']} to {parsed_markets[-1]['candle_start_utc']})")

# UP/DOWN balance
up_count = sum(1 for m in parsed_markets if m['outcome'] == 'UP')
dn_count = len(parsed_markets) - up_count
print(f"  Overall: UP {up_count:,} ({up_count/len(parsed_markets)*100:.1f}%) | DOWN {dn_count:,} ({dn_count/len(parsed_markets)*100:.1f}%)")


# ──────────────────────────────────────────────────────────
# 4. LOAD BINANCE KLINES (fetch fresh from API)
# ──────────────────────────────────────────────────────────
print("\nFetching Binance 1-minute klines from API...")

# We need klines from earliest market to latest market
earliest = parsed_markets[0]['candle_start_utc']
latest = parsed_markets[-1]['candle_close_utc']

print(f"  Date range needed: {earliest} to {latest}")

# Binance API allows max 1000 candles per request
# We'll fetch in chunks
import requests

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


def calculate_ema(prices, period):
    """Calculate EMA."""
    return prices.ewm(span=period, adjust=False).mean()


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

    if len(historical) < 21:  # Need at least 21 candles for EMA21
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

    # EMA9 and EMA21
    ema9_series = calculate_ema(closes, 9)
    ema21_series = calculate_ema(closes, 21)

    if pd.isna(ema9_series.iloc[-1]) or pd.isna(ema21_series.iloc[-1]):
        return None

    ema9 = ema9_series.iloc[-1]
    ema21 = ema21_series.iloc[-1]
    ema_gap_bp = (ema9 - ema21) / ema21 * 10000  # basis points

    return {
        'mom_10m': mom_10m,
        'mom_5m': mom_5m,
        'rsi': rsi,
        'ema_gap_bp': ema_gap_bp,
        'current_price': current_price,
    }


# ──────────────────────────────────────────────────────────
# 6. DEFINE STRATEGY VARIATIONS
# ──────────────────────────────────────────────────────────
variations = {
    # A. Momentum thresholds
    'mom_baseline': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': {7, 13}},
    'mom_aggressive': {'mom_10m': 0.05, 'mom_5m': 0.02, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': {7, 13}},
    'mom_conservative': {'mom_10m': 0.12, 'mom_5m': 0.05, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': {7, 13}},
    'mom_strong': {'mom_10m': 0.20, 'mom_5m': 0.08, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': {7, 13}},

    # B. RSI variations
    'rsi_none': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': None, 'rsi_dn': None, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': {7, 13}},
    'rsi_current': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': {7, 13}},
    'rsi_extreme': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 65, 'rsi_dn': 35, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': {7, 13}},
    'rsi_very_extreme': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 70, 'rsi_dn': 30, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': {7, 13}},

    # C. EMA gap filter
    'ema_none': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': None, 'entry_min': 2, 'skip_hours': {7, 13}},
    'ema_current': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': {7, 13}},
    'ema_tight': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 5, 'entry_min': 2, 'skip_hours': {7, 13}},
    'ema_wide': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 10, 'entry_min': 2, 'skip_hours': {7, 13}},

    # D. Entry timing
    'entry_2min': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': {7, 13}},
    'entry_5min': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 5, 'skip_hours': {7, 13}},
    'entry_7min': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 7, 'skip_hours': {7, 13}},
    'entry_10min': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 10, 'skip_hours': {7, 13}},

    # E. Hour filters
    'hours_all': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': set()},
    'hours_skip_7_13': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': {7, 13}},
    'hours_skip_7_8_13': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': {7, 8, 13}},
    'hours_skip_7_to_13': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': {7, 8, 9, 10, 11, 12, 13}},
    'hours_overnight_only': {'mom_10m': 0.08, 'mom_5m': 0.03, 'rsi_up': 55, 'rsi_dn': 45, 'ema_gap': 2, 'entry_min': 2, 'skip_hours': {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}},
}


# ──────────────────────────────────────────────────────────
# 7. RUN BACKTEST FOR ALL VARIATIONS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RUNNING BACKTEST FOR ALL VARIATIONS")
print("=" * 80)

results = {}

for var_name, params in variations.items():
    print(f"\nTesting: {var_name}...")

    signals = 0
    correct = 0
    pnl = 0.0

    for market in parsed_markets:
        # Entry time = close_time - entry_min minutes
        entry_time = market['candle_close_utc'] - timedelta(minutes=params['entry_min'])

        # Skip hours filter
        if market['candle_start_utc'].hour in params['skip_hours']:
            continue

        # Get momentum features
        features = get_momentum_features(btc_klines, entry_time)
        if features is None:
            continue

        # Generate signal
        signal = None

        # UP signal
        if (features['mom_10m'] > params['mom_10m'] and
            features['mom_5m'] > params['mom_5m']):
            # Check RSI
            if params['rsi_up'] is None or features['rsi'] > params['rsi_up']:
                # Check EMA gap
                if params['ema_gap'] is None or features['ema_gap_bp'] > params['ema_gap']:
                    signal = 'UP'

        # DOWN signal
        if (features['mom_10m'] < -params['mom_10m'] and
            features['mom_5m'] < -params['mom_5m']):
            # Check RSI
            if params['rsi_dn'] is None or features['rsi'] < params['rsi_dn']:
                # Check EMA gap
                if params['ema_gap'] is None or features['ema_gap_bp'] < -params['ema_gap']:
                    signal = 'DOWN'

        if signal is None:
            continue

        # Record signal
        signals += 1

        # Check if correct
        actual = market['outcome']
        if signal == actual:
            correct += 1
            # Win: assume $10 bet at 0.50 entry -> +$10 profit
            pnl += 10.0
        else:
            # Loss: -$10
            pnl -= 10.0

    # Store results
    wr = correct / signals * 100 if signals > 0 else 0.0
    results[var_name] = {
        'signals': signals,
        'correct': correct,
        'win_rate': wr,
        'pnl': pnl,
        'avg_pnl_per_trade': pnl / signals if signals > 0 else 0.0,
        'params': params,
    }

    print(f"  Signals: {signals:,} | Correct: {correct:,} | WR: {wr:.2f}% | PnL: ${pnl:+.2f}")


# ──────────────────────────────────────────────────────────
# 8. FIND BEST COMBINATION
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FINDING BEST COMBINATION")
print("=" * 80)

# Best from each dimension
best_mom = max([k for k in results.keys() if k.startswith('mom_')], key=lambda k: results[k]['win_rate'])
best_rsi = max([k for k in results.keys() if k.startswith('rsi_')], key=lambda k: results[k]['win_rate'])
best_ema = max([k for k in results.keys() if k.startswith('ema_')], key=lambda k: results[k]['win_rate'])
best_entry = max([k for k in results.keys() if k.startswith('entry_')], key=lambda k: results[k]['win_rate'])
best_hours = max([k for k in results.keys() if k.startswith('hours_')], key=lambda k: results[k]['win_rate'])

print(f"  Best momentum: {best_mom} (WR: {results[best_mom]['win_rate']:.2f}%)")
print(f"  Best RSI: {best_rsi} (WR: {results[best_rsi]['win_rate']:.2f}%)")
print(f"  Best EMA: {best_ema} (WR: {results[best_ema]['win_rate']:.2f}%)")
print(f"  Best entry: {best_entry} (WR: {results[best_entry]['win_rate']:.2f}%)")
print(f"  Best hours: {best_hours} (WR: {results[best_hours]['win_rate']:.2f}%)")

# Combine best settings
combined_params = {
    'mom_10m': variations[best_mom]['mom_10m'],
    'mom_5m': variations[best_mom]['mom_5m'],
    'rsi_up': variations[best_rsi]['rsi_up'],
    'rsi_dn': variations[best_rsi]['rsi_dn'],
    'ema_gap': variations[best_ema]['ema_gap'],
    'entry_min': variations[best_entry]['entry_min'],
    'skip_hours': variations[best_hours]['skip_hours'],
}

print(f"\nTesting combined best settings...")
print(f"  Params: {combined_params}")

# Run combined backtest
signals = 0
correct = 0
pnl = 0.0

for market in parsed_markets:
    entry_time = market['candle_close_utc'] - timedelta(minutes=combined_params['entry_min'])

    if market['candle_start_utc'].hour in combined_params['skip_hours']:
        continue

    features = get_momentum_features(btc_klines, entry_time)
    if features is None:
        continue

    signal = None

    # UP signal
    if (features['mom_10m'] > combined_params['mom_10m'] and
        features['mom_5m'] > combined_params['mom_5m']):
        if combined_params['rsi_up'] is None or features['rsi'] > combined_params['rsi_up']:
            if combined_params['ema_gap'] is None or features['ema_gap_bp'] > combined_params['ema_gap']:
                signal = 'UP'

    # DOWN signal
    if (features['mom_10m'] < -combined_params['mom_10m'] and
        features['mom_5m'] < -combined_params['mom_5m']):
        if combined_params['rsi_dn'] is None or features['rsi'] < combined_params['rsi_dn']:
            if combined_params['ema_gap'] is None or features['ema_gap_bp'] < -combined_params['ema_gap']:
                signal = 'DOWN'

    if signal is None:
        continue

    signals += 1
    actual = market['outcome']
    if signal == actual:
        correct += 1
        pnl += 10.0
    else:
        pnl -= 10.0

wr = correct / signals * 100 if signals > 0 else 0.0
results['combined_best'] = {
    'signals': signals,
    'correct': correct,
    'win_rate': wr,
    'pnl': pnl,
    'avg_pnl_per_trade': pnl / signals if signals > 0 else 0.0,
    'params': combined_params,
}

print(f"  Signals: {signals:,} | Correct: {correct:,} | WR: {wr:.2f}% | PnL: ${pnl:+.2f}")


# ──────────────────────────────────────────────────────────
# 9. SAVE RESULTS TO JSON
# ──────────────────────────────────────────────────────────
output_path = r"C:/Users/Star/.local/bin/star-polymarket/backtest_momentum_variations_results.json"

# Convert sets to lists for JSON serialization
results_json = {}
for k, v in results.items():
    v_copy = v.copy()
    if 'skip_hours' in v_copy['params']:
        v_copy['params']['skip_hours'] = list(v_copy['params']['skip_hours'])
    results_json[k] = v_copy

with open(output_path, 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"\nResults saved to: {output_path}")


# ──────────────────────────────────────────────────────────
# 10. PRINT SUMMARY TABLE
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

# Sort by win rate
sorted_results = sorted(results.items(), key=lambda x: x[1]['win_rate'], reverse=True)

print(f"\n{'Variation':<25} {'Signals':>8} {'Wins':>8} {'WR%':>8} {'PnL':>10} {'$/Trade':>10}")
print("-" * 80)

for var_name, res in sorted_results:
    print(f"{var_name:<25} {res['signals']:>8,} {res['correct']:>8,} {res['win_rate']:>7.2f}% "
          f"${res['pnl']:>+9.2f} ${res['avg_pnl_per_trade']:>+9.4f}")

print("-" * 80)

# ──────────────────────────────────────────────────────────
# 11. TOP RECOMMENDATIONS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TOP RECOMMENDATIONS")
print("=" * 80)

# Filter for variations with at least 50 signals
qualified = {k: v for k, v in results.items() if v['signals'] >= 50}

if qualified:
    # Best by win rate
    best_wr = max(qualified.items(), key=lambda x: x[1]['win_rate'])
    print(f"\nBest Win Rate: {best_wr[0]}")
    print(f"  WR: {best_wr[1]['win_rate']:.2f}% ({best_wr[1]['correct']:,}/{best_wr[1]['signals']:,})")
    print(f"  PnL: ${best_wr[1]['pnl']:+.2f} (${best_wr[1]['avg_pnl_per_trade']:+.4f}/trade)")
    print(f"  Params: {best_wr[1]['params']}")

    # Best by PnL
    best_pnl = max(qualified.items(), key=lambda x: x[1]['pnl'])
    print(f"\nBest PnL: {best_pnl[0]}")
    print(f"  PnL: ${best_pnl[1]['pnl']:+.2f} (${best_pnl[1]['avg_pnl_per_trade']:+.4f}/trade)")
    print(f"  WR: {best_pnl[1]['win_rate']:.2f}% ({best_pnl[1]['correct']:,}/{best_pnl[1]['signals']:,})")
    print(f"  Params: {best_pnl[1]['params']}")

    # Best sharpe (PnL per trade / volatility approximation)
    # Simple metric: high WR + high signal count
    best_sharpe = max(qualified.items(), key=lambda x: x[1]['win_rate'] * (x[1]['signals'] ** 0.5))
    print(f"\nBest Risk-Adjusted: {best_sharpe[0]}")
    print(f"  WR: {best_sharpe[1]['win_rate']:.2f}% ({best_sharpe[1]['correct']:,}/{best_sharpe[1]['signals']:,})")
    print(f"  PnL: ${best_sharpe[1]['pnl']:+.2f} (${best_sharpe[1]['avg_pnl_per_trade']:+.4f}/trade)")
    print(f"  Params: {best_sharpe[1]['params']}")
else:
    print("\n  No variations with 50+ signals")

print("\n" + "=" * 80)
print("BACKTEST COMPLETE")
print("=" * 80)
