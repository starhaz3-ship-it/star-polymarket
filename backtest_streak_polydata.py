"""Streak Reversal Backtest using PolyData (50K+ resolved BTC/ETH candle markets).

Tests the hypothesis: after N consecutive same-direction outcomes (e.g., 3 UPs in a row),
does the NEXT candle tend to reverse (mean reversion) or continue (momentum)?

Strategy variants:
  - REVERSAL: Bet AGAINST the streak after N consecutive same-direction outcomes
  - CONTINUATION: Bet WITH the streak (baseline comparison)
  - Streak lengths: N = 2, 3, 4, 5, 6+

Groups: BTC 5M, BTC 15M, ETH 5M, ETH 15M (independent sequences)
"""
import pandas as pd
import numpy as np
import json, re, os, glob, sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Optional
from scipy import stats as scipy_stats

# ──────────────────────────────────────────────────────────
# 1. LOAD POLYDATA MARKETS
# ──────────────────────────────────────────────────────────
POLYDATA = r"C:\Users\Star\Documents\polymarket-assistant-main\PolyData"
MARKET_DIR = os.path.join(POLYDATA, "data", "polymarket", "markets")

print("=" * 80)
print("STREAK REVERSAL BACKTEST — PolyData BTC/ETH 5M/15M Candle Markets")
print("=" * 80)

print("\nLoading market parquet files...")
market_files = glob.glob(os.path.join(MARKET_DIR, "*.parquet"))
print(f"  Found {len(market_files)} parquet files")
all_markets = pd.concat([pd.read_parquet(f) for f in market_files], ignore_index=True)
print(f"  Total markets in dataset: {len(all_markets):,}")

# Filter for BTC/ETH Up or Down with closed status
updown = all_markets[
    all_markets['question'].str.contains('Up or Down', case=False, na=False) &
    all_markets['question'].str.contains('Bitcoin|Ethereum', case=False, na=False) &
    all_markets['closed'].fillna(False)
].copy()
print(f"  BTC/ETH Up or Down (closed): {len(updown):,}")

# ──────────────────────────────────────────────────────────
# 2. PARSE CANDLE TIME + OUTCOME (copied from backtest_skew_polydata.py)
# ──────────────────────────────────────────────────────────
ET = timezone(timedelta(hours=-5))

def parse_question(q: str) -> Optional[dict]:
    """Parse market question into asset, candle_start (UTC), duration_min, outcome."""
    asset = "BTC" if "Bitcoin" in q else "ETH" if "Ethereum" in q else None
    if not asset:
        return None

    # Try format: "Bitcoin Up or Down - February 16, 1:35PM-1:40PM ET"
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

        # Only 5min and 15min candles
        if dur not in (5, 15):
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
        except (ValueError, OverflowError):
            return None

        return {"asset": asset, "candle_start_utc": utc_dt, "duration_min": dur}

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
print("\nParsing markets...")
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

print(f"  Parsed: {len(parsed_markets):,} markets (parse_fail={parse_fails:,}, outcome_fail={outcome_fails:,})")

# Sort by candle_start_utc
parsed_markets.sort(key=lambda x: x['candle_start_utc'])

if parsed_markets:
    print(f"  Date range: {parsed_markets[0]['candle_start_utc']} to {parsed_markets[-1]['candle_start_utc']}")

# Breakdown
groups_summary = {}
for asset in ['BTC', 'ETH']:
    for dur in [5, 15]:
        key = f"{asset}_{dur}M"
        cnt = sum(1 for m in parsed_markets if m['asset'] == asset and m['duration_min'] == dur)
        groups_summary[key] = cnt

print(f"  Groups: " + " | ".join(f"{k}: {v:,}" for k, v in groups_summary.items()))

# UP/DOWN balance overall
up_count = sum(1 for m in parsed_markets if m['outcome'] == 'UP')
dn_count = len(parsed_markets) - up_count
print(f"  Overall: UP {up_count:,} ({up_count/len(parsed_markets)*100:.1f}%) | DOWN {dn_count:,} ({dn_count/len(parsed_markets)*100:.1f}%)")


# ──────────────────────────────────────────────────────────
# 4. GROUP BY (ASSET, DURATION) AND BUILD SEQUENCES
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("BUILDING CHRONOLOGICAL SEQUENCES")
print("=" * 80)

sequences = defaultdict(list)  # key: (asset, duration_min) -> list of outcomes in order

for m in parsed_markets:
    key = (m['asset'], m['duration_min'])
    sequences[key].append(m['outcome'])

for key, seq in sorted(sequences.items()):
    up = sum(1 for o in seq if o == 'UP')
    dn = len(seq) - up
    print(f"  {key[0]} {key[1]:>2}M: {len(seq):>6,} candles | UP {up:,} ({up/len(seq)*100:.1f}%) | DOWN {dn:,} ({dn/len(seq)*100:.1f}%)")


# ──────────────────────────────────────────────────────────
# 5. STREAK ANALYSIS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STREAK DISTRIBUTION")
print("=" * 80)

def compute_streaks(outcomes):
    """Compute all streaks from a sequence of UP/DOWN outcomes.
    Returns list of (direction, streak_length, next_outcome) tuples.
    """
    if len(outcomes) < 2:
        return []

    streaks = []
    current_dir = outcomes[0]
    current_len = 1

    for i in range(1, len(outcomes)):
        if outcomes[i] == current_dir:
            current_len += 1
        else:
            # Streak ended, record it with the outcome that broke it
            streaks.append((current_dir, current_len, outcomes[i]))
            current_dir = outcomes[i]
            current_len = 1

    # Last streak has no "next" outcome
    return streaks


def compute_running_streaks(outcomes):
    """For each position i (starting from 1), compute:
    - The current streak length ending at position i-1
    - The direction of that streak
    - The actual outcome at position i (the "next" candle)

    This lets us test: "after seeing N in a row, what happens next?"
    """
    if len(outcomes) < 2:
        return []

    results = []
    for i in range(1, len(outcomes)):
        # Look backwards from i-1 to count streak
        streak_dir = outcomes[i - 1]
        streak_len = 1
        j = i - 2
        while j >= 0 and outcomes[j] == streak_dir:
            streak_len += 1
            j -= 1

        results.append({
            'streak_dir': streak_dir,
            'streak_len': streak_len,
            'next_outcome': outcomes[i],
            'position': i,
        })

    return results


# Compute streak distributions
all_streak_data = []

for key, seq in sorted(sequences.items()):
    asset, dur = key
    group_name = f"{asset} {dur}M"

    running = compute_running_streaks(seq)
    for r in running:
        r['group'] = group_name
        r['asset'] = asset
        r['duration'] = dur
    all_streak_data.extend(running)

    # Show streak distribution
    streak_counts = defaultdict(int)
    for r in running:
        streak_counts[r['streak_len']] += 1

    print(f"\n  {group_name} streak distribution (what we see BEFORE each trade):")
    for slen in sorted(streak_counts.keys()):
        if slen <= 10:
            pct = streak_counts[slen] / len(running) * 100
            print(f"    Streak {slen:>2}: {streak_counts[slen]:>6,} occurrences ({pct:>5.1f}%)")
    longer = sum(v for k, v in streak_counts.items() if k > 10)
    if longer:
        print(f"    Streak 11+: {longer:>6,} occurrences ({longer/len(running)*100:>5.1f}%)")

print(f"\n  Total streak data points: {len(all_streak_data):,}")


# ──────────────────────────────────────────────────────────
# 6. BACKTEST REVERSAL vs CONTINUATION
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STREAK REVERSAL vs CONTINUATION BACKTEST")
print("=" * 80)

TRADE_SIZE = 3.00  # $3 per trade
ENTRY_PRICE = 0.50  # Assume 50/50 odds at entry

def compute_pnl(bet_dir, actual_outcome, trade_size=TRADE_SIZE, entry_price=ENTRY_PRICE):
    """
    Compute PnL for a single trade.
    Buy at entry_price, resolve at 1.0 (win) or 0.0 (lose).
    Shares = trade_size / entry_price
    Win: shares * (1.0 - entry_price) = trade_size * (1/entry_price - 1) = $3
    Lose: -trade_size
    At $0.50 entry: Win = +$3, Lose = -$3
    """
    if bet_dir == actual_outcome:
        return trade_size * (1.0 / entry_price - 1.0)  # +$3 at 0.50
    else:
        return -trade_size  # -$3


# Strategy definitions
STREAK_THRESHOLDS = [2, 3, 4, 5, 6]  # Minimum streak length to trigger

results_table = []

for strategy_type in ['REVERSAL', 'CONTINUATION']:
    for min_streak in STREAK_THRESHOLDS:
        # Track by group and overall
        group_results = defaultdict(lambda: {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0, 'outcomes': []})
        overall = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0, 'outcomes': []}

        for sd in all_streak_data:
            # Only trigger when streak >= min_streak
            if sd['streak_len'] < min_streak:
                continue

            streak_dir = sd['streak_dir']

            if strategy_type == 'REVERSAL':
                # Bet AGAINST the streak
                bet_dir = 'DOWN' if streak_dir == 'UP' else 'UP'
            else:
                # Bet WITH the streak (continuation)
                bet_dir = streak_dir

            actual = sd['next_outcome']
            won = (bet_dir == actual)
            pnl = compute_pnl(bet_dir, actual)

            group_key = sd['group']
            group_results[group_key]['trades'] += 1
            group_results[group_key]['wins'] += 1 if won else 0
            group_results[group_key]['losses'] += 0 if won else 1
            group_results[group_key]['pnl'] += pnl
            group_results[group_key]['outcomes'].append(1 if won else 0)

            overall['trades'] += 1
            overall['wins'] += 1 if won else 0
            overall['losses'] += 0 if won else 1
            overall['pnl'] += pnl
            overall['outcomes'].append(1 if won else 0)

        results_table.append({
            'strategy': strategy_type,
            'min_streak': min_streak,
            'overall': overall,
            'groups': dict(group_results),
        })


# ──────────────────────────────────────────────────────────
# 7. PRINT RESULTS TABLE
# ──────────────────────────────────────────────────────────
print("\n" + "-" * 110)
print(f"{'Strategy':<14} {'MinStrk':<8} {'Trades':>8} {'Wins':>8} {'Losses':>8} {'WR%':>8} {'PnL':>10} {'$/Trade':>10} {'Chi2-p':>10}")
print("-" * 110)

for r in results_table:
    o = r['overall']
    if o['trades'] == 0:
        continue
    wr = o['wins'] / o['trades'] * 100
    avg_pnl = o['pnl'] / o['trades']

    # Chi-squared test: is WR significantly different from 50%?
    observed = [o['wins'], o['losses']]
    expected = [o['trades'] / 2, o['trades'] / 2]
    if o['trades'] >= 5:
        chi2, p_value = scipy_stats.chisquare(observed, expected)
    else:
        chi2, p_value = 0, 1.0

    sig = "***" if p_value < 0.001 else "** " if p_value < 0.01 else "*  " if p_value < 0.05 else "   "

    label = f"streak>={r['min_streak']}"
    if r['min_streak'] == 6:
        label = f"streak>=6"

    print(f"{r['strategy']:<14} {label:<8} {o['trades']:>8,} {o['wins']:>8,} {o['losses']:>8,} "
          f"{wr:>7.2f}% {o['pnl']:>+10.2f} {avg_pnl:>+10.4f} {p_value:>9.6f} {sig}")

print("-" * 110)


# ──────────────────────────────────────────────────────────
# 8. DETAILED BREAKDOWN BY GROUP
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("DETAILED BREAKDOWN BY GROUP (REVERSAL STRATEGY)")
print("=" * 80)

for r in results_table:
    if r['strategy'] != 'REVERSAL':
        continue

    min_streak = r['min_streak']
    print(f"\n  REVERSAL (streak >= {min_streak}):")
    print(f"  {'Group':<12} {'Trades':>8} {'Wins':>8} {'WR%':>8} {'PnL':>10} {'$/Trade':>10}")
    print(f"  {'-'*60}")

    for gname in sorted(r['groups'].keys()):
        g = r['groups'][gname]
        if g['trades'] == 0:
            continue
        wr = g['wins'] / g['trades'] * 100
        avg = g['pnl'] / g['trades']
        print(f"  {gname:<12} {g['trades']:>8,} {g['wins']:>8,} {wr:>7.2f}% {g['pnl']:>+10.2f} {avg:>+10.4f}")


# ──────────────────────────────────────────────────────────
# 9. EXACT STREAK ANALYSIS (streak == N, not >=)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("EXACT STREAK LENGTH ANALYSIS (what happens after exactly N in a row?)")
print("=" * 80)

print(f"\n  {'Streak':<10} {'Next UP':>10} {'Next DN':>10} {'Total':>10} {'UP%':>8} {'Reversal%':>10} {'Chi2-p':>10}")
print(f"  {'-'*70}")

for exact_len in range(1, 11):
    up_after = 0
    dn_after = 0

    for sd in all_streak_data:
        if sd['streak_len'] != exact_len:
            continue
        if sd['next_outcome'] == 'UP':
            up_after += 1
        else:
            dn_after += 1

    total = up_after + dn_after
    if total == 0:
        continue

    up_pct = up_after / total * 100

    # Reversal % depends on streak direction
    # For UP streaks of length N: reversal = next is DOWN
    # For DOWN streaks of length N: reversal = next is UP
    # We need to split by streak direction
    rev_count = 0
    cont_count = 0
    for sd in all_streak_data:
        if sd['streak_len'] != exact_len:
            continue
        if sd['next_outcome'] != sd['streak_dir']:
            rev_count += 1
        else:
            cont_count += 1

    rev_pct = rev_count / total * 100

    if total >= 5:
        chi2, p_val = scipy_stats.chisquare([rev_count, cont_count], [total/2, total/2])
    else:
        chi2, p_val = 0, 1.0

    sig = "***" if p_val < 0.001 else "** " if p_val < 0.01 else "*  " if p_val < 0.05 else "   "

    print(f"  {exact_len:<10} {up_after:>10,} {dn_after:>10,} {total:>10,} {up_pct:>7.2f}% {rev_pct:>9.2f}% {p_val:>9.6f} {sig}")

# Also check 11+ combined
up_long = sum(1 for sd in all_streak_data if sd['streak_len'] > 10 and sd['next_outcome'] == 'UP')
dn_long = sum(1 for sd in all_streak_data if sd['streak_len'] > 10 and sd['next_outcome'] == 'DOWN')
total_long = up_long + dn_long
if total_long > 0:
    rev_long = sum(1 for sd in all_streak_data if sd['streak_len'] > 10 and sd['next_outcome'] != sd['streak_dir'])
    cont_long = total_long - rev_long
    rev_pct = rev_long / total_long * 100
    up_pct = up_long / total_long * 100
    if total_long >= 5:
        chi2, p_val = scipy_stats.chisquare([rev_long, cont_long], [total_long/2, total_long/2])
    else:
        chi2, p_val = 0, 1.0
    sig = "***" if p_val < 0.001 else "** " if p_val < 0.01 else "*  " if p_val < 0.05 else "   "
    print(f"  {'11+':<10} {up_long:>10,} {dn_long:>10,} {total_long:>10,} {up_pct:>7.2f}% {rev_pct:>9.2f}% {p_val:>9.6f} {sig}")


# ──────────────────────────────────────────────────────────
# 10. DIRECTION-SPECIFIC ANALYSIS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("DIRECTION-SPECIFIC STREAK ANALYSIS")
print("=" * 80)

for streak_dir in ['UP', 'DOWN']:
    print(f"\n  After {streak_dir} streaks:")
    print(f"  {'Length':<10} {'Reverses':>10} {'Continues':>10} {'Total':>10} {'Rev%':>8} {'Chi2-p':>10}")
    print(f"  {'-'*60}")

    for slen in range(2, 9):
        rev = sum(1 for sd in all_streak_data
                  if sd['streak_len'] >= slen and sd['streak_dir'] == streak_dir
                  and sd['next_outcome'] != streak_dir)
        cont = sum(1 for sd in all_streak_data
                   if sd['streak_len'] >= slen and sd['streak_dir'] == streak_dir
                   and sd['next_outcome'] == streak_dir)
        total = rev + cont
        if total == 0:
            continue
        rev_pct = rev / total * 100

        if total >= 5:
            chi2, p_val = scipy_stats.chisquare([rev, cont], [total/2, total/2])
        else:
            p_val = 1.0
        sig = "***" if p_val < 0.001 else "** " if p_val < 0.01 else "*  " if p_val < 0.05 else "   "

        print(f"  >={slen:<8} {rev:>10,} {cont:>10,} {total:>10,} {rev_pct:>7.2f}% {p_val:>9.6f} {sig}")


# ──────────────────────────────────────────────────────────
# 11. ASSET-SPECIFIC STREAK ANALYSIS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("ASSET-SPECIFIC STREAK REVERSAL RATES")
print("=" * 80)

for group_name in sorted(set(sd['group'] for sd in all_streak_data)):
    group_data = [sd for sd in all_streak_data if sd['group'] == group_name]

    print(f"\n  {group_name}:")
    print(f"  {'Streak>=':<10} {'Reverses':>10} {'Continues':>10} {'Rev%':>8} {'PnL(Rev)':>10} {'p-value':>10}")
    print(f"  {'-'*60}")

    for min_s in [2, 3, 4, 5, 6, 8]:
        subset = [sd for sd in group_data if sd['streak_len'] >= min_s]
        if not subset:
            continue

        rev = sum(1 for sd in subset if sd['next_outcome'] != sd['streak_dir'])
        cont = len(subset) - rev
        rev_pct = rev / len(subset) * 100

        # PnL if we played reversal
        pnl = sum(compute_pnl(
            'DOWN' if sd['streak_dir'] == 'UP' else 'UP',
            sd['next_outcome']
        ) for sd in subset)

        if len(subset) >= 5:
            chi2, p_val = scipy_stats.chisquare([rev, cont], [len(subset)/2, len(subset)/2])
        else:
            p_val = 1.0
        sig = "***" if p_val < 0.001 else "** " if p_val < 0.01 else "*  " if p_val < 0.05 else "   "

        print(f"  >={min_s:<8} {rev:>10,} {cont:>10,} {rev_pct:>7.2f}% {pnl:>+10.2f} {p_val:>9.6f} {sig}")


# ──────────────────────────────────────────────────────────
# 12. TIME-OF-DAY STREAK EFFECT
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TIME-OF-DAY EFFECT ON STREAK REVERSALS (streak >= 3)")
print("=" * 80)

# Rebuild with time info
streak_with_time = []
for key, seq in sorted(sequences.items()):
    asset, dur = key
    group_name = f"{asset} {dur}M"
    group_markets = [m for m in parsed_markets if m['asset'] == asset and m['duration_min'] == dur]

    for i in range(1, len(group_markets)):
        streak_dir = group_markets[i-1]['outcome']
        streak_len = 1
        j = i - 2
        while j >= 0 and group_markets[j]['outcome'] == streak_dir:
            streak_len += 1
            j -= 1

        if streak_len >= 3:
            utc_hour = group_markets[i]['candle_start_utc'].hour
            actual = group_markets[i]['outcome']
            reversed_flag = actual != streak_dir
            streak_with_time.append({
                'utc_hour': utc_hour,
                'reversed': reversed_flag,
                'group': group_name,
            })

print(f"\n  {'UTC Hour':<10} {'Reverses':>10} {'Continues':>10} {'Rev%':>8} {'Trades':>8}")
print(f"  {'-'*50}")

hour_stats = defaultdict(lambda: [0, 0])
for s in streak_with_time:
    if s['reversed']:
        hour_stats[s['utc_hour']][0] += 1
    else:
        hour_stats[s['utc_hour']][1] += 1

for h in sorted(hour_stats.keys()):
    rev, cont = hour_stats[h]
    total = rev + cont
    rev_pct = rev / total * 100 if total > 0 else 0
    bar = "#" * int(rev_pct / 2)
    print(f"  {h:>2}:00 UTC  {rev:>10,} {cont:>10,} {rev_pct:>7.2f}% {total:>8,}  {bar}")


# ──────────────────────────────────────────────────────────
# 13. CUMULATIVE PNL SIMULATION
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("CUMULATIVE PNL SIMULATION (Best streak strategies)")
print("=" * 80)

for strat, min_s in [('REVERSAL', 3), ('REVERSAL', 4), ('REVERSAL', 5), ('CONTINUATION', 3)]:
    cum_pnl = 0.0
    trade_count = 0
    max_dd = 0.0
    peak_pnl = 0.0
    win_streak = 0
    lose_streak = 0
    max_win_streak = 0
    max_lose_streak = 0

    for sd in all_streak_data:
        if sd['streak_len'] < min_s:
            continue

        streak_dir = sd['streak_dir']
        if strat == 'REVERSAL':
            bet = 'DOWN' if streak_dir == 'UP' else 'UP'
        else:
            bet = streak_dir

        pnl = compute_pnl(bet, sd['next_outcome'])
        cum_pnl += pnl
        trade_count += 1

        if pnl > 0:
            win_streak += 1
            lose_streak = 0
            max_win_streak = max(max_win_streak, win_streak)
        else:
            lose_streak += 1
            win_streak = 0
            max_lose_streak = max(max_lose_streak, lose_streak)

        peak_pnl = max(peak_pnl, cum_pnl)
        dd = peak_pnl - cum_pnl
        max_dd = max(max_dd, dd)

    if trade_count > 0:
        wr = sum(1 for sd in all_streak_data if sd['streak_len'] >= min_s and
                 compute_pnl('DOWN' if sd['streak_dir'] == 'UP' else 'UP' if strat == 'REVERSAL' else sd['streak_dir'],
                             sd['next_outcome']) > 0) / trade_count * 100
        print(f"\n  {strat} (streak >= {min_s}):")
        print(f"    Trades: {trade_count:,} | WR: {wr:.2f}%")
        print(f"    Final PnL: ${cum_pnl:+.2f} | Peak: ${peak_pnl:+.2f} | Max DD: ${max_dd:.2f}")
        print(f"    Avg PnL/trade: ${cum_pnl/trade_count:+.4f}")
        print(f"    Max win streak: {max_win_streak} | Max lose streak: {max_lose_streak}")


# ──────────────────────────────────────────────────────────
# 14. AUTOCORRELATION ANALYSIS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("AUTOCORRELATION ANALYSIS (lag-1 to lag-5)")
print("=" * 80)

for key, seq in sorted(sequences.items()):
    asset, dur = key
    group_name = f"{asset} {dur}M"

    # Convert to numeric: UP=1, DOWN=0
    numeric = [1 if o == 'UP' else 0 for o in seq]
    n = len(numeric)
    mean = sum(numeric) / n
    var = sum((x - mean) ** 2 for x in numeric) / n

    if var < 1e-10:
        continue

    print(f"\n  {group_name} (n={n:,}):")
    for lag in range(1, 6):
        if n <= lag:
            continue
        cov = sum((numeric[i] - mean) * (numeric[i - lag] - mean) for i in range(lag, n)) / (n - lag)
        autocorr = cov / var
        # Test significance: |r| > 2/sqrt(n)
        threshold = 2.0 / (n ** 0.5)
        sig = " *SIGNIFICANT*" if abs(autocorr) > threshold else ""
        bar = "+" * int(abs(autocorr) * 100) if autocorr > 0 else "-" * int(abs(autocorr) * 100)
        print(f"    Lag {lag}: r={autocorr:+.6f} (threshold={threshold:.6f}){sig}  {bar}")


# ──────────────────────────────────────────────────────────
# 15. RUNS TEST (non-randomness)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RUNS TEST FOR RANDOMNESS")
print("=" * 80)

for key, seq in sorted(sequences.items()):
    asset, dur = key
    group_name = f"{asset} {dur}M"

    n = len(seq)
    n_up = sum(1 for o in seq if o == 'UP')
    n_dn = n - n_up

    # Count runs
    runs = 1
    for i in range(1, n):
        if seq[i] != seq[i-1]:
            runs += 1

    # Expected runs and variance under null hypothesis (random)
    if n_up == 0 or n_dn == 0:
        continue

    expected_runs = 1 + (2 * n_up * n_dn) / n
    var_runs = (2 * n_up * n_dn * (2 * n_up * n_dn - n)) / (n * n * (n - 1))

    if var_runs <= 0:
        continue

    z = (runs - expected_runs) / (var_runs ** 0.5)
    p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

    sig = "***" if p_value < 0.001 else "** " if p_value < 0.01 else "*  " if p_value < 0.05 else "   "

    print(f"  {group_name}: runs={runs:,} expected={expected_runs:.1f} Z={z:+.4f} p={p_value:.6f} {sig}")
    if z < -1.96:
        print(f"    -> CLUSTERING detected (fewer runs than expected = streaks are longer)")
    elif z > 1.96:
        print(f"    -> ALTERNATION detected (more runs than expected = outcomes alternate)")
    else:
        print(f"    -> RANDOM (cannot reject null hypothesis)")


# ──────────────────────────────────────────────────────────
# 16. FINAL VERDICT
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

# Find best strategy
best_rev = None
best_rev_edge = -999
best_cont = None
best_cont_edge = -999

for r in results_table:
    o = r['overall']
    if o['trades'] < 100:
        continue
    wr = o['wins'] / o['trades'] * 100
    edge = wr - 50.0

    if r['strategy'] == 'REVERSAL' and edge > best_rev_edge:
        best_rev_edge = edge
        best_rev = r
    elif r['strategy'] == 'CONTINUATION' and edge > best_cont_edge:
        best_cont_edge = edge
        best_cont = r

print(f"\n  BEST REVERSAL:    streak >= {best_rev['min_streak']}, "
      f"WR = {best_rev['overall']['wins']/best_rev['overall']['trades']*100:.2f}%, "
      f"edge = {best_rev_edge:+.2f}pp, "
      f"PnL = ${best_rev['overall']['pnl']:+.2f} over {best_rev['overall']['trades']:,} trades"
      if best_rev else "  NO REVERSAL with 100+ trades")

print(f"  BEST CONTINUATION: streak >= {best_cont['min_streak']}, "
      f"WR = {best_cont['overall']['wins']/best_cont['overall']['trades']*100:.2f}%, "
      f"edge = {best_cont_edge:+.2f}pp, "
      f"PnL = ${best_cont['overall']['pnl']:+.2f} over {best_cont['overall']['trades']:,} trades"
      if best_cont else "  NO CONTINUATION with 100+ trades")

# Overall assessment
if best_rev and best_rev_edge > 2.0:
    print(f"\n  STREAK REVERSAL EDGE EXISTS: {best_rev_edge:+.2f}pp above random")
    print(f"  After {best_rev['min_streak']}+ consecutive same-direction outcomes,")
    print(f"  betting AGAINST the streak yields {best_rev['overall']['wins']/best_rev['overall']['trades']*100:.2f}% WR")
    print(f"  Recommendation: Implement streak-based position sizing in maker bot")
elif best_cont and best_cont_edge > 2.0:
    print(f"\n  STREAK CONTINUATION EDGE EXISTS: {best_cont_edge:+.2f}pp above random")
    print(f"  After {best_cont['min_streak']}+ consecutive same-direction outcomes,")
    print(f"  betting WITH the streak yields {best_cont['overall']['wins']/best_cont['overall']['trades']*100:.2f}% WR")
    print(f"  Recommendation: Implement momentum-based position sizing in maker bot")
else:
    print(f"\n  NO SIGNIFICANT STREAK EDGE DETECTED")
    print(f"  Both reversal ({best_rev_edge:+.2f}pp) and continuation ({best_cont_edge:+.2f}pp)")
    print(f"  are within noise of 50% baseline.")
    print(f"  Candle outcomes appear to be INDEPENDENT — no usable streak signal.")
    print(f"  Recommendation: Continue 50/50 paired trades, do NOT skew based on streaks.")

print(f"\n{'=' * 80}")
print("Done.")
