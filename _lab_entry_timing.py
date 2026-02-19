"""
BTC Up/Down 15M Entry Timing Backtest
=====================================
Tests optimal entry timing for Polymarket 15-minute BTC Up/Down markets.
Simulates entering at 2/4/6/8 minutes before candle close using 1m candle data.
"""

import requests
import time
import json
from datetime import datetime, timezone

BET_SIZE = 3.0  # $3 per trade
BINANCE_BASE = "https://api.binance.com/api/v3/klines"

# ─────────────────────────────────────────────
# 1. Fetch 15M candles (last 14 days)
# ─────────────────────────────────────────────
def fetch_15m_candles():
    """Fetch ~1344 fifteen-minute candles (14 days)."""
    all_candles = []
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 14 * 24 * 60 * 60 * 1000  # 14 days ago

    cursor = start_ms
    while cursor < now_ms:
        params = {
            "symbol": "BTCUSDT",
            "interval": "15m",
            "startTime": cursor,
            "limit": 1000,
        }
        resp = requests.get(BINANCE_BASE, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_candles.extend(batch)
        cursor = batch[-1][6] + 1  # close_time + 1
        time.sleep(0.3)

    print(f"[+] Fetched {len(all_candles)} 15M candles")
    return all_candles


# ─────────────────────────────────────────────
# 2. Fetch 1M candles (last 14 days)
# ─────────────────────────────────────────────
def fetch_1m_candles():
    """Fetch 1-minute candles for the last 14 days. Multiple requests needed."""
    all_candles = []
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 14 * 24 * 60 * 60 * 1000

    cursor = start_ms
    while cursor < now_ms:
        params = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "startTime": cursor,
            "limit": 1000,
        }
        resp = requests.get(BINANCE_BASE, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_candles.extend(batch)
        cursor = batch[-1][6] + 1
        time.sleep(0.3)

    print(f"[+] Fetched {len(all_candles)} 1M candles")
    return all_candles


# ─────────────────────────────────────────────
# 3. Build 1M lookup: open_time_ms -> close_price
# ─────────────────────────────────────────────
def build_1m_lookup(candles_1m):
    """Map each 1m candle open_time -> close price for fast lookup."""
    lookup = {}
    for c in candles_1m:
        open_time = c[0]
        close_price = float(c[4])
        lookup[open_time] = close_price
    return lookup


# ─────────────────────────────────────────────
# 4. Entry pricing model
# ─────────────────────────────────────────────
def calc_entry_price(btc_at_entry, candle_open, candle_high, candle_low):
    """
    Approximate Polymarket UP contract price at entry time.
    If BTC is above 15M open -> UP looks more likely -> UP ask is higher.
    Model: 0.50 + displacement_fraction * 0.30, capped [0.35, 0.65].
    Returns the ask price for the side we'd bet on.
    """
    candle_range = candle_high - candle_low
    if candle_range < 0.01:
        return 0.50

    displacement = btc_at_entry - candle_open
    frac = displacement / candle_range
    # Clamp fraction to [-1, 1]
    frac = max(-1.0, min(1.0, frac))
    up_price = 0.50 + frac * 0.30
    up_price = max(0.35, min(0.65, up_price))
    return up_price


# ─────────────────────────────────────────────
# 5. Run simulation
# ─────────────────────────────────────────────
def run_simulation():
    print("=" * 70)
    print("  BTC 15M Entry Timing Backtest")
    print("  Polymarket Up/Down Market Simulation")
    print("=" * 70)
    print()

    # Fetch data
    candles_15m = fetch_15m_candles()
    candles_1m = fetch_1m_candles()
    lookup_1m = build_1m_lookup(candles_1m)

    # Entry timing offsets: minutes before 15M candle close
    entry_offsets_min = [2, 4, 6, 8]

    # Results storage: {offset: {wins, losses, total_pnl, entry_prices, ...}}
    results = {}
    results_filtered = {}  # Only trades with >$50 displacement

    for offset in entry_offsets_min:
        results[offset] = {
            "wins": 0, "losses": 0, "total_pnl": 0.0,
            "entry_prices": [], "pnls": [],
            "correct_signals": 0, "total_signals": 0,
        }
        results_filtered[offset] = {
            "wins": 0, "losses": 0, "total_pnl": 0.0,
            "entry_prices": [], "pnls": [],
            "correct_signals": 0, "total_signals": 0,
        }

    skipped = 0
    processed = 0

    for candle in candles_15m:
        c_open_time = candle[0]       # ms
        c_open = float(candle[1])
        c_high = float(candle[2])
        c_low = float(candle[3])
        c_close = float(candle[4])
        c_close_time = candle[6]      # ms

        # Skip candles with tiny range (no meaningful movement)
        candle_range = c_high - c_low
        if candle_range < 1.0:
            skipped += 1
            continue

        # Actual outcome
        actual_up = c_close >= c_open

        for offset in entry_offsets_min:
            # Entry time = close_time - offset minutes
            # close_time is the end of the candle period
            # 1m candle open_time that contains our entry point
            entry_time_ms = c_close_time - (offset * 60 * 1000)

            # Round down to nearest minute boundary for 1m lookup
            entry_1m_open = (entry_time_ms // 60000) * 60000

            if entry_1m_open not in lookup_1m:
                continue

            btc_at_entry = lookup_1m[entry_1m_open]

            # Our signal: if BTC > 15M open -> bet UP, else bet DOWN
            signal_up = btc_at_entry > c_open

            # Handle exact equality: skip (no signal)
            if btc_at_entry == c_open:
                continue

            # Entry price model
            up_ask = calc_entry_price(btc_at_entry, c_open, c_high, c_low)

            # If we bet UP: entry_cost = up_ask
            # If we bet DOWN: entry_cost = (1 - up_ask) = down_ask
            if signal_up:
                entry_cost = up_ask
                correct = actual_up
            else:
                entry_cost = 1.0 - up_ask
                correct = not actual_up

            # PnL
            if correct:
                pnl = (1.0 - entry_cost) * BET_SIZE
            else:
                pnl = -entry_cost * BET_SIZE

            # Store in main results
            r = results[offset]
            r["total_signals"] += 1
            if correct:
                r["wins"] += 1
                r["correct_signals"] += 1
            else:
                r["losses"] += 1
            r["total_pnl"] += pnl
            r["entry_prices"].append(entry_cost)
            r["pnls"].append(pnl)

            # Filtered: only when displacement > $50
            displacement = abs(btc_at_entry - c_open)
            if displacement > 50.0:
                rf = results_filtered[offset]
                rf["total_signals"] += 1
                if correct:
                    rf["wins"] += 1
                    rf["correct_signals"] += 1
                else:
                    rf["losses"] += 1
                rf["total_pnl"] += pnl
                rf["entry_prices"].append(entry_cost)
                rf["pnls"].append(pnl)

        processed += 1

    print(f"\n[+] Processed {processed} candles, skipped {skipped} (tiny range)")
    print()

    # ─────────────────────────────────────────
    # 6. Print results table
    # ─────────────────────────────────────────

    # --- ALL TRADES ---
    print("=" * 70)
    print("  ALL TRADES (no displacement filter)")
    print("=" * 70)
    header = f"{'Entry':>8} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'AvgEntry':>9} {'AvgPnL':>8} {'TotalPnL':>10} {'Sharpe':>7}"
    print(header)
    print("-" * 70)

    for offset in entry_offsets_min:
        r = results[offset]
        n = r["total_signals"]
        if n == 0:
            print(f"  {offset}m pre  {'N/A':>7}")
            continue
        wr = r["wins"] / n * 100
        avg_entry = sum(r["entry_prices"]) / n
        avg_pnl = r["total_pnl"] / n
        total_pnl = r["total_pnl"]

        # Simple Sharpe approximation
        import statistics
        if len(r["pnls"]) > 1:
            std = statistics.stdev(r["pnls"])
            sharpe = (avg_pnl / std) if std > 0 else 0.0
        else:
            sharpe = 0.0

        label = f"{offset}m pre"
        print(f"{label:>8} {n:>7} {r['wins']:>6} {wr:>6.1f}% {avg_entry:>8.3f} {avg_pnl:>+7.4f} {total_pnl:>+9.2f} {sharpe:>+6.3f}")

    # --- FILTERED TRADES (>$50 displacement) ---
    print()
    print("=" * 70)
    print("  HIGH CONVICTION ONLY (displacement > $50 from 15M open)")
    print("=" * 70)
    print(header)
    print("-" * 70)

    for offset in entry_offsets_min:
        rf = results_filtered[offset]
        n = rf["total_signals"]
        if n == 0:
            print(f"  {offset}m pre  {'N/A':>7}   (no trades passed filter)")
            continue
        wr = rf["wins"] / n * 100
        avg_entry = sum(rf["entry_prices"]) / n
        avg_pnl = rf["total_pnl"] / n
        total_pnl = rf["total_pnl"]

        if len(rf["pnls"]) > 1:
            std = statistics.stdev(rf["pnls"])
            sharpe = (avg_pnl / std) if std > 0 else 0.0
        else:
            sharpe = 0.0

        label = f"{offset}m pre"
        print(f"{label:>8} {n:>7} {rf['wins']:>6} {wr:>6.1f}% {avg_entry:>8.3f} {avg_pnl:>+7.4f} {total_pnl:>+9.2f} {sharpe:>+6.3f}")

    # --- Additional displacement thresholds ---
    print()
    print("=" * 70)
    print("  DISPLACEMENT SENSITIVITY (8m pre-close entry)")
    print("=" * 70)
    thresholds = [0, 25, 50, 75, 100, 150, 200]
    print(f"{'MinDisp':>8} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'AvgEntry':>9} {'AvgPnL':>8} {'TotalPnL':>10}")
    print("-" * 65)

    # Recompute for 8m with various thresholds
    offset = 8
    for thresh in thresholds:
        wins = 0
        losses = 0
        total_pnl = 0.0
        entry_prices = []
        total = 0

        for candle in candles_15m:
            c_open_time = candle[0]
            c_open = float(candle[1])
            c_high = float(candle[2])
            c_low = float(candle[3])
            c_close = float(candle[4])
            c_close_time = candle[6]

            candle_range = c_high - c_low
            if candle_range < 1.0:
                continue

            actual_up = c_close >= c_open

            entry_time_ms = c_close_time - (offset * 60 * 1000)
            entry_1m_open = (entry_time_ms // 60000) * 60000

            if entry_1m_open not in lookup_1m:
                continue

            btc_at_entry = lookup_1m[entry_1m_open]
            displacement = abs(btc_at_entry - c_open)

            if displacement < thresh:
                continue

            if btc_at_entry == c_open:
                continue

            signal_up = btc_at_entry > c_open
            up_ask = calc_entry_price(btc_at_entry, c_open, c_high, c_low)

            if signal_up:
                entry_cost = up_ask
                correct = actual_up
            else:
                entry_cost = 1.0 - up_ask
                correct = not actual_up

            if correct:
                pnl = (1.0 - entry_cost) * BET_SIZE
            else:
                pnl = -entry_cost * BET_SIZE

            total += 1
            if correct:
                wins += 1
            else:
                losses += 1
            total_pnl += pnl
            entry_prices.append(entry_cost)

        if total == 0:
            print(f"  ${thresh:>4}  {'N/A':>7}")
            continue
        wr = wins / total * 100
        avg_entry = sum(entry_prices) / total
        avg_pnl = total_pnl / total
        print(f"  ${thresh:>4} {total:>7} {wins:>6} {wr:>6.1f}% {avg_entry:>8.3f} {avg_pnl:>+7.4f} {total_pnl:>+9.2f}")

    # --- Contrarian analysis (bet AGAINST the signal) ---
    print()
    print("=" * 70)
    print("  CONTRARIAN TEST: Bet AGAINST displacement direction")
    print("=" * 70)
    print(f"{'Entry':>8} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'AvgEntry':>9} {'AvgPnL':>8} {'TotalPnL':>10}")
    print("-" * 65)

    for offset in entry_offsets_min:
        wins = 0
        losses = 0
        total_pnl = 0.0
        entry_prices = []
        total = 0

        for candle in candles_15m:
            c_open = float(candle[1])
            c_high = float(candle[2])
            c_low = float(candle[3])
            c_close = float(candle[4])
            c_close_time = candle[6]

            candle_range = c_high - c_low
            if candle_range < 1.0:
                continue

            actual_up = c_close >= c_open
            entry_time_ms = c_close_time - (offset * 60 * 1000)
            entry_1m_open = (entry_time_ms // 60000) * 60000

            if entry_1m_open not in lookup_1m:
                continue

            btc_at_entry = lookup_1m[entry_1m_open]
            if btc_at_entry == c_open:
                continue

            # CONTRARIAN: bet opposite to displacement
            signal_up = btc_at_entry < c_open  # flipped!
            up_ask = calc_entry_price(btc_at_entry, c_open, c_high, c_low)

            if signal_up:
                entry_cost = up_ask
                correct = actual_up
            else:
                entry_cost = 1.0 - up_ask
                correct = not actual_up

            if correct:
                pnl = (1.0 - entry_cost) * BET_SIZE
            else:
                pnl = -entry_cost * BET_SIZE

            total += 1
            if correct:
                wins += 1
            else:
                losses += 1
            total_pnl += pnl
            entry_prices.append(entry_cost)

        if total == 0:
            print(f"  {offset}m pre  {'N/A':>7}")
            continue
        wr = wins / total * 100
        avg_entry = sum(entry_prices) / total
        avg_pnl = total_pnl / total
        label = f"{offset}m pre"
        print(f"{label:>8} {total:>7} {wins:>6} {wr:>6.1f}% {avg_entry:>8.3f} {avg_pnl:>+7.4f} {total_pnl:>+9.2f}")

    # --- Time-of-day analysis (8m entry) ---
    print()
    print("=" * 70)
    print("  TIME-OF-DAY BREAKDOWN (8m pre-close, all trades)")
    print("=" * 70)
    print(f"{'Hour(UTC)':>10} {'Trades':>7} {'WR%':>7} {'AvgPnL':>8} {'TotalPnL':>10}")
    print("-" * 50)

    hourly = {}
    for candle in candles_15m:
        c_open_time = candle[0]
        c_open = float(candle[1])
        c_high = float(candle[2])
        c_low = float(candle[3])
        c_close = float(candle[4])
        c_close_time = candle[6]

        candle_range = c_high - c_low
        if candle_range < 1.0:
            continue

        actual_up = c_close >= c_open
        entry_time_ms = c_close_time - (8 * 60 * 1000)
        entry_1m_open = (entry_time_ms // 60000) * 60000

        if entry_1m_open not in lookup_1m:
            continue

        btc_at_entry = lookup_1m[entry_1m_open]
        if btc_at_entry == c_open:
            continue

        signal_up = btc_at_entry > c_open
        up_ask = calc_entry_price(btc_at_entry, c_open, c_high, c_low)

        if signal_up:
            entry_cost = up_ask
            correct = actual_up
        else:
            entry_cost = 1.0 - up_ask
            correct = not actual_up

        if correct:
            pnl = (1.0 - entry_cost) * BET_SIZE
        else:
            pnl = -entry_cost * BET_SIZE

        hour = datetime.fromtimestamp(c_open_time / 1000, tz=timezone.utc).hour
        if hour not in hourly:
            hourly[hour] = {"wins": 0, "total": 0, "pnl": 0.0}
        hourly[hour]["total"] += 1
        if correct:
            hourly[hour]["wins"] += 1
        hourly[hour]["pnl"] += pnl

    for h in sorted(hourly.keys()):
        d = hourly[h]
        wr = d["wins"] / d["total"] * 100 if d["total"] > 0 else 0
        avg_pnl = d["pnl"] / d["total"] if d["total"] > 0 else 0
        print(f"  {h:02d}:00 {d['total']:>7} {wr:>6.1f}% {avg_pnl:>+7.4f} {d['pnl']:>+9.2f}")

    # --- Summary ---
    print()
    print("=" * 70)
    print("  SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    best_offset = None
    best_pnl = -999999
    for offset in entry_offsets_min:
        r = results[offset]
        if r["total_signals"] > 0 and r["total_pnl"] > best_pnl:
            best_pnl = r["total_pnl"]
            best_offset = offset

    best_filt_offset = None
    best_filt_pnl = -999999
    for offset in entry_offsets_min:
        rf = results_filtered[offset]
        if rf["total_signals"] > 0 and rf["total_pnl"] > best_filt_pnl:
            best_filt_pnl = rf["total_pnl"]
            best_filt_offset = offset

    if best_offset:
        r = results[best_offset]
        wr = r["wins"] / r["total_signals"] * 100
        print(f"  Best ALL trades:  {best_offset}m pre-close | WR={wr:.1f}% | PnL=${best_pnl:+.2f} over {r['total_signals']} trades")

    if best_filt_offset:
        rf = results_filtered[best_filt_offset]
        if rf["total_signals"] > 0:
            wr = rf["wins"] / rf["total_signals"] * 100
            print(f"  Best FILTERED:    {best_filt_offset}m pre-close | WR={wr:.1f}% | PnL=${best_filt_pnl:+.2f} over {rf['total_signals']} trades")

    # Check if momentum or contrarian is better
    print()
    print("  Key takeaways:")
    print("  - Momentum = bet in direction of current displacement from 15M open")
    print("  - Contrarian = bet against displacement (mean reversion)")
    print("  - Earlier entry (8m) gives more time for signal but prices closer to 0.50")
    print("  - Later entry (2m) gives stronger signal but worse pricing")
    print()


if __name__ == "__main__":
    run_simulation()
