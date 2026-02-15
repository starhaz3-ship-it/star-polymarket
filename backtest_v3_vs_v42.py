"""
Head-to-head backtest: Momentum V3.0 vs V4.2
Uses 7 days of BTC 1-minute data from Binance.
Simulates both strategies on the same 5-min rounds.

V3.0: Wait 2min, if momentum > 15bp, buy direction at mid+1c. Flat $5/trade.
V4.2: Wait 45s+, 10min lookback, vol gate, trend gate, impulse gate, edge gate, $50-150 sizing.
"""

import json
import math
import time
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Literal

Side = Literal["UP", "DOWN"]

# ============================================================
# DATA FETCHING
# ============================================================

def fetch_binance_klines(symbol="BTCUSDT", interval="1m", days=7):
    """Fetch 1-minute candles from Binance."""
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000

    current = start_ms
    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "limit": 1000,
        }
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if not data:
            break
        all_klines.extend(data)
        current = data[-1][0] + 60000
        time.sleep(0.15)

    bars = []
    for k in all_klines:
        bars.append({
            "ts": k[0],
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return bars


def aggregate_5m(bars_1m):
    """Group 1-min bars into 5-min bars."""
    bars_5m = []
    for i in range(0, len(bars_1m) - 4, 5):
        chunk = bars_1m[i:i+5]
        if len(chunk) < 5:
            break
        bars_5m.append({
            "ts": chunk[0]["ts"],
            "open": chunk[0]["open"],
            "high": max(c["high"] for c in chunk),
            "low": min(c["low"] for c in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(c["volume"] for c in chunk),
            "minutes": chunk,  # keep 1-min detail
        })
    return bars_5m


# ============================================================
# V3.0 STRATEGY SIMULATION
# ============================================================

def simulate_v30(bars_5m, bars_1m):
    """
    V3.0: Wait 2 minutes into the 5-min bar, check momentum vs open.
    If > 15bp, buy that direction at mid+1c (~$0.51).
    Flat $5/trade. Max 3 concurrent.
    """
    WAIT_SECONDS = 120  # 2 minutes
    MAX_WAIT_SECONDS = 210  # 3.5 minutes
    THRESHOLD_BPS = 15.0
    SIZE_USD = 5.0
    BUY_PRICE = 0.51  # mid(0.50) + 0.01

    trades = []

    for bar in bars_5m:
        mins = bar["minutes"]
        if len(mins) < 5:
            continue

        open_price = mins[0]["open"]  # BTC price at bar open

        # Check at 2-minute mark (end of minute 2)
        price_at_2m = mins[1]["close"]  # end of 2nd minute

        elapsed = 120  # 2 minutes
        if elapsed < WAIT_SECONDS or elapsed > MAX_WAIT_SECONDS:
            continue

        mom_bps = (price_at_2m - open_price) / open_price * 10000

        if abs(mom_bps) < THRESHOLD_BPS:
            continue

        direction = "UP" if mom_bps > 0 else "DOWN"
        bar_close = bar["close"]
        actual_dir = "UP" if bar_close > open_price else "DOWN"

        shares = max(1, math.floor(SIZE_USD / BUY_PRICE))
        cost = BUY_PRICE * shares

        if direction == actual_dir:
            pnl = shares * 1.0 - cost
        else:
            pnl = -cost

        trades.append({
            "direction": direction,
            "actual": actual_dir,
            "win": direction == actual_dir,
            "momentum_bps": mom_bps,
            "pnl": pnl,
            "cost": cost,
            "shares": shares,
            "buy_price": BUY_PRICE,
            "bar_open": open_price,
            "bar_close": bar_close,
        })

    return trades


# ============================================================
# V4.2 HELPER FUNCTIONS
# ============================================================

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def _bps(p0, p1):
    if p0 <= 0:
        return 0.0
    return (p1 - p0) / p0 * 10000.0

def compute_vol_pct(prices):
    if len(prices) < 3:
        return 0.0
    mx = max(prices)
    mn = min(prices)
    mid = (mx + mn) / 2.0
    if mid <= 0:
        return 0.0
    return (mx - mn) / mid * 100.0

def ema_series(values, n):
    if not values:
        return []
    a = 2.0 / (n + 1.0)
    out = [values[0]]
    for i in range(1, len(values)):
        out.append(a * values[i] + (1.0 - a) * out[i - 1])
    return out

def fair_prob_from_momentum(mom_bps, accel_bps, vol_pct, trend_ok):
    score = 0.0
    score += mom_bps / 18.0
    score += accel_bps / 10.0
    score += 0.35 if trend_ok else -0.35
    score -= max(0.0, (vol_pct - 0.8)) * 0.35
    p = _sigmoid(score)
    return _clamp(p, 0.08, 0.92)


# ============================================================
# V4.2 STRATEGY SIMULATION
# ============================================================

def simulate_v42(bars_5m, bars_1m):
    """
    V4.2:
    - 10-min lookback for momentum (uses prior bars + current)
    - Vol gating: 0.10% - 1.80%
    - Trend gating: EMA 9 vs 21
    - Impulse gate: momentum must be accelerating
    - Edge gate: fair_prob - ask >= 1.2c
    - Sizing: $50-$150 scaled by edge + strength
    """
    THRESHOLD_BPS = 12.0
    ACCEL_BPS = 3.0
    VOL_MIN = 0.10
    VOL_MAX = 1.80
    MIN_EDGE = 0.012
    SPREAD_GUESS = 0.01
    SIZE_BASE = 50.0
    SIZE_MAX = 150.0
    TREND_FAST = 9
    TREND_SLOW = 21

    trades = []

    # Build a running price history from 1-min closes for trend/vol
    # We need enough history before the first 5m bar
    price_history = []

    # Index 1-min bars by timestamp for quick lookup
    min_bar_map = {}
    for b in bars_1m:
        min_bar_map[b["ts"]] = b

    for bar_idx, bar in enumerate(bars_5m):
        mins = bar["minutes"]
        if len(mins) < 5:
            continue

        open_price = mins[0]["open"]

        # Build price history up to this bar (use 1-min closes from prior bars)
        # We need ~21+ 5m bars of history for EMA trend
        # Use close prices of prior 5m bars
        hist_start = max(0, bar_idx - 30)
        price_hist = [bars_5m[j]["close"] for j in range(hist_start, bar_idx)]

        # Also add per-minute prices within this bar for vol calculation
        intra_prices = [m["close"] for m in mins]

        # Check at each minute within the bar (V4.2 can trigger anytime after 45s)
        # We'll check at minute 1 (60s), minute 2 (120s), minute 3 (180s)
        for check_min in [1, 2, 3]:
            if check_min >= len(mins):
                continue

            current_price = mins[check_min]["close"]
            elapsed_sec = (check_min + 1) * 60

            if elapsed_sec < 45:  # MOMENTUM_WAIT_SECONDS
                continue

            # 10-min lookback momentum: need price from 10 min ago
            # That's 2 bars back (2 * 5min = 10min)
            if bar_idx < 2:
                continue

            lookback_bar = bars_5m[bar_idx - 2]
            p0 = lookback_bar["open"]  # price ~10 min ago
            mom_bps = _bps(p0, current_price)

            # Acceleration: compare last 5min vs full 10min
            if bar_idx < 1:
                continue
            half_bar = bars_5m[bar_idx - 1]
            p_half = half_bar["open"]  # price ~5 min ago
            mom_half = _bps(p_half, current_price)
            accel = mom_half - mom_bps

            # Vol filter: realized vol over the lookback window
            vol_prices = []
            for j in range(max(0, bar_idx - 2), bar_idx):
                vol_prices.extend([bars_5m[j]["high"], bars_5m[j]["low"]])
            vol_prices.extend(intra_prices[:check_min+1])
            vol_pct = compute_vol_pct(vol_prices)

            if vol_pct < VOL_MIN or vol_pct > VOL_MAX:
                continue

            # Direction
            direction = None
            if mom_bps >= THRESHOLD_BPS:
                direction = "UP"
                if accel < -ACCEL_BPS:
                    direction = None  # impulse fading
            elif mom_bps <= -THRESHOLD_BPS:
                direction = "DOWN"
                if accel > ACCEL_BPS:
                    direction = None  # impulse fading

            if not direction:
                continue

            # Trend gate
            if len(price_hist) >= TREND_SLOW + 5:
                ef = ema_series(price_hist[-(TREND_SLOW + 20):], TREND_FAST)[-1]
                es = ema_series(price_hist[-(TREND_SLOW + 20):], TREND_SLOW)[-1]
                if direction == "UP" and ef <= es:
                    continue
                if direction == "DOWN" and ef >= es:
                    continue

            # Edge gate
            mid = 0.50  # Polymarket mid for 5m Up/Down
            ask = mid + SPREAD_GUESS / 2.0  # ~0.505

            fair_yes = fair_prob_from_momentum(abs(mom_bps), abs(accel), vol_pct, True)
            edge = fair_yes - ask

            if edge < MIN_EDGE:
                continue

            # Sizing
            strength = _clamp((abs(mom_bps) - THRESHOLD_BPS) / 40.0, 0.0, 1.0)
            edge_strength = _clamp((edge - MIN_EDGE) / 0.04, 0.0, 1.0)
            size_usd = SIZE_BASE + (0.55 * strength + 0.45 * edge_strength) * (SIZE_MAX - SIZE_BASE)

            # Buy price: fair_yes - min_edge (best we can pay while keeping edge)
            max_edge_pay = fair_yes - MIN_EDGE
            buy_price = _clamp(round(min(0.95, max_edge_pay, ask), 2), 0.05, 0.95)

            shares = max(1, math.floor(size_usd / buy_price))
            cost = buy_price * shares

            # Actual outcome
            bar_close = bar["close"]
            actual_dir = "UP" if bar_close > open_price else "DOWN"

            if direction == actual_dir:
                pnl = shares * 1.0 - cost
            else:
                pnl = -cost

            trades.append({
                "direction": direction,
                "actual": actual_dir,
                "win": direction == actual_dir,
                "momentum_bps": mom_bps,
                "accel_bps": accel,
                "vol_pct": vol_pct,
                "edge": edge,
                "fair_yes": fair_yes,
                "pnl": pnl,
                "cost": cost,
                "shares": shares,
                "buy_price": buy_price,
                "size_usd": size_usd,
                "bar_open": open_price,
                "bar_close": bar_close,
                "check_min": check_min,
            })
            break  # Only one trade per bar

    return trades


# ============================================================
# ANALYSIS
# ============================================================

def analyze(trades, label):
    if not trades:
        print(f"\n{'='*60}")
        print(f"  {label}: NO TRADES")
        print(f"{'='*60}")
        return {}

    total = len(trades)
    wins = sum(1 for t in trades if t["win"])
    losses = total - wins
    wr = wins / total * 100

    total_pnl = sum(t["pnl"] for t in trades)
    avg_pnl = total_pnl / total
    total_cost = sum(t["cost"] for t in trades)
    total_revenue = sum(t["shares"] * 1.0 for t in trades if t["win"])

    win_pnls = [t["pnl"] for t in trades if t["win"]]
    loss_pnls = [t["pnl"] for t in trades if not t["win"]]
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0

    # Max drawdown
    cum = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cum += t["pnl"]
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    # Streaks
    max_win_streak = 0
    max_loss_streak = 0
    cur_win = 0
    cur_loss = 0
    for t in trades:
        if t["win"]:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win_streak = max(max_win_streak, cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    # Daily breakdown
    daily = {}
    for t in trades:
        # Approximate day from bar timestamp
        day = "day"
        daily.setdefault(day, []).append(t["pnl"])

    # Avg buy price
    avg_buy = sum(t["buy_price"] for t in trades) / total
    avg_shares = sum(t["shares"] for t in trades) / total
    avg_cost = total_cost / total

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades:       {total}")
    print(f"  Wins:         {wins} ({wr:.1f}%)")
    print(f"  Losses:       {losses} ({100-wr:.1f}%)")
    print(f"  Total PnL:    ${total_pnl:+,.2f}")
    print(f"  Avg PnL:      ${avg_pnl:+.2f}/trade")
    print(f"  Avg Win:      ${avg_win:+.2f}")
    print(f"  Avg Loss:     ${avg_loss:+.2f}")
    print(f"  W:L Ratio:    {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "  W:L Ratio:    N/A")
    print(f"  Total Cost:   ${total_cost:,.2f}")
    print(f"  Total Rev:    ${total_revenue:,.2f}")
    print(f"  Max Drawdown: ${max_dd:,.2f}")
    print(f"  Win Streak:   {max_win_streak}")
    print(f"  Loss Streak:  {max_loss_streak}")
    print(f"  Avg Buy:      ${avg_buy:.2f}")
    print(f"  Avg Shares:   {avg_shares:.1f}")
    print(f"  Avg Cost:     ${avg_cost:.2f}/trade")

    # Momentum distribution for winning vs losing trades
    win_moms = [abs(t["momentum_bps"]) for t in trades if t["win"]]
    loss_moms = [abs(t["momentum_bps"]) for t in trades if not t["win"]]
    if win_moms:
        print(f"  Avg |Mom| (wins):   {sum(win_moms)/len(win_moms):.1f}bp")
    if loss_moms:
        print(f"  Avg |Mom| (losses): {sum(loss_moms)/len(loss_moms):.1f}bp")

    # Per-trade capital efficiency
    roi = total_pnl / total_cost * 100 if total_cost > 0 else 0
    print(f"  ROI:          {roi:+.2f}%")

    return {
        "label": label,
        "trades": total,
        "wins": wins,
        "win_rate": round(wr, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "max_drawdown": round(max_dd, 2),
        "total_cost": round(total_cost, 2),
        "roi_pct": round(roi, 2),
        "avg_buy_price": round(avg_buy, 3),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }


# ============================================================
# POLYMARKET-REALISTIC SIMULATION
# ============================================================

def simulate_polymarket_realistic(bars_5m, bars_1m):
    """
    More realistic simulation that accounts for Polymarket mechanics:
    - Buy price varies (not always 0.50-0.51)
    - Win = shares * $1.00 - cost
    - Loss = -cost (shares worth $0)
    - V3.0 buys at $0.51 (mid + 1c)
    - V4.2 buys at edge-optimized price

    We also simulate what happens with $5 bets vs $50-150 bets
    given a $43 account balance.
    """
    print("\n" + "="*60)
    print("  CAPITAL-ADJUSTED COMPARISON ($43 account)")
    print("="*60)

    # V3.0 with $5 bets - can afford many trades
    v30_trades = simulate_v30(bars_5m, bars_1m)
    v30_capital_needed = 5.0 * 3  # max 3 concurrent * $5 = $15

    # V4.2 with $50-150 bets - can afford 1 maybe 2 trades at a time
    v42_trades = simulate_v42(bars_5m, bars_1m)
    v42_avg_cost = sum(t["cost"] for t in v42_trades) / len(v42_trades) if v42_trades else 50
    v42_capital_needed = v42_avg_cost * 2  # max 2 concurrent

    print(f"\n  V3.0: {len(v30_trades)} trades, needs ~${v30_capital_needed:.0f} capital (${5}/trade x 3 concurrent)")
    print(f"  V4.2: {len(v42_trades)} trades, needs ~${v42_capital_needed:.0f} capital (${v42_avg_cost:.0f}/trade x 2 concurrent)")

    if v42_capital_needed > 43:
        print(f"\n  WARNING: V4.2 needs ${v42_capital_needed:.0f} but account only has $43!")
        print(f"  V4.2 would need to reduce size to ~${43/2:.0f}/trade max")

        # Re-simulate V4.2 with capped sizing
        cap_factor = min(1.0, 43.0 / (v42_capital_needed * 1.2))
        print(f"  Scaling V4.2 PnL by {cap_factor:.2f}x to match $43 account")
        for t in v42_trades:
            t["pnl"] *= cap_factor
            t["cost"] *= cap_factor


def simulate_v42_with_cap(bars_5m, bars_1m, max_size=20.0):
    """V4.2 but with size capped to fit $43 account."""
    THRESHOLD_BPS = 12.0
    ACCEL_BPS = 3.0
    VOL_MIN = 0.10
    VOL_MAX = 1.80
    MIN_EDGE = 0.012
    SPREAD_GUESS = 0.01
    SIZE_BASE = max_size  # Capped
    SIZE_MAX = max_size   # Capped (same as base for fair comparison)
    TREND_FAST = 9
    TREND_SLOW = 21

    trades = []

    for bar_idx, bar in enumerate(bars_5m):
        mins = bar["minutes"]
        if len(mins) < 5:
            continue

        open_price = mins[0]["open"]
        price_hist = [bars_5m[j]["close"] for j in range(max(0, bar_idx - 30), bar_idx)]
        intra_prices = [m["close"] for m in mins]

        for check_min in [1, 2, 3]:
            if check_min >= len(mins):
                continue

            current_price = mins[check_min]["close"]
            if bar_idx < 2:
                continue

            lookback_bar = bars_5m[bar_idx - 2]
            p0 = lookback_bar["open"]
            mom_bps = _bps(p0, current_price)

            half_bar = bars_5m[bar_idx - 1]
            p_half = half_bar["open"]
            mom_half = _bps(p_half, current_price)
            accel = mom_half - mom_bps

            vol_prices = []
            for j in range(max(0, bar_idx - 2), bar_idx):
                vol_prices.extend([bars_5m[j]["high"], bars_5m[j]["low"]])
            vol_prices.extend(intra_prices[:check_min+1])
            vol_pct = compute_vol_pct(vol_prices)

            if vol_pct < VOL_MIN or vol_pct > VOL_MAX:
                continue

            direction = None
            if mom_bps >= THRESHOLD_BPS:
                direction = "UP"
                if accel < -ACCEL_BPS:
                    direction = None
            elif mom_bps <= -THRESHOLD_BPS:
                direction = "DOWN"
                if accel > ACCEL_BPS:
                    direction = None

            if not direction:
                continue

            if len(price_hist) >= TREND_SLOW + 5:
                ef = ema_series(price_hist[-(TREND_SLOW + 20):], TREND_FAST)[-1]
                es = ema_series(price_hist[-(TREND_SLOW + 20):], TREND_SLOW)[-1]
                if direction == "UP" and ef <= es:
                    continue
                if direction == "DOWN" and ef >= es:
                    continue

            mid = 0.50
            ask = mid + SPREAD_GUESS / 2.0
            fair_yes = fair_prob_from_momentum(abs(mom_bps), abs(accel), vol_pct, True)
            edge = fair_yes - ask

            if edge < MIN_EDGE:
                continue

            buy_price = _clamp(round(min(0.95, fair_yes - MIN_EDGE, ask), 2), 0.05, 0.95)
            shares = max(1, math.floor(max_size / buy_price))
            cost = buy_price * shares

            bar_close = bar["close"]
            actual_dir = "UP" if bar_close > open_price else "DOWN"

            if direction == actual_dir:
                pnl = shares * 1.0 - cost
            else:
                pnl = -cost

            trades.append({
                "direction": direction,
                "actual": actual_dir,
                "win": direction == actual_dir,
                "momentum_bps": mom_bps,
                "accel_bps": accel,
                "vol_pct": vol_pct,
                "edge": edge,
                "fair_yes": fair_yes,
                "pnl": pnl,
                "cost": cost,
                "shares": shares,
                "buy_price": buy_price,
                "bar_open": open_price,
                "bar_close": bar_close,
            })
            break

    return trades


# ============================================================
# MAIN
# ============================================================

def main():
    print("Fetching 7 days of BTC 1-min data from Binance...")
    bars_1m = fetch_binance_klines(days=7)
    print(f"  Got {len(bars_1m)} 1-min bars")

    bars_5m = aggregate_5m(bars_1m)
    print(f"  Aggregated to {len(bars_5m)} 5-min bars")

    if len(bars_5m) < 50:
        print("ERROR: Not enough data")
        return

    date_start = datetime.fromtimestamp(bars_1m[0]["ts"] / 1000, tz=timezone.utc)
    date_end = datetime.fromtimestamp(bars_1m[-1]["ts"] / 1000, tz=timezone.utc)
    natural_up = sum(1 for b in bars_5m if b["close"] > b["open"]) / len(bars_5m) * 100
    print(f"  Date range: {date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}")
    print(f"  Natural UP%: {natural_up:.1f}%")

    # ---- V3.0 ----
    print("\nSimulating V3.0...")
    v30_trades = simulate_v30(bars_5m, bars_1m)
    v30_stats = analyze(v30_trades, "V3.0 — Simple Momentum (2min wait, 15bp, $5 flat)")

    # ---- V4.2 full size ----
    print("\nSimulating V4.2 (full $50-150 sizing)...")
    v42_trades = simulate_v42(bars_5m, bars_1m)
    v42_stats = analyze(v42_trades, "V4.2 — Full ($50-150 sizing, all gates)")

    # ---- V4.2 capped to $5 for fair WR comparison ----
    print("\nSimulating V4.2 ($5 sizing — fair WR comparison)...")
    v42_5_trades = simulate_v42_with_cap(bars_5m, bars_1m, max_size=5.0)
    v42_5_stats = analyze(v42_5_trades, "V4.2 — $5 sizing (fair WR comparison)")

    # ---- V4.2 capped to $20 for $43 account ----
    print("\nSimulating V4.2 ($20 sizing — fits $43 account)...")
    v42_20_trades = simulate_v42_with_cap(bars_5m, bars_1m, max_size=20.0)
    v42_20_stats = analyze(v42_20_trades, "V4.2 — $20 sizing (fits $43 account)")

    # ---- HEAD TO HEAD ----
    print("\n" + "="*60)
    print("  HEAD-TO-HEAD COMPARISON")
    print("="*60)

    def row(label, v30, v42, v42_5, v42_20):
        print(f"  {label:20s}  {v30:>12s}  {v42:>12s}  {v42_5:>12s}  {v42_20:>12s}")

    row("", "V3.0 ($5)", "V4.2 ($50-150)", "V4.2 ($5)", "V4.2 ($20)")
    row("-"*20, "-"*12, "-"*12, "-"*12, "-"*12)

    for label, key, fmt in [
        ("Trades", "trades", "d"),
        ("Win Rate", "win_rate", ".1f"),
        ("Total PnL", "total_pnl", "+,.2f"),
        ("Avg PnL/trade", "avg_pnl", "+.4f"),
        ("Avg Win", "avg_win", "+.4f"),
        ("Avg Loss", "avg_loss", "+.4f"),
        ("Max Drawdown", "max_drawdown", ",.2f"),
        ("Total Cost", "total_cost", ",.2f"),
        ("ROI%", "roi_pct", "+.2f"),
        ("Win Streak", "max_win_streak", "d"),
        ("Loss Streak", "max_loss_streak", "d"),
    ]:
        vals = []
        for s in [v30_stats, v42_stats, v42_5_stats, v42_20_stats]:
            v = s.get(key, 0) if s else 0
            if fmt.endswith("f"):
                vals.append(f"${v:{fmt}}" if "pnl" in key.lower() or key in ["avg_win","avg_loss","max_drawdown","total_cost"] else f"{v:{fmt}}")
            elif fmt == "d":
                vals.append(f"{v}")
            else:
                vals.append(f"{v:{fmt}}")
        row(label, *vals)

    # ---- VERDICT ----
    print("\n" + "="*60)
    print("  VERDICT")
    print("="*60)

    v30_wr = v30_stats.get("win_rate", 0) if v30_stats else 0
    v42_5_wr = v42_5_stats.get("win_rate", 0) if v42_5_stats else 0
    v30_pnl = v30_stats.get("total_pnl", 0) if v30_stats else 0
    v42_20_pnl = v42_20_stats.get("total_pnl", 0) if v42_20_stats else 0
    v30_trades_n = v30_stats.get("trades", 0) if v30_stats else 0
    v42_5_trades_n = v42_5_stats.get("trades", 0) if v42_5_stats else 0

    print(f"  Win Rate:  V3.0={v30_wr:.1f}%  vs  V4.2={v42_5_wr:.1f}%  (same $5 sizing)")
    print(f"  Trades:    V3.0={v30_trades_n}  vs  V4.2={v42_5_trades_n}")
    print(f"  PnL ($20): V3.0=${v30_pnl:+,.2f}  vs  V4.2=${v42_20_pnl:+,.2f}")

    if v42_5_wr > v30_wr + 3:
        print(f"\n  >> V4.2 has significantly higher WR (+{v42_5_wr - v30_wr:.1f}pp)")
        print(f"     BUT trades {v42_5_trades_n} vs {v30_trades_n} times (fewer opportunities)")
    elif v30_wr > v42_5_wr + 3:
        print(f"\n  >> V3.0 has significantly higher WR (+{v30_wr - v42_5_wr:.1f}pp)")
    else:
        print(f"\n  >> Similar WR. Difference is {abs(v42_5_wr - v30_wr):.1f}pp")

    if v42_5_trades_n < v30_trades_n * 0.3:
        print(f"  >> V4.2 gates out {(1 - v42_5_trades_n/v30_trades_n)*100:.0f}% of trades")
        print(f"     More selective but fewer opportunities = less total PnL potential")

    # Save results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": f"{date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}",
        "bars_1m": len(bars_1m),
        "bars_5m": len(bars_5m),
        "natural_up_pct": natural_up,
        "v30": v30_stats,
        "v42_full": v42_stats,
        "v42_5usd": v42_5_stats,
        "v42_20usd": v42_20_stats,
    }

    out = Path(__file__).parent / "backtest_v3_vs_v42_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
