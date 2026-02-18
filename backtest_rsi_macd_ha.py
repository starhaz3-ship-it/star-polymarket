"""
Backtest: RSI + MACD on Heikin Ashi 1-min BTC candles for Polymarket 5M/15M markets.

Strategy (from screenshot guide):
- SHORT/DOWN: RSI overbought (70+) + turning down, MACD histogram shrinking/crossing down
- LONG/UP: RSI oversold (30-) + turning up, MACD histogram growing/crossing up
- Heikin Ashi candles smooth noise
- Best hours: 04:00-10:00 AM EST (09:00-15:00 UTC)
- 15M: enter 7-9 min into period
- 5M: enter last 2 min

Simulates binary outcome: does BTC close higher or lower than open for each N-min window?
Entry prices modeled at various CLOB price levels.
"""

import json
import math
import time
import requests
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict


# ─── Data Fetching ───────────────────────────────────────────────────────────

def fetch_binance_klines(symbol="BTCUSDT", interval="1m", days=14):
    """Fetch 1-minute candles from Binance REST API."""
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    current = start_ms
    print(f"Fetching {days}d of {interval} {symbol} candles...")
    while current < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": current, "limit": 1000}
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if not data:
            break
        all_klines.extend(data)
        current = data[-1][0] + 60000
        time.sleep(0.12)
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
    print(f"  Fetched {len(bars)} candles")
    return bars


# ─── Heikin Ashi ─────────────────────────────────────────────────────────────

def to_heikin_ashi(bars):
    """Convert OHLC bars to Heikin Ashi."""
    ha = []
    for i, b in enumerate(bars):
        ha_close = (b["open"] + b["high"] + b["low"] + b["close"]) / 4
        if i == 0:
            ha_open = (b["open"] + b["close"]) / 2
        else:
            ha_open = (ha[i - 1]["open"] + ha[i - 1]["close"]) / 2
        ha_high = max(b["high"], ha_open, ha_close)
        ha_low = min(b["low"], ha_open, ha_close)
        ha.append({
            "ts": b["ts"],
            "open": ha_open,
            "high": ha_high,
            "low": ha_low,
            "close": ha_close,
            "is_green": ha_close >= ha_open,
            "volume": b["volume"],
        })
    return ha


# ─── Indicators ──────────────────────────────────────────────────────────────

def calc_rsi_series(closes, period=14):
    """Calculate RSI series using Wilder's smoothing (matches TradingView)."""
    rsi = np.full(len(closes), np.nan)
    if len(closes) < period + 1:
        return rsi

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Initial SMA seed
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100 - (100 / (1 + avg_gain / avg_loss))

    # Wilder smoothing
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100 - (100 / (1 + avg_gain / avg_loss))

    return rsi


def calc_ema(values, period):
    """Calculate EMA series."""
    ema = np.full(len(values), np.nan)
    k = 2.0 / (period + 1)

    # Find first valid value
    start = 0
    while start < len(values) and np.isnan(values[start]):
        start += 1
    if start >= len(values):
        return ema

    # Seed with SMA
    if start + period - 1 < len(values):
        ema[start + period - 1] = np.mean(values[start:start + period])
        for i in range(start + period, len(values)):
            ema[i] = values[i] * k + ema[i - 1] * (1 - k)
    return ema


def calc_macd(closes, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram series."""
    closes = np.array(closes, dtype=float)
    ema_fast = np.full(len(closes), np.nan)
    ema_slow = np.full(len(closes), np.nan)

    k_f = 2.0 / (fast + 1)
    k_s = 2.0 / (slow + 1)

    # Seed EMAs with SMA
    if len(closes) >= fast:
        ema_fast[fast - 1] = np.mean(closes[:fast])
        for i in range(fast, len(closes)):
            ema_fast[i] = closes[i] * k_f + ema_fast[i - 1] * (1 - k_f)

    if len(closes) >= slow:
        ema_slow[slow - 1] = np.mean(closes[:slow])
        for i in range(slow, len(closes)):
            ema_slow[i] = closes[i] * k_s + ema_slow[i - 1] * (1 - k_s)

    macd_line = ema_fast - ema_slow

    # Signal line = EMA of MACD line
    signal_line = calc_ema(macd_line, signal)

    # Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# ─── Strategy Signal Logic ───────────────────────────────────────────────────

def check_rsi_macd_signal(rsi_series, hist_series, idx, lookback=3):
    """
    Check for RSI+MACD signal at given index.

    DOWN signal:
      - RSI was overbought (>70) in recent lookback OR currently >65 and turning down
      - RSI slope is negative (turning down)
      - MACD histogram is shrinking (getting less positive) or crossing below zero

    UP signal:
      - RSI was oversold (<30) in recent lookback OR currently <35 and turning up
      - RSI slope is positive (turning up)
      - MACD histogram is growing (getting less negative) or crossing above zero

    Returns: "UP", "DOWN", or None
    """
    if idx < lookback + 1:
        return None

    rsi_now = rsi_series[idx]
    if np.isnan(rsi_now):
        return None

    hist_now = hist_series[idx]
    if np.isnan(hist_now):
        return None

    # RSI values over lookback window
    rsi_window = rsi_series[idx - lookback:idx + 1]
    hist_window = hist_series[idx - lookback:idx + 1]

    if any(np.isnan(rsi_window)) or any(np.isnan(hist_window)):
        return None

    rsi_max_recent = np.max(rsi_window)
    rsi_min_recent = np.min(rsi_window)
    rsi_slope = rsi_window[-1] - rsi_window[-2]

    # Histogram slope (is it shrinking or growing?)
    hist_slope = hist_window[-1] - hist_window[-2]

    # ──── DOWN / SHORT signal ────
    # RSI was overbought or near overbought, now turning down
    rsi_overbought = rsi_max_recent >= 65 and rsi_slope < 0
    # MACD histogram shrinking (positive going smaller, or crossing negative)
    macd_bearish = (hist_now > 0 and hist_slope < 0) or (hist_now < 0 and hist_window[-2] >= 0)

    if rsi_overbought and macd_bearish:
        # Stronger if RSI was actually 70+
        return "DOWN"

    # ──── UP / LONG signal ────
    rsi_oversold = rsi_min_recent <= 35 and rsi_slope > 0
    macd_bullish = (hist_now < 0 and hist_slope > 0) or (hist_now > 0 and hist_window[-2] <= 0)

    if rsi_oversold and macd_bullish:
        return "UP"

    return None


def check_strict_signal(rsi_series, hist_series, idx, lookback=3):
    """Strict version: requires RSI 70+/30- (as in the screenshot)."""
    if idx < lookback + 1:
        return None

    rsi_now = rsi_series[idx]
    hist_now = hist_series[idx]
    if np.isnan(rsi_now) or np.isnan(hist_now):
        return None

    rsi_window = rsi_series[idx - lookback:idx + 1]
    hist_window = hist_series[idx - lookback:idx + 1]
    if any(np.isnan(rsi_window)) or any(np.isnan(hist_window)):
        return None

    rsi_slope = rsi_window[-1] - rsi_window[-2]
    hist_slope = hist_window[-1] - hist_window[-2]

    # STRICT DOWN: RSI was 70+ and turning down
    if np.max(rsi_window) >= 70 and rsi_slope < 0:
        if (hist_now > 0 and hist_slope < 0) or (hist_now < 0 and hist_window[-2] >= 0):
            return "DOWN"

    # STRICT UP: RSI was 30- and turning up
    if np.min(rsi_window) <= 30 and rsi_slope > 0:
        if (hist_now < 0 and hist_slope > 0) or (hist_now > 0 and hist_window[-2] <= 0):
            return "UP"

    return None


# ─── Backtest Engine ─────────────────────────────────────────────────────────

def run_backtest(bars_1m, window_minutes=5, entry_offset_minutes=None, hour_filter=None):
    """
    Backtest RSI+MACD Heikin Ashi strategy on N-minute Polymarket windows.

    Args:
        bars_1m: raw 1-min OHLCV bars
        window_minutes: 5 or 15 (market duration)
        entry_offset_minutes: minutes into window to check signal
            - 15M default: 8 (enters at minute 8 of 15)
            - 5M default: 3 (enters at minute 3 of 5, i.e., last 2 min)
        hour_filter: tuple (start_utc, end_utc) to filter best hours, e.g. (9, 15) for 04-10 EST
    """
    if entry_offset_minutes is None:
        entry_offset_minutes = 8 if window_minutes == 15 else 3

    # Convert to Heikin Ashi
    ha_bars = to_heikin_ashi(bars_1m)
    ha_closes = np.array([b["close"] for b in ha_bars])

    # Also keep raw closes for outcome determination
    raw_closes = np.array([b["close"] for b in bars_1m])
    raw_opens = np.array([b["open"] for b in bars_1m])

    # Calculate indicators on HA closes
    rsi = calc_rsi_series(ha_closes, period=14)
    macd_line, signal_line, histogram = calc_macd(ha_closes, fast=12, slow=26, signal=9)

    # Build N-minute windows aligned to clean boundaries
    # Polymarket 5M/15M markets start on exact 5/15 min boundaries (UTC)
    results = {"relaxed": [], "strict": []}
    timestamps = [b["ts"] for b in bars_1m]

    # Find window boundaries
    first_ts = timestamps[0]
    first_dt = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)
    # Align to window boundary
    minute_align = (first_dt.minute // window_minutes) * window_minutes
    start_dt = first_dt.replace(minute=minute_align, second=0, microsecond=0)
    if start_dt < first_dt:
        start_dt += timedelta(minutes=window_minutes)

    current_window_start = start_dt
    end_dt = datetime.fromtimestamp(timestamps[-1] / 1000, tz=timezone.utc)

    while current_window_start + timedelta(minutes=window_minutes) <= end_dt:
        window_end = current_window_start + timedelta(minutes=window_minutes)
        entry_time = current_window_start + timedelta(minutes=entry_offset_minutes)

        # Hour filter (UTC)
        if hour_filter:
            h = current_window_start.hour
            if hour_filter[0] <= hour_filter[1]:
                if not (hour_filter[0] <= h < hour_filter[1]):
                    current_window_start += timedelta(minutes=window_minutes)
                    continue
            else:  # wraps midnight
                if not (h >= hour_filter[0] or h < hour_filter[1]):
                    current_window_start += timedelta(minutes=window_minutes)
                    continue

        # Find bar indices
        window_start_ms = int(current_window_start.timestamp() * 1000)
        window_end_ms = int(window_end.timestamp() * 1000)
        entry_ms = int(entry_time.timestamp() * 1000)

        # Find entry bar index (closest bar at or before entry time)
        entry_idx = None
        for i, ts in enumerate(timestamps):
            if ts <= entry_ms:
                entry_idx = i
            else:
                break

        # Find window open and close bars for outcome
        open_idx = None
        close_idx = None
        for i, ts in enumerate(timestamps):
            if ts >= window_start_ms and open_idx is None:
                open_idx = i
            if ts < window_end_ms:
                close_idx = i

        if entry_idx is None or open_idx is None or close_idx is None:
            current_window_start += timedelta(minutes=window_minutes)
            continue
        if close_idx <= open_idx:
            current_window_start += timedelta(minutes=window_minutes)
            continue

        # Determine actual outcome: did BTC go UP or DOWN in this window?
        window_open_price = bars_1m[open_idx]["open"]
        window_close_price = bars_1m[close_idx]["close"]
        pct_move = (window_close_price - window_open_price) / window_open_price * 100

        if abs(pct_move) < 0.001:
            # Flat — skip (too close to call, Polymarket would be ~50/50)
            current_window_start += timedelta(minutes=window_minutes)
            continue

        actual_direction = "UP" if pct_move > 0 else "DOWN"

        # Check signals
        for variant, check_fn in [("relaxed", check_rsi_macd_signal), ("strict", check_strict_signal)]:
            signal = check_fn(rsi, histogram, entry_idx, lookback=3)
            if signal is not None:
                win = signal == actual_direction
                results[variant].append({
                    "time": current_window_start.strftime("%Y-%m-%d %H:%M UTC"),
                    "hour_utc": current_window_start.hour,
                    "signal": signal,
                    "actual": actual_direction,
                    "win": win,
                    "pct_move": round(pct_move, 4),
                    "rsi_at_entry": round(float(rsi[entry_idx]), 1),
                    "hist_at_entry": round(float(histogram[entry_idx]), 4),
                })

        current_window_start += timedelta(minutes=window_minutes)

    return results


# ─── PnL Simulation ─────────────────────────────────────────────────────────

def simulate_pnl(trades, entry_price=0.65, size_usd=5.0):
    """
    Simulate PnL for binary outcome trades.
    Win pays $1/share - cost. Loss pays -cost.
    Polymarket fee ~2% on winnings.
    """
    FEE_RATE = 0.02
    total_pnl = 0.0
    wins = 0
    losses = 0

    for t in trades:
        shares = math.floor(size_usd / entry_price)
        if shares < 1:
            continue
        cost = entry_price * shares

        if t["win"]:
            gross = shares * 1.0  # $1 per share on win
            profit = gross - cost
            fee = profit * FEE_RATE
            net = profit - fee
            total_pnl += net
            wins += 1
        else:
            total_pnl -= cost
            losses += 1

    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / total, 3) if total > 0 else 0,
        "entry_price": entry_price,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    DAYS = 14
    SIZE_USD = 5.0

    # Fetch data
    bars_1m = fetch_binance_klines(days=DAYS)

    total_windows_5m = DAYS * 24 * 12   # 12 per hour
    total_windows_15m = DAYS * 24 * 4   # 4 per hour

    print(f"\n{'='*80}")
    print(f"RSI + MACD HEIKIN ASHI BACKTEST — {DAYS} days of 1-min BTC candles")
    print(f"{'='*80}")

    # ──── Run all variants ────
    configs = [
        # (label, window_min, entry_offset, hour_filter)
        ("5M — All Hours (entry min 3)", 5, 3, None),
        ("5M — EST 4-10AM / UTC 9-15 (entry min 3)", 5, 3, (9, 15)),
        ("5M — All Hours (entry min 4, last 1 min)", 5, 4, None),
        ("15M — All Hours (entry min 8)", 15, 8, None),
        ("15M — EST 4-10AM / UTC 9-15 (entry min 8)", 15, 8, (9, 15)),
        ("15M — All Hours (entry min 10)", 15, 10, None),
    ]

    all_results = {}

    for label, wm, eo, hf in configs:
        res = run_backtest(bars_1m, window_minutes=wm, entry_offset_minutes=eo, hour_filter=hf)
        all_results[label] = res

        print(f"\n{'-'*80}")
        print(f"  {label}")
        print(f"{'-'*80}")

        for variant in ["relaxed", "strict"]:
            trades = res[variant]
            if not trades:
                print(f"  [{variant.upper()}] No signals generated")
                continue

            wins = sum(1 for t in trades if t["win"])
            total = len(trades)
            wr = wins / total * 100

            # Signal frequency
            if wm == 5:
                total_possible = total_windows_5m
                if hf:
                    total_possible = int(total_possible * 6 / 24)
            else:
                total_possible = total_windows_15m
                if hf:
                    total_possible = int(total_possible * 6 / 24)

            freq = total / total_possible * 100 if total_possible > 0 else 0

            print(f"\n  [{variant.upper()}] {wins}W / {total - wins}L  ({wr:.1f}% WR)  |  "
                  f"Signal frequency: {total}/{total_possible} windows ({freq:.1f}%)")

            # PnL at different entry prices
            print(f"  {'Entry Price':>12}  {'WR':>6}  {'PnL':>8}  {'Avg PnL':>8}  {'Trades':>6}")
            for ep in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
                pnl = simulate_pnl(trades, entry_price=ep, size_usd=SIZE_USD)
                marker = " ***" if pnl["total_pnl"] > 0 else ""
                print(f"  ${ep:.2f}          {pnl['win_rate']:>5.1f}%  ${pnl['total_pnl']:>+7.2f}  "
                      f"${pnl['avg_pnl']:>+7.3f}  {pnl['trades']:>5}{marker}")

            # Direction breakdown
            up_signals = [t for t in trades if t["signal"] == "UP"]
            dn_signals = [t for t in trades if t["signal"] == "DOWN"]
            up_wr = sum(1 for t in up_signals if t["win"]) / len(up_signals) * 100 if up_signals else 0
            dn_wr = sum(1 for t in dn_signals if t["win"]) / len(dn_signals) * 100 if dn_signals else 0
            print(f"  Direction: UP signals {len(up_signals)} ({up_wr:.0f}% WR) | "
                  f"DOWN signals {len(dn_signals)} ({dn_wr:.0f}% WR)")

            # Hour breakdown
            hour_stats = defaultdict(lambda: {"w": 0, "l": 0})
            for t in trades:
                h = t["hour_utc"]
                if t["win"]:
                    hour_stats[h]["w"] += 1
                else:
                    hour_stats[h]["l"] += 1

            if hour_stats:
                print(f"  Hour breakdown (UTC):")
                for h in sorted(hour_stats.keys()):
                    s = hour_stats[h]
                    total_h = s["w"] + s["l"]
                    wr_h = s["w"] / total_h * 100
                    est_h = (h - 5) % 24
                    bar = "#" * s["w"] + "." * s["l"]
                    print(f"    {h:02d}:00 UTC ({est_h:02d} EST): {s['w']}W/{s['l']}L "
                          f"({wr_h:.0f}%) {bar}")

    # ──── Summary comparison ────
    print(f"\n{'='*80}")
    print(f"  SUMMARY — Best Configurations (at $0.65 entry, ${SIZE_USD} bet)")
    print(f"{'='*80}")

    summary = []
    for label, res in all_results.items():
        for variant in ["relaxed", "strict"]:
            trades = res[variant]
            if not trades:
                continue
            pnl = simulate_pnl(trades, entry_price=0.65, size_usd=SIZE_USD)
            summary.append({
                "label": f"{label} [{variant}]",
                **pnl,
            })

    summary.sort(key=lambda x: x["total_pnl"], reverse=True)
    print(f"  {'Configuration':<55} {'Trades':>6} {'WR':>6} {'PnL':>9}")
    print(f"  {'-'*55} {'-'*6} {'-'*6} {'-'*9}")
    for s in summary:
        marker = " <<<" if s["total_pnl"] > 0 else ""
        print(f"  {s['label']:<55} {s['trades']:>6} {s['win_rate']:>5.1f}% ${s['total_pnl']:>+7.2f}{marker}")

    # ──── Breakeven analysis ────
    print(f"\n{'='*80}")
    print(f"  BREAKEVEN ANALYSIS")
    print(f"{'='*80}")
    print(f"  At 2% Polymarket fee, breakeven win rates by entry price:")
    for ep in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        # Win: (1 - ep) * 0.98 per share. Loss: ep per share.
        # Breakeven: WR * win_per = (1-WR) * loss_per
        win_per = (1 - ep) * 0.98
        loss_per = ep
        be_wr = loss_per / (win_per + loss_per) * 100
        print(f"  ${ep:.2f} entry → need {be_wr:.1f}% WR to break even")

    # Save results
    out = {}
    for label, res in all_results.items():
        out[label] = {}
        for variant in ["relaxed", "strict"]:
            trades = res[variant]
            out[label][variant] = {
                "trade_count": len(trades),
                "win_rate": round(sum(1 for t in trades if t["win"]) / len(trades) * 100, 1) if trades else 0,
                "pnl_at_065": simulate_pnl(trades, 0.65, SIZE_USD)["total_pnl"] if trades else 0,
                "trades": trades[-20:],  # last 20 for inspection
            }

    outpath = Path(__file__).parent / "backtest_rsi_macd_ha_results.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
