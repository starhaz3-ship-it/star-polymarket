#!/usr/bin/env python3
"""
Open-Entry 5M Momentum Backtest for Polymarket BTC prediction markets.

Strategy: Enter EVERY 5M BTC candle at open, use first N seconds of price action
as directional signal. If BTC price > candle open -> bet UP, if < open -> bet DOWN.
Hold to resolution (5M candle close).

Data: Binance REST API (BTCUSDT 5M + 1M klines)
"""

import httpx
import numpy as np
import json
import time
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

# ─────────────────────────── CONFIG ───────────────────────────

SYMBOL = "BTCUSDT"
MIN_BARS_5M = 2000       # ~7 days of 5M bars
BATCH_SIZE = 1000         # Binance max per request
BASE_URL = "https://api.binance.com"
COST_PER_TRADE = 1.0      # $1 per trade for PnL sim

# Parameter grid
THRESHOLD_BPS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]
ENTRY_DELAY_SECS = [15, 20, 30, 45, 60, 90, 120, 180]
ENTRY_PRICES = [0.48, 0.50, 0.52, 0.55, 0.60, 0.65]

# Signal enhancements
RSI_PERIOD = 14
EMA_SHORT = 5
EMA_LONG = 20

OUTPUT_JSON = Path(__file__).parent / "backtest_open_entry_results.json"

# ─────────────────────────── DATA FETCH ───────────────────────────

def fetch_klines(symbol: str, interval: str, limit: int = 1000,
                 end_time: int | None = None) -> list[list]:
    """Fetch klines from Binance REST API."""
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time is not None:
        params["endTime"] = end_time

    with httpx.Client(timeout=30) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def fetch_all_klines(symbol: str, interval: str, min_bars: int) -> list[list]:
    """Page back from current time to get at least min_bars klines."""
    all_klines = []
    end_time = None
    retries = 0

    print(f"Fetching {interval} klines for {symbol}...")
    while len(all_klines) < min_bars:
        try:
            batch = fetch_klines(symbol, interval, BATCH_SIZE, end_time)
            if not batch:
                break
            all_klines = batch + all_klines
            # Next page ends before the earliest bar we have
            end_time = batch[0][0] - 1
            print(f"  ...{len(all_klines)} bars fetched so far")
            time.sleep(0.15)  # Rate limit courtesy
            retries = 0
        except Exception as e:
            retries += 1
            if retries > 5:
                print(f"  Failed after 5 retries: {e}")
                break
            print(f"  Retry {retries}: {e}")
            time.sleep(2 ** retries)

    # Deduplicate by open_time
    seen = set()
    deduped = []
    for k in all_klines:
        if k[0] not in seen:
            seen.add(k[0])
            deduped.append(k)
    deduped.sort(key=lambda x: x[0])
    print(f"  Total unique {interval} bars: {len(deduped)}")
    return deduped


def parse_klines(raw: list[list]) -> dict:
    """Parse raw Binance klines into numpy arrays.
    Kline format: [open_time, open, high, low, close, volume, close_time, ...]
    """
    n = len(raw)
    data = {
        "open_time": np.array([int(k[0]) for k in raw], dtype=np.int64),
        "open": np.array([float(k[1]) for k in raw], dtype=np.float64),
        "high": np.array([float(k[2]) for k in raw], dtype=np.float64),
        "low": np.array([float(k[3]) for k in raw], dtype=np.float64),
        "close": np.array([float(k[4]) for k in raw], dtype=np.float64),
        "volume": np.array([float(k[5]) for k in raw], dtype=np.float64),
        "close_time": np.array([int(k[6]) for k in raw], dtype=np.int64),
    }
    return data


# ─────────────────────────── INDICATORS ───────────────────────────

def compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI on close prices."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi = np.full(len(prices), np.nan)
    if len(deltas) < period:
        return rsi

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    # Fill the period-th element
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        first_avg_gain = np.mean(gains[:period])
        first_avg_loss = np.mean(losses[:period])
        if first_avg_loss == 0:
            rsi[period] = 100.0
        else:
            rsi[period] = 100.0 - (100.0 / (1.0 + first_avg_gain / first_avg_loss))

    return rsi


def compute_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA."""
    ema = np.full(len(prices), np.nan)
    if len(prices) < period:
        return ema
    multiplier = 2.0 / (period + 1)
    ema[period - 1] = np.mean(prices[:period])
    for i in range(period, len(prices)):
        ema[i] = prices[i] * multiplier + ema[i - 1] * (1 - multiplier)
    return ema


# ─────────────────────────── SIGNAL MAPPING ───────────────────────────

def build_1m_lookup(data_1m: dict) -> dict:
    """Build a dict mapping 1M open_time_ms -> close price for fast lookup."""
    lookup = {}
    for i in range(len(data_1m["open_time"])):
        lookup[int(data_1m["open_time"][i])] = data_1m["close"][i]
    return lookup


def get_price_at_delay(candle_open_time_ms: int, delay_sec: int,
                       lookup_1m: dict, candle_open_price: float) -> float | None:
    """Get the BTC price approximately delay_sec after the 5M candle opens.

    We find the 1M candle that contains the timestamp (open + delay_sec),
    and return its close as an approximation.

    For delay_sec=30, the relevant 1M candle starts at candle_open_time_ms + 0ms
    (the first minute candle). Its close approximates price at T+60s.
    For better granularity at 30s, we interpolate between open and close of
    the first 1M candle.
    """
    # Which 1M candle contains the target time?
    target_ms = candle_open_time_ms + delay_sec * 1000
    # 1M candle that contains this time starts at:
    minute_start = (target_ms // 60000) * 60000

    if minute_start in lookup_1m:
        close_of_minute = lookup_1m[minute_start]
        # The 1M candle spans [minute_start, minute_start + 60000)
        # Target is at some fraction through this minute
        frac = (target_ms - minute_start) / 60000.0
        # We also need the open of this 1M candle - approximate from previous close
        # For simplicity, if it's the first minute of the 5M bar, use candle open
        if minute_start == candle_open_time_ms:
            minute_open = candle_open_price
        else:
            # Use previous minute's close as this minute's open
            prev_minute = minute_start - 60000
            minute_open = lookup_1m.get(prev_minute, candle_open_price)

        # Linear interpolation within the minute
        return minute_open + frac * (close_of_minute - minute_open)

    # Fallback: try surrounding minutes
    for offset in [0, -60000, 60000]:
        check = minute_start + offset
        if check in lookup_1m:
            return lookup_1m[check]

    return None


# ─────────────────────────── CORE BACKTEST ───────────────────────────

def run_backtest(data_5m: dict, lookup_1m: dict, threshold_bp: float,
                 entry_delay_sec: int, entry_price: float,
                 rsi_filter: bool = False, ema_filter: bool = False,
                 volume_filter: bool = False,
                 rsi_values: np.ndarray | None = None,
                 ema_short: np.ndarray | None = None,
                 ema_long: np.ndarray | None = None,
                 volume_median: float = 0.0,
                 indices: np.ndarray | None = None) -> dict:
    """Run backtest on given data subset.

    Returns dict with WR, PnL, trades, sharpe, etc.
    """
    if indices is None:
        indices = np.arange(len(data_5m["open_time"]))

    wins = 0
    losses = 0
    skips = 0
    pnls = []

    for idx in indices:
        i = int(idx)
        candle_open = data_5m["open"][i]
        candle_close = data_5m["close"][i]
        candle_open_time = int(data_5m["open_time"][i])
        candle_volume = data_5m["volume"][i]

        # Skip doji candles (open == close)
        if candle_open == candle_close:
            skips += 1
            continue

        # Get price at delay
        price_at_delay = get_price_at_delay(candle_open_time, entry_delay_sec,
                                            lookup_1m, candle_open)
        if price_at_delay is None:
            skips += 1
            continue

        # Compute signal in basis points
        signal_bp = ((price_at_delay - candle_open) / candle_open) * 10000.0

        # Threshold filter
        if abs(signal_bp) < threshold_bp:
            skips += 1
            continue

        # Direction from signal
        direction = "UP" if signal_bp > 0 else "DOWN"

        # RSI filter
        if rsi_filter and rsi_values is not None:
            rsi_val = rsi_values[i]
            if np.isnan(rsi_val):
                skips += 1
                continue
            if direction == "UP" and rsi_val < 50:
                skips += 1
                continue
            if direction == "DOWN" and rsi_val >= 50:
                skips += 1
                continue

        # EMA filter
        if ema_filter and ema_short is not None and ema_long is not None:
            ema_s = ema_short[i]
            ema_l = ema_long[i]
            if np.isnan(ema_s) or np.isnan(ema_l):
                skips += 1
                continue
            if direction == "UP" and ema_s < ema_l:
                skips += 1
                continue
            if direction == "DOWN" and ema_s >= ema_l:
                skips += 1
                continue

        # Volume filter: only trade high-volume candles (above median)
        if volume_filter and candle_volume < volume_median:
            skips += 1
            continue

        # Resolution: UP if close > open, DOWN if close < open
        outcome = "UP" if candle_close > candle_open else "DOWN"

        # PnL
        if direction == outcome:
            pnl = (1.0 / entry_price - 1.0) * COST_PER_TRADE
            wins += 1
        else:
            pnl = -COST_PER_TRADE
            losses += 1
        pnls.append(pnl)

    total_trades = wins + losses
    wr = wins / total_trades * 100 if total_trades > 0 else 0.0
    total_pnl = sum(pnls) if pnls else 0.0
    avg_pnl = np.mean(pnls) if pnls else 0.0
    std_pnl = np.std(pnls) if len(pnls) > 1 else 0.0
    sharpe = (avg_pnl / std_pnl * np.sqrt(252 * 288)) if std_pnl > 0 else 0.0  # annualized

    return {
        "wins": wins,
        "losses": losses,
        "skips": skips,
        "total_trades": total_trades,
        "win_rate": round(wr, 2),
        "total_pnl": round(total_pnl, 4),
        "avg_pnl": round(avg_pnl, 6),
        "sharpe": round(sharpe, 4),
        "pnls": pnls,  # for bootstrap
    }


# ─────────────────────────── BOOTSTRAP ───────────────────────────

def bootstrap_confidence(pnls: list[float], n_samples: int = 10000,
                         ci: float = 0.95) -> dict:
    """Bootstrap confidence intervals on WR and PnL."""
    if len(pnls) < 10:
        return {"wr_ci": [0, 0], "pnl_ci": [0, 0], "n_trades": len(pnls)}

    pnls_arr = np.array(pnls)
    n = len(pnls_arr)
    rng = np.random.default_rng(42)

    boot_wrs = []
    boot_pnls = []

    for _ in range(n_samples):
        sample = rng.choice(pnls_arr, size=n, replace=True)
        boot_wins = np.sum(sample > 0)
        boot_wr = boot_wins / n * 100
        boot_wrs.append(boot_wr)
        boot_pnls.append(np.sum(sample))

    alpha = (1 - ci) / 2
    wr_low = np.percentile(boot_wrs, alpha * 100)
    wr_high = np.percentile(boot_wrs, (1 - alpha) * 100)
    pnl_low = np.percentile(boot_pnls, alpha * 100)
    pnl_high = np.percentile(boot_pnls, (1 - alpha) * 100)

    return {
        "wr_ci_95": [round(wr_low, 2), round(wr_high, 2)],
        "pnl_ci_95": [round(pnl_low, 4), round(pnl_high, 4)],
        "n_trades": n,
        "wr_mean": round(np.mean(boot_wrs), 2),
        "pnl_mean": round(np.mean(boot_pnls), 4),
    }


# ─────────────────────────── MAIN ───────────────────────────

def main():
    print("=" * 80)
    print("  OPEN-ENTRY 5M MOMENTUM BACKTEST")
    print("  Strategy: Enter every 5M candle, use first N seconds as signal")
    print("=" * 80)
    print()

    # ── Fetch data ──
    # 5M klines: need ~2000+ bars (~7 days)
    raw_5m = fetch_all_klines(SYMBOL, "5m", MIN_BARS_5M)
    if len(raw_5m) < 500:
        print(f"ERROR: Only got {len(raw_5m)} 5M bars, need at least 500")
        sys.exit(1)

    data_5m = parse_klines(raw_5m)
    print(f"\n5M data range: {datetime.fromtimestamp(data_5m['open_time'][0]/1000, tz=timezone.utc)} "
          f"to {datetime.fromtimestamp(data_5m['open_time'][-1]/1000, tz=timezone.utc)}")
    print(f"Total 5M bars: {len(data_5m['open_time'])}")

    # 1M klines: need to cover the same period for signal lookup
    # Each 5M bar has 5 x 1M bars, so we need ~5x as many
    start_ms = int(data_5m["open_time"][0])
    end_ms = int(data_5m["close_time"][-1])

    # Fetch 1M klines covering the same period
    raw_1m = []
    current_end = end_ms
    print(f"\nFetching 1M klines for signal interpolation...")
    while True:
        batch = fetch_klines(SYMBOL, "1m", BATCH_SIZE, current_end)
        if not batch:
            break
        raw_1m = batch + raw_1m
        earliest = batch[0][0]
        if earliest <= start_ms:
            break
        current_end = earliest - 1
        time.sleep(0.15)
        print(f"  ...{len(raw_1m)} 1M bars fetched")

    # Deduplicate
    seen = set()
    deduped_1m = []
    for k in raw_1m:
        if k[0] not in seen:
            seen.add(k[0])
            deduped_1m.append(k)
    deduped_1m.sort(key=lambda x: x[0])
    print(f"  Total unique 1M bars: {len(deduped_1m)}")

    data_1m = parse_klines(deduped_1m)
    lookup_1m = build_1m_lookup(data_1m)

    # ── Precompute indicators on 5M data ──
    rsi_values = compute_rsi(data_5m["close"], RSI_PERIOD)
    ema_short = compute_ema(data_5m["close"], EMA_SHORT)
    ema_long = compute_ema(data_5m["close"], EMA_LONG)
    volume_median = float(np.median(data_5m["volume"]))

    n_total = len(data_5m["open_time"])
    n_train = int(n_total * 0.7)
    train_indices = np.arange(n_train)
    test_indices = np.arange(n_train, n_total)

    print(f"\nTrain set: {n_train} bars ({n_train/288:.1f} days)")
    print(f"Test set:  {n_total - n_train} bars ({(n_total-n_train)/288:.1f} days)")

    # ── Parameter sweep ──
    print(f"\nRunning parameter sweep: {len(THRESHOLD_BPS)} x {len(ENTRY_DELAY_SECS)} x {len(ENTRY_PRICES)} = "
          f"{len(THRESHOLD_BPS)*len(ENTRY_DELAY_SECS)*len(ENTRY_PRICES)} combos")
    print("-" * 120)

    all_results = []
    combo_count = 0
    total_combos = len(THRESHOLD_BPS) * len(ENTRY_DELAY_SECS) * len(ENTRY_PRICES)

    for threshold_bp, delay_sec, entry_price in product(THRESHOLD_BPS, ENTRY_DELAY_SECS, ENTRY_PRICES):
        combo_count += 1
        if combo_count % 50 == 0:
            print(f"  Progress: {combo_count}/{total_combos}")

        # Full backtest
        result_full = run_backtest(
            data_5m, lookup_1m, threshold_bp, delay_sec, entry_price,
        )

        # Walk-forward: train
        result_train = run_backtest(
            data_5m, lookup_1m, threshold_bp, delay_sec, entry_price,
            indices=train_indices,
        )

        # Walk-forward: test
        result_test = run_backtest(
            data_5m, lookup_1m, threshold_bp, delay_sec, entry_price,
            indices=test_indices,
        )

        combo_result = {
            "threshold_bp": threshold_bp,
            "entry_delay_sec": delay_sec,
            "entry_price": entry_price,
            "full": {k: v for k, v in result_full.items() if k != "pnls"},
            "train": {k: v for k, v in result_train.items() if k != "pnls"},
            "test": {k: v for k, v in result_test.items() if k != "pnls"},
            "_pnls_full": result_full["pnls"],  # Keep for bootstrap on best
        }
        all_results.append(combo_result)

    # ── Signal enhancements ──
    print("\n\nRunning signal enhancement tests...")
    print("-" * 120)

    # Use middle-ground parameters for enhancement comparison
    base_threshold = 2.0
    base_delay = 30
    base_entry = 0.50

    # Baseline (no filters)
    baseline = run_backtest(data_5m, lookup_1m, base_threshold, base_delay, base_entry)

    # RSI filter
    rsi_result = run_backtest(
        data_5m, lookup_1m, base_threshold, base_delay, base_entry,
        rsi_filter=True, rsi_values=rsi_values,
    )

    # EMA filter
    ema_result = run_backtest(
        data_5m, lookup_1m, base_threshold, base_delay, base_entry,
        ema_filter=True, ema_short=ema_short, ema_long=ema_long,
    )

    # Volume filter
    vol_result = run_backtest(
        data_5m, lookup_1m, base_threshold, base_delay, base_entry,
        volume_filter=True,
    )

    # RSI + EMA combined
    rsi_ema_result = run_backtest(
        data_5m, lookup_1m, base_threshold, base_delay, base_entry,
        rsi_filter=True, ema_filter=True,
        rsi_values=rsi_values, ema_short=ema_short, ema_long=ema_long,
    )

    # RSI + Volume combined
    rsi_vol_result = run_backtest(
        data_5m, lookup_1m, base_threshold, base_delay, base_entry,
        rsi_filter=True, volume_filter=True,
        rsi_values=rsi_values,
    )

    # All filters
    all_filters_result = run_backtest(
        data_5m, lookup_1m, base_threshold, base_delay, base_entry,
        rsi_filter=True, ema_filter=True, volume_filter=True,
        rsi_values=rsi_values, ema_short=ema_short, ema_long=ema_long,
    )

    enhancements = {
        "base_params": {"threshold_bp": base_threshold, "delay_sec": base_delay, "entry_price": base_entry},
        "baseline": {k: v for k, v in baseline.items() if k != "pnls"},
        "rsi_filter": {k: v for k, v in rsi_result.items() if k != "pnls"},
        "ema_filter": {k: v for k, v in ema_result.items() if k != "pnls"},
        "volume_filter": {k: v for k, v in vol_result.items() if k != "pnls"},
        "rsi_ema_combined": {k: v for k, v in rsi_ema_result.items() if k != "pnls"},
        "rsi_volume_combined": {k: v for k, v in rsi_vol_result.items() if k != "pnls"},
        "all_filters": {k: v for k, v in all_filters_result.items() if k != "pnls"},
    }

    # ── Sort by full PnL and find best ──
    all_results.sort(key=lambda x: x["full"]["total_pnl"], reverse=True)
    best = all_results[0]

    # Bootstrap on best combo
    print("\nBootstrapping confidence intervals on best combo...")
    bootstrap = bootstrap_confidence(best["_pnls_full"])

    # ── Walk-forward on best ──
    wf_train_wr = best["train"]["win_rate"]
    wf_test_wr = best["test"]["win_rate"]
    wf_train_pnl = best["train"]["total_pnl"]
    wf_test_pnl = best["test"]["total_pnl"]

    # ── Print results ──
    print("\n" + "=" * 120)
    print("  TOP 30 PARAMETER COMBOS (sorted by PnL)")
    print("=" * 120)
    header = f"{'Rank':>4} | {'Thresh':>6} | {'Delay':>5} | {'Entry$':>6} | {'Trades':>6} | {'WR%':>6} | {'PnL':>10} | {'Sharpe':>8} | {'Train WR':>8} | {'Test WR':>7} | {'Test PnL':>9}"
    print(header)
    print("-" * 120)

    for rank, r in enumerate(all_results[:30], 1):
        print(f"{rank:>4} | {r['threshold_bp']:>6.1f} | {r['entry_delay_sec']:>5} | {r['entry_price']:>6.2f} | "
              f"{r['full']['total_trades']:>6} | {r['full']['win_rate']:>5.1f}% | "
              f"${r['full']['total_pnl']:>9.2f} | {r['full']['sharpe']:>8.2f} | "
              f"{r['train']['win_rate']:>7.1f}% | {r['test']['win_rate']:>6.1f}% | "
              f"${r['test']['total_pnl']:>8.2f}")

    # Bottom 10
    print(f"\n{'-'*120}")
    print("  BOTTOM 10 (worst combos)")
    print(f"{'-'*120}")
    for rank, r in enumerate(all_results[-10:], len(all_results) - 9):
        print(f"{rank:>4} | {r['threshold_bp']:>6.1f} | {r['entry_delay_sec']:>5} | {r['entry_price']:>6.2f} | "
              f"{r['full']['total_trades']:>6} | {r['full']['win_rate']:>5.1f}% | "
              f"${r['full']['total_pnl']:>9.2f} | {r['full']['sharpe']:>8.2f} | "
              f"{r['train']['win_rate']:>7.1f}% | {r['test']['win_rate']:>6.1f}% | "
              f"${r['test']['total_pnl']:>8.2f}")

    # Best combo summary
    print(f"\n{'='*80}")
    print(f"  BEST COMBO")
    print(f"{'='*80}")
    print(f"  Threshold:     {best['threshold_bp']} bp")
    print(f"  Entry Delay:   {best['entry_delay_sec']}s")
    print(f"  Entry Price:   ${best['entry_price']}")
    print(f"  Total Trades:  {best['full']['total_trades']}")
    print(f"  Win Rate:      {best['full']['win_rate']}%")
    print(f"  Total PnL:     ${best['full']['total_pnl']:.2f}")
    print(f"  Sharpe:        {best['full']['sharpe']:.4f}")
    print(f"\n  Walk-Forward:")
    print(f"    Train WR: {wf_train_wr}%  |  Test WR: {wf_test_wr}%")
    print(f"    Train PnL: ${wf_train_pnl:.2f}  |  Test PnL: ${wf_test_pnl:.2f}")
    print(f"\n  Bootstrap 95% CI (10,000 samples):")
    print(f"    WR:  [{bootstrap['wr_ci_95'][0]}%, {bootstrap['wr_ci_95'][1]}%]  mean={bootstrap['wr_mean']}%")
    print(f"    PnL: [${bootstrap['pnl_ci_95'][0]}, ${bootstrap['pnl_ci_95'][1]}]  mean=${bootstrap['pnl_mean']}")

    # Signal enhancements
    print(f"\n{'='*80}")
    print(f"  SIGNAL ENHANCEMENTS (threshold={base_threshold}bp, delay={base_delay}s, entry=${base_entry})")
    print(f"{'='*80}")
    print(f"{'Filter':<25} | {'Trades':>6} | {'WR%':>6} | {'PnL':>10} | {'Sharpe':>8} | {'WR Delta':>8}")
    print("-" * 80)
    base_wr = enhancements["baseline"]["win_rate"]
    for name, key in [("No Filter (baseline)", "baseline"),
                      ("RSI(14) filter", "rsi_filter"),
                      ("EMA(5/20) cross", "ema_filter"),
                      ("Volume > median", "volume_filter"),
                      ("RSI + EMA", "rsi_ema_combined"),
                      ("RSI + Volume", "rsi_volume_combined"),
                      ("All filters", "all_filters")]:
        r = enhancements[key]
        delta = r["win_rate"] - base_wr
        sign = "+" if delta >= 0 else ""
        print(f"{name:<25} | {r['total_trades']:>6} | {r['win_rate']:>5.1f}% | "
              f"${r['total_pnl']:>9.2f} | {r['sharpe']:>8.2f} | {sign}{delta:>6.1f}%")

    # ── Save results ──
    output = {
        "metadata": {
            "symbol": SYMBOL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_5m_bars": len(data_5m["open_time"]),
            "total_1m_bars": len(data_1m["open_time"]),
            "data_start": datetime.fromtimestamp(data_5m["open_time"][0]/1000, tz=timezone.utc).isoformat(),
            "data_end": datetime.fromtimestamp(data_5m["open_time"][-1]/1000, tz=timezone.utc).isoformat(),
            "train_bars": int(n_train),
            "test_bars": int(n_total - n_train),
            "cost_per_trade": COST_PER_TRADE,
        },
        "best_combo": {
            "threshold_bp": best["threshold_bp"],
            "entry_delay_sec": best["entry_delay_sec"],
            "entry_price": best["entry_price"],
            "full": best["full"],
            "train": best["train"],
            "test": best["test"],
            "bootstrap": bootstrap,
        },
        "top_20_combos": [
            {k: v for k, v in r.items() if k != "_pnls_full"}
            for r in all_results[:20]
        ],
        "bottom_10_combos": [
            {k: v for k, v in r.items() if k != "_pnls_full"}
            for r in all_results[-10:]
        ],
        "signal_enhancements": enhancements,
        "all_combos_summary": [
            {
                "threshold_bp": r["threshold_bp"],
                "entry_delay_sec": r["entry_delay_sec"],
                "entry_price": r["entry_price"],
                "wr": r["full"]["win_rate"],
                "pnl": r["full"]["total_pnl"],
                "trades": r["full"]["total_trades"],
                "sharpe": r["full"]["sharpe"],
                "test_wr": r["test"]["win_rate"],
                "test_pnl": r["test"]["total_pnl"],
            }
            for r in all_results
        ],
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_JSON}")

    # ── Key insights ──
    print(f"\n{'='*80}")
    print("  KEY INSIGHTS")
    print(f"{'='*80}")

    # Win rate by entry price
    print("\n  Win Rate by Entry Price (averaged across all combos):")
    for ep in ENTRY_PRICES:
        subset = [r for r in all_results if r["entry_price"] == ep]
        avg_wr = np.mean([r["full"]["win_rate"] for r in subset])
        avg_pnl = np.mean([r["full"]["total_pnl"] for r in subset])
        print(f"    Entry ${ep:.2f}: avg WR={avg_wr:.1f}%, avg PnL=${avg_pnl:.2f}")

    print("\n  Win Rate by Threshold (averaged across all combos):")
    for tb in THRESHOLD_BPS:
        subset = [r for r in all_results if r["threshold_bp"] == tb]
        avg_wr = np.mean([r["full"]["win_rate"] for r in subset])
        avg_trades = np.mean([r["full"]["total_trades"] for r in subset])
        print(f"    Threshold {tb:>5.1f}bp: avg WR={avg_wr:.1f}%, avg trades={avg_trades:.0f}")

    print("\n  Win Rate by Entry Delay (averaged across all combos):")
    for ds in ENTRY_DELAY_SECS:
        subset = [r for r in all_results if r["entry_delay_sec"] == ds]
        avg_wr = np.mean([r["full"]["win_rate"] for r in subset])
        avg_trades = np.mean([r["full"]["total_trades"] for r in subset])
        print(f"    Delay {ds:>3}s: avg WR={avg_wr:.1f}%, avg trades={avg_trades:.0f}")

    # Walk-forward stability
    profitable_train = sum(1 for r in all_results if r["train"]["total_pnl"] > 0)
    profitable_test = sum(1 for r in all_results if r["test"]["total_pnl"] > 0)
    both_profitable = sum(1 for r in all_results
                          if r["train"]["total_pnl"] > 0 and r["test"]["total_pnl"] > 0)
    print(f"\n  Walk-Forward Stability:")
    print(f"    Combos profitable in-sample:  {profitable_train}/{total_combos}")
    print(f"    Combos profitable out-sample: {profitable_test}/{total_combos}")
    print(f"    Combos profitable in BOTH:    {both_profitable}/{total_combos}")

    # Breakeven WR for different entry prices
    print(f"\n  Breakeven Win Rate by Entry Price:")
    for ep in ENTRY_PRICES:
        # WIN = (1/ep - 1) * cost, LOSS = -cost
        # Breakeven: WR * win_per_trade + (1-WR) * (-cost) = 0
        # WR * (1/ep - 1) * cost = (1-WR) * cost
        # WR * (1/ep - 1) = 1 - WR
        # WR * (1/ep - 1) + WR = 1
        # WR * (1/ep - 1 + 1) = 1
        # WR * (1/ep) = 1
        # WR = ep
        print(f"    Entry ${ep:.2f}: breakeven WR = {ep*100:.1f}%")

    print(f"\n{'='*80}")
    print("  DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
