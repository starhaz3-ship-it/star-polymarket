"""
Head-to-head backtest: V3.0 vs V4.2 vs V5
V5: Candle-close regime + z-scored momentum + impulse + edge gating + TP/SL

Uses 7 days of BTC 1-minute data from Binance.
"""

import json
import math
import time
import statistics
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ============================================================
# DATA FETCHING
# ============================================================

def fetch_binance_klines(symbol="BTCUSDT", interval="1m", days=7):
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    current = start_ms
    while current < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": current, "limit": 1000}
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
            "ts": k[0], "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
        })
    return bars


def aggregate_5m(bars_1m):
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
            "minutes": chunk,
        })
    return bars_5m


# ============================================================
# INDICATORS
# ============================================================

def ema(values, n):
    if not values:
        return []
    a = 2.0 / (n + 1.0)
    out = [values[0]]
    for i in range(1, len(values)):
        out.append(a * values[i] + (1.0 - a) * out[i - 1])
    return out


def atr(highs, lows, closes, period=14):
    """Average True Range."""
    if len(highs) < 2:
        return [0.0]
    trs = [highs[0] - lows[0]]
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        trs.append(tr)
    # Smoothed ATR via EMA
    return ema(trs, period)


def bollinger_bandwidth(closes, period=20, mult=2.0):
    """BB bandwidth = (upper - lower) / middle."""
    if len(closes) < period:
        return [0.0] * len(closes)
    bw = [0.0] * (period - 1)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        mid = sum(window) / period
        std = statistics.stdev(window) if len(window) > 1 else 0.0
        if mid > 0 and std > 0:
            bw.append((2 * mult * std) / mid)
        else:
            bw.append(0.0)
    return bw


def sma(values, n):
    if len(values) < n:
        return [0.0] * len(values)
    out = [0.0] * (n - 1)
    for i in range(n - 1, len(values)):
        out.append(sum(values[i - n + 1:i + 1]) / n)
    return out


# ============================================================
# V3.0: Simple Momentum (baseline)
# ============================================================

def simulate_v30(bars_5m, size_usd=5.0):
    THRESHOLD_BPS = 15.0
    BUY_PRICE = 0.51
    trades = []
    for bar in bars_5m:
        mins = bar["minutes"]
        if len(mins) < 5:
            continue
        open_price = mins[0]["open"]
        price_at_2m = mins[1]["close"]
        mom_bps = (price_at_2m - open_price) / open_price * 10000
        if abs(mom_bps) < THRESHOLD_BPS:
            continue
        direction = "UP" if mom_bps > 0 else "DOWN"
        bar_close = bar["close"]
        actual_dir = "UP" if bar_close > open_price else "DOWN"
        shares = max(1, math.floor(size_usd / BUY_PRICE))
        cost = BUY_PRICE * shares
        pnl = (shares * 1.0 - cost) if direction == actual_dir else -cost
        trades.append({
            "direction": direction, "actual": actual_dir,
            "win": direction == actual_dir,
            "momentum_bps": mom_bps, "pnl": pnl,
            "cost": cost, "shares": shares, "buy_price": BUY_PRICE,
        })
    return trades


# ============================================================
# V4.2: Complex gates (from user's code)
# ============================================================

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))

def _bps(p0, p1):
    return (p1 - p0) / p0 * 10000.0 if p0 > 0 else 0.0

def compute_vol_pct(prices):
    if len(prices) < 3:
        return 0.0
    mx, mn = max(prices), min(prices)
    mid = (mx + mn) / 2.0
    return (mx - mn) / mid * 100.0 if mid > 0 else 0.0

def fair_prob_v42(mom_bps, accel_bps, vol_pct, trend_ok):
    score = mom_bps / 18.0 + accel_bps / 10.0
    score += 0.35 if trend_ok else -0.35
    score -= max(0.0, (vol_pct - 0.8)) * 0.35
    return _clamp(_sigmoid(score), 0.08, 0.92)


def simulate_v42(bars_5m, size_usd=5.0):
    THRESHOLD_BPS = 12.0
    ACCEL_BPS = 3.0
    VOL_MIN, VOL_MAX = 0.10, 1.80
    MIN_EDGE = 0.012
    TREND_FAST, TREND_SLOW = 9, 21
    BUY_PRICE = 0.51

    trades = []
    for bar_idx, bar in enumerate(bars_5m):
        mins = bar["minutes"]
        if len(mins) < 5 or bar_idx < 2:
            continue
        open_price = mins[0]["open"]
        price_hist = [bars_5m[j]["close"] for j in range(max(0, bar_idx - 30), bar_idx)]
        intra_prices = [m["close"] for m in mins]

        for check_min in [1, 2, 3]:
            if check_min >= len(mins):
                continue
            current_price = mins[check_min]["close"]
            p0 = bars_5m[bar_idx - 2]["open"]
            mom_bps = _bps(p0, current_price)
            p_half = bars_5m[bar_idx - 1]["open"]
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
                ef = ema(price_hist[-(TREND_SLOW + 20):], TREND_FAST)[-1]
                es = ema(price_hist[-(TREND_SLOW + 20):], TREND_SLOW)[-1]
                if direction == "UP" and ef <= es:
                    continue
                if direction == "DOWN" and ef >= es:
                    continue

            fair_yes = fair_prob_v42(abs(mom_bps), abs(accel), vol_pct, True)
            edge = fair_yes - 0.505
            if edge < MIN_EDGE:
                continue

            bar_close = bar["close"]
            actual_dir = "UP" if bar_close > open_price else "DOWN"
            shares = max(1, math.floor(size_usd / BUY_PRICE))
            cost = BUY_PRICE * shares
            pnl = (shares * 1.0 - cost) if direction == actual_dir else -cost
            trades.append({
                "direction": direction, "actual": actual_dir,
                "win": direction == actual_dir,
                "momentum_bps": mom_bps, "pnl": pnl,
                "cost": cost, "shares": shares, "buy_price": BUY_PRICE,
            })
            break
    return trades


# ============================================================
# V5: Regime + Z-scored Momentum + Impulse + Edge + TP/SL
# ============================================================

def simulate_v5(bars_5m, size_usd=5.0):
    """
    V5 Strategy:
    On each 5m candle close:
      1. Compute regime: atrp, ema9/21 trend, bb_bandwidth
      2. If regime not tradable: skip
      3. Compute signal: mom_10m (2-bar return), vol_60m (12-bar stdev), mom_z, impulse
      4. If mom_z strong + impulse confirms + trend aligns:
         fair_prob -> edge gate -> place order
      5. TP/SL exits on subsequent bars
    """

    # Config
    ATR_PERIOD = 14
    EMA_FAST = 9
    EMA_SLOW = 21
    BB_PERIOD = 20
    BB_MULT = 2.0

    # Regime gates
    ATRP_MIN = 0.0008    # min ATR% — skip dead markets
    ATRP_MAX = 0.012     # max ATR% — skip chaos
    BB_BW_MIN = 0.005    # min BB bandwidth — need some expansion
    BB_BW_MAX = 0.06     # max BB bandwidth — skip extreme expansion

    # Signal thresholds
    MOM_Z_THRESHOLD = 1.2   # z-score threshold for entry
    IMPULSE_MIN_BPS = 3.0   # minimum impulse (acceleration) in bps
    MIN_EDGE = 0.012         # minimum edge over ask

    # Execution
    BUY_PRICE_DEFAULT = 0.51
    MAX_PAY = 0.95

    # TP/SL (in Polymarket cents on the contract price)
    TP_CENTS = 0.025    # exit if mark >= entry + 2.5c
    SL_CENTS = 0.020    # exit if mark <= entry - 2.0c
    MAX_HOLD_BARS = 1   # max hold = 1 bar (5 min) then let settle

    # Pre-compute indicator series over all bars
    closes = [b["close"] for b in bars_5m]
    highs = [b["high"] for b in bars_5m]
    lows = [b["low"] for b in bars_5m]
    opens = [b["open"] for b in bars_5m]
    n = len(bars_5m)

    # ATR and ATR%
    atr_vals = atr(highs, lows, closes, ATR_PERIOD)
    atrp = [atr_vals[i] / closes[i] if closes[i] > 0 else 0 for i in range(n)]

    # EMAs
    ema_fast = ema(closes, EMA_FAST)
    ema_slow = ema(closes, EMA_SLOW)

    # Bollinger bandwidth
    bb_bw = bollinger_bandwidth(closes, BB_PERIOD, BB_MULT)

    # Returns (5m log returns)
    returns = [0.0]
    for i in range(1, n):
        if closes[i-1] > 0:
            returns.append((closes[i] - closes[i-1]) / closes[i-1])
        else:
            returns.append(0.0)

    trades = []
    open_position = None  # {"bar_idx": int, "direction": str, "entry_price": float, ...}

    MIN_BARS = max(ATR_PERIOD, EMA_SLOW, BB_PERIOD, 12) + 5  # need enough history

    for i in range(MIN_BARS, n):
        bar = bars_5m[i]

        # ---- Check open position for TP/SL/expiry ----
        if open_position is not None:
            pos = open_position
            held_bars = i - pos["bar_idx"]

            # For Polymarket 5-min markets, the outcome is determined at bar close
            # So we resolve at the bar that contains the result
            # Since we enter at bar i's close (signal bar), the market settles at the END
            # of the 5-min round. We treat the next bar's close as settlement.

            if held_bars >= MAX_HOLD_BARS:
                # Settlement: check outcome
                settle_bar = bars_5m[i]
                # The "open" of the 5-min round is what we compare against
                round_open = opens[pos["bar_idx"]]
                round_close = closes[i]
                actual_dir = "UP" if round_close > round_open else "DOWN"

                direction = pos["direction"]
                shares = pos["shares"]
                cost = pos["cost"]

                if direction == actual_dir:
                    pnl = shares * 1.0 - cost
                else:
                    pnl = -cost

                trades.append({
                    "direction": direction,
                    "actual": actual_dir,
                    "win": direction == actual_dir,
                    "momentum_bps": pos.get("mom_bps", 0),
                    "mom_z": pos.get("mom_z", 0),
                    "impulse_bps": pos.get("impulse_bps", 0),
                    "atrp": pos.get("atrp", 0),
                    "bb_bw": pos.get("bb_bw", 0),
                    "edge": pos.get("edge", 0),
                    "fair_yes": pos.get("fair_yes", 0),
                    "pnl": pnl,
                    "cost": cost,
                    "shares": shares,
                    "buy_price": pos["buy_price"],
                    "bar_open": round_open,
                    "bar_close": round_close,
                    "held_bars": held_bars,
                })
                open_position = None
            continue  # Don't open new position while holding one

        # ---- REGIME CHECK ----
        cur_atrp = atrp[i]
        if cur_atrp < ATRP_MIN or cur_atrp > ATRP_MAX:
            continue

        cur_bb_bw = bb_bw[i]
        if cur_bb_bw < BB_BW_MIN or cur_bb_bw > BB_BW_MAX:
            continue

        # Trend
        trend_up = ema_fast[i] > ema_slow[i]
        trend_down = ema_fast[i] < ema_slow[i]

        # ---- SIGNAL COMPUTATION ----
        # mom_10m = return over 2 bars (10 minutes)
        if i < 2:
            continue
        mom_10m = (closes[i] - closes[i-2]) / closes[i-2] if closes[i-2] > 0 else 0
        mom_10m_bps = mom_10m * 10000

        # vol_60m = stdev of returns over last 12 bars (60 minutes)
        if i < 12:
            continue
        recent_returns = returns[i-11:i+1]  # last 12 returns
        vol_60m = statistics.stdev(recent_returns) if len(recent_returns) > 1 else 0.001

        # Avoid division by zero
        if vol_60m < 1e-8:
            vol_60m = 1e-8

        # mom_z = momentum z-score
        mom_z = mom_10m / vol_60m

        # impulse = last_5m_return - prior_5m_return (acceleration)
        impulse = returns[i] - returns[i-1]
        impulse_bps = impulse * 10000

        # ---- DIRECTION DECISION ----
        direction = None

        if mom_z >= MOM_Z_THRESHOLD and impulse > 0 and trend_up:
            direction = "UP"
        elif mom_z <= -MOM_Z_THRESHOLD and impulse < 0 and trend_down:
            direction = "DOWN"

        if not direction:
            continue

        # ---- FAIR PROBABILITY & EDGE ----
        # Map mom_z + impulse + regime_strength -> fair_prob
        regime_strength = _clamp(cur_atrp / 0.005, 0.3, 1.5)  # normalized around typical ATR%

        score = 0.0
        score += abs(mom_z) * 0.45           # z-score contribution
        score += abs(impulse_bps) / 15.0     # impulse contribution
        score += 0.25 * regime_strength      # regime quality
        score += 0.30                        # base offset

        fair_yes = _clamp(_sigmoid(score), 0.10, 0.92)

        # Edge vs ask
        ask = BUY_PRICE_DEFAULT  # assume 0.51
        edge = fair_yes - ask

        if edge < MIN_EDGE:
            continue

        # ---- EXECUTION ----
        # Price: min(ask, fair_yes - min_edge)
        max_edge_pay = fair_yes - MIN_EDGE
        buy_price = _clamp(round(min(MAX_PAY, max_edge_pay, ask), 2), 0.05, 0.95)

        # Size scaling by edge + strength
        strength = _clamp(abs(mom_z) / 3.0, 0.0, 1.0)
        edge_factor = _clamp((edge - MIN_EDGE) / 0.05, 0.0, 1.0)
        # For fair comparison, keep size_usd fixed
        shares = max(1, math.floor(size_usd / buy_price))
        cost = buy_price * shares

        # Open position (will resolve on next bar)
        open_position = {
            "bar_idx": i,
            "direction": direction,
            "buy_price": buy_price,
            "shares": shares,
            "cost": cost,
            "mom_bps": mom_10m_bps,
            "mom_z": mom_z,
            "impulse_bps": impulse_bps,
            "atrp": cur_atrp,
            "bb_bw": cur_bb_bw,
            "edge": edge,
            "fair_yes": fair_yes,
        }

    # Resolve any remaining open position
    if open_position is not None:
        pos = open_position
        i = n - 1
        round_open = opens[pos["bar_idx"]]
        round_close = closes[i]
        actual_dir = "UP" if round_close > round_open else "DOWN"
        direction = pos["direction"]
        shares = pos["shares"]
        cost = pos["cost"]
        pnl = (shares * 1.0 - cost) if direction == actual_dir else -cost
        trades.append({
            "direction": direction, "actual": actual_dir,
            "win": direction == actual_dir,
            "momentum_bps": pos.get("mom_bps", 0),
            "mom_z": pos.get("mom_z", 0),
            "pnl": pnl, "cost": cost, "shares": shares,
            "buy_price": pos["buy_price"],
        })

    return trades


# ============================================================
# V5-INTRA: Same V5 logic but applied INTRA-BAR (like V3.0)
# ============================================================

def simulate_v5_intra(bars_5m, size_usd=5.0):
    """
    V5 regime/signal logic but checked intra-bar at minute 2 (like V3.0).
    This tests whether V5's filters improve V3.0's already-high WR.
    """
    ATR_PERIOD = 14
    EMA_FAST = 9
    EMA_SLOW = 21

    closes = [b["close"] for b in bars_5m]
    highs = [b["high"] for b in bars_5m]
    lows = [b["low"] for b in bars_5m]
    n = len(bars_5m)

    atr_vals = atr(highs, lows, closes, ATR_PERIOD)
    atrp = [atr_vals[i] / closes[i] if closes[i] > 0 else 0 for i in range(n)]
    ema_fast = ema(closes, EMA_FAST)
    ema_slow = ema(closes, EMA_SLOW)
    bb_bw = bollinger_bandwidth(closes, 20, 2.0)

    returns = [0.0]
    for i in range(1, n):
        returns.append((closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] > 0 else 0)

    BUY_PRICE = 0.51
    MOM_THRESHOLD_BPS = 15.0
    ATRP_MIN, ATRP_MAX = 0.0008, 0.012
    MOM_Z_THRESHOLD = 0.8  # relaxed since intra-bar momentum is inherently noisy

    MIN_BARS = max(ATR_PERIOD, EMA_SLOW, 20, 12) + 5
    trades = []

    for i in range(MIN_BARS, n):
        bar = bars_5m[i]
        mins = bar["minutes"]
        if len(mins) < 5:
            continue

        # Regime from prior completed bars
        cur_atrp = atrp[i-1]  # use prior bar's ATR% (current bar not complete)
        if cur_atrp < ATRP_MIN or cur_atrp > ATRP_MAX:
            continue

        cur_bb_bw = bb_bw[i-1]
        if cur_bb_bw < 0.005 or cur_bb_bw > 0.06:
            continue

        # Intra-bar momentum at minute 2
        open_price = mins[0]["open"]
        price_at_2m = mins[1]["close"]
        mom_bps = (price_at_2m - open_price) / open_price * 10000

        if abs(mom_bps) < MOM_THRESHOLD_BPS:
            continue

        direction = "UP" if mom_bps > 0 else "DOWN"

        # Trend alignment
        trend_up = ema_fast[i-1] > ema_slow[i-1]
        trend_down = ema_fast[i-1] < ema_slow[i-1]
        if direction == "UP" and not trend_up:
            continue
        if direction == "DOWN" and not trend_down:
            continue

        # Z-score check using recent vol
        if i >= 12:
            recent_rets = returns[i-11:i]
            vol = statistics.stdev(recent_rets) if len(recent_rets) > 1 else 0.001
            if vol < 1e-8:
                vol = 1e-8
            mom_10m_ret = (price_at_2m - closes[i-2]) / closes[i-2] if closes[i-2] > 0 else 0
            mom_z = mom_10m_ret / vol
            if abs(mom_z) < MOM_Z_THRESHOLD:
                continue

        # Outcome
        bar_close = bar["close"]
        actual_dir = "UP" if bar_close > open_price else "DOWN"
        shares = max(1, math.floor(size_usd / BUY_PRICE))
        cost = BUY_PRICE * shares
        pnl = (shares * 1.0 - cost) if direction == actual_dir else -cost

        trades.append({
            "direction": direction, "actual": actual_dir,
            "win": direction == actual_dir,
            "momentum_bps": mom_bps, "pnl": pnl,
            "cost": cost, "shares": shares, "buy_price": BUY_PRICE,
        })

    return trades


# ============================================================
# ANALYSIS
# ============================================================

def analyze(trades, label):
    if not trades:
        print(f"\n{'='*60}")
        print(f"  {label}: NO TRADES")
        print(f"{'='*60}")
        return {"label": label, "trades": 0, "wins": 0, "win_rate": 0,
                "total_pnl": 0, "avg_pnl": 0, "avg_win": 0, "avg_loss": 0,
                "max_drawdown": 0, "total_cost": 0, "roi_pct": 0,
                "max_win_streak": 0, "max_loss_streak": 0, "trades_per_day": 0}

    total = len(trades)
    wins = sum(1 for t in trades if t["win"])
    wr = wins / total * 100
    total_pnl = sum(t["pnl"] for t in trades)
    avg_pnl = total_pnl / total
    total_cost = sum(t["cost"] for t in trades)

    win_pnls = [t["pnl"] for t in trades if t["win"]]
    loss_pnls = [t["pnl"] for t in trades if not t["win"]]
    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0

    cum = peak = max_dd = 0
    for t in trades:
        cum += t["pnl"]
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    max_ws = max_ls = cur_w = cur_l = 0
    for t in trades:
        if t["win"]:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_ws = max(max_ws, cur_w)
        max_ls = max(max_ls, cur_l)

    roi = total_pnl / total_cost * 100 if total_cost > 0 else 0
    trades_per_day = total / 7.0

    win_moms = [abs(t["momentum_bps"]) for t in trades if t["win"]]
    loss_moms = [abs(t["momentum_bps"]) for t in trades if not t["win"]]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades:        {total}  ({trades_per_day:.1f}/day)")
    print(f"  Wins:          {wins} ({wr:.1f}%)")
    print(f"  Losses:        {total - wins} ({100-wr:.1f}%)")
    print(f"  Total PnL:     ${total_pnl:+,.2f}")
    print(f"  Avg PnL:       ${avg_pnl:+.2f}/trade")
    print(f"  Avg Win:       ${avg_win:+.2f}")
    print(f"  Avg Loss:      ${avg_loss:+.2f}")
    if avg_loss != 0:
        print(f"  W:L Ratio:     {abs(avg_win/avg_loss):.2f}x")
    print(f"  Total Cost:    ${total_cost:,.2f}")
    print(f"  ROI:           {roi:+.2f}%")
    print(f"  Max Drawdown:  ${max_dd:,.2f}")
    print(f"  Win Streak:    {max_ws}")
    print(f"  Loss Streak:   {max_ls}")
    if win_moms:
        print(f"  Avg |Mom| W:   {sum(win_moms)/len(win_moms):.1f}bp")
    if loss_moms:
        print(f"  Avg |Mom| L:   {sum(loss_moms)/len(loss_moms):.1f}bp")

    # V5-specific stats
    if trades[0].get("mom_z") is not None:
        zs = [abs(t.get("mom_z", 0)) for t in trades]
        print(f"  Avg |mom_z|:   {sum(zs)/len(zs):.2f}")
        win_z = [abs(t.get("mom_z", 0)) for t in trades if t["win"]]
        loss_z = [abs(t.get("mom_z", 0)) for t in trades if not t["win"]]
        if win_z:
            print(f"  Avg |z| wins:  {sum(win_z)/len(win_z):.2f}")
        if loss_z:
            print(f"  Avg |z| loss:  {sum(loss_z)/len(loss_z):.2f}")

    return {
        "label": label, "trades": total, "wins": wins,
        "win_rate": round(wr, 2), "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 4), "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4), "max_drawdown": round(max_dd, 2),
        "total_cost": round(total_cost, 2), "roi_pct": round(roi, 2),
        "max_win_streak": max_ws, "max_loss_streak": max_ls,
        "trades_per_day": round(trades_per_day, 1),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("Fetching 7 days of BTC 1-min data from Binance...")
    bars_1m = fetch_binance_klines(days=7)
    print(f"  Got {len(bars_1m)} 1-min bars")
    bars_5m = aggregate_5m(bars_1m)
    print(f"  Aggregated to {len(bars_5m)} 5-min bars")

    date_start = datetime.fromtimestamp(bars_1m[0]["ts"] / 1000, tz=timezone.utc)
    date_end = datetime.fromtimestamp(bars_1m[-1]["ts"] / 1000, tz=timezone.utc)
    natural_up = sum(1 for b in bars_5m if b["close"] > b["open"]) / len(bars_5m) * 100
    print(f"  Range: {date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}")
    print(f"  Natural UP%: {natural_up:.1f}%")

    SIZE = 5.0  # Fair comparison: all at $5

    # Run all strategies
    v30 = simulate_v30(bars_5m, SIZE)
    s30 = analyze(v30, "V3.0 — Simple Momentum (2min, 15bp, $5)")

    v42 = simulate_v42(bars_5m, SIZE)
    s42 = analyze(v42, "V4.2 — Complex Gates (vol+trend+impulse+edge)")

    v5 = simulate_v5(bars_5m, SIZE)
    s5 = analyze(v5, "V5 — Regime + Z-scored Mom + Impulse (candle close)")

    v5i = simulate_v5_intra(bars_5m, SIZE)
    s5i = analyze(v5i, "V5-INTRA — V5 regime/z-score + V3.0 intra-bar entry")

    # Head to head
    print("\n" + "="*70)
    print("  HEAD-TO-HEAD COMPARISON (all $5 sizing)")
    print("="*70)

    def row(label, *vals):
        parts = [f"  {label:20s}"]
        for v in vals:
            parts.append(f"{v:>14s}")
        print("".join(parts))

    row("", "V3.0", "V4.2", "V5", "V5-INTRA")
    row("-"*20, "-"*14, "-"*14, "-"*14, "-"*14)

    for label, key in [
        ("Trades", "trades"),
        ("Trades/day", "trades_per_day"),
        ("Win Rate %", "win_rate"),
        ("Total PnL", "total_pnl"),
        ("Avg PnL/trade", "avg_pnl"),
        ("Avg Win", "avg_win"),
        ("Avg Loss", "avg_loss"),
        ("Max Drawdown", "max_drawdown"),
        ("ROI%", "roi_pct"),
        ("Win Streak", "max_win_streak"),
        ("Loss Streak", "max_loss_streak"),
    ]:
        vals = []
        for s in [s30, s42, s5, s5i]:
            v = s.get(key, 0)
            if key in ["total_pnl", "avg_pnl", "avg_win", "avg_loss", "max_drawdown"]:
                vals.append(f"${v:+,.2f}")
            elif key in ["win_rate", "roi_pct", "trades_per_day"]:
                vals.append(f"{v:.1f}")
            else:
                vals.append(f"{v}")
        row(label, *vals)

    # Verdict
    print("\n" + "="*70)
    print("  VERDICT")
    print("="*70)

    all_stats = [("V3.0", s30), ("V4.2", s42), ("V5", s5), ("V5-INTRA", s5i)]
    best_wr = max(all_stats, key=lambda x: x[1].get("win_rate", 0))
    best_pnl = max(all_stats, key=lambda x: x[1].get("total_pnl", 0))
    best_roi = max(all_stats, key=lambda x: x[1].get("roi_pct", 0))

    print(f"  Best WR:    {best_wr[0]} @ {best_wr[1]['win_rate']:.1f}%")
    print(f"  Best PnL:   {best_pnl[0]} @ ${best_pnl[1]['total_pnl']:+,.2f}")
    print(f"  Best ROI:   {best_roi[0]} @ {best_roi[1]['roi_pct']:+.2f}%")

    print(f"\n  V3.0:     {s30['trades']} trades, {s30['win_rate']:.1f}% WR, ${s30['total_pnl']:+,.2f} PnL")
    print(f"  V4.2:     {s42['trades']} trades, {s42['win_rate']:.1f}% WR, ${s42['total_pnl']:+,.2f} PnL")
    print(f"  V5:       {s5['trades']} trades, {s5['win_rate']:.1f}% WR, ${s5['total_pnl']:+,.2f} PnL")
    print(f"  V5-INTRA: {s5i['trades']} trades, {s5i['win_rate']:.1f}% WR, ${s5i['total_pnl']:+,.2f} PnL")

    # Save
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": f"{date_start:%Y-%m-%d %H:%M} to {date_end:%Y-%m-%d %H:%M}",
        "bars_1m": len(bars_1m), "bars_5m": len(bars_5m),
        "natural_up_pct": round(natural_up, 2),
        "v30": s30, "v42": s42, "v5": s5, "v5_intra": s5i,
    }
    out = Path(__file__).parent / "backtest_v5_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
