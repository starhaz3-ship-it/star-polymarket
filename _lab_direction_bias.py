"""
BTC Directional Bias Analysis for Polymarket Betting Optimization
Analyzes 5M and 15M candles to determine UP vs DOWN bias.
"""

import requests
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict

SEPARATOR = "=" * 80
SUB_SEP = "-" * 60

def fetch_candles(symbol, interval, limit):
    """Fetch candles from Binance."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    candles = []
    for k in data:
        candles.append({
            "open_time": k[0],
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "close_time": k[6],
            "dt": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
        })
    return candles


def classify(candle):
    """UP if close >= open, else DOWN."""
    return "UP" if candle["close"] >= candle["open"] else "DOWN"


def pct(count, total):
    return (count / total * 100) if total > 0 else 0.0


def filter_candles_by_hours(candles, hours):
    """Filter candles from the last N hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return [c for c in candles if c["dt"] >= cutoff]


def print_direction_stats(candles, label):
    """Print UP/DOWN stats for a set of candles."""
    if not candles:
        print(f"  {label}: No data")
        return
    up = sum(1 for c in candles if classify(c) == "UP")
    dn = len(candles) - up
    up_pct = pct(up, len(candles))
    dn_pct = pct(dn, len(candles))
    bias = "UP" if up_pct > dn_pct else ("DOWN" if dn_pct > up_pct else "NEUTRAL")
    strength = abs(up_pct - 50)
    print(f"  {label:>12s}: {len(candles):4d} candles | UP {up:4d} ({up_pct:5.1f}%) | DOWN {dn:4d} ({dn_pct:5.1f}%) | Bias: {bias} ({strength:.1f}pp)")


def streak_analysis(candles):
    """Analyze streaks of consecutive UP or DOWN candles."""
    if not candles:
        return
    directions = [classify(c) for c in candles]

    # Current streak
    current_dir = directions[-1]
    current_streak = 0
    for d in reversed(directions):
        if d == current_dir:
            current_streak += 1
        else:
            break

    # All streaks
    streaks = []
    streak_dir = directions[0]
    streak_len = 1
    for d in directions[1:]:
        if d == streak_dir:
            streak_len += 1
        else:
            streaks.append((streak_dir, streak_len))
            streak_dir = d
            streak_len = 1
    streaks.append((streak_dir, streak_len))

    up_streaks = [s[1] for s in streaks if s[0] == "UP"]
    dn_streaks = [s[1] for s in streaks if s[0] == "DOWN"]
    max_up = max(up_streaks) if up_streaks else 0
    max_dn = max(dn_streaks) if dn_streaks else 0
    avg_up = sum(up_streaks) / len(up_streaks) if up_streaks else 0
    avg_dn = sum(dn_streaks) / len(dn_streaks) if dn_streaks else 0

    print(f"  Current streak: {current_streak} {current_dir}")
    print(f"  Max UP streak:  {max_up} candles | Avg UP streak:  {avg_up:.1f}")
    print(f"  Max DOWN streak:{max_dn} candles | Avg DOWN streak:{avg_dn:.1f}")
    print(f"  Total streaks:  {len(streaks)} (UP: {len(up_streaks)}, DOWN: {len(dn_streaks)})")


def magnitude_analysis(candles, label):
    """Analyze magnitude of UP vs DOWN moves."""
    up_moves = []
    dn_moves = []
    for c in candles:
        move_usd = c["close"] - c["open"]
        move_bps = (move_usd / c["open"]) * 10000
        if classify(c) == "UP":
            up_moves.append((move_usd, move_bps))
        else:
            dn_moves.append((abs(move_usd), abs(move_bps)))

    if up_moves:
        avg_up_usd = sum(m[0] for m in up_moves) / len(up_moves)
        avg_up_bps = sum(m[1] for m in up_moves) / len(up_moves)
        med_up_usd = sorted(m[0] for m in up_moves)[len(up_moves)//2]
    else:
        avg_up_usd = avg_up_bps = med_up_usd = 0

    if dn_moves:
        avg_dn_usd = sum(m[0] for m in dn_moves) / len(dn_moves)
        avg_dn_bps = sum(m[1] for m in dn_moves) / len(dn_moves)
        med_dn_usd = sorted(m[0] for m in dn_moves)[len(dn_moves)//2]
    else:
        avg_dn_usd = avg_dn_bps = med_dn_usd = 0

    print(f"\n  {label}")
    print(f"  UP candles   ({len(up_moves):4d}): avg ${avg_up_usd:8.2f} ({avg_up_bps:5.2f} bps) | median ${med_up_usd:8.2f}")
    print(f"  DOWN candles ({len(dn_moves):4d}): avg ${avg_dn_usd:8.2f} ({avg_dn_bps:5.2f} bps) | median ${med_dn_usd:8.2f}")

    if avg_up_usd > 0 and avg_dn_usd > 0:
        ratio = avg_up_usd / avg_dn_usd
        if ratio > 1.05:
            asym = f"UP moves are {ratio:.2f}x larger -- BULLISH asymmetry"
        elif ratio < 0.95:
            asym = f"DOWN moves are {1/ratio:.2f}x larger -- BEARISH asymmetry"
        else:
            asym = "Roughly symmetric moves"
        print(f"  Asymmetry: {asym}")

    return up_moves, dn_moves


def simulate_betting(candles, direction, bet_size=3.0):
    """
    Simulate betting $bet_size on a direction every candle at $0.50 entry.
    If direction matches candle, win $bet_size (pay $0.50, receive $1.00, profit = $bet_size * 1.0).
    Actually: bet_size shares at $0.50 each = cost $bet_size * 0.50.
    If win: receive bet_size * $1.00 = $bet_size. Profit = $bet_size - $bet_size*0.50 = $bet_size*0.50
    If lose: lose the cost = $bet_size * 0.50
    Wait -- this is Polymarket logic. $3 bet at $0.50 = 6 shares.
    Win: 6 * $1 = $6, profit = $3. Lose: $0, loss = -$3.
    So each bet: +$3 if correct, -$3 if wrong. Net = (wins - losses) * $3.
    """
    wins = sum(1 for c in candles if classify(c) == direction)
    losses = len(candles) - wins
    net = (wins - losses) * bet_size
    wr = pct(wins, len(candles))
    return wins, losses, net, wr


def time_of_day_analysis(candles):
    """Analyze direction bias by UTC hour."""
    hourly = defaultdict(lambda: {"up": 0, "dn": 0, "up_moves": [], "dn_moves": []})
    for c in candles:
        hour = c["dt"].hour
        if classify(c) == "UP":
            hourly[hour]["up"] += 1
            hourly[hour]["up_moves"].append(c["close"] - c["open"])
        else:
            hourly[hour]["dn"] += 1
            hourly[hour]["dn_moves"].append(abs(c["close"] - c["open"]))

    print(f"\n  {'Hour':>4s} | {'Total':>5s} | {'UP':>4s}  {'UP%':>5s} | {'DN':>4s}  {'DN%':>5s} | {'Bias':>7s} | {'AvgUP$':>8s} | {'AvgDN$':>8s}")
    print(f"  {'-'*4}-+-{'-'*5}-+-{'-'*4}--{'-'*5}-+-{'-'*4}--{'-'*5}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")

    strong_hours = []
    for hour in range(24):
        h = hourly[hour]
        total = h["up"] + h["dn"]
        if total == 0:
            continue
        up_p = pct(h["up"], total)
        dn_p = pct(h["dn"], total)
        avg_up = sum(h["up_moves"]) / len(h["up_moves"]) if h["up_moves"] else 0
        avg_dn = sum(h["dn_moves"]) / len(h["dn_moves"]) if h["dn_moves"] else 0

        if up_p >= 60:
            bias = "UP ***"
            strong_hours.append((hour, "UP", up_p))
        elif dn_p >= 60:
            bias = "DN ***"
            strong_hours.append((hour, "DOWN", dn_p))
        else:
            bias = "neutral"

        print(f"  {hour:4d} | {total:5d} | {h['up']:4d}  {up_p:5.1f}% | {h['dn']:4d}  {dn_p:5.1f}% | {bias:>7s} | ${avg_up:7.2f} | ${avg_dn:7.2f}")

    return strong_hours


def regime_detection(candles_15m):
    """
    Simple regime detection using 4-hour rolling UP%.
    4 hours = 16 candles of 15M.
    """
    window = 16
    if len(candles_15m) < window:
        print("  Not enough data for regime detection")
        return None, None, None

    regimes = []
    for i in range(window, len(candles_15m) + 1):
        window_candles = candles_15m[i - window:i]
        up_count = sum(1 for c in window_candles if classify(c) == "UP")
        up_pct_val = pct(up_count, window)

        if up_pct_val > 60:
            regime = "BULLISH"
        elif up_pct_val < 40:
            regime = "BEARISH"
        else:
            regime = "NEUTRAL"

        regimes.append({
            "dt": candles_15m[i - 1]["dt"],
            "up_pct": up_pct_val,
            "regime": regime,
            "candle_idx": i - 1,
        })

    # Current regime
    current = regimes[-1]
    # How long has current regime been active?
    duration = 1
    for r in reversed(regimes[:-1]):
        if r["regime"] == current["regime"]:
            duration += 1
        else:
            break

    duration_hours = duration * 0.25  # each entry = 15min

    print(f"\n  Current regime: {current['regime']} (rolling 4h UP% = {current['up_pct']:.1f}%)")
    print(f"  Active for: {duration} periods ({duration_hours:.1f} hours)")
    print(f"  As of: {current['dt'].strftime('%Y-%m-%d %H:%M UTC')}")

    # Regime distribution over all data
    regime_counts = defaultdict(int)
    for r in regimes:
        regime_counts[r["regime"]] += 1
    total = len(regimes)
    print(f"\n  Regime distribution (14d):")
    for reg in ["BULLISH", "NEUTRAL", "BEARISH"]:
        cnt = regime_counts[reg]
        print(f"    {reg:8s}: {cnt:4d} periods ({pct(cnt, total):5.1f}%)")

    # When in BULLISH regime, what's the WR of next candle being UP?
    # When in BEARISH regime, what's the WR of next candle being DOWN?
    bull_next_up = 0
    bull_total = 0
    bear_next_dn = 0
    bear_total = 0

    for i, r in enumerate(regimes[:-1]):
        next_candle_idx = r["candle_idx"] + 1
        if next_candle_idx < len(candles_15m):
            next_dir = classify(candles_15m[next_candle_idx])
            if r["regime"] == "BULLISH":
                bull_total += 1
                if next_dir == "UP":
                    bull_next_up += 1
            elif r["regime"] == "BEARISH":
                bear_total += 1
                if next_dir == "DOWN":
                    bear_next_dn += 1

    print(f"\n  Regime-following accuracy:")
    if bull_total > 0:
        bull_wr = pct(bull_next_up, bull_total)
        print(f"    BULLISH regime -> next candle UP:   {bull_next_up}/{bull_total} = {bull_wr:.1f}%")
    else:
        print(f"    BULLISH regime: No data")
        bull_wr = 50.0

    if bear_total > 0:
        bear_wr = pct(bear_next_dn, bear_total)
        print(f"    BEARISH regime -> next candle DOWN: {bear_next_dn}/{bear_total} = {bear_wr:.1f}%")
    else:
        print(f"    BEARISH regime: No data")
        bear_wr = 50.0

    # Recent regime transitions (last 48h = 192 candles)
    print(f"\n  Recent regime transitions (last 48h):")
    recent = regimes[-192:] if len(regimes) >= 192 else regimes
    prev_regime = recent[0]["regime"]
    transitions = 0
    for r in recent[1:]:
        if r["regime"] != prev_regime:
            transitions += 1
            print(f"    {r['dt'].strftime('%m-%d %H:%M')} UTC: {prev_regime} -> {r['regime']} (UP%={r['up_pct']:.0f}%)")
            prev_regime = r["regime"]
    if transitions == 0:
        print(f"    No transitions -- stable {current['regime']} regime")

    return current["regime"], bull_wr if bull_total > 0 else None, bear_wr if bear_total > 0 else None


def main():
    print(SEPARATOR)
    print("  BTC DIRECTIONAL BIAS ANALYSIS FOR POLYMARKET")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(SEPARATOR)

    # Fetch data
    print("\n[Fetching data from Binance...]")
    candles_15m = fetch_candles("BTCUSDT", "15m", 1000)
    time.sleep(0.2)
    candles_5m = fetch_candles("BTCUSDT", "5m", 1000)

    span_15m = (candles_15m[-1]["dt"] - candles_15m[0]["dt"]).total_seconds() / 86400
    span_5m = (candles_5m[-1]["dt"] - candles_5m[0]["dt"]).total_seconds() / 86400
    btc_price = candles_15m[-1]["close"]
    print(f"  15M candles: {len(candles_15m)} ({span_15m:.1f} days)")
    print(f"  5M candles:  {len(candles_5m)} ({span_5m:.1f} days)")
    print(f"  Current BTC price: ${btc_price:,.2f}")

    # =========================================================================
    # PART 1: Rolling Direction Stats
    # =========================================================================
    print(f"\n{SEPARATOR}")
    print("  PART 1: ROLLING DIRECTION STATS")
    print(SEPARATOR)

    print(f"\n  15-Minute Candles:")
    print_direction_stats(filter_candles_by_hours(candles_15m, 24), "Last 24h")
    print_direction_stats(filter_candles_by_hours(candles_15m, 72), "Last 3d")
    print_direction_stats(filter_candles_by_hours(candles_15m, 168), "Last 7d")
    print_direction_stats(candles_15m, f"All ({span_15m:.0f}d)")

    print(f"\n  5-Minute Candles:")
    print_direction_stats(filter_candles_by_hours(candles_5m, 24), "Last 24h")
    print_direction_stats(filter_candles_by_hours(candles_5m, 72), "Last 3d")
    print_direction_stats(candles_5m, f"All ({span_5m:.0f}d)")

    print(f"\n  Streak Analysis (15M):")
    streak_analysis(candles_15m)

    print(f"\n  Streak Analysis (5M):")
    streak_analysis(candles_5m)

    # =========================================================================
    # PART 2: Magnitude Analysis
    # =========================================================================
    print(f"\n{SEPARATOR}")
    print("  PART 2: MAGNITUDE ANALYSIS")
    print(SEPARATOR)

    magnitude_analysis(candles_15m, "15-Minute Candles (full dataset)")
    magnitude_analysis(filter_candles_by_hours(candles_15m, 24), "15-Minute Candles (last 24h)")
    magnitude_analysis(candles_5m, "5-Minute Candles (full dataset)")
    magnitude_analysis(filter_candles_by_hours(candles_5m, 24), "5-Minute Candles (last 24h)")

    print(f"\n{SUB_SEP}")
    print("  Simulated $3 Polymarket Betting (at $0.50 entry, 15M candles)")
    print(SUB_SEP)

    for period_label, hours in [("Last 24h", 24), ("Last 3d", 72), ("Last 7d", 168), ("Full dataset", None)]:
        subset = filter_candles_by_hours(candles_15m, hours) if hours else candles_15m
        up_w, up_l, up_net, up_wr = simulate_betting(subset, "UP", 3.0)
        dn_w, dn_l, dn_net, dn_wr = simulate_betting(subset, "DOWN", 3.0)
        better = "UP" if up_net > dn_net else "DOWN"
        print(f"\n  {period_label} ({len(subset)} candles):")
        print(f"    Always UP:   {up_w}W/{up_l}L ({up_wr:.1f}% WR) -> Net ${up_net:+.2f}")
        print(f"    Always DOWN: {dn_w}W/{dn_l}L ({dn_wr:.1f}% WR) -> Net ${dn_net:+.2f}")
        print(f"    Better: {better} by ${abs(up_net - dn_net):.2f}")

    # =========================================================================
    # PART 3: Time-of-Day Direction Bias
    # =========================================================================
    print(f"\n{SEPARATOR}")
    print("  PART 3: TIME-OF-DAY DIRECTION BIAS (15M, 14 days)")
    print(SEPARATOR)

    strong_hours = time_of_day_analysis(candles_15m)

    if strong_hours:
        print(f"\n  Strong bias hours (>60% one direction):")
        for hour, direction, strength in strong_hours:
            print(f"    {hour:02d}:00 UTC -> {direction} ({strength:.1f}%)")
    else:
        print(f"\n  No hours with >60% directional bias found.")

    # =========================================================================
    # PART 4: Regime Detection
    # =========================================================================
    print(f"\n{SEPARATOR}")
    print("  PART 4: REGIME DETECTION (4-hour rolling window)")
    print(SEPARATOR)

    current_regime, bull_wr, bear_wr = regime_detection(candles_15m)

    # =========================================================================
    # PART 5: Recommendations
    # =========================================================================
    print(f"\n{SEPARATOR}")
    print("  PART 5: RECOMMENDATIONS")
    print(SEPARATOR)

    # Gather signals
    last_24h_15m = filter_candles_by_hours(candles_15m, 24)
    last_3d_15m = filter_candles_by_hours(candles_15m, 72)

    up_24h = sum(1 for c in last_24h_15m if classify(c) == "UP")
    up_pct_24h = pct(up_24h, len(last_24h_15m)) if last_24h_15m else 50

    up_3d = sum(1 for c in last_3d_15m if classify(c) == "UP")
    up_pct_3d = pct(up_3d, len(last_3d_15m)) if last_3d_15m else 50

    # Score: weighted average of signals
    signals = []

    # 1. 24h direction
    if up_pct_24h > 55:
        signals.append(("24h direction bias", "UP", up_pct_24h - 50))
    elif up_pct_24h < 45:
        signals.append(("24h direction bias", "DOWN", 50 - up_pct_24h))
    else:
        signals.append(("24h direction bias", "NEUTRAL", 0))

    # 2. 3d direction
    if up_pct_3d > 53:
        signals.append(("3d direction bias", "UP", up_pct_3d - 50))
    elif up_pct_3d < 47:
        signals.append(("3d direction bias", "DOWN", 50 - up_pct_3d))
    else:
        signals.append(("3d direction bias", "NEUTRAL", 0))

    # 3. Current regime
    if current_regime:
        if current_regime == "BULLISH":
            signals.append(("4h regime", "UP", 10))
        elif current_regime == "BEARISH":
            signals.append(("4h regime", "DOWN", 10))
        else:
            signals.append(("4h regime", "NEUTRAL", 0))

    # 4. Regime predictive power
    if bull_wr and bull_wr > 55:
        signals.append(("Regime follow-through (BULL)", "UP", bull_wr - 50))
    if bear_wr and bear_wr > 55:
        signals.append(("Regime follow-through (BEAR)", "DOWN", bear_wr - 50))

    # 5. Magnitude asymmetry (last 24h)
    up_mag_24h = [c["close"] - c["open"] for c in last_24h_15m if classify(c) == "UP"]
    dn_mag_24h = [abs(c["close"] - c["open"]) for c in last_24h_15m if classify(c) == "DOWN"]
    avg_up_mag = sum(up_mag_24h) / len(up_mag_24h) if up_mag_24h else 0
    avg_dn_mag = sum(dn_mag_24h) / len(dn_mag_24h) if dn_mag_24h else 0
    if avg_up_mag > 0 and avg_dn_mag > 0:
        mag_ratio = avg_up_mag / avg_dn_mag
        if mag_ratio > 1.1:
            signals.append(("Magnitude asymmetry (24h)", "UP", min((mag_ratio - 1) * 20, 10)))
        elif mag_ratio < 0.9:
            signals.append(("Magnitude asymmetry (24h)", "DOWN", min((1/mag_ratio - 1) * 20, 10)))

    print(f"\n  Signal Summary:")
    up_score = 0
    dn_score = 0
    for name, direction, strength in signals:
        symbol = "^" if direction == "UP" else ("v" if direction == "DOWN" else "-")
        print(f"    [{symbol}] {name}: {direction} (strength: {strength:.1f})")
        if direction == "UP":
            up_score += strength
        elif direction == "DOWN":
            dn_score += strength

    net_score = up_score - dn_score
    if net_score > 5:
        overall = "UP"
        confidence = "HIGH" if net_score > 15 else ("MODERATE" if net_score > 8 else "LOW")
    elif net_score < -5:
        overall = "DOWN"
        confidence = "HIGH" if net_score < -15 else ("MODERATE" if net_score < -8 else "LOW")
    else:
        overall = "NEUTRAL"
        confidence = "N/A"

    print(f"\n  {'*' * 60}")
    print(f"  OVERALL RECOMMENDATION: {overall} (confidence: {confidence})")
    print(f"  UP score: {up_score:.1f} | DOWN score: {dn_score:.1f} | Net: {net_score:+.1f}")
    print(f"  {'*' * 60}")

    # Specific hour recommendations
    if strong_hours:
        print(f"\n  Hour-specific recommendations:")
        for hour, direction, strength in strong_hours:
            # Convert to ET for reference
            et_hour = (hour - 5) % 24
            print(f"    {hour:02d}:00 UTC ({et_hour:02d}:00 ET): Favor {direction} ({strength:.1f}%)")

    # Should regime override signals?
    print(f"\n  Regime override assessment:")
    if current_regime in ("BULLISH", "BEARISH"):
        regime_dir = "UP" if current_regime == "BULLISH" else "DOWN"
        regime_wr = bull_wr if current_regime == "BULLISH" else bear_wr
        if regime_wr and regime_wr > 55:
            print(f"    YES - {current_regime} regime has {regime_wr:.1f}% follow-through WR.")
            print(f"    When regime says {regime_dir}, next candle IS {regime_dir} {regime_wr:.1f}% of the time.")
            if regime_wr > 60:
                print(f"    STRONG recommendation: Override signal-based entries toward {regime_dir}.")
            else:
                print(f"    MODERATE recommendation: Weight {regime_dir} more heavily, but don't fully override signals.")
        else:
            print(f"    NO - {current_regime} regime follow-through is weak ({regime_wr:.1f}% if available).")
            print(f"    Stick with signal-based entries.")
    else:
        print(f"    NO - NEUTRAL regime. No directional override. Use signal-based entries.")

    # Practical Polymarket implications
    print(f"\n  Polymarket implications:")
    print(f"    - Momentum 15M V1.6 streak reversal should be aware of regime.")
    if overall != "NEUTRAL":
        print(f"    - Consider weighting {overall} bets slightly higher in {overall.lower()} regime.")
        print(f"    - If signal conflicts with {overall} regime, reduce bet size or skip.")
    else:
        print(f"    - No strong directional bias detected. Continue with balanced signal-based entries.")

    # Net candle movement (where is BTC actually going?)
    print(f"\n  Net BTC price movement:")
    for label, hours in [("24h", 24), ("3d", 72), ("7d", 168)]:
        subset = filter_candles_by_hours(candles_15m, hours)
        if subset:
            net_move = subset[-1]["close"] - subset[0]["open"]
            net_bps = (net_move / subset[0]["open"]) * 10000
            direction = "UP" if net_move > 0 else "DOWN"
            print(f"    {label}: ${net_move:+,.2f} ({net_bps:+.1f} bps) -> {direction}")

    print(f"\n{SEPARATOR}")
    print("  END OF ANALYSIS")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
