#!/usr/bin/env python
"""
Consensus Signal Analysis: Do multiple paper/live bots agreeing improve WR?

Loads trades from:
  - fifteen_min_strategies_results.json  (paper 15M strategies)
  - momentum_15m_results.json + archived (live/paper momentum 15M)
  - sniper_5m_results.json + archived    (paper sniper 5M)
  - sniper_5m_live_results.json + archived (live sniper 5M)

Groups trades by market close time, analyzes consensus vs solo WR.
"""

import json
import re
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict, Counter
from itertools import combinations

BASE = r"C:\Users\Star\.local\bin\star-polymarket"

# ── Helpers ──────────────────────────────────────────────────────────────

def load_json(path):
    full = os.path.join(BASE, path)
    if not os.path.exists(full):
        return []
    with open(full, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("resolved", [])


ET = timezone(timedelta(hours=-5))   # Eastern Time (EST, approximate)
UTC = timezone.utc

TITLE_RE = re.compile(
    r"(?:Bitcoin|Ethereum|ETH|BTC)\s+Up or Down\s*-\s*"
    r"(\w+)\s+(\d{1,2}),?\s+"
    r"(\d{1,2}:\d{2}(?:AM|PM))\s*-\s*(\d{1,2}:\d{2}(?:AM|PM))\s*ET",
    re.IGNORECASE,
)

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_close_time(title):
    """Extract market close time as UTC datetime from title/question string."""
    m = TITLE_RE.search(title or "")
    if not m:
        return None
    month_str, day_str, _start_str, end_str = m.groups()
    month = MONTH_MAP.get(month_str.lower())
    if not month:
        return None
    day = int(day_str)
    # Parse end time (close time)
    end_time = datetime.strptime(end_str.upper(), "%I:%M%p")
    # Handle midnight crossover: "11:55PM-12:00AM" means next day midnight
    close_et = datetime(2026, month, day, end_time.hour, end_time.minute, tzinfo=ET)
    # If end time is 12:00AM and it's a "-12:00AM" close, it means midnight = start of next day
    if end_time.hour == 0 and end_time.minute == 0:
        close_et = datetime(2026, month, day, 0, 0, tzinfo=ET) + timedelta(days=1)
    return close_et.astimezone(UTC)


def parse_entry_time(raw):
    """Parse entry_time which may be ISO string or float timestamp."""
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw, tz=UTC)
    if isinstance(raw, str):
        # Strip trailing Z or handle +00:00
        raw = raw.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return None


def round_to_interval(dt, minutes):
    """Round datetime to nearest N-minute interval."""
    if dt is None:
        return None
    ts = dt.timestamp()
    interval = minutes * 60
    rounded = round(ts / interval) * interval
    return datetime.fromtimestamp(rounded, tz=UTC)


def detect_interval(title):
    """Detect whether a market is 5-min or 15-min from its title."""
    m = TITLE_RE.search(title or "")
    if not m:
        return 15  # default to 15
    start_str, end_str = m.group(3), m.group(4)
    start = datetime.strptime(start_str.upper(), "%I:%M%p")
    end = datetime.strptime(end_str.upper(), "%I:%M%p")
    diff = (end - start).total_seconds() / 60
    if diff < 0:
        diff += 12 * 60  # AM/PM crossover
    return int(diff) if diff > 0 else 15


# ── Data Loading ─────────────────────────────────────────────────────────

def load_all_trades():
    """Load and normalize all trades from all sources."""
    trades = []

    # 1) Fifteen-min strategies (paper)
    for t in load_json("fifteen_min_strategies_results.json"):
        if t.get("status") != "closed":
            continue
        trades.append({
            "source": "15m_strategies",
            "strategy": t.get("strategy", "unknown"),
            "side": t.get("side", "").upper(),
            "pnl": t.get("pnl", 0),
            "won": t.get("pnl", 0) > 0,
            "title": t.get("title", ""),
            "entry_time": parse_entry_time(t.get("entry_time")),
            "close_time": parse_close_time(t.get("title", "")),
            "interval": detect_interval(t.get("title", "")),
            "live": False,
        })

    # 2) Momentum 15M (live + archived paper)
    momentum_files = [
        "momentum_15m_results.json",
        "archive/momentum_15m_results_paper_20260218_203743.json",
        "archive/momentum_15m_results_paper_20260219.json",
    ]
    for mf in momentum_files:
        for t in load_json(mf):
            if t.get("status") != "closed":
                continue
            trades.append({
                "source": "momentum_15m",
                "strategy": t.get("strategy", "momentum_strong"),
                "side": t.get("side", "").upper(),
                "pnl": t.get("pnl", 0),
                "won": t.get("pnl", 0) > 0,
                "title": t.get("title", ""),
                "entry_time": parse_entry_time(t.get("entry_time")),
                "close_time": parse_close_time(t.get("title", "")),
                "interval": detect_interval(t.get("title", "")),
                "live": "order_id" in t and t.get("order_id") is not None,
            })

    # 3) Sniper 5M (paper + archived)
    sniper_paper_files = [
        "sniper_5m_results.json",
        "archive/sniper_5m_results_v1.0_20260218.json",
    ]
    for sf in sniper_paper_files:
        for t in load_json(sf):
            if t.get("status") != "closed":
                continue
            trades.append({
                "source": "sniper_5m",
                "strategy": t.get("strategy", "sniper_5m"),
                "side": t.get("side", "").upper(),
                "pnl": t.get("pnl", 0),
                "won": t.get("pnl", 0) > 0,
                "title": t.get("question", ""),
                "entry_time": parse_entry_time(t.get("entry_time")),
                "close_time": parse_close_time(t.get("question", "")),
                "interval": detect_interval(t.get("question", "")),
                "live": False,
            })

    # 4) Sniper 5M live + archived
    sniper_live_files = [
        "sniper_5m_live_results.json",
        "archive/sniper_5m_live_results_pre_v1.1.json",
    ]
    for sf in sniper_live_files:
        for t in load_json(sf):
            if t.get("status") != "closed":
                continue
            trades.append({
                "source": "sniper_5m_live",
                "strategy": t.get("strategy", "sniper_5m_live"),
                "side": t.get("side", "").upper(),
                "pnl": t.get("pnl", 0),
                "won": t.get("pnl", 0) > 0,
                "title": t.get("question", ""),
                "entry_time": parse_entry_time(t.get("entry_time")),
                "close_time": parse_close_time(t.get("question", "")),
                "interval": detect_interval(t.get("question", "")),
                "live": True,
            })

    # De-duplicate by (source, entry_time, side, title)
    seen = set()
    deduped = []
    for t in trades:
        key = (t["source"], str(t["entry_time"]), t["side"], t["title"])
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    return deduped


# ── Analysis ─────────────────────────────────────────────────────────────

def analyze_consensus(trades):
    """Group trades by market close time and analyze consensus."""

    # ---------- Per-source summary ----------
    print("=" * 80)
    print("DATA OVERVIEW")
    print("=" * 80)
    source_stats = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        s = source_stats[t["source"]]
        s["n"] += 1
        s["wins"] += int(t["won"])
        s["pnl"] += t["pnl"]

    print(f"{'Source':<25} {'Trades':>7} {'Wins':>6} {'WR':>7} {'PnL':>10}")
    print("-" * 60)
    for src in sorted(source_stats):
        s = source_stats[src]
        wr = s["wins"] / s["n"] * 100 if s["n"] else 0
        print(f"{src:<25} {s['n']:>7} {s['wins']:>6} {wr:>6.1f}% ${s['pnl']:>8.2f}")
    total_n = sum(s["n"] for s in source_stats.values())
    total_w = sum(s["wins"] for s in source_stats.values())
    total_pnl = sum(s["pnl"] for s in source_stats.values())
    wr_all = total_w / total_n * 100 if total_n else 0
    print("-" * 60)
    print(f"{'TOTAL':<25} {total_n:>7} {total_w:>6} {wr_all:>6.1f}% ${total_pnl:>8.2f}")

    # ---------- Strategy-level summary ----------
    print(f"\n{'Strategy':<25} {'Trades':>7} {'Wins':>6} {'WR':>7} {'PnL':>10}")
    print("-" * 60)
    strat_stats = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        s = strat_stats[t["strategy"]]
        s["n"] += 1
        s["wins"] += int(t["won"])
        s["pnl"] += t["pnl"]
    for strat in sorted(strat_stats, key=lambda x: -strat_stats[x]["n"]):
        s = strat_stats[strat]
        wr = s["wins"] / s["n"] * 100 if s["n"] else 0
        print(f"{strat:<25} {s['n']:>7} {s['wins']:>6} {wr:>6.1f}% ${s['pnl']:>8.2f}")

    # ---------- Group by market (15M markets only for cross-bot consensus) ----------
    print("\n" + "=" * 80)
    print("15-MINUTE MARKET CONSENSUS ANALYSIS")
    print("=" * 80)

    # Group all trades by close_time (market identifier)
    markets_15m = defaultdict(list)
    for t in trades:
        if t["close_time"] is None:
            continue
        if t["interval"] == 15:
            markets_15m[t["close_time"]].append(t)

    # Classify each market
    solo_trades = []         # Only 1 signal
    agree_trades = []        # 2+ signals, all same direction
    disagree_markets = []    # 2+ signals, mixed directions

    for close_time, mkt_trades in sorted(markets_15m.items()):
        # Get unique (strategy, side) pairs to avoid double-counting same strategy
        strat_signals = {}
        for t in mkt_trades:
            key = t["strategy"]
            if key not in strat_signals:
                strat_signals[key] = t

        unique_trades = list(strat_signals.values())

        if len(unique_trades) == 1:
            solo_trades.append(unique_trades[0])
        else:
            sides = set(t["side"] for t in unique_trades)
            if len(sides) == 1:
                # All agree
                agree_trades.extend(unique_trades)
            else:
                # Disagree
                disagree_markets.append(unique_trades)

    # Solo WR
    solo_wins = sum(1 for t in solo_trades if t["won"])
    solo_n = len(solo_trades)
    solo_wr = solo_wins / solo_n * 100 if solo_n else 0

    # All-agree WR (per market, not per trade)
    # Group agree trades by market
    agree_by_mkt = defaultdict(list)
    for t in agree_trades:
        agree_by_mkt[t["close_time"]].append(t)
    agree_mkt_wins = 0
    agree_mkt_total = 0
    agree_trade_wins = sum(1 for t in agree_trades if t["won"])
    agree_trade_n = len(agree_trades)
    for close_time, mkt_trades in agree_by_mkt.items():
        agree_mkt_total += 1
        # "Market won" = majority of trades in that market won
        if sum(1 for t in mkt_trades if t["won"]) > len(mkt_trades) / 2:
            agree_mkt_wins += 1
    agree_mkt_wr = agree_mkt_wins / agree_mkt_total * 100 if agree_mkt_total else 0
    agree_trade_wr = agree_trade_wins / agree_trade_n * 100 if agree_trade_n else 0

    # Disagree analysis
    disagree_total_trades = sum(len(m) for m in disagree_markets)
    disagree_winning_side_wins = 0
    disagree_winning_side_total = 0
    disagree_majority_wins = 0
    disagree_market_count = len(disagree_markets)
    for mkt_trades in disagree_markets:
        # Group by side
        by_side = defaultdict(list)
        for t in mkt_trades:
            by_side[t["side"]].append(t)
        # Which side had more signals?
        majority_side = max(by_side.keys(), key=lambda s: len(by_side[s]))
        majority_trades = by_side[majority_side]
        majority_won = sum(1 for t in majority_trades if t["won"]) > len(majority_trades) / 2
        if majority_won:
            disagree_majority_wins += 1
        # For all trades in disagreement markets
        for t in mkt_trades:
            disagree_winning_side_total += 1
            if t["won"]:
                disagree_winning_side_wins += 1
    disagree_wr = disagree_winning_side_wins / disagree_winning_side_total * 100 if disagree_winning_side_total else 0

    print(f"\n{'Category':<30} {'Markets':>8} {'Trades':>8} {'WR (trade)':>12} {'WR (mkt)':>10}")
    print("-" * 72)
    print(f"{'Solo (1 signal)':<30} {solo_n:>8} {solo_n:>8} {solo_wr:>11.1f}% {solo_wr:>9.1f}%")
    print(f"{'All Agree (2+ same dir)':<30} {agree_mkt_total:>8} {agree_trade_n:>8} {agree_trade_wr:>11.1f}% {agree_mkt_wr:>9.1f}%")
    if disagree_market_count:
        disagree_mkt_wr = disagree_majority_wins / disagree_market_count * 100
        print(f"{'Disagree (mixed dirs)':<30} {disagree_market_count:>8} {disagree_total_trades:>8} {disagree_wr:>11.1f}% {disagree_mkt_wr:>9.1f}%")
    else:
        print(f"{'Disagree (mixed dirs)':<30} {'0':>8} {'0':>8} {'N/A':>12} {'N/A':>10}")

    # Show detail on agree markets
    print("\n--- Markets where 2+ strategies agreed ---")
    for close_time, mkt_trades in sorted(agree_by_mkt.items()):
        strats = [t["strategy"] for t in mkt_trades]
        side = mkt_trades[0]["side"]
        won = all(t["won"] for t in mkt_trades)
        result = "WIN" if won else "LOSS" if all(not t["won"] for t in mkt_trades) else "MIXED"
        close_str = close_time.strftime("%b %d %H:%M UTC")
        print(f"  {close_str} | {side:<5} | {result:<5} | {' + '.join(strats)}")

    # Show detail on disagree markets
    if disagree_markets:
        print("\n--- Markets where strategies disagreed ---")
        for mkt_trades in disagree_markets:
            close_time = mkt_trades[0]["close_time"]
            close_str = close_time.strftime("%b %d %H:%M UTC") if close_time else "?"
            for t in mkt_trades:
                won_str = "W" if t["won"] else "L"
                print(f"  {close_str} | {t['strategy']:<20} | {t['side']:<5} | {won_str} | pnl=${t['pnl']:.2f}")

    # ---------- 5-Minute markets: sniper consensus ----------
    print("\n" + "=" * 80)
    print("5-MINUTE MARKET ANALYSIS (Sniper bots)")
    print("=" * 80)

    markets_5m = defaultdict(list)
    for t in trades:
        if t["close_time"] is None:
            continue
        if t["interval"] == 5:
            markets_5m[t["close_time"]].append(t)

    sniper_solo = 0
    sniper_solo_wins = 0
    sniper_multi = 0
    sniper_multi_wins = 0
    for close_time, mkt_trades in markets_5m.items():
        strat_signals = {}
        for t in mkt_trades:
            key = (t["source"], t["strategy"])
            if key not in strat_signals:
                strat_signals[key] = t
        unique = list(strat_signals.values())
        if len(unique) == 1:
            sniper_solo += 1
            if unique[0]["won"]:
                sniper_solo_wins += 1
        else:
            # Paper + live on same market
            sniper_multi += 1
            if sum(1 for t in unique if t["won"]) > len(unique) / 2:
                sniper_multi_wins += 1

    print(f"  Solo 5M signals:  {sniper_solo} markets, {sniper_solo_wins} wins ({sniper_solo_wins/sniper_solo*100:.1f}% WR)" if sniper_solo else "  Solo 5M signals:  0 markets")
    print(f"  Multi 5M signals: {sniper_multi} markets, {sniper_multi_wins} wins ({sniper_multi_wins/sniper_multi*100:.1f}% WR)" if sniper_multi else "  Multi 5M signals: 0 markets")

    # ---------- Cross-timeframe: can 5M snipers confirm 15M signals? ----------
    print("\n" + "=" * 80)
    print("CROSS-TIMEFRAME: 5M Sniper within same 15M window")
    print("=" * 80)

    # For each 15M market, check if any 5M sniper traded within its window
    cross_confirmed = 0
    cross_confirmed_wins = 0
    cross_unconfirmed = 0
    cross_unconfirmed_wins = 0

    for close_15m, trades_15m in markets_15m.items():
        # 15M window: close_15m - 15min to close_15m
        window_start = close_15m - timedelta(minutes=15)
        # Find 5M trades in this window
        sniper_in_window = []
        for close_5m, trades_5m in markets_5m.items():
            if window_start <= close_5m <= close_15m:
                sniper_in_window.extend(trades_5m)

        # Get 15M direction
        sides_15m = set(t["side"] for t in trades_15m)
        if len(sides_15m) != 1:
            continue  # Skip disagreement markets
        dir_15m = sides_15m.pop()

        # Did any sniper agree?
        sniper_agrees = any(t["side"] == dir_15m for t in sniper_in_window)
        market_won = sum(1 for t in trades_15m if t["won"]) > len(trades_15m) / 2

        if sniper_in_window and sniper_agrees:
            cross_confirmed += 1
            if market_won:
                cross_confirmed_wins += 1
        elif not sniper_in_window:
            cross_unconfirmed += 1
            if market_won:
                cross_unconfirmed_wins += 1

    if cross_confirmed:
        print(f"  15M markets confirmed by 5M sniper: {cross_confirmed}, WR={cross_confirmed_wins/cross_confirmed*100:.1f}%")
    else:
        print(f"  15M markets confirmed by 5M sniper: 0 (no overlapping 5M data)")
    if cross_unconfirmed:
        print(f"  15M markets without 5M confirm:     {cross_unconfirmed}, WR={cross_unconfirmed_wins/cross_unconfirmed*100:.1f}%")

    # ---------- Strategy pair agreement matrix ----------
    print("\n" + "=" * 80)
    print("STRATEGY AGREEMENT MATRIX (15M markets)")
    print("=" * 80)

    # Build: for each market, which strategies signaled which direction
    market_signals = {}  # close_time -> {strategy: side}
    market_outcomes = {}  # close_time -> won (True if UP won)
    for close_time, mkt_trades in markets_15m.items():
        sigs = {}
        for t in mkt_trades:
            sigs[t["strategy"]] = {"side": t["side"], "won": t["won"]}
        market_signals[close_time] = sigs

    # Get all strategies with >= 3 trades
    all_strats = sorted(set(
        t["strategy"] for t in trades if t["interval"] == 15
    ))
    # Filter to strategies with meaningful volume
    strat_counts = Counter(t["strategy"] for t in trades if t["interval"] == 15)
    active_strats = [s for s in all_strats if strat_counts[s] >= 3]

    print(f"\nStrategies with 3+ trades: {active_strats}")
    print(f"\n{'Pair':<45} {'Both':>6} {'Agree':>6} {'Agr%':>6} {'AgrWR':>7} {'DisWR':>7}")
    print("-" * 80)

    pair_data = {}  # (s1, s2) -> {agree_n, agree_wins, disagree_n, disagree_wins, both_n}

    for i, s1 in enumerate(active_strats):
        for s2 in active_strats[i + 1:]:
            both = 0
            agree = 0
            agree_wins = 0
            disagree = 0
            disagree_wins = 0
            for close_time, sigs in market_signals.items():
                if s1 in sigs and s2 in sigs:
                    both += 1
                    if sigs[s1]["side"] == sigs[s2]["side"]:
                        agree += 1
                        # Both won or both lost (same direction)
                        if sigs[s1]["won"]:
                            agree_wins += 1
                    else:
                        disagree += 1
                        # Count wins for either side
                        if sigs[s1]["won"]:
                            disagree_wins += 1
                        if sigs[s2]["won"]:
                            disagree_wins += 1

            if both > 0:
                pair_data[(s1, s2)] = {
                    "both": both,
                    "agree": agree,
                    "agree_wins": agree_wins,
                    "disagree": disagree,
                    "disagree_wins": disagree_wins,
                }
                agr_pct = agree / both * 100
                agr_wr = agree_wins / agree * 100 if agree else 0
                dis_wr = disagree_wins / (disagree * 2) * 100 if disagree else 0  # /2 because 2 trades per disagree
                pair_label = f"{s1} + {s2}"
                print(f"{pair_label:<45} {both:>6} {agree:>6} {agr_pct:>5.0f}% {agr_wr:>6.1f}% {dis_wr:>6.1f}%")

    # ---------- Best N-strategy combos ----------
    print("\n" + "=" * 80)
    print("BEST STRATEGY COMBINATIONS (ranked by WR, min 5 consensus markets)")
    print("=" * 80)

    # Include momentum_strong in the mix
    all_15m_strats = list(set(
        t["strategy"] for t in trades if t["interval"] == 15
    ))

    combo_results = []

    # Test pairs
    for combo in combinations(all_15m_strats, 2):
        consensus_wins = 0
        consensus_total = 0
        consensus_pnl = 0.0
        for close_time, sigs in market_signals.items():
            if all(s in sigs for s in combo):
                sides = set(sigs[s]["side"] for s in combo)
                if len(sides) == 1:  # All agree
                    consensus_total += 1
                    # Use average pnl of the combo's trades
                    avg_won = all(sigs[s]["won"] for s in combo)
                    if avg_won:
                        consensus_wins += 1
                    consensus_pnl += sum(sigs[s]["won"] * 1 - (not sigs[s]["won"]) * 1 for s in combo)
        if consensus_total >= 3:  # Lower threshold given limited data
            wr = consensus_wins / consensus_total * 100
            combo_results.append({
                "combo": combo,
                "size": 2,
                "consensus_markets": consensus_total,
                "wins": consensus_wins,
                "wr": wr,
            })

    # Test triples
    for combo in combinations(all_15m_strats, 3):
        consensus_wins = 0
        consensus_total = 0
        for close_time, sigs in market_signals.items():
            if all(s in sigs for s in combo):
                sides = set(sigs[s]["side"] for s in combo)
                if len(sides) == 1:
                    consensus_total += 1
                    if all(sigs[s]["won"] for s in combo):
                        consensus_wins += 1
        if consensus_total >= 2:
            wr = consensus_wins / consensus_total * 100
            combo_results.append({
                "combo": combo,
                "size": 3,
                "consensus_markets": consensus_total,
                "wins": consensus_wins,
                "wr": wr,
            })

    # Sort by WR then by volume
    combo_results.sort(key=lambda x: (-x["wr"], -x["consensus_markets"]))

    print(f"\n{'Rank':>4} {'Combo':<55} {'Markets':>8} {'Wins':>6} {'WR':>7}")
    print("-" * 85)
    for i, cr in enumerate(combo_results[:15]):
        combo_str = " + ".join(cr["combo"])
        print(f"{i+1:>4} {combo_str:<55} {cr['consensus_markets']:>8} {cr['wins']:>6} {cr['wr']:>6.1f}%")

    if not combo_results:
        print("  (No combos found with sufficient consensus markets)")

    # ---------- Trade frequency analysis ----------
    print("\n" + "=" * 80)
    print("TRADE FREQUENCY ANALYSIS")
    print("=" * 80)

    # Calculate date range
    valid_dates = [t["entry_time"].date() for t in trades if t["entry_time"] and t["interval"] == 15]
    if valid_dates:
        min_date = min(valid_dates)
        max_date = max(valid_dates)
        n_days = (max_date - min_date).days + 1
    else:
        n_days = 1

    total_15m = len([t for t in trades if t["interval"] == 15])
    total_solo_mkts = solo_n
    total_agree_mkts = agree_mkt_total
    total_disagree_mkts = disagree_market_count
    total_mkts = total_solo_mkts + total_agree_mkts + total_disagree_mkts

    print(f"  Date range: {min_date} to {max_date} ({n_days} days)")
    print(f"  Total 15M trades: {total_15m} ({total_15m / n_days:.1f}/day)")
    print(f"  Total 15M markets touched: {total_mkts} ({total_mkts / n_days:.1f}/day)")
    print(f"  Solo markets: {total_solo_mkts} ({total_solo_mkts / n_days:.1f}/day)")
    print(f"  Consensus (agree) markets: {total_agree_mkts} ({total_agree_mkts / n_days:.1f}/day)")
    print(f"  Consensus filter (2+ agree only): {total_agree_mkts} trades/period = {total_agree_mkts / n_days:.1f}/day")
    print(f"  Without filter: {total_mkts} markets/period = {total_mkts / n_days:.1f}/day")
    if total_agree_mkts and total_mkts:
        print(f"  Filter would REMOVE {total_mkts - total_agree_mkts} markets ({(total_mkts - total_agree_mkts)/total_mkts*100:.0f}%) but keep only high-confidence ones")

    # ---------- Hourly breakdown of consensus ----------
    print("\n  Consensus by hour (UTC):")
    hour_agree = defaultdict(lambda: {"n": 0, "wins": 0})
    hour_solo = defaultdict(lambda: {"n": 0, "wins": 0})
    for t in solo_trades:
        if t["close_time"]:
            h = t["close_time"].hour
            hour_solo[h]["n"] += 1
            hour_solo[h]["wins"] += int(t["won"])
    for close_time, mkt_trades in agree_by_mkt.items():
        h = close_time.hour
        hour_agree[h]["n"] += 1
        hour_agree[h]["wins"] += int(sum(1 for t in mkt_trades if t["won"]) > len(mkt_trades) / 2)

    all_hours = sorted(set(list(hour_agree.keys()) + list(hour_solo.keys())))
    print(f"  {'Hour':>6} | {'Solo':>6} {'SoloWR':>8} | {'Agree':>6} {'AgrWR':>8}")
    for h in all_hours:
        s = hour_solo[h]
        a = hour_agree[h]
        s_wr = f"{s['wins']/s['n']*100:.0f}%" if s["n"] else "N/A"
        a_wr = f"{a['wins']/a['n']*100:.0f}%" if a["n"] else "N/A"
        print(f"  {h:>4}h  | {s['n']:>6} {s_wr:>8} | {a['n']:>6} {a_wr:>8}")

    # ---------- RECOMMENDATION ----------
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    print(f"""
  BASELINE (all 15M trades):
    Solo:    {solo_n:>4} markets, {solo_wr:.1f}% WR
    Agree:   {agree_mkt_total:>4} markets, {agree_mkt_wr:.1f}% WR (trade-level: {agree_trade_wr:.1f}%)
    Disagree:{disagree_market_count:>4} markets, {disagree_wr:.1f}% avg trade WR
""")

    # Compute the WR lift
    if agree_mkt_total and solo_n:
        lift = agree_mkt_wr - solo_wr
        if lift > 5:
            verdict = "YES - Consensus filter shows meaningful WR improvement"
            detail = (
                f"  When 2+ strategies agree, WR is {agree_mkt_wr:.1f}% vs {solo_wr:.1f}% solo "
                f"(+{lift:.1f}pp lift).\n"
                f"  Cost: lose {total_mkts - total_agree_mkts} markets/period "
                f"({(total_mkts - total_agree_mkts)/total_mkts*100:.0f}% fewer trades).\n"
                f"  At {agree_mkt_total / n_days:.1f} consensus markets/day, "
                f"this is still tradeable volume."
            )
        elif lift > 0:
            verdict = "MARGINAL - Small WR lift, may not justify reduced volume"
            detail = (
                f"  Lift is only +{lift:.1f}pp ({agree_mkt_wr:.1f}% vs {solo_wr:.1f}%).\n"
                f"  You'd lose {(total_mkts - total_agree_mkts)/total_mkts*100:.0f}% of trades "
                f"for minimal improvement.\n"
                f"  Consider: use consensus as a SIZE MULTIPLIER instead of a gate."
            )
        else:
            verdict = "NO - Consensus does not improve WR"
            detail = (
                f"  Solo WR ({solo_wr:.1f}%) >= Consensus WR ({agree_mkt_wr:.1f}%).\n"
                f"  Adding more signals doesn't help. The best single strategy may be sufficient.\n"
                f"  Focus on strategy quality, not quantity."
            )
    else:
        verdict = "INSUFFICIENT DATA"
        detail = "  Not enough consensus markets to draw conclusions."

    print(f"  VERDICT: {verdict}")
    print(detail)

    # Additional nuance
    print(f"""
  ADDITIONAL CONSIDERATIONS:
  - Data covers {n_days} days. Minimum ~2 weeks (14 days) for statistical confidence.
  - 15M strategies file has {strat_counts.get('FRAMA_CMO', 0)} FRAMA_CMO trades dominating.
    Most "solo" trades are just FRAMA_CMO. Consensus = FRAMA_CMO + another = rarer.
  - The 5M sniper runs on DIFFERENT markets (5-min intervals).
    Cross-timeframe confirmation is possible but has very limited overlap.
  - Momentum 15M (live) only has {sum(1 for t in trades if t['source']=='momentum_15m')} trades
    in the current file. Need more data to assess momentum+FRAMA consensus.
""")

    # Best actionable combo
    if combo_results:
        best = combo_results[0]
        print(f"  BEST COMBO: {' + '.join(best['combo'])}")
        print(f"    {best['consensus_markets']} consensus markets, {best['wr']:.1f}% WR")
        print(f"    (vs overall solo WR of {solo_wr:.1f}%)")
    print()


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trades = load_all_trades()
    print(f"Loaded {len(trades)} total trades from all sources.\n")

    if not trades:
        print("ERROR: No trade data found. Check file paths.")
        exit(1)

    analyze_consensus(trades)
