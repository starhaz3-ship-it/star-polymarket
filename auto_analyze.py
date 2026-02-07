#!/usr/bin/env python3
"""
Auto-Analysis Pipeline — runs every 6 hours via Task Scheduler.
Generates latest_analysis.json for Claude's Evolution Protocol.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

DIR = Path(__file__).parent
LIVE_FILE = DIR / "ta_live_results.json"
PAPER_FILE = DIR / "ta_paper_results.json"
OUTPUT_FILE = DIR / "latest_analysis.json"


def load_json(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def analyze_trades(trades_dict, label=""):
    """Analyze a dict of trades and return stats breakdown."""
    closed = [t for t in trades_dict.values() if t.get("status") == "closed"]
    if not closed:
        return {"total": 0, "label": label}

    wins = [t for t in closed if t.get("pnl", 0) > 0]
    losses = [t for t in closed if t.get("pnl", 0) <= 0]
    total_pnl = sum(t.get("pnl", 0) for t in closed)

    # By side
    by_side = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0})
    for t in closed:
        side = t.get("side", "UNKNOWN")
        by_side[side]["trades"] += 1
        by_side[side]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            by_side[side]["wins"] += 1
        else:
            by_side[side]["losses"] += 1

    # By asset
    by_asset = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0})
    for t in closed:
        title = t.get("market_title", "") or ""
        asset = "BTC"
        if "ethereum" in title.lower() or "eth" in title.lower():
            asset = "ETH"
        elif "solana" in title.lower() or "sol" in title.lower():
            asset = "SOL"
        by_asset[asset]["trades"] += 1
        by_asset[asset]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            by_asset[asset]["wins"] += 1
        else:
            by_asset[asset]["losses"] += 1

    # By entry hour
    by_hour = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0})
    for t in closed:
        try:
            entry_time = t.get("entry_time", "")
            if isinstance(entry_time, str) and entry_time:
                hour = datetime.fromisoformat(entry_time).hour
            else:
                continue
        except Exception:
            continue
        by_hour[hour]["trades"] += 1
        by_hour[hour]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            by_hour[hour]["wins"] += 1
        else:
            by_hour[hour]["losses"] += 1

    # By price bucket
    by_price = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0})
    for t in closed:
        ep = t.get("entry_price", 0.5)
        if ep < 0.20:
            bucket = "<0.20"
        elif ep < 0.30:
            bucket = "0.20-0.30"
        elif ep < 0.35:
            bucket = "0.30-0.35"
        elif ep < 0.40:
            bucket = "0.35-0.40"
        elif ep < 0.45:
            bucket = "0.40-0.45"
        elif ep < 0.50:
            bucket = "0.45-0.50"
        else:
            bucket = "0.50+"
        by_price[bucket]["trades"] += 1
        by_price[bucket]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) > 0:
            by_price[bucket]["wins"] += 1
        else:
            by_price[bucket]["losses"] += 1

    # Worst performers (lowest WR with 3+ trades)
    losers = []
    for bucket, stats in by_price.items():
        if stats["trades"] >= 3:
            wr = stats["wins"] / stats["trades"]
            if wr < 0.45:
                losers.append({"bucket": bucket, "wr": round(wr, 3), "pnl": round(stats["pnl"], 2), "trades": stats["trades"]})
    losers.sort(key=lambda x: x["wr"])

    return {
        "label": label,
        "total": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed), 4) if closed else 0,
        "total_pnl": round(total_pnl, 2),
        "by_side": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in by_side.items()},
        "by_asset": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in by_asset.items()},
        "by_hour": {str(k): {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in sorted(by_hour.items())},
        "by_price_bucket": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in by_price.items()},
        "worst_buckets": losers[:5],
    }


def analyze_shadows(paper_data):
    """Analyze skip-hour shadow trades."""
    shadows = paper_data.get("skip_hour_shadows", {})
    stats = paper_data.get("skip_hour_stats", {})

    closed = [s for s in shadows.values() if s.get("status") == "closed"]
    open_count = sum(1 for s in shadows.values() if s.get("status") == "open")

    verdicts = {}
    for h_str, s in stats.items():
        h = int(h_str)
        total = s.get("wins", 0) + s.get("losses", 0)
        if total > 0:
            wr = s["wins"] / total
            verdict = "REOPEN" if wr >= 0.55 and total >= 10 else "KEEP_SKIP" if total >= 10 else "COLLECTING"
            verdicts[h] = {
                "wins": s["wins"], "losses": s["losses"],
                "pnl": round(s.get("pnl", 0), 2),
                "wr": round(wr, 4), "trades": total,
                "verdict": verdict,
            }

    return {
        "total_shadows": len(shadows),
        "closed": len(closed),
        "open": open_count,
        "total_pnl": round(sum(s.get("pnl", 0) for s in closed), 2),
        "per_hour_verdicts": verdicts,
        "reopen_candidates": [h for h, v in verdicts.items() if v["verdict"] == "REOPEN"],
    }


def generate_suggestions(live_analysis, paper_analysis, shadow_analysis):
    """Generate actionable suggestions based on analysis."""
    suggestions = []

    # Check for asset underperformance
    for asset, stats in live_analysis.get("by_asset", {}).items():
        if stats["trades"] >= 5:
            wr = stats["wins"] / stats["trades"]
            if wr < 0.40:
                suggestions.append(f"WARN: {asset} has {wr:.0%} WR in {stats['trades']} live trades — consider disabling")

    # Check for side imbalance
    for side, stats in live_analysis.get("by_side", {}).items():
        if stats["trades"] >= 5:
            wr = stats["wins"] / stats["trades"]
            if side == "UP" and wr < 0.40:
                suggestions.append(f"WARN: UP trades at {wr:.0%} WR — tighten UP filters or go DOWN-only")

    # Shadow hour reopens
    for h in shadow_analysis.get("reopen_candidates", []):
        v = shadow_analysis["per_hour_verdicts"][h]
        suggestions.append(f"REOPEN: UTC {h} has {v['wr']:.0%} WR in {v['trades']} shadow trades (${v['pnl']:+.2f})")

    # Price bucket warnings
    for bucket_info in live_analysis.get("worst_buckets", []):
        suggestions.append(f"LOSER BUCKET: {bucket_info['bucket']} = {bucket_info['wr']:.0%} WR ({bucket_info['trades']}T, ${bucket_info['pnl']:+.2f})")

    return suggestions


def main():
    print(f"[AUTO-ANALYZE] Running at {datetime.now(timezone.utc).isoformat()}")

    live_data = load_json(LIVE_FILE)
    paper_data = load_json(PAPER_FILE)

    live_analysis = analyze_trades(live_data.get("trades", {}), label="live")
    paper_analysis = analyze_trades(paper_data.get("trades", {}), label="paper")
    shadow_analysis = analyze_shadows(paper_data)
    suggestions = generate_suggestions(live_analysis, paper_analysis, shadow_analysis)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live": live_analysis,
        "paper": paper_analysis,
        "skip_hour_shadows": shadow_analysis,
        "suggestions": suggestions,
        "hourly_stats_live": live_data.get("hourly_stats", {}),
        "hourly_stats_paper": paper_data.get("hourly_stats", {}),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[AUTO-ANALYZE] Written to {OUTPUT_FILE}")
    print(f"  Live: {live_analysis['total']}T {live_analysis.get('win_rate', 0):.0%} WR ${live_analysis.get('total_pnl', 0):+.2f}")
    print(f"  Paper: {paper_analysis['total']}T {paper_analysis.get('win_rate', 0):.0%} WR ${paper_analysis.get('total_pnl', 0):+.2f}")
    print(f"  Shadows: {shadow_analysis['total_shadows']} ({shadow_analysis['closed']} closed)")
    print(f"  Suggestions: {len(suggestions)}")
    for s in suggestions:
        print(f"    -> {s}")


if __name__ == "__main__":
    main()
