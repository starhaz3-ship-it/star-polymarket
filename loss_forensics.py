"""
Loss Forensics Engine V1.0 - Automated Post-Loss Analysis

Inspired by @sharbel's "why did this lose? fix it" loop, but automated.
Runs after every loss to:
1. Analyze the losing trade's features vs winning patterns
2. Detect recurring loss patterns across trade history
3. Generate specific parameter fix recommendations
4. Write forensic reports to loss_autopsy.json

Usage:
    # Auto-called from live trader after every loss
    from loss_forensics import ForensicsEngine
    engine = ForensicsEngine()
    engine.analyze_loss(trade_dict, all_trades_dict)

    # Standalone: analyze all historical losses
    python loss_forensics.py
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from functools import partial

print = partial(print, flush=True)

AUTOPSY_FILE = Path(__file__).parent / "loss_autopsy.json"
LIVE_RESULTS = Path(__file__).parent / "ta_live_results.json"
PAPER_RESULTS = Path(__file__).parent / "ta_paper_results.json"


class ForensicsEngine:
    """Automated post-loss forensic analyzer."""

    # Known danger zones from historical data
    KNOWN_PATTERNS = {
        "UP_expensive": {"field": "entry_price", "op": ">", "val": 0.50, "side": "UP",
                         "fix": "Tighten MAX_ENTRY_PRICE for UP trades"},
        "DOWN_death_zone": {"field": "entry_price", "op": "range", "val": (0.40, 0.45), "side": "DOWN",
                            "fix": "DOWN $0.40-0.45 = 14% WR death zone, block entirely"},
        "RSI_extreme_UP": {"field": "rsi", "op": ">", "val": 75, "side": "UP",
                           "fix": "RSI>75 UP = overbought, price likely to reverse DOWN"},
        "RSI_extreme_DOWN": {"field": "rsi", "op": "<", "val": 30, "side": "DOWN",
                             "fix": "RSI<30 DOWN = oversold, price likely to bounce UP"},
        "VWAP_dislocated": {"field": "vwap_distance", "op": "abs>", "val": 0.50,
                            "fix": "VWAP distance > 0.50 = price far from fair value, avoid"},
        "counter_trend": {"field": "_with_trend", "op": "==", "val": False,
                          "fix": "Counter-trend trades have lower WR, reduce size or skip"},
        "low_edge": {"field": "edge_at_entry", "op": "<", "val": 0.30,
                     "fix": "Edge < 30% = thin margin, raise MIN_EDGE"},
        "heiken_contradiction_UP": {"field": "heiken_bullish", "op": "==", "val": False, "side": "UP",
                                    "extra_field": "heiken_count", "extra_op": ">=", "extra_val": 3,
                                    "fix": "UP trade with 3+ bearish Heiken candles = contradiction"},
        "late_entry": {"field": "time_remaining", "op": ">", "val": 12.0,
                       "fix": "Entry > 12 min before expiry = too early, tighten MAX_TIME_REMAINING"},
    }

    def __init__(self):
        self.autopsy_data = self._load_autopsy()

    def _load_autopsy(self) -> dict:
        if AUTOPSY_FILE.exists():
            try:
                return json.load(open(AUTOPSY_FILE))
            except Exception:
                pass
        return {
            "version": "1.0",
            "total_losses_analyzed": 0,
            "reports": [],
            "pattern_counts": {},
            "active_recommendations": [],
            "applied_fixes": [],
        }

    def _save_autopsy(self):
        with open(AUTOPSY_FILE, 'w') as f:
            json.dump(self.autopsy_data, f, indent=2)

    def _extract_features(self, trade: dict) -> dict:
        """Flatten trade + features into single dict for analysis."""
        flat = {
            "side": trade.get("side", ""),
            "entry_price": trade.get("entry_price", 0),
            "exit_price": trade.get("exit_price", 0),
            "pnl": trade.get("pnl", 0),
            "size_usd": trade.get("size_usd", 0),
            "edge_at_entry": trade.get("edge_at_entry", 0),
            "kl_divergence": trade.get("kl_divergence", 0),
            "market_title": trade.get("market_title", ""),
            "entry_time": trade.get("entry_time", ""),
            "exit_time": trade.get("exit_time", ""),
        }
        features = trade.get("features", {})
        flat.update(features)
        # Extract asset from market_title
        title = trade.get("market_title", "")
        if title.startswith("["):
            flat["asset"] = title[1:title.index("]")] if "]" in title else "?"
        else:
            flat["asset"] = "?"
        # Extract entry hour
        try:
            flat["entry_hour"] = datetime.fromisoformat(trade["entry_time"]).hour
        except Exception:
            flat["entry_hour"] = -1
        return flat

    def _check_pattern(self, pattern: dict, flat: dict) -> bool:
        """Check if a known danger pattern matches this trade."""
        # Side filter
        if "side" in pattern and flat.get("side") != pattern["side"]:
            return False

        field = pattern["field"]
        # Get value from flat features or top-level
        val = flat.get(field)
        if val is None:
            return False

        op = pattern["op"]
        threshold = pattern["val"]

        if op == ">":
            match = val > threshold
        elif op == "<":
            match = val < threshold
        elif op == ">=":
            match = val >= threshold
        elif op == "==":
            match = val == threshold
        elif op == "range":
            match = threshold[0] <= val < threshold[1]
        elif op == "abs>":
            match = abs(val) > threshold
        else:
            match = False

        # Check extra condition if present
        if match and "extra_field" in pattern:
            extra_val = flat.get(pattern["extra_field"])
            if extra_val is None:
                return False
            extra_op = pattern["extra_op"]
            extra_threshold = pattern["extra_val"]
            if extra_op == ">=":
                match = extra_val >= extra_threshold
            elif extra_op == "==":
                match = extra_val == extra_threshold

        return match

    def analyze_loss(self, trade: dict, all_trades: dict = None) -> dict:
        """
        Analyze a single losing trade. Returns forensic report.

        Args:
            trade: The losing trade dict
            all_trades: All trades dict (for pattern comparison)
        """
        flat = self._extract_features(trade)

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade_id": trade.get("trade_id", "unknown"),
            "asset": flat["asset"],
            "side": flat["side"],
            "entry_price": flat["entry_price"],
            "exit_price": flat["exit_price"],
            "pnl": flat["pnl"],
            "size_usd": flat["size_usd"],
            "entry_hour": flat["entry_hour"],
            "triggers": [],
            "diagnosis": "",
            "recommendation": "",
            "severity": "LOW",  # LOW, MEDIUM, HIGH, CRITICAL
        }

        # Check all known patterns
        triggered = []
        for pattern_name, pattern in self.KNOWN_PATTERNS.items():
            if self._check_pattern(pattern, flat):
                triggered.append({
                    "pattern": pattern_name,
                    "fix": pattern["fix"],
                    "value": flat.get(pattern["field"]),
                })
                # Track pattern frequency
                self.autopsy_data["pattern_counts"][pattern_name] = \
                    self.autopsy_data["pattern_counts"].get(pattern_name, 0) + 1

        report["triggers"] = triggered

        # Dynamic analysis: compare against winning averages
        if all_trades:
            wins = [v for v in all_trades.values()
                    if v.get("status") == "closed" and v.get("pnl", 0) > 0]
            if wins:
                win_features = [self._extract_features(w) for w in wins]
                deviations = self._find_deviations(flat, win_features)
                if deviations:
                    report["deviations_from_winners"] = deviations

        # Generate diagnosis
        if len(triggered) >= 3:
            report["severity"] = "CRITICAL"
            report["diagnosis"] = f"CRITICAL: {len(triggered)} danger patterns triggered simultaneously"
        elif len(triggered) == 2:
            report["severity"] = "HIGH"
            report["diagnosis"] = f"HIGH: Multiple danger patterns ({', '.join(t['pattern'] for t in triggered)})"
        elif len(triggered) == 1:
            report["severity"] = "MEDIUM"
            report["diagnosis"] = f"MEDIUM: {triggered[0]['pattern']} - {triggered[0]['fix']}"
        else:
            report["severity"] = "LOW"
            report["diagnosis"] = "No known pattern matched - novel loss type. Needs manual review."

        # Generate specific recommendation
        if triggered:
            # Prioritize by pattern frequency (most common = most important to fix)
            sorted_triggers = sorted(triggered,
                                     key=lambda t: self.autopsy_data["pattern_counts"].get(t["pattern"], 0),
                                     reverse=True)
            top = sorted_triggers[0]
            count = self.autopsy_data["pattern_counts"].get(top["pattern"], 1)
            report["recommendation"] = f"[FIX #{count}] {top['fix']} (triggered {count}x total)"

            # Check for repeat offenders: same pattern 3+ times = escalate
            if count >= 3:
                rec = {
                    "pattern": top["pattern"],
                    "fix": top["fix"],
                    "occurrences": count,
                    "severity": "URGENT",
                    "auto_fixable": top["pattern"] in self._get_auto_fixable(),
                }
                # Add to active recommendations if not already there
                existing = [r["pattern"] for r in self.autopsy_data["active_recommendations"]]
                if top["pattern"] not in existing:
                    self.autopsy_data["active_recommendations"].append(rec)

        # Store report
        self.autopsy_data["total_losses_analyzed"] += 1
        self.autopsy_data["reports"].append(report)
        # Keep last 100 reports
        if len(self.autopsy_data["reports"]) > 100:
            self.autopsy_data["reports"] = self.autopsy_data["reports"][-100:]

        self._save_autopsy()

        # Print summary
        print(f"[FORENSICS] Loss #{self.autopsy_data['total_losses_analyzed']}: "
              f"{flat['asset']} {flat['side']} @${flat['entry_price']:.3f} -> ${flat['pnl']:+.2f}")
        print(f"  Severity: {report['severity']} | Triggers: {len(triggered)}")
        for t in triggered:
            print(f"    - {t['pattern']}: {t['fix']} (val={t['value']})")
        if report.get("recommendation"):
            print(f"  FIX: {report['recommendation']}")

        return report

    def _find_deviations(self, loss_features: dict, win_features: list) -> list:
        """Find how the losing trade deviates from winning trade averages."""
        deviations = []
        numeric_fields = ["rsi", "vwap_distance", "entry_price", "edge_at_entry",
                          "kl_divergence", "time_remaining", "heiken_count",
                          "macd_histogram", "_atr_ratio"]

        for field in numeric_fields:
            loss_val = loss_features.get(field)
            if loss_val is None or not isinstance(loss_val, (int, float)):
                continue

            win_vals = [w.get(field) for w in win_features
                        if w.get(field) is not None and isinstance(w.get(field), (int, float))]
            if not win_vals:
                continue

            win_avg = sum(win_vals) / len(win_vals)
            win_std = (sum((v - win_avg) ** 2 for v in win_vals) / len(win_vals)) ** 0.5

            if win_std > 0:
                z_score = (loss_val - win_avg) / win_std
                if abs(z_score) >= 1.5:
                    direction = "above" if z_score > 0 else "below"
                    deviations.append({
                        "field": field,
                        "loss_value": round(loss_val, 4),
                        "win_average": round(win_avg, 4),
                        "z_score": round(z_score, 2),
                        "note": f"{field} is {abs(z_score):.1f} std devs {direction} winning average"
                    })

        return sorted(deviations, key=lambda d: abs(d["z_score"]), reverse=True)

    def _get_auto_fixable(self) -> set:
        """Patterns that can be auto-fixed by parameter adjustment."""
        return {
            "UP_expensive",
            "VWAP_dislocated",
            "low_edge",
            "late_entry",
            "RSI_extreme_UP",
            "RSI_extreme_DOWN",
        }

    def run_full_audit(self, source: str = "live") -> dict:
        """
        Run full forensic audit on all historical losses.

        Args:
            source: "live" or "paper"
        """
        results_file = LIVE_RESULTS if source == "live" else PAPER_RESULTS
        if not results_file.exists():
            print(f"[FORENSICS] {results_file} not found")
            return {}

        data = json.load(open(results_file))
        all_trades = data.get("trades", {})
        closed = {k: v for k, v in all_trades.items() if v.get("status") == "closed"}
        losses = {k: v for k, v in closed.items() if v.get("pnl", 0) <= 0}
        wins = {k: v for k, v in closed.items() if v.get("pnl", 0) > 0}

        print(f"\n{'='*70}")
        print(f"LOSS FORENSICS FULL AUDIT - {source.upper()}")
        print(f"{'='*70}")
        print(f"Total: {len(closed)} trades | {len(wins)}W / {len(losses)}L | "
              f"WR: {len(wins)/max(1,len(closed))*100:.0f}%")
        print()

        # Analyze each loss
        reports = []
        for trade_id, trade in sorted(losses.items(),
                                       key=lambda x: x[1].get("entry_time", "")):
            report = self.analyze_loss(trade, all_trades)
            reports.append(report)
            print()

        # Pattern summary
        print(f"{'='*70}")
        print("PATTERN FREQUENCY (most common loss causes):")
        print(f"{'='*70}")
        for pattern, count in sorted(self.autopsy_data["pattern_counts"].items(),
                                     key=lambda x: -x[1]):
            pct = count / max(1, len(losses)) * 100
            bar = "#" * min(30, int(pct / 3))
            fix = self.KNOWN_PATTERNS.get(pattern, {}).get("fix", "")
            print(f"  {pattern:30s} {count:3d} ({pct:4.0f}%) {bar}")
            print(f"    -> {fix}")
        print()

        # Asset breakdown
        print("ASSET BREAKDOWN:")
        asset_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
        for v in closed.values():
            title = v.get("market_title", "")
            asset = title[1:title.index("]")] if title.startswith("[") and "]" in title else "?"
            won = v.get("pnl", 0) > 0
            asset_stats[asset]["wins" if won else "losses"] += 1
            asset_stats[asset]["pnl"] += v.get("pnl", 0)
        for asset, s in sorted(asset_stats.items()):
            total = s["wins"] + s["losses"]
            wr = s["wins"] / total * 100 if total > 0 else 0
            print(f"  {asset}: {total}T ({s['wins']}W/{s['losses']}L) {wr:.0f}% WR ${s['pnl']:+.2f}")
        print()

        # Side breakdown
        print("SIDE BREAKDOWN:")
        side_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
        for v in closed.values():
            side = v.get("side", "?")
            won = v.get("pnl", 0) > 0
            side_stats[side]["wins" if won else "losses"] += 1
            side_stats[side]["pnl"] += v.get("pnl", 0)
        for side, s in sorted(side_stats.items()):
            total = s["wins"] + s["losses"]
            wr = s["wins"] / total * 100 if total > 0 else 0
            print(f"  {side}: {total}T ({s['wins']}W/{s['losses']}L) {wr:.0f}% WR ${s['pnl']:+.2f}")
        print()

        # Hour breakdown
        print("HOUR BREAKDOWN (UTC):")
        hour_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
        for v in closed.values():
            try:
                h = datetime.fromisoformat(v["entry_time"]).hour
                won = v.get("pnl", 0) > 0
                hour_stats[h]["wins" if won else "losses"] += 1
                hour_stats[h]["pnl"] += v.get("pnl", 0)
            except Exception:
                pass
        for h in sorted(hour_stats.keys()):
            s = hour_stats[h]
            total = s["wins"] + s["losses"]
            wr = s["wins"] / total * 100 if total > 0 else 0
            tag = " ** DANGER" if wr < 40 and total >= 2 else ""
            print(f"  UTC {h:02d}: {total}T ({s['wins']}W/{s['losses']}L) {wr:.0f}% WR ${s['pnl']:+.2f}{tag}")
        print()

        # Price bucket breakdown
        print("ENTRY PRICE BUCKETS:")
        price_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
        for v in closed.values():
            p = v.get("entry_price", 0)
            bucket = f"${int(p*20)/20:.2f}-{int(p*20)/20+0.05:.2f}"
            won = v.get("pnl", 0) > 0
            price_stats[bucket]["wins" if won else "losses"] += 1
            price_stats[bucket]["pnl"] += v.get("pnl", 0)
        for bucket in sorted(price_stats.keys()):
            s = price_stats[bucket]
            total = s["wins"] + s["losses"]
            wr = s["wins"] / total * 100 if total > 0 else 0
            tag = " ** DANGER" if wr < 40 and total >= 2 else ""
            print(f"  {bucket}: {total}T ({s['wins']}W/{s['losses']}L) {wr:.0f}% WR ${s['pnl']:+.2f}{tag}")
        print()

        # Active recommendations
        if self.autopsy_data["active_recommendations"]:
            print(f"{'='*70}")
            print("ACTIVE RECOMMENDATIONS (pattern triggered 3+ times):")
            print(f"{'='*70}")
            for rec in self.autopsy_data["active_recommendations"]:
                auto = " [AUTO-FIXABLE]" if rec.get("auto_fixable") else ""
                print(f"  [{rec['severity']}] {rec['pattern']} ({rec['occurrences']}x){auto}")
                print(f"    -> {rec['fix']}")
            print()

        self._save_autopsy()
        return self.autopsy_data


def generate_fix_code(autopsy_data: dict) -> str:
    """Generate Python code snippets for recommended fixes."""
    fixes = []
    for rec in autopsy_data.get("active_recommendations", []):
        pattern = rec["pattern"]
        if pattern == "UP_expensive":
            fixes.append("# Tighten UP entry price\nMAX_ENTRY_PRICE = 0.45  # Was higher")
        elif pattern == "VWAP_dislocated":
            fixes.append("# Tighten VWAP filter\n# Block trades with abs(vwap_distance) > 0.40")
        elif pattern == "counter_trend":
            fixes.append("# Increase counter-trend penalty\n# position_size *= 0.70  # Was 0.85")
        elif pattern == "low_edge":
            fixes.append("# Raise edge floor\nMIN_EDGE = 0.25  # Was lower")
        elif pattern == "late_entry":
            fixes.append("# Tighten entry window\nMAX_TIME_REMAINING = 10.0  # Was higher")
        elif pattern == "RSI_extreme_UP":
            fixes.append("# Block UP trades when RSI > 75\n# if signal.side == 'UP' and rsi > 75: skip")
        elif pattern == "RSI_extreme_DOWN":
            fixes.append("# Block DOWN trades when RSI < 30\n# if signal.side == 'DOWN' and rsi < 30: skip")
    return "\n\n".join(fixes) if fixes else "# No auto-fixes recommended yet"


if __name__ == "__main__":
    engine = ForensicsEngine()
    engine.run_full_audit("live")
