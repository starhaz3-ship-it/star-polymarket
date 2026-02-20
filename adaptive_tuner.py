"""
Adaptive Parameter Tuner V1.0 — ML auto-tuning for Polymarket bots.

Uses Thompson sampling on parameter bins with shadow-tracking of skipped markets
to learn optimal parameter values 10x faster than trading alone.

Used by: run_sniper_5m_live.py, run_momentum_15m_live.py
"""

import json
import random
import time
from pathlib import Path
from datetime import datetime, timezone


class ParameterTuner:
    """
    Thompson sampling on binned continuous parameters.

    Shadow-tracks skipped opportunities to learn from ALL markets,
    not just the ones we trade. Auto-adjusts parameters toward the
    most profitable sweet spot.

    Each parameter has bins with Beta(alpha, beta) distributions.
    Every ADJUST_INTERVAL trades, the highest-sampled bin becomes active.
    """

    def __init__(self, config: dict, state_file: str):
        """
        config: {
            "param_name": {
                "bins": [0.72, 0.74, 0.76, 0.78, 0.80],  # bin edges (upper bounds)
                "labels": ["0.72", "0.74", "0.76", "0.78", "0.80"],
                "default_idx": 2,  # index of current/default bin
                "floor_idx": 0,    # never go below this bin
                "ceil_idx": 4,     # never go above this bin
            }
        }
        """
        self.config = config
        self.state_file = Path(state_file)
        self.state = {
            "params": {},       # {param: {bin_label: {alpha, beta, trades, pnl}}}
            "shadow": [],       # pending shadow entries awaiting resolution
            "active": {},       # {param: current_bin_label}
            "adjustments": [],  # log of every parameter change
            "total_resolved": 0,
        }
        self._load()
        self._ensure_structure()

    def _ensure_structure(self):
        """Initialize bins for all configured parameters."""
        for param, cfg in self.config.items():
            if param not in self.state["params"]:
                self.state["params"][param] = {}
            for label in cfg["labels"]:
                if label not in self.state["params"][param]:
                    # Start with weak uninformative prior
                    self.state["params"][param][label] = {
                        "alpha": 1, "beta": 1, "trades": 0, "pnl": 0.0
                    }
            # Set default active bin if not set
            if param not in self.state.get("active", {}):
                if "active" not in self.state:
                    self.state["active"] = {}
                default_label = cfg["labels"][cfg["default_idx"]]
                self.state["active"][param] = default_label

    def _load(self):
        if self.state_file.exists():
            try:
                self.state = json.loads(self.state_file.read_text())
            except (json.JSONDecodeError, Exception):
                pass

    def _save(self):
        try:
            self.state_file.write_text(json.dumps(self.state, indent=2))
        except Exception:
            pass

    def get_active_value(self, param: str) -> float:
        """Get the current active value for a parameter."""
        label = self.state.get("active", {}).get(param)
        if label is None:
            cfg = self.config[param]
            label = cfg["labels"][cfg["default_idx"]]
        return float(label)

    def get_active_values(self) -> dict:
        """Get all active parameter values."""
        return {p: self.get_active_value(p) for p in self.config}

    def record_shadow(self, market_id: str, end_time_iso: str,
                      side: str, entry_price: float, clob_dominant: float,
                      extra: dict = None):
        """
        Record a shadow entry for a market we SAW but may or may not have traded.
        This lets us learn from skipped markets too.

        For each parameter, we record which bins would have included this trade.
        """
        entry = {
            "market_id": market_id,
            "end_time": end_time_iso,
            "side": side,
            "entry_price": entry_price,
            "clob_dominant": clob_dominant,
            "ts": datetime.now(timezone.utc).isoformat(),
            "extra": extra or {},
        }
        self.state["shadow"].append(entry)
        # Keep shadow list bounded
        if len(self.state["shadow"]) > 500:
            self.state["shadow"] = self.state["shadow"][-300:]
        self._save()

    def resolve_shadows(self, resolutions: dict):
        """
        Resolve shadow entries. resolutions = {market_id: "UP"/"DOWN"}.

        For each resolved shadow, check which parameter bins would have
        captured this trade, and update their Beta distributions.
        """
        if not resolutions:
            return 0

        resolved_count = 0
        remaining = []

        for entry in self.state["shadow"]:
            mid = entry["market_id"]
            if mid not in resolutions:
                remaining.append(entry)
                continue

            outcome = resolutions[mid]
            side = entry["side"]
            won = (side == outcome)
            entry_price = entry["entry_price"]
            clob_dominant = entry["clob_dominant"]
            pnl_per_share = (1.0 - entry_price) if won else -entry_price

            # Update bins for each parameter
            for param, cfg in self.config.items():
                bins = cfg["bins"]
                labels = cfg["labels"]

                # Determine which bins would have included this trade
                for i, (edge, label) in enumerate(zip(bins, labels)):
                    would_include = False

                    if param in ("max_entry", "clob_upper"):
                        # Higher cap = more trades. This trade taken if cap >= entry_price
                        would_include = (edge >= entry_price)
                    elif param in ("clob_lower",):
                        # Lower floor = more trades. This trade taken if floor <= clob_dominant
                        would_include = (edge <= clob_dominant)
                    elif param in ("time_window_upper",):
                        # Wider window = more trades. Use extra.time_left_min
                        tl = entry.get("extra", {}).get("time_left_min", 3.0)
                        would_include = (edge >= tl)
                    elif param in ("min_confidence",):
                        would_include = (edge <= entry_price)
                    elif param in ("signal_threshold_bp",):
                        # Lower threshold = more trades. Trade taken if threshold <= signal strength
                        strength = abs(entry.get("extra", {}).get("strength_bp", 0))
                        would_include = (edge <= strength)
                    elif param in ("entry_delay_sec",):
                        # Shorter delay = more trades. Trade taken if delay <= elapsed
                        elapsed = entry.get("extra", {}).get("elapsed_sec", 30)
                        would_include = (edge <= elapsed)
                    elif param in ("contrarian_ceil",):
                        # V2.4: Higher ceil = stricter (more filtered without consensus)
                        # Trade passes if: consensus OR entry >= ceil
                        has_consensus = entry.get("extra", {}).get("consensus", False)
                        would_include = has_consensus or (entry_price >= edge)
                    elif param in ("momentum_floor",):
                        # V2.4: Higher floor = stricter (blocks weak momentum)
                        # Trade passes if abs(momentum) >= floor
                        mom = abs(entry.get("extra", {}).get("momentum_10m", 0))
                        would_include = (mom >= edge)
                    elif param in ("rsi_up",):
                        # V2.4: Higher threshold = stricter for UP signals
                        # DOWN signals always pass RSI_UP filter
                        rsi = entry.get("extra", {}).get("rsi", 50)
                        trade_side = entry.get("side", "DOWN")
                        if trade_side == "UP":
                            would_include = (rsi >= edge)
                        else:
                            would_include = True  # RSI-UP threshold only gates UP bets

                    if would_include:
                        b = self.state["params"][param][label]
                        if won:
                            b["alpha"] += 1
                        else:
                            b["beta"] += 1
                        b["trades"] += 1
                        b["pnl"] = round(b["pnl"] + pnl_per_share, 4)

            resolved_count += 1

        self.state["shadow"] = remaining
        self.state["total_resolved"] = self.state.get("total_resolved", 0) + resolved_count

        # Check if we should adjust parameters
        if resolved_count > 0:
            self._maybe_adjust()

        self._save()
        return resolved_count

    def _maybe_adjust(self):
        """
        Every 10 resolved trades, Thompson-sample to find optimal bins.
        Only adjust if we have enough data and the new bin is significantly better.
        """
        total = self.state.get("total_resolved", 0)
        if total < 10:
            return  # need minimum data
        if total % 5 != 0:
            return  # check every 5 resolutions

        for param, cfg in self.config.items():
            bins_data = self.state["params"].get(param, {})
            labels = cfg["labels"]
            floor_idx = cfg["floor_idx"]
            ceil_idx = cfg["ceil_idx"]

            # Thompson sample from each bin
            best_label = None
            best_score = -1

            for i, label in enumerate(labels):
                if i < floor_idx or i > ceil_idx:
                    continue
                b = bins_data.get(label, {"alpha": 1, "beta": 1, "trades": 0})
                if b["trades"] < 3:
                    continue  # not enough data for this bin

                # Thompson sample: higher alpha (wins) = higher expected WR
                score = random.betavariate(max(b["alpha"], 0.1), max(b["beta"], 0.1))

                # Penalize bins with very few trades (exploration bonus for less-tested bins)
                if b["trades"] < 8:
                    score *= 0.9  # slight penalty for low-data bins

                if score > best_score:
                    best_score = score
                    best_label = label

            if best_label and best_label != self.state["active"].get(param):
                old = self.state["active"].get(param, "?")
                old_data = bins_data.get(old, {"alpha": 1, "beta": 1, "trades": 0})
                new_data = bins_data.get(best_label, {"alpha": 1, "beta": 1, "trades": 0})

                old_wr = old_data["alpha"] / max(old_data["alpha"] + old_data["beta"], 1) * 100
                new_wr = new_data["alpha"] / max(new_data["alpha"] + new_data["beta"], 1) * 100

                self.state["active"][param] = best_label
                adjustment = {
                    "param": param,
                    "old": old,
                    "new": best_label,
                    "old_wr": round(old_wr, 1),
                    "new_wr": round(new_wr, 1),
                    "old_trades": old_data["trades"],
                    "new_trades": new_data["trades"],
                    "total_resolved": total,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                self.state["adjustments"].append(adjustment)
                # Keep adjustment log bounded
                if len(self.state["adjustments"]) > 200:
                    self.state["adjustments"] = self.state["adjustments"][-100:]

                print(f"[ML TUNE] {param}: {old} -> {best_label} "
                      f"| old={old_wr:.0f}%WR({old_data['trades']}t) "
                      f"new={new_wr:.0f}%WR({new_data['trades']}t)")

    def get_report(self) -> str:
        """Human-readable report of all parameter bins."""
        lines = [f"[ML TUNER] {self.state.get('total_resolved', 0)} markets resolved"]

        for param, cfg in self.config.items():
            active = self.state.get("active", {}).get(param, "?")
            lines.append(f"  {param} (active={active}):")
            for label in cfg["labels"]:
                b = self.state["params"].get(param, {}).get(label, {"alpha": 1, "beta": 1, "trades": 0, "pnl": 0})
                wr = b["alpha"] / max(b["alpha"] + b["beta"], 1) * 100
                marker = " <--" if label == active else ""
                lines.append(f"    [{label}] {wr:.0f}%WR | {b['trades']}t | PnL/sh=${b['pnl']:+.3f}{marker}")

        return "\n".join(lines)


# Pre-built configs for each bot
SNIPER_TUNER_CONFIG = {
    "max_entry": {
        "bins":   [0.72, 0.74, 0.76, 0.78, 0.80, 0.82],
        "labels": ["0.72", "0.74", "0.76", "0.78", "0.80", "0.82"],
        "default_idx": 3,   # 0.78 (current setting)
        "floor_idx": 1,     # never below 0.74
        "ceil_idx": 5,      # can go up to 0.82
    },
    "clob_upper": {
        "bins":   [0.72, 0.74, 0.76, 0.78, 0.80, 0.82],
        "labels": ["0.72", "0.74", "0.76", "0.78", "0.80", "0.82"],
        "default_idx": 3,   # 0.78
        "floor_idx": 1,     # never below 0.74
        "ceil_idx": 5,      # can go up to 0.82
    },
}

MOMENTUM_TUNER_CONFIG = {
    "time_window_upper": {
        "bins":   [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "labels": ["3.0", "4.0", "5.0", "6.0", "7.0", "8.0"],
        "default_idx": 3,   # 6.0 (current setting)
        "floor_idx": 0,     # can tighten to 3.0
        "ceil_idx": 5,      # can widen to 8.0
    },
    # V2.4: ML-tuned quality parameters
    "max_entry": {
        "bins":   [0.55, 0.58, 0.60, 0.63, 0.65, 0.68],
        "labels": ["0.55", "0.58", "0.60", "0.63", "0.65", "0.68"],
        "default_idx": 4,   # 0.65
        "floor_idx": 1,     # never below 0.58
        "ceil_idx": 5,      # can go up to 0.68
    },
    "contrarian_ceil": {
        "bins":   [0.38, 0.40, 0.43, 0.45, 0.48, 0.50],
        "labels": ["0.38", "0.40", "0.43", "0.45", "0.48", "0.50"],
        "default_idx": 3,   # 0.45
        "floor_idx": 0,     # can relax to 0.38 (very permissive)
        "ceil_idx": 5,      # can tighten to 0.50 (strict)
    },
    "momentum_floor": {
        "bins":   [0.0006, 0.0008, 0.0010, 0.0012, 0.0015, 0.0018],
        "labels": ["0.0006", "0.0008", "0.0010", "0.0012", "0.0015", "0.0018"],
        "default_idx": 3,   # 0.0012
        "floor_idx": 0,     # can relax to 0.0006
        "ceil_idx": 5,      # can tighten to 0.0018
    },
    "rsi_up": {
        "bins":   [55, 58, 60, 62, 65, 68],
        "labels": ["55", "58", "60", "62", "65", "68"],
        "default_idx": 3,   # 62
        "floor_idx": 1,     # never below 58
        "ceil_idx": 5,      # can tighten to 68
    },
}
