"""
Adaptive Parameter Tuner V2.1 — ML profit maximizer for Polymarket bots.

V2.1 UPGRADES (2026-02-21):
  - Brier Score calibration: measures prediction accuracy by confidence bucket
  - Detects overconfidence (paying too much) and underconfidence (leaving edge on table)
  - Rolling drift detection: last 20 trades vs overall, flags degradation
  - Direction-split Brier: separate scores for UP vs DOWN signals

V2.0 UPGRADES (2026-02-21):
  - Exponential recency weighting (decay_factor=0.95 per hour)
  - Profit-per-hour objective (rewards profitable volume, not just WR)
  - Adjustment interval reduced: every 3 resolved markets (was 5)
  - Gradient estimation: neighboring-bin awareness for faster convergence
  - Hour-of-day weighting: auto-skip negative EV hours
  - Optimal size multiplier: logistic regression on trade features
  - New tuner configs: binance_threshold, min_confidence, momentum_ceiling
  - Updated momentum max_entry for V4.2

Uses Thompson sampling on parameter bins with shadow-tracking of skipped markets
to learn optimal parameter values 10x faster than trading alone.

Used by: run_sniper_5m_live.py, run_momentum_15m_live.py
"""

import json
import math
import random
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict


class ParameterTuner:
    """
    Thompson sampling on binned continuous parameters.

    Shadow-tracks skipped opportunities to learn from ALL markets,
    not just the ones we trade. Auto-adjusts parameters toward the
    most profitable sweet spot.

    Each parameter has bins with Beta(alpha, beta) distributions.
    Every ADJUST_INTERVAL resolved markets, the highest-scored bin becomes active.

    V2.0: Recency weighting, profit-per-hour objective, gradient adjustment,
    hour-of-day tracking, and optimal sizing.
    """

    ADJUST_INTERVAL = 3   # V2.0: was 5, now 3 for faster convergence
    DECAY_FACTOR = 0.95   # V2.0: per-hour exponential decay for recency weighting

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
            "params": {},       # {param: {bin_label: {alpha, beta, trades, pnl, timestamps}}}
            "shadow": [],       # pending shadow entries awaiting resolution
            "active": {},       # {param: current_bin_label}
            "adjustments": [],  # log of every parameter change
            "total_resolved": 0,
            "hour_stats": {},   # V2.0: {hour_str: {wins, losses, pnl, trades}}
            "first_resolve_ts": None,  # V2.0: timestamp of first resolution (for profit/hr)
            "resolved_log": [],  # V2.0: list of {ts, pnl, hour, won, features} for sizing model
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
                        "alpha": 1, "beta": 1, "trades": 0, "pnl": 0.0,
                        "timestamps": [],  # V2.0: track when each update happened
                    }
                else:
                    # Ensure timestamps field exists for legacy state
                    if "timestamps" not in self.state["params"][param][label]:
                        self.state["params"][param][label]["timestamps"] = []
            # Set default active bin if not set
            if param not in self.state.get("active", {}):
                if "active" not in self.state:
                    self.state["active"] = {}
                default_label = cfg["labels"][cfg["default_idx"]]
                self.state["active"][param] = default_label

        # V2.0: Ensure hour_stats structure
        if "hour_stats" not in self.state:
            self.state["hour_stats"] = {}
        if "first_resolve_ts" not in self.state:
            self.state["first_resolve_ts"] = None
        if "resolved_log" not in self.state:
            self.state["resolved_log"] = []

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

    def _compute_recency_weight(self, ts_iso: str) -> float:
        """
        V2.0: Compute exponential recency weight for a timestamp.
        Returns decay^(age_in_hours). Recent trades get weight ~1.0,
        old trades decay toward 0.
        """
        try:
            ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_hours = max((now - ts).total_seconds() / 3600, 0)
            return self.DECAY_FACTOR ** age_hours
        except (ValueError, TypeError):
            return 0.5  # fallback for malformed timestamps

    def _compute_weighted_stats(self, bin_data: dict) -> dict:
        """
        V2.0: Compute recency-weighted stats for a bin.
        Returns {weighted_pnl, weighted_trades, weighted_alpha, weighted_beta}.
        """
        timestamps = bin_data.get("timestamps", [])
        trades = bin_data.get("trades", 0)

        if not timestamps or trades == 0:
            return {
                "weighted_pnl": bin_data.get("pnl", 0.0),
                "weighted_trades": trades,
                "weighted_alpha": bin_data.get("alpha", 1),
                "weighted_beta": bin_data.get("beta", 1),
            }

        # Each timestamp entry: {ts, won, pnl_contrib}
        w_pnl = 0.0
        w_trades = 0.0
        w_alpha = 1.0  # base prior
        w_beta = 1.0

        for entry in timestamps:
            w = self._compute_recency_weight(entry.get("ts", ""))
            w_pnl += entry.get("pnl_contrib", 0.0) * w
            w_trades += w
            if entry.get("won", False):
                w_alpha += w
            else:
                w_beta += w

        return {
            "weighted_pnl": round(w_pnl, 6),
            "weighted_trades": round(w_trades, 4),
            "weighted_alpha": w_alpha,
            "weighted_beta": w_beta,
        }

    def resolve_shadows(self, resolutions: dict):
        """
        Resolve shadow entries. resolutions = {market_id: "UP"/"DOWN"}.

        For each resolved shadow, check which parameter bins would have
        captured this trade, and update their Beta distributions.

        V2.0: Also tracks timestamps for recency weighting and hour stats.
        """
        if not resolutions:
            return 0

        resolved_count = 0
        remaining = []
        now_iso = datetime.now(timezone.utc).isoformat()
        now_hour = datetime.now(timezone.utc).hour

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

            # V2.0: Track hour stats
            hour_str = str(now_hour)
            if hour_str not in self.state["hour_stats"]:
                self.state["hour_stats"][hour_str] = {
                    "wins": 0, "losses": 0, "pnl": 0.0, "trades": 0
                }
            hs = self.state["hour_stats"][hour_str]
            hs["trades"] += 1
            hs["pnl"] = round(hs["pnl"] + pnl_per_share, 4)
            if won:
                hs["wins"] += 1
            else:
                hs["losses"] += 1

            # V2.0: Track first resolve timestamp for profit/hr calculation
            if self.state["first_resolve_ts"] is None:
                self.state["first_resolve_ts"] = now_iso

            # V2.0: Add to resolved_log for sizing model
            log_entry = {
                "ts": now_iso,
                "pnl": round(pnl_per_share, 4),
                "hour": now_hour,
                "won": won,
                "entry_price": entry_price,
                "side": side,
                "momentum": entry.get("extra", {}).get("momentum_10m", 0),
                "rsi": entry.get("extra", {}).get("rsi", 50),
            }
            self.state["resolved_log"].append(log_entry)
            # Bound resolved_log to last 500 entries
            if len(self.state["resolved_log"]) > 500:
                self.state["resolved_log"] = self.state["resolved_log"][-400:]

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
                    elif param in ("binance_threshold",):
                        # Lower threshold = more sensitive to Binance moves
                        strength = abs(entry.get("extra", {}).get("binance_move", 0))
                        would_include = (edge <= strength)
                    elif param in ("signal_threshold_bp",):
                        # Lower threshold = more trades. Trade taken if threshold <= signal strength
                        strength = abs(entry.get("extra", {}).get("strength_bp", 0))
                        would_include = (edge <= strength)
                    elif param in ("entry_delay_sec",):
                        # Shorter delay = more trades. Trade taken if delay <= elapsed
                        elapsed = entry.get("extra", {}).get("elapsed_sec", 30)
                        would_include = (edge <= elapsed)
                    elif param in ("contrarian_ceil",):
                        # Higher ceil = stricter (more filtered without consensus)
                        has_consensus = entry.get("extra", {}).get("consensus", False)
                        would_include = has_consensus or (entry_price >= edge)
                    elif param in ("momentum_floor",):
                        # Higher floor = stricter (blocks weak momentum)
                        mom = abs(entry.get("extra", {}).get("momentum_10m", 0))
                        would_include = (mom >= edge)
                    elif param in ("momentum_ceiling",):
                        # Lower ceiling = stricter (blocks excessive momentum / overextended)
                        mom = abs(entry.get("extra", {}).get("momentum_10m", 0))
                        would_include = (mom <= edge)
                    elif param in ("rsi_up",):
                        # Higher threshold = stricter for UP signals
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
                        # V2.0: Track timestamp for recency weighting
                        b["timestamps"].append({
                            "ts": now_iso,
                            "won": won,
                            "pnl_contrib": round(pnl_per_share, 4),
                        })
                        # Bound timestamps to last 200 per bin
                        if len(b["timestamps"]) > 200:
                            b["timestamps"] = b["timestamps"][-150:]

            resolved_count += 1

        self.state["shadow"] = remaining
        self.state["total_resolved"] = self.state.get("total_resolved", 0) + resolved_count

        # Check if we should adjust parameters
        if resolved_count > 0:
            self._maybe_adjust()

        self._save()
        return resolved_count

    def _gradient_adjust(self, param: str, best_idx: int, labels: list,
                         bins_data: dict, floor_idx: int, ceil_idx: int,
                         scores: dict) -> int:
        """
        V2.0: Gradient estimation for faster convergence.

        For the best bin, check if neighboring bins score higher.
        If a neighbor in a consistent direction scores better, nudge toward it.
        Returns the adjusted best index.
        """
        if len(scores) < 2:
            return best_idx

        # Check left and right neighbors
        left_idx = best_idx - 1 if best_idx > floor_idx else None
        right_idx = best_idx + 1 if best_idx < ceil_idx else None

        best_score = scores.get(labels[best_idx], -999)

        # Compute gradient: positive = move right, negative = move left
        gradient = 0.0

        if left_idx is not None and labels[left_idx] in scores:
            left_score = scores[labels[left_idx]]
            gradient -= (best_score - left_score)  # negative grad if left is worse

        if right_idx is not None and labels[right_idx] in scores:
            right_score = scores[labels[right_idx]]
            gradient += (right_score - best_score)  # positive grad if right is better

        # If gradient strongly points to a neighbor, move one step
        GRADIENT_THRESHOLD = 0.02  # need meaningful difference to nudge

        if gradient > GRADIENT_THRESHOLD and right_idx is not None:
            right_label = labels[right_idx]
            right_data = bins_data.get(right_label, {"trades": 0})
            if right_data.get("trades", 0) >= 2:
                return right_idx

        if gradient < -GRADIENT_THRESHOLD and left_idx is not None:
            left_label = labels[left_idx]
            left_data = bins_data.get(left_label, {"trades": 0})
            if left_data.get("trades", 0) >= 2:
                return left_idx

        return best_idx

    def _maybe_adjust(self):
        """
        Every ADJUST_INTERVAL resolved markets, find the optimal bin for each
        parameter.

        V2.0 OBJECTIVE: Maximize PROFIT PER HOUR.
          - Uses recency-weighted stats so recent data dominates
          - Scores bins by weighted_pnl / hours_of_data (profit rate)
          - Applies gradient estimation for faster convergence
          - Exploration bonus (UCB1-style) ensures under-tested bins get tried

        This makes a bin with 70% WR / 8 trades / +$6.40 beat 85% WR / 2 trades / +$3.00.
        """
        total = self.state.get("total_resolved", 0)
        if total < 6:
            return  # need minimum data
        if total % self.ADJUST_INTERVAL != 0:
            return  # V2.0: check every 3 resolutions (was 5)

        # V2.0: Compute hours of data for profit-per-hour scoring
        hours_of_data = 1.0  # minimum 1 hour to avoid division by zero
        first_ts = self.state.get("first_resolve_ts")
        if first_ts:
            try:
                first_dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                hours_of_data = max((now - first_dt).total_seconds() / 3600, 1.0)
            except (ValueError, TypeError):
                pass

        for param, cfg in self.config.items():
            bins_data = self.state["params"].get(param, {})
            labels = cfg["labels"]
            floor_idx = cfg["floor_idx"]
            ceil_idx = cfg["ceil_idx"]

            best_label = None
            best_score = -999
            best_idx = -1
            scores = {}  # V2.0: track all scores for gradient

            for i, label in enumerate(labels):
                if i < floor_idx or i > ceil_idx:
                    continue
                b = bins_data.get(label, {"alpha": 1, "beta": 1, "trades": 0, "pnl": 0,
                                          "timestamps": []})
                if b["trades"] < 3:
                    continue  # not enough data for this bin

                # V2.0: Use recency-weighted stats
                ws = self._compute_weighted_stats(b)

                # V2.0: PROFIT-PER-HOUR SCORING
                # weighted_pnl captures recency; divide by hours for rate
                profit_rate = ws["weighted_pnl"] / hours_of_data

                # Also factor in trade density (more trades = more confidence)
                # weighted_trades / hours = trade frequency
                trade_freq = ws["weighted_trades"] / hours_of_data

                # Combined: profit_rate with a small bonus for higher frequency
                # (all else equal, prefer bins that generate more opportunities)
                score = profit_rate + 0.01 * trade_freq

                # UCB1 exploration bonus - encourages trying under-tested bins
                # sqrt(2 * ln(total_resolved) / weighted_trades)
                if ws["weighted_trades"] > 0:
                    exploration = math.sqrt(
                        2 * math.log(max(total, 1)) / max(ws["weighted_trades"], 0.5)
                    )
                else:
                    exploration = 2.0  # high exploration for untested
                score += 0.10 * exploration

                # Thompson noise for stochastic exploration (prevents getting stuck)
                wr_sample = random.betavariate(
                    max(ws["weighted_alpha"], 0.1),
                    max(ws["weighted_beta"], 0.1)
                )
                score += 0.05 * (wr_sample - 0.5)

                scores[label] = score

                if score > best_score:
                    best_score = score
                    best_label = label
                    best_idx = i

            # V2.0: Apply gradient adjustment
            if best_idx >= 0 and len(scores) >= 2:
                adjusted_idx = self._gradient_adjust(
                    param, best_idx, labels, bins_data, floor_idx, ceil_idx, scores
                )
                if adjusted_idx != best_idx:
                    best_label = labels[adjusted_idx]
                    best_idx = adjusted_idx

            if best_label and best_label != self.state["active"].get(param):
                old = self.state["active"].get(param, "?")
                old_data = bins_data.get(old, {"alpha": 1, "beta": 1, "trades": 0,
                                               "pnl": 0, "timestamps": []})
                new_data = bins_data.get(best_label, {"alpha": 1, "beta": 1, "trades": 0,
                                                      "pnl": 0, "timestamps": []})

                old_wr = old_data["alpha"] / max(old_data["alpha"] + old_data["beta"], 1) * 100
                new_wr = new_data["alpha"] / max(new_data["alpha"] + new_data["beta"], 1) * 100
                old_pnl_avg = old_data["pnl"] / max(old_data["trades"], 1)
                new_pnl_avg = new_data["pnl"] / max(new_data["trades"], 1)

                # V2.0: Also log recency-weighted scores
                old_ws = self._compute_weighted_stats(old_data)
                new_ws = self._compute_weighted_stats(new_data)

                self.state["active"][param] = best_label
                adjustment = {
                    "param": param,
                    "old": old,
                    "new": best_label,
                    "old_wr": round(old_wr, 1),
                    "new_wr": round(new_wr, 1),
                    "old_pnl": round(old_data["pnl"], 3),
                    "new_pnl": round(new_data["pnl"], 3),
                    "old_trades": old_data["trades"],
                    "new_trades": new_data["trades"],
                    "old_weighted_pnl": round(old_ws["weighted_pnl"], 4),
                    "new_weighted_pnl": round(new_ws["weighted_pnl"], 4),
                    "hours_of_data": round(hours_of_data, 2),
                    "total_resolved": total,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                self.state["adjustments"].append(adjustment)
                # Keep adjustment log bounded
                if len(self.state["adjustments"]) > 200:
                    self.state["adjustments"] = self.state["adjustments"][-100:]

                print(f"[ML TUNE V2] {param}: {old} -> {best_label} "
                      f"| old={old_wr:.0f}%WR {old_data['trades']}t ${old_data['pnl']:+.2f} "
                      f"(${old_pnl_avg:+.3f}/t, w${old_ws['weighted_pnl']:+.3f}) "
                      f"| new={new_wr:.0f}%WR {new_data['trades']}t ${new_data['pnl']:+.2f} "
                      f"(${new_pnl_avg:+.3f}/t, w${new_ws['weighted_pnl']:+.3f}) "
                      f"| {hours_of_data:.1f}h data")

    def get_hourly_stats(self) -> dict:
        """
        V2.0: Returns hour-of-day stats from resolved trade data.

        Returns: {
            hour_int: {
                "wr": float (0-100),
                "pnl": float,
                "trades": int,
                "ev": float (expected pnl per trade),
                "skip": bool (True if negative EV)
            }
        }

        Bots can call this to dynamically skip or boost certain hours.
        Example: if get_hourly_stats()[7]["skip"] is True, skip that hour.
        """
        result = {}
        for hour in range(24):
            hour_str = str(hour)
            hs = self.state.get("hour_stats", {}).get(hour_str)
            if hs and hs.get("trades", 0) > 0:
                wr = (hs["wins"] / hs["trades"]) * 100 if hs["trades"] > 0 else 0
                ev = hs["pnl"] / hs["trades"] if hs["trades"] > 0 else 0
                result[hour] = {
                    "wr": round(wr, 1),
                    "pnl": round(hs["pnl"], 4),
                    "trades": hs["trades"],
                    "ev": round(ev, 4),
                    "skip": (ev < 0 and hs["trades"] >= 5),  # need 5+ trades to conclude
                }
            else:
                result[hour] = {
                    "wr": 0, "pnl": 0.0, "trades": 0, "ev": 0.0, "skip": False,
                }
        return result

    def get_negative_ev_hours(self) -> list:
        """
        V2.0: Convenience method — returns list of hours that should be skipped.
        Only marks hours with 5+ trades and negative average PnL.
        """
        stats = self.get_hourly_stats()
        return [h for h, s in stats.items() if s["skip"]]

    def get_brier_score(self, last_n: int = 0) -> dict:
        """
        V2.1: Brier Score calibration — measures how well our confidence
        (entry price) predicts outcomes.

        Brier Score = mean((predicted_prob - outcome)^2)
          0.00 = perfect | 0.10 = excellent | 0.20 = good | 0.25 = coin flip

        Also returns calibration buckets: predicted confidence vs actual WR.
        Detects overconfidence (we pay $0.90 but only win 70%) and
        underconfidence (we pay $0.75 but win 95%).
        """
        resolved = self.state.get("resolved_log", [])
        if last_n > 0:
            resolved = resolved[-last_n:]
        if len(resolved) < 10:
            return {"score": None, "rating": "insufficient_data", "trades": len(resolved)}

        bucket_ranges = [
            (0.50, 0.65, "0.50-0.65"), (0.65, 0.75, "0.65-0.75"),
            (0.75, 0.85, "0.75-0.85"), (0.85, 0.95, "0.85-0.95"),
            (0.95, 1.01, "0.95-1.00"),
        ]
        buckets = {label: {"pred_sum": 0.0, "wins": 0, "count": 0}
                   for _, _, label in bucket_ranges}

        brier_sum = 0.0
        up_sq, down_sq = [], []

        for r in resolved:
            pred = r.get("entry_price", 0.5)
            outcome = 1 if r.get("won", False) else 0
            sq_err = (pred - outcome) ** 2
            brier_sum += sq_err

            side = r.get("side", "")
            if side == "UP":
                up_sq.append(sq_err)
            elif side == "DOWN":
                down_sq.append(sq_err)

            for b_lo, b_hi, b_label in bucket_ranges:
                if b_lo <= pred < b_hi:
                    buckets[b_label]["pred_sum"] += pred
                    buckets[b_label]["wins"] += outcome
                    buckets[b_label]["count"] += 1
                    break

        brier = brier_sum / len(resolved)
        rating = ("excellent" if brier < 0.10 else "good" if brier < 0.20
                  else "fair" if brier < 0.25 else "poor")

        # Calibration per bucket
        calibration = {}
        for b_label, bd in buckets.items():
            if bd["count"] >= 3:
                avg_pred = bd["pred_sum"] / bd["count"]
                actual_wr = bd["wins"] / bd["count"]
                gap = actual_wr - avg_pred
                calibration[b_label] = {
                    "avg_predicted": round(avg_pred, 3),
                    "actual_wr": round(actual_wr, 3),
                    "gap": round(gap, 3),
                    "trades": bd["count"],
                    "status": ("OVERCONFIDENT" if gap < -0.10
                               else "UNDERCONFIDENT" if gap > 0.10
                               else "calibrated"),
                }

        result = {
            "score": round(brier, 4),
            "rating": rating,
            "trades": len(resolved),
            "calibration": calibration,
        }
        if up_sq:
            result["up_brier"] = round(sum(up_sq) / len(up_sq), 4)
        if down_sq:
            result["down_brier"] = round(sum(down_sq) / len(down_sq), 4)

        # Rolling drift: last 20 vs overall
        if len(resolved) >= 40:
            recent = resolved[-20:]
            recent_brier = sum(
                (r.get("entry_price", 0.5) - (1 if r.get("won", False) else 0)) ** 2
                for r in recent
            ) / 20
            result["recent_brier"] = round(recent_brier, 4)
            result["drift"] = round(recent_brier - brier, 4)

        return result

    def get_optimal_size_mult(self, hour: int = None, direction: str = None,
                              entry_price: float = None, momentum: float = None,
                              rsi: float = None) -> float:
        """
        V2.0: Returns a size multiplier (0.5x to 2.0x) based on conditional
        probability model using resolved trade history.

        Uses simple logistic-regression-style scoring on trade features:
          - Hour-of-day edge
          - Direction edge (UP vs DOWN)
          - Entry price edge (cheaper = better historically)
          - Momentum strength
          - RSI alignment

        Each factor contributes a score component; combined into a sigmoid
        mapped to [0.5, 2.0]. Requires at least 20 resolved trades to activate;
        otherwise returns 1.0 (neutral).

        Args:
            hour: UTC hour (0-23). Defaults to current hour.
            direction: "UP" or "DOWN"
            entry_price: Entry price (0-1)
            momentum: Absolute momentum value
            rsi: RSI value (0-100)

        Returns:
            float: Size multiplier in [0.5, 2.0]
        """
        resolved = self.state.get("resolved_log", [])
        if len(resolved) < 20:
            return 1.0  # not enough data

        if hour is None:
            hour = datetime.now(timezone.utc).hour

        # ---- Compute conditional edges from resolved history ----
        score = 0.0
        factors = 0

        # 1. Hour edge: WR at this hour vs overall WR
        hour_stats = self.get_hourly_stats()
        overall_wins = sum(1 for r in resolved if r.get("won", False))
        overall_wr = overall_wins / len(resolved) if resolved else 0.5
        if hour in hour_stats and hour_stats[hour]["trades"] >= 5:
            hour_wr = hour_stats[hour]["wr"] / 100
            hour_edge = hour_wr - overall_wr
            score += hour_edge * 2.0  # weight: 2x
            factors += 1

        # 2. Direction edge: UP vs DOWN historical WR
        if direction:
            dir_trades = [r for r in resolved if r.get("side") == direction]
            if len(dir_trades) >= 5:
                dir_wins = sum(1 for r in dir_trades if r.get("won", False))
                dir_wr = dir_wins / len(dir_trades)
                dir_edge = dir_wr - overall_wr
                score += dir_edge * 1.5  # weight: 1.5x
                factors += 1

        # 3. Entry price edge: bin by price ranges
        if entry_price is not None:
            # Bin: <0.40, 0.40-0.55, 0.55-0.70, >0.70
            bins_ranges = [(0, 0.40), (0.40, 0.55), (0.55, 0.70), (0.70, 1.0)]
            for lo, hi in bins_ranges:
                if lo <= entry_price < hi:
                    bin_trades = [r for r in resolved
                                  if lo <= r.get("entry_price", 0.5) < hi]
                    if len(bin_trades) >= 3:
                        bin_wins = sum(1 for r in bin_trades if r.get("won", False))
                        bin_wr = bin_wins / len(bin_trades)
                        price_edge = bin_wr - overall_wr
                        score += price_edge * 1.5
                        factors += 1
                    break

        # 4. Momentum strength edge
        if momentum is not None:
            abs_mom = abs(momentum)
            # Strong momentum historically better?
            strong_trades = [r for r in resolved
                             if abs(r.get("momentum", 0)) >= abs_mom * 0.8]
            if len(strong_trades) >= 5:
                strong_wins = sum(1 for r in strong_trades if r.get("won", False))
                strong_wr = strong_wins / len(strong_trades)
                mom_edge = strong_wr - overall_wr
                score += mom_edge * 1.0
                factors += 1

        # 5. RSI alignment edge
        if rsi is not None and direction:
            # For UP: RSI > 50 is aligned. For DOWN: RSI < 50 is aligned.
            if direction == "UP":
                aligned = rsi >= 50
            else:
                aligned = rsi <= 50
            aligned_trades = [r for r in resolved if r.get("rsi") is not None]
            if len(aligned_trades) >= 5:
                # Check if aligned trades win more
                a_wins = 0
                a_total = 0
                u_wins = 0
                u_total = 0
                for r in aligned_trades:
                    r_rsi = r.get("rsi", 50)
                    r_side = r.get("side", "UP")
                    is_aligned = (r_side == "UP" and r_rsi >= 50) or \
                                 (r_side == "DOWN" and r_rsi <= 50)
                    if is_aligned:
                        a_total += 1
                        if r.get("won", False):
                            a_wins += 1
                    else:
                        u_total += 1
                        if r.get("won", False):
                            u_wins += 1
                if a_total >= 3 and u_total >= 3:
                    a_wr = a_wins / a_total
                    u_wr = u_wins / u_total
                    rsi_edge = (a_wr - u_wr) * (1 if aligned else -1)
                    score += rsi_edge * 1.0
                    factors += 1

        # Normalize score if we had factors
        if factors > 0:
            score = score / factors

        # Sigmoid mapping: score in roughly [-1, 1] -> multiplier in [0.5, 2.0]
        # sigmoid(score * 3) maps to [0, 1], then scale to [0.5, 2.0]
        try:
            sig = 1.0 / (1.0 + math.exp(-score * 3.0))
        except OverflowError:
            sig = 0.0 if score < 0 else 1.0

        multiplier = 0.5 + sig * 1.5  # maps [0,1] -> [0.5, 2.0]

        # Clamp just in case
        return round(max(0.5, min(2.0, multiplier)), 3)

    def get_report(self) -> str:
        """Human-readable report of all parameter bins."""
        lines = [f"[ML TUNER V2.0] {self.state.get('total_resolved', 0)} markets resolved "
                 f"| Objective: MAX PROFIT/HOUR"]

        # V2.0: Show profit-per-hour
        first_ts = self.state.get("first_resolve_ts")
        if first_ts:
            try:
                first_dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                hours = max((datetime.now(timezone.utc) - first_dt).total_seconds() / 3600, 0.01)
                total_pnl = sum(r.get("pnl", 0) for r in self.state.get("resolved_log", []))
                pph = total_pnl / hours
                lines.append(f"  Profit rate: ${pph:+.4f}/hr over {hours:.1f}h "
                             f"(total ${total_pnl:+.3f} from {len(self.state.get('resolved_log', []))} resolutions)")
            except (ValueError, TypeError):
                pass

        for param, cfg in self.config.items():
            active = self.state.get("active", {}).get(param, "?")
            lines.append(f"  {param} (active={active}):")
            for label in cfg["labels"]:
                b = self.state["params"].get(param, {}).get(
                    label, {"alpha": 1, "beta": 1, "trades": 0, "pnl": 0, "timestamps": []})
                wr = b["alpha"] / max(b["alpha"] + b["beta"], 1) * 100
                avg_pnl = b["pnl"] / max(b["trades"], 1)
                ws = self._compute_weighted_stats(b)
                marker = " <-- ACTIVE" if label == active else ""
                lines.append(f"    [{label}] {wr:.0f}%WR | {b['trades']}t | "
                             f"total=${b['pnl']:+.3f} avg=${avg_pnl:+.3f}/t "
                             f"w${ws['weighted_pnl']:+.3f}{marker}")

        # V2.0: Show hour stats summary
        neg_hours = self.get_negative_ev_hours()
        if neg_hours:
            lines.append(f"  SKIP HOURS (UTC): {sorted(neg_hours)} (neg EV, 5+ trades)")

        # V2.1: Brier Score calibration
        brier = self.get_brier_score()
        if brier.get("score") is not None:
            drift_str = ""
            if "drift" in brier:
                arrow = "^" if brier["drift"] > 0.02 else "v" if brier["drift"] < -0.02 else "="
                drift_str = f" | recent={brier['recent_brier']:.3f} {arrow}"
            lines.append(f"  BRIER: {brier['score']:.3f} ({brier['rating']}){drift_str}")
            if brier.get("up_brier") is not None and brier.get("down_brier") is not None:
                lines.append(f"    UP={brier['up_brier']:.3f} | DOWN={brier['down_brier']:.3f}")
            for bucket, cal in brier.get("calibration", {}).items():
                flag = ""
                if cal["status"] == "OVERCONFIDENT":
                    flag = " *** OVERCONFIDENT"
                elif cal["status"] == "UNDERCONFIDENT":
                    flag = " ** underconfident"
                lines.append(f"    {bucket}: pred={cal['avg_predicted']:.0%} "
                             f"actual={cal['actual_wr']:.0%} gap={cal['gap']:+.0%} "
                             f"({cal['trades']}t){flag}")

        return "\n".join(lines)


# Pre-built configs for each bot
SNIPER_TUNER_CONFIG = {
    "max_entry": {
        "bins":   [0.76, 0.78, 0.80, 0.82, 0.84],
        "labels": ["0.76", "0.78", "0.80", "0.82", "0.84"],
        "default_idx": 2,   # V3.2: 0.80 — 0.74 removed (below MIN_CONFIDENCE=0.75, caused 3hr halt)
        "floor_idx": 0,     # never below 0.76 (must be >= MIN_CONFIDENCE $0.75)
        "ceil_idx": 4,      # can go up to 0.84
    },
    # V3.1: clob_upper no longer used (CLOB fallback now uses MAX_ENTRY_PRICE directly)
    "clob_upper": {
        "bins":   [0.76, 0.78, 0.80, 0.82, 0.84],
        "labels": ["0.76", "0.78", "0.80", "0.82", "0.84"],
        "default_idx": 2,   # V3.2: 0.80 — 0.74 removed
        "floor_idx": 0,     # never below 0.76
        "ceil_idx": 4,      # can go up to 0.84
    },
    # V2.0: Binance price move threshold for sniper entries
    "binance_threshold": {
        "bins": [0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        "labels": ["0.02", "0.03", "0.04", "0.05", "0.06", "0.07"],
        "default_idx": 2,   # 0.04 (current V3.1 setting)
        "floor_idx": 0,
        "ceil_idx": 5,
    },
    # V2.0: Minimum confidence threshold for sniper entries
    "min_confidence": {
        "bins": [0.55, 0.60, 0.65, 0.70, 0.72, 0.75],
        "labels": ["0.55", "0.60", "0.65", "0.70", "0.72", "0.75"],
        "default_idx": 4,   # 0.72 -- paper data: $0.80 entries = 96.8% WR
        "floor_idx": 0,
        "ceil_idx": 5,
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
    # V4.2: Updated max_entry bins and default
    "max_entry": {
        "bins": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        "labels": ["0.50", "0.55", "0.60", "0.65", "0.70", "0.75"],
        "default_idx": 3,   # 0.65 (V4.2 setting)
        "floor_idx": 0,
        "ceil_idx": 5,
    },
    "contrarian_ceil": {
        "bins":   [0.38, 0.40, 0.43, 0.45, 0.48, 0.50],
        "labels": ["0.38", "0.40", "0.43", "0.45", "0.48", "0.50"],
        "default_idx": 3,   # 0.45
        "floor_idx": 0,     # can relax to 0.38 (very permissive)
        "ceil_idx": 5,      # can tighten to 0.50 (strict)
    },
    "momentum_floor": {
        "bins":   [0.0004, 0.0006, 0.0008, 0.0010, 0.0012, 0.0015],
        "labels": ["0.0004", "0.0006", "0.0008", "0.0010", "0.0012", "0.0015"],
        "default_idx": 2,   # 0.0008 -- lowered from 0.0012, was blocking too many signals
        "floor_idx": 0,     # can relax to 0.0004
        "ceil_idx": 5,      # can tighten to 0.0015
    },
    "rsi_up": {
        "bins":   [50, 53, 55, 58, 60, 63],
        "labels": ["50", "53", "55", "58", "60", "63"],
        "default_idx": 2,   # 55 -- lowered from 62 -- BTC RSI neutral most of the time
        "floor_idx": 0,     # can relax to 50 (no RSI filter)
        "ceil_idx": 5,      # can tighten to 63
    },
    # V2.0: Momentum ceiling -- blocks overextended entries (too much momentum = reversal risk)
    "momentum_ceiling": {
        "bins": [0.0008, 0.0010, 0.0012, 0.0015, 0.0020, 0.0030],
        "labels": ["0.0008", "0.0010", "0.0012", "0.0015", "0.0020", "0.0030"],
        "default_idx": 2,   # 0.0012 (V4.2 setting)
        "floor_idx": 0,
        "ceil_idx": 5,
    },
}
