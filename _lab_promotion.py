"""
15M Strategy Lab — Promotion Analysis
Filters to post-baseline trades (>= 2026-02-17T00:00:00Z), computes per-strategy:
  - Win rate, total trades, total PnL
  - BCa Bootstrap 95% CI for win rate (2000 resamples)
  - Walk-forward: 3 sequential time windows, each profitable?
  - Average entry price, average PnL per trade
  - Profit factor (gross wins / gross losses)
  - Promotion verdict: PROMOTE / WATCH / SKIP
"""

import json
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

# ── Load & filter ──────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\Star\.local\bin\star-polymarket\fifteen_min_strategies_results.json"
BASELINE_CUTOFF = datetime(2026, 2, 17, 0, 0, 0, tzinfo=timezone.utc)

with open(DATA_PATH, "r") as f:
    data = json.load(f)

resolved = data.get("resolved", [])
print(f"Total resolved trades in file: {len(resolved)}")

# Parse entry_time and filter post-baseline
trades = []
for t in resolved:
    if t.get("status") != "closed":
        continue
    entry_str = t.get("entry_time", "")
    try:
        # Handle both +00:00 and Z suffixes
        entry_dt = datetime.fromisoformat(entry_str.replace("Z", "+00:00"))
    except Exception:
        continue
    if entry_dt >= BASELINE_CUTOFF:
        t["_entry_dt"] = entry_dt
        trades.append(t)

print(f"Post-baseline trades (>= 2026-02-17): {len(trades)}")
if not trades:
    print("No post-baseline trades found. Nothing to analyze.")
    raise SystemExit(0)

# ── Group by strategy ──────────────────────────────────────────────────────
by_strat = defaultdict(list)
for t in trades:
    by_strat[t["strategy"]].append(t)

# Sort each strategy's trades by entry time
for s in by_strat:
    by_strat[s].sort(key=lambda t: t["_entry_dt"])

# ── BCa Bootstrap 95% CI ──────────────────────────────────────────────────
def bca_bootstrap_ci(outcomes, n_resamples=2000, alpha=0.05):
    """
    BCa (Bias-Corrected and Accelerated) Bootstrap 95% CI for win rate.
    outcomes: list/array of 1 (win) / 0 (loss)
    Returns (lower, upper) bounds.
    """
    outcomes = np.array(outcomes, dtype=float)
    n = len(outcomes)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (outcomes[0], outcomes[0])

    theta_hat = np.mean(outcomes)

    # Bootstrap resamples
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(outcomes, size=n, replace=True))
        for _ in range(n_resamples)
    ])

    # Bias correction factor z0
    from scipy.stats import norm
    prop_below = np.mean(boot_means < theta_hat)
    # Clamp to avoid inf
    prop_below = np.clip(prop_below, 1e-10, 1 - 1e-10)
    z0 = norm.ppf(prop_below)

    # Acceleration factor a (jackknife)
    jackknife_means = np.array([
        (np.sum(outcomes) - outcomes[i]) / (n - 1)
        for i in range(n)
    ])
    jk_mean = np.mean(jackknife_means)
    diff = jk_mean - jackknife_means
    a = np.sum(diff ** 3) / (6.0 * (np.sum(diff ** 2) ** 1.5 + 1e-30))

    # Adjusted percentiles
    z_alpha = norm.ppf(alpha / 2)
    z_1alpha = norm.ppf(1 - alpha / 2)

    def adjusted_pct(z_val):
        num = z0 + z_val
        denom = 1 - a * num
        if abs(denom) < 1e-10:
            denom = 1e-10
        return norm.cdf(z0 + num / denom)

    pct_lo = adjusted_pct(z_alpha)
    pct_hi = adjusted_pct(z_1alpha)

    # Clamp
    pct_lo = np.clip(pct_lo, 0, 1)
    pct_hi = np.clip(pct_hi, 0, 1)

    lower = np.percentile(boot_means, pct_lo * 100)
    upper = np.percentile(boot_means, pct_hi * 100)

    return (float(lower), float(upper))


# ── Walk-forward (3 windows) ──────────────────────────────────────────────
def walk_forward_grade(strat_trades):
    """
    Split trades into 3 sequential time windows.
    Returns (windows_profitable: int out of 3, grade: str)
    """
    n = len(strat_trades)
    if n < 3:
        return (0, "WEAK")

    chunk_size = n // 3
    windows = [
        strat_trades[:chunk_size],
        strat_trades[chunk_size:2 * chunk_size],
        strat_trades[2 * chunk_size:],
    ]

    profitable_windows = 0
    for w in windows:
        w_pnl = sum(t["pnl"] for t in w)
        if w_pnl > 0:
            profitable_windows += 1

    if profitable_windows >= 3:
        grade = "ROBUST"
    elif profitable_windows == 2:
        grade = "OK"
    else:
        grade = "WEAK"

    return (profitable_windows, grade)


# ── Compute metrics ───────────────────────────────────────────────────────
results = []

for strat_name in sorted(by_strat.keys()):
    strat_trades = by_strat[strat_name]
    n_trades = len(strat_trades)

    wins = [t for t in strat_trades if t["pnl"] > 0]
    losses = [t for t in strat_trades if t["pnl"] <= 0]
    n_wins = len(wins)
    n_losses = len(losses)
    wr = n_wins / n_trades if n_trades > 0 else 0

    total_pnl = sum(t["pnl"] for t in strat_trades)
    avg_pnl = total_pnl / n_trades if n_trades > 0 else 0
    avg_entry = np.mean([t["entry_price"] for t in strat_trades]) if n_trades > 0 else 0

    gross_wins = sum(t["pnl"] for t in wins) if wins else 0
    gross_losses = abs(sum(t["pnl"] for t in losses)) if losses else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # BCa bootstrap
    outcomes = [1 if t["pnl"] > 0 else 0 for t in strat_trades]
    ci_lo, ci_hi = bca_bootstrap_ci(outcomes)

    # Walk-forward
    wf_profitable, wf_grade = walk_forward_grade(strat_trades)

    # Verdict
    if n_trades >= 20 and ci_lo > 0.50 and wf_grade in ("ROBUST", "OK") and total_pnl > 0:
        verdict = "PROMOTE"
    elif 10 <= n_trades <= 19 and wr > 0.60:
        verdict = "WATCH"
    elif n_trades >= 20 and wr > 0.60 and total_pnl > 0:
        # Borderline — passes most criteria but CI or walk-forward failed
        verdict = "WATCH"
    else:
        verdict = "SKIP"

    results.append({
        "strategy": strat_name,
        "n_trades": n_trades,
        "wins": n_wins,
        "losses": n_losses,
        "wr": wr,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_pass": ci_lo > 0.50,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "avg_entry": avg_entry,
        "profit_factor": profit_factor,
        "wf_profitable": wf_profitable,
        "wf_grade": wf_grade,
        "verdict": verdict,
    })

# ── Print table ───────────────────────────────────────────────────────────
print()
print("=" * 120)
print("  15M STRATEGY LAB  --  PROMOTION ANALYSIS  (Post-Baseline >= 2026-02-17)")
print("=" * 120)

# Header
hdr = (
    f"{'Strategy':<20} {'Trades':>6} {'W':>4} {'L':>4} "
    f"{'WR%':>6} {'CI_Lo':>6} {'CI_Hi':>6} {'CI>50':>6} "
    f"{'PnL':>9} {'Avg PnL':>8} {'Avg Entry':>9} {'PF':>7} "
    f"{'WF':>5} {'WF Grade':>8} {'Verdict':>8}"
)
print(hdr)
print("-" * 120)

for r in results:
    pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "INF"
    line = (
        f"{r['strategy']:<20} {r['n_trades']:>6} {r['wins']:>4} {r['losses']:>4} "
        f"{r['wr']*100:>5.1f}% {r['ci_lo']*100:>5.1f}% {r['ci_hi']*100:>5.1f}% "
        f"{'YES' if r['ci_pass'] else 'NO':>6} "
        f"${r['total_pnl']:>8.2f} ${r['avg_pnl']:>6.2f} "
        f"${r['avg_entry']:>7.3f} {pf_str:>7} "
        f"{r['wf_profitable']:>2}/3 {r['wf_grade']:>8} {r['verdict']:>8}"
    )
    print(line)

# Totals
total_trades = sum(r["n_trades"] for r in results)
total_wins = sum(r["wins"] for r in results)
total_losses = sum(r["losses"] for r in results)
total_pnl = sum(r["total_pnl"] for r in results)
overall_wr = total_wins / total_trades if total_trades > 0 else 0

print("-" * 120)
print(
    f"{'TOTAL':<20} {total_trades:>6} {total_wins:>4} {total_losses:>4} "
    f"{overall_wr*100:>5.1f}%"
    f"{'':>22}"
    f"${total_pnl:>8.2f}"
)
print()

# ── Detailed verdicts ─────────────────────────────────────────────────────
print("=" * 80)
print("  PROMOTION VERDICTS")
print("=" * 80)

for r in results:
    tag = r["verdict"]
    reasons = []
    if r["n_trades"] < 10:
        reasons.append(f"only {r['n_trades']} trades (need 10+)")
    elif r["n_trades"] < 20:
        reasons.append(f"only {r['n_trades']} trades (need 20+ for PROMOTE)")
    if not r["ci_pass"]:
        reasons.append(f"CI lower bound {r['ci_lo']*100:.1f}% <= 50%")
    if r["wf_grade"] == "WEAK":
        reasons.append(f"walk-forward WEAK ({r['wf_profitable']}/3 profitable)")
    if r["total_pnl"] <= 0:
        reasons.append(f"negative PnL (${r['total_pnl']:.2f})")
    if r["ci_pass"] and r["n_trades"] >= 20 and r["wf_grade"] in ("ROBUST", "OK") and r["total_pnl"] > 0:
        reasons.append("ALL criteria met")

    reason_str = "; ".join(reasons) if reasons else "criteria partially met"
    symbol = {
        "PROMOTE": "[+]",
        "WATCH": "[~]",
        "SKIP": "[-]",
    }[tag]

    print(f"  {symbol} {r['strategy']:<20} -> {tag:>8}  ({reason_str})")

print()
print("Criteria: PROMOTE = 20+ trades, CI_lo > 50%, walk-forward ROBUST/OK, PnL > 0")
print("          WATCH   = 10-19 trades with WR > 60%, or 20+ borderline")
print("          SKIP    = <10 trades, CI_lo <= 50%, or walk-forward WEAK")
print()
