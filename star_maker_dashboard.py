"""
  ╔═══════════════════════════════════════════════════════════╗
  ║            STAR-MAKER LIVE DASHBOARD v1.1                ║
  ║         10-Minute PnL Tracker — Maker Bot Only           ║
  ╚═══════════════════════════════════════════════════════════╝

  Tracks new trades from launch time. W/L and PnL start at zero.
  Balance starts at $100. Updates every 10 minutes with rolling chart.
"""
import json, time, sys, io
from datetime import datetime, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MAKER_PATH = r"C:\Users\Star\.local\bin\star-polymarket\maker_results.json"
INTERVAL = 600  # 10 minutes
STARTING_BALANCE = 100.00

def load_maker():
    with open(MAKER_PATH) as f:
        return json.load(f)

def get_snapshot(data):
    resolved = data.get('resolved', [])
    paired = [r for r in resolved if r.get('paired')]
    partial = [r for r in resolved if r.get('partial')]
    paired_pnl = sum(r.get('pnl', 0) for r in paired)
    partial_pnl = sum(r.get('pnl', 0) for r in partial)
    wins = len([r for r in paired if r.get('pnl', 0) > 0])
    losses = len([r for r in paired if r.get('pnl', 0) <= 0])
    return {
        'resolved': len(resolved),
        'paired': len(paired),
        'partial': len(partial),
        'paired_pnl': paired_pnl,
        'partial_pnl': partial_pnl,
        'net_pnl': paired_pnl + partial_pnl,
        'wins': wins,
        'losses': losses,
    }

# === BASELINE ===
baseline_data = load_maker()
baseline = get_snapshot(baseline_data)
start_time = datetime.now(timezone.utc)

# Track history for chart
history = []

SEP = "\n" + "=" * 75 + "\n"

def render(tick_num):
    now = datetime.now(timezone.utc)
    elapsed = (now - start_time).total_seconds()
    elapsed_min = elapsed / 60

    try:
        current_data = load_maker()
        current = get_snapshot(current_data)
    except Exception as e:
        print(f"  Error reading maker data: {e}")
        sys.stdout.flush()
        return

    # Deltas from baseline
    d_paired = current['paired'] - baseline['paired']
    d_partial = current['partial'] - baseline['partial']
    d_resolved = current['resolved'] - baseline['resolved']
    d_pnl = current['net_pnl'] - baseline['net_pnl']
    d_paired_pnl = current['paired_pnl'] - baseline['paired_pnl']
    d_partial_pnl = current['partial_pnl'] - baseline['partial_pnl']
    d_wins = current['wins'] - baseline['wins']
    d_losses = current['losses'] - baseline['losses']
    wr = d_wins / (d_wins + d_losses) * 100 if (d_wins + d_losses) > 0 else 0

    balance = STARTING_BALANCE + d_pnl

    # Record history
    history.append({
        'time': now,
        'tick': tick_num,
        'pnl': d_pnl,
        'balance': balance,
        'wins': d_wins,
        'losses': d_losses,
        'paired': d_paired,
        'partial': d_partial,
        'elapsed_min': elapsed_min,
    })

    # Hourly rate projection
    if elapsed_min > 1:
        hourly_rate = d_pnl / (elapsed_min / 60)
        daily_proj = hourly_rate * 24
    else:
        hourly_rate = 0
        daily_proj = 0

    # Use separator instead of clear (works in background mode)
    print(SEP)
    print("  ╔═══════════════════════════════════════════════════════════════════╗")
    print("  ║              S T A R - M A K E R   D A S H B O A R D            ║")
    print("  ║                    Live PnL Tracker v1.1                         ║")
    print("  ╚═══════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Started: {start_time.strftime('%H:%M:%S')} UTC | Now: {now.strftime('%H:%M:%S')} UTC | Elapsed: {int(elapsed_min)}m")
    print(f"  Update #{tick_num} (every 10 min)")
    print()

    # Balance + PnL box
    bal_bar_len = max(0, min(int(balance / 2), 50))
    bal_bar = "#" * bal_bar_len

    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │  BALANCE:      ${balance:>8.2f}  (started ${STARTING_BALANCE:.2f})               │")
    print(f"  │  [{bal_bar:<50}] │")
    print(f"  │                                                                 │")
    print(f"  │  SESSION PnL:  ${d_pnl:>+8.2f}                                      │")
    print(f"  │    Paired:     ${d_paired_pnl:>+8.2f}   ({d_paired} trades)                     │")
    print(f"  │    Partials:   ${d_partial_pnl:>+8.2f}   ({d_partial} partials)                   │")
    print(f"  │                                                                 │")
    print(f"  │  W / L:        {d_wins:>3} / {d_losses:<3}    WR: {wr:>5.1f}%                        │")
    print(f"  │  Hourly Rate:  ${hourly_rate:>+8.2f}                                      │")
    print(f"  │  Daily Proj:   ${daily_proj:>+8.2f}                                      │")
    print("  └─────────────────────────────────────────────────────────────────┘")
    print()

    # === 10-MINUTE PnL CHART ===
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │                    10-MINUTE PnL CHART                         │")
    print("  ├─────────────────────────────────────────────────────────────────┤")

    if len(history) <= 1:
        print("  │  Collecting first data point... chart starts at update #1      │")
    else:
        # Calculate interval deltas
        intervals = []
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            interval_pnl = curr['pnl'] - prev['pnl']
            interval_wins = curr['wins'] - prev['wins']
            interval_losses = curr['losses'] - prev['losses']
            intervals.append({
                'tick': curr['tick'],
                'time': curr['time'],
                'pnl': interval_pnl,
                'cum_pnl': curr['pnl'],
                'balance': curr['balance'],
                'wins': interval_wins,
                'losses': interval_losses,
            })

        # Find max abs PnL for scaling
        max_abs = max(abs(iv['pnl']) for iv in intervals) if intervals else 1
        max_abs = max(max_abs, 0.50)  # minimum scale $0.50
        chart_width = 25

        for iv in intervals[-24:]:  # show last 24 intervals (4 hours)
            t = iv['time'].strftime('%H:%M')
            pnl = iv['pnl']
            bal = iv['balance']
            w = iv['wins']
            l = iv['losses']

            # Bar
            bar_len = int(abs(pnl) / max_abs * chart_width)
            bar_len = max(bar_len, 0)

            if pnl >= 0:
                bar = " " * chart_width + "|" + "#" * bar_len
            else:
                pad = chart_width - bar_len
                bar = " " * pad + "-" * bar_len + "|"

            wl_str = f"{w}W/{l}L" if (w + l) > 0 else " --- "
            print(f"  │ {t} {bar} ${pnl:>+5.2f} {wl_str:>5} bal:${bal:>7.2f} │")

    print("  └─────────────────────────────────────────────────────────────────┘")
    print()

    # === BALANCE SPARKLINE ===
    if len(history) > 1:
        bals = [h['balance'] for h in history]
        min_b = min(bals)
        max_b = max(bals)
        rng = max_b - min_b if max_b != min_b else 1
        spark_chars = "_.-~*#"
        spark = ""
        for b in bals:
            idx = int((b - min_b) / rng * (len(spark_chars) - 1))
            spark += spark_chars[idx]
        print(f"  Balance trend: ${min_b:.2f} [{spark}] ${max_b:.2f}")
        print()

    # New trades detail
    if d_resolved > 0:
        new_resolved = current_data.get('resolved', [])[-d_resolved:]
        print("  Recent trades this session:")
        for r in new_resolved[-10:]:
            asset = r.get('asset', '?')
            pnl = r.get('pnl', 0)
            is_paired = r.get('paired', False)
            is_partial = r.get('partial', False)
            status = "PAIRED" if is_paired else ("PARTIAL" if is_partial else "UNFILLED")
            icon = "W" if pnl > 0 else "L" if pnl < 0 else "="
            print(f"    {icon} {asset:>4} {status:<8} ${pnl:>+6.2f}")
        if d_resolved > 10:
            print(f"    ... and {d_resolved - 10} more")
        print()

    # Lifetime reference
    print(f"  Lifetime (all-time): ${current['net_pnl']:>+.2f} | {current['wins']}W/{current['losses']}L | {current['paired']} paired / {current['partial']} partial")
    print()

    next_min = (now.minute // 10 + 1) * 10
    if next_min >= 60:
        next_min = 0
    print(f"  Next update: ~{next_min:02d} min mark | Dashboard running...")
    sys.stdout.flush()

# === MAIN LOOP ===
print()
print("  ╔═══════════════════════════════════════════════════════════════════╗")
print("  ║         S T A R - M A K E R   D A S H B O A R D   v1.1         ║")
print("  ╚═══════════════════════════════════════════════════════════════════╝")
print()
print(f"  Starting balance: ${STARTING_BALANCE:.2f}")
print(f"  Baseline maker PnL: ${baseline['net_pnl']:>+.2f} | {baseline['wins']}W/{baseline['losses']}L | {baseline['paired']} paired")
print(f"  W/L zeroed, PnL zeroed, balance set to ${STARTING_BALANCE:.2f}")
print(f"  First chart data in 10 minutes...")
print()
sys.stdout.flush()
time.sleep(2)

tick = 0
render(tick)

while True:
    time.sleep(INTERVAL)
    tick += 1
    render(tick)
