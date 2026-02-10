"""Quick check of Hydra A/B tracking status."""
import json

data = json.load(open("ta_paper_results.json"))
hq = data.get("hydra_quarantine", {})
hqr = data.get("hydra_quarantine_raw", {})
hfs = data.get("hydra_filter_stats", {})
hp = data.get("hydra_pending", {})

print("=== HYDRA A/B TRACKING STATUS ===")
print()

open_p = sum(1 for p in hp.values() if p.get("status") == "open")
filtered_p = sum(1 for p in hp.values() if p.get("filtered"))
total_p = len(hp)
print(f"Pending: {open_p} open, {filtered_p} filtered, {total_p} total entries")
print()

print(f"{'STRATEGY':25s} | {'FILTERED (acted on)':22s} | {'RAW (A/B comparison)':22s}")
print("-" * 75)
all_names = sorted(set(list(hq.keys()) + list(hqr.keys())))
for name in all_names:
    q = hq.get(name, {})
    qr = hqr.get(name, {})
    n = q.get("predictions", 0)
    nr = qr.get("predictions", 0)
    if n > 0:
        wr = q.get("correct", 0) / n * 100
        f_str = f"{n}T {wr:.0f}%WR ${q.get('pnl', 0):+.2f}"
    else:
        f_str = "no signals pass"
    if nr > 0:
        wr_r = qr.get("correct", 0) / nr * 100
        r_str = f"{nr}T {wr_r:.0f}%WR ${qr.get('pnl', 0):+.2f}"
    else:
        r_str = "-"
    status = q.get("status", "Q")
    print(f"  {name:23s} {status:4s} | {f_str:22s} | {r_str:22s}")

# Totals
filt_total = sum(q.get("predictions", 0) for q in hq.values())
filt_correct = sum(q.get("correct", 0) for q in hq.values())
filt_pnl = sum(q.get("pnl", 0) for q in hq.values())
raw_total = sum(q.get("predictions", 0) for q in hqr.values())
raw_correct = sum(q.get("correct", 0) for q in hqr.values())
raw_pnl = sum(q.get("pnl", 0) for q in hqr.values())
print("-" * 75)
filt_wr = filt_correct / max(1, filt_total) * 100
raw_wr = raw_correct / max(1, raw_total) * 100
print(f"  {'TOTAL':23s}      | {filt_total}T {filt_wr:.0f}%WR ${filt_pnl:+.2f}      | {raw_total}T {raw_wr:.0f}%WR ${raw_pnl:+.2f}")

if hfs:
    print()
    print("=== FILTER EFFECTIVENESS ===")
    total_saved = 0
    for fname, fs in sorted(hfs.items()):
        net = fs["saved_pnl"]
        total_saved += net
        verdict = "SAVING" if net > 0 else "COSTING"
        print(f"  {fname:30s} blocked={fs['blocked']} | would_win={fs['would_win']} would_lose={fs['would_lose']} | net=${net:+.2f} ({verdict})")
    print(f"  {'TOTAL FILTER VALUE':30s} ${total_saved:+.2f}")

# Show some recent pending entries with filter status
filtered_entries = [(k, v) for k, v in hp.items() if v.get("filtered")]
if filtered_entries:
    print(f"\n=== RECENT FILTERED SIGNALS ({len(filtered_entries)} total) ===")
    for k, v in list(filtered_entries)[-5:]:
        print(f"  {v['strategy']:20s} {v['asset']} {v['direction']} | {v.get('filter_reason', '?')} | conf={v['confidence']:.0%}")
