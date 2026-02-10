"""Quick weather bot status check."""
import json

data = json.load(open("weather_paper_results.json"))
trades = data["trades"]
o = sum(1 for t in trades.values() if t["status"] == "open")
c = sum(1 for t in trades.values() if t["status"] == "closed")
print("=== WEATHER BOT ===")
print(f"Scans: {data['scan_count']} | Trades: {len(trades)} (Open: {o}, Closed: {c})")
print(f"PnL: ${data['total_pnl']:+.2f} | Wins: {data['wins']} Losses: {data['losses']}")
cids = set(t["condition_id"] for t in trades.values())
print(f"Unique markets: {len(cids)}")
print()
for t in trades.values():
    r = ""
    if t["status"] == "closed":
        r = f" -> {t['exit_reason']} ${t['pnl']:+.2f}"
    print(f"  {t['location']:15s} {t['temp_bucket']:15s} @ ${t['entry_price']:.4f} | NOAA={t['noaa_probability']:.0%} edge={t['edge']:.0%} | {t['status']}{r}")
print()
ls = data.get("location_stats", {})
if ls:
    print("Location stats:")
    for loc, s in sorted(ls.items()):
        wr = s["wins"] / max(1, s["trades"]) * 100
        print(f"  {loc}: {s['trades']}T {wr:.0f}%WR ${s['pnl']:+.2f}")
