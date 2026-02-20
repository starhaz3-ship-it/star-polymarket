"""Analyze 7thStaircase's recent trades to understand their strategy."""
import httpx
import json
from collections import defaultdict
from datetime import datetime

wallet = "0x0ac97e4f5c542cd98c226ae8e1736ae78b489641"

# Fetch recent positions
r = httpx.get(
    f"https://data-api.polymarket.com/positions?user={wallet}&limit=200&sortBy=startDate&sortDir=desc",
    timeout=15,
)
positions = r.json()
print(f"Total positions returned: {len(positions)}")

wins = losses = 0
total_pnl = 0.0
both_sides = defaultdict(list)  # group by market title+time to detect hedging
sizes = []
entry_prices = []
strategies = {"5m_up": 0, "5m_down": 0, "15m_up": 0, "15m_down": 0, "other": 0}
hourly_pnl = defaultdict(float)
hourly_count = defaultdict(int)

for p in positions:
    market = p.get("title", "")
    side = p.get("outcome", "?")
    size = float(p.get("initialValue", 0))
    cur_val = float(p.get("currentValue", 0))
    pnl = cur_val - size
    resolved = p.get("resolved", False)
    entry = float(p.get("avgPrice", 0))
    shares = float(p.get("size", 0))
    start = p.get("startDate", "")

    if resolved:
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        total_pnl += pnl

    sizes.append(size)
    if entry > 0:
        entry_prices.append(entry)

    # Categorize
    is_5m = "5 min" in market.lower() or (":" in market and "PM" in market and "5" in market)
    is_15m = "15 min" in market.lower()
    # Actually detect by time range in title
    if "6:50PM-6:55PM" in market or "6:55PM-7:00PM" in market or "6:40PM-6:45PM" in market or "6:45PM-6:50PM" in market or "6:35PM-6:40PM" in market:
        tf = "5m"
    elif "6:45PM-7:00PM" in market or "6:30PM-6:45PM" in market or "6:15PM-6:30PM" in market:
        tf = "15m"
    else:
        tf = "other"

    key = f"{tf}_{side.lower()}" if tf != "other" else "other"
    if key in strategies:
        strategies[key] += 1
    else:
        strategies["other"] += 1

    # Group by market for hedging detection
    market_key = market.split(" - ")[-1] if " - " in market else market
    both_sides[market_key].append({"side": side, "size": size, "entry": entry, "pnl": pnl, "shares": shares})

    # Hourly PnL
    if start and resolved:
        try:
            dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            hourly_pnl[dt.hour] += pnl
            hourly_count[dt.hour] += 1
        except:
            pass

print(f"\n=== OVERALL (last {len(positions)} positions) ===")
print(f"Record: {wins}W/{losses}L ({100*wins/(wins+losses) if wins+losses else 0:.0f}% WR)")
print(f"Total PnL: ${total_pnl:+,.2f}")
print(f"Avg trade size: ${sum(sizes)/len(sizes) if sizes else 0:.2f}")
print(f"Avg entry price: ${sum(entry_prices)/len(entry_prices) if entry_prices else 0:.2f}")
print(f"Max trade size: ${max(sizes) if sizes else 0:.2f}")

print(f"\n=== STRATEGY BREAKDOWN ===")
for k, v in sorted(strategies.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} trades")

print(f"\n=== HEDGING ANALYSIS ===")
hedged = 0
unhedged = 0
for mkt, trades in both_sides.items():
    sides_in_market = set(t["side"] for t in trades)
    if len(sides_in_market) > 1:
        hedged += 1
        total_cost = sum(t["size"] for t in trades)
        total_return = sum(t["size"] + t["pnl"] for t in trades)
        net = total_return - total_cost
        if len(list(both_sides.items())) <= 60:  # only print if manageable
            print(f"  HEDGED: {mkt[:50]} | cost=${total_cost:.2f} return=${total_return:.2f} net=${net:+.2f}")
    else:
        unhedged += 1

print(f"\nHedged markets: {hedged} | Unhedged: {unhedged}")

print(f"\n=== RECENT 20 TRADES ===")
for p in positions[:20]:
    market = p.get("title", "")[:55]
    side = p.get("outcome", "?")
    size = float(p.get("initialValue", 0))
    pnl = float(p.get("currentValue", 0)) - size
    entry = float(p.get("avgPrice", 0))
    shares = float(p.get("size", 0))
    resolved = p.get("resolved", False)
    status = "W" if pnl > 0 else "L" if resolved else "?"
    print(f"  {status} | {side:4s} @ ${entry:.2f} | ${size:>8.2f} cost | PnL: ${pnl:+8.2f} | {market}")

print(f"\n=== HOURLY PnL (UTC) ===")
for h in sorted(hourly_pnl.keys()):
    print(f"  Hour {h:02d}: ${hourly_pnl[h]:+8.2f} ({hourly_count[h]} trades)")
