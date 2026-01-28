"""Find best trades with broader criteria."""
import httpx
import json
import re
from datetime import datetime, timezone

# Get current BTC price
r = httpx.get('https://api.binance.com/api/v3/ticker/price', params={'symbol': 'BTCUSDT'})
btc_price = float(r.json()['price'])
print(f"Current BTC: ${btc_price:,.2f}")
print()

# Get whale positions
WHALE = "0xe9c6312464b52aa3eff13d822b003282075995c9"
r = httpx.get("https://data-api.polymarket.com/positions", params={"user": WHALE, "sizeThreshold": 0})
positions = r.json()

print("=" * 70)
print("ANALYZING ALL WHALE POSITIONS")
print("=" * 70)

btc_opportunities = []
other_opportunities = []

for pos in positions:
    title = pos.get("title", "")
    outcome = pos.get("outcome", "")
    current_price = float(pos.get("curPrice", 0) or 0)
    entry_price = float(pos.get("avgPrice", 0) or 0)
    whale_pnl = float(pos.get("cashPnl", 0) or 0)
    value = float(pos.get("currentValue", 0) or 0)
    condition_id = pos.get("conditionId", "")
    token_id = pos.get("asset", "")
    end_date = pos.get("endDate", "")

    # Skip closed/settled
    if current_price <= 0.01 or current_price >= 0.99:
        continue

    # Skip very small positions
    if value < 100:
        continue

    # Momentum
    momentum = 0
    if entry_price > 0:
        momentum = (current_price - entry_price) / entry_price

    opp = {
        "title": title[:55],
        "outcome": outcome,
        "condition_id": condition_id,
        "token_id": token_id,
        "current_price": current_price,
        "entry_price": entry_price,
        "momentum": momentum,
        "whale_pnl": whale_pnl,
        "whale_value": value,
        "end_date": end_date[:10] if end_date else "",
    }

    # Check if BTC market
    if "BTC" in title.upper() or "BITCOIN" in title.upper():
        match = re.search(r'\$?([\d,]+)', title)
        if match:
            threshold = float(match.group(1).replace(",", ""))
            distance = (btc_price - threshold) / threshold
            opp["threshold"] = threshold
            opp["btc_distance"] = distance

            # Calculate probability based on distance
            if outcome == "Yes":
                if distance > 0:  # BTC above threshold
                    opp["estimated_prob"] = min(0.98, 0.60 + distance * 3)
                    opp["status"] = f"BTC is {distance:.1%} ABOVE - FAVORABLE"
                else:
                    opp["estimated_prob"] = max(0.05, 0.40 + distance * 2)
                    opp["status"] = f"BTC is {abs(distance):.1%} BELOW - AT RISK"
            else:  # No
                if distance < 0:  # BTC below threshold
                    opp["estimated_prob"] = min(0.98, 0.60 + abs(distance) * 3)
                    opp["status"] = f"BTC is {abs(distance):.1%} BELOW - FAVORABLE"
                else:
                    opp["estimated_prob"] = max(0.05, 0.40 - distance * 2)
                    opp["status"] = f"BTC is {distance:.1%} ABOVE - AT RISK"

            opp["edge"] = opp["estimated_prob"] - current_price
            btc_opportunities.append(opp)
    else:
        other_opportunities.append(opp)

# Sort BTC opportunities by edge
btc_opportunities.sort(key=lambda x: x.get("edge", 0), reverse=True)

print(f"\n### BTC MARKETS ({len(btc_opportunities)} active) ###\n")

for opp in btc_opportunities[:15]:
    edge = opp.get("edge", 0)
    prob = opp.get("estimated_prob", 0)
    status = "***" if edge > 0.05 and prob > 0.70 else ""

    print(f"{status}{opp['outcome']} @ ${opp['current_price']:.2f} | {opp['title']}")
    print(f"   Threshold: ${opp.get('threshold', 0):,.0f} | {opp.get('status', '')}")
    print(f"   Est Prob: {prob:.0%} | Edge: {edge:+.1%} | Whale P&L: ${opp['whale_pnl']:+,.0f}")
    print(f"   Expires: {opp['end_date']} | Token: {opp['token_id'][:20]}...")
    print()

# Find BEST trade
print("=" * 70)
print("BEST TRADE RECOMMENDATION")
print("=" * 70)

# Filter for:
# 1. YES position (our ML shows 43% win vs 6% for NO)
# 2. Edge > 3%
# 3. Whale is profitable
# 4. Entry 50-90%

best_trades = [
    o for o in btc_opportunities
    if o["outcome"] == "Yes"
    and o.get("edge", 0) > 0.03
    and o["whale_pnl"] > 0
    and 0.50 <= o["current_price"] <= 0.90
]

if not best_trades:
    # Relax to include NO positions if no YES available
    best_trades = [
        o for o in btc_opportunities
        if o.get("edge", 0) > 0.05
        and 0.50 <= o["current_price"] <= 0.95
    ]

if best_trades:
    best = best_trades[0]
    print(f"\nMarket: {best['title']}")
    print(f"Action: BUY {best['outcome']}")
    print(f"Current Price: ${best['current_price']:.3f}")
    print(f"Estimated Win Prob: {best.get('estimated_prob', 0):.0%}")
    print(f"Edge: {best.get('edge', 0):+.1%}")
    print(f"Status: {best.get('status', '')}")
    print(f"Whale P&L: ${best['whale_pnl']:+,.0f}")
    print(f"Expires: {best['end_date']}")
    print(f"\nCondition ID: {best['condition_id']}")
    print(f"Token ID: {best['token_id']}")

    # Calculate recommended size
    kelly = best.get("edge", 0) * 0.25 / (1 - best["current_price"])
    kelly = max(0, min(kelly, 0.10))  # Cap at 10%
    print(f"\nKelly suggests: {kelly:.1%} of bankroll")
    print(f"For $100 bankroll: ${kelly * 100:.2f}")
else:
    print("\nNo high-conviction trades found at current prices.")
    print("The market may have already priced in the BTC move.")
