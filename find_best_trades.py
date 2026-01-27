"""Find best high-probability trades right now."""
import httpx
import json
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
print("HIGH PROBABILITY OPPORTUNITIES")
print("=" * 70)

opportunities = []

for pos in positions:
    title = pos.get("title", "")
    outcome = pos.get("outcome", "")
    current_price = float(pos.get("curPrice", 0) or 0)
    entry_price = float(pos.get("avgPrice", 0) or 0)
    whale_pnl = float(pos.get("cashPnl", 0) or 0)
    value = float(pos.get("currentValue", 0) or 0)
    condition_id = pos.get("conditionId", "")
    token_id = pos.get("asset", "")

    # Skip closed/settled
    if current_price <= 0.01 or current_price >= 0.99:
        continue

    # Skip small positions
    if value < 500:
        continue

    # Calculate edge based on criteria
    edge = 0
    confidence = 0
    reason = ""

    # BTC markets - compare to spot price
    if "BTC" in title.upper() or "BITCOIN" in title.upper():
        import re
        match = re.search(r'\$?([\d,]+)', title)
        if match:
            threshold = float(match.group(1).replace(",", ""))

            # Calculate how far BTC is from threshold
            distance_pct = (btc_price - threshold) / threshold

            if outcome == "Yes":
                # YES bet - BTC needs to be ABOVE threshold
                if distance_pct > 0.02:  # BTC is 2%+ above threshold
                    # High probability - calculate edge
                    actual_prob = min(0.95, 0.70 + distance_pct * 2)
                    edge = actual_prob - current_price
                    confidence = min(0.95, 0.5 + distance_pct * 3)
                    reason = f"BTC ${btc_price:,.0f} is {distance_pct:.1%} ABOVE ${threshold:,.0f}"

            elif outcome == "No":
                # NO bet - BTC needs to be BELOW threshold
                if distance_pct < -0.02:  # BTC is 2%+ below threshold
                    actual_prob = min(0.95, 0.70 + abs(distance_pct) * 2)
                    edge = actual_prob - current_price
                    confidence = min(0.95, 0.5 + abs(distance_pct) * 3)
                    reason = f"BTC ${btc_price:,.0f} is {abs(distance_pct):.1%} BELOW ${threshold:,.0f}"

    # Apply our ML criteria
    # 1. YES positions preferred (43% vs 6% win rate from backtest)
    if outcome == "Yes":
        confidence *= 1.2
    else:
        confidence *= 0.5  # Penalize NO positions

    # 2. Entry 50-85% is optimal
    if 0.50 <= current_price <= 0.85:
        confidence *= 1.1
    elif current_price > 0.90:
        confidence *= 0.8  # Less upside

    # 3. Whale is profitable
    if whale_pnl > 0:
        confidence *= 1.1
    else:
        confidence *= 0.7

    # 4. Positive momentum
    if entry_price > 0:
        momentum = (current_price - entry_price) / entry_price
        if momentum > 0:
            confidence *= 1.1

    # Only include if edge > 5% and confidence > 50%
    if edge > 0.05 and confidence > 0.50:
        # Calculate Kelly position size
        win_return = (1 - current_price) / current_price
        kelly = (confidence * win_return - (1 - confidence)) / win_return
        kelly = max(0, kelly) * 0.25  # Quarter Kelly
        kelly = min(kelly, 0.15)  # Max 15%

        opportunities.append({
            "title": title[:60],
            "outcome": outcome,
            "condition_id": condition_id,
            "token_id": token_id,
            "current_price": current_price,
            "edge": edge,
            "confidence": confidence,
            "whale_pnl": whale_pnl,
            "kelly_fraction": kelly,
            "reason": reason,
            "expected_return": edge * kelly,
        })

# Sort by expected return
opportunities.sort(key=lambda x: x["expected_return"], reverse=True)

if not opportunities:
    print("No high-probability opportunities found matching criteria.")
    print("\nCriteria: YES positions, 50-85% entry, edge > 5%, confidence > 50%")
else:
    print(f"\nFound {len(opportunities)} opportunities:\n")

    for i, opp in enumerate(opportunities[:10], 1):
        print(f"{i}. {opp['title']}")
        print(f"   Side: {opp['outcome']} @ ${opp['current_price']:.3f}")
        print(f"   Edge: {opp['edge']:.1%} | Confidence: {opp['confidence']:.0%}")
        print(f"   Whale P&L: ${opp['whale_pnl']:+,.0f}")
        print(f"   Kelly Size: {opp['kelly_fraction']:.1%} of bankroll")
        print(f"   Reason: {opp['reason']}")
        print()

    # Best trade recommendation
    if opportunities:
        best = opportunities[0]
        print("=" * 70)
        print("RECOMMENDED TRADE")
        print("=" * 70)
        print(f"Market: {best['title']}")
        print(f"Action: BUY {best['outcome']}")
        print(f"Price: ${best['current_price']:.3f}")
        print(f"Edge: {best['edge']:.1%}")
        print(f"Confidence: {best['confidence']:.0%}")
        print(f"Kelly suggests: {best['kelly_fraction']:.1%} of bankroll")
        print(f"For $100 bankroll: ${best['kelly_fraction'] * 100:.2f}")
        print(f"Condition ID: {best['condition_id']}")
        print(f"Token ID: {best['token_id']}")
