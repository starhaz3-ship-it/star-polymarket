"""Find high-probability non-BTC trades."""
import httpx
import json
from datetime import datetime, timezone

print("=" * 70)
print("SCANNING FOR NON-BTC HIGH PROBABILITY TRADES")
print("=" * 70)
print()

# Get all active markets
r = httpx.get("https://gamma-api.polymarket.com/markets", params={
    "active": "true",
    "closed": "false",
    "limit": 300
})
markets = r.json()

# Get whale positions for signal
WHALE = "0xe9c6312464b52aa3eff13d822b003282075995c9"
r = httpx.get("https://data-api.polymarket.com/positions", params={"user": WHALE, "sizeThreshold": 0})
whale_positions = {p.get("conditionId"): p for p in r.json()}

now = datetime.now(timezone.utc)
opportunities = []

for m in markets:
    question = m.get("question", "")
    condition_id = m.get("conditionId", "")

    # Skip BTC/Bitcoin markets
    if "BTC" in question.upper() or "BITCOIN" in question.upper():
        continue

    # Get prices
    try:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            prices = json.loads(prices)
        if not prices:
            continue
        yes_price = float(prices[0])
        no_price = float(prices[1]) if len(prices) > 1 else 1 - yes_price
    except:
        continue

    # Skip settled or illiquid
    if yes_price <= 0.02 or yes_price >= 0.98:
        continue

    # Get end date
    end_str = m.get("endDate", "")
    if not end_str:
        continue

    try:
        end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        hours_left = (end_date - now).total_seconds() / 3600
    except:
        continue

    # Only short duration (< 7 days)
    if hours_left <= 0 or hours_left > 168:
        continue

    # Check if whale has position
    whale_pos = whale_positions.get(condition_id)
    whale_pnl = 0
    whale_side = None
    if whale_pos:
        whale_pnl = float(whale_pos.get("cashPnl", 0) or 0)
        whale_side = whale_pos.get("outcome")

    # Calculate potential returns
    yes_return = (1 / yes_price - 1) * 100 if yes_price > 0 else 0
    no_return = (1 / no_price - 1) * 100 if no_price > 0 else 0

    # Look for high probability (>70%) with decent return (>20%)
    if yes_price >= 0.70 and yes_return >= 15:
        opportunities.append({
            "question": question[:65],
            "condition_id": condition_id,
            "tokens": m.get("clobTokenIds", ""),
            "side": "YES",
            "price": yes_price,
            "potential_return": yes_return,
            "hours_left": hours_left,
            "whale_pnl": whale_pnl,
            "whale_agrees": whale_side == "Yes" if whale_side else False,
            "volume": float(m.get("volume", 0) or 0),
        })

    if no_price >= 0.70 and no_return >= 15:
        opportunities.append({
            "question": question[:65],
            "condition_id": condition_id,
            "tokens": m.get("clobTokenIds", ""),
            "side": "NO",
            "price": no_price,
            "potential_return": no_return,
            "hours_left": hours_left,
            "whale_pnl": whale_pnl,
            "whale_agrees": whale_side == "No" if whale_side else False,
            "volume": float(m.get("volume", 0) or 0),
        })

# Sort by: high probability, short duration, good return
opportunities.sort(key=lambda x: (x["price"], -x["hours_left"], x["potential_return"]), reverse=True)

print(f"Found {len(opportunities)} opportunities\n")

# Show top opportunities
for i, opp in enumerate(opportunities[:15], 1):
    whale_indicator = " [WHALE]" if opp["whale_agrees"] else ""
    print(f"{i}. {opp['question']}")
    print(f"   {opp['side']} @ ${opp['price']:.2f} | Return: +{opp['potential_return']:.0f}%{whale_indicator}")
    print(f"   Expires: {opp['hours_left']:.0f}h | Volume: ${opp['volume']:,.0f}")
    print()

# Find BEST opportunity
print("=" * 70)
print("TOP RECOMMENDATION")
print("=" * 70)

# Prefer: whale agrees, high volume, >80% probability, short duration
best = None
for opp in opportunities:
    # High probability
    if opp["price"] < 0.80:
        continue
    # Decent volume
    if opp["volume"] < 10000:
        continue
    # Not too short (avoid last-minute volatility)
    if opp["hours_left"] < 6:
        continue

    best = opp
    break

if best:
    print(f"\nMarket: {best['question']}")
    print(f"Action: BUY {best['side']}")
    print(f"Price: ${best['price']:.3f}")
    print(f"Probability: {best['price']*100:.0f}%")
    print(f"Return if wins: +{best['potential_return']:.0f}%")
    print(f"Expires in: {best['hours_left']:.0f} hours")
    print(f"Volume: ${best['volume']:,.0f}")
    print(f"Condition ID: {best['condition_id']}")

    tokens = best.get("tokens", "")
    if tokens:
        try:
            token_list = json.loads(tokens) if isinstance(tokens, str) else tokens
            if best["side"] == "YES":
                print(f"Token ID: {token_list[0]}")
            else:
                print(f"Token ID: {token_list[1] if len(token_list) > 1 else token_list[0]}")
        except:
            print(f"Tokens: {tokens}")
else:
    print("\nNo high-conviction non-BTC trades found matching criteria.")
    print("Criteria: >80% probability, >$10K volume, 6-168 hours to expiry")
