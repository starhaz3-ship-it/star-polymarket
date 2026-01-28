"""Find interesting non-BTC trades - broader search."""
import httpx
import json
from datetime import datetime, timezone

print("=" * 70)
print("SCANNING ALL NON-BTC/CRYPTO MARKETS")
print("=" * 70)
print()

# Get all active markets
r = httpx.get("https://gamma-api.polymarket.com/markets", params={
    "active": "true",
    "closed": "false",
    "limit": 500
})
markets = r.json()

# Get whale positions
WHALE = "0xe9c6312464b52aa3eff13d822b003282075995c9"
r = httpx.get("https://data-api.polymarket.com/positions", params={"user": WHALE, "sizeThreshold": 0})
whale_positions = {p.get("conditionId"): p for p in r.json()}

now = datetime.now(timezone.utc)
opportunities = []

crypto_keywords = ["BTC", "BITCOIN", "ETH", "ETHER", "SOL", "SOLANA", "DOGE", "XRP", "CRYPTO"]

for m in markets:
    question = m.get("question", "")
    condition_id = m.get("conditionId", "")
    q_upper = question.upper()

    # Skip crypto markets
    if any(kw in q_upper for kw in crypto_keywords):
        continue

    # Get prices
    try:
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            prices = json.loads(prices)
        if not prices:
            continue
        yes_price = float(prices[0])
    except:
        continue

    # Skip settled
    if yes_price <= 0.03 or yes_price >= 0.97:
        continue

    # Get end date
    end_str = m.get("endDate", "")
    if not end_str:
        continue

    try:
        end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        hours_left = (end_date - now).total_seconds() / 3600
        days_left = hours_left / 24
    except:
        continue

    # Only reasonably short duration (< 14 days)
    if hours_left <= 0 or hours_left > 336:
        continue

    volume = float(m.get("volume", 0) or 0)

    # Check whale position
    whale_pos = whale_positions.get(condition_id)
    whale_pnl = 0
    whale_side = None
    whale_value = 0
    if whale_pos:
        whale_pnl = float(whale_pos.get("cashPnl", 0) or 0)
        whale_side = whale_pos.get("outcome")
        whale_value = float(whale_pos.get("currentValue", 0) or 0)

    # Look for opportunities
    no_price = 1 - yes_price

    for side, price in [("YES", yes_price), ("NO", no_price)]:
        if price < 0.50:  # Only high probability
            continue

        potential_return = (1 / price - 1) * 100

        # Skip if return too low
        if potential_return < 5:
            continue

        opportunities.append({
            "question": question[:60],
            "condition_id": condition_id,
            "tokens": m.get("clobTokenIds", ""),
            "side": side,
            "price": price,
            "return": potential_return,
            "hours": hours_left,
            "days": days_left,
            "volume": volume,
            "whale_side": whale_side,
            "whale_pnl": whale_pnl,
            "whale_value": whale_value,
        })

# Sort by probability * return / sqrt(time)
for opp in opportunities:
    # Score: higher prob, higher return, shorter time
    opp["score"] = opp["price"] * opp["return"] / (opp["days"] + 1) ** 0.5

opportunities.sort(key=lambda x: x["score"], reverse=True)

print(f"Found {len(opportunities)} non-crypto opportunities\n")

# Categories
categories = {
    "politics": ["TRUMP", "BIDEN", "ELECTION", "PRESIDENT", "CONGRESS", "SENATE", "GOP", "DEM"],
    "sports": ["NFL", "NBA", "MLB", "SUPER BOWL", "GAME", "WIN", "CHAMPIONSHIP"],
    "tech": ["AI", "OPENAI", "GOOGLE", "APPLE", "TESLA", "ELON", "MUSK"],
    "world": ["UKRAINE", "RUSSIA", "CHINA", "WAR", "NATO", "EU"],
}

def categorize(q):
    q_upper = q.upper()
    for cat, keywords in categories.items():
        if any(kw in q_upper for kw in keywords):
            return cat
    return "other"

# Show by category
for cat in ["politics", "sports", "tech", "world", "other"]:
    cat_opps = [o for o in opportunities if categorize(o["question"]) == cat][:5]
    if cat_opps:
        print(f"\n### {cat.upper()} ###\n")
        for opp in cat_opps:
            whale = " [WHALE]" if opp["whale_side"] == opp["side"] else ""
            print(f"{opp['side']} @ ${opp['price']:.2f} | +{opp['return']:.0f}% | {opp['days']:.1f}d{whale}")
            print(f"  {opp['question']}")
            print()

# Best overall
print("=" * 70)
print("BEST OPPORTUNITIES (Score = Prob x Return / sqrt(Days))")
print("=" * 70)

for i, opp in enumerate(opportunities[:10], 1):
    whale = " [WHALE AGREES]" if opp["whale_side"] == opp["side"] else ""
    whale_info = f" (Whale PnL: ${opp['whale_pnl']:+,.0f})" if opp["whale_value"] > 0 else ""

    print(f"\n{i}. {opp['question']}")
    print(f"   {opp['side']} @ ${opp['price']:.2f} ({opp['price']*100:.0f}% prob)")
    print(f"   Return: +{opp['return']:.0f}% | Expires: {opp['days']:.1f} days")
    print(f"   Volume: ${opp['volume']:,.0f}{whale}{whale_info}")

    tokens = opp.get("tokens", "")
    if tokens and i <= 3:
        try:
            token_list = json.loads(tokens) if isinstance(tokens, str) else tokens
            idx = 0 if opp["side"] == "YES" else 1
            if len(token_list) > idx:
                print(f"   Token: {token_list[idx]}")
        except:
            pass
