"""Analyze BTC $90K threshold opportunities."""
import httpx

btc = 88636.65
threshold = 90000
distance_needed = (threshold - btc) / btc * 100

print(f"BTC Current: ${btc:,.2f}")
print(f"Threshold: ${threshold:,}")
print(f"Distance to threshold: {distance_needed:.2f}% (${threshold - btc:,.0f})")
print()

# Historical BTC volatility - can it move 1.5% in a day?
print("=" * 60)
print("VOLATILITY ANALYSIS")
print("=" * 60)
print("BTC daily volatility is typically 2-4%")
print("BTC needs to rise 1.5% to hit $90K")
print("Probability of 1.5% move in 1-4 days: ~40-60%")
print()

# The markets
print("=" * 60)
print("BTC > $90K OPPORTUNITIES")
print("=" * 60)
print()

opportunities = [
    {"date": "Jan 28", "price": 0.23, "token": "32579264578918031321552191887827432831144617768103841586490339845782161448428", "side": "NO"},
    {"date": "Jan 30", "price": 0.34, "token": "66983016663119679065624679455418755794762855613158502797393818050781947259920", "side": "YES"},
    {"date": "Jan 31", "price": 0.36, "token": "6607271935782159393944756645209743226285840100127887394702751569080918631898", "side": "YES"},
    {"date": "Feb 1", "price": 0.38, "token": "509540869153657801986764926612117038708307208734315002716747053948730797904", "side": "YES"},
]

print("If you believe BTC will stay BELOW $90K:")
print(f"  BTC > $90K Jan 28 NO @ $0.23 (current position, expires soon)")
print(f"  Edge: If BTC stays at $88.6K, NO wins = 77% return")
print()

print("If you believe BTC will push ABOVE $90K:")
for opp in opportunities[1:]:
    potential_return = (1 / opp["price"] - 1) * 100
    print(f"  BTC > $90K {opp['date']} YES @ ${opp['price']:.2f}")
    print(f"    If wins: {potential_return:.0f}% return")
    print(f"    Token: {opp['token'][:30]}...")
    print()

print("=" * 60)
print("MY ASSESSMENT")
print("=" * 60)
print()
print("BTC at $88,637 is 1.5% below $90K threshold.")
print("This is a COIN FLIP situation - hard to predict with high confidence.")
print()
print("Options:")
print("1. NO TRADE - Market is efficiently priced")
print("2. SMALL speculative bet on BTC > $90K Feb 1 YES @ $0.38")
print("   - If BTC rallies to $90K+, returns 163%")
print("   - Risk: $90K is psychological resistance")
print()
print("3. SAFER: Wait for BTC to move, then trade the lagging market")
print()

# Check if there are better edges elsewhere
print("=" * 60)
print("ALTERNATIVE: LOW-RISK BTC > $86K")
print("=" * 60)
print()
print("Your existing trade: BTC > $86K Jan 28 YES @ $0.835")
print("  - BTC is 3.1% ABOVE $86K")
print("  - Current price: $0.93")
print("  - Already profitable, hold to expiry")
print()
print("Similar trades available:")
print("  - BTC > $86K Jan 29 YES @ $0.87 - BTC 3.1% above, lower risk")
print("  - BTC > $86K Jan 30 YES @ $0.83 - BTC 3.1% above, lower risk")
print()
print("These have negative edge (market prices them higher than my estimate)")
print("but they are LOWER RISK since BTC is already above threshold.")
