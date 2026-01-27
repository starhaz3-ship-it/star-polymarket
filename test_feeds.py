"""Test the arbitrage bot components."""

from arbitrage.polymarket_feed import PolymarketFeed, BTCMarket
from arbitrage.spot_feed import get_btc_price
from arbitrage.detector import ArbitrageDetector

def main():
    # Get spot price
    spot = get_btc_price()
    print(f"BTC Spot: ${spot:,.2f}")
    print()

    # Fetch markets
    feed = PolymarketFeed()
    markets = feed.fetch_btc_markets()

    print(f"Found {len(markets)} 15-min BTC markets")
    print()

    if not markets:
        print("No 15-min markets found. Checking all BTC markets...")
        # Try broader search
        import httpx
        client = httpx.Client(timeout=30.0)
        r = client.get("https://gamma-api.polymarket.com/markets", params={
            "active": "true",
            "closed": "false",
            "limit": 20,
        })
        data = r.json()
        btc_markets = [m for m in data if "BTC" in m.get("question", "") or "Bitcoin" in m.get("question", "")]
        print(f"Found {len(btc_markets)} total BTC markets:")
        for m in btc_markets[:5]:
            print(f"  - {m.get('question', '')[:60]}")
        return

    # Update prices and scan
    for m in markets[:10]:
        feed.update_market_prices(m)
        print(f"{m.question[:55]}...")
        print(f"  Strike: ${m.strike_price:,.0f} | YES: {m.yes_price:.3f} | NO: {m.no_price:.3f}")
        print(f"  Time: {m.time_remaining_sec:.0f}s | Gap: {m.arbitrage_gap:.4f}")
        print()

    # Run detector
    detector = ArbitrageDetector()
    signals = detector.scan_all(markets, spot)

    print(f"\nFound {len(signals)} arbitrage signals:")
    for s in signals[:5]:
        print(f"  {s}")

if __name__ == "__main__":
    main()
