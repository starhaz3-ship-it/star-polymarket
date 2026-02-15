"""
Polymarket BTC 5-Minute Order Book Liquidity Analysis
=====================================================
Analyzes bid/ask depth for BTC 5m Up/Down markets to find
optimal trade size for 24/7 maker bot operation.

Uses the same API patterns as run_maker.py for market discovery.
"""

import httpx
import json
import time
import re
import sys
from datetime import datetime, timezone
from collections import defaultdict

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

UA = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 15


def fetch_5m_events():
    """Fetch active, non-closed 5M events (same logic as run_maker.py)."""
    r = httpx.get(
        "https://gamma-api.polymarket.com/events",
        params={
            "tag_slug": "5M",
            "active": "true",
            "closed": "false",
            "limit": 200,
            "order": "endDate",
            "ascending": "true",
        },
        headers=UA,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def fetch_orderbook(token_id):
    """Fetch order book from CLOB API."""
    r = httpx.get(
        "https://clob.polymarket.com/book",
        params={"token_id": token_id},
        headers=UA,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def parse_duration_from_title(title):
    """Parse actual duration in minutes from market title.
    E.g. 'Bitcoin Up or Down - Feb 15, 1:30PM-1:35PM ET' -> 5"""
    m = re.search(r"(\d{1,2}):(\d{2})(AM|PM)-(\d{1,2}):(\d{2})(AM|PM)", title)
    if not m:
        return None
    h1, m1, p1 = int(m.group(1)), int(m.group(2)), m.group(3)
    h2, m2, p2 = int(m.group(4)), int(m.group(5)), m.group(6)
    t1 = (h1 % 12 + (12 if p1 == "PM" else 0)) * 60 + m1
    t2 = (h2 % 12 + (12 if p2 == "PM" else 0)) * 60 + m2
    dur = t2 - t1 if t2 > t1 else t2 + 1440 - t1
    return dur


def analyze_book_side(orders, side_name):
    """Analyze one side of the order book."""
    if not orders:
        return None

    levels = []
    for o in orders:
        price = float(o["price"])
        size = float(o["size"])
        levels.append((price, size))

    if side_name == "bids":
        levels.sort(key=lambda x: -x[0])
    else:
        levels.sort(key=lambda x: x[0])

    if not levels:
        return None

    best_price = levels[0][0]
    total_size = sum(s for _, s in levels)
    total_usd = sum(p * s for p, s in levels)

    depth = []
    cum_size = 0
    cum_usd = 0
    for price, size in levels:
        cum_size += size
        cum_usd += price * size
        depth.append(
            {
                "price": price,
                "size": size,
                "cum_size": cum_size,
                "cum_usd": cum_usd,
                "price_move": abs(price - best_price),
            }
        )

    impact = {}
    for threshold in [0.00, 0.01, 0.02, 0.03, 0.05, 0.10]:
        shares_available = 0
        usd_available = 0
        for d in depth:
            if d["price_move"] <= threshold + 0.0001:
                shares_available = d["cum_size"]
                usd_available = d["cum_usd"]
        impact[threshold] = {"shares": shares_available, "usd": usd_available}

    return {
        "best_price": best_price,
        "total_size": total_size,
        "total_usd": total_usd,
        "num_levels": len(levels),
        "depth": depth,
        "impact": impact,
    }


def main():
    print("=" * 80)
    print("POLYMARKET BTC 5-MINUTE ORDER BOOK LIQUIDITY ANALYSIS")
    now = datetime.now(timezone.utc)
    print(f"Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)

    # 1. Fetch events
    print("\n[1] Fetching active 5M events (closed=false, ascending endDate)...")
    events = fetch_5m_events()
    print(f"    Total 5M events returned: {len(events)}")

    # Filter for BTC only, future endDate, and actual 5-min duration
    btc_markets = []
    for event in events:
        title = event.get("title", "")
        if "bitcoin" not in title.lower() and "btc" not in title.lower():
            continue

        for mkt in event.get("markets", []):
            if mkt.get("closed", True):
                continue

            end_date_str = mkt.get("endDate", "")
            if end_date_str:
                try:
                    end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                    if end_dt < now:
                        continue  # Already expired
                except Exception:
                    pass

            # Check actual duration
            q = mkt.get("question", "") or title
            dur = parse_duration_from_title(q)
            if dur is not None and dur > 5:
                continue

            # Parse outcomes and token IDs
            outcomes = mkt.get("outcomes", [])
            token_ids = mkt.get("clobTokenIds", [])
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)

            if len(outcomes) != 2 or len(token_ids) != 2:
                continue

            # Identify Up/Down
            up_idx = down_idx = None
            for i, o in enumerate(outcomes):
                ol = str(o).lower()
                if ol in ("up", "yes"):
                    up_idx = i
                elif ol in ("down", "no"):
                    down_idx = i
            if up_idx is None or down_idx is None:
                continue

            btc_markets.append(
                {
                    "question": q,
                    "end_date": end_date_str,
                    "condition_id": mkt.get("conditionId", ""),
                    "market_id": mkt.get("id", ""),
                    "up_token": token_ids[up_idx],
                    "down_token": token_ids[down_idx],
                    "outcomes": outcomes,
                    "duration": dur or 5,
                }
            )

    print(f"    BTC 5m markets with future endDate: {len(btc_markets)}")
    if not btc_markets:
        print("\n    WARNING: No live BTC 5m markets found!")
        print("    This could mean:")
        print("    - Markets are between rounds (check back in a few minutes)")
        print("    - API issue")
        print("\n    Listing ALL non-closed 5M events for debugging:")
        for event in events[:10]:
            title = event.get("title", "N/A")
            mkts = event.get("markets", [])
            print(f"    - {title} ({len(mkts)} markets)")
            for m in mkts[:2]:
                q = m.get("question", "N/A")
                end = m.get("endDate", "N/A")
                closed = m.get("closed", "N/A")
                print(f"      {q} | end={end} | closed={closed}")
        return

    # 2. Fetch order books for multiple markets
    print(f"\n[2] Fetching order books for up to {min(len(btc_markets), 5)} markets...\n")

    all_analyses = []
    books_fetched = 0

    for i, mkt in enumerate(btc_markets[:5]):
        print("-" * 80)
        print(f"MARKET {i+1}: {mkt['question']}")
        print(f"  End: {mkt['end_date']}")
        mins_until = ""
        try:
            end_dt = datetime.fromisoformat(mkt["end_date"].replace("Z", "+00:00"))
            delta = (end_dt - now).total_seconds() / 60
            mins_until = f" ({delta:.1f} min from now)"
        except Exception:
            pass
        print(f"  Time remaining: {mins_until}")

        for side_label, token_id in [
            ("UP", mkt["up_token"]),
            ("DOWN", mkt["down_token"]),
        ]:
            try:
                book = fetch_orderbook(token_id)
            except httpx.HTTPStatusError as e:
                print(f"  {side_label}: HTTP {e.response.status_code} (market may be expired/resolved)")
                continue
            except Exception as e:
                print(f"  {side_label}: Error: {e}")
                continue

            bids_raw = book.get("bids", [])
            asks_raw = book.get("asks", [])
            books_fetched += 1

            bid_analysis = analyze_book_side(bids_raw, "bids")
            ask_analysis = analyze_book_side(asks_raw, "asks")

            print(f"\n  === {side_label} TOKEN ===")
            print(f"  Bids: {len(bids_raw)} levels | Asks: {len(asks_raw)} levels")

            # ASK side = cost to BUY this token
            if ask_analysis:
                print(f"\n  ASK SIDE (cost to BUY {side_label}):")
                print(f"    Best ask: ${ask_analysis['best_price']:.2f}")
                print(
                    f"    Total ask liquidity: {ask_analysis['total_size']:.1f} shares"
                    f" (${ask_analysis['total_usd']:.2f})"
                )
                print(f"    Price levels: {ask_analysis['num_levels']}")
                print(f"\n    {'Price':>8} {'Size':>10} {'CumSize':>10} {'CumUSD':>10} {'Slip':>8}")
                for d in ask_analysis["depth"][:20]:
                    print(
                        f"    ${d['price']:.2f} {d['size']:>10.1f} "
                        f"{d['cum_size']:>10.1f} ${d['cum_usd']:>9.2f} "
                        f"+{d['price_move']:.2f}"
                    )

                print(f"\n    IMPACT ANALYSIS (buying {side_label}):")
                for threshold in [0.00, 0.01, 0.02, 0.03, 0.05, 0.10]:
                    data = ask_analysis["impact"].get(threshold, {"shares": 0, "usd": 0})
                    if data["shares"] > 0:
                        print(
                            f"      At best ask +${threshold:.2f}: "
                            f"{data['shares']:.0f} shares (${data['usd']:.2f})"
                        )
            else:
                print(f"  NO ASKS for {side_label}")

            # BID side = exit liquidity
            if bid_analysis:
                print(f"\n  BID SIDE (exit/sell liquidity for {side_label}):")
                print(f"    Best bid: ${bid_analysis['best_price']:.2f}")
                print(
                    f"    Total bid liquidity: {bid_analysis['total_size']:.1f} shares"
                    f" (${bid_analysis['total_usd']:.2f})"
                )
                print(f"    Price levels: {bid_analysis['num_levels']}")
                print(f"\n    {'Price':>8} {'Size':>10} {'CumSize':>10} {'CumUSD':>10} {'Slip':>8}")
                for d in bid_analysis["depth"][:20]:
                    print(
                        f"    ${d['price']:.2f} {d['size']:>10.1f} "
                        f"{d['cum_size']:>10.1f} ${d['cum_usd']:>9.2f} "
                        f"-{d['price_move']:.2f}"
                    )
            else:
                print(f"  NO BIDS for {side_label}")

            # Spread
            spread_val = None
            if bid_analysis and ask_analysis:
                spread_val = ask_analysis["best_price"] - bid_analysis["best_price"]
                print(f"\n  SPREAD: ${spread_val:.2f}")

            all_analyses.append(
                {
                    "market": mkt["question"],
                    "side": side_label,
                    "bid": bid_analysis,
                    "ask": ask_analysis,
                    "spread": spread_val,
                }
            )

        time.sleep(0.5)

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 80}")
    print("SUMMARY: OPTIMAL TRADE SIZE ANALYSIS")
    print(f"{'=' * 80}")
    print(f"\nBooks fetched: {books_fetched} ({len(all_analyses)} token sides)")

    if not all_analyses:
        print("\nNo order book data collected. Cannot determine optimal size.")
        return

    # Separate UP and DOWN analyses
    up_analyses = [a for a in all_analyses if a["side"] == "UP" and a["ask"]]
    down_analyses = [a for a in all_analyses if a["side"] == "DOWN" and a["ask"]]

    print(f"\nUP token books with asks: {len(up_analyses)}")
    print(f"DOWN token books with asks: {len(down_analyses)}")

    # Aggregate across all books
    ask_at_best = []
    ask_within_1c = []
    ask_within_2c = []
    ask_within_3c = []
    ask_within_5c = []
    total_ask_sizes = []
    spreads = []
    best_ask_prices = []

    for a in all_analyses:
        if a["ask"]:
            ask = a["ask"]
            total_ask_sizes.append(ask["total_size"])
            best_ask_prices.append(ask["best_price"])
            ask_at_best.append(ask["impact"].get(0.00, {"shares": 0})["shares"])
            ask_within_1c.append(ask["impact"].get(0.01, {"shares": 0})["shares"])
            ask_within_2c.append(ask["impact"].get(0.02, {"shares": 0})["shares"])
            ask_within_3c.append(ask["impact"].get(0.03, {"shares": 0})["shares"])
            ask_within_5c.append(ask["impact"].get(0.05, {"shares": 0})["shares"])
        if a["spread"] is not None:
            spreads.append(a["spread"])

    avg_ask_price = sum(best_ask_prices) / len(best_ask_prices) if best_ask_prices else 0.50

    print(f"\n--- SPREADS ---")
    if spreads:
        print(f"  Min: ${min(spreads):.2f}")
        print(f"  Max: ${max(spreads):.2f}")
        print(f"  Avg: ${sum(spreads)/len(spreads):.2f}")
    else:
        print("  No spread data")

    print(f"\n--- TOTAL ASK-SIDE LIQUIDITY PER BOOK ---")
    if total_ask_sizes:
        print(f"  Min: {min(total_ask_sizes):.0f} shares (${min(total_ask_sizes)*avg_ask_price:.2f})")
        print(f"  Max: {max(total_ask_sizes):.0f} shares (${max(total_ask_sizes)*avg_ask_price:.2f})")
        print(f"  Avg: {sum(total_ask_sizes)/len(total_ask_sizes):.0f} shares")

    print(f"\n--- MARKET IMPACT: SHARES AVAILABLE AT EACH SLIPPAGE LEVEL ---")

    def summarize(label, data_list):
        if not data_list or max(data_list) == 0:
            print(f"  {label}: No liquidity")
            return 0
        nonzero = [x for x in data_list if x > 0]
        if not nonzero:
            print(f"  {label}: No liquidity")
            return 0
        mn = min(nonzero)
        mx = max(nonzero)
        avg = sum(nonzero) / len(nonzero)
        print(f"  {label}:")
        print(f"    Min: {mn:.0f} shares (${mn*avg_ask_price:.2f})")
        print(f"    Max: {mx:.0f} shares (${mx*avg_ask_price:.2f})")
        print(f"    Avg: {avg:.0f} shares (${avg*avg_ask_price:.2f})")
        print(f"    Observations with liquidity: {len(nonzero)}/{len(data_list)}")
        return mn

    min_best = summarize("AT best ask (0 slippage)", ask_at_best)
    min_1c = summarize("Within +$0.01 of best ask", ask_within_1c)
    min_2c = summarize("Within +$0.02 of best ask", ask_within_2c)
    min_3c = summarize("Within +$0.03 of best ask", ask_within_3c)
    min_5c = summarize("Within +$0.05 of best ask", ask_within_5c)

    print(f"\n{'=' * 80}")
    print("RECOMMENDATION: MAXIMUM TRADE SIZE")
    print(f"{'=' * 80}")

    print(f"""
Average best ask price: ${avg_ask_price:.2f}
Average spread: ${sum(spreads)/len(spreads):.2f}""" if spreads else "")

    print(f"""
For 24/7 reliable fills with minimal market impact:

  TIER 1 - ZERO SLIPPAGE (buy at best ask only):
    Max shares per side: {min_best:.0f}
    Max USD per side:    ${min_best * avg_ask_price:.2f}
    Both sides total:    ${min_best * avg_ask_price * 2:.2f}

  TIER 2 - SMALL IMPACT (<= $0.01 slippage):
    Max shares per side: {min_1c:.0f}
    Max USD per side:    ${min_1c * avg_ask_price:.2f}
    Both sides total:    ${min_1c * avg_ask_price * 2:.2f}

  TIER 3 - MODERATE IMPACT (<= $0.02 slippage):
    Max shares per side: {min_2c:.0f}
    Max USD per side:    ${min_2c * avg_ask_price:.2f}
    Both sides total:    ${min_2c * avg_ask_price * 2:.2f}

  TIER 4 - AGGRESSIVE (<= $0.03 slippage):
    Max shares per side: {min_3c:.0f}
    Max USD per side:    ${min_3c * avg_ask_price:.2f}
    Both sides total:    ${min_3c * avg_ask_price * 2:.2f}

  NOTE: These are WORST-CASE minimums across {len(all_analyses)} sampled books.
  Average liquidity is typically higher. Night hours (0-8 UTC) may be thinner.

  MAKER BOT CONTEXT:
  - Current SIZE_PER_SIDE_USD = $25
  - Maker places LIMIT orders (not market orders), so actual impact is lower
  - But limit orders may not fill if book is thin
  - For MARKET orders (aggressive), use Tier 1-2 sizing
  - For LIMIT orders (passive), can use Tier 3-4 but expect partial fills
""")


if __name__ == "__main__":
    main()
