"""
Reversal Monitor — Track 5M BTC final-seconds pricing & reversal rates.
Tests the theory: buy "dead" side at $0.01-0.10 in last seconds = profitable?

Run: python -u reversal_monitor.py > reversal_monitor.log 2>&1 &
"""
import asyncio, httpx, json, time, os
from datetime import datetime, timezone
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "reversal_monitor_results.json"
SCAN_INTERVAL = 5        # seconds between market scans
FINAL_WINDOW = 30        # monitor last N seconds before close
SNAPSHOT_INTERVAL = 2    # seconds between order book snapshots in final window

# CLOB client for order books
from py_clob_client.client import ClobClient
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

def init_clob():
    host = "https://clob.polymarket.com"
    key = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    # Decrypt if needed
    if key.startswith("ENC:"):
        from arbitrage.executor import Executor
        e = Executor()
        return e.client if e._initialized else None
    return ClobClient(host, chain_id=137)

def load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {"markets": [], "summary": {}}

def save_results(data):
    RESULTS_FILE.write_text(json.dumps(data, indent=2, default=str))


async def discover_5m_btc():
    """Find active 5M BTC Up/Down markets."""
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://gamma-api.polymarket.com/events",
            params={"tag_slug": "5M", "active": "true", "closed": "false",
                    "limit": 50, "order": "endDate", "ascending": "true"},
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code != 200:
            return []

        pairs = []
        for event in r.json():
            title = event.get("title", "")
            if "bitcoin" not in title.lower() and "btc" not in title.lower():
                continue
            markets = event.get("markets", [])
            if len(markets) != 2:
                continue

            up_market = down_market = None
            for m in markets:
                outcome = m.get("groupItemTitle", m.get("outcome", "")).lower()
                if "up" in outcome:
                    up_market = m
                elif "down" in outcome:
                    down_market = m

            if not up_market or not down_market:
                continue

            end_str = event.get("endDate", up_market.get("endDate", ""))
            if not end_str:
                continue

            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            secs_left = (end_dt - now).total_seconds()

            pairs.append({
                "event_id": event.get("id", ""),
                "title": title,
                "end_dt": end_dt,
                "secs_left": secs_left,
                "up_token": up_market.get("clobTokenIds", ["", ""])[0] if up_market.get("clobTokenIds") else "",
                "down_token": down_market.get("clobTokenIds", ["", ""])[0] if down_market.get("clobTokenIds") else "",
                "condition_id": up_market.get("conditionId", ""),
                "up_id": up_market.get("id", ""),
                "down_id": down_market.get("id", ""),
            })
        return pairs


async def get_book(clob, token_id):
    """Get best bid/ask for a token."""
    try:
        book = clob.get_order_book(token_id)
        bids = book.get("bids", []) if isinstance(book, dict) else getattr(book, 'bids', [])
        asks = book.get("asks", []) if isinstance(book, dict) else getattr(book, 'asks', [])
        best_bid = float(bids[0]["price"] if isinstance(bids[0], dict) else bids[0].price) if bids else 0.0
        best_ask = float(asks[0]["price"] if isinstance(asks[0], dict) else asks[0].price) if asks else 1.0
        return {"bid": best_bid, "ask": best_ask}
    except Exception:
        return None


async def check_outcome(condition_id, up_id, down_id):
    """Check which side won after resolution."""
    await asyncio.sleep(15)  # Wait for resolution to propagate
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # Try UP market
                r = await client.get(f"https://gamma-api.polymarket.com/markets/{up_id}",
                                     headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200:
                    data = r.json()
                    price = float(data.get("lastTradePrice", data.get("outcomePrices", "0.5")))
                    if price > 0.9:
                        return "UP"
                    elif price < 0.1:
                        return "DOWN"
                # Try DOWN market
                r2 = await client.get(f"https://gamma-api.polymarket.com/markets/{down_id}",
                                      headers={"User-Agent": "Mozilla/5.0"})
                if r2.status_code == 200:
                    data2 = r2.json()
                    price2 = float(data2.get("lastTradePrice", data2.get("outcomePrices", "0.5")))
                    if price2 > 0.9:
                        return "DOWN"
                    elif price2 < 0.1:
                        return "UP"
        except Exception:
            pass
        await asyncio.sleep(10)
    return "UNKNOWN"


async def monitor_market(clob, market):
    """Monitor a single 5M BTC market in its final seconds."""
    title = market["title"]
    end_dt = market["end_dt"]
    snapshots = []

    print(f"\n[MONITOR] {title}")
    print(f"  Watching final {FINAL_WINDOW}s...")

    # Wait until final window
    while True:
        now = datetime.now(timezone.utc)
        secs_left = (end_dt - now).total_seconds()
        if secs_left <= FINAL_WINDOW:
            break
        if secs_left <= 0:
            return None
        await asyncio.sleep(min(SCAN_INTERVAL, secs_left - FINAL_WINDOW))

    # Snapshot order books every 2s in final window
    while True:
        now = datetime.now(timezone.utc)
        secs_left = (end_dt - now).total_seconds()
        if secs_left <= -5:
            break

        up_book = await get_book(clob, market["up_token"])
        dn_book = await get_book(clob, market["down_token"])

        if up_book and dn_book:
            cheap_side = "UP" if up_book["ask"] < dn_book["ask"] else "DOWN"
            cheap_ask = min(up_book["ask"], dn_book["ask"])
            expensive_ask = max(up_book["ask"], dn_book["ask"])

            snap = {
                "secs_left": round(secs_left, 1),
                "up_ask": up_book["ask"],
                "up_bid": up_book["bid"],
                "down_ask": dn_book["ask"],
                "down_bid": dn_book["bid"],
                "cheap_side": cheap_side,
                "cheap_ask": cheap_ask,
            }
            snapshots.append(snap)
            print(f"  [{secs_left:5.1f}s] UP: ${up_book['ask']:.2f}/{up_book['bid']:.2f} | "
                  f"DN: ${dn_book['ask']:.2f}/{dn_book['bid']:.2f} | "
                  f"Cheap: {cheap_side} @ ${cheap_ask:.2f}")

        await asyncio.sleep(SNAPSHOT_INTERVAL)

    if not snapshots:
        return None

    # Check outcome
    print(f"  Checking outcome...")
    winner = await check_outcome(market["condition_id"], market["up_id"], market["down_id"])

    # Analyze: was the cheap side the winner?
    last_snap = snapshots[-1]
    cheapest_snap = min(snapshots, key=lambda s: s["cheap_ask"])
    reversal = (cheapest_snap["cheap_side"] == winner)

    result = {
        "title": title,
        "end_time": end_dt.isoformat(),
        "winner": winner,
        "snapshots": len(snapshots),
        "cheapest_side": cheapest_snap["cheap_side"],
        "cheapest_ask": cheapest_snap["cheap_ask"],
        "cheapest_at_secs_left": cheapest_snap["secs_left"],
        "last_cheap_side": last_snap["cheap_side"],
        "last_cheap_ask": last_snap["cheap_ask"],
        "reversal": reversal,
        "all_snapshots": snapshots,
    }

    tag = "REVERSAL!" if reversal else "no reversal"
    print(f"  >> Winner: {winner} | Cheap side: {cheapest_snap['cheap_side']} @ ${cheapest_snap['cheap_ask']:.2f} | {tag}")

    return result


async def main():
    print("=" * 60)
    print("REVERSAL MONITOR — 5M BTC Final-Seconds Tracker")
    print(f"Monitoring last {FINAL_WINDOW}s of each 5M BTC market")
    print(f"Snapshots every {SNAPSHOT_INTERVAL}s | Results: {RESULTS_FILE.name}")
    print("=" * 60)

    clob = init_clob()
    if not clob:
        print("[FATAL] Could not init CLOB client")
        return

    results = load_results()
    tracked = set()  # event_ids we've already monitored

    while True:
        try:
            markets = await discover_5m_btc()
            for m in markets:
                eid = m["event_id"]
                if eid in tracked:
                    continue
                if m["secs_left"] > FINAL_WINDOW + 60:
                    # Too early, skip for now
                    continue
                if m["secs_left"] < 0:
                    tracked.add(eid)
                    continue

                tracked.add(eid)
                result = await monitor_market(clob, m)
                if result:
                    results["markets"].append(result)

                    # Update summary
                    total = len(results["markets"])
                    reversals = sum(1 for r in results["markets"] if r.get("reversal"))
                    cheap_under_10 = [r for r in results["markets"] if r["cheapest_ask"] <= 0.10]
                    rev_under_10 = sum(1 for r in cheap_under_10 if r.get("reversal"))
                    cheap_under_05 = [r for r in results["markets"] if r["cheapest_ask"] <= 0.05]
                    rev_under_05 = sum(1 for r in cheap_under_05 if r.get("reversal"))

                    results["summary"] = {
                        "total_markets": total,
                        "total_reversals": reversals,
                        "reversal_rate": f"{reversals/total*100:.1f}%" if total else "0%",
                        "markets_cheap_under_10c": len(cheap_under_10),
                        "reversals_under_10c": rev_under_10,
                        "rate_under_10c": f"{rev_under_10/len(cheap_under_10)*100:.1f}%" if cheap_under_10 else "N/A",
                        "markets_cheap_under_5c": len(cheap_under_05),
                        "reversals_under_5c": rev_under_05,
                        "rate_under_5c": f"{rev_under_05/len(cheap_under_05)*100:.1f}%" if cheap_under_05 else "N/A",
                    }
                    save_results(results)

                    print(f"\n  [SUMMARY] {total} markets | {reversals} reversals ({results['summary']['reversal_rate']}) | "
                          f"<10c: {rev_under_10}/{len(cheap_under_10)} | <5c: {rev_under_05}/{len(cheap_under_05)}")

            await asyncio.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            print("\n[MONITOR] Stopped.")
            break
        except Exception as e:
            print(f"[ERR] {e}")
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
