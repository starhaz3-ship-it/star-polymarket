"""Check Polymarket account: positions, balances, open orders."""

import json
import sys
import os
from pathlib import Path
from functools import partial

from dotenv import load_dotenv

print = partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()


def check_account():
    from arbitrage.executor import Executor
    executor = Executor()

    if not executor._initialized:
        print("ERROR: Executor not initialized")
        return

    client = executor.client

    print("=" * 70)
    print("POLYMARKET ACCOUNT STATUS")
    print("=" * 70)

    # 1. Check USDC balance
    try:
        # Try to get balance info
        balance = client.get_balance_allowance()
        if balance:
            print(f"\nBALANCE/ALLOWANCE:")
            if isinstance(balance, dict):
                for k, v in balance.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {balance}")
    except Exception as e:
        print(f"\nBALANCE: Could not fetch ({e})")

    # 2. Check open orders
    try:
        orders = client.get_orders()
        print(f"\nOPEN ORDERS: {len(orders) if orders else 0}")
        if orders:
            for o in orders[:20]:
                side = o.get("side", "?")
                price = o.get("price", "?")
                size = o.get("original_size", o.get("size", "?"))
                status = o.get("status", "?")
                market = o.get("market", o.get("token_id", "?"))
                oid = o.get("id", o.get("orderID", ""))[:16]
                print(f"  {side} {size} @ ${price} | status: {status} | {oid}...")
    except Exception as e:
        print(f"\nORDERS: Could not fetch ({e})")

    # 3. Check trades/fills
    try:
        trades = client.get_trades()
        print(f"\nRECENT TRADES/FILLS: {len(trades) if trades else 0}")
        if trades:
            for t in trades[:20]:
                side = t.get("side", "?")
                price = t.get("price", "?")
                size = t.get("size", "?")
                status = t.get("status", t.get("match_time", "?"))
                market = t.get("market", "")[:16] if t.get("market") else ""
                print(f"  {side} {size} @ ${price} | {status} | {market}...")
    except Exception as e:
        print(f"\nTRADES: Could not fetch ({e})")

    # 4. Try to get market positions via API
    try:
        # The CLOB client may have different methods
        # Try various approaches
        methods = ['get_positions', 'get_balances', 'get_all_balances']
        for method_name in methods:
            method = getattr(client, method_name, None)
            if method:
                try:
                    result = method()
                    print(f"\n{method_name.upper()}: {len(result) if result else 0}")
                    if result:
                        for item in (result[:20] if isinstance(result, list) else [result]):
                            print(f"  {item}")
                except Exception as e2:
                    print(f"\n{method_name}: Error - {e2}")
    except Exception as e:
        print(f"\nPOSITIONS: Could not fetch ({e})")

    # 5. Try the gamma API for positions
    try:
        import httpx
        proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
        if proxy:
            print(f"\nPROXY WALLET: {proxy}")
            r = httpx.get(
                f"https://gamma-api.polymarket.com/query",
                params={
                    "query": json.dumps({
                        "type": "positions",
                        "address": proxy,
                        "limit": 50
                    })
                },
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15
            )
            if r.status_code == 200:
                positions = r.json()
                print(f"  Positions found: {len(positions) if positions else 0}")
                if positions:
                    for p in positions[:20]:
                        print(f"  {p}")
    except Exception as e:
        print(f"\nGAMMA POSITIONS: {e}")

    # 6. Try data API for portfolio
    try:
        import httpx
        proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
        if proxy:
            r = httpx.get(
                f"https://data-api.polymarket.com/positions",
                params={"user": proxy, "sizeThreshold": "0"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15
            )
            if r.status_code == 200:
                positions = r.json()
                print(f"\nDATA API POSITIONS: {len(positions) if positions else 0}")
                if positions:
                    total_value = 0
                    for p in positions:
                        title = p.get("title", p.get("market", ""))[:50]
                        size = p.get("size", 0)
                        cur_price = p.get("curPrice", p.get("currentPrice", 0))
                        outcome = p.get("outcome", p.get("side", "?"))
                        initial = p.get("initialValue", 0)
                        current = p.get("currentValue", 0)
                        pnl = p.get("pnl", 0)
                        value = float(size) * float(cur_price) if size and cur_price else 0
                        total_value += value
                        print(f"  {outcome} | size:{size} @ ${cur_price} | "
                              f"init:${initial} cur:${current} pnl:${pnl} | {title}")
                    print(f"\n  TOTAL POSITION VALUE: ${total_value:.2f}")
            else:
                print(f"\nDATA API: Status {r.status_code}")
    except Exception as e:
        print(f"\nDATA API: {e}")

    # 7. Check profile/balance via profile API
    try:
        import httpx
        proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
        if proxy:
            r = httpx.get(
                f"https://data-api.polymarket.com/profile",
                params={"user": proxy},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15
            )
            if r.status_code == 200:
                profile = r.json()
                print(f"\nPROFILE:")
                if isinstance(profile, dict):
                    for k in ['balance', 'portfolioValue', 'totalDeposits',
                              'totalWithdrawals', 'pnl', 'volume', 'marketsTraded']:
                        if k in profile:
                            print(f"  {k}: {profile[k]}")
                else:
                    print(f"  {profile}")
    except Exception as e:
        print(f"\nPROFILE: {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    check_account()
