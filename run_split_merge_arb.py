"""
Split+Merge Arbitrage Bot for Polymarket 15-Minute Crypto Markets

Scans for mispricings where YES + NO != $1.00 and executes:
  - Buy+Merge: YES_ask + NO_ask < $1.00 → buy both, merge to USDC
  - Split+Sell: YES_bid + NO_bid > $1.00 → split USDC, sell both tokens

All split/merge operations are GASLESS via Polymarket's relayer (Builder API).
"""
import sys
import asyncio
import json
import time
import os
import math
from datetime import datetime, timezone
from pathlib import Path
from functools import partial

import httpx
from dotenv import load_dotenv

# Force unbuffered output
print = partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
SCAN_INTERVAL = 10           # Seconds between scans
MIN_PROFIT_PER_UNIT = 0.005  # $0.005 min edge per $1 unit after fees
MAX_POSITION_SIZE = 50.0     # Max $ per arb trade
MIN_SHARES = 10              # Min shares (orderbook depth threshold)
RESULTS_FILE = Path(__file__).parent / "arb_results.json"

ASSETS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
}


def _init_clob_client():
    """Initialize ClobClient with Magic wallet credentials."""
    from py_clob_client.client import ClobClient
    from crypto_utils import decrypt_key

    password = os.getenv("POLYMARKET_PASSWORD", "")
    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if pk.startswith("ENC:"):
        pk = decrypt_key(pk[4:], os.getenv("POLYMARKET_KEY_SALT", ""), password)
    if not pk:
        raise RuntimeError("No private key available")

    proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
    client = ClobClient(
        host="https://clob.polymarket.com",
        key=pk,
        chain_id=137,
        signature_type=1,
        funder=proxy if proxy else None,
    )
    creds = client.derive_api_key()
    client.set_api_creds(creds)
    return client


def _init_poly_web3(pk=None):
    """Initialize PolyWeb3Service for gasless split/merge."""
    from py_clob_client.client import ClobClient
    from py_builder_relayer_client.client import RelayClient
    from py_builder_signing_sdk.config import BuilderConfig
    from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds
    from poly_web3 import PolyWeb3Service
    from crypto_utils import decrypt_key

    password = os.getenv("POLYMARKET_PASSWORD", "")
    if pk is None:
        pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        if pk.startswith("ENC:"):
            pk = decrypt_key(pk[4:], os.getenv("POLYMARKET_KEY_SALT", ""), password)

    proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
    client = ClobClient(
        host="https://clob.polymarket.com", key=pk, chain_id=137,
        signature_type=1, funder=proxy if proxy else None,
    )
    creds = client.derive_api_key()
    client.set_api_creds(creds)

    builder_key = decrypt_key(os.getenv("POLY_BUILDER_API_KEY", "")[4:],
                              os.getenv("POLY_BUILDER_API_KEY_SALT", ""), password)
    builder_secret = decrypt_key(os.getenv("POLY_BUILDER_SECRET", "")[4:],
                                 os.getenv("POLY_BUILDER_SECRET_SALT", ""), password)
    builder_pass = decrypt_key(os.getenv("POLY_BUILDER_PASSPHRASE", "")[4:],
                               os.getenv("POLY_BUILDER_PASSPHRASE_SALT", ""), password)

    relay = RelayClient(
        relayer_url="https://relayer-v2.polymarket.com", chain_id=137,
        private_key=pk,
        builder_config=BuilderConfig(
            local_builder_creds=BuilderApiKeyCreds(
                key=builder_key, secret=builder_secret, passphrase=builder_pass)),
    )
    return PolyWeb3Service(clob_client=client, relayer_client=relay,
                           rpc_url="https://polygon-bor.publicnode.com")


def parse_market(m):
    """Extract token IDs and outcomes from a market dict."""
    token_ids = m.get("clobTokenIds", "[]")
    if isinstance(token_ids, str):
        token_ids = json.loads(token_ids)
    outcomes = m.get("outcomes", [])
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)
    condition_id = m.get("conditionId", "")
    question = m.get("question", m.get("title", ""))
    end_date = m.get("endDate", "")

    if len(token_ids) != 2 or len(outcomes) != 2:
        return None

    return {
        "condition_id": condition_id,
        "question": question,
        "end_date": end_date,
        "token_0": token_ids[0],  # First outcome token
        "token_1": token_ids[1],  # Second outcome token
        "outcome_0": str(outcomes[0]).lower(),
        "outcome_1": str(outcomes[1]).lower(),
    }


class SplitMergeArb:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.clob = _init_clob_client()
        self.web3_service = None if dry_run else _init_poly_web3()

        # Stats
        self.scans = 0
        self.opportunities_found = 0
        self.arbs_executed = 0
        self.total_profit = 0.0
        self.trades = []
        self.start_time = datetime.now(timezone.utc).isoformat()

        # Load existing results
        self._load_results()

    def _load_results(self):
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.arbs_executed = data.get("arbs_executed", 0)
                self.total_profit = data.get("total_profit", 0.0)
                self.trades = data.get("trades", [])
            except Exception:
                pass

    def _save_results(self):
        data = {
            "start_time": self.start_time,
            "last_update": datetime.now(timezone.utc).isoformat(),
            "scans": self.scans,
            "opportunities_found": self.opportunities_found,
            "arbs_executed": self.arbs_executed,
            "total_profit": round(self.total_profit, 4),
            "trades": self.trades[-100:],  # Keep last 100
        }
        RESULTS_FILE.write_text(json.dumps(data, indent=2))

    async def fetch_markets(self, client: httpx.AsyncClient):
        """Fetch all active 15-min crypto markets."""
        markets = []
        try:
            r = await client.get(
                "https://gamma-api.polymarket.com/events",
                params={"tag_slug": "15M", "active": "true", "closed": "false", "limit": 50},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            if r.status_code != 200:
                return markets

            for event in r.json():
                title = event.get("title", "").lower()
                asset = None
                for a, keywords in ASSETS.items():
                    if any(kw in title for kw in keywords):
                        asset = a
                        break
                if not asset:
                    continue
                for m in event.get("markets", []):
                    if m.get("closed"):
                        continue
                    parsed = parse_market(m)
                    if parsed:
                        parsed["asset"] = asset
                        markets.append(parsed)
        except Exception as e:
            print(f"[API] Market fetch error: {e}")
        return markets

    def get_orderbook(self, token_id):
        """Get best bid/ask for a token from the CLOB orderbook."""
        try:
            book = self.clob.get_order_book(token_id)
            best_bid = float(book.bids[0].price) if book.bids else 0.0
            best_ask = float(book.asks[0].price) if book.asks else 1.0
            bid_size = float(book.bids[0].size) if book.bids else 0.0
            ask_size = float(book.asks[0].size) if book.asks else 0.0
            return best_bid, best_ask, bid_size, ask_size
        except Exception:
            return 0.0, 1.0, 0.0, 0.0

    def get_fee_rate(self, token_id):
        """Get fee rate in decimal (e.g., 0.02 = 2%)."""
        try:
            bps = self.clob.get_fee_rate_bps(token_id)
            return int(bps) / 10000.0
        except Exception:
            return 0.02  # Conservative 2% default

    def check_buy_merge(self, market, ob0, ob1):
        """Check for Buy+Merge opportunity: YES_ask + NO_ask < 1.0."""
        _, ask_0, _, ask_size_0 = ob0
        _, ask_1, _, ask_size_1 = ob1

        if ask_size_0 < MIN_SHARES or ask_size_1 < MIN_SHARES:
            return None

        fee_0 = self.get_fee_rate(market["token_0"])
        fee_1 = self.get_fee_rate(market["token_1"])

        # Cost to buy 1 unit of each side + fees
        cost_per_unit = ask_0 * (1 + fee_0) + ask_1 * (1 + fee_1)
        profit_per_unit = 1.0 - cost_per_unit

        if profit_per_unit < MIN_PROFIT_PER_UNIT:
            return None

        # Size limited by available liquidity and max position
        max_shares = min(ask_size_0, ask_size_1, MAX_POSITION_SIZE / max(ask_0 + ask_1, 0.01))
        max_shares = math.floor(max_shares)
        if max_shares < MIN_SHARES:
            return None

        total_cost = max_shares * (ask_0 + ask_1)
        total_profit = max_shares * profit_per_unit

        return {
            "type": "BUY_MERGE",
            "market": market,
            "ask_0": ask_0, "ask_1": ask_1,
            "fee_0": fee_0, "fee_1": fee_1,
            "shares": max_shares,
            "cost": round(total_cost, 4),
            "profit": round(total_profit, 4),
            "edge_pct": round(profit_per_unit * 100, 3),
        }

    def check_split_sell(self, market, ob0, ob1):
        """Check for Split+Sell opportunity: YES_bid + NO_bid > 1.0."""
        bid_0, _, bid_size_0, _ = ob0
        bid_1, _, bid_size_1, _ = ob1

        if bid_size_0 < MIN_SHARES or bid_size_1 < MIN_SHARES:
            return None

        fee_0 = self.get_fee_rate(market["token_0"])
        fee_1 = self.get_fee_rate(market["token_1"])

        # Revenue from selling 1 unit of each side after fees
        revenue_per_unit = bid_0 * (1 - fee_0) + bid_1 * (1 - fee_1)
        profit_per_unit = revenue_per_unit - 1.0  # Cost is $1.00 per split

        if profit_per_unit < MIN_PROFIT_PER_UNIT:
            return None

        max_shares = min(bid_size_0, bid_size_1, MAX_POSITION_SIZE)
        max_shares = math.floor(max_shares)
        if max_shares < MIN_SHARES:
            return None

        total_profit = max_shares * profit_per_unit

        return {
            "type": "SPLIT_SELL",
            "market": market,
            "bid_0": bid_0, "bid_1": bid_1,
            "fee_0": fee_0, "fee_1": fee_1,
            "shares": max_shares,
            "cost": round(max_shares * 1.0, 4),  # $1 per unit split
            "profit": round(total_profit, 4),
            "edge_pct": round(profit_per_unit * 100, 3),
        }

    async def execute_buy_merge(self, opp):
        """Execute Buy+Merge: buy both sides, merge to USDC."""
        market = opp["market"]
        shares = opp["shares"]
        print(f"[ARB] Executing BUY+MERGE on {market['question'][:50]}")
        print(f"  {market['outcome_0']}@${opp['ask_0']:.3f} + {market['outcome_1']}@${opp['ask_1']:.3f} = ${opp['ask_0']+opp['ask_1']:.4f} < $1.00")
        print(f"  {shares} shares | cost=${opp['cost']:.2f} | profit=${opp['profit']:.4f} ({opp['edge_pct']:.2f}%)")

        if self.dry_run:
            print(f"  [DRY RUN] Would execute")
            return True

        from py_clob_client.clob_types import MarketOrderArgs, OrderType, BalanceAllowanceParams, AssetType
        from py_clob_client.order_builder.constants import BUY

        try:
            # Buy outcome 0 (FOK)
            fee_0_bps = int(opp["fee_0"] * 10000)
            order_0 = MarketOrderArgs(
                token_id=market["token_0"],
                amount=float(shares) * opp["ask_0"],  # $ amount for BUY
                side=BUY,
                fee_rate_bps=fee_0_bps,
            )
            signed_0 = self.clob.create_market_order(order_0)
            resp_0 = self.clob.post_order(signed_0, OrderType.FOK)
            if not resp_0.get("success"):
                print(f"  [FAIL] {market['outcome_0']} buy failed: {resp_0.get('errorMsg', '?')}")
                return False
            print(f"  [OK] Bought {market['outcome_0']}: {resp_0.get('orderID', '?')[:16]}...")

            # Buy outcome 1 (FOK)
            fee_1_bps = int(opp["fee_1"] * 10000)
            order_1 = MarketOrderArgs(
                token_id=market["token_1"],
                amount=float(shares) * opp["ask_1"],
                side=BUY,
                fee_rate_bps=fee_1_bps,
            )
            signed_1 = self.clob.create_market_order(order_1)
            resp_1 = self.clob.post_order(signed_1, OrderType.FOK)
            if not resp_1.get("success"):
                print(f"  [WARN] {market['outcome_1']} buy failed - holding {market['outcome_0']} directionally")
                return False
            print(f"  [OK] Bought {market['outcome_1']}: {resp_1.get('orderID', '?')[:16]}...")

            # Merge both into USDC
            print(f"  [MERGE] Merging {shares} pairs → ${shares:.2f} USDC...")
            result = self.web3_service.merge(
                condition_id=market["condition_id"],
                amount=shares,
            )
            if result:
                tx = result.get("transactionHash", "?")[:20] if isinstance(result, dict) else "?"
                print(f"  [OK] Merged! tx={tx}...")
                return True
            else:
                print(f"  [WARN] Merge may have failed - check positions")
                return False

        except Exception as e:
            print(f"  [ERROR] {str(e)[:80]}")
            return False

    async def execute_split_sell(self, opp):
        """Execute Split+Sell: split USDC into tokens, sell both."""
        market = opp["market"]
        shares = opp["shares"]
        print(f"[ARB] Executing SPLIT+SELL on {market['question'][:50]}")
        print(f"  {market['outcome_0']}@${opp['bid_0']:.3f} + {market['outcome_1']}@${opp['bid_1']:.3f} = ${opp['bid_0']+opp['bid_1']:.4f} > $1.00")
        print(f"  {shares} shares | cost=${opp['cost']:.2f} | profit=${opp['profit']:.4f} ({opp['edge_pct']:.2f}%)")

        if self.dry_run:
            print(f"  [DRY RUN] Would execute")
            return True

        from py_clob_client.clob_types import OrderArgs, OrderType, BalanceAllowanceParams, AssetType
        from py_clob_client.order_builder.constants import SELL

        try:
            # Split USDC into token pairs
            print(f"  [SPLIT] Splitting ${shares:.2f} USDC → {shares} token pairs...")
            result = self.web3_service.split(
                condition_id=market["condition_id"],
                amount=shares,
            )
            if not result:
                print(f"  [FAIL] Split failed")
                return False
            tx = result.get("transactionHash", "?")[:20] if isinstance(result, dict) else "?"
            print(f"  [OK] Split done! tx={tx}...")

            # Approve + Sell outcome 0
            try:
                self.clob.update_balance_allowance(
                    BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=market["token_0"]))
            except Exception:
                pass
            order_0 = OrderArgs(
                price=opp["bid_0"],
                size=float(shares),
                side=SELL,
                token_id=market["token_0"],
            )
            signed_0 = self.clob.create_order(order_0)
            resp_0 = self.clob.post_order(signed_0, OrderType.FOK)
            if resp_0.get("success"):
                print(f"  [OK] Sold {market['outcome_0']}: {resp_0.get('orderID', '?')[:16]}...")
            else:
                print(f"  [WARN] {market['outcome_0']} sell failed - tokens held for resolution")

            # Approve + Sell outcome 1
            try:
                self.clob.update_balance_allowance(
                    BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=market["token_1"]))
            except Exception:
                pass
            order_1 = OrderArgs(
                price=opp["bid_1"],
                size=float(shares),
                side=SELL,
                token_id=market["token_1"],
            )
            signed_1 = self.clob.create_order(order_1)
            resp_1 = self.clob.post_order(signed_1, OrderType.FOK)
            if resp_1.get("success"):
                print(f"  [OK] Sold {market['outcome_1']}: {resp_1.get('orderID', '?')[:16]}...")
            else:
                print(f"  [WARN] {market['outcome_1']} sell failed - tokens held for resolution")

            return True

        except Exception as e:
            print(f"  [ERROR] {str(e)[:80]}")
            return False

    async def scan_once(self, client: httpx.AsyncClient):
        """Scan all markets for arb opportunities."""
        self.scans += 1
        markets = await self.fetch_markets(client)
        if not markets:
            return

        best_opp = None
        best_profit = 0.0
        min_ask_sum = 2.0
        max_bid_sum = 0.0

        for market in markets:
            # Get orderbooks for both sides
            ob0 = self.get_orderbook(market["token_0"])
            ob1 = self.get_orderbook(market["token_1"])

            bid_0, ask_0 = ob0[0], ob0[1]
            bid_1, ask_1 = ob1[0], ob1[1]
            ask_sum = round(ask_0 + ask_1, 4)
            bid_sum = round(bid_0 + bid_1, 4)
            min_ask_sum = min(min_ask_sum, ask_sum)
            max_bid_sum = max(max_bid_sum, bid_sum)

            # Log only near-opportunity markets (within 5% of $1.00)
            if ask_sum < 1.05 or bid_sum > 0.95:
                q = market["question"][:40]
                print(f"  [{market['asset']}] {q} | asks={ask_sum} bids={bid_sum}")

            # Check both arb types
            buy_merge = self.check_buy_merge(market, ob0, ob1)
            if buy_merge and buy_merge["profit"] > best_profit:
                best_opp = buy_merge
                best_profit = buy_merge["profit"]

            split_sell = self.check_split_sell(market, ob0, ob1)
            if split_sell and split_sell["profit"] > best_profit:
                best_opp = split_sell
                best_profit = split_sell["profit"]

        if not best_opp:
            print(f"  No opps | {len(markets)} mkts | best asks={min_ask_sum:.3f} bids={max_bid_sum:.3f}")
            return

        if best_opp:
            self.opportunities_found += 1
            m = best_opp["market"]
            print(f"\n{'='*60}")
            print(f"[ARB] OPPORTUNITY #{self.opportunities_found}: {best_opp['type']}")

            if best_opp["type"] == "BUY_MERGE":
                success = await self.execute_buy_merge(best_opp)
            else:
                success = await self.execute_split_sell(best_opp)

            if success:
                self.arbs_executed += 1
                self.total_profit += best_opp["profit"]
                self.trades.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "type": best_opp["type"],
                    "market": m["question"][:60],
                    "asset": m["asset"],
                    "shares": best_opp["shares"],
                    "edge_pct": best_opp["edge_pct"],
                    "profit": best_opp["profit"],
                    "dry_run": self.dry_run,
                })
            self._save_results()
            print(f"{'='*60}\n")

    async def run(self):
        """Main loop."""
        mode = "DRY RUN" if self.dry_run else "LIVE"
        print(f"{'='*60}")
        print(f"  Split+Merge Arbitrage Bot [{mode}]")
        print(f"  Min edge: ${MIN_PROFIT_PER_UNIT}/unit | Max size: ${MAX_POSITION_SIZE}")
        print(f"  Scanning {len(ASSETS)} assets every {SCAN_INTERVAL}s")
        print(f"  Prior: {self.arbs_executed} arbs, ${self.total_profit:.2f} profit")
        print(f"{'='*60}\n")

        async with httpx.AsyncClient() as client:
            while True:
                try:
                    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    print(f"[Scan {self.scans+1}] {now} UTC")
                    await self.scan_once(client)
                except Exception as e:
                    print(f"[ERROR] Scan failed: {str(e)[:80]}")

                await asyncio.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv or "-d" in sys.argv
    bot = SplitMergeArb(dry_run=dry_run)
    asyncio.run(bot.run())
