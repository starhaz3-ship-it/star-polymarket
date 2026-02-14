"""
Momentum 5M Directional Paper Trader V1.1 — Chop Filter

Standalone implementation of the 5M momentum strategy from run_ta_paper.py.
Paper-proven: 155 trades, 64.5% WR, +$269 PnL.
  - momentum_strong: 71% WR, +$215 (best sub-strategy)
  - momentum: 55% WR, +$54

Strategy:
  1. Fetch BTC 1-minute candles from Binance (real-time momentum)
  2. Calculate 10-min and 5-min momentum (price change over last 2-3 bars)
  3. Find active Polymarket 5M BTC Up/Down markets
  4. If upward momentum -> buy UP token
  5. If downward momentum -> buy DOWN token
  6. Wait for market resolution (5 min expiry)

Entry rules:
  - momentum_strong: 10m AND 5m momentum aligned in same direction
  - momentum: 10m momentum only (5m may be flat/counter)
  - Time window: 0.5 - 4.8 min before market close
  - Entry price range: $0.10 - $0.60 (avoid extremes)
  - Min momentum: 0.03% (10m price change)
  - Paper spread: +$0.02 on entry (realistic fill simulation)

Usage:
  python run_momentum_5m.py          # Paper mode (default)
  python run_momentum_5m.py --live   # Live mode (real CLOB orders)
"""

import sys
import json
import time
import os
import asyncio
import argparse
from datetime import datetime, timezone
from pathlib import Path
from functools import partial as fn_partial
from typing import Dict, Optional

import httpx
from dotenv import load_dotenv

print = fn_partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

# ============================================================================
# PID LOCK (prevent duplicate instances)
# ============================================================================
from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# CONFIG
# ============================================================================
SCAN_INTERVAL = 30          # seconds between cycles
SIZE_STRONG = 5.0           # momentum_strong bet size
SIZE_WEAK = 3.0             # momentum (weak) bet size
MAX_CONCURRENT = 3          # max open positions
MIN_MOMENTUM_10M = 0.0003   # 0.03% price change over 10 min
MIN_ENTRY = 0.10            # minimum entry price (avoid noise)
MAX_ENTRY = 0.60            # maximum entry price (avoid paying premium)
TIME_WINDOW = (0.5, 4.8)    # minutes before expiry to enter
SPREAD_OFFSET = 0.02        # paper fill spread simulation
RESOLVE_AGE_MIN = 6.0       # minutes before forcing resolution via API
# V1.1: Chop filter — reject ranging markets where momentum is noise
EMA_GAP_MIN_BP = 6          # EMA(9)-EMA(21) gap must be >= 6 basis points
RESULTS_FILE = Path(__file__).parent / "momentum_5m_results.json"

# ============================================================================
# AUTO-REDEEM (gasless relayer) — same as maker bot
# ============================================================================
_poly_web3_service = None

def _get_poly_web3_service():
    global _poly_web3_service
    if _poly_web3_service is not None:
        return _poly_web3_service
    try:
        from py_clob_client.client import ClobClient
        from py_builder_relayer_client.client import RelayClient
        from py_builder_signing_sdk.config import BuilderConfig
        from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds
        from poly_web3 import PolyWeb3Service
        from crypto_utils import decrypt_key

        password = os.getenv("POLYMARKET_PASSWORD", "")
        if not password:
            return None
        pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        if pk.startswith("ENC:"):
            pk = decrypt_key(pk[4:], os.getenv("POLYMARKET_KEY_SALT", ""), password)
        if not pk:
            return None
        proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
        client = ClobClient(
            host="https://clob.polymarket.com", key=pk, chain_id=137,
            signature_type=1, funder=proxy_address if proxy_address else None,
        )
        creds = client.derive_api_key()
        client.set_api_creds(creds)

        builder_key = decrypt_key(os.getenv("POLY_BUILDER_API_KEY", "")[4:],
                                  os.getenv("POLY_BUILDER_API_KEY_SALT", ""), password)
        builder_secret = decrypt_key(os.getenv("POLY_BUILDER_SECRET", "")[4:],
                                     os.getenv("POLY_BUILDER_SECRET_SALT", ""), password)
        builder_passphrase = decrypt_key(os.getenv("POLY_BUILDER_PASSPHRASE", "")[4:],
                                         os.getenv("POLY_BUILDER_PASSPHRASE_SALT", ""), password)
        builder_config = BuilderConfig(
            local_builder_creds=BuilderApiKeyCreds(
                key=builder_key, secret=builder_secret, passphrase=builder_passphrase))
        relay_client = RelayClient(
            relayer_url="https://relayer-v2.polymarket.com", chain_id=137,
            private_key=pk, builder_config=builder_config,
        )
        _poly_web3_service = PolyWeb3Service(
            clob_client=client, relayer_client=relay_client,
            rpc_url="https://polygon-bor.publicnode.com",
        )
        print(f"[REDEEM] PolyWeb3Service initialized")
        return _poly_web3_service
    except Exception as e:
        print(f"[REDEEM] Failed to init: {str(e)[:80]}")
        return None

def auto_redeem_winnings():
    try:
        proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
        if not proxy_address:
            return 0.0
        service = _get_poly_web3_service()
        if not service:
            return 0.0
        positions = service.fetch_positions(user_address=proxy_address)
        if not positions:
            return 0.0
        total_shares = sum(float(p.get("size", 0) or 0) for p in positions)
        print(f"[REDEEM] Found {len(positions)} winning ({total_shares:.0f} shares, ~${total_shares:.2f})")
        results = service.redeem_all(batch_size=10)
        if results:
            print(f"[REDEEM] Claimed {len(results)} batch(es) (~${total_shares:.2f} USDC)")
            return total_shares
        return 0.0
    except Exception as e:
        if "No winning" not in str(e):
            print(f"[REDEEM] Error: {str(e)[:80]}")
        return 0.0


# ============================================================================
# MOMENTUM 5M TRADER
# ============================================================================
class Momentum5MTrader:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.client = None  # ClobClient for live mode
        self.trades: Dict[str, dict] = {}  # {trade_key: trade_dict}
        self.resolved: list = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0, "start_time": datetime.now(timezone.utc).isoformat()}
        self._load()

        if not self.paper:
            self._init_clob()

    def _init_clob(self):
        try:
            from py_clob_client.client import ClobClient
            from crypto_utils import decrypt_key

            password = os.getenv("POLYMARKET_PASSWORD", "")
            pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
            if pk.startswith("ENC:"):
                pk = decrypt_key(pk[4:], os.getenv("POLYMARKET_KEY_SALT", ""), password)
            proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "")

            self.client = ClobClient(
                host="https://clob.polymarket.com", key=pk, chain_id=137,
                signature_type=1, funder=proxy if proxy else None,
            )
            creds = self.client.derive_api_key()
            self.client.set_api_creds(creds)
            print(f"[CLOB] Client initialized - LIVE MODE")
        except Exception as e:
            print(f"[CLOB] Init failed: {e}")
            print(f"[CLOB] Falling back to PAPER mode")
            self.paper = True

    def _load(self):
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.trades = data.get("active", {})
                self.resolved = data.get("resolved", [])
                self.stats = data.get("stats", self.stats)
                print(f"[LOAD] {len(self.resolved)} resolved, {len(self.trades)} active | "
                      f"PnL: ${self.stats['pnl']:+.2f} | {self.stats['wins']}W/{self.stats['losses']}L")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        data = {
            "active": self.trades,
            "resolved": self.resolved,
            "stats": self.stats,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            RESULTS_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    # ========================================================================
    # DATA FETCHING
    # ========================================================================

    async def fetch_binance_candles(self) -> list:
        """Fetch 1-minute BTC/USDT candles from Binance."""
        url = "https://api.binance.com/api/v3/klines"
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    r = await client.get(url, params={"symbol": "BTCUSDT", "interval": "1m", "limit": 30})
                    r.raise_for_status()
                    return [{"time": int(k[0]), "close": float(k[4])} for k in r.json()]
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                print(f"[BINANCE] Error: {e}")
                return []

    async def discover_5m_markets(self) -> list:
        """Find active Polymarket 5M BTC Up/Down markets."""
        markets = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    "https://gamma-api.polymarket.com/events",
                    params={"tag_slug": "15M", "active": "true", "closed": "false",
                            "limit": 200},
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                if r.status_code != 200:
                    print(f"[API] Error: {r.status_code}")
                    return markets

                now = datetime.now(timezone.utc)
                for event in r.json():
                    title = event.get("title", "").lower()
                    if "bitcoin" not in title and "btc" not in title:
                        continue
                    for m in event.get("markets", []):
                        if m.get("closed", True):
                            continue
                        # Skip expired
                        end_date = m.get("endDate", "")
                        if end_date:
                            try:
                                end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                                if end_dt < now:
                                    continue
                                time_left = (end_dt - now).total_seconds() / 60
                                m["_time_left"] = time_left
                            except Exception:
                                continue
                        if not m.get("question"):
                            m["question"] = event.get("title", "")
                        markets.append(m)
        except Exception as e:
            print(f"[API] Error: {e}")
        return markets

    # ========================================================================
    # MARKET HELPERS
    # ========================================================================

    def get_prices(self, market: dict) -> tuple:
        """Extract UP/DOWN prices."""
        outcomes = market.get("outcomes", [])
        prices = market.get("outcomePrices", [])
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(prices, str):
            prices = json.loads(prices)
        up_price, down_price = None, None
        for i, o in enumerate(outcomes):
            if i < len(prices):
                p = float(prices[i])
                if str(o).lower() == "up":
                    up_price = p
                elif str(o).lower() == "down":
                    down_price = p
        return up_price, down_price

    def get_token_ids(self, market: dict) -> tuple:
        """Extract UP/DOWN token IDs."""
        outcomes = market.get("outcomes", [])
        token_ids = market.get("clobTokenIds", [])
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        up_tid, down_tid = None, None
        for i, o in enumerate(outcomes):
            if i < len(token_ids):
                if str(o).lower() == "up":
                    up_tid = token_ids[i]
                elif str(o).lower() == "down":
                    down_tid = token_ids[i]
        return up_tid, down_tid

    # ========================================================================
    # RESOLVE
    # ========================================================================

    async def resolve_trades(self):
        """Resolve expired 5m trades."""
        now = datetime.now(timezone.utc)
        for tid, trade in list(self.trades.items()):
            if trade.get("status") != "open":
                continue
            try:
                entry_dt = datetime.fromisoformat(trade["entry_time"])
                age_min = (now - entry_dt).total_seconds() / 60
            except Exception:
                age_min = 999

            if age_min < RESOLVE_AGE_MIN:
                continue

            # Resolve via Gamma API
            nid = trade.get("market_numeric_id")
            if nid:
                try:
                    async with httpx.AsyncClient(timeout=8) as cl:
                        r = await cl.get(f"https://gamma-api.polymarket.com/markets/{nid}",
                                         headers={"User-Agent": "Mozilla/5.0"})
                        if r.status_code == 200:
                            rm = r.json()
                            up_p, down_p = self.get_prices(rm)
                            if up_p is not None:
                                price = up_p if trade["side"] == "UP" else down_p
                                # BINARY ONLY: wait for full resolution
                                if price >= 0.95:
                                    # WIN: shares pay $1 each
                                    shares = trade["size_usd"] / trade["entry_price"]
                                    trade["pnl"] = round(shares - trade["size_usd"], 2)
                                    trade["exit_price"] = 1.0
                                elif price <= 0.05:
                                    # LOSS: shares worth $0
                                    trade["pnl"] = round(-trade["size_usd"], 2)
                                    trade["exit_price"] = 0.0
                                else:
                                    # Not resolved yet - skip
                                    continue

                                trade["exit_time"] = now.isoformat()
                                trade["status"] = "closed"

                                won = trade["pnl"] > 0
                                self.stats["wins" if won else "losses"] += 1
                                self.stats["pnl"] += trade["pnl"]
                                self.resolved.append(trade)
                                del self.trades[tid]

                                w, l = self.stats["wins"], self.stats["losses"]
                                wr = w / (w + l) * 100 if (w + l) > 0 else 0
                                tag = "WIN" if won else "LOSS"
                                print(f"[{tag}] {trade['side']} ${trade['pnl']:+.2f} | "
                                      f"entry=${trade['entry_price']:.2f} exit=${trade['exit_price']:.2f} | "
                                      f"strat={trade.get('strategy', '?')} | "
                                      f"{w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f} | "
                                      f"{trade.get('title', '')[:40]}")
                except Exception as e:
                    print(f"[RESOLVE] API error: {e}")
            elif age_min > 15:
                # Very old, no numeric ID — mark as loss
                trade["status"] = "closed"
                trade["pnl"] = -trade["size_usd"]
                self.stats["losses"] += 1
                self.stats["pnl"] += trade["pnl"]
                self.resolved.append(trade)
                del self.trades[tid]
                print(f"[LOSS] {trade['side']} ${trade['pnl']:+.2f} | aged out (no market ID)")

    # ========================================================================
    # ENTRY
    # ========================================================================

    async def find_entries(self, candles: list, markets: list):
        """Find momentum-based entries on 5M BTC markets."""
        now = datetime.now(timezone.utc)
        open_count = sum(1 for t in self.trades.values() if t.get("status") == "open")
        if open_count >= MAX_CONCURRENT:
            return

        # Calculate momentum from 1-min Binance candles
        if len(candles) < 3:
            return

        recent_closes = [c["close"] for c in candles[-3:]]
        if recent_closes[-1] == 0 or recent_closes[0] == 0:
            return

        momentum_10m = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
        momentum_5m = (recent_closes[-1] - recent_closes[-2]) / recent_closes[-2]
        btc_price = recent_closes[-1]

        # V1.1: EMA chop filter — reject ranging markets
        all_closes = [c["close"] for c in candles]
        if len(all_closes) >= 21:
            # EMA(9)
            ema9 = sum(all_closes[:9]) / 9
            m9 = 2 / (9 + 1)
            for p in all_closes[9:]:
                ema9 = (p - ema9) * m9 + ema9
            # EMA(21)
            ema21 = sum(all_closes[:21]) / 21
            m21 = 2 / (21 + 1)
            for p in all_closes[21:]:
                ema21 = (p - ema21) * m21 + ema21
            ema_gap_bp = abs(ema9 - ema21) / btc_price * 10000.0
            if ema_gap_bp < EMA_GAP_MIN_BP:
                print(f"[CHOP] EMA gap {ema_gap_bp:.1f}bp < {EMA_GAP_MIN_BP}bp — ranging market, skipping cycle")
                return

        # Debug: show momentum state
        has_momentum = abs(momentum_10m) > MIN_MOMENTUM_10M
        in_window = sum(1 for m in markets if TIME_WINDOW[0] <= m.get("_time_left", 99) <= TIME_WINDOW[1])
        if in_window > 0 or has_momentum:
            direction = "UP" if momentum_10m > 0 else "DOWN"
            print(f"[SCAN] BTC ${btc_price:,.0f} | mom={momentum_10m:+.4%} ({direction}) | "
                  f"markets_in_window={in_window}/{len(markets)} | "
                  f"threshold={'PASS' if has_momentum else 'MISS'}")

        for market in markets:
            if open_count >= MAX_CONCURRENT:
                break

            time_left = market.get("_time_left", 99)
            if time_left < TIME_WINDOW[0] or time_left > TIME_WINDOW[1]:
                continue

            up_price, down_price = self.get_prices(market)
            if up_price is None or down_price is None:
                continue

            cid = market.get("conditionId", "")
            question = market.get("question", "")
            nid = market.get("id")

            # Skip if already trading this market
            if f"5m_{cid}_UP" in self.trades or f"5m_{cid}_DOWN" in self.trades:
                continue

            # === MOMENTUM STRATEGY ===
            side = None
            entry_price = None
            strategy = "momentum"

            if momentum_10m > MIN_MOMENTUM_10M:
                # Upward momentum -> buy UP
                if MIN_ENTRY <= up_price <= MAX_ENTRY:
                    side = "UP"
                    entry_price = round(up_price + SPREAD_OFFSET, 2)
                    if momentum_5m > 0:
                        strategy = "momentum_strong"
            elif momentum_10m < -MIN_MOMENTUM_10M:
                # Downward momentum -> buy DOWN
                if MIN_ENTRY <= down_price <= MAX_ENTRY:
                    side = "DOWN"
                    entry_price = round(down_price + SPREAD_OFFSET, 2)
                    if momentum_5m < 0:
                        strategy = "momentum_strong"

            if not side or not entry_price or entry_price > MAX_ENTRY:
                continue

            # Size based on signal strength
            trade_size = SIZE_STRONG if strategy == "momentum_strong" else SIZE_WEAK

            # === PLACE TRADE ===
            trade_key = f"5m_{cid}_{side}"
            order_id = None

            if not self.paper and self.client:
                # LIVE: Place actual CLOB order
                try:
                    from py_clob_client.clob_types import OrderArgs, OrderType
                    from py_clob_client.order_builder.constants import BUY

                    up_tid, down_tid = self.get_token_ids(market)
                    token_id = up_tid if side == "UP" else down_tid
                    if not token_id:
                        print(f"[SKIP] No token ID for {side}")
                        continue

                    shares = round(trade_size / entry_price, 2)
                    order_args = OrderArgs(
                        price=round(entry_price, 2),
                        size=shares,
                        side=BUY,
                        token_id=token_id,
                    )
                    signed = self.client.create_order(order_args)
                    resp = self.client.post_order(signed, OrderType.GTC)
                    if resp.get("success"):
                        order_id = resp.get("orderID", "?")
                        print(f"[LIVE] Placed {side} order {order_id[:20]}... @ ${entry_price:.2f} x {shares:.0f}")
                    else:
                        print(f"[LIVE] Order failed: {resp.get('errorMsg', '?')}")
                        continue
                except Exception as e:
                    print(f"[LIVE] Order error: {e}")
                    continue

            self.trades[trade_key] = {
                "side": side,
                "entry_price": entry_price,
                "size_usd": trade_size,
                "entry_time": now.isoformat(),
                "condition_id": cid,
                "market_numeric_id": nid,
                "title": question,
                "strategy": strategy,
                "momentum_10m": round(momentum_10m, 6),
                "momentum_5m": round(momentum_5m, 6),
                "btc_price": btc_price,
                "order_id": order_id,
                "status": "open",
                "pnl": 0.0,
            }
            open_count += 1

            mode = "LIVE" if not self.paper else "PAPER"
            print(f"[ENTRY] {side} ${entry_price:.2f} ${trade_size:.0f} | {strategy} | "
                  f"mom_10m={momentum_10m:+.4%} mom_5m={momentum_5m:+.4%} | "
                  f"BTC=${btc_price:,.0f} | time={time_left:.1f}m | [{mode}] | "
                  f"{question[:50]}")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        mode = "PAPER" if self.paper else "LIVE"
        print("=" * 70)
        print(f"MOMENTUM 5M DIRECTIONAL TRADER - {mode} MODE")
        print(f"Strategy: BTC momentum continuation on Polymarket 5M markets")
        print(f"Proven: 155T, 64.5% WR, +$269 PnL (paper)")
        print(f"  momentum_strong: 71% WR | momentum: 55% WR")
        print(f"Size: ${SIZE_STRONG} strong / ${SIZE_WEAK} weak | Max concurrent: {MAX_CONCURRENT}")
        print(f"Entry window: {TIME_WINDOW[0]}-{TIME_WINDOW[1]} min before close")
        print(f"Min momentum: {MIN_MOMENTUM_10M:.4%} (10m)")
        print(f"Scan interval: {SCAN_INTERVAL}s")
        print("=" * 70)

        if self.resolved:
            w, l = self.stats["wins"], self.stats["losses"]
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            print(f"[RESUME] {w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
                  f"{len(self.resolved)} resolved, {len(self.trades)} active")

        cycle = 0
        last_redeem = 0
        while True:
            try:
                cycle += 1
                now_ts = time.time()

                # Fetch data in parallel
                candles, markets = await asyncio.gather(
                    self.fetch_binance_candles(),
                    self.discover_5m_markets(),
                )

                if not candles:
                    print(f"[WARN] No candles from Binance")
                    await asyncio.sleep(10)
                    continue

                # Resolve expired trades
                await self.resolve_trades()

                # Find new entries
                await self.find_entries(candles, markets)

                # Status every 5 cycles
                if cycle % 5 == 0:
                    open_count = len(self.trades)
                    w, l = self.stats["wins"], self.stats["losses"]
                    total = w + l
                    wr = w / total * 100 if total > 0 else 0
                    btc = candles[-1]["close"] if candles else 0
                    print(f"\n--- Cycle {cycle} | {mode} | "
                          f"Active: {open_count} | "
                          f"{w}W/{l}L {wr:.0f}%WR | "
                          f"PnL: ${self.stats['pnl']:+.2f} | "
                          f"BTC: ${btc:,.0f} ---")

                # Auto-redeem every 45s (live only)
                if not self.paper and now_ts - last_redeem >= 45:
                    try:
                        claimed = auto_redeem_winnings()
                        if claimed > 0:
                            print(f"[REDEEM] Claimed ${claimed:.2f} back to USDC")
                    except Exception:
                        pass
                    last_redeem = now_ts

                # Save
                self._save()

                await asyncio.sleep(SCAN_INTERVAL)

            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Saving state...")
                self._save()
                break
            except Exception as e:
                print(f"[ERROR] Cycle {cycle}: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(10)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Live mode (real CLOB orders)")
    parser.add_argument("--paper", action="store_true", help="Paper mode (default)")
    args = parser.parse_args()

    lock = acquire_pid_lock("momentum_5m")
    try:
        trader = Momentum5MTrader(paper=not args.live)
        asyncio.run(trader.run())
    finally:
        release_pid_lock("momentum_5m")
