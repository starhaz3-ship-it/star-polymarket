"""
Pairs Arbitrage Bot V1.0

Reverse-engineered from whale wallets 0xe1DF and 0x8151:
  - Every 15-min BTC Up/Down market, place GTC limit buys at $0.45 on BOTH sides
  - Combined cost ~$0.90 -> guaranteed $1.00 payout = $0.10/pair profit (11.1% ROI)
  - Risk: one side doesn't fill -> exposed to binary outcome

Bail-out mechanism (our edge over the whales):
  - After self.bail_timeout: cancel unfilled order, market-sell the filled side
  - Bail loss: ~$0.05/share (sell at ~$0.40 what we bought at $0.45)
  - Whales lose $0.45/share (ride exposed to expiry, ~50% total loss)

Fill probability from whale data (23 markets):
  - Within 1 min: 30%
  - Within 2 min: 52%
  - Within 3 min: 61%  <- self.bail_timeout
  - Within 5 min: 65%
  - Within 15 min: 100%

Expected value per market at $5/side:
  - 60% hedged x $1.11 profit = +$0.67
  - 40% bail x $0.55 loss = -$0.22
  - NET: +$0.45/market = ~$2.70/hour (6 markets/hour)

Usage:
  python run_pairs_arb.py          # Paper mode (default)
  python run_pairs_arb.py --live   # Live mode (real CLOB orders)
"""

import sys
import json
import time
import os
import asyncio
import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
from functools import partial as fn_partial

import httpx
from dotenv import load_dotenv

print = fn_partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# CONFIG — set by market_mode (15M or 5M)
# ============================================================================
BID_PRICE = 0.45            # limit bid price for each side (fallback)
MAX_BID_PRICE = 0.48        # absolute max we'll bid per side
MIN_COMBINED = 0.94         # skip if our combined bids > this (need 6c+ margin)
MAX_ASK_SPREAD = 0.15       # V1.2: symmetry filter — skip if |up_ask - dn_ask| > this
TRADE_SIZE_PER_SIDE = 3.00  # USD per side ($6 total per market)
MIN_SHARES = 5              # CLOB minimum
DAILY_LOSS_LIMIT = 20.0     # stop if session losses exceed this

# Mode-specific defaults (overridden by CLI --mode)
MODE_CONFIGS = {
    "15M": {
        "tag_slug": "15M",
        "scan_interval": 15,
        "bail_timeout": 300,     # 5 min — longer wait = more 2nd-side fills
        "max_concurrent": 2,
        "entry_window": (1.0, 13.0),
        "results_file": "pairs_arb_results.json",
        "lock_name": "pairs_arb",
    },
    "5M": {
        "tag_slug": "5M",
        "scan_interval": 8,      # faster scan for shorter markets
        "bail_timeout": 60,      # 1 min — proportional (3min/15min * 5min)
        "max_concurrent": 2,
        "entry_window": (0.5, 4.0),  # enter 0.5-4 min before close
        "results_file": "pairs_arb_5m_results.json",
        "lock_name": "pairs_arb_5m",
    },
}

# Paper mode fill simulation (from whale timing data)
# Cumulative probability of second side filling by elapsed seconds
PAPER_FILL_CDF = [
    (30, 0.0), (60, 0.30), (90, 0.43), (120, 0.52),
    (180, 0.61), (240, 0.61), (300, 0.65), (420, 0.74),
    (600, 0.87), (900, 1.0),
]


class PairsArbBot:
    def __init__(self, paper=True, mode="15M"):
        self.paper = paper
        self.mode = mode
        self.cfg = MODE_CONFIGS[mode]
        self.client = None
        self.running = True
        self.session_pnl = 0.0

        # Mode-specific settings
        self.scan_interval = self.cfg["scan_interval"]
        self.bail_timeout = self.cfg["bail_timeout"]
        self.max_concurrent = self.cfg["max_concurrent"]
        self.entry_window = self.cfg["entry_window"]
        self.results_file = Path(__file__).parent / self.cfg["results_file"]

        # Active arb attempts: {condition_id: {...}}
        self.active_arbs = {}
        self.resolved = []
        self.stats = {"wins": 0, "losses": 0, "bails": 0, "pnl": 0.0,
                      "hedged": 0, "one_sided": 0, "markets_entered": 0}
        self.attempted_cids = set()

        self._load()
        if not paper:
            self._init_clob()
        self._cleanup_stale()

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
            # COLLATERAL (USDC) allowance at startup — CONDITIONAL needs token_id (set at entry)
            try:
                from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
                self.client.update_balance_allowance(
                    BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
                print(f"[CLOB] USDC allowance set")
            except Exception as e:
                print(f"[CLOB] Allowance warning: {e}")
            print(f"[CLOB] Client initialized - LIVE MODE")
        except Exception as e:
            print(f"[CLOB] Init failed: {e}")
            print(f"[CLOB] Falling back to PAPER mode")
            self.paper = True

    def _load(self):
        if self.results_file.exists():
            try:
                data = json.loads(self.results_file.read_text())
                self.resolved = data.get("resolved", [])
                self.active_arbs = data.get("active", {})
                self.stats = {**self.stats, **data.get("stats", {})}
                self.session_pnl = data.get("session_pnl", 0.0)
                n_active = len(self.active_arbs)
                print(f"[LOAD] {len(self.resolved)} resolved"
                      f"{f', {n_active} active' if n_active else ''} | "
                      f"PnL: ${self.stats['pnl']:+.2f} | "
                      f"Hedged: {self.stats['hedged']} | Bails: {self.stats['bails']}")
                # Re-add active CIDs to attempted set so we don't re-enter them
                for cid in self.active_arbs:
                    self.attempted_cids.add(cid)
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _cleanup_stale(self):
        """On startup, fix stale 'exposed'/'cancelled' entries and recalculate stats."""
        now = time.time()
        fixed = 0
        recalc_needed = False

        # Phase 1: Clean up stale active entries (status != open/hedged)
        stale_active = [cid for cid, arb in self.active_arbs.items()
                        if arb.get("status") in ("cancelled", "bailed", "exposed",
                                                   "resolved_hedged")]
        for cid in stale_active:
            arb = self.active_arbs.pop(cid)
            # Only add to resolved if not already there
            if not any(r.get("condition_id") == cid and r.get("entry_time") == arb.get("entry_time")
                       for r in self.resolved):
                self.resolved.append(arb)
            print(f"  [CLEANUP] Removed stale active: {arb.get('question', '?')[:45]} "
                  f"(status={arb['status']})")
            fixed += 1
            recalc_needed = True

        # Phase 2: For cancelled entries, verify via CLOB if orders were actually filled
        for trade in self.resolved:
            status = trade.get("status", "")

            if status == "cancelled" and self.client and not self.paper:
                # Check CLOB for actual fill status
                up_oid = trade.get("up_order", {}).get("order_id", "")
                dn_oid = trade.get("down_order", {}).get("order_id", "")
                if up_oid and dn_oid and not up_oid.startswith("paper_"):
                    try:
                        up_info = self.check_order_fill(up_oid)
                        dn_info = self.check_order_fill(dn_oid)
                        up_matched = up_info.get("filled", False)
                        dn_matched = dn_info.get("filled", False)

                        if up_matched:
                            trade["up_filled"] = True
                            if up_info.get("size_matched", 0) > 0:
                                trade["up_order"]["shares"] = up_info["size_matched"]
                                p = up_info.get("price") or BID_PRICE
                                trade["up_order"]["price"] = p
                                trade["up_order"]["cost"] = round(p * up_info["size_matched"], 2)
                        if dn_matched:
                            trade["down_filled"] = True
                            if dn_info.get("size_matched", 0) > 0:
                                trade["down_order"]["shares"] = dn_info["size_matched"]
                                p = dn_info.get("price") or BID_PRICE
                                trade["down_order"]["price"] = p
                                trade["down_order"]["cost"] = round(p * dn_info["size_matched"], 2)

                        if up_matched and dn_matched:
                            up_cost = trade["up_order"]["cost"]
                            dn_cost = trade["down_order"]["cost"]
                            total_cost = up_cost + dn_cost
                            hedged_shares = min(trade["up_order"]["shares"],
                                                trade["down_order"]["shares"])
                            profit = hedged_shares - total_cost
                            trade["pnl"] = profit
                            trade["status"] = "resolved_hedged"
                            print(f"  [CLEANUP] CLOB verified hedge: {trade['question'][:45]} "
                                  f"pnl=${profit:+.2f}")
                            fixed += 1
                            recalc_needed = True
                        elif up_matched or dn_matched:
                            # One side filled — record as exposed loss
                            filled_key = "up_order" if up_matched else "down_order"
                            cost = trade.get(filled_key, {}).get("cost", 4.95)
                            trade["pnl"] = -cost
                            trade["status"] = "exposed"
                            print(f"  [CLEANUP] CLOB verified one-sided: {trade['question'][:45]} "
                                  f"pnl=${trade['pnl']:+.2f}")
                            fixed += 1
                            recalc_needed = True
                    except Exception as e:
                        print(f"  [CLEANUP] CLOB check error: {e}")

            # Fix exposed trades: market has ended, record full loss
            if status == "exposed" and trade.get("pnl", 0) == 0:
                end_dt_str = trade.get("end_dt", "")
                if end_dt_str:
                    try:
                        end_dt = datetime.fromisoformat(end_dt_str)
                        if now > end_dt.timestamp() + 60:
                            for side in ["up_order", "down_order"]:
                                if trade.get(f"{side.split('_')[0]}_filled"):
                                    cost = trade.get(side, {}).get("cost", 4.95)
                                    trade["pnl"] = -cost
                                    break
                            print(f"  [CLEANUP] Exposed trade fixed: {trade['question'][:45]} "
                                  f"pnl=${trade['pnl']:+.2f}")
                            fixed += 1
                            recalc_needed = True
                    except Exception:
                        pass

            # Fix cancelled entries that were actually double-hedged
            # (both sides show up_filled=True and down_filled=True but status=cancelled)
            if trade.get("status") == "cancelled":
                if trade.get("up_filled") and trade.get("down_filled"):
                    up_cost = trade.get("up_order", {}).get("cost", 4.95)
                    dn_cost = trade.get("down_order", {}).get("cost", 4.95)
                    total_cost = up_cost + dn_cost
                    hedged_shares = min(
                        trade.get("up_order", {}).get("shares", 11),
                        trade.get("down_order", {}).get("shares", 11),
                    )
                    profit = hedged_shares - total_cost
                    trade["pnl"] = profit
                    trade["status"] = "resolved_hedged"
                    print(f"  [CLEANUP] Surprise hedge recovered: {trade['question'][:45]} "
                          f"pnl=${profit:+.2f}")
                    fixed += 1
                    recalc_needed = True

        # Always recalculate stats from scratch on startup to catch any drift
        self.stats["hedged"] = 0
        self.stats["bails"] = 0
        self.stats["wins"] = 0
        self.stats["losses"] = 0
        self.stats["one_sided"] = 0
        self.stats["pnl"] = 0.0
        for trade in self.resolved:
            s = trade.get("status", "")
            pnl = trade.get("pnl", 0)
            self.stats["pnl"] += pnl
            if s == "resolved_hedged":
                self.stats["hedged"] += 1
                self.stats["wins"] += 1
            elif s == "bailed":
                self.stats["bails"] += 1
                self.stats["one_sided"] += 1
            elif s == "exposed":
                self.stats["losses"] += 1
                self.stats["one_sided"] += 1
        if fixed or recalc_needed:
            print(f"  [CLEANUP] Stats recalculated: {fixed} fixes | PnL: ${self.stats['pnl']:+.2f}")
        self._save_inner()

    def _save_inner(self):
        """Save without circular dependency."""
        active_clean = {
            cid: arb for cid, arb in self.active_arbs.items()
            if arb.get("status") in ("open", "hedged")
        }
        data = {
            "active": active_clean,
            "resolved": self.resolved[-500:],
            "stats": self.stats,
            "session_pnl": self.session_pnl,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self.results_file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self.results_file)
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    def _save(self):
        # Filter active to only include non-terminal entries
        active_clean = {
            cid: arb for cid, arb in self.active_arbs.items()
            if arb.get("status") in ("open", "hedged")
        }
        data = {
            "active": active_clean,
            "resolved": self.resolved[-500:],
            "stats": self.stats,
            "session_pnl": self.session_pnl,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        # Atomic write: tmp -> rename
        tmp = self.results_file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self.results_file)
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    # ========================================================================
    # MARKET DISCOVERY
    # ========================================================================

    async def discover_15m_markets(self) -> list:
        """Find active BTC 15M Up/Down markets."""
        markets = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    "https://gamma-api.polymarket.com/events",
                    params={"tag_slug": self.cfg["tag_slug"], "active": "true",
                            "closed": "false", "limit": 200},
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                if r.status_code != 200:
                    return markets

                now = datetime.now(timezone.utc)
                for event in r.json():
                    title = event.get("title", "").lower()
                    if "bitcoin" not in title and "btc" not in title:
                        continue

                    for m in event.get("markets", []):
                        if m.get("closed", True):
                            continue
                        end_date = m.get("endDate", "")
                        if not end_date:
                            continue
                        try:
                            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                            if end_dt < now:
                                continue
                            time_left = (end_dt - now).total_seconds() / 60
                            m["_time_left"] = time_left
                            m["_end_dt"] = end_dt.isoformat()
                        except Exception:
                            continue
                        if not m.get("question"):
                            m["question"] = event.get("title", "")
                        markets.append(m)
        except Exception as e:
            print(f"[API] Error: {e}")
        return markets

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
    # ORDER MANAGEMENT
    # ========================================================================

    def place_gtc_limit(self, token_id: str, price: float, size_usd: float,
                        side_label: str) -> dict:
        """Place a GTC limit BUY at specified price. Returns order info (may not be filled yet)."""
        shares = int(size_usd / price)
        if shares < MIN_SHARES:
            shares = MIN_SHARES

        if self.paper:
            return {
                "success": True,
                "order_id": f"paper_{int(time.time()*1000)}_{side_label}",
                "shares": shares, "price": price,
                "cost": round(shares * price, 2),
                "status": "LIVE",  # Posted, not yet filled
            }

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            order_args = OrderArgs(
                price=round(price, 2), size=shares, side=BUY, token_id=token_id,
            )
            signed = self.client.create_order(order_args)
            resp = self.client.post_order(signed, OrderType.GTC)

            if resp.get("success"):
                order_id = resp.get("orderID", "?")
                status = resp.get("status", "LIVE")
                print(f"  [{side_label}] GTC limit posted: {shares}sh @ ${price:.2f} "
                      f"(order={order_id[:16]}...) status={status}")
                return {
                    "success": True, "order_id": order_id,
                    "shares": shares, "price": price,
                    "cost": round(shares * price, 2),
                    "status": status,
                }
            else:
                print(f"  [{side_label}] GTC order REJECTED: {resp.get('errorMsg', '?')}")
                return {"success": False}
        except Exception as e:
            print(f"  [{side_label}] Order error: {e}")
            return {"success": False}

    def check_order_fill(self, order_id: str) -> dict:
        """Check if a GTC order has been filled."""
        if self.paper:
            return {"filled": False, "size_matched": 0}

        try:
            info = self.client.get_order(order_id)
            # CRITICAL: get_order() may return a string, not a dict
            if isinstance(info, str):
                try:
                    info = json.loads(info)
                except json.JSONDecodeError:
                    return {"filled": False, "size_matched": 0, "error": "parse_error"}
            if not isinstance(info, dict):
                return {"filled": False, "size_matched": 0, "error": "bad_type"}

            status = (info.get("status", "") or "UNKNOWN").upper()
            size_matched = float(info.get("size_matched", 0) or 0)
            # "MATCHED" = fully filled, "ACTIVE" = on the book, "CANCELLED" = cancelled
            is_filled = status == "MATCHED" or size_matched >= MIN_SHARES
            fill_price = 0
            try:
                assoc = info.get("associate_trades", [])
                if assoc and isinstance(assoc[0], dict) and assoc[0].get("price"):
                    fill_price = float(assoc[0]["price"])
            except (IndexError, TypeError, ValueError):
                fill_price = BID_PRICE if is_filled else 0
            return {
                "filled": is_filled,
                "size_matched": size_matched,
                "status": status,
                "price": fill_price,
            }
        except Exception as e:
            print(f"  [FILL_CHECK] Error for {order_id[:16]}: {e}")
            return {"filled": False, "size_matched": 0, "error": str(e)}

    def cancel_order(self, order_id: str, side_label: str) -> str:
        """Cancel a GTC order. Returns 'cancelled', 'already_matched', or 'error'."""
        if self.paper:
            return "cancelled"

        try:
            resp = self.client.cancel(order_id)
            canceled = resp.get("canceled", [])
            not_canceled = resp.get("not_canceled", {})

            if order_id in canceled:
                print(f"  [{side_label}] Order cancelled")
                return "cancelled"

            reason = not_canceled.get(order_id, "")
            if "matched" in str(reason).lower():
                print(f"  [{side_label}] Order already MATCHED (filled while we waited)")
                return "already_matched"
            if reason == "ORDER_ALREADY_INACTIVE":
                print(f"  [{side_label}] Order already inactive")
                return "cancelled"

            print(f"  [{side_label}] Cancel response: {resp}")
            return "cancelled"
        except Exception as e:
            print(f"  [{side_label}] Cancel error: {e}")
            return "error"

    def _ensure_sell_allowance(self, token_id: str):
        """Ensure CONDITIONAL token allowance is set for selling.
        V1.3: Per-token approval (ERC-1155 requires specific token_id)."""
        if self.paper or not token_id:
            return
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            self.client.update_balance_allowance(
                BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=token_id)
            )
        except Exception as e:
            print(f"  [ALLOWANCE] {str(e)[:80]}")

    def place_market_sell(self, token_id: str, shares: float, side_label: str) -> dict:
        """Market-sell shares via GTC limit at aggressive price (bail-out)."""
        if self.paper:
            bail_price = BID_PRICE - 0.05
            return {
                "success": True, "shares": shares, "price": bail_price,
                "proceeds": round(shares * bail_price, 2),
            }

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import SELL

            # V1.3: Aggressive allowance — set ALL methods before selling
            self._ensure_sell_allowance(token_id)
            time.sleep(1)  # Give chain time to confirm approval

            # Sell at slight discount to guarantee fill
            sell_price = round(max(0.30, BID_PRICE - 0.05), 2)  # $0.40
            order_args = OrderArgs(
                price=sell_price, size=int(shares), side=SELL, token_id=token_id,
            )
            signed = self.client.create_order(order_args)
            resp = self.client.post_order(signed, OrderType.GTC)

            if resp.get("success"):
                order_id = resp.get("orderID", "?")
                status = resp.get("status", "")
                if status == "matched":
                    time.sleep(1)
                    try:
                        info = self.client.get_order(order_id)
                        size_matched = float(info.get("size_matched", 0) or 0)
                        assoc = info.get("associate_trades", [])
                        fill_price = float(assoc[0]["price"]) if assoc and assoc[0].get("price") else sell_price
                        return {
                            "success": size_matched > 0,
                            "shares": size_matched, "price": fill_price,
                            "proceeds": round(fill_price * size_matched, 2),
                        }
                    except Exception as e:
                        print(f"  [{side_label}] Sell fill check: {e}")
                        return {"success": True, "shares": int(shares), "price": sell_price,
                                "proceeds": round(sell_price * shares, 2)}
                else:
                    print(f"  [{side_label}] GTC sell posted (order={order_id[:16]}...)")
                    return {"success": True, "shares": int(shares), "price": sell_price,
                            "proceeds": round(sell_price * shares, 2)}
            else:
                print(f"  [{side_label}] Sell REJECTED: {resp.get('errorMsg', '?')}")
                return {"success": False}
        except Exception as e:
            print(f"  [{side_label}] Sell error: {e}")
            return {"success": False}

    # ========================================================================
    # PAPER FILL SIMULATION
    # ========================================================================

    def simulate_paper_fills(self, arb: dict) -> tuple:
        """Simulate fill timing for paper mode based on whale CDF data.
        Returns (up_filled, down_filled)."""
        elapsed = time.time() - arb["entry_time"]

        # Get fill probability for current elapsed time
        prob = 0.0
        for t, p in PAPER_FILL_CDF:
            if elapsed <= t:
                prob = p
                break
        else:
            prob = 1.0

        # Each side has `prob` chance of having filled by now
        # Use deterministic seed per arb so results are consistent across checks
        rng = random.Random(hash(arb["condition_id"]))
        up_roll = rng.random()
        dn_roll = rng.random()

        # Simulate: first side always fills quickly (within ~30s like whales)
        # Second side follows the CDF
        first_filled = elapsed >= 5  # First side fills in ~5s
        second_filled = (up_roll < prob) if arb["first_side"] == "UP" else (dn_roll < prob)

        if arb["first_side"] == "UP":
            return first_filled, second_filled
        else:
            return second_filled, first_filled

    async def _resolve_binance(self, end_dt) -> str:
        """Check Binance to determine if BTC went UP or DOWN in this window."""
        try:
            start_dt = end_dt - __import__('datetime').timedelta(minutes=15)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get("https://api.binance.com/api/v3/klines", params={
                    "symbol": "BTCUSDT", "interval": "15m",
                    "startTime": start_ms, "endTime": end_ms, "limit": 2,
                })
                r.raise_for_status()
                klines = r.json()
                if klines:
                    o = float(klines[0][1])
                    c = float(klines[0][4])
                    return "UP" if c >= o else "DOWN"
        except Exception:
            pass
        return "DOWN"  # conservative fallback

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        mode = "PAPER" if self.paper else "LIVE"
        print(f"\n{'='*60}")
        print(f"  PAIRS ARBITRAGE BOT V1.2 -- {self.mode} {mode} MODE")
        print(f"  Bid: ${BID_PRICE:.2f}/side | Bail: {self.bail_timeout}s ({self.bail_timeout/60:.0f}min)")
        print(f"  Size: ${TRADE_SIZE_PER_SIDE:.2f}/side (${TRADE_SIZE_PER_SIDE*2:.2f} total)")
        print(f"  Entry: {self.entry_window[0]}-{self.entry_window[1]}min before close")
        print(f"  Loss limit: ${DAILY_LOSS_LIMIT:.2f} | Max concurrent: {self.max_concurrent}")
        print(f"{'='*60}\n")

        cycle = 0
        while self.running:
            try:
                if self.session_pnl < -DAILY_LOSS_LIMIT:
                    print(f"[STOP] Session loss ${self.session_pnl:.2f} hit limit")
                    break

                # Check active arbs every cycle
                await self.check_active_arbs()

                # Look for new markets
                if len(self.active_arbs) < self.max_concurrent:
                    await self.scan_and_enter()

                # Status every 2 min
                cycle += 1
                if cycle % 8 == 0:
                    active_str = ""
                    for cid, arb in self.active_arbs.items():
                        elapsed = time.time() - arb["entry_time"]
                        active_str += f"\n    {arb['status']:8} +{elapsed:.0f}s {arb['question'][:40]}"
                    if active_str:
                        print(f"[STATUS] Active arbs:{active_str}")
                    h = self.stats["hedged"]
                    b = self.stats["bails"]
                    total = h + b + self.stats["losses"]
                    if total > 0:
                        print(f"  Lifetime: {total} resolved | Hedged: {h} ({h/total*100:.0f}%) "
                              f"| Bails: {b} | PnL: ${self.stats['pnl']:+.2f}")

                await asyncio.sleep(self.scan_interval)

            except KeyboardInterrupt:
                print("\n[STOP] Interrupted")
                break
            except Exception as e:
                print(f"[ERROR] Main loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(self.scan_interval)

        # Cleanup: cancel all active orders
        if not self.paper:
            for cid, arb in self.active_arbs.items():
                for key in ["up_order", "down_order"]:
                    order = arb.get(key, {})
                    if order and order.get("order_id") and not order.get("filled"):
                        self.cancel_order(order["order_id"], key)

        self._save()
        total = self.stats["hedged"] + self.stats["bails"] + self.stats["losses"]
        print(f"\n[FINAL] Session PnL: ${self.session_pnl:+.2f} | "
              f"{total} resolved | Hedged: {self.stats['hedged']} | "
              f"Bails: {self.stats['bails']} | PnL: ${self.stats['pnl']:+.2f}")

    async def scan_and_enter(self):
        """Find new 15m markets and place GTC limits on both sides.
        V1.2: Sort by freshest (most time left), symmetry filter, dynamic pricing."""
        markets = await self.discover_15m_markets()

        # V1.2: Sort by time_left DESC — enter freshest markets first (most symmetric)
        eligible = []
        for market in markets:
            cid = market.get("conditionId", "")
            if not cid or cid in self.attempted_cids or cid in self.active_arbs:
                continue
            time_left = market.get("_time_left", 0)
            if time_left < self.entry_window[0] or time_left > self.entry_window[1]:
                continue
            eligible.append(market)
        eligible.sort(key=lambda m: m.get("_time_left", 0), reverse=True)

        for market in eligible:
            if len(self.active_arbs) >= self.max_concurrent:
                break

            cid = market.get("conditionId", "")
            time_left = market.get("_time_left", 0)
            up_tid, down_tid = self.get_token_ids(market)
            if not up_tid or not down_tid:
                continue

            question = market.get("question", "?")[:60]

            # V1.2: Read orderbook prices, check symmetry, dynamic bid
            up_bid = BID_PRICE
            dn_bid = BID_PRICE
            up_ask = 0.0
            dn_ask = 0.0
            if self.client and not self.paper:
                try:
                    up_resp = self.client.get_price(up_tid, "sell")
                    dn_resp = self.client.get_price(down_tid, "sell")
                    up_ask = float(up_resp.get("price", 0) if isinstance(up_resp, dict) else up_resp or 0)
                    dn_ask = float(dn_resp.get("price", 0) if isinstance(dn_resp, dict) else dn_resp or 0)

                    if up_ask > 0 and dn_ask > 0:
                        # Symmetry filter — skip lopsided markets
                        ask_spread = abs(up_ask - dn_ask)
                        if ask_spread > MAX_ASK_SPREAD:
                            print(f"  [SKIP] {question[:40]} | asks UP=${up_ask:.2f} DN=${dn_ask:.2f} "
                                  f"spread=${ask_spread:.2f} > ${MAX_ASK_SPREAD} — too lopsided")
                            self.attempted_cids.add(cid)
                            continue

                        # Dynamic bid: 1c below ask, capped at MAX_BID_PRICE, floored at $0.40
                        up_bid = round(max(0.40, min(up_ask - 0.01, MAX_BID_PRICE)), 2)
                        dn_bid = round(max(0.40, min(dn_ask - 0.01, MAX_BID_PRICE)), 2)

                        # Profit margin check
                        if up_bid + dn_bid > MIN_COMBINED:
                            print(f"  [SKIP] Combined ${up_bid + dn_bid:.2f} > ${MIN_COMBINED} — no margin")
                            self.attempted_cids.add(cid)
                            continue
                    else:
                        # Can't read prices — use fallback
                        print(f"  [PRICE] No asks available, using ${BID_PRICE}")
                except Exception as e:
                    print(f"  [PRICE] Fallback to ${BID_PRICE}: {str(e)[:60]}")

            self.attempted_cids.add(cid)
            self.stats["markets_entered"] += 1

            now_mst = datetime.now(timezone.utc).strftime("%H:%M UTC")
            print(f"\n{'='*55}")
            print(f"[ENTER] {question}")
            print(f"  {now_mst} | {time_left:.1f}min left | UP@${up_bid:.2f} DOWN@${dn_bid:.2f}"
                  f" (asks: {up_ask:.2f}/{dn_ask:.2f})")

            # V1.3: Pre-approve sell allowance BEFORE placing orders (ensures bail works)
            if not self.paper:
                self._ensure_sell_allowance(up_tid)
                self._ensure_sell_allowance(down_tid)

            # Place GTC limits on both UP and DOWN with dynamic pricing
            up_order = self.place_gtc_limit(up_tid, up_bid, TRADE_SIZE_PER_SIDE, "UP")
            dn_order = self.place_gtc_limit(down_tid, dn_bid, TRADE_SIZE_PER_SIDE, "DOWN")

            if not up_order["success"] and not dn_order["success"]:
                print(f"  [SKIP] Neither side posted")
                continue

            # Determine first side (for paper simulation ordering)
            first_side = "UP" if random.random() < 0.5 else "DOWN"

            arb = {
                "condition_id": cid,
                "question": question,
                "entry_time": time.time(),
                "end_dt": market.get("_end_dt", ""),
                "up_tid": up_tid,
                "down_tid": down_tid,
                "up_order": up_order if up_order["success"] else None,
                "down_order": dn_order if dn_order["success"] else None,
                "up_filled": False,
                "down_filled": False,
                "up_fill_time": None,
                "down_fill_time": None,
                "first_side": first_side,
                "status": "open",  # open -> hedged -> resolved | open -> bailing -> bailed
                "bail_deadline": time.time() + self.bail_timeout,
            }

            # Check if either order filled immediately (possible on CLOB)
            if not self.paper:
                if up_order["success"] and up_order.get("status") == "matched":
                    arb["up_filled"] = True
                    arb["up_fill_time"] = time.time()
                    self._ensure_sell_allowance(up_tid)  # V1.2: pre-approve sell
                    print(f"  [INSTANT] UP filled immediately!")
                if dn_order["success"] and dn_order.get("status") == "matched":
                    arb["down_filled"] = True
                    arb["down_fill_time"] = time.time()
                    self._ensure_sell_allowance(down_tid)  # V1.2: pre-approve sell
                    print(f"  [INSTANT] DOWN filled immediately!")

                if arb["up_filled"] and arb["down_filled"]:
                    arb["status"] = "hedged"
                    print(f"  [HEDGED] Both sides filled instantly!")

            self.active_arbs[cid] = arb
            self._save()

    async def check_active_arbs(self):
        """Check fill status, handle bail-outs and resolutions."""
        now = time.time()
        to_remove = []

        for cid, arb in list(self.active_arbs.items()):

            if arb["status"] == "open":
                elapsed = now - arb["entry_time"]

                # Check fills
                if self.paper:
                    up_f, dn_f = self.simulate_paper_fills(arb)
                    if up_f and not arb["up_filled"]:
                        arb["up_filled"] = True
                        arb["up_fill_time"] = now
                        print(f"  [FILL] UP +{elapsed:.0f}s | {arb['question'][:40]}")
                        self._save()
                    if dn_f and not arb["down_filled"]:
                        arb["down_filled"] = True
                        arb["down_fill_time"] = now
                        print(f"  [FILL] DOWN +{elapsed:.0f}s | {arb['question'][:40]}")
                        self._save()
                else:
                    # Live: check order status via CLOB
                    if not arb["up_filled"] and arb.get("up_order"):
                        info = self.check_order_fill(arb["up_order"]["order_id"])
                        if info["filled"]:
                            arb["up_filled"] = True
                            arb["up_fill_time"] = now
                            actual_price = info.get("price") or BID_PRICE
                            arb["up_order"]["price"] = actual_price
                            arb["up_order"]["cost"] = round(actual_price * info["size_matched"], 2)
                            arb["up_order"]["shares"] = info["size_matched"]
                            self._ensure_sell_allowance(arb["up_tid"])  # V1.2: pre-approve
                            print(f"  [FILL] UP +{elapsed:.0f}s {info['size_matched']:.0f}sh "
                                  f"@ ${actual_price:.3f} | {arb['question'][:40]}")

                    if not arb["down_filled"] and arb.get("down_order"):
                        info = self.check_order_fill(arb["down_order"]["order_id"])
                        if info["filled"]:
                            arb["down_filled"] = True
                            arb["down_fill_time"] = now
                            actual_price = info.get("price") or BID_PRICE
                            arb["down_order"]["price"] = actual_price
                            arb["down_order"]["cost"] = round(actual_price * info["size_matched"], 2)
                            arb["down_order"]["shares"] = info["size_matched"]
                            self._ensure_sell_allowance(arb["down_tid"])  # V1.2: pre-approve
                            print(f"  [FILL] DOWN +{elapsed:.0f}s {info['size_matched']:.0f}sh "
                                  f"@ ${actual_price:.3f} | {arb['question'][:40]}")

                # Both filled -> hedged!
                if arb["up_filled"] and arb["down_filled"]:
                    arb["status"] = "hedged"
                    up_cost = arb["up_order"]["cost"]
                    dn_cost = arb["down_order"]["cost"]
                    total_cost = up_cost + dn_cost
                    hedged_shares = min(arb["up_order"]["shares"], arb["down_order"]["shares"])
                    profit = hedged_shares - total_cost
                    gap = abs((arb["up_fill_time"] or 0) - (arb["down_fill_time"] or 0))
                    print(f"  [HEDGED] Both sides filled! gap={gap:.0f}s "
                          f"cost=${total_cost:.2f} profit=${profit:.2f} | {arb['question'][:40]}")
                    self._save()

                # Bail-out check
                elif now > arb["bail_deadline"]:
                    filled_side = None
                    unfilled_side = None

                    if arb["up_filled"] and not arb["down_filled"]:
                        filled_side, unfilled_side = "UP", "DOWN"
                    elif arb["down_filled"] and not arb["up_filled"]:
                        filled_side, unfilled_side = "DOWN", "UP"
                    elif not arb["up_filled"] and not arb["down_filled"]:
                        # Neither side filled — cancel both, but check for surprise fills
                        print(f"  [CANCEL] Neither side filled after {elapsed:.0f}s — "
                              f"cancelling | {arb['question'][:40]}")
                        up_cancel = "cancelled"
                        dn_cancel = "cancelled"
                        if arb.get("up_order"):
                            up_cancel = self.cancel_order(arb["up_order"]["order_id"], "UP")
                        if arb.get("down_order"):
                            dn_cancel = self.cancel_order(arb["down_order"]["order_id"], "DOWN")

                        # Detect surprise fills from cancel responses
                        if up_cancel == "already_matched":
                            arb["up_filled"] = True
                            arb["up_fill_time"] = now
                            if not self.paper:
                                fi = self.check_order_fill(arb["up_order"]["order_id"])
                                if fi.get("size_matched", 0) > 0:
                                    arb["up_order"]["shares"] = fi["size_matched"]
                                    p = fi.get("price") or BID_PRICE
                                    arb["up_order"]["price"] = p
                                    arb["up_order"]["cost"] = round(p * fi["size_matched"], 2)
                        if dn_cancel == "already_matched":
                            arb["down_filled"] = True
                            arb["down_fill_time"] = now
                            if not self.paper:
                                fi = self.check_order_fill(arb["down_order"]["order_id"])
                                if fi.get("size_matched", 0) > 0:
                                    arb["down_order"]["shares"] = fi["size_matched"]
                                    p = fi.get("price") or BID_PRICE
                                    arb["down_order"]["price"] = p
                                    arb["down_order"]["cost"] = round(p * fi["size_matched"], 2)

                        if arb["up_filled"] and arb["down_filled"]:
                            # Surprise double-hedge!
                            arb["status"] = "hedged"
                            up_cost = arb["up_order"]["cost"]
                            dn_cost = arb["down_order"]["cost"]
                            total_cost = up_cost + dn_cost
                            hedged_shares = min(arb["up_order"]["shares"], arb["down_order"]["shares"])
                            profit = hedged_shares - total_cost
                            print(f"  [SURPRISE HEDGE] Both sides were filled! "
                                  f"cost=${total_cost:.2f} profit=${profit:.2f}")
                            self._save()
                            continue  # Stay active, resolve when market ends
                        elif arb["up_filled"] or arb["down_filled"]:
                            # One side surprise-filled — bail it
                            s_filled = "UP" if arb["up_filled"] else "DOWN"
                            print(f"  [SURPRISE FILL] {s_filled} was actually filled! Bailing...")
                            s_key = "up_order" if s_filled == "UP" else "down_order"
                            s_tid = arb["up_tid"] if s_filled == "UP" else arb["down_tid"]
                            s_order = arb[s_key]
                            sell_r = self.place_market_sell(s_tid, s_order["shares"], s_filled)
                            if sell_r["success"]:
                                bail_loss = s_order["cost"] - sell_r["proceeds"]
                                print(f"  Sold {sell_r['shares']:.0f}sh @ ${sell_r['price']:.3f} "
                                      f"= ${sell_r['proceeds']:.2f} (loss: ${bail_loss:.2f})")
                                arb["pnl"] = -bail_loss
                                arb["status"] = "bailed"
                                arb["bail_result"] = sell_r
                                self.stats["bails"] += 1
                                self.stats["one_sided"] += 1
                                self.stats["pnl"] += arb["pnl"]
                                self.session_pnl += arb["pnl"]
                            else:
                                # V1.3: Don't record as full loss — shares still in wallet
                                # Let it ride to resolution instead of marking dead
                                print(f"  [HOLD] Bail sell failed — holding to expiry")
                                arb["status"] = "hedged"  # treat as one-sided hedge
                                arb["bail_failed"] = True
                                self._save()
                                continue  # stay active, resolve at market end
                        else:
                            # Truly neither side filled — clean cancel
                            arb["status"] = "cancelled"
                            arb["pnl"] = 0.0
                            self.resolved.append(arb)
                            to_remove.append(cid)
                            self._save()
                            continue

                    # One side filled — attempt bail
                    print(f"\n  [BAIL] {arb['question'][:45]}")
                    print(f"  {filled_side} filled, {unfilled_side} not filled after {elapsed:.0f}s")

                    # Cancel the unfilled order — but check if it actually filled!
                    unfilled_key = f"{'down' if unfilled_side == 'DOWN' else 'up'}_order"
                    cancel_result = "cancelled"
                    if arb.get(unfilled_key):
                        cancel_result = self.cancel_order(arb[unfilled_key]["order_id"], unfilled_side)

                    # If cancel says "already_matched", the order filled while we waited!
                    # Also re-check via get_order to be sure
                    if cancel_result == "already_matched":
                        # Both sides actually filled — this is hedged, not a bail!
                        if unfilled_side == "UP":
                            arb["up_filled"] = True
                            arb["up_fill_time"] = now
                        else:
                            arb["down_filled"] = True
                            arb["down_fill_time"] = now

                        # Try to get actual fill details
                        fill_info = self.check_order_fill(arb[unfilled_key]["order_id"])
                        if fill_info.get("size_matched", 0) > 0:
                            arb[unfilled_key]["shares"] = fill_info["size_matched"]
                            actual_p = fill_info.get("price") or BID_PRICE
                            arb[unfilled_key]["price"] = actual_p
                            arb[unfilled_key]["cost"] = round(actual_p * fill_info["size_matched"], 2)

                        arb["status"] = "hedged"
                        up_cost = arb["up_order"]["cost"]
                        dn_cost = arb["down_order"]["cost"]
                        total_cost = up_cost + dn_cost
                        hedged_shares = min(arb["up_order"]["shares"], arb["down_order"]["shares"])
                        profit = hedged_shares - total_cost
                        print(f"  [SURPRISE HEDGE] {unfilled_side} actually filled! "
                              f"cost=${total_cost:.2f} profit=${profit:.2f}")
                        self._save()
                        continue  # Don't bail — it's hedged now

                    # Truly one-sided — market-sell the filled side
                    filled_key = f"{'up' if filled_side == 'UP' else 'down'}_order"
                    filled_order = arb[filled_key]
                    filled_tid = arb["up_tid"] if filled_side == "UP" else arb["down_tid"]

                    sell_result = self.place_market_sell(
                        filled_tid, filled_order["shares"], filled_side
                    )

                    # Retry sell up to 3 times if it fails
                    if not sell_result["success"]:
                        for _retry in range(2):
                            print(f"  [RETRY] Sell attempt {_retry+2}/3...")
                            time.sleep(2)
                            sell_result = self.place_market_sell(
                                filled_tid, filled_order["shares"], filled_side
                            )
                            if sell_result["success"]:
                                break

                    if sell_result["success"]:
                        bail_loss = filled_order["cost"] - sell_result["proceeds"]
                        print(f"  Sold {sell_result['shares']:.0f}sh @ ${sell_result['price']:.3f} "
                              f"= ${sell_result['proceeds']:.2f} (loss: ${bail_loss:.2f})")
                        arb["pnl"] = -bail_loss
                    else:
                        # V1.3: All retries failed — hold to expiry instead of marking full loss
                        # The shares are still in wallet and might win at resolution
                        print(f"  [HOLD] Bail sell failed after 3 attempts — holding to expiry")
                        arb["status"] = "hedged"  # treat as one-sided, resolve at market end
                        arb["bail_failed"] = True
                        arb["exposed_side"] = filled_side
                        self._save()
                        continue  # stay active

                    arb["status"] = "bailed"
                    arb["bail_result"] = sell_result
                    self.stats["bails"] += 1
                    self.stats["one_sided"] += 1
                    self.stats["pnl"] += arb["pnl"]
                    self.session_pnl += arb["pnl"]
                    self.resolved.append(arb)
                    to_remove.append(cid)
                    self._save()

            elif arb["status"] == "hedged":
                # Wait for market resolution
                end_dt_str = arb.get("end_dt", "")
                if not end_dt_str:
                    continue
                try:
                    end_dt = datetime.fromisoformat(end_dt_str)
                    if now > end_dt.timestamp() + 90:  # 90s after close

                        if arb.get("bail_failed"):
                            # V1.3: One-sided position — resolve based on market outcome
                            exposed_side = arb.get("exposed_side")
                            if not exposed_side:
                                # Figure out which side we hold
                                if arb.get("up_filled") and not arb.get("down_filled"):
                                    exposed_side = "UP"
                                elif arb.get("down_filled") and not arb.get("up_filled"):
                                    exposed_side = "DOWN"
                                else:
                                    exposed_side = "UP"  # fallback

                            filled_key = "up_order" if exposed_side == "UP" else "down_order"
                            filled_order = arb.get(filled_key, {})
                            cost = filled_order.get("cost", 3.0)
                            shares = filled_order.get("shares", 6)

                            # Check market outcome via Binance
                            try:
                                outcome = await self._resolve_binance(end_dt)
                            except Exception:
                                outcome = None

                            if outcome == exposed_side:
                                # We won! Shares pay $1.00 each
                                payout = shares * 1.0
                                profit = payout - cost
                                arb["pnl"] = profit
                                arb["status"] = "resolved_exposed_win"
                                self.stats["wins"] += 1
                                print(f"\n  [EXPOSED WIN] {arb['question'][:45]}")
                                print(f"  Held {exposed_side} {shares}sh, cost=${cost:.2f} "
                                      f"-> Payout=${payout:.2f} = +${profit:.2f}")
                            else:
                                # We lost — shares worth $0
                                arb["pnl"] = -cost
                                arb["status"] = "resolved_exposed_loss"
                                self.stats["losses"] += 1
                                print(f"\n  [EXPOSED LOSS] {arb['question'][:45]}")
                                print(f"  Held {exposed_side} {shares}sh, cost=${cost:.2f} -> $0")

                            self.stats["one_sided"] += 1
                            self.stats["pnl"] += arb["pnl"]
                            self.session_pnl += arb["pnl"]
                            self.resolved.append(arb)
                            to_remove.append(cid)
                            self._save()
                        else:
                            # Normal hedge — guaranteed profit
                            up_cost = arb["up_order"]["cost"]
                            dn_cost = arb["down_order"]["cost"]
                            total_cost = up_cost + dn_cost
                            hedged_shares = min(arb["up_order"]["shares"],
                                                arb["down_order"]["shares"])
                            payout = hedged_shares  # $1.00/share guaranteed
                            profit = payout - total_cost
                            arb["pnl"] = profit
                            arb["status"] = "resolved_hedged"
                            self.stats["hedged"] += 1
                            self.stats["wins"] += 1
                            self.stats["pnl"] += profit
                            self.session_pnl += profit
                            print(f"\n  [PROFIT] {arb['question'][:45]}")
                            print(f"  Cost=${total_cost:.2f} -> Payout=${payout:.2f} "
                                  f"= +${profit:.2f}")
                            self.resolved.append(arb)
                            to_remove.append(cid)
                            self._save()
                except Exception:
                    pass

        for cid in to_remove:
            del self.active_arbs[cid]


# ============================================================================
# MAIN
# ============================================================================
async def main():
    parser = argparse.ArgumentParser(description="Pairs Arbitrage Bot V1.0")
    parser.add_argument("--live", action="store_true", help="Live trading mode")
    parser.add_argument("--mode", choices=["15M", "5M"], default="15M",
                        help="Market mode: 15M (default) or 5M")
    args = parser.parse_args()

    mode = args.mode.upper()
    lock_name = MODE_CONFIGS[mode]["lock_name"]

    pid_file = acquire_pid_lock(lock_name)
    try:
        bot = PairsArbBot(paper=not args.live, mode=mode)
        await bot.run()
    finally:
        release_pid_lock(lock_name)

if __name__ == "__main__":
    asyncio.run(main())
