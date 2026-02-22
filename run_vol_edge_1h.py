"""
Volatility Edge 1H Trader V2.0

Uses Black-Scholes-style binary option pricing to find mispriced
Polymarket hourly crypto Up/Down contracts.

Core idea (from Synth/OpenClaw research):
  With 15-25 minutes remaining, many contracts trade at ~85c but haven't
  fully converged. If realized volatility says the true probability is 92c,
  that's a 7c edge per share.

How it works:
  1. Fetch BTC/ETH 1-minute candles from Binance (realized volatility)
  2. Find active Polymarket 1H Up/Down markets with 15-25 min remaining
  3. Get current price + compute strike from market context
  4. P(above) = N(d1) where d1 = ln(S/K) / (sigma * sqrt(T))
  5. Compare P(above) to market's YES price
  6. If edge > min_edge (5%), bet using Kelly sizing
  7. Resolve via Binance 1H candle

Key differences from momentum bot:
  - Uses MATH (probability estimation) not momentum signals
  - Enters LATE (15-25 min remaining) when contracts are near 85c
  - Uses Kelly sizing (not fixed $2.50)
  - Only trades when estimated edge > 5%

Usage:
  python -u run_vol_edge_1h.py          # Paper mode (default)
  python -u run_vol_edge_1h.py --live   # Live mode (real CLOB orders)
"""

import sys
import json
import time
import os
import math
import asyncio
import functools
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

import argparse

import numpy as np
import httpx
from dotenv import load_dotenv

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from pid_lock import acquire_pid_lock, release_pid_lock

# ============================================================================
# CONFIG
# ============================================================================
SCAN_INTERVAL = 20          # seconds between scan cycles
LOG_INTERVAL = 120          # print status every 2 min
MAX_CONCURRENT = 4          # max open positions
MIN_BET = 2.50              # minimum bet (CLOB minimum)
MAX_BET = 5.00              # maximum bet per trade
BANKROLL = 50.0             # virtual bankroll for Kelly sizing
KELLY_FRACTION = 0.25       # quarter Kelly (conservative)

# Entry window: 15-25 min before market close (late entry)
TIME_WINDOW_MIN = 15.0      # min time remaining to enter
TIME_WINDOW_MAX = 25.0      # max time remaining to enter

# Price filter: only enter contracts between $0.60 and $0.92
# (high probability, not fully converged)
MIN_ENTRY_PRICE = 0.60
MAX_ENTRY_PRICE = 0.92

# Edge threshold: minimum estimated edge to trade
MIN_EDGE = 0.05             # 5% edge required

# Volatility estimation
VOL_WINDOW_1M = 60          # 1-minute candles for vol estimation (1 hour)
VOL_ANNUALIZATION = math.sqrt(365.25 * 24 * 60)  # annualize from 1-min returns

# Daily loss limit
DAILY_LOSS_LIMIT = 15.0

# Skip hours (UTC) — copy from momentum bot proven dead hours
SKIP_HOURS = {7, 8, 11, 12, 13, 14, 15}

FOK_SLIPPAGE = 0.02         # slippage above ask for GTC fill
MIN_SHARES = 5              # CLOB minimum order size

RESULTS_FILE = Path(__file__).parent / "vol_edge_1h_results.json"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
GAMMA_API_URL = "https://gamma-api.polymarket.com/events"

ASSET_KEYWORDS = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
}
BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
}


# ============================================================================
# MATH: Binary Option Pricing
# ============================================================================

def normal_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc for precision."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def estimate_probability(current_price: float, strike_price: float,
                         sigma_1min: float, minutes_remaining: float) -> float:
    """Estimate P(price > strike at expiry) using log-normal model.

    This is essentially the Black-Scholes formula for a binary call option
    (cash-or-nothing), assuming zero drift (no risk-free rate for short horizon).

    Args:
        current_price: Current Binance price
        strike_price: Strike/threshold for the contract
        sigma_1min: Standard deviation of 1-minute log returns
        minutes_remaining: Minutes until contract expiry

    Returns:
        Probability that price will be above strike at expiry (0 to 1)
    """
    if current_price <= 0 or strike_price <= 0 or minutes_remaining <= 0:
        return 0.5
    if sigma_1min <= 0:
        # Zero vol — deterministic
        return 1.0 if current_price > strike_price else 0.0

    # Scale 1-min vol to remaining time
    sigma_t = sigma_1min * math.sqrt(minutes_remaining)

    # d1 = ln(S/K) / (sigma * sqrt(T))
    # For binary option with zero drift: P(S_T > K) = N(d1)
    # Actually for zero-drift GBM: d = (ln(S/K) + 0.5*sigma^2*T) / (sigma*sqrt(T))
    # But for short horizons (20 min), drift term is negligible
    # Simpler: d = ln(S/K) / (sigma*sqrt(T))
    try:
        d = math.log(current_price / strike_price) / sigma_t
    except (ValueError, ZeroDivisionError):
        return 0.5

    return normal_cdf(d)


def compute_realized_vol(candles_1m: list) -> float:
    """Compute realized volatility from 1-minute candle closes.

    Returns sigma of 1-minute log returns.
    """
    if len(candles_1m) < 10:
        return 0.0

    closes = [c["close"] for c in candles_1m]
    log_returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] > 0 and closes[i] > 0:
            log_returns.append(math.log(closes[i] / closes[i - 1]))

    if len(log_returns) < 5:
        return 0.0

    # Standard deviation of 1-minute log returns
    mean_r = sum(log_returns) / len(log_returns)
    var_r = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
    return math.sqrt(var_r)


def kelly_bet_size(prob_win: float, entry_price: float,
                   bankroll: float, fraction: float = 0.25) -> float:
    """Kelly criterion for binary option bet sizing.

    For a binary bet at price p that pays $1 if win:
      b = (1 - p) / p  (odds ratio)
      Kelly f* = (b * prob - (1 - prob)) / b
               = prob - (1 - prob) / b
               = prob - (1 - prob) * p / (1 - p)

    Args:
        prob_win: Our estimated probability of winning
        entry_price: Price per share (cost)
        bankroll: Total available bankroll
        fraction: Kelly fraction (0.25 = quarter Kelly)

    Returns:
        Bet size in USD (capped at MAX_BET, floored at MIN_BET)
    """
    if entry_price <= 0 or entry_price >= 1:
        return 0.0

    # Payout odds: pay p, receive 1 if win (profit = 1 - p)
    b = (1.0 - entry_price) / entry_price  # odds

    # Kelly formula
    f_star = (b * prob_win - (1.0 - prob_win)) / b

    if f_star <= 0:
        return 0.0  # No edge — don't bet

    # Apply fraction and bankroll
    bet = f_star * fraction * bankroll

    # Clamp
    bet = max(MIN_BET, min(MAX_BET, bet))
    return round(bet, 2)


def infer_strike_from_prices(up_price: float, down_price: float,
                             current_price: float, end_dt: datetime) -> float:
    """Infer the strike price from the market's UP/DOWN prices.

    For hourly Up/Down markets, the strike is typically the open price
    of the candle covering the market window (i.e., the price at the
    start of the hour).

    If UP = 0.85 and DOWN = 0.15, the market implies P(up) = 85%.
    Using inverse normal CDF, we can back out the implied strike:
      P(up) = N(ln(S/K) / sigma_t)
      K = S * exp(-sigma_t * N_inv(P(up)))

    But this requires knowing vol, creating a circular dependency.
    Instead, for hourly contracts, the strike IS the open price of the
    hour. We approximate it as:
      strike ≈ current_price * (1 - small_adjustment_based_on_UP_price)

    In practice, for Polymarket hourly contracts:
      - "Will BTC be UP this hour?" = close > open of that hour's candle
      - Strike = open price of that hour
      - We fetch the open price directly from Binance
    """
    # This is a placeholder — the actual strike is fetched from Binance
    # in the main scan loop (the hourly candle open price)
    return current_price


# ============================================================================
# VOLATILITY EDGE PAPER TRADER
# ============================================================================

class VolEdge1HTrader:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.client = None
        self.active: Dict[str, dict] = {}
        self.resolved: list = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0}
        self.session_pnl = 0.0
        self.attempted_cids: set = set()
        self.running = True
        self._last_log_time = 0.0
        self._load()
        if not self.paper:
            self._init_clob()

    def _load(self):
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.active = data.get("active", {})
                self.resolved = data.get("resolved", [])
                self.stats = {**self.stats, **data.get("stats", {})}
                for key in self.active:
                    cid = self.active[key].get("condition_id", "")
                    if cid:
                        self.attempted_cids.add(cid)
                for trade in self.resolved:
                    cid = trade.get("condition_id", "")
                    if cid:
                        self.attempted_cids.add(cid)
                w, l = self.stats["wins"], self.stats["losses"]
                total = w + l
                wr = w / total * 100 if total > 0 else 0
                print(f"[LOAD] {len(self.resolved)} resolved, {len(self.active)} active | "
                      f"{w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f}")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        data = {
            "active": self.active,
            "resolved": self.resolved[-500:],
            "stats": self.stats,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        tmp = RESULTS_FILE.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(RESULTS_FILE)
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    def _init_clob(self):
        """Initialize CLOB client for live trading."""
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

    def get_token_ids(self, market: dict) -> tuple:
        """Extract UP/DOWN token IDs from market data."""
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

    def _verify_fill(self, order_id: str, entry_price: float, shares: float,
                     trade_size: float) -> tuple:
        """Check if a GTC order has been filled."""
        try:
            order_info = self.client.get_order(order_id)
            if isinstance(order_info, str):
                try:
                    order_info = json.loads(order_info)
                except json.JSONDecodeError:
                    return (False, entry_price, shares, trade_size)
            if not isinstance(order_info, dict):
                return (False, entry_price, shares, trade_size)
            size_matched = float(order_info.get("size_matched", 0) or 0)
            if size_matched >= MIN_SHARES:
                assoc = order_info.get("associate_trades", [])
                if assoc and assoc[0].get("price"):
                    entry_price = float(assoc[0]["price"])
                    trade_size = round(entry_price * size_matched, 2)
                    shares = size_matched
                print(f"[LIVE] FILLED: {size_matched:.0f}sh @ ${entry_price:.2f} = ${trade_size:.2f}")
                return (True, entry_price, shares, trade_size)
            return (False, entry_price, shares, trade_size)
        except Exception as e:
            print(f"[LIVE] Fill check: {e}")
            return (False, entry_price, shares, trade_size)

    # ========================================================================
    # MARKET DISCOVERY
    # ========================================================================

    async def discover_1h_markets(self) -> list:
        """Find active Polymarket 1H Up/Down markets."""
        markets = []
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    GAMMA_API_URL,
                    params={"tag_slug": "1H", "active": "true",
                            "closed": "false", "limit": 200},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if r.status_code != 200:
                    return markets

                now = datetime.now(timezone.utc)
                for event in r.json():
                    title = event.get("title", "").lower()
                    asset = None
                    for name, keywords in ASSET_KEYWORDS.items():
                        if any(kw in title for kw in keywords):
                            asset = name
                            break
                    if not asset:
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
                            time_left_min = (end_dt - now).total_seconds() / 60
                            m["_time_left_min"] = time_left_min
                            m["_end_dt"] = end_dt.isoformat()
                            m["_end_dt_parsed"] = end_dt
                            m["_asset"] = asset
                        except Exception:
                            continue

                        if not m.get("question"):
                            m["question"] = event.get("title", "")
                        markets.append(m)
        except Exception as e:
            print(f"[API] Discovery error: {e}")
        return markets

    def get_prices(self, market: dict) -> tuple:
        """Extract UP/DOWN prices from market data."""
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

    # ========================================================================
    # BINANCE DATA
    # ========================================================================

    async def fetch_binance_1m_candles(self, symbol: str, limit: int = 65) -> list:
        """Fetch 1-minute candles from Binance for vol estimation.

        Default limit=65 gives ~1 hour of data for realized vol calculation.
        """
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    r = await client.get(BINANCE_REST_URL, params={
                        "symbol": symbol, "interval": "1m", "limit": limit,
                    })
                    r.raise_for_status()
                    return [
                        {
                            "time": int(k[0]),
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                        }
                        for k in r.json()
                    ]
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                print(f"[BINANCE] Error fetching 1m candles ({symbol}): {e}")
                return []

    async def get_hourly_open_price(self, symbol: str, end_dt: datetime) -> Optional[float]:
        """Get the open price of the hourly candle that the market covers.

        For a market ending at end_dt, the hour started at (end_dt - 1h).
        The strike is the open price of that hour.
        """
        try:
            start_ms = int((end_dt - timedelta(hours=1)).timestamp() * 1000)
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(BINANCE_REST_URL, params={
                    "symbol": symbol, "interval": "1h",
                    "startTime": start_ms, "limit": 2,
                })
                r.raise_for_status()
                klines = r.json()
                if klines:
                    return float(klines[0][1])  # open price
        except Exception as e:
            print(f"[BINANCE] Error fetching hourly open ({symbol}): {e}")
        return None

    async def resolve_via_binance(self, end_dt: datetime, asset: str) -> Optional[str]:
        """Determine if asset went UP or DOWN during the 1-hour window."""
        binance_symbol = BINANCE_SYMBOLS.get(asset, "BTCUSDT")
        try:
            start_ms = int((end_dt - timedelta(hours=1)).timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(BINANCE_REST_URL, params={
                    "symbol": binance_symbol, "interval": "1h",
                    "startTime": start_ms, "endTime": end_ms, "limit": 5,
                })
                r.raise_for_status()
                klines = r.json()

                if not klines:
                    r2 = await client.get(BINANCE_REST_URL, params={
                        "symbol": binance_symbol, "interval": "5m",
                        "startTime": start_ms, "endTime": end_ms, "limit": 15,
                    })
                    r2.raise_for_status()
                    klines_5m = r2.json()
                    if not klines_5m:
                        return None
                    open_price = float(klines_5m[0][1])
                    close_price = float(klines_5m[-1][4])
                else:
                    best_candle = min(klines, key=lambda k: abs(int(k[0]) - start_ms))
                    open_price = float(best_candle[1])
                    close_price = float(best_candle[4])

                if close_price > open_price:
                    return "UP"
                elif close_price < open_price:
                    return "DOWN"
                else:
                    return "DOWN"
        except Exception as e:
            print(f"[BINANCE] Resolution error ({asset}): {e}")
            return None

    # ========================================================================
    # SIGNAL: VOLATILITY EDGE ESTIMATION
    # ========================================================================

    def compute_edge(self, current_price: float, strike_price: float,
                     sigma_1min: float, minutes_remaining: float,
                     up_price: float, down_price: float) -> dict:
        """Compute estimated probability and edge vs market prices.

        Returns dict with:
          - prob_up: estimated P(price > strike)
          - prob_down: estimated P(price < strike)
          - edge_up: prob_up - up_price (positive = underpriced UP)
          - edge_down: prob_down - down_price (positive = underpriced DOWN)
          - best_side: 'UP' or 'DOWN' (whichever has more edge)
          - best_edge: edge of best side
          - sigma_1min: realized vol used
        """
        prob_up = estimate_probability(current_price, strike_price,
                                       sigma_1min, minutes_remaining)
        prob_down = 1.0 - prob_up

        edge_up = prob_up - up_price
        edge_down = prob_down - down_price

        if edge_up > edge_down:
            best_side = "UP"
            best_edge = edge_up
        else:
            best_side = "DOWN"
            best_edge = edge_down

        return {
            "prob_up": round(prob_up, 4),
            "prob_down": round(prob_down, 4),
            "edge_up": round(edge_up, 4),
            "edge_down": round(edge_down, 4),
            "best_side": best_side,
            "best_edge": round(best_edge, 4),
            "sigma_1min": round(sigma_1min, 8),
        }

    # ========================================================================
    # ENTRY
    # ========================================================================

    async def find_entries(self, markets: list):
        """Find volatility-edge entries on 1H markets."""
        now = datetime.now(timezone.utc)

        if now.hour in SKIP_HOURS:
            return

        open_count = len(self.active)
        if open_count >= MAX_CONCURRENT:
            return

        if self.session_pnl <= -DAILY_LOSS_LIMIT:
            return

        active_assets = set()
        for trade in self.active.values():
            active_assets.add(trade.get("_asset", ""))

        # Group markets by asset
        best_per_asset: Dict[str, dict] = {}
        for market in markets:
            asset = market.get("_asset", "")
            time_left = market.get("_time_left_min", 99)

            if time_left < TIME_WINDOW_MIN or time_left > TIME_WINDOW_MAX:
                continue
            if asset in active_assets:
                continue

            cid = market.get("conditionId", "")
            if not cid or cid in self.attempted_cids:
                continue

            if asset not in best_per_asset or time_left < best_per_asset[asset].get("_time_left_min", 99):
                best_per_asset[asset] = market

        if not best_per_asset:
            return

        # Fetch data for each candidate asset
        for asset, market in list(best_per_asset.items()):
            if open_count >= MAX_CONCURRENT:
                break

            symbol = BINANCE_SYMBOLS.get(asset)
            if not symbol:
                continue

            cid = market.get("conditionId", "")
            end_dt_str = market.get("_end_dt", "")
            time_left = market.get("_time_left_min", 0)

            try:
                end_dt = datetime.fromisoformat(end_dt_str)
            except Exception:
                continue

            # Fetch 1-min candles + hourly open in parallel
            candles_task = self.fetch_binance_1m_candles(symbol, limit=65)
            open_task = self.get_hourly_open_price(symbol, end_dt)
            candles_1m, strike_price = await asyncio.gather(candles_task, open_task)

            if not candles_1m or len(candles_1m) < 10 or strike_price is None:
                continue

            current_price = candles_1m[-1]["close"]
            sigma_1min = compute_realized_vol(candles_1m)

            if sigma_1min <= 0:
                continue

            # Get market prices
            up_price, down_price = self.get_prices(market)
            if up_price is None or down_price is None:
                continue

            # Compute edge
            edge_info = self.compute_edge(
                current_price, strike_price,
                sigma_1min, time_left,
                up_price, down_price,
            )

            best_side = edge_info["best_side"]
            best_edge = edge_info["best_edge"]
            prob_win = edge_info["prob_up"] if best_side == "UP" else edge_info["prob_down"]
            entry_price = up_price if best_side == "UP" else down_price

            # Log the analysis even if we don't trade
            sigma_hourly = sigma_1min * math.sqrt(60)
            print(f"[SCAN] {asset} {time_left:.0f}min | "
                  f"S=${current_price:,.2f} K=${strike_price:,.2f} | "
                  f"sig1m={sigma_1min:.6f} sig1h={sigma_hourly:.4f} | "
                  f"P(up)={edge_info['prob_up']:.1%} mkt={up_price:.2f} | "
                  f"P(dn)={edge_info['prob_down']:.1%} mkt={down_price:.2f} | "
                  f"edge={best_edge:+.1%} -> {best_side}")

            # Entry filters
            if best_edge < MIN_EDGE:
                continue

            if entry_price < MIN_ENTRY_PRICE or entry_price > MAX_ENTRY_PRICE:
                continue

            # Kelly sizing
            bet_size = kelly_bet_size(prob_win, entry_price, BANKROLL, KELLY_FRACTION)
            if bet_size < MIN_BET:
                continue

            # Compute shares (integer for CLOB)
            entry_price_adj = round(entry_price + 0.01, 2)
            shares = int(bet_size / entry_price_adj)
            if shares < MIN_SHARES:
                shares = MIN_SHARES
            cost = round(shares * entry_price_adj, 2)

            self.attempted_cids.add(cid)

            # === PLACE TRADE ===
            trade_key = f"vol1h_{cid}_{best_side}"
            order_id = None
            fill_confirmed = False

            if not self.paper and self.client:
                try:
                    from py_clob_client.clob_types import OrderArgs, OrderType
                    from py_clob_client.order_builder.constants import BUY
                    import time as _t

                    up_tid, down_tid = self.get_token_ids(market)
                    token_id = up_tid if best_side == "UP" else down_tid
                    if not token_id:
                        print(f"[SKIP] No token ID for {asset}:{best_side}")
                        continue

                    gtc_price = min(round(entry_price + FOK_SLIPPAGE, 2), 0.99)

                    order_args = OrderArgs(
                        price=gtc_price,
                        size=shares,
                        side=BUY,
                        token_id=token_id,
                    )
                    print(f"[LIVE] GTC {asset} {best_side} {shares}sh @ ${gtc_price:.2f} "
                          f"(ask=${entry_price:.2f})")
                    signed = self.client.create_order(order_args)
                    resp = self.client.post_order(signed, OrderType.GTC)

                    if not resp.get("success"):
                        print(f"[LIVE] GTC order failed: {resp.get('errorMsg', '?')}")
                        continue

                    order_id = resp.get("orderID", "?")
                    status = resp.get("status", "")
                    print(f"[LIVE] GTC posted: {order_id[:20]}... status={status}")

                    if status == "matched":
                        _t.sleep(2)
                        fill_confirmed, entry_price_adj, shares, cost = self._verify_fill(
                            order_id, entry_price_adj, shares, cost)
                        if not fill_confirmed:
                            _t.sleep(3)
                            fill_confirmed, entry_price_adj, shares, cost = self._verify_fill(
                                order_id, entry_price_adj, shares, cost)
                        if not fill_confirmed:
                            fill_confirmed = True
                            entry_price_adj = gtc_price
                            cost = round(shares * entry_price_adj, 2)
                            print(f"[LIVE] Matched -- trusting fill {shares}sh @ ${entry_price_adj:.2f}")
                    else:
                        print(f"[LIVE] Polling for fill (30s max)...")
                        for wait_round in range(6):
                            _t.sleep(5)
                            fill_confirmed, entry_price_adj, shares, cost = self._verify_fill(
                                order_id, entry_price_adj, shares, cost)
                            if fill_confirmed:
                                break
                        if not fill_confirmed:
                            try:
                                self.client.cancel(order_id)
                                print(f"[LIVE] Unfilled after 30s -- cancelled")
                            except Exception:
                                pass
                            continue
                except Exception as e:
                    print(f"[LIVE] Order error: {e}")
                    continue

            trade = {
                "condition_id": cid,
                "market_numeric_id": market.get("id"),
                "question": market.get("question", "?"),
                "side": best_side,
                "entry_price": round(entry_price_adj, 4),
                "shares": shares,
                "size_usd": cost,
                "prob_model": round(prob_win, 4),
                "prob_market": round(entry_price, 4),
                "edge": round(best_edge, 4),
                "sigma_1min": round(sigma_1min, 8),
                "strike_price": round(strike_price, 2),
                "current_price": round(current_price, 2),
                "kelly_fraction": round(bet_size / BANKROLL, 4),
                "_asset": asset,
                "time_left_min": round(time_left, 1),
                "entry_time": now.isoformat(),
                "end_dt": end_dt_str,
                "order_id": order_id,
                "_token_ids": list(self.get_token_ids(market)),
                "_fill_confirmed": fill_confirmed or self.paper,
                "status": "open",
                "pnl": 0.0,
                "result": None,
            }

            self.active[trade_key] = trade
            open_count += 1

            mode = "LIVE" if not self.paper else "PAPER"
            print(f"\n[{mode}] {asset}:{best_side} @ ${entry_price_adj:.2f} ${cost:.2f} ({shares}sh) | "
                  f"edge={best_edge:+.1%} P(win)={prob_win:.1%} mkt={entry_price:.2f} | "
                  f"Kelly={bet_size / BANKROLL:.1%} bet=${bet_size:.2f} | "
                  f"sig1m={sigma_1min:.6f} S/K={current_price / strike_price:.5f} | "
                  f"{time_left:.0f}min left | {market.get('question', '?')[:50]}")

        self._save()

    # ========================================================================
    # RESOLUTION
    # ========================================================================

    async def resolve_trades(self):
        """Resolve trades whose 1H windows have closed."""
        now = datetime.now(timezone.utc)
        to_remove = []

        for key, trade in list(self.active.items()):
            if trade.get("status") != "open":
                continue

            end_dt_str = trade.get("end_dt", "")
            if not end_dt_str:
                continue

            try:
                end_dt = datetime.fromisoformat(end_dt_str)
            except Exception:
                continue

            seconds_past_close = (now - end_dt).total_seconds()
            if seconds_past_close < 30:
                continue

            asset = trade.get("_asset", "BTC")
            outcome = await self.resolve_via_binance(end_dt, asset)

            if outcome is None:
                if seconds_past_close > 300:
                    trade["status"] = "closed"
                    trade["result"] = "unknown"
                    trade["pnl"] = round(-trade["size_usd"], 2)
                    trade["resolve_time"] = now.isoformat()
                    self.stats["losses"] += 1
                    self.stats["pnl"] += trade["pnl"]
                    self.session_pnl += trade["pnl"]
                    self.resolved.append(trade)
                    to_remove.append(key)
                continue

            won = (trade["side"] == outcome)
            if won:
                pnl = round(trade["shares"] * 1.0 - trade["size_usd"], 2)
                trade["result"] = "WIN"
            else:
                pnl = round(-trade["size_usd"], 2)
                trade["result"] = "LOSS"

            trade["status"] = "closed"
            trade["pnl"] = pnl
            trade["market_outcome"] = outcome
            trade["resolve_time"] = now.isoformat()

            self.stats["wins" if won else "losses"] += 1
            self.stats["pnl"] += pnl
            self.session_pnl += pnl

            self.resolved.append(trade)
            to_remove.append(key)

            w, l = self.stats["wins"], self.stats["losses"]
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            tag = "WIN" if won else "LOSS"

            # Show edge accuracy
            edge = trade.get("edge", 0)
            prob_model = trade.get("prob_model", 0)

            print(f"[{tag}] {asset}:{trade['side']} ${pnl:+.2f} | "
                  f"model={prob_model:.1%} edge={edge:+.1%} | "
                  f"entry=${trade['entry_price']:.2f} | "
                  f"market={outcome} | "
                  f"{w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f} | "
                  f"session=${self.session_pnl:+.2f}")

        for key in to_remove:
            del self.active[key]

        if to_remove:
            self._save()

    # ========================================================================
    # STATUS
    # ========================================================================

    def print_status(self, markets: list):
        now = time.time()
        if now - self._last_log_time < LOG_INTERVAL:
            return
        self._last_log_time = now

        w, l = self.stats["wins"], self.stats["losses"]
        total = w + l
        wr = w / total * 100 if total > 0 else 0

        # Count markets in our entry window
        in_window = sum(1 for m in markets
                        if TIME_WINDOW_MIN <= m.get("_time_left_min", 99) <= TIME_WINDOW_MAX)

        now_utc = datetime.now(timezone.utc)
        print(f"[STATUS] {now_utc.strftime('%H:%M')}UTC | "
              f"{w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f} | "
              f"session=${self.session_pnl:+.2f} | "
              f"{len(self.active)} open | "
              f"{in_window}/{len(markets)} in window | "
              f"{'BLOCKED' if now_utc.hour in SKIP_HOURS else 'ACTIVE'}")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        mode = "LIVE" if not self.paper else "PAPER"
        print("=" * 65)
        print(f"VOLATILITY EDGE 1H {mode} TRADER V2.0")
        print("Binary option pricing model (Black-Scholes style)")
        print(f"Entry window: {TIME_WINDOW_MIN}-{TIME_WINDOW_MAX} min before close")
        print(f"Price range: ${MIN_ENTRY_PRICE}-${MAX_ENTRY_PRICE}")
        print(f"Min edge: {MIN_EDGE:.0%} | Kelly fraction: {KELLY_FRACTION:.0%}")
        print(f"Bankroll: ${BANKROLL} | Bet range: ${MIN_BET}-${MAX_BET}")
        print(f"Skip hours (UTC): {sorted(SKIP_HOURS)}")
        print("=" * 65)

        while self.running:
            try:
                markets = await self.discover_1h_markets()
                await self.resolve_trades()
                await self.find_entries(markets)
                self.print_status(markets)
            except Exception as e:
                print(f"[ERROR] {e}")
                traceback.print_exc()

            await asyncio.sleep(SCAN_INTERVAL)


# ============================================================================
# MAIN
# ============================================================================

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Live mode (real CLOB orders)")
    args = parser.parse_args()

    lock = acquire_pid_lock("vol_edge_1h")
    if not lock:
        print("[FATAL] Another vol_edge_1h instance is running. Exiting.")
        sys.exit(1)

    trader = VolEdge1HTrader(paper=not args.live)

    try:
        await trader.run()
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down...")
    finally:
        release_pid_lock(lock)
        trader._save()
        print("[DONE] Final save complete.")


if __name__ == "__main__":
    asyncio.run(main())
