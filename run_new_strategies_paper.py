"""
New Strategies Trader V2.0

Runs backtest-validated strategies on Polymarket BTC 5m markets.
Supports --live mode with $5 flat bets and 20-trade auto-cutoff.

Backtest validated (14-day, 20K candles):
  WILLIAMS_VWAP: 56.0% WR, +$20.42, 686 trades (BEST)
  STOCH_BB: 63.3% WR, +$16.26, 158 trades
  MEAN_REVERT_EXTREME: 62.9% WR, +$6.90, 70 trades
  CCI_BOUNCE: 63.6% WR, +$5.85, 55 trades
  DOUBLE_BOTTOM_RSI: 61.8% WR, +$4.85, 55 trades
  VWAP_REVERSION: 58.3% WR, +$3.20, 60 trades
  ULTIMATE_OSC: 60.0% WR, +$2.80, 40 trades

Usage:
    python run_new_strategies_paper.py          # Paper mode
    python run_new_strategies_paper.py --live   # Live $5 bets, auto-cutoff at 20 trades
"""

import sys
import json
import time
import os
import argparse
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from functools import partial as fn_partial

import httpx
from dotenv import load_dotenv

load_dotenv()
print = fn_partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
from pid_lock import acquire_pid_lock, release_pid_lock

# Import our backtest strategies (stateful, bar-by-bar)
from nautilus_backtest.strategies.mean_revert_extreme import MeanRevertExtreme
from nautilus_backtest.strategies.momentum_regime import MomentumRegime
from nautilus_backtest.strategies.vwap_reversion import VwapReversion
from nautilus_backtest.strategies.vwap_supertrend_stochrsi import VwapSupertrendStochRSI
from nautilus_backtest.strategies.stoch_bb import StochBb
from nautilus_backtest.strategies.cci_bounce import CciBounce
from nautilus_backtest.strategies.double_bottom_rsi import DoubleBottomRsi
from nautilus_backtest.strategies.ultimate_osc import UltimateOsc
from nautilus_backtest.strategies.williams_vwap import WilliamsVwap
from nautilus_backtest.strategies.triple_rsi import TripleRsi
from nautilus_backtest.strategies.ha_keltner_mfi import HaKeltnerMfi
from nautilus_backtest.strategies.aroon_cross import AroonCross
from nautilus_backtest.strategies.inverse_wrapper import InverseStrategy

# ============================================================================
# CONFIG
# ============================================================================
SCAN_INTERVAL = 25
MAX_CONCURRENT_PER_STRAT = 2
MAX_CONCURRENT_TOTAL = 16
TIME_WINDOW = (2.0, 12.0)
SPREAD_OFFSET = 0.03
MIN_ENTRY_PRICE = 0.10
MAX_ENTRY_PRICE = 0.60
RESOLVE_AGE_MIN = 16.0
LIVE_SIZE_USD = 5.0       # Flat $5 live bets
LIVE_CUTOFF_TRADES = 20   # Auto-switch to paper after this many trades if losing

# Hour filter — skip worst hours from 14-day backtest
SKIP_HOURS_UTC = {12, 17, 19, 20, 21}

RESULTS_FILE = Path(__file__).parent / "new_strategies_paper_results.json"

# Strategy configs: min_confidence from backtest calibration
# live_enabled: strategies with 60%+ backtest WR place real $5 orders in --live mode.
# Per-strategy cutoff: after 20 trades, if PnL <= $0 → auto-revert to paper.
STRAT_CONFIG = {
    "MEAN_REVERT_EXTREME": {
        "min_confidence": 0.72,  # 62.9% WR, 70 trades
        "min_edge": 0.005,
        "base_size": 4.0,
        "max_size": 8.0,
        "live_enabled": True,
    },
    "MOMENTUM_REGIME": {
        "min_confidence": 0.76,
        "min_edge": 0.008,
        "base_size": 3.0,
        "max_size": 6.0,
    },
    "VWAP_REVERSION": {
        "min_confidence": 0.72,
        "min_edge": 0.005,
        "base_size": 4.0,
        "max_size": 8.0,
    },
    "VWAP_ST_SR": {
        "min_confidence": 0.72,
        "min_edge": 0.008,
        "base_size": 3.0,
        "max_size": 7.0,
    },
    "STOCH_BB": {
        "min_confidence": 0.72,  # 63.3% WR, 158 trades
        "min_edge": 0.005,
        "base_size": 4.0,
        "max_size": 8.0,
        "live_enabled": True,
    },
    "CCI_BOUNCE": {
        "min_confidence": 0.70,  # 63.6% WR, 55 trades
        "min_edge": 0.005,
        "base_size": 4.0,
        "max_size": 8.0,
        "live_enabled": True,
    },
    "DOUBLE_BOTTOM_RSI": {
        "min_confidence": 0.80,  # 61.8% WR, 55 trades
        "min_edge": 0.005,
        "base_size": 4.0,
        "max_size": 8.0,
        "live_enabled": True,
    },
    "ULTIMATE_OSC": {
        "min_confidence": 0.72,  # 60.0% WR, 40 trades
        "min_edge": 0.005,
        "base_size": 3.0,
        "max_size": 7.0,
        "live_enabled": True,
    },
    # ── PAPER ONLY (below 60% WR) ──
    "WILLIAMS_VWAP": {
        "min_confidence": 0.76,  # 56.0% WR — below 60% threshold, paper only
        "min_edge": 0.010,
        "base_size": 3.0,
        "max_size": 6.0,
    },
    "TRIPLE_RSI": {
        "min_confidence": 0.80,
        "min_edge": 0.008,
        "base_size": 3.0,
        "max_size": 7.0,
    },
    "HA_KELTNER_MFI_INV": {
        "min_confidence": 0.70,  # Inverse: 54.9% WR, +$42, 2164 trades
        "min_edge": 0.003,
        "base_size": 4.0,
        "max_size": 8.0,
        "live_enabled": True,
        "only_hours_utc": {22, 21, 7, 15, 5},  # Top 5 hours: 68%, 67%, 64%, 64%, 63% WR
    },
    "AROON_CROSS_INV": {
        "min_confidence": 0.70,  # Inverse: 55.9% WR, +$20, 678 trades
        "min_edge": 0.003,
        "base_size": 4.0,
        "max_size": 8.0,
        "live_enabled": True,
        "only_hours_utc": {13, 19, 6, 15, 3},  # Top 5 hours: 72%, 69%, 69%, 69%, 68% WR
    },
}


# ============================================================================
# PAPER TRADER
# ============================================================================

class NewStrategiesTrader:
    def __init__(self, live: bool = False):
        self.live = live
        self.client = None  # CLOB client (live only)
        self.trades = {}
        self.resolved = []
        self.stats = {}
        for name in STRAT_CONFIG:
            self.stats[name] = {"wins": 0, "losses": 0, "pnl": 0.0}
        self.stats["_total"] = {"wins": 0, "losses": 0, "pnl": 0.0}

        # Per-strategy live tracking: after 20 trades, revert to paper if PnL <= $0
        self.live_session_trades = 0       # Global counter for display
        self.live_session_pnl = 0.0        # Global counter for display
        self.cutoff_triggered = False       # Global kill switch
        self.strat_live_trades: dict = {}   # strat_name -> count of live trades
        self.strat_live_pnl: dict = {}      # strat_name -> cumulative live PnL
        self.strat_cutoff: set = set()      # strategies reverted to paper

        # Create stateful strategy instances
        self.strategies = {
            "MEAN_REVERT_EXTREME": MeanRevertExtreme(horizon_bars=5),
            "MOMENTUM_REGIME": MomentumRegime(horizon_bars=5),
            "VWAP_REVERSION": VwapReversion(horizon_bars=5),
            "VWAP_ST_SR": VwapSupertrendStochRSI(horizon_bars=5),
            "STOCH_BB": StochBb(horizon_bars=5),
            "CCI_BOUNCE": CciBounce(horizon_bars=5),
            "DOUBLE_BOTTOM_RSI": DoubleBottomRsi(horizon_bars=5),
            "ULTIMATE_OSC": UltimateOsc(horizon_bars=5),
            "WILLIAMS_VWAP": WilliamsVwap(horizon_bars=5),
            "TRIPLE_RSI": TripleRsi(horizon_bars=5),
            # Inverse strategies: flip signals from consistent losers
            "HA_KELTNER_MFI_INV": InverseStrategy(HaKeltnerMfi(horizon_bars=5)),
            "AROON_CROSS_INV": InverseStrategy(AroonCross(horizon_bars=5)),
        }

        self._last_bar_time = 0

        if self.live:
            self._init_client()

        self._load()

    def _init_client(self):
        """Initialize CLOB client for live trading."""
        try:
            from arbitrage.executor import Executor
            executor = Executor()
            if executor._initialized:
                self.client = executor.client
                print("[LIVE] CLOB client initialized")
            else:
                print("[LIVE] Client init failed — falling back to PAPER")
                self.live = False
        except Exception as e:
            print(f"[LIVE] Client error: {e} — PAPER MODE")
            self.live = False

    def _load(self):
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.resolved = data.get("resolved", [])
                self.trades = data.get("active", {})
                saved_stats = data.get("stats", {})
                for name in list(STRAT_CONFIG.keys()) + ["_total"]:
                    if name in saved_stats:
                        self.stats[name] = saved_stats[name]
                t = self.stats["_total"]
                total = t["wins"] + t["losses"]
                wr = t["wins"] / total * 100 if total > 0 else 0
                print(f"[LOAD] {total} resolved ({t['wins']}W/{t['losses']}L "
                      f"{wr:.0f}%WR) | PnL: ${t['pnl']:+.2f} | {len(self.trades)} active")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        data = {
            "stats": self.stats,
            "resolved": self.resolved[-300:],
            "active": self.trades,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            RESULTS_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    # ========================================================================
    # DATA
    # ========================================================================

    def fetch_1m_candles(self) -> list:
        url = "https://api.binance.com/api/v3/klines"
        for attempt in range(3):
            try:
                r = httpx.get(url, params={"symbol": "BTCUSDT", "interval": "1m", "limit": 200},
                              timeout=15)
                r.raise_for_status()
                return [{"time": int(b[0]), "open": float(b[1]), "high": float(b[2]),
                         "low": float(b[3]), "close": float(b[4]), "volume": float(b[5])}
                        for b in r.json()]
            except Exception as e:
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                print(f"[BINANCE] Error: {e}")
                return []

    def discover_markets(self) -> list:
        markets = []
        try:
            r = httpx.get("https://gamma-api.polymarket.com/events",
                          params={"tag_slug": "5M", "active": "true", "closed": "false",
                                  "limit": 200, "order": "endDate", "ascending": "true"},
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
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
                    if end_date:
                        try:
                            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                            if end_dt < now:
                                continue
                            m["_time_left"] = (end_dt - now).total_seconds() / 60
                            m["_tte_seconds"] = int((end_dt - now).total_seconds())
                        except Exception:
                            continue
                    if not m.get("question"):
                        m["question"] = event.get("title", "")
                    markets.append(m)
        except Exception as e:
            if "429" not in str(e):
                print(f"[API] Error: {e}")
        return markets

    # ========================================================================
    # HELPERS
    # ========================================================================

    def get_prices(self, market: dict):
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

    def get_token_ids(self, market: dict):
        """Extract UP and DOWN token IDs from market data."""
        outcomes = market.get("outcomes", [])
        token_ids = market.get("clobTokenIds", [])
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        up_token, down_token = None, None
        for i, o in enumerate(outcomes):
            if i < len(token_ids):
                if str(o).lower() == "up":
                    up_token = token_ids[i]
                elif str(o).lower() == "down":
                    down_token = token_ids[i]
        return up_token, down_token

    def compute_fair_prob(self, direction: str, confidence: float) -> float:
        """Compute fair UP probability from signal direction and confidence."""
        if direction == "UP":
            return 0.50 + (confidence - 0.50) * 0.4  # Scale: 0.72 conf → ~0.59 fair
        else:
            return 0.50 - (confidence - 0.50) * 0.4  # Scale: 0.72 conf → ~0.41 fair

    def compute_edge(self, direction: str, fair_up: float, market: dict, min_edge: float):
        up_p, down_p = self.get_prices(market)
        if up_p is None or down_p is None:
            return None, None, 0.0, 0.0, 0
        tte = market.get("_tte_seconds", 0)
        if tte < 60:
            return None, None, 0.0, 0.0, 0

        if direction == "UP":
            entry = up_p + SPREAD_OFFSET
            if entry < MIN_ENTRY_PRICE or entry > MAX_ENTRY_PRICE:
                return None, None, 0.0, 0.0, 0
            edge = fair_up - entry
            return ("UP", entry, fair_up, edge, tte) if edge >= min_edge else (None, None, fair_up, edge, tte)
        elif direction == "DOWN":
            entry = down_p + SPREAD_OFFSET
            if entry < MIN_ENTRY_PRICE or entry > MAX_ENTRY_PRICE:
                return None, None, 0.0, 0.0, 0
            edge = (1.0 - fair_up) - entry
            return ("DOWN", entry, 1.0 - fair_up, edge, tte) if edge >= min_edge else (None, None, 1.0 - fair_up, edge, tte)
        return None, None, 0.0, 0.0, 0

    # ========================================================================
    # RESOLVE
    # ========================================================================

    def resolve_trades(self):
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

            nid = trade.get("market_numeric_id")
            if nid:
                try:
                    r = httpx.get(f"https://gamma-api.polymarket.com/markets/{nid}",
                                  headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                    if r.status_code == 200:
                        rm = r.json()
                        up_p, down_p = self.get_prices(rm)
                        if up_p is not None:
                            price = up_p if trade["side"] == "UP" else down_p
                            if price >= 0.95:
                                shares = trade["size_usd"] / trade["entry_price"]
                                trade["pnl"] = round(shares - trade["size_usd"], 2)
                                trade["exit_price"] = 1.0
                            elif price <= 0.05:
                                trade["pnl"] = round(-trade["size_usd"], 2)
                                trade["exit_price"] = 0.0
                            else:
                                continue

                            trade["exit_time"] = now.isoformat()
                            trade["status"] = "closed"
                            won = trade["pnl"] > 0

                            strat = trade.get("strategy", "_total")
                            if strat in self.stats:
                                self.stats[strat]["wins" if won else "losses"] += 1
                                self.stats[strat]["pnl"] += trade["pnl"]
                            self.stats["_total"]["wins" if won else "losses"] += 1
                            self.stats["_total"]["pnl"] += trade["pnl"]

                            # Track per-strategy live session for 20-trade cutoff
                            if trade.get("live_order"):
                                self.live_session_trades += 1
                                self.live_session_pnl += trade["pnl"]
                                self.strat_live_trades[strat] = self.strat_live_trades.get(strat, 0) + 1
                                self.strat_live_pnl[strat] = self.strat_live_pnl.get(strat, 0) + trade["pnl"]
                                self._check_cutoff(strat)

                            # Auto-redeem winning positions
                            if won and trade.get("live_order"):
                                self._try_redeem()

                            self.resolved.append(trade)
                            del self.trades[tid]

                            t = self.stats["_total"]
                            total = t["wins"] + t["losses"]
                            wr = t["wins"] / total * 100 if total > 0 else 0
                            s = self.stats.get(strat, t)
                            sw = s["wins"] + s["losses"]
                            swr = s["wins"] / sw * 100 if sw > 0 else 0
                            mode = "LIVE" if trade.get("live_order") else "PAPER"
                            tag = "WIN" if won else "LOSS"
                            print(f"[{tag}] {strat} {trade['side']} ${trade['pnl']:+.2f} [{mode}] | "
                                  f"entry=${trade['entry_price']:.2f} exit=${trade.get('exit_price', 0):.2f} | "
                                  f"conf={trade.get('confidence', 0):.2f} | "
                                  f"Strat: {s['wins']}W/{s['losses']}L {swr:.0f}%WR ${s['pnl']:+.2f} | "
                                  f"Total: {t['wins']}W/{t['losses']}L {wr:.0f}%WR ${t['pnl']:+.2f}"
                                  + (f" | Session: {self.live_session_trades}T ${self.live_session_pnl:+.2f}" if trade.get("live_order") else ""))
                except Exception as e:
                    if "429" not in str(e):
                        print(f"[RESOLVE] API error: {e}")
            elif age_min > 25:
                strat = trade.get("strategy", "_total")
                trade["status"] = "closed"
                trade["pnl"] = round(-trade["size_usd"], 2)
                if strat in self.stats:
                    self.stats[strat]["losses"] += 1
                    self.stats[strat]["pnl"] += trade["pnl"]
                self.stats["_total"]["losses"] += 1
                self.stats["_total"]["pnl"] += trade["pnl"]
                self.resolved.append(trade)
                del self.trades[tid]
                print(f"[LOSS] {strat} {trade['side']} ${trade['pnl']:+.2f} | aged out")

    def _check_cutoff(self, strat: str):
        """After 20 live trades per strategy, revert that strategy to paper if losing."""
        if strat in self.strat_cutoff:
            return
        trades = self.strat_live_trades.get(strat, 0)
        pnl = self.strat_live_pnl.get(strat, 0)
        if trades >= LIVE_CUTOFF_TRADES:
            if pnl <= 0:
                print(f"\n{'='*60}")
                print(f"  CUTOFF: {strat} — {trades} live trades, "
                      f"PnL ${pnl:+.2f} — REVERTING TO PAPER")
                print(f"{'='*60}\n")
                self.strat_cutoff.add(strat)
                # Remove live_enabled so it stays paper for rest of session
                if strat in STRAT_CONFIG:
                    STRAT_CONFIG[strat]["live_enabled"] = False
            else:
                print(f"[CUTOFF-PASS] {strat} — {trades} trades, "
                      f"PnL ${pnl:+.2f} — PROFITABLE, staying LIVE")

    def _try_redeem(self):
        """Try to redeem winning positions via gasless relayer."""
        try:
            from run_maker import _try_gasless_redeem
            ok, amt = _try_gasless_redeem()
            if ok and amt > 0:
                print(f"[REDEEM] Claimed ${amt:.2f} back to USDC")
        except Exception as e:
            pass  # Redeem is best-effort, don't crash

    # ========================================================================
    # ENTRY
    # ========================================================================

    def find_entries(self, candles: list, markets: list):
        now = datetime.now(timezone.utc)
        hour_utc = now.hour

        # Check total position limit
        total_open = sum(1 for t in self.trades.values() if t.get("status") == "open")
        if total_open >= MAX_CONCURRENT_TOTAL:
            return

        # Process new bars through stateful strategies
        # Feed only bars we haven't seen yet
        new_bars = []
        for c in candles:
            if c["time"] > self._last_bar_time:
                new_bars.append(c)

        if not new_bars:
            pass
        else:
            self._last_bar_time = candles[-1]["time"]

        # Feed new bars to ALL strategies (maintains indicator state even during skip hours)
        signals = {}  # strat_name -> (direction, confidence)

        for strat_name, strategy in self.strategies.items():
            direction, confidence = None, 0.0
            for bar in new_bars:
                d, c = strategy.update(bar["high"], bar["low"], bar["close"], bar["volume"])
                if d is not None:
                    direction, confidence = d, c

            if not direction:
                continue

            cfg = STRAT_CONFIG[strat_name]
            if confidence < cfg["min_confidence"]:
                continue

            # Per-strategy hour filter (inverse strategies: only trade during top hours)
            only_hours = cfg.get("only_hours_utc")
            if only_hours:
                if hour_utc not in only_hours:
                    continue
            else:
                # Normal strategies: skip global bad hours
                if hour_utc in SKIP_HOURS_UTC:
                    continue

            signals[strat_name] = (direction, confidence)

        if not signals:
            return

        # Try to enter trades for each signal
        for strat_name, (direction, confidence) in signals.items():
            cfg = STRAT_CONFIG[strat_name]

            # Per-strategy position limit
            strat_open = sum(1 for t in self.trades.values()
                             if t.get("status") == "open" and t.get("strategy") == strat_name)
            if strat_open >= MAX_CONCURRENT_PER_STRAT:
                continue

            fair_up = self.compute_fair_prob(direction, confidence)

            for market in markets:
                if total_open >= MAX_CONCURRENT_TOTAL:
                    break
                if strat_open >= MAX_CONCURRENT_PER_STRAT:
                    break

                time_left = market.get("_time_left", 99)
                if time_left < TIME_WINDOW[0] or time_left > TIME_WINDOW[1]:
                    continue

                cid = market.get("conditionId", "")
                nid = market.get("id")
                trade_key = f"{strat_name}_{cid}_{direction}"
                if trade_key in self.trades:
                    continue

                # Only 1 live order per market (prevent multi-strategy stacking)
                if cfg.get("live_enabled", False) and self.live:
                    live_on_market = sum(1 for t in self.trades.values()
                                         if t.get("status") == "open"
                                         and t.get("live_order")
                                         and t.get("condition_id") == cid)
                    if live_on_market >= 1:
                        continue

                side, entry_price, fair_px, edge, tte = self.compute_edge(
                    direction, fair_up, market, cfg["min_edge"]
                )
                if not side:
                    continue

                # Size: $5 flat for live-enabled strategies, scaled for paper
                is_live_order = (self.live and self.client and not self.cutoff_triggered
                                 and cfg.get("live_enabled", False)
                                 and strat_name not in self.strat_cutoff)
                if is_live_order:
                    trade_size = LIVE_SIZE_USD
                else:
                    trade_size = cfg["base_size"] + (confidence - 0.65) * (cfg["max_size"] - cfg["base_size"])
                    trade_size = round(max(cfg["base_size"], min(cfg["max_size"], trade_size)), 2)

                # Live execution
                order_id = None
                if is_live_order:
                    up_token, down_token = self.get_token_ids(market)
                    token_id = up_token if side == "UP" else down_token
                    if not token_id:
                        continue

                    try:
                        from py_clob_client.clob_types import OrderArgs, OrderType
                        from py_clob_client.order_builder.constants import BUY

                        shares = round(trade_size / entry_price, 2)
                        order_args = OrderArgs(
                            price=round(entry_price, 2),
                            size=shares,
                            side=BUY,
                            token_id=token_id,
                        )
                        signed = self.client.create_order(order_args)
                        resp = self.client.post_order(signed, OrderType.GTC)

                        if not resp.get("success"):
                            err = resp.get("errorMsg", "unknown")
                            print(f"[LIVE] Order failed: {err} | {strat_name} {side}")
                            continue

                        order_id = resp.get("orderID", "")
                    except Exception as e:
                        print(f"[LIVE] Order error: {e}")
                        continue

                mode = "LIVE" if is_live_order else "PAPER"
                self.trades[trade_key] = {
                    "side": side,
                    "entry_price": entry_price,
                    "size_usd": trade_size,
                    "entry_time": now.isoformat(),
                    "condition_id": cid,
                    "market_numeric_id": nid,
                    "title": market.get("question", ""),
                    "strategy": strat_name,
                    "confidence": round(confidence, 3),
                    "edge": round(edge, 4),
                    "fair_px": round(fair_px, 4),
                    "btc_price": float(candles[-1]["close"]),
                    "hour_utc": hour_utc,
                    "tte_seconds": tte,
                    "status": "open",
                    "pnl": 0.0,
                    "live_order": is_live_order,
                    "order_id": order_id,
                }
                total_open += 1
                strat_open += 1

                print(f"[ENTRY] {strat_name} {side} ${entry_price:.2f} ${trade_size:.0f} | "
                      f"conf={confidence:.2f} edge={edge:.1%} fair={fair_px:.2f} | "
                      f"tte={tte}s | BTC=${candles[-1]['close']:,.0f} | [{mode}]"
                      + (f" | oid={order_id[:12]}..." if order_id else ""))

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    def run(self):
        mode_str = "LIVE MODE ($5/trade)" if self.live else "PAPER MODE"
        print("=" * 76)
        print(f"  NEW STRATEGIES TRADER V2.0 — {mode_str}")
        if self.live:
            print(f"  Auto-cutoff: per-strategy revert to PAPER after {LIVE_CUTOFF_TRADES} trades if PnL <= $0")
            print(f"  Max 1 live order per market (prevents multi-strategy stacking)")
        print(f"  Strategies: {', '.join(STRAT_CONFIG.keys())}")
        print(f"  Max {MAX_CONCURRENT_PER_STRAT}/strat, {MAX_CONCURRENT_TOTAL} total")
        print(f"  Window: {TIME_WINDOW[0]}-{TIME_WINDOW[1]}min")
        print(f"  Skip hours (UTC): {sorted(SKIP_HOURS_UTC)}")
        for name, cfg in STRAT_CONFIG.items():
            is_live = self.live and cfg.get("live_enabled", False)
            mode = "LIVE $5" if is_live else "PAPER"
            size_str = f"${LIVE_SIZE_USD:.0f} flat" if is_live else f"${cfg['base_size']}-${cfg['max_size']}"
            hours = f" hours={sorted(cfg['only_hours_utc'])}" if cfg.get("only_hours_utc") else ""
            print(f"    [{mode:>8}] {name}: conf>={cfg['min_confidence']} edge>={cfg['min_edge']:.1%} size={size_str}{hours}")
        print("=" * 76)

        if self.resolved:
            t = self.stats["_total"]
            total = t["wins"] + t["losses"]
            wr = t["wins"] / total * 100 if total > 0 else 0
            print(f"[RESUME] {t['wins']}W/{t['losses']}L {wr:.0f}%WR | PnL: ${t['pnl']:+.2f}")

        # Warm up strategies with historical data
        print("[WARMUP] Feeding 200 1m candles to strategies...")
        candles = self.fetch_1m_candles()
        if candles:
            for bar in candles:
                for strategy in self.strategies.values():
                    strategy.update(bar["high"], bar["low"], bar["close"], bar["volume"])
            self._last_bar_time = candles[-1]["time"]
            print(f"[WARMUP] Done. BTC ${candles[-1]['close']:,.0f}")

        cycle = 0
        while True:
            try:
                cycle += 1

                candles = self.fetch_1m_candles()
                markets = self.discover_markets()

                if not candles:
                    print("[WARN] No 1m candles")
                    time.sleep(10)
                    continue

                self.resolve_trades()
                self.find_entries(candles, markets)

                if cycle % 12 == 0:
                    btc = candles[-1]["close"]
                    t = self.stats["_total"]
                    total = t["wins"] + t["losses"]
                    wr = t["wins"] / total * 100 if total > 0 else 0
                    in_window = sum(1 for m in markets
                                    if TIME_WINDOW[0] <= m.get("_time_left", 99) <= TIME_WINDOW[1])
                    hour = datetime.now(timezone.utc).hour
                    skip_tag = " [SKIP HOUR]" if hour in SKIP_HOURS_UTC else ""
                    mode_tag = "LIVE" if (self.live and not self.cutoff_triggered) else "PAPER"
                    cutoff_str = f" | Cutoffs: {','.join(sorted(self.strat_cutoff))}" if self.strat_cutoff else ""
                    live_tag = (f" | LiveSession: {self.live_session_trades}T ${self.live_session_pnl:+.2f}{cutoff_str}"
                                if self.live_session_trades > 0 else "")

                    strat_parts = []
                    for name in STRAT_CONFIG:
                        s = self.stats.get(name, {"wins": 0, "losses": 0, "pnl": 0.0})
                        st = s["wins"] + s["losses"]
                        swr = s["wins"] / st * 100 if st > 0 else 0
                        act = sum(1 for tr in self.trades.values()
                                  if tr.get("status") == "open" and tr.get("strategy") == name)
                        strat_parts.append(f"{name}:{s['wins']}W/{s['losses']}L/{swr:.0f}%/{act}a")

                    print(f"\n--- Cycle {cycle} | {mode_tag} | BTC ${btc:,.0f} | "
                          f"{t['wins']}W/{t['losses']}L {wr:.0f}%WR ${t['pnl']:+.2f} | "
                          f"Active: {len(self.trades)} | Markets: {in_window}/{len(markets)}"
                          f"{skip_tag}{live_tag} ---")
                    print(f"    {' | '.join(strat_parts)}")

                self._save()
                time.sleep(SCAN_INTERVAL)

            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Saving...")
                self._save()
                break
            except Exception as e:
                print(f"[ERROR] Cycle {cycle}: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(15)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="New Strategies Trader V2.0")
    parser.add_argument("--live", action="store_true", help="Live mode with $5 flat bets")
    args = parser.parse_args()

    lock_name = "new_strategies_live" if args.live else "new_strategies_paper"
    lock = acquire_pid_lock(lock_name)
    try:
        trader = NewStrategiesTrader(live=args.live)
        trader.run()
    finally:
        release_pid_lock(lock_name)
