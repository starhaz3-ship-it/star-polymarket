"""
New Strategies Paper Trader V1.2

Runs MEAN_REVERT_EXTREME + MOMENTUM_REGIME + VWAP_REVERSION + VWAP_ST_SR + HA_KELTNER_MFI on Polymarket BTC markets.
All use 1-minute BTC candles for signal generation.

Backtest results (14-day):
  MEAN_REVERT_EXTREME_5m: 62.9% WR, +$6.90, 70 trades (BEST strategy)
  VWAP_REVERSION_5m: 59.3% WR, +$3.73, 59 trades (64.2% with hour filter)
  MOMENTUM_REGIME_5m: 53.2% WR, +$0.34, 222 trades (56.7% with hour filter)

Usage:
    python run_new_strategies_paper.py
"""

import sys
import json
import time
import os
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from functools import partial as fn_partial

import httpx

print = fn_partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
from pid_lock import acquire_pid_lock, release_pid_lock

# Import our backtest strategies (stateful, bar-by-bar)
from nautilus_backtest.strategies.mean_revert_extreme import MeanRevertExtreme
from nautilus_backtest.strategies.momentum_regime import MomentumRegime
from nautilus_backtest.strategies.vwap_reversion import VwapReversion
from nautilus_backtest.strategies.vwap_supertrend_stochrsi import VwapSupertrendStochRSI
from nautilus_backtest.strategies.ha_keltner_mfi import HaKeltnerMfi

# ============================================================================
# CONFIG
# ============================================================================
SCAN_INTERVAL = 25
MAX_CONCURRENT_PER_STRAT = 2
MAX_CONCURRENT_TOTAL = 10
TIME_WINDOW = (2.0, 12.0)
SPREAD_OFFSET = 0.03
MIN_ENTRY_PRICE = 0.10
MAX_ENTRY_PRICE = 0.60
RESOLVE_AGE_MIN = 16.0
BASE_SIZE = 3.0
MAX_SIZE = 8.0

# Hour filter — skip worst hours from 14-day backtest
SKIP_HOURS_UTC = {12, 17, 19, 20, 21}

RESULTS_FILE = Path(__file__).parent / "new_strategies_paper_results.json"

# Strategy configs: min_confidence from backtest calibration
STRAT_CONFIG = {
    "MEAN_REVERT_EXTREME": {
        "min_confidence": 0.72,  # Lower threshold — strategy is already selective (5/day)
        "min_edge": 0.005,      # Low edge threshold since strategy has inherent edge
        "base_size": 4.0,       # Bigger base — 62.9% WR is strong
        "max_size": 8.0,
    },
    "MOMENTUM_REGIME": {
        "min_confidence": 0.76,  # Slightly higher — more signals, need quality filter
        "min_edge": 0.008,
        "base_size": 3.0,
        "max_size": 6.0,
    },
    "VWAP_REVERSION": {
        "min_confidence": 0.72,  # Same as MRE — similar selectivity (59 trades / 14d)
        "min_edge": 0.005,
        "base_size": 4.0,       # 59.3% WR raw, 64.2% filtered — strong
        "max_size": 8.0,
    },
    "VWAP_ST_SR": {
        "min_confidence": 0.72,  # VWAP pullback + Supertrend + StochRSI cross
        "min_edge": 0.008,      # Slightly higher — new strategy, be cautious
        "base_size": 3.0,
        "max_size": 7.0,
    },
    "HA_KELTNER_MFI": {
        "min_confidence": 0.72,  # Heikin-Ashi trend + KC breakout + MFI
        "min_edge": 0.008,
        "base_size": 3.0,
        "max_size": 7.0,
    },
}


# ============================================================================
# PAPER TRADER
# ============================================================================

class NewStrategiesPaperTrader:
    def __init__(self):
        self.trades = {}
        self.resolved = []
        self.stats = {}
        for name in STRAT_CONFIG:
            self.stats[name] = {"wins": 0, "losses": 0, "pnl": 0.0}
        self.stats["_total"] = {"wins": 0, "losses": 0, "pnl": 0.0}

        # Create stateful strategy instances (maintain state across cycles)
        # Use 5m horizon for both since that was the winning variant
        # But we use ALL 1m candles to compute indicators — the horizon is just for cooldown
        self.strategies = {
            "MEAN_REVERT_EXTREME": MeanRevertExtreme(horizon_bars=5),
            "MOMENTUM_REGIME": MomentumRegime(horizon_bars=5),
            "VWAP_REVERSION": VwapReversion(horizon_bars=5),
            "VWAP_ST_SR": VwapSupertrendStochRSI(horizon_bars=5),
            "HA_KELTNER_MFI": HaKeltnerMfi(horizon_bars=5),
        }

        # Track which bars we've already processed to avoid duplicate signals
        self._last_bar_time = 0

        self._load()

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
                          params={"tag_slug": "5M", "active": "true", "closed": "false", "limit": 50},
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

                            self.resolved.append(trade)
                            del self.trades[tid]

                            t = self.stats["_total"]
                            total = t["wins"] + t["losses"]
                            wr = t["wins"] / total * 100 if total > 0 else 0
                            s = self.stats.get(strat, t)
                            sw = s["wins"] + s["losses"]
                            swr = s["wins"] / sw * 100 if sw > 0 else 0
                            tag = "WIN" if won else "LOSS"
                            print(f"[{tag}] {strat} {trade['side']} ${trade['pnl']:+.2f} | "
                                  f"entry=${trade['entry_price']:.2f} exit=${trade.get('exit_price', 0):.2f} | "
                                  f"conf={trade.get('confidence', 0):.2f} | "
                                  f"Strat: {s['wins']}W/{s['losses']}L {swr:.0f}%WR ${s['pnl']:+.2f} | "
                                  f"Total: {t['wins']}W/{t['losses']}L {wr:.0f}%WR ${t['pnl']:+.2f}")
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

    # ========================================================================
    # ENTRY
    # ========================================================================

    def find_entries(self, candles: list, markets: list):
        now = datetime.now(timezone.utc)
        hour_utc = now.hour

        if hour_utc in SKIP_HOURS_UTC:
            return

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
            # No new bars — use last candle for signal check
            # (strategies already have state from previous bars)
            pass
        else:
            self._last_bar_time = candles[-1]["time"]

        # Feed new bars to strategies and check for signals
        signals = {}  # strat_name -> (direction, confidence)

        for strat_name, strategy in self.strategies.items():
            # Feed ALL new bars to maintain accurate state
            direction, confidence = None, 0.0
            for bar in new_bars:
                d, c = strategy.update(bar["high"], bar["low"], bar["close"], bar["volume"])
                if d is not None:
                    direction, confidence = d, c  # Keep the latest signal

            if direction:
                cfg = STRAT_CONFIG[strat_name]
                if confidence >= cfg["min_confidence"]:
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

                side, entry_price, fair_px, edge, tte = self.compute_edge(
                    direction, fair_up, market, cfg["min_edge"]
                )
                if not side:
                    continue

                # Size: scale by confidence
                trade_size = cfg["base_size"] + (confidence - 0.65) * (cfg["max_size"] - cfg["base_size"])
                trade_size = round(max(cfg["base_size"], min(cfg["max_size"], trade_size)), 2)

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
                }
                total_open += 1
                strat_open += 1

                print(f"[ENTRY] {strat_name} {side} ${entry_price:.2f} ${trade_size:.0f} | "
                      f"conf={confidence:.2f} edge={edge:.1%} fair={fair_px:.2f} | "
                      f"tte={tte}s | BTC=${candles[-1]['close']:,.0f} | [PAPER]")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    def run(self):
        print("=" * 76)
        print("  NEW STRATEGIES PAPER TRADER V1.0")
        print(f"  Strategies: {', '.join(STRAT_CONFIG.keys())}")
        print(f"  Max {MAX_CONCURRENT_PER_STRAT}/strat, {MAX_CONCURRENT_TOTAL} total")
        print(f"  Window: {TIME_WINDOW[0]}-{TIME_WINDOW[1]}min")
        print(f"  Skip hours (UTC): {sorted(SKIP_HOURS_UTC)}")
        for name, cfg in STRAT_CONFIG.items():
            print(f"    {name}: conf>={cfg['min_confidence']} edge>={cfg['min_edge']:.1%} "
                  f"size=${cfg['base_size']}-${cfg['max_size']}")
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

                    strat_parts = []
                    for name in STRAT_CONFIG:
                        s = self.stats.get(name, {"wins": 0, "losses": 0, "pnl": 0.0})
                        st = s["wins"] + s["losses"]
                        swr = s["wins"] / st * 100 if st > 0 else 0
                        act = sum(1 for tr in self.trades.values()
                                  if tr.get("status") == "open" and tr.get("strategy") == name)
                        strat_parts.append(f"{name}:{s['wins']}W/{s['losses']}L/{swr:.0f}%/{act}a")

                    print(f"\n--- Cycle {cycle} | BTC ${btc:,.0f} | "
                          f"{t['wins']}W/{t['losses']}L {wr:.0f}%WR ${t['pnl']:+.2f} | "
                          f"Active: {len(self.trades)} | Markets: {in_window}/{len(markets)}"
                          f"{skip_tag} ---")
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
    lock = acquire_pid_lock("new_strategies_paper")
    try:
        trader = NewStrategiesPaperTrader()
        trader.run()
    finally:
        release_pid_lock("new_strategies_paper")
