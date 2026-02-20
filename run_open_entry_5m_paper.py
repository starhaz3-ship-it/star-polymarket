#!/usr/bin/env python3
"""
Open-Entry 5M BTC Paper Trader v2.0 — Adaptive ML
===================================================
Inspired by 7thStaircase strategy ($44.5K all-time, 2,485 predictions).

V2.0 changes (W:L ratio fix):
  1. Hard price cap ($0.60) — skip when signal is already priced in
  2. Adaptive delay — volatility regime (ATR/RSI) adjusts entry timing
  3. Contrarian mode — bet AGAINST when CLOB > $0.85 (market overconfident)

Three entry modes:
  MOMENTUM:   CLOB <= $0.60, bet WITH signal. Fair risk/reward.
  CONTRARIAN: CLOB >= $0.85, bet AGAINST signal. Cheap entry, big upside.
  SKIP:       $0.60-$0.85 dead zone. Signal priced in but not extreme enough.
"""

import asyncio
import json
import math
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np

from pid_lock import acquire_pid_lock, release_pid_lock
from adaptive_tuner import ParameterTuner

# ── Config ──────────────────────────────────────────────────────────────────
VERSION = "2.1"
TRADE_SIZE = 3.00               # flat fallback (used when not enough data for Kelly)
BANKROLL = 119.00               # current Polymarket balance for Kelly sizing
KELLY_FRACTION = 0.5            # Half Kelly (captures 75% of growth, much less variance)
MIN_KELLY_SIZE = 2.00           # floor: never bet less than this
MAX_KELLY_SIZE = 15.00          # ceiling: cap risk while proving edge
MIN_KELLY_TRADES = 10           # need this many trades per mode before Kelly kicks in
MIN_SHARES = 5                  # CLOB minimum
SCAN_INTERVAL = 15              # seconds between Gamma API polls
ENTRY_WINDOW = (15, 270)        # enter 15-270 sec after market open (adaptive delay controls actual)
RESOLVE_DELAY = 20              # seconds after close to check resolution
SIGNAL_THRESHOLD_BP = 1.0       # min 1bp move to pick a direction (else skip)
MAX_CONCURRENT = 5              # max open positions at once
ASSET = "BTC"

# V2.0: W:L ratio fixes
MAX_ENTRY_PRICE = 0.60          # Hard cap: skip momentum entries above this
CONTRARIAN_MIN_PRICE = 0.85     # Flip direction when CLOB dominant is above this
RSI_OVERBOUGHT = 70             # Skip UP momentum when RSI > this
RSI_OVERSOLD = 30               # Skip DOWN momentum when RSI < this
VOL_DELAY_MULT = {              # Multiply ML base delay by this
    "HIGH": 0.25,               # High vol → enter early (direction clear fast)
    "NORMAL": 0.6,              # Normal vol → moderate delay
    "LOW": 1.0,                 # Low vol → full delay (need time for signal)
}
GAMMA_URL = "https://gamma-api.polymarket.com/events"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"
CLOB_BOOK_URL = "https://clob.polymarket.com/book"
RESULTS_FILE = Path(__file__).parent / "open_entry_5m_results.json"
LOG_EVERY_N = 5                 # print summary every N cycles

# ── ML Tuner Config ─────────────────────────────────────────────────────────
OPEN_ENTRY_TUNER_CONFIG = {
    "signal_threshold_bp": {
        # How many basis points of BTC move needed before we pick a direction
        # Backtest: 0.5bp=75.5%WR/+$1090, 2.0bp=78.3%WR/+$1030, 3.0bp=80.2%WR/+$960
        "bins": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
        "labels": ["0.5", "1.0", "2.0", "3.0", "5.0", "8.0"],
        "default_idx": 2,  # start at 2.0bp (backtest sweet spot: high WR + high PnL)
        "floor_idx": 0,
        "ceil_idx": 5,
    },
    "entry_delay_sec": {
        # How many seconds after market open to enter
        # Backtest: 180s=75-80%WR (best), 120s=70-73%WR, 60s=65%WR, 30s=~50%WR
        "bins": [30, 60, 90, 120, 150, 180],
        "labels": ["30", "60", "90", "120", "150", "180"],
        "default_idx": 4,  # start at 150s (close to backtest optimal 180s)
        "floor_idx": 0,
        "ceil_idx": 5,
    },
}
TUNER_STATE_FILE = str(Path(__file__).parent / "open_entry_tuner_state.json")


def compute_kelly_size(win_rate: float, entry_price: float,
                       bankroll: float = BANKROLL) -> float:
    """Compute Half Kelly bet size for a binary Polymarket contract.

    Kelly fraction = p - (1-p)/b
    where p = win probability, b = payout odds = (1 - entry) / entry.
    Returns dollar amount to bet (Half Kelly, clamped to [MIN, MAX]).
    """
    if entry_price <= 0.02 or entry_price >= 0.98 or win_rate <= 0:
        return TRADE_SIZE  # fallback to flat size

    b = (1.0 - entry_price) / entry_price  # payout odds
    q = 1.0 - win_rate
    kelly_f = win_rate - q / b  # Kelly fraction of bankroll

    if kelly_f <= 0:
        return 0.0  # negative Kelly = don't bet

    half_kelly = kelly_f * KELLY_FRACTION * bankroll
    return max(MIN_KELLY_SIZE, min(MAX_KELLY_SIZE, round(half_kelly, 2)))


# ── Binance Live Feed ───────────────────────────────────────────────────────
class BinanceFeed:
    """Lightweight Binance BTC price feed via WebSocket."""

    def __init__(self):
        self.current_price = 0.0
        self.last_update = 0.0
        self._task = None
        self._prices_1m = []    # (timestamp, price) tuples for last 5 min
        self._candle_opens = {} # {candle_start_ts: open_price} for 5M candles
        self._indicators_cache = None
        self._indicators_ts = 0

    def start(self):
        self._task = asyncio.create_task(self._ws_loop())

    async def seed(self):
        """Seed current price from REST."""
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{BINANCE_REST_URL}?symbol=BTCUSDT&interval=1m&limit=5",
                timeout=10,
            )
            klines = r.json()
            if klines:
                self.current_price = float(klines[-1][4])
                self.last_update = time.time()
                for k in klines:
                    self._prices_1m.append((k[0] / 1000, float(k[4])))

    async def _ws_loop(self):
        import websockets
        while True:
            try:
                async with websockets.connect(BINANCE_WS_URL, ping_interval=20) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        price = float(data["p"])
                        now = time.time()
                        self.current_price = price
                        self.last_update = now
                        self._prices_1m.append((now, price))
                        # Trim to last 5 minutes
                        cutoff = now - 300
                        while self._prices_1m and self._prices_1m[0][0] < cutoff:
                            self._prices_1m.pop(0)
            except Exception as e:
                print(f"[BINANCE-WS] Error: {e}. Reconnecting in 3s...")
                await asyncio.sleep(3)

    def get_5m_candle_open(self, market_start_ts: float) -> float:
        """Get the BTC price at the start of a 5M candle."""
        # Find the first price after market_start_ts
        for ts, price in self._prices_1m:
            if ts >= market_start_ts:
                return price
        # Fallback: use REST
        return 0.0

    def get_direction_signal(self, market_start_ts: float):
        """
        Quick directional signal: compare current price vs candle open.
        Returns (side, strength_bp, detail).
        """
        now = time.time()
        if now - self.last_update > 10:
            return None, 0, "stale"

        open_price = self.get_5m_candle_open(market_start_ts)
        if open_price <= 0:
            return None, 0, "no_open"

        change_bp = (self.current_price - open_price) / open_price * 10000

        if abs(change_bp) < SIGNAL_THRESHOLD_BP:
            return None, change_bp, f"flat ({change_bp:+.1f}bp)"

        side = "UP" if change_bp > 0 else "DOWN"
        return side, change_bp, f"{side} {change_bp:+.1f}bp (open=${open_price:.2f} now=${self.current_price:.2f})"

    def is_stale(self, max_age=10):
        return time.time() - self.last_update > max_age

    async def fetch_1m_indicators(self):
        """Fetch 1M candles from Binance REST and compute ATR, RSI, EMA slope.
        Cached for 30 seconds."""
        now = time.time()
        if self._indicators_cache and now - self._indicators_ts < 30:
            return self._indicators_cache

        default = {"regime": "NORMAL", "atr_bp": 0, "rsi": 50.0, "ema_slope_bp": 0}
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    f"{BINANCE_REST_URL}?symbol=BTCUSDT&interval=1m&limit=30",
                    timeout=10,
                )
                klines = r.json()
        except Exception:
            return default

        if not klines or len(klines) < 16:
            return default

        closes = np.array([float(k[4]) for k in klines])
        highs = np.array([float(k[2]) for k in klines])
        lows = np.array([float(k[3]) for k in klines])

        # ATR(14) in basis points
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            ),
        )
        atr = np.mean(tr[-14:])
        atr_bp = atr / closes[-1] * 10000

        # RSI(14)
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses_arr = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses_arr[-14:])
        rs = avg_gain / max(avg_loss, 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # EMA(5) slope in basis points
        ema5 = np.mean(closes[-5:])
        ema5_prev = np.mean(closes[-6:-1])
        ema_slope_bp = (ema5 - ema5_prev) / closes[-1] * 10000

        # Volatility regime classification
        if atr_bp > 8:
            regime = "HIGH"
        elif atr_bp < 3:
            regime = "LOW"
        else:
            regime = "NORMAL"

        result = {
            "regime": regime,
            "atr_bp": round(atr_bp, 2),
            "rsi": round(rsi, 1),
            "ema_slope_bp": round(ema_slope_bp, 2),
        }
        self._indicators_cache = result
        self._indicators_ts = now
        return result


# ── Paper Trader ────────────────────────────────────────────────────────────
class OpenEntry5MPaper:

    def __init__(self):
        self.active = {}        # cid -> trade dict
        self.resolved = []      # list of resolved trades
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0, "skipped": 0, "entered": 0}
        # Per-mode tracking for Kelly sizing
        self.mode_stats = {
            "momentum": {"wins": 0, "losses": 0},
            "contrarian": {"wins": 0, "losses": 0},
        }
        self.attempted_cids = set()
        self.binance = BinanceFeed()
        self.tuner = ParameterTuner(OPEN_ENTRY_TUNER_CONFIG, TUNER_STATE_FILE)
        self.cycle = 0
        self.session_pnl = 0.0
        self._running = True
        self._current_indicators = {"regime": "NORMAL", "atr_bp": 0, "rsi": 50.0, "ema_slope_bp": 0}

    def load(self):
        """Load prior state from JSON."""
        if RESULTS_FILE.exists():
            try:
                data = json.loads(RESULTS_FILE.read_text())
                self.resolved = data.get("resolved", [])
                self.active = data.get("active", {})
                self.stats = data.get("stats", self.stats)
                # Rebuild attempted CIDs and per-mode stats from history
                for t in self.resolved:
                    self.attempted_cids.add(t.get("condition_id", ""))
                    mode = t.get("strategy", "momentum")
                    if mode not in self.mode_stats:
                        self.mode_stats[mode] = {"wins": 0, "losses": 0}
                    if t.get("result") == "WIN":
                        self.mode_stats[mode]["wins"] += 1
                    elif t.get("result") == "LOSS":
                        self.mode_stats[mode]["losses"] += 1
                for cid in self.active:
                    self.attempted_cids.add(cid)
                w, l = self.stats["wins"], self.stats["losses"]
                total = w + l
                wr = 100 * w / total if total else 0
                mode_summary = " | ".join(
                    f"{m}:{s['wins']}W/{s['losses']}L"
                    for m, s in self.mode_stats.items() if s['wins'] + s['losses'] > 0
                )
                print(f"[LOAD] {total} resolved, {len(self.active)} active | "
                      f"PnL: ${self.stats['pnl']:+.2f} | {w}W/{l}L {wr:.0f}%WR | {mode_summary}")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def save(self):
        """Atomic save to JSON."""
        data = {
            "active": self.active,
            "resolved": self.resolved[-500:],
            "stats": self.stats,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        tmp = RESULTS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(RESULTS_FILE)

    async def discover_markets(self):
        """Find 5M BTC markets that just opened (within entry window).

        Uses slug-based discovery: constructs the expected slug from the current
        5-minute timestamp slot and fetches directly. This is reliable because
        Polymarket 5M markets follow a deterministic naming pattern:
        btc-updown-5m-{unix_timestamp_of_candle_start}
        """
        now = datetime.now(timezone.utc)
        now_ts = int(now.timestamp())
        markets = []

        # Check current and next 5M slots
        slot_len = 300  # 5 minutes
        current_slot = (now_ts // slot_len) * slot_len
        slots_to_check = [current_slot, current_slot + slot_len]

        for slot_ts in slots_to_check:
            slug = f"btc-updown-5m-{slot_ts}"
            start_dt = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
            end_dt = start_dt + timedelta(minutes=5)
            elapsed_sec = (now - start_dt).total_seconds()

            # Adaptive delay: base ML delay × volatility multiplier
            ml_delay = self.tuner.get_active_value("entry_delay_sec")
            regime = self._current_indicators.get("regime", "NORMAL")
            vol_mult = VOL_DELAY_MULT.get(regime, 0.6)
            effective_delay = max(15, ml_delay * vol_mult)
            if not (effective_delay <= elapsed_sec <= ENTRY_WINDOW[1]):
                continue

            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(
                        f"{GAMMA_URL}?slug={slug}",
                        headers={"User-Agent": "Mozilla/5.0"},
                        timeout=10,
                    )
                    events = r.json()
            except Exception as e:
                print(f"[SCAN] Gamma API error for {slug}: {e}")
                continue

            if not events:
                continue

            event = events[0]
            for m in event.get("markets", []):
                if m.get("closed", True):
                    continue

                cid = m.get("conditionId", "")
                if cid in self.attempted_cids:
                    continue

                # Extract token IDs
                outcomes = m.get("outcomes", [])
                token_ids = m.get("clobTokenIds", [])
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                if isinstance(token_ids, str):
                    token_ids = json.loads(token_ids)

                up_token = down_token = None
                for i, o in enumerate(outcomes):
                    if str(o).lower() == "up":
                        up_token = token_ids[i]
                    elif str(o).lower() == "down":
                        down_token = token_ids[i]

                if up_token and down_token:
                    markets.append({
                        "cid": cid,
                        "title": m.get("question", event.get("title", "")),
                        "end_dt": end_dt,
                        "start_dt": start_dt,
                        "elapsed_sec": elapsed_sec,
                        "up_token": up_token,
                        "down_token": down_token,
                        "slug": slug,
                    })

        return markets

    async def get_clob_price(self, token_id: str) -> float:
        """Get best ask from CLOB order book."""
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    CLOB_BOOK_URL,
                    params={"token_id": token_id},
                    timeout=5,
                )
                book = r.json()
                asks = book.get("asks", [])
                if asks:
                    return min(float(a["price"]) for a in asks)
        except:
            pass
        return 0.50  # default to $0.50 at open

    async def enter_trade(self, market: dict):
        """Enter a paper trade with adaptive mode selection.

        Three modes based on CLOB price:
          MOMENTUM:   price <= $0.60 → bet WITH signal (fair risk/reward)
          CONTRARIAN: price >= $0.85 → bet AGAINST signal (cheap entry, big upside)
          SKIP:       $0.60-$0.85 → dead zone (signal priced in, not extreme enough)
        """
        cid = market["cid"]
        self.attempted_cids.add(cid)

        if len(self.active) >= MAX_CONCURRENT:
            self.stats["skipped"] += 1
            return

        # Get directional signal from Binance (ML-tuned threshold)
        start_ts = market["start_dt"].timestamp()
        ml_threshold = self.tuner.get_active_value("signal_threshold_bp")
        side, strength_bp, detail = self.binance.get_direction_signal(start_ts)

        # Shadow-track ALL markets for ML tuner learning
        shadow_side = "UP" if strength_bp > 0 else "DOWN" if strength_bp < 0 else None

        if side is None or abs(strength_bp) < ml_threshold:
            if shadow_side:
                self.tuner.record_shadow(
                    market_id=cid,
                    end_time_iso=market["end_dt"].isoformat(),
                    side=shadow_side,
                    entry_price=0.50,
                    clob_dominant=abs(strength_bp),
                    extra={"elapsed_sec": market["elapsed_sec"], "strength_bp": strength_bp},
                )
            self.stats["skipped"] += 1
            return

        # Shadow-track this trade too
        self.tuner.record_shadow(
            market_id=cid,
            end_time_iso=market["end_dt"].isoformat(),
            side=side,
            entry_price=0.50,
            clob_dominant=abs(strength_bp),
            extra={"elapsed_sec": market["elapsed_sec"], "strength_bp": strength_bp},
        )

        # ── Get BOTH CLOB prices ──────────────────────────────────────
        up_price = await self.get_clob_price(market["up_token"])
        down_price = await self.get_clob_price(market["down_token"])
        momentum_price = up_price if side == "UP" else down_price
        contrarian_price = down_price if side == "UP" else up_price

        # ── Mode selection based on CLOB price ────────────────────────
        indicators = self._current_indicators
        entry_side = side
        entry_price = 0.0
        strategy = None

        if momentum_price >= CONTRARIAN_MIN_PRICE:
            # CONTRARIAN MODE: market overconfident, bet the other way
            entry_side = "DOWN" if side == "UP" else "UP"
            entry_price = contrarian_price
            strategy = "contrarian"

            # Sanity: cheap side must actually be cheap
            if entry_price > (1.0 - CONTRARIAN_MIN_PRICE + 0.05):
                self.stats["skipped"] += 1
                return

        elif momentum_price <= MAX_ENTRY_PRICE:
            # MOMENTUM MODE: price is fair, bet with signal
            entry_side = side
            entry_price = momentum_price
            strategy = "momentum"

            # RSI filter: skip when indicator says move is exhausted
            rsi = indicators.get("rsi", 50.0)
            if entry_side == "UP" and rsi > RSI_OVERBOUGHT:
                self.stats["skipped"] += 1
                print(f"  [SKIP-RSI] UP but RSI={rsi:.0f} > {RSI_OVERBOUGHT} (overbought)")
                return
            if entry_side == "DOWN" and rsi < RSI_OVERSOLD:
                self.stats["skipped"] += 1
                print(f"  [SKIP-RSI] DOWN but RSI={rsi:.0f} < {RSI_OVERSOLD} (oversold)")
                return

        else:
            # DEAD ZONE: signal priced in but not extreme enough for contrarian
            self.stats["skipped"] += 1
            print(f"  [SKIP] Dead zone ${momentum_price:.2f} "
                  f"(>{MAX_ENTRY_PRICE} but <{CONTRARIAN_MIN_PRICE})")
            return

        # Final sanity
        if entry_price <= 0.02 or entry_price >= 0.98:
            self.stats["skipped"] += 1
            return

        # ── Kelly sizing ───────────────────────────────────────────────
        mode_s = self.mode_stats.get(strategy, {"wins": 0, "losses": 0})
        mode_total = mode_s["wins"] + mode_s["losses"]
        if mode_total >= MIN_KELLY_TRADES:
            mode_wr = mode_s["wins"] / mode_total
            trade_size = compute_kelly_size(mode_wr, entry_price)
            sizing_method = f"kelly({mode_wr:.0%})"
            if trade_size <= 0:
                self.stats["skipped"] += 1
                print(f"  [SKIP-KELLY] Negative Kelly for {strategy} "
                      f"({mode_wr:.0%}WR @ ${entry_price:.2f})")
                return
        else:
            trade_size = TRADE_SIZE
            sizing_method = f"flat(need {MIN_KELLY_TRADES - mode_total} more)"

        shares = max(MIN_SHARES, math.floor(trade_size / entry_price))
        cost = round(shares * entry_price, 4)
        token_id = market["up_token"] if entry_side == "UP" else market["down_token"]

        trade = {
            "condition_id": cid,
            "question": market["title"][:80],
            "side": entry_side,
            "entry_price": round(entry_price, 4),
            "shares": shares,
            "cost": round(cost, 4),
            "trade_size": trade_size,
            "sizing_method": sizing_method,
            "signal_strength_bp": round(strength_bp, 2),
            "signal_detail": detail,
            "signal_side": side,
            "elapsed_at_entry_sec": round(market["elapsed_sec"], 1),
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "end_dt": market["end_dt"].isoformat(),
            "start_dt": market["start_dt"].isoformat(),
            "btc_price_at_entry": round(self.binance.current_price, 2),
            "status": "open",
            "pnl": 0.0,
            "result": None,
            "strategy": strategy,
            "indicators": indicators,
            "paper": True,
        }

        self.active[cid] = trade
        self.stats["entered"] += 1
        self.save()

        mode_tag = "CONTRA" if strategy == "contrarian" else "MOM"
        regime = indicators.get("regime", "?")
        rsi_val = indicators.get("rsi", 0)
        print(f"  [{mode_tag}] {entry_side} @ ${entry_price:.2f} ({shares}sh/${cost:.2f}) | "
              f"Size: {sizing_method} | Signal: {detail} | Vol:{regime} RSI:{rsi_val:.0f} | "
              f"{market['title'][:40]}")

    async def resolve_trades(self):
        """Check if any active trades have resolved."""
        now = datetime.now(timezone.utc)
        to_resolve = []

        for cid, trade in list(self.active.items()):
            end_dt = datetime.fromisoformat(trade["end_dt"])
            elapsed_past_close = (now - end_dt).total_seconds()

            if elapsed_past_close < RESOLVE_DELAY:
                continue

            # Resolve via Binance REST
            outcome = await self._resolve_via_binance(trade)
            if outcome:
                to_resolve.append((cid, outcome))
            elif elapsed_past_close > 180:
                # Timeout — mark as loss
                to_resolve.append((cid, "TIMEOUT"))

        for cid, outcome in to_resolve:
            trade = self.active.pop(cid)
            won = (trade["side"] == outcome) if outcome != "TIMEOUT" else False

            mode = trade.get("strategy", "momentum")
            if mode not in self.mode_stats:
                self.mode_stats[mode] = {"wins": 0, "losses": 0}

            if won:
                pnl = round(trade["shares"] * 1.0 - trade["cost"], 4)
                trade["result"] = "WIN"
                self.stats["wins"] += 1
                self.mode_stats[mode]["wins"] += 1
            else:
                pnl = round(-trade["cost"], 4)
                trade["result"] = "LOSS"
                self.stats["losses"] += 1
                self.mode_stats[mode]["losses"] += 1

            trade["pnl"] = pnl
            trade["market_outcome"] = outcome
            trade["resolve_time"] = now.isoformat()
            trade["status"] = "closed"

            self.stats["pnl"] = round(self.stats["pnl"] + pnl, 4)
            self.session_pnl = round(self.session_pnl + pnl, 4)
            self.resolved.append(trade)

            w, l = self.stats["wins"], self.stats["losses"]
            wr = 100 * w / (w + l) if (w + l) else 0
            tag = "WIN" if won else "LOSS"
            print(f"  [{tag}] {trade['side']} | PnL: ${pnl:+.2f} | "
                  f"Entry: ${trade['entry_price']:.2f} | Outcome: {outcome} | "
                  f"Running: {w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f}")

            # Resolve shadow entries in tuner
            self.tuner.resolve_shadows({cid: outcome})

        if to_resolve:
            self.save()

    async def _resolve_shadow_entries(self):
        """Resolve shadow-tracked markets via Binance for ML tuner learning."""
        shadows = self.tuner.state.get("shadow", [])
        if not shadows:
            return

        now = datetime.now(timezone.utc)
        resolutions = {}

        for entry in shadows:
            end_time = entry.get("end_time", "")
            mid = entry.get("market_id", "")
            if not end_time or not mid:
                continue

            try:
                end_dt = datetime.fromisoformat(end_time)
            except:
                continue

            # Only resolve if market is past close + delay
            if (now - end_dt).total_seconds() < RESOLVE_DELAY:
                continue

            # Already resolved?
            if mid in resolutions:
                continue

            # Resolve via Binance
            start_dt = end_dt - timedelta(minutes=5)
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            try:
                async with httpx.AsyncClient() as client:
                    r = await client.get(
                        BINANCE_REST_URL,
                        params={
                            "symbol": "BTCUSDT",
                            "interval": "5m",
                            "startTime": start_ms,
                            "endTime": end_ms,
                            "limit": 5,
                        },
                        timeout=10,
                    )
                    klines = r.json()

                if klines:
                    best = klines[0]
                    for k in klines:
                        if abs(k[0] - start_ms) < 60000:
                            best = k
                            break
                    o, c = float(best[1]), float(best[4])
                    resolutions[mid] = "UP" if c > o else "DOWN"
            except:
                pass

        if resolutions:
            n = self.tuner.resolve_shadows(resolutions)
            if n > 0:
                total = self.tuner.state.get("total_resolved", 0)
                if total % 10 == 0:
                    print(f"[ML TUNER] {total} shadows resolved. Report:")
                    print(self.tuner.get_report())

    async def _resolve_via_binance(self, trade: dict) -> str:
        """Resolve a 5M market using Binance klines."""
        try:
            start_dt = datetime.fromisoformat(trade["start_dt"])
            end_dt = datetime.fromisoformat(trade["end_dt"])
            start_ms = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)

            async with httpx.AsyncClient() as client:
                r = await client.get(
                    BINANCE_REST_URL,
                    params={
                        "symbol": "BTCUSDT",
                        "interval": "5m",
                        "startTime": start_ms,
                        "endTime": end_ms,
                        "limit": 5,
                    },
                    timeout=10,
                )
                klines = r.json()

            if not klines:
                return None

            # Find the candle that best matches our window
            best = None
            for k in klines:
                k_open_ts = k[0]
                if abs(k_open_ts - start_ms) < 60000:  # within 1 min
                    best = k
                    break
            if not best:
                best = klines[0]

            open_price = float(best[1])
            close_price = float(best[4])

            if close_price > open_price:
                return "UP"
            elif close_price < open_price:
                return "DOWN"
            else:
                return "DOWN"  # flat = DOWN by convention

        except Exception as e:
            print(f"  [RESOLVE] Binance error: {e}")
            return None

    async def run(self):
        """Main event loop."""
        print(f"[PID] {Path('open_entry_5m').name} running as PID {__import__('os').getpid()}")

        # Seed Binance feed
        await self.binance.seed()
        self.binance.start()
        print(f"[BINANCE] WebSocket started | BTC=${self.binance.current_price:,.2f}")

        # Wait for WS to get fresh data
        await asyncio.sleep(2)

        w, l = self.stats["wins"], self.stats["losses"]
        total = w + l
        wr = 100 * w / total if total else 0
        print(f"[RESUME] {w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
              f"{total} resolved, {len(self.active)} active")

        ml_thresh = self.tuner.get_active_value("signal_threshold_bp")
        ml_delay = self.tuner.get_active_value("entry_delay_sec")
        print(f"""
======================================================================
OPEN ENTRY 5M PAPER TRADER v{VERSION} — ADAPTIVE ML
Strategy: 3-mode entry (momentum / contrarian / skip)
Signal: Binance momentum + RSI filter + volatility regime
MOMENTUM:   CLOB <= ${MAX_ENTRY_PRICE} -> bet WITH signal
CONTRARIAN: CLOB >= ${CONTRARIAN_MIN_PRICE} -> bet AGAINST signal
SKIP:       ${MAX_ENTRY_PRICE}-${CONTRARIAN_MIN_PRICE} dead zone
ML threshold: {ml_thresh}bp | Base delay: {ml_delay}s
Adaptive delay: HIGH x{VOL_DELAY_MULT["HIGH"]} / NORMAL x{VOL_DELAY_MULT["NORMAL"]} / LOW x{VOL_DELAY_MULT["LOW"]}
RSI filter: skip UP>{RSI_OVERBOUGHT} / DOWN<{RSI_OVERSOLD}
Sizing: Half Kelly (${MIN_KELLY_SIZE}-${MAX_KELLY_SIZE}), flat ${TRADE_SIZE} until {MIN_KELLY_TRADES} trades/mode
Max concurrent: {MAX_CONCURRENT}
======================================================================
""")

        while self._running:
            self.cycle += 1

            # 0. Refresh volatility indicators (cached 30s)
            self._current_indicators = await self.binance.fetch_1m_indicators()

            # 1. Discover new markets (uses adaptive delay from indicators)
            markets = await self.discover_markets()

            # 2. Enter new trades (uses mode selection from indicators)
            for market in markets:
                await self.enter_trade(market)

            # 3. Resolve completed trades
            await self.resolve_trades()

            # 4. Resolve shadow entries for ML tuner (every 4 cycles = ~60s)
            if self.cycle % 4 == 0:
                await self._resolve_shadow_entries()

            # 5. Status update
            if self.cycle % LOG_EVERY_N == 0:
                w, l = self.stats["wins"], self.stats["losses"]
                total = w + l
                wr = 100 * w / total if total else 0
                stale = " STALE!" if self.binance.is_stale() else ""
                btc_str = f"BTC=${self.binance.current_price:,.2f}" if self.binance.current_price else "BTC=?"
                ml_thresh = self.tuner.get_active_value("signal_threshold_bp")
                ml_delay = self.tuner.get_active_value("entry_delay_sec")
                regime = self._current_indicators.get("regime", "?")
                rsi = self._current_indicators.get("rsi", 0)
                atr = self._current_indicators.get("atr_bp", 0)
                vol_mult = VOL_DELAY_MULT.get(regime, 0.6)
                eff_delay = max(15, ml_delay * vol_mult)
                print(f"--- Cycle {self.cycle} | PAPER | Active: {len(self.active)} | "
                      f"{w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
                      f"Session: ${self.session_pnl:+.2f} | "
                      f"Entered: {self.stats['entered']} Skip: {self.stats['skipped']} | "
                      f"Vol:{regime} ATR:{atr:.1f}bp RSI:{rsi:.0f} delay:{eff_delay:.0f}s | "
                      f"{btc_str}{stale} ---")

            await asyncio.sleep(SCAN_INTERVAL)

    def stop(self):
        self._running = False


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    lock = acquire_pid_lock("open_entry_5m")

    trader = OpenEntry5MPaper()
    trader.load()

    # Graceful shutdown
    def handle_signal(signum, frame):
        print("\n[SHUTDOWN] Saving state...")
        trader.stop()
        trader.save()
        release_pid_lock("open_entry_5m")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        asyncio.run(trader.run())
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] KeyboardInterrupt — saving...")
    finally:
        trader.save()
        release_pid_lock("open_entry_5m")


if __name__ == "__main__":
    main()
