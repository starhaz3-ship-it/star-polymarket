"""
Momentum 15M Directional Live Trader V1.0

Adapted from run_momentum_5m.py V1.1 for 15-minute markets.
Paper-proven: 224W/134L (358T), 63% WR, +$1,459 PnL @ $10/trade over 2 days.
  - ONLY trades momentum_strong (10m + 5m aligned)

Strategy:
  1. Fetch BTC/ETH/SOL 1-minute candles from Binance (real-time momentum)
  2. Calculate 10-min and 5-min momentum (price change over last 2-3 bars)
  3. Find active Polymarket 15M Up/Down markets
  4. If upward momentum -> buy UP token
  5. If downward momentum -> buy DOWN token
  6. Wait for market resolution (15 min expiry)

Entry rules:
  - ONLY momentum_strong: 10m AND 5m momentum aligned in same direction
  - Time window: 2.0 - 12.0 min before market close
  - Entry price range: $0.10 - $0.60 (avoid extremes)
  - Min momentum: 0.05% (10m price change)
  - $2.50/trade (CLOB minimum viable)

Safety:
  - Daily loss limit: $15
  - Max 3 concurrent positions
  - Session PnL tracking with auto-stop

Usage:
  python run_momentum_15m_live.py          # Paper mode (default)
  python run_momentum_15m_live.py --live   # Live mode (real CLOB orders)
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
import websockets
from collections import deque
from dotenv import load_dotenv

print = fn_partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

# ============================================================================
# PID LOCK (prevent duplicate instances)
# ============================================================================
from pid_lock import acquire_pid_lock, release_pid_lock
from hmm_regime_filter import HMMRegimeFilter

# ============================================================================
# CONFIG
# ============================================================================
SCAN_INTERVAL = 30          # seconds between cycles
MAX_CONCURRENT = 3          # max open positions
MIN_MOMENTUM_10M = 0.0005   # V1.9b: Loosened from 0.08% to 0.05% — more trades
MIN_MOMENTUM_5M = 0.0002    # V1.9b: Loosened from 0.03% to 0.02%
RSI_CONFIRM_UP = 60         # V1.9b: Loosened from 65 to 60 — still filters junk, more signals
RSI_CONFIRM_DOWN = 35       # V1.7: Tightened from 45 to 35
# V1.9: Contrarian DISABLED (backtest: 8% WR)
RSI_CONTRARIAN_LO = 25
RSI_CONTRARIAN_HI = 35
# V1.9: Moderate DOWN band — RSI 30-50 with bearish momentum → bet DOWN
RSI_MODERATE_DOWN_LO = 30   # V1.9b: Widened from 35 to 30
RSI_MODERATE_DOWN_HI = 50
MIN_ENTRY = 0.35            # V1.1: Raised from $0.10 — entries <$0.35 were 0W/3L (0% WR, -$7.50)
MAX_ENTRY = 0.60            # maximum entry price (avoid paying premium)
TIME_WINDOW = (2.0, 8.0)    # V1.9b: Widened from (2,5) to (2,8) — more entry opportunities
SPREAD_OFFSET = 0.02        # paper fill spread simulation (ignored in live)
RESOLVE_AGE_MIN = 18.0      # minutes before forcing resolution via API
EMA_GAP_MIN_BP = None        # V1.7: REMOVED — was 2bp, cost 24% signals for 0.9% WR gain
DAILY_LOSS_LIMIT = 15.0     # stop trading if session losses exceed $15
AUTO_KILL_TRADES = 20       # V1.2: auto-shutdown after this many trades if WR < 50%
AUTO_KILL_MIN_WR = 0.50     # V1.2: minimum win rate to stay alive
MIN_SHARES = 5              # CLOB minimum order size
FOK_SLIPPAGE = 0.03         # cents above best ask to sweep multiple price levels

# V1.6: Streak reversal — bet AGAINST long streaks (backtested on 50K markets: 55% WR)
STREAK_REVERSAL_MIN_STREAK = 3   # Minimum streak length to trigger reversal entry
STREAK_REVERSAL_SIZE = 2.50      # Always minimal size (user directive)

# ============================================================================
# PROMOTION PIPELINE: PROBATION -> PROMOTED -> CHAMPION
# ============================================================================
TIERS = {
    "PROBATION":  {"size": 3.00, "min_trades": 0,  "promote_after": 20, "promote_wr": 0.55, "promote_profitable": True},
    "PROMOTED":   {"size": 5.00, "min_trades": 20, "promote_after": 50, "promote_wr": 0.65, "promote_profitable": True},
    "CHAMPION":   {"size": 10.00, "min_trades": 50, "promote_after": None, "promote_wr": None, "promote_profitable": None},
}
TIER_ORDER = ["PROBATION", "PROMOTED", "CHAMPION"]

# Multi-asset support (matches paper bot scope)
ASSETS = {
    "BTC": {"symbol": "BTCUSDT", "keywords": ["bitcoin", "btc"]},
    # ETH PAUSED V1.7 — 58% WR vs BTC 81%, underperforming
    # "ETH": {"symbol": "ETHUSDT", "keywords": ["ethereum", "eth"]},
    # SOL BLOCKED — never trade SOL
}

RESULTS_FILE = Path(__file__).parent / "momentum_15m_results.json"


# ============================================================================
# V1.4: ADAPTIVE FILTER ML — Thompson sampling on filter presets
# ============================================================================
class AdaptiveFilterML:
    """
    Learns optimal momentum/RSI filter strictness via Thompson sampling.

    4 presets from relaxed to ultra-strict. Each has Beta(alpha, beta) prior.
    On each entry opportunity: sample all presets, pick highest, use its filters.
    On resolution: update all presets that would have taken the trade.
    Shadow-tracks what each preset would have done for faster learning.
    """
    PRESETS = {
        "relaxed":  {"m10": 0.0005, "m5": 0.0002, "rsi_up": 60, "rsi_dn": 40},  # V1.7: shifted up
        "moderate": {"m10": 0.0008, "m5": 0.0003, "rsi_up": 65, "rsi_dn": 35},  # V1.7: was 55/45
        "strict":   {"m10": 0.0012, "m5": 0.0005, "rsi_up": 68, "rsi_dn": 32},  # V1.7: was 58/42
        "ultra":    {"m10": 0.0018, "m5": 0.0008, "rsi_up": 72, "rsi_dn": 28},  # V1.7: was 60/40
    }
    STATE_FILE = Path(__file__).parent / "momentum_ml_filter_state.json"
    # Initial priors: moderate gets a slight head start (our V1.3 default)
    DEFAULT_PRIORS = {
        "relaxed":  [2, 2],
        "moderate": [3, 1],
        "strict":   [2, 2],
        "ultra":    [1, 2],
    }

    def __init__(self):
        self.betas = {}  # {preset_name: [alpha, beta]}
        self.history = []  # [{preset, won, m10, m5, rsi, ...}]
        self._load()

    def _load(self):
        try:
            with open(self.STATE_FILE) as f:
                data = json.load(f)
            self.betas = data.get("betas", {})
            self.history = data.get("history", [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        # Ensure all presets have priors
        for name, prior in self.DEFAULT_PRIORS.items():
            if name not in self.betas:
                self.betas[name] = list(prior)

    def _save(self):
        with open(self.STATE_FILE, "w") as f:
            json.dump({"betas": self.betas, "history": self.history[-500:]}, f, indent=2)

    def would_pass(self, preset_name: str, abs_m10: float, abs_m5: float,
                   rsi: float, side: str) -> bool:
        """Check if a candidate entry would pass a given preset's filters."""
        p = self.PRESETS[preset_name]
        if abs_m10 < p["m10"] or abs_m5 < p["m5"]:
            return False
        if side == "UP" and rsi < p["rsi_up"]:
            return False
        if side == "DOWN" and rsi > p["rsi_dn"]:
            return False
        return True

    def select_preset(self) -> str:
        """Thompson sampling: sample from each preset's Beta, return highest."""
        import random
        best_name, best_sample = None, -1
        for name, (a, b) in self.betas.items():
            sample = random.betavariate(max(a, 0.1), max(b, 0.1))
            if sample > best_sample:
                best_sample = sample
                best_name = name
        return best_name

    def get_active_filters(self) -> dict:
        """Return the filter thresholds for the selected preset."""
        # V1.9b: Lock to relaxed — Thompson sampling with no data randomly picks
        # strict/ultra which blocks most signals. Let it learn, but floor at relaxed.
        selected = self.select_preset()
        # Use the LESS restrictive of selected vs relaxed for each threshold
        relaxed = self.PRESETS["relaxed"]
        active = self.PRESETS[selected]
        merged = {
            "preset": selected,
            "m10": min(active["m10"], relaxed["m10"]),
            "m5": min(active["m5"], relaxed["m5"]),
            "rsi_up": min(active["rsi_up"], relaxed["rsi_up"]),
            "rsi_dn": max(active["rsi_dn"], relaxed["rsi_dn"]),
        }
        return merged

    def record_outcome(self, won: bool, abs_m10: float, abs_m5: float,
                       rsi: float, side: str, preset_used: str):
        """Update all presets that would have taken this trade."""
        record = {
            "preset_used": preset_used, "won": won,
            "abs_m10": abs_m10, "abs_m5": abs_m5, "rsi": rsi, "side": side,
        }
        shadow = {}
        for name in self.PRESETS:
            if self.would_pass(name, abs_m10, abs_m5, rsi, side):
                # This preset would have taken this trade — update it
                if won:
                    self.betas[name][0] += 1  # alpha++
                else:
                    self.betas[name][1] += 1  # beta++
                shadow[name] = "W" if won else "L"
        record["shadow"] = shadow
        self.history.append(record)
        self._save()
        return shadow

    def get_report(self) -> str:
        """Summary of each preset's learned performance."""
        lines = ["[ML FILTER] Preset performance (Thompson sampling):"]
        for name in ["relaxed", "moderate", "strict", "ultra"]:
            a, b = self.betas.get(name, [1, 1])
            total = a + b - sum(self.DEFAULT_PRIORS.get(name, [0, 0]))
            wr = a / (a + b) * 100 if (a + b) > 0 else 0
            lines.append(f"  {name:10s} | alpha={a:.0f} beta={b:.0f} | "
                         f"~{wr:.0f}%WR | {total:.0f} data points")
        # Current favorite
        best = max(self.betas.items(), key=lambda x: x[1][0] / (x[1][0] + x[1][1]))
        lines.append(f"  Favorite: {best[0]} ({best[1][0]/(best[1][0]+best[1][1])*100:.0f}% mean)")
        return "\n".join(lines)


# ============================================================================
# V1.7 CHANGE TRACKER — shadow-tracks old vs new settings for auto-revert
# ============================================================================
class V17ChangeTracker:
    """
    Tracks V1.7 parameter changes vs V1.6 baseline.
    Shadow-records what old settings would have done on each trade.
    If new settings underperform old by >10% WR over 30+ trades, flags for revert.
    """
    STATE_FILE = Path(__file__).parent / "v17_change_tracker.json"
    OLD_SETTINGS = {
        "time_window": (2.0, 12.0),
        "rsi_up": 55, "rsi_dn": 45,
        "ema_gap_bp": 2,
    }
    NEW_SETTINGS = {
        "time_window": (2.0, 5.0),
        "rsi_up": 65, "rsi_dn": 35,
        "ema_gap_bp": None,
    }

    def __init__(self):
        self.trades = []  # [{won, time_left, rsi, ema_gap_bp, old_would_enter, new_entered}]
        self._load()

    def _load(self):
        try:
            with open(self.STATE_FILE) as f:
                self.trades = json.load(f).get("trades", [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def _save(self):
        with open(self.STATE_FILE, "w") as f:
            json.dump({"trades": self.trades[-200:]}, f, indent=2)

    def record_trade(self, won: bool, time_left: float, rsi: float,
                     ema_gap_bp: float, side: str):
        """Record a trade and compute what old settings would have done."""
        old_time_ok = self.OLD_SETTINGS["time_window"][0] <= time_left <= self.OLD_SETTINGS["time_window"][1]
        old_rsi_ok = True
        if side == "UP" and rsi < self.OLD_SETTINGS["rsi_up"]:
            old_rsi_ok = False
        if side == "DOWN" and rsi > self.OLD_SETTINGS["rsi_dn"]:
            old_rsi_ok = False
        old_ema_ok = ema_gap_bp is None or ema_gap_bp >= self.OLD_SETTINGS["ema_gap_bp"]
        old_would_enter = old_time_ok and old_rsi_ok and old_ema_ok

        self.trades.append({
            "won": won, "time_left": round(time_left, 1),
            "rsi": round(rsi, 1) if rsi else None,
            "ema_gap_bp": round(ema_gap_bp, 1) if ema_gap_bp else None,
            "side": side, "old_would_enter": old_would_enter,
        })
        self._save()

    def record_skipped(self, time_left: float, rsi: float, ema_gap_bp: float,
                       side: str, reason: str):
        """Record an entry the NEW settings skipped (for tracking missed signals)."""
        old_time_ok = self.OLD_SETTINGS["time_window"][0] <= time_left <= self.OLD_SETTINGS["time_window"][1]
        old_rsi_ok = True
        if side == "UP" and rsi < self.OLD_SETTINGS["rsi_up"]:
            old_rsi_ok = False
        if side == "DOWN" and rsi > self.OLD_SETTINGS["rsi_dn"]:
            old_rsi_ok = False
        old_ema_ok = ema_gap_bp is None or ema_gap_bp >= self.OLD_SETTINGS["ema_gap_bp"]
        if old_time_ok and old_rsi_ok and old_ema_ok:
            self.trades.append({
                "won": None, "time_left": round(time_left, 1),
                "rsi": round(rsi, 1) if rsi else None,
                "ema_gap_bp": round(ema_gap_bp, 1) if ema_gap_bp else None,
                "side": side, "old_would_enter": True,
                "new_skipped": True, "skip_reason": reason,
            })
            self._save()

    def get_report(self) -> str:
        """Compare new vs old settings performance."""
        new_trades = [t for t in self.trades if t.get("won") is not None]
        old_overlap = [t for t in new_trades if t.get("old_would_enter")]
        new_only = [t for t in new_trades if not t.get("old_would_enter")]
        old_skipped = [t for t in self.trades if t.get("new_skipped")]

        new_w = sum(1 for t in new_trades if t["won"])
        new_l = len(new_trades) - new_w
        new_wr = new_w / len(new_trades) * 100 if new_trades else 0

        lines = [f"[V1.7 TRACKER] {len(new_trades)} trades | {new_w}W/{new_l}L ({new_wr:.0f}% WR)"]
        if new_only:
            nw = sum(1 for t in new_only if t["won"])
            lines.append(f"  New-only entries (EMA removed): {len(new_only)} trades, {nw}W/{len(new_only)-nw}L")
        if old_skipped:
            lines.append(f"  Old would have entered (new skipped): {len(old_skipped)} "
                        f"(time_window: {sum(1 for t in old_skipped if t.get('skip_reason')=='time_window')}, "
                        f"rsi: {sum(1 for t in old_skipped if t.get('skip_reason')=='rsi')})")

        # Revert check
        if len(new_trades) >= 30 and new_wr < 70:
            lines.append(f"  *** WARNING: V1.7 WR={new_wr:.0f}% below 70% threshold — CONSIDER REVERTING ***")
        return "\n".join(lines)


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
# V2.0: BINANCE LIVE FEED — WebSocket tick stream → candles + spike detection
# ============================================================================
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
BINANCE_REST_URL = "https://api.binance.com/api/v3/klines"


class BinanceLiveFeed:
    """Real-time Binance price feed with 1-min candle aggregation and spike detection."""

    def __init__(self):
        self.current_price: Optional[float] = None
        self.last_update: float = 0
        self._running = False
        self._ws = None
        # Rolling 1-min candle buffer (same format as fetch_binance_candles)
        self._candles: deque = deque(maxlen=35)  # extra headroom over 30
        self._current_minute: int = 0  # minute boundary for candle bucketing
        self._minute_ticks: list = []  # ticks in current minute
        # Tick buffer for spike detection (last 60s of ticks)
        self._tick_buffer: deque = deque(maxlen=3000)  # ~50 ticks/sec worst case
        self._connected = False
        self._ws_started = False

    async def start(self):
        """Start WebSocket feed as background task. Seeds candles from REST first."""
        self._running = True
        # Seed candle buffer from REST so indicators work immediately
        await self._seed_from_rest()
        # Launch WS in background
        asyncio.create_task(self._ws_loop())
        self._ws_started = True
        print(f"[WS] Binance live feed started | {len(self._candles)} candles seeded")

    async def _seed_from_rest(self):
        """Fetch initial 30 candles from REST to bootstrap RSI/EMA."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(BINANCE_REST_URL, params={
                    "symbol": "BTCUSDT", "interval": "1m", "limit": 30
                })
                r.raise_for_status()
                for k in r.json():
                    self._candles.append({"time": int(k[0]), "close": float(k[4])})
                if self._candles:
                    self.current_price = self._candles[-1]["close"]
                    self.last_update = time.time()
                    self._current_minute = int(time.time()) // 60
        except Exception as e:
            print(f"[WS] REST seed error: {e}")

    async def _ws_loop(self):
        """Persistent WebSocket connection with auto-reconnect."""
        consecutive_failures = 0
        while self._running:
            try:
                async with websockets.connect(
                    BINANCE_WS_URL,
                    ping_interval=20, ping_timeout=10,
                    close_timeout=5, open_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    consecutive_failures = 0
                    print(f"[WS] Connected to Binance BTC/USDT trade stream")

                    async for message in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(message)
                            price = float(data["p"])
                            now = time.time()
                            self.current_price = price
                            self.last_update = now

                            # Store tick for spike detection
                            self._tick_buffer.append((now, price))

                            # Aggregate into 1-min candles
                            minute = int(now) // 60
                            if minute != self._current_minute:
                                # Close previous minute candle
                                if self._minute_ticks:
                                    self._candles.append({
                                        "time": self._current_minute * 60 * 1000,
                                        "close": self._minute_ticks[-1]
                                    })
                                self._current_minute = minute
                                self._minute_ticks = [price]
                            else:
                                self._minute_ticks.append(price)

                        except (KeyError, ValueError):
                            pass

            except websockets.exceptions.ConnectionClosed:
                consecutive_failures += 1
                self._connected = False
                print(f"[WS] Connection closed, reconnecting... ({consecutive_failures})")
                await asyncio.sleep(1)
            except Exception as e:
                consecutive_failures += 1
                self._connected = False
                if consecutive_failures >= 3:
                    print(f"[WS] Unstable ({e}), REST fallback active")
                    consecutive_failures = 0
                await asyncio.sleep(2)

    def get_candles(self) -> list:
        """Return candle buffer in same format as fetch_binance_candles().
        If WS has a partial current candle, append it as the latest."""
        candles = list(self._candles)
        # Append in-progress candle if we have ticks
        if self._minute_ticks and self.current_price:
            candles.append({
                "time": self._current_minute * 60 * 1000,
                "close": self._minute_ticks[-1]
            })
        return candles[-30:]  # return last 30

    def get_spike(self) -> dict:
        """Detect rapid price moves over last 5/15/30 seconds.
        Returns percentage changes (e.g., 0.002 = 0.2%)."""
        now = time.time()
        result = {"delta_5s": 0.0, "delta_15s": 0.0, "delta_30s": 0.0}
        if not self._tick_buffer or self.current_price is None:
            return result
        current = self.current_price
        for label, lookback in [("delta_5s", 5), ("delta_15s", 15), ("delta_30s", 30)]:
            cutoff = now - lookback
            # Find oldest tick within lookback window
            old_price = None
            for ts, p in self._tick_buffer:
                if ts >= cutoff:
                    old_price = p
                    break
            if old_price and old_price > 0:
                result[label] = (current - old_price) / old_price
        return result

    def is_stale(self, max_age_sec: float = 10.0) -> bool:
        """Check if data is stale (WS disconnected)."""
        if self.current_price is None:
            return True
        return (time.time() - self.last_update) > max_age_sec

    def stop(self):
        self._running = False
        if self._ws:
            try:
                asyncio.create_task(self._ws.close())
            except Exception:
                pass


# ============================================================================
# MOMENTUM 15M TRADER
# ============================================================================
class Momentum15MTrader:
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.client = None  # ClobClient for live mode
        self.trades: Dict[str, dict] = {}  # {trade_key: trade_dict}
        self.resolved: list = []
        self.stats = {"wins": 0, "losses": 0, "pnl": 0.0, "start_time": datetime.now(timezone.utc).isoformat()}
        self.session_pnl = 0.0  # session loss tracking for daily limit
        self.running = True     # V1.2: auto-kill sets this to False
        self.tier = "PROBATION"  # current promotion tier
        self.filter_ml = AdaptiveFilterML()  # V1.4: adaptive filter ML
        self.v17_tracker = V17ChangeTracker()  # V1.7: track parameter changes
        self.feed = BinanceLiveFeed()  # V2.0: WebSocket price feed
        self.hmm = HMMRegimeFilter()   # V2.1: HMM regime filter
        self._load()
        self._init_streak_tracker()  # V1.5: streak detection

        if not self.paper:
            self._init_clob()

    # ========================================================================
    # V1.5: STREAK DETECTION — track consecutive market outcomes per asset
    # ========================================================================

    def _init_streak_tracker(self):
        """Initialize streak tracking from resolved outcomes."""
        # Per-asset outcome history: {"BTC": ["UP", "DOWN", "UP", ...], "ETH": [...]}
        self._streak_outcomes: Dict[str, list] = {"BTC": [], "ETH": []}
        self._streak_shadow: list = []  # Shadow log for streak reversal strategy

        # Rebuild from resolved trades (they record actual market outcome)
        for trade in self.resolved:
            asset = trade.get("_asset", trade.get("asset", ""))
            side = trade.get("side", "")
            won = trade.get("pnl", 0) > 0
            if not asset or not side:
                continue
            # The market outcome is the OPPOSITE of our side if we lost
            outcome = side if won else ("DOWN" if side == "UP" else "UP")
            if asset in self._streak_outcomes:
                self._streak_outcomes[asset].append(outcome)

        # Keep last 20 outcomes per asset
        for asset in self._streak_outcomes:
            self._streak_outcomes[asset] = self._streak_outcomes[asset][-20:]

    def record_market_outcome(self, asset: str, outcome: str):
        """Record a resolved market outcome (UP or DOWN) for streak tracking."""
        if asset not in self._streak_outcomes:
            self._streak_outcomes[asset] = []
        self._streak_outcomes[asset].append(outcome)
        # Keep last 20
        if len(self._streak_outcomes[asset]) > 20:
            self._streak_outcomes[asset] = self._streak_outcomes[asset][-20:]

    def get_streak(self, asset: str) -> tuple:
        """Get current streak for an asset.
        Returns (streak_length, streak_direction) e.g. (4, "UP") or (2, "DOWN").
        streak_length=0 means no streak (mixed recent outcomes)."""
        outcomes = self._streak_outcomes.get(asset, [])
        if not outcomes:
            return (0, None)

        last = outcomes[-1]
        streak = 0
        for o in reversed(outcomes):
            if o == last:
                streak += 1
            else:
                break
        return (streak, last)

    def streak_confluence(self, asset: str, proposed_side: str) -> dict:
        """Evaluate streak signal for a proposed entry.
        Returns {streak_len, streak_dir, alignment, boost}.
        alignment: 'WITH' if betting same as streak, 'AGAINST' if reversal, 'NEUTRAL'.
        boost: multiplier for entry confidence (1.0 = no effect)."""
        streak_len, streak_dir = self.get_streak(asset)

        if streak_len < 2 or streak_dir is None:
            return {"streak_len": streak_len, "streak_dir": streak_dir,
                    "alignment": "NEUTRAL", "boost": 1.0}

        if proposed_side == streak_dir:
            alignment = "WITH"
        else:
            alignment = "AGAINST"

        # Reversal rates from 50,348-market PolyData backtest (Feb 2026):
        # Conservative estimates — old 570-market sample was overly optimistic
        reversal_rates = {2: 0.53, 3: 0.55, 4: 0.56, 5: 0.56}
        capped_len = min(streak_len, 5)
        reversal_prob = reversal_rates.get(capped_len, 0.50)

        if alignment == "WITH":
            # Betting WITH a streak — continuation is less likely at long streaks
            continuation_prob = 1.0 - reversal_prob
            if streak_len >= 4:
                boost = 0.7  # Moderate penalty — 56% reversal rate (50K backtest)
            elif streak_len >= 3:
                boost = 0.85  # Slight penalty — 55% reversal rate
            else:
                boost = 1.0
        else:
            # Betting AGAINST a streak — reversal expected
            if streak_len >= 4:
                boost = 1.15  # Modest boost — 56% reversal rate (50K backtest)
            elif streak_len >= 3:
                boost = 1.05  # Slight boost
            else:
                boost = 1.0

        return {
            "streak_len": streak_len, "streak_dir": streak_dir,
            "alignment": alignment, "boost": boost,
            "reversal_prob": reversal_prob,
        }

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
                self.tier = data.get("tier", "PROBATION")
                print(f"[LOAD] {len(self.resolved)} resolved, {len(self.trades)} active | "
                      f"PnL: ${self.stats['pnl']:+.2f} | {self.stats['wins']}W/{self.stats['losses']}L | "
                      f"Tier: {self.tier} (${TIERS[self.tier]['size']}/trade)")
            except Exception as e:
                print(f"[LOAD] Error: {e}")

    def _save(self):
        data = {
            "active": self.trades,
            "resolved": self.resolved,
            "stats": self.stats,
            "tier": self.tier,
            "session_pnl": self.session_pnl,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            RESULTS_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"[SAVE] Error: {e}")

    # ========================================================================
    # DATA FETCHING
    # ========================================================================

    async def fetch_binance_candles(self, symbol: str = "BTCUSDT") -> list:
        """Fetch 1-minute candles from Binance."""
        url = "https://api.binance.com/api/v3/klines"
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    r = await client.get(url, params={"symbol": symbol, "interval": "1m", "limit": 30})
                    r.raise_for_status()
                    return [{"time": int(k[0]), "close": float(k[4])} for k in r.json()]
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                print(f"[BINANCE] Error ({symbol}): {e}")
                return []

    async def discover_15m_markets(self) -> list:
        """Find active Polymarket 15M Up/Down markets for all assets."""
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

                    # Detect which asset this market is for
                    asset = None
                    for asset_name, cfg in ASSETS.items():
                        if any(kw in title for kw in cfg["keywords"]):
                            asset = asset_name
                            break
                    if not asset:
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
                                time_left = (end_dt - now).total_seconds() / 60
                                m["_time_left"] = time_left
                            except Exception:
                                continue
                        if not m.get("question"):
                            m["question"] = event.get("title", "")
                        m["_asset"] = asset
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

    def _verify_fill(self, order_id: str, entry_price: float, shares: float,
                     trade_size: float) -> tuple:
        """Check if a GTC order has been filled. Returns (filled, price, shares, cost)."""
        try:
            order_info = self.client.get_order(order_id)
            # Handle string response from get_order()
            if isinstance(order_info, str):
                try:
                    order_info = json.loads(order_info)
                except json.JSONDecodeError:
                    return (False, entry_price, shares, trade_size)
            if not isinstance(order_info, dict):
                return (False, entry_price, shares, trade_size)

            size_matched = float(order_info.get("size_matched", 0) or 0)
            if size_matched >= MIN_SHARES:
                # Use actual fill price if available
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
    # PROMOTION PIPELINE
    # ========================================================================

    def check_promotion(self):
        """Check if we should promote to the next tier based on live results."""
        tier_cfg = TIERS[self.tier]
        promote_after = tier_cfg["promote_after"]
        if promote_after is None:
            return  # Already at max tier (CHAMPION)

        total_trades = self.stats["wins"] + self.stats["losses"]
        if total_trades < promote_after:
            return  # Not enough trades yet

        wr = self.stats["wins"] / total_trades if total_trades > 0 else 0
        profitable = self.stats["pnl"] > 0

        # Check promotion criteria
        if wr >= tier_cfg["promote_wr"] and profitable:
            current_idx = TIER_ORDER.index(self.tier)
            next_tier = TIER_ORDER[current_idx + 1]
            old_size = tier_cfg["size"]
            new_size = TIERS[next_tier]["size"]

            self.tier = next_tier
            print(f"\n{'='*60}")
            print(f"[PROMOTED] {TIER_ORDER[current_idx]} -> {next_tier}!")
            print(f"  Trades: {total_trades} | WR: {wr:.1%} | PnL: ${self.stats['pnl']:+.2f}")
            print(f"  Size: ${old_size:.2f} -> ${new_size:.2f}/trade")
            print(f"{'='*60}\n")
        else:
            # Log why not promoted
            reasons = []
            if wr < tier_cfg["promote_wr"]:
                reasons.append(f"WR {wr:.1%} < {tier_cfg['promote_wr']:.0%}")
            if not profitable:
                reasons.append(f"PnL ${self.stats['pnl']:+.2f} not profitable")
            if total_trades == promote_after:  # Only log once at threshold
                print(f"[TIER] {self.tier} review @ {total_trades}T: NOT promoted — {', '.join(reasons)}")

    def check_auto_kill(self):
        """V1.2: Auto-shutdown if WR < 50% after 20+ trades."""
        total = self.stats["wins"] + self.stats["losses"]
        if total < AUTO_KILL_TRADES:
            return False
        wr = self.stats["wins"] / total
        if wr < AUTO_KILL_MIN_WR:
            print(f"\n{'='*60}")
            print(f"[AUTO-KILL] {total}T | {self.stats['wins']}W/{self.stats['losses']}L | {wr:.1%} WR < {AUTO_KILL_MIN_WR:.0%}")
            print(f"  PnL: ${self.stats['pnl']:+.2f} | Strategy not viable, shutting down.")
            print(f"{'='*60}\n")
            self.save_results()
            self.running = False
            return True
        return False

    # ========================================================================
    # RESOLVE
    # ========================================================================

    async def resolve_trades(self):
        """Resolve expired 15m trades."""
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

            # V1.8: Verify on-chain position before resolving (skip phantoms)
            if not self.paper and not trade.get("_fill_confirmed", False):
                try:
                    proxy = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
                    async with httpx.AsyncClient(timeout=8) as cl:
                        pos_r = await cl.get(
                            "https://data-api.polymarket.com/positions",
                            params={"user": proxy, "sizeThreshold": 0},
                            headers={"User-Agent": "Mozilla/5.0"},
                        )
                        if pos_r.status_code == 200:
                            positions = pos_r.json()
                            held = any(
                                p.get("conditionId") == trade.get("condition_id")
                                for p in positions
                            )
                            if not held:
                                trade["status"] = "cancelled"
                                trade["pnl"] = 0.0
                                trade["_phantom"] = True
                                del self.trades[tid]
                                print(f"[PHANTOM] {trade.get('_asset', '?')}:{trade['side']} — no on-chain position, order never filled | {trade.get('title', '')[:40]}")
                                continue
                            else:
                                trade["_fill_confirmed"] = True
                except Exception as e:
                    print(f"[VERIFY] Position check error: {e}")

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
                                    shares = trade.get("_shares", trade["size_usd"] / trade["entry_price"])
                                    trade["pnl"] = round(shares - trade["size_usd"], 2)
                                    trade["exit_price"] = 1.0
                                elif price <= 0.05:
                                    # LOSS: shares worth $0
                                    trade["pnl"] = round(-trade["size_usd"], 2)
                                    trade["exit_price"] = 0.0
                                else:
                                    continue

                                trade["exit_time"] = now.isoformat()
                                trade["status"] = "closed"

                                won = trade["pnl"] > 0
                                self.stats["wins" if won else "losses"] += 1
                                self.stats["pnl"] += trade["pnl"]
                                self.session_pnl += trade["pnl"]
                                trade["tier"] = self.tier  # record which tier this trade was at
                                self.resolved.append(trade)
                                del self.trades[tid]

                                w, l = self.stats["wins"], self.stats["losses"]
                                wr = w / (w + l) * 100 if (w + l) > 0 else 0
                                tag = "WIN" if won else "LOSS"
                                print(f"[{tag}] {trade['_asset']}:{trade['side']} ${trade['pnl']:+.2f} | "
                                      f"entry=${trade['entry_price']:.2f} exit=${trade['exit_price']:.2f} | "
                                      f"strat={trade.get('strategy', '?')} | "
                                      f"{w}W/{l}L {wr:.0f}%WR ${self.stats['pnl']:+.2f} | "
                                      f"session=${self.session_pnl:+.2f} | "
                                      f"[{self.tier}] | {trade.get('title', '')[:40]}")

                                # V1.5: Record market outcome for streak tracking
                                asset = trade.get("_asset", trade.get("asset", ""))
                                # Market outcome: if we won, outcome = our side; if lost, outcome = opposite
                                market_outcome = trade["side"] if won else ("DOWN" if trade["side"] == "UP" else "UP")
                                self.record_market_outcome(asset, market_outcome)
                                streak_len, streak_dir = self.get_streak(asset)
                                if streak_len >= 3:
                                    print(f"[STREAK] {asset}: {streak_len}x {streak_dir} consecutive")

                                # V1.4: Update ML filter with outcome
                                ml_rsi = trade.get("rsi") or 50
                                ml_shadow = self.filter_ml.record_outcome(
                                    won=won,
                                    abs_m10=abs(trade.get("momentum_10m", 0)),
                                    abs_m5=abs(trade.get("momentum_5m", 0)),
                                    rsi=ml_rsi,
                                    side=trade["side"],
                                    preset_used=trade.get("ml_preset", "moderate"),
                                )
                                if ml_shadow:
                                    print(f"[ML] Shadow: {ml_shadow}")

                                # V1.7: Track parameter change impact
                                if trade.get("strategy") == "momentum_strong":
                                    self.v17_tracker.record_trade(
                                        won=won,
                                        time_left=trade.get("_time_left", 5.0),
                                        rsi=trade.get("rsi") or 50,
                                        ema_gap_bp=trade.get("_ema_gap_bp"),
                                        side=trade["side"],
                                    )

                                # Check promotion or auto-kill after each resolved trade
                                self.check_promotion()
                                if self.check_auto_kill():
                                    return
                except Exception as e:
                    print(f"[RESOLVE] API error: {e}")
            elif age_min > 25:
                # Very old, no numeric ID — mark as loss
                trade["status"] = "closed"
                trade["pnl"] = -trade["size_usd"]
                self.stats["losses"] += 1
                self.stats["pnl"] += trade["pnl"]
                self.session_pnl += trade["pnl"]
                self.resolved.append(trade)
                del self.trades[tid]
                print(f"[LOSS] {trade.get('_asset', '?')}:{trade['side']} ${trade['pnl']:+.2f} | aged out (no market ID)")
                if self.check_auto_kill():
                    return

    # ========================================================================
    # ENTRY
    # ========================================================================

    async def find_entries(self, all_candles: dict, markets: list):
        """Find momentum-based entries on 15M markets across all assets."""
        now = datetime.now(timezone.utc)

        # V1.9b: Only skip proven dead hours (2AM ET + 8AM ET)
        # Reduced from 6 skipped hours — was too aggressive on 20 data points
        SKIP_HOURS = {7, 13}
        if now.hour in SKIP_HOURS:
            return

        # V2.1: HMM regime filter — skip entries in CHOPPY regime
        if not self.hmm.is_favorable("momentum"):
            regime = self.hmm.current_regime()
            conf = self.hmm.get_regime_confidence()
            # Only block if confidence is high enough (>60%)
            if conf > 0.60:
                return

        open_count = sum(1 for t in self.trades.values() if t.get("status") == "open")
        if open_count >= MAX_CONCURRENT:
            return

        # Daily loss limit check
        if self.session_pnl <= -DAILY_LOSS_LIMIT:
            print(f"[STOP] Daily loss limit hit: ${self.session_pnl:+.2f} (limit: -${DAILY_LOSS_LIMIT})")
            return

        for market in markets:
            if open_count >= MAX_CONCURRENT:
                break

            asset = market.get("_asset", "BTC")
            candles = all_candles.get(asset, [])
            if len(candles) < 3:
                continue

            time_left = market.get("_time_left", 99)
            if time_left < TIME_WINDOW[0] or time_left > TIME_WINDOW[1]:
                continue

            # Calculate momentum from 1-min candles
            recent_closes = [c["close"] for c in candles[-3:]]
            if recent_closes[-1] == 0 or recent_closes[0] == 0:
                continue

            momentum_10m = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            momentum_5m = (recent_closes[-1] - recent_closes[-2]) / recent_closes[-2]
            asset_price = recent_closes[-1]

            # EMA chop filter
            all_closes = [c["close"] for c in candles]
            ema_gap_bp = None
            if len(all_closes) >= 21:
                ema9 = sum(all_closes[:9]) / 9
                m9 = 2 / (9 + 1)
                for p in all_closes[9:]:
                    ema9 = (p - ema9) * m9 + ema9
                ema21 = sum(all_closes[:21]) / 21
                m21 = 2 / (21 + 1)
                for p in all_closes[21:]:
                    ema21 = (p - ema21) * m21 + ema21
                ema_gap_bp = abs(ema9 - ema21) / asset_price * 10000.0
                if EMA_GAP_MIN_BP is not None and ema_gap_bp < EMA_GAP_MIN_BP:
                    continue

            # V1.3: RSI(14) confirmation
            rsi_val = None
            if len(all_closes) >= 15:
                deltas = [all_closes[i] - all_closes[i-1] for i in range(1, len(all_closes))]
                gains = [d if d > 0 else 0 for d in deltas]
                losses_r = [-d if d < 0 else 0 for d in deltas]
                avg_gain = sum(gains[:14]) / 14
                avg_loss = sum(losses_r[:14]) / 14
                for i in range(14, len(deltas)):
                    avg_gain = (avg_gain * 13 + gains[i]) / 14
                    avg_loss = (avg_loss * 13 + losses_r[i]) / 14
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi_val = 100 - (100 / (1 + rs))
                else:
                    rsi_val = 100.0

            up_price, down_price = self.get_prices(market)
            if up_price is None or down_price is None:
                continue

            cid = market.get("conditionId", "")
            question = market.get("question", "")
            nid = market.get("id")

            # Skip if already trading this market
            if f"15m_{cid}_UP" in self.trades or f"15m_{cid}_DOWN" in self.trades:
                continue

            # === V1.9 MOMENTUM ENTRY LOGIC ===
            # On-chain verified: UP 7W/1L (88%), DOWN 6W/6L (50%)
            # Strategy: UP-primary + contrarian (RSI 25-35 bearish → bet UP) + moderate DOWN (RSI 35-50)
            side = None
            entry_price = None
            strategy = "momentum_strong"
            ml_filters = self.filter_ml.get_active_filters()
            ml_preset = ml_filters["preset"]
            ml_m10 = ml_filters["m10"]
            ml_m5 = ml_filters["m5"]

            # V2.0: SPIKE DETECTION — Binance moves faster than Polymarket odds
            spike = self.feed.get_spike()
            rsi_override = False
            spike_boosted = False
            s15 = spike["delta_15s"]

            if abs(s15) > 0.003:
                # MEGA SPIKE: >0.3% in 15s — override RSI, spike IS the signal
                rsi_override = True
                ml_m10 *= 0.3
                ml_m5 *= 0.3
                spike_boosted = True
            elif abs(s15) > 0.0015:
                # SPIKE BOOST: >0.15% in 15s — halve momentum thresholds
                ml_m10 *= 0.5
                ml_m5 *= 0.5
                spike_boosted = True

            if momentum_10m > ml_m10 and momentum_5m > ml_m5:
                # NORMAL UP: Bullish momentum + RSI confirms → bet UP (on-chain: 88% WR)
                if not rsi_override and rsi_val is not None and rsi_val < RSI_CONFIRM_UP:
                    continue  # RSI too weak for UP
                if MIN_ENTRY <= up_price <= MAX_ENTRY:
                    side = "UP"
                    entry_price = round(up_price + (SPREAD_OFFSET if self.paper else 0), 2)
            elif momentum_10m < -ml_m10 and momentum_5m < -ml_m5:
                # Bearish momentum — only take moderate DOWN (RSI 35-50)
                # V1.9: Contrarian (RSI 25-35 → bet UP) DISABLED — backtest 7.95% WR, -$370
                # V1.9: RSI <35 DOWN DISABLED — on-chain 0W/3L
                if not rsi_override and rsi_val is not None and not (RSI_MODERATE_DOWN_LO < rsi_val <= RSI_MODERATE_DOWN_HI):
                    continue
                # MODERATE DOWN (or mega spike override): bearish momentum → bet DOWN
                # Backtest: 91.40% WR, +$385 over 93 trades
                if MIN_ENTRY <= down_price <= MAX_ENTRY:
                    side = "DOWN"
                    entry_price = round(down_price + (SPREAD_OFFSET if self.paper else 0), 2)

            if not side or not entry_price or entry_price > MAX_ENTRY:
                continue

            # V1.5: Streak confluence — skip if betting WITH a long streak (likely to reverse)
            streak_info = self.streak_confluence(asset, side)
            streak_boost = streak_info["boost"]
            if streak_boost < 0.7:
                print(f"[STREAK-SKIP] {asset}:{side} | {streak_info['streak_len']}x {streak_info['streak_dir']} "
                      f"streak | boost={streak_boost:.1f} (betting WITH, reversal likely)")
                continue

            # Size based on current promotion tier
            trade_size = TIERS[self.tier]["size"]
            shares = int(trade_size / entry_price)  # V1.9b: integer shares — CLOB rejects excess decimals

            # Enforce CLOB minimum shares
            if shares < MIN_SHARES:
                shares = MIN_SHARES
            trade_size = round(shares * entry_price, 2)

            # === PLACE TRADE ===
            trade_key = f"15m_{cid}_{side}"
            order_id = None

            fill_confirmed = False
            if not self.paper and self.client:
                # V2.2: GTC limit order + fill polling (FOK was failing on thin books)
                try:
                    from py_clob_client.clob_types import OrderArgs, OrderType
                    from py_clob_client.order_builder.constants import BUY
                    import time as _t

                    up_tid, down_tid = self.get_token_ids(market)
                    token_id = up_tid if side == "UP" else down_tid
                    if not token_id:
                        print(f"[SKIP] No token ID for {asset}:{side}")
                        continue

                    # GTC limit at ask + slippage to be top of book
                    gtc_price = min(round(entry_price + FOK_SLIPPAGE, 2), 0.99)

                    order_args = OrderArgs(
                        price=gtc_price,
                        size=shares,
                        side=BUY,
                        token_id=token_id,
                    )
                    spike_tag = f" SPIKE({s15:+.3%})" if spike_boosted else ""
                    print(f"[LIVE] GTC {asset} {side} {shares}sh @ ${gtc_price:.2f} "
                          f"(ask=${entry_price:.2f}){spike_tag}")
                    signed = self.client.create_order(order_args)
                    resp = self.client.post_order(signed, OrderType.GTC)

                    if not resp.get("success"):
                        print(f"[LIVE] GTC order failed: {resp.get('errorMsg', '?')}")
                        continue

                    order_id = resp.get("orderID", "?")
                    status = resp.get("status", "")
                    print(f"[LIVE] GTC posted: {order_id[:20]}... status={status}")

                    # If immediately matched, confirm fill
                    if status == "matched":
                        _t.sleep(2)
                        fill_confirmed, entry_price, shares, trade_size = self._verify_fill(
                            order_id, entry_price, shares, trade_size)
                        if not fill_confirmed:
                            # Retry once more after 3s — API may be slow
                            _t.sleep(3)
                            fill_confirmed, entry_price, shares, trade_size = self._verify_fill(
                                order_id, entry_price, shares, trade_size)
                        if not fill_confirmed:
                            # Matched but can't verify — trust fill at GTC limit price
                            fill_confirmed = True
                            entry_price = gtc_price  # use actual CLOB price, not Gamma API
                            trade_size = round(shares * entry_price, 2)
                            print(f"[LIVE] Matched — trusting fill {shares}sh @ ${entry_price:.2f} (gtc limit)")
                    else:
                        # Status is "live" — poll for fill up to 30 seconds
                        print(f"[LIVE] Polling for fill (30s max)...")
                        for wait_round in range(6):
                            _t.sleep(5)
                            fill_confirmed, entry_price, shares, trade_size = self._verify_fill(
                                order_id, entry_price, shares, trade_size)
                            if fill_confirmed:
                                break

                        if not fill_confirmed:
                            # Cancel unfilled GTC order
                            try:
                                self.client.cancel(order_id)
                                print(f"[LIVE] Unfilled after 30s — cancelled")
                            except Exception:
                                pass
                            continue

                except Exception as e:
                    print(f"[LIVE] Order error: {e}")
                    continue

            self.trades[trade_key] = {
                "side": side,
                "entry_price": entry_price,
                "size_usd": trade_size,
                "_shares": shares,
                "entry_time": now.isoformat(),
                "condition_id": cid,
                "market_numeric_id": nid,
                "title": question,
                "strategy": strategy,
                "asset": asset,
                "_asset": asset,
                "momentum_10m": round(momentum_10m, 6),
                "momentum_5m": round(momentum_5m, 6),
                "rsi": round(rsi_val, 1) if rsi_val is not None else None,
                "ml_preset": ml_preset,
                "asset_price": asset_price,
                "order_id": order_id,
                "_fill_confirmed": fill_confirmed or self.paper,
                "spike_15s": round(s15, 6) if spike_boosted else 0,
                "spike_boosted": spike_boosted,
                "streak_len": streak_info["streak_len"],
                "streak_dir": streak_info["streak_dir"],
                "streak_alignment": streak_info["alignment"],
                "streak_boost": streak_boost,
                "_time_left": round(time_left, 1),  # V1.7: for change tracker
                "_ema_gap_bp": round(ema_gap_bp, 1) if ema_gap_bp is not None else None,
                "status": "open",
                "pnl": 0.0,
            }
            open_count += 1

            mode = "LIVE" if not self.paper else "PAPER"
            rsi_str = f"RSI={rsi_val:.0f}" if rsi_val is not None else "RSI=?"
            streak_str = ""
            if streak_info["streak_len"] >= 2:
                streak_str = f" | streak={streak_info['streak_len']}x{streak_info['streak_dir']} {streak_info['alignment']} b={streak_boost:.1f}"
            print(f"[ENTRY] {asset}:{side} ${entry_price:.2f} ${trade_size:.2f} ({shares:.0f}sh) | momentum_strong | "
                  f"mom_10m={momentum_10m:+.4%} mom_5m={momentum_5m:+.4%} {rsi_str} | "
                  f"[ML:{ml_preset}] | {asset}=${asset_price:,.2f} | time={time_left:.1f}m | [{mode}]{streak_str} | "
                  f"{question[:50]}")

        # ================================================================
        # V1.6: STREAK REVERSAL ENTRIES — DISABLED V1.7
        # Live data: 46% WR (13 trades, 6W/7L, -$4.20) — worse than coin flip
        # Backtested 55% WR did not hold in live trading
        # ================================================================
        if False and open_count < MAX_CONCURRENT and self.session_pnl > -DAILY_LOSS_LIMIT:
            for market in markets:
                if open_count >= MAX_CONCURRENT:
                    break

                asset = market.get("_asset", "BTC")
                streak_len, streak_dir = self.get_streak(asset)
                if streak_len < STREAK_REVERSAL_MIN_STREAK or not streak_dir:
                    continue

                time_left = market.get("_time_left", 99)
                if time_left < TIME_WINDOW[0] or time_left > TIME_WINDOW[1]:
                    continue

                cid = market.get("conditionId", "")
                question = market.get("question", "")
                nid = market.get("id")

                # Skip if already trading this market (from momentum OR streak)
                if f"15m_{cid}_UP" in self.trades or f"15m_{cid}_DOWN" in self.trades:
                    continue

                up_price, down_price = self.get_prices(market)
                if up_price is None or down_price is None:
                    continue

                reversal_side = "DOWN" if streak_dir == "UP" else "UP"
                rev_price = up_price if reversal_side == "UP" else down_price
                if rev_price < MIN_ENTRY or rev_price > MAX_ENTRY:
                    continue

                entry_price = round(rev_price + (SPREAD_OFFSET if self.paper else 0), 2)
                if entry_price > MAX_ENTRY:
                    continue

                trade_size = STREAK_REVERSAL_SIZE
                shares = round(trade_size / entry_price, 2)
                if shares < MIN_SHARES:
                    shares = MIN_SHARES
                    trade_size = round(shares * entry_price, 2)

                trade_key = f"15m_{cid}_{reversal_side}"
                order_id = None

                if not self.paper and self.client:
                    try:
                        from py_clob_client.clob_types import OrderArgs, OrderType
                        from py_clob_client.order_builder.constants import BUY

                        up_tid, down_tid = self.get_token_ids(market)
                        token_id = up_tid if reversal_side == "UP" else down_tid
                        if not token_id:
                            continue

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
                            print(f"[LIVE] STREAK {asset} {reversal_side} order {order_id[:20]}... @ ${entry_price:.2f} x {shares:.0f}sh")
                        else:
                            print(f"[LIVE] Streak order failed: {resp.get('errorMsg', '?')}")
                            continue
                    except Exception as e:
                        print(f"[LIVE] Streak order error: {e}")
                        continue

                self.trades[trade_key] = {
                    "side": reversal_side,
                    "entry_price": entry_price,
                    "size_usd": trade_size,
                    "entry_time": now.isoformat(),
                    "condition_id": cid,
                    "market_numeric_id": nid,
                    "title": question,
                    "strategy": "streak_reversal",
                    "asset": asset,
                    "_asset": asset,
                    "streak_len": streak_len,
                    "streak_dir": streak_dir,
                    "streak_alignment": "REVERSAL",
                    "streak_boost": 1.0,
                    "status": "open",
                    "pnl": 0.0,
                }
                open_count += 1

                mode = "LIVE" if not self.paper else "PAPER"
                print(f"[ENTRY] {asset}:{reversal_side} ${entry_price:.2f} ${trade_size:.2f} ({shares:.0f}sh) | "
                      f"streak_reversal | {streak_len}x{streak_dir} -> bet {reversal_side} | "
                      f"time={time_left:.1f}m | [{mode}] | {question[:50]}")

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    async def run(self):
        mode = "PAPER" if self.paper else "LIVE"
        print("=" * 70)
        tier_size = TIERS[self.tier]["size"]
        print(f"MOMENTUM 15M DIRECTIONAL TRADER V2.0 - {mode} MODE")
        print(f"Strategy: UP-primary + contrarian reversal + moderate DOWN | BTC only")
        print(f"On-chain verified: BTC UP 7W/1L (88%), DOWN 6W/6L (50%)")
        print(f"Assets: {', '.join(ASSETS.keys())}")
        print(f"Tier: {self.tier} (${tier_size:.2f}/trade) | Max concurrent: {MAX_CONCURRENT}")
        print(f"Entry price: ${MIN_ENTRY:.2f}-${MAX_ENTRY:.2f}")
        print(f"Entry window: {TIME_WINDOW[0]}-{TIME_WINDOW[1]} min before close")
        print(f"V1.9 entry rules (backtest: 93% WR, +$1,110 over 258 trades):")
        print(f"  UP signal:     RSI >{RSI_CONFIRM_UP} + bullish momentum -> bet UP (backtest: 94% WR)")
        print(f"  MODERATE DOWN: RSI {RSI_MODERATE_DOWN_LO}-{RSI_MODERATE_DOWN_HI} + bearish momentum -> bet DOWN (backtest: 91% WR)")
        print(f"  CONTRARIAN:    DISABLED (backtest: 8% WR, -$370 — mean reversion FAILED)")
        print(f"FOK orders + fill verify (V1.8) | ML filter reset (clean priors)")
        print(f"Skip hours (UTC): {{7,11,12,13,14,15}} (2AM + 6-10AM ET — on-chain losers)")
        print(f"Daily loss limit: ${DAILY_LOSS_LIMIT}")
        print(f"Scan interval: {SCAN_INTERVAL}s")
        print(f"V2.0: Binance WebSocket live feed + spike detection")
        print("=" * 70)

        # V2.0: Start WebSocket feed
        await self.feed.start()

        # V2.1: Start HMM regime filter (seeds from REST for full 120 bars)
        await self.hmm.start()
        print(self.hmm.get_report())

        if self.resolved:
            w, l = self.stats["wins"], self.stats["losses"]
            wr = w / (w + l) * 100 if (w + l) > 0 else 0
            print(f"[RESUME] {w}W/{l}L {wr:.0f}%WR | PnL: ${self.stats['pnl']:+.2f} | "
                  f"{len(self.resolved)} resolved, {len(self.trades)} active")

        cycle = 0
        last_redeem = 0
        while self.running:
            try:
                cycle += 1
                now_ts = time.time()

                # V2.0: Get candles from WebSocket feed (real-time, no REST delay)
                if self.feed.is_stale():
                    # WS disconnected — fallback to REST
                    candle_tasks = {
                        asset: self.fetch_binance_candles(cfg["symbol"])
                        for asset, cfg in ASSETS.items()
                    }
                    results = await asyncio.gather(*candle_tasks.values())
                    all_candles = dict(zip(candle_tasks.keys(), results))
                else:
                    # WS live — use feed candles (BTC only for now)
                    all_candles = {"BTC": self.feed.get_candles()}

                # V2.1: Update HMM with latest price
                if self.feed.current_price:
                    self.hmm.update(self.feed.current_price)

                # Fetch markets
                markets = await self.discover_15m_markets()

                if not any(all_candles.values()):
                    print(f"[WARN] No candles from Binance")
                    await asyncio.sleep(10)
                    continue

                # Resolve expired trades
                await self.resolve_trades()

                # Find new entries
                await self.find_entries(all_candles, markets)

                # Status every 5 cycles
                if cycle % 5 == 0:
                    open_count = len(self.trades)
                    w, l = self.stats["wins"], self.stats["losses"]
                    total = w + l
                    wr = w / total * 100 if total > 0 else 0

                    # Show momentum for each asset
                    mom_str = ""
                    for asset, candles in all_candles.items():
                        if len(candles) >= 3:
                            closes = [c["close"] for c in candles[-3:]]
                            mom = (closes[-1] - closes[0]) / closes[0]
                            mom_str += f" {asset}={mom:+.3%}"

                    # V2.0: Spike info
                    spike = self.feed.get_spike()
                    spike_str = ""
                    if not self.feed.is_stale():
                        s15 = spike["delta_15s"]
                        if abs(s15) > 0.0005:
                            spike_str = f" | spike15s={s15:+.3%}"
                        ws_tag = "WS" if self.feed._connected else "REST"
                    else:
                        ws_tag = "REST"

                    tier_size = TIERS[self.tier]["size"]
                    print(f"\n--- Cycle {cycle} | {mode}({ws_tag}) | {self.tier} ${tier_size:.2f} | "
                          f"Active: {open_count} | "
                          f"{w}W/{l}L {wr:.0f}%WR | "
                          f"PnL: ${self.stats['pnl']:+.2f} | "
                          f"Session: ${self.session_pnl:+.2f} |{mom_str}{spike_str} ---")

                    # Market count by asset
                    in_window = {}
                    for m in markets:
                        a = m.get("_asset", "?")
                        tl = m.get("_time_left", 99)
                        if TIME_WINDOW[0] <= tl <= TIME_WINDOW[1]:
                            in_window[a] = in_window.get(a, 0) + 1
                    if in_window:
                        mkt_str = " ".join(f"{a}={n}" for a, n in sorted(in_window.items()))
                        print(f"    Markets in window: {mkt_str}")

                # V1.4: ML filter report every 25 cycles
                if cycle % 25 == 0 and cycle > 0:
                    print(self.filter_ml.get_report())
                    print(self.v17_tracker.get_report())
                    print(self.hmm.get_report())

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

    lock = acquire_pid_lock("momentum_15m")
    try:
        trader = Momentum15MTrader(paper=not args.live)
        asyncio.run(trader.run())
    finally:
        release_pid_lock("momentum_15m")
