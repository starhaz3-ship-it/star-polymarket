"""
Paper trader for EMA crossover + RSI strategy on BTC/USDT 5m futures.
Fetches real candles from Binance public API. No API keys needed.

Strategy:
- LONG: EMA(9) > EMA(21) AND RSI(14) > 55
- SHORT: EMA(9) < EMA(21) AND RSI(14) < 45
- TP: entry +/- ATR(14) * 2.5
- SL: entry +/- ATR(14) * 1.2
- Risk: 1% of balance per trade, 3x leverage
"""

import time
import json
import httpx
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# === CONFIG (matches user's script exactly) ===
SYMBOL = "BTCUSDT"
TIMEFRAME_MS = 5 * 60 * 1000  # 5 minutes
LIMIT = 500
RISK_PER_TRADE = 0.01
LEVERAGE = 3
FAST_EMA = 9
SLOW_EMA = 21
RSI_LEN = 14
ATR_LEN = 14
TP_MULT = 2.5
SL_MULT = 1.2
STARTING_BALANCE = 10000.0  # Paper balance
CHECK_INTERVAL = 60  # Check every 60 seconds (candles are 5m)
REPORT_INTERVAL = 1800  # Report every 30 minutes

OUTPUT_FILE = Path(__file__).parent / "ema_rsi_paper_results.json"
LOG_FILE = Path(__file__).parent / "ema_rsi_paper.log"


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# === INDICATORS (exact match to user's code) ===
def ema(series, length):
    """Exponential moving average (pandas-equivalent ewm)."""
    result = np.zeros_like(series, dtype=float)
    alpha = 2.0 / (length + 1)
    result[0] = series[0]
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def rsi(closes, length):
    """RSI using simple rolling mean (matches pandas rolling().mean())."""
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Simple rolling mean (like pandas .rolling(length).mean())
    rsi_vals = np.full(len(closes), np.nan)
    for i in range(length, len(deltas)):
        avg_gain = np.mean(gains[i - length + 1:i + 1])
        avg_loss = np.mean(losses[i - length + 1:i + 1])
        if avg_loss == 0:
            rsi_vals[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_vals[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return rsi_vals


def atr(highs, lows, closes, length):
    """Average True Range."""
    tr = np.zeros(len(closes))
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    atr_vals = np.full(len(closes), np.nan)
    for i in range(length, len(closes)):
        atr_vals[i] = np.mean(tr[i - length + 1:i + 1])
    return atr_vals


# === DATA ===
def fetch_candles():
    """Fetch 5m BTC/USDT candles from Binance public API with retry."""
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=5m&limit={LIMIT}"
    for attempt in range(3):
        try:
            with httpx.Client(timeout=15) as client:
                r = client.get(url)
                r.raise_for_status()
                data = r.json()
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise

    candles = []
    for bar in data:
        candles.append({
            "time": int(bar[0]),
            "open": float(bar[1]),
            "high": float(bar[2]),
            "low": float(bar[3]),
            "close": float(bar[4]),
            "volume": float(bar[5]),
        })
    return candles


def apply_indicators(candles):
    """Apply all indicators to candle data."""
    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])

    ema_fast = ema(closes, FAST_EMA)
    ema_slow = ema(closes, SLOW_EMA)
    rsi_vals = rsi(closes, RSI_LEN)
    atr_vals = atr(highs, lows, closes, ATR_LEN)

    for i, c in enumerate(candles):
        c["ema_fast"] = ema_fast[i]
        c["ema_slow"] = ema_slow[i]
        c["rsi"] = rsi_vals[i]
        c["atr"] = atr_vals[i]

    return candles


def get_signal(candles):
    """Generate signal from latest candle."""
    last = candles[-1]
    ef = last["ema_fast"]
    es = last["ema_slow"]
    r = last["rsi"]

    if np.isnan(r) or np.isnan(last["atr"]):
        return None

    if ef > es and r > 55:
        return "long"
    elif ef < es and r < 45:
        return "short"
    return None


# === PAPER TRADER ===
class PaperTrader:
    def __init__(self):
        self.balance = STARTING_BALANCE
        self.position = None  # {"side", "entry", "size", "sl", "tp", "time"}
        self.trades = []  # Closed trades
        self.last_report = time.time()
        self.last_candle_time = 0
        self._load()

    def _load(self):
        if OUTPUT_FILE.exists():
            try:
                data = json.loads(OUTPUT_FILE.read_text())
                self.balance = data.get("balance", STARTING_BALANCE)
                self.position = data.get("position", None)
                self.trades = data.get("trades", [])
                self.last_candle_time = data.get("last_candle_time", 0)
                log(f"Loaded state: ${self.balance:.2f} balance, {len(self.trades)} trades, pos={'OPEN' if self.position else 'NONE'}")
            except Exception as e:
                log(f"Load error: {e}")

    def _save(self):
        data = {
            "balance": self.balance,
            "position": self.position,
            "trades": self.trades,
            "last_candle_time": self.last_candle_time,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        OUTPUT_FILE.write_text(json.dumps(data, indent=2))

    def check_exit(self, candle):
        """Check if current position hit TP or SL."""
        if not self.position:
            return

        high = candle["high"]
        low = candle["low"]
        pos = self.position

        if pos["side"] == "long":
            # Check SL first (worse case)
            if low <= pos["sl"]:
                pnl = (pos["sl"] - pos["entry"]) * pos["size"] * LEVERAGE
                self._close_position(pnl, pos["sl"], "SL", candle)
            elif high >= pos["tp"]:
                pnl = (pos["tp"] - pos["entry"]) * pos["size"] * LEVERAGE
                self._close_position(pnl, pos["tp"], "TP", candle)
        else:  # short
            if high >= pos["sl"]:
                pnl = (pos["entry"] - pos["sl"]) * pos["size"] * LEVERAGE
                self._close_position(pnl, pos["sl"], "SL", candle)
            elif low <= pos["tp"]:
                pnl = (pos["entry"] - pos["tp"]) * pos["size"] * LEVERAGE
                self._close_position(pnl, pos["tp"], "TP", candle)

    def _close_position(self, pnl, exit_price, reason, candle):
        pos = self.position
        self.balance += pnl
        trade = {
            "side": pos["side"],
            "entry": pos["entry"],
            "exit": exit_price,
            "size": pos["size"],
            "pnl": round(pnl, 2),
            "reason": reason,
            "entry_time": pos["time"],
            "exit_time": datetime.fromtimestamp(candle["time"] / 1000, tz=timezone.utc).isoformat(),
            "atr": pos.get("atr", 0),
            "rsi": pos.get("rsi", 0),
        }
        self.trades.append(trade)
        tag = "WIN" if pnl > 0 else "LOSS"
        log(f"[{tag}] {pos['side'].upper()} {reason} | entry=${pos['entry']:,.2f} exit=${exit_price:,.2f} | "
            f"PnL: ${pnl:+,.2f} | bal=${self.balance:,.2f}")
        self.position = None

    def try_entry(self, sig, candle):
        """Try to open a new position."""
        if self.position:
            return
        if not sig:
            return

        price = candle["close"]
        atr_val = candle["atr"]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        # Position sizing: risk 1% of balance, stop = ATR * SL_MULT
        risk_amount = self.balance * RISK_PER_TRADE
        stop_distance = atr_val * SL_MULT
        size = risk_amount / stop_distance
        size = round(size, 6)

        if size <= 0:
            return

        if sig == "long":
            sl = price - atr_val * SL_MULT
            tp = price + atr_val * TP_MULT
        else:
            sl = price + atr_val * SL_MULT
            tp = price - atr_val * TP_MULT

        self.position = {
            "side": sig,
            "entry": price,
            "size": size,
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "time": datetime.fromtimestamp(candle["time"] / 1000, tz=timezone.utc).isoformat(),
            "atr": round(atr_val, 2),
            "rsi": round(candle["rsi"], 1),
        }
        log(f"[ENTRY] {sig.upper()} @ ${price:,.2f} | size={size:.6f} | "
            f"SL=${sl:,.2f} TP=${tp:,.2f} | ATR=${atr_val:.2f} RSI={candle['rsi']:.1f}")

    def report(self, candle, force=False):
        """Print 30-minute performance report."""
        now = time.time()
        if not force and (now - self.last_report) < REPORT_INTERVAL:
            return
        self.last_report = now

        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        total_pnl = sum(t["pnl"] for t in self.trades)
        total = len(self.trades)
        wr = len(wins) / total * 100 if total > 0 else 0

        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        profit_factor = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else float('inf')

        # Recent trades (last 5)
        recent = self.trades[-5:] if self.trades else []

        # Per-side breakdown
        longs = [t for t in self.trades if t["side"] == "long"]
        shorts = [t for t in self.trades if t["side"] == "short"]
        long_pnl = sum(t["pnl"] for t in longs)
        short_pnl = sum(t["pnl"] for t in shorts)
        long_wr = sum(1 for t in longs if t["pnl"] > 0) / len(longs) * 100 if longs else 0
        short_wr = sum(1 for t in shorts if t["pnl"] > 0) / len(shorts) * 100 if shorts else 0

        # TP vs SL
        tp_trades = [t for t in self.trades if t["reason"] == "TP"]
        sl_trades = [t for t in self.trades if t["reason"] == "SL"]

        # Unrealized PnL
        unreal = 0
        pos_str = "FLAT"
        if self.position:
            p = self.position
            current = candle["close"]
            if p["side"] == "long":
                unreal = (current - p["entry"]) * p["size"] * LEVERAGE
            else:
                unreal = (p["entry"] - current) * p["size"] * LEVERAGE
            pos_str = f"{p['side'].upper()} @ ${p['entry']:,.2f} (unreal: ${unreal:+,.2f})"

        sep = "=" * 65
        log(sep)
        log(f"  EMA/RSI PAPER REPORT | BTC/USDT 5m | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        log(sep)
        log(f"  Balance: ${self.balance:,.2f} (started ${STARTING_BALANCE:,.2f}) | Net: ${self.balance - STARTING_BALANCE:+,.2f}")
        log(f"  Total PnL: ${total_pnl:+,.2f} | Trades: {total} | WR: {wr:.1f}%")
        log(f"  Avg Win: ${avg_win:+,.2f} | Avg Loss: ${avg_loss:+,.2f} | PF: {profit_factor:.2f}")
        log(f"  LONG:  {len(longs)}T {long_wr:.0f}%WR ${long_pnl:+,.2f}")
        log(f"  SHORT: {len(shorts)}T {short_wr:.0f}%WR ${short_pnl:+,.2f}")
        log(f"  TP hits: {len(tp_trades)} | SL hits: {len(sl_trades)}")
        log(f"  Position: {pos_str}")
        log(f"  BTC: ${candle['close']:,.2f} | EMA9={candle['ema_fast']:,.2f} EMA21={candle['ema_slow']:,.2f} | RSI={candle['rsi']:.1f}")
        if recent:
            log(f"  Recent trades:")
            for t in recent:
                tag = "W" if t["pnl"] > 0 else "L"
                log(f"    [{tag}] {t['side'].upper():5s} {t['reason']} ${t['pnl']:+,.2f} | "
                    f"${t['entry']:,.2f}->${t['exit']:,.2f} | RSI={t.get('rsi',0):.0f}")
        log(sep)

    def run(self):
        log("=" * 65)
        log("  EMA(9/21) + RSI(14) Paper Trader v1.0")
        log(f"  BTC/USDT 5m | TP={TP_MULT}x ATR | SL={SL_MULT}x ATR | {LEVERAGE}x leverage")
        log(f"  Risk: {RISK_PER_TRADE*100}% per trade | Balance: ${self.balance:,.2f}")
        log("=" * 65)

        while True:
            try:
                candles = fetch_candles()
                candles = apply_indicators(candles)

                latest = candles[-1]
                latest_time = latest["time"]

                # Only process new candles
                if latest_time > self.last_candle_time:
                    self.last_candle_time = latest_time

                    # Check exits on ALL candles since last check (catch TP/SL on intermediate bars)
                    if self.position:
                        # Check the last few candles for TP/SL hits
                        for c in candles[-5:]:
                            if c["time"] > self.last_candle_time - 5 * TIMEFRAME_MS:
                                self.check_exit(c)
                                if not self.position:
                                    break

                    # Generate signal
                    sig = get_signal(candles)

                    # Close on signal reversal (don't hold against the trend)
                    if self.position and sig and sig != self.position["side"]:
                        pos = self.position
                        price = latest["close"]
                        if pos["side"] == "long":
                            pnl = (price - pos["entry"]) * pos["size"] * LEVERAGE
                        else:
                            pnl = (pos["entry"] - price) * pos["size"] * LEVERAGE
                        self._close_position(pnl, price, "REVERSAL", latest)

                    # Try entry
                    self.try_entry(sig, latest)

                    # Status line
                    sig_str = sig.upper() if sig else "NONE"
                    pos_str = self.position["side"].upper() if self.position else "FLAT"
                    log(f"[TICK] ${latest['close']:,.2f} | EMA9={latest['ema_fast']:,.2f} EMA21={latest['ema_slow']:,.2f} | "
                        f"RSI={latest['rsi']:.1f} | sig={sig_str} pos={pos_str} | {len(self.trades)}T ${sum(t['pnl'] for t in self.trades):+,.2f}")

                    self._save()

                # 30-minute report
                self.report(candles[-1])

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                log("Stopped by user")
                self.report(candles[-1], force=True)
                self._save()
                break
            except Exception as e:
                log(f"ERROR: {type(e).__name__}: {e}")
                time.sleep(60)


if __name__ == "__main__":
    trader = PaperTrader()
    trader.run()
