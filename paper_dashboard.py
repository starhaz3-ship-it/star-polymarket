"""
Paper Trading Dashboard - Real-time monitoring with Rich UI
Based on polymarket-assistant by @SolSt1ne

Shows: BTC price, PM prices, TA signals, ML predictions, trade stats
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import httpx

# Rich imports for beautiful terminal UI
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# Import our ML engine
sys.path.insert(0, str(Path(__file__).parent))
from arbitrage.ta_signals import TASignalGenerator, Candle
from arbitrage.bregman_optimizer import BregmanOptimizer

console = Console(force_terminal=True)

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

BINANCE_REST = "https://api.binance.com/api/v3"
PM_GAMMA = "https://gamma-api.polymarket.com/events"
REFRESH = 5  # seconds between updates
RESULTS_FILE = Path(__file__).parent / "ta_paper_results.json"
TUNER_FILE = Path(__file__).parent / "ml_tuner_state.json"


# ══════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DashState:
    # Binance data
    btc_price: float = 0.0
    klines: List[dict] = None

    # Polymarket data
    pm_up: float = 0.0
    pm_down: float = 0.0
    market_title: str = ""

    # TA signals
    rsi: float = 50.0
    vwap: float = 0.0
    regime: str = "range"
    ha_greens: int = 0
    squeeze_on: bool = False
    squeeze_momentum: float = 0.0
    squeeze_fired: bool = False

    # ML state
    ml_up_pct: float = 50.0
    ml_tuned_params: dict = None

    # Trade stats
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    open_trades: int = 0

    def __post_init__(self):
        if self.klines is None:
            self.klines = []
        if self.ml_tuned_params is None:
            self.ml_tuned_params = {}


# ══════════════════════════════════════════════════════════════════════
# INDICATORS (from polymarket-assistant)
# ══════════════════════════════════════════════════════════════════════

def calc_rsi(klines: List[dict], period: int = 14) -> Optional[float]:
    """RSI calculation."""
    closes = [k["c"] for k in klines]
    if len(closes) < period + 1:
        return None

    changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [max(c, 0) for c in changes[:period]]
    losses = [max(-c, 0) for c in changes[:period]]

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for c in changes[period:]:
        avg_gain = (avg_gain * (period - 1) + max(c, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-c, 0)) / period

    if avg_loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1 + avg_gain / avg_loss)


def calc_vwap(klines: List[dict]) -> float:
    """VWAP calculation."""
    if not klines:
        return 0.0
    tp_v = sum((k["h"] + k["l"] + k["c"]) / 3 * k["v"] for k in klines)
    v = sum(k["v"] for k in klines)
    return tp_v / v if v else 0.0


def calc_ema(values: List[float], period: int) -> Optional[float]:
    """EMA of last value."""
    if len(values) < period:
        return None
    mult = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * mult + ema * (1 - mult)
    return ema


def heikin_ashi_trend(klines: List[dict]) -> Tuple[int, str]:
    """Returns (green_count_last_3, trend_string)."""
    if len(klines) < 3:
        return 0, "—"

    ha = []
    for i, k in enumerate(klines):
        c = (k["o"] + k["h"] + k["l"] + k["c"]) / 4
        o = (k["o"] + k["c"]) / 2 if i == 0 else (ha[i-1]["o"] + ha[i-1]["c"]) / 2
        ha.append({"o": o, "c": c, "green": c >= o})

    last3 = ha[-3:]
    greens = sum(1 for h in last3 if h["green"])

    if greens >= 2:
        return greens, "UP"
    elif greens <= 1:
        return greens, "DOWN"
    return greens, "MIXED"


def calc_ttm_squeeze(klines: List[dict], bb_length: int = 20, kc_length: int = 20) -> Tuple[bool, float, bool]:
    """
    Calculate TTM Squeeze indicator.
    Returns: (squeeze_on, momentum, squeeze_fired)
    """
    if len(klines) < max(bb_length, kc_length) + 5:
        return False, 0.0, False

    closes = [k["c"] for k in klines]
    highs = [k["h"] for k in klines]
    lows = [k["l"] for k in klines]

    # Bollinger Bands: SMA + 2*StdDev
    bb_closes = closes[-bb_length:]
    bb_sma = sum(bb_closes) / bb_length
    variance = sum((x - bb_sma) ** 2 for x in bb_closes) / bb_length
    bb_std = variance ** 0.5
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std

    # Keltner Channels: EMA + 1.5*ATR
    kc_ema = calc_ema(closes[-kc_length:], kc_length)
    if kc_ema is None:
        return False, 0.0, False

    # ATR calculation
    trs = []
    for i in range(len(klines) - kc_length, len(klines)):
        if i == 0:
            tr = highs[i] - lows[i]
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
        trs.append(tr)
    atr = sum(trs) / len(trs)

    kc_upper = kc_ema + 1.5 * atr
    kc_lower = kc_ema - 1.5 * atr

    # Squeeze detection: BB inside KC
    squeeze_on = (bb_lower > kc_lower) and (bb_upper < kc_upper)

    # Momentum: deviation from midline
    highest = max(highs[-bb_length:])
    lowest = min(lows[-bb_length:])
    midline = (highest + lowest) / 2 + bb_sma
    momentum = (closes[-1] - midline / 2) / (bb_std if bb_std > 0 else 1) * 10

    # Check previous squeeze state for "fired" detection
    squeeze_fired = False
    if len(klines) > bb_length + 1:
        prev_closes = closes[-(bb_length + 1):-1]
        prev_sma = sum(prev_closes) / bb_length
        prev_var = sum((x - prev_sma) ** 2 for x in prev_closes) / bb_length
        prev_std = prev_var ** 0.5
        prev_bb_upper = prev_sma + 2 * prev_std
        prev_bb_lower = prev_sma - 2 * prev_std
        prev_kc_ema = calc_ema(closes[-(kc_length + 1):-1], kc_length)
        if prev_kc_ema:
            prev_kc_upper = prev_kc_ema + 1.5 * atr
            prev_kc_lower = prev_kc_ema - 1.5 * atr
            prev_squeeze = (prev_bb_lower > prev_kc_lower) and (prev_bb_upper < prev_kc_upper)
            squeeze_fired = prev_squeeze and not squeeze_on

    return squeeze_on, momentum, squeeze_fired


# ══════════════════════════════════════════════════════════════════════
# DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════

async def fetch_binance_klines(symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 100) -> List[dict]:
    """Fetch klines from Binance."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{BINANCE_REST}/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit}
        )
        data = resp.json()
        return [
            {"t": r[0]/1000, "o": float(r[1]), "h": float(r[2]),
             "l": float(r[3]), "c": float(r[4]), "v": float(r[5])}
            for r in data
        ]


async def fetch_pm_market() -> Tuple[float, float, str]:
    """Fetch current BTC 15m market from Polymarket."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://clob.polymarket.com/markets",
                params={"tag_slug": "15M"}
            )
            markets = resp.json()

            # Find BTC market closest to expiry
            btc_markets = [m for m in markets if "Bitcoin" in m.get("question", "") and "Up or Down" in m.get("question", "")]
            if not btc_markets:
                return 0.0, 0.0, "No market"

            # Get the one expiring soonest
            btc_markets.sort(key=lambda m: m.get("end_date_iso", ""))
            market = btc_markets[0]

            tokens = market.get("tokens", [])
            up_price = 0.0
            down_price = 0.0
            for t in tokens:
                if t.get("outcome") == "Up":
                    up_price = float(t.get("price", 0))
                elif t.get("outcome") == "Down":
                    down_price = float(t.get("price", 0))

            return up_price, down_price, market.get("question", "")[:50]
    except Exception as e:
        return 0.0, 0.0, f"Error: {e}"


def load_paper_results() -> dict:
    """Load paper trading results."""
    if RESULTS_FILE.exists():
        try:
            return json.load(open(RESULTS_FILE))
        except:
            pass
    return {"total_pnl": 0.0, "wins": 0, "losses": 0, "trades": {}}


def load_tuner_state() -> dict:
    """Load ML tuner state."""
    if TUNER_FILE.exists():
        try:
            return json.load(open(TUNER_FILE))
        except:
            pass
    return {}


# ══════════════════════════════════════════════════════════════════════
# DASHBOARD RENDERING
# ══════════════════════════════════════════════════════════════════════

def color_val(val: float) -> str:
    """Return color based on value."""
    if val > 0:
        return "green"
    elif val < 0:
        return "red"
    return "yellow"


def render_header(state: DashState) -> Panel:
    """Render header with price and trend."""
    # Determine trend
    if state.rsi > 60 and state.ha_greens >= 2:
        trend, trend_col = "BULLISH", "green"
    elif state.rsi < 40 and state.ha_greens <= 1:
        trend, trend_col = "BEARISH", "red"
    else:
        trend, trend_col = "NEUTRAL", "yellow"

    parts = [
        ("  BTC ", "bold white on dark_blue"),
        (f" ${state.btc_price:,.2f} ", "bold white"),
    ]

    if state.pm_up > 0:
        parts.append((f"  PM: ↑{state.pm_up:.2f} ↓{state.pm_down:.2f}  ", "cyan"))

    parts.append((f" {trend} ", f"bold white on {trend_col}"))
    parts.append(("\n", ""))
    parts.append(("  Star Polymarket Paper Trader", "dim white"))
    parts.append(("  |  ML V3 + LightGBM", "dim cyan"))

    return Panel(
        Text.assemble(*parts),
        title="PAPER TRADING DASHBOARD",
        box=box.DOUBLE,
        expand=True,
    )


def render_ta_panel(state: DashState) -> Panel:
    """Render technical analysis panel."""
    t = Table(box=None, show_header=False, pad_edge=False, expand=True)
    t.add_column("label", style="dim", width=14)
    t.add_column("value", width=16)
    t.add_column("signal", width=12)

    # RSI
    rsi_col = "red" if state.rsi > 70 else "green" if state.rsi < 30 else "yellow"
    rsi_sig = "OVERBOUGHT" if state.rsi > 70 else "OVERSOLD" if state.rsi < 30 else ""
    t.add_row("RSI(14)", f"[{rsi_col}]{state.rsi:.1f}[/{rsi_col}]", f"[{rsi_col}]{rsi_sig}[/{rsi_col}]")

    # VWAP
    vwap_col = "green" if state.btc_price > state.vwap else "red"
    vwap_sig = "above" if state.btc_price > state.vwap else "below"
    t.add_row("VWAP", f"${state.vwap:,.2f}", f"[{vwap_col}]price {vwap_sig}[/{vwap_col}]")

    # Regime
    regime_col = "green" if state.regime == "trend_up" else "red" if state.regime == "trend_down" else "yellow"
    t.add_row("Regime", f"[{regime_col}]{state.regime}[/{regime_col}]", "")

    # Heikin Ashi
    ha_col = "green" if state.ha_greens >= 2 else "red"
    ha_dots = "▲" * state.ha_greens + "▼" * (3 - state.ha_greens)
    t.add_row("Heikin Ashi", f"[{ha_col}]{ha_dots}[/{ha_col}]", f"[{ha_col}]{state.ha_greens}/3 green[/{ha_col}]")

    # TTM Squeeze (volatility compression)
    if state.squeeze_fired:
        sq_status = "[bold magenta]FIRED![/bold magenta]"
        sq_sig = "BREAKOUT"
    elif state.squeeze_on:
        sq_status = "[yellow]ON[/yellow]"
        sq_sig = "compressing"
    else:
        sq_status = "[dim]OFF[/dim]"
        sq_sig = ""
    mom_col = "green" if state.squeeze_momentum > 0 else "red"
    t.add_row("Squeeze", sq_status, f"[{mom_col}]mom {state.squeeze_momentum:+.1f}[/{mom_col}]")

    return Panel(t, title="TECHNICAL", box=box.ROUNDED, expand=True)


def render_ml_panel(state: DashState) -> Panel:
    """Render ML predictions panel."""
    t = Table(box=None, show_header=False, pad_edge=False, expand=True)
    t.add_column("label", style="dim", width=14)
    t.add_column("value", width=20)

    # ML prediction
    ml_col = "green" if state.ml_up_pct > 55 else "red" if state.ml_up_pct < 45 else "yellow"
    t.add_row("ML Prediction", f"[{ml_col}]{state.ml_up_pct:.1f}% UP[/{ml_col}]")

    # Tuned params
    params = state.ml_tuned_params
    if params:
        t.add_row("UP Filter", f"<${params.get('up_max_price', 0.5):.2f} @ {params.get('up_min_confidence', 0.6)*100:.0f}%")
        t.add_row("DOWN Filter", f"<${params.get('down_max_price', 0.5):.2f} @ {params.get('down_min_confidence', 0.6)*100:.0f}%")
        t.add_row("Min Edge", f">{params.get('min_edge', 0.06)*100:.0f}%")
        t.add_row("Tune Cycle", f"#{params.get('tune_count', 0)}")

    return Panel(t, title="ML ENGINE", box=box.ROUNDED, expand=True)


def render_stats_panel(state: DashState) -> Panel:
    """Render trading stats panel."""
    t = Table(box=None, show_header=False, pad_edge=False, expand=True)
    t.add_column("label", style="dim", width=14)
    t.add_column("value", width=16)

    # PnL
    pnl_col = color_val(state.total_pnl)
    t.add_row("Total PnL", f"[{pnl_col} bold]${state.total_pnl:+,.2f}[/{pnl_col} bold]")

    # Win/Loss
    total = state.wins + state.losses
    wr = state.wins / total * 100 if total > 0 else 0
    wr_col = "green" if wr >= 55 else "red" if wr < 50 else "yellow"
    t.add_row("Win/Loss", f"{state.wins}/{state.losses}")
    t.add_row("Win Rate", f"[{wr_col}]{wr:.1f}%[/{wr_col}]")

    # Open trades
    t.add_row("Open Trades", f"{state.open_trades}")

    return Panel(t, title="TRADING STATS", box=box.ROUNDED, expand=True)


def render_signals_panel(state: DashState) -> Panel:
    """Render active signals panel."""
    signals = []

    # RSI signals
    if state.rsi > 70:
        signals.append("[red]RSI -> OVERBOUGHT (sell signal)[/red]")
    elif state.rsi < 30:
        signals.append("[green]RSI -> OVERSOLD (buy signal)[/green]")

    # VWAP signals
    if state.btc_price > state.vwap * 1.002:
        signals.append("[green]Price above VWAP (bullish)[/green]")
    elif state.btc_price < state.vwap * 0.998:
        signals.append("[red]Price below VWAP (bearish)[/red]")

    # HA signals
    if state.ha_greens >= 3:
        signals.append("[green]HA -> 3 green candles (uptrend)[/green]")
    elif state.ha_greens == 0:
        signals.append("[red]HA -> 3 red candles (downtrend)[/red]")

    # ML signals
    if state.ml_up_pct > 65:
        signals.append(f"[green]ML -> Strong UP ({state.ml_up_pct:.0f}%)[/green]")
    elif state.ml_up_pct < 35:
        signals.append(f"[red]ML -> Strong DOWN ({100-state.ml_up_pct:.0f}%)[/red]")

    # PM arbitrage opportunity
    if state.pm_up > 0 and state.pm_down > 0:
        spread = state.pm_up + state.pm_down
        if spread < 0.98:
            signals.append(f"[green]PM Spread {spread:.3f} (arb opportunity!)[/green]")

    if not signals:
        signals.append("[dim]No strong signals[/dim]")

    # Add current time
    now = datetime.now(timezone.utc)
    signals.append("[dim]─────────────────────────────[/dim]")
    signals.append(f"[dim]Updated: {now.strftime('%H:%M:%S')} UTC[/dim]")

    return Panel("\n".join(signals), title="SIGNALS", box=box.ROUNDED, expand=True)


def render_dashboard(state: DashState) -> Table:
    """Render full dashboard."""
    # Main grid
    grid = Table(box=None, pad_edge=False, show_header=False, expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)

    grid.add_row(render_ta_panel(state), render_ml_panel(state))
    grid.add_row(render_stats_panel(state), render_signals_panel(state))

    # Combine header + grid
    outer = Table(box=None, pad_edge=False, show_header=False, expand=True)
    outer.add_column()
    outer.add_row(render_header(state))
    outer.add_row(grid)

    return outer


# ══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════

async def update_state(state: DashState):
    """Update all state data."""
    # Fetch Binance klines
    try:
        state.klines = await fetch_binance_klines()
        if state.klines:
            state.btc_price = state.klines[-1]["c"]
            state.rsi = calc_rsi(state.klines) or 50.0
            state.vwap = calc_vwap(state.klines)
            state.ha_greens, _ = heikin_ashi_trend(state.klines)

            # TTM Squeeze calculation
            sq_on, sq_mom, sq_fired = calc_ttm_squeeze(state.klines)
            state.squeeze_on = sq_on
            state.squeeze_momentum = sq_mom
            state.squeeze_fired = sq_fired

            # Determine regime
            closes = [k["c"] for k in state.klines]
            ema20 = calc_ema(closes, 20)
            if ema20:
                if state.btc_price > ema20 * 1.001:
                    state.regime = "trend_up"
                elif state.btc_price < ema20 * 0.999:
                    state.regime = "trend_down"
                else:
                    state.regime = "range"
    except Exception as e:
        pass

    # Fetch PM prices
    try:
        state.pm_up, state.pm_down, state.market_title = await fetch_pm_market()
    except:
        pass

    # Load paper results
    results = load_paper_results()
    state.total_pnl = results.get("total_pnl", 0.0)
    state.wins = results.get("wins", 0)
    state.losses = results.get("losses", 0)
    state.open_trades = sum(1 for t in results.get("trades", {}).values() if t.get("status") == "open")

    # Load ML tuner state
    state.ml_tuned_params = load_tuner_state()

    # Calculate ML prediction (simplified)
    if state.rsi > 60:
        state.ml_up_pct = 60 + (state.rsi - 60) * 0.5
    elif state.rsi < 40:
        state.ml_up_pct = 40 - (40 - state.rsi) * 0.5
    else:
        state.ml_up_pct = 50.0


async def main():
    """Main dashboard loop."""
    console.print("\n[bold magenta]═══ PAPER TRADING DASHBOARD ═══[/bold magenta]\n")
    console.print("Starting dashboard...\n")

    state = DashState()

    # Initial fetch
    await update_state(state)

    with Live(console=console, refresh_per_second=1, transient=False) as live:
        while True:
            await update_state(state)
            live.update(render_dashboard(state))
            await asyncio.sleep(REFRESH)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")
