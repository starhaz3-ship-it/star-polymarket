"""Fetch BTC 1-min candles from Binance and convert to nautilus Bar objects."""
import time
import requests
import numpy as np
from datetime import datetime, timezone, timedelta

from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from nautilus_backtest.config import (
    BAR_TYPE_1M, BINANCE_KLINES_URL, DATA_SYMBOL, DATA_INTERVAL, DATA_DAYS,
)


def fetch_binance_klines(days: int = DATA_DAYS) -> list[list]:
    """Fetch BTC/USDT 1-min klines from Binance REST API."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (days * 24 * 60 * 60 * 1000)
    all_klines = []
    current_ms = start_ms

    print(f"[DATA] Fetching {days} days of BTC/USDT 1-min candles from Binance...")
    while current_ms < end_ms:
        params = {
            "symbol": DATA_SYMBOL,
            "interval": DATA_INTERVAL,
            "startTime": current_ms,
            "limit": 1000,
        }
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        klines = resp.json()
        if not klines:
            break
        all_klines.extend(klines)
        current_ms = klines[-1][0] + 60_000  # Next minute
        if len(all_klines) % 5000 == 0:
            print(f"  Fetched {len(all_klines)} candles...")
        time.sleep(0.1)  # Rate limit

    print(f"[DATA] Fetched {len(all_klines)} candles ({days} days)")
    return all_klines


def klines_to_bars(klines: list[list], bar_type: BarType = BAR_TYPE_1M) -> list[Bar]:
    """Convert Binance klines to nautilus Bar objects."""
    bars = []
    instrument = TestInstrumentProvider.btcusdt_binance()

    for k in klines:
        # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
        ts_event_ns = int(k[0]) * 1_000_000  # ms -> ns
        ts_init_ns = int(k[6]) * 1_000_000   # close time in ns

        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(str(k[1])),
            high=Price.from_str(str(k[2])),
            low=Price.from_str(str(k[3])),
            close=Price.from_str(str(k[4])),
            volume=Quantity.from_str(str(k[5])),
            ts_event=ts_event_ns,
            ts_init=ts_init_ns,
        )
        bars.append(bar)

    print(f"[DATA] Converted {len(bars)} bars")
    return bars


def fetch_and_convert(days: int = DATA_DAYS) -> list[Bar]:
    """Full pipeline: fetch from Binance, convert to nautilus Bars."""
    klines = fetch_binance_klines(days)
    return klines_to_bars(klines)


if __name__ == "__main__":
    bars = fetch_and_convert(7)
    print(f"First bar: {bars[0]}")
    print(f"Last bar: {bars[-1]}")
