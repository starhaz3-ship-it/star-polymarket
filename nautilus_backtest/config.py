"""Shared configuration for nautilus backtester."""
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId

INSTRUMENT_ID = InstrumentId.from_str("BTCUSDT.BINANCE")
BAR_TYPE_1M = BarType.from_str("BTCUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL")

# Default backtest params
DEFAULT_HORIZON_5M = 5
DEFAULT_HORIZON_15M = 15
DEFAULT_ENTRY_PRICE = 0.50   # Simulated Polymarket mid-price
DEFAULT_SPREAD = 0.03        # Realistic spread offset

# Data
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
DATA_SYMBOL = "BTCUSDT"
DATA_INTERVAL = "1m"
DATA_DAYS = 30
