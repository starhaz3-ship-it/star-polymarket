"""
Configuration for the arbitrage bot.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Bot configuration."""

    # Polymarket settings
    CLOB_API_URL: str = "https://clob.polymarket.com"
    GAMMA_API_URL: str = "https://gamma-api.polymarket.com"
    DATA_API_URL: str = "https://data-api.polymarket.com"
    CHAIN_ID: int = 137  # Polygon

    # Wallet (loaded from .env)
    PRIVATE_KEY: str = ""
    PROXY_ADDRESS: str = ""
    SIGNATURE_TYPE: int = 1  # Magic wallet

    # Binance settings
    BINANCE_WS_URL: str = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    BINANCE_REST_URL: str = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

    # Trading parameters
    MIN_EDGE_PERCENT: float = 3.5  # Minimum edge after fees (%)
    MIN_NEGRISK_EDGE: float = 2.0  # Lower threshold for NegRisk (more reliable)
    MAX_POSITION_SIZE: float = 100.0  # Max USD per trade
    MIN_POSITION_SIZE: float = 5.0  # Min USD per trade (Polymarket min ~$1)
    MAX_DAILY_LOSS: float = 50.0  # Stop trading if daily loss exceeds this
    MAX_OPEN_POSITIONS: int = 3  # Max concurrent positions

    # NegRisk settings
    NEGRISK_MIN_OUTCOMES: int = 3  # Min outcomes for multi-outcome arb
    NEGRISK_MAX_OUTCOMES: int = 20  # Max outcomes to consider
    ENDGAME_PROB_THRESHOLD: float = 0.95  # Min probability for endgame strategy
    ENDGAME_TIME_HOURS: int = 48  # Max hours to resolution for endgame

    # Timing
    POLL_INTERVAL_MS: int = 500  # How often to check for opportunities
    ORDER_TIMEOUT_SEC: int = 10  # Cancel order if not filled

    # Fees (Polymarket 2026 fee structure)
    TAKER_FEE_MAX: float = 0.03  # 3% max near 50%
    TAKER_FEE_MIN: float = 0.0  # 0% near 0% or 100%

    def __post_init__(self):
        """Load secrets from environment."""
        self.PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        self.PROXY_ADDRESS = os.getenv("POLYMARKET_PROXY_ADDRESS", "")

    @staticmethod
    def estimate_fee(probability: float) -> float:
        """
        Estimate taker fee based on probability.
        Fees are highest near 50%, lowest near 0% or 100%.
        """
        # Distance from 50%
        distance_from_mid = abs(probability - 0.5)
        # Fee scales inversely with distance from 50%
        # At 50%: fee = 3%, at 0% or 100%: fee = 0%
        fee = Config.TAKER_FEE_MAX * (1 - distance_from_mid * 2)
        return max(fee, Config.TAKER_FEE_MIN)


# Global config instance
config = Config()
