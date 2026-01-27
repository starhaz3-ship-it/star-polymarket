"""
Star Polymarket - Polymarket API Client

A Python client for interacting with the Polymarket API.
"""

import os
import json
import getpass
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _decrypt_private_key() -> str | None:
    """Decrypt the private key from environment if encrypted."""
    key = os.getenv("POLYMARKET_PRIVATE_KEY")
    if not key:
        return None

    if key.startswith("ENC:"):
        from crypto_utils import decrypt_key
        salt = os.getenv("POLYMARKET_KEY_SALT")
        if not salt:
            raise ValueError("Encrypted key found but no salt")
        password = getpass.getpass("Enter wallet password: ")
        try:
            return decrypt_key(key[4:], salt, password)
        except Exception:
            raise ValueError("Invalid password")

    return key

# Polymarket API Base URLs
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
DATA_API_URL = "https://data-api.polymarket.com"

# Chain ID for Polygon
POLYGON_CHAIN_ID = 137


class StarPolymarket:
    """Client for interacting with Polymarket APIs."""

    def __init__(self, private_key: str = None, wallet_address: str = None, unlock: bool = False):
        """Initialize the client.

        Args:
            private_key: Wallet private key (passed directly, never stored/displayed)
            wallet_address: Wallet address (or from POLYMARKET_WALLET_ADDRESS env var)
            unlock: If True, prompt for password to decrypt stored key
        """
        self.client = httpx.Client(timeout=30.0)
        self.wallet_address = wallet_address or os.getenv("POLYMARKET_WALLET_ADDRESS")
        self.clob_client = None
        self._private_key = None  # Never expose this

        if private_key:
            self._private_key = private_key
        elif unlock:
            self._private_key = _decrypt_private_key()

        if self._private_key:
            self._init_clob_client()

    def _init_clob_client(self):
        """Initialize the CLOB client for authenticated operations."""
        try:
            from py_clob_client.client import ClobClient

            # Get proxy address for Magic wallet setup
            proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS")

            if proxy_address:
                # Magic wallet: use signature_type=1 and funder
                self.clob_client = ClobClient(
                    host=CLOB_API_URL,
                    key=self._private_key,
                    chain_id=POLYGON_CHAIN_ID,
                    signature_type=1,
                    funder=proxy_address,
                )
            else:
                # Standard EOA wallet
                self.clob_client = ClobClient(
                    host=CLOB_API_URL,
                    key=self._private_key,
                    chain_id=POLYGON_CHAIN_ID,
                )

            # Set up API credentials
            creds = self.clob_client.derive_api_key()
            self.clob_client.set_api_creds(creds)

        except ImportError:
            print("Warning: py_clob_client not installed. Trading features unavailable.")
        except Exception as e:
            print(f"Warning: Could not initialize CLOB client: {e}")

    @property
    def is_authenticated(self) -> bool:
        """Check if the client has authentication credentials."""
        return self._private_key is not None

    def get_positions(self, wallet_address: str = None) -> list[dict]:
        """Get current positions for a wallet.

        Args:
            wallet_address: Wallet address (defaults to configured wallet)

        Returns:
            List of position dictionaries
        """
        address = wallet_address or self.wallet_address
        if not address:
            raise ValueError("Wallet address required")

        response = self.client.get(f"{DATA_API_URL}/positions", params={"user": address})
        response.raise_for_status()
        return response.json()

    def get_trades(self, wallet_address: str = None, limit: int = 50) -> list[dict]:
        """Get trade history for a wallet.

        Args:
            wallet_address: Wallet address (defaults to configured wallet)
            limit: Maximum number of trades to return

        Returns:
            List of trade dictionaries
        """
        address = wallet_address or self.wallet_address
        if not address:
            raise ValueError("Wallet address required")

        response = self.client.get(
            f"{DATA_API_URL}/trades",
            params={"user": address, "limit": limit}
        )
        response.raise_for_status()
        return response.json()

    def get_activity(self, wallet_address: str = None) -> list[dict]:
        """Get account activity.

        Args:
            wallet_address: Wallet address (defaults to configured wallet)

        Returns:
            List of activity records
        """
        address = wallet_address or self.wallet_address
        if not address:
            raise ValueError("Wallet address required")

        response = self.client.get(f"{DATA_API_URL}/activity", params={"user": address})
        response.raise_for_status()
        return response.json()

    def get_markets(self, limit: int = 10, active: bool = True, order: str = "volume24hr", ascending: bool = False) -> list[dict]:
        """Fetch a list of markets from Polymarket.

        Args:
            limit: Maximum number of markets to return
            active: If True, only return active markets
            order: Sort field (volume24hr, liquidity, startDate, endDate)
            ascending: Sort direction

        Returns:
            List of market dictionaries
        """
        params = {
            "limit": limit,
            "active": str(active).lower(),
            "closed": "false",
            "order": order,
            "ascending": str(ascending).lower(),
        }
        response = self.client.get(f"{GAMMA_API_URL}/markets", params=params)
        response.raise_for_status()
        return response.json()

    def get_market(self, market_id: str) -> dict:
        """Fetch a specific market by ID.

        Args:
            market_id: The market's condition ID or slug

        Returns:
            Market details dictionary
        """
        response = self.client.get(f"{GAMMA_API_URL}/markets/{market_id}")
        response.raise_for_status()
        return response.json()

    def get_events(self, limit: int = 10) -> list[dict]:
        """Fetch a list of events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        params = {"limit": limit}
        response = self.client.get(f"{GAMMA_API_URL}/events", params=params)
        response.raise_for_status()
        return response.json()

    def get_event(self, event_id: str) -> dict:
        """Fetch a specific event by ID or slug.

        Args:
            event_id: The event's ID or slug

        Returns:
            Event details dictionary
        """
        response = self.client.get(f"{GAMMA_API_URL}/events/{event_id}")
        response.raise_for_status()
        return response.json()

    def search(self, query: str) -> dict:
        """Search for markets, events, or profiles.

        Args:
            query: Search query string

        Returns:
            Search results dictionary
        """
        params = {"query": query}
        response = self.client.get(f"{GAMMA_API_URL}/search", params=params)
        response.raise_for_status()
        return response.json()

    def get_prices(self, token_ids: list[str]) -> dict:
        """Get current prices for market tokens.

        Args:
            token_ids: List of token IDs to get prices for

        Returns:
            Dictionary mapping token IDs to prices
        """
        params = {"token_ids": ",".join(token_ids)}
        response = self.client.get(f"{CLOB_API_URL}/prices", params=params)
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """Example usage of Star Polymarket client."""
    import sys

    # Check if user wants to unlock wallet
    unlock = "--unlock" in sys.argv or "-u" in sys.argv

    with StarPolymarket(unlock=unlock) as client:
        # Show account info if authenticated
        if client.wallet_address:
            print(f"=== Star Polymarket ===")
            print(f"Wallet: {client.wallet_address}\n")

            # Get positions
            print("Your Positions:")
            print("-" * 40)
            try:
                positions = client.get_positions()
                if positions:
                    for pos in positions[:10]:  # Show first 10
                        title = pos.get("title", pos.get("market", {}).get("question", "Unknown"))
                        size = pos.get("size", 0)
                        outcome = pos.get("outcome", "?")
                        print(f"  {title[:50]}...")
                        print(f"    {outcome}: {size} shares")
                else:
                    print("  No open positions")
            except Exception as e:
                print(f"  Could not fetch positions: {e}")

            print()

        # Show top markets
        print("Top Markets by Volume:")
        print("-" * 40)
        markets = client.get_markets(limit=5)

        for market in markets:
            question = market.get("question", "Unknown")
            outcome_prices = market.get("outcomePrices", [])
            volume_24h = market.get("volume24hr", 0)

            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except:
                    outcome_prices = []

            yes_price = float(outcome_prices[0]) * 100 if len(outcome_prices) > 0 else 0
            no_price = float(outcome_prices[1]) * 100 if len(outcome_prices) > 1 else 0

            print(f"  {question[:60]}")
            print(f"    Yes: {yes_price:.1f}%  |  No: {no_price:.1f}%  |  24h: ${volume_24h:,.0f}")
            print()


if __name__ == "__main__":
    main()
