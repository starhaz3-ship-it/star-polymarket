"""
Trade executor for Polymarket.
"""

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from dotenv import load_dotenv

from .config import config
from .detector import ArbitrageSignal, SignalType

load_dotenv()


class OrderStatus(Enum):
    """Status of an order."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ExecutionResult:
    """Result of a trade execution."""
    success: bool
    order_id: Optional[str]
    status: OrderStatus
    filled_size: float
    filled_price: float
    fees_paid: float
    error_message: Optional[str]
    timestamp: float

    @property
    def net_cost(self) -> float:
        """Total cost including fees."""
        return self.filled_size * self.filled_price + self.fees_paid


class Executor:
    """Executes trades on Polymarket."""

    def __init__(self):
        self.client = None
        self._initialized = False
        self._init_client()

    def _decrypt_key(self) -> Optional[str]:
        """Decrypt the private key from environment."""
        key = os.getenv("POLYMARKET_PRIVATE_KEY", "")

        if key.startswith("ENC:"):
            # Encrypted key - need password
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from crypto_utils import decrypt_key

                salt = os.getenv("POLYMARKET_KEY_SALT", "")
                password = os.getenv("POLYMARKET_PASSWORD", "")

                if not password:
                    # Check if running interactively - need both isatty and fileno
                    try:
                        import sys as _sys
                        import os as _os
                        is_interactive = _sys.stdin.isatty() and _sys.stdin.fileno() >= 0
                        # Also check if we're in a proper terminal
                        if is_interactive and _os.isatty(_sys.stdin.fileno()):
                            import getpass
                            password = getpass.getpass("Enter wallet password: ")
                        else:
                            print("[Executor] No password and not interactive - skipping", flush=True)
                            return None
                    except (OSError, ValueError):
                        print("[Executor] No stdin available - skipping", flush=True)
                        return None

                return decrypt_key(key[4:], salt, password)
            except Exception as e:
                print(f"[Executor] Failed to decrypt key: {e}")
                return None

        return key if key else None

    def _init_client(self):
        """Initialize the CLOB client."""
        try:
            from py_clob_client.client import ClobClient

            private_key = self._decrypt_key()
            if not private_key:
                print("[Executor] No private key available")
                return

            proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")

            self.client = ClobClient(
                host=config.CLOB_API_URL,
                key=private_key,
                chain_id=config.CHAIN_ID,
                signature_type=config.SIGNATURE_TYPE,
                funder=proxy_address if proxy_address else None,
            )

            # Set up API credentials
            creds = self.client.derive_api_key()
            self.client.set_api_creds(creds)

            self._initialized = True
            print("[Executor] CLOB client initialized")

        except ImportError:
            print("[Executor] py_clob_client not installed")
        except Exception as e:
            print(f"[Executor] Init error: {e}")

    def execute_signal(self, signal: ArbitrageSignal) -> ExecutionResult:
        """Execute a trading signal."""
        if not self._initialized:
            return ExecutionResult(
                success=False,
                order_id=None,
                status=OrderStatus.FAILED,
                filled_size=0,
                filled_price=0,
                fees_paid=0,
                error_message="Client not initialized",
                timestamp=time.time()
            )

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            if signal.signal_type == SignalType.BUY_BOTH:
                # Same-market arbitrage: buy both YES and NO
                return self._execute_both_sides(signal)

            elif signal.signal_type == SignalType.BUY_YES:
                return self._execute_single(
                    signal,
                    token_id=signal.market.yes_token_id,
                    price=signal.market.yes_ask,
                    side="YES"
                )

            elif signal.signal_type == SignalType.BUY_NO:
                return self._execute_single(
                    signal,
                    token_id=signal.market.no_token_id,
                    price=signal.market.no_ask,
                    side="NO"
                )

        except Exception as e:
            return ExecutionResult(
                success=False,
                order_id=None,
                status=OrderStatus.FAILED,
                filled_size=0,
                filled_price=0,
                fees_paid=0,
                error_message=str(e),
                timestamp=time.time()
            )

    def _execute_single(
        self,
        signal: ArbitrageSignal,
        token_id: str,
        price: float,
        side: str
    ) -> ExecutionResult:
        """Execute a single-side order."""
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY

        # Calculate size in shares
        size = signal.recommended_size / price

        order_args = OrderArgs(
            price=price,
            size=size,
            side=BUY,
            token_id=token_id,
        )

        print(f"[Executor] Placing {side} order: {size:.2f} shares @ ${price:.4f}")

        try:
            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order, OrderType.GTC)

            success = response.get("success", False)
            order_id = response.get("orderID", "")
            status_str = response.get("status", "")

            if success and status_str == "matched":
                status = OrderStatus.FILLED
                filled_size = float(response.get("takingAmount", 0))
                filled_price = price
            elif success:
                status = OrderStatus.PENDING
                filled_size = 0
                filled_price = 0
            else:
                status = OrderStatus.FAILED
                filled_size = 0
                filled_price = 0

            return ExecutionResult(
                success=success,
                order_id=order_id,
                status=status,
                filled_size=filled_size,
                filled_price=filled_price,
                fees_paid=filled_size * filled_price * config.estimate_fee(price),
                error_message=response.get("errorMsg"),
                timestamp=time.time()
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                order_id=None,
                status=OrderStatus.FAILED,
                filled_size=0,
                filled_price=0,
                fees_paid=0,
                error_message=str(e),
                timestamp=time.time()
            )

    def _execute_both_sides(self, signal: ArbitrageSignal) -> ExecutionResult:
        """Execute same-market arbitrage (buy both YES and NO)."""
        # Execute YES side
        yes_result = self._execute_single(
            signal,
            token_id=signal.market.yes_token_id,
            price=signal.market.yes_ask,
            side="YES"
        )

        if not yes_result.success:
            return yes_result

        # Execute NO side
        no_result = self._execute_single(
            signal,
            token_id=signal.market.no_token_id,
            price=signal.market.no_ask,
            side="NO"
        )

        # Combine results
        return ExecutionResult(
            success=yes_result.success and no_result.success,
            order_id=f"{yes_result.order_id},{no_result.order_id}",
            status=OrderStatus.FILLED if (yes_result.status == OrderStatus.FILLED and
                                          no_result.status == OrderStatus.FILLED) else OrderStatus.PARTIAL,
            filled_size=yes_result.filled_size + no_result.filled_size,
            filled_price=(yes_result.filled_price + no_result.filled_price) / 2,
            fees_paid=yes_result.fees_paid + no_result.fees_paid,
            error_message=yes_result.error_message or no_result.error_message,
            timestamp=time.time()
        )

    def get_open_orders(self) -> list:
        """Get all open orders."""
        if not self._initialized:
            return []

        try:
            return self.client.get_orders()
        except Exception as e:
            print(f"[Executor] Error getting orders: {e}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self._initialized:
            return False

        try:
            self.client.cancel(order_id)
            return True
        except Exception as e:
            print(f"[Executor] Cancel error: {e}")
            return False


if __name__ == "__main__":
    # Test executor initialization
    executor = Executor()
    print(f"Initialized: {executor._initialized}")
