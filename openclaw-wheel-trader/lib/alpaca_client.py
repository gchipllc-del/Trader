"""
Alpaca API Client — thin wrapper with rate limiting and error handling.
All broker communication goes through this module.
"""

import os
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone

import yaml

from lib.audit import log_event

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")


class RateLimiter:
    """Sliding window rate limiter."""

    def __init__(self, max_calls: int = 120, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window = window_seconds
        self.calls: deque[float] = deque()
        self.backoff_seconds = 2
        self.max_backoff = 60

    def wait_if_needed(self):
        now = time.time()
        # Remove calls outside the window
        while self.calls and self.calls[0] < now - self.window:
            self.calls.popleft()

        if len(self.calls) >= self.max_calls:
            sleep_time = min(self.backoff_seconds, self.max_backoff)
            log_event("rate_limit", "throttled", {
                "calls_in_window": len(self.calls),
                "sleep_seconds": sleep_time,
            })
            time.sleep(sleep_time)
            self.backoff_seconds = min(self.backoff_seconds * 2, self.max_backoff)
        else:
            self.backoff_seconds = 2  # Reset backoff

        self.calls.append(time.time())


class AlpacaClient:
    """
    Wrapper around Alpaca API.
    
    In paper mode, uses paper-api.alpaca.markets.
    Validates credentials on init. Rate-limits all calls.
    """

    def __init__(self):
        self.api_key = os.environ.get("ALPACA_API_KEY")
        self.secret_key = os.environ.get("ALPACA_SECRET_KEY")
        self.base_url = os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )

        if not self.api_key or not self.secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
            )

        # Verify we're in paper mode
        with open(CONFIG_PATH, "r") as f:
            settings = yaml.safe_load(f)

        if settings.get("mode") == "paper":
            if "paper" not in self.base_url:
                raise EnvironmentError(
                    "Settings say paper mode but ALPACA_BASE_URL is not paper endpoint"
                )

        # Load rate limit settings
        rate_config = settings.get("rate_limits", {})
        self.limiter = RateLimiter(
            max_calls=rate_config.get("alpaca_calls_per_minute", 120)
        )

        self._api = None
        self._trading_client = None

        log_event("startup", "alpaca_client_init", {
            "base_url": self.base_url,
            "mode": settings.get("mode"),
        }, result="success")

    def _get_trading_client(self):
        """Lazy-load the Alpaca trading client."""
        if self._trading_client is None:
            try:
                from alpaca.trading.client import TradingClient
                self._trading_client = TradingClient(
                    self.api_key, self.secret_key, paper=("paper" in self.base_url)
                )
            except ImportError:
                raise ImportError(
                    "Install alpaca-py: pip install alpaca-py"
                )
        return self._trading_client

    def get_account(self) -> dict:
        """Get account info (cash, portfolio value, buying power)."""
        self.limiter.wait_if_needed()
        client = self._get_trading_client()
        account = client.get_account()
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "status": account.status,
        }

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        self.limiter.wait_if_needed()
        client = self._get_trading_client()
        positions = client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_pl": float(p.unrealized_pl),
                "market_value": float(p.market_value),
                "side": p.side,
            }
            for p in positions
        ]

    def get_open_orders(self) -> list[dict]:
        """Get all open/pending orders."""
        self.limiter.wait_if_needed()
        client = self._get_trading_client()
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = client.get_orders(filter=request)
        return [
            {
                "id": str(o.id),
                "symbol": o.symbol,
                "side": str(o.side),
                "qty": str(o.qty),
                "type": str(o.type),
                "status": str(o.status),
                "limit_price": str(o.limit_price) if o.limit_price else None,
                "submitted_at": str(o.submitted_at),
            }
            for o in orders
        ]

    def _build_option_symbol(self, ticker: str, expiration: str, option_type: str, strike: float) -> str:
        """Build OCC option symbol: AAPL240621P00170000"""
        from datetime import datetime as dt
        exp_date = dt.strptime(expiration, "%Y-%m-%d")
        date_str = exp_date.strftime("%y%m%d")
        side_char = "P" if option_type == "put" else "C"
        strike_str = f"{int(strike * 1000):08d}"
        return f"{ticker}{date_str}{side_char}{strike_str}"

    def submit_order(self, intent) -> dict:
        """
        Submit an order to Alpaca.
        This should ONLY be called from order_gate.step3_execute.

        Handles both equity and option orders. For options, builds the
        OCC symbol and uses the appropriate order class.
        """
        self.limiter.wait_if_needed()
        client = self._get_trading_client()

        from alpaca.trading.requests import (
            MarketOrderRequest,
            LimitOrderRequest,
        )
        from alpaca.trading.enums import OrderSide, TimeInForce

        side_map = {
            "buy": OrderSide.BUY,
            "sell": OrderSide.SELL,
            "sell_to_open": OrderSide.SELL,
            "buy_to_close": OrderSide.BUY,
        }

        side = side_map.get(intent.side, OrderSide.BUY)

        if intent.asset_type == "option":
            symbol = self._build_option_symbol(
                intent.ticker, intent.expiration, intent.option_type, intent.strike,
            )
        else:
            symbol = intent.ticker

        if intent.order_type == "market":
            request = MarketOrderRequest(
                symbol=symbol,
                qty=intent.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        else:
            request = LimitOrderRequest(
                symbol=symbol,
                qty=intent.quantity,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=intent.limit_price,
            )

        order = client.submit_order(request)

        return {
            "id": str(order.id),
            "status": str(order.status),
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": str(order.side),
            "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price else None,
        }

    def cancel_all_orders(self) -> int:
        """KILL SWITCH: Cancel all open orders. Returns count cancelled."""
        self.limiter.wait_if_needed()
        client = self._get_trading_client()
        cancelled = client.cancel_orders()
        count = len(cancelled) if cancelled else 0
        log_event("kill_switch", "all_orders_cancelled", {
            "count": count,
        }, result="success")
        return count

    def close_all_positions(self) -> int:
        """KILL SWITCH: Close all open positions at market. Returns count closed."""
        self.limiter.wait_if_needed()
        client = self._get_trading_client()
        closed = client.close_all_positions(cancel_orders=True)
        count = len(closed) if closed else 0
        log_event("kill_switch", "all_positions_closed", {
            "count": count,
        }, result="success")
        return count
