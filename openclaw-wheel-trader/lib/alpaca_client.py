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
        self._stock_data_client = None
        self._option_data_client = None

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

    def _get_stock_data_client(self):
        """Lazy-load the Alpaca stock historical data client."""
        if self._stock_data_client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            self._stock_data_client = StockHistoricalDataClient(
                self.api_key, self.secret_key
            )
        return self._stock_data_client

    def _get_option_data_client(self):
        """Lazy-load the Alpaca option historical data client."""
        if self._option_data_client is None:
            from alpaca.data.historical import OptionHistoricalDataClient
            self._option_data_client = OptionHistoricalDataClient(
                self.api_key, self.secret_key
            )
        return self._option_data_client

    def get_bars(
        self,
        tickers: list[str],
        timeframe: str = "1Day",
        limit: int = 252,
    ) -> dict[str, "pd.DataFrame"]:
        """
        Fetch historical OHLCV bars for multiple tickers in one call.

        Args:
            tickers: List of stock symbols
            timeframe: "1Day" or "1Week"
            limit: Number of bars per ticker

        Returns:
            {ticker: DataFrame with open, high, low, close, volume}
        """
        import pandas as pd
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        self.limiter.wait_if_needed()
        client = self._get_stock_data_client()

        tf_map = {"1Day": TimeFrame.Day, "1Week": TimeFrame.Week}
        tf = tf_map.get(timeframe, TimeFrame.Day)

        start = datetime.now(timezone.utc) - __import__("datetime").timedelta(days=400)

        request = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=tf,
            start=start,
            limit=limit,
        )

        bars = client.get_stock_bars(request)

        result = {}
        for ticker in tickers:
            ticker_bars = bars[ticker] if ticker in bars else []
            if not ticker_bars:
                continue
            rows = []
            for b in ticker_bars:
                rows.append({
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": int(b.volume),
                    "timestamp": b.timestamp,
                })
            df = pd.DataFrame(rows)
            df.index = pd.to_datetime(df["timestamp"])
            df = df.drop(columns=["timestamp"])
            result[ticker] = df

        log_event("data", "bars_fetched", {
            "tickers": tickers,
            "timeframe": timeframe,
            "bars_per_ticker": {t: len(result.get(t, [])) for t in tickers},
        })

        return result

    def get_option_contracts(
        self,
        ticker: str,
        expiration_gte: str,
        expiration_lte: str,
        option_type: str | None = None,
        strike_price_gte: float | None = None,
        strike_price_lte: float | None = None,
    ) -> list[dict]:
        """
        Fetch available option contracts for a ticker.

        Returns list of contract dicts with symbol, strike, expiration, type, etc.
        """
        self.limiter.wait_if_needed()
        client = self._get_trading_client()

        from alpaca.trading.requests import GetOptionContractsRequest

        params = {
            "underlying_symbols": [ticker],
            "expiration_date_gte": expiration_gte,
            "expiration_date_lte": expiration_lte,
            "status": "active",
        }
        if option_type:
            params["type"] = option_type
        if strike_price_gte is not None:
            params["strike_price_gte"] = str(strike_price_gte)
        if strike_price_lte is not None:
            params["strike_price_lte"] = str(strike_price_lte)

        request = GetOptionContractsRequest(**params)
        response = client.get_option_contracts(request)

        contracts = []
        for c in (response.option_contracts or []):
            contracts.append({
                "symbol": c.symbol,
                "underlying": ticker,
                "strike": float(c.strike_price),
                "expiration": str(c.expiration_date),
                "option_type": str(c.type).split(".")[-1].lower(),
                "open_interest": int(c.open_interest) if c.open_interest else 0,
                "status": str(c.status),
            })

        return contracts

    def get_option_quotes(self, option_symbols: list[str]) -> dict[str, dict]:
        """
        Fetch latest quotes (bid/ask) for option symbols.

        Args:
            option_symbols: List of OCC option symbols

        Returns:
            {symbol: {"bid", "ask", "bid_size", "ask_size"}}
        """
        if not option_symbols:
            return {}

        from alpaca.data.requests import OptionLatestQuoteRequest

        self.limiter.wait_if_needed()
        client = self._get_option_data_client()

        request = OptionLatestQuoteRequest(symbol_or_symbols=option_symbols)
        quotes = client.get_option_latest_quote(request)

        result = {}
        for sym, q in quotes.items():
            result[sym] = {
                "bid": float(q.bid_price) if q.bid_price else 0,
                "ask": float(q.ask_price) if q.ask_price else 0,
                "bid_size": int(q.bid_size) if q.bid_size else 0,
                "ask_size": int(q.ask_size) if q.ask_size else 0,
            }

        return result

    def get_option_snapshots(self, option_symbols: list[str]) -> dict[str, dict]:
        """
        Fetch snapshots (quote + greeks) for option symbols.

        Returns:
            {symbol: {"bid", "ask", "delta", "gamma", "theta", "vega", "implied_volatility"}}
        """
        if not option_symbols:
            return {}

        from alpaca.data.requests import OptionSnapshotRequest

        self.limiter.wait_if_needed()
        client = self._get_option_data_client()

        request = OptionSnapshotRequest(symbol_or_symbols=option_symbols)
        snapshots = client.get_option_snapshot(request)

        result = {}
        for sym, snap in snapshots.items():
            entry = {
                "bid": 0, "ask": 0,
                "delta": 0, "gamma": 0, "theta": 0, "vega": 0,
                "implied_volatility": 0,
            }
            if snap.latest_quote:
                entry["bid"] = float(snap.latest_quote.bid_price or 0)
                entry["ask"] = float(snap.latest_quote.ask_price or 0)
            if snap.greeks:
                entry["delta"] = float(snap.greeks.delta or 0)
                entry["gamma"] = float(snap.greeks.gamma or 0)
                entry["theta"] = float(snap.greeks.theta or 0)
                entry["vega"] = float(snap.greeks.vega or 0)
            if snap.implied_volatility is not None:
                entry["implied_volatility"] = float(snap.implied_volatility)
            result[sym] = entry

        return result

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
