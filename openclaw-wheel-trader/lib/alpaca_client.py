"""
Alpaca API Client — thin wrapper with rate limiting and error handling.
All broker communication goes through this module.
"""

import os
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone, timedelta

import yaml

from lib.audit import log_event

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")


def _retry_on_network_error(fn, *, attempts: int = 4, base_delay: float = 2.0):
    """Run ``fn()`` with exponential-backoff retry on transient broker
    network errors.

    Targets the post-wake / brief-outage pattern where Alpaca's API
    briefly refuses connections (Errno 61) right after the laptop
    resumes. Without this, a single launchd cron firing during the
    blip exits 1 and the operator sees a spurious failed-run alert.

    Only catches :class:`requests.ConnectionError` and
    :class:`requests.Timeout` — auth, 4xx, and 5xx are surfaced
    immediately so real bugs aren't masked.
    """
    import requests  # local import — alpaca SDK pulls it in transitively
    delay = base_delay
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            if attempt == attempts:
                break
            log_event("alpaca", "network_retry", {
                "attempt": attempt,
                "delay_seconds": delay,
                "error": str(e)[:200],
            }, result="degraded")
            time.sleep(delay)
            delay *= 2
    assert last_exc is not None
    raise last_exc


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

    # ── User-Agent identification ──────────────────────────────────
    # Stamping every Alpaca request with a unique UA makes it easy to
    # filter the bot's traffic in Alpaca's support / abuse logs when
    # debugging rate-limit or order-routing issues. Pattern from
    # alpacahq/options-wheel core/user_agent_mixin.py.
    _USER_AGENT = "OPENCLAW-WHEEL"

    @classmethod
    def _ua_mixin(cls):
        """Mixin that injects the bot's User-Agent into Alpaca SDK requests.

        Built lazily so import-time failures in the alpaca SDK don't
        bubble up to module load. Returns a mixin class; combine with
        any concrete SDK client via multiple inheritance.
        """
        class _UAMixin:
            def _get_default_headers(self) -> dict:
                headers = super()._get_default_headers()
                headers["User-Agent"] = cls._USER_AGENT
                return headers
        return _UAMixin

    def _get_trading_client(self):
        """Lazy-load the Alpaca trading client."""
        if self._trading_client is None:
            try:
                from alpaca.trading.client import TradingClient
                class _Stamped(self._ua_mixin(), TradingClient):
                    pass
                self._trading_client = _Stamped(
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
            class _Stamped(self._ua_mixin(), StockHistoricalDataClient):
                pass
            self._stock_data_client = _Stamped(
                self.api_key, self.secret_key
            )
        return self._stock_data_client

    def _get_option_data_client(self):
        """Lazy-load the Alpaca option historical data client."""
        if self._option_data_client is None:
            from alpaca.data.historical import OptionHistoricalDataClient
            class _Stamped(self._ua_mixin(), OptionHistoricalDataClient):
                pass
            self._option_data_client = _Stamped(
                self.api_key, self.secret_key
            )
        return self._option_data_client

    def _get_crypto_data_client(self):
        """Lazy-load the Alpaca crypto historical data client.

        Crypto market data is FREE on Alpaca — no API key required. Trading
        crypto still goes through the same TradingClient as stocks.
        """
        if not hasattr(self, "_crypto_data_client") or self._crypto_data_client is None:
            from alpaca.data.historical import CryptoHistoricalDataClient
            class _Stamped(self._ua_mixin(), CryptoHistoricalDataClient):
                pass
            self._crypto_data_client = _Stamped()
        return self._crypto_data_client

    def get_crypto_bars(
        self,
        symbols: list[str],
        timeframe: str = "1Day",
        days_back: int = 365,
    ) -> dict[str, "pd.DataFrame"]:
        """Fetch crypto OHLCV bars from Alpaca.

        Args:
            symbols: list of slash-format symbols, e.g. ["BTC/USD", "ETH/USD"]
            timeframe: "1Day", "4Hour", "1Hour"
            days_back: history depth (default 365 for 1Y vol calc)

        Returns:
            {symbol: DataFrame[open, high, low, close, volume]}
        """
        import pandas as pd
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame

        self.limiter.wait_if_needed()
        client = self._get_crypto_data_client()

        tf_map = {
            "1Day": TimeFrame.Day,
            "1Hour": TimeFrame.Hour,
        }
        tf = tf_map.get(timeframe, TimeFrame.Day)

        start = datetime.now(timezone.utc) - timedelta(days=days_back)
        req = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start,
        )

        try:
            bars = client.get_crypto_bars(req)
            df = bars.df  # MultiIndex (symbol, timestamp)
            if df.empty:
                return {}
            result = {}
            for sym in symbols:
                if sym in df.index.get_level_values("symbol"):
                    sub = df.xs(sym, level="symbol").copy()
                    sub.index = pd.to_datetime(sub.index)
                    result[sym] = sub
            return result
        except Exception as e:
            log_event("market_data", "crypto_bars_failed", {
                "symbols": symbols,
                "error": str(e)[:200],
            }, result="failed")
            return {}

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

        start = datetime.now(timezone.utc) - timedelta(days=400)

        # Fetch in batches per ticker via raw API for reliability
        import requests as _requests

        result = {}
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        tf_str = timeframe  # Pass through directly: "1Day" or "1Week"

        for ticker in tickers:
            self.limiter.wait_if_needed()
            url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
            params = {
                "timeframe": tf_str,
                "limit": 1000,  # Max per page
                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "adjustment": "split",
            }
            try:
                resp = _requests.get(url, params=params, headers=headers, timeout=15)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                all_bars = data.get("bars", [])

                # Paginate until we have enough bars or no more pages
                while data.get("next_page_token") and len(all_bars) < limit:
                    self.limiter.wait_if_needed()
                    params["page_token"] = data["next_page_token"]
                    resp = _requests.get(url, params=params, headers=headers, timeout=15)
                    if resp.status_code != 200:
                        break
                    data = resp.json()
                    all_bars.extend(data.get("bars", []))

                # Keep only the last `limit` bars
                if len(all_bars) > limit:
                    all_bars = all_bars[-limit:]

                if not all_bars:
                    continue

                rows = []
                for b in all_bars:
                    rows.append({
                        "open": float(b["o"]),
                        "high": float(b["h"]),
                        "low": float(b["l"]),
                        "close": float(b["c"]),
                        "volume": int(b["v"]),
                        "timestamp": b["t"],
                    })
                df = pd.DataFrame(rows)
                df.index = pd.to_datetime(df["timestamp"])
                df = df.drop(columns=["timestamp"])
                result[ticker] = df

            except Exception as e:
                log_event("data", "bars_fetch_error", {"ticker": ticker, "error": str(e)})
                continue

        log_event("data", "bars_fetched", {
            "tickers": list(result.keys()),
            "timeframe": tf_str,
            "count": len(result),
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

        # Paginate: Alpaca returns a `next_page_token` when the result set
        # is larger than a single page (current cap is around 100 rows).
        # Without this loop, a broad chain scan silently truncates.
        # Pattern mirrored from alpacahq/options-wheel core/broker_client.py.
        contracts: list[dict] = []
        page_token: str | None = None
        while True:
            if page_token:
                request.page_token = page_token
            response = _retry_on_network_error(
                lambda: client.get_option_contracts(request)
            )
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
            page_token = getattr(response, "next_page_token", None)
            if not page_token:
                break

        return contracts

    # Alpaca's option-data endpoints cap a single request at ~100 symbols
    # — beyond that, results are silently truncated. Both get_option_quotes
    # and get_option_snapshots iterate in 100-symbol chunks.
    _OPTION_SYMBOL_BATCH = 100

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

        client = self._get_option_data_client()
        result: dict[str, dict] = {}
        for i in range(0, len(option_symbols), self._OPTION_SYMBOL_BATCH):
            batch = option_symbols[i:i + self._OPTION_SYMBOL_BATCH]
            self.limiter.wait_if_needed()
            request = OptionLatestQuoteRequest(symbol_or_symbols=batch)
            quotes = _retry_on_network_error(
                lambda: client.get_option_latest_quote(request)
            )
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

        client = self._get_option_data_client()
        result: dict[str, dict] = {}
        for i in range(0, len(option_symbols), self._OPTION_SYMBOL_BATCH):
            batch = option_symbols[i:i + self._OPTION_SYMBOL_BATCH]
            self.limiter.wait_if_needed()
            request = OptionSnapshotRequest(symbol_or_symbols=batch)
            snapshots = _retry_on_network_error(
                lambda: client.get_option_snapshot(request)
            )
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
        """Get account info (cash, portfolio value, buying power, PDT count)."""
        self.limiter.wait_if_needed()
        client = self._get_trading_client()
        account = _retry_on_network_error(client.get_account)
        return {
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "status": account.status,
            # FINRA day-trade count over the last 5 business days, per the
            # broker. Source of truth — positions.json can be stale across
            # concurrent processes (Wave 2 #8 fix).
            "daytrade_count": int(getattr(account, "daytrade_count", 0) or 0),
            "pattern_day_trader": bool(
                getattr(account, "pattern_day_trader", False)
            ),
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

        Handles equity, option, and crypto orders. For options, builds the
        OCC symbol. For crypto buys, uses notional (dollars) with TimeInForce.GTC.
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

        if intent.asset_type == "crypto":
            # Crypto: GTC required, supports notional buys, fractional qty sells.
            tif = TimeInForce.GTC
            kwargs = {"symbol": symbol, "side": side, "time_in_force": tif}
            if intent.notional and intent.side == "buy":
                kwargs["notional"] = round(float(intent.notional), 2)
            else:
                kwargs["qty"] = round(float(intent.quantity), 9)
            request = MarketOrderRequest(**kwargs)
        elif intent.order_type == "market":
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
            "qty": str(order.qty) if order.qty else None,
            "notional": str(order.notional) if getattr(order, "notional", None) else None,
            "side": str(order.side),
            "filled_qty": str(order.filled_qty) if order.filled_qty else "0",
            "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price else None,
        }

    def wait_for_fill(self, order_id: str, timeout_seconds: float = 10.0,
                      poll_interval: float = 0.5) -> dict:
        """
        Poll an order until it reaches a terminal state, or until the timeout.

        Wave 2 #9 + #10 fix: prevents recording an open position from the
        post-submit response (which often shows status="pending_new" with
        filled_qty=0) — without polling, positions.json would record the
        intended share count even if the order is later rejected, leaving
        a zombie entry that subsequent scans treat as held.

        Terminal statuses: filled, canceled, expired, rejected, suspended,
        done_for_day. NOTE: `partially_filled` is intentionally NOT terminal —
        it means the order is still working and more fills may arrive in
        milliseconds (especially for multi-share market orders). Returning
        on partially_filled would record fewer shares than we actually own.

        Returns the final order dict (same shape as submit_order); the
        caller decides whether the fill is good enough to record.
        Returns whatever the broker reports at timeout if no terminal
        status is reached — caller should treat as "unknown" and audit.
        """
        import time as _time
        terminal = {"filled", "canceled", "cancelled",
                    "expired", "rejected", "suspended", "done_for_day"}
        deadline = _time.time() + max(0.5, timeout_seconds)

        client = self._get_trading_client()
        last = None
        while _time.time() < deadline:
            self.limiter.wait_if_needed()
            try:
                order = client.get_order_by_id(order_id)
            except Exception as e:
                log_event("alpaca_client", "wait_for_fill_poll_error",
                          {"order_id": order_id, "error": str(e)[:200]},
                          result="degraded")
                _time.sleep(poll_interval)
                continue
            last = {
                "id": str(order.id),
                "status": str(order.status).split(".")[-1].lower(),
                "symbol": str(order.symbol),
                "qty": str(order.qty) if order.qty else None,
                "filled_qty": str(order.filled_qty) if order.filled_qty else "0",
                "filled_avg_price": str(order.filled_avg_price)
                                     if order.filled_avg_price else None,
                "side": str(order.side),
            }
            if last["status"] in terminal:
                return last
            _time.sleep(poll_interval)

        # Timeout — return whatever we last saw, or a synthetic "unknown".
        if last is None:
            return {"id": order_id, "status": "unknown_timeout"}
        last["status"] = last["status"] + "_timeout"
        return last

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

    def liquidate_wheel_book(self) -> dict:
        """Planned wheel reset — close options first, then equity.

        Unlike ``close_all_positions`` (which fires every close in
        parallel), this sequences the unwind to avoid creating an
        uncovered short call mid-process: if Alpaca closes shares
        before the short call attached to them, the call goes naked
        for a brief window. Sequence: options first → equity second.

        Pattern lifted from alpacahq/options-wheel
        core/broker_client.py.liquidate_all_positions.

        Returns ``{"options_closed", "stocks_closed", "errors"}``.
        """
        client = self._get_trading_client()
        self.limiter.wait_if_needed()
        positions = list(client.get_all_positions())

        from alpaca.trading.enums import AssetClass
        options = [p for p in positions if getattr(p, "asset_class", None) == AssetClass.US_OPTION]
        stocks = [p for p in positions if getattr(p, "asset_class", None) != AssetClass.US_OPTION]

        result = {"options_closed": 0, "stocks_closed": 0, "errors": []}

        for p in options:
            try:
                self.limiter.wait_if_needed()
                client.close_position(p.symbol)
                result["options_closed"] += 1
            except Exception as e:
                result["errors"].append(f"option:{p.symbol}: {e}")

        for p in stocks:
            try:
                self.limiter.wait_if_needed()
                client.close_position(p.symbol)
                result["stocks_closed"] += 1
            except Exception as e:
                result["errors"].append(f"stock:{p.symbol}: {e}")

        log_event("wheel_reset", "liquidation_complete", {
            "options_closed": result["options_closed"],
            "stocks_closed": result["stocks_closed"],
            "errors": result["errors"][:3],
        }, result="success" if not result["errors"] else "partial")

        return result
