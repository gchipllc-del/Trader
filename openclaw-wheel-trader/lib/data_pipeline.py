"""
Data Pipeline — fetches all market data needed by the trading engines.

Orchestrates: bars → options chains → quotes/greeks → IV calculation
Produces the exact data structures that csp_engine and cc_engine expect.
"""

import yaml
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from lib.alpaca_client import AlpacaClient
from lib.iv_rank import calculate_historical_volatility, evaluate_premium_environment
from lib.audit import log_event

CONFIG_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"


def _load_strategy_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def fetch_all_data(
    client: AlpacaClient,
    tickers: list[str] | None = None,
) -> dict:
    """
    Fetch all market data needed for CSP/CC scanning.

    Returns:
        {
            "daily_data":     {ticker: pd.DataFrame},
            "weekly_data":    {ticker: pd.DataFrame},
            "options_chains": {ticker: [list of option dicts]},
            "iv_data":        {ticker: iv evaluation dict},
        }
    """
    config = _load_strategy_config()
    if tickers is None:
        tickers = config.get("tickers", [])

    log_event("data_pipeline", "fetch_started", {"tickers": tickers})

    # Step 1: Fetch daily bars (one batch call for all tickers)
    daily_data = client.get_bars(tickers, timeframe="1Day", limit=252)

    # Step 2: Resample daily → weekly (no API call needed)
    weekly_data = {}
    for ticker, df in daily_data.items():
        weekly = df.resample("W-FRI").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        weekly_data[ticker] = weekly

    # Step 3: Fetch options chains for each ticker
    csp_cfg = config.get("csp", {})
    cc_cfg = config.get("cc", {})
    dte_min = min(csp_cfg.get("dte_min", 30), cc_cfg.get("dte_min", 30))
    dte_max = max(csp_cfg.get("dte_max", 45), cc_cfg.get("dte_max", 45))

    today = datetime.now(timezone.utc).date()
    exp_gte = (today + timedelta(days=dte_min)).isoformat()
    exp_lte = (today + timedelta(days=dte_max + 7)).isoformat()

    options_chains = {}
    for ticker in tickers:
        if ticker not in daily_data:
            continue
        current_price = daily_data[ticker]["close"].iloc[-1]

        # Fetch both puts and calls within strike range
        strike_low = current_price * 0.85
        strike_high = current_price * 1.15

        chain = _fetch_enriched_chain(
            client, ticker, exp_gte, exp_lte,
            strike_low, strike_high, today,
        )
        if chain:
            options_chains[ticker] = chain

    # Step 4: Calculate IV data
    iv_data = _calculate_iv_data(daily_data, options_chains, config)

    log_event("data_pipeline", "fetch_complete", {
        "tickers_with_bars": list(daily_data.keys()),
        "tickers_with_options": list(options_chains.keys()),
        "tickers_with_iv": list(iv_data.keys()),
    })

    return {
        "daily_data": daily_data,
        "weekly_data": weekly_data,
        "options_chains": options_chains,
        "iv_data": iv_data,
    }


def _fetch_enriched_chain(
    client: AlpacaClient,
    ticker: str,
    exp_gte: str,
    exp_lte: str,
    strike_low: float,
    strike_high: float,
    today,
) -> list[dict]:
    """Fetch option contracts and enrich with quotes/greeks."""
    contracts = client.get_option_contracts(
        ticker=ticker,
        expiration_gte=exp_gte,
        expiration_lte=exp_lte,
        strike_price_gte=strike_low,
        strike_price_lte=strike_high,
    )

    if not contracts:
        return []

    # Batch fetch snapshots for greeks + quotes
    symbols = [c["symbol"] for c in contracts]

    # Fetch in batches of 100
    snapshots = {}
    for i in range(0, len(symbols), 100):
        batch = symbols[i:i + 100]
        try:
            batch_snaps = client.get_option_snapshots(batch)
            snapshots.update(batch_snaps)
        except Exception as e:
            log_event("data_pipeline", "snapshot_fetch_failed", {
                "ticker": ticker, "batch_size": len(batch), "error": str(e),
            })
            # Fall back to quotes only
            try:
                batch_quotes = client.get_option_quotes(batch)
                for sym, q in batch_quotes.items():
                    snapshots[sym] = {**q, "delta": 0, "gamma": 0,
                                      "theta": 0, "vega": 0, "implied_volatility": 0}
            except Exception:
                pass

    # Merge contracts with snapshots into the format screener expects
    enriched = []
    for contract in contracts:
        sym = contract["symbol"]
        snap = snapshots.get(sym, {})

        bid = snap.get("bid", 0)
        ask = snap.get("ask", 0)

        # Skip illiquid options
        if bid <= 0 and ask <= 0:
            continue

        exp_date = datetime.strptime(contract["expiration"], "%Y-%m-%d").date()
        dte = (exp_date - today).days

        enriched.append({
            "strike": contract["strike"],
            "expiration": contract["expiration"],
            "option_type": contract["option_type"],
            "bid": bid,
            "ask": ask,
            "delta": snap.get("delta", 0),
            "dte": dte,
            "open_interest": contract.get("open_interest", 0),
            "implied_volatility": snap.get("implied_volatility", 0),
            "occ_symbol": sym,
        })

    return enriched


def _calculate_iv_data(
    daily_data: dict[str, pd.DataFrame],
    options_chains: dict[str, list[dict]],
    config: dict,
) -> dict[str, dict]:
    """
    Calculate IV environment for each ticker.

    Strategy: prefer ATM option implied_volatility when available,
    fall back to historical volatility as proxy.
    """
    iv_cfg = config.get("iv", {})
    min_iv_rank = iv_cfg.get("min_iv_rank", 30) / 100  # Convert from percentage

    iv_data = {}
    for ticker, df in daily_data.items():
        if len(df) < 50:
            continue

        # Calculate historical volatility series as baseline
        hv_series = calculate_historical_volatility(df["close"], window=20)
        hv_series = hv_series.dropna()

        if len(hv_series) < 20:
            continue

        # Try to get ATM implied volatility from options chain
        current_price = df["close"].iloc[-1]
        current_iv = float(hv_series.iloc[-1])  # Default to HV

        chain = options_chains.get(ticker, [])
        if chain:
            # Find ATM option (closest strike to current price)
            atm = min(chain, key=lambda o: abs(o["strike"] - current_price))
            if atm.get("implied_volatility", 0) > 0:
                current_iv = atm["implied_volatility"]

        iv_eval = evaluate_premium_environment(
            current_iv=current_iv,
            iv_history=hv_series,
            min_iv_rank=min_iv_rank,
        )
        iv_data[ticker] = iv_eval

    return iv_data


def fetch_option_prices_for_positions(
    client: AlpacaClient,
    positions: list[dict],
) -> dict[str, float]:
    """
    Fetch current mid-prices for open option positions.
    Used by the monitoring loop for early close checks.

    Returns:
        {ticker: mid_price}
    """
    symbols = []
    ticker_to_symbol = {}

    for pos in positions:
        if pos.get("type") not in ("csp", "cc") or pos.get("status") != "open":
            continue

        ticker = pos.get("ticker", "")
        strike = pos.get("strike", 0)
        expiration = pos.get("expiration", "")
        opt_type = "put" if pos.get("type") == "csp" else "call"

        if ticker and strike and expiration:
            occ = client._build_option_symbol(ticker, expiration, opt_type, strike)
            symbols.append(occ)
            ticker_to_symbol[ticker] = occ

    if not symbols:
        return {}

    try:
        quotes = client.get_option_quotes(symbols)
    except Exception as e:
        log_event("data_pipeline", "position_quotes_failed", {"error": str(e)})
        return {}

    prices = {}
    for ticker, sym in ticker_to_symbol.items():
        q = quotes.get(sym, {})
        bid = q.get("bid", 0)
        ask = q.get("ask", 0)
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0
        prices[ticker] = mid

    return prices
