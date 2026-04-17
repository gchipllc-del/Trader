# OpenClaw Wheel Trader - Brief Overview

## What Is It?
An autonomous trading bot that buys and sells stocks (and eventually options) on autopilot using the Alpaca brokerage API. It scans the market 4x daily, scores every stock on a 0-13 scale using quantitative data, technical analysis, and momentum indicators, then executes trades that pass all safety checks. After market close, a self-optimization agent (Hermes) reviews performance and tunes the strategy parameters automatically.

## What Does It Trade?
Currently 17 stocks across tech, fintech, travel, commodities, and EV sectors. Every stock must pass a quantitative screen (Sharpe ratio, drawdown, volatility), a technical screen (trend direction, support/resistance zones, candlestick patterns), and a momentum screen (RSI, MACD, volume surge, rate of change) before the bot will buy it.

## How Does It Make Money?
- **Buys** stocks at support zones with bullish confirmation signals
- **Sells** when price hits resistance targets, trailing stops lock in gains, or bearish reversal patterns appear
- **Cuts losses** quickly with 3.5% stop losses
- As the portfolio grows past $5,000, it graduates to **selling options** (The Wheel Strategy) for premium income

## What Makes It Safe?
- Paper trading only (no real money until explicitly approved)
- 6 circuit breakers: daily loss limit, position size cap, cooldown after losses
- Pattern Day Trader guard (FINRA 3-trade rule)
- 3-step order pipeline: no single function can place an order
- Kill switch for emergency liquidation
- Every action logged to an append-only audit trail

## What Runs Automatically?
- 4 market scans per day (8:33, 10:15, 12:30, 1:45 CT)
- Position monitoring every 3 minutes (stops, targets, exit signals)
- Hermes self-optimization after market close (reviews trades, adjusts parameters)
- Web dashboard at localhost:5051 (polybot uses 5050)

## Tech Stack
Python 3.11, Alpaca REST API, Flask, SQLite, ChromaDB, pandas, numpy, Rich, Chart.js
