# OpenClaw Wheel Trader

An autonomous stock and options trading bot that executes **The Wheel Strategy** on US equities via [Alpaca](https://alpaca.markets). Built with a multi-layered analysis framework, self-optimizing parameters, and institutional-grade safety systems.

> **Status:** Phase 1 (Stock Trading) active on Alpaca paper account. Fully automated with 4x daily scans, 3-minute monitoring, and self-optimization.

---

## How It Works

OpenClaw grows a portfolio through three phases:

| Phase | Portfolio Size | Strategy |
|-------|---------------|----------|
| **Phase 1** | < $5,000 | Swing-trade stocks using quant + technical + momentum scoring |
| **Phase 2** | $5,000 - $10,000 | Add cash-secured puts (CSPs) on cheap underlyings |
| **Phase 3** | $10,000+ | Full Wheel: sell puts, get assigned, sell covered calls, repeat |

### The Wheel Strategy

```
  Sell Cash-Secured Put (CSP)
         |
    +---------+---------+
    |                   |
  Expires OTM       Assigned
  (keep premium)    (own 100 shares)
    |                   |
    v                   v
  Repeat            Sell Covered Call (CC)
                         |
                    +---------+---------+
                    |                   |
                  Expires OTM       Called Away
                  (keep premium)    (sell shares + premium)
                    |                   |
                    v                   v
                  Repeat            Back to CSPs
```

---

## Stock Selection: Three-Gate System

Every trade must pass through three independent filters before execution.

### Gate 1: Quantitative Screen
Ranks the ticker universe by risk-adjusted metrics using 252 days of historical data.

| Metric | What It Measures |
|--------|-----------------|
| Sharpe Ratio | Risk-adjusted return (> 0.3 to pass) |
| Sortino Ratio | Downside-adjusted return |
| Max Drawdown | Worst peak-to-trough decline (> -55% to pass) |
| Annualized Volatility | Price stability |
| Average Daily Volume | Liquidity |

Composite quant score: **0-10**. Verdict: STRONG / OK / WEAK / AVOID.

### Gate 2: Technical Screen
Applies the [Candlestick Trading Bible](https://www.amazon.com/Candlestick-Trading-Bible/dp/B08XN7Y1ZQ) + [Naked Forex](https://www.amazon.com/Naked-Forex-High-Probability-Techniques/dp/1118114019) frameworks.

| Component | Score | Method |
|-----------|-------|--------|
| **Trend** | 0-3 | Multi-timeframe analysis (weekly + daily). Higher highs/lows, SMA alignment |
| **Level** | 0-3 | Support/resistance zones with minimum touches, room-to-left validation |
| **Signal** | 0-3 | 10 candlestick patterns: engulfing, morning/evening star, hammer, doji, harami, tweezers |

### Gate 3: Momentum Screen
Real-time momentum indicators for fast-moving opportunities.

| Indicator | Score | Trigger |
|-----------|-------|---------|
| **RSI** | +1 | Recovering from oversold (< 35) or in neutral zone (40-65) |
| **MACD** | +1 | Bullish cross or rising positive histogram |
| **Volume** | +1 | Surge >= 1.5x 20-day average |
| **ROC** | +1 | 5-day rate of change > 1% |

**Total composite score: 0-13** (trend + level + signal + momentum). Minimum score to enter: configurable (default 3).

---

## Architecture

```
                         Alpaca API
                             |
                      alpaca_client.py
                     (rate-limited REST)
                             |
                      data_pipeline.py
                    (bars, options, IV)
                             |
          +------------------+------------------+
          |                  |                  |
       trend.py          zones.py        candlestick.py
        (0-3)             (0-3)             (0-3)
          |                  |                  |
          |    momentum.py   |    iv_rank.py    |
          |      (0-4)       |                  |
          +--------+---------+---------+--------+
                   |                   |
           quant_screener.py    stock_engine.py
             (Gate 1)          (Gates 2 + 3)
                   |                   |
                   +--------+----------+
                            |
              +-------------+-------------+
              |             |             |
        order_gate.py  circuit_breaker  pdt_guard.py
        (3-step gate)   (6 checks)     (FINRA rule)
              |             |             |
              +-------------+-------------+
                            |
                     Alpaca Submit
                            |
              +-------------+-------------+
              |             |             |
         positions.json  audit.jsonl  memory_palace
```

### Module Map

| Module | Purpose | Lines |
|--------|---------|-------|
| `lib/alpaca_client.py` | Broker API wrapper with rate limiting | ~250 |
| `lib/data_pipeline.py` | Data fetching and transformation orchestrator | ~180 |
| `lib/stock_engine.py` | Phase 1 stock trading (score, buy, monitor, sell) | ~560 |
| `lib/quant_screener.py` | Quantitative universe screening (Sharpe, DD, vol) | ~225 |
| `lib/momentum.py` | RSI, MACD, volume surge, rate of change | ~130 |
| `lib/trend.py` | Multi-timeframe trend identification | ~140 |
| `lib/zones.py` | Support/resistance zone detection | ~200 |
| `lib/candlestick.py` | 10 candlestick pattern detectors | ~350 |
| `lib/iv_rank.py` | IV Rank and premium environment assessment | ~100 |
| `lib/screener.py` | Options candidate scoring (CSP + CC) | ~200 |
| `lib/csp_engine.py` | Cash-secured put scan and execution | ~180 |
| `lib/cc_engine.py` | Covered call scan and execution | ~220 |
| `lib/order_gate.py` | 3-step order pipeline (propose/validate/execute) | ~120 |
| `lib/circuit_breaker.py` | 6 hard safety checks | ~130 |
| `lib/pdt_guard.py` | Pattern day trader rule enforcement | ~100 |
| `lib/monitor.py` | Continuous position monitoring loop | ~400 |
| `lib/kill_switch.py` | Emergency full liquidation | ~50 |
| `lib/audit.py` | Append-only forensic audit logger | ~100 |
| `lib/memory_palace.py` | Persistent memory (ChromaDB + SQLite + diaries) | ~350 |
| `lib/backtest.py` | Monte Carlo simulation engine | ~200 |
| `lib/dashboard_data.py` | Dashboard data aggregation layer | ~300 |
| `lib/dashboard_web.py` | Flask web dashboard server | ~60 |
| `lib/dashboard_terminal.py` | Rich terminal dashboard | ~120 |
| `agents/hermes_optimizer.py` | Self-tuning parameter optimization | ~300 |
| `agents/strategy_agent.py` | Trade proposal agent | ~80 |
| `agents/risk_agent.py` | Risk validation agent (can VETO) | ~120 |
| `agents/compliance_agent.py` | Regulatory compliance agent | ~100 |
| `agents/consensus.py` | 3-agent unanimous consent protocol | ~60 |
| `main.py` | CLI entry point (10 commands) | ~250 |

---

## Safety Systems

### Circuit Breakers
| Breaker | Limit | Action |
|---------|-------|--------|
| Daily loss | -$300 | Halt all trading for the day |
| Position size | 20% of portfolio | Reject oversized orders |
| Sector concentration | 40% of portfolio | Prevent sector overweight |
| Open orders | 8 max | Prevent order pile-up |
| Cooldown | 15 min after loss | Prevent revenge trading |
| Paper mode | Enforced | Cannot go live without explicit approval |

### PDT Guard
Tracks round-trip day trades across a rolling 5-business-day window. Blocks trades that would breach the FINRA 3-day-trade limit for accounts under $25,000.

### Order Gate (3-Step Pipeline)
1. **PROPOSE** - Log intent, check for duplicates (60s dedup window via SHA-256 hash)
2. **VALIDATE** - Run all circuit breakers, verify minimum score
3. **EXECUTE** - Only proceeds if step 2 set `_validated = True`

No single function call can place an order. Steps must be called sequentially.

### Kill Switch
Emergency liquidation callable from CLI, Telegram, or automated failsafe:
- Cancel all open orders
- Close all positions at market
- Log everything to audit trail

Triggers automatically after 10 consecutive missed monitoring checks.

### Audit Trail
Append-only JSONL logger. Every action is logged **before** execution. Secrets are automatically redacted. File-locked for concurrent safety. Each event gets a UUID.

---

## Hermes Self-Optimization Agent

After market close each day, Hermes reviews trade performance and adjusts strategy parameters.

### Optimization Loop
1. **REVIEW** - Analyze closed trades: win rate, avg win/loss, expectancy, stop-out rate, profit factor
2. **DIAGNOSE** - Identify issues: "stops too tight?", "win rate supports more trades?", "leaving money on the table?"
3. **TUNE** - Adjust parameters in small steps within hard safety bounds
4. **LOG** - Record every change with reasoning to `data/hermes_log.jsonl`
5. **VALIDATE** - Ensure all values remain within bounds

### Tunable Parameters and Bounds

| Parameter | Min | Max | Step Size |
|-----------|-----|-----|-----------|
| `stop_loss_pct` | 2% | 8% | 0.5% |
| `default_target_pct` | 5% | 20% | 1% |
| `min_composite_score` | 2 | 6 | 1 |
| `max_position_pct` | 10% | 30% | 2.5% |
| `max_trades_per_scan` | 1 | 5 | 1 |
| `trailing_stop_pct` | 0% | 6% | 0.5% |
| `max_concurrent_positions` | 2 | 8 | 1 |

### Diagnostic Logic
- **Stop-out rate > 50%** -> widen stops (they're too tight)
- **Win rate < 40%** -> raise minimum score (be more selective)
- **R:R ratio < 1.0** -> widen profit targets
- **Positive expectancy + win rate > 50%** -> increase position size
- **Many target hits** -> add/increase trailing stop to catch bigger moves

---

## Memory Palace

Persistent memory system across sessions. Three storage layers:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Drawers** | ChromaDB (vector DB) | Verbatim trade reasoning, searchable by semantic similarity |
| **Knowledge Graph** | SQLite | Temporal facts with valid_from/valid_to: `(AAPL, entered_csp, 170P)` |
| **Agent Diaries** | JSONL per agent | Compressed activity logs: `"AAPL\|STOCK_BUY\|12sh@$150\|score_8/13"` |

---

## Dashboard

### Web Dashboard
Dark trading terminal UI at `http://localhost:5051` with:
- Real-time portfolio value and P/L
- Positions table with entry, current, target, stop, score
- Quant scores table for full universe
- Circuit breaker progress bars
- P/L chart (Chart.js)
- Auto-refresh every 30 seconds

### Terminal Dashboard
Rich-library colored output via `python main.py status`:
- Portfolio header panel
- Positions table with colored P/L
- Circuit breaker status

### API Endpoints
| Endpoint | Returns |
|----------|---------|
| `GET /api/state` | Full dashboard state |
| `GET /api/quant` | Quant scores (15-min cache) |
| `GET /api/portfolio` | Portfolio summary |
| `GET /api/positions` | Positions with P/L |
| `GET /api/events` | Recent audit events |
| `GET /api/history` | Trade history + P/L series |
| `GET /api/breakers` | Circuit breaker status |

---

## Automation Schedule

Runs via macOS `launchd` agents (Mon-Fri during market hours):

| Time (CT) | Job | What It Does |
|-----------|-----|--------------|
| Every 3 min | Monitor | Check stops, trailing stops, exit signals, assignments |
| 8:33 AM | Scan 1 | Market open — fresh opportunities |
| 10:15 AM | Scan 2 | Mid-morning momentum |
| 12:30 PM | Scan 3 | Lunch dip buys |
| 1:45 PM | Scan 4 | Power hour |
| 3:15 PM | Hermes | Post-close self-optimization |

---

## Quick Start

### Prerequisites
- Python 3.11+
- [Alpaca](https://alpaca.markets) account (paper trading)
- macOS (for launchd automation) or Linux (use cron)

### Installation
```bash
git clone git@github.com:gchipllc-del/Trader.git
cd Trader

pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your Alpaca API key and secret
```

### Configuration
```bash
# Strategy parameters (tickers, scores, stops, targets)
nano config/wheel_strategy.yaml

# Runtime settings (circuit breakers, monitoring, PDT)
nano config/settings.yaml
```

### Usage
```bash
# Run a market scan (auto-detects phase, scores tickers, executes trades)
python main.py scan

# Start continuous monitoring
python main.py monitor

# Check portfolio status
python main.py status
python main.py status --full    # Include quant scores

# Launch web dashboard
python main.py dashboard

# Run Hermes self-optimization
python main.py hermes              # Live mode
python main.py hermes --dry-run    # Analysis only

# Check PDT day trade status
python main.py pdt

# Run Monte Carlo backtest
python main.py backtest --ticker SPY

# Emergency liquidation
python main.py kill --reason "manual stop"

# Paper-to-live migration checklist
python main.py migrate
```

### Automation Setup (macOS)
```bash
# Load launchd agents for automated trading
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.openclaw.scan1.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.openclaw.monitor.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.openclaw.hermes.plist

# Keep Mac awake
caffeinate -dims &
```

---

## File Structure
```
openclaw-wheel-trader/
  config/
    wheel_strategy.yaml     # Strategy parameters (Hermes-tunable)
    settings.yaml           # Runtime settings (safety-critical)
  lib/
    alpaca_client.py        # Broker API wrapper
    data_pipeline.py        # Data fetching orchestrator
    stock_engine.py         # Phase 1 stock trading engine
    quant_screener.py       # Quantitative universe screening
    momentum.py             # RSI, MACD, volume, ROC indicators
    trend.py                # Multi-timeframe trend analysis
    zones.py                # Support/resistance detection
    candlestick.py          # 10 candlestick pattern detectors
    iv_rank.py              # IV environment assessment
    screener.py             # Options candidate scoring
    csp_engine.py           # Cash-secured put engine
    cc_engine.py            # Covered call engine
    order_gate.py           # 3-step order safety pipeline
    circuit_breaker.py      # Hard safety limits
    pdt_guard.py            # Pattern day trader protection
    monitor.py              # Continuous position monitoring
    kill_switch.py          # Emergency liquidation
    audit.py                # Append-only audit logger
    memory_palace.py        # Persistent memory system
    backtest.py             # Monte Carlo simulation
    dashboard_data.py       # Dashboard data aggregation
    dashboard_web.py        # Flask web server
    dashboard_terminal.py   # Rich terminal display
  agents/
    hermes_optimizer.py     # Self-tuning parameter optimizer
    strategy_agent.py       # Trade proposal agent
    risk_agent.py           # Risk validation (can VETO)
    compliance_agent.py     # Regulatory compliance checks
    consensus.py            # 3-agent unanimous consent
  scripts/
    run_scan.sh             # Cron/launchd scan runner
    run_monitor.sh          # Cron/launchd monitor runner
    run_hermes.sh           # Cron/launchd Hermes runner
  templates/
    dashboard.html          # Web dashboard UI
  data/
    positions.json          # Current position state
    trade_history.json      # Completed trade records
    palace/                 # Memory palace storage
  logs/
    audit_log.jsonl         # Forensic audit trail
  tests/
    test_sprint01.py        # 34 tests for Sprint 0+1
    test_sprint02_09.py     # 41 tests for Sprints 2-9
  main.py                   # CLI entry point
  .env                      # API keys (git-ignored)
```

---

## Disclaimer

This software is for **educational and paper trading purposes only**. It is not financial advice. Trading stocks and options involves significant risk of loss. Past performance does not guarantee future results. Always do your own research before trading with real money.

---

## License

Private repository. All rights reserved.

---

Built by Jesse @ GChip LLC
