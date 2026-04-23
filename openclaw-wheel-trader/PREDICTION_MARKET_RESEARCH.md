# Prediction Market Trading Bot — Comprehensive Research Report

**Prepared for:** Jesse  
**Date:** April 15, 2026  
**Purpose:** Overnight research to inform cloning openclaw-wheel-trader for prediction market trading

---

## 1. PREDICTION MARKET PLATFORMS — Which Ones Allow Automated Trading?

### Platform Comparison Matrix

| Feature | Polymarket | Kalshi | Manifold | Metaculus | PredictIt |
|---------|-----------|--------|----------|-----------|-----------|
| Programmatic Trading | Yes | Yes | Yes | Forecasts only | No |
| Python SDK | Official | Official | Community | Unofficial | None |
| REST API | Yes | Yes | Yes | Yes | No |
| WebSocket | Yes | Yes | No | No | No |
| US Legal | Gray area | **Yes (CFTC)** | Yes (play $) | Yes | Uncertain |
| Fees | ~2% winnings | 7% profit | Free (play) | Free | 10%+5% |
| Market Volume | Very High | Medium | Low (play) | N/A | Low |
| Market Variety | Excellent | Good | Excellent | Good | Poor |
| Bot Friendliness | 9/10 | 8/10 | 7/10 | 4/10 | 2/10 |

**Recommendation:** Primary targets should be **Polymarket** (volume, variety, bot-friendly) and **Kalshi** (US-legal, regulated, clean API). Use **Manifold** for paper trading / strategy validation. Use **Metaculus** as a data source.

---

### 1A. Polymarket

**Overview:** The largest prediction market by volume. Built on Polygon (Ethereum L2). Uses a CLOB (Central Limit Order Book) model through on-chain settlement.

**API:**
- REST API at `https://clob.polymarket.com` (CLOB API)
- REST API at `https://gamma-api.polymarket.com` (Gamma API — market data, events, pricing)
- WebSocket support for real-time order book and trade feeds
- Full programmatic trading: place, cancel, amend orders
- Authentication via Polygon wallet (private key signs CLOB API key credentials)

**Key Endpoints:**
- `GET /markets` — list all active markets
- `GET /book` — order book for a specific market
- `POST /order` — place a limit order
- `DELETE /order/{id}` — cancel order
- `GET /trades` — trade history
- `GET /positions` — your current positions

**Python SDK:**
- Official: `py-clob-client` (pip install py-clob-client)
- Also: `polymarket-py` community wrapper

**Fees:**
- No maker fees (you earn rebates for providing liquidity)
- Taker fees ~2% of winnings on resolution (not on trade execution)
- No deposit/withdrawal fees beyond gas

**Markets:** Politics (largest category), crypto prices, world events, sports, entertainment, AI milestones, weather, elections, geopolitical events. Thousands of active markets.

**US Restrictions:** Polymarket settled with the CFTC in January 2022 for $1.4M and is technically not available to US users. They geo-block US IPs and require non-US attestation. US users trading on Polymarket operate in a legal gray area.

---

### 1B. Kalshi

**Overview:** The only CFTC-regulated prediction market exchange in the US. Legally operates as a Designated Contract Market (DCM). Binary event contracts.

**API:**
- REST API at `https://trading-api.kalshi.com/trade-api/v2`
- WebSocket feed for market data streaming
- Full programmatic trading supported
- Authentication via API key (generated in dashboard)

**Key Endpoints:**
- `GET /exchange/status` — exchange status
- `GET /markets` — list markets with filtering
- `GET /markets/{ticker}/orderbook` — order book
- `POST /portfolio/orders` — place order
- `DELETE /portfolio/orders/{order_id}` — cancel
- `GET /portfolio/positions` — current positions
- `GET /portfolio/settlements` — settlement history

**Python SDK:**
- Official: `kalshi-python` (pip install kalshi-python)
- Well-documented with examples

**Fees:**
- Exchange fee: 7% of profit per contract on settlement (no fee on losing contracts)
- No per-trade commission

**Markets:** Weather (temperature, hurricane), economics (CPI, Fed rate, GDP), politics (elections, legislation), finance (stock prices, crypto), sports, world events.

**US Restrictions:** Fully legal for US users. CFTC-regulated, US-domiciled.

---

### 1C. Manifold Markets

**Overview:** Play-money prediction market (Mana currency). Very open, anyone can create markets. Good for paper trading / strategy testing.

**API:**
- REST API at `https://api.manifold.markets/v0`
- Full programmatic trading supported
- Authentication via API key (from profile settings)

**Key Endpoints:**
- `GET /v0/markets` — list markets
- `POST /v0/bet` — place a bet
- `POST /v0/market` — create a new market

**Python SDK:** Community: `manifoldpy` (pip install manifoldpy)

---

### 1D. Metaculus

Not a trading venue — forecasting platform only. Useful as a **data source** for calibration signals. Metaculus community predictions are well-calibrated. Scraping their forecasts as a signal source for Polymarket/Kalshi trades is a legitimate strategy.

---

### 1E. PredictIt

No public API for trading. 10% profit fee + 5% withdrawal fee. Declining relevance. **Not recommended for automated trading.**

---

## 2. OPEN SOURCE PREDICTION MARKET BOTS — GitHub Repos to Review

### Key Repos/SDKs

**Official Platform SDKs:**
- `github.com/Polymarket/py-clob-client` — Official Python client for Polymarket CLOB API. Handles wallet signing, order management, market data. Production-grade.
- `github.com/Kalshi/kalshi-python` — Official Kalshi API client. Authentication, order management, market data.

**Community Bots to Search:**
```bash
# Run these searches tomorrow morning
gh search repos "polymarket bot" --sort stars --limit 20
gh search repos "kalshi trading bot" --sort stars --limit 20
gh search repos "prediction market trading" --sort stars --limit 20
gh search repos "polymarket python" --sort stars --limit 20
gh search repos "polymarket AI agent" --sort stars --limit 20
gh search repos "prediction market arbitrage" --sort stars --limit 20
```

**Common Patterns in Existing Repos:**
- **LLM-powered:** Feed market description + news to GPT-4/Claude, ask for probability estimate, compare to market price, trade the discrepancy
- **Arbitrage:** Cross-platform price discrepancies (Polymarket vs Kalshi)
- **Weather bots:** Pull NOAA data and trade temperature contracts on Kalshi
- **Forecasting aggregators:** Combine Metaculus + polls + models into ensemble predictions

### Red Flags When Evaluating Repos

- No tests — prediction market code without tests is dangerous
- Hardcoded private keys — security issue, signals amateur code
- "100% win rate" claims — scam or delusion
- Requires depositing through their smart contract — potential rug pull
- No audit logging — you need to know what the bot did
- No rate limiting — will get you banned from the API
- Last commit >6 months ago — APIs change frequently, stale code likely broken

---

## 3. WINNING STRATEGIES FOR PREDICTION MARKETS

### Strategy Comparison

| Strategy | Edge | Win Rate | Capital Needed | Complexity | Bot Fit |
|----------|------|----------|---------------|------------|---------|
| Arbitrage | Small, consistent | Very high | High (capital on 2+ platforms) | Medium | Excellent |
| News Alpha | High when right | Medium | Medium | High | Good |
| Calibration | Moderate, durable | Medium-high | Low | Medium | Excellent |
| Market Making | Small, consistent | High | High | High | Good |
| Info Aggregation | Moderate | Medium-high | Medium | Medium | Excellent |
| Contrarian | Low rate, big payoff | Low | Low | Low | Good |

**Recommendation for v1:** Start with **Calibration Edge** (Bayesian updating) combined with **Information Aggregation**. These map most naturally to your existing architecture. Add **News Alpha** as v2.

### 3A. Arbitrage
Exploit price discrepancies across platforms or within a platform. Same event priced differently on Polymarket vs Kalshi = guaranteed profit minus fees. Expected edge: 1-3% per trade, requires speed and volume.

### 3B. News Alpha (LLM-Powered)
Use LLMs/NLP to analyze breaking news faster than the market can reprice. Highest-alpha strategy. Market can take minutes to hours to fully reprice after breaking news. Risk: LLM hallucinations, being late.

### 3C. Calibration Edge (Bayesian Updating)
Maintain better-calibrated probability estimates than the market. Start with base rate, apply Bayesian updating as new information arrives, trade when your confidence interval doesn't overlap with market price. Your memory palace is perfect for storing evidence chains.

### 3D. Market Making
Post both buy and sell orders, earn the bid-ask spread. Maps directly to Polymarket's CLOB model (maker rebates!). Main risk: adverse selection.

### 3E. Information Aggregation
Combine Metaculus + Polymarket + LLM + base rates + expert polls + betting odds into a weighted ensemble. Hermes tracks which sources are most accurate over time.

### 3F. Contrarian / Mean Reversion
Fade extreme positions (>92c or <8c). Low win rate, high payoff (20:1). Kelly criterion is critical.

---

## 4. ARCHITECTURE TRANSLATION: OPENCLAW WHEEL TRADER → OPENCLAW ORACLE TRADER

### What Stays (Almost Unchanged)

| Current Module | Prediction Market Version | Changes |
|---------------|--------------------------|---------|
| `lib/audit.py` | Keep as-is | None |
| `lib/circuit_breaker.py` | Keep with mods | Add max_per_market, category limits |
| `lib/order_gate.py` | Keep as-is | Update OrderIntent fields |
| `lib/kill_switch.py` | Keep as-is | Swap Alpaca for Polymarket/Kalshi |
| `lib/memory_palace.py` | Keep with new wings | Wings = market categories, rooms = event types |
| `agents/consensus.py` | Keep as-is | Same flow, different criteria |
| `agents/hermes_optimizer.py` | Keep with new params | Tune calibration weights instead of stops/targets |
| `main.py` | Keep with new commands | Add: `forecast`, `arb-scan`, `calibrate` |

### What Changes Significantly

**`lib/alpaca_client.py` → `lib/market_client.py`**
```python
class PredictionMarketClient:
    """Unified client supporting multiple prediction market platforms."""
    
    def __init__(self, platform: str = "polymarket"):
        if platform == "polymarket":
            self.client = PolymarketClient(private_key=os.environ["POLY_PRIVATE_KEY"])
        elif platform == "kalshi":
            self.client = KalshiClient(api_key=os.environ["KALSHI_API_KEY"])
        elif platform == "manifold":
            self.client = ManifoldClient(api_key=os.environ["MANIFOLD_API_KEY"])
    
    def get_markets(self, category=None, status="active") -> list[Market]
    def get_orderbook(self, market_id: str) -> OrderBook
    def place_order(self, market_id, side, price, quantity) -> Order
    def cancel_order(self, order_id) -> bool
    def get_positions(self) -> list[Position]
    def get_balance(self) -> float
```

**`lib/screener.py` → `lib/market_scanner.py`**
```python
@dataclass
class MarketCandidate:
    market_id: str
    question: str               # "Will X happen by Y date?"
    platform: str               # polymarket, kalshi, manifold
    current_price: float        # 0.00-1.00 (market's implied probability)
    our_probability: float      # Our estimate
    edge: float                 # our_probability - current_price
    confidence: float
    resolution_date: str
    volume_24h: float
    
    # New scoring (replaces trend/level/signal)
    evidence_score: int         # 0-3
    calibration_score: int      # 0-3
    edge_score: int             # 0-3
    composite_score: int        # 0-9
    
    side: str                   # "YES" or "NO"
    kelly_fraction: float       # Kelly criterion position size
```

**`lib/candlestick.py` + `lib/zones.py` + `lib/trend.py` → `lib/forecaster.py`**
```python
class Forecaster:
    """Generates calibrated probability estimates."""
    
    def estimate_probability(self, market: Market) -> ForecastResult:
        # Aggregate: base rate + LLM + Metaculus + news + expert polls
        
    def bayesian_update(self, prior: float, evidence: Evidence) -> float:
        """Update probability given new evidence."""
        
    def calculate_edge(self, our_prob: float, market_prob: float) -> float:
        """Edge = our probability minus market probability."""
```

### What's Entirely New

1. **`lib/news_feed.py`** — Real-time news monitoring (NewsAPI, Reuters RSS, Twitter, Reddit)
2. **`lib/resolution_tracker.py`** — Settlement management and P/L on resolution
3. **`lib/arb_scanner.py`** — Cross-platform arbitrage detection
4. **`lib/kelly.py`** — Kelly Criterion position sizing for binary outcomes

### Agent Roles Translation

| Agent | Stock Bot | Prediction Market Bot |
|-------|-----------|----------------------|
| Strategy | "AAPL at support with hammer, sell 170P" | "Fed rate market at 42c, our Bayesian estimate is 58%, buy YES" |
| Risk | "Position 8% of portfolio, tech sector 22%" | "Position 3% of portfolio, politics category 18%, uncorrelated" |
| Compliance | "No wash sale, no earnings conflict" | "Market active, resolution clear, within platform limits" |

### Circuit Breaker Translation

| Current | Prediction Market |
|---------|-------------------|
| max_daily_loss: -$500 | max_daily_loss: -$500 (same) |
| max_position_pct: 10% | max_per_market: 10% |
| max_open_orders: 5 | max_open_positions: 20 (more diversification) |
| cooldown_after_loss: 30min | cooldown_after_bad_forecast: 60min |
| **NEW** | max_category_exposure: 30% |
| **NEW** | max_resolution_date_exposure: 25% |
| **NEW** | min_liquidity: $10k volume |

### Hermes Tunable Parameters

| Current Param | Prediction Market Param | Bounds |
|--------------|------------------------|--------|
| stop_loss_pct | early_exit_threshold | 5-20% |
| default_target_pct | take_profit_threshold | 10-40% |
| min_composite_score | min_evidence_score | 4-8 / 9 |
| max_position_pct | max_per_market_pct | 3-15% |
| **NEW** | llm_weight | 0.1-0.5 |
| **NEW** | metaculus_weight | 0.1-0.5 |
| **NEW** | news_sensitivity | 0.1-0.8 |
| **NEW** | kelly_multiplier (fraction of full Kelly) | 0.25-0.75 |

---

## 5. KEY TECHNICAL CHALLENGES

### 5A. Pricing Prediction Market Positions
- A 65c contract = market implies 65% probability of YES
- Your job: determine if TRUE probability differs from MARKET probability
- **Expected Value:** EV = (your_prob * payout) - (1 - your_prob) * cost
- **Only trade when edge > fee + uncertainty margin** (minimum ~5-10% edge)

### 5B. Risk Management for Binary Outcomes
- **You can lose 100%** of a position (vs stocks rarely going to $0)
- **Kelly Criterion:** f* = (p*b - q) / b. Always use fractional Kelly (quarter or half)
- **Diversification:** 20+ small uncorrelated positions dramatically reduces variance
- **Correlation awareness:** "Will X win?" and "Will X's party win?" are the same bet

### 5C. Market Resolution and Settlement
- Polymarket: UMA Optimistic Oracle, can be disputed, on-chain settlement
- Kalshi: centralized resolution based on official data (NOAA, BLS, AP), same/next-day
- Your bot needs: resolution date tracking, disputed resolution handling, P/L calculation with fees

### 5D. Data Sources for Predictions
- **News:** NewsAPI, GDELT, Reuters RSS, Twitter/X, Reddit
- **Forecasts:** Metaculus API, Good Judgment Open, INFER
- **Domain:** NOAA (weather), BLS (economics), FRED (indicators), FiveThirtyEight (politics)
- **Official:** PACER (courts), congress.gov (legislation), SEC EDGAR

---

## 6. LEGAL/REGULATORY LANDSCAPE

- **Kalshi:** Fully legal for US users. CFTC-regulated DCM. Automated trading explicitly supported.
- **Polymarket:** Gray area for US users. Geo-blocks US IPs. Regulatory risk exists.
- **Manifold:** Legal. Play money / sweepstakes model.
- **CFTC trend:** Generally warming to prediction markets after Kalshi v. CFTC court victory.
- **Practical advice:** Start with Kalshi (unambiguously legal), use Manifold for paper trading.

---

## 7. IMPLEMENTATION ROADMAP

### Phase 1: Foundation (Week 1-2)
- [ ] Fork openclaw-wheel-trader to openclaw-oracle-trader
- [ ] Replace `lib/alpaca_client.py` with `lib/market_client.py` (Kalshi first)
- [ ] Modify `OrderIntent` for binary contracts (YES/NO, market_id, probability)
- [ ] Update circuit breakers for prediction market limits
- [ ] Set up Manifold as paper trading platform
- [ ] Install SDKs: `pip install kalshi-python py-clob-client manifoldpy`

### Phase 2: Forecasting Engine (Week 2-3)
- [ ] Build `lib/forecaster.py` with Bayesian updating
- [ ] Integrate Metaculus API as forecast source
- [ ] Build LLM-powered market analysis (Claude API)
- [ ] Replace `lib/screener.py` with `lib/market_scanner.py`
- [ ] Implement Kelly criterion position sizing

### Phase 3: Agent Adaptation (Week 3-4)
- [ ] Update strategy/risk/compliance agents for prediction markets
- [ ] Update memory palace wings/rooms for market categories
- [ ] Adapt Hermes for calibration weight tuning

### Phase 4: News Alpha (Week 4-5)
- [ ] Build `lib/news_feed.py` for real-time news monitoring
- [ ] Build LLM pipeline: news → probability impact → trade signal

### Phase 5: Arbitrage (Week 5-6)
- [ ] Build `lib/arb_scanner.py` for cross-platform comparison
- [ ] Implement intra-market arb detection

### Phase 6: Production Hardening (Week 6-7)
- [ ] Backtest on historical data
- [ ] Calibration tracking dashboard
- [ ] Paper trading on Kalshi for 2+ weeks before real money

---

## 8. TOMORROW MORNING CHECKLIST

1. **Search GitHub** for repos:
   ```bash
   gh search repos "polymarket bot" --sort stars --limit 20
   gh search repos "kalshi trading bot" --sort stars --limit 20
   gh search repos "prediction market AI" --sort stars --limit 20
   ```

2. **Sign up for Kalshi** at kalshi.com — generate API key

3. **Get a Manifold API key** at manifold.markets — paper trading

4. **Install SDKs:**
   ```bash
   pip install kalshi-python py-clob-client manifoldpy
   ```

5. **Review your repos** against this architecture plan

6. **Test the APIs** — fetch active markets from each platform

---

**Key takeaway:** Your existing architecture is surprisingly well-suited for prediction markets. The 3-gate order pipeline, 3-agent consensus, circuit breakers, memory palace, and Hermes optimizer all translate directly. The main work is replacing the technical analysis layer (candlesticks, S/R zones, IV rank) with a forecasting layer (Bayesian updating, LLM analysis, news monitoring) and swapping Alpaca for prediction market platform clients. The safety infrastructure you already built is the hard part — and it's done.

---

*Research compiled by Claude — April 15, 2026*
