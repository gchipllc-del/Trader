# GitHub Scan — 2026-05-27

Follow-up to `docs/github_scan_2026-05-20.md`. Scoped to the bot's
*current* gaps after the research-pass merge (Turtle + PEAD + confluence
+ paper-live realism) and today's paper-vs-live divergence fix.

## What we already have

- Turtle pre-filter (regime + Donchian) — now live + backtest
- Confluence filter (5-signal: Turtle + Markov + Bayesian + PEAD + Bull/Bear)
- Bull/Bear agents (one-shot scoring, ~2 LLM calls)
- News sentiment veto + reranking
- PEAD signal (post-earnings drift)
- Forever-hold protection (TSM, MU, AVGO, META, MSFT, CBRS)
- Paper-to-live realism overlay (slippage, gap stops, drift detector)
- Hermes self-improving scientific loop
- Wheel strategy (CSP → assignment → CC) with options pricing + Greeks

## What we DON'T have (the gaps this scan targets)

- ❌ Fundamentals analysis (EPS growth, margins, debt/equity)
- ❌ Dedicated Sentiment Analyst (current news_sentiment is passive)
- ❌ Structured multi-round bull/bear debate (currently one-shot scoring)
- ❌ LangGraph-style orchestration (current agents are ad-hoc Python)
- ❌ Persistent decision-log reflection (Hermes is param-level, not decision-level)
- ❌ Options-flow / unusual-activity data (paid APIs only at $35+/mo)
- ❌ Earnings-call transcript sentiment (separate from news headlines)

---

## Ranked findings

### 1. ⭐ TauricResearch/TradingAgents — 80.1k stars

- **Repo**: https://github.com/TauricResearch/TradingAgents
- **License**: Apache-2.0
- **Last release**: v0.2.5 (2026-05-11)
- **Stack**: Python 3.13, LangGraph, Claude 4.x supported
- **What it is**: 7-agent debate framework in 3 teams
  - Analyst Team (4): Fundamentals, Sentiment, News, Technical
  - Research Team (2): Bull, Bear
  - Decision Team (2): Trader, Portfolio Manager
- **Decision flow**: Multi-round structured debate → Trader synthesizes
  → Portfolio Manager final risk gate

**Why it matters for us:**
1. We're missing 3 of the 4 Analyst types (no Fundamentals, no Sentiment,
   no News agent — only passive signals)
2. Our Bull/Bear are one-shot scorers; theirs debate in rounds
3. Their LangGraph orchestration is a proven framework; ours is ad-hoc

**Integration sketch (1-2 weeks):**
- Replace `agents/bull_agent.py` + `agents/bear_agent.py` ad-hoc orchestration
  with their LangGraph workflow
- Keep our wheel-specific intelligence (Turtle, PEAD, CSP/CC engines)
  as `tools` the LangGraph agents call
- Add new `agents/fundamentals_agent.py` (uses Finnhub API we already have)
- Keep our Hermes scientific loop on top as the meta-optimizer

**Risk:** Big lift, may not produce WR gain proportional to effort.
Validate first by reading their backtest results in their paper.

---

### 2. ⭐ hopit-ai/india-trade-cli (Vibe Trading) — 54 stars, MIT

- **Repo**: https://github.com/hopit-ai/india-trade-cli
- **License**: MIT
- **What it is**: 7 LLM analyst agents (Technical, Fundamental, Options,
  News/Macro, Sentiment, Sector Rotation, Risk Manager), then a **5-round
  bull-vs-bear debate** (Bull → Bear → Bull rebuttal → Bear rebuttal →
  Facilitator synthesis)
- **Cost**: 8–11 LLM calls per decision (cheap on Gemini/Ollama free tiers)
- **Output**: 3 risk-profiled trade plans (Aggressive/Neutral/Conservative)
  with entry, stop, target

**Why it matters for us:**

The **5-round debate pattern** is the easiest win in this scan. We currently
ask each agent for ONE score; they don't see each other's argument. A
rebuttal round lets the Bear specifically counter the Bull's case (and
vice versa), which is exactly the kind of disagreement-aware reasoning
that catches setups our one-shot scorers miss.

**Integration sketch (2-3 days):**
- Modify `agents/bull_agent.py` / `bear_agent.py` to support 5-round mode
- Round 1-2: each makes opening argument (current behavior)
- Round 3-4: each agent sees the other's argument and must rebut it
  (new — small prompt change)
- Round 5: facilitator synthesizes (could be the existing trader or a
  new lightweight LLM call)
- Adds 30-60s per stock evaluation; with 5 candidates/scan it's tolerable

**Risk:** Low. Easy to A/B test by gating behind a YAML toggle.

---

### 3. alpacahq/options-wheel — official Alpaca template

- **Repo**: https://github.com/alpacahq/options-wheel
- **What it is**: Official Alpaca template for the wheel — much simpler
  than ours (no zones, no signals, no agents, no PEAD, no Hermes, no
  earnings filter, no risk caps beyond buying power)
- **Their 2-week paper result**: +0.95% on $100k. Our 10d result:
  +1.17% on $1.5k with 142 trades.

**One worth-stealing nugget — their put score formula:**

```
score = (1 - |Δ|) × (250 / (DTE + 5)) × (bid price / strike price)
```

Clean closed-form for "delta safety × theta decay × premium yield". Our
`csp_engine.py` scoring may be overcomplicated — worth comparing.

**Integration sketch (30 min):**
- Read `lib/csp_engine.py` ranking function
- Compare to this formula
- If ours has artifacts that this doesn't (or vice versa), tune

---

### 4. vahagn-madatyan/wheel-it — 1 star, but ONE great idea

- **Repo**: https://github.com/vahagn-madatyan/wheel-it
- **License**: Apache-2.0, last release v0.4.0 (2026-03-21)
- **Best idea**: **Rank options by EXTRINSIC premium only**, not total
  premium. Prevents ITM contracts from inflating apparent annualized
  yield (an ITM put's "premium" is mostly intrinsic value, not actual
  edge collected).

**Integration sketch (30-60 min):**
- Audit `lib/csp_engine.py` and `lib/cc_engine.py` premium-scoring paths
- For any in-the-money option, subtract intrinsic value: `extrinsic = bid - max(0, strike - underlying)` for puts; for calls flip the sign
- Re-rank by extrinsic only
- This is a correctness fix, not a strategy change — should never make
  us worse

**Risk:** Near-zero. Pure correctness improvement.

---

### 5. rj694/earnings-sentiment — VALIDATION ONLY, don't build

- **Repo**: https://github.com/rj694/earnings-sentiment
- **What it is**: FinBERT + Loughran-McDonald dictionary on earnings call
  transcripts
- **Critical finding**: sentiment correlates with 1-day post-earnings
  return at ρ≈0.3, **but vanishes by day 5**

**Why I'm flagging this as DON'T BUILD:**

We have PEAD already (60-day decay window). Adding a sentiment-of-call
overlay would only help in the FIRST 24-48 hours after earnings. Our PEAD
already captures the multi-day drift component cleanly. Adding sentiment
gives us a same-day flash signal that's hard to act on with our 6-min
scan cadence.

Logging this as a "researched and rejected" so we don't keep coming back
to it.

---

## Out of scope (skip until $5k bankroll)

- **Unusual Options Activity scanners** (Unusual Whales, FlowAlgo) — paid
  ($35+/mo). Real edge but cost-prohibitive at current bankroll. Revisit
  when we cross the sector_rotation activation threshold.
- **TradingAgents UI/dashboard** — we already have our own.

---

## Recommended order of operations

1. **WEEK 1** (low risk, high value)
   - `wheel-it` extrinsic-only ranking (30-60 min audit + fix)
   - `alpacahq/options-wheel` score formula comparison (30 min)
   - `india-trade-cli` 5-round debate pattern (2-3 days)
2. **WEEK 2-3** (only if Week 1 lifts WR materially)
   - `TauricResearch/TradingAgents` Fundamentals Analyst integration
     (start with JUST the Fundamentals agent before adopting the whole
     LangGraph stack)

Don't bite TradingAgents whole-hog in Week 1 — too much surface area and
the WR improvement from the 5-round debate alone may make it unnecessary.

## Sources

- [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
- [TradingAgents homepage](https://tradingagents-ai.github.io/)
- [hopit-ai/india-trade-cli (Vibe Trading)](https://github.com/hopit-ai/india-trade-cli)
- [alpacahq/options-wheel](https://github.com/alpacahq/options-wheel)
- [vahagn-madatyan/wheel-it](https://github.com/vahagn-madatyan/wheel-it)
- [rj694/earnings-sentiment](https://github.com/rj694/earnings-sentiment)
- [cdubiel08/Earnings-Calls-NLP](https://github.com/cdubiel08/Earnings-Calls-NLP)
- [alex-jb/orallexa-ai-trading-agent](https://github.com/alex-jb/orallexa-ai-trading-agent)
- [PickMyTrade: Build a Multi-Agent AI Trading System (2026)](https://blog.pickmytrade.io/build-a-multi-agent-ai-trading-system-with-trading-agents-2026/)
- [Ultra Lab: 5 Hottest AI Finance Projects on GitHub in 2026](https://ultralab.tw/en/blog/ai-finance-github-projects-2026)
- [DEV: Build an Unusual Options Activity Scanner with Python](https://dev.to/orthogonalinfo/build-an-unusual-options-activity-scanner-with-python-and-free-data-kka)
- [hudson-and-thames/backtest_tutorial — Transaction Costs](https://github.com/hudson-and-thames/backtest_tutorial/blob/main/Intro_Transaction_Costs.ipynb)
