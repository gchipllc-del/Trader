# OpenClaw Trader — Optimization Backlog

Prioritized improvements toward the **$1,500 → $5,000 → $25,000** goal.

## ✅ Recently Shipped (2026-04-16)

| # | Feature | Impact |
|---|---------|--------|
| ✅ | Scale-out rules (50% at +15%) | Locks gains on winners like SOFI, frees capital |
| ✅ | Raised `min_composite_score` 5→7 | Only high-conviction entries, blocks weak setups |
| ✅ | Fast-recycle after exits | Freed slot immediately redeployed in same cycle |
| ✅ | Kronos AI price gate | Vetoes stocks predicted to drop |
| ✅ | News sentiment gate | Vetoes stocks with strongly bearish news |
| ✅ | Calibration tracking | Win rate by score, pattern accuracy, Kronos accuracy |
| ✅ | Trailing stops (verified working) | Ratchets up 3% below peak price |

---

## 🎯 Next Up (High Impact, do soon)

### #4 — Partial entry on high-conviction setups
**Current**: Full position entered at first signal
**Proposed**: For 9/9 composite scores, enter 60% position immediately, add remaining 40% if price confirms (pulls back to zone, or breaks out above signal high)
**Benefit**: Better fills on top picks, reduces risk on fake breakouts
**Files to touch**: `lib/stock_engine.py` (split `execute_stock_buy` into initial + scaling)
**Effort**: M (1-2 hours)
**Prereq**: None

### #5 — Prune ticker universe dynamically
**Current**: 18 tickers in `tickers_phase1` — scans all of them every cycle
**Proposed**: Rolling quality filter — auto-drop tickers that haven't passed Gate 1 (quant) in 14+ days, promote new candidates from a watchlist
**Benefit**: Faster scans, focus on tickers with current edge
**Files to touch**: `lib/quant_screener.py`, `config/wheel_strategy.yaml`
**Effort**: M (2-3 hours)
**Prereq**: Need 2 weeks of scan data to calibrate

### #6 — Phase 2 premium collection (at $5k portfolio)
**Current**: Stock-only (Phase 1)
**Proposed**: At $5k portfolio, activate CSPs on cheap tickers (F, SOFI, NIO, AAL). Premium income compounds faster than stock P/L.
**Benefit**: Additional income stream, less directional risk
**Files to touch**: Already implemented in `lib/csp_engine.py` — just need to reach $5k
**Effort**: L (already built)
**Prereq**: Portfolio value ≥ $5,000

### #7 — Close the Hermes feedback loop
**Current**: `trade_history.json` is empty — no closed trades yet for Hermes to learn from
**Proposed**: Once 10+ trades close, run `python main.py hermes --dry-run` weekly to see tuning recommendations. Accept good ones.
**Benefit**: Self-improving parameters (Kelly, stop %, score threshold)
**Files to touch**: None (just operational cadence)
**Effort**: S (policy change)
**Prereq**: 10+ completed trades (will happen naturally as positions exit)

---

## 🔬 Medium-Impact Ideas (explore later)

### #8 — Volatility-based position sizing
**Idea**: Instead of fixed 30% max, size inverse to ATR. Low-vol stocks get larger positions, high-vol get smaller.
**Files**: `lib/stock_engine.py:calculate_position_size`
**Effort**: M

### #9 — Pre-market gap trading
**Idea**: Scan overnight gappers at 9:25 AM ET. Gap down >3% at support zone = high-probability fade.
**Files**: New `lib/gap_scanner.py`
**Effort**: M
**Caution**: Higher slippage, requires pre-market data feed

### #10 — Earnings season adaptive mode
**Idea**: During heavy earnings weeks, auto-lower max_concurrent to 1 and raise min_score to 9 (extreme caution)
**Files**: `lib/stock_engine.py`, `lib/enhancements.py`
**Effort**: S
**Prereq**: Earnings calendar already implemented

### #11 — Correlation-aware position sizing
**Idea**: Don't hold 2 positions with >0.8 correlation (e.g., F + NIO both EV-adjacent). Force diversification.
**Files**: `agents/risk_agent.py`
**Effort**: M

### #12 — Regime-adaptive parameters
**Idea**: Use memory_palace's `get_current_regime()` to swap parameter sets (bull: aggressive, bear: defensive, sideways: wheel/CSPs)
**Files**: `lib/stock_engine.py`, `config/wheel_strategy.yaml` (add `regime_presets` section)
**Effort**: L

---

## 🧪 Experimental (interesting but risky)

### #13 — Prediction markets as sentiment feed
**Idea**: Polymarket/Kalshi markets on "SPY above $600 by Friday?" as a crowdsourced signal
**Files**: Bridge to polybot's `lib/market_scanner.py`
**Effort**: M
**Caution**: Low liquidity on specific stock markets

### #14 — Options IV term structure as regime indicator
**Idea**: VIX9D/VIX1M/VIX3M ratios predict regime changes
**Files**: New `lib/term_structure.py`
**Effort**: M

### #15 — Multi-timeframe Kronos ensemble
**Idea**: Run Kronos at 1h AND 1d intervals, combine predictions
**Files**: `lib/kronos_forecaster.py`
**Effort**: M
**Caution**: Expensive (2x inference time)

---

## ❌ Skipped — Analyzed and Rejected

| Repo / Idea | Why Skipped |
|-------------|-------------|
| `forrestchang/andrej-karpathy-skills` | Just 4 markdown files of prompt guidelines — zero code, zero trading content |
| `kyegomez/BitNet` | No pretrained weights, would require weeks of from-scratch training, doesn't solve any current bot need |
| Full port of polybot NegRisk arbitrage | Only works on multi-outcome prediction markets — N/A for stocks |
| Metaculus community forecasts | No stock-specific markets at relevant granularity |

---

## 📊 Goal Tracking

| Milestone | Portfolio Value | Unlocks |
|-----------|-----------------|---------|
| Phase 1 (current) | $1,561 | Stock trading only |
| Phase 1.5 | $3,000 | Hermes has enough data to tune |
| **Phase 2** | **$5,000** | **CSP selling on cheap tickers** |
| Phase 3 | $10,000 | Full Wheel (CSP + CC) on all tickers |
| Short-term goal | $25,000 | Remove PDT restrictions, unlock day trading |
| Long-term goal | $250,000 | Algorithmic fund scale |

---

## Operating Principles

1. **Measure before optimizing.** No change without a calibration baseline.
2. **One change at a time.** Don't ship 3 optimizations in the same deploy.
3. **Hermes has veto power.** If it flags a parameter change as harmful, revert.
4. **Paper first, live never without migration checklist** (`python main.py migrate`).
5. **Circuit breakers are sacred.** Never tune them past safety bounds.
