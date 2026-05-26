"""
LLM Stock Analyst — Multi-provider LLM-powered stock setup analysis.

Supports:
    - DeepSeek (deepseek-chat / deepseek-reasoner) — OpenAI-compatible API
    - Claude (claude-haiku-4-5 / claude-sonnet / claude-opus) — Anthropic API

Uses an LLM to add qualitative reasoning on top of quantitative signals. Great
for high-conviction setups where extra context can improve probability estimates.

Provider is selected in config/wheel_strategy.yaml under `llm.provider`. The
corresponding API key is read from env vars:
    - DeepSeek:  DEEPSEEK_API_KEY
    - Claude:    ANTHROPIC_API_KEY

Security:
    - API keys loaded ONLY from environment variables
    - Responses treated as untrusted (defensive parsing)
    - Responses cached with TTL to control costs
    - Rate limiting enforced
    - Graceful fallback if API key missing, provider unreachable, or credits exhausted
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

from lib.audit import log_event

CONFIG_PATH = Path(__file__).parent.parent / "config"
CACHE_DIR = Path(__file__).parent.parent / "data" / "llm_cache"

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_TIMEOUT = 60  # Seconds — deepseek-reasoner can be slow


# ── Structured-output path (instructor library) ──────────────────
# Optional, opt-in via llm.use_instructor in wheel_strategy.yaml.
# When enabled, instructor + pydantic validate the LLM response automatically
# and retry on validation failure — killing the fragile brace-counting regex
# in _parse_response(). Falls back cleanly if instructor isn't installed.
try:
    import instructor  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore
    _HAS_INSTRUCTOR = True
except Exception:
    _HAS_INSTRUCTOR = False


if _HAS_INSTRUCTOR:
    class _OptionAnalysisSchema(BaseModel):
        win_probability: float = Field(ge=0.0, le=1.0,
            description="Probability the sold option expires worthless (our win)")
        confidence: float = Field(ge=0.0, le=1.0,
            description="Self-assessed confidence in this analysis")
        bullish_factors: list[str] = Field(default_factory=list, max_length=5)
        bearish_factors: list[str] = Field(default_factory=list, max_length=5)
        reasoning: str = Field(default="", max_length=600)
        suggested_action: str = Field(default="wait",
            description="sell | wait | skip")

    class _StockAnalysisSchema(BaseModel):
        win_probability: float = Field(ge=0.0, le=1.0)
        confidence: float = Field(ge=0.0, le=1.0)
        bullish_factors: list[str] = Field(default_factory=list, max_length=5)
        bearish_factors: list[str] = Field(default_factory=list, max_length=5)
        reasoning: str = Field(default="", max_length=600)
        suggested_action: str = Field(default="wait",
            description="buy | wait | skip")


def _use_instructor(llm_cfg: dict) -> bool:
    """Should we route this call through instructor? Both the lib + config must be on."""
    return _HAS_INSTRUCTOR and bool(llm_cfg.get("use_instructor", True))


def _call_with_instructor(
    system_prompt: str,
    user_prompt: str,
    provider: str,
    model: str,
    max_tokens: int,
    schema_class: type,
    ticker: str,
):
    """Call provider via instructor with a pydantic schema. Returns model instance or None.

    Handles DeepSeek (OpenAI-compatible) and Claude. Any exception (missing key,
    API error, validation retries exhausted) → returns None so the caller can
    fall back to the legacy parse path.
    """
    try:
        if provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                log_event("llm_analyst", "no_api_key",
                          {"ticker": ticker, "provider": "deepseek"})
                return None
            from openai import OpenAI  # instructor[openai] extra pulls this in
            client = instructor.from_openai(
                OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=api_key),
                mode=instructor.Mode.JSON,  # DeepSeek supports JSON mode
            )
            return client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                response_model=schema_class,
                max_retries=2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        elif provider == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                log_event("llm_analyst", "no_api_key",
                          {"ticker": ticker, "provider": "claude"})
                return None
            from anthropic import Anthropic
            client = instructor.from_anthropic(Anthropic())
            return client.messages.create(
                model=model,
                max_tokens=max_tokens,
                response_model=schema_class,
                max_retries=2,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        else:
            return None
    except Exception as e:
        log_event("llm_analyst", "instructor_failed", {
            "ticker": ticker, "provider": provider, "error": str(e)[:200],
        })
        return None


def _load_strategy() -> dict:
    with open(CONFIG_PATH / "wheel_strategy.yaml", "r") as f:
        return yaml.safe_load(f)


@dataclass
class StockAnalysis:
    """Structured output from LLM stock analysis."""
    ticker: str
    win_probability: float       # Estimated prob trade hits target (0-1)
    confidence: float            # Self-assessed 0-1
    bullish_factors: list[str]
    bearish_factors: list[str]
    reasoning: str
    suggested_action: str        # "buy", "wait", "skip"
    raw_response: str
    provider: str = "unknown"    # "deepseek" | "claude"
    model: str = "unknown"
    cached: bool = False


# ── Rate Limiter ─────────────────────────────────────────────────

_last_call = 0.0


def _rate_limit(min_interval: float = 2.0):
    global _last_call
    now = time.time()
    elapsed = now - _last_call
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_call = time.time()


# ── Cache ────────────────────────────────────────────────────────

def _cache_key(ticker: str, setup_hash: str, provider: str, model: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
    raw = f"stock_llm:{provider}:{model}:{ticker}:{setup_hash}:{today}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_get(key: str, ttl_minutes: int = 60):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        # Check TTL (file mtime vs now)
        mtime = path.stat().st_mtime
        if time.time() - mtime > ttl_minutes * 60:
            return None
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_put(key: str, data: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


# ── Prompt Builder ───────────────────────────────────────────────

def _build_prompt(
    ticker: str,
    current_price: float,
    target_price: float,
    stop_loss: float,
    composite_score: int,
    pattern: str | None,
    momentum_score: int,
    kronos_direction: str | None,
    kronos_expected_return: float | None,
    news_sentiment: float | None,
    recent_headlines: list[str] | None,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) pair."""
    headlines_str = "\n".join(f"- {h}" for h in (recent_headlines or [])[:5]) or "(none)"

    # Past outcomes & reflections — the learning loop. Without this, every
    # decision is made cold; with it, the LLM sees how prior setups on
    # this ticker actually played out and any lessons recorded by the
    # reflection pass after they closed.
    try:
        from lib.memory_palace import get_past_outcomes
        past_context = get_past_outcomes(ticker)
    except Exception:
        past_context = ""
    past_block = (
        f"\n\nPRIOR OUTCOMES & LESSONS:\n{past_context}\n"
        if past_context else ""
    )

    system_prompt = (
        "You are a disciplined quantitative stock analyst advising a retail trader using "
        "The Wheel Strategy (CSPs, covered calls, and occasional swing stock trades). "
        "You synthesize technical signals, AI forecasts, and news sentiment into a calibrated "
        "win-probability estimate and a clear action recommendation. "
        "Be skeptical. Penalize setups with bearish AI forecasts, negative news momentum, or "
        "thin/low-conviction patterns. Reward confluence across signals. "
        "You MUST respond with valid JSON only — no markdown, no commentary outside JSON."
    )

    user_prompt = f"""Analyze this stock trade setup and estimate the probability it hits target before stop.

TICKER: {ticker}
PRICE: ${current_price:.2f}
TARGET: ${target_price:.2f} ({(target_price / current_price - 1) * 100:+.1f}%)
STOP:   ${stop_loss:.2f} ({(stop_loss / current_price - 1) * 100:+.1f}%)
R/R:    {((target_price - current_price) / max(1e-6, current_price - stop_loss)):.2f}:1

TECHNICAL SIGNALS:
- Composite score: {composite_score}/13
- Candlestick pattern: {pattern or "none"}
- Momentum score: {momentum_score}/4
- Kronos AI forecast: {kronos_direction or "n/a"} ({f"{kronos_expected_return:+.1%}" if kronos_expected_return is not None else "n/a"})
- News sentiment: {f"{news_sentiment:.2f}" if news_sentiment is not None else "n/a"} (0=bearish, 0.5=neutral, 1=bullish)

RECENT HEADLINES:
{headlines_str}{past_block}

## Step 0: Knowledge expansion (think before you score)
Before producing the JSON, briefly enumerate to yourself:
- What you actually know about {ticker} as a business (sector, recent earnings cadence, well-known catalysts).
- What macro/sector regime applies right now (rate environment, sector rotation, recent index trend).
- What you DON'T know — gaps in the data above (e.g., is there a binary catalyst inside the trade window? is the news sentiment driven by a single old article?).

CRITICAL anti-fabrication rule: if you don't actually know a fact about this ticker (recent guidance, analyst targets, a specific catalyst date), DO NOT make one up. Either omit it or explicitly flag it as uncertain in `bearish_factors` (e.g., "unknown earnings proximity"). Hallucinated specifics that drive a high `win_probability` are worse than admitting ignorance.

## Output
Return your analysis as strict JSON (no markdown, no prose outside JSON):
{{
  "win_probability": <float between 0 and 1>,
  "confidence": <float between 0 and 1 — how sure are you in your estimate>,
  "bullish_factors": [<short string>, ...],
  "bearish_factors": [<short string>, ...],
  "reasoning": "<2-3 sentence summary of your thinking>",
  "suggested_action": "<buy|wait|skip>"
}}

Guidance:
- If Kronos forecast is negative and news sentiment < 0.4 → suggest "skip".
- If R/R > 2:1 and composite >= 9/13 and no major red flags → suggest "buy".
- Otherwise → "wait".
- Put confidence LOW (< 0.4) if signals conflict, data is thin, or your Step 0 surfaced material unknowns."""

    return system_prompt, user_prompt


# ── Provider: DeepSeek ───────────────────────────────────────────

def _call_deepseek(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    ticker: str,
) -> str | None:
    """Call DeepSeek chat completion API. Returns raw text or None on failure."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        log_event("llm_analyst", "no_api_key", {"ticker": ticker, "provider": "deepseek"})
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,   # Lower = more deterministic for analysis
        "stream": False,
    }

    # DeepSeek V4 chat models support structured JSON output natively.
    # Match both the legacy alias `deepseek-chat` and the V4 family
    # (`deepseek-v4-flash`, `deepseek-v4-pro`).
    if model in ("deepseek-chat", "deepseek-v4-flash", "deepseek-v4-pro"):
        payload["response_format"] = {"type": "json_object"}

    try:
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=DEEPSEEK_TIMEOUT,
        )
        if resp.status_code != 200:
            log_event("llm_analyst", "api_http_error", {
                "ticker": ticker,
                "provider": "deepseek",
                "status": resp.status_code,
                "body": resp.text[:300],
            })
            return None
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            log_event("llm_analyst", "no_choices", {"ticker": ticker, "provider": "deepseek"})
            return None
        return choices[0].get("message", {}).get("content", "") or None
    except requests.exceptions.Timeout:
        log_event("llm_analyst", "timeout", {"ticker": ticker, "provider": "deepseek"})
        return None
    except Exception as e:
        log_event("llm_analyst", "api_failed", {
            "ticker": ticker,
            "provider": "deepseek",
            "error": str(e)[:200],
        })
        return None


# ── Provider: Claude ─────────────────────────────────────────────

def _call_claude(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    ticker: str,
) -> str | None:
    """Call Anthropic Claude API. Returns raw text or None on failure."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log_event("llm_analyst", "no_api_key", {"ticker": ticker, "provider": "claude"})
        return None

    try:
        from anthropic import Anthropic
        client = Anthropic()

        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return resp.content[0].text if resp.content else None
    except Exception as e:
        log_event("llm_analyst", "api_failed", {
            "ticker": ticker,
            "provider": "claude",
            "error": str(e)[:200],
        })
        return None


# ── Response Parser ──────────────────────────────────────────────

def _parse_response(raw: str, ticker: str, provider: str) -> dict | None:
    """Extract JSON object from LLM response. Returns dict or None."""
    if not raw:
        return None

    # DeepSeek-reasoner may emit reasoning content before the JSON — strip think tags if any
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)

    # Find the first JSON object (non-greedy won't work for nested braces — use greedy + brace matching)
    # Find first { ... } block by brace counting
    start = raw.find("{")
    if start == -1:
        log_event("llm_analyst", "parse_failed", {"ticker": ticker, "provider": provider, "raw": raw[:200]})
        return None

    depth = 0
    end = -1
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        log_event("llm_analyst", "no_closing_brace", {"ticker": ticker, "provider": provider, "raw": raw[:200]})
        return None

    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        log_event("llm_analyst", "json_decode_failed", {"ticker": ticker, "provider": provider, "raw": raw[start:end][:200]})
        return None


# ── Main API ─────────────────────────────────────────────────────

def analyze_stock_setup(
    ticker: str,
    current_price: float,
    target_price: float,
    stop_loss: float,
    composite_score: int,
    pattern: str | None = None,
    momentum_score: int = 0,
    kronos_direction: str | None = None,
    kronos_expected_return: float | None = None,
    news_sentiment: float | None = None,
    recent_headlines: list[str] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> StockAnalysis | None:
    """
    Ask the configured LLM to analyze a stock setup and return a win probability estimate.

    Provider/model resolution:
        1. Explicit args (provider=, model=)
        2. config/wheel_strategy.yaml `llm.provider` + `llm.model`
        3. Defaults: provider="deepseek", model="deepseek-chat"

    Returns None if:
        - API key for the chosen provider is not set
        - API call fails (credits, rate limit, timeout)
        - Response parsing fails

    Non-blocking: the caller can proceed without LLM analysis if this returns None.
    """
    strategy = _load_strategy()
    llm_cfg = strategy.get("llm", {}) or {}

    provider = (provider or llm_cfg.get("provider") or "deepseek").lower()
    if provider == "deepseek":
        model = model or llm_cfg.get("model") or "deepseek-v4-flash"
    elif provider == "claude":
        model = model or llm_cfg.get("claude_model") or llm_cfg.get("model") or "claude-haiku-4-5"
    else:
        log_event("llm_analyst", "unknown_provider", {"ticker": ticker, "provider": provider})
        return None

    max_tokens = int(llm_cfg.get("max_tokens", 1000))
    cache_ttl = int(llm_cfg.get("cache_ttl_minutes", 60))

    # Cache key based on setup characteristics + provider + model
    setup_sig = f"{composite_score}:{pattern}:{momentum_score}:{kronos_direction}:{kronos_expected_return}:{news_sentiment}"
    setup_hash = hashlib.md5(setup_sig.encode()).hexdigest()[:8]

    key = _cache_key(ticker, setup_hash, provider, model)
    cached = _cache_get(key, ttl_minutes=cache_ttl)
    if cached:
        try:
            cached_copy = dict(cached)
            cached_copy["cached"] = True
            # Strip any extra fields not in the dataclass
            allowed = set(StockAnalysis.__annotations__.keys())
            return StockAnalysis(**{k: v for k, v in cached_copy.items() if k in allowed})
        except Exception:
            pass  # Fall through to re-fetch

    system_prompt, user_prompt = _build_prompt(
        ticker=ticker,
        current_price=current_price,
        target_price=target_price,
        stop_loss=stop_loss,
        composite_score=composite_score,
        pattern=pattern,
        momentum_score=momentum_score,
        kronos_direction=kronos_direction,
        kronos_expected_return=kronos_expected_return,
        news_sentiment=news_sentiment,
        recent_headlines=recent_headlines,
    )

    _rate_limit(min_interval=2.0)

    # Preferred: instructor + pydantic (validated, auto-retries). Falls back
    # to legacy raw + brace-counting parse on any failure so existing behavior
    # is preserved.
    used_instructor = False
    data: dict | None = None
    raw: str | None = None

    if _use_instructor(llm_cfg):
        result = _call_with_instructor(
            system_prompt, user_prompt, provider, model, max_tokens,
            _StockAnalysisSchema, ticker,
        )
        if result is not None:
            data = result.model_dump()
            raw = json.dumps(data)
            used_instructor = True

    if data is None:
        if provider == "deepseek":
            raw = _call_deepseek(system_prompt, user_prompt, model, max_tokens, ticker)
        else:  # claude
            raw = _call_claude(system_prompt, user_prompt, model, max_tokens, ticker)
        if not raw:
            return None
        data = _parse_response(raw, ticker, provider)
        if data is None:
            return None

    # Validate and sanitize
    try:
        win_prob = float(data.get("win_probability", 0.5))
        win_prob = max(0.05, min(0.95, win_prob))
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        log_event("llm_analyst", "validation_failed", {"ticker": ticker, "data": str(data)[:200]})
        return None

    analysis = StockAnalysis(
        ticker=ticker,
        win_probability=round(win_prob, 4),
        confidence=round(confidence, 4),
        bullish_factors=[str(x)[:200] for x in (data.get("bullish_factors") or [])[:5]],
        bearish_factors=[str(x)[:200] for x in (data.get("bearish_factors") or [])[:5]],
        reasoning=str(data.get("reasoning", ""))[:500],
        suggested_action=str(data.get("suggested_action", "wait"))[:20].lower(),
        raw_response=(raw or "")[:2000],
        provider=provider + (":instructor" if used_instructor else ""),
        model=model,
        cached=False,
    )

    # Cache
    _cache_put(key, {
        "ticker": analysis.ticker,
        "win_probability": analysis.win_probability,
        "confidence": analysis.confidence,
        "bullish_factors": analysis.bullish_factors,
        "bearish_factors": analysis.bearish_factors,
        "reasoning": analysis.reasoning,
        "suggested_action": analysis.suggested_action,
        "raw_response": analysis.raw_response,
        "provider": analysis.provider,
        "model": analysis.model,
    })

    log_event("llm_analyst", "analyzed", {
        "ticker": ticker,
        "provider": provider,
        "model": model,
        "win_prob": analysis.win_probability,
        "action": analysis.suggested_action,
    })

    return analysis


# ── Options Setup Analysis (CSP / CC) ────────────────────────────

@dataclass
class OptionAnalysis:
    """Structured LLM assessment for a wheel option trade."""
    ticker: str
    trade_type: str              # "csp" | "cc"
    win_probability: float       # Prob the sold option expires worthless (our win)
    confidence: float            # Self-assessed 0-1
    bullish_factors: list[str]
    bearish_factors: list[str]
    reasoning: str
    suggested_action: str        # "sell" | "wait" | "skip"
    raw_response: str
    provider: str = "unknown"
    model: str = "unknown"
    cached: bool = False


def _build_option_prompt(
    ticker: str,
    trade_type: str,
    strike: float,
    premium: float,
    delta: float,
    dte: int,
    composite_score: int,
    zone_level: float,
    zone_touches: int,
    iv_rank: float,
    candlestick_pattern: str | None,
    annualized_return: float,
    cost_basis: float | None,
) -> tuple[str, str]:
    """Build (system, user) prompt pair for CSP/CC assessment."""
    is_csp = trade_type == "csp"

    direction = "bullish (want price ABOVE strike)" if is_csp else "bearish (want price BELOW strike)"
    zone_kind = "support" if is_csp else "resistance"
    leg = f"{strike}P" if is_csp else f"{strike}C"
    cb_str = f"Cost basis: ${cost_basis:.2f}" if (not is_csp and cost_basis) else ""

    system_prompt = (
        "You are a disciplined options analyst advising on The Wheel Strategy. "
        "Your job: assess whether selling this specific option is a good trade right now, "
        "given the technical setup and confluence. You never recommend buying options. "
        "You estimate the probability the sold option expires worthless (our win condition). "
        "Penalize thin premium, weak zone support, contradictory patterns, and near-earnings setups. "
        "Reward strong zones, ≥2 prior touches, confirming candlesticks, high IV rank. "
        "You MUST respond with valid JSON only — no markdown, no prose outside JSON."
    )

    user_prompt = f"""Analyze this wheel option trade and estimate the probability it expires worthless (win).

TICKER: {ticker}
TRADE: SELL-TO-OPEN {leg} exp in {dte} DTE for ${premium} credit
BIAS NEEDED: {direction}
{cb_str}

TECHNICAL SIGNALS:
- Composite score: {composite_score}/9 (trend + level + candlestick)
- {zone_kind.capitalize()} zone: ${zone_level:.2f} ({zone_touches} historical touches)
- Candlestick pattern: {candlestick_pattern or "none"}
- Delta: {delta:+.2f} (prob of assignment at current model)
- IV rank: {iv_rank:.0%} (higher = richer premium)
- Annualized return: {annualized_return:.1%}

Return your analysis as strict JSON:
{{
  "win_probability": <float 0-1 — prob option expires OTM = we keep full credit>,
  "confidence": <float 0-1>,
  "bullish_factors": [<short string>, ...],
  "bearish_factors": [<short string>, ...],
  "reasoning": "<2-3 sentences>",
  "suggested_action": "<sell|wait|skip>"
}}

Guidance:
- Composite ≥ 7/9 AND strong {zone_kind} AND IV rank ≥ 30% AND confirming pattern → "sell".
- Composite ≤ 5/9 OR zone broken OR opposing pattern → "skip".
- Otherwise → "wait".
- Confidence LOW (<0.4) if signals conflict or IV rank is weak."""

    return system_prompt, user_prompt


def analyze_option_setup(
    ticker: str,
    trade_type: str,
    strike: float,
    premium: float,
    delta: float,
    dte: int,
    composite_score: int,
    zone_level: float,
    zone_touches: int,
    iv_rank: float,
    annualized_return: float,
    candlestick_pattern: str | None = None,
    cost_basis: float | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> OptionAnalysis | None:
    """Ask the configured LLM to assess a CSP or CC candidate.

    Returns None if: LLM disabled, no API key, call fails, or parse fails.
    Non-blocking: caller should proceed when this returns None.
    """
    strategy = _load_strategy()
    llm_cfg = strategy.get("llm", {}) or {}
    if not llm_cfg.get("enabled", False):
        return None

    provider = (provider or llm_cfg.get("provider") or "deepseek").lower()
    if provider == "deepseek":
        model = model or llm_cfg.get("model") or "deepseek-v4-flash"
    elif provider == "claude":
        model = model or llm_cfg.get("claude_model") or llm_cfg.get("model") or "claude-haiku-4-5"
    else:
        log_event("llm_analyst", "unknown_provider", {"ticker": ticker, "provider": provider})
        return None

    max_tokens = int(llm_cfg.get("max_tokens", 1000))
    cache_ttl = int(llm_cfg.get("cache_ttl_minutes", 60))

    setup_sig = f"{trade_type}:{strike}:{dte}:{composite_score}:{candlestick_pattern}:{zone_level}:{iv_rank:.2f}"
    setup_hash = hashlib.md5(setup_sig.encode()).hexdigest()[:8]
    key = _cache_key(ticker, "opt_" + setup_hash, provider, model)

    cached = _cache_get(key, ttl_minutes=cache_ttl)
    if cached:
        try:
            cached_copy = dict(cached)
            cached_copy["cached"] = True
            allowed = set(OptionAnalysis.__annotations__.keys())
            return OptionAnalysis(**{k: v for k, v in cached_copy.items() if k in allowed})
        except Exception:
            pass

    system_prompt, user_prompt = _build_option_prompt(
        ticker=ticker,
        trade_type=trade_type,
        strike=strike,
        premium=premium,
        delta=delta,
        dte=dte,
        composite_score=composite_score,
        zone_level=zone_level,
        zone_touches=zone_touches,
        iv_rank=iv_rank,
        candlestick_pattern=candlestick_pattern,
        annualized_return=annualized_return,
        cost_basis=cost_basis,
    )

    _rate_limit(min_interval=2.0)

    # Preferred path: instructor + pydantic (validated, auto-retries on bad JSON).
    # Fall back to legacy raw-text + brace-counting parse if instructor is off
    # or the structured call itself fails.
    used_instructor = False
    data: dict | None = None
    raw: str | None = None

    if _use_instructor(llm_cfg):
        result = _call_with_instructor(
            system_prompt, user_prompt, provider, model, max_tokens,
            _OptionAnalysisSchema, ticker,
        )
        if result is not None:
            data = result.model_dump()
            raw = json.dumps(data)  # keep raw_response populated for audit parity
            used_instructor = True

    if data is None:
        if provider == "deepseek":
            raw = _call_deepseek(system_prompt, user_prompt, model, max_tokens, ticker)
        else:
            raw = _call_claude(system_prompt, user_prompt, model, max_tokens, ticker)
        if not raw:
            return None
        data = _parse_response(raw, ticker, provider)
        if data is None:
            return None

    try:
        win_prob = max(0.05, min(0.95, float(data.get("win_probability", 0.5))))
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        log_event("llm_analyst", "option_validation_failed", {"ticker": ticker, "data": str(data)[:200]})
        return None

    analysis = OptionAnalysis(
        ticker=ticker,
        trade_type=trade_type,
        win_probability=round(win_prob, 4),
        confidence=round(confidence, 4),
        bullish_factors=[str(x)[:200] for x in (data.get("bullish_factors") or [])[:5]],
        bearish_factors=[str(x)[:200] for x in (data.get("bearish_factors") or [])[:5]],
        reasoning=str(data.get("reasoning", ""))[:500],
        suggested_action=str(data.get("suggested_action", "wait"))[:20].lower(),
        raw_response=(raw or "")[:2000],
        provider=provider + (":instructor" if used_instructor else ""),
        model=model,
        cached=False,
    )

    _cache_put(key, {
        "ticker": analysis.ticker,
        "trade_type": analysis.trade_type,
        "win_probability": analysis.win_probability,
        "confidence": analysis.confidence,
        "bullish_factors": analysis.bullish_factors,
        "bearish_factors": analysis.bearish_factors,
        "reasoning": analysis.reasoning,
        "suggested_action": analysis.suggested_action,
        "raw_response": analysis.raw_response,
        "provider": analysis.provider,
        "model": analysis.model,
    })

    log_event("llm_analyst", "option_analyzed", {
        "ticker": ticker,
        "trade_type": trade_type,
        "provider": provider,
        "win_prob": analysis.win_probability,
        "action": analysis.suggested_action,
    })

    return analysis


# ── Convenience: is the LLM configured and reachable? ────────────

def llm_status() -> dict:
    """Return a summary of what providers are configured and have keys set."""
    strategy = _load_strategy()
    llm_cfg = strategy.get("llm", {}) or {}
    provider = (llm_cfg.get("provider") or "deepseek").lower()

    return {
        "enabled": bool(llm_cfg.get("enabled", False)),
        "provider": provider,
        "model": llm_cfg.get("model") if provider == "deepseek" else (llm_cfg.get("claude_model") or llm_cfg.get("model")),
        "deepseek_key_set": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }
