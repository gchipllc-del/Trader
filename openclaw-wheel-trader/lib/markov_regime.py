"""Markov Regime Model — quantitative state-transition forecasting.

Adapted from the "hedge fund method" framework (Rowan's row-and-chain
quant approach popularized May 2026). Replaces vibes/trendline reading
with a 3-state Markov chain whose transition probabilities are learned
from historical state sequences.

The 10-element pipeline:

  1. STATE — classify a 20-day window by cumulative return:
       bull   if  return >= +5%
       bear   if  return <= -5%
       else   sideways
  2. LABEL HISTORY — assign a state to every historical day going back
     to bar 20 (the first day where a 20-day return is computable).
  3. MARKOV PROPERTY — tomorrow's distribution depends ONLY on today's
     state, not the path that led there.
  4. TRANSITION MATRIX — count every (today_state, tomorrow_state) pair
     in the history; normalize each row to probabilities summing to 1.
  5. PERSISTENCE — diagonal of the matrix (P(bull|bull), P(bear|bear),
     P(side|side)) — the "stickiness" of each regime.
  6. MULTI-DAY FORECAST — M^N gives the N-day-ahead distribution from
     any starting state.
  7. STATIONARY DISTRIBUTION — limit of M^N as N→∞; the long-run mix.
  8. SIGNAL — P(bull_horizon) - P(bear_horizon). Sign = direction,
     magnitude = position size.
  9. WALK-FORWARD VALIDATION — re-build the matrix at every bar using
     only data available up to that bar (no look-ahead bias).
 10. HMM CROSS-CHECK — Hidden-Markov pattern discovery (without the
     subjective 5% labels) cross-validates the rule-based labels.

This module is pure-Python (no pandas / numpy) so it runs cleanly under
the current env's NumPy 2 / torch incompatibility. Math is exact via
list-of-list 3x3 matrix operations.

Public API:
    classify_state(return_pct) -> "bear" | "sideways" | "bull"
    label_states(prices, window=20) -> list[str]
    build_transition_matrix(states) -> dict[from][to] = prob
    matrix_power(matrix, n) -> dict[from][to]
    forecast(matrix, today_state, horizon=1) -> {state: prob}
    signal_strength(matrix, today_state, horizon=1) -> float in [-1, 1]
    stationary_distribution(matrix, iterations=200) -> {state: prob}
    walk_forward_backtest(prices, window=20, train_window=252)
        -> list[dict] per-bar with state, signal, forward_return
    markov_summary(ticker, lookback_days=730, horizon=1) -> dict
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Literal

STATES: tuple[str, str, str] = ("bear", "sideways", "bull")

# Default thresholds — match the video's prescription. Tuneable in
# config/wheel_strategy.yaml under markov.* once the operator wants to
# experiment per-asset (small caps tend to need wider bands; ETFs
# narrower).
BULL_THRESHOLD = 0.05          # +5% over the window → bull
BEAR_THRESHOLD = -0.05         # -5% or worse → bear
DEFAULT_WINDOW = 20            # bars looking back for the state calc
DEFAULT_TRAIN_WINDOW = 252     # ~1 trading year of history for the matrix

State = Literal["bear", "sideways", "bull"]


# ───────────────────────── 1. State classification ─────────────────────

def classify_state(return_pct: float) -> State:
    """Map a cumulative return to one of the three states.

    return_pct is a decimal fraction (e.g. 0.07 = +7%). Thresholds are
    fixed BULL_THRESHOLD / BEAR_THRESHOLD; sideways is everything between.
    """
    if return_pct >= BULL_THRESHOLD:
        return "bull"
    if return_pct <= BEAR_THRESHOLD:
        return "bear"
    return "sideways"


# ───────────────────────── 2. Label history ────────────────────────────

def label_states(prices: list[float], window: int = DEFAULT_WINDOW) -> list[str]:
    """For each bar from index ``window`` onward, label its state from
    the cumulative return over the prior ``window`` bars.

    Returns a list of length len(prices) - window. The first label
    corresponds to bar index ``window`` in the original prices list.
    """
    if window < 1 or len(prices) <= window:
        return []
    labels: list[str] = []
    for i in range(window, len(prices)):
        p0 = prices[i - window]
        p1 = prices[i]
        if p0 <= 0:
            labels.append("sideways")
            continue
        ret = (p1 - p0) / p0
        labels.append(classify_state(ret))
    return labels


# ───────────────────────── 3-4. Transition matrix ──────────────────────

def build_transition_matrix(
    states: list[str], smoothing: float = 1.0
) -> dict[str, dict[str, float]]:
    """Build the 3x3 transition probability matrix from a state sequence.

    Uses Laplace (additive) smoothing with default α=1.0 so that no
    transition pair is ever zero — important because rare transitions
    (bear→bull direct) would otherwise produce undefined forecasts. With
    smoothing=0 you get pure empirical probabilities.

    Returns matrix[from_state][to_state] = probability, with each row
    summing to 1.0 exactly.
    """
    counts = {s: {t: float(smoothing) for t in STATES} for s in STATES}
    for i in range(len(states) - 1):
        if states[i] in STATES and states[i + 1] in STATES:
            counts[states[i]][states[i + 1]] += 1.0
    matrix: dict[str, dict[str, float]] = {}
    for s in STATES:
        row_total = sum(counts[s].values())
        if row_total <= 0:
            # No data for this row — uniform prior
            matrix[s] = {t: 1.0 / len(STATES) for t in STATES}
        else:
            matrix[s] = {t: counts[s][t] / row_total for t in STATES}
    return matrix


# ───────────────────────── 5. Persistence ──────────────────────────────

def stickiness(matrix: dict) -> dict[str, float]:
    """Diagonal of the matrix — P(state stays the same tomorrow).
    Higher = stickier regime. Bull and bear typically score 0.7-0.9 on
    daily bars for liquid assets; sideways lower (~0.6).
    """
    return {s: matrix[s][s] for s in STATES}


# ───────────────────────── 6. Matrix exponentiation ────────────────────

def _matrix_to_list(matrix: dict) -> list[list[float]]:
    return [[matrix[s][t] for t in STATES] for s in STATES]


def _list_to_matrix(M: list[list[float]]) -> dict[str, dict[str, float]]:
    return {STATES[i]: {STATES[j]: M[i][j] for j in range(3)} for i in range(3)}


def _matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    n = len(A)
    return [
        [sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]


def matrix_power(matrix: dict, n: int) -> dict[str, dict[str, float]]:
    """Raise a transition matrix to the integer power ``n`` (via binary
    exponentiation for efficiency on large horizons).

    M^1 = M (one-day forecast).
    M^N gives the N-day-ahead distribution: row = today_state,
    column = state N days ahead.
    """
    if n < 1:
        raise ValueError(f"matrix_power needs n >= 1, got {n}")
    base = _matrix_to_list(matrix)
    # Initialize result to identity
    R = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    while n > 0:
        if n & 1:
            R = _matmul(R, base)
        n >>= 1
        if n:
            base = _matmul(base, base)
    return _list_to_matrix(R)


# ───────────────────────── 6 (cont.). Forecasting ──────────────────────

def forecast(
    matrix: dict, today_state: str, horizon: int = 1
) -> dict[str, float]:
    """Return P(state in `horizon` days | today_state) as a dict."""
    if today_state not in STATES:
        raise ValueError(f"unknown state: {today_state!r}")
    Mh = matrix_power(matrix, horizon)
    return dict(Mh[today_state])


# ───────────────────────── 7. Stationary distribution ──────────────────

def stationary_distribution(
    matrix: dict, iterations: int = 200, tol: float = 1e-9
) -> dict[str, float]:
    """Power-iterate from uniform to find π such that π·M = π.

    Converges fast (typically <50 iterations) for ergodic chains, which
    a smoothed transition matrix always is.
    """
    pi = [1.0 / len(STATES)] * len(STATES)
    M = _matrix_to_list(matrix)
    for _ in range(iterations):
        new = [sum(pi[i] * M[i][j] for i in range(3)) for j in range(3)]
        # Normalize (defensive — should already sum to 1)
        s = sum(new) or 1.0
        new = [x / s for x in new]
        delta = max(abs(new[i] - pi[i]) for i in range(3))
        pi = new
        if delta < tol:
            break
    return {STATES[i]: pi[i] for i in range(3)}


# ───────────────────────── 8. Signal generation ────────────────────────

def signal_strength(
    matrix: dict, today_state: str, horizon: int = 1
) -> float:
    """P(bull at horizon) - P(bear at horizon). Range [-1, +1].

    Positive → bias long, sized by magnitude.
    Negative → bias short, sized by |magnitude|.
    Near zero → no clear signal; stand aside.
    """
    f = forecast(matrix, today_state, horizon=horizon)
    return f["bull"] - f["bear"]


def signal_to_position_size(
    signal: float, max_pct: float = 0.20, deadzone: float = 0.10
) -> float:
    """Translate a Markov signal in [-1, +1] into a position fraction in
    [-max_pct, +max_pct]. Anything inside |signal| < deadzone returns 0.

    The video author leaves sizing to each fund; this implementation
    does linear scaling above the deadzone:
      |signal|=0.10 → 0.0   (boundary)
      |signal|=0.55 → max_pct/2
      |signal|=1.00 → max_pct
    """
    if abs(signal) < deadzone:
        return 0.0
    sign = 1.0 if signal > 0 else -1.0
    scaled = (abs(signal) - deadzone) / (1.0 - deadzone)
    return sign * max_pct * scaled


# ───────────────────────── 9. Walk-forward ─────────────────────────────

def walk_forward_backtest(
    prices: list[float],
    window: int = DEFAULT_WINDOW,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    horizon: int = 1,
) -> list[dict]:
    """At each bar from `window+train_window` onward:
      1. Build a matrix using ONLY states known up to that bar
         (no look-ahead). Specifically the last `train_window`
         transitions before today.
      2. Classify today's state.
      3. Compute the horizon-day signal.
      4. Record forward return (actual close after `horizon` bars vs
         today's close) — for outcome attribution.

    Returns a list of per-bar dicts with bar_idx, state, signal,
    forecast distribution, and forward_return (None at the right edge
    where forward data isn't yet available).
    """
    if window < 1 or train_window < 3:
        raise ValueError("window>=1, train_window>=3")
    all_states = label_states(prices, window=window)
    if len(all_states) < train_window + horizon + 1:
        return []
    out: list[dict] = []
    # The state at index k in all_states corresponds to bar (window + k)
    # in the original prices list.
    for k in range(train_window, len(all_states)):
        train_slice = all_states[k - train_window:k]
        today_state = all_states[k]
        matrix = build_transition_matrix(train_slice)
        sig = signal_strength(matrix, today_state, horizon=horizon)
        fc = forecast(matrix, today_state, horizon=horizon)
        bar_idx = window + k
        fwd_return: float | None
        if bar_idx + horizon < len(prices):
            p0 = prices[bar_idx]
            p1 = prices[bar_idx + horizon]
            fwd_return = (p1 - p0) / p0 if p0 > 0 else None
        else:
            fwd_return = None
        out.append({
            "bar_idx": bar_idx,
            "state": today_state,
            "signal": round(sig, 4),
            "p_bull": round(fc["bull"], 4),
            "p_side": round(fc["sideways"], 4),
            "p_bear": round(fc["bear"], 4),
            "fwd_return": round(fwd_return, 6) if fwd_return is not None else None,
        })
    return out


# ───────────────────────── 10. HMM cross-check (lightweight) ───────────
# A full Baum-Welch HMM needs scipy; we ship a pragmatic substitute:
# K-state clustering by (rolling-return, rolling-vol) without the
# subjective +/-5% labels, then map clusters to {bear, side, bull} by
# their mean return. Operator can compare against the rule-based labels.

def _rolling_return(prices: list[float], i: int, window: int) -> float:
    p0 = prices[i - window]
    return (prices[i] - p0) / p0 if p0 > 0 else 0.0


def _rolling_vol(prices: list[float], i: int, window: int) -> float:
    rets = []
    for k in range(i - window + 1, i + 1):
        if k <= 0:
            continue
        p0, p1 = prices[k - 1], prices[k]
        if p0 > 0:
            rets.append((p1 - p0) / p0)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


def hmm_lite_labels(
    prices: list[float],
    window: int = DEFAULT_WINDOW,
    iterations: int = 25,
) -> list[str]:
    """Label-free state discovery.

    Cluster each bar's (rolling_return, rolling_vol) into 3 regimes using
    K-means, then map clusters by their mean return to bull (highest),
    sideways (middle), bear (lowest). This isn't a true Baum-Welch HMM
    — it's a lightweight stand-in that doesn't require numpy/scipy and
    captures the same idea: "let the data reveal the regimes."

    Output length matches label_states() so the two can be compared bar
    for bar to score agreement (the video's "green-light" check).
    """
    n = len(prices)
    if n <= window + 2:
        return []
    pts: list[tuple[float, float]] = []
    for i in range(window, n):
        pts.append((_rolling_return(prices, i, window),
                    _rolling_vol(prices, i, window)))

    # Init centroids by spreading across the return range
    rets = sorted(p[0] for p in pts)
    centroids = [
        (rets[len(rets) // 6], 0.0),     # low-return seed
        (rets[len(rets) // 2], 0.0),     # mid-return seed
        (rets[5 * len(rets) // 6], 0.0), # high-return seed
    ]

    def _dist(a: tuple, b: tuple) -> float:
        # Equal-weight return and vol after scaling vol up so it isn't
        # dwarfed by raw return magnitude.
        return (a[0] - b[0]) ** 2 + ((a[1] - b[1]) * 20) ** 2

    cluster = [0] * len(pts)
    for _ in range(iterations):
        changed = False
        # Assign
        for idx, p in enumerate(pts):
            best = min(range(3), key=lambda c: _dist(p, centroids[c]))
            if cluster[idx] != best:
                cluster[idx] = best
                changed = True
        # Update
        new_centroids = []
        for c in range(3):
            members = [pts[i] for i in range(len(pts)) if cluster[i] == c]
            if not members:
                new_centroids.append(centroids[c])
                continue
            mr = sum(m[0] for m in members) / len(members)
            mv = sum(m[1] for m in members) / len(members)
            new_centroids.append((mr, mv))
        centroids = new_centroids
        if not changed:
            break

    # Map clusters to states by ascending mean return
    order = sorted(range(3), key=lambda c: centroids[c][0])
    cluster_to_state = {order[0]: "bear", order[1]: "sideways", order[2]: "bull"}
    return [cluster_to_state[c] for c in cluster]


def agreement_score(
    rule_labels: list[str], hmm_labels: list[str]
) -> float:
    """Fraction of bars where rule-based and HMM-lite labels agree.
    The video calls this the "green-light" overlap — quants take the
    signal more seriously when both label streams confirm.
    """
    if not rule_labels or not hmm_labels:
        return 0.0
    n = min(len(rule_labels), len(hmm_labels))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if rule_labels[i] == hmm_labels[i])
    return matches / n


# ───────────────────────── Convenience: ticker pipeline ────────────────

def _fetch_alpaca_daily_closes(
    ticker: str,
    lookback_days: int = 730,
) -> list[float]:
    """Pure-Python daily close fetch — bypasses pandas to dodge the
    NumPy 2 / pandas issue. Uses the same Alpaca creds as alpaca_client.

    Returns oldest→newest close prices. Empty list on failure.
    """
    import os
    import requests
    api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not secret:
        # Try the .env file the running scripts source
        env_path = (
            __import__("pathlib").Path(__file__).resolve().parent.parent / ".env"
        )
        if env_path.exists():
            for raw in env_path.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k in ("ALPACA_API_KEY", "APCA_API_KEY_ID") and not api_key:
                    api_key = v
                elif k in ("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY") and not secret:
                    secret = v
    if not api_key or not secret:
        raise RuntimeError("No Alpaca credentials found in env or .env")

    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=lookback_days)
    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
    closes: list[float] = []
    page_token = None
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret,
    }
    while True:
        params = {
            "timeframe": "1Day",
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 10000,
            "adjustment": "all",
        }
        if page_token:
            params["page_token"] = page_token
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        bars = data.get("bars", []) or []
        for b in bars:
            try:
                closes.append(float(b["c"]))
            except (KeyError, TypeError, ValueError):
                continue
        page_token = data.get("next_page_token")
        if not page_token:
            break
    return closes


def markov_summary(
    ticker: str,
    lookback_days: int = 730,
    horizon: int = 1,
    window: int = DEFAULT_WINDOW,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    include_hmm: bool = True,
) -> dict:
    """End-to-end pipeline for one ticker. Fetches prices, builds
    matrix, computes signal, runs walk-forward backtest, and (optionally)
    cross-checks against the HMM-lite labels.

    The output dict is designed to feed both the CLI report and the
    dashboard panel with zero further computation.
    """
    closes = _fetch_alpaca_daily_closes(ticker, lookback_days=lookback_days)
    if len(closes) < window + 30:
        return {
            "ticker": ticker,
            "error": f"insufficient history ({len(closes)} bars; need ≥ {window + 30})",
        }

    rule_labels = label_states(closes, window=window)
    matrix = build_transition_matrix(rule_labels)
    today_state = rule_labels[-1]
    today_return = (closes[-1] - closes[-1 - window]) / closes[-1 - window]
    fc = forecast(matrix, today_state, horizon=horizon)
    sig = signal_strength(matrix, today_state, horizon=horizon)
    stick = stickiness(matrix)
    stat_dist = stationary_distribution(matrix)
    position_pct = signal_to_position_size(sig)

    # Walk-forward backtest hit rate
    wf = walk_forward_backtest(
        closes, window=window, train_window=train_window, horizon=horizon
    )
    if wf:
        correct = 0
        sized = 0
        gross_pnl = 0.0
        for row in wf:
            fr = row["fwd_return"]
            if fr is None:
                continue
            sized += 1
            # Direction match: positive signal & positive return = correct
            sign_sig = 1 if row["signal"] > 0 else (-1 if row["signal"] < 0 else 0)
            sign_fr = 1 if fr > 0 else (-1 if fr < 0 else 0)
            if sign_sig != 0 and sign_sig == sign_fr:
                correct += 1
            # Simple position-fraction × return PnL contribution
            pos = signal_to_position_size(row["signal"])
            gross_pnl += pos * fr
        wf_stats = {
            "bars": sized,
            "directional_accuracy": round(correct / sized, 4) if sized else None,
            "cumulative_pnl_per_unit_bankroll": round(gross_pnl, 4),
        }
    else:
        wf_stats = {"bars": 0, "directional_accuracy": None,
                    "cumulative_pnl_per_unit_bankroll": 0.0}

    result = {
        "ticker": ticker,
        "lookback_days": lookback_days,
        "horizon": horizon,
        "window": window,
        "bars_used": len(closes),
        "today_window_return": round(today_return, 4),
        "today_state": today_state,
        "matrix": matrix,
        "stickiness": stick,
        "forecast": fc,
        "signal": round(sig, 4),
        "position_size_pct": round(position_pct, 4),
        "stationary": stat_dist,
        "walk_forward": wf_stats,
    }

    if include_hmm:
        hmm_labels = hmm_lite_labels(closes, window=window)
        result["hmm_lite_today"] = hmm_labels[-1] if hmm_labels else None
        result["hmm_agreement"] = round(
            agreement_score(rule_labels, hmm_labels), 4
        )
        result["green_light"] = (
            result["hmm_lite_today"] == today_state
            and abs(sig) >= 0.15
        )

    return result


def render_summary(result: dict) -> str:
    """Human-readable single-ticker report."""
    if "error" in result:
        return f"{result['ticker']}: ERROR — {result['error']}"
    lines = []
    lines.append("=" * 70)
    lines.append(f"MARKOV REGIME — {result['ticker']}  "
                 f"(window={result['window']}d, horizon={result['horizon']}d)")
    lines.append("=" * 70)
    lines.append(f"Bars used: {result['bars_used']}  "
                 f"({result['lookback_days']}-day lookback)")
    lines.append(f"Today's {result['window']}d return: "
                 f"{result['today_window_return']:+.2%}  → "
                 f"state = {result['today_state'].upper()}")
    lines.append("")
    lines.append("Transition matrix (P(tomorrow | today)):")
    lines.append(f"            {'bear':>10} {'sideways':>10} {'bull':>10}")
    for s in STATES:
        row = result["matrix"][s]
        lines.append(
            f"  {s:>8}  {row['bear']:>10.2%} {row['sideways']:>10.2%} "
            f"{row['bull']:>10.2%}"
        )
    lines.append("")
    lines.append("Stickiness (diagonal):")
    for s, v in result["stickiness"].items():
        lines.append(f"  {s:<10} {v:.2%}")
    lines.append("")
    lines.append(f"Horizon-{result['horizon']} forecast from today "
                 f"({result['today_state']}):")
    for s, p in result["forecast"].items():
        lines.append(f"  P({s}) = {p:.2%}")
    lines.append("")
    sig = result["signal"]
    direction = "LONG" if sig > 0 else ("SHORT" if sig < 0 else "FLAT")
    lines.append(f"SIGNAL: {sig:+.4f}  → {direction}  "
                 f"(suggested position {result['position_size_pct']:+.2%} of bankroll)")
    lines.append("")
    lines.append("Stationary (long-run mix):")
    for s, p in result["stationary"].items():
        lines.append(f"  π({s}) = {p:.2%}")
    lines.append("")
    wf = result["walk_forward"]
    da = wf["directional_accuracy"]
    lines.append(f"Walk-forward over {wf['bars']} bars: "
                 f"directional acc = "
                 f"{(da*100 if da is not None else 0):.1f}% "
                 f"| cumulative PnL/unit = {wf['cumulative_pnl_per_unit_bankroll']:+.4f}")
    if "hmm_agreement" in result:
        lines.append(f"HMM-lite today: {result.get('hmm_lite_today')}  "
                     f"(agrees rule-based on {result['hmm_agreement']:.1%} of history)")
        if result.get("green_light"):
            lines.append("GREEN LIGHT ✓ — rule + HMM concur AND |signal| ≥ 0.15")
        else:
            lines.append("no green light — either labels disagree or signal weak")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "STATES", "BULL_THRESHOLD", "BEAR_THRESHOLD",
    "DEFAULT_WINDOW", "DEFAULT_TRAIN_WINDOW",
    "classify_state", "label_states", "build_transition_matrix",
    "stickiness", "matrix_power", "forecast", "signal_strength",
    "signal_to_position_size", "stationary_distribution",
    "walk_forward_backtest", "hmm_lite_labels", "agreement_score",
    "markov_summary", "render_summary",
]


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    result = markov_summary(ticker, horizon=horizon)
    print(render_summary(result))
