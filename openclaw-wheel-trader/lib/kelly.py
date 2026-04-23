"""
Kelly Criterion — optimal position sizing for stocks.

Adapted from polybot's binary-outcome Kelly for stock trading.
For stocks, Kelly uses:
    - win_prob: probability the trade hits target (from composite score + Kronos)
    - reward: % gain if target hit (target_price / entry_price - 1)
    - risk: % loss if stop hit (1 - stop_loss / entry_price)

Kelly fraction: f = (p*reward - q*risk) / (reward*risk)
    where p = win_prob, q = 1-p

Full Kelly is too aggressive — always use fractional Kelly (0.25 to 0.50).
"""

from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"


def _load_strategy() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def composite_to_win_prob(composite_score: int, max_score: int = 13) -> float:
    """
    Map composite score (0-13) to estimated win probability.

    Empirically (will tune with calibration data):
    - 0-4:  35% win rate (weak setups)
    - 5-7:  50% win rate (mediocre)
    - 8-10: 62% win rate (good)
    - 11-13: 72% win rate (excellent)

    Args:
        composite_score: 0-13 composite (trend+level+signal+momentum)
        max_score: Maximum possible score (13 for current system)

    Returns:
        Estimated win probability (0.35 to 0.75).
    """
    if max_score <= 0:
        return 0.5

    normalized = composite_score / max_score  # 0.0 to 1.0

    # Linear interpolation from 0.35 at score 0 to 0.75 at max
    # This is empirical — Hermes will tune based on actual calibration data
    return 0.35 + (normalized * 0.40)


def kelly_fraction_stock(
    win_prob: float,
    reward_pct: float,
    risk_pct: float,
) -> float:
    """
    Calculate full Kelly fraction for a stock trade.

    For a stock where:
      - Win probability: p
      - Reward if win: reward_pct (e.g., 0.10 for 10% gain to target)
      - Loss if stop hit: risk_pct (e.g., 0.035 for 3.5% loss to stop)

    Kelly = (p*b - q) / b
    where b = reward_pct / risk_pct (reward-to-risk ratio)

    Args:
        win_prob: Estimated probability trade hits target (0.0-1.0)
        reward_pct: Expected gain if target hits (e.g., 0.10 = 10%)
        risk_pct: Expected loss if stop hits (e.g., 0.035 = 3.5%)

    Returns:
        Full Kelly fraction. Can be negative (don't trade) or >1 (very strong).
    """
    if risk_pct <= 0 or win_prob <= 0 or win_prob >= 1:
        return 0.0

    b = reward_pct / risk_pct  # reward-to-risk ratio
    p = win_prob
    q = 1.0 - p

    f = (p * b - q) / b
    return f


def fractional_kelly_stock(
    win_prob: float,
    reward_pct: float,
    risk_pct: float,
    fraction: float | None = None,
) -> float:
    """
    Fractional Kelly for stocks — multiply full Kelly by a safety factor.

    Quarter Kelly (0.25) is conservative, half Kelly (0.50) is moderate.
    Never use full Kelly — one bad estimate = ruin.

    Args:
        win_prob: Estimated probability
        reward_pct: Target % gain
        risk_pct: Stop-loss % loss
        fraction: Kelly multiplier (default from config, 0.25)

    Returns:
        Fractional Kelly bet size (0.0 to 1.0, clamped).
    """
    if fraction is None:
        strategy = _load_strategy()
        fraction = strategy.get("stock_params", {}).get("kelly_multiplier", 0.25)

    full_k = kelly_fraction_stock(win_prob, reward_pct, risk_pct)

    if full_k <= 0:
        return 0.0

    return min(full_k * fraction, 1.0)


def kelly_position_size(
    portfolio_value: float,
    current_price: float,
    target_price: float,
    stop_loss: float,
    composite_score: int,
    kronos_expected_return: float | None = None,
    max_position_pct: float | None = None,
    fraction: float | None = None,
) -> dict:
    """
    Calculate Kelly-optimal position size in shares.

    Combines:
    1. Composite score → base win probability
    2. Kronos forecast → adjusts win probability up/down
    3. Reward/risk from target + stop
    4. Fractional Kelly for safety
    5. Caps by max_position_pct circuit breaker

    Args:
        portfolio_value: Current portfolio $
        current_price: Entry price $
        target_price: Profit target $
        stop_loss: Stop loss $
        composite_score: 0-13 composite from screener
        kronos_expected_return: Optional Kronos forecast (e.g., +0.05 = +5%)
        max_position_pct: Hard cap (default 0.30)
        fraction: Kelly multiplier (default 0.25)

    Returns:
        {
            "shares": int,
            "position_value": float,
            "win_prob": float,
            "reward_pct": float,
            "risk_pct": float,
            "full_kelly": float,
            "fractional_kelly": float,
            "pct_of_portfolio": float,
            "reason": str,
        }
    """
    if portfolio_value <= 0 or current_price <= 0:
        return {"shares": 0, "reason": "invalid_inputs"}

    # Calculate reward and risk %
    reward_pct = (target_price - current_price) / current_price
    risk_pct = (current_price - stop_loss) / current_price

    if reward_pct <= 0 or risk_pct <= 0:
        return {"shares": 0, "reason": "invalid_target_or_stop",
                "reward_pct": reward_pct, "risk_pct": risk_pct}

    # Base win probability from composite score
    win_prob = composite_to_win_prob(composite_score)

    # Kronos adjustment: if Kronos is bullish, bump win_prob up; bearish, down
    kronos_adjustment = 0.0
    if kronos_expected_return is not None:
        # Kronos +5% return → +5% win prob bump
        # Kronos -5% return → -5% win prob bump
        kronos_adjustment = max(-0.15, min(0.15, kronos_expected_return * 1.0))
        win_prob = max(0.20, min(0.85, win_prob + kronos_adjustment))

    # Calculate Kelly
    full_k = kelly_fraction_stock(win_prob, reward_pct, risk_pct)

    if full_k <= 0:
        return {
            "shares": 0, "reason": "negative_edge",
            "win_prob": round(win_prob, 4),
            "reward_pct": round(reward_pct, 4),
            "risk_pct": round(risk_pct, 4),
            "full_kelly": round(full_k, 4),
        }

    frac_k = fractional_kelly_stock(win_prob, reward_pct, risk_pct, fraction)

    # Apply circuit breaker cap
    if max_position_pct is None:
        strategy = _load_strategy()
        max_position_pct = strategy.get("stock_params", {}).get("max_position_pct", 0.30)

    pct_of_portfolio = min(frac_k, max_position_pct)
    position_value = portfolio_value * pct_of_portfolio
    shares = int(position_value / current_price)

    reason = "kelly_sized"
    if pct_of_portfolio == max_position_pct and frac_k > max_position_pct:
        reason = f"capped_at_{max_position_pct:.0%}"

    return {
        "shares": shares,
        "position_value": round(shares * current_price, 2),
        "win_prob": round(win_prob, 4),
        "kronos_adjustment": round(kronos_adjustment, 4),
        "reward_pct": round(reward_pct, 4),
        "risk_pct": round(risk_pct, 4),
        "reward_to_risk": round(reward_pct / risk_pct, 2),
        "full_kelly": round(full_k, 4),
        "fractional_kelly": round(frac_k, 4),
        "pct_of_portfolio": round(pct_of_portfolio, 4),
        "reason": reason,
    }


def expected_value_stock(
    win_prob: float,
    reward_pct: float,
    risk_pct: float,
) -> float:
    """
    Expected value per dollar bet on a stock trade.

    EV = win_prob * reward_pct - (1-win_prob) * risk_pct

    Args:
        win_prob: Probability of hitting target
        reward_pct: Gain if win (e.g., 0.10)
        risk_pct: Loss if stop (e.g., 0.035)

    Returns:
        Expected value per $1 bet. Positive = profitable.
    """
    return win_prob * reward_pct - (1.0 - win_prob) * risk_pct


# ── Legacy prediction-market functions (kept for polybot compatibility) ──

def kelly_fraction(our_prob: float, market_prob: float) -> float:
    """Legacy prediction-market Kelly. Use kelly_fraction_stock for stocks."""
    if market_prob <= 0 or market_prob >= 1:
        return 0.0
    if our_prob <= 0 or our_prob >= 1:
        return 0.0
    b = (1.0 - market_prob) / market_prob
    p = our_prob
    q = 1.0 - p
    return (p * b - q) / b
