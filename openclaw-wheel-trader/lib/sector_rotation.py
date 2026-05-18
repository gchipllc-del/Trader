"""
Sector Rotation Overlay — regime-aware candidate ranking modifier.

Adapted from FinRL-X's "Adaptive Rotation" pattern (github.com/AI4Finance-
Foundation/FinRL-Trading): bias capital toward sleeves whose risk profile
matches the current regime. Bull regimes favor growth tech, chop favors
balanced/defensive, bear favors defensive.

We do NOT literally allocate buckets of cash (the original FinRL-X
pattern). Instead, this is a *candidate-ranking modifier*: after the
existing gates (quant + technical + Kelly + correlation), each
candidate's `composite_score` gets a small bonus or penalty based on
whether its sleeve aligns with the current regime's preferred sleeve(s).

Why this design over literal bucket allocation:
  • Bucket allocation only makes sense above a bankroll threshold
    where you have enough capital to populate each sleeve meaningfully.
    With $1.5K and min_position_pct=0.10 (= $150/pos), 3 sleeves with
    3 positions each ≈ $1350 — barely fits. At $5K it's roomy.
  • Even at small bankrolls, *biasing selection* toward regime-aligned
    sleeves still helps without forcing diversification math you can't
    afford. The bias is gated by `activation_bankroll`.

Public API:
    apply_rotation_bias(candidates, regime, bankroll, cfg) -> list[dict]
    get_sleeve_preferences(regime, cfg) -> dict[sleeve, weight]
    classify_sleeve(ticker) -> str  # one of growth_tech / real_assets / defensive / other

Returns candidates with new fields:
    rotation_sleeve  : which sleeve this ticker maps to
    rotation_bonus   : multiplier applied (1.0 = no change, >1 = favored, <1 = de-emphasized)
    rotation_active  : True if rotation modified scoring, False otherwise
"""

from __future__ import annotations

from lib.audit import log_event


# ── Sleeve classification ──────────────────────────────────────────
# Map correlation-group names (from lib/correlation.py) to FinRL-X
# style sleeves. Tickers inherit the sleeve of their correlation group.
GROUP_TO_SLEEVE: dict[str, str] = {
    # GROWTH TECH — high beta, AI/innovation, sensitive to growth regimes
    "big_tech": "growth_tech",
    "ai_defense": "growth_tech",
    "intel_semi": "growth_tech",
    "fintech": "growth_tech",
    "ev_autos": "growth_tech",      # behaves more like tech than autos for trading
    "entertainment": "growth_tech", # streaming/digital lean

    # REAL ASSETS — cyclical, commodity-tied, value-leaning
    "steel_mining": "real_assets",
    "airlines": "real_assets",
    "cruise": "real_assets",
    "mobility": "real_assets",
    "banks": "real_assets",          # rate-sensitive; behaves cyclically

    # DEFENSIVE — staples, pharma, low-beta
    "pharma": "defensive",
    "consumer": "defensive",

    # OTHER — Chinese ADRs are their own thing (geopolitical risk)
    "chinese_adr": "other",
}

# Sleeve preference weights per regime. Each row sums to roughly 1.0
# but the value used is the per-sleeve multiplier (not a literal
# allocation %). Default 1.0 = no bias.
REGIME_SLEEVE_BIAS: dict[str, dict[str, float]] = {
    "bull": {
        "growth_tech": 1.20,    # favor growth in bull
        "real_assets": 1.05,    # neutral-positive
        "defensive": 0.85,      # under-weight defensive
        "other": 0.95,
    },
    "bear": {
        "growth_tech": 0.75,    # avoid growth in bear
        "real_assets": 0.95,
        "defensive": 1.25,      # favor defensive
        "other": 0.85,          # extra penalty on China risk
    },
    "chop": {
        "growth_tech": 0.90,    # slight under-weight
        "real_assets": 1.05,
        "defensive": 1.10,      # mild favor
        "other": 0.90,
    },
    "unknown": {                # neutral when regime indeterminate
        "growth_tech": 1.0,
        "real_assets": 1.0,
        "defensive": 1.0,
        "other": 1.0,
    },
}

# Default activation: rotation is dormant until bankroll crosses this.
# At small bankrolls, diversification math doesn't have enough capital
# to populate sleeves; biasing selection is fine but the lift is small.
DEFAULT_ACTIVATION_BANKROLL = 5000.0


def classify_sleeve(ticker: str) -> str:
    """Return sleeve name for a ticker. 'other' for unmapped tickers."""
    try:
        from lib.correlation import CORRELATION_GROUPS
    except Exception:
        return "other"
    upper = ticker.upper()
    for group_name, members in CORRELATION_GROUPS.items():
        if upper in [m.upper() for m in members]:
            return GROUP_TO_SLEEVE.get(group_name, "other")
    return "other"


def get_sleeve_preferences(regime: str, cfg: dict | None = None) -> dict[str, float]:
    """Return per-sleeve multiplier for a given regime. Allows config
    override of the defaults via cfg.regime_bias[regime]."""
    cfg = cfg or {}
    overrides = (cfg.get("regime_bias") or {}).get(regime)
    if overrides and isinstance(overrides, dict):
        # Merge: override only the sleeves the user specified
        base = dict(REGIME_SLEEVE_BIAS.get(regime, REGIME_SLEEVE_BIAS["unknown"]))
        base.update(overrides)
        return base
    return REGIME_SLEEVE_BIAS.get(regime, REGIME_SLEEVE_BIAS["unknown"])


def apply_rotation_bias(
    candidates: list[dict],
    regime: str,
    bankroll: float,
    cfg: dict | None = None,
) -> list[dict]:
    """Apply regime-aware sleeve bias to candidate composite_scores.

    Mutates each candidate to add `rotation_sleeve`, `rotation_bonus`,
    `rotation_active`. The bias is multiplicative on `composite_score`
    so downstream sorting/gating sees the adjusted score directly.

    Dormant when:
        cfg.enabled is False
        bankroll < cfg.activation_bankroll (default $5000)
        regime is None or empty
        candidates list is empty

    Failures fall back to no-op (returns candidates unchanged) — never
    breaks the scan path.
    """
    cfg = cfg or {}
    if not cfg.get("enabled", True):
        return candidates
    if not candidates:
        return candidates

    activation = float(cfg.get("activation_bankroll", DEFAULT_ACTIVATION_BANKROLL))
    if bankroll < activation:
        # Dormant — annotate so the diagnostic is visible but don't
        # touch scoring.
        for c in candidates:
            c["rotation_sleeve"] = classify_sleeve(c.get("ticker", ""))
            c["rotation_bonus"] = 1.0
            c["rotation_active"] = False
        log_event("sector_rotation", "dormant", {
            "regime": regime,
            "bankroll": round(bankroll, 2),
            "activation_threshold": activation,
        })
        return candidates

    regime_key = (regime or "unknown").lower()
    sleeve_weights = get_sleeve_preferences(regime_key, cfg)
    # Clamp the effect: never let rotation bias more than ±20% (defensive
    # — rotation is a tiebreaker, not a primary signal). This guards
    # against accidentally-extreme config.
    max_swing = float(cfg.get("max_bonus_swing", 0.25))
    floor = max(0.0, 1.0 - max_swing)
    ceiling = 1.0 + max_swing

    for c in candidates:
        ticker = c.get("ticker", "")
        sleeve = classify_sleeve(ticker)
        raw = float(sleeve_weights.get(sleeve, 1.0))
        bonus = max(floor, min(ceiling, raw))
        c["rotation_sleeve"] = sleeve
        c["rotation_bonus"] = round(bonus, 4)
        c["rotation_active"] = True
        # Apply to the composite_score used by downstream sort/gates.
        # Round to int because composite_score is integer-valued upstream.
        base_score = float(c.get("composite_score", 0))
        biased = base_score * bonus
        c["composite_score_pre_rotation"] = c.get("composite_score", 0)
        c["composite_score"] = int(round(biased))

    log_event("sector_rotation", "applied", {
        "regime": regime_key,
        "bankroll": round(bankroll, 2),
        "n_candidates": len(candidates),
        "sleeve_weights": sleeve_weights,
    })
    return candidates


def summary_line(c: dict) -> str:
    """One-line annotation for per-candidate output."""
    if not c.get("rotation_active"):
        return f"rotation: dormant (bankroll < threshold)"
    return (
        f"rotation: sleeve={c.get('rotation_sleeve')} "
        f"bonus={c.get('rotation_bonus')} "
        f"({c.get('composite_score_pre_rotation')} → {c.get('composite_score')})"
    )
