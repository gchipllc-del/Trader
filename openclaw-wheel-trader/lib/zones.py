"""
Support/Resistance Zone Detection — Naked Forex Method

Zones are "big fat beer bellies" not thin lines.
A zone is valid when price has bounced from it multiple times.
We score zones by: touches, recency, width, and "room to the left."

Source: Naked Forex Ch. 4-6
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config" / "wheel_strategy.yaml"


def _load_zone_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("zones", {})


@dataclass
class Zone:
    """A support or resistance zone."""
    level: float              # Center price of the zone
    zone_type: str            # "support" or "resistance"
    upper: float              # Top of zone
    lower: float              # Bottom of zone
    touches: int              # Number of times price bounced here
    first_touch: str          # Date of first touch
    last_touch: str           # Date of most recent touch
    room_to_left: bool        # Clear space before first touch
    score: float              # Composite quality score (0-10)


def find_swing_points(
    df: pd.DataFrame, window: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find swing highs and swing lows.
    A swing high: high is the highest in a window of N bars on each side.
    A swing low: low is the lowest in a window of N bars on each side.
    """
    highs = []
    lows = []

    high_vals = df["high"].values
    low_vals = df["low"].values

    for i in range(window, len(df) - window):
        # Swing high
        if high_vals[i] == max(high_vals[i - window : i + window + 1]):
            highs.append({"date": df.index[i], "price": high_vals[i], "idx": i})

        # Swing low
        if low_vals[i] == min(low_vals[i - window : i + window + 1]):
            lows.append({"date": df.index[i], "price": low_vals[i], "idx": i})

    return pd.DataFrame(highs), pd.DataFrame(lows)


def cluster_levels(prices: list[float], tolerance_pct: float = 0.02) -> list[list[float]]:
    """
    Cluster nearby price levels into zones.
    Prices within tolerance_pct of each other belong to the same cluster.
    """
    if not prices:
        return []

    sorted_prices = sorted(prices)
    clusters = [[sorted_prices[0]]]

    for price in sorted_prices[1:]:
        cluster_center = np.mean(clusters[-1])
        if abs(price - cluster_center) / cluster_center <= tolerance_pct:
            clusters[-1].append(price)
        else:
            clusters.append([price])

    return clusters


def check_room_to_left(
    df: pd.DataFrame, zone_level: float, zone_width_pct: float, first_touch_idx: int, min_bars: int = 10
) -> bool:
    """
    'Room to the left' — Naked Forex concept.
    Before the first touch of a zone, price should NOT have been hanging
    around that level. There should be clear space — the zone should feel
    like price arrived at a wall, not that it was already sitting there.
    """
    if first_touch_idx < min_bars:
        return False

    upper = zone_level * (1 + zone_width_pct)
    lower = zone_level * (1 - zone_width_pct)

    # Check the N bars before the first touch
    lookback = df.iloc[max(0, first_touch_idx - min_bars) : first_touch_idx]
    bars_in_zone = ((lookback["close"] >= lower) & (lookback["close"] <= upper)).sum()

    # If fewer than 30% of lookback bars are in the zone, there's room
    return bars_in_zone / len(lookback) < 0.3 if len(lookback) > 0 else False


def detect_zones(
    df: pd.DataFrame,
    current_price: float | None = None,
) -> list[Zone]:
    """
    Detect support and resistance zones from OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume] and DatetimeIndex
        current_price: Current price to classify zones as support/resistance

    Returns:
        List of Zone objects sorted by score (best first)
    """
    cfg = _load_zone_config()
    min_touches = cfg.get("min_touches", 2)
    zone_width_pct = cfg.get("zone_width_pct", 0.02)
    lookback_days = cfg.get("lookback_days", 120)
    require_room = cfg.get("room_to_left", True)

    # Trim to lookback window
    if len(df) > lookback_days:
        df = df.iloc[-lookback_days:]

    if current_price is None:
        current_price = df["close"].iloc[-1]

    # Find swing points
    swing_highs, swing_lows = find_swing_points(df, window=5)

    # Combine all swing levels
    all_levels = []
    if not swing_highs.empty:
        for _, row in swing_highs.iterrows():
            all_levels.append(
                {"price": row["price"], "date": row["date"], "idx": int(row["idx"]), "type": "high"}
            )
    if not swing_lows.empty:
        for _, row in swing_lows.iterrows():
            all_levels.append(
                {"price": row["price"], "date": row["date"], "idx": int(row["idx"]), "type": "low"}
            )

    if not all_levels:
        return []

    # Cluster nearby levels into zones
    prices = [lv["price"] for lv in all_levels]
    clusters = cluster_levels(prices, tolerance_pct=zone_width_pct)

    zones = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue

        level = float(np.mean(cluster))
        upper = level * (1 + zone_width_pct)
        lower = level * (1 - zone_width_pct)

        # Find all touches of this zone
        touch_data = [
            lv for lv in all_levels
            if lower <= lv["price"] <= upper
        ]

        if len(touch_data) < min_touches:
            continue

        first_touch = min(touch_data, key=lambda x: x["idx"])
        last_touch = max(touch_data, key=lambda x: x["idx"])

        # Room to the left check
        has_room = check_room_to_left(
            df, level, zone_width_pct, first_touch["idx"]
        )

        if require_room and not has_room:
            continue

        # Classify as support or resistance
        zone_type = "support" if level < current_price else "resistance"

        # Score the zone (0-10)
        touch_score = min(len(touch_data) / 5, 1.0) * 3          # Max 3 for touches
        recency_score = (last_touch["idx"] / len(df)) * 3         # Max 3 for recency
        room_score = 2.0 if has_room else 0.0                     # 2 for room to left
        width_score = 2.0 if zone_width_pct <= 0.03 else 1.0      # Tighter = better
        total_score = touch_score + recency_score + room_score + width_score

        zones.append(Zone(
            level=round(level, 2),
            zone_type=zone_type,
            upper=round(upper, 2),
            lower=round(lower, 2),
            touches=len(touch_data),
            first_touch=str(first_touch["date"]),
            last_touch=str(last_touch["date"]),
            room_to_left=has_room,
            score=round(total_score, 2),
        ))

    # Sort by score descending
    zones.sort(key=lambda z: z.score, reverse=True)
    return zones


def get_nearest_support(zones: list[Zone], current_price: float) -> Zone | None:
    """Get the highest-scoring support zone below current price."""
    support_zones = [z for z in zones if z.zone_type == "support" and z.level < current_price]
    return support_zones[0] if support_zones else None


def get_nearest_resistance(zones: list[Zone], current_price: float) -> Zone | None:
    """Get the highest-scoring resistance zone above current price."""
    resistance_zones = [z for z in zones if z.zone_type == "resistance" and z.level > current_price]
    return resistance_zones[0] if resistance_zones else None
