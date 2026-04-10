"""
Last Kiss Detection — Naked Forex Ch. 5

A "Last Kiss" is a breakout + retest pattern:
1. Price breaks through a support/resistance zone
2. Price comes BACK to the zone (the "last kiss")
3. The zone that was resistance becomes support (or vice versa)
4. Entry on the retest with tight stop

This is used to confirm zone validity when selecting put strikes.
If a former resistance zone has flipped to support via a Last Kiss,
it's a high-confidence zone to sell puts at.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from lib.zones import Zone


@dataclass
class LastKissSignal:
    """A detected breakout + retest (Last Kiss) pattern."""
    zone: Zone
    breakout_date: str
    retest_date: str
    flip_type: str           # "resistance_to_support" or "support_to_resistance"
    retest_quality: float    # 0-1, how cleanly price retested the zone
    confirmed: bool          # Did price hold the zone on retest?


def detect_last_kiss(
    df: pd.DataFrame,
    zones: list[Zone],
    lookback: int = 30,
) -> list[LastKissSignal]:
    """
    Scan for Last Kiss patterns at known zones.

    Logic:
    - For each zone, check if price broke through it and came back
    - A breakout above resistance → zone becomes potential support
    - Price retests the zone from above → Last Kiss confirmed if it holds

    Args:
        df: Daily OHLCV DataFrame
        zones: List of Zone objects from zones.py
        lookback: How many bars to scan

    Returns:
        List of LastKissSignal objects
    """
    if len(df) < lookback + 10:
        return []

    signals = []
    recent = df.iloc[-lookback:]

    for zone in zones:
        upper = zone.upper
        lower = zone.lower
        mid = zone.level

        # Track: was price below zone, then above, then retesting?
        # (resistance → support flip)
        phase = "waiting"  # waiting → broke_above → retesting
        breakout_date = None

        for i in range(len(recent)):
            row = recent.iloc[i]
            close = row["close"]
            low = row["low"]

            if phase == "waiting":
                # Price is below the zone
                if close < lower:
                    phase = "below"

            elif phase == "below":
                # Price breaks above the zone
                if close > upper:
                    phase = "broke_above"
                    breakout_date = str(recent.index[i])

            elif phase == "broke_above":
                # Price comes back to retest the zone from above
                if low <= upper and close > lower:
                    # Retest quality: how close did price get to zone center?
                    distance = abs(low - mid) / mid if mid > 0 else 1
                    quality = max(0, 1 - distance * 10)

                    # Confirmed if the candle closed above the zone
                    confirmed = close > upper

                    signals.append(LastKissSignal(
                        zone=zone,
                        breakout_date=breakout_date or "",
                        retest_date=str(recent.index[i]),
                        flip_type="resistance_to_support",
                        retest_quality=round(quality, 2),
                        confirmed=confirmed,
                    ))
                    phase = "waiting"  # Reset for this zone

                # If price falls back through zone, breakout failed
                elif close < lower:
                    phase = "waiting"

        # Also check support → resistance flip
        phase = "waiting"
        breakout_date = None

        for i in range(len(recent)):
            row = recent.iloc[i]
            close = row["close"]
            high = row["high"]

            if phase == "waiting":
                if close > upper:
                    phase = "above"

            elif phase == "above":
                if close < lower:
                    phase = "broke_below"
                    breakout_date = str(recent.index[i])

            elif phase == "broke_below":
                if high >= lower and close < upper:
                    distance = abs(high - mid) / mid if mid > 0 else 1
                    quality = max(0, 1 - distance * 10)
                    confirmed = close < lower

                    signals.append(LastKissSignal(
                        zone=zone,
                        breakout_date=breakout_date or "",
                        retest_date=str(recent.index[i]),
                        flip_type="support_to_resistance",
                        retest_quality=round(quality, 2),
                        confirmed=confirmed,
                    ))
                    phase = "waiting"

                elif close > upper:
                    phase = "waiting"

    # Only return confirmed signals, sorted by quality
    confirmed = [s for s in signals if s.confirmed]
    confirmed.sort(key=lambda s: s.retest_quality, reverse=True)
    return confirmed


def get_flipped_support_zones(
    df: pd.DataFrame,
    zones: list[Zone],
) -> list[Zone]:
    """
    Return zones that have flipped from resistance to support
    via a confirmed Last Kiss. These are the highest-confidence
    zones for selling cash-secured puts.
    """
    last_kisses = detect_last_kiss(df, zones)
    r_to_s = [lk for lk in last_kisses if lk.flip_type == "resistance_to_support"]

    # Return the underlying zones
    flipped_zones = []
    for lk in r_to_s:
        z = lk.zone
        # Upgrade the zone type
        flipped = Zone(
            level=z.level,
            zone_type="support",  # Now confirmed support
            upper=z.upper,
            lower=z.lower,
            touches=z.touches + 1,  # The retest counts as a touch
            first_touch=z.first_touch,
            last_touch=lk.retest_date,
            room_to_left=z.room_to_left,
            score=min(z.score + 2, 10),  # Bonus for Last Kiss confirmation
        )
        flipped_zones.append(flipped)

    return flipped_zones
