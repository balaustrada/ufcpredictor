"""
Utility functions for the UFC predictor project.

This module contains various utility functions used throughout the project, including
functions for converting between different time and odds formats, as well as other
miscellaneous helper functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from typing import List, Optional

    import numpy as np
    from numpy.typing import NDArray


weight_dict = {
    "Heavyweight": 265,
    "Welterweight": 170,
    "Women's Flyweight": 125,
    "Light Heavyweight": 205,
    "Middleweight": 185,
    "Women's Featherweight": 145,
    "Bantamweight": 135,
    "Lightweight": 155,
    "Flyweight": 125,
    "Women's Strawweight": 115,
    "Women's Bantamweight": 135,
    "Featherweight": 145,
}


def convert_minutes_to_seconds(time_str: str) -> Optional[int]:
    """
    Convert a time string from minutes:seconds format to seconds.

    Args:
        time_str: Time string in minutes:seconds format.

    Returns:
        Time in seconds. If the input string is "--", returns 0. If the input is None
            or "NULL", or if the input is NaN, returns None.
    """
    if time_str == "--":
        return 0
    elif time_str in (None, "NULL") or pd.isna(time_str):
        return None
    else:
        minutes, seconds = map(int, time_str.split(":"))
        return minutes * 60 + seconds


def convert_odds_to_decimal(
    odds: List[int | float] | NDArray[np.float64 | np.int_],
) -> NDArray[np.float64]:
    """
    Convert odds from American format to decimal format.

    Args:
        odds: Odds in American format.

    Returns:
        Odds in decimal format.
    """
    if not isinstance(odds, np.ndarray):
        odds = np.asarray(odds, dtype=np.float64)
    else:
        odds = odds.astype(np.float64)

    msk = odds > 0

    odds[msk] = odds[msk] / 100 + 1
    odds[~msk] = 100 / -odds[~msk] + 1

    return odds


def convert_odds_to_moneyline(
    odds: NDArray[np.float64] | List[float],
) -> NDArray[np.int_]:
    """
    Convert odds from decimal format to moneyline format.

    Args:
        odds: Odds in decimal format.

    Returns:
        Odds in moneyline format.
    """
    if not isinstance(odds, np.ndarray):
        odds = np.asarray(odds, dtype=np.float64)

    msk = odds > 2

    odds[msk] = (odds[msk] - 1) * 100
    odds[~msk] = 100 / (1 - odds[~msk])

    return np.round(odds).astype(int)
