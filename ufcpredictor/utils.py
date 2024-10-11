from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional

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
    if time_str == "--":
        return 0
    elif time_str in (None, "NULL") or pd.isna(time_str):
        return None
    else:
        minutes, seconds = map(int, time_str.split(":"))
        return minutes * 60 + seconds


def convert_odds_to_decimal(odds: NDArray[np.float64]) -> NDArray[np.float64]:
    msk = odds > 0

    odds[msk] = odds[msk] / 100 + 1
    odds[~msk] = 100 / -odds[~msk] + 1

    return odds


def convert_odds_to_moneyline(odds: NDArray[np.float64]) -> NDArray[np.float64]:
    msk = odds > 2

    odds[msk] = (odds[msk] - 1) * 100
    odds[~msk] = 100 / (1 - odds[~msk])

    return odds
