import pandas as pd

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

def convert_minutes_to_seconds(time_str: str) -> int:
    if time_str == "--":
        return 0
    elif time_str in (None, "NULL") or pd.isna(time_str):
        return None
    else:
        minutes, seconds = map(int, time_str.split(":"))
        return minutes * 60 + seconds
