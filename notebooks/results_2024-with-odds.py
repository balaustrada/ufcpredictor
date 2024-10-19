# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: kaggle
#     language: python
#     name: kaggle
# ---

# %% [markdown]
# # Predicting 2024 fights using opening odds

# %%
import jupyter_black

jupyter_black.load()

# %%
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import snapshot_download

from ufcpredictor.data_processor import WOSRDataProcessor as DataProcessor
from ufcpredictor.datasets import BasicDataset, ForecastDataset
from ufcpredictor.loss_functions import BettingLoss
from ufcpredictor.models import SymmetricFightNet
from ufcpredictor.utils import convert_odds_to_decimal
from ufcpredictor.plot_tools import PredictionPlots
from ufcpredictor.trainer import Trainer

from ufcscraper.ufc_scraper import UFCScraper
from ufcscraper.odds_scraper import BestFightOddsScraper

from typing import Optional

# %%
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", None)

# %%
logger = logging.getLogger(__name__)

# %%
# Selecting the set of attributes that we will use as features 
# for our model
X_set = [
    "clinch_strikes_att_opponent_per_minute",
    "time_since_last_fight",
    "total_strikes_succ_opponent_per_minute",
    "takedown_succ_per_minute",
    "KO_opponent_per_minute",
    "takedown_att_per_minute",
    "takedown_succ_opponent_per_minute",
    "win_opponent_per_fight",
    "head_strikes_succ_opponent_per_minute",
    "clinch_strikes_succ_opponent_per_minute",
    "ground_strikes_succ_opponent_per_minute",
    "ground_strikes_att_per_minute",
    "head_strikes_succ_per_minute",
    "age",
    "distance_strikes_succ_per_minute",
    "body_strikes_succ_per_minute",
    "strikes_succ_opponent_per_minute",
    "leg_strikes_att_per_minute",
    "reversals_opponent_per_minute",
    "strikes_succ_per_minute",
    "distance_strikes_att_opponent_per_minute",
    "Sub_opponent_per_fight",
    "distance_strikes_att_per_minute",
    "knockdowns_per_minute",
    "OSR",
]

# %%
# From the full dataset load all the event dates
general_data_processor = DataProcessor(Path(".").resolve().parents[1] / "data")
general_data_processor.load_data()
general_data_processor.aggregate_data()
general_data_processor.add_per_minute_and_fight_stats()
event_dates = general_data_processor.scraper.event_scraper.data["event_date"].unique()

# %%
# Preserve only fights for fighters that have at least
# 3 previous fights in the UFC.
general_invalid_fights = set(
    general_data_processor.data_aggregated[
        general_data_processor.data_aggregated["num_fight"] < 4
    ]["fight_id"]
)

general_valid_fights = (
    set(general_data_processor.data["fight_id"]) - general_invalid_fights
)

# %%
stats = []
wins = 0
bets = 0

# %%
# Starting date to make the predictions
starting_date = "2024-01-01"

# %%
# Now iterate over events to start adding data.
# We train the model before each event
# before making predictions
for event_date in sorted(event_dates[event_dates > starting_date]):
    if (
        len(
            general_data_processor.data[
                general_data_processor.data["event_date"] == event_date
            ]
        )
        == 0
    ):
        continue

    data_processor = DataProcessor(Path(".").resolve().parents[1] / "data")

    # Removing fights with date < 4 days before
    # I have to perform it before loading and filtering to ensure
    # no contamination
    data_processor.scraper.event_scraper.data = (
        data_processor.scraper.event_scraper.data[
            pd.to_datetime(data_processor.scraper.event_scraper.data["event_date"])
            < pd.to_datetime(event_date) - pd.Timedelta(days=4)  # 4 days just in case
        ]
    )
    data_processor.load_data()
    data_processor.aggregate_data()
    data_processor.add_per_minute_and_fight_stats()
    data_processor.normalize_data()

    ###########################################
    # Training phase
    ###########################################
    # Check which fights are invalid:
    invalid_fights = set(
        data_processor.data_aggregated[data_processor.data_aggregated["num_fight"] < 3][
            "fight_id"
        ]
    )

    # This is the split for the quality (later) sample where training is
    # performed for some extra epochs
    early_split = pd.to_datetime(event_date) - pd.DateOffset(
        years=5
    )  # Maybe this should be string(?) @@

    early_train_fights = data_processor.data["fight_id"]
    train_fights = data_processor.data["fight_id"][
        data_processor.data["event_date"] >= early_split
    ]

    early_train_fights = set(early_train_fights) - set(invalid_fights)
    train_fights = set(train_fights) - set(invalid_fights)

    # I am going to train with everything left.
    early_train_dataset = BasicDataset(
        data_processor=data_processor,
        fight_ids=early_train_fights,
        X_set=X_set,
    )
    train_dataset = BasicDataset(
        data_processor=data_processor,
        fight_ids=train_fights,
        X_set=X_set,
    )

    early_train_dataloader = torch.utils.data.DataLoader(
        early_train_dataset, batch_size=64, shuffle=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )

    seed = 43
    torch.manual_seed(seed)
    import random

    random.seed(seed)
    np.random.seed(seed)

    model = SymmetricFightNet(
        input_size=len(train_dataset.X_set),
        dropout_prob=0.35,
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    trainer = Trainer(
        train_loader=train_dataloader,
        test_loader=None,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=BettingLoss(),
    )

    # First train
    trainer.train(
        epochs=5,
        train_loader=early_train_dataloader,
        silent=True,
    )

    # Second quality train
    trainer.train(epochs=3, train_loader=train_dataloader, silent=True)

    # Now the model is trained, let's load the ForecastDataset and predict
    predict_dataset = ForecastDataset(
        data_processor=data_processor,
        X_set=train_dataset.X_set,
    )

    # Iterate over the fights happening in the date.
    fight_data = general_data_processor.data[
        general_data_processor.data["event_date"] == event_date
    ]

    fight_data = fight_data[fight_data["fight_id"].isin(general_valid_fights)]

    if len(fight_data) == 0:  # @This should be fixed
        continue

    fight_data = fight_data.merge(
        fight_data,
        left_on="fight_id",
        right_on="fight_id",
        how="inner",
        suffixes=("_x", "_y"),
    )

    fight_data = fight_data[fight_data["fighter_id_x"] != fight_data["fighter_id_y"]]
    fight_data = fight_data.drop_duplicates(subset=["fight_id"], keep="first")

    fight_ids = fight_data["fight_id"].values.tolist()
    fighter_ids = fight_data["fighter_id_x"].values.tolist()
    opponent_ids = fight_data["opponent_id_x"].values.tolist()
    fighter_odds = fight_data["opening_x"].values.tolist()
    opponent_odds = fight_data["opening_y"].values.tolist()
    event_dates = (
        fight_data["event_date_x"]
        .apply(lambda x: x.strftime("%Y-%m-%d"))
        .astype(str)
        .values
    ).tolist()

    Y = fight_data["fighter_id_x"] != fight_data["winner_x"]

    p1, p2 = predict_dataset.get_forecast_prediction(
        fighter_names=fighter_ids,
        opponent_names=opponent_ids,
        event_dates=event_dates,
        fighter_odds=fighter_odds,
        opponent_odds=opponent_odds,
        model=model,
        parse_ids=True,
    )

    for p1i, p2i, Yi, fighter_odd, opponent_odd, fight_id in zip(
        p1, p2, Y, fighter_odds, opponent_odds, fight_ids
    ):
        prediction = (p1i[0] + p2i[0]) * 0.5

        bet = np.abs(prediction - 0.5) * 2 * 10

        if round(prediction) == Yi:  # correct
            if Yi == 0:
                win = bet * fighter_odd
            else:
                win = bet * opponent_odd
        else:
            win = 0

        wins += win
        bets += bet

        stats.append(
            [
                prediction,
                Yi,
                fighter_odd,
                opponent_odd,
                prediction == Yi,
                bet,
                win,
                fight_id,
            ]
        )
    print(
        event_date,
        round(wins, 2),
        round(bets, 2),
        round(wins - bets, 2),
        round(wins / bets * 100, 2),
        end="\r",
    )

# %%

# %%
# Grouping the stats of each prediction into a dataframe
df = pd.DataFrame(
    stats,
    columns=[
        "Prediction",
        "result",
        "fighter_odds",
        "opponent_odds",
        "correct",
        "bet",
        "win",
        "fight_id",
    ],
)

bet = df["bet"].sum()
win = df["win"].sum()
print(f"win ({win:.2f}) - bet ({bet:.2f}) = {win - bet:.2f}")
# print(f"{win:.2f} - {bet:.2f} = {win - bet:.2f}")

# %%
total_number_of_valid_fights = (
    (general_data_processor.data["event_date"] >= starting_date)
    & (general_data_processor.data["fight_id"].isin(general_valid_fights))
).sum()
print("Total number of valid fights")
print(total_number_of_valid_fights)

# %%
# Getting bets per date
df = df.merge(
    general_data_processor.data[["fight_id", "event_date"]],
    on="fight_id",
)
results = df.groupby("event_date")[["bet", "win"]].sum().reset_index()

# %%
fig, ax = plt.subplots()

ax.plot(
    results["event_date"],
    np.cumsum(results["win"] - results["bet"]) / np.cumsum(results["bet"]) * 100,
)
# Put x labels rotated 90 degrees
ax.tick_params(axis="x", labelrotation=90)

ax.grid()
ax.axhline(0, color="black")

ax.set_xlabel("Event date")
ax.set_ylabel("Return (%)");

# %%
from ufcpredictor.utils import convert_odds_to_moneyline, convert_odds_to_decimal

# %%
# Compare performance vs just picking favourite
df["fighter_p"] = 1 / df["fighter_odds"] * 100
df["opponent_p"] = 1 / df["opponent_odds"] * 100

df["fighter_oddsm"] = convert_odds_to_moneyline(df["fighter_odds"])
df["opponent_oddsm"] = convert_odds_to_moneyline(df["opponent_odds"])

excess = (df["fighter_p"] + df["opponent_p"]) - 100

df["fighter_p"] = df["fighter_p"] - excess / 2
df["opponent_p"] = df["opponent_p"] - excess / 2

df["house_pred"] = df["opponent_p"] / 100

df["correct"] = round(df["Prediction"]) == df["result"]
df["house correct"] = round(df["house_pred"]) == df["result"]

# %%
model_correct_fraction = df["correct"].mean()
favourite_correct_faction = df["house correct"].mean()

print(f"Model correct fraction: {model_correct_fraction:.4f}")
print(f"Favourite correct fraction: {favourite_correct_faction:.4f}")
