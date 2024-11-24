# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

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

from ufcpredictor.data_processor import DataProcessor

from ufcpredictor.datasets import BasicDataset, ForecastDataset
from ufcpredictor.data_enhancers import SumFlexibleELO, RankedFields
from ufcpredictor.data_aggregator import WeightedDataAggregator
from ufcpredictor.datasets import BasicDataset
from ufcpredictor.trainer import Trainer

from ufcscraper.ufc_scraper import UFCScraper
from ufcscraper.odds_scraper import BestFightOddsScraper

from typing import Optional

# %%
from ufcpredictor.models import SymmetricFightNet
from ufcpredictor.loss_functions import BettingLoss

# %%
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", None)

# %%
logger = logging.getLogger(__name__)

# %%
data_processor_kwargs = {
    "data_folder": "/home/cramirpe/UFC/UFCfightdata",
    "data_aggregator": WeightedDataAggregator(alpha=-0.0001),
    "data_enhancers": [
        SumFlexibleELO(
            scaling_factor=0.5,
            K_factor=30,
        ),
        RankedFields(
            fields=["age", "fighter_height_cm"],
            exponents=[1.2, 1.2],
        ),
    ],
}

data_processor = DataProcessor(**data_processor_kwargs)

# %%
if True:
    X_set = [
        "age",
        "body_strikes_att_opponent_per_minute",
        "body_strikes_att_per_minute",
        "body_strikes_succ_opponent_per_minute",
        "body_strikes_succ_per_minute",
        "clinch_strikes_att_opponent_per_minute",
        "clinch_strikes_att_per_minute",
        "clinch_strikes_succ_opponent_per_minute",
        "clinch_strikes_succ_per_minute",
        "ctrl_time_opponent_per_minute",
        "ctrl_time_per_minute",
        "distance_strikes_att_opponent_per_minute",
        "distance_strikes_att_per_minute",
        "distance_strikes_succ_opponent_per_minute",
        "distance_strikes_succ_per_minute",
        "fighter_height_cm",
        "ground_strikes_att_opponent_per_minute",
        "ground_strikes_att_per_minute",
        "ground_strikes_succ_opponent_per_minute",
        "ground_strikes_succ_per_minute",
        "head_strikes_att_opponent_per_minute",
        "head_strikes_att_per_minute",
        "head_strikes_succ_opponent_per_minute",
        "head_strikes_succ_per_minute",
        "knockdowns_opponent_per_minute",
        "knockdowns_per_minute",
        "KO_opponent_per_fight",
        "KO_opponent_per_minute",
        "KO_per_fight",
        "KO_per_minute",
        "leg_strikes_att_opponent_per_minute",
        "leg_strikes_att_per_minute",
        "leg_strikes_succ_opponent_per_minute",
        "leg_strikes_succ_per_minute",
        "num_fight",
        "reversals_opponent_per_minute",
        "reversals_per_minute",
        "strikes_att_opponent_per_minute",
        "strikes_att_per_minute",
        "strikes_succ_opponent_per_minute",
        "strikes_succ_per_minute",
        "Sub_opponent_per_fight",
        "Sub_opponent_per_minute",
        "Sub_per_fight",
        "Sub_per_minute",
        "submission_att_opponent_per_minute",
        "submission_att_per_minute",
        "takedown_att_opponent_per_minute",
        "takedown_att_per_minute",
        "takedown_succ_opponent_per_minute",
        "takedown_succ_per_minute",
        "time_since_last_fight",
        "total_strikes_att_opponent_per_minute",
        "total_strikes_att_per_minute",
        "total_strikes_succ_opponent_per_minute",
        "total_strikes_succ_per_minute",
        "win_opponent_per_fight",
        "win_per_fight",
        "ELO",
    ]
else:
    X_set = None
    X_set = BasicDataset.X_set + [
        "ELO",
    ]

# %%
Xf_set = ["num_rounds", "weight"]

# %%
# Set the starting date for the process
starting_date = "2023-01-01"  # "2023-01-01"

# %%
data_processor.load_data()
data_processor.aggregate_data()
data_processor.add_per_minute_and_fight_stats()

# for field in X_set:
#     if field in ["ELO", "age"]:
#         continue
#     self.data_aggregated[field] = (self.data_aggregated[field].rank(pct=True) * 100) ** 1.2

data_processor.normalize_data()

# %%
general_invalid_fights = set(
    data_processor.data[data_processor.data["num_fight"] < 4]["fight_id"]
)  # The usual is 4

general_valid_fights = set(data_processor.data["fight_id"]) - set(
    general_invalid_fights
)
event_dates = data_processor.data_aggregated["event_date"].unique()

# %%

# %%
stats = []
wins = 0
bets = 0

# %%
cash = [
    200,
]
invest = [
    200,
]

# %%
starting_date = "2023-01-01"

# convert it into pandas datetime
starting_date = pd.to_datetime(starting_date)

# %%

# %%
# Now iterate over events to start adding data.
# (this should encapsulate the following cells as well)
for i, event_date in enumerate(sorted(event_dates[event_dates > starting_date])):
    if len(data_processor.data[data_processor.data["event_date"] == event_date]) == 0:
        continue

    invalid_fights = general_invalid_fights | set(
        data_processor.data["fight_id"][
            data_processor.data["event_date"] > (event_date - pd.Timedelta(days=3))
        ]
    )

    ###########################################
    # Training phase
    ###########################################
    # This is the split for the quality (later) sample where training is
    # performed for some extra epochs
    early_split = pd.to_datetime(event_date) - pd.DateOffset(
        years=5
    )  # Maybe this should be string(?) @@

    early_train_fights = data_processor.data["fight_id"][
        data_processor.data["event_date"] < (event_date - pd.Timedelta(days=3))
    ]
    train_fights = data_processor.data["fight_id"][
        (
            (data_processor.data["event_date"] < (event_date - pd.Timedelta(days=3)))
            & (data_processor.data["event_date"] >= early_split)
        )
    ]

    early_train_fights = set(early_train_fights) - set(invalid_fights)
    train_fights = set(train_fights) - set(invalid_fights)

    # I am going to train with everything left.
    early_train_dataset = BasicDataset(
        data_processor=data_processor,
        fight_ids=early_train_fights,
        X_set=X_set,
        Xf_set=Xf_set,
    )
    train_dataset = BasicDataset(
        data_processor=data_processor,
        fight_ids=train_fights,
        X_set=X_set,
        Xf_set=Xf_set,
    )

    early_train_dataloader = torch.utils.data.DataLoader(
        early_train_dataset, batch_size=256, shuffle=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True
    )

    seed = 43
    torch.manual_seed(seed)
    import random

    random.seed(seed)
    np.random.seed(seed)

    model = SymmetricFightNet(
        input_size=len(train_dataset.X_set),
        input_size_f=len(train_dataset.Xf_set),
        dropout_prob=0.35,
        fighter_network_shape=[256, 512, 1024, 512],
        network_shape=[2048, 1024, 512, 128, 64, 1],
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=2
    )

    trainer = Trainer(
        train_loader=train_dataloader,
        test_loader=None,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=BettingLoss(
            max_bet=max(
                cash[-1] * 0.1,
                5,
            ),
        ),
        mlflow_tracking=False,
    )

    # First train
    trainer.train(
        epochs=5,
        train_loader=early_train_dataloader,
        silent=True,
    )

    # Second quality train
    trainer.train(epochs=30, train_loader=train_dataloader, silent=True)

    # Now the model is trained, let's load the ForecastDataset and predict
    predict_dataset = ForecastDataset(
        data_processor=data_processor,
        X_set=train_dataset.X_set,
        Xf_set=train_dataset.Xf_set,
    )

    # Iterate over the fights happening in the date.
    fight_data = data_processor.data[data_processor.data["event_date"] == event_date]

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

    num_rounds = fight_data["num_rounds_x"].values.tolist()
    weights = fight_data["weight_x"].values.tolist()

    fight_features = list(np.asarray([num_rounds, weights]).T)

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
        fight_features=fight_features,
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

        extra_added = max(bet - cash[-1], 0)
        cash_i = cash[-1] + win - min(bet, cash[-1])

        invest.append(invest[-1] + extra_added)
        cash.append(cash_i)

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
        round(cash[-1], 2),
        round(invest[-1], 2),
        round(cash[-1] - invest[-1], 2),
        # round(wins, 2),
        # round(bets, 2),
        # round(wins - bets, 2),
        # round(wins / bets * 100, 2),
        end="\r",
    )

# %%
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

df = df.merge(
    data_processor.data[
        ["fight_id", "fighter_id", "event_date", "event_id", "weight_class"]
    ],
    on="fight_id",
)

df.to_csv("full_predictions_20241111_1825.csv")

# %%
results = df.groupby("event_date")[["bet", "win"]].sum().reset_index()

# %%
cash = [
    200,
]
invest = [
    200,
]

for i, (win, bet) in enumerate(zip(results["win"], results["bet"])):
    extra_added = max(bet - cash[-1], 0)
    cash_i = cash[-1] + win - min(bet, cash[-1])

    invest.append(invest[-1] + extra_added)
    cash.append(cash_i)

# %%
fig, ax = plt.subplots()

ax.plot(
    results["event_date"],
    invest[1:],
    label="invest",
)

ax.plot(
    results["event_date"],
    cash[1:],
    label="cash",
)

ax.plot(
    results["event_date"],
    [x - y for x, y in zip(cash, invest)][1:],
    label="profit",
)

ax.axhline(0, c="k")
ax.legend()
ax.grid()

# %%

# %%

# %%
bet = df["bet"].sum()
win = df["win"].sum()
print(f"{win:.2f} - {bet:.2f} = {win - bet:.2f}")

# %%

# %%

# %%

# %%
df

# %%
total_number_of_valid_fights = (
    (general_data_processor.data["event_date"] >= starting_date)
    & (general_data_processor.data["fight_id"].isin(general_valid_fights))
).sum()
print(total_number_of_valid_fights)

# %%

# %%

# %%
df = df.merge(
    general_data_processor.data[["fight_id", "event_date"]],
    on="fight_id",
)

# %%
df

# %%
results = df.groupby("event_date")[["bet", "win"]].sum().reset_index()
results

# %%
fig, ax = plt.subplots()

ax.plot(
    results["event_date"],
    np.cumsum(results["win"] - results["bet"]) / np.cumsum(results["bet"]),
)
# Put x labels rotated 90 degrees
ax.tick_params(axis="x", labelrotation=90)

ax.grid()
ax.axhline(0, color="black")

# %%

# %%
from ufcpredictor.utils import convert_odds_to_moneyline, convert_odds_to_decimal

# %%
df["fighter_p"] = 1 / df["fighter_odds"] * 100
df["opponent_p"] = 1 / df["opponent_odds"] * 100

df["fighter_oddsm"] = convert_odds_to_moneyline(df["fighter_odds"])
df["opponent_oddsm"] = convert_odds_to_moneyline(df["opponent_odds"])

excess = (df["fighter_p"] + df["opponent_p"]) - 100

df["fighter_p"] = df["fighter_p"] - excess / 2
df["opponent_p"] = df["opponent_p"] - excess / 2

df["house_pred"] = df["opponent_p"] / 100

# %%
df["correct"] = round(df["Prediction"]) == df["result"]
df["house correct"] = round(df["house_pred"]) == df["result"]
df

# %%
df[["correct", "house correct"]].mean()

# %%
(df["correct"].mean() - df["house correct"].mean()) * 100

# %%

# %%
