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

from typing import Optional

# %%
import matplotlib.pyplot as plt
import pandas as pd

# %%
logger = logging.getLogger(__name__)

# %%
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
# Set the starting date for the process
starting_date = "2020-01-01"

# %%
# From the full dataset load all the event dates
general_data_processor = DataProcessor(Path(".").resolve().parents[1] / "data")
general_data_processor.load_data()
event_dates = general_data_processor.scraper.event_scraper.data["event_date"].unique()

# %%
# Now iterate over events to start adding data.
# (this should encapsulate the following cells as well)
for event_date in sorted(event_dates[event_dates > starting_date]):
    pass

event_date = "2024-10-05"
# event_date = "2024-12-05"

# %%
data_processor = DataProcessor(Path(".").resolve().parents[1] / "data")
data_processor.load_data()
data_processor.data = data_processor.data[
    data_processor.data["event_date"]
    < (
        pd.to_datetime(event_date) - pd.Timedelta(days=4)
    )  # Just for robustness use 4 days before as limit
]

# %%
data_processor.aggregate_data()
data_processor.add_per_minute_and_fight_stats()
data_processor.normalize_data()

# %%
###########################################
# Training phase
###########################################
# Check which fights are invalid:
invalid_fights = set(
    data_processor.data_aggregated[data_processor.data_aggregated["num_fight"] < 4][
        "fight_id"
    ]
)

# This is the split for the quality (later) sample where training is
# performed for some extra epochs
# early_split = pd.to_datetime(event_date) - pd.DateOffset(
#     years=4
# )  # Maybe this should be string(?) @@
early_split = "2018-01-01"

early_train_fights = data_processor.data["fight_id"]
train_fights = data_processor.data["fight_id"][
    data_processor.data["event_date"] >= early_split
]

early_train_fights = set(early_train_fights) - set(invalid_fights)
train_fights = set(train_fights) - set(invalid_fights)

# %%
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

# %%
print("First train...")
# First train
trainer.train(
    epochs=5,
    train_loader=early_train_dataloader,
    silent=False,
)

print("Second train..")
# Second quality train
trainer.train(epochs=3, train_loader=train_dataloader, silent=False)

# %%
# Now the model is trained, let's load the ForecastDataset and predict
predict_dataset = ForecastDataset(
    data_processor=data_processor,
    X_set=train_dataset.X_set,
)

# %%
# Iterate over the fights happening in the date.

# %%
fight_data = general_data_processor.data[
    general_data_processor.data["event_date"] == event_date
]

# %%
fight_data = fight_data.merge(
    fight_data,
    left_on="fight_id",
    right_on="fight_id",
    how="inner",
    suffixes=("_x", "_y"),
)

fight_data = fight_data[fight_data["fighter_id_x"] != fight_data["fighter_id_y"]]
fight_data = fight_data.drop_duplicates(subset=["fight_id"], keep="first")

# %%
fight_data

# %%
fighter_ids = fight_data["fighter_id_x"].values.tolist()
opponent_ids = fight_data["opponent_id_x"].values.tolist()
winner = fighter_ids == fight_data["winner_x"].values.tolist()
fighter_odds = fight_data["opening_x"].values.tolist()
opponent_odds = fight_data["opening_y"].values.tolist()
event_dates = (
    fight_data["event_date_x"]
    .apply(lambda x: x.strftime("%Y-%m-%d"))
    .astype(str)
    .values
).tolist()

# %%
model = SymmetricFightNet(
    input_size=len(X_set),
    dropout_prob=0.35,
)
model.load_state_dict(torch.load("models/model.pth"))

# %%
predict_dataset.get_forecast_prediction(
    fighter_names=fighter_ids,
    opponent_names=opponent_ids,
    event_dates=event_dates,
    fighter_odds=fighter_odds,
    opponent_odds=opponent_odds,
    model=model,
    parse_ids=True,
)

# %%
predict_dataset.get_single_forecast_prediction(
    fighter_name="Ilia Topuria",
    opponent_name="Max Holloway",
    event_date="2024-10-26",
    odds1=convert_odds_to_decimal(-188),
    odds2=convert_odds_to_decimal(188),
    model=model,
)

# %%
fighter_names = [
    "Ilia Topuria",
]
opponent_names = [
    "Max Holloway",
]
event_dates = [
    "2024-10-26",
]
fighter_odds = convert_odds_to_decimal(
    [
        -188,
    ]
)
opponent_odds = convert_odds_to_decimal(
    [
        188,
    ]
)
model = model
parse_ids = False

# %%
self = predict_dataset

# %%
if not parse_ids:
    fighter_ids = [self.data_processor.get_fighter_id(x) for x in fighter_names]
    opponent_ids = [self.data_processor.get_fighter_id(x) for x in opponent_names]
else:
    fighter_ids = fighter_names
    opponent_ids = opponent_names

match_data = pd.DataFrame(
    {
        "fighter_id": fighter_ids + opponent_ids,
        "event_date_forecast": event_dates * 2,
        "opening": np.concatenate((fighter_odds, opponent_odds)),
    }
)

match_data = match_data.merge(
    self.data_processor.data_normalized,
    left_on="fighter_id",
    right_on="fighter_id",
)

match_data = match_data[match_data["event_date"] < match_data["event_date_forecast"]]
match_data = match_data.sort_values(
    by=["fighter_id", "event_date"],
    ascending=[True, False],
)
match_data = match_data.drop_duplicates(
    subset=["fighter_id", "event_date_forecast"],
    keep="first",
)
match_data["id_"] = (
    match_data["fighter_id"].astype(str)
    + "_"
    + match_data["event_date_forecast"].astype(str)
)

# This data dict is used to facilitate the construction of the tensors
data_dict = {
    id_: data
    for id_, data in zip(
        match_data["id_"].values,
        np.asarray([match_data[x] for x in self.X_set]).T,
    )
}

data = [
    torch.FloatTensor(
        np.asarray(
            [
                data_dict[fighter_id + "_" + str(event_date)]
                for fighter_id, event_date in zip(fighter_ids, event_dates)
            ]
        )
    ),  # X1
    torch.FloatTensor(
        np.asarray(
            [
                data_dict[fighter_id + "_" + str(event_date)]
                for fighter_id, event_date in zip(opponent_ids, event_dates)
            ]
        )
    ),  # X2
    torch.FloatTensor(np.asarray(fighter_odds)).reshape(-1, 1),  # Odds1,
    torch.FloatTensor(np.asarray(opponent_odds)).reshape(-1, 1),  # Odds2
]

# %%
data

# %%
X1, X2, odds1, odds2 = data

# %%
model.eval()
with torch.no_grad():
    predictions_1 = model(X1, X2, odds1, odds2).detach().numpy()

# %%
predictions_1

# %%
