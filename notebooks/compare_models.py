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
#     display_name: ufc
#     language: python
#     name: ufc
# ---

# %%
import jupyter_black

jupyter_black.load()

# %%
import pandas as pd

pd.set_option("display.max_columns", None)

# %%
from ufcpredictor.data_processor import (
    OSRDataProcessor,
    SumFlexibleELODataProcessor,
    WOSRDataProcessor,
    ELODataProcessor,
    FlexibleELODataProcessor,
)
from ufcpredictor.datasets import BasicDataset
from ufcpredictor.trainer import Trainer
import torch
import numpy as np

# %%
import mlflow.pytorch
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# %%
mlflow.set_experiment("test0")

# %%
data_folder = "/home/cramirpe/UFC/UFCfightdata"

# %%
DataProcessor = SumFlexibleELODataProcessor
data_processor_kwargs = {
    "data_folder": "/home/cramirpe/UFC/UFCfightdata",
    "scaling_factor": 0.5,
    # "boost_values": [1, 2, 3],
    "K_factor": 30,
}

# %%
self = data_processor = DataProcessor(**data_processor_kwargs)

# %%
self.load_data()
self.aggregate_data()
self.add_per_minute_and_fight_stats()
self.normalize_data()

# %%
if True:
    X_set = [
        "clinch_strikes_att_opponent_per_minute",
        "time_since_last_fight",
        "total_strikes_succ_opponent_per_minute",
        "takedown_succ_per_minute",
        "KO_opponent_per_minute",
        "KO_per_minute",
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
        "ELO",
        "fighter_height_cm",
        "weight",
    ]
else:
    X_set = None
    X_set = BasicDataset.X_set + [
        "ELO",
    ]

# %% [markdown]
# ----

# %%
fight_ids = self.data["fight_id"].unique()

# %%
invalid_fights = set(
    self.data_aggregated[self.data_aggregated["num_fight"] < 4]["fight_id"]
)  # The usual is 4

# invalid_fights |= set(self.data_aggregated[self.data_aggregated["event_date"] < "2013-01-01"]["fight_id"])

# %%
early_split_date = "2018-01-01"
split_date = "2023-05-01"  # "2023-08-01"
max_date = "2024-11-01"

early_train_fights = self.data["fight_id"][self.data["event_date"] < split_date]

train_fights = self.data["fight_id"][
    (
        (self.data["event_date"] < split_date)
        & (self.data["event_date"] >= early_split_date)
    )
]


test_fights = self.data["fight_id"][
    (self.data["event_date"] >= split_date) & (self.data["event_date"] <= max_date)
]

early_train_fights = set(early_train_fights) - set(invalid_fights)
train_fights = set(train_fights) - set(invalid_fights)
test_fights = set(test_fights) - set(invalid_fights)

# %%
# Now I generate a data_processor specifically for training, so
# I avoid any possible contamination from the test sample:
train_data_processor = DataProcessor(**data_processor_kwargs)
train_data_processor.scraper.event_scraper.data = (
    train_data_processor.scraper.event_scraper.data[
        pd.to_datetime(train_data_processor.scraper.event_scraper.data["event_date"])
        < pd.to_datetime(split_date)
    ]
)

train_data_processor.load_data()
train_data_processor.aggregate_data()
train_data_processor.add_per_minute_and_fight_stats()
train_data_processor.normalize_data()

# %%
early_train_dataset = BasicDataset(
    train_data_processor,
    early_train_fights,
    X_set=X_set,
)

train_dataset = BasicDataset(
    train_data_processor,
    train_fights,
    X_set=X_set,
)

test_dataset = BasicDataset(
    data_processor,
    test_fights,
    X_set=X_set,
)

# %%
early_train_dataloader = torch.utils.data.DataLoader(
    early_train_dataset, batch_size=256, shuffle=True
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=256, shuffle=False
)

# %%
from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F
from torch import nn
from ufcpredictor.loss_functions import BettingLoss


# %%
from typing import List


# %%
class FighterNet(nn.Module):
    """
    A neural network model designed to predict the outcome of a fight based on a single
    fighter's characteristics.

    The model takes into account the characteristics of the fighter and the odds of the
    fight. It can be used to make predictions on the outcome of a fight and to
    calculate the benefit of a bet.
    """

    params: List[str] = [
        "dropout_prob",
    ]

    def __init__(self, input_size: int, dropout_prob: float = 0.0) -> None:
        super(FighterNet, self).__init__()

        layer_sizes = [input_size, 128, 256, 512, 256, 127]
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p=dropout_prob) for _ in range(len(layer_sizes) - 1)]
        )

        self.dropout_prob = dropout_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc, bn, dropout in zip(self.layers, self.bns, self.dropouts):
            x = F.relu(bn(fc(x)))
            x = dropout(x)

        return x


class SymmetricFightNet(nn.Module):
    """
    A neural network model designed to predict the outcome of a fight between two
    fighters.

    The model takes into account the characteristics of both fighters and the odds of
    the fight. It uses a symmetric architecture to ensure that the model is fair and
    unbiased towards either fighter.

    The model can be used to make predictions on the outcome of a fight and to calculate
    the benefit of a bet.
    """

    params: List[str] = [
        "dropout_prob",
    ]

    def __init__(self, input_size: int, dropout_prob: float = 0.0) -> None:
        super(SymmetricFightNet, self).__init__()
        self.fighter_net = FighterNet(input_size=input_size, dropout_prob=dropout_prob)

        layer_sizes = [257, 512, 128, 64, 1]
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p=dropout_prob) for _ in range(len(layer_sizes) - 1)]
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout_prob = dropout_prob

    def forward(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        odds1: torch.Tensor,
        odds2: torch.Tensor,
    ) -> torch.Tensor:
        out1 = self.fighter_net(X1[:, :-1])
        out2 = self.fighter_net(X2[:, :-1])

        prob1 = 1 / odds1
        prob2 = 1 / odds2
        prob1 = prob1 / (prob1 + prob2)
        prob2 = 1 - prob1

        out1 = torch.cat((out1, prob1), dim=1)  # Appending the weight there.
        out2 = torch.cat((out2, prob2), dim=1)

        x = torch.cat((out1, out2, X1[:, -1:]), dim=1)
        # x = torch.cat((out1 - out2, out2 - out1, X1[:, -1:]), dim=1)

        for fc, bn, dropout in zip(self.layers[:-1], self.bns, self.dropouts):
            x = F.relu(bn(fc(x)))
            x = dropout(x)

        x = self.sigmoid(self.layers[-1](x))
        return x


# %%

# %%
seed = 20
torch.manual_seed(seed)
import random

random.seed(seed)
np.random.seed(seed)

# %%
# run = mlflow.start_run()

# mlflow.log_params(
#     {
#         "seed": seed,
#         "date_early_split": early_split_date,
#         "date_split": split_date,
#         "date_max": max_date,
#     }
# )
model = SymmetricFightNet(
    input_size=len(train_dataset.X_set) - 1,
    dropout_prob=0.35,  # 0.35
)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=2e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.7, patience=2
)

trainer = Trainer(
    train_loader=train_dataloader,
    test_loader=test_dataloader,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=BettingLoss(),
    mlflow_tracking=False,
)

trainer.train(
    epochs=5,
    train_loader=early_train_dataloader,
    test_loader=test_dataloader,
)

trainer.train(epochs=20)  # ~8 is a good match if dropout to 0.35

# %%

# %%
# mlflow.pytorch.log_model(model, "model")

# %%

# %%

# %%
# Save model dict

torch.save(model.state_dict(), "model.pth")

# %%
from ufcpredictor.plot_tools import PredictionPlots
import matplotlib.pyplot as plt

# %%
fig, ax = plt.subplots()

stats = PredictionPlots.show_fight_prediction_detail_from_dataset(
    model=trainer.model,
    dataset=test_dataset,
    fight_ids=None,
    print_info=False,
    show_plot=True,
    ax=ax,
)

ax.set_ylim(-10, 30)
ax.grid()

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

# %%
fig, ax = plt.subplots()


results = df.groupby("event_date")[["bet", "win"]].sum().reset_index()
ax.plot(
    results["event_date"],
    np.cumsum(results["win"] - results["bet"]),  # / np.cumsum(results["bet"]) * 100,
    ls="--",
    c="k",
)


# Generate evenly spaced colors from the rainbow colormap
# unique_weight_classes = df["weight_class"].unique()
weight_classes = [
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
]
colors = plt.cm.rainbow(np.linspace(0, 1, len(weight_classes)))

# Plot each weight class with its unique color
for color, weight_class in zip(colors, reversed(weight_classes)):
    results = (
        df[df["weight_class"] == weight_class]
        .groupby("event_date")[["win", "bet"]]
        .sum()
    )
    cumsum_values = np.cumsum(results["win"] - results["bet"])
    ax.plot(results.index, cumsum_values, label=weight_class, color=color)

    # Place label at the last point of each line
    last_event_date = results.index[-1]
    last_cumsum_value = cumsum_values.iloc[-1]
    ax.text(
        last_event_date,
        last_cumsum_value,
        weight_class,
        color=color,
        fontsize=8,
        va="center",
    )

# ax.set_ylim(-20, 20)
# Put x labels rotated 90 degrees
ax.tick_params(axis="x", labelrotation=90)
ax.set_ylim(None, None)
ax.grid()
ax.axhline(0, color="black")
ax.legend()

# %% [raw]
#

# %%

# %%

# %%
X1, X2, Y, odds1, odds2, _, _ = test_dataset.get_fight_data_from_ids(None)

X1.requires_grad = True
X2.requires_grad = True
odds1.requires_grad = True
odds2.requires_grad = True

output = model(X1, X2, odds1.reshape(-1, 1), odds2.reshape(-1, 1))
output.sum().backward()

# %%
odds1.grad.sum()

# %%
fig, ax = plt.subplots(figsize=(5, 12))

labels = test_dataset.X_set + ["odds"]
values = list(abs(X1.grad.sum(axis=0))) + [
    odds1.grad.sum(),
]

sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
labels = [labels[i] for i in sorted_indices]
values = [values[i] for i in sorted_indices]

ax.barh(labels, values)
ax.grid()

# %%

# %%

# %%
df

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
