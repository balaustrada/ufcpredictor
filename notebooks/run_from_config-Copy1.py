# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from ufcpredictor import UFCPredictor
predictor = UFCPredictor("/home/cramirpe/UFC/ufcpredictor/config.yaml")

# %%
import jupyter_black

jupyter_black.load()

# %%
import importlib

# %%
import ufcpredictor
from ufcpredictor.trainer import Trainer

# %%
import torch

import random
import numpy as np

# %%
from pathlib import Path
import yaml
from datetime import datetime
import pandas as pd

config = yaml.safe_load(Path("/home/cramirpe/UFC/ufcpredictor/config.yaml").read_text())

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
from ufcpredictor import UFCPredictor

predictor = UFCPredictor("/home/cramirpe/UFC/ufcpredictor/config.yaml")

data_processor = predictor.data_processor


# %%
def load_trainer(self) -> None:
    """
    Loads the trainer with the datasets, model, optimizer, scheduler, and loss function.
    """
    fight_ids = self.data_processor.data["fight_id"].unique()

    if self.config.get("filters", {}).get("minimum fight number", 0) > 0:
        invalid_fights = set(
            self.data_processor.data[self.data_processor.data["num_fight"] < 5][
                "fight_id"
            ]
        )
    else:
        invalid_fights = set()

    # TODO check this, because we are using the inverse and naming it the same.
    # Consider creating an inverse, or whatever, maybe introduce it inside
    # of data_enhancer. with factor -1 (?)
    # minimum_notice_days = self.config.get("filters", {}).get("minimum notice days", 0)
    # if minimum_notice_days > 0:
    #     invalid_fights.update(
    #         self.data_processor.data[
    #             self.data_processor.data["notice_days"] > 1 / minimum_notice_days
    #         ]["fight_id"]
    #     )
    invalid_fights.update(
        self.data_processor.data[self.data_processor.data["notice_days"] != 1 / 60][
            "fight_id"
        ]
    )

    early_split_date = pd.to_datetime(
        self.config.get("filters", {}).get("early split date", None)
    )
    split_date = pd.to_datetime(self.config["filters"]["split date"])
    max_date = self.config.get("filters", {}).get(
        "max_date", datetime.now().strftime("%Y-%m-%d")
    )

    if early_split_date is not None:
        early_train_fights = self.data_processor.data["fight_id"][
            self.data_processor.data["event_date"] < split_date
        ]
        train_fights = self.data_processor.data["fight_id"][
            (self.data_processor.data["event_date"] < split_date)
            & (self.data_processor.data["event_date"] >= early_split_date)
        ]
    else:
        train_fights = self.data_processor.data["fight_id"][
            self.data_processor.data["event_date"] < split_date
        ]
        early_train_fights = set()

    test_fights = self.data_processor.data["fight_id"][
        (self.data_processor.data["event_date"] >= split_date)
        & (self.data_processor.data["event_date"] <= max_date)
    ]

    early_train_fights = set(early_train_fights) - set(invalid_fights)
    train_fights = set(train_fights) - set(invalid_fights)
    test_fights = set(test_fights) - set(invalid_fights)

    self.early_train = len(early_train_fights) > 0

    # Loading datasets
    dataset_cfg = self.config.get("dataset", {})

    if dataset_cfg.get("class") is not None:
        Dataset = getattr(ufcpredictor.datasets, dataset_cfg.get("class"))
    else:
        raise ValueError("No dataset specified in self.config file")

    if self.early_train:
        self.early_train_dataset = early_train_dataset = Dataset(
            data_processor=self.data_processor,
            fight_ids=early_train_fights,
            **dataset_cfg.get("args", {}),
        )

    self.train_dataset = train_dataset = Dataset(
        data_processor=self.data_processor,
        fight_ids=train_fights,
        **dataset_cfg.get("args", {}),
    )

    self.test_dataset = test_dataset = Dataset(
        data_processor=self.data_processor,
        fight_ids=test_fights,
        **dataset_cfg.get("args", {}),
    )

    # Initialize dataloaders

    batch_size = self.config["training"]["batch size"]

    if self.early_train:
        self.early_train_dataloader = torch.utils.data.DataLoader(
            early_train_dataset, batch_size=batch_size, shuffle=True
        )

    self.train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    self.test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Setting random seed for reproducibility
    seed = self.config["training"]["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Loading model


# %%

# %%
load_trainer(predictor)
early_train_dataset = predictor.early_train_dataset
train_dataset = predictor.train_dataset
test_dataset = predictor.test_dataset

early_train = True

early_train_dataloader = predictor.early_train_dataloader
train_dataloader = predictor.train_dataloader
test_dataloader = predictor.test_dataloader

# %%
# Loading model

self = predictor

model_cfg = config.get("model", {})

if model_cfg.get("class") is not None:
    self.model = getattr(ufcpredictor.models, model_cfg["class"])(
        **model_cfg.get("args", {}),
    )
else:
    raise ValueError("Model class not defined")

# Loading optimimzer
optimizer_cfg = self.config.get("optimizer", {})

if optimizer_cfg.get("class") is not None:
    optimizer = getattr(ufcpredictor.optimizers, optimizer_cfg["class"])(
        self.model.parameters(),
        **optimizer_cfg.get("args", {}),
    )
else:
    raise ValueError("Optimizer class not defined")

# Loading scheduler
scheduler_cfg = self.config.get("scheduler", {})

if scheduler_cfg.get("class") is not None:
    scheduler = getattr(ufcpredictor.schedulers, scheduler_cfg["class"])(
        optimizer,
        **scheduler_cfg.get("args", {}),
    )
else:
    raise ValueError("Scheduler class not defined")

# Loading loss
loss_cfg = self.config.get("loss", {})

if loss_cfg.get("class") is not None:
    loss = getattr(ufcpredictor.loss_functions, loss_cfg["class"])(
        **loss_cfg.get("args", {}),
    )
else:
    raise ValueError("Loss class not defined")

trainer = trainer = Trainer(
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    model=self.model,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss,
    mlflow_tracking=False,
    device=device,
)


# %%
def load_models(self):

    # Setting random seed for reproducibility
    seed = predictor.config["training"]["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model_cfg = self.config.get("model", {})

    if model_cfg.get("class") is not None:
        self.model = getattr(ufcpredictor.models, model_cfg["class"])(
            **model_cfg.get("args", {}),
        )
    else:
        raise ValueError("Model class not defined")

    # Loading optimimzer
    optimizer_cfg = self.config.get("optimizer", {})

    if optimizer_cfg.get("class") is not None:
        optimizer = getattr(ufcpredictor.optimizers, optimizer_cfg["class"])(
            self.model.parameters(),
            **optimizer_cfg.get("args", {}),
        )
    else:
        raise ValueError("Optimizer class not defined")

    # Loading scheduler
    scheduler_cfg = self.config.get("scheduler", {})

    if scheduler_cfg.get("class") is not None:
        scheduler = getattr(ufcpredictor.schedulers, scheduler_cfg["class"])(
            optimizer,
            **scheduler_cfg.get("args", {}),
        )
    else:
        scheduler = None

    # Loading loss
    loss_cfg = self.config.get("loss", {})

    if loss_cfg.get("class") is not None:
        loss = getattr(ufcpredictor.loss_functions, loss_cfg["class"])(
            **loss_cfg.get("args", {}),
        )
    else:
        raise ValueError("Loss class not defined")

    self.trainer = Trainer(
        train_dataloader=self.train_dataloader,
        test_dataloader=self.test_dataloader,
        model=self.model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss,
        mlflow_tracking=False,
        device=self.device,
    )


# load_models(predictor)
# trainer = predictor.trainer

# %%

# %%

# %%
trainer.train(
    epochs=1,
    train_dataloader=early_train_dataloader,
    test_dataloader=test_dataloader,
)

# %%

# %%

# %%
if early_train:
    trainer.train(
        epochs=config["training"]["early train epochs"],
        train_dataloader=early_train_dataloader,
        test_dataloader=test_dataloader,
    )

trainer.train(epochs=config["training"]["train epochs"], test_dataloader=test_dataloader)

# %%

# %%

# %%
import matplotlib.pyplot as plt
from ufcpredictor.plot_tools import PredictionPlots

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
df["confidence"] = abs((df["Prediction"] - 0.5) * 2)

# df = df[df["confidence"] > ]

cash0 = 100

df = df.sort_values(by="event_date")

cash = [
    cash0,
]
invest = [
    cash0,
]
dates = [
    None,
]

print("Max confidence: ", df["confidence"].max())
print("Max bet: ", df["bet"].max())


for date, group in df.groupby("event_date"):
    # max_bet = max(cash[-1] * 0.5, 10)

    # win = (group["confidence"]*group["win"]).sum() * max_bet / 10 / group["confidence"].sum()
    # bet = (group["confidence"]*group["bet"]).sum() * max_bet / 10 / group["confidence"].sum()

    # extra_added = max(bet - cash[-1], 0)
    # cash_i = cash[-1] + win - min(bet, cash[-1])

    # invest.append(invest[-1] + extra_added)
    # cash.append(cash_i)
    # dates.append(date)

    max_bet = max(cash[-1] * 0.1, 20) / df["confidence"].max()
    win = (group["confidence"] * group["win"] * max_bet / 10).sum()
    bet = (group["confidence"] * group["bet"] * max_bet / 10).sum()

    if bet > max_bet:
        win = win / bet * max_bet
        bet = max_bet

    extra_added = max(bet - cash[-1], 0)
    cash_i = cash[-1] + win - min(bet, cash[-1])

    invest.append(invest[-1] + extra_added)
    cash.append(cash_i)
    dates.append(date)


cash = cash[1:]
invest = invest[1:]
dates = dates[1:]


# %%
import re

# Check which events are numbered
event_df = data_processor.scraper.event_scraper.data

event_df["isnumbered"] = event_df["event_name"].apply(
    lambda x: re.search(r"\d+", x) is not None
)

df = df.merge(
    event_df[["event_date", "isnumbered"]],
    how="left",
)

# %%
# group by weith_class, show sum of bet and win
# I want to show also mean bet
grouped_df = df.groupby(["weight_class"]).agg(
    bet_sum=("bet", "sum"),
    win_sum=("win", "sum"),
    confidence_mean=("confidence", "mean"),
    count=("bet", "count"),
)

grouped_df["win_percentage"] = grouped_df["win_sum"] / grouped_df["bet_sum"]
grouped_df["bet_per_fight"] = grouped_df["bet_sum"] / grouped_df["count"]
grouped_df = grouped_df.reset_index()

classes = [
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
]
grouped_df["weight_class"] = pd.Categorical(
    grouped_df["weight_class"], categories=classes, ordered=True
)

# use the custom order above
grouped_df = grouped_df.sort_values(by=["weight_class"])

grouped_df

# %%
# groupby isnumbered, show sum of bet and win
grouped_df = df.groupby(["isnumbered"]).agg({"bet": "sum", "win": "sum"})
grouped_df["win_percentage"] = (grouped_df["win"] / grouped_df["bet"]) * 100
grouped_df

# %%
fig, ax = plt.subplots()

ax.plot(
    dates,
    invest,
    label="invest",
)

ax.plot(
    dates,
    cash,
    label="cash",
)

ax.plot(
    dates,
    [x - y for x, y in zip(cash, invest)],
    label="profit",
)

ax.axhline(0, c="k")
ax.tick_params(axis="x", labelrotation=45)


ax.legend()
ax.grid()

# %%

# %% [markdown]
# ## Forecasts

# %%
from ufcpredictor.utils import (
    pad_or_truncate,
    convert_odds_to_decimal,
    convert_odds_to_moneyline,
)
from ufcpredictor.utils_sheets import read_fights_sheet
from datetime import datetime

# %%
credentials_file = "/home/cramirpe/UFC/gsheets_token.json"
spreadsheet_id = "1sBdtiCPPJyMupocgZsaNLZaX4rGoC4XIOs6k0la3cpk"

fields_to_extract = {
    "Date": "datetime64[D]",
    "Fighter Name": str,
    "Opponent Name": str,
    "Fighter Odds": int,
    "Opponent Odds": int,
    "Weight": int,
    "Rounds": int,
}

(
    event_dates,
    fighter_names,
    opponent_names,
    fighter_odds,
    opponent_odds,
    weight,
    rounds,
) = read_fights_sheet(
    spreadsheet_id=spreadsheet_id,
    creds_file=credentials_file,
    fields_to_read=fields_to_extract.keys(),
    dtypes_=list(fields_to_extract.values()),
)

fighter_odds = convert_odds_to_decimal(fighter_odds)
opponent_odds = convert_odds_to_decimal(opponent_odds)

event_dates = list(event_dates)
fight_parameters_values = [[w, r] for w, r in zip(weight, rounds)]

# %%
self = forecast_dataset

fighter_ids = [self.data_processor.get_fighter_id(x) for x in fighter_names]
opponent_ids = [self.data_processor.get_fighter_id(x) for x in opponent_names]


counts = self.data_processor.data_normalized["fighter_id"].value_counts()
fighter_counts = pd.Series(fighter_ids + opponent_ids).map(counts).fillna(0).astype(int)
fighter_counts = pd.DataFrame(
    {
        "name": [
            self.data_processor.get_fighter_name(id_)
            for id_ in fighter_ids + opponent_ids
        ],
        "count": fighter_counts,
    }
)

print(fighter_counts.sort_values(by="count"))
invalid_fighters = fighter_counts[fighter_counts["count"] <= 3]["name"].to_list()
invalid_fighters_count = fighter_counts[fighter_counts["count"] <= 3]["count"].to_list()

# %%
p1, p2 = forecast_dataset.get_forecast_prediction(
    fighter_names,
    opponent_names,
    event_dates,
    fighter_odds,
    opponent_odds,
    model,
    fight_parameters_values,
    parse_ids=False,
    device=device,
)

# %%

# %%
value = (p1 + p2) / 2
confidence = abs((value - 0.5) * 2)

# %%
max_bet = 20
confidence = abs(p1 + p2 - 1)
bet = max_bet / 10 * confidence
bet = bet.numpy().flatten().round(2)

# %%

# %%
import sys


# %%
for f, o, fightfeat, p1h, p2h, beth, fodds, oodds in zip(
    fighter_names,
    opponent_names,
    fight_parameters_values,
    p1,
    p2,
    bet,
    fighter_odds,
    opponent_odds,
):
    fodds = convert_odds_to_moneyline(fodds)
    oodds = convert_odds_to_moneyline(oodds)

    if f in invalid_fighters:
        f_inv = "NOT ENOUGH FIGHTS"
        fc = invalid_fighters_count[invalid_fighters.index(f)]
    else:
        f_inv = ""
        fc = ""

    if o in invalid_fighters:
        o_inv = "NOT ENOUGH FIGHTS"
        oc = invalid_fighters_count[invalid_fighters.index(o)]
    else:
        o_inv = ""
        oc = ""

    print(
        f"\t{f}({fodds:d})\t{f_inv}  {fc}\n\t{o}({oodds:d})\t{o_inv}  {oc}\n\t{fightfeat[0]:d}\t{fightfeat[1]:d}\n\t{(p1h[0] + p2h[0]) / 2:.5f}+-{abs(p1h[0]-p2h[0]):.5f}\n"
        f"\tSuggested bet: {beth:.2f}\n"
    )

# %%
bet.sum()

# %%
