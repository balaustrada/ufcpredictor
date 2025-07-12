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
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# %%
import os


# %%
import mlflow
import mlflow.pytorch
import torch

# Enable autologging for PyTorch
# mlflow.pytorch.autolog()

# mlflow.set_tracking_uri("http://127.0.0.1:5000") 
# mlflow.set_experiment('Diferent tries')

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
import pandas as pd
pd.set_option("display.max_columns", None)

# %%
from ufcpredictor.utils import convert_odds_to_decimal

# %%
from ufcpredictor.data_processor import DataProcessor
from ufcpredictor.data_enhancers import SumFlexibleELO, RankedFields
from ufcpredictor.data_aggregator import WeightedDataAggregator
from ufcpredictor.datasets import DatasetWithTimeEvolution, ForecastDatasetTimeEvolution
from ufcpredictor.trainer import Trainer
from ufcpredictor.plot_tools import PredictionPlots
import torch
import numpy as np

import matplotlib.pyplot as plt

# %%
# DataProcessor = ELODataProcebssor
# data_processor_kwargs = {
#     "data_folder": "/home/cramirpe/UFC/UFCfightdata",
#     # "scaling_factor": 0.5,
#     # "boost_values": [1, 2, 3],
#     # "K_factor": 30,
# }

data_processor_kwargs = {
    # "data_folder": "/home/cramirez/kaggle/ufc_scraper/UFCfightdata",
    "data_folder": "/home/cramirpe/UFC/UFCfightdata",
    "data_aggregator": WeightedDataAggregator(alpha=-0.0001),
    "data_enhancers": [
        SumFlexibleELO(
            scaling_factor=0, #0.5
            K_factor = 30, # 30
        ),
        RankedFields(
            fields=["age", "fighter_height_cm"],
            exponents=[1.2, 1.2],
        ),
    ],
}

# %%
data_processor = DataProcessor(
    **data_processor_kwargs
)

# %%
if True:
    fighter_fight_statistics = [
        "age",
        # "notice_days",
        # "body_strikes_att_opponent_per_minute",
        # "body_strikes_att_per_minute",
        "body_strikes_succ_opponent_per_minute",
        "body_strikes_succ_per_minute",
        # "clinch_strikes_att_opponent_per_minute",
        # "clinch_strikes_att_per_minute",
        "clinch_strikes_succ_opponent_per_minute",
        "clinch_strikes_succ_per_minute",
        "ctrl_time_opponent_per_minute",
        "ctrl_time_per_minute",
        # "distance_strikes_att_opponent_per_minute",
        # "distance_strikes_att_per_minute",
        "distance_strikes_succ_opponent_per_minute",
        "distance_strikes_succ_per_minute",
        "fighter_height_cm",
        # "ground_strikes_att_opponent_per_minute",
        # "ground_strikes_att_per_minute",
        "ground_strikes_succ_opponent_per_minute",
        "ground_strikes_succ_per_minute",
        # "head_strikes_att_opponent_per_minute",
        # "head_strikes_att_per_minute",
        "head_strikes_succ_opponent_per_minute",
        "head_strikes_succ_per_minute",
        "knockdowns_opponent_per_minute",
        "knockdowns_per_minute",
        # "KO_opponent_per_fight",
        "KO_opponent_per_minute",
        "KO_per_fight",
        "KO_per_minute",
        # "leg_strikes_att_opponent_per_minute",
        # "leg_strikes_att_per_minute",
        "leg_strikes_succ_opponent_per_minute",
        "leg_strikes_succ_per_minute",
        "num_fight",
        # "reversals_opponent_per_minute",
        # "reversals_per_minute",
        # "strikes_att_opponent_per_minute",
        # "strikes_att_per_minute",
        "strikes_succ_opponent_per_minute",
        "strikes_succ_per_minute",
        # "Sub_opponent_per_fight",
        "Sub_opponent_per_minute",
        # "Sub_per_fight",
        "Sub_per_minute",
        "submission_att_opponent_per_minute",
        "submission_att_per_minute",
        # "takedown_att_opponent_per_minute",
        # "takedown_att_per_minute",
        "takedown_succ_opponent_per_minute",
        "takedown_succ_per_minute",
        "time_since_last_fight",
        # "total_strikes_att_opponent_per_minute",
        # "total_strikes_att_per_minute",
        "total_strikes_succ_opponent_per_minute",
        "total_strikes_succ_per_minute",
        "win_opponent_per_fight",
        "win_per_fight",
        "ELO",
    ]
else:
    fighter_fight_statistics = None
    fighter_fight_statistics = BasicDataset.fighter_fight_statistics + [
        "ELO",
    ]

previous_fights_statistics = [
        "age",
        # "notice_days",
        # "body_strikes_att_opponent_per_minute",
        # "body_strikes_att_per_minute",
        "body_strikes_succ_opponent_per_minute",
        "body_strikes_succ_per_minute",
        # "clinch_strikes_att_opponent_per_minute",
        # "clinch_strikes_att_per_minute",
        "clinch_strikes_succ_opponent_per_minute",
        "clinch_strikes_succ_per_minute",
        "ctrl_time_opponent_per_minute",
        "ctrl_time_per_minute",
        # "distance_strikes_att_opponent_per_minute",
        # "distance_strikes_att_per_minute",
        "distance_strikes_succ_opponent_per_minute",
        "distance_strikes_succ_per_minute",
        # "fighter_height_cm",
        # "ground_strikes_att_opponent_per_minute",
        # "ground_strikes_att_per_minute",
        "ground_strikes_succ_opponent_per_minute",
        "ground_strikes_succ_per_minute",
        # "head_strikes_att_opponent_per_minute",
        # "head_strikes_att_per_minute",
        "head_strikes_succ_opponent_per_minute",
        "head_strikes_succ_per_minute",
        "knockdowns_opponent_per_minute",
        "knockdowns_per_minute",
        # "KO_opponent_per_fight",
        "KO_opponent_per_minute",
        # "KO_per_fight",
        "KO_per_minute",
        # "leg_strikes_att_opponent_per_minute",
        # "leg_strikes_att_per_minute",
        "leg_strikes_succ_opponent_per_minute",
        "leg_strikes_succ_per_minute",
        # "num_fight",
        # "reversals_opponent_per_minute",
        # "reversals_per_minute",
        # "strikes_att_opponent_per_minute",
        # "strikes_att_per_minute",
        "strikes_succ_opponent_per_minute",
        "strikes_succ_per_minute",
        # "Sub_opponent_per_fight",
        "Sub_opponent_per_minute",
        # "Sub_per_fight",
        "Sub_per_minute",
        "submission_att_opponent_per_minute",
        "submission_att_per_minute",
        # "takedown_att_opponent_per_minute",
        # "takedown_att_per_minute",
        "takedown_succ_opponent_per_minute",
        "takedown_succ_per_minute",
        "time_since_last_fight", # Adding this somehow slowed the convergence and is not as good (why?) maybe because of the default value used(?) it was the mean (~ 7months)
        # "total_strikes_att_opponent_per_minute",
        # "total_strikes_att_per_minute",
        "total_strikes_succ_opponent_per_minute",
        "total_strikes_succ_per_minute",
        # "win_opponent_per_fight",
        # "win_per_fight",
        "ELO",
]

# %%
len(fighter_fight_statistics)

# %%
data_processor.load_data()
data_processor.aggregate_data()
data_processor.add_per_minute_and_fight_stats()

# for field in fighter_fight_statistics:
#     if field in ["ELO", "age"]:
#         continue
#     self.data_aggregated[field] = (self.data_aggregated[field].rank(pct=True) * 100) ** 1.2

data_processor.normalize_data()

# %% [markdown]
# ___

# %% [markdown]
# ----

# %%

# %%
fight_ids = data_processor.data["fight_id"].unique()

# %%
invalid_fights = set(data_processor.data[data_processor.data["num_fight"] < 5]["fight_id"]) # The usual is 4

# invalid_fights |= set(self.data_aggregated[self.data_aggregated["event_date"] < "2013-01-01"]["fight_id"])

# %%
len(invalid_fights)

# %%
invalid_fights.update(
    data_processor.data[
        data_processor.data["notice_days"] != 1/60
    ]["fight_id"]
)

# %%
len(invalid_fights)

# %%
early_split_date = "2018-01-01"#"2017-01-01"
split_date = "2023-08-01"#"2023-08-01"
max_date = "2025-11-11" 

early_train_fights = data_processor.data["fight_id"][data_processor.data["event_date"] < split_date]

train_fights = data_processor.data["fight_id"][(
        (data_processor.data["event_date"] < split_date)
        & (data_processor.data["event_date"] >= early_split_date)
    )
]

    
test_fights  = data_processor.data["fight_id"][(data_processor.data["event_date"] >= split_date) & (data_processor.data["event_date"] <= max_date)]

early_train_fights = set(early_train_fights) - set(invalid_fights)
train_fights = set(train_fights) - set(invalid_fights)
test_fights = set(test_fights) - set(invalid_fights)

# %%
from ufcpredictor.models import SimpleFightNetWithTimeEvolution
from ufcpredictor.loss_functions import BettingLoss


# %%
status_array_size = 10
fight_parameters = ["num_rounds", "weight"]
previous_fights_parameters = ["num_rounds", "weight", "winner"]

# fight_parameters = []
early_train_dataset = DatasetWithTimeEvolution(
    data_processor,
    early_train_fights,
    fighter_fight_statistics=fighter_fight_statistics,
    fight_parameters=fight_parameters,
    previous_fights_statistics=previous_fights_statistics,
    previous_fights_parameters=previous_fights_parameters,
    status_array_size=status_array_size,
)

train_dataset = DatasetWithTimeEvolution(
    data_processor,
    train_fights,
    fighter_fight_statistics=fighter_fight_statistics,
    fight_parameters=fight_parameters,
    previous_fights_statistics=previous_fights_statistics,
    previous_fights_parameters=previous_fights_parameters,
    status_array_size=status_array_size,
)

test_dataset = DatasetWithTimeEvolution(
    data_processor,
    test_fights,
    fighter_fight_statistics=fighter_fight_statistics,
    fight_parameters=fight_parameters,
    previous_fights_statistics=previous_fights_statistics,
    previous_fights_parameters=previous_fights_parameters,
    status_array_size=status_array_size,
)

forecast_dataset = ForecastDatasetTimeEvolution(
    data_processor=data_processor,
    fighter_fight_statistics=fighter_fight_statistics,
    fight_parameters=fight_parameters,
    previous_fights_statistics=previous_fights_statistics,
    previous_fights_parameters=previous_fights_parameters,
    status_array_size=status_array_size,
)

# %%

# %%
batch_size = 32  # 64 # 2048
early_train_dataloader = torch.utils.data.DataLoader(
    early_train_dataset, batch_size=batch_size, shuffle=True
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

# %%

# %%
seed = 4
torch.manual_seed(seed)
import random

random.seed(seed)
np.random.seed(seed)

# %%

# %%
dropout = 0.4  # 0.35 seemed to work good, but also 0.45 or even 0.5
model = SimpleFightNetWithTimeEvolution(
    input_size=2*len(fighter_fight_statistics)+ len(fight_parameters) + 2 + 2*status_array_size, # 2 are the odds,
    # input_size_f=len(fight_parameters),
    dropout_prob=dropout,
    # fighter_network_shape=[256, 512, 1024, 512],
    # network_shape=[2048, 1024, 512, 128, 64, 1],
    # network_shape=[122, 1024, 2048, 1024, 512, 256, 128, 64, 1],
    # network_shape=[512,1024, 512, 256, 128, 64, 1],
    # network_shape=[256, 512, 256, 128, 64, 1],  # This was the best one so far
    # network_shape=[512, 128, 64, 1],
    network_shape=[128, 64, 32, 1],
    status_array_size=status_array_size,
    # network_shape=[122, 1024, 512, 1024, 512, 256, 128, 64, 1],
    fighter_transformer_kwargs=dict(
        state_dim=status_array_size,  # 20,
        stat_dim=len(previous_fights_statistics),
        fight_parameters_size=len(previous_fights_parameters),
        layer_sizes=[128, 64],
        # layer_sizes=[128, 64, 10], # This better(?)
        # layer_sizes=[128, 512, 256, 128, 64, 10], # This worked
        dropout=dropout * 0.9,
    ),
)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-3, weight_decay=2e-5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.9, patience=4
# )
# mlflow.end_run()
# mlflow.start_run()

optimizer = torch.optim.Adam(
    params=model.parameters(), lr=1.3e-3, weight_decay=1e-5#1e-5
)  # , weight_decay=2e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.7, patience=2
)

trainer = Trainer(
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=BettingLoss(),
    mlflow_tracking=False,
    device=device,
)

# %%

trainer.train(
    epochs=10,
    train_dataloader=early_train_dataloader,
    test_dataloader=test_dataloader,
)

# %%
trainer.train(epochs=10)  # ~8 is a good match if drÇopout to 0.35

# %%
# Save modeÇdict
# torch.save(model.state_dict(), 'model.pth')

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

# %%
from datetime import date

# %%
self = forecast_dataset

# %%
forecast_dataset.update_data_trans(model.evolver)

# %%
from ufcpredictor.utils import pad_or_truncate, convert_odds_to_decimal, convert_odds_to_moneyline

padding = 20

# %%
convert_odds_to_decimal([100,])

# %%
fighter_names = [
    "Ilia Topuria",
    "Alexandre Pantoja",
    "Brandon Royval",
    "Beneil Dariush",
    "Payton Talbott",
    "Jack Hermansson",
    # "Hyder Amil",
    # "Viviane Araujo",
    "Terrance McKinney",
    # "Sedriques Dumas",
    # "Jhonata Diniz",
    # "Niko Price",
]
fighter_odds = convert_odds_to_decimal([
    -425,#-130,
    -265,
    -125,
    136,
    150,
    160,
    -185,
])
opponent_names = [
    "Charles Oliveira",
    "Kai Kara-France",
    "Joshua Van",
    "Renato Moicano",
    "Felipe Lima",
    "Gregory Rodrigues",
    # "Jose Delgado",
    # "Tracy Cortez",
    "Viacheslav Borshchev",
    # "Jackson McVey",
    # "Alvin Hines",
    # "Jacobe Smith"
]
opponent_odds = convert_odds_to_decimal([
    330,
    215,
    105,
    -162,
    -185,
    -190,
    150,
])
fight_parameters_values = [
    [5, 155],
    [5, 125],
    [3, 125],
    [3, 155], 
    [3, 135],
    [3, 185],
    # [3, 145],
    # [3, 115],
    [3, 155],
    # [3, 185],
    # [3, 265],
    # [3, 170],
]
event_dates = [date(2025, 6, 20)] * len(fighter_names)
# fighter_odds = [1] * len(fighter_names)
# opponent_odds = [1] * len(opponent_names)
model = trainer.model
parse_ids: bool = False
device: str = "cpu"
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

# %%

# %%
p1, p2 = forecast_dataset.get_forecast_prediction(
    fighter_names,
    opponent_names,
    event_dates,
    fighter_odds,
    opponent_odds,
    model,
    fight_parameters_values,
    parse_ids = False,
    device= device
)

# %%

# %%
value = (p1 + p2)/2
confidence = abs((value - 0.5)*2)

# %%
max_bet = 20
confidence = abs(p1 + p2 -1)
bet =  max_bet / 10 *confidence
bet = bet.numpy().flatten().round(2)

# %%
for f, o, fightfeat, p1h, p2h, beth, fodds, oodds in zip(
    fighter_names, opponent_names, fight_parameters_values, p1, p2, bet, fighter_odds, opponent_odds,
):
    fodds = convert_odds_to_moneyline(fodds)
    oodds = convert_odds_to_moneyline(oodds)
    
    print(
        f"\t{f}({fodds})\n\t{o}({oodds})\n\t{fightfeat}\n\t{(p1h[0] + p2h[0]) / 2:.5f}+-{abs(p1h[0]-p2h[0]):.5f}\n"
        f"\tSuggested bet: {2*beth:.2f}\n"
    )

# %%
bet.sum()

# %%

# %%

# %%

# %% [markdown]
# ### El problema es como pasar los diferentes pasos...

# %%

# %%

# %%

# %%
len(forecast_dataset.data)

# %%

# %%
names_f = ["Maheshate", "Nyamjargal Tumendemberel", "Shi ming", "Kiru Sahota", "Baergeng Jieleyisi","Volkan Oezdemir", "Song Kenan", "Petr Yan"]
names_o = ["Nikolas Motta", "Carlos Hernandez", "Feng Xiaocan", "DongHun Choi", "SuYoung You", "Carlos Ulberg", "Muslim Salikhov", "Deiveson Figueiredo"]
odds_bfo_f = [-200, -127, 295, 120, 110, 205, 125, -260]
odds_bfo_o = [160, 102, -385, -150, -150, -265, -157, 196]
odds_365_f = [-188, 130, 300, -125, 130, 200, 150, -334]
odds_365_o = [162, -160, -400, 100, -160, -250, -175, 250]



assert len(names_f) == len(names_o) == len(odds_bfo_f) == len(odds_bfo_o) == len(odds_365_f) == len(odds_365_o)
# for i in range(len(names_f)):
#     print(names_f[i], odds_bfo_f[i], odds_365_f[i])
#     print(names_o[i], odds_bfo_o[i], odds_365_o[i])
#     print('--')

for i in range(len(names_f)):
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    print('noodds')
    PredictionPlots.plot_single_prediction(
        model=model,
        dataset=forecast_dataset,
        fighter_name=names_f[i],
        opponent_name=names_o[i],
        fight_parameters_values=[],
        event_date="2024-11-22",
        odds1=convert_odds_to_decimal(1),
        odds2=convert_odds_to_decimal(1),
        ax=axs[0],
    )
    print('bfo')
    PredictionPlots.plot_single_prediction(
        model=model,
        dataset=forecast_dataset,
        fighter_name=names_f[i],
        opponent_name=names_o[i],
        fight_parameters_values=[],
        event_date="2024-11-22",
        odds1=convert_odds_to_decimal(odds_bfo_f[i]),
        odds2=convert_odds_to_decimal(odds_bfo_o[i]),
        ax=axs[1],
        )
    print('365')
    PredictionPlots.plot_single_prediction(
        model=model,
        dataset=forecast_dataset,
        fighter_name=names_f[i],
        opponent_name=names_o[i],
        fight_parameters_values=[],
        event_date="2024-11-22",
        odds1=convert_odds_to_decimal(odds_365_f[i]),
        odds2=convert_odds_to_decimal(odds_365_o[i]),
        ax=axs[2],
        )

# %%
PredictionPlots.plot_single_prediction(
    model=model,
    dataset=forecast_dataset,
    fighter_name="Ming Shi",
    opponent_name="Feng Xiaocan",
    fight_parameters_values=[],
    event_date="2024-11-22",
    odds1=convert_odds_to_decimal(205),
    odds2=convert_odds_to_decimal(-265),
    )

# %%
PredictionPlots.plot_single_prediction(
    model=model,
    dataset=forecast_dataset,
    fighter_name="Petr Yan",
    opponent_name="Deiveson Figueiredo",
    fight_parameters_values=[],
    event_date="2024-11-22",
    odds1=convert_odds_to_decimal(1),
    odds2=convert_odds_to_decimal(1),
    )

# %%
mlflow.end_run()

# %%
cash0 = 200

df_weight = df.sort_values(by="event_date")

cash = [cash0,]
invest = [cash0,]
dates = [None,]


for date, group in df_weight.groupby("event_date"):  
    max_bet = max(cash[-1] * 0.5, 10)
    
    win = (group["confidence"]*group["win"]).sum() * max_bet / 10 / group["confidence"].sum()
    bet = (group["confidence"]*group["bet"]).sum() * max_bet / 10 / group["confidence"].sum()

    group["bet"] = group["confidence"]*group["win"] * max_bet / 10 / group["confidence"].sum()
    group["win"] = group["confidence"]*group["bet"] * max_bet / 10 / group["confidence"].sum()
    

    extra_added = max(bet - cash[-1], 0)
    cash_i = cash[-1] + win - min(bet, cash[-1])

    invest.append(invest[-1] + extra_added)
    cash.append(cash_i)
    dates.append(date)
    

cash = cash[1:]
invest = invest[1:]
dates = dates[1:]

# %%
fig, ax = plt.subplots()

msk = df_weight["weight_class"] != "Middleweight"

results = df_weight[msk].groupby("event_date")[["bet", "win"]].sum().reset_index()
ax.plot(
    results["event_date"],
    np.cumsum(results["win"] - results["bet"]), #/ np.cumsum(results["bet"]) * 100,
    ls='--',
    c='k',
)

invest = []
invest.append(results["bet"].iloc[0])


results = df_weight.groupby("event_date")[["bet", "win"]].sum().reset_index()


# Generate evenly spaced colors from the rainbow colormap
#unique_weight_classes = df_weight["weight_class"].unique()
weight_classes = ["Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"]
colors = plt.cm.rainbow(np.linspace(0, 1, len(weight_classes)))

# Plot each weight class with its unique color
for color, weight_class in zip(colors, reversed(weight_classes)):
    results = df_weight[df_weight["weight_class"] == weight_class].groupby("event_date")[["win", "bet"]].sum()
    cumsum_values = np.cumsum(results["win"] - results["bet"])
    ax.plot(results.index, cumsum_values, label=weight_class, color=color)

    # Place label at the last point of each line
    last_event_date = results.index[-1]
    last_cumsum_value = cumsum_values.iloc[-1]
    ax.text(last_event_date, last_cumsum_value, weight_class, color=color, fontsize=8, va='center')

# ax.set_ylim(-20, 20)
# Put x labels rotated 90 degrees
ax.tick_params(axis="x", labelrotation=90)
ax.set_ylim(None, None)
ax.grid()
ax.axhline(0, color="black")
ax.legend()

# %%
data = df.merge(
    data_processor.data[["fight_id", "fighter_name", "fighter_name_opponent", "event_date"]],
    on="fight_id",
).drop_duplicates(subset=["fight_id"])

# %%
# sort by win-bet
data["win-bet"] = data["win"] - data["bet"]
data = data.sort_values(by='win-bet', ascending=False)

# %%
results["bet"].sum()

# %%

# %%
X1, X2, X3, Y, odds1, odds2, _, _ = test_dataset.get_fight_data_from_ids(None)

X1.requires_grad = True
X2.requires_grad = True
odds1.requires_grad = True
odds2.requires_grad = True

output = model(X1, X2, X3, odds1.reshape(-1,1), odds2.reshape(-1,1))
output.sum().backward()

# %%
odds1.grad.sum()

# %%
fig, ax = plt.subplots(figsize=(5, 12))

labels = test_dataset.fighter_fight_statistics + ["odds"]
values = list(abs(X1.grad.sum(axis=0))) + [odds1.grad.sum(),]

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

df["fighter_p"] = df["fighter_p"] - excess/2
df["opponent_p"] = df["opponent_p"] - excess/2

df["house_pred"] = df["opponent_p"]/100

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
data_processor.scraper.event_scraper

# %%
data_processor.scraper.fight_scraper.data.columns

# %%
df = df.merge(
    data_processor.scraper.fight_scraper.data[["fight_id", "weight_class"]],
    on="fight_id",
)

df = df.merge(
    data_processor.scraper.event_scraper.data[["event_id", "event_name"]],
    on="event_id",
)

# %%
group = df.groupby("weight_class")

# %%
(df["win"] - df["bet"]).sum()

# %%
weight_classes = ["Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"]
len(weight_classes)

# %%
((group["win"].sum() - group["bet"].sum()) / group["bet"].count() )[weight_classes]

# %%
group["bet"].count()

# %%
fight_night = df["event_name"].str.contains("Fight Night")

group = df.groupby(fight_night)

# %%
group["win"].sum() - group["bet"].sum()

# %%
group["bet"].count()

# %%

# %%
forecast_dataset.get_forecast_prediction(
    fighter_names = ["Ilia Topuria",],
    opponent_names = ["Islam Makhachev",],
    event_dates = ["20250808",],
    fighter_odds = [100,],
    opponent_odds = [100,],
    model = model,
)

# %%
