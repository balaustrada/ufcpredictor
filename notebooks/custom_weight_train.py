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
import os

# %%
import mlflow
import mlflow.pytorch
import torch

# Enable autologging for PyTorch
# mlflow.pytorch.autolog()

mlflow.set_tracking_uri("http://127.0.0.1:5000") 
mlflow.set_experiment('custom_weight_train')

# %%
import pandas as pd
pd.set_option("display.max_columns", None)

# %%
from ufcpredictor.data_processor import DataProcessor
from ufcpredictor.data_enhancers import SumFlexibleELO, RankedFields
from ufcpredictor.data_aggregator import WeightedDataAggregator
from ufcpredictor.datasets import BasicDataset
from ufcpredictor.trainer import Trainer
from ufcpredictor.plot_tools import PredictionPlots
import torch
import numpy as np

import copy 

import matplotlib.pyplot as plt

# %%
# DataProcessor = ELODataProcessor
# data_processor_kwargs = {
#     "data_folder": "/home/cramirpe/UFC/UFCfightdata",
#     # "scaling_factor": 0.5,
#     # "boost_values": [1, 2, 3],
#     # "K_factor": 30,
# }

data_processor_kwargs = {
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
len(X_set)

# %%
data_processor.load_data()
data_processor.aggregate_data()
data_processor.add_per_minute_and_fight_stats()

# for field in X_set:
#     if field in ["ELO", "age"]:
#         continue
#     self.data_aggregated[field] = (self.data_aggregated[field].rank(pct=True) * 100) ** 1.2

data_processor.normalize_data()

# %% [markdown]
# ----

# %%
fight_ids = data_processor.data["fight_id"].unique()

# %%
invalid_fights = set(data_processor.data[data_processor.data["num_fight"] < 4]["fight_id"]) # The usual is 4

# invalid_fights |= set(self.data_aggregated[self.data_aggregated["event_date"] < "2013-01-01"]["fight_id"])

# %%
early_split_date = "2017-01-01" #"2017-01-01"
split_date = "2023-06-01"#"2023-08-01"
max_date = "2024-11-11" 

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
 Xf_set = ["num_rounds","weight"]
# Xf_set = []
early_train_dataset = BasicDataset(
    data_processor,
    early_train_fights,
    X_set=X_set,
    Xf_set = Xf_set,
)

train_dataset = BasicDataset(
    data_processor,
    train_fights,
    X_set=X_set,
    Xf_set = Xf_set,
)

test_dataset = BasicDataset(
    data_processor,
    test_fights,
    X_set=X_set,
    Xf_set = Xf_set,
)

# %%
early_train_dataloader = torch.utils.data.DataLoader(early_train_dataset, batch_size=2048, shuffle=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2048, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=False)

# %%
from ufcpredictor.models import SymmetricFightNet
from ufcpredictor.loss_functions import BettingLoss


# %%
model = SymmetricFightNet(
        input_size=len(train_dataset.X_set),
        input_size_f=len(Xf_set),
        dropout_prob=0.05, # 0.25
        fighter_network_shape=[256, 512, 1024, 512],
        # network_shape=[2048, 1024, 512, 128, 64, 1],
        # network_shape=[122, 1024, 2048, 1024, 512, 256, 128, 64, 1],
        network_shape=[122, 1024, 512, 256, 128, 64, 1],  # This was the best one so far
        # network_shape=[122, 1024, 512, 1024, 512, 256, 128, 64, 1],
        
)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-3, weight_decay=2e-5)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.9, patience=4
# )
mlflow.end_run()
mlflow.start_run()

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=2e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=6
)

trainer = Trainer(
    train_loader = train_dataloader,
    test_loader = test_dataloader,
    model = model,
    optimizer = optimizer,
    scheduler= scheduler,
    loss_fn =BettingLoss(),
    mlflow_tracking=True,
)   


# trainer.train(
#     epochs=3,
#     train_loader=early_train_dataloader,
#     test_loader=test_dataloader,
# )

# %%
weight_classes = data_processor.data["weight_class"].unique()

trainers = dict()

for weight_class in weight_classes:
    print(weight_class)

    seed = 2 # 42
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)

    wc_model = copy.deepcopy(model)

    
    valid_early_fights = set(data_processor.data["fight_id"][data_processor.data["weight_class"] == weight_class]) & early_train_fights
    wc_early_dataset = BasicDataset(
        data_processor,
        valid_early_fights,
        X_set=X_set,
        Xf_set = Xf_set,
    )


    valid_fights = set(data_processor.data["fight_id"][data_processor.data["weight_class"] == weight_class]) & train_fights
    wc_dataset = BasicDataset(
        data_processor,
        valid_fights,
        X_set=X_set,
        Xf_set = Xf_set,
    )

    valid_test_fights = set(data_processor.data["fight_id"][data_processor.data["weight_class"] == weight_class]) & test_fights
    wc_test_dataset = BasicDataset(
        data_processor,
        valid_test_fights,
        X_set=X_set,
        Xf_set=Xf_set,
    )

    wc_early_dataloader = torch.utils.data.DataLoader(wc_early_dataset, batch_size=2048, shuffle=True)
    wc_dataloader = torch.utils.data.DataLoader(wc_dataset, batch_size=2048, shuffle=True)
    wc_test_dataloader = torch.utils.data.DataLoader(wc_test_dataset, batch_size=2048, shuffle=False)
    
    model = SymmetricFightNet(
            input_size=len(train_dataset.X_set),
            input_size_f=len(Xf_set),
            dropout_prob=0.05, # 0.25
            fighter_network_shape=[256, 512, 1024, 512],
            # network_shape=[2048, 1024, 512, 128, 64, 1],
            # network_shape=[122, 1024, 2048, 1024, 512, 256, 128, 64, 1],
            network_shape=[122, 1024, 512, 256, 128, 64, 1],  # This was the best one so far
            # network_shape=[122, 1024, 512, 1024, 512, 256, 128, 64, 1],
            
    )
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-3, weight_decay=2e-5)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, mode="min", factor=0.9, patience=4
    # )
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=6
    )

    trainer = Trainer(
        train_loader = train_dataloader,
        test_loader = test_dataloader,
        model = model,
        optimizer = optimizer,
        scheduler= scheduler,
        loss_fn =BettingLoss(),
        mlflow_tracking=False,
    )   
    trainer.train(epochs=15, train_loader=early_train_dataloader,silent=True)
    trainer.train(epochs=10, silent=True)


    
        
    trainers[weight_class] = Trainer(
        train_loader = wc_dataloader,
        test_loader = wc_test_dataloader,
        model = model,
        optimizer = optimizer,
        scheduler= scheduler,
        loss_fn =BettingLoss(),
        mlflow_tracking=False,
    )
    trainers[weight_class].train(
        epochs=5,
        train_loader=wc_early_dataloader,
        test_loader=wc_test_dataloader,
        silent=True,
    )
    trainers[weight_class].train(epochs=30, silent=True)

# %%
weight_classes


# %%

# %%
# Save model dict
#torch.save(model.state_dict(), 'model.pth')

# %%
trainers[weight_class].test_loader.dataset

# %%

fig, ax = plt.subplots()
stats = []

for weight_class in weight_classes:
    print(weight_class)
    stats.append(PredictionPlots.show_fight_prediction_detail_from_dataset(
        model=trainers[weight_class].model,
        dataset=trainers[weight_class].test_loader.dataset,
        fight_ids=test_dataset.data_processor.data["fight_id"][test_dataset.data_processor.data["weight_class"] == weight_class],
        print_info=False,
        show_plot=True,
        ax=ax,
    ))

ax.set_ylim(-10,30)
ax.grid()

# %%
stats  = flattened_stats = [item for sublist in stats for item in sublist]

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
        "fight_id"
    ],
)

df = df.merge(
    data_processor.data[["fight_id", "fighter_id", "event_date", "event_id", "weight_class"]],
    on="fight_id",
)

df["confidence"] = abs((df["Prediction"] - 0.5) *2)


# %%
cash0 = 200

df = df.sort_values(by="event_date")

cash = [cash0,]
invest = [cash0,]
dates = [None,]


for date, group in df.groupby("event_date"):  
    max_bet = max(cash[-1] * 0.4, 10)
    
    win = (group["confidence"]*group["win"]).sum() * max_bet / 10 / group["confidence"].sum()
    bet = (group["confidence"]*group["bet"]).sum() * max_bet / 10 / group["confidence"].sum()

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
    [x-y for x,y in zip(cash,invest)],
    label="profit",
)

ax.axhline(0, c='k')
ax.legend()
ax.grid()

# %%
cash0 = 200

df_weight = df.sort_values(by="event_date")

cash = [cash0,]
invest = [cash0,]
dates = [None,]


for date, group in df_weight.groupby("event_date"):  
    max_bet = max(cash[-1] * 0.4, 10)
    
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

msk = df_weight["weight_class"] != None#"Middleweight"

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

labels = test_dataset.X_set + ["odds"]
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
