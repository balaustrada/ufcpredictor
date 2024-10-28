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
import pandas as pd
pd.set_option("display.max_columns", None)

# %%
from ufcpredictor.data_processor import ELODataProcessor as DataProcessor
from ufcpredictor.datasets import BasicDataset
from ufcpredictor.trainer import Trainer
import torch
import numpy as np

# %%
self = data_processor = DataProcessor(
    data_folder="/home/cramirpe/UFC/UFCfightdata",
)

# %%
self.load_data()
self.aggregate_data()
self.add_per_minute_and_fight_stats()
self.normalize_data()

# %%
if True:
    X_set = [
        'clinch_strikes_att_opponent_per_minute',
        'time_since_last_fight',
        'total_strikes_succ_opponent_per_minute',
        'takedown_succ_per_minute',
        'KO_opponent_per_minute',
        'takedown_att_per_minute',
        'takedown_succ_opponent_per_minute',
        'win_opponent_per_fight',
        'head_strikes_succ_opponent_per_minute',
        'clinch_strikes_succ_opponent_per_minute',
        'ground_strikes_succ_opponent_per_minute',
        'ground_strikes_att_per_minute',
        'head_strikes_succ_per_minute',
        'age',
        'distance_strikes_succ_per_minute',
        'body_strikes_succ_per_minute',
        'strikes_succ_opponent_per_minute',
        'leg_strikes_att_per_minute',
        'reversals_opponent_per_minute',
        'strikes_succ_per_minute',
        'distance_strikes_att_opponent_per_minute',
        'Sub_opponent_per_fight',
        'distance_strikes_att_per_minute',
        'knockdowns_per_minute',
        'ELO',
    ]
else:
    X_set = None

# X_set=BasicDataset.X_set + ["OSR",]

# %% [markdown]
# ----

# %%
fight_ids = self.data["fight_id"].unique()

# %%
invalid_fights = set(self.data_aggregated[self.data_aggregated["num_fight"] < 3]["fight_id"]) # The usual is 4

# invalid_fights |= set(self.data_aggregated[self.data_aggregated["event_date"] < "2013-01-01"]["fight_id"])

# %%
early_split_date = "2019-11-01"
split_date = "2024-10-25"

early_train_fights = self.data["fight_id"][self.data["event_date"] < split_date]

train_fights = self.data["fight_id"][(self.data["event_date"] >= early_split_date)]


early_train_fights = set(early_train_fights) - set(invalid_fights)
train_fights = set(train_fights) - set(invalid_fights)

# %%
# Now I generate a data_processor specifically for training, so 
# I avoid any possible contamination from the test sample:
train_data_processor = DataProcessor(
    "/home/cramirpe/UFC/UFCfightdata",
)
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

# %%
early_train_dataloader = torch.utils.data.DataLoader(early_train_dataset, batch_size=64, shuffle=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# %%
from ufcpredictor.models import SymmetricFightNet
from ufcpredictor.loss_functions import BettingLoss


# %%
seed = 43
torch.manual_seed(seed)
import random
random.seed(seed)
np.random.seed(seed)

# %%
model = SymmetricFightNet(
        input_size=len(train_dataset.X_set),
        dropout_prob=0.35, # 0.35
)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
)

trainer = Trainer(
    train_loader = train_dataloader,
    test_loader = None,
    model = model,
    optimizer = optimizer,
    scheduler= scheduler,
    loss_fn =BettingLoss(),
)   

# %%
trainer.train(
    epochs=5,
    train_loader=early_train_dataloader,
    test_loader=None,
)

# %%
trainer.train(epochs=3) # ~8 is a good match if dropout to 0.35 

# %%
# Save model dict
torch.save(model.state_dict(), 'model.pth')

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

ax.set_ylim(-10,30)
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
        "fight_id"
    ],
)

df = df.merge(
    data_processor.data[["fight_id", "event_date"]],
    on="fight_id",
)

results = df.groupby("event_date")[["bet", "win"]].sum().reset_index()
results

# %%
fig, ax = plt.subplots()

ax.plot(
    results["event_date"],
    np.cumsum(results["win"] - results["bet"])
)
# Put x labels rotated 90 degrees
ax.tick_params(axis="x", labelrotation=90)

ax.grid()
ax.axhline(0, color="black")

# %%

# %%

# %%

# %%
X1, X2, Y, odds1, odds2, _, _ = test_dataset.get_fight_data_from_ids(None)

X1.requires_grad = True
X2.requires_grad = True
odds1.requires_grad = True
odds2.requires_grad = True

output = model(X1, X2, odds1.reshape(-1,1), odds2.reshape(-1,1))
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
