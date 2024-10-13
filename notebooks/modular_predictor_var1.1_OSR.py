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
import pandas as pd
pd.set_option("display.max_columns", None)

# %%
from ufcpredictor.data_processor import WOSRDataProcessor as DataProcessor
from ufcpredictor.datasets import BasicDataset
from ufcpredictor.trainer import Trainer
import torch
import numpy as np

# %%
self = data_processor = DataProcessor(
    data_folder="/home/cramirez/kaggle/ufc_scraper/data",
    weights=[0.3, 0.3, 0.3]
)

# %%
self.load_data()
self.aggregate_data()
self.add_per_minute_and_fight_stats()
self.normalize_data()

    # %%
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
 'OSR',
    ]

# X_set=BasicDataset.X_set + ["OSR",]

# %% [markdown]
# ----

# %%
fight_ids = self.data["fight_id"].unique()

# %%
invalid_fights = set(self.data_aggregated[self.data_aggregated["num_fight"] < 4]["fight_id"]) # The usual is 4

# invalid_fights |= set(self.data_aggregated[self.data_aggregated["event_date"] < "2013-01-01"]["fight_id"])

# %%
split_date = "2021-12-01"#"2023-08-01"
train_fights = self.data["fight_id"][self.data["event_date"] < split_date]
test_fights  = self.data["fight_id"][self.data["event_date"] >= split_date]

train_fights = set(train_fights) - set(invalid_fights)
test_fights = set(test_fights) - set(invalid_fights)

# %%
train_dataset = BasicDataset(
    data_processor,
    train_fights,
    X_set=X_set,
)

test_dataset = BasicDataset(
    data_processor,
    test_fights,
    X_set=X_set,
)

# %%
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

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
    test_loader = test_dataloader,
    model = model,
    optimizer = optimizer,
    scheduler= scheduler,
    loss_fn =BettingLoss(),
)   

# %%
trainer.train(epochs=8) # ~8 is a good match if dropout to 0.35 

# %%
# Save model dict
torch.save(model.state_dict(), 'model.pth')

# %%
from ufcpredictor.plot_tools import PredictionPlots
import matplotlib.pyplot as plt

# %%
fig, ax = plt.subplots()

PredictionPlots.show_fight_prediction_detail_from_dataset(
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
