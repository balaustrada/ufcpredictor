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

# %%
from ufcpredictor.data_processor import DataProcessor
from ufcpredictor.datasets import BasicDataset
import torch
import numpy as np

# %%
torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)

# %%
self = data_processor = DataProcessor(
    "/home/cramirez/kaggle/ufc_scraper/data"
)

# %%
self.load_data()
self.aggregate_data()
self.add_per_minute_and_fight_stats()
self.normalize_data()

# %% [markdown]
# ----

# %%
fight_ids = self.data["fight_id"].unique()

# %%
invalid_fights = list(self.data_aggregated[self.data_aggregated["num_fight"] < 5]["fight_id"])

# %%
split_date = "2023-12-01"#"2023-08-01"
train_fights = self.data["fight_id"][self.data["event_date"] < split_date]
test_fights  = self.data["fight_id"][self.data["event_date"] >= split_date]

train_fights = set(train_fights) - set(invalid_fights)
test_fights = set(test_fights) - set(invalid_fights)

# %%
train_dataset = BasicDataset(
    data_processor,
    train_fights,
)

test_dataset = BasicDataset(
    data_processor,
    test_fights,
)

# %%
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# %%
train_dataset[0][-1]

# %%
from torch import nn
class BettingLoss(nn.Module):
    def __init__(self):
        super(BettingLoss, self).__init__()

    def get_bet(self, prediction):
        return prediction

    def forward(self, predictions, targets, odds_1, odds_2):
        msk = torch.round(predictions) == targets

        return_fighter_1 = self.get_bet(0.5 - predictions) * odds_1
        return_fighter_2 = self.get_bet(predictions - 0.5) * odds_2

        losses = torch.where(
            torch.round(predictions) == 0,
            self.get_bet(0.5 - predictions),
            self.get_bet(predictions - 0.5),
        )

        earnings = torch.zeros_like(losses)
        earnings[msk & (targets == 0)] = return_fighter_1[msk & (targets == 0)]
        earnings[msk & (targets == 1)] = return_fighter_2[msk & (targets == 1)]

        return (losses - earnings).mean()


class SymmetricFightNet(nn.Module):
    
    def __init__(self, input_size, dropout_prob=0):
        super(SymmetricFightNet, self).__init__()
        self.fighter_net = FighterNet(input_size=input_size, dropout_prob=dropout_prob)

        self.fc1 = nn.Linear(256, 512)
        # self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

        # Use the global dropout probability
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.dropout4 = nn.Dropout(p=dropout_prob)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X1, X2, odds1, odds2):
        out1 = self.fighter_net(X1)
        out2 = self.fighter_net(X2)

        # import pdb; pdb.set_trace()

        out1 = torch.cat((out1, odds1), dim=1)
        out2 = torch.cat((out2, odds2), dim=1)

        x = torch.cat((out1 - out2, out2 - out1), dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after the first ReLU
        # x = self.relu(self.fc2(x))
        # x = self.dropout2(x)  # Apply dropout after the second ReLU
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)  # Apply dropout after the third ReLU
        x = self.relu(self.fc4(x))
        # x = self.dropout4(x)  # Apply dropout after the fourth ReLU
        x = self.fc5(x)
        return x


# %%

# %%
def train(
    model, optimizer, train_dataloader, val_dataloader, scheduler, device, num_epochs
):
    model.to(device)

    criterion = {"target": BettingLoss().to(device)}

    best_loss = 999999
    best_model = None

    target_preds = []
    target_labels = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = []

        for X1, X2, Y, odds1, odds2 in tqdm(iter(train_dataloader)):
            X1, X2, odds1, odds2 = (
                X1.to(device),
                X2.to(device),
                odds1.to(device),
                odds2.to(device),
            )
            Y = Y.to(device)

            optimizer.zero_grad()
            target_logit = model(X1, X2, odds1, odds2)
            # target_logit_2 = model(X2, X1)

            # target_logit = (target_logit + (1-target_logit_2)) / 2

            # w = target_logit <= 0
            # w &= target_logit > 1
            # if w.sum() > 0:
            #     import pdb; pdb.set_trace()

            # w = Y <= 0
            # w &= Y > 1
            # if w.sum() > 0:
            #     import pdb; pdb.set_trace()

            loss = criterion["target"](target_logit, Y, odds1, odds2)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            target_preds += torch.round(target_logit).detach().cpu().numpy().tolist()
            target_labels += Y.detach().cpu().numpy().tolist()

        match = np.asarray(target_preds).reshape(-1) == np.asarray(
            target_labels
        ).reshape(-1)

        val_loss, val_target_f1, correct, _, _ = validation(
            model, val_dataloader, criterion, device
        )

        print(f"Train acc: [{match.sum() / len(match):.5f}]")
        print(
            f"Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Disaster? F1 : [{val_target_f1:.5f}] Correct: [{correct*100:.2f}]"
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        if best_loss > val_loss:
            best_los = val_loss
            best_model = model
            torch.save(model, "./best-model.pth")
            print("Model saved!")

    return best_model


# %%
# validation function
def validation(model, val_dataloader, criterion, device):
    model.eval()
    val_loss = []

    target_preds = []
    target = []
    target_labels = []

    correct = []
    count = []

    with torch.no_grad():
        for X1, X2, Y, odds1, odds2 in tqdm(iter(val_dataloader)):
            X1, X2, odds1, odds2 = (
                X1.to(device),
                X2.to(device),
                odds1.to(device),
                odds2.to(device),
            )
            Y = Y.to(device)

            target_logit = model(X1, X2, odds1, odds2)
            # target_logit_2 = model(X2, X1)

            # target_logit_2 = 1 - target_logit_2
            # target_logit = (target_logit + target_logit_2) / 2

            loss = criterion["target"](target_logit, Y, odds1, odds2)

            val_loss.append(loss.item())

            target += target_logit
            target_preds += torch.round(target_logit).detach().cpu().numpy().tolist()
            target_labels += Y.detach().cpu().numpy().tolist()

    match = np.asarray(target_preds).reshape(-1) == np.asarray(target_labels).reshape(
        -1
    )

    target_f1 = f1_score(target_labels, target_preds, average="macro")

    return np.mean(val_loss), target_f1, match.sum() / len(match), target, target_labels


# %%
from ufcpredictor.models import SymmetricFightNet

# %%
model = SymmetricFightNet(input_size=58, dropout_prob=0.35)
model.eval()

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
scheduler = None
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# %%
from ufcpredictor.loss_functions import BettingLoss

from tqdm import tqdm
from sklearn.metrics import f1_score

# %%
infer_model = train(
    model,
    optimizer,
    train_dataloader,
    test_dataloader,
    scheduler,
    device="cpu",
    num_epochs=8,
)

# %% [markdown]
# _____

# %%
import matplotlib.pyplot as plt

def get_fight_data(self, fight_ids):
    if fight_ids is not None:
        fight_data = self.fight_data[
            self.fight_data["fight_id"].isin(fight_ids)
        ]
    else:
        fight_data = self.fight_data.copy()
    
    data = [
        torch.FloatTensor(
            np.asarray([fight_data[x + "_x"].values for x in self.X_set]).T
        ),
        torch.FloatTensor(
            np.asarray([fight_data[x + "_y"].values for x in self.X_set]).T
        ),
        torch.FloatTensor(
            (fight_data["winner_x"] != fight_data["fighter_id_x"]).values
        ),
        torch.FloatTensor(fight_data["opening_x"].values),
        torch.FloatTensor(fight_data["opening_y"].values),
    ]
    
    fighter_names = fight_data["fighter_name_x"].values
    opponent_names = fight_data["fighter_name_y"].values
    
    X1, X2, Y, odds1, odds2 = data

    return X1, X2, Y, odds1, odds2
    
def show_fight_prediction_detail(
    self,
    fight_ids,
    print_info=False,
    show_plot=False,
    ax=None,
):  
    
    if fight_ids is not None:
        fight_data = self.fight_data[
            self.fight_data["fight_id"].isin(fight_ids)
        ]
    else:
        fight_data = self.fight_data.copy()
    
    data = [
        torch.FloatTensor(
            np.asarray([fight_data[x + "_x"].values for x in self.X_set]).T
        ),
        torch.FloatTensor(
            np.asarray([fight_data[x + "_y"].values for x in self.X_set]).T
        ),
        torch.FloatTensor(
            (fight_data["winner_x"] != fight_data["fighter_id_x"]).values
        ),
        torch.FloatTensor(fight_data["opening_x"].values),
        torch.FloatTensor(fight_data["opening_y"].values),
    ]
    
    fighter_names = fight_data["fighter_name_x"].values
    opponent_names = fight_data["fighter_name_y"].values
    
    with torch.no_grad():
        X1, X2, Y, odds1, odds2 = data
        predictions_1 = model(X1, X2, odds1.reshape(-1,1), odds2.reshape(-1,1)).detach().cpu().numpy().reshape(-1)
        predictions_2 = 1- model(X2, X1, odds2.reshape(-1,1), odds1.reshape(-1,1)).detach().cpu().numpy().reshape(-1)
    
        predictions = 0.5*(predictions_1 + predictions_2)
        shifts = abs(predictions_2 - predictions_1)
    
        corrects = predictions.round() == Y.numpy()
    
        odds1 = odds1.numpy().reshape(-1)
        odds2 = odds2.numpy().reshape(-1)
        
    
        invested = 0
        earnings = 0
        fights = 0
        nbets = 0

        invest_progress = []
        earning_progress = []
        
        for fighter, opponent, prediction, shift, odd1, odd2, correct in zip(
            fighter_names,
            opponent_names,
            predictions,
            shifts,
            odds1,
            odds2,
            corrects,
        ):
            prediction = round(float(prediction),3)
            shift = round(float(shift), 3)
        
            if prediction > 0.5:
                bet = 2*10*(prediction-0.5)
                earning = odd2*bet if correct else 0
            else:
                bet = 2*10*(0.5-prediction)
                earning = odd1*bet if correct else 0
        
            invested += bet
            earnings += earning

            invest_progress.append(bet)
            earning_progress.append(earning)
            
            fights += 1
            nbets += 1

            if print_info:
                print(fighter, "vs", opponent)
                print(odd1, "vs", odd2)
                print(prediction, shift)
        
                print(f"bet: {bet:.2f}, earn: {earning:.2f}")
                print(f"invested: {invested:.2f}, earnings: {earnings:.2f}, nbets: {nbets}, fights: {fights}")
                print(f"benefits: {(earnings/invested-1)*100:.2f}%")
                
                print()

    if show_plot:
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(np.cumsum(invest_progress), (np.cumsum(earning_progress)-np.cumsum(invest_progress))/np.cumsum(invest_progress)*100)
        ax.axhline(0, c='k')

# %%
fight_ids = [
    "b1f2ec122beda7a5",
    "f995ed679ef8bdf4",
    "3fa8ee3fdc04fe36",
]

fig, ax = plt.subplots()
show_fight_prediction_detail(
    test_dataset,
    fight_ids=None, 
    print_info=True,
    show_plot=True,
    ax=ax,
)

ax.set_ylim(-10, 10)
ax.grid()

# %%
X1, X2, Y, odds1, odds2 = get_fight_data(test_dataset, None)

X1.requires_grad = True
X2.requires_grad = True
odds1.requires_grad = True
odds2.requires_grad = True

output = model(X1, X2, odds1.reshape(-1,1), odds2.reshape(-1,1))
output.sum().backward()

# %%
fig, ax = plt.subplots(figsize=(5, 12))

labels = test_dataset.X_set + ["odds"]
values = list(abs(X1.grad.sum(axis=0))) + [4.3132,]

sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
labels = [labels[i] for i in sorted_indices]
values = [values[i] for i in sorted_indices]

ax.barh(labels, values)
ax.grid()

# %%

# %%
len(values)

# %%
labels

# %%
