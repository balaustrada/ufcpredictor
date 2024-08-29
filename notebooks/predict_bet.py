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
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
plt.style.use('tableau-colorblind10')


# Show all columns for each dataframe (horizontal scroll)
pd.set_option("display.max_columns", None)


# %%
data_folder = Path("/home/cramirez/kaggle/ufc_scraper/data")

# %%
fight_data = pd.read_csv(
    data_folder / "fight_data.csv"
)


# %%
round_data = pd.read_csv(
    data_folder / "round_data.csv"
)

###########################################################
# Now we are going to group the round data to get stats
# per match for each player
###########################################################

# Fix the column ctrl time to show time in seconds
def convert_to_seconds(time_str:str) -> int:
    if time_str == "--":
        return 0
    elif time_str in (None, "NULL") or pd.isna(time_str):
        return None
    else:
        minutes, seconds = map(int, time_str.split(":"))
        return minutes * 60 + seconds

round_data["ctrl_time"] = round_data["ctrl_time"].apply(convert_to_seconds)


# %%
fighter_data = pd.read_csv(
    data_folder / "fighter_data.csv"
)

# Adding the name by combingin first and last name
fighter_data["fighter_name"] = (
    fighter_data["fighter_f_name"] + " " + fighter_data["fighter_l_name"].fillna("")
)

# %%
event_data = pd.read_csv(
    data_folder / "event_data.csv"
)

# %%
###########################################################
# I want to create two rows per match, one row for each fighter
###########################################################
# Hence I need to duplicate the current fight data 
# Assining fighter and opponent to each other
data = pd.concat(
    [
        fight_data.rename(
            columns={"fighter_1": "opponent_id", "fighter_2": "fighter_id"}
        ),
        fight_data.rename(
            columns={"fighter_2": "opponent_id", "fighter_1": "fighter_id"}
        ),
    ]
)

# I am merging the fighter data to the previous table 
# This includes height, reach etc...
fighter_fields = ["fighter_id", "fighter_name", "fighter_nickname"]
data = data.merge(
    fighter_data,#[fighter_fields],
    on="fighter_id",
    how="left",
)        

data = data.merge(
    fighter_data[["fighter_id", "fighter_name", "fighter_nickname"]],
    left_on="opponent_id",
    right_on="fighter_id",
    how="left",
    suffixes=("", "_opponent"),
)

# %%
odds_data = pd.read_csv(data_folder / "BestFightOdds_odds.csv")

#################################################################
# Convert into European format which makes more sense
# In european returns = odds*bet
#################################################################
for field in "opening", "closing_range_min", "closing_range_max":
    msk = odds_data[field] > 0

    odds_data.loc[msk, field] = odds_data.loc[msk, field] / 100 + 1
    odds_data.loc[~msk, field] = 100 / -odds_data.loc[~msk, field] + 1

# %%
###########################################################
# Each round data row only includes the stats of one fighter
# but opponent stats are also useful, now we include them
# by joining round with itself
###########################################################
# Get the different names of stats
stat_names = [col for col in round_data.columns if col not in ["fight_id", "fighter_id", "round"]]

# Merging columsn
round_data = pd.merge(
    round_data,
    round_data,
    on=["fight_id", "round"],
    suffixes=("", "_opponent"),
)

# And then remove the match of the fighter with itself
round_data = round_data[
    round_data["fighter_id"] != round_data["fighter_id_opponent"]
]

# %%
##############################################################
# Now we sum all the stats per round
##############################################################
fight_stats = round_data.groupby(
    ["fight_id", "fighter_id", "fighter_id_opponent"]
).sum().reset_index().drop("round", axis=1)

# %%
#######################################################################
# This is a single row example
#######################################################################
max_len = max(len(c) for c in fight_stats.columns)

for c, v in zip(fight_stats.columns, fight_stats.values[0]):
    print(f"{c.ljust(max_len)}\t\t{v}")

# %%
###############################################################################
# Now we have stats per fight, we can join it with our previous data
# To get general fight stats like number of rounds, finish time etc etc
##############################################################################
data = data.merge(
    fight_stats,
    on=["fight_id", "fighter_id"],
)

data["finish_time"] = data["finish_time"].apply(convert_to_seconds)
data["total_time"] = (data["finish_round"]-1)*5*60 + data["finish_time"]
data = data[
    data["time_format"].isin(['3 Rnd (5-5-5)', '5 Rnd (5-5-5-5-5)'])
]

# %%
##############################################################
# Add odds data
###############################################################
data = data.merge(
    odds_data[["fight_id", "fighter_id", "opening"]], # I am only adding opening info
    on=["fight_id", "fighter_id"],
)

# %%
# ##############################################################
# # Add the opponent name 
# ###############################################################
# data = data.merge(
#     fighter_data[["fighter_id", "fighter_name"]],
#     left_on=["fighter_id_opponent"],
#     right_on=["fighter_id"],
#     suffixes=("", "_opponent"),
# )

# %%
##################################################################################
# Adding the weight in lbs from Weight Class (*!! weight classes changed in the past...)
##################################################################################
weight_dict = {
    "Heavyweight": 265,
    "Welterweight": 170,
    "Women's Flyweight": 125,
    "Light Heavyweight": 205,
    "Middleweight": 185,
    "Women's Featherweight": 145,
    "Bantamweight": 135,
    "Lightweight": 155,
    "Flyweight": 125,
    "Women's Strawweight": 115,
    "Women's Bantamweight": 135,
    "Featherweight": 145,
}

data.loc[:, "weight"] = data["weight_class"].map(weight_dict)

##################################################################################
# Remmove null weight classes, or open weight or catch weight (agreed weight outside a weight class)
##################################################################################
data = data[
    (data["weight_class"] != "NULL")
    & (data["weight_class"] != "Catch Weight")
    & (data["weight_class"] != "Open Weight")
]

# %%
# Only keep male fights for now.
data = data[data["gender"] == "M"]

# %%
# Remove disqualified and doctor's stoppage
data = data[data["result"].isin(["Decision", "KO/TKO", "Submission"])]

# %%
#############################################
# Add some missing stats
# KO, Submission and Win
#############################################
# Whether fighters has KOd his opponent
data["KO"] = np.where(
    (data["result"].str.contains("KO")) & (data["winner"] == data["fighter_id"]), 1, 0
)

# Whether the fighter has been KOd by his opponent
data["KO_opponent"] = np.where(
    (data["result"].str.contains("KO")) & (data["winner"] != data["fighter_id"]), 1, 0
)

# Same for busmission
data["Sub"] = np.where(
    (data["result"].str.contains("Submission"))
    & (data["winner"] == data["fighter_id"]),
    1,
    0,
)

data["Sub_opponent"] = np.where(
    (data["result"].str.contains("Submission"))
    & (data["winner"] != data["fighter_id"]),
    1,
    0,
)

data["win"] = np.where(data["winner"] == data["fighter_id"], 1, 0)
data["win_opponent"] = np.where(data["winner"] != data["fighter_id"], 1, 0)

stat_names += ["KO", "Sub", "win"]

# %%
####################################################################
# Add the date of the event to the dataframe
# Remove old fights since I don't have odds for previous dates
# + UFC was a nonsense
#####################################################################
data = data.merge(
    event_data[["event_id", "event_date"]], # I only need the date for now,
    on="event_id",
)   

data["event_date"] = pd.to_datetime(data["event_date"])
data["fighter_dob"] = pd.to_datetime(data["fighter_dob"])
data = data[data["event_date"].dt.date >= datetime.date(2008, 8, 1)]
data["age"]  = (data["event_date"] - data["fighter_dob"]).dt.days / 365
 

data = data.sort_values(by=["fighter_id", "event_date"])

# %%
##########################################################################
# The input data for the network should be the data summed/averaged for all previous
# fights to the one to be predicted.
###########################################################################
data_aggregated = data.copy()
data_aggregated["num_fight"] = data_aggregated.groupby("fighter_id").cumcount() + 1

data_aggregated["previous_fight_date"] = data_aggregated.groupby("fighter_id")["event_date"].shift(1)
data_aggregated["time_since_last_fight"] = (
    data_aggregated["event_date"] - data_aggregated["previous_fight_date"]
).dt.days

# Convert stats into float and then sum  
for column in stat_names:
    for suffix in ["", "_opponent"]:
        data_aggregated[f"{column}{suffix}"] = data_aggregated[f"{column}{suffix}"].astype(float)
        data_aggregated[f"{column}{suffix}"] = data_aggregated.groupby(
            "fighter_id"
        )[f"{column}{suffix}"].cumsum()

# Apply cumsum also to total time (total time in the octagon)
data_aggregated["total_time"] = data_aggregated.groupby("fighter_id")["total_time"].cumsum()

# %%
##########################################################################
# Check aggregated data for Jailton Almeida
# https://ufcstats.com/fighter-details/41e83a89929d1327
###########################################################################
data_aggregated[data_aggregated["fighter_id"] == "150ff4cc642270b9"]

# %%
############################################################################
# Generate per minute and per fight statistics
# some are relevant per minute (strikes, takedowns etc)
# some are relevant per fight (KO, submission, wins, etc)
# but all are computed here to choose later
############################################################################
new_columns = {}

for colum in stat_names:
    for suffix in ["", "_opponent"]:
        new_columns[f"{colum}{suffix}_per_minute"] = data_aggregated[f"{colum}{suffix}"] / data_aggregated["total_time"]
        new_columns[f"{colum}{suffix}_per_fight"] = data_aggregated[f"{colum}{suffix}"] / data_aggregated["num_fight"]

new_columns_df = pd.DataFrame(new_columns)
data_aggregated = pd.concat([data_aggregated, new_columns_df], axis=1)
data_aggregated = data_aggregated.copy() # Defragment dataframe

# %%
##########################################################################
# Check aggregated data for Jailton Almeida 
# https://ufcstats.com/fighter-details/41e83a89929d1327
###########################################################################
data_aggregated[data_aggregated["fighter_id"] == "150ff4cc642270b9"]

# %% [markdown]
# ## Normalize fields

# %%
for colum in stat_names:
    for suffix1 in ["", "_opponent"]:
        for suffix2 in "", "_per_minute", "_per_fight":
            mean = data_aggregated[colum + suffix1 + suffix2].mean()
            data_aggregated[column + suffix1 + suffix2] = data_aggregated[colum] / mean
            
for column in ["age", "time_since_last_fight", "fighter_height_cm"]:
    mean = data_aggregated[column].mean()
    data_aggregated[column] = data_aggregated[column] / mean

# %% [markdown]
# ## Now entering NN related things

# %%
#############################################
# Define the input features for the NN
#############################################
input_set = [
    "knockdowns_per_minute",
    "strikes_att_per_minute",
    "strikes_succ_per_minute",
    "head_strikes_att_per_minute",
    "head_strikes_succ_per_minute",
    "body_strikes_att_per_minute",
    "body_strikes_succ_per_minute",
    "leg_strikes_att_per_minute",
    "leg_strikes_succ_per_minute",
    "distance_strikes_att_per_minute",
    "distance_strikes_succ_per_minute",
    "ground_strikes_att_per_minute",
    "ground_strikes_succ_per_minute",
    "clinch_strikes_att_per_minute",
    "clinch_strikes_succ_per_minute",
    "total_strikes_att_per_minute",
    "total_strikes_succ_per_minute",
    "takedown_att_per_minute",
    "takedown_succ_per_minute",
    "submission_att_per_minute",
    "reversals_per_minute",
    "ctrl_time_per_minute",
    "KO_per_minute",
    "Sub_per_minute",
    "KO_per_fight",  # KO_per_minute is KO power, KO_per_fight is related to finishing power, not exactly the same...
    "Sub_per_fight",  # Sub_per_minute is kind of grappling strength, while Sub_per_fight is related to finishing power
    "time_since_last_fight",  # Relevant
    "win_per_fight",
    "age",
    "fighter_height_cm",  # Prob. better to use BMI
    # "fighter_reach_cm", # Need to remove nans
    "num_fight",  # There's a limited amount of hits you can receive...
]

opponent_stats = []
# For the stats _per_ features
# Add also the ones absorved (_opponent ones)
for x in input_set:
    if "_per_" in x:
        index_of_per = x.index("_per_")
        suffix = x[index_of_per:]

        opponent_stats.append(x[:index_of_per] + "_opponent" + suffix)

input_set += opponent_stats

# %%

# %%
from pyspark.sql.dataframe import DataFrame

# %%
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import Adam

# %%
from sklearn.metrics import f1_score

# %%
len(data_aggregated) / 2  # This is the number of data fights

# %%
############################################
# I won't predict fights for fighters with less than 3 fights in the UFC
#############################################
invalid_fights = list(data_aggregated[data_aggregated["num_fight"] < 3]["fight_id"])
valid_fights = set(data_aggregated["fight_id"]) - set(invalid_fights)

# %%
len(valid_fights)


# %%
###################################################################
# This functions gets a fight id and returns
# the features for each fighter and the winner
###################################################################
def from_id_to_fight(id_, print_fighters=False, print_odds=False):
    # Get fighters
    fights = data_aggregated[data_aggregated["fight_id"] == id_]
    fight = fights.iloc[0]
    fight_opponent = fights.iloc[1]
    
    f1 = fight["fighter_id"]
    f2 = fight["opponent_id"]
    date = fight["event_date"]
    winner = fight["winner"]

    f1p = data_aggregated[
        (data_aggregated["event_date"] < date) & (data_aggregated["fighter_id"] == f1)
    ]
    f1p = f1p.iloc[f1p["event_date"].argmax()]

    f2p = data_aggregated[
        (data_aggregated["event_date"] < date) & (data_aggregated["fighter_id"] == f2)
    ]
    f2p = f2p.iloc[f2p["event_date"].argmax()]

    x1 = [f1p[x] for x in input_set]
    x2 = [f2p[x] for x in input_set]

    if print_fighters:
        print(fight["fighter_name"], " vs ", fight["fighter_name_opponent"])

    if print_odds:
        fight_mask = odds_data["fight_id"] == id_

        odds = []
        for fighter_id in (fight["fighter_id"], fight["fighter_id_opponent"]):
            fighter_mask = fight_mask & (odds_data["fighter_id"] == fighter_id)
            odds.append(odds_data[fighter_mask]["opening"].values[0])
        print(" vs ".join([str(odd) for odd in odds]))

    # if np.isnan(x1).any():
    #     import pdb; pdb.set_trace()


    odds_1 = fight["opening"]
    odds_2 = fight_opponent["opening"]
    
    return (
        torch.FloatTensor(x1),
        torch.FloatTensor(x2),
        torch.FloatTensor([float(winner != fight["fighter_id"])]),
        torch.FloatTensor([odds_1, odds_2]),
    )


# %%
class CustomDataset(Dataset):
    def __init__(self, data, mode="train"):
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X1, X2, Y, odds = self.data[idx]  # Ignore odds here
        odds1, odds2 = odds

        odds1 = torch.FloatTensor(
            [
                odds1,
            ]
        )
        odds2 = torch.FloatTensor(
            [
                odds2,
            ]
        )

        ################################
        # Randomly flip fighters, I think the veterans
        # are always on the right side
        # and is slightly biased towards it
        ################################
        if np.random.random() >= 0.5:
            X1, X2 = X2, X1
            Y = 1 - Y
            odds1, odds2 = odds2, odds1

        if self.mode == "train":
            return X1, X2, odds1, odds2, Y
        else:
            return X1, X2, odds1, odds2


# %%
train_fights = sorted(
    data_aggregated["fight_id"][data_aggregated["event_date"] < "2023-08-01"]
    .unique()
    .tolist()
)
test_fights = sorted(
    data_aggregated["fight_id"][data_aggregated["event_date"] >= "2023-08-01"]
    .unique()
    .tolist()
)
validation_fights = test_fights

# %%
validation_fights = test_fights

# %%
#############################################
# Get the input data as pytorch arrays
# for training and test
#############################################

train_data = []
valid_train_fights = []
for id_ in train_fights:
    if id_ in valid_fights:
        valid_train_fights.append(id_)
        train_data.append(from_id_to_fight(id_))

test_data = []
valid_test_fights = []
for id_ in test_fights:
    if id_ in valid_fights:
        valid_test_fights.append(id_)
        test_data.append(from_id_to_fight(id_))

val_data = test_data

# %%
print(f"Train data: {len(train_data)}\nTest data: {len(test_data)}")

# %%
train_dt = CustomDataset(train_data, mode="train")
val_dt = CustomDataset(val_data, mode="train")
full_dt = CustomDataset(data, mode="train")

train_dataloader = torch.utils.data.DataLoader(train_dt, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dt, batch_size=64, shuffle=False)
full_dataloader = torch.utils.data.DataLoader(full_dt, batch_size=64, shuffle=False)

# %%
len(train_data[0][1])

# %%
DROPOUT_PROB = 0


class FighterNet(nn.Module):
    def __init__(self):
        super(FighterNet, self).__init__()
        self.fc1 = nn.Linear(len(train_data[0][1]), 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 127)

        # Use the global dropout probability
        self.dropout1 = nn.Dropout(p=DROPOUT_PROB)
        self.dropout2 = nn.Dropout(p=DROPOUT_PROB)
        self.dropout3 = nn.Dropout(p=DROPOUT_PROB)
        self.dropout4 = nn.Dropout(p=DROPOUT_PROB)
        self.dropout5 = nn.Dropout(p=DROPOUT_PROB)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after the first ReLU
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after the second ReLU
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)  # Apply dropout after the third ReLU
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)  # Apply dropout after the fourth ReLU
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)  # Apply dropout after the fifth ReLU

        return x


class SymmetricFightNet(nn.Module):
    def __init__(self):
        super(SymmetricFightNet, self).__init__()
        self.fighter_net = FighterNet()

        self.fc1 = nn.Linear(256, 512)
        # self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

        # Use the global dropout probability
        self.dropout1 = nn.Dropout(p=DROPOUT_PROB)
        self.dropout2 = nn.Dropout(p=DROPOUT_PROB)
        self.dropout3 = nn.Dropout(p=DROPOUT_PROB)
        self.dropout4 = nn.Dropout(p=DROPOUT_PROB)

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
        x = self.dropout4(x)  # Apply dropout after the fourth ReLU
        x = self.sigmoid(self.fc5(x))
        return x


# %%
class BettingLoss(nn.Module):
    def __init__(self):
        super(BettingLoss, self).__init__()

    def get_bet(self, prediction):
        return prediction * 2 * 10

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

        for X1, X2, odds1, odds2, Y in tqdm(iter(train_dataloader)):
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
        for X1, X2, odds1, odds2, Y in tqdm(iter(val_dataloader)):
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
model = SymmetricFightNet()
model.eval()

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
scheduler = None
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# %%
infer_model = train(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    scheduler,
    device="cpu",
    num_epochs=10,
)


# %%

# %% [markdown]
# ### Now let's predict the quantity to bet

# %%
def compare_fighters_from_id(id_):
    x1, x2, y, (odds1, odds2) = from_id_to_fight(
        id_, print_fighters=True, print_odds=False
    )

    # odds1 = odds[0]
    # odds2 = odds[1]

    # odds1 = torch.FloatTensor([odds[0]]).reshape((1, -1))
    # odds2 = torch.FloatTensor([odds[1]]).reshape((1, -1))

    odds1 = torch.reshape(odds1, (1, -1))
    odds2 = torch.reshape(odds2, (1, -1))
    x1 = torch.reshape(x1, (1, -1))
    x2 = torch.reshape(x2, (1, -1))

    model.eval()
    with torch.no_grad():
        value1 = float(model(x1, x2, odds1, odds2))
        value2 = 1 - float(model(x2, x1, odds2, odds1))

    value = (value1 + value2) / 2

    print(round(value, 3), round(np.abs(value1 - value2), 3))


def compare_fighters(
    fighter1,
    fighter2,
    date,
    show_distance,
    min_confidence=0,
    max_diff=0.2,
    max_bet=10000,
):
    fighter1 = data_aggregated["fighter_id"][
        data_aggregated["fighter_name"].str.contains(fighter1)
    ].iloc[0]
    fighter2 = data_aggregated["fighter_id"][
        data_aggregated["fighter_name"].str.contains(fighter2)
    ].iloc[0]
    f1p = data_aggregated[
        (data_aggregated["event_date"].dt.date < date)
        & (data_aggregated["fighter_id"].str.contains(fighter1))
    ]
    f1p = f1p.iloc[f1p["event_date"].argmax()]

    f2p = data_aggregated[
        (data_aggregated["event_date"].dt.date < date)
        & (data_aggregated["fighter_id"].str.contains(fighter2))
    ]
    f2p = f2p.iloc[f2p["event_date"].argmax()]

    x1 = [f1p[x] for x in X_set]
    x2 = [f2p[x] for x in X_set]

    x1 = torch.reshape(torch.FloatTensor(x1), (1, -1))
    x2 = torch.reshape(torch.FloatTensor(x2), (1, -1))

    print(f1p["UFC_names"], " vs ", f2p["UFC_names"])

    value1 = float(model(x1, x2))
    value2 = 1 - float(model(x2, x1))

    value = (value1 + value2) / 2
    diff = np.abs(value1 - value2)
    confidence = np.abs(value - 0.5)

    if show_distance:
        distance = np.abs(0.5 - value)
        print(round(value, 3), round(np.abs(value1 - value2), 3), round(distance, 3))
    else:
        print(round(value, 3), round(np.abs(value1 - value2), 3))

    bet = max_bet * ((confidence - min_confidence) / (0.5 - min_confidence)) ** 4
    print(f"Suggested bet: {bet:.3f}")


# %%
fights = [
    "cf5e8f98159c3971",
    "bec3154a11df3299",
    "e4931f3ab3bf4141",
    "66eddbe35056c06d",
    "c3ef3cb03edde8bb",
    "07cb64236ae7aaea",
    "eda1e8c285a0a788",
    "db49619221b282af",
    "90e46589a18064bc",
    "4a00db8aeff200a2",
    "1394ce1832ddefe0",
    "eba46dbf9cdd83c0",
    "14e53999507c76a7",
    "894c44c3d04aaf6f",
    "6e73ebbd089aff5e",
    "0133019a4f1055ee",
]

for id_ in fights:
    try:
        compare_fighters_from_id(id_)
    except:
        pass


# %%
def bet_result(fight, fighter, bet):
    fight_mask = odds_data["fight_id"] == fight
    fight_mask = fight_mask & (odds_data["fighter_id"] == fighter)

    # if len(odds_data[fight_mask]) == 0:
    #     return None

    american = odds_data[fight_mask]["opening"].values[0]

    if american > 0:
        return (american / 100 + 1) * bet
    else:
        return (100 / abs(american) + 1) * bet


# %%

# %%
import jupyter_black

jupyter_black.load()

# %%
odds_data = pd.read_csv(data_folder / "BestFightOdds_odds.csv")


# %%
def simulate_bets(min_confidence=0, max_diff=0.2, max_bet=5, print_info=False):
    bets = 0
    earnings = 0
    nbets = 0

    fights = 0
    corrects = 0
    fight_ids = []

    # confidence = 0.2

    X1 = torch.tensor(np.asarray([val_data[i][0] for i in range(len(val_data))]))

    X2 = torch.tensor(np.asarray([val_data[i][1] for i in range(len(val_data))]))

    odds1 = torch.tensor(np.asarray([val_data[i][3][0] for i in range(len(val_data))]))
    odds2 = torch.tensor(np.asarray([val_data[i][3][1] for i in range(len(val_data))]))

    Ys = torch.tensor(np.asarray([val_data[i][2] for i in range(len(val_data))]))

    odds1 = odds1.reshape(-1, 1)
    odds2 = odds2.reshape(-1, 1)

    model.eval()
    with torch.no_grad():
        predictions_1 = model(X1, X2, odds1, odds2).detach().numpy()
        predictions_2 = 1 - model(X2, X1, odds2, odds1).detach().numpy()

    if print_info:
        print("")

    for fight_id, Y, prediction_1, prediction_2 in zip(
        valid_test_fights,
        Ys.reshape(-1),
        predictions_1.reshape(-1),
        predictions_2.reshape(-1),
    ):
        fights += 1
        fight_row = data_aggregated[data_aggregated["fight_id"] == fight_id].iloc[0]
        f1 = fight_row["fighter_id"]
        f2 = fight_row["opponent_id"]

        winner = fight_row["winner"]
        loser = f1 if winner == f2 else f2
        diff = abs(prediction_1 - prediction_2)

        prediction = 0.5 * (prediction_1 + prediction_2)
        confidence = np.abs(prediction - 0.5)
        # if confidence < min_confidence:
        #     continue
        # elif diff > max_diff:
        #     continue

        if print_info:
            if winner == f1:
                arrow = "<-"
            else:
                arrow = "->"

            print(
                fight_row["fighter_name"],
                " vs ",
                fight_row["fighter_name_opponent"],
                arrow,
            )

            # Getting odds
            odds = []
            fight_mask = odds_data["fight_id"] == fight_id
            for fighter_id in (fight_row["fighter_id"], fight_row["opponent_id"]):
                fighter_mask = fight_mask & (odds_data["fighter_id"] == fighter_id)
                odds.append(odds_data[fighter_mask]["opening"].values[0])
            print(" vs ".join(map(str, odds)))

        # bet = (confidence - min_confidence) / (0.5 - min_confidence) * max_bet
        bet = max_bet * np.abs(prediction - 0.5) * 2
        # bet = max_bet * ((confidence - min_confidence) / (0.5 - min_confidence)) ** 4

        if Y == round(prediction):
            bets += bet
            nbets += 1
            earning = bet_result(fight_id, winner, bet)
            earnings += earning
            corrects += 1
            fight_ids.append(fight_id)
            if print_info:
                if (winner == f1 and prediction > 0.5) or (
                    winner == f2 and prediction < 0.5
                ):
                    # This weird thing is because my X1 X2 might be swapped, and there is
                    # No way I can know that for now
                    prediction = 1 - prediction
                print(round(prediction, 3), round(diff, 3))

                print(f"bet: {bet:.2f}, earn: {earning:.2f}")

        else:
            bets += bet
            nbets += 1
            if print_info:
                if (winner == f1 and prediction < 0.5) or (
                    winner == f2 and prediction > 0.5
                ):
                    # This weird thing is because my X1 X2 might be swapped, and there is
                    # No way I can know that for now
                    prediction = 1 - prediction
                print(round(prediction, 3), round(diff, 3))
                print(f"bet: {bet:.2f}, earn: {0}")

        if print_info:
            print(
                f"invested: {bets:.2f}, earnings: {earnings:.2f}, nbets: {nbets}, fights: {fights}"
            )
            print(
                f"benefits: {(earnings-bets)/bets*100:.2f}%",
            )
            print("")

        # if diff >= 0.3:
        #     continue

    # print("                                                          ", end="\r")
    # print(f"{min_confidence:.2f}\t{bets}\t{earnings:.2f}\t{nbets}", end="\r")
    return earnings, bets, nbets, corrects, fights, fight_ids


# %%
# _ = simulate_bets(min_confidence=0, max_diff=0.1, max_bet=10000, print_info=True)
_ = simulate_bets(min_confidence=0.4, max_diff=0.01, max_bet=100, print_info=True)
# _ = simulate_bets(min_confidence=0, max_diff=0.1, max_bet=1000, print_info=True)

# %%
