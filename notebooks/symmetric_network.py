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
plt.style.use('tableau-colorblind10')

# %%
from ufcscraper.ufc_scraper import UFCScraper
from ufcscraper.odds_scraper import BestFightOddsScraper

# %%
scraper = UFCScraper(data_folder = "/home/cramirez/kaggle/ufc_scraper/data")
bfo_scraper = BestFightOddsScraper(
    data_folder = "/home/cramirez/kaggle/ufc_scraper/data",
    n_sessions = -1, # do not open any session.
)

# %%
# This has fight data by fighter, and can also be merged with odds data easily
data = bfo_scraper.get_ufcstats_data()

# %%
# Now it also has odds data
data = data.merge(
    bfo_scraper.data,
    on=["fight_id", "fighter_id"],
)

# %%
# Now we need to fill the stats, they are given by round, so we first need to group by fight,
# but let's take a quick look at the data first.
round_data = scraper.fight_scraper.rounds_handler.data
round_data.head()


# %%
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

fight_data = round_data.groupby(
    ["fight_id", "fighter_id"]
).sum().reset_index().drop("round", axis=1)

# %%
max_len = max(len(c) for c in fight_data.columns)

for c, v in zip(fight_data.columns, fight_data.values[0]):
    print(f"{c.ljust(max_len)}\t\t{v}")

# %%
# We can now join with the other data:
data = data.merge(
    fight_data,
    on=["fight_id", "fighter_id"],
)

# Join with column to get general data from the fight
data = data.merge(
    scraper.fight_scraper.data[["fight_id", "winner", "num_rounds", "weight_class", "gender", "result", "result_details", "finish_round", "finish_time", "time_format"]],
    on=["fight_id"],
)

data["finish_time"] = data["finish_time"].apply(convert_to_seconds)
data["total_time"] = (data["finish_round"]-1)*5*60 + data["finish_time"]
data = data[
    data["time_format"].isin(['3 Rnd (5-5-5)', '5 Rnd (5-5-5-5-5)'])
]

# %%
data = data[
    (data["weight_class"] != "NULL") & 
    (data["weight_class"] != "Catch Weight") & 
    (data["weight_class"] != "Open Weight")
]

# %%
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

# %%
# Only keep male fights for now.
data = data[data["gender"] == "M"]

# %%
# Remove disqualified and doctor's stoppage
data = data[data["result"].isin(["Decision", "KO/TKO", "Submission"])]

# %%

# %%
data['KO'] = np.where(
    (data['result'].str.contains("KO")) & (data['winner'] == data['fighter_id']), 1, 0
)

data['KO_opp'] = np.where(
    (data['result'].str.contains("KO")) & (data['winner'] != data['fighter_id']), 1, 0
)

data['Sub'] = np.where(
    (data['result'].str.contains("Submission")) & (data['winner'] == data['fighter_id']), 1, 0
)

data['Sub_opp'] = np.where(
    (data['result'].str.contains("Submission")) & (data['winner'] != data['fighter_id']), 1, 0
)

data['win'] = np.where(data['winner'] == data['fighter_id'], 1, 0)

# %%
data["UFC_names"] = data["UFC_names"].apply(lambda x: x[0])
data["opponent_UFC_names"] = data["opponent_UFC_names"].apply(lambda x: x[0])
data = data.sort_values(by=["fighter_id", "event_date"])

# %%
to_sum_columns = [
	"knockdowns",
	"strikes_att",
	"strikes_succ",
	"head_strikes_att",
	"head_strikes_succ",
	"body_strikes_att",
	"body_strikes_succ",
	"leg_strikes_att",
	"leg_strikes_succ",
	"distance_strikes_att",
	"distance_strikes_succ",
	"ground_strikes_att",
	"ground_strikes_succ",
	"clinch_strikes_att",
	"clinch_strikes_succ",
	"total_strikes_att",
	"total_strikes_succ",
	"takedown_att",
	"takedown_succ",
	"submission_att",
	"reversals",
	"ctrl_time",
	"num_rounds",
	"total_time",
	"KO",
	"KO_opp",
	"Sub",
	"Sub_opp",
	"win",
]

# %%
data_ag = data.sort_values(by=["fighter_id", "event_date"])
data_ag["num_fight"] = data_ag.groupby("fighter_id").cumcount() + 1

# %%
data_ag[data_ag["UFC_names"].str.contains("Holloway")]

# %%
for column in to_sum_columns:
    data_ag[column] = data_ag[column].astype(float)
    data_ag[column] = data_ag.groupby("fighter_id")[column].cumsum()

# %%
data_ag[data_ag["UFC_names"].str.contains("Holloway")]

# %%
for column in to_sum_columns:
    data_ag[f"{column}_per_min"] = data_ag[column] / data_ag['total_time']
    data_ag[f"{column}_per_fight"] = data_ag[column] / data_ag['num_fight']

# %%
# existing_columns = list(set(df.columns) - set(to_sum_columns) - set(initial_columns))

# # Select relevant columns
# selected_columns = existing_columns + [col for col in df.columns if col not in existing_columns]

# # Final DataFrame with selected columns
# df_selected = df[selected_columns]

# %% [markdown]
# ## Normalize data fields

# %%
"knockdowns_per_min" in data_ag.columns

# %%
X_set = [
	"knockdowns_per_min",
	"strikes_att_per_min",
	"strikes_succ_per_min",
	"head_strikes_att_per_min",
	"head_strikes_succ_per_min",
	"body_strikes_att_per_min",
	"body_strikes_succ_per_min",
	"leg_strikes_att_per_min",
	"leg_strikes_succ_per_min",
	"distance_strikes_att_per_min",
	"distance_strikes_succ_per_min",
	"ground_strikes_att_per_min",
	"ground_strikes_succ_per_min",
	"clinch_strikes_att_per_min",
	"clinch_strikes_succ_per_min",
	"total_strikes_att_per_min",
	"total_strikes_succ_per_min",
	"takedown_att_per_min",
	"takedown_succ_per_min",
	"submission_att_per_min",
	"reversals_per_min",
	"ctrl_time_per_min",
	"KO_per_min",
	"KO_opp_per_min",
	"Sub_per_min",
	"Sub_opp_per_min",
	"win_per_min",
]

# %%
for x in X_set:
    assert x in data_ag.columns

# %%
means = {f"{c}_mean": data_ag[c].mean() for c in X_set}

# %%
data_ag["num_fights_normalized"] = data_ag["num_fight"]

# %%
for column in X_set:
    mean_value = means[f"{column}_mean"]
    data_ag[column] = data_ag[column] / mean_value

# %% [markdown]
# ______________________

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
invalid_fights = list(data_ag[data_ag["total_strikes_succ_per_min"].isna()]["fight_id"])
data_ag = data_ag[~data_ag["fight_id"].isin(invalid_fights)]

# %%
invalid_fights = list(data_ag[data_ag["num_fights_normalized"] < 3]['fight_id'])
valid_fights = set(data_ag['fight_id']) - set(invalid_fights)

# %%
len(valid_fights)


# %%
def from_id_to_fight(id_, print_fighters=False):
    # Get fighters
    fight = data_ag[data_ag["fight_id"] == id_].iloc[0] 
    f1 = fight['fighter_id']
    f2 = fight['opponent_id']  
    date = fight['event_date']
    winner = fight['winner']

    f1p = data_ag[
        (data_ag["event_date"] < date) &
        (data_ag["fighter_id"] == f1)
    ]
    f1p = f1p.iloc[f1p["event_date"].argmax()]
    
    f2p = data_ag[
        (data_ag["event_date"] < date) &
        (data_ag["fighter_id"] == f2)
    ]
    f2p = f2p.iloc[f2p["event_date"].argmax()]


    x1 = [f1p[x] for x in X_set]
    x2 = [f2p[x] for x in X_set]


    if print_fighters:
        print(fight["UFC_names"], " vs ", fight["opponent_UFC_names"])
        
    return (
        torch.FloatTensor(x1), 
        torch.FloatTensor(x2), 
        torch.FloatTensor(
            [float(winner == f2p['fighter_id'])]
        )
    )

# %%
data = []
for id_ in sorted(valid_fights):
    data.append(from_id_to_fight(id_))


# %%
class CustomDataset(Dataset):
    def __init__(self, data, mode = "train"):
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X1, X2, Y = self.data[idx]

        if np.random.random() >= 0.5:
            X1, X2 = X2, X1
            Y = 1 -Y

        if self.mode == "train":
            return X1, X2, Y
        else:
            return X1, X2
            


# %%
train_data = data[:int(len(data)*0.75)]
val_data = data[int(len(data)*0.75):]

# %%
len(data[0][1])*2

# %%
train_dt = CustomDataset(train_data, mode = "train")
val_dt = CustomDataset(val_data, mode="train")
full_dt = CustomDataset(data, mode="train")

train_dataloader = torch.utils.data.DataLoader(train_dt, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dt, batch_size=64, shuffle=False)
full_dataloader = torch.utils.data.DataLoader(full_dt, batch_size=64, shuffle=False)


# %%
class FighterNet(nn.Module):
    def __init__(self):
        super(FighterNet, self).__init__()
        self.fc1 = nn.Linear(len(data[0][1]), 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class SymmetricFightNet(nn.Module):
    def __init__(self):
        super(SymmetricFightNet, self).__init__()
        self.fighter_net = FighterNet()
        
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X1, X2):
        out1 = self.fighter_net(X1)
        out2 = self.fighter_net(X2)

        x = torch.cat((out1 - out2, out2-out1), dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x
        



# %%
DROPOUT_PROB = 0.3
class FighterNet(nn.Module):
    def __init__(self):
        super(FighterNet, self).__init__()
        self.fc1 = nn.Linear(len(data[0][1]), 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Use the global dropout probability
        self.dropout1 = nn.Dropout(p=DROPOUT_PROB)
        self.dropout2 = nn.Dropout(p=DROPOUT_PROB)
        self.dropout3 = nn.Dropout(p=DROPOUT_PROB)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after the first ReLU
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after the second ReLU
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)  # Apply dropout after the third ReLU
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

    def forward(self, X1, X2):
        out1 = self.fighter_net(X1)
        out2 = self.fighter_net(X2)

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
def train(model, optimizer, train_dataloader, val_dataloader, scheduler, device, num_epochs):
    model.to(device)

    criterion = {
        'target': nn.BCELoss().to(device)
    }

    best_loss = 999999
    best_model = None

    target_preds = []
    target_labels = []

    for epoch in range(1, num_epochs+1):
        model.train()
        train_loss = []

        for X1, X2, Y in tqdm(iter(train_dataloader)):
            X1, X2 = X1.to(device), X2.to(device)
            Y = Y.to(device)


            optimizer.zero_grad()
            target_logit = model(X1, X2)
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

            try:
                loss = criterion['target'](target_logit, Y)
            except:
                import pdb; pdb.set_trace()

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            target_preds += torch.round(target_logit).detach().cpu().numpy().tolist()
            target_labels += Y.detach().cpu().numpy().tolist()

        
        match = np.asarray(target_preds).reshape(-1) == np.asarray(target_labels).reshape(-1)
        
        val_loss, val_target_f1, correct, _, _ = validation(model, val_dataloader, criterion, device)

        print(f"Train acc: [{match.sum() / len(match):.5f}]")
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Disaster? F1 : [{val_target_f1:.5f}] Correct: [{correct*100:.2f}]')

        if scheduler is not None:
            scheduler.step(val_loss)

        if best_loss > val_loss:
            best_los = val_loss
            best_model = model
            torch.save(model, "./best-model.pth")
            print("Model saved!")

    return best_model

# %%

# %%
odds_data = bfo_scraper.data


# %%

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
        for X1, X2, Y in tqdm(iter(val_dataloader)):
            X1, X2 = X1.to(device), X2.to(device)
            Y = Y.to(device)
            
            target_logit = model(X1, X2)
            # target_logit_2 = model(X2, X1)
            
            # target_logit_2 = 1 - target_logit_2
            # target_logit = (target_logit + target_logit_2) / 2

            loss = criterion["target"](target_logit, Y)

            val_loss.append(loss.item())

            
            target += target_logit
            target_preds += torch.round(target_logit).detach().cpu().numpy().tolist()
            target_labels += Y.detach().cpu().numpy().tolist()



           
    match = np.asarray(target_preds).reshape(-1) == np.asarray(target_labels).reshape(-1)          
    
    target_f1 = f1_score(target_labels, target_preds, average='macro')

    return np.mean(val_loss), target_f1, match.sum()/len(match), target, target_labels

# %%
# train!
model = SymmetricFightNet()
model.eval()

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.5e-3)
scheduler = None
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

# %%
infer_model = train(model, optimizer, train_dataloader, val_dataloader, scheduler, device="cpu", num_epochs=20)


# %%
def compare_fighters_from_id(id_):
    x1, x2, y = from_id_to_fight(
        id_,
        print_fighters=True
    )

    x1 = torch.reshape(x1, (1, -1))
    x2 = torch.reshape(x2, (1, -1))

    value1 = float(model(x1, x2))
    value2 = 1-float(model(x2, x1))
    
    value = (value1 + value2)/2

    print(round(value, 3), round(np.abs(value1-value2), 3))
    

def compare_fighters(fighter1, fighter2, date):
    fighter1 = data_ag["fighter_id"][data_ag["UFC_names"].str.contains(fighter1)].iloc[0]
    fighter2 = data_ag["fighter_id"][data_ag["UFC_names"].str.contains(fighter2)].iloc[0]
    f1p = data_ag[
        (data_ag["event_date"].dt.date < date) &
        (data_ag["fighter_id"].str.contains(fighter1))
    ]
    f1p = f1p.iloc[f1p["event_date"].argmax()]

    f2p = data_ag[
        (data_ag["event_date"].dt.date < date) &
        (data_ag["fighter_id"].str.contains(fighter2))
    ]
    f2p = f2p.iloc[f2p["event_date"].argmax()]

    x1 = [f1p[x] for x in X_set]
    x2 = [f2p[x] for x in X_set]

    x1 = torch.reshape(torch.FloatTensor(x1), (1, -1))
    x2 = torch.reshape(torch.FloatTensor(x2), (1, -1))

    print(f1p['UFC_names'], " vs ", f2p['UFC_names'])

    value1= float(model(x1, x2))
    value2 = 1-float(model(x2, x1))

   
    value = (value1 + value2)/2
    diff = np.abs(value1-value2)
    
    
    
    print(round(value, 3), round(np.abs(value1-value2), 3))
    


        



# %%
from datetime import date

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
compare_fighters("France", "Erceg", date.today())

# %%

# %%

# %%

# %%

# %%
len(data)

# %%
len(valid_fights)


# %%
def bet_result(fight, fighter):
    fight_mask = odds_data["fight_id"] == fight
    fight_mask = fight_mask & (odds_data["fighter_id"] == fighter)

    # if len(odds_data[fight_mask]) == 0:
    #     return None
        
    american = odds_data[fight_mask]["opening"].values[0]

    if american > 0:
        return american/100
    else:
        return 100/abs(american)


# %%
validation_fights = sorted(valid_fights)[int(len(data)*0.75):]

# %%
index = 0
fight_id = validation_fights[index]

# %%
device = "cpu"
with torch.no_grad():
    X1, X2, Y=  val_data[index]
    X1, X2 = X1.to(device), X2.to(device)
    Y = Y.to(device)

    target_logit = model(X1.reshape(1, -1), X2.reshape(1, -1))


# %%
def simulate_bets(confidence=0):
    bets = 0 
    earnings = 0
    # confidence = 0.2
    
    for i, fight_id in enumerate(validation_fights):
        X1, X2, Y = val_data[i]
    
        X1, X2 = X1.to(device), X2.to(device)
        Y = Y.to(device)
        
        target_logit = model(X1.reshape(1, -1), X2.reshape(1, -1))
    
        fight_row = data_ag[data_ag["fight_id"] == fight_id].iloc[0]
        f1 = fight_row["fighter_id"]
        f2=  fight_row["opponent_id"]
        
        winner = fight_row["winner"]
        prediction = target_logit[0][0]
    
        # print(Y[0], prediction)
    
        if (prediction <= 0.5 - confidence) or (prediction >= 0.5 + confidence):
            if Y[0] == torch.round(prediction):
                _ = bet_result(fight_id, winner)
                if _ is not None:
                    bets +=1
                    earnings += _
                else:
                    print("None")
            else:
                bets+=1
                earnings -=1
    print(confidence, end='\r')
    return earnings, bets


# %%
import jupyter_black
jupyter_black.load()


# %%
def simulate_bets(confidence=0):
    bets = 0
    earnings = 0
    # confidence = 0.2

    X1 = torch.tensor(np.asarray([val_data[i][0] for i in range(len(val_data))]))

    X2 = torch.tensor(np.asarray([val_data[i][1] for i in range(len(val_data))]))

    Ys = torch.tensor(np.asarray([val_data[i][2] for i in range(len(val_data))]))

    predictions_1 = model(X1, X2)
    predictions_2 = model(X1, X2)

    for fight_id, Y, prediction_1, prediction_2 in zip(
        validation_fights,
        Ys.reshape(-1),
        predictions_1.reshape(-1),
        predictions_2.reshape(-1),
    ):
        fight_row = data_ag[data_ag["fight_id"] == fight_id].iloc[0]
        f1 = fight_row["fighter_id"]
        f2 = fight_row["opponent_id"]

        winner = fight_row["winner"]
        loser = f1 if winner == f2 else f2

        if torch.round(prediction_1) == torch.round(prediction_2):
            prediction = 0.5 * (prediction_1 + prediction_2)
        else:
            continue

        # print(Y[0], prediction)
        if (prediction <= 0.5 - confidence) or (prediction >= 0.5 + confidence):
            if Y == torch.round(prediction):
                _ = bet_result(fight_id, winner)
                if _ > 1.5:
                    bets += 1
                    earnings += _

            else:
                _ = bet_result(fight_id, loser)
                if _ > 1.5:
                    bets += 1
                    earnings -= 1
    print("                                       ", end="\r")
    print(f"{confidence:.2f}\t{bets}\t{earnings:.2f}", end="\r")
    return earnings, bets


# %%
confidences = np.linspace(0, 0.5, 15)
earnings = []
bets = []

for c in confidences:
    e, b = simulate_bets(c)
    earnings.append(e)
    bets.append(b)

# %%
import matplotlib.pyplot as plt

# %%
eranings = np.asarray(earnings)
bets = np.asarray(bets)

# %%
fig, ax = plt.subplots()

ax.plot(confidences, earnings / bets)
ax.axhline(y=0, color="k")

# %%
fig, ax = plt.subplots()

ax.plot(confidences, bets)

# %%
