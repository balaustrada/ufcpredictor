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
from typing import *

# %%
import numpy as np

# %%
from datetime import date

# %%
from ufcpredictor.win_predictor import WinPredictor

# %%
from ufcpredictor.train_methods import train, validation
from ufcpredictor.networks import SymmetricFightNet

# %%
import torch
import random


# %%
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU
np.random.seed(seed)
random.seed(seed)

# For CuDNN backend (GPU)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %%
predictor = WinPredictor(
    "/home/cramirez/kaggle/ufc_scraper/data"
)

# %%
train_fights = random.sample(
    sorted(predictor.data_processor.valid_fights),
    int(len(predictor.data_processor.valid_fights) * 0.7),
)
test_fights = list(predictor.data_processor.valid_fights - set(train_fights))

# %%
data = predictor.data_processor.data
train_fights = (
    data["fight_id"][data["event_date"] < "2024-01-01"].unique().tolist()
)
test_fights = (
    data["fight_id"][data["event_date"] >= "2024-01-01"].unique().tolist()
)

train_fights = [id_ for id_ in train_fights if id_ in predictor.data_processor.valid_fights]
test_fights =  [id_ for id_ in test_fights if id_ in predictor.data_processor.valid_fights]

# %%
train_dataloader = predictor.data_processor.get_data_loader(
    fight_ids= train_fights, 
    X_set = predictor.X_set,
    batch_size=64,
)

# %%
test_dataloader = predictor.data_processor.get_data_loader(
    fight_ids = test_fights,
    X_set=predictor.X_set,
    batch_size=64,
)

# %%
model = SymmetricFightNet(input_size=len(predictor.X_set))
model.eval()

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.5e-3)
scheduler = None
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# %%
infer_model = train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    device="cpu",
    num_epochs=5,
)

# %%
import jupyter_black

jupyter_black.load()


# %%
def compare_fighters_from_id(id_):
    x1, x2, y = predictor.data_processor.from_id_to_fight(
        X_set=predictor.X_set,
        id_=id_,
        print_info=True,
    )

    x1 = torch.reshape(x1, (1, -1))
    x2 = torch.reshape(x2, (1, -1))

    model.eval()
    with torch.no_grad():
        value1 = float(model(x1, x2))
        value2 = 1 - float(model(x2, x1))

    value = (value1 + value2) / 2

    print(round(value, 3), round(np.abs(value1 - value2), 3))


# %%
def compare_fighters(
    fighter1,
    fighter2,
    date,
    show_distance,
    min_confidence=0,
    max_diff=0.2,
    max_bet=10000,
):
    data = predictor.data_processor

    fighter1 = data["fighter_id"][data["UFC_names"].str.contains(fighter1)].iloc[0]
    fighter2 = data["fighter_id"][data["UFC_names"].str.contains(fighter2)].iloc[0]
    f1p = data[
        (data["event_date"].dt.date < date)
        & (data["fighter_id"].str.contains(fighter1))
    ]
    f1p = f1p.iloc[f1p["event_date"].argmax()]

    f2p = data[
        (data["event_date"].dt.date < date)
        & (data["fighter_id"].str.contains(fighter2))
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
        print("")
    except:
        pass

# %%

    odds_data = predictor.data_processor.bfo_scraper.data


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
def simulate_bets(min_confidence=0, max_diff=0.2, max_bet=5, print_info=False):
    bets = 0
    earnings = 0
    nbets = 0

    fights = 0
    corrects = 0
    fight_ids = []

    # confidence = 0.2

    predictions_1 = []
    predictions_2 = []
    Ys = []

    model.eval()
    with torch.no_grad():
        for batch_X1, batch_X2, batch_Ys in test_dataloader:
            predictions_1.append(model(batch_X1, batch_X2).detach().numpy())
            predictions_2.append(1 - model(batch_X2, batch_X1).detach().numpy())
            Ys.append(batch_Ys)

    predictions_1 = np.concatenate(predictions_1).reshape(-1)
    predictions_2 = np.concatenate(predictions_2).reshape(-1)
    Ys = np.concatenate(Ys).reshape(-1)

    data = predictor.data_processor.data

    if print_info:
        print("")

    for fight_id, Y, prediction_1, prediction_2 in zip(
        test_fights,
        Ys,
        predictions_1,
        predictions_2,
    ):
        fights += 1
        fight_row = data[data["fight_id"] == fight_id].iloc[0]
        f1 = fight_row["fighter_id"]
        f2 = fight_row["opponent_id"]

        winner = fight_row["winner"]
        loser = f1 if winner == f2 else f2
        diff = abs(prediction_1 - prediction_2)

        prediction = 0.5 * (prediction_1 + prediction_2)
        confidence = np.abs(prediction - 0.5)
        if confidence < min_confidence:
            continue
        elif diff > max_diff:
            continue

        if print_info:
            if winner == f1:
                arrow = "<-"
            else:
                arrow = "->"

            print(
                fight_row["UFC_names"], " vs ", fight_row["opponent_UFC_names"], arrow
            )

            # Getting odds
            odds = []
            fight_mask = odds_data["fight_id"] == fight_id
            for fighter_id in (fight_row["fighter_id"], fight_row["opponent_id"]):
                fighter_mask = fight_mask & (odds_data["fighter_id"] == fighter_id)
                odds.append(odds_data[fighter_mask]["opening"].values[0])
            print(" vs ".join(map(str, odds)))

        # if (Y == round(prediction)) and (
        #     (winner == f1 and prediction > 0.5) or (winner == f2 and prediction < 0.5)
        # ):
        #     odds = odds[::-1]
        # elif (winner == f1 and prediction < 0.5) or (winner == f2 and prediction > 0.5):
        #     odds = odds[::-1]

        # if prediction > 0.5:
        #     odd = odds[1]
        # else:
        #     odd = odds[0]

        # if odd > 0:
        #     odd = odd / 100 + 1
        # else:
        #     odd = 100 / odd + 1

        # bet = odd * np.abs(0.5 - prediction)
        # # bet = (confidence - min_confidence) / (0.5 - min_confidence) * max_bet
        # bet *= max_bet * ((confidence - min_confidence) / (0.5 - min_confidence)) ** 4
        bet = max_bet * ((confidence - min_confidence) / (0.5 - min_confidence)) ** 4

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
            print("")

        # if diff >= 0.3:
        #     continue

    # print("                                                          ", end="\r")
    # print(f"{min_confidence:.2f}\t{bets}\t{earnings:.2f}\t{nbets}", end="\r")
    return earnings, bets, nbets, corrects, fights, fight_ids


# %%
_ = simulate_bets(min_confidence=0.05, max_diff=0.1, max_bet=10000, print_info=True)
# _ = simulate_bets(min_confidence=0, max_diff=0.1, max_bet=1000, print_info=True)

# %%

# %%

# %%
data = predictor.data_processor.data

# %%
import datetime

# %%
X_set = predictor.X_set

# %%
compare_fighters(
    "Malley",
    "Merab",
    datetime.date(2024, 9, 15),
    True,
    max_diff=0.1,
    max_bet=1000,
    min_confidence=0.05,
)

# %%
compare_fighters(
    "Topuria",
    "Holloway",
    datetime.date(2024, 10, 15),
    True,
    max_diff=0.1,
    max_bet=1000,
    min_confidence=0.05,
)

# %%
compare_fighters(
    "Malley",
    "Merab",
    datetime.date(2024, 9, 15),
    True,
    max_diff=0.1,
    max_bet=1000,
    min_confidence=0.05,
)


# %%
def compare_fighters(
    fighter1,
    fighter2,
    date,
    show_distance,
    min_confidence=0,
    max_diff=0.2,
    max_bet=10000,
):
    fighter1 = data["fighter_id"][data["UFC_names"].str.contains(fighter1)].iloc[0]
    fighter2 = data["fighter_id"][data["UFC_names"].str.contains(fighter2)].iloc[0]
    f1p = data[
        (data["event_date"].dt.date < date)
        & (data["fighter_id"].str.contains(fighter1))
    ]
    f1p = f1p.iloc[f1p["event_date"].argmax()]

    f2p = data[
        (data["event_date"].dt.date < date)
        & (data["fighter_id"].str.contains(fighter2))
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

# %%
