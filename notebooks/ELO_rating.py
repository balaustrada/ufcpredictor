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
import matplotlib.pyplot as plt

# %%
import jupyter_black

jupyter_black.load()

# %%
import pandas as pd

pd.set_option("display.max_columns", None)

# %%
from ufcpredictor.data_processor import FlexibleELODataProcessor
from ufcpredictor.data_processor import ELODataProcessor, SumFlexibleELODataProcessor
from ufcpredictor.datasets import BasicDataset
from ufcpredictor.trainer import Trainer
import torch
import numpy as np

# %%
elodp = ELODataProcessor(
    data_folder="/home/cramirpe/UFC/UFCfightdata",
    K_factor=40,
)
elodp.load_data()

# %%
felodp = FlexibleELODataProcessor(
    data_folder="/home/cramirpe/UFC/UFCfightdata",
    K_factor=40,
    boost_values=[1, 1.2, 1.5],
)
felodp.load_data()

# %%
sfelodp = SumFlexibleELODataProcessor(
    scaling_factor=0.5,  # If performance in percentile 100, then counts like half victory (or half lose if 0).
    K_factor=30,
    data_folder="/home/cramirpe/UFC/UFCfightdata",
)
sfelodp.load_data()

# %%
X = felodp.data[
    [
        "fighter_name",
        "fighter_name_opponent",
        "event_date",
        "ELO",
        "ELO_opponent",
        "match_score",
    ]
]
X = X[X["event_date"] > "2023-01-01"]
# Only keep last match for each fighte
X = X.sort_values(by="event_date", ascending=False).groupby("fighter_name").head(1)
X = X.sort_values(by="ELO", ascending=False)
print("flexible elo data processor")
X.head(20)

# %%
X[X["fighter_name"].str.contains("Dustin")]

# %%
X = elodp.data[
    ["fighter_name", "fighter_name_opponent", "event_date", "ELO", "ELO_opponent"]
]
X = X[X["event_date"] > "2023-01-01"]
# Only keep last match for each fighte
X = X.sort_values(by="event_date", ascending=False).groupby("fighter_name").head(1)
X = X.sort_values(by="ELO", ascending=False)
print("elo_data_processor")
X.head(20)

# %%
X[X["fighter_name"].str.contains("Dustin")]

# %%
X = sfelodp.data[
    [
        "fighter_name",
        "fighter_name_opponent",
        "event_date",
        "ELO",
        "ELO_opponent",
        "match_score",
    ]
]
X = X[X["event_date"] > "2023-01-01"]
# Only keep last match for each fighte
X = X.sort_values(by="event_date", ascending=False).groupby("fighter_name").head(1)
X = X.sort_values(by="ELO", ascending=False)
print("sum_flex_elo_data_processor")
X.head(20)

# %%
X[X["fighter_name"].str.contains("Dustin")]

# %%
100 / 6

# %%
boost_factors = [1, 1.2, 1.4]
boost_factors = [1 / x for x in boost_factors][::-1] + boost_factors[1:]


def z_score(x):
    return (x - x.mean()) / x.std()


def get_percentile(x):
    return z_score(x).rank(pct=True) * 100


def get_score(x):
    n_degrees = 3  # Number ofs possible boosts

    bins = np.concatenate(
        (
            np.linspace(0, 50, 3 + 1)[:-1],
            np.linspace(50, 100, 3 + 1)[1:],
        )
    )

    return pd.cut(
        get_percentile(x), bins=bins, labels=boost_factors, right=False
    ).astype(float)


# %%
df = self.data

# %% [markdown]
# Which stats should we use to power the K-factor?
# - Difference between strikes and opponent.
# - Difference between takedowns and opponent.
# - Difference in control time.
# - Submission 
# - KO
#
#

# %%
self.data["KO"]

# %%
boost_factors

# %%
strikes_score = get_score(
    self.data["strikes_succ"] - self.data["strikes_succ_opponent"]
)

takedowns_scores = get_score(
    self.data["takedown_succ"] - self.data["takedown_succ_opponent"]
)

control_score = get_score(self.data["ctrl_time"] - self.data["ctrl_time_opponent"])

knockdown_score = get_score(self.data["knockdowns"] - self.data["knockdowns_opponent"])

submission_score = (self.data["Sub"] - self.data["Sub_opponent"]).apply(
    lambda x: boost_factors[-1] if x == 1 else (1 if x == 0 else boost_factors[0])
)

KO_score = (self.data["KO"] - self.data["KO_opponent"]).apply(
    lambda x: boost_factors[-1] if x == 1 else (1 if x == 0 else boost_factors[0])
)

win_score = (self.data["winner"] == self.data["fighter_id"]).apply(
    lambda x: boost_factors[-1] if x else 1
)

points_score = get_score(self.data["fighter_score"] - self.data["opponent_score"])


all_score = (
    strikes_score
    * takedowns_scores
    * control_score
    * knockdown_score
    * submission_score
    * KO_score
    * win_score
    * points_score
)

# %%
all_score.mean()

# %%
all_score.hist(bins=100)

# %%

# %%
get_percentile(self.data["strikes_succ"] - self.data["strikes_succ_opponent"]).hist(
    bins=50
)

# %%

# %%

# %%
z_score(self.data["strikes_succ"] - self.data["strikes_succ_opponent"]).rank(
    pct=True
).hist(bins=50)

# %%

# %%
self.scraper.fight_scraper.data

# %%

# %%
self.load_data()

# %%
df = self.data

# %%
df = df.sort_values(by="event_date", ascending=True)


# Initialize ratings
initial_rating = 1000
ratings = {}


# Function to calculate expected score
def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))


# List to keep track of updated ratings
updated_ratings = []

# K-factor for rating update
K = 100  # 32

# %%
updated_ratings = []

for _, fight in df.iterrows():
    fighter_id = fight["fighter_id"]
    opponent_id = fight["opponent_id"]
    winner_id = fight["winner"]

    # Get current ratings, or initialize if not present
    rating_fighter = ratings.get(fighter_id, initial_rating)
    rating_opponent = ratings.get(opponent_id, initial_rating)

    # Calculate expected scores
    E_fighter = expected_score(rating_fighter, rating_opponent)
    E_opponent = expected_score(rating_opponent, rating_fighter)

    # Determine scores based on the winner
    S_fighter = 1 if winner_id == fighter_id else 0
    S_opponent = 1 if winner_id == opponent_id else 0

    # Update ratings
    new_rating_fighter = rating_fighter + 100 * (S_fighter - E_fighter)
    new_rating_opponent = rating_opponent + 100 * (S_opponent - E_opponent)

    # Store the updated ratings
    ratings[fighter_id] = new_rating_fighter
    ratings[opponent_id] = new_rating_opponent

    # Append the updated ratings to the list
    updated_ratings.append(
        {
            "fight_id": fight["fight_id"],
            "fighter_id": fighter_id,
            "opponent_id": opponent_id,
            "winner": winner_id,
            "event_date": fight["event_date"],
            "ELO": new_rating_fighter,  # Store the ELO rating after the fight
        }
    )

updated_ratings_df = pd.DataFrame(updated_ratings)

# %%
data = self.data.merge(
    updated_ratings_df[["fight_id", "fighter_id", "ELO"]],
    on=["fight_id", "fighter_id"],
)

# %%
data

# %%
X = data[["fighter_name", "fighter_name_opponent", "event_date", "ELO"]]
X = X.sort_values(by="event_date", ascending=False)

# %%
# Only keep last match for each fighte
X = X.groupby("fighter_name").head(1)

# %%
X.sort_values(by="ELO", ascending=False).head(30)

# %%
X[X["fighter_name"].str.contains("Topuria")]

# %%
self.data["match_score"].isna().sum()

# %%
data = self.data

# %%
strikes_score = self.get_scores(data["strikes_succ"] - data["strikes_succ_opponent"])

takedowns_scores = self.get_scores(
    data["takedown_succ"] - data["takedown_succ_opponent"]
)

control_scores = self.get_scores(data["ctrl_time"] - data["ctrl_time_opponent"])

knockdown_scores = self.get_scores(data["knockdowns"] - data["knockdowns_opponent"])

submission_scores = self.get_scores(
    (data["Sub"] - data["Sub_opponent"]).apply(
        lambda x: (
            self.boost_factors[-1]
            if x == 1
            else (1 if x == 0 else self.boost_factors[0])
        )
    )
)

KO_scores = self.get_scores(
    (data["KO"] - data["KO_opponent"]).apply(
        lambda x: (
            self.boost_factors[-1]
            if x == 1
            else (1 if x == 0 else self.boost_factors[0])
        )
    )
)

win_score = (data["winner"] == data["fighter_id"]).apply(
    lambda x: self.boost_factors[-1] if x else 1
)

points_score = self.get_scores(self.data["fighter_score"] - self.data["opponent_score"])

# %%
strikes_score.isna().sum()

# %%
