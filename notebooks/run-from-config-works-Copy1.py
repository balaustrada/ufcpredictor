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
predictor = UFCPredictor("/home/cramirpe/UFC/ufcpredictor/config.yaml", device="cuda")

# %%
predictor.load_trainer()

# %%
predictor.train_model()

# %%
#predictor.save_model()

# %%
predictor.load_model()
predictor.model.eval()

# %%
import matplotlib.pyplot as plt
from ufcpredictor.plot_tools import PredictionPlots

# %%
fig, ax = plt.subplots()

stats = PredictionPlots.show_fight_prediction_detail_from_dataset(
    model=predictor.model,
    dataset=predictor.test_dataset,
    fight_ids=None,
    print_info=False,
    show_plot=True,
    ax=ax,
)

ax.set_ylim(-10, 30)
ax.grid()

# %%
import pandas as pd

# %%
data_processor = predictor.data_processor

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
import pandas as pd
import numpy as np

# %%
_ = predictor.load_forecast_dataset()
forecast_dataset = predictor.forecast_dataset

# %%
credentials_file = "/home/cramirpe/UFC/reader_creds.json"
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
fight_parameters_values = [[r, w] for w, r in zip(weight, rounds)]

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
predictor.model.eval()
p1, p2 = forecast_dataset.get_forecast_prediction(
    fighter_names,
    opponent_names,
    event_dates,
    fighter_odds,
    opponent_odds,
    predictor.model,
        #     self.fight_parameters, np.asarray(fight_parameters_values).T
        # ):
        #     match_data[feature_name] = np.concatenate((stats, stats))
    fight_parameters_values,
    parse_ids=False,
    device="cpu",
)

# %%

# %%
value = (p1 + p2) / 2
confidence = abs((value - 0.5) * 2)

# %%
max_bet = 20
confidence = abs(p1 + p2 - 1)
bet = max_bet / 10 * confidence
bet = np.asarray(bet).flatten().round(2)

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

# %% [markdown]
# -----------

# %%
from ufcpredictor.plot_tools import PredictionPlots
import matplotlib.pyplot as plt
from datetime import datetime

# %%
predictor.model.eval();

# %%
fig, ax = plt.subplots()

PredictionPlots.plot_single_prediction(
    model=predictor.model,
    dataset=predictor.forecast_dataset,
    fighter_name='0d7b51c9d2649a6e',
    opponent_name='767755fd74662dbf',
    fight_parameters_values=[185, 5],
    event_date='2025-07-13',
    odds1=2.85,
    odds2=1.4444444444,
    ax=ax,
    parse_id=True
)


# %%
p1, p2 = predictor.forecast_dataset.get_forecast_prediction(
    ['0d7b51c9d2649a6e',],
    ['767755fd74662dbf',],
    ['2025-07-13',],
    [2.85,],
    [1.4444,],
    predictor.model,
    [[5,185],],
    parse_ids=True,
    device="cpu",
)

# %%
#new p1 p2
(p1 + p2)/2

# %%
p1

# %%
p2

# %%
#old_p1_p2

predictor.model.eval()
values = forecast_dataset.get_forecast_prediction(
    [fighter_names[-1],],
    [opponent_names[-1],],
    [event_dates[-1],],
    [fighter_odds[-1],],
    [opponent_odds[-1],],
    predictor.model,
    [fight_parameters_values[-1],],
    parse_ids=False,
    device="cpu",
)
p1, p2 = values
(p1 + p2)/2

# %%
predictor.model.eval()
values = forecast_dataset.get_forecast_prediction(
    fighter_names,
    opponent_names,
    event_dates,
    fighter_odds,
    opponent_odds,
    predictor.model,
    fight_parameters_values,
    parse_ids=False,
    device="cpu",
)
p1, p2= values


# %%
(p1 + p2)/2

# %%
(p1[-1] + p2[-1])/2

# %%
fighter_ids = [predictor.data_processor.get_fighter_id(x) for x in fighter_names]
opponent_ids = [predictor.data_processor.get_fighter_id(x) for x in opponent_names]

match_data_multiple= forecast_dataset.get_match_data_for_predictions(
    fighter_ids=fighter_ids,
    opponent_ids=opponent_ids,
    event_dates = event_dates,
    fighter_odds = fighter_odds,
    opponent_odds = opponent_odds,
    fight_parameters_values = fight_parameters_values,
)

match_data_single = forecast_dataset.get_match_data_for_predictions(
    fighter_ids=[fighter_ids[-1],],
    opponent_ids=[opponent_ids[-1],],
    event_dates=[event_dates[-1],],
    fighter_odds=[fighter_odds[-1],],
    opponent_odds=[opponent_odds[-1],],
    fight_parameters_values=[fight_parameters_values[-1],],
)

# %%

# %%

# %%
match_data_single

# %%
match_data_multiple[
    match_data_multiple["fighter_id"].isin(["0d7b51c9d2649a6e", "767755fd74662dbf"])
]

# %% [markdown]
# ---

# %%
from ufcpredictor.datasets import *

    # %%
    from typing import List
    def get_match_data_for_predictions(
        self,
        fighter_ids,
        opponent_ids,
        event_dates,
        fighter_odds: List[float],
        opponent_odds: List[float],
        fight_parameters_values: List[List[float]] = [],
    ) -> pd.DataFrame:
        # Start the match data with the ids and event dates
        match_data = pd.DataFrame(
            {
                "fighter_id": fighter_ids + opponent_ids,
                "event_date_forecast": event_dates * 2,
                "opening": np.concatenate((fighter_odds, opponent_odds)),
            }
        )

        # If fight features are provided, we add them to the match data
        for feature_name, stats in zip(self.fight_parameters, np.asarray(fight_parameters_values).T):
            match_data[feature_name] = np.concatenate((stats, stats))

        # We add the fighter normalized data to the match data.
        match_data = match_data.merge(
            self.data_processor.data_normalized,
            left_on="fighter_id",
            right_on="fighter_id",
        )

        # We only consider statistics prior to the fight date.
        match_data = match_data[
            match_data["event_date"] < match_data["event_date_forecast"]
        ]
        match_data = match_data.sort_values(
            by=["fighter_id", "event_date"],
            ascending=[True, False],
        )
        # Keep more up to date statistics for each fighter
        # (previous to the event_date_forecast)
        match_data = match_data.drop_duplicates(
            subset=["fighter_id", "event_date_forecast"],
            keep="first",
        )
        match_data["id_"] = (
            match_data["fighter_id"].astype(str)
            + "_"
            + match_data["event_date_forecast"].astype(str)
        )

        # Weight is the same for both fighters, so we use the first one
        match_data = match_data.rename(
            columns={
                "weight_x": "weight",
            }
        )

        
        ###############################################################
        # Now we need to fix some fields to adapt them to the match to
        # be predicted, since we are modifying the last line we are
        # modifying on top of the last fight.
        ###############################################################
        # Add time_since_last_fight information
        match_data["event_date_forecast"] = pd.to_datetime(
            match_data["event_date_forecast"]
        )
        match_data["time_since_last_fight"] = (
            match_data["event_date_forecast"] - match_data["event_date"]
        ).dt.days

        match_data["age"] = (
            match_data["event_date_forecast"] - match_data["fighter_dob"]
        ).dt.days / 365

        # We add the number of fights (is the previous + 1)
        match_data["num_fight"] = match_data["num_fight"] + 1


        new_fields = ["age", "time_since_last_fight"] + self.fight_parameters
        # Now we iterate over enhancers, in case it is a RankedField
        # we need to pass the appropriate fields to rank them.
        fields = []
        exponents = []
        for data_enhancer in self.data_processor.data_enhancers:
            if isinstance(data_enhancer, RankedFields):
                for field, exponent in zip(
                    data_enhancer.fields, data_enhancer.exponents
                ):
                    if field in new_fields:
                        exponents.append(exponent)
                        fields.append(field)


        # If there are fields to be ranked, we do so by going back to 
        # the original data and ranking them again.
        # @TODO: Revisit this and check it is working.
        if len(fields) > 0:
            ranked_fields = RankedFields(fields, exponents)

            original_df = self.data_processor.data[
                [field + "not_ranked" for field in fields]
            ].rename(columns={field + "not_ranked": field for field in fields})

            match_data[fields] = ranked_fields.add_data_fields(
                pd.concat([original_df, match_data[fields]])
            ).iloc[len(self.data_processor.data) :][fields]

       #  # Now we will normalize the fields that need to be normalized.
       #  for field in new_fields:
       #      if field in self.data_processor.normalization_factors.keys():
       #          match_data[field] /= self.data_processor.normalization_factors[field]

       # # Add fight parameters to both fighters. 
       #  # TODO: Check that concatenate is the right choice (first fighters, then 
       #  # opponents?)
       #  # It is likely that I need to add a checker on the match_data merge
       #  # to ensure that each fighter gets previous data.
       #  for feature_name, stats in zip(
       #      self.fight_parameters, np.asarray(fight_parameters_values).T
       #  ):
       #      match_data[feature_name] = np.concatenate((stats, stats))

            
        return match_data

# %%
fighter_ids = [predictor.data_processor.get_fighter_id(x) for x in fighter_names]
opponent_ids = [predictor.data_processor.get_fighter_id(x) for x in opponent_names]

match_data_multiple= get_match_data_for_predictions(
    predictor.forecast_dataset,
    fighter_ids=fighter_ids,
    opponent_ids=opponent_ids,
    event_dates = event_dates,
    fighter_odds = fighter_odds,
    opponent_odds = opponent_odds,
    fight_parameters_values = fight_parameters_values,
)

match_data_single = get_match_data_for_predictions(
    predictor.forecast_dataset,
    fighter_ids=[fighter_ids[-1],],
    opponent_ids=[opponent_ids[-1],],
    event_dates=[event_dates[-1],],
    fighter_odds=[fighter_odds[-1],],
    opponent_odds=[opponent_odds[-1],],
    fight_parameters_values=[fight_parameters_values[-1],],
)

# %%
match_data_multiple = match_data_multiple[
    match_data_multiple["fighter_id"].isin(["0d7b51c9d2649a6e", "767755fd74662dbf"])
]

# %%
match_data_single.reset_index(drop=True).compare(match_data_multiple.reset_index(drop=True))

        # %%
        match_data = pd.DataFrame(
            {
                "fighter_id": fighter_ids + opponent_ids,
                "event_date_forecast": event_dates * 2,
                "opening": np.concatenate((fighter_odds, opponent_odds)),
            }
        )

        # If fight features are provided, we add them to the match data
        for feature_name, stats in zip(self.fight_parameters, np.asarray(fight_parameters_values).T):
            match_data[feature_name] = np.concatenate((stats, stats))

        # We add the fighter normalized data to the match data.
        match_data = match_data.merge(
            self.data_processor.data_normalized,
            left_on="fighter_id",
            right_on="fighter_id",
        )

        # We only consider statistics prior to the fight date.
        match_data = match_data[
            match_data["event_date"] < match_data["event_date_forecast"]
        ]
        match_data = match_data.sort_values(
            by=["fighter_id", "event_date"],
            ascending=[True, False],
        )
        # Keep more up to date statistics for each fighter
        # (previous to the event_date_forecast)
        match_data = match_data.drop_duplicates(
            subset=["fighter_id", "event_date_forecast"],
            keep="first",
        )
        match_data["id_"] = (
            match_data["fighter_id"].astype(str)
            + "_"
            + match_data["event_date_forecast"].astype(str)
        )

        # Weight is the same for both fighters, so we use the first one
        match_data = match_data.rename(
            columns={
                "weight_x": "weight",
            }
        )

        ###############################################################
        # Now we need to fix some fields to adapt them to the match to
        # be predicted, since we are modifying the last line we are
        # modifying on top of the last fight.
        ###############################################################
        # Add time_since_last_fight information
        match_data["event_date_forecast"] = pd.to_datetime(
            match_data["event_date_forecast"]
        )
        match_data["time_since_last_fight"] = (
            match_data["event_date_forecast"] - match_data["event_date"]
        ).dt.days

        match_data["age"] = (
            match_data["event_date_forecast"] - match_data["fighter_dob"]
        ).dt.days / 365

        # We add the number of fights (is the previous + 1)
        match_data["num_fight"] = match_data["num_fight"] + 1

        new_fields = ["age", "time_since_last_fight"] + self.fight_parameters
        # Now we iterate over enhancers, in case it is a RankedField
        # we need to pass the appropriate fields to rank them.
        fields = []
        exponents = []
        for data_enhancer in self.data_processor.data_enhancers:
            if isinstance(data_enhancer, RankedFields):
                for field, exponent in zip(
                    data_enhancer.fields, data_enhancer.exponents
                ):
                    if field in new_fields:
                        exponents.append(exponent)
                        fields.append(field)

        # If there are fields to be ranked, we do so by going back to 
        # the original data and ranking them again.
        # @TODO: Revisit this and check it is working.
        if len(fields) > 0:
            ranked_fields = RankedFields(fields, exponents)

            original_df = self.data_processor.data[
                [field + "not_ranked" for field in fields]
            ].rename(columns={field + "not_ranked": field for field in fields})

            match_data[fields] = ranked_fields.add_data_fields(
                pd.concat([original_df, match_data[fields]])
            ).iloc[len(self.data_processor.data) :][fields]

        # Now we will normalize the fields that need to be normalized.
        for field in new_fields:
            if field in self.data_processor.normalization_factors.keys():
                match_data[field] /= self.data_processor.normalization_factors[field]
