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
import textwrap
import yaml
from pathlib import Path

# %%
config = textwrap.dedent(
    """
statistics:
  fighter fight statistics: &fighter_fight_statistics
    - age
    # - notice_days
    # - body_strikes_att_opponent_per_minute
    # - body_strikes_att_per_minute
    - body_strikes_succ_opponent_per_minute
    - body_strikes_succ_per_minute
    # - clinch_strikes_att_opponent_per_minute
    # - clinch_strikes_att_per_minute
    - clinch_strikes_succ_opponent_per_minute
    - clinch_strikes_succ_per_minute
    - ctrl_time_opponent_per_minute
    - ctrl_time_per_minute
    # - distance_strikes_att_opponent_per_minute
    # - distance_strikes_att_per_minute
    - distance_strikes_succ_opponent_per_minute
    - distance_strikes_succ_per_minute
    - fighter_height_cm
    # - ground_strikes_att_opponent_per_minute
    # - ground_strikes_att_per_minute
    - ground_strikes_succ_opponent_per_minute
    - ground_strikes_succ_per_minute
    # - head_strikes_att_opponent_per_minute
    # - head_strikes_att_per_minute
    - head_strikes_succ_opponent_per_minute
    - head_strikes_succ_per_minute
    - knockdowns_opponent_per_minute
    - knockdowns_per_minute
    # - KO_opponent_per_fight
    - KO_opponent_per_minute
    - KO_per_fight
    - KO_per_minute
    # - leg_strikes_att_opponent_per_minute
    # - leg_strikes_att_per_minute
    - leg_strikes_succ_opponent_per_minute
    - leg_strikes_succ_per_minute
    - num_fight
    - reversals_opponent_per_minute
    - reversals_per_minute
    # - strikes_att_opponent_per_minute
    # - strikes_att_per_minute
    - strikes_succ_opponent_per_minute
    - strikes_succ_per_minute
    - Sub_opponent_per_fight
    - Sub_opponent_per_minute
    - Sub_per_fight
    - Sub_per_minute
    - submission_att_opponent_per_minute
    - submission_att_per_minute
    - takedown_att_opponent_per_minute
    - takedown_att_per_minute
    - takedown_succ_opponent_per_minute
    - takedown_succ_per_minute
    - time_since_last_fight
    # - total_strikes_att_opponent_per_minute
    # - total_strikes_att_per_minute
    - total_strikes_succ_opponent_per_minute
    - total_strikes_succ_per_minute
    - win_opponent_per_fight
    - win_per_fight
    - ELO
  
  previous fights statistics: &previous_fights_statistics
    - "age"
    # - "notice_days"
    # - "body_strikes_att_opponent_per_minute"
    # - "body_strikes_att_per_minute"
    - "body_strikes_succ_opponent_per_minute"
    - "body_strikes_succ_per_minute"
    # - "clinch_strikes_att_opponent_per_minute"
    # - "clinch_strikes_att_per_minute"
    - "clinch_strikes_succ_opponent_per_minute"
    - "clinch_strikes_succ_per_minute"
    - "ctrl_time_opponent_per_minute"
    - "ctrl_time_per_minute"
    # - "distance_strikes_att_opponent_per_minute"
    # - "distance_strikes_att_per_minute"
    - "distance_strikes_succ_opponent_per_minute"
    - "distance_strikes_succ_per_minute"
    # - "fighter_height_cm"
    # - "ground_strikes_att_opponent_per_minute"
    # - "ground_strikes_att_per_minute"
    - "ground_strikes_succ_opponent_per_minute"
    - "ground_strikes_succ_per_minute"
    # - "head_strikes_att_opponent_per_minute"
    # - "head_strikes_att_per_minute"
    - "head_strikes_succ_opponent_per_minute"
    - "head_strikes_succ_per_minute"
    - "knockdowns_opponent_per_minute"
    - "knockdowns_per_minute"
    # - "KO_opponent_per_fight"
    - "KO_opponent_per_minute"
    # - "KO_per_fight"
    - "KO_per_minute"
    # - "leg_strikes_att_opponent_per_minute"
    # - "leg_strikes_att_per_minute"
    - "leg_strikes_succ_opponent_per_minute"
    - "leg_strikes_succ_per_minute"
    # - "num_fight"
    # - "reversals_opponent_per_minute"
    # - "reversals_per_minute"
    # - "strikes_att_opponent_per_minute"
    # - "strikes_att_per_minute"
    - "strikes_succ_opponent_per_minute"
    - "strikes_succ_per_minute"
    # - "Sub_opponent_per_fight"
    - "Sub_opponent_per_minute"
    # - "Sub_per_fight"
    - "Sub_per_minute"
    - "submission_att_opponent_per_minute"
    - "submission_att_per_minute"
    # - "takedown_att_opponent_per_minute"
    # - "takedown_att_per_minute"
    - "takedown_succ_opponent_per_minute"
    - "takedown_succ_per_minute"
    - "time_since_last_fight" # Adding this somehow slowed the convergence and is not as good (why?) maybe because of the default value used(?) it was the mean (~ 7months)
    # "total_strikes_att_opponent_per_minute"
    # - "total_strikes_att_per_minute"
    - "total_strikes_succ_opponent_per_minute"
    - "total_strikes_succ_per_minute"
    # - "win_opponent_per_fight"
    # - "win_per_fight"
    - "ELO"
  
  fight parameters: &fight_parameters
    - num_rounds
    - weight

  previous fights parameters: &previous_fights_parameters
    - num_rounds
    - weight
    - winner

general:
  state size: &state_size 6
  mlflow tracking: False
  model filename: tmp.pt

data processor:
    class: DataProcessor
    args:
      data_folder: /home/cramirpe/UFC/UFCfightdata
    
    data aggregator: 
      class: WeightedDataAggregator
      args:
        alpha: -0.0001
    
    data enhancers:
      - class: SumFlexibleELO
        args:
          scaling_factor: 0
          K_factor: 30
      - class: RankedFields
        args:
          fields:
            - age
            - fighter_height_cm
          exponents:
            - 1.2
            - 1.2

filters:
  minimum fight number: 5
  minimum notice days: 60
  early split date: 2018-01-01
  split date: 2023-08-01
  max date: 2025-11-11

training:
  seed: 102
  batch size: 64
  early train epochs: 10
  train epochs: 10

dataset:
  class: DatasetWithTimeEvolution
  args:
    fighter_fight_statistics: *fighter_fight_statistics
    fight_parameters: *fight_parameters
    previous_fights_parameters: *previous_fights_parameters
    previous_fights_statistics: *previous_fights_statistics
    state_size: *state_size

forecast dataset:
  class: ForecastDatasetTimeEvolution
  args:
    fighter_fight_statistics: *fighter_fight_statistics
    fight_parameters: *fight_parameters
    previous_fights_parameters: *previous_fights_parameters
    previous_fights_statistics: *previous_fights_statistics
    state_size: *state_size


model:
  class: SimpleFightNetWithTimeEvolution
  args:
    fighter_fight_statistics: *fighter_fight_statistics
    fight_parameters: *fight_parameters
    dropout_prob: 0.45
    network_shape: [64, 32, 16,  1]
    state_size: *state_size
    fighter_transformer_kwargs:
      state_size: *state_size
      fighter_fight_statistics: *previous_fights_statistics
      fight_parameters: *previous_fights_parameters

      network_shape: [64, 32]
      dropout: 0.405

optimizer:
  class: Adam
  args:
    lr: 1.3e-3
    # weight_decay: 1.e-5
  
# scheduler:
#   class: ReduceLROnPlateau
#   args:
#     mode: min
#     factor: 0.7
#     patience: 2

loss:
  class: BettingLoss
  args: {}
"""
)

# %%
Path("/tmp/config.yaml").write_text(config)

# %%
i = 2
print("Loading model for seed: ", i, end="\r")
predictor = UFCPredictor(
    "/tmp/config.yaml",
    device="cpu",
)

predictor.load_trainer()
# predictors[-1].load_model()
# predictors[-1].train_model()
# predictors[-1].model.eval();

# %%
predictor.config["training"]["seed"]

# %%
self = predictor

# %%
predictor.trainer.train(
    epochs=20,
    train_dataloader=predictor.early_train_dataloader,
    test_dataloader=predictor.test_dataloader,
)

# %%
predictor.trainer.train(
    epochs=5,
    test_dataloader=predictor.test_dataloader,
)

# %%
import jupyter_black


jupyter_black.load()

# %%
fig, ax = plt.subplots()
simulate_betting(
    predictor,
    ax1=ax,
    initial_cash=100,
    max_bet_prop=0.2,
    min_max_bet=20,
    max_parlay_size=1,
)

# %%

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ufcpredictor.plot_tools import PredictionPlots

from itertools import combinations

# %%
import math


def bet_strategy(
    confidence: float,
    max_bet: float,
    odds: float = None,
):
    use = 2
    match use:
        case 1:
            return confidence**4 * max_bet
        case 2:
            threshold = 0.3
            growth = 2
            base = 5  # math.e

            scale = (confidence - threshold) * growth
            return max(max_bet * (base**scale - 1) / (base - 1), 0)
        case 3:
            bankroll = max_bet
            edge = confidence**2 * (odds - 1) - (1 - confidence**2)
            return max(bankroll * edge / (odds - 1), 0)
        case _:
            raise ValueError


def simulate_betting(
    predictor: UFCPredictor,
    ax1,
    initial_cash=100,
    max_parlay_size=1,
    max_bet_prop=0.4,  # Max bet in proportion with available cash
    min_max_bet=20,  # Minimum maximum bet (even if cash is lower)
):
    stats = PredictionPlots.show_fight_prediction_detail_from_dataset(
        model=predictor.model,
        dataset=predictor.test_dataset,
        fight_ids=None,
        print_info=False,
        show_plot=True,
        ax=ax1,
    )

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

    df["betting_odds"] = np.where(
        df["Prediction"] > 0.5, df["opponent_odds"], df["fighter_odds"]
    )
    df = df.merge(
        predictor.data_processor.data[
            ["fight_id", "fighter_id", "event_date", "event_id", "weight_class"]
        ],
        on="fight_id",
    )
    df["confidence"] = abs((df["Prediction"] - 0.5) * 2)
    df = df.drop_duplicates(
        subset=["fight_id"], keep="last"
    )  # keep first or last should be the same
    # df = df[df["confidence"] > ]

    df = df.sort_values(by="event_date")

    cash = [
        initial_cash,
    ]
    invest = [
        initial_cash,
    ]
    dates = [
        None,
    ]

    print("Max confidence: ", df["confidence"].max())
    print("Max bet: ", df["bet"].max())

    for i, (date, group) in enumerate(df.groupby("event_date")):
        max_bet = (
            max(cash[-1] * max_bet_prop, min_max_bet) / df["confidence"].max()
        )  # In principle max(confidence) = 1

        date_bet = 0
        date_win = 0

        for combination_size in range(1, max_parlay_size + 1):
            for combo in list(combinations(group.index, combination_size)):
                parlay_df = group.loc[list(combo)]
                parlay_odds = parlay_df["betting_odds"].prod()
                parlay_confidence = parlay_df["confidence"].prod()

                parlay_correct = parlay_df["correct"].prod()

                # Here, the /10 corrects for the 10 used in
                # show_fight_prediction_detail_from_dataset,

                # parlay_bet = parlay_confidence * max_bet / 10 * parlay_df["bet"].sum()
                parlay_bet = bet_strategy(parlay_confidence, max_bet, parlay_odds)
                parlay_win = parlay_correct * parlay_odds * parlay_bet

                date_bet += parlay_bet
                date_win += parlay_win

        if date_bet > max_bet:
            date_win = date_win / date_bet * max_bet
            date_bet = max_bet

        extra_added = max(
            date_bet - cash[-1], 0
        )  # I might not have money to play, therefore we add
        cash_i = (
            cash[-1] + date_win - min(date_bet, cash[-1])
        )  # We have the previous money + the win - the bet

        invest.append(invest[-1] + extra_added)
        cash.append(cash_i)
        dates.append(date)

    cash = cash[1:]
    invest = invest[1:]
    dates = dates[1:]

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
    ax.set_title(predictor.config["general"]["model filename"])

    ax.legend()
    ax.grid()


# %%

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
fight_parameters_values = [[w, r] for w, r in zip(weight, rounds)]

# %%
predictor.load_forecast_dataset()
self = predictor.forecast_dataset
forecast_dataset = self

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
p1, p2 = forecast_dataset.get_forecast_prediction(
    fighter_names,
    opponent_names,
    event_dates,
    fighter_odds,
    opponent_odds,
    predictor.model,
    fight_parameters_values,
    parse_ids=False,
    device=predictor.device,
)

# %%
value = (p1 + p2) / 2
confidence = abs((value - 0.5) * 2)

# %%
max_bet = 40
confidence = abs(p1 + p2 - 1).numpy().reshape(-1)

bet = np.asarray([bet_strategy(c, max_bet) for c in confidence])
bet = bet.flatten().round(2)

# %%
bet.sum()

# %%
# Readjust to max_bet
bet = bet * max_bet / bet.sum()

# %%
bet.sum()

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

# %%
bet.sum()

# %%
bets = []
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

    if f in invalid_fighters or o in invalid_fighters:
        bets.append(-1)
    else:
        bets.append((p1h[0] + p2h[0]) / 2)

# %%
[print(f"{bet:.3f}") for bet in bets]

# %%

# %%
