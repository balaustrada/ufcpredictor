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
  seed: 3
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
self.trainer.train(
    epochs=20,
    train_dataloader=self.early_train_dataloader,
    test_dataloader=self.test_dataloader,
)

# %%
self.trainer.train(
    epochs=5,
    test_dataloader=self.test_dataloader,
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

                parlay_bet = parlay_confidence * max_bet / 10 * parlay_df["bet"].sum()
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
