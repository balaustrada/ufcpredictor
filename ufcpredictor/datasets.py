from __future__ import annotations

import logging

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ufcscraper.ufc_scraper import UFCScraper
from ufcscraper.odds_scraper import BestFightOddsScraper
from ufcpredictor.utils import convert_minutes_to_seconds, weight_dict
from ufcpredictor.data_processor import DataProcessor
from ufcpredictor.models import SymmetricFightNet
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import f1_score


if TYPE_CHECKING:  # pragma: no cover
    import datetime
    from typing import Any, Callable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class Dataset:
    pass

class BasicDataset(Dataset):
    X_set = [
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
        "KO_per_fight",
        "Sub_per_fight",
        "time_since_last_fight",
        "win_per_fight",
        "age",
        "fighter_height_cm",
        "num_fight",
        "knockdowns_opponent_per_minute",
        "strikes_att_opponent_per_minute",
        "strikes_succ_opponent_per_minute",
        "head_strikes_att_opponent_per_minute",
        "head_strikes_succ_opponent_per_minute",
        "body_strikes_att_opponent_per_minute",
        "body_strikes_succ_opponent_per_minute",
        "leg_strikes_att_opponent_per_minute",
        "leg_strikes_succ_opponent_per_minute",
        "distance_strikes_att_opponent_per_minute",
        "distance_strikes_succ_opponent_per_minute",
        "ground_strikes_att_opponent_per_minute",
        "ground_strikes_succ_opponent_per_minute",
        "clinch_strikes_att_opponent_per_minute",
        "clinch_strikes_succ_opponent_per_minute",
        "total_strikes_att_opponent_per_minute",
        "total_strikes_succ_opponent_per_minute",
        "takedown_att_opponent_per_minute",
        "takedown_succ_opponent_per_minute",
        "submission_att_opponent_per_minute",
        "reversals_opponent_per_minute",
        "ctrl_time_opponent_per_minute",
        "KO_opponent_per_minute",
        "Sub_opponent_per_minute",
        "KO_opponent_per_fight",
        "Sub_opponent_per_fight",
        "win_opponent_per_fight",
    ]

    def __init__(self, data_processor: DataProcessor, fight_ids: List[str], X_set: List[str]=None) -> None:
        """
        fight_ids: Fight ids to load (usually train fights or test fights)
        """
        self.data_processor = data_processor
        self.fight_ids = fight_ids

        if X_set is not None:
            self.X_set = X_set

        not_found = []
        for column in self.X_set:
            if column not in self.data_processor.data_normalized.columns:
                not_found.append(column)

        if len(not_found) > 0:
            raise ValueError(f"Columns not found in normalized data: {not_found}")

        self.load_data()

    def load_data(self):
        reduced_data = self.data_processor.data_normalized.copy()

        # We shift stats because the input for the model should be the
        # stats prior to the fight
        for x in self.X_set:
            reduced_data[x] = reduced_data.groupby("fighter_id")[x].shift(1)

        # We remove invalid fights
        reduced_data = reduced_data[reduced_data["fight_id"].isin(self.fight_ids)]

        # We now merge stats with itself to get one row per match with the data
        # from the two fighters
        fight_data = reduced_data.merge(
            reduced_data,
            left_on="fight_id",
            right_on="fight_id",
            how="inner",
            suffixes=("_x", "_y"),
        )

        # Remove matchings of the fighter with itself and also only keep
        # one row per match (fighter1 vs fighter2 is the same as fighter 2 vs fighter 1)
        fight_data = fight_data[
            fight_data["fighter_id_x"] != fight_data["fighter_id_y"]
        ]
        fight_data = fight_data.drop_duplicates(subset=["fight_id"], keep="first")

        # Now we load the data into torch tensors
        # This is a list of FloatTensors each having a size equal to the number
        # of fights.
        self.data = [
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

        self.fight_data = fight_data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        X, Y, winner, odds_1, odds_2 = self.data
        X, Y, winner, odds_1, odds_2 = (
            X[idx],
            Y[idx],
            winner[idx],
            odds_1[idx],
            odds_2[idx],
        )

        if np.random.random() >= 0.5:
            X, Y = Y, X
            winner = 1 - winner
            odds_1, odds_2 = odds_2, odds_1

        return X, Y, winner.reshape(-1), odds_1.reshape(-1), odds_2.reshape(-1)

    def get_fight_data_from_ids(self, fight_ids: Optional[List[str]] = None) -> Tuple[torch.Tensor]:
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

        return X1, X2, Y, odds1, odds2, fighter_names, opponent_names