from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ufcscraper.odds_scraper import BestFightOddsScraper
from ufcscraper.ufc_scraper import UFCScraper

from ufcpredictor.data_processor import DataProcessor
from ufcpredictor.models import SymmetricFightNet
from ufcpredictor.utils import convert_minutes_to_seconds, weight_dict

if TYPE_CHECKING:  # pragma: no cover
    import datetime
    from typing import Any, Callable, List, Optional, Set, Tuple

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        data_processor: DataProcessor,
        fight_ids: List[str],
        X_set: Optional[List[str]] = None,
    ) -> None:
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

    def load_data(self) -> None:
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

    def __len__(self) -> int:
        return len(self.data[0])

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # X, Y, winner, odds_1, odds_2 = self.data
        # X, Y, winner, odds_1, odds_2 = (
        #     X[idx],
        #     Y[idx],
        #     winner[idx],
        #     odds_1[idx],
        #     odds_2[idx],
        # )
        X, Y, winner, odds_1, odds_2 = [x[idx] for x in self.data]

        if np.random.random() >= 0.5:
            X, Y = Y, X
            winner = 1 - winner
            odds_1, odds_2 = odds_2, odds_1

        return X, Y, winner.reshape(-1), odds_1.reshape(-1), odds_2.reshape(-1)

    def get_fight_data_from_ids(self, fight_ids: Optional[List[str]] = None) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        NDArray[np.str_],
        NDArray[np.str_],
    ]:
        if fight_ids is not None:
            fight_data = self.fight_data[self.fight_data["fight_id"].isin(fight_ids)]
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

        fighter_names=  np.array(fight_data["fighter_name_x"].values)
        opponent_names= np.array(fight_data["fighter_name_y"].values)

        X1, X2, Y, odds1, odds2 = data

        return X1, X2, Y, odds1, odds2, fighter_names, opponent_names


class ForecastDataset(Dataset):
    X_set = BasicDataset.X_set

    def __init__(
        self,
        data_processor: DataProcessor,
        X_set: Optional[List[str]] = None,
    ) -> None:
        self.data_processor = data_processor

        if X_set is not None:
            self.X_set = X_set

        not_found = []
        for column in self.X_set:
            if column not in self.data_processor.data_normalized.columns:
                not_found.append(column)

        if len(not_found) > 0:
            raise ValueError(f"Columns not found in normalized data: {not_found}")

    def get_forecast_prediction(
        self,
        fighter_ids: List[str],
        opponent_ids: List[str],
        event_dates: List[str | datetime.date],
        fighter_odds: List[float],
        opponent_odds: List[float],
        model: nn.Module,
    ) -> Tuple[NDArray, NDArray]:
        match_data = pd.DataFrame(
            {
                "fighter_id": fighter_ids + opponent_ids,
                "event_date_forecast": event_dates * 2,
                "opening": np.concatenate((fighter_odds, opponent_odds)),
            }
        )

        match_data = match_data.merge(
            self.data_processor.data_normalized,
            left_on="fighter_id",
            right_on="fighter_id",
        )

        match_data = match_data[
            match_data["event_date"] < match_data["event_date_forecast"]
        ]
        match_data = match_data.sort_values(
            by=["fighter_id", "event_date"],
            ascending=[True, False],
        )
        match_data = match_data.drop_duplicates(
            subset=["fighter_id", "event_date_forecast"],
            keep="first",
        )
        match_data["id_"] = (
            match_data["fighter_id"].astype(str)
            + "_"
            + match_data["event_date_forecast"].astype(str)
        )

        # This data dict is used to facilitate the construction of the tensors
        data_dict = {
            id_: data
            for id_, data in zip(
                match_data["id_"].values,
                np.asarray([match_data[x] for x in self.X_set]).T,
            )
        }

        data = [
            torch.FloatTensor(
                np.asarray(
                    [
                        data_dict[fighter_id + "_" + str(event_date)]
                        for fighter_id, event_date in zip(fighter_ids, event_dates)
                    ]
                )
            ),  # X1
            torch.FloatTensor(
                np.asarray(
                    [
                        data_dict[fighter_id + "_" + str(event_date)]
                        for fighter_id, event_date in zip(opponent_ids, event_dates)
                    ]
                )
            ),  # X2
            torch.FloatTensor(np.asarray(fighter_odds)).reshape(-1, 1),  # Odds1,
            torch.FloatTensor(np.asarray(opponent_odds)).reshape(-1, 1),  # Odds2
        ]

        X1, X2, odds1, odds2 = data
        model.eval()
        with torch.no_grad():
            predictions_1 = model(X1, X2, odds1, odds2).detach().numpy()
            predictions_2 = 1 - model(X2, X1, odds2, odds1).detach().numpy()

        return predictions_1, predictions_2
