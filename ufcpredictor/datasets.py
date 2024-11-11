"""
This module contains dataset classes designed to handle UFC fight data for training 
and testing neural network models.

The dataset classes provide a structured way to store and retrieve data for fighter 
characteristics, fight outcomes, and odds. They are designed to work with the 
DataProcessor class to prepare and normalize the data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ufcpredictor.data_processor import DataProcessor

if TYPE_CHECKING:  # pragma: no cover
    import datetime
    from typing import List, Optional, Tuple

    from numpy.typing import NDArray
    from torch import nn

logger = logging.getLogger(__name__)


class BasicDataset(Dataset):
    """
    A basic dataset class designed to handle UFC fight data for training and testing
    neural network models.

    This class provides a simple way to store and retrieve data for fighter
    characteristics, fight outcomes, and odds. It is designed to be used with the
    SymmetricFightNet model and other UFC prediction models.
    """

    X_set = [
        "age",
        "body_strikes_att_opponent_per_minute",
        "body_strikes_att_per_minute",
        "body_strikes_succ_opponent_per_minute",
        "body_strikes_succ_per_minute",
        "clinch_strikes_att_opponent_per_minute",
        "clinch_strikes_att_per_minute",
        "clinch_strikes_succ_opponent_per_minute",
        "clinch_strikes_succ_per_minute",
        "ctrl_time_opponent_per_minute",
        "ctrl_time_per_minute",
        "distance_strikes_att_opponent_per_minute",
        "distance_strikes_att_per_minute",
        "distance_strikes_succ_opponent_per_minute",
        "distance_strikes_succ_per_minute",
        "fighter_height_cm",
        "ground_strikes_att_opponent_per_minute",
        "ground_strikes_att_per_minute",
        "ground_strikes_succ_opponent_per_minute",
        "ground_strikes_succ_per_minute",
        "head_strikes_att_opponent_per_minute",
        "head_strikes_att_per_minute",
        "head_strikes_succ_opponent_per_minute",
        "head_strikes_succ_per_minute",
        "knockdowns_opponent_per_minute",
        "knockdowns_per_minute",
        "KO_opponent_per_fight",
        "KO_opponent_per_minute",
        "KO_per_fight",
        "KO_per_minute",
        "leg_strikes_att_opponent_per_minute",
        "leg_strikes_att_per_minute",
        "leg_strikes_succ_opponent_per_minute",
        "leg_strikes_succ_per_minute",
        "num_fight",
        "reversals_opponent_per_minute",
        "reversals_per_minute",
        "strikes_att_opponent_per_minute",
        "strikes_att_per_minute",
        "strikes_succ_opponent_per_minute",
        "strikes_succ_per_minute",
        "Sub_opponent_per_fight",
        "Sub_opponent_per_minute",
        "Sub_per_fight",
        "Sub_per_minute",
        "submission_att_opponent_per_minute",
        "submission_att_per_minute",
        "takedown_att_opponent_per_minute",
        "takedown_att_per_minute",
        "takedown_succ_opponent_per_minute",
        "takedown_succ_per_minute",
        "time_since_last_fight",
        "total_strikes_att_opponent_per_minute",
        "total_strikes_att_per_minute",
        "total_strikes_succ_opponent_per_minute",
        "total_strikes_succ_per_minute",
        "win_opponent_per_fight",
        "win_per_fight",
    ]

    Xf_set = []

    def __init__(
        self,
        data_processor: DataProcessor,
        fight_ids: List[str],
        X_set: Optional[List[str]] = None,
        Xf_set: Optional[List[str]] = None,
    ) -> None:
        """
        Constructor for ForecastDataset.

        Args:
            data_processor: The DataProcessor instance that contains the data.
            fight_ids: The list of fight ids to include in the dataset.
            X_set: The list of columns to include in the dataset. If None, use all
            columns.

        Raises:
            ValueError: If some columns are not found in the normalized data.
        """
        self.data_processor = data_processor
        self.fight_ids = fight_ids

        if X_set is not None:
            self.X_set = X_set

        if Xf_set is not None:
            self.Xf_set = Xf_set

        not_found = []
        for column in self.X_set + self.Xf_set:
            if column not in self.data_processor.data_normalized.columns:
                not_found.append(column)

        if len(not_found) > 0:
            raise ValueError(f"Columns not found in normalized data: {not_found}")

        self.load_data()

    def load_data(self) -> None:
        """
        Loads the data into a format that can be used to train a model.

        The data is first reduced to only include the columns specified in X_set.
        Then, the stats are shifted to get the stats prior to each fight.
        The data is then merged with itself to get one row per match with the data
        from the two fighters.
        The matchings of the fighter with itself are removed and only one row per
        match is kept.
        Finally, the data is loaded into torch tensors.
        """
        reduced_data = self.data_processor.data_normalized.copy()

        # We shift stats because the input for the model should be the
        # stats prior to the fight
        for x in self.X_set:
            if x not in ["age", "num_fight", "time_since_last_fight"]:
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
                np.asarray([fight_data[xf + "_x"].values for xf in self.Xf_set]).T
            ),
            torch.FloatTensor(
                (fight_data["winner_x"] != fight_data["fighter_id_x"]).values
            ),
            torch.FloatTensor(fight_data["opening_x"].values),
            torch.FloatTensor(fight_data["opening_y"].values),
        ]

        if len(self.Xf_set) == 0:
            self.data[2] = torch.empty(len(fight_data["winner_x"]), 0)

        self.fight_data = fight_data

    def __len__(self) -> int:
        """Returns the size of the dataset.

        Returns:
            The size of the dataset.
        """
        return len(self.data[0])

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of (X, Y, winner, odds_1, odds_2) for the given index.

        The data is randomly flipped to simulate the possibility of a fight being
        between two fighters in either order.

        Args:
            idx: The index of the data to return.

        Returns:
            A tuple of (X, Y, winner, odds_1, odds_2) where X and Y are the
            input data for the two fighters, winner is a tensor of size 1
            indicating which fighter won, and odds_1 and odds_2 are the opening
            odds for the two fighters.
        """
        X1, X2, X3, winner, odds_1, odds_2 = [x[idx] for x in self.data]

        if np.random.random() >= 0.5:
            X1, X2 = X2, X1
            winner = 1 - winner
            odds_1, odds_2 = odds_2, odds_1

        return X1, X2, X3, winner.reshape(-1), odds_1.reshape(-1), odds_2.reshape(-1)

    def get_fight_data_from_ids(self, fight_ids: Optional[List[str]] = None) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        NDArray[np.str_],
        NDArray[np.str_],
    ]:
        """
        Returns a tuple of (X, Y, winner, odds_1, odds_2, fighter_names, opponent_names)
        for the given fight ids.

        If fight_ids is None, returns all the data in the dataset.

        Args:
            fight_ids: The list of fight ids to include in the dataset. If None,
                use all the data in the dataset.

        Returns:
            A tuple of (X, Y, winner, odds_1, odds_2, fighter_names, opponent_names)
            where X and Y are the input data for the two fighters, winner is a tensor
            of size 1 indicating which fighter won, and odds_1 and odds_2 are the
            opening odds for the two fighters. fighter_names and opponent_names are
            the names of the fighters and their opponents.
        """
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
                np.asarray([fight_data[x + "_x"].values for x in self.Xf_set]).T
            ),
            torch.FloatTensor(
                (fight_data["winner_x"] != fight_data["fighter_id_x"]).values
            ),
            torch.FloatTensor(fight_data["opening_x"].values),
            torch.FloatTensor(fight_data["opening_y"].values),
        ]

        fighter_names = np.array(fight_data["fighter_name_x"].values)
        opponent_names = np.array(fight_data["fighter_name_y"].values)

        X1, X2, X3, Y, odds1, odds2 = data

        return X1, X2, X3, Y, odds1, odds2, fighter_names, opponent_names


class ForecastDataset(Dataset):
    """
    A dataset class designed to handle forecasting data for UFC fight predictions.

    This class provides a structured way to store and retrieve data for training and
    testing neural network models. It is designed to work with the DataProcessor class
    to prepare and normalize the data.
    """

    X_set = BasicDataset.X_set
    Xf_set = BasicDataset.Xf_set

    def __init__(
        self,
        data_processor: DataProcessor,
        X_set: Optional[List[str]] = None,
        Xf_set: Optional[List[str]] = None,
    ) -> None:
        """
        Constructor for ForecastDataset.

        Args:
            data_processor: The DataProcessor instance that contains the data.
            X_set: The list of columns to include in the dataset. If None, use all
                columns.

        Raises:
            ValueError: If some columns are not found in the normalized data.
        """
        self.data_processor = data_processor

        if X_set is not None:
            self.X_set = X_set

        if Xf_set is not None:
            self.Xf_set = Xf_set

        not_found = []
        for column in self.X_set + self.Xf_set:
            if column not in self.data_processor.data_normalized.columns:
                not_found.append(column)

        if len(not_found) > 0:
            raise ValueError(f"Columns not found in normalized data: {not_found}")

    def get_single_forecast_prediction(
        self,
        fighter_name: str,
        opponent_name: str,
        event_date: str | datetime.date,
        odds1: int,
        odds2: int,
        model: nn.Module,
        fight_features: Optional[List[float]] = None,
        parse_ids: bool = False,
    ) -> Tuple[float, float]:
        """
        Make a prediction for a single match. Either providing the names of the
        fighters and their opponents, or providing the ids of the fighters and
        their opponents.

        Args:
            fighter_name: The name of the fighter.
            opponent_name: The name of the opponent.
            event_date: The date of the fight.
            odds1: The odds of the first fighter.
            odds2: The odds of the second fighter.
            model: The model to make the prediction with.
            parse_ids: Whether to parse the ids of the fighters and opponents. Ids
                are parsed in fields "fighter_name" and "opponent_name"if True,
                and names are parsed if False.

        Returns: The predicted odds for the first and second fighters.
        """
        p1, p2 = self.get_forecast_prediction(
            [
                fighter_name,
            ],
            [
                opponent_name,
            ],
            [
                event_date,
            ],
            [
                odds1,
            ],
            [
                odds2,
            ],
            model=model,
            fight_features=[
                fight_features,
            ],
            parse_ids=parse_ids,
        )

        return p1[0][0], p2[0][0]

    def get_forecast_prediction(
        self,
        fighter_names: List[str],
        opponent_names: List[str],
        event_dates: List[str | datetime.date],
        fighter_odds: List[float],
        opponent_odds: List[float],
        model: nn.Module,
        fight_features: List[List[float]] = [],
        parse_ids: bool = False,
        device: str = "cpu",
    ) -> Tuple[NDArray, NDArray]:
        """
        Make a prediction for a given list of matches. Either providing the names of
        the fighters and their opponents, or providing the ids of the fighters and
        their opponents.

        Args:
            fighters_names: The list of fighters names.
            opponent_names: The list of opponent names.
            event_dates: The list of event dates.
            fighter_odds: The list of fighter odds.
            opponent_odds: The list of opponent odds.
            model: The model to make the prediction with.
            parse_ids: Whether to parse the ids of the fighters and opponents. Ids
                are parsed in fields "fighter_names" and "opponent_names"if True,
                and names are parsed if False.
            device: The device to use for the prediction.

        Returns:
            A tuple of two numpy arrays, each containing the predictions for one of the
            fighters.
        """
        if not parse_ids:
            fighter_ids = [self.data_processor.get_fighter_id(x) for x in fighter_names]
            opponent_ids = [
                self.data_processor.get_fighter_id(x) for x in opponent_names
            ]
        else:
            fighter_ids = fighter_names
            opponent_ids = opponent_names

        match_data = pd.DataFrame(
            {
                "fighter_id": fighter_ids + opponent_ids,
                "event_date_forecast": event_dates * 2,
                "opening": np.concatenate((fighter_odds, opponent_odds)),
            }
        )

        for feature_name, stats in zip(self.Xf_set, np.asarray(fight_features).T):
            match_data[feature_name] = np.concatenate((stats, stats))

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

        for feature_name, stats in zip(self.Xf_set, np.asarray(fight_features).T):
            match_data[feature_name] = np.concatenate((stats, stats))

        if len(self.Xf_set) > 0:
            fight_data_dict = {
                id_: data
                for id_, data in zip(
                    match_data["id_"].values,
                    np.asarray([match_data[x] for x in self.Xf_set]).T,
                )
            }
        else:
            fight_data_dict = {id_: [] for id_ in match_data["id_"].values}

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
            torch.FloatTensor(
                np.asarray(
                    [
                        fight_data_dict[fighter_id + "_" + str(event_date)]
                        for fighter_id, event_date in zip(fighter_ids, event_dates)
                    ]
                )
            ),  # X3
            torch.FloatTensor(np.asarray(fighter_odds)).reshape(-1, 1),  # Odds1,
            torch.FloatTensor(np.asarray(opponent_odds)).reshape(-1, 1),  # Odds2
        ]

        X1, X2, X3, odds1, odds2 = data
        X1, X2, X3, odds1, odds2, model = (
            X1.to(device),
            X2.to(device),
            X3.to(device),
            odds1.to(device),
            odds2.to(device),
            model.to(device),
        )
        model.eval()
        with torch.no_grad():
            predictions_1 = model(X1, X2, X3, odds1, odds2).detach().cpu().numpy()
            predictions_2 = 1 - model(X2, X1, X3, odds2, odds1).detach().cpu().numpy()

        return predictions_1, predictions_2
