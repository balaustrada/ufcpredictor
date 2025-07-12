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
from ufcpredictor.data_enhancers import RankedFields
from ufcpredictor.utils import pad_or_truncate

if TYPE_CHECKING:  # pragma: no cover
    import datetime
    from typing import List, Optional, Tuple

    from numpy.typing import NDArray
    from torch import nn

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    A base dataset class designed to handle UFC fight data for training and testing
    neural network models.

    This class is supposed to not be used directly, but rather to provide a
    common interface for datasets that handle UFC fight data. It provides the
    basic functionality to load and process the data, including fighter statistics,
    fight parameters, and fight ids. It also provides a method to load the data
    into torch tensors for training and testing.

    Attributes
        fighter_fight_statistics: A list of columns that contain fighter
            statistics for each fight. These statistics are used to train the model.
        fight_parameters: A list of columns that contain fight parameters, such as
            fight date, weight class, and location. These parameters are used to
            train the model.
        data_processor: An instance of DataProcessor that contains the data to be
            used.
        fight_ids: A list of fight ids to include in the dataset. If None, all
            fights are included.
        data: A list of torch tensors containing the data for each fight. Each
            tensor contains the fighter statistics, fight parameters, and opening
            odds for each fighter in the fight.
        fight_data: A pandas DataFrame containing the data for each fight, including
            fighter statistics, fight parameters, and opening odds. This DataFrame
            is used to retrieve the data for each fight when needed.
        data: A list of torch tensors containing the data for each fight. Each tensor
            contains the information needed to train or test the model.
    """

    fighter_fight_statistics: List[str] = [
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

    fight_parameters: List[str] = []

    def __init__(
        self,
        data_processor: DataProcessor,
        fight_ids: Optional[List[str]] = None,
        fighter_fight_statistics: Optional[List[str]] = None,
        fight_parameters: Optional[List[str]] = None,
    ) -> None:
        """
        Constructor for BaseDataset.

        Args:
            data_processor: The DataProcessor instance that contains the data.
            fight_ids: The list of fight ids to include in the dataset. If None,
                all fights are included.
            fighter_fight_statistics: The list of columns to include in the dataset.
                If None, use default columns defined in cls.fighter_fight_statistics.
            fight_parameters: The list of fight parameters to include in the model.
                If None, use an empty list.

        Raises:
            ValueError: If any column is not found in the normalized data.
        """
        self.data_processor = data_processor
        self.fight_ids = fight_ids

        if fighter_fight_statistics is not None:
            self.fighter_fight_statistics = fighter_fight_statistics

        if fight_parameters is not None:
            self.fight_parameters = fight_parameters

        not_found = []
        for column in (
            self.fighter_fight_statistics + self.fight_parameters
        ):  # pragma: no cover
            if column not in self.data_processor.data_normalized.columns:
                not_found.append(column)

        if len(not_found) > 0:  # pragma: no cover
            raise ValueError(f"Columns not found in normalized data: {not_found}")

        self.load_data()

    def load_data(self) -> None:
        """
        Loads the data into a format that can be used to train a model.

        The data is first reduced to only include the columns specified in
        fighter_fight_statistics.

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
        for x in self.fighter_fight_statistics:
            if x not in ["age", "num_fight", "time_since_last_fight"]:
                reduced_data[x] = reduced_data.groupby("fighter_id")[x].shift(1)

        # We remove invalid fights

        if self.fight_ids is not None:
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
        self.data: List[torch.Tensor] = [
            torch.FloatTensor(
                np.asarray(
                    [fight_data[x + "_x"].values for x in self.fighter_fight_statistics]
                ).T
            ),
            torch.FloatTensor(
                np.asarray(
                    [fight_data[x + "_y"].values for x in self.fighter_fight_statistics]
                ).T
            ),
            torch.FloatTensor(
                np.asarray(
                    [fight_data[xf + "_x"].values for xf in self.fight_parameters]
                ).T
            ),
            torch.FloatTensor(
                (fight_data["winner_x"] != fight_data["fighter_id_x"]).values
            ),
            torch.FloatTensor(fight_data["opening_x"].values),
            torch.FloatTensor(fight_data["opening_y"].values),
        ]

        if len(self.fight_parameters) == 0:
            self.data[2] = torch.empty(len(fight_data["winner_x"]), 0)

        self.fight_data = fight_data

    def __len__(self) -> int:
        """Returns the size of the dataset.

        Returns:
            The size of the dataset.
        """
        return len(self.data[0])


class ForecastDataset(BaseDataset):
    """
    A dataset class designed to handle forecasting data for UFC fight predictions.

    This class provides functionality to retrieve the inputs for forecasting
    predictions.
    """

    fighter_fight_statistics = BaseDataset.fighter_fight_statistics
    fight_parameters = BaseDataset.fight_parameters

    def __init__(
        self,
        data_processor: DataProcessor,
        fighter_fight_statistics: Optional[List[str]] = None,
        fight_parameters: Optional[List[str]] = None,
    ) -> None:
        """
        Constructor for ForecastDataset.

        Args:
            data_processor: The DataProcessor instance that contains the data.
            fighter_fight_statistics: The list of columns to include in the dataset. If None, use all
                columns.
            fight_parameters: The list of fight parameters to include in the
                model. If None, use an empty list.

        Raises:
            ValueError: If some columns are not found in the normalized data.
        """
        self.data_processor = data_processor

        if fighter_fight_statistics is not None:
            self.fighter_fight_statistics = fighter_fight_statistics

        if fight_parameters is not None:
            self.fight_parameters = fight_parameters

        not_found = []
        for column in self.fighter_fight_statistics + self.fight_parameters:
            if column not in self.data_processor.data_normalized.columns:
                not_found.append(column)

        if len(not_found) > 0:
            raise ValueError(f"Columns not found in normalized data: {not_found}")

        self.fight_ids = None
        self.load_data()

    def get_single_forecast_prediction(
        self,
        fighter_name: str,
        opponent_name: str,
        event_date: str | datetime.date,
        odds1: int,
        odds2: int,
        model: nn.Module,
        fight_parameters_values: List[float] = [],
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

        Returns:
            A tuple of two numpy arrays, each one evaluating the model switching
            between the two fighters. For symmetric models, they should be the same.
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
            fight_parameters_values=[
                fight_parameters_values,
            ],
            parse_ids=parse_ids,
        )

        return p1[0][0], p2[0][0]

    def get_match_data_for_predictions(
        self,
        fighter_ids: List[str],
        opponent_ids: List[str],
        event_dates: List[str | datetime.date],
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
        for feature_name, stats in zip(
            self.fight_parameters, np.asarray(fight_parameters_values).T
        ):
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
        for feature_name in self.fight_parameters:
            if feature_name + "_x" in match_data.columns:
                match_data = match_data.rename(
                    columns={
                        feature_name + "_x": feature_name,
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

        return match_data

    def get_forecast_prediction(
        self,
        fighter_names: List[str],
        opponent_names: List[str],
        event_dates: List[str | datetime.date],
        fighter_odds: List[float],
        opponent_odds: List[float],
        model: nn.Module,
        fight_parameters_values: List[List[float]] = [],
        parse_ids: bool = False,
        device: str | torch.device = "cpu",
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
            A tuple of two numpy arrays, each one evaluating the model switching
            between the two fighters. For symmetric models, they should be the same.
        """
        if not parse_ids:
            fighter_ids = [self.data_processor.get_fighter_id(x) for x in fighter_names]
            opponent_ids = [
                self.data_processor.get_fighter_id(x) for x in opponent_names
            ]
        else:
            fighter_ids = fighter_names
            opponent_ids = opponent_names

        match_data = self.get_match_data_for_predictions(
            fighter_ids=fighter_ids,
            opponent_ids=opponent_ids,
            event_dates=event_dates,
            fighter_odds=fighter_odds,
            opponent_odds=opponent_odds,
            fight_parameters_values=fight_parameters_values,
        )

        ###############################################################
        # Now we start building the tensor to input to the model
        ###############################################################
        # This data dict is used to facilitate the construction of the tensors
        data_dict = {
            id_: data
            for id_, data in zip(
                match_data["id_"].values,
                np.asarray([match_data[x] for x in self.fighter_fight_statistics]).T,
            )
        }

        # We add fight parameters to the arrays.
        if len(self.fight_parameters) > 0:
            fight_data_dict = {
                id_: data
                for id_, data in zip(
                    match_data["id_"].values,
                    np.asarray([match_data[x] for x in self.fight_parameters]).T,
                )
            }
        else:
            fight_data_dict = {id_: [] for id_ in match_data["id_"].values}

        # We convert the arrays into torch tensors.
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


class BasicDataset(BaseDataset):
    """
    A basic dataset class designed to that implements the basic functionality in
    BaseDataset, but does not include any time evolution or forecasting functionality.
    """

    def __getitem__(self, idx: int) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Getter for the dataset.

        Data is augmented by randomly switching the fighters position in the network.

        Args:
            idx: The index of the data to return.

        Returns:
            A tuple of ((X1, X2, X3), winner, (odds_1, odds_2)) where X1,X2 are the
            input data for the two fighters, X3 are the fight parameters, winner is
            the winner of the fight and (odds1, odds2) the odds for each of the
            fighters.
        """
        X1, X2, X3, winner, odds_1, odds_2 = [x[idx] for x in self.data]

        if np.random.random() >= 0.5:
            X1, X2 = X2, X1
            winner = 1 - winner
            odds_1, odds_2 = odds_2, odds_1

        return (
            (
                X1,
                X2,
                X3,
            ),
            winner.reshape(-1),
            (
                odds_1.reshape(-1),
                odds_2.reshape(-1),
            ),
        )

    def get_fight_data_from_ids(self, fight_ids: Optional[List[str]] = None) -> Tuple[
        Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor],
        torch.FloatTensor,
        Tuple[torch.FloatTensor, torch.FloatTensor],
        NDArray[np.str_],
        NDArray[np.str_],
    ]:
        """
        Get the fight information for the given fight ids.


        Args:
            fight_ids: The list of fight ids to include from the dataset. If None,
                use all the data in the dataset.

        Returns:
            Returns a tuple of ((X1, X2, X3), Y,  (odds_1, odds_2), fighter_names,
            opponent_names) where X1 and X2 are the input data for the two fighters,
            X3 are the fight parameters, Y is the winner of the fight, odds_1 and
            odds_2 are the opening odds for each fighter, and fighter_names and
            opponent_names are the names of the fighters and their opponents.
        """
        if fight_ids is not None:
            fight_data = self.fight_data[self.fight_data["fight_id"].isin(fight_ids)]
        else:
            fight_data = self.fight_data.copy()

        data = [
            torch.FloatTensor(
                np.asarray(
                    [fight_data[x + "_x"].values for x in self.fighter_fight_statistics]
                ).T
            ),
            torch.FloatTensor(
                np.asarray(
                    [fight_data[x + "_y"].values for x in self.fighter_fight_statistics]
                ).T
            ),
            torch.FloatTensor(
                np.asarray(
                    [fight_data[x + "_x"].values for x in self.fight_parameters]
                ).T
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

        return (X1, X2, X3), Y, (odds1, odds2), fighter_names, opponent_names


class DatasetWithTimeEvolution(BaseDataset):
    """
    A dataset class designed to handle UFC fight data with time evolution for
    predictions.

    This class extends the BaseDataset class to include functionality for
    handling time evolution of fight state. This takes into account the statistics
    of previous UFC fights to define the state of the fighters in the current fight.

    Attributes:
        data: A list of torch tensors containing the data for each fight and the
            position of next and previous fights. The first data is prepared to
            be sent to the model, while the positional data is used to locate
            previous and next fights in the first data, so it can also be sent to
            the model.
        fighter_history_tensor: A tensor containing the history of the fighters
            in the dataset. This is used to define the state of the fighters in the
            current fight. It contains the statistics of the fighters and the
            state vector.
        previous_fights_statistics: A list of columns that contain statistics from
            previous fights. These statistics are used to define the state of the
            fighters in the current fight.
        previous_fights_parameters: A list of columns that contain parameters from
            previous fights. These parameters are used to define the state of the
            fighters in the current fight.
        state_size: The size of the state vector for each fighter.
        num_past_fights: The number of past fights to consider for each fighter when
            defining the state of the fighters in the current fight.
        f1_positions: For each fighter/fight pair, it defines the location in the
            fighter_history_tensor of the previous fights.
        f2_positions: For each fighter/fight pair, it defines the location in the
            fighter_history_tensor of the opponent's previous fights.
        next_f1_positions: For each fighter/fight pair, it defines the location in
            the fighter_history_tensor of the next fight of the fighter.
        next_f2_positions: For each fighter/fight pair, it defines the location in
            the fighter_history_tensor of the next fight of the opponent.
    """

    fighter_history_tensor: torch.Tensor

    previous_fights_statistics: List[str] = [
        "body_strikes_att_per_minute",
        "clinch_strikes_att_per_minute",
        "knockdowns_per_minute",
        "ELO",
        "knockdowns_per_minute",
    ]

    previous_fights_parameters: List[str] = [
        "winner",
    ]

    state_size: int = 5
    num_past_fights: int = 20
    f1_positions: List[List[int]]
    f2_positions: List[List[int]]
    next_f1_positions: List[List[int]]
    next_f2_positions: List[List[int]]

    def __init__(
        self,
        data_processor: DataProcessor,
        fight_ids: Optional[List[str]] = None,
        fighter_fight_statistics: Optional[List[str]] = None,
        fight_parameters: Optional[List[str]] = None,
        previous_fights_statistics: Optional[List[str]] = None,
        previous_fights_parameters: Optional[List[str]] = None,
        state_size: Optional[int] = None,
        num_past_fights: Optional[int] = None,
    ) -> None:
        """
        Constructor for ForecastDataset.

        Args:
            data_processor: The DataProcessor instance that contains the data.
            fight_ids: The list of fight ids to include in the dataset. If None,
                all fights are included.
            fighter_fight_statistics: The list of columns to include in the dataset.
                If None, use default columns defined in cls.fighter_fight_statistics.
            fight_parameters: The list of fight parameters to include in the model.
                If None, use an empty list. If None, use default columns defined in
                cls.fight_parameters.
            previous_fights_statistics: The list of columns to use for previous
                fights statistics. If None, use default columns defined in
                cls.previous_fights_statistics.
            previous_fights_parameters: The list of columns to use for previous
                fights parameters. If None, use default columns defined in
                cls.previous_fights_parameters.
            state_size: The size of the state vector for each fighter. If None,
                use the default value defined in cls.state_size.
            num_past_fights: The number of past fights to consider for each fighter
                when defining the state of the fighters in the current fight. If
                None, use the default value defined in cls.num_past_fights.

        Raises:
            ValueError: If some columns are not found in the normalized data.
        """
        self.data_processor = data_processor
        self.fight_ids = fight_ids

        self.fighter_fight_statistics = (
            fighter_fight_statistics or self.fighter_fight_statistics
        )

        self.fight_parameters = fight_parameters or self.fight_parameters

        self.previous_fights_parameters = (
            previous_fights_parameters or self.previous_fights_parameters
        )

        self.previous_fights_statistics = (
            previous_fights_statistics or self.previous_fights_statistics
        )

        self.state_size = state_size or self.state_size

        self.num_past_fights = num_past_fights or self.num_past_fights

        not_found = []
        for column in self.fighter_fight_statistics + self.fight_parameters:
            if (
                column not in self.data_processor.data_normalized.columns
            ):  # pragma: no cover
                not_found.append(column)

        if len(not_found) > 0:  # pragma: no cover
            raise ValueError(f"Columns not found in normalized data: {not_found}")

        self.load_data()

    def get_indices_previous_and_next(
        self, include_current: bool = False
    ) -> pd.DataFrame:
        """
        Get the indices of the previous and next fights for each fighter in the dataset.

        Args:
            include_current: Whether to include the current fight in the previous
            fights. This is only used in forecasting.

        Returns:
            A pandas DataFrame containing the indices of the previous and next fights
        """
        # We first retrieve the non aggregated data,
        # only keeping the relevant fields
        previous_and_next_indices = (
            self.data_processor.data_normalized_nonagg.copy()
            .sort_values(by=["event_date", "fight_id"])
            .reset_index(drop=True)[
                ["fight_id", "fighter_id", "event_date", "num_fight", "opponent_id"]
                + self.previous_fights_statistics
                + self.previous_fights_parameters
            ]
        )

        # We now add the index of the fighter/fight pair
        previous_and_next_indices["index"] = previous_and_next_indices.index

        previous_and_next_indices["winner"] = (
            previous_and_next_indices["winner"]
            == previous_and_next_indices["fighter_id"]
        ).astype(int)

        # Fill missing values in time_since_last_fight, with the mean value
        # this is for first fights where there is no previous fight.
        # It is not needed in BasicDataset since the first three fights of
        # a fighter are not usually included.
        if "time_since_last_fight" in previous_and_next_indices.columns:
            previous_and_next_indices["time_since_last_fight"] = (
                previous_and_next_indices["time_since_last_fight"].fillna(
                    previous_and_next_indices["time_since_last_fight"].mean()
                )
            )

        # To find the index of the opponent, we merge the data with itself
        # but matching fighter_id with opponent_id.
        # Will link each fight row to the opponent row.
        previous_and_next_indices = (
            previous_and_next_indices.merge(
                previous_and_next_indices[["fight_id", "opponent_id", "index"]],
                left_on=["fight_id", "fighter_id"],
                right_on=["fight_id", "opponent_id"],
            )
            .rename(
                columns={
                    "opponent_id_x": "opponent_id",
                    "index_x": "index",
                    "index_y": "opponent_row",
                }
            )
            .drop(columns=["opponent_id_y", "opponent_id"])
        )

        # Now we need to see which are the previous fights of each row
        # And also the next fight of each row, we start by defining a
        # simplified dataframe
        indices_df = previous_and_next_indices[["fighter_id", "index", "opponent_row"]]

        # Then we merge with the original dataframe to match
        # each fighter's fight to all past and future fights
        indices_df = indices_df.merge(
            indices_df,
            on="fighter_id",
        )

        # We preselect the previous fights by looking at index_y < index_x
        # After that, we need to aggregate all of them in a list, this will
        # define all previous fights of a given fight.
        # If there are no previous fights, we just insert an empty list
        if include_current:
            previous_indices_df = indices_df[
                indices_df["index_x"] >= indices_df["index_y"]
            ]
        else:
            previous_indices_df = indices_df[
                indices_df["index_x"] > indices_df["index_y"]
            ]

        previous_indices_df = (
            previous_indices_df.groupby("index_x")
            .agg(
                previous_fights=("index_y", list),
                previous_opponents=("opponent_row_y", list),
            )
            .reset_index()
            .set_index("index_x")
            .reindex(range(0, len(previous_and_next_indices)), fill_value=pd.NA)
            .reset_index()
        )
        # Fill missing lists with empty lists
        previous_indices_df["previous_fights"] = previous_indices_df[
            "previous_fights"
        ].apply(lambda x: x if isinstance(x, list) else [])
        previous_indices_df["previous_opponents"] = previous_indices_df[
            "previous_opponents"
        ].apply(lambda x: x if isinstance(x, list) else [])

        # We now add this previous fights information to previous_and_next_indices
        previous_and_next_indices = previous_and_next_indices.merge(
            previous_indices_df,
            left_on="index",
            right_on="index_x",
        )

        # We similarly preselect the future fights by looking at index_y > index_x
        # After that, we group by fight/fighter (index_x) and only will keep
        # the immediately next fight.
        # If there are no future fights, we just set this value to -1.
        next_indices_df = indices_df[indices_df["index_x"] < indices_df["index_y"]]
        next_indices_df = (
            next_indices_df.sort_values(by=["index_x", "index_y"])
            .drop_duplicates(
                subset="index_x",
                keep="first",
            )[["index_x", "index_y"]]
            .set_index("index_x")
            .reindex(range(0, len(previous_and_next_indices)), fill_value=-1)
            .rename(columns={"index_y": "next_fight"})
        )
        previous_and_next_indices = previous_and_next_indices.merge(
            next_indices_df,
            left_on="index",
            right_on="index_x",
        )

        return previous_and_next_indices

    def compute_position_data(
        self, previous_and_next_indices_trans: pd.DataFrame
    ) -> None:
        """
        Compute the position data for each fighter based on their fight history.

        This method generates two helper arrays that will help get the position
        of the previous and the next fights for each fighter/fight pair in the
        tensors. Ensuring that the state vector can be efficiently updated at each
        iteration of the model training.
        """
        previous_and_next_indices_trans = previous_and_next_indices_trans.copy()

        preserved_fields = ["fight_id", "fighter_id", "num_fight", "next_fight"]
        fight_data_nonag = previous_and_next_indices_trans[preserved_fields].merge(
            previous_and_next_indices_trans[preserved_fields],
            left_on="fight_id",
            right_on="fight_id",
            how="inner",
            suffixes=("_x", "_y"),
        )

        fight_data_nonag = fight_data_nonag[
            fight_data_nonag["fighter_id_x"] != fight_data_nonag["fighter_id_y"]
        ]
        fight_data_nonag = fight_data_nonag.drop_duplicates(
            subset=["fight_id"], keep="first"
        )
        fight_data_nonag["max_num_fight"] = fight_data_nonag[
            ["num_fight_x", "num_fight_y"]
        ].max(axis=1)

        previous_and_next_indices_trans = previous_and_next_indices_trans.reset_index(
            drop=True
        )
        previous_and_next_indices_trans["Index"] = previous_and_next_indices_trans.index

        X = (
            fight_data_nonag.merge(
                previous_and_next_indices_trans[["fight_id", "fighter_id", "Index"]],
                left_on=["fight_id", "fighter_id_x"],
                right_on=["fight_id", "fighter_id"],
            )
            .rename(columns={"Index": "Index_x"})
            .drop(columns="fighter_id")
            .merge(
                previous_and_next_indices_trans[["fight_id", "fighter_id", "Index"]],
                left_on=["fight_id", "fighter_id_y"],
                right_on=["fight_id", "fighter_id"],
            )
            .rename(columns={"Index": "Index_y"})
            .drop(columns="fighter_id")
        )

        f1_positions = []
        f2_positions = []
        next_f1_positions = []
        next_f2_positions = []

        for max_fight in sorted(X["max_num_fight"].unique()):
            # Filter rows for the current max_fight value
            filtered_rows = X[X["max_num_fight"] == max_fight]

            f1_positions.append(filtered_rows["Index_x"].values)
            f2_positions.append(filtered_rows["Index_y"].values)
            next_f1_positions.append(filtered_rows["next_fight_x"].values)
            next_f2_positions.append(filtered_rows["next_fight_y"].values)

        self.f1_positions = f1_positions
        self.f2_positions = f2_positions
        self.next_f1_positions = next_f1_positions
        self.next_f2_positions = next_f2_positions

    def update_data_trans(
        self, transformer: nn.Module, device: str | torch.device = "cpu"
    ) -> None:
        for i, (
            f1_position,
            f2_position,
            next_f1_position,
            next_f2_position,
        ) in enumerate(
            zip(
                self.f1_positions,
                self.f2_positions,
                self.next_f1_positions,
                self.next_f2_positions,
            )
        ):
            self.fighter_history_tensor = self.fighter_history_tensor.to(device)
            X1 = self.fighter_history_tensor[f1_position][:, : self.state_size]
            X2 = self.fighter_history_tensor[f2_position][:, : self.state_size]
            s1 = self.fighter_history_tensor[f1_position][
                :, self.state_size : -len(self.previous_fights_parameters)
            ]
            s2 = self.fighter_history_tensor[f2_position][
                :, self.state_size : -len(self.previous_fights_parameters)
            ]

            m = self.fighter_history_tensor[f1_position][
                :, -len(self.previous_fights_parameters) :
            ]

            X1, X2 = transformer(X1, X2, s1, s2, m)

            msk = np.asarray(next_f1_position) > 0
            self.fighter_history_tensor[next_f1_position[msk], : self.state_size] = X1[
                msk
            ]

            msk = np.asarray(next_f2_position) > 0
            self.fighter_history_tensor[next_f2_position[msk], : self.state_size] = X2[
                msk
            ]

    def load_data(self) -> None:
        """
        This method applies the same data processing as the BasicDataset, but
        it also incorporates the previous and next fights indices to the data.
        """
        reduced_data = self.data_processor.data_normalized.copy()

        # We shift stats because the input for the model should be the
        # stats prior to the fight
        for x in self.fighter_fight_statistics:
            if x not in ["age", "num_fight", "time_since_last_fight"]:
                reduced_data[x] = reduced_data.groupby("fighter_id")[x].shift(1)

        # We remove invalid fights
        if self.fight_ids is not None:
            reduced_data = reduced_data[reduced_data["fight_id"].isin(self.fight_ids)]

        # We add the last fights and next fight indices to the data.
        previous_and_next_indices = self.get_indices_previous_and_next()
        previous_and_next_indices = previous_and_next_indices.loc[
            :, ~previous_and_next_indices.columns.duplicated()
        ].copy()

        # With this same indices we generate the position data. This will
        # be used later to update the fighter state tensor.
        self.compute_position_data(previous_and_next_indices)

        # Incorporate previous fights to the reduced data.
        reduced_data = reduced_data.merge(
            previous_and_next_indices[
                ["fight_id", "fighter_id", "previous_fights", "previous_opponents"]
            ],
        )

        # Define the fighter history tensor, which will contain the
        # history of the fighters in the dataset.
        # Start by getting the previous fights statistics and parameters
        self.fighter_history_tensor = torch.FloatTensor(
            [previous_and_next_indices[x] for x in self.previous_fights_statistics]
        ).T

        # Add an empty tensor for the state vector of the fighters (initial)
        # And add fight parameters as well.
        self.fighter_history_tensor = torch.concat(
            (
                torch.zeros(
                    (self.fighter_history_tensor.size()[0], self.state_size),
                    dtype=torch.float,
                ),
                self.fighter_history_tensor,
                torch.FloatTensor(
                    [
                        previous_and_next_indices[x]
                        for x in self.previous_fights_parameters
                    ]
                ).T,
            ),
            dim=1,
        )

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

        # Now we load the data into torch tensors.
        # This is a list of FloatTensors each having a size equal to the number
        # fights.
        self.data: List[torch.Tensor] = [
            torch.FloatTensor(
                np.asarray(
                    [fight_data[x + "_x"].values for x in self.fighter_fight_statistics]
                ).T
            ),
            torch.FloatTensor(
                np.asarray(
                    [fight_data[x + "_y"].values for x in self.fighter_fight_statistics]
                ).T
            ),
            torch.FloatTensor(
                np.asarray(
                    [fight_data[xf + "_x"].values for xf in self.fight_parameters]
                ).T
            ),
            torch.FloatTensor(
                (fight_data["winner_x"] != fight_data["fighter_id_x"]).values
            ),
            torch.FloatTensor(fight_data["opening_x"].values),
            torch.FloatTensor(fight_data["opening_y"].values),
            fight_data["previous_fights_x"].values,
            fight_data["previous_fights_y"].values,
            fight_data["previous_opponents_x"].values,
            fight_data["previous_opponents_y"].values,
        ]

        if len(self.fight_parameters) == 0:
            self.data[2] = torch.empty(len(fight_data["winner_x"]), 0)

        self.fight_data = fight_data

    def __len__(self) -> int:
        """Returns the size of the dataset.

        Returns:
            The size of the dataset.
        """
        return len(self.data[0])

    def __getitem__(self, idx: int) -> Tuple[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Getter for the dataset.

        Data is augmented by randomly switching the fighters position in the network.

        Args:
            idx: The index of the data to return.

        Returns:
            A tuple of ((X1, X2, X3, fighter_prev_data, opponent_prev_data,
            fighter_prev_opponents_data, opponent_prev  _opponents_data), winner,
            (odds_1, odds_2)) where X1 and X2 are the statistics for the two
            fighters, X3 are the fight parameters, winner is the winner of the fight
            and (odds1, odds2) the odds for each of the fighters.
            The rest of elements are the statistics of the previous fights to be
            used in the model.
        """
        X1, X2, X3, winner, odds_1, odds_2, f_prev_f, o_prev_f, f_prev_o, o_prev_o = [
            x[idx] for x in self.data
        ]

        if np.random.random() >= 0.5:
            X1, X2 = X2, X1
            winner = 1 - winner
            odds_1, odds_2 = odds_2, odds_1
            f_prev_f, o_prev_f = o_prev_f, f_prev_f
            f_prev_o, o_prev_o = o_prev_o, f_prev_o

        fighter_prev_data = self.fighter_history_tensor[f_prev_f]
        opponent_prev_data = self.fighter_history_tensor[o_prev_f]
        fighter_prev_opponents_data = self.fighter_history_tensor[f_prev_o]
        opponent_prev_opponents_data = self.fighter_history_tensor[o_prev_o]

        return (
            (
                X1,
                X2,
                X3,
                pad_or_truncate(fighter_prev_data, self.num_past_fights),
                pad_or_truncate(opponent_prev_data, self.num_past_fights),
                pad_or_truncate(fighter_prev_opponents_data, self.num_past_fights),
                pad_or_truncate(opponent_prev_opponents_data, self.num_past_fights),
            ),
            winner.reshape(-1),
            (
                odds_1.reshape(-1),
                odds_2.reshape(-1),
            ),
        )

    def get_fight_data_from_ids(self, fight_ids: Optional[List[str]] = None) -> Tuple[
        Tuple[
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
        ],
        torch.FloatTensor,
        Tuple[torch.Tensor, torch.Tensor],
        NDArray[np.str_],
        NDArray[np.str_],
    ]:
        """
        Return data for the given fight ids.

        If fight_ids is None, returns all the data in the dataset.

        Args:
            fight_ids: The list of fight ids to include in the dataset. If None,
                use all the data in the dataset.

        Returns:
            A tuple of ((X1, X2, X3, fighter_prev_data, opponent_prev_data,
            fighter_prev_opponents_data, opponent_prev_opponents_data), winner,
            (odds_1, odds_2)) where X1 and X2 are the statistics for the two
            fighters, X3 are the fight parameters, winner is the winner of the fight
            and (odds1, odds2) the odds for each of the fighters.
            The rest of elements are the statistics of the previous fights to be
            used in the model.
        """
        if fight_ids is not None:
            fight_data = self.fight_data[self.fight_data["fight_id"].isin(fight_ids)]
        else:  # pragma: no cover
            fight_data = self.fight_data.copy()

        data = [
            torch.FloatTensor(
                np.asarray(
                    [fight_data[x + "_x"].values for x in self.fighter_fight_statistics]
                ).T
            ),
            torch.FloatTensor(
                np.asarray(
                    [fight_data[x + "_y"].values for x in self.fighter_fight_statistics]
                ).T
            ),
            torch.FloatTensor(
                np.asarray(
                    [fight_data[x + "_x"].values for x in self.fight_parameters]
                ).T
            ),
            torch.FloatTensor(
                (fight_data["winner_x"] != fight_data["fighter_id_x"]).values
            ),
            torch.FloatTensor(fight_data["opening_x"].values),
            torch.FloatTensor(fight_data["opening_y"].values),
        ]

        fighter_names = np.array(fight_data["fighter_name_x"].values)
        opponent_names = np.array(fight_data["fighter_name_y"].values)

        fighter_prev_data = torch.FloatTensor(
            torch.stack(
                [
                    pad_or_truncate(
                        self.fighter_history_tensor[prev], self.num_past_fights
                    )
                    for prev in fight_data["previous_fights_x"].values
                ]
            )
        )
        opponent_prev_data = torch.FloatTensor(
            torch.stack(
                [
                    pad_or_truncate(
                        self.fighter_history_tensor[prev], self.num_past_fights
                    )
                    for prev in fight_data["previous_fights_y"].values
                ]
            )
        )
        fighter_prev_opponent_data = torch.FloatTensor(
            torch.stack(
                [
                    pad_or_truncate(
                        self.fighter_history_tensor[prev], self.num_past_fights
                    )
                    for prev in fight_data["previous_opponents_x"].values
                ]
            )
        )
        opponent_prev_opponent_data = torch.FloatTensor(
            torch.stack(
                [
                    pad_or_truncate(
                        self.fighter_history_tensor[prev], self.num_past_fights
                    )
                    for prev in fight_data["previous_opponents_y"].values
                ]
            )
        )

        X1, X2, X3, Y, odds1, odds2 = data

        return (
            (
                X1,
                X2,
                X3,
                fighter_prev_data,
                opponent_prev_data,
                fighter_prev_opponent_data,
                opponent_prev_opponent_data,
            ),
            Y,
            (
                odds1.reshape(-1, 1),
                odds2.reshape(-1, 1),
            ),
            fighter_names,
            opponent_names,
        )


class ForecastDatasetTimeEvolution(ForecastDataset, DatasetWithTimeEvolution):
    """
    A dataset class designed to handle forecasting data for UFC fight predictions.

    This class extends ForecastDataset to perform forecasting on models with time
    evolution.
    """

    fighter_fight_statistics = DatasetWithTimeEvolution.fighter_fight_statistics
    fight_parameters = DatasetWithTimeEvolution.fight_parameters
    previous_fights_statistics = DatasetWithTimeEvolution.previous_fights_statistics
    previous_fights_parameters = DatasetWithTimeEvolution.previous_fights_parameters

    def __init__(
        self,
        data_processor: DataProcessor,
        fighter_fight_statistics: Optional[List[str]] = None,
        fight_parameters: Optional[List[str]] = None,
        previous_fights_statistics: Optional[List[str]] = None,
        previous_fights_parameters: Optional[List[str]] = None,
        state_size: Optional[int] = None,
    ) -> None:
        """
        Constructor for ForecastDataset.

        Args:
            data_processor: The DataProcessor instance that contains the data.
            fighter_fight_statistics: The list of columns to include in the dataset. If None, use all
                columns.
            fight_parameters: The list of fight parameters to include in the
                model. If None, use an empty list.
            previous_fights_statistics: The list of columns to use for previous
                fights statistics. If None, use default columns defined in
                cls.previous_fights_statistics.
            previous_fights_parameters: The list of columns to use for previous
                fights parameters. If None, use default columns defined in
                cls.previous_fights_parameters.
            state_size: The size of the state vector for each fighter. If None,
                use the default value defined in cls.state_size.

        Raises:
            ValueError: If some columns are not found in the normalized data.
        """
        self.data_processor = data_processor

        if fighter_fight_statistics is not None:
            self.fighter_fight_statistics = fighter_fight_statistics

        if fight_parameters is not None:
            self.fight_parameters = fight_parameters

        if previous_fights_statistics is not None:
            self.previous_fights_statistics = previous_fights_statistics

        if previous_fights_parameters is not None:
            self.previous_fights_parameters = previous_fights_parameters

        if state_size is not None:
            self.state_size = state_size

        not_found = []
        for column in self.fighter_fight_statistics + self.fight_parameters:
            if (
                column not in self.data_processor.data_normalized.columns
            ):  # pragma: no cover
                not_found.append(column)

        if len(not_found) > 0:  #  pragma: no cover
            raise ValueError(f"Columns not found in normalized data: {not_found}")

        self.fight_ids = None
        self.load_data()

    def get_single_forecast_prediction(
        self,
        fighter_name: str,
        opponent_name: str,
        event_date: str | datetime.date,
        odds1: int,
        odds2: int,
        model: nn.Module,
        fight_parameters_values: List[float] = [],
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

        Returns:
            A tuple of two numpy arrays, each one evaluating the model switching
            between the two fighters. For symmetric models, they should be the same.
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
            fight_parameters_values=[
                fight_parameters_values,
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
        fight_parameters_values: List[List[float]] = [],
        parse_ids: bool = False,
        device: str | torch.device = "cpu",
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
            A tuple of two numpy arrays, each one evaluating the model switching
            between the two fighters. For symmetric models, they should be the same.
        """
        if not parse_ids:
            fighter_ids = [self.data_processor.get_fighter_id(x) for x in fighter_names]
            opponent_ids = [
                self.data_processor.get_fighter_id(x) for x in opponent_names
            ]
        else:  # pragma: no cover
            fighter_ids = fighter_names
            opponent_ids = opponent_names

        match_data = self.get_match_data_for_predictions(
            fighter_ids=fighter_ids,
            opponent_ids=opponent_ids,
            event_dates=event_dates,
            fighter_odds=fighter_odds,
            opponent_odds=opponent_odds,
            fight_parameters_values=fight_parameters_values,
        )

        previous_and_next_indices = self.get_indices_previous_and_next(
            include_current=True
        )

        match_data = match_data.merge(
            previous_and_next_indices[
                ["fight_id", "fighter_id", "previous_fights", "previous_opponents"]
            ]
        )

        ###############################################################
        # Now we start building the tensor to input to the model
        ###############################################################
        # This data dict is used to facilitate the construction of the tensors
        data_dict = {
            id_: data
            for id_, data in zip(
                match_data["id_"].values,
                np.asarray([match_data[x] for x in self.fighter_fight_statistics]).T,
            )
        }

        if len(self.fight_parameters) > 0:
            fight_data_dict = {
                id_: data
                for id_, data in zip(
                    match_data["id_"].values,
                    np.asarray([match_data[x] for x in self.fight_parameters]).T,
                )
            }
        else:  # pragma: no cover
            fight_data_dict = {id_: [] for id_ in match_data["id_"].values}

        fighter_history_dict = {
            id_: data
            for id_, data in zip(
                match_data["id_"].values,
                np.asarray(
                    [
                        pad_or_truncate(
                            self.fighter_history_tensor[idxs], self.num_past_fights
                        )
                        .detach()
                        .numpy()
                        for idxs in match_data["previous_fights"].values
                    ]
                ),
            )
        }

        opponent_history_dict = {
            id_: data
            for id_, data in zip(
                match_data["id_"].values,
                np.asarray(
                    [
                        pad_or_truncate(
                            self.fighter_history_tensor[idxs], self.num_past_fights
                        )
                        .detach()
                        .numpy()
                        for idxs in match_data["previous_opponents"].values
                    ]
                ),
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
            torch.FloatTensor(
                np.asarray(
                    [
                        fighter_history_dict[fighter_id + "_" + str(event_date)]
                        for fighter_id, event_date in zip(fighter_ids, event_dates)
                    ]
                )
            ),
            torch.FloatTensor(
                np.asarray(
                    [
                        fighter_history_dict[fighter_id + "_" + str(event_date)]
                        for fighter_id, event_date in zip(opponent_ids, event_dates)
                    ]
                )
            ),
            torch.FloatTensor(
                np.asarray(
                    [
                        opponent_history_dict[fighter_id + "_" + str(event_date)]
                        for fighter_id, event_date in zip(fighter_ids, event_dates)
                    ]
                )
            ),
            torch.FloatTensor(
                np.asarray(
                    [
                        opponent_history_dict[fighter_id + "_" + str(event_date)]
                        for fighter_id, event_date in zip(opponent_ids, event_dates)
                    ]
                )
            ),
        ]

        (
            X1,
            X2,
            X3,
            odds1,
            odds2,
            fighter_prev_data,
            opponent_prev_data,
            fighter_prev_opponents_data,
            opponent_prev_opponents_data,
        ) = data
        data = [x.to(device) for x in data]

        model.eval()
        with torch.no_grad():
            predictions_1 = model(
                X1,
                X2,
                X3,
                fighter_prev_data,
                opponent_prev_data,
                fighter_prev_opponents_data,
                opponent_prev_opponents_data,
                odds1,
                odds2,
            )
            predictions_2 = 1 - model(
                X2,
                X1,
                X3,
                opponent_prev_data,
                fighter_prev_data,
                opponent_prev_opponents_data,
                fighter_prev_opponents_data,
                odds2,
                odds1,
            )

        return predictions_1, predictions_2
