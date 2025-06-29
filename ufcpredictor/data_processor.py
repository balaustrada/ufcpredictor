"""
Data processing module for UFC fight data.

Provides classes to prepare and normalize data for model training and evaluation.
Handles data transformation, normalization, and feature engineering.
"""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from fuzzywuzzy.process import extractOne
from ufcscraper.odds_scraper import BestFightOddsScraper
from ufcscraper.ufc_scraper import UFCScraper

from ufcpredictor.data_aggregator import DefaultDataAggregator
from ufcpredictor.utils import convert_minutes_to_seconds, weight_dict

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path
    from typing import Dict, List, Optional

    from ufcpredictor.data_aggregator import DataAggregator
    from ufcpredictor.data_enhancers import DataEnhancer

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    A data processor class designed to prepare and normalize UFC fight data for
    training and testing neural network models.

    This class provides a way to handle and transform raw data into a format suitable
    for model training and evaluation.

    The DataProcessor is designed to work with the dataset classes in
    ufcpredictor.datasets to provide a seamless data preparation workflow.
    """

    mlflow_params: List[str] = []
    normalization_factors: Dict[str, float] = {}

    def __init__(
        self,
        data_folder: Optional[Path | str] = None,
        ufc_scraper: Optional[UFCScraper] = None,
        bfo_scraper: Optional[BestFightOddsScraper] = None,
        data_aggregator: Optional[DataAggregator] = None,
        data_enhancers: List[DataEnhancer] = [],
    ) -> None:
        """
        Constructor for DataProcessor.

        Args:
            data_folder: The folder containing the data.
            ufc_scraper: The scraper to use for ufc data.
            bfo_scraper: The scraper to use for best fight odds data.
            data_aggregator: The data aggregator to use for aggregating data.
            data_enhancers: The list of data enhancers to apply to the data.

        Raises:
            ValueError: If data_folder is None and both ufc_scraper and
                bfo_scraper are None.
        """
        if data_folder is None and (ufc_scraper is None or bfo_scraper is None):
            raise ValueError(
                "If data_folder is None, both ufc_scraper and bfo_scraper "
                "should be provided"
            )

        self.scraper = ufc_scraper or UFCScraper(data_folder=data_folder)
        self.bfo_scraper = bfo_scraper or BestFightOddsScraper(
            data_folder=data_folder, n_sessions=-1
        )

        self.data_aggregator = data_aggregator or DefaultDataAggregator()
        self.data_enhancers = data_enhancers

    def load_data(self) -> None:
        """
        Loads and processes all the data.

        First, it joins all the relevant dataframes (fight, fighter, event, and odds).
        Then, it fixes the date and time fields, converts the odds to decimal format,
        fills the weight for each fighter (if not available), adds key statistics
        (KO, Submission, and Win), and applies filters to the data.
        Finally, it groups the round data by fighter and fight, and assigns the result
        to the data attribute.

        This method should be called before any other method.
        """
        data = self.join_dataframes()
        data = self.fix_date_and_time_fields(data)
        data = self.convert_odds_to_decimal(data)
        data = self.fill_weight(data)
        data = self.add_key_stats(data)
        data = self.apply_filters(data)
        self.data = self.group_round_data(data)
        self.data["num_fight"] = self.data.groupby("fighter_id").cumcount() + 1

        for data_enhancer in self.data_enhancers:
            self.data = data_enhancer.add_data_fields(self.data)

        names = self.data["fighter_name"].values
        ids = self.data["fighter_id"].values

        self.fighter_names = {id_: name_ for id_, name_ in zip(ids, names)}
        self.fighter_ids = {name_: id_ for id_, name_ in zip(ids, names)}

    def get_fighter_name(self, id_: str) -> str:
        """
        Returns the name of the fighter with the given id.

        Args:cla
            id_: The id of the fighter.

        Returns:
            The name of the fighter.
        """
        return self.fighter_names[id_]

    def get_fighter_id(self, name: str) -> str:
        """
        Returns the id of the fighter with the given name.
        Search is performed using fuzzywuzzy.
        If multiple matches are found, the first one is returned.

        Args:
            name: The name of the fighter.

        Returns:
            The id of the fighter.
        """
        best_name, score = extractOne(name, self.fighter_ids.keys())

        if score < 100:
            logger.warning(
                f"Fighter found for {name} with {score}% accuracy: {best_name}"
            )
        return self.fighter_ids[best_name]

    def join_dataframes(self) -> pd.DataFrame:
        """
        Joins all the relevant dataframes (fight, fighter, event, and odds).

        It duplicates the current fight data to create two rows per match,
        one row for each fighter, and assigns fighter and opponent to each other.
        Then, it merges the fighter data, round data, and odds data to the
        previous table. Finally, it adds the date of the event to the dataframe.

        Returns:
            The joined dataframe.
        """
        fight_data = self.scraper.fight_scraper.data
        round_data = self.scraper.fight_scraper.rounds_handler.data
        fighter_data = self.scraper.fighter_scraper.data
        event_data = self.scraper.event_scraper.data
        replacement_data = self.scraper.replacement_scraper.data

        odds_data = self.bfo_scraper.data

        ###########################################################
        # I want to create two rows per match, one row for each fighter
        ###########################################################
        # Hence I need to duplicate the current fight data
        # Assigning fighter and opponent to each other
        data = pd.concat(
            [
                fight_data.rename(
                    columns={
                        "fighter_1": "opponent_id",
                        "fighter_2": "fighter_id",
                        "scores_1": "opponent_score",
                        "scores_2": "fighter_score",
                    }
                ),
                fight_data.rename(
                    columns={
                        "fighter_2": "opponent_id",
                        "fighter_1": "fighter_id",
                        "scores_2": "opponent_score",
                        "scores_1": "fighter_score",
                    }
                ),
            ]
        )

        # I am merging the fighter data to the previous table
        # This includes height, reach etc...
        fighter_data["fighter_name"] = (
            fighter_data["fighter_f_name"]
            + " "
            + fighter_data["fighter_l_name"].fillna("")
        )
        data = data.merge(
            fighter_data,  # [fighter_fields],
            on="fighter_id",
            how="left",
        )

        data = data.merge(
            fighter_data[["fighter_id", "fighter_name", "fighter_nickname"]],
            left_on="opponent_id",
            right_on="fighter_id",
            how="left",
            suffixes=("", "_opponent"),
        )

        # Also merging the replacement data if available
        data = data.merge(
            replacement_data,
            on=["fight_id", "fighter_id"],
            how="left",
        )
        data["notice_days"] = 1 / data["notice_days"].fillna(60)

        #############################################################
        # Add round data.
        #############################################################

        # Merging columns
        round_data = pd.merge(
            round_data,
            round_data,
            on=["fight_id", "round"],
            suffixes=("", "_opponent"),
        )

        # And then remove the match of the fighter with itself
        round_data = round_data[
            round_data["fighter_id"] != round_data["fighter_id_opponent"]
        ]

        data = data.merge(
            round_data,
            on=[
                "fight_id",
                "fighter_id",
                "fighter_id_opponent",
            ],
        )

        ##############################################################
        # Add odds data
        ###############################################################
        data = data.merge(
            odds_data,
            on=["fight_id", "fighter_id"],
        )

        # Add the date of the event to the dataframe
        data = data.merge(
            event_data[["event_id", "event_date"]],  # I only need the date for now,
            on="event_id",
        )

        return data

    @staticmethod
    def fix_date_and_time_fields(data: pd.DataFrame) -> pd.DataFrame:
        """
        Fix date and time fields in the dataframe.

        This function takes care of converting control time, finish time
        and total time from minutes to seconds. It also converts the
        event date and fighter date of birth to datetime objects.

        The dataframe is then sorted by fighter id and event date.

        Args:
            data: The dataframe to be processed.

        Returns:
            The dataframe with the fields fixed.
        """
        data["ctrl_time"] = data["ctrl_time"].apply(convert_minutes_to_seconds)
        data["ctrl_time_opponent"] = data["ctrl_time_opponent"].apply(
            convert_minutes_to_seconds
        )
        data["finish_time"] = data["finish_time"].apply(convert_minutes_to_seconds)
        data["total_time"] = (data["finish_round"] - 1) * 5 * 60 + data["finish_time"]
        data["event_date"] = pd.to_datetime(data["event_date"])
        data["fighter_dob"] = pd.to_datetime(data["fighter_dob"])

        data = data.sort_values(by=["fighter_id", "event_date"])

        return data

    @staticmethod
    def convert_odds_to_decimal(data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert odds from American format to decimal format.

        Args:
            data: The dataframe with the odds in American format.

        Returns:
            The dataframe with the odds in decimal format.
        """
        for field in "opening", "closing_range_min", "closing_range_max":
            data[field] = data[field].astype(float)
            msk = data[field] > 0

            data.loc[msk, field] = data.loc[msk, field] / 100 + 1
            data.loc[~msk, field] = 100 / -data.loc[~msk, field] + 1

        return data

    @staticmethod
    def fill_weight(data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill the weight column using the weight_class column and the weight_dict.

        The weight_dict is a dictionary mapping the weight classes to their
        corresponding weights in lbs. The weights are then filled in the weight
        column according to the weight classes in the weight_class column.

        This function also removes rows with null weight classes, or open weight
        or catch weight (agreed weight outside a weight class).

        Args:
            data: The dataframe to be processed.

        Returns:
            The dataframe with the weight column filled.
        """
        data.loc[:, "weight"] = data["weight_class"].map(weight_dict)

        ##################################################################################
        # Remove null weight classes, or open weight or catch weight (agreed weight outside a weight class)
        ##################################################################################
        data = data[
            (data["weight_class"] != "NULL")
            & (data["weight_class"] != "Catch Weight")
            & (data["weight_class"] != "Open Weight")
        ]

        return data

    @staticmethod
    def add_key_stats(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add key statistics to the dataframe.

        This function adds columns to the dataframe indicating whether a fighter
        has won a fight via KO, submission or decision, and whether the opponent
        has won a fight via KO, submission or decision. It also adds a column
        indicating the age of the fighter at the time of the fight.

        Args:
            data: The dataframe to be processed.

        Returns:
            The dataframe with the added columns.
        """
        #############################################
        # Add some missing stats
        # KO, Submission and Win
        #############################################
        # Whether fighter has KOd his opponent
        data["KO"] = np.where(
            (data["result"].str.contains("KO"))
            & (data["winner"] == data["fighter_id"]),
            1,
            0,
        )

        # Whether the fighter has been KOd by his opponent
        data["KO_opponent"] = np.where(
            (data["result"].str.contains("KO"))
            & (data["winner"] != data["fighter_id"]),
            1,
            0,
        )

        # Same for submission
        data["Sub"] = np.where(
            (data["result"].str.contains("Submission"))
            & (data["winner"] == data["fighter_id"]),
            1,
            0,
        )

        data["Sub_opponent"] = np.where(
            (data["result"].str.contains("Submission"))
            & (data["winner"] != data["fighter_id"]),
            1,
            0,
        )

        data["win"] = np.where(data["winner"] == data["fighter_id"], 1, 0)
        data["win_opponent"] = np.where(data["winner"] != data["fighter_id"], 1, 0)
        data["age"] = (data["event_date"] - data["fighter_dob"]).dt.days / 365

        return data

    @staticmethod
    def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filters to the dataframe.

        This function applies filters to the dataframe to remove fights:
        - Before 2008, 8, 1, since I don't have odds for these
        - With non-standard fight formats (time_format not in ["3 Rnd (5-5-5)", "5 Rnd (5-5-5-5-5)"])
        - With female fighters (gender not in ["M"])
        - With disqualified or doctor's stoppage results (result not in ["Decision", "KO/TKO", "Submission"])
        - With draws or invalid winners (winner not in ("Draw", "NC") or winner.isna())

        Args:
            data: The dataframe to be processed.

        Returns:
            The dataframe with the applied filters.
        """
        # Remove old fights since I don't have odds for these
        data = data[data["event_date"].dt.date >= datetime.date(2008, 8, 1)]

        # Remove non-standard fight format
        data = data[data["time_format"].isin(["3 Rnd (5-5-5)", "5 Rnd (5-5-5-5-5)"])]

        # Remove female fights
        data = data[data["gender"] == "M"]

        # Remove disqualified and doctor's stoppage
        data = data[data["result"].isin(["Decision", "KO/TKO", "Submission"])]

        # Remove draws and invalid and NC
        data = data[(~data["winner"].isin(["Draw", "NC"])) & (~data["winner"].isna())]

        return data

    @property
    def round_stat_names(self) -> List[str]:
        """
        The names of the round statistics.

        This property returns the names of the columns in the rounds data
        that are not in ["fight_id", "fighter_id", "round"]. It also returns
        the same names with "_opponent" appended, to represent the opponent's
        statistics.

        Returns:
            A list of strings, the names of the round statistics.
        """
        return [
            c
            for c in self.scraper.fight_scraper.rounds_handler.dtypes.keys()
            if c not in ["fight_id", "fighter_id", "round"]
        ] + [
            c + "_opponent"
            for c in self.scraper.fight_scraper.rounds_handler.dtypes.keys()
            if c not in ["fight_id", "fighter_id", "round"]
        ]

    @property
    def stat_names(self) -> List[str]:
        """
        The names of the statistics.

        This property returns the names of the columns in the rounds data
        that are not in ["fight_id", "fighter_id", "round"]. It also returns
        the same names with "_opponent" appended, to represent the opponent's
        statistics, and the names of the columns "KO", "Sub" and "win",
        which are the result of the fight, with "_opponent" appended to
        represent the opponent's result.

        Returns:
            A list of strings, the names of the statistics.
        """
        stat_names = self.round_stat_names
        for field in ("KO", "Sub", "win"):
            stat_names += [field, field + "_opponent"]

        return stat_names

    @property
    def aggregated_fields(self) -> List[str]:
        """
        The fields that are aggregated over the fighter's history.

        This property returns all the statistic names, including the ones
        with "_opponent" appended to represent the opponent's statistics.
        It also returns the aggregated fields added by the data enhancers.

        Returns:
            A list of strings, the names of the aggregated fields.
        """
        aggregated_fields = self.stat_names

        for data_enhancer in self.data_enhancers:
            aggregated_fields += data_enhancer.aggregated_fields

        return aggregated_fields

    @property
    def normalized_fields(self) -> List[str]:
        """
        The fields that are normalized over the fighter's history.

        These fields are normalized in the sense that they are divided by
        their mean value in the history of the fighter. This is done to
        reduce the effect of outliers and to make the data more comparable
        between different fighters.

        The fields normalized are:
        - "age"
        - "time_since_last_fight"
        - "fighter_height_cm"
        - "weight",
        - All the aggregated fields (see :meth:`aggregated_fields`),
          and the same fields with "_per_minute" and "_per_fight" appended,
          which represent the aggregated fields per minute and per fight,
          respectively.

        It also returns the normalized fields added by the data enhancers.

        Returns:
            A list of strings, the names of the normalized fields.
        """
        normalized_fields = [
            "age",
            "time_since_last_fight",
            "fighter_height_cm",
            "weight",
        ]

        for field in self.aggregated_fields:
            normalized_fields += [field, field + "_per_minute", field + "_per_fight"]

        for data_enhancer in self.data_enhancers:
            normalized_fields += data_enhancer.normalized_fields

        return normalized_fields

    def group_round_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Group the round data by the fixed fields and sum the round statistics.

        The fixed fields are the columns in the data that are not in the round
        statistics and not in ["round"]. The round statistics are the columns
        in the data that are in the round statistics and not in ["round"].

        Args:
            data: The data to be grouped.

        Returns:
            The grouped data, with the round statistics summed.
        """
        fixed_fields = [
            c
            for c in data.columns
            if c
            not in self.round_stat_names
            + [
                "round",
            ]
        ]

        return (
            data.groupby(
                fixed_fields, dropna=False
            )  # Important to group nans as valid values.
            .sum()
            .reset_index()
            .drop("round", axis=1)
        ).sort_values(by=["fighter_id", "event_date"])

    def aggregate_data(self) -> None:
        """
        Aggregate the data by combining the round statistics over the history of the
        fighters.

        The aggregated data is stored in the attribute data_aggregated.

        The specific implementation depends on the DataAggregator used.
        """
        self.data_aggregated = self.data_aggregator.aggregate_data(self)

        for data_enhancer in self.data_enhancers:
            self.data_aggregated = data_enhancer.add_aggregated_fields(
                self.data_aggregated
            )

    def add_per_minute_and_fight_stats(self) -> None:
        """
        Add two new columns to the aggregated data for each statistic.

        The first column is the statistic per minute, computed by dividing the
        statistic by the total time in the octagon. The second column is the
        statistic per fight, computed by dividing the statistic by the number
        of fights.

        The new columns are named <statistic>_per_minute and <statistic>_per_fight,
        where <statistic> is the name of the statistic.

        Args:
            None

        Returns:
            None
        """
        new_columns = {}
        new_columns_data = {}

        for column in self.aggregated_fields:
            new_columns[column + "_per_minute"] = (
                self.data_aggregated[column]
                / self.data_aggregated["weighted_total_time"]
            )
            new_columns[column + "_per_fight"] = (
                self.data_aggregated[column]
                / self.data_aggregated["weighted_num_fight"]
            )

            # Also aggregate data per fight in data attribute
            new_columns_data[column + "_per_minute"] = (
                self.data[column] / self.data["total_time"]
            )

        self.data_aggregated = pd.concat(
            [self.data_aggregated, pd.DataFrame(new_columns)], axis=1
        ).copy()

        self.data = pd.concat(
            [self.data, pd.DataFrame(new_columns_data)], axis=1
        ).copy()

    def normalize_data(self) -> None:
        """
        Normalize the aggregated data by dividing each column by its mean.

        This is done so that the data is more comparable between different fighters.
        The fields normalized are the ones in normalized_fields.

        Args:
            None

        Returns:
            None
        """
        data_normalized = self.data_aggregated.copy()

        logger.info(f"Fields to be normalized: {self.normalized_fields}")

        for column in self.normalized_fields:
            mean = self.data_aggregated[column].mean()
            data_normalized[column] = data_normalized[column] / mean

            self.normalization_factors[column] = mean

        self.data_normalized = data_normalized

        data_normalized_nonagg = self.data.copy()

        data_normalized_nonagg = data_normalized_nonagg.merge(
            self.data_normalized[
                ["fight_id", "fighter_id", "time_since_last_fight", "num_fight"]
            ],
            how="left",
        )

        for column in self.normalized_fields:
            if "_per_fight" not in column:
                mean = data_normalized_nonagg[column].mean()
                data_normalized_nonagg[column] = data_normalized_nonagg[column] / mean

        self.data_normalized_nonagg = data_normalized_nonagg
