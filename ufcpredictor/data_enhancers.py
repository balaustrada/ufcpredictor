"""
Data Enhancer Module for UFC fight data.

Provides classes to add derived features to the UFC fight data,
such as an ELO rating for each fighter based on opponent history.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, Dict, List

    from ufcpredictor.data_processor import DataProcessor


logger = logging.getLogger(__name__)


class DataEnhancer:
    """
    This class provides a base class for adding derived features to the UFC fight data.

    It is called by the DataProcessor class to add derived features to the data, after
    the data has been processed and after it has been aggregated.
    """

    mlflow_params: List[str] = []

    @property
    def aggregated_fields(self) -> List[str]:
        """
        The fields added by the data enhancer that need to be aggregated
        by the DataProcessor DataAggregator instance.

        Returns:
            A list of strings, the names of the aggregated fields.
        """
        return []

    @property
    def normalized_fields(self) -> List[str]:
        """
        The fields added by the data enhancer that need to be normalized
        by the DataProcessor instance.

        Returns:
            A list of strings, the names of the normalized fields.
        """
        return []

    def add_data_fields(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This method can be overridden by subclasses to add derived features to the data.

        Args:
            data: The data to add the derived features to.

        Returns:
            The data with the derived features added.
        """
        return data

    def add_aggregated_fields(self, data_aggregated: pd.DataFrame) -> pd.DataFrame:
        """
        This method can be overridden by subclasses to add derived features to the
        aggregated data.

        Args:
            data_aggregated: The aggregated data to add the derived features to.

        Returns:
            The aggregated data with the derived features added.
        """
        return data_aggregated


class RankedFields(DataEnhancer):
    """
    This class adds ranked fields to the data.

    The ranked fields are used to rank the fighters based on their statistics.

    Rank is calculated using the rank function from pandas as follows:
        data[field] = (data[field].rank(pct=True)*100)**exponent
    """

    mlflow_params: List[str] = ["fields", "exponents"]

    def __init__(self, fields: List[str], exponents: List[float] | float):
        """
        Initializes the RankedFields instance.

        Args:
            fields: The fields to rank.
            exponents: The exponents to use for the ranking.
        """
        if isinstance(exponents, float):  # pragma: no cover
            exponents = [exponents] * len(fields)

        self.fields = fields
        self.exponents = exponents

    def add_data_fields(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert fields into ranked fields.

        Args:
            data: The data to add the ranked fields to.

        Returns:
            The data with the ranked fields added.
        """
        for field, exponent in zip(self.fields, self.exponents):
            data[field + "not_ranked"] = data[field]
            data[field] = (data[field].rank(pct=True) * 100) ** exponent
        return data


class OSR(DataEnhancer):
    """
    Extends the DataProcessor class to add OSR information.

    The OSR shows the strength of a given fighter by showing its win/loss ratio
    with contributions from their opponents win/loss ratio.

    The OSR is computed iteratively, for each iteration the new OSR is computed
    as:
        new_OSR = (old_OSR + mean_OSR_opponents + wins/n_fights)
    """

    mlflow_params: List[str] = []

    def add_aggregated_fields(self, data_aggregated: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data by computing the fighters' statistics and OSR.

        This method aggregates the data by computing the fighters' statistics and
        OSR (Opponent Strength Rating). The OSR is computed as the average of the
        opponent's OSR and the fighter's win rate.

        The OSR is calculated iteratively until the difference between the new and
        old values is less than 0.1.

        Args:
            data_aggregated: The aggregated data to add the derived features to.

        Returns:
            The aggregated data with OSR added.
        """
        # Adding OSR information
        df = data_aggregated[
            ["fighter_id", "fight_id", "opponent_id", "event_date"]
        ].copy()
        df["S"] = data_aggregated["win"] / data_aggregated["num_fight"]
        df["OSR"] = df["S"]

        diff = 1
        new_OSR = df["S"]

        while diff > 0.1:
            df["OSR"] = new_OSR
            df["OSR_past"] = df.groupby("fighter_id")["OSR"].shift(1)

            merged_df = df.merge(
                df,
                left_on="fighter_id",
                right_on="opponent_id",
                suffixes=("_x", "_y"),
                how="left",
            )

            merged_df = merged_df[
                (merged_df["event_date_x"] > merged_df["event_date_y"])
            ]

            OSR_opponent = merged_df.groupby(["fighter_id_x", "fight_id_x"])[
                "OSR_y"
            ].mean()

            df = (
                df[
                    [
                        "fighter_id",
                        "fight_id",
                        "opponent_id",
                        "event_date",
                        "S",
                        "OSR",
                        "OSR_past",
                    ]
                ]
                .merge(
                    OSR_opponent,
                    left_on=["fighter_id", "fight_id"],
                    right_on=["fighter_id_x", "fight_id_x"],
                    how="left",
                )
                .rename(columns={"OSR_y": "OSR_opp"})
            )

            new_OSR = df[["S", "OSR_opp", "OSR_past"]].mean(axis=1)

            diff = abs(new_OSR - df["OSR"]).sum()

        data_aggregated["OSR"] = new_OSR

        return data_aggregated


class WOSR(OSR):
    """
    Extends the OSRDataProcessor class to add weights to the different components
    of the OSR estimation.

    The OSR is computed iteratively, for each iteration the new OSR is computed
    as:
        new_OSR = (w1*old_OSR + w2*mean_OSR_opponents + w3*wins/n_fights)
    the weights are [w1, w2, w3]
    """

    mlflow_params: List[str] = [
        "weights",
    ]

    def __init__(self, weights: List[float] = [0.3, 0.3, 0.3]):
        """
        Initializes the WOSR instance.

        Args:
            weights: The weights to use for the OSR estimation.
        """
        self.skills_weight, self.past_OSR_weight, self.opponent_OSR_weight = weights

    def add_aggregated_fields(self, data_aggregated: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data by computing the fighters' statistics and OSR.

        This method aggregates the data by computing the fighters' statistics and
        OSR (Opponent Strength Rating). The OSR is computed as the average of the
        opponent's OSR and the fighter's win rate.

        The OSR is calculated iteratively until the difference between the new and
        old values is less than 0.1.

        Args:
            data_aggregated: The aggregated data to add the derived features to.

        Returns:
            The aggregated data with OSR added.
        """
        # Adding OSR information
        df = data_aggregated[
            ["fighter_id", "fight_id", "opponent_id", "event_date"]
        ].copy()
        df["S"] = data_aggregated["win"] / data_aggregated["num_fight"]
        df["OSR"] = df["S"]

        diff = 1
        new_OSR = df["S"]

        while diff > 0.1:
            df["OSR"] = new_OSR
            df["OSR_past"] = df.groupby("fighter_id")["OSR"].shift(1)

            merged_df = df.merge(
                df,
                left_on="fighter_id",
                right_on="opponent_id",
                suffixes=("_x", "_y"),
                how="left",
            )

            merged_df = merged_df[
                (merged_df["event_date_x"] > merged_df["event_date_y"])
            ]

            OSR_opponent = merged_df.groupby(["fighter_id_x", "fight_id_x"])[
                "OSR_y"
            ].mean()

            df = (
                df[
                    [
                        "fighter_id",
                        "fight_id",
                        "opponent_id",
                        "event_date",
                        "S",
                        "OSR",
                        "OSR_past",
                    ]
                ]
                .merge(
                    OSR_opponent,
                    left_on=["fighter_id", "fight_id"],
                    right_on=["fighter_id_x", "fight_id_x"],
                    how="left",
                )
                .rename(columns={"OSR_y": "OSR_opp"})
            )

            new_OSR = (
                df["S"].fillna(0) * self.skills_weight
                + df["OSR_past"].fillna(0) * self.past_OSR_weight
                + df["OSR_opp"].fillna(0) * self.opponent_OSR_weight
            )
            weight_sum = (
                (~df["S"].isna()) * self.skills_weight
                + (~df["OSR_past"].isna()) * self.past_OSR_weight
                + (~df["OSR_opp"].isna()) * self.opponent_OSR_weight
            )
            new_OSR /= weight_sum

            # new_OSR = df[["S", "OSR_opp", "OSR_past"]].mean(axis=1)

            diff = abs(new_OSR - df["OSR"]).sum()

        data_aggregated["OSR"] = new_OSR

        return data_aggregated


class ELO(DataEnhancer):
    """
    This class adds a traditional ELO ratings to the data based
    on the results of the fights.
    """

    mlflow_params: List[str] = ["initial_rating", "K_factor"]

    def __init__(self, initial_rating: float = 1000, K_factor: float = 32):
        """
        Initializes the ELO instance.

        Args:
            initial_rating: The initial rating of the fighters.
            K_factor: The K-factor of the Elo rating system.
        """
        super().__init__()

        self.initial_rating = initial_rating
        self.K_factor = K_factor

    @property
    def normalized_fields(self) -> List[str]:
        """
        The fields added by the data enhancer that need to be normalized
        by the DataProcessor instance.

        Returns:
            A list of strings, the names of the normalized fields.
        """
        return [
            "ELO",
        ]

    def add_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This method adds a score to a fight.

        In this class is not implemented and just adds 1 to each
        fight score.

        Args:
            data: The data to add the scores to.

        Returns:
            The data with the scores added.
        """
        data["match_score"] = 1
        return data

    def compute_new_rating(
        self, rating: float, S: float, E: float, match_score: float
    ) -> float:
        """
        Compute the new rating based on the ELO rating system.

        Args:
            rating: The current rating of the fighter.
            S: The score of the fighter.
            E: The expected score of the fighter.
            match_score: The score of the match.

        Returns:
            The new rating of the fighter.
        """
        return rating + self.K_factor * (S - E) * match_score

    def add_data_fields(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and add ELO ratings to the dataframe.

        Args:
            data: The dataframe to add the ELO ratings to.

        Returns:
            The dataframe with ELO ratings added.
        """
        ratings: Dict[int, float] = {}
        updated_ratings: List[Dict[str, float | str]] = []

        data = self.add_scores(data)
        unique_fights = data.drop_duplicates(subset="fight_id")

        unique_fights = unique_fights.merge(
            data[["fight_id", "fighter_id", "match_score"]],
            left_on=["fight_id", "opponent_id"],
            right_on=["fight_id", "fighter_id"],
            suffixes=("", "_opponent"),
        )

        for _, fight in unique_fights.sort_values(
            by="event_date", ascending=True
        ).iterrows():
            fight_id = fight["fight_id"]
            fighter_id = fight["fighter_id"]
            opponent_id = fight["opponent_id"]
            winner_id = fight["winner"]
            match_score = fight["match_score"]
            match_score_opponent = fight["match_score_opponent"]

            # Get current ratings, or initialize if not present
            rating_fighter = ratings.get(fighter_id, self.initial_rating)
            rating_opponent = ratings.get(opponent_id, self.initial_rating)

            # Calculate expected scores
            E_fighter = self.expected_ELO_score(rating_fighter, rating_opponent)
            E_opponent = self.expected_ELO_score(rating_opponent, rating_fighter)

            # Determine scores based on the winner
            S_fighter = 1 if winner_id == fighter_id else 0
            S_opponent = 1 if winner_id == opponent_id else 0

            # Update ratings
            new_rating_fighter = self.compute_new_rating(
                rating_fighter, S_fighter, E_fighter, match_score
            )
            new_rating_opponent = self.compute_new_rating(
                rating_opponent, S_opponent, E_opponent, match_score_opponent
            )

            # Store the updated ratings
            ratings[fighter_id] = new_rating_fighter
            ratings[opponent_id] = new_rating_opponent

            # Append the updated ratings to the list
            updated_ratings.extend(
                [
                    {
                        "fight_id": fight_id,
                        "fighter_id": fighter_id,
                        "ELO": new_rating_fighter,
                        "ELO_opponent": new_rating_opponent,
                    },
                    {
                        "fight_id": fight_id,
                        "fighter_id": opponent_id,
                        "ELO": new_rating_opponent,
                        "ELO_opponent": new_rating_fighter,
                    },
                ]
            )

        updated_ratings_df = pd.DataFrame(updated_ratings)

        return data.merge(
            updated_ratings_df,
            on=["fight_id", "fighter_id"],
        )

    @staticmethod
    def expected_ELO_score(r1: float, r2: float) -> float:
        """
        Calculate the expected ELO score for a match between two fighters.

        Args:
            r1: The rating of the first fighter.
            r2: The rating of the second fighter.

        Returns:
            The expected score for the match between the two fighters.
        """
        return 1 / (1 + 10 ** ((r2 - r1) / 400))


class FlexibleELO(ELO):
    """
    This class adds a flexible ELO ratings to the data based on the results
    of the fights.

    The strength of the win is calculated based on different statistics, and
    the possible boosts are controlled by the boost_values and n_boost_bins
    parameters.

    The rating of the fighter is transformed as:
        new_rating_fighter = rating_fighter + K_factor * (S_fighter - E_fighter) * boost
    """

    mlflow_params: List[str] = ["n_boost_bins", "boost_values"]

    def __init__(
        self,
        *args: Any,
        n_boost_bins: int = 3,
        boost_values: List[float] = [1, 1.2, 1.4],
        **kwargs: Any,
    ):
        """
        Initializes the flexible ELO instance, which adds a flexible ELO based
        on the results of the fights.

        The strength of the win is calculated based on different statistics, and
        the possible boosts are controlled by the boost_values and n_boost_bins
        parameters.

        The n_boost_bins parameters controls the number of bins to consider, and the
        boost_values controls de boost for each bin. E.g. if n_boost_bins = 3 and
        boost_values = [1, 1.2, 1.4], then close fights will get a boost of 1, fights
        with a clear win will get a boost of 1.2, and fights with a very clear win
        will get a boost of 1.4.

        Args:
            initial_rating: The initial rating of the fighters.
            K_factor: The K-factor of the Elo rating system.
            n_boost_bins: The number of bins to use for the boost values.
            boost_values: The values to use for the boost values.
        """
        super().__init__(*args, **kwargs)

        self.n_boost_bins = n_boost_bins
        self.boost_values = boost_values

        self.boost_bins = np.concatenate(
            (
                np.linspace(0, 50, n_boost_bins + 1)[:-1],
                np.linspace(50, 100, n_boost_bins + 1)[1:],
            )
        )
        self.boost_factors = [1 / x for x in boost_values][::-1] + boost_values[1:]

    def get_scores(self, series: pd.Series) -> pd.Series:
        """
        Calculate the rank of a series of values and apply a boost based on the
        class attribute boost_factors.

        Args:
            series: The series of values to calculate the rank for.

        Returns:
            The rank of the values in the series, with the boost applied.
        """
        z_score = (series - series.mean()) / series.std()
        percentile = z_score.rank(pct=True) * 100

        return pd.cut(
            percentile,
            bins=self.boost_bins,
            labels=self.boost_factors,
            right=False,
        ).astype(float)

    def add_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This method adds a score to a fight based on different statistics
        to provide the strength of the win.

        Args:
            data: The data to add the scores to.

        Returns:
            The data with the scores added.
        """
        strikes_score = self.get_scores(
            data["strikes_succ"] - data["strikes_succ_opponent"]
        )

        takedowns_scores = self.get_scores(
            data["takedown_succ"] - data["takedown_succ_opponent"]
        )

        control_scores = self.get_scores(data["ctrl_time"] - data["ctrl_time_opponent"])

        knockdown_scores = self.get_scores(
            data["knockdowns"] - data["knockdowns_opponent"]
        )

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

        points_score = self.get_scores(data["fighter_score"] - data["opponent_score"])

        match_score = 1
        for score in (
            strikes_score,
            takedowns_scores,
            control_scores,
            knockdown_scores,
            submission_scores,
            KO_scores,
            win_score,
            points_score,
        ):
            match_score *= score.fillna(1)

        data["match_score"] = match_score

        return data


class SumFlexibleELO(ELO):
    """
    This class adds a flexible ELO ratings to the data based on the results
    of the fights.

    The strength of the win is calculated based on different statistics and is
    stored in the "match_score" column.

    The rating of the fighter is transformed as:
        new_rating_fighter = rating_fighter + K_factor * (
            S_fighter - E_fighter * scaling_factor * (match_score -50) / 100)
        )
    """

    mlflow_params: List[str] = ["scaling_factor"]

    def __init__(self, *args: Any, scaling_factor: float = 0.5, **kwargs: Any):
        """
        Initializes the SumFlexibleELO instance.

        The strength of the win is calculated based on different statistics, and
        the possible boosts are controlled by the boost_values and n_boost_bins
        parameters.

        The rating of the fighter is transformed as:
            new_rating_fighter = rating_fighter + K_factor * (
                S_fighter - E_fighter * scaling_factor * (match_score -50) / 100)
            )

        Args:
            initial_rating: The initial rating of the fighters.
            K_factor: The learning rate.
            scaling_factor: The scaling factor that modulates the relevance of the
                match score.
        """
        super().__init__(*args, **kwargs)

        self.scaling_factor = scaling_factor

    def get_scores(self, series: pd.Series) -> pd.Series:
        """
        Calculates the strength of a statistic based on the mean and standard
        deviation of the statistic.

        Args:
            series: The statistic to be transformed.

        Returns:
            The transformed statistic as a rank percentile.
        """
        z_score = (series - series.mean()) / series.std()
        return z_score.rank(pct=True)

    def add_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        strikes_score = self.get_scores(
            data["strikes_succ"] - data["strikes_succ_opponent"]
        )

        takedowns_scores = self.get_scores(
            data["takedown_succ"] - data["takedown_succ_opponent"]
        )

        control_scores = self.get_scores(data["ctrl_time"] - data["ctrl_time_opponent"])

        knockdown_scores = self.get_scores(
            data["knockdowns"] - data["knockdowns_opponent"]
        )

        submission_scores = self.get_scores(
            (data["Sub"] - data["Sub_opponent"]).apply(lambda x: 1 if x == 1 else 0)
        )

        KO_scores = self.get_scores(
            (data["KO"] - data["KO_opponent"]).apply(lambda x: 1 if x == 1 else 0)
        )

        win_score = (data["winner"] == data["fighter_id"]).apply(
            lambda x: 1 if x else 0
        )

        points_score = self.get_scores(data["fighter_score"] - data["opponent_score"])

        match_score = 0
        for score in (
            strikes_score,
            takedowns_scores,
            control_scores,
            knockdown_scores,
            submission_scores,
            KO_scores,
            win_score,
            points_score,
        ):
            match_score += score.fillna(0)

        data["match_score"] = (
            strikes_score
            + takedowns_scores
            + control_scores
            + knockdown_scores
            + submission_scores
            + KO_scores
            + win_score
            + points_score
        ) / 8

        return data

    def compute_new_rating(
        self, rating: float, S: float, E: float, match_score: float
    ) -> float:
        """
        Compute the new rating based on the ELO rating system.

        Args:
            rating: The current rating of the fighter.
            S: The score of the fighter.
            E: The expected score of the fighter.
            match_score: The score of the match.

        Returns:
            The new rating of the fighter.
        """
        return rating + self.K_factor * (
            S - E + self.scaling_factor * (match_score - 50) / 100
        )
