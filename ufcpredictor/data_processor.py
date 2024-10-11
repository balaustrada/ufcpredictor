from __future__ import annotations

import logging

from pathlib import Path
from typing import TYPE_CHECKING

import datetime
import numpy as np
import pandas as pd
import torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import f1_score

from ufcscraper.ufc_scraper import UFCScraper
from ufcscraper.odds_scraper import BestFightOddsScraper
from ufcpredictor.utils import convert_minutes_to_seconds, weight_dict

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, Callable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(
        self,
        data_folder: Optional[Path | str],
        ufc_scraper: Optional[UFCScraper] = None,
        bfo_scraper: Optional[BestFightOddsScraper] = None,
    ) -> None:
        if data_folder is None and (ufc_scraper is None or bfo_scraper is None):
            raise ValueError(
                "If data_folder is None, both ufc_scraper and bfo_scraper "
                "should be provided"
            )

        self.scraper = ufc_scraper or UFCScraper(data_folder=data_folder)
        self.bfo_scraper = bfo_scraper or BestFightOddsScraper(
            data_folder=data_folder, n_sessions=-1
        )

    def load_data(self) -> None:
        data = self.join_dataframes()
        data = self.fix_date_and_time_fields(data)
        data = self.convert_odds_to_european(data)
        data = self.fill_weight(data)
        data = self.add_key_stats(data)
        data = self.apply_filters(data)
        self.data = self.group_round_data(data)

    def join_dataframes(self) -> pd.DataFrame:
        fight_data = self.scraper.fight_scraper.data
        round_data = self.scraper.fight_scraper.rounds_handler.data
        fighter_data = self.scraper.fighter_scraper.data
        event_data = self.scraper.event_scraper.data

        odds_data = self.bfo_scraper.data

        ###########################################################
        # I want to create two rows per match, one row for each fighter
        ###########################################################
        # Hence I need to duplicate the current fight data
        # Assigning fighter and opponent to each other
        data = pd.concat(
            [
                fight_data.rename(
                    columns={"fighter_1": "opponent_id", "fighter_2": "fighter_id"}
                ),
                fight_data.rename(
                    columns={"fighter_2": "opponent_id", "fighter_1": "fighter_id"}
                ),
            ]
        )

        # I am merging the fighter data to the previous table
        # This includes height, reach etc...
        fighter_fields = ["fighter_id", "fighter_name", "fighter_nickname"]
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
    def convert_odds_to_european(data: pd.DataFrame) -> pd.DataFrame:
        for field in "opening", "closing_range_min", "closing_range_max":
            msk = data[field] > 0

            data.loc[msk, field] = data.loc[msk, field] / 100 + 1
            data.loc[~msk, field] = 100 / -data.loc[~msk, field] + 1

        return data

    @staticmethod
    def fill_weight(data: pd.DataFrame) -> pd.DataFrame:
        data.loc[:, "weight"] = data["weight_class"].map(weight_dict)

        ##################################################################################
        # Remmove null weight classes, or open weight or catch weight (agreed weight outside a weight class)
        ##################################################################################
        data = data[
            (data["weight_class"] != "NULL")
            & (data["weight_class"] != "Catch Weight")
            & (data["weight_class"] != "Open Weight")
        ]

        return data

    @staticmethod
    def add_key_stats(data: pd.DataFrame) -> pd.DataFrame:
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
        # Remove old fights since I don't have odds for these
        data = data[data["event_date"].dt.date >= datetime.date(2008, 8, 1)]

        # Remove non-standard fight format
        data = data[data["time_format"].isin(["3 Rnd (5-5-5)", "5 Rnd (5-5-5-5-5)"])]

        # Remove female fights
        data = data[data["gender"] == "M"]

        # Remove disqualified and doctor's stoppage
        data = data[data["result"].isin(["Decision", "KO/TKO", "Submission"])]

        return data

    @property
    def round_stat_names(self) -> List[str]:
        return [
            c
            for c in self.scraper.fight_scraper.rounds_handler.columns
            if c not in ["fight_id", "fighter_id", "round"]
        ] + [
            c + "_opponent"
            for c in self.scraper.fight_scraper.rounds_handler.columns
            if c not in ["fight_id", "fighter_id", "round"]
        ]

    @property
    def stat_names(self):
        stat_names = self.round_stat_names
        for field in ("KO", "Sub", "win"):
            stat_names += [field, field + "_opponent"]

        return stat_names

    @property
    def aggregated_fields(self):
        return self.stat_names

    @property
    def normalized_fields(self):
        normalized_fields = ["age", "time_since_last_fight", "fighter_height_cm"]

        for field in self.aggregated_fields:
            normalized_fields += [field, field + "_per_minute", field + "_per_fight"]

        return normalized_fields

    def group_round_data(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def aggregate_data(self):
        logger.info(f"Fields to be aggregated: {self.aggregated_fields}")

        data_aggregated = self.data.copy()
        data_aggregated["num_fight"] = (
            data_aggregated.groupby("fighter_id").cumcount() + 1
        )

        data_aggregated["previous_fight_date"] = data_aggregated.groupby("fighter_id")[
            "event_date"
        ].shift(1)
        data_aggregated["time_since_last_fight"] = (
            data_aggregated["event_date"] - data_aggregated["previous_fight_date"]
        ).dt.days

        for column in self.aggregated_fields:
            data_aggregated[column] = data_aggregated[column].astype(float)
            data_aggregated[column] = data_aggregated.groupby("fighter_id")[
                column
            ].cumsum()

        data_aggregated["total_time"] = data_aggregated.groupby("fighter_id")[
            "total_time"
        ].cumsum()

        self.data_aggregated = data_aggregated

    def add_per_minute_and_fight_stats(self):
        new_columns = {}

        for column in self.aggregated_fields:
            new_columns[column + "_per_minute"] = (
                self.data_aggregated[column] / self.data_aggregated["total_time"]
            )
            new_columns[column + "_per_fight"] = (
                self.data_aggregated[column] / self.data_aggregated["num_fight"]
            )

        self.data_aggregated = pd.concat(
            [self.data_aggregated, pd.DataFrame(new_columns)], axis=1
        ).copy()

    def normalize_data(self):
        data_normalized = self.data_aggregated.copy()

        logger.info(f"Fields to be normalized: {self.normalized_fields}")

        for column in self.normalized_fields:
            mean = self.data_aggregated[column].mean()
            data_normalized[column] = data_normalized[column] / mean

        self.data_normalized = data_normalized

    # def from_id_to_fight(self, X_set: List[str], id_: str, print_info: bool = False):
    #     # Get fighters
    #     fight = self.data[self.data["fight_id"] == id_].iloc[0]
    #     f1 = fight["fighter_id"]
    #     f2 = fight["opponent_id"]
    #     date = fight["event_date"]
    #     winner = fight["winner"]

    #     # Get previous data to the match
    #     # Getting all previous matches and getting information from the last match
    #     # Remember this is cumulative data
    #     f1p = self.data[
    #         (self.data["event_date"] < date) & (self.data["fighter_id"] == f1)
    #     ]
    #     f1p = f1p.iloc[f1p["event_date"].argmax()]

    #     f2p = self.data[
    #         (self.data["event_date"] < date) & (self.data["fighter_id"] == f2)
    #     ]
    #     f2p = f2p.iloc[f2p["event_date"].argmax()]

    #     # We collect the input data defined in X_set
    #     x1 = [f1p[x] for x in X_set]
    #     x2 = [f2p[x] for x in X_set]

    #     if print_info:
    #         print(fight["UFC_names"], " vs ", fight["opponent_UFC_names"])

    #         odds_data = self.bfo_scraper.data
    #         fight_mask = odds_data["fight_id"] == id_
    #         fighter_odds = odds_data[
    #             fight_mask & (odds_data["fighter_id"] == fight["fighter_id"])
    #         ]["opening"].values[0]
    #         opponent_odds = odds_data[
    #             fight_mask & (odds_data["fighter_id"] == fight["opponent_id"])
    #         ]["opening"].values[0]
    #         print(f"{fighter_odds} vs {opponent_odds}")

    #     return (
    #         torch.FloatTensor(x1),
    #         torch.FloatTensor(x2),
    #         torch.FloatTensor([float(winner == f2p["fighter_id"])]),
    #     )

    # def get_dataset(self, fight_ids: List[str], X_set: List[str]):
    #     data = []
    #     for id_ in fight_ids:
    #         data.append(
    #             self.from_id_to_fight(
    #                 X_set=X_set,
    #                 id_=id_,
    #             )
    #         )

    #     class CustomDataset(Dataset):
    #         def __init__(self, data, mode="train"):
    #             self.data = data
    #             self.mode = mode  # @TODO revisit this it doesn't make sense

    #         def __len__(self):
    #             return len(self.data)

    #         def __getitem__(self, idx):
    #             X1, X2, Y = self.data[idx]

    #             # Flip data to avoid RED or BLUE
    #             # predictions
    #             if np.random.random() >= 0.5:
    #                 (
    #                     X1,
    #                     X2,
    #                 ) = (
    #                     X2,
    #                     X1,
    #                 )
    #                 Y = 1 - Y

    #             if self.mode == "train":
    #                 return X1, X2, Y
    #             else:
    #                 return X1, X2

    #     return CustomDataset(data)

    # def get_data_loader(self, fight_ids: List[str], X_set: List[str], *args, **kwargs):
    #     dataset = self.get_dataset(fight_ids, X_set)
    #     return DataLoader(dataset, *args, **kwargs)


class OSRDataProcessor(DataProcessor):
    def aggregate_data(self):
        super().aggregate_data()

        # Adding OSR information
        df = self.data_aggregated[
            ["fighter_id", "fight_id", "opponent_id", "event_date"]
        ].copy()
        df["S"] = self.data_aggregated["win"] / self.data_aggregated["num_fight"]
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

        self.data_aggregated["OSR"] = new_OSR


class WOSRDataProcessor(DataProcessor):
    def __init__(self, *args, weights: List[float] = [0.3, 0.3, 0.3], **kwargs):
        super().__init__(*args, **kwargs)

        self.skills_weight, self.past_OSR_weight, self.opponent_OSR_weight = weights

    def aggregate_data(self):
        super().aggregate_data()

        # Adding OSR information
        df = self.data_aggregated[
            ["fighter_id", "fight_id", "opponent_id", "event_date"]
        ].copy()
        df["S"] = self.data_aggregated["win"] / self.data_aggregated["num_fight"]
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

        self.data_aggregated["OSR"] = new_OSR
