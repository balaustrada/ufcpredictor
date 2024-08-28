from __future__ import annotations

import logging

from pathlib import Path
from typing import TYPE_CHECKING

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
    import datetime
    from typing import Any, Callable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DataProcessor:
    input_fields = [
        "knockdowns_per_min",
        "strikes_att_per_min",
        "strikes_succ_per_min",
        "head_strikes_att_per_min",
        "head_strikes_succ_per_min",
        "body_strikes_att_per_min",
        "body_strikes_succ_per_min",
        "leg_strikes_att_per_min",
        "leg_strikes_succ_per_min",
        "distance_strikes_att_per_min",
        "distance_strikes_succ_per_min",
        "ground_strikes_att_per_min",
        "ground_strikes_succ_per_min",
        "clinch_strikes_att_per_min",
        "clinch_strikes_succ_per_min",
        "total_strikes_att_per_min",
        "total_strikes_succ_per_min",
        "takedown_att_per_min",
        "takedown_succ_per_min",
        "submission_att_per_min",
        "reversals_per_min",
        "ctrl_time_per_min",
        "KO_per_min",
        "KO_opp_per_min",
        "Sub_per_min",
        "Sub_opp_per_min",
        "win_per_min",
        "time_since_last_fight",
    ]
    averaged_columns = [
        "knockdowns",
        "strikes_att",
        "strikes_succ",
        "head_strikes_att",
        "head_strikes_succ",
        "body_strikes_att",
        "body_strikes_succ",
        "leg_strikes_att",
        "leg_strikes_succ",
        "distance_strikes_att",
        "distance_strikes_succ",
        "ground_strikes_att",
        "ground_strikes_succ",
        "clinch_strikes_att",
        "clinch_strikes_succ",
        "total_strikes_att",
        "total_strikes_succ",
        "takedown_att",
        "takedown_succ",
        "submission_att",
        "reversals",
        "ctrl_time",
        "num_rounds",
        "total_time",
        "KO",
        "KO_opp",
        "Sub",
        "Sub_opp",
        "win",
    ]

    def __init__(self, data_folder: Path | str) -> None:
        self.data_folder = data_folder
        self.bfo_scraper = BestFightOddsScraper(
            data_folder=self.data_folder,
            n_sessions=-1,
        )
        self.scraper = UFCScraper(
            data_folder=self.data_folder,
        )

    def prepare_fight_data(self) -> pd.DataFrame:
        # We prepare UFCStats data
        data = self.bfo_scraper.get_ufcstats_data()

        # Now we add odds information
        data = data.merge(
            self.bfo_scraper.data,
            on=["fight_id", "fighter_id"],
        )

        # We now join with round data, which we previously
        # sum by fight to get full fight statistics
        # (we don't yet handle round information separately)
        round_data = self.scraper.fight_scraper.rounds_handler.data
        round_data["ctrl_time"] = round_data["ctrl_time"].apply(
            convert_minutes_to_seconds
        )
        data = data.merge(
            round_data.groupby(["fight_id", "fighter_id"])
            .sum()
            .reset_index()
            .drop("round", axis=1),
            on=["fight_id", "fighter_id"],
        )

        # We now join with fight data to get winner,
        # winner round, gender, weight,  etc...
        data = data.merge(
            self.scraper.fight_scraper.data[
                [
                    "fight_id",
                    "winner",
                    "num_rounds",
                    "weight_class",
                    "gender",
                    "result",
                    "result_details",
                    "finish_round",
                    "finish_time",
                    "time_format",
                ]
            ],
            on=["fight_id"],
        )
        data["finish_time"] = data["finish_time"].apply(convert_minutes_to_seconds)
        data["total_time"] = (data["finish_round"] - 1) * 5 * 60 + data["finish_time"]
        # This means we only support the now usual 2 formats.
        data = data[data["time_format"].isin(["3 Rnd (5-5-5)", "5 Rnd (5-5-5-5-5)"])]

        # Remove catch, open and invalid weights
        data = data[
            (data["weight_class"] != "NULL")
            & (data["weight_class"] != "Catch Weight")
            & (data["weight_class"] != "Open Weight")
        ]

        # Translate weight into lbs
        data.loc[:, "weight"] = data["weight_class"].map(weight_dict)

        data = data[data["gender"] == "M"]

        # Remove disqualified and doctor's stoppage
        data = data[data["result"].isin(["Decision", "KO/TKO", "Submission"])]

        # Add extra fields for KO Sub and win
        data["KO"] = np.where(
            (data["result"].str.contains("KO"))
            & (data["winner"] == data["fighter_id"]),
            1,
            0,
        )
        data["KO_opp"] = np.where(
            (data["result"].str.contains("KO"))
            & (data["winner"] != data["fighter_id"]),
            1,
            0,
        )
        data["Sub"] = np.where(
            (data["result"].str.contains("Submission"))
            & (data["winner"] == data["fighter_id"]),
            1,
            0,
        )
        data["Sub_opp"] = np.where(
            (data["result"].str.contains("Submission"))
            & (data["winner"] != data["fighter_id"]),
            1,
            0,
        )
        data["win"] = np.where(data["winner"] == data["fighter_id"], 1, 0)

        # We preserve only one name (this is just for ease of use)
        data["UFC_names"] = data["UFC_names"].apply(lambda x: x[0])
        data["opponent_UFC_names"] = data["opponent_UFC_names"].apply(lambda x: x[0])

        # Start adding aggregated info
        data = data.sort_values(by=["fighter_id", "event_date"])
        data["num_fight"] = data.groupby("fighter_id").cumcount() + 1
        data["prev_fight_date"] = data.groupby("fighter_id")["event_date"].shift(1)
        data["time_since_last_fight"] = (
            data["event_date"] - data["prev_fight_date"]
        ).dt.days

        for column in self.averaged_columns:
            data[column] = data[column].astype(float)
            data[column] = data.groupby("fighter_id")[column].cumsum()

        # Average required columns
        new_columns = {}
        for column in self.averaged_columns:
            new_columns[f"{column}_per_min"] = data[column] / data["total_time"]
            new_columns[f"{column}_per_fight"] = data[column] / data["num_fight"]

        data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)

        # Remove invalid fights
        invalid_fights = set(
            data[data["total_strikes_succ_per_min"].isna()]["fight_id"].tolist()
        )
        if len(invalid_fights) > 0:
            raise ValueError("Invalid fights found")
        
        invalid_fights.update(
            data[data["num_fight"] < 3]["fight_id"].tolist()
        )

        self.valid_fights = set(data["fight_id"]) - set(invalid_fights)

        # Normalize input fields
        for column in self.input_fields:
            data[column] = data[column] / data[column].mean()
        self.data = data

    def from_id_to_fight(self, X_set: List[str], id_: str, print_info: bool = False):
        # Get fighters
        fight = self.data[self.data["fight_id"] == id_].iloc[0]
        f1 = fight["fighter_id"]
        f2 = fight["opponent_id"]
        date = fight["event_date"]
        winner = fight["winner"]

        # Get previous data to the match
        # Getting all previous matches and getting information from the last match
        # Remember this is cumulative data
        f1p = self.data[
            (self.data["event_date"] < date) & (self.data["fighter_id"] == f1)
        ]
        f1p = f1p.iloc[f1p["event_date"].argmax()]

        f2p = self.data[
            (self.data["event_date"] < date) & (self.data["fighter_id"] == f2)
        ]
        f2p = f2p.iloc[f2p["event_date"].argmax()]

        # We collect the input data defined in X_set
        x1 = [f1p[x] for x in X_set]
        x2 = [f2p[x] for x in X_set]

        if print_info:
            print(fight["UFC_names"], " vs ", fight["opponent_UFC_names"])

            odds_data = self.bfo_scraper.data
            fight_mask = odds_data["fight_id"] == id_
            fighter_odds = odds_data[
                fight_mask & (odds_data["fighter_id"] == fight["fighter_id"])
            ]["opening"].values[0]
            opponent_odds = odds_data[
                fight_mask & (odds_data["fighter_id"] == fight["opponent_id"])
            ]["opening"].values[0]
            print(f"{fighter_odds} vs {opponent_odds}")

        return (
            torch.FloatTensor(x1),
            torch.FloatTensor(x2),
            torch.FloatTensor([float(winner == f2p["fighter_id"])]),
        )

    def get_dataset(self, fight_ids: List[str], X_set: List[str]):
        data = []
        for id_ in fight_ids:
            data.append(
                self.from_id_to_fight(
                    X_set=X_set,
                    id_=id_,
                )
            )

        class CustomDataset(Dataset):
            def __init__(self, data, mode="train"):
                self.data = data
                self.mode = mode  # @TODO revisit this it doesn't make sense

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                X1, X2, Y = self.data[idx]

                # Flip data to avoid RED or BLUE
                # predictions
                if np.random.random() >= 0.5:
                    (
                        X1,
                        X2,
                    ) = (
                        X2,
                        X1,
                    )
                    Y = 1 - Y

                if self.mode == "train":
                    return X1, X2, Y
                else:
                    return X1, X2

        return CustomDataset(data)
    
    def get_data_loader(self, fight_ids: List[str], X_set: List[str], *args, **kwargs):
        dataset = self.get_dataset(fight_ids, X_set)
        return DataLoader(dataset, *args, **kwargs)
    