from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path
    from typing import Any, List, Optional
    
    from ufcpredictor.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class DataAggregator:
    pass


class DefaultDataAggregator(DataAggregator):
    def aggregate_data(self, data_processor: DataProcessor) -> pd.DataFrame:
        logger.info(f"Fields to be aggregated: {data_processor.aggregated_fields}")

        data_aggregated = data_processor.data.copy()
        data_aggregated["num_fight"] = (
            data_aggregated.groupby("fighter_id").cumcount() + 1
        )

        data_aggregated["previous_fight_date"] = data_aggregated.groupby("fighter_id")[
            "event_date"
        ].shift(1)
        data_aggregated["time_since_last_fight"] = (
            data_aggregated["event_date"] - data_aggregated["previous_fight_date"]
        ).dt.days

        for column in data_processor.aggregated_fields:
            data_aggregated[column] = data_aggregated[column].astype(float)
            data_aggregated[column] = data_aggregated.groupby("fighter_id")[
                column
            ].cumsum()

        data_aggregated["total_time"] = data_aggregated.groupby("fighter_id")[
            "total_time"
        ].cumsum()
        data_aggregated["weighted_total_time"] = data_aggregated["total_time"]
        data_aggregated["weighted_num_fight"] = data_aggregated["num_fight"]

        return data_aggregated


class WeightedDataAggregator(DataAggregator):
    def __init__(self, alpha=-0.0004):
        self.alpha = alpha

    def aggregate_data(self, data_processor: DataProcessor) -> pd.DataFrame:
        logger.info(f"Fields to be aggregated: {data_processor.aggregated_fields}")

        data = data_processor.data[
            ["fight_id", "fighter_id", "event_date", "total_time"]
            + data_processor.aggregated_fields
        ]

        data_merged = data.drop(
            columns=data_processor.aggregated_fields + ["total_time"]
        ).merge(
            data,
            on="fighter_id",
            suffixes=("", "_prev"),
        )

        # Now only preserve combinations where the previous fight is
        # before the current fight.
        data_merged = data_merged[
            data_merged["event_date_prev"] < data_merged["event_date"]
        ]

        # Compute the distance from previous to current
        data_merged["w"] = np.exp(
            self.alpha
            * (data_merged["event_date"] - data_merged["event_date_prev"]).dt.days
        )

        for column in data_processor.aggregated_fields:
            data_merged[column] = data_merged[column].astype(float)
            data_merged[column] = data_merged[column] * data_merged["w"]

        data_merged["num_fight"] = 1  # This will sum up to the total number of fights.
        data_merged["total_time"] = data_merged["total_time"]
        data_merged["weighted_num_fight"] = data_merged["num_fight"] * data_merged["w"]
        data_merged["weighted_total_time"] = (
            data_merged["total_time"] * data_merged["w"]
        )

        data_aggregated = (
            data_merged.drop(columns=["event_date", "event_date_prev", "fight_id_prev"])
            .groupby(["fighter_id", "fight_id"])
            .sum()
            .reset_index()
        )

        for (
            column
        ) in (
            data_processor.aggregated_fields
        ):  # + ["weighted_num_fight", "weighted_total_time"]: # Not clear If I should normalize these two quantities.
            data_aggregated[column] = data_aggregated[column] / data_aggregated["w"]

        data_aggregated = data_processor.data.drop(
            columns=data_processor.aggregated_fields
        ).merge(
            data_aggregated.drop(columns=["w"]),
            on=["fighter_id", "fight_id"],
        )

        data_aggregated["previous_fight_date"] = data_aggregated.groupby("fighter_id")[
            "event_date"
        ].shift(1)
        data_aggregated["time_since_last_fight"] = (
            data_aggregated["event_date"] - data_aggregated["previous_fight_date"]
        ).dt.days

        return data_aggregated
