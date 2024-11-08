from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import snapshot_download

from ufcpredictor.data_processor import DataProcessor
from ufcpredictor.extra_fields import SumFlexibleELOExtraField
from ufcpredictor.data_aggregator import WeightedDataAggregator
from ufcpredictor.datasets import ForecastDataset
from ufcpredictor.models import SymmetricFightNet
from ufcpredictor.utils import convert_odds_to_decimal
from ufcpredictor.plot_tools import PredictionPlots

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional


logger = logging.getLogger(__name__)

X_set = [
        "age",
        # "body_strikes_att_opponent_per_minute",
        # "body_strikes_att_per_minute",
        "body_strikes_succ_opponent_per_minute",
        "body_strikes_succ_per_minute",
        # "clinch_strikes_att_opponent_per_minute",
        # "clinch_strikes_att_per_minute",
        "clinch_strikes_succ_opponent_per_minute",
        "clinch_strikes_succ_per_minute",
        "ctrl_time_opponent_per_minute",
        "ctrl_time_per_minute",
        # "distance_strikes_att_opponent_per_minute",
        # "distance_strikes_att_per_minute",
        "distance_strikes_succ_opponent_per_minute",
        "distance_strikes_succ_per_minute",
        "fighter_height_cm",
        # "ground_strikes_att_opponent_per_minute",
        # "ground_strikes_att_per_minute",
        "ground_strikes_succ_opponent_per_minute",
        "ground_strikes_succ_per_minute",
        # "head_strikes_att_opponent_per_minute",
        # "head_strikes_att_per_minute",
        "head_strikes_succ_opponent_per_minute",
        "head_strikes_succ_per_minute",
        "knockdowns_opponent_per_minute",
        "knockdowns_per_minute",
        # "KO_opponent_per_fight",
        "KO_opponent_per_minute",
        "KO_per_fight",
        "KO_per_minute",
        # "leg_strikes_att_opponent_per_minute",
        # "leg_strikes_att_per_minute",
        "leg_strikes_succ_opponent_per_minute",
        "leg_strikes_succ_per_minute",
        "num_fight",
        # "reversals_opponent_per_minute",
        # "reversals_per_minute",
        # "strikes_att_opponent_per_minute",
        # "strikes_att_per_minute",
        "strikes_succ_opponent_per_minute",
        "strikes_succ_per_minute",
        # "Sub_opponent_per_fight",
        "Sub_opponent_per_minute",
        # "Sub_per_fight",
        "Sub_per_minute",
        "submission_att_opponent_per_minute",
        "submission_att_per_minute",
        # "takedown_att_opponent_per_minute",
        # "takedown_att_per_minute",
        "takedown_succ_opponent_per_minute",
        "takedown_succ_per_minute",
        "time_since_last_fight",
        # "total_strikes_att_opponent_per_minute",
        # "total_strikes_att_per_minute",
        "total_strikes_succ_opponent_per_minute",
        "total_strikes_succ_per_minute",
        "win_opponent_per_fight",
        "win_per_fight",
        "ELO",
]


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = get_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=args.log_level,
        format="%(levelname)s:%(message)s",
    )

    if args.download_dataset:  # pragma: no cover
        logger.info("Downloading dataset...")
        if "DATASET_TOKEN" not in os.environ:  # pragma: no cover
            raise ValueError(
                "'DATASET_TOKEN' must be set as an environmental variable"
                "to download the dataset. Please make sure you have access "
                "to the Hugging Face dataset."
            )
        snapshot_download(
            repo_id="balaustrada/UFCfightdata",
            allow_patterns=["*.csv"],
            token=os.environ["DATASET_TOKEN"],
            repo_type="dataset",
            local_dir=args.data_folder,
        )
    data_processor_kwargs = {
        "data_folder": args.data_folder,
        "data_aggregator": WeightedDataAggregator(),
        "extra_fields": [
            SumFlexibleELOExtraField(
                scaling_factor=0.5,
                K_factor = 40,
            )
        ],
    }
    logger.info("Loading data...")
    data_processor = DataProcessor(**data_processor_kwargs)
    data_processor.load_data()
    data_processor.aggregate_data()
    data_processor.add_per_minute_and_fight_stats()
    data_processor.normalize_data()

    logger.info("Creating dataset from loaded data...")
    dataset = ForecastDataset(
        data_processor=data_processor,
        X_set=X_set,
    )

    logger.info("Loading model...")
    model = SymmetricFightNet(
        input_size=len(X_set),
        dropout_prob=0.35,
    )
    model.load_state_dict(torch.load(args.model_path))

    fighter_names = sorted(
        list(data_processor.scraper.fighter_scraper.data["fighter_name"].values)
    )

    with gr.Blocks() as demo:
        event_date = gr.DateTime(
            label="Event Date",
            include_time=False,
            value=datetime.now().strftime("%Y-%m-%d"),
        )
        fighter_name = gr.Dropdown(
            label="Fighter Name",
            choices=fighter_names,
            value="Ilia Topuria",
            interactive=True,
        )
        opponent_name = gr.Dropdown(
            label="Opponent Name",
            choices=fighter_names,
            value="Max Holloway",
            interactive=True,
        )
        odds1 = gr.Number(label="Fighter odds", value=100)
        odds2 = gr.Number(label="Opponent odds", value=100)

        btn = gr.Button("Predict")

        output = gr.Plot(label="")
        # output = gr.Text(label="Prediction Output")

        def get_forecast_single_prediction(
            fighter_name: str, opponent_name: str, event_date: float, odds1: int, odds2: int
        ) -> plt.Figure:
            fig, ax = plt.subplots(figsize=(6.4, 1.7))

            PredictionPlots.plot_single_prediction(
                model=model,
                dataset=dataset,
                fighter_name=fighter_name,
                opponent_name=opponent_name,
                event_date=datetime.fromtimestamp(event_date).strftime("%Y-%m-%d"),
                odds1=convert_odds_to_decimal(
                    [
                        odds1,
                    ]
                )[0],
                odds2=convert_odds_to_decimal(
                    [
                        odds2,
                    ]
                )[0],
                ax=ax,
            )

            fig.subplots_adjust(
                top=0.75, bottom=0.2
            )  # Adjust margins as needed

            return fig

        btn.click(
            get_forecast_single_prediction,
            inputs=[fighter_name, opponent_name, event_date, odds1, odds2],
            outputs=output,
        )

    demo.launch(server_name=args.server_name, server_port=args.port)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"],
    )

    parser.add_argument(
        "--server-name",
        default="127.0.0.1",
        type=str,
    )

    parser.add_argument(
        "--download-dataset",
        action="store_true",
    )

    parser.add_argument(
        "--data-folder",
        type=Path,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
    )

    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    main()
