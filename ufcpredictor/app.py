from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

from ufcpredictor.data_aggregator import WeightedDataAggregator
from ufcpredictor.data_processor import DataProcessor
from ufcpredictor.datasets import BasicDataset, ForecastDataset
from ufcpredictor.data_enhancers import SumFlexibleELO
from ufcpredictor.loss_functions import BettingLoss
from ufcpredictor.models import SymmetricFightNet
from ufcpredictor.plot_tools import PredictionPlots
from ufcpredictor.trainer import Trainer
from ufcpredictor.utils import convert_odds_to_decimal

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


def get_model_parameters(
    X_set: List[str],
    Xf_set: List[str],
) -> tuple[
    torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau
]:
    seed = 30
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = SymmetricFightNet(
        input_size=len(X_set),
        input_size_f=len(Xf_set),
        dropout_prob=0.35,
        # fighter_network_shape=[256, 512, 1024, 512],
        # network_shape=[2048, 1024, 512, 128, 64, 1],
    )
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=2
    )
    return model, optimizer, scheduler


def get_data_parameters() -> Tuple[List[str], Dict[str, Any], int, int, int]:
    data_processor_kwargs = {
        "data_aggregator": WeightedDataAggregator(),
        "data_enhancers": [
            SumFlexibleELO(
                scaling_factor=0.5,
                K_factor=40,
            )
        ],
    }
    days_to_early_split = 1825
    min_num_fights = 4
    batch_size = 128

    # Xf_set = ["num_rounds", "weight"]
    Xf_set = []

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

    return (
        X_set,
        Xf_set,
        data_processor_kwargs,
        days_to_early_split,
        batch_size,
        min_num_fights,
    )


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

    (
        X_set,
        Xf_set,
        data_processor_kwargs,
        days_to_early_split,
        batch_size,
        min_num_fights,
    ) = get_data_parameters()

    data_processor_kwargs = {
        "data_folder": args.data_folder,
        **data_processor_kwargs,
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
        Xf_set=Xf_set,
    )

    logger.info("Training model (testing)...")
    model = train_model(
        data_processor=data_processor,
        X_set=X_set,
        Xf_set=Xf_set,
        days_to_early_split=days_to_early_split,
        batch_size=batch_size,
        min_num_fights=min_num_fights,
        test=True,
    )
    logger.info("Training model (final)...")
    model = train_model(
        data_processor=data_processor,
        X_set=X_set,
        Xf_set=Xf_set,
        days_to_early_split=days_to_early_split,
        batch_size=batch_size,
        min_num_fights=min_num_fights,
        test=False,
    )

    fighter_names = sorted(
        list(data_processor.scraper.fighter_scraper.data["fighter_name"].values)
    )

    with gr.Blocks() as demo:
        event_date = gr.DateTime(
            label="Event Date",
            include_time=False,
            value=datetime.now().strftime("%Y-%m-%d"),
        )

        fight_features = [gr.Number(label=label, value=0) for label in Xf_set]

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
            fighter_name: str,
            opponent_name: str,
            event_date: float,
            odds1: int,
            odds2: int,
            *fight_features: float,
        ) -> plt.Figure:
            fig, ax = plt.subplots(figsize=(6.4, 1.7))

            PredictionPlots.plot_single_prediction(
                model=model,
                dataset=dataset,
                fighter_name=fighter_name,
                opponent_name=opponent_name,
                fight_features=list(fight_features),
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

            fig.subplots_adjust(top=0.75, bottom=0.2)  # Adjust margins as needed

            plt.close(fig)

            return fig

        btn.click(
            get_forecast_single_prediction,
            inputs=[
                fighter_name,
                opponent_name,
                event_date,
                odds1,
                odds2,
                *fight_features,
            ],
            outputs=output,
        )

    demo.launch(server_name=args.server_name, server_port=args.port)


def train_model(
    data_processor: DataProcessor,
    X_set: List[str],
    Xf_set: List[str],
    days_to_early_split: int,
    batch_size: int,
    min_num_fights: int,
    test: bool,
) -> torch.nn.Module:
    invalid_fights = set(
        data_processor.data_aggregated[
            data_processor.data_aggregated["num_fight"] < min_num_fights
        ]["fight_id"]
    )  # The usual is 4
    # Early split date should be today - 5 years
    early_split_date = (
        datetime.now() - pd.Timedelta(days=days_to_early_split)
    ).strftime("%Y-%m-%d")
    early_train_fights = data_processor.data["fight_id"]
    train_fights = data_processor.data["fight_id"][
        data_processor.data["event_date"] >= early_split_date
    ]
    # Use last 3 months as test
    if test:
        test_split_date = (datetime.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        test_fights = data_processor.data["fight_id"][
            data_processor.data["event_date"] >= test_split_date
        ]
        test_fights = set(test_fights) - set(invalid_fights)
        train_fights = set(train_fights) - set(test_fights)

    train_fights = set(train_fights) - set(invalid_fights)
    early_train_fights = set(early_train_fights) - set(invalid_fights)

    early_train_dataset = BasicDataset(
        data_processor,
        early_train_fights,
        X_set=X_set,
        Xf_set=Xf_set,
    )

    train_dataset = BasicDataset(
        data_processor,
        train_fights,
        X_set=X_set,
        Xf_set=Xf_set,
    )

    if test:
        test_dataset = BasicDataset(
            data_processor,
            test_fights,
            X_set=X_set,
            Xf_set=Xf_set,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
    else:
        test_dataloader = None

    early_train_dataloader = torch.utils.data.DataLoader(
        early_train_dataset, batch_size=batch_size, shuffle=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    model, optimizer, scheduler = get_model_parameters(X_set, Xf_set)

    trainer = Trainer(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=BettingLoss(),
    )

    trainer.train(
        epochs=5,
        train_loader=early_train_dataloader,
        test_loader=test_dataloader,
    )

    trainer.train(epochs=30)

    return trainer.model


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
        "--port",
        type=int,
        default=7860,
    )

    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    main()
