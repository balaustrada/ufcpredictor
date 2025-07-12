from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import cast, TYPE_CHECKING

import gradio as gr
import matplotlib.pyplot as plt
import torch
from huggingface_hub import snapshot_download

from ufcpredictor import UFCPredictor
from ufcpredictor.datasets import ForecastDataset
from ufcpredictor.plot_tools import PredictionPlots
from ufcpredictor.utils import convert_odds_to_decimal

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional


logger = logging.getLogger(__name__)


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

    predictor = UFCPredictor(
        args.config_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    predictor.load_model()
    predictor.load_forecast_dataset()
    predictor.model.eval()

    # Only keep fighters with at least «minimum fight number» fights
    counts = predictor.data_processor.data_normalized["fighter_id"].value_counts()
    fighter_ids = counts[
        counts >= predictor.config["filters"]["minimum fight number"]
    ].index
    fighter_names = [
        predictor.data_processor.get_fighter_name(id_) for id_ in fighter_ids
    ]

    # There might be fighters with the same name, so we need to add the id to the name
    name_counts = Counter(fighter_names)
    show_names = []
    for name, id_ in zip(fighter_names, fighter_ids):
        if name_counts[name] > 1:
            show_names.append(f"{name} ({id_})")
        else:
            show_names.append(name)

    ##############################
    ## This block here is used to create the app
    ##############################

    with gr.Blocks() as demo:
        event_date = gr.DateTime(
            label="Event Date",
            include_time=False,
            value=datetime.now().strftime("%Y-%m-%d"),
        )

        fight_parameters = predictor.config.get("statistics", {}).get(
            "fight parameters", []
        )
        print(fight_parameters)

        fight_parameters_values = [
            gr.Number(label=label.replace("_", " "), value=0)
            for label in fight_parameters
        ]

        fighter_name = gr.Dropdown(
            label="Fighter Name",
            choices=show_names,
            value="Ilia Topuria",
            interactive=True,
        )
        opponent_name = gr.Dropdown(
            label="Opponent Name",
            choices=show_names,
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
            *fight_parameters_values: float,
        ) -> plt.Figure:
            fig, ax = plt.subplots(figsize=(6.4, 1.7))

            PredictionPlots.plot_single_prediction(
                model=predictor.model,
                dataset=cast(ForecastDataset, predictor.forecast_dataset),
                fighter_name=fighter_ids[show_names.index(fighter_name)],
                opponent_name=fighter_ids[show_names.index(opponent_name)],
                fight_parameters_values=list(fight_parameters_values),
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
                parse_id=True,
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
                *fight_parameters_values,
            ],
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
        "--config-path",
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
