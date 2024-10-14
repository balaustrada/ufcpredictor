from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt

import gradio as gr
import numpy as np
import torch

from ufcpredictor.data_processor import WOSRDataProcessor as DataProcessor
from ufcpredictor.datasets import ForecastDataset
from ufcpredictor.models import SymmetricFightNet
from ufcpredictor.utils import convert_odds_to_decimal

if TYPE_CHECKING:
    from typing import Optional


logger = logging.getLogger(__name__)

X_set = [
    "clinch_strikes_att_opponent_per_minute",
    "time_since_last_fight",
    "total_strikes_succ_opponent_per_minute",
    "takedown_succ_per_minute",
    "KO_opponent_per_minute",
    "takedown_att_per_minute",
    "takedown_succ_opponent_per_minute",
    "win_opponent_per_fight",
    "head_strikes_succ_opponent_per_minute",
    "clinch_strikes_succ_opponent_per_minute",
    "ground_strikes_succ_opponent_per_minute",
    "ground_strikes_att_per_minute",
    "head_strikes_succ_per_minute",
    "age",
    "distance_strikes_succ_per_minute",
    "body_strikes_succ_per_minute",
    "strikes_succ_opponent_per_minute",
    "leg_strikes_att_per_minute",
    "reversals_opponent_per_minute",
    "strikes_succ_per_minute",
    "distance_strikes_att_opponent_per_minute",
    "Sub_opponent_per_fight",
    "distance_strikes_att_per_minute",
    "knockdowns_per_minute",
    "OSR",
]


def predict(a, b, c, d):
    return a, b, c, d


def greet(name):
    return "Hello " + name + "!"

def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = get_args()

    logging.basicConfig(
        stream=sys.stdout,
        level=args.log_level,
        format="%(levelname)s:%(message)s",
    )

    logger.info("Loading data...")
    data_processor = DataProcessor(args.data_folder)
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
    model.load_state_dict(
        torch.load(args.model_path)
    )

    fighter_names = list(
        data_processor.scraper.fighter_scraper.data["fighter_name"].values
    )

    with gr.Blocks() as demo:
        event_date = gr.DateTime(label="Event Date", include_time=False, value=datetime.now().strftime("%Y-%m-%d"))
        fighter_name = gr.Dropdown(
            label="Fighter Name",
            choices=fighter_names,
            interactive=True,
        )
        opponent_name = gr.Dropdown(
            label="Opponent Name",
            choices=fighter_names,
            interactive=True,
        )
        odds1 = gr.Number(label="Odds 1", value=100)
        odds2 = gr.Number(label="Odds 2", value=100)

        btn = gr.Button("Predict")

        output = gr.Plot(label="")
        # output = gr.Text(label="Prediction Output")


        def get_forecast_single_prediction(fighter_name, opponent_name, event_date, odds1, odds2):
            event_date = [datetime.fromtimestamp(event_date).strftime("%Y-%m-%d"),]
            odds1 = convert_odds_to_decimal([odds1,])
            odds2 = convert_odds_to_decimal([odds2,])

            #print(fighter_name, opponent_name, event_date, odds1, odds2)
            p1, p2 =  dataset.get_forecast_prediction(
                [fighter_name,],
                [opponent_name,],
                event_date,
                odds1,
                odds2,
                model=model
            )
            prediction = (p1 + p2)[0][0] - 1 # Prediction between -1 and 1
            shift = np.abs(p1 - p2)[0][0] * 2

            prediction *=100
            shift*=100

            fig, ax = plt.subplots(figsize=(6.4, 1.7))
            red = "tab:red"
            blue = "tab:blue"

            color = red if prediction < 0 else blue

            ax.barh(
                0,
                prediction,
                xerr=shift,
                color=color,
                capsize=5,
                height=0.7,
            )
            ax.set_ylim([-1, 1])

            ax.set_xlim([-100, 100])
            ax_right = ax.twinx()
            ax_right.set_yticks([])
            ax.set_yticks([])

            ticks = np.arange(-100, 101, 25, dtype=int)
            ax.set_xticks(ticks)
            ax.set_xticklabels([abs(tick) for tick in ticks])

            ax.text(
                ax.get_xlim()[0],
                ax.get_ylim()[1]*1.3,
                fighter_name,
                color=red,
                ha="left",
                va="center",
                fontsize=12,
                fontweight='bold',
            )

            ax.text(
                ax.get_xlim()[1],
                ax.get_ylim()[1]*1.3,
                opponent_name,
                color=blue,
                ha="right",
                va="center",
                fontsize=12,
                fontweight='bold',
            )

            ax.axvline(x=0, color='lightgray', lw=1)
            ax.text(
                prediction*1.2,
                ax.get_ylim()[1]*0.5,
                f"{abs(prediction):.2f}±{shift:.2f}",
                color=color,
                ha='left' if prediction > 0 else "right",
                va="center",
                fontsize=11,
                fontweight='bold',
            )
            fig.subplots_adjust(left=0.1, right=0.9, top=0.75, bottom=0.2)  # Adjust margins as needed

    
            return fig
            #return f"{prediction[0][0]:.3f} ({shift[0][0]:.3f})"
        
        btn.click(
            get_forecast_single_prediction,
            inputs=[fighter_name, opponent_name, event_date, odds1, odds2], 
            outputs=output
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

if __name__ == "__main__":
    main()
