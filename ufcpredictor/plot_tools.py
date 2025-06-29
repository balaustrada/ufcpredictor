"""
This module provides tools for plotting and visualizing predictions made by UFC
predictor models.
"""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover
    from typing import List, Optional, Sequence, Tuple

    import torch
    from numpy.typing import NDArray
    from torch import nn

    from ufcpredictor.datasets import BasicDataset, ForecastDataset

logger = logging.getLogger(__name__)


class PredictionPlots:
    """
    Provides tools for visualizing and analyzing the predictions made by UFC predictor
    models.

    This class contains methods for displaying the prediction details of a fight,
    including the prediction, shift, odds, and correctness. It also calculates and
    displays the total invested, earnings, number of bets, and number of fights.
    Additionally, it can show a plot of the benefit of the model over time.
    """

    @staticmethod
    def show_fight_prediction_detail(
        model: nn.Module,
        data: Tuple[
            Sequence[torch.Tensor],
            torch.Tensor,
            Sequence[torch.Tensor],
            NDArray[np.str_],
            NDArray[np.str_],
        ],
        print_info: bool = False,
        show_plot: bool = False,
        ax: Optional[plt.Axes] = None,
        device: str | torch.device = "cpu",
    ) -> List[Tuple[float, int, float, float, bool, float, float]]:
        """
        Shows the prediction detail of a fight and the benefit of the model.

        It prints the prediction, shift, odd1, odd2, and correct for each fight.
        It also prints the total invested, earnings, number of bets and number
            of fights.
        Finally, it prints the benefit of the model as a percentage.

        Args:
            model : The model to use to make predictions.
            data : The data to use to make predictions. It should contain the fighter
                and opponent data, the label, the odds and the names of the fighters.
            print_info : If True, print the prediction, shift, odd1, odd2, and correct
                for each fight. If False, do not print anything.
            show_plot : If True, show a plot of the benefit of the model over time.
            ax : The axes to use to show the plot. If None, a new figure will be
                created.
        """
        X, Y, odds, fighter_names, opponent_names = data
        X = [Xi.to(device) for Xi in X]
        Y = Y.to(device)
        odds = [od.to(device) for od in odds]
        model = model.to(device)

        stats = []

        with torch.no_grad():
            predictions_1 = model(*X, *odds).detach().cpu().numpy().reshape(-1)
            predictions_2 = 1 - model(
                *X, *odds, invert=True
            ).detach().cpu().numpy().reshape(-1)

            predictions = 0.5 * (predictions_1 + predictions_2)
            shifts = abs(predictions_2 - predictions_1)

            corrects = predictions.round() == Y.cpu().numpy()

            invested = 0.0
            earnings = 0.0
            fights = 0
            nbets = 0

            invest_progress = []
            earning_progress = []

            odds1, odds2 = odds

            for fighter, opponent, prediction, shift, odd1, odd2, correct, Yi in zip(
                fighter_names,
                opponent_names,
                predictions,
                shifts,
                odds1.cpu().numpy().reshape(-1),
                odds2.cpu().numpy().reshape(-1),
                corrects,
                Y.cpu().numpy().tolist(),
            ):
                prediction = round(float(prediction), 3)
                shift = round(float(shift), 3)

                if prediction > 0.5:
                    bet = 2 * 10 * (prediction - 0.5)
                    earning = odd2 * bet if correct else 0
                else:
                    bet = 2 * 10 * (0.5 - prediction)
                    earning = odd1 * bet if correct else 0

                invested += bet
                earnings += earning

                invest_progress.append(bet)
                earning_progress.append(earning)

                fights += 1
                nbets += 1

                if print_info:  # pragma: no cover
                    print(fighter, "vs", opponent)
                    print(odd1, "vs", odd2)
                    print(prediction, shift)

                    print(f"bet: {bet:.2f}, earn: {earning:.2f}")
                    print(
                        f"invested: {invested:.2f}, earnings: {earnings:.2f}, nbets: {nbets}, fights: {fights}"
                    )
                    print(f"benefits: {(earnings/invested-1)*100:.2f}%")

                    print()
                stats.append(
                    (
                        prediction,
                        Yi,
                        odd1,
                        odd2,
                        correct,
                        bet,
                        earning,
                    )
                )

        if show_plot:
            if ax is None:  # pragma: no cover
                fig, ax = plt.subplots()

            ax.plot(
                np.cumsum(invest_progress),
                (np.cumsum(earning_progress) - np.cumsum(invest_progress))
                / np.cumsum(invest_progress)
                * 100,
            )
            ax.axhline(0, c="k")

        return stats

    @staticmethod
    def show_fight_prediction_detail_from_dataset(
        model: nn.Module,
        dataset: BasicDataset,
        fight_ids: Optional[List[str]] = None,
        print_info: bool = False,
        show_plot: bool = False,
        ax: Optional[plt.Axes] = None,
        device: str | torch.device = "cpu",
    ) -> List[Tuple[float, int, float, float, bool, float, float, str]]:
        """
        Shows the prediction detail of a fight and the benefit of the model.

        It uses the dataset to get the data for the specified fight ids.
        It then calls show_fight_prediction_detail with the model and the data.

        Args:
            model : The model to use to make predictions.
            dataset : The dataset to use to get the data.
            fight_ids : The id of the fight to use. If None, it will use all the data
                in the dataset.
            print_info : If True, print the prediction, shift, odd1, odd2, and correct
                for each fight. If False, do not print anything.
            show_plot : If True, show a plot of the benefit of the model over time.
            ax : The axes to use to show the plot. If None, a new figure will be
                created.
        """
        X, Y, odds, fighter_names, opponent_names = dataset.get_fight_data_from_ids(
            fight_ids
        )

        stats = PredictionPlots.show_fight_prediction_detail(
            model,
            (X, Y, odds, fighter_names, opponent_names),
            print_info,
            show_plot,
            ax,
            device=device,
        )

        return [
            fight_stats + (fight_id,)
            for fight_stats, fight_id in zip(
                stats, dataset.fight_data["fight_id"].values
            )
        ]

    @staticmethod
    def plot_single_prediction(
        model: nn.Module,
        dataset: ForecastDataset,
        fighter_name: str,
        opponent_name: str,
        fight_parameters_values: List[float],
        event_date: str | datetime.date,
        odds1: int,
        odds2: int,
        ax: Optional[plt.Axes] = None,
        parse_id: bool = False,
    ) -> None:
        """
        Plots the prediction for a single fight.

        Args:
            model : The model to use to make predictions.
            dataset : The dataset to use to get the data.
            fighter_name : The name of the first fighter.
            opponent_name : The name of the second fighter.
            event_date : The date of the fight.
            odds1 : The odds for the first fighter (decimal).
            odds2 : The odds for the second fighter (decimal).
            ax : The axes to use to show the plot. If None, a new figure will be
                created.
            parse_id : If True, the id of the fighters is parsed instead of the name.
        """
        p1, p2 = dataset.get_single_forecast_prediction(
            fighter_name,
            opponent_name,
            event_date,
            odds1,
            odds2,
            model,
            fight_parameters_values,
            parse_id,
        )

        if parse_id:
            names = dataset.data_processor.data["fighter_name"]
            ids = dataset.data_processor.data["fighter_id"]

            display_fighter_name = names[ids == fighter_name].values[0]
            display_opponent_name = names[ids == opponent_name].values[0]
        else:  # pragma: no cover
            display_fighter_name = fighter_name
            display_opponent_name = opponent_name

        if ax is None:  # pragma: no cover
            fig, ax = plt.subplots()

        prediction = ((p1 + p2) - 1) * 100
        shift = np.abs(p1 - p2) * 2 * 100

        red = "tab:red"
        blue = "tab:blue"

        color = red if prediction <= 0 else blue

        ax.barh(
            0,
            prediction,
            xerr=shift,
            color=color,
            capsize=5,
            height=0.7,
        )
        ax.set_ylim(-1, 1)
        ax.set_xlim(-100, 100)

        ticks = np.arange(-100, 101, 25, dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([abs(tick) for tick in ticks])

        ax.text(
            ax.get_xlim()[0],
            ax.get_ylim()[1] * 1.3,
            display_fighter_name,
            color=red,
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        ax.text(
            ax.get_xlim()[1],
            ax.get_ylim()[1] * 1.3,
            display_opponent_name,
            color=blue,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        ax.axvline(x=0, color="lightgray", lw=1)
        ax.text(
            prediction,
            ax.get_ylim()[1] * 0.5,
            f"{abs(prediction):.2f}±{shift:.2f}",
            color=color,
            ha="center",  # "left" if prediction > 0 else "right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
