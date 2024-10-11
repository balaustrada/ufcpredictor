from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, Callable, List, Optional, Set, Tuple
    from torch import nn
    from ufcpredictor.datasets import Dataset

logger = logging.getLogger(__name__)


class PredictionPlots:
    @staticmethod
    def show_fight_prediction_detail(
        model: nn.Module,
        data: Tuple[torch.Tensor],
        print_info: bool = False,
        show_plot: bool = False,
        ax: Optional[plt.Axes] = None,
    ):
        X1, X2, Y, odds1, odds2, fighter_names, opponent_names = data

        with torch.no_grad():
            predictions_1 = (
                model(X1, X2, odds1.reshape(-1, 1), odds2.reshape(-1, 1))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )
            predictions_2 = 1 - model(
                X2, X1, odds2.reshape(-1, 1), odds1.reshape(-1, 1)
            ).detach().cpu().numpy().reshape(-1)

            predictions = 0.5 * (predictions_1 + predictions_2)
            shifts = abs(predictions_2 - predictions_1)

            corrects = predictions.round() == Y.numpy()

            odds1 = odds1.numpy().reshape(-1)
            odds2 = odds2.numpy().reshape(-1)

            invested = 0
            earnings = 0
            fights = 0
            nbets = 0

            invest_progress = []
            earning_progress = []

            for fighter, opponent, prediction, shift, odd1, odd2, correct in zip(
                fighter_names,
                opponent_names,
                predictions,
                shifts,
                odds1,
                odds2,
                corrects,
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

                if print_info: # pragma: no cover
                    print(fighter, "vs", opponent)
                    print(odd1, "vs", odd2)
                    print(prediction, shift)

                    print(f"bet: {bet:.2f}, earn: {earning:.2f}")
                    print(
                        f"invested: {invested:.2f}, earnings: {earnings:.2f}, nbets: {nbets}, fights: {fights}"
                    )
                    print(f"benefits: {(earnings/invested-1)*100:.2f}%")

                    print()

        if show_plot:
            if ax is None: # pragma: no cover
                fig, ax = plt.subplots()

            ax.plot(
                np.cumsum(invest_progress),
                (np.cumsum(earning_progress) - np.cumsum(invest_progress))
                / np.cumsum(invest_progress)
                * 100,
            )
            ax.axhline(0, c="k")

    @staticmethod
    def show_fight_prediction_detail_from_dataset(
        model: nn.Module,
        dataset: Dataset,
        fight_ids: Optional[List[str]] = None,
        print_info: bool = False,
        show_plot: bool = False,
        ax: Optional[plt.Axes] = None,
    ):
        X1, X2, Y, odds1, odds2, fighter_names, opponent_names = (
            dataset.get_fight_data_from_ids(fight_ids)
        )

        PredictionPlots.show_fight_prediction_detail(
            model,
            (X1, X2, Y, odds1, odds2, fighter_names, opponent_names),
            print_info,
            show_plot,
            ax,
        )
