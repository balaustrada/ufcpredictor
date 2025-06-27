"""
This module contains loss functions designed to train neural network models to predict
the outcome of UFC fights.

The loss functions take into account the predictions made by the model and the actual
outcomes of the fights, and are used to optimize the model's performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:  # pragma: no cover
    from typing import List

# import torch.nn.functional as F


class BettingLoss(nn.Module):

    mlflow_params: List[str] = [
        "max_bet",
    ]

    def __init__(self, max_bet: float = 10) -> None:
        """
        Initializes the BettingLoss instance.

        This function calls the constructor of the parent class and performs no
        other actions.
        """
        super(BettingLoss, self).__init__()
        self.max_bet = max_bet

    def get_bet(self, prediction: torch.Tensor | float) -> torch.Tensor | float:
        """
        Computes the bet for the given prediction.

        This function takes a prediction between 0 and 1 and returns the
        corresponding bet between 0 and 20. The bet is computed as the
        prediction times 2 times 10. This is just an approximation
        that seems to work well.

        Args:
            prediction: A tensor or float between 0 and 1 representing a
                prediction.

        Returns:
            A tensor or float between 0 and 20 representing the bet.
        """
        return prediction * 2 * self.max_bet

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        odds_1: torch.Tensor,
        odds_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the betting loss for the given predictions and targets.

        This function takes a tensor of predictions between 0 and 1, a tensor of
        targets (0 or 1), and two tensors of odds. It returns a tensor with the
        computed betting loss, which is the mean of the losses minus the earnings,
        this is the net profit.

        The betting loss returned is the negative profit.

        Args:
            predictions: A tensor of predictions between 0 and 1.
            targets: A tensor of targets (0 or 1).
            odds_1: A tensor of odds for fighter 1.
            odds_2: A tensor of odds for fighter 2.

        Returns:
            A tensor with the computed betting loss.
        """
        msk = torch.round(predictions) == targets

        return_fighter_1 = self.get_bet(0.5 - predictions) * odds_1
        return_fighter_2 = self.get_bet(predictions - 0.5) * odds_2

        losses = torch.where(
            torch.round(predictions) == 0,
            self.get_bet(0.5 - predictions),
            self.get_bet(predictions - 0.5),
        )

        earnings = torch.zeros_like(losses)
        earnings[msk & (targets == 0)] = return_fighter_1[msk & (targets == 0)]
        earnings[msk & (targets == 1)] = return_fighter_2[msk & (targets == 1)]

        return (losses - earnings).mean()

        # return F.binary_cross_entropy(predictions, targets.float())

        # # Soft approximation of rounding using sigmoid
        # sharpness = 3  # increase for sharper transition
        # soft_pred_1 = torch.sigmoid(sharpness * (0.5 - predictions))  # approximates prediction < 0.5
        # soft_pred_2 = torch.sigmoid(sharpness * (predictions - 0.5))  # approximates prediction > 0.5

        # # Betting amounts
        # bet_1 = self.get_bet(0.5 - predictions)
        # bet_2 = self.get_bet(predictions - 0.5)

        # # Total loss (total amount bet)
        # losses = bet_1 * soft_pred_1 + bet_2 * soft_pred_2

        # # Soft masks for correct prediction
        # soft_correct = torch.sigmoid(sharpness * (1.0 - torch.abs(predictions - targets)))

        # # Target masks (0 or 1)
        # target_is_0 = (1 - targets).float()
        # target_is_1 = targets.float()

        # # Earnings (softly weighted by match quality)
        # earnings_1 = bet_1 * odds_1 * soft_correct * target_is_0
        # earnings_2 = bet_2 * odds_2 * soft_correct * target_is_1

        # # Total earnings (note: no hard indexing!)
        # earnings = earnings_1 + earnings_2

        # return (losses - earnings).mean()
