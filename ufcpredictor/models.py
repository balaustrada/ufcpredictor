"""
This module contains neural network models designed to predict the outcome of UFC 
fights.

The models take into account various characteristics of the fighters and the odds 
of the fights, and can be used to make predictions on the outcome of a fight and 
to calculate the benefit of a bet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:  # pragma: no cover
    from typing import List, Optional


class FighterNet(nn.Module):
    """
    A neural network model designed to predict the outcome of a fight based on a single
    fighter's characteristics.

    The model takes into account the characteristics of the fighter and the odds of the
    fight. It can be used to make predictions on the outcome of a fight and to
    calculate the benefit of a bet.
    """

    mlflow_params: List[str] = ["dropout_prob", "network_shape"]

    def __init__(
        self,
        input_size: int,
        dropout_prob: float = 0.0,
        network_shape: List[int] = [128, 256, 512, 256, 127],
    ) -> None:
        """
        Initialize the FighterNet model with the given input size and dropout
        probability.

        Args:
            input_size: The size of the input to the model.
            dropout_prob: The probability of dropout.
            network_shape: Shape of the network layers (except input layer).
        """
        super(FighterNet, self).__init__()
        self.network_shape = [input_size] + network_shape
        self.fcs = nn.ModuleList(
            [
                nn.Linear(input_, output)
                for input_, output in zip(
                    self.network_shape[:-1], self.network_shape[1:]
                )
            ]
        )
        self.dropouts = nn.ModuleList(
            [nn.Dropout(p=dropout_prob) for _ in range(len(self.network_shape) - 1)]
        )
        self.dropout_prob = dropout_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the output of the model given the input tensor x.

        Args:
            x: The input tensor to the model.

        Returns:
            The output of the model.
        """
        for fc, dropout in zip(self.fcs, self.dropouts):
            x = F.relu(fc(x))
            x = dropout(x)

        return x


class SymmetricFightNet(nn.Module):
    """
    A neural network model designed to predict the outcome of a fight between two
    fighters.

    The model takes into account the characteristics of both fighters and the odds of
    the fight. It uses a symmetric architecture to ensure that the model is fair and
    unbiased towards either fighter.

    The model can be used to make predictions on the outcome of a fight and to calculate
    the benefit of a bet.
    """

    mlflow_params: List[str] = [
        "dropout_prob", "network_shape", "fighter_network_shape"
    ]

    def __init__(
        self,
        input_size: int,
        input_size_f: int,
        dropout_prob: float = 0.0,
        network_shape: List[int] = [512, 128, 64, 1],
        fighter_network_shape: Optional[List[int]] = None,
    ) -> None:
        """
        Initialize the SymmetricFightNet model with the given input size and dropout
        probability.

        Args:
            input_size: The size of the input to the model.
            dropout_prob: The probability of dropout.
            network_shape: Shape of the network layers (except input layer).
            fighter_network_shape: Shape of the network layers for the fighter
                network (except input layer).
        """
        super(SymmetricFightNet, self).__init__()

        fighter_network_args = {
            "input_size": input_size,
            "dropout_prob": dropout_prob,
        }
        if fighter_network_shape is not None:
            fighter_network_args["network_shape"] = fighter_network_shape

        self.fighter_net = FighterNet(**fighter_network_args)
        self.fighter_network_shape = self.fighter_net.network_shape

        self.network_shape = [
            self.fighter_network_shape[-1] * 2 + 2 + input_size_f
        ] + network_shape

        self.fcs = nn.ModuleList(
            [
                nn.Linear(input_, output)
                for input_, output in zip(
                    self.network_shape[:-1], self.network_shape[1:]
                )
            ]
        )
        self.dropouts = nn.ModuleList(
            [
                nn.Dropout(p=dropout_prob)
                for _ in range(len(self.network_shape) - 1)  # This should be -2
            ]
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout_prob = dropout_prob

    def forward(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        X3: torch.Tensor,
        odds1: torch.Tensor,
        odds2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the output of the SymmetricFightNet model.

        Args:
            X1: The input tensor for the first fighter.
            X2: The input tensor for the second fighter.
            X3: The input tensor for the fight features.
            odds1: The odds tensor for the first fighter.
            odds2: The odds tensor for the second fighter.

        Returns:
            The output of the SymmetricFightNet model.
        """
        out1 = self.fighter_net(X1)
        out2 = self.fighter_net(X2)

        out1 = torch.cat((out1, odds1), dim=1)
        out2 = torch.cat((out2, odds2), dim=1)

        x = torch.cat((out1 - out2, out2 - out1, X3), dim=1)

        for fc, dropout in zip(self.fcs[:-1], self.dropouts):
            x = self.relu(fc(x))
            x = dropout(x)

        x = self.fcs[-1](x)
        x = self.sigmoid(x)
        return x
