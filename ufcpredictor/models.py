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
    from typing import List


class FighterNet(nn.Module):
    """
    A neural network model designed to predict the outcome of a fight based on a single
    fighter's characteristics.

    The model takes into account the characteristics of the fighter and the odds of the
    fight. It can be used to make predictions on the outcome of a fight and to
    calculate the benefit of a bet.
    """

    mlflow_params: List[str] = [
        "dropout_prob",
    ]

    def __init__(self, input_size: int, dropout_prob: float = 0.0) -> None:
        """
        Initialize the FighterNet model with the given input size and dropout
        probability.

        Args:
            input_size: The size of the input to the model.
            dropout_prob: The probability of dropout.
        """
        super(FighterNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 127)

        # Use the global dropout probability
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.dropout4 = nn.Dropout(p=dropout_prob)
        self.dropout5 = nn.Dropout(p=dropout_prob)

        self.dropout_prob = dropout_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the output of the model given the input tensor x.

        Args:
            x: The input tensor to the model.

        Returns:
            The output of the model.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after the first ReLU
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after the second ReLU
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)  # Apply dropout after the third ReLU
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)  # Apply dropout after the fourth ReLU
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)  # Apply dropout after the fifth ReLU

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
        "dropout_prob",
    ]

    def __init__(self, input_size: int, dropout_prob: float = 0.0) -> None:
        """
        Initialize the SymmetricFightNet model with the given input size and dropout
        probability.

        Args:
            input_size: The size of the input to the model.
            dropout_prob: The probability of dropout.
        """
        super(SymmetricFightNet, self).__init__()
        self.fighter_net = FighterNet(input_size=input_size, dropout_prob=dropout_prob)

        self.fc1 = nn.Linear(256, 512)
        # self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)

        # Use the global dropout probability
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.dropout4 = nn.Dropout(p=dropout_prob)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout_prob = dropout_prob

    def forward(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        odds1: torch.Tensor,
        odds2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the output of the SymmetricFightNet model.

        Args:
            X1: The input tensor for the first fighter.
            X2: The input tensor for the second fighter.
            odds1: The odds tensor for the first fighter.
            odds2: The odds tensor for the second fighter.

        Returns:
            The output of the SymmetricFightNet model.
        """
        out1 = self.fighter_net(X1)
        out2 = self.fighter_net(X2)

        out1 = torch.cat((out1, odds1), dim=1)
        out2 = torch.cat((out2, odds2), dim=1)

        x = torch.cat((out1 - out2, out2 - out1), dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after the first ReLU
        # x = self.relu(self.fc2(x))
        # x = self.dropout2(x)  # Apply dropout after the second ReLU
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)  # Apply dropout after the third ReLU
        x = self.relu(self.fc4(x))
        x = self.dropout4(x)  # Apply dropout after the fourth ReLU
        x = self.sigmoid(self.fc5(x))
        return x
