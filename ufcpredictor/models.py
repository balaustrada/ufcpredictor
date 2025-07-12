"""
This module contains neural network models designed to predict the outcome of UFC
fights.

The models take into account various characteristics of the fighters and the odds
of the fights, and can be used to make predictions on the outcome of a fight and
to calculate the benefit of a bet.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from ufcpredictor.datasets import DatasetWithTimeEvolution

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any, Dict, List, Optional, Tuple


class Model(nn.Module, ABC):
    """
    Base class for all models in the ufcpredictor package.

    This class inherits from torch.(Model) and serves as a base class for all
    models in the ufcpredictor package. It can be extended to create custom models
    for predicting fight outcomes.
    """

    # shouldn't be instantiated directly, should raise error
    def __init__(self) -> None:
        """
        Initialize the Model class.
        """
        super(Model, self).__init__()


class FighterNet(Model):
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
        fighter_fight_statistics: list[str],
        dropout_prob: float = 0.0,
        network_shape: List[int] = [128, 256, 512, 256, 127],
    ) -> None:
        """
        Initialize the FighterNet model with the given input size and dropout
        probability.

        Args:
            fighter_fight_statistics: Statistics of the fighters for the fight.
            fight_parameters: Fight parameters for the fight, such as weight class.
            network_shape: Shape of the network layers (except input layer).
        """
        super(FighterNet, self).__init__()

        input_size = len(fighter_fight_statistics)
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


class SymmetricFightNet(Model):
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
        "network_shape",
        "fighter_network_shape",
    ]

    def __init__(
        self,
        fighter_fight_statistics: list[str],
        fight_parameters: list[str],
        dropout_prob: float = 0.0,
        network_shape: List[int] = [512, 128, 64, 1],
        fighter_network_shape: Optional[List[int]] = None,
    ):
        """
        Initialize the SymmetricFightNet model with the given input size and dropout
        probability.

        Args:
            fighter_fight_statistics: Statistics of the fighters for the fight.
            fight_parameters: Fight parameters for the fight, such as weight class.
            network_shape: Shape of the network layers (except input layer).
            fighter_network_shape: Shape of the network layers for the fighter
                network (except input layer).
        """
        super(SymmetricFightNet, self).__init__()

        fighter_network_args: Dict[str, Any] = {
            "fighter_fight_statistics": fighter_fight_statistics,
            "dropout_prob": dropout_prob,
        }
        if fighter_network_shape is not None:  # pragma: no cover
            fighter_network_args["network_shape"] = fighter_network_shape

        self.fighter_net = FighterNet(**fighter_network_args)
        self.fighter_network_shape = self.fighter_net.network_shape

        self.network_shape = [
            self.fighter_network_shape[-1] * 2 + 2 + len(fight_parameters)
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
        invert: bool = False,
    ) -> torch.Tensor:
        """
        Compute the output of the SymmetricFightNet model.

        Args:
            X1: The input tensor for the first fighter.
            X2: The input tensor for the second fighter.
            X3: The input tensor for the fight features.
            odds1: The odds tensor for the first fighter.
            odds2: The odds tensor for the second fighter.
            invert: If True, invert the input order (not used in this model).

        Returns:
            The output of the SymmetricFightNet model.
        """
        if invert:  # pragma: no cover
            X1, X2 = X2, X1
            odds1, odds2 = odds2, odds1

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


class SimpleFightNet(Model):
    """
    A neural network model designed to predict the outcome of a fight between two
    fighters.

    The model takes into account the characteristics of both fighters and the odds of
    the fight. It combines the features of both fighters as an input to the model.

    The model can be used to make predictions on the outcome of a fight and to calculate
    the benefit of a bet.
    """

    mlflow_params: List[str] = ["dropout_prob", "network_shape"]

    state_size = 5

    def __init__(
        self,
        fighter_fight_statistics: list[str],
        fight_parameters: list[str],
        dropout_prob: float = 0.0,
        network_shape: List[int] = [1024, 512, 256, 128, 64, 1],
    ):
        """
        Initialize the SimpleFightNet model with the given input size and dropout
        probability.

        Args:
            fighter_fight_statistics: Statistics of the fighters for the fight.
            fight_parameters: Fight parameters for the fight, such as weight class.
            dropout_prob: The probability of dropout.
            network_shape: Shape of the network layers (except input layer).
        """
        super().__init__()

        input_size = (
            2 * len(fighter_fight_statistics)
            + len(fight_parameters)
            + 2  # odds for the fighter and opponent
        )
        self.network_shape = [
            input_size,
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
            [nn.Dropout(p=dropout_prob) for _ in range(len(self.network_shape) - 1)]
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
        invert: bool = False,
    ) -> torch.Tensor:
        """
        Compute the output of the SimpleFightNet model.

        Args:
            X1: Fighter 1 stats tensor of shape (batch_size, _).
            X2: Fighter 2 stats tensor of shape (batch_size, _).
            X3: Fight parameters tensor of shape (batch_size, _).
            odds1: Odds for Fighter 1 tensor of shape (batch_size, 1).
            odds2: Odds for Fighter 2 tensor of shape (batch_size, 1).
            invert: If True, invert the input order. Used in non-symmetric
            models to generate stronger predictions.

        Returns:
            The output of the SimpleFightNet model.
        """
        if invert:  # pragma: no cover
            X1, X2 = X2, X1
            odds1, odds2 = odds2, odds1

        x = torch.cat((X1, X2, X3, odds1, odds2), dim=1)

        for fc, dropout in zip(self.fcs[:-1], self.dropouts):
            x = self.relu(fc(x))
            x = dropout(x)

        x = self.fcs[-1](x)
        x = self.sigmoid(x)
        return x


class SimpleFightNetWithTimeEvolution(Model):
    """
    A neural network model designed to predict the outcome of a fight between
    two fighters.

    This model extends the functionality of SimpleFightNet by incorporating a
    FighterStateEvolver, which allows to evolve the fighters' states over
    time.
    """

    def __init__(
        self,
        fighter_fight_statistics: list[str],
        fight_parameters: list[str],
        dropout_prob: float = 0.0,
        network_shape: List[int] = [1024, 512, 256, 128, 64, 1],
        fighter_transformer_kwargs: Dict = dict(),
        state_size: int = 8,
        num_past_fights: int = DatasetWithTimeEvolution.num_past_fights,
    ):
        """
        Initialize the SimpleFightNet model with the given input size and dropout
        probability.

        Args:
            fighter_fight_statistics: Statistics of the fighters for the fight.
            fight_parameters: Fight parameters for the fight, such as weight class,
            dropout_prob: The probability of dropout.
            network_shape: Shape of the network layers (except input layer).
            fighter_transformer_kwargs: Keyword arguments for the
                FighterStateEvolver model.
            state_size: The size of the state tensor.
            num_past_fights: The number of past fights to consider for the
                FighterStateEvolver model. Defaults to the value from
                DatasetWithTimeEvolution.num_past_fights.
        """
        super().__init__()

        self.state_size = state_size

        self.network_shape = [
            2 * len(fighter_fight_statistics)
            + 2 * len(fight_parameters)
            + 2 * state_size,
        ] + network_shape

        self.num_past_fights = num_past_fights

        self.evolver = FighterStateEvolver(**fighter_transformer_kwargs)

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
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout_prob = dropout_prob

    def forward(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        X3: torch.Tensor,
        fighter_prev_data: torch.Tensor,
        opponent_prev_data: torch.Tensor,
        fighter_prev_opponents_data: torch.Tensor,
        opponent_prev_opponents_data: torch.Tensor,
        odds1: torch.Tensor,
        odds2: torch.Tensor,
        invert: bool = False,
    ) -> torch.Tensor:
        """
        Compute the output of the SimpleFightNet model.

        Args:
            X1: Fighter 1 stats tensor of shape (batch_size, _).
            X2: Fighter 2 stats tensor of shape (batch_size, _).
            X3: Fight parameters tensor of shape (batch_size, _).
            fighter_prev_data: Fighter 1 previous fights data tensor of shape
                (batch_size, num_past_fights, _).
            opponents_prev_data: Fighter2 previous fights data tensor of shape
                (batch_size, num_past_fights, _).
            fighter_prev_opponents_data: Fighter 1 previous opponents data tensor of
                shape (batch_size, num_past_fights, _).
            opponent_prev_opponents_data: Fighter 2 previous opponents data tensor of
                shape (batch_size, num_past_fights, _).
            odds1: Odds for Fighter 1 tensor of shape (batch_size, 1).
            odds2: Odds for Fighter 2 tensor of shape (batch_size, 1).
            invert: If True, invert the input order. Used in non-symmetric
            models to generate stronger predictions.

        Returns:
            The output of the SimpleFightNet model.
        """
        if invert:  # pragma: no cover
            X1, X2 = X2, X1
            odds1, odds2 = odds2, odds1
            (
                fighter_prev_data,
                opponent_prev_data,
                fighter_prev_opponents_data,
                opponent_prev_opponents_data,
            ) = (
                opponent_prev_data,
                fighter_prev_data,
                opponent_prev_opponents_data,
                fighter_prev_opponents_data,
            )

        # odds1 = odds1 / odds1
        # odds2 = odds2 / odds2

        # We start with the initial states of both fighters, to evolve them
        S1 = torch.zeros(X1.shape[0], self.state_size).to(X1.device)
        S2 = torch.zeros(X2.shape[0], self.state_size).to(X1.device)

        for i in range(self.num_past_fights):
            ff = fighter_prev_data[:, i, :]
            of = opponent_prev_data[:, i, :]
            fo = fighter_prev_opponents_data[:, i, :]
            oo = opponent_prev_opponents_data[:, i, :]

            S1, _ = self.evolver(
                S1,
                fo[:, : self.state_size],
                ff[:, self.state_size : -len(self.evolver.fight_parameters)],
                fo[:, self.state_size : -len(self.evolver.fight_parameters)],
                ff[:, -len(self.evolver.fight_parameters) :],
            )

            # @TODO: It is inconsistent using ff and then oo (for the last term)
            # It should be equivalent, but the test fails...
            S2, _ = self.evolver(
                S2,
                oo[:, : self.state_size],
                of[:, self.state_size : -len(self.evolver.fight_parameters)],
                oo[:, self.state_size : -len(self.evolver.fight_parameters)],
                oo[:, -len(self.evolver.fight_parameters) :],
            )

        # x = torch.cat((X1, X2, X3, odds1, odds2, S1-S2, S2-S1), dim=1)
        x = torch.cat((X1, X2, X3, odds1, odds2, S1 - S2, S2 - S1), dim=1)

        for fc, dropout in zip(self.fcs[:-1], self.dropouts):
            x = self.relu(fc(x))
            x = dropout(x)

        x = self.fcs[-1](x)
        x = self.sigmoid(x)
        return x


class FighterStateEvolver(Model):
    """
    A neural network model designed to predict the evolution of a fighter's state after a fight.

    The model takes into account the current state of both fighters, their
    fight statistics, and the fight parameters.
    """

    def __init__(
        self,
        state_size: int,
        fighter_fight_statistics: list[str],
        fight_parameters: list[str],
        network_shape: List[int],
        dropout: float = 0.1,
    ):
        """
        Initialize the FighterStateEvolver model.

        Args:
            state_size: Size of the fighters state tensor.
            fighter_fight_statistics: Statistics of the fighters for the fight
            fight_parameters: Fight parameters for the fight, such as weight class,
            network_shape: List specifying the sizes of hidden
                layers.
            dropout (float): Dropout probability.
        """
        super().__init__()

        # Calculate the input dimension
        input_dim = (
            2 * state_size + 2 * len(fighter_fight_statistics) + len(fight_parameters)
        )

        self.state_size = state_size
        self.fighter_fight_statistics = fighter_fight_statistics
        self.fight_parameters = fight_parameters

        # Create the layers of the feedforward network
        layers: List[nn.Module] = []
        previous_dim = input_dim
        for layer_size in network_shape:
            layers.append(nn.Linear(previous_dim, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = layer_size

        self.feedforward = nn.Sequential(*layers)

        # Output projections for X1 and X2
        self.output_X1 = nn.Linear(previous_dim, state_size)
        self.output_X2 = nn.Linear(previous_dim, state_size)

    def forward(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
        m: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the FighterStateEvolver model.

        Args:
            X1 (tensor): Fighter 1 state tensor of shape (batch_size, state_size).
            X2 (tensor): Fighter 2 state tensor of shape (batch_size, state_size).
            s1 (tensor): Fighter 1 fight statistics tensor of shape
                (batch_size, fighter_fight_statistics_size).
            s2 (tensor): Fighter 2 fight statistics tensor of shape
                (batch_size, fighter_fight_statistics_size).
            m (tensor): Match fight statistics tensor of shape
                (batch_size, fight_parameters_size).

        Returns:
            X1_new (tensor): Fighter 1 new state tensor of shape
                (batch_size, state_size).
            X2_new (tensor): Fighter 2 new state tensor of shape
                (batch_size, state_size).
        """

        # Concatenate all inputs
        combined_input = torch.cat([X1, X2, s1, s2, m], dim=-1)

        # Pass through the feedforward network
        hidden_output = self.feedforward(combined_input)

        # Compute outputs for X1 and X2
        X1_new = self.output_X1(hidden_output)
        X2_new = self.output_X2(hidden_output)

        return X1_new, X2_new
