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
    from typing import Any, Dict, List, Optional


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

        fighter_network_args: Dict[str, Any] = {
            "input_size": input_size,
            "dropout_prob": dropout_prob,
        }
        if fighter_network_shape is not None: # pragma: no cover
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

class SimpleFightNet(nn.Module):
    """
    A neural network model designed to predict the outcome of a fight between two
    fighters.

    The model takes into account the characteristics of both fighters and the odds of
    the fight. It combines the features of both fighters as an input to the model.

    The model can be used to make predictions on the outcome of a fight and to calculate
    the benefit of a bet.
    """

    mlflow_params: List[str] = [
        "dropout_prob",
        "network_shape"
    ]

    def __init__(
        self,
        input_size: int,
        dropout_prob: float = 0.0,
        network_shape: List[int] = [1024, 512, 256, 128, 64, 1],
        fighter_transformer_kwargs = Dict(),
    ):
        """
        Initialize the SimpleFightNet model with the given input size and dropout
        probability.

        Args:
            dropout_prob: The probability of dropout.
            network_shape: Shape of the network layers (except input layer).
        """
        super().__init__()

        self.network_shape = [input_size,] + network_shape

        self.transformer = FighterTransformer(**fighter_transformer_kwargs)

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
            ff_data,
            of_data,
            fo_data,
            oo_data,
    ) -> torch.Tensor:
        """
        Compute the output of the SimpleFightNet model.

        Args:
            X1: The input tensor for the first fighter.
            X2: The input tensor for the second fighter.
            X3: The input tensor for the fight features.
            odds1: The odds tensor for the first fighter.
            odds2: The odds tensor for the second fighter.

        Returns:
            The output of the SimpleFightNet model.
        """
        S1, S2 = self.transformer()
        x = torch.cat((X1, X2, X3, odds1, odds2), dim=1)

        for fc, dropout in zip(self.fcs[:-1], self.dropouts):
            x = self.relu(fc(x))
            x = dropout(x)

        x = self.fcs[-1](x)
        x = self.sigmoid(x)
        return x
    
class FighterTransformer(nn.Module):
    def __init__(self, state_dim, stat_dim, match_dim, hidden_dim, num_heads, num_layers, dropout=0.1):
        """
        Args:
            state_dim (int): Dimension of the fighter states (X1, X2).
            stat_dim (int): Dimension of the fighter stats (s1, s2).
            match_dim (int): Dimension of the match stats (m).
            hidden_dim (int): Dimension of the transformer hidden embeddings.
            num_heads (int): Number of attention heads in the transformer.
            num_layers (int): Number of transformer encoder layers.
            dropout (float): Dropout probability.
        """
        super().__init__()
        
        # Input embeddings
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        self.stat_embedding = nn.Linear(stat_dim, hidden_dim)
        self.match_embedding = nn.Linear(match_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, X1, X2, s1, s2, m):
        """
        Args:
            X1 (tensor): Fighter 1 state tensor of shape (batch_size, state_dim).
            X2 (tensor): Fighter 2 state tensor of shape (batch_size, state_dim).
            s1 (tensor): Fighter 1 stats tensor of shape (batch_size, stat_dim).
            s2 (tensor): Fighter 2 stats tensor of shape (batch_size, stat_dim).
            m (tensor): Match stats tensor of shape (batch_size, match_dim).
        
        Returns:
            X1_new (tensor): Fighter 1 new state tensor of shape (batch_size, state_dim).
            X2_new (tensor): Fighter 2 new state tensor of shape (batch_size, state_dim).
        """
        # Embed inputs
        X1_embed = self.state_embedding(X1)
        X2_embed = self.state_embedding(X2)
        s1_embed = self.stat_embedding(s1)
        s2_embed = self.stat_embedding(s2)
        m_embed = self.match_embedding(m)
        
        # Concatenate embeddings for the transformer input
        # Shape: (batch_size, sequence_length, hidden_dim)
        transformer_input = torch.stack([X1_embed, X2_embed, s1_embed, s2_embed, m_embed], dim=1)
        
        # Pass through the transformer
        # Shape: (batch_size, sequence_length, hidden_dim)
        transformer_output = self.transformer(transformer_input)
        
        # Extract new states for X1 and X2
        X1_new = self.output_projection(transformer_output[:, 0, :])  # First token corresponds to X1
        X2_new = self.output_projection(transformer_output[:, 1, :])  # Second token corresponds to X2
        
        return X1_new, X2_new