import torch
from torch import nn


class BettingLoss(nn.Module):
    def __init__(self) -> None:
        super(BettingLoss, self).__init__()

    def get_bet(self, prediction: torch.Tensor | float) -> torch.Tensor | float:
        return prediction * 2 * 10

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        odds_1: torch.Tensor,
        odds_2: torch.Tensor,
    ) -> torch.Tensor:
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
