from __future__ import annotations

import unittest
from pathlib import Path
from shutil import rmtree

import torch

from ufcpredictor.loss_functions import BettingLoss

THIS_DIR = Path(__file__).parent


class TestBettingLoss(unittest.TestCase):
    def setUp(self) -> None:
        Path(THIS_DIR / "test_files/run_files").mkdir(exist_ok=True)
        self.loss_fn = BettingLoss()

    def tearDown(self) -> None:
        rmtree(THIS_DIR / "test_files/run_files/")

    def test_get_bet(self):
        # Test get_bet for a few sample predictions
        predictions = torch.tensor([0.4, 0.5, 0.6])
        expected_bets = torch.tensor([8.0, 10.0, 12.0])

        torch.testing.assert_close(self.loss_fn.get_bet(predictions), expected_bets)

    def test_loss(self):
        # Test the loss function when some predictions match the targets
        predictions = torch.tensor([0.4, 0.8, 0.3, 0.7], requires_grad=True)
        targets = torch.tensor([0, 1, 0, 0])
        odds_1 = torch.tensor([1.5, 2.0, 2.5, 3.0])
        odds_2 = torch.tensor([1.8, 2.3, 2.0, 2.7])

        loss = self.loss_fn(predictions, targets, odds_1, odds_2)

        # Compute expected values manually for partial matches and assert the loss is correct
        earnings_1 = self.loss_fn.get_bet(0.5 - predictions) * odds_1
        earnings_2 = self.loss_fn.get_bet(predictions - 0.5) * odds_2
        losses = self.loss_fn.get_bet(torch.abs(predictions - 0.5))

        mask_0 = torch.round(predictions) == 0
        mask_1 = torch.round(predictions) == 1

        expected_earnings = torch.zeros_like(losses)
        expected_earnings[mask_0 & (targets == 0)] = earnings_1[mask_0 & (targets == 0)]
        expected_earnings[mask_1 & (targets == 1)] = earnings_2[mask_1 & (targets == 1)]

        expected_loss = (losses - expected_earnings).mean()

        self.assertAlmostEqual(loss.item(), expected_loss.item())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
