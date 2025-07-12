import random
import unittest

import numpy as np
import torch

from ufcpredictor.models import FighterNet, SymmetricFightNet

# Assuming FighterNet and SymmetricFightNet are imported here


class TestFighterNet(unittest.TestCase):
    def setUp(self):
        # Create 10 random strings as statistics
        self.input_size = 10
        fighter_fight_statistics = [f"stat_{i}" for i in range(self.input_size)]
        self.dropout_prob = 0.5
        self.model = FighterNet(
            fighter_fight_statistics=fighter_fight_statistics,
            dropout_prob=self.dropout_prob,
        )

    def test_forward_pass(self):
        # Create a dummy input tensor of shape (batch_size, input_size)
        batch_size = 32
        dummy_input = torch.randn(batch_size, self.input_size)

        # Run a forward pass
        output = self.model(dummy_input)

        # Check the output shape
        expected_output_size = 127  # Expected output size based on the model definition
        self.assertEqual(output.shape, (batch_size, expected_output_size))


class TestSymmetricFightNet(unittest.TestCase):

    def setUp(self):
        seed = 30
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        # Crate 10 random strings as statistics
        self.fighter_fight_statistics = [f"stat_{i}" for i in range(4)]
        self.fight_parameters = []
        self.input_size = 10
        self.dropout_prob = 0.5
        self.model = SymmetricFightNet(
            fighter_fight_statistics=self.fighter_fight_statistics,
            fight_parameters=self.fight_parameters,
            dropout_prob=self.dropout_prob,
        )

    def test_forward_pass(self):
        # Create dummy input tensors of shape (batch_size, input_size)
        batch_size = 32
        X1 = torch.randn(batch_size, len(self.fighter_fight_statistics))
        X2 = torch.randn(batch_size, len(self.fighter_fight_statistics))
        X3 = torch.empty(batch_size, len(self.fight_parameters))
        odds1 = torch.randn(batch_size, 1)
        odds2 = torch.randn(batch_size, 1)

        # Run a forward pass
        output = self.model(X1, X2, X3, odds1, odds2)

        # Check the output shape (since it's binary classification, output should be (batch_size, 1))
        self.assertEqual(output.shape, (batch_size, 1))

    def test_symmetric_behavior(self):
        # Check if symmetric inputs produce consistent outputs
        batch_size = 32
        X1 = torch.randn(batch_size, len(self.fighter_fight_statistics))
        X2 = torch.randn(batch_size, len(self.fighter_fight_statistics))
        X3 = torch.empty(batch_size, len(self.fight_parameters))
        odds1 = torch.randn(batch_size, 1)
        odds2 = torch.randn(batch_size, 1)

        # Run two forward passes with flipped inputs
        self.model.eval()
        with torch.no_grad():
            output1 = self.model(X1, X2, X3, odds1, odds2)
            output2 = self.model(X2, X1, X3, odds2, odds1)

        # Since the model should be symmetric, the two outputs should be very similar
        self.assertTrue(torch.allclose(output1, output2, atol=1e-2))

    def test_model_output(self):
        batch_size = 32
        X1 = torch.randn(batch_size, len(self.fighter_fight_statistics))
        X2 = torch.randn(batch_size, len(self.fighter_fight_statistics))
        X3 = torch.empty(batch_size, len(self.fight_parameters))
        odds1 = torch.randn(batch_size, 1)
        odds2 = torch.randn(batch_size, 1)

        # Run two forward passes with flipped inputs
        self.model.eval()
        with torch.no_grad():
            output1 = self.model(X1, X2, X3, odds1, odds2)

        print(output1)
        torch.testing.assert_close(
            output1,
            torch.tensor(
                [
                    [0.5216],
                    [0.5228],
                    [0.5233],
                    [0.5223],
                    [0.5226],
                    [0.5229],
                    [0.5208],
                    [0.5210],
                    [0.5224],
                    [0.5213],
                    [0.5212],
                    [0.5214],
                    [0.5210],
                    [0.5233],
                    [0.5217],
                    [0.5210],
                    [0.5212],
                    [0.5221],
                    [0.5210],
                    [0.5212],
                    [0.5211],
                    [0.5210],
                    [0.5228],
                    [0.5211],
                    [0.5213],
                    [0.5211],
                    [0.5214],
                    [0.5207],
                    [0.5210],
                    [0.5226],
                    [0.5211],
                    [0.5210],
                ],
            ),
            atol=1e-3,
            rtol=1e-3,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
