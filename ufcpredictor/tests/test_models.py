import random
import unittest

import numpy as np
import torch

from ufcpredictor.models import FighterNet, SymmetricFightNet

# Assuming FighterNet and SymmetricFightNet are imported here


class TestFighterNet(unittest.TestCase):
    def setUp(self):
        self.input_size = 10  # Example input size
        self.dropout_prob = 0.5
        self.model = FighterNet(
            input_size=self.input_size, dropout_prob=self.dropout_prob
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
        self.input_size = 10  # Example input size
        self.dropout_prob = 0.5
        self.model = SymmetricFightNet(
            input_size=self.input_size,
            input_size_f=0,
            dropout_prob=self.dropout_prob,
        )

    def test_forward_pass(self):
        # Create dummy input tensors of shape (batch_size, input_size)
        batch_size = 32
        X1 = torch.randn(batch_size, self.input_size)
        X2 = torch.randn(batch_size, self.input_size)
        X3 = torch.empty(batch_size, 0)
        odds1 = torch.randn(batch_size, 1)
        odds2 = torch.randn(batch_size, 1)

        # Run a forward pass
        output = self.model(X1, X2, X3, odds1, odds2)

        # Check the output shape (since it's binary classification, output should be (batch_size, 1))
        self.assertEqual(output.shape, (batch_size, 1))

    def test_symmetric_behavior(self):
        # Check if symmetric inputs produce consistent outputs
        batch_size = 32
        X1 = torch.randn(batch_size, self.input_size)
        X2 = torch.randn(batch_size, self.input_size)
        X3 = torch.empty(batch_size, 0)
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
        X1 = torch.randn(batch_size, self.input_size)
        X2 = torch.randn(batch_size, self.input_size)
        X3 = torch.empty(batch_size, 0)
        odds1 = torch.randn(batch_size, 1)
        odds2 = torch.randn(batch_size, 1)

        # Run two forward passes with flipped inputs
        self.model.eval()
        with torch.no_grad():
            output1 = self.model(X1, X2, X3, odds1, odds2)

        torch.testing.assert_close(
            output1,
            torch.tensor(
                [
                    [0.5053],
                    [0.5041],
                    [0.5051],
                    [0.5050],
                    [0.5054],
                    [0.5044],
                    [0.5040],
                    [0.5045],
                    [0.5040],
                    [0.5041],
                    [0.5042],
                    [0.5042],
                    [0.5047],
                    [0.5043],
                    [0.5047],
                    [0.5042],
                    [0.5039],
                    [0.5042],
                    [0.5054],
                    [0.5042],
                    [0.5055],
                    [0.5046],
                    [0.5050],
                    [0.5039],
                    [0.5041],
                    [0.5054],
                    [0.5049],
                    [0.5039],
                    [0.5044],
                    [0.5040],
                    [0.5052],
                    [0.5053],
                ],
            ),
            atol=1e-3,
            rtol=1e-3,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
