import unittest

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

    def test_dropout_effect(self):
        # Check if dropout layers have the correct probability
        self.assertEqual(self.model.dropout1.p, self.dropout_prob)
        self.assertEqual(self.model.dropout2.p, self.dropout_prob)
        self.assertEqual(self.model.dropout3.p, self.dropout_prob)
        self.assertEqual(self.model.dropout4.p, self.dropout_prob)
        self.assertEqual(self.model.dropout5.p, self.dropout_prob)


class TestSymmetricFightNet(unittest.TestCase):

    def setUp(self):
        self.input_size = 10  # Example input size
        self.dropout_prob = 0.5
        self.model = SymmetricFightNet(
            input_size=self.input_size, dropout_prob=self.dropout_prob
        )

    def test_forward_pass(self):
        # Create dummy input tensors of shape (batch_size, input_size)
        batch_size = 32
        X1 = torch.randn(batch_size, self.input_size)
        X2 = torch.randn(batch_size, self.input_size)
        odds1 = torch.randn(batch_size, 1)
        odds2 = torch.randn(batch_size, 1)

        # Run a forward pass
        output = self.model(X1, X2, odds1, odds2)

        # Check the output shape (since it's binary classification, output should be (batch_size, 1))
        self.assertEqual(output.shape, (batch_size, 1))

    def test_dropout_effect(self):
        # Check if dropout layers have the correct probability
        self.assertEqual(self.model.dropout1.p, self.dropout_prob)
        self.assertEqual(self.model.dropout2.p, self.dropout_prob)
        self.assertEqual(self.model.dropout3.p, self.dropout_prob)
        self.assertEqual(self.model.dropout4.p, self.dropout_prob)

    def test_symmetric_behavior(self):
        # Check if symmetric inputs produce consistent outputs
        batch_size = 32
        X1 = torch.randn(batch_size, self.input_size)
        X2 = torch.randn(batch_size, self.input_size)
        odds1 = torch.randn(batch_size, 1)
        odds2 = torch.randn(batch_size, 1)

        # Run two forward passes with flipped inputs
        self.model.eval()
        with torch.no_grad():
            output1 = self.model(X1, X2, odds1, odds2)
            output2 = self.model(X2, X1, odds2, odds1)

        # Since the model should be symmetric, the two outputs should be very similar
        self.assertTrue(torch.allclose(output1, output2, atol=1e-2))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
