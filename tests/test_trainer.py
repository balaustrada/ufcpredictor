import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ufcpredictor.loss_functions import BettingLoss
from ufcpredictor.trainer import Trainer


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 1)

    def forward(self, X1, X2, X3, odds1=None, odds2=None):

        return torch.sigmoid(self.fc1(X1) + self.fc1(X2))


from torch.utils.data import Dataset
import torch


class CustomFightDataset(Dataset):
    def __init__(self, X1, X2, X3, Y, odds1, odds2):
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.Y = Y
        self.odds1 = odds1
        self.odds2 = odds2

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (
            (self.X1[idx], self.X2[idx], self.X3[idx]),
            self.Y[idx],
            (self.odds1[idx], self.odds2[idx]),
        )


class TestTrainer(unittest.TestCase):
    def setUp(self):
        # Create some dummy data
        X1 = torch.rand(100, 10)
        X2 = torch.rand(100, 10)
        X3 = torch.rand(100, 0)
        Y = torch.randint(0, 2, (100, 1)).float()
        odds1 = torch.rand(100, 1)
        odds2 = torch.rand(100, 1)

        dataset = CustomFightDataset(X1, X2, X3, Y, odds1, odds2)
        self.train_dataloader = DataLoader(dataset, batch_size=10)
        self.test_dataloader = DataLoader(dataset, batch_size=10)

        # Initialize a simple model, optimizer, and loss function
        self.model = SimpleNet()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss_fn = BettingLoss()
        self.device = "cpu"

        # Initialize Trainer
        self.trainer = Trainer(
            train_dataloader=self.train_dataloader,
            test_dataloader=self.test_dataloader,
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            device=self.device,
        )

    def test_train(self):
        # Test training loop for a single epoch
        self.trainer.train(epochs=1)
        self.assertIsNotNone(self.model)
        # Ensure the model was trained without errors

    def test_test(self):
        # Test evaluation loop
        val_loss, target_f1, correct, target, target_labels = self.trainer.test()
        self.assertGreaterEqual(target_f1, 0)
        # self.assertLessEqual(val_loss, 1.0)
        # Ensure validation runs properly and outputs valid metrics

    def test_empty_loader(self):
        trainer = Trainer(
            train_dataloader=self.train_dataloader,
            test_dataloader=None,
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            device=self.device,
        )
        result = trainer.test(test_dataloader=None)
        self.assertEqual(
            result,
            (0, 0, 0, [], []),
        )


class TestTrainerWithScheduler(unittest.TestCase):
    def setUp(self):
        # Create some dummy data
        X1 = torch.rand(100, 10)
        X2 = torch.rand(100, 10)
        X3 = torch.rand(100, 0)
        Y = torch.randint(0, 2, (100, 1)).float()
        odds1 = torch.rand(100, 1)
        odds2 = torch.rand(100, 1)

        dataset = CustomFightDataset(X1, X2, X3, Y, odds1, odds2)
        self.train_dataloader = DataLoader(dataset, batch_size=10)
        self.test_dataloader = DataLoader(dataset, batch_size=10)

        # Initialize a simple model, optimizer, and loss function
        self.model = SimpleNet()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss_fn = BettingLoss()
        self.device = "cpu"
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )

        # Initialize Trainer
        self.trainer = Trainer(
            train_dataloader=self.train_dataloader,
            test_dataloader=self.test_dataloader,
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            device=self.device,
            scheduler=self.scheduler,
        )

    def test_train(self):
        # Test training loop for a single epoch
        self.trainer.train(epochs=1)
        self.assertIsNotNone(self.model)
        # Ensure the model was trained without errors

    def test_test(self):
        # Test evaluation loop
        val_loss, target_f1, correct, target, target_labels = self.trainer.test()
        self.assertGreaterEqual(target_f1, 0)
        # self.assertLessEqual(val_loss, 1.0)
        # Ensure validation runs properly and outputs valid metrics

    # def test_train(self):
    #     # Test training loop for multiple epochs to ensure convergence
    #     initial_loss = None
    #     final_loss = None
    #     train_losses = []

    #     for epoch in range(5):
    #         self.trainer.train(epochs=1)  # Train for 1 epoch at a time
    #         val_loss, target_f1, correct, _, _ = (
    #             self.trainer.test()
    #         )  # Test to get validation loss

    #         if initial_loss is None:
    #             initial_loss = val_loss  # Track the initial loss at the start
    #         final_loss = val_loss  # Track the final loss after multiple epochs
    #         train_losses.append(val_loss)

    #     # Ensure the model converges by checking that the loss decreases over epochs
    #     self.assertLess(final_loss, initial_loss, "Training did not reduce loss.")

    #     # Ensure that losses are decreasing
    #     decreasing_losses = all(
    #         x > y for x, y in zip(train_losses[:-1], train_losses[1:])
    #     )
    #     self.assertTrue(
    #         decreasing_losses, "Losses did not consistently decrease over epochs."
    #     )

    #     # Ensure that model is training (loss should not be too large)
    #     self.assertLess(
    #         final_loss, 0.7, "Final loss is too high, indicating poor training."
    #     )

    # def test_test(self):
    #     # Test the evaluation loop
    #     val_loss, val_target_f1, val_correct, _, _ = self.trainer.test()
    #     train_loss, train_target_f1, train_correct, _, _ = self.trainer.test(
    #         self.train_dataloader
    #     )

    #     # Ensure the F1 score is reasonable
    #     self.assertGreaterEqual(val_target_f1, 0, "F1 score is below zero.")
    #     self.assertLessEqual(val_loss, 1.0, "Validation loss is too high.")

    #     # Ensure validation results compared to train results show no overfitting
    #     self.assertGreaterEqual(
    #         val_loss,
    #         train_loss,
    #         "Validation loss is unexpectedly lower than training loss (possible overfitting).",
    #     )
    #     self.assertAlmostEqual(
    #         val_loss,
    #         train_loss,
    #         delta=0.2,
    #         msg="Validation loss is much higher than training loss, indicating overfitting.",
    #     )

    #     # Check if training and validation accuracy/F1 score are close
    #     self.assertAlmostEqual(
    #         val_target_f1,
    #         train_target_f1,
    #         delta=0.2,
    #         msg="Validation F1 score is much lower than training F1 score, indicating overfitting.",
    #     )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
