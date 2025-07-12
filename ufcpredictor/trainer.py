"""
This module provides a Trainer class for training and testing PyTorch models using a
specific workflow.

The Trainer class encapsulates the training and testing data, model, optimizer, loss
function, and learning rate scheduler, providing a simple way to train and test a
PyTorch model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

import mlflow
from ufcpredictor.data_aggregator import DataAggregator
from ufcpredictor.data_processor import DataProcessor
from ufcpredictor.datasets import BasicDataset

if TYPE_CHECKING:  # pragma: no cover
    from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for training and testing a PyTorch model.

    This class provides a simple way to train and test a PyTorch model using a specific
    training and testing workflow.

    Attributes:
        train_dataloader (torch.utils.data.DataLoader): A DataLoader for the training data.
        test_dataloader (torch.utils.data.DataLoader): A DataLoader for the test data.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        loss_fn (torch.nn.Module): The loss function to be used.
        scheduler (Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]): The learning
            rate scheduler to be used.
        device (str | torch.device): The device to be used for training. Defaults to
            "cpu".
    """

    def __init__(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        test_dataloader: Optional[torch.utils.data.DataLoader] = None,
        scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
        device: str | torch.device = "cpu",
        mlflow_tracking: bool = False,
    ):
        """
        Initialize the Trainer object.

        Args:
            train_dataloader: A DataLoader for the training data.
            test_dataloader: A DataLoader for the test data.
            model: The model to be trained.
            optimizer: The optimizer to be used.
            loss_fn: The loss function to be used.
            scheduler: The learning rate scheduler to be used.
            device: The device to be used for training. Defaults to "cpu".
        """
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_fn = loss_fn.to(device)
        self.epoch_counter: int = 0
        self.mlflow_tracking = mlflow_tracking

        if self.mlflow_tracking:  # pragma: no cover
            params = {
                "optimizer": self.optimizer.__class__.__name__,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "scheduler": (
                    self.scheduler.__class__.__name__ if self.scheduler else None
                ),
                "scheduler_mode": self.scheduler.mode if self.scheduler else None,
                "scheduler_factor": self.scheduler.factor if self.scheduler else None,
                "scheduler_patience": (
                    self.scheduler.patience if self.scheduler else None
                ),
            }
            data_processor = cast(
                BasicDataset, self.train_dataloader.dataset
            ).data_processor
            data_aggregator = data_processor.data_aggregator

            for label, object_ in zip(
                ["loss_function", "model", "data_processor", "data_aggregator"],
                [self.loss_fn, self.model, data_processor, data_aggregator],
            ):
                params[label] = object_.__class__.__name__
                if hasattr(object_, "mlflow_params"):
                    for param in object_.mlflow_params:
                        params[label + "_" + param] = getattr(object_, param)

            data_enhancers = data_processor.data_enhancers
            # sort extra fields by name
            data_enhancers.sort(key=lambda x: x.__class__.__name__)

            for i, data_enhancer in enumerate(data_processor.data_enhancers):
                params["data_enhancer_" + str(i)] = data_enhancer.__class__.__name__
                for param in data_enhancer.mlflow_params:
                    params["data_enhancer_" + str(i) + "_" + param] = getattr(
                        data_enhancer, param
                    )

            for set_ in "fighter_fight_statistics", "fight_parameters":
                if hasattr(self.train_dataloader.dataset, set_):
                    params[set_] = sorted(getattr(self.train_dataloader.dataset, set_))

            mlflow.log_params(dict(sorted(params.items())))

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader | None = None,
        test_dataloader: torch.utils.data.DataLoader | None = None,
        epochs: int = 10,
        silent: bool = False,
    ) -> None:
        """
        Train the model for a given number of epochs.

        Args:
            train_dataloader: The DataLoader for the training data. Defaults to the
                DataLoader passed to the Trainer constructor.
            test_dataloader: The DataLoader for the test data. Defaults to the
                DataLoader passed to the Trainer constructor.
            epochs: The number of epochs to train for. Defaults to 10.
            silent: Whether to not print training progress. Defaults to False.

        Returns:
            None
        """
        if train_dataloader is None:
            train_dataloader = self.train_dataloader

        self.model.to(self.device)

        for epoch in range(1, epochs + 1):
            self.epoch_counter += 1
            self.model.train()
            train_loss = []
            target_preds: List[float] = []
            target_labels: List[float] = []

            for X, Y, odds in tqdm(iter(train_dataloader), disable=silent):
                X = [xi.to(self.device) for xi in X]
                odds = [oddsi.to(self.device) for oddsi in odds]
                Y = Y.to(self.device)

                self.optimizer.zero_grad()
                target_logit = self.model(*X, *odds)
                loss = self.loss_fn(target_logit, Y, *odds)

                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())
                target_preds += torch.round(target_logit).detach().cpu().numpy().flatten().tolist()  # type: ignore
                target_labels += Y.detach().cpu().numpy().flatten().tolist()

                if hasattr(train_dataloader.dataset, "update_data_trans"):
                    with torch.no_grad():
                        train_dataloader.dataset.update_data_trans(
                            self.model.evolver, self.device
                        )

            match = np.asarray(target_preds).reshape(-1) == np.asarray(
                target_labels
            ).reshape(-1)

            val_loss, val_target_f1, correct, _, _ = self.test(
                test_dataloader, silent=silent
            )

            if not silent:
                print(f"Train acc: [{match.sum() / len(match):.5f}]")
                print(
                    f"Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] "
                    f"Val Loss : [{val_loss:.5f}] Disaster? F1 : [{val_target_f1:.5f}] "
                    f"Correct: [{correct*100:.2f}]"
                )

            if self.mlflow_tracking:  # pragma: no cover
                mlflow.log_metric(
                    "train_loss", np.mean(train_loss), step=self.epoch_counter
                )
                mlflow.log_metric(
                    "val_loss", cast(float, np.mean(val_loss)), step=self.epoch_counter
                )
                mlflow.log_metric(
                    "val_f1_score", val_target_f1, step=self.epoch_counter
                )

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

    def test(
        self,
        test_dataloader: torch.utils.data.DataLoader | None = None,
        silent: bool = False,
    ) -> Tuple[float, float, float, List, List]:
        """
        Evaluates the model on the test data and returns the validation loss, target F1
        score, proportion of correct predictions, target predictions, and target labels.

        Args:
            test_dataloader: The DataLoader for the test data. Defaults to the DataLoader
                passed to the Trainer constructor.
            silent: Whether to not print training progress. Defaults to False.

        Returns:
            A tuple containing the validation loss, target F1 score, proportion of correct
            predictions, target predictions, and target labels.
        """
        if test_dataloader is None:
            if self.test_dataloader is None:
                return 0, 0, 0, [], []
            else:
                test_dataloader = self.test_dataloader

        self.model.eval()
        val_loss = []

        target_preds: List[float] = []
        target_labels: List[float] = []

        with torch.no_grad():
            for X, Y, odds in tqdm(iter(test_dataloader), disable=silent):
                X = [xi.to(self.device) for xi in X]
                odds = [oddsi.to(self.device) for oddsi in odds]
                Y = Y.to(self.device)

                self.optimizer.zero_grad()
                target_logit = self.model(*X, *odds)
                loss = self.loss_fn(target_logit, Y, *odds)
                val_loss.append(loss.item())

                target_preds += (
                    torch.round(target_logit).detach().cpu().numpy().tolist()  # type: ignore
                )
                target_labels += Y.detach().cpu().numpy().tolist()

        match = np.asarray(target_preds).reshape(-1) == np.asarray(
            target_labels
        ).reshape(-1)

        target_f1 = f1_score(target_labels, target_preds, average="macro")

        return (
            np.mean(val_loss),
            target_f1,
            match.sum() / len(match),
            target_preds,
            target_labels,
        )
