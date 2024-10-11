from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

if TYPE_CHECKING:  # pragma: no cover
    from typing import List, Optional, Tuple

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
        device: str | torch.device = "cpu",
    ):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_fn = loss_fn.to(device)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader | None = None,
        test_loader: torch.utils.data.DataLoader | None = None,
        epochs: int = 10,
    ) -> None:
        if train_loader is None:
            train_loader = self.train_loader

        self.model.to(self.device)

        target_preds = []
        target_labels = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = []

            for X1, X2, Y, odds1, odds2 in tqdm(iter(train_loader)):
                X1, X2, Y, odds1, odds2 = (
                    X1.to(self.device),
                    X2.to(self.device),
                    Y.to(self.device),
                    odds1.to(self.device),
                    odds2.to(self.device),
                )

                self.optimizer.zero_grad()
                target_logit = self.model(X1, X2, odds1, odds2)
                loss = self.loss_fn(target_logit, Y, odds1, odds2)

                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())

                target_preds += (
                    torch.round(target_logit).detach().cpu().numpy().tolist()
                )
                target_labels += Y.detach().cpu().numpy().tolist()

            match = np.asarray(target_preds).reshape(-1) == np.asarray(
                target_labels
            ).reshape(-1)

            val_loss, val_target_f1, correct, _, _ = self.test(test_loader)

            print(f"Train acc: [{match.sum() / len(match):.5f}]")
            print(
                f"Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] "
                f"Val Loss : [{val_loss:.5f}] Disaster? F1 : [{val_target_f1:.5f}] "
                f"Correct: [{correct*100:.2f}]"
            )

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

    def test(
        self, test_loader: torch.utils.data.DataLoader | None = None
    ) -> Tuple[float, float, float, List, List]:
        if test_loader is None:
            test_loader = self.test_loader

        self.model.eval()
        val_loss = []

        target_preds = []
        target = []
        target_labels = []

        with torch.no_grad():
            for X1, X2, Y, odds1, odds2 in tqdm(iter(test_loader)):
                X1, X2, Y, odds1, odds2 = (
                    X1.to(self.device),
                    X2.to(self.device),
                    Y.to(self.device),
                    odds1.to(self.device),
                    odds2.to(self.device),
                )
                target_logit = self.model(X1, X2, odds1, odds2)
                loss = self.loss_fn(target_logit, Y, odds1, odds2)
                val_loss.append(loss.item())

                target += target_logit
                target_preds += (
                    torch.round(target_logit).detach().cpu().numpy().tolist()
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
            target,
            target_labels,
        )
