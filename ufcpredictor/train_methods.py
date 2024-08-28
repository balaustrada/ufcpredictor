from __future__ import annotations

import logging

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ufcscraper.ufc_scraper import UFCScraper
from ufcscraper.odds_scraper import BestFightOddsScraper
from ufcpredictor.utils import convert_minutes_to_seconds, weight_dict
from ufcpredictor.data_processor import DataProcessor
from ufcpredictor.networks import SymmetricFightNet
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import f1_score




if TYPE_CHECKING:  # pragma: no cover
    import datetime
    from typing import Any, Callable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

def train(
        model, train_dataloader, test_dataloader, optimizer, scheduler, device, num_epochs
):
    model.to(device)
    criterion = nn.BCELoss().to(device)

    best_loss = 99999999
    best_model = None

    target_preds = []
    target_labels = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = []

        for X1, X2, Y in tqdm(iter(train_dataloader)):
            X1 = X1.to(device)
            X2 = X2.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            out = model(X1, X2)
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            target_preds += torch.round(out).detach().cpu().numpy().tolist()
            target_labels += Y.detach().cpu().numpy().tolist()

        match = np.asarray(target_preds).reshape(-1) == np.asarray(
            target_labels
        ).reshape(-1)

        val_loss, val_target_f1, correct, _, _ = validation(
            model, test_dataloader, criterion, device
        )


        print(f"Train acc: [{match.sum() / len(match):.5f}]")
        print(
            f"Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Disaster? F1 : [{val_target_f1:.5f}] Correct: [{correct*100:.2f}]"
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        if best_loss > val_loss:
            best_loss = val_loss
            best_model = model
            torch.save(model, "./best-model.pth")
            print("Model saved!")

    return best_model

# validation function
def validation(model, test_dataloader, criterion, device):
    model.eval()
    val_loss = []

    target_preds = []
    target = []
    target_labels = []

    with torch.no_grad():
        for X1, X2, Y in tqdm(iter(test_dataloader)):
            X1, X2 = X1.to(device), X2.to(device)
            Y = Y.to(device)

            out = model(X1, X2)
            loss = criterion(out, Y)

            val_loss.append(loss.item())

            target += out
            target_preds += torch.round(out).detach().cpu().numpy().tolist()
            target_labels += Y.detach().cpu().numpy().tolist()

    match = np.asarray(target_preds).reshape(-1) == np.asarray(target_labels).reshape(
        -1
    )

    target_f1 = f1_score(target_labels, target_preds, average="macro")

    return np.mean(val_loss), target_f1, match.sum() / len(match), target, target_labels