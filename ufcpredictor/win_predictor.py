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


class WinPredictor:
    X_set = [
        "knockdowns_per_min",
        "strikes_att_per_min",
        "strikes_succ_per_min",
        "head_strikes_att_per_min",
        "head_strikes_succ_per_min",
        "body_strikes_att_per_min",
        "body_strikes_succ_per_min",
        "leg_strikes_att_per_min",
        "leg_strikes_succ_per_min",
        "distance_strikes_att_per_min",
        "distance_strikes_succ_per_min",
        "ground_strikes_att_per_min",
        "ground_strikes_succ_per_min",
        "clinch_strikes_att_per_min",
        "clinch_strikes_succ_per_min",
        "total_strikes_att_per_min",
        "total_strikes_succ_per_min",
        "takedown_att_per_min",
        "takedown_succ_per_min",
        "submission_att_per_min",
        "reversals_per_min",
        "ctrl_time_per_min",
        "KO_per_min",
        "KO_opp_per_min",
        "Sub_per_min",
        "Sub_opp_per_min",
        "win_per_min",
        "time_since_last_fight",
    ]

    def __init__(self, data_folder: Path | str) -> None:
        self.data_folder = data_folder
        self.data_processor = DataProcessor(
            data_folder = self.data_folder
        )
        self.data = self.data_processor.prepare_fight_data()    
