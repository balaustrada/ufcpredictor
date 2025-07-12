from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
import logging
from typing import TYPE_CHECKING
from importlib_resources import files

import numpy as np
import pandas as pd
import torch
import yaml

import ufcpredictor
from ufcpredictor.trainer import Trainer
from ufcpredictor import pretrained_models

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional


logger = logging.getLogger(__name__)


class UFCPredictor:
    config_path: Path | str
    config: dict
    device: torch.device
    early_train: bool = False

    data_processor: ufcpredictor.data_processor.DataProcessor
    early_train_dataset: Optional[ufcpredictor.datasets.Dataset] = None
    train_dataset: Optional[ufcpredictor.datasets.Dataset] = None
    test_dataset: Optional[ufcpredictor.datasets.Dataset] = None
    forecast_dataset: Optional[ufcpredictor.datasets.Dataset]

    early_train_dataloader: Optional[torch.utils.data.DataLoader] = None
    train_dataloader: Optional[torch.utils.data.DataLoader] = None
    test_dataloader: Optional[torch.utils.data.DataLoader] = None
    model: ufcpredictor.models.Model
    trainer: Trainer

    def __init__(
        self, config_path: Path | str, device: torch.device | str = "cpu"
    ) -> None:
        """
        Initializes the UFCPredictor instance.

        Args:
            config_path: Path or string representing the configuration file.
            device: Device to run the model on, e.g., "cpu" or "cuda".
        """
        self.config = yaml.safe_load(Path(config_path).read_text())

        self.device = torch.device(device)
        if (
            self.device.type == "cuda" and not torch.cuda.is_available()
        ):  # pragma: no cover
            raise ValueError("CUDA is not available on this device.")

        self.load_data_processor()

    def load_trainer(self) -> None:
        """
        Loads the trainer with the datasets, model, optimizer, scheduler, and loss function.
        """
        if self.config.get("filters", {}).get("minimum fight number", 0) > 0:
            invalid_fights = set(
                self.data_processor.data[self.data_processor.data["num_fight"] < 5][
                    "fight_id"
                ]
            )
        else:  # pragma: no cover
            invalid_fights = set()

        # TODO check this, because we are using the inverse and naming it the same.
        # Consider creating an inverse, or whatever, maybe introduce it inside
        # of data_enhancer. with factor -1 (?)
        # minimum_notice_days = self.config.get("filters", {}).get("minimum notice days", 0)
        # if minimum_notice_days > 0:
        #     invalid_fights.update(
        #         self.data_processor.data[
        #             self.data_processor.data["notice_days"] > 1 / minimum_notice_days
        #         ]["fight_id"]
        #     )
        minimum_notice_days = self.config.get("filters", {}).get(
            "minimum notice days", 1
        )
        if minimum_notice_days > 0:  # pragma: no cover
            invalid_fights.update(
                self.data_processor.data[
                    self.data_processor.data["notice_days"] > 1 / minimum_notice_days
                ]["fight_id"]
            )

        early_split_date = pd.to_datetime(
            self.config.get("filters", {}).get("early split date", None)
        )
        split_date = pd.to_datetime(self.config["filters"]["split date"])
        max_date = pd.to_datetime(
            self.config.get("filters", {}).get(
                "max date", datetime.now().strftime("%Y-%m-%d")
            )
        )

        include_test = split_date != max_date

        if early_split_date:
            early_train_fights = self.data_processor.data["fight_id"][
                self.data_processor.data["event_date"] < split_date
            ]
            train_fights = self.data_processor.data["fight_id"][
                (self.data_processor.data["event_date"] < split_date)
                & (self.data_processor.data["event_date"] >= early_split_date)
            ]
        else:  # pragma: no cover
            train_fights = self.data_processor.data["fight_id"][
                self.data_processor.data["event_date"] < split_date
            ]
            early_train_fights = set()

        early_train_fights = set(early_train_fights) - set(invalid_fights)
        train_fights = set(train_fights) - set(invalid_fights)

        if include_test:
            test_fights = self.data_processor.data["fight_id"][
                (self.data_processor.data["event_date"] >= split_date)
                & (self.data_processor.data["event_date"] <= max_date)
            ]
            test_fights = set(test_fights) - set(invalid_fights)

        self.early_train = len(early_train_fights) > 0

        # Loading datasets
        dataset_cfg = self.config.get("dataset", {})

        if dataset_cfg.get("class") is not None:
            Dataset = getattr(ufcpredictor.datasets, dataset_cfg.get("class"))
        else:  # pragma: no cover
            raise ValueError("No dataset specified in self.config file")

        if self.early_train:
            self.early_train_dataset = Dataset(
                data_processor=self.data_processor,
                fight_ids=early_train_fights,
                **dataset_cfg.get("args", {}),
            )

        self.train_dataset = Dataset(
            data_processor=self.data_processor,
            fight_ids=train_fights,
            **dataset_cfg.get("args", {}),
        )

        if include_test:
            self.test_dataset = Dataset(
                data_processor=self.data_processor,
                fight_ids=test_fights,
                **dataset_cfg.get("args", {}),
            )

        # Initialize dataloaders

        batch_size = self.config["training"]["batch size"]

        if self.early_train_dataset:
            self.early_train_dataloader = torch.utils.data.DataLoader(
                self.early_train_dataset, batch_size=batch_size, shuffle=True
            )

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )

        if self.test_dataset:
            self.test_dataloader = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=batch_size, shuffle=False
            )

        # Setting random seed for reproducibility
        seed = self.config["training"]["seed"]
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Loading model

        model_cfg = self.config.get("model", {})

        if model_cfg.get("class") is not None:
            self.model = getattr(ufcpredictor.models, model_cfg["class"])(
                **model_cfg.get("args", {}),
            )
        else:  # pragma: no cover
            raise ValueError("Model class not defined")

        # Loading optimimzer
        optimizer_cfg = self.config.get("optimizer", {})

        if optimizer_cfg.get("class") is not None:
            optimizer = getattr(ufcpredictor.optimizers, optimizer_cfg["class"])(
                self.model.parameters(),
                **optimizer_cfg.get("args", {}),
            )
        else:  # pragma: no cover
            raise ValueError("Optimizer class not defined")

        # Loading scheduler
        scheduler_cfg = self.config.get("scheduler", {})

        if scheduler_cfg.get("class") is not None:
            scheduler = getattr(ufcpredictor.schedulers, scheduler_cfg["class"])(
                optimizer,
                **scheduler_cfg.get("args", {}),
            )
        else:  # pragma: no cover
            scheduler = None

        # Loading loss
        loss_cfg = self.config.get("loss", {})

        if loss_cfg.get("class") is not None:
            loss = getattr(ufcpredictor.loss_functions, loss_cfg["class"])(
                **loss_cfg.get("args", {}),
            )
        else:  # pragma: no cover
            raise ValueError("Loss class not defined")

        self.trainer = Trainer(
            train_dataloader=self.train_dataloader,
            test_dataloader=self.test_dataloader,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss,
            mlflow_tracking=False,
            device=self.device,
        )

    def load_forecast_dataset(self) -> None:
        """
        Loads the forecast dataset based on the configuration.
        This is used for making predictions on new data.
        """
        forecast_dataset_cfg = self.config.get("forecast dataset", {})

        if forecast_dataset_cfg.get("class") is not None:
            Dataset = getattr(ufcpredictor.datasets, forecast_dataset_cfg.get("class"))
        else:  # pragma: no cover
            raise ValueError("No dataset specified in config file")

        self.forecast_dataset = Dataset(
            data_processor=self.data_processor,
            **forecast_dataset_cfg.get("args", {}),
        )

    def train_model(self) -> None:
        """
        Trains the model using the trainer with the specified configuration.
        """
        # Load trainer (if not already loaded)
        if not self.trainer:  # pragma: no cover
            self.load_trainer()

        if self.early_train:
            self.trainer.train(
                epochs=self.config["training"]["early train epochs"],
                train_dataloader=self.early_train_dataloader,
                test_dataloader=self.test_dataloader,
            )

        self.trainer.train(
            epochs=self.config["training"]["train epochs"],
            test_dataloader=self.test_dataloader,
        )

    def save_model(self) -> None:
        """
        Saves the trained model to the specified path in the configuration.
        """
        if not self.model:  # pragma: no cover
            raise ValueError(
                "Model is not loaded. Please train the model or load it from file first."
            )

        model_filename = self.config.get("general", {}).get("model filename", None)

        if model_filename:
            model_filename = Path(model_filename)
            if not (
                model_filename.is_absolute() or model_filename.parent != Path(".")
            ):  # pragma: no cover
                model_filename = files(pretrained_models).joinpath(model_filename)
        else:  # pragma: no cover
            raise ValueError("Model filename not specified in the configuration.")

        torch.save(
            self.model.state_dict(), Path("ufcpredictor/models") / model_filename
        )

    def load_model(self) -> None:
        """
        Loads a pre-trained model from the specified file in the configuration.
        """
        model_cfg = self.config.get("model", {})

        if model_cfg.get("class") is not None:
            self.model = getattr(ufcpredictor.models, model_cfg["class"])(
                **model_cfg.get("args", {}),
            )
        else:  # pragma: no cover
            raise ValueError("Model class not defined")

        model_filename = self.config.get("general", {}).get("model filename", None)

        if model_filename:
            model_filename = Path(model_filename)
            if not (
                model_filename.is_absolute() or model_filename.parent != Path(".")
            ):  # pragma: no cover
                model_filename = files(pretrained_models).joinpath(model_filename)
        else:  # pragma: no cover
            raise ValueError("Model filename not specified in the configuration.")

        if not model_filename.exists():  # pragma: no cover
            raise FileNotFoundError(f"Model file {model_filename} does not exist.")

        self.model.load_state_dict(torch.load(model_filename, weights_only=True))

    def load_data_processor(self) -> None:
        """
        Loads the data processor based on the configuration.
        Initializes the data aggregator and enhancers, and processes the data.
        """
        # Initialize data aggregator
        data_aggregator_cfg = self.config.get("data processor", {}).get(
            "data aggregator", {}
        )

        if data_aggregator_cfg.get("class") is not None:
            data_aggregator = getattr(
                ufcpredictor.data_aggregator, data_aggregator_cfg.get("class")
            )(**data_aggregator_cfg.get("args", {}))
        else:  # pragma: no cover
            raise Exception("Missing data_aggregator class")

        # Initialize data enhancers
        data_enhancers = []
        for data_enhancer_cfg in self.config.get("data processor", {}).get(
            "data enhancers", []
        ):
            data_enhancers.append(
                getattr(ufcpredictor.data_enhancers, data_enhancer_cfg.get("class"))(
                    **data_enhancer_cfg.get("args", {})
                )
            )

        # Initialize data processor
        data_processor_cfg = self.config.get("data processor", {})

        if data_processor_cfg.get("class") is not None:
            data_processor = getattr(
                ufcpredictor.data_processor, data_processor_cfg.get("class")
            )(
                data_aggregator=data_aggregator,
                data_enhancers=data_enhancers,
                **data_processor_cfg.get("args", {}),
            )
        else:  # pragma: no cover
            raise Exception("Missing data_processor class")

        # Load data in data processor
        data_processor.load_data()
        data_processor.aggregate_data()
        data_processor.add_per_minute_and_fight_stats()
        data_processor.normalize_data()

        self.data_processor = data_processor
