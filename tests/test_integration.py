import shutil
import unittest
from pathlib import Path

import numpy as np
import re
import torch

from ufcpredictor import UFCPredictor
from ufcpredictor.data_aggregator import WeightedDataAggregator
from ufcpredictor.data_enhancers import RankedFields, SumFlexibleELO
from ufcpredictor.data_processor import DataProcessor
from ufcpredictor.datasets import (
    BasicDataset,
    DatasetWithTimeEvolution,
    ForecastDataset,
    ForecastDatasetTimeEvolution,
)
from ufcpredictor.loss_functions import BettingLoss
from ufcpredictor.models import SimpleFightNet, SimpleFightNetWithTimeEvolution
from ufcpredictor.trainer import Trainer

THIS_DIR = Path(__file__).parent


class TestSimpleModel(unittest.TestCase):
    def test_it(self):
        data_processor_kwargs = {
            "data_folder": THIS_DIR / "test_files",
            "data_aggregator": WeightedDataAggregator(alpha=-0.0001),
            "data_enhancers": [
                SumFlexibleELO(
                    scaling_factor=0,  # 0.5
                    K_factor=30,  # 30
                ),
                RankedFields(
                    fields=["age", "fighter_height_cm"],
                    exponents=[1.2, 1.2],
                ),
            ],
        }

        data_processor = DataProcessor(**data_processor_kwargs)

        fighter_fight_statistics = [
            "age",
            # "body_strikes_att_opponent_per_minute",
            # "body_strikes_att_per_minute",
            "body_strikes_succ_opponent_per_minute",
            "body_strikes_succ_per_minute",
            # "clinch_strikes_att_opponent_per_minute",
            # "clinch_strikes_att_per_minute",
            "clinch_strikes_succ_opponent_per_minute",
            "clinch_strikes_succ_per_minute",
            "ctrl_time_opponent_per_minute",
            "ctrl_time_per_minute",
            # "distance_strikes_att_opponent_per_minute",
            # "distance_strikes_att_per_minute",
            "distance_strikes_succ_opponent_per_minute",
            "distance_strikes_succ_per_minute",
            "fighter_height_cm",
            # "ground_strikes_att_opponent_per_minute",
            # "ground_strikes_att_per_minute",
            "ground_strikes_succ_opponent_per_minute",
            "ground_strikes_succ_per_minute",
            # "head_strikes_att_opponent_per_minute",
            # "head_strikes_att_per_minute",
            "head_strikes_succ_opponent_per_minute",
            "head_strikes_succ_per_minute",
            "knockdowns_opponent_per_minute",
            "knockdowns_per_minute",
            # "KO_opponent_per_fight",
            "KO_opponent_per_minute",
            "KO_per_fight",
            "KO_per_minute",
            # "leg_strikes_att_opponent_per_minute",
            # "leg_strikes_att_per_minute",
            "leg_strikes_succ_opponent_per_minute",
            "leg_strikes_succ_per_minute",
            "num_fight",
            # "reversals_opponent_per_minute",
            # "reversals_per_minute",
            # "strikes_att_opponent_per_minute",
            # "strikes_att_per_minute",
            "strikes_succ_opponent_per_minute",
            "strikes_succ_per_minute",
            # "Sub_opponent_per_fight",
            "Sub_opponent_per_minute",
            # "Sub_per_fight",
            "Sub_per_minute",
            "submission_att_opponent_per_minute",
            "submission_att_per_minute",
            # "takedown_att_opponent_per_minute",
            # "takedown_att_per_minute",
            "takedown_succ_opponent_per_minute",
            "takedown_succ_per_minute",
            "time_since_last_fight",
            # "total_strikes_att_opponent_per_minute",
            # "total_strikes_att_per_minute",
            "total_strikes_succ_opponent_per_minute",
            "total_strikes_succ_per_minute",
            "win_opponent_per_fight",
            "win_per_fight",
            "ELO",
        ]
        fight_parameters = ["num_rounds", "weight"]

        # build data processor
        data_processor.load_data()
        data_processor.aggregate_data()
        data_processor.add_per_minute_and_fight_stats()
        data_processor.normalize_data()

        # Create splits
        fight_ids = data_processor.data["fight_id"].unique()
        invalid_fights = set(
            data_processor.data[data_processor.data["num_fight"] < 5]["fight_id"]
        )  # The usual is 4

        early_split_date = "2022-01-01"
        split_date = "2024-01-01"
        max_date = "2024-10-10"

        early_train_fights = data_processor.data["fight_id"][
            data_processor.data["event_date"] < split_date
        ]

        train_fights = data_processor.data["fight_id"][
            (
                (data_processor.data["event_date"] < split_date)
                & (data_processor.data["event_date"] >= early_split_date)
            )
        ]

        early_train_fights = set(early_train_fights) - set(invalid_fights)
        train_fights = set(train_fights) - set(invalid_fights)

        # Create datasets
        early_train_dataset = BasicDataset(
            data_processor,
            early_train_fights,
            fighter_fight_statistics=fighter_fight_statistics,
            fight_parameters=fight_parameters,
        )

        train_dataset = BasicDataset(
            data_processor,
            train_fights,
            fighter_fight_statistics=fighter_fight_statistics,
            fight_parameters=fight_parameters,
        )

        forecast_dataset = ForecastDataset(
            data_processor=data_processor,
            fighter_fight_statistics=fighter_fight_statistics,
            fight_parameters=fight_parameters,
        )

        batch_size = 64  # 2048
        early_train_dataloader = torch.utils.data.DataLoader(
            early_train_dataset, batch_size=batch_size, shuffle=True
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        seed = 21
        torch.manual_seed(seed)
        import random

        random.seed(seed)
        np.random.seed(seed)

        model = SimpleFightNet(
            fighter_fight_statistics=train_dataset.fighter_fight_statistics,
            fight_parameters=train_dataset.fight_parameters,
            dropout_prob=0.05,  # 0.25
        )

        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=1e-3, weight_decay=2e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=6
        )

        trainer = Trainer(
            train_dataloader=train_dataloader,
            test_dataloader=None,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=BettingLoss(),
        )

        trainer.train(epochs=1, train_dataloader=early_train_dataloader)
        trainer.train(epochs=10)
        model.eval()

        p1, p2 = forecast_dataset.get_single_forecast_prediction(
            fighter_name="47ffb45b4bac 6156bda3868d",
            opponent_name="5e228b7c95fd f1140f24a3a9",
            event_date="2024-11-11",
            odds1=1.1,
            odds2=1.2,
            model=model,
            fight_parameters_values=[5, 140],
        )

        self.assertAlmostEqual(p1, 0.48208007, places=3)
        self.assertAlmostEqual(p2, 0.5160773, places=3)

    def test_time_evolution(self):
        data_processor_kwargs = {
            "data_folder": THIS_DIR / "test_files",
            "data_aggregator": WeightedDataAggregator(alpha=-0.0001),
            "data_enhancers": [
                SumFlexibleELO(
                    scaling_factor=0,  # 0.5
                    K_factor=30,  # 30
                ),
                RankedFields(
                    fields=["age", "fighter_height_cm"],
                    exponents=[1.2, 1.2],
                ),
            ],
        }

        data_processor = DataProcessor(**data_processor_kwargs)

        fighter_fight_statistics = [
            "age",
            # "notice_days",
            # "body_strikes_att_opponent_per_minute",
            # "body_strikes_att_per_minute",
            "body_strikes_succ_opponent_per_minute",
            "body_strikes_succ_per_minute",
            # "clinch_strikes_att_opponent_per_minute",
            # "clinch_strikes_att_per_minute",
            "clinch_strikes_succ_opponent_per_minute",
            "clinch_strikes_succ_per_minute",
            "ctrl_time_opponent_per_minute",
            "ctrl_time_per_minute",
            # "distance_strikes_att_opponent_per_minute",
            # "distance_strikes_att_per_minute",
            "distance_strikes_succ_opponent_per_minute",
            "distance_strikes_succ_per_minute",
            "fighter_height_cm",
            # "ground_strikes_att_opponent_per_minute",
            # "ground_strikes_att_per_minute",
            "ground_strikes_succ_opponent_per_minute",
            "ground_strikes_succ_per_minute",
            # "head_strikes_att_opponent_per_minute",
            # "head_strikes_att_per_minute",
            "head_strikes_succ_opponent_per_minute",
            "head_strikes_succ_per_minute",
            "knockdowns_opponent_per_minute",
            "knockdowns_per_minute",
            # "KO_opponent_per_fight",
            "KO_opponent_per_minute",
            "KO_per_fight",
            "KO_per_minute",
            # "leg_strikes_att_opponent_per_minute",
            # "leg_strikes_att_per_minute",
            "leg_strikes_succ_opponent_per_minute",
            "leg_strikes_succ_per_minute",
            "num_fight",
            # "reversals_opponent_per_minute",
            # "reversals_per_minute",
            # "strikes_att_opponent_per_minute",
            # "strikes_att_per_minute",
            "strikes_succ_opponent_per_minute",
            "strikes_succ_per_minute",
            # "Sub_opponent_per_fight",
            "Sub_opponent_per_minute",
            # "Sub_per_fight",
            "Sub_per_minute",
            "submission_att_opponent_per_minute",
            "submission_att_per_minute",
            # "takedown_att_opponent_per_minute",
            # "takedown_att_per_minute",
            "takedown_succ_opponent_per_minute",
            "takedown_succ_per_minute",
            "time_since_last_fight",
            # "total_strikes_att_opponent_per_minute",
            # "total_strikes_att_per_minute",
            "total_strikes_succ_opponent_per_minute",
            "total_strikes_succ_per_minute",
            "win_opponent_per_fight",
            "win_per_fight",
            "ELO",
        ]
        previous_fights_statistics = [
            "age",
            # "notice_days",
            # "body_strikes_att_opponent_per_minute",
            # "body_strikes_att_per_minute",
            "body_strikes_succ_opponent_per_minute",
            "body_strikes_succ_per_minute",
            # "clinch_strikes_att_opponent_per_minute",
            # "clinch_strikes_att_per_minute",
            "clinch_strikes_succ_opponent_per_minute",
            "clinch_strikes_succ_per_minute",
            "ctrl_time_opponent_per_minute",
            "ctrl_time_per_minute",
            # "distance_strikes_att_opponent_per_minute",
            # "distance_strikes_att_per_minute",
            "distance_strikes_succ_opponent_per_minute",
            "distance_strikes_succ_per_minute",
            # "fighter_height_cm",
            # "ground_strikes_att_opponent_per_minute",
            # "ground_strikes_att_per_minute",
            "ground_strikes_succ_opponent_per_minute",
            "ground_strikes_succ_per_minute",
            # "head_strikes_att_opponent_per_minute",
            # "head_strikes_att_per_minute",
            "head_strikes_succ_opponent_per_minute",
            "head_strikes_succ_per_minute",
            "knockdowns_opponent_per_minute",
            "knockdowns_per_minute",
            # "KO_opponent_per_fight",
            "KO_opponent_per_minute",
            # "KO_per_fight",
            "KO_per_minute",
            # "leg_strikes_att_opponent_per_minute",
            # "leg_strikes_att_per_minute",
            "leg_strikes_succ_opponent_per_minute",
            "leg_strikes_succ_per_minute",
            # "num_fight",
            # "reversals_opponent_per_minute",
            # "reversals_per_minute",
            # "strikes_att_opponent_per_minute",
            # "strikes_att_per_minute",
            "strikes_succ_opponent_per_minute",
            "strikes_succ_per_minute",
            # "Sub_opponent_per_fight",
            "Sub_opponent_per_minute",
            # "Sub_per_fight",
            "Sub_per_minute",
            "submission_att_opponent_per_minute",
            "submission_att_per_minute",
            # "takedown_att_opponent_per_minute",
            # "takedown_att_per_minute",
            "takedown_succ_opponent_per_minute",
            "takedown_succ_per_minute",
            "time_since_last_fight",  # Adding this somehow slowed the convergence and is not as good (why?) maybe because of the default value used(?) it was the mean (~ 7months)
            # "total_strikes_att_opponent_per_minute",
            # "total_strikes_att_per_minute",
            "total_strikes_succ_opponent_per_minute",
            "total_strikes_succ_per_minute",
            # "win_opponent_per_fight",
            # "win_per_fight",
            "ELO",
        ]
        state_size = 8
        fight_parameters = ["num_rounds", "weight"]
        previous_fights_parameters = ["num_rounds", "weight", "winner"]

        # build data processor
        data_processor.load_data()
        data_processor.aggregate_data()
        data_processor.add_per_minute_and_fight_stats()
        data_processor.normalize_data()

        # Create splits
        fight_ids = data_processor.data["fight_id"].unique()
        invalid_fights = set(
            data_processor.data[data_processor.data["num_fight"] < 5]["fight_id"]
        )  # The usual is 4

        invalid_fights.update(
            data_processor.data[data_processor.data["notice_days"] != 1 / 60][
                "fight_id"
            ]
        )  # Short notices worsen the result, even when introduced in the model...

        early_split_date = "2022-01-01"
        split_date = "2024-01-01"
        max_date = "2024-10-10"

        early_train_fights = data_processor.data["fight_id"][
            data_processor.data["event_date"] < split_date
        ]

        train_fights = data_processor.data["fight_id"][
            (
                (data_processor.data["event_date"] < split_date)
                & (data_processor.data["event_date"] >= early_split_date)
            )
        ]

        early_train_fights = set(early_train_fights) - set(invalid_fights)
        train_fights = set(train_fights) - set(invalid_fights)

        # Create datasets
        early_train_dataset = DatasetWithTimeEvolution(
            data_processor,
            early_train_fights,
            fighter_fight_statistics=fighter_fight_statistics,
            fight_parameters=fight_parameters,
            previous_fights_statistics=previous_fights_statistics,
            previous_fights_parameters=previous_fights_parameters,
            state_size=state_size,
        )

        train_dataset = DatasetWithTimeEvolution(
            data_processor,
            train_fights,
            fighter_fight_statistics=fighter_fight_statistics,
            fight_parameters=fight_parameters,
            previous_fights_statistics=previous_fights_statistics,
            previous_fights_parameters=previous_fights_parameters,
            state_size=state_size,
        )

        forecast_dataset = ForecastDatasetTimeEvolution(
            data_processor=data_processor,
            fighter_fight_statistics=fighter_fight_statistics,
            fight_parameters=fight_parameters,
            previous_fights_statistics=previous_fights_statistics,
            previous_fights_parameters=previous_fights_parameters,
            state_size=state_size,
        )

        batch_size = 64  # 2048
        early_train_dataloader = torch.utils.data.DataLoader(
            early_train_dataset, batch_size=batch_size, shuffle=True
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        seed = 21
        torch.manual_seed(seed)
        import random

        random.seed(seed)
        np.random.seed(seed)

        dropout = 0.01
        model = SimpleFightNetWithTimeEvolution(
            fighter_fight_statistics=fighter_fight_statistics,
            fight_parameters=fight_parameters,
            dropout_prob=dropout,
            network_shape=[128, 64, 32, 1],
            state_size=state_size,
            # network_shape=[122, 1024, 512, 1024, 512, 256, 128, 64, 1],
            fighter_transformer_kwargs=dict(
                state_size=state_size,  # 20,
                fighter_fight_statistics=previous_fights_statistics,
                fight_parameters=previous_fights_parameters,
                network_shape=[128, 64],
                dropout=dropout * 0.9,
            ),
        )

        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=1.3e-3, weight_decay=1e-5  # 1e-5
        )  # , weight_decay=2e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.7, patience=2
        )

        trainer = Trainer(
            train_dataloader=train_dataloader,
            test_dataloader=None,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=BettingLoss(),
            mlflow_tracking=False,
        )

        trainer.train(epochs=1, train_dataloader=early_train_dataloader)
        trainer.train(epochs=10)
        model.eval()

        p1, p2 = forecast_dataset.get_single_forecast_prediction(
            fighter_name="47ffb45b4bac 6156bda3868d",
            opponent_name="5e228b7c95fd f1140f24a3a9",
            event_date="2024-11-11",
            odds1=1.1,
            odds2=1.2,
            model=model,
            fight_parameters_values=[5, 140],
        )

        self.assertAlmostEqual(float(p1), 0.5147984, places=3)
        self.assertAlmostEqual(float(p2), 0.5147038, places=3)


class TestModelUsingConfig(unittest.TestCase):
    def setUp(self):
        # Config file is under THIS_DIR / "test_files" / "config_simple.yaml"
        # I need to copy this file, and change inside of it the data_folder to point
        # to THIS_DIR / "test_files" it appears twice in the file

        config_file = THIS_DIR / "test_files" / "config_simple.yaml"
        # copy file into /tmp and edit it to modify lines

        self.temp_config_file = Path("/tmp/config_simple.yaml")
        shutil.copy(config_file, self.temp_config_file)

        # use read the file as read fieldata = file.read() then use
        # .replace(pattern, replacement) then save
        with open(self.temp_config_file, "r") as file:
            filedata = file.read()

        # Replace the data_folder line
        filedata = filedata.replace("data_folder_replace", f"{THIS_DIR / 'test_files'}")

        with open(self.temp_config_file, "w") as file:
            file.write(filedata)

    def tearDown(self):
        # Remove the temporary config file if it exists
        if self.temp_config_file.exists():
            self.temp_config_file.unlink()

    def test_it_with_module(self):
        predictor = UFCPredictor(
            config_path="/tmp/config_simple.yaml",
            device="cpu",
        )
        predictor.load_trainer()
        predictor.train_model()
        predictor.load_forecast_dataset()
        predictor.model.eval()

        p1, p2 = predictor.forecast_dataset.get_single_forecast_prediction(
            fighter_name="47ffb45b4bac 6156bda3868d",
            opponent_name="5e228b7c95fd f1140f24a3a9",
            event_date="2024-11-11",
            odds1=1.1,
            odds2=1.2,
            model=predictor.model,
            fight_parameters_values=[5, 140],
        )

        self.assertAlmostEqual(p1, 0.52276415, places=3)
        self.assertAlmostEqual(p2, 0.4773013, places=3)

    def test_save_model(self):
        with open(self.temp_config_file, "r") as file:
            filedata = file.read()

        # I want to replace the full line that contains simple_fighter_network.pt into
        # /tmp/simple_fighter_network.pt
        # The full line, not just the matching
        filedata = re.sub(
            r"model filename: .+",
            "model filename: /tmp/simple_fighter_network.pt",
            filedata,
        )

        with open(self.temp_config_file, "w") as file:
            file.write(filedata)

        predictor = UFCPredictor(
            config_path="/tmp/config_simple.yaml",
            device="cpu",
        )
        predictor.load_trainer()
        predictor.train_model()

        predictor.save_model()

        # Check that the saved model matches the one in test files
        saved_model_path = Path("/tmp/simple_fighter_network.pt")
        expected_model_path = THIS_DIR / "test_files" / "simple_fighter_network.pt"

        self.assertTrue(saved_model_path.exists(), "Saved model file does not exist")
        self.assertTrue(
            expected_model_path.exists(), "Expected model file does not exist"
        )

        # Load both models and compare their state_dicts
        saved_model = torch.load(saved_model_path, map_location="cpu")
        expected_model = torch.load(expected_model_path, map_location="cpu")

        # Compare the tensors using torch
        # This will check if the tensors are equal
        for key in saved_model:
            np.testing.assert_allclose(
                saved_model[key].numpy(),
                expected_model[key].numpy(),
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"Tensor mismatch for key: {key}",
            )

    def test_load_model(self):
        predictor = UFCPredictor(
            config_path="/tmp/config_simple.yaml",
            device="cpu",
        )

        predictor.load_model()
        predictor.load_forecast_dataset()
        predictor.model.eval()

        p1, p2 = predictor.forecast_dataset.get_single_forecast_prediction(
            fighter_name="47ffb45b4bac 6156bda3868d",
            opponent_name="5e228b7c95fd f1140f24a3a9",
            event_date="2024-11-11",
            odds1=1.1,
            odds2=1.2,
            model=predictor.model,
            fight_parameters_values=[5, 140],
        )

        self.assertAlmostEqual(p1, 0.5227641, places=3)
        self.assertAlmostEqual(p2, 0.4773013, places=3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
