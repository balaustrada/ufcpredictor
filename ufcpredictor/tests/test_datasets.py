import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from ufcpredictor.datasets import BasicDataset, ForecastDataset


def mock_call_return_args(*args, **kwargs):
    return torch.Tensor(np.array(([arg for arg in args if arg.numel() != 0])))


class TestBasicDataset(unittest.TestCase):
    X_set = ["knockdowns_per_minute"]

    def test_basic_dataset_initialization(self):
        # Mock data
        mock_data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f3", "f1", "f2", "f3"],
                "fight_id": [
                    "fight1",
                    "fight1",
                    "fight2",
                    "fight2",
                    "fight3",
                    "fight3",
                ],
                "knockdowns_per_minute": [1, 2, 3, 4, 5, 6],
                "winner": ["f1", "f2", "f1", "f2", "f3", "f1"],
                "opening": [1.5, 2.0, 1.2, 1.8, 2.5, 1.9],
            }
        )

        mock_processor = MagicMock()
        mock_processor.data_normalized = mock_data

        fight_ids = ["fight1", "fight2"]

        # Check initialization without errors
        dataset = BasicDataset(
            data_processor=mock_processor, fight_ids=fight_ids, X_set=self.X_set
        )
        assert len(dataset.data) == 6  # We expect 5 tensors in dataset.data
        assert isinstance(dataset.data[0], torch.FloatTensor)  # Check tensor type
        assert len(dataset) == 2  # 2 fights in fight_ids

        # Check columns not found error
        with self.assertRaises(ValueError):
            BasicDataset(
                data_processor=mock_processor,
                fight_ids=fight_ids,
                X_set=["invalid_column"],
            )

    def test_basic_dataset_load_data(self):
        # Mock data
        mock_data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f3", "f1", "f2", "f3"],
                "fight_id": [
                    "fight1",
                    "fight1",
                    "fight2",
                    "fight2",
                    "fight3",
                    "fight3",
                ],
                "knockdowns_per_minute": [1, 2, 3, 4, 5, 6],
                "winner": ["f1", "f2", "f1", "f2", "f3", "f1"],
                "opening": [1.5, 2.0, 1.2, 1.8, 2.5, 1.9],
            }
        )

        mock_processor = MagicMock()
        mock_processor.data_normalized = mock_data

        fight_ids = ["fight1"]

        # Check data loading
        dataset = BasicDataset(
            data_processor=mock_processor, fight_ids=fight_ids, X_set=self.X_set
        )
        assert dataset.fight_data.shape[0] == 1  # Only one fight should be loaded

    def test_basic_dataset_getitem(self):
        # Mock data
        mock_data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f3", "f1", "f2", "f3"],
                "fight_id": [
                    "fight1",
                    "fight1",
                    "fight2",
                    "fight2",
                    "fight3",
                    "fight3",
                ],
                "knockdowns_per_minute": [1, 2, 3, 4, 5, 6],
                "winner": ["f1", "f2", "f1", "f2", "f3", "f1"],
                "opening": [1.5, 2.0, 1.2, 1.8, 2.5, 1.9],
            }
        )

        mock_processor = MagicMock()
        mock_processor.data_normalized = mock_data

        fight_ids = ["fight1"]

        dataset = BasicDataset(
            data_processor=mock_processor, fight_ids=fight_ids, X_set=self.X_set
        )
    
        # Retrieve an item
        X1, X2, X3, winner, odds_1, odds_2 = dataset[0]

        assert isinstance(
            X1, torch.FloatTensor
        )  # Check that the data is in tensor form
        assert winner.shape == torch.Size([1])  # Check the shape of the winner tensor
        assert odds_1.shape == torch.Size([1])  # Check odds shapes

    def test_basic_dataset_getitem_swap(self):
        # Mock data
        mock_data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f3", "f1", "f2", "f3"],
                "fight_id": [
                    "fight1",
                    "fight1",
                    "fight2",
                    "fight2",
                    "fight3",
                    "fight3",
                ],
                "knockdowns_per_minute": [1, 2, 3, 4, 5, 6],
                "winner": ["f1", "f2", "f1", "f2", "f3", "f1"],
                "opening": [1.5, 2.0, 1.2, 1.8, 2.5, 1.9],
            }
        )

        mock_processor = MagicMock()
        mock_processor.data_normalized = mock_data

        fight_ids = ["fight1"]

        dataset = BasicDataset(
            data_processor=mock_processor, fight_ids=fight_ids, X_set=self.X_set
        )

        # Retrieve an item multiple times to check for swapping
        with patch("numpy.random.random", side_effect=[0.1, 0.8]):
            original = dataset[0]
            swapped = dataset[0]

        assert not torch.equal(
            original[0], swapped[0]
        )  # X should be different after swap
        assert not torch.equal(original[1], swapped[1])  # Y should be swapped as well
        assert not torch.equal(original[3], swapped[3])  # Winner should be swapped

    def test_get_fight_data_from_ids(self):
        # Mock data
        mock_data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f3", "f1", "f2", "f3"],
                "fight_id": [
                    "fight1",
                    "fight1",
                    "fight2",
                    "fight2",
                    "fight3",
                    "fight3",
                ],
                "knockdowns_per_minute": [1, 2, 3, 4, 5, 6],
                "winner": ["f1", "f2", "f1", "f2", "f3", "f1"],
                "opening": [1.5, 2.0, 1.2, 1.8, 2.5, 1.9],
                "fighter_name": ["John", "Doe", "Jane", "John", "Doe", "Jane"],
            }
        )

        mock_processor = MagicMock()
        mock_processor.data_normalized = mock_data

        dataset = BasicDataset(
            data_processor=mock_processor,
            fight_ids=["fight1", "fight2", "fight3"],
            X_set=self.X_set,
        )

        # Test retrieving specific fight data
        X1, X2, X3, Y, odds1, odds2, fighter_names, opponent_names = (
            dataset.get_fight_data_from_ids(fight_ids=["fight1"])
        )

        assert len(fighter_names) == 1  # Only 1 fight in the result
        assert fighter_names[0] == "John"  # Correct fighter name

    def test_get_fight_data_from_ids_all(self):
        # Mock data
        mock_data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f3", "f1", "f2", "f3"],
                "fight_id": [
                    "fight1",
                    "fight1",
                    "fight2",
                    "fight2",
                    "fight3",
                    "fight3",
                ],
                "knockdowns_per_minute": [1, 2, 3, 4, 5, 6],
                "winner": ["f1", "f2", "f1", "f2", "f3", "f1"],
                "opening": [1.5, 2.0, 1.2, 1.8, 2.5, 1.9],
                "fighter_name": ["John", "Doe", "Jane", "John", "Doe", "Jane"],
            }
        )

        mock_processor = MagicMock()
        mock_processor.data_normalized = mock_data

        dataset = BasicDataset(
            data_processor=mock_processor,
            fight_ids=["fight1", "fight2", "fight3"],
            X_set=self.X_set,
        )

        # Test retrieving specific fight data
        X1, X2, X3, Y, odds1, odds2, fighter_names, opponent_names = (
            dataset.get_fight_data_from_ids(fight_ids=None)
        )

        assert (fighter_names == ["John", "Jane", "Doe"]).all()  # Correct fighter name


class TestForecastDataset(unittest.TestCase):
    X_set = ["knockdowns_per_minute"]

    def test_get_forecast_prediction(self):
        # Mock data
        mock_data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f3", "f1", "f2", "f3"],
                "fight_id": [
                    "fight1",
                    "fight1",
                    "fight2",
                    "fight2",
                    "fight3",
                    "fight3",
                ],
                "knockdowns_per_minute": [1, 2, 3, 4, 5, 6],
                "winner": ["f1", "f2", "f1", "f2", "f3", "f1"],
                "opening": [1.5, 2.0, 1.2, 1.8, 2.5, 1.9],
                "fighter_name": ["John", "Doe", "Jane", "John", "Doe", "Jane"],
                "event_date": [
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-03",
                ],
            }
        )

        mock_processor = MagicMock()
        mock_processor.data_normalized = mock_data

        forecast_dataset = ForecastDataset(
            data_processor=mock_processor, X_set=self.X_set
        )

        # Prepare mock input data
        fighter_ids = ["f1", "f2"]
        opponent_ids = ["f2", "f1"]
        event_dates = ["2023-02-01", "2023-03-02"]
        fighter_odds = [1.5, 2.0]
        opponent_odds = [1.8, 1.2]

        # Mock the model's forward method to return fixed predictions
        mock_model = MagicMock()
        mock_model.side_effect = mock_call_return_args

        mock_model.to = MagicMock()
        mock_model.to.return_value = mock_model

        # Call the method under test
        predictions_1, predictions_2 = forecast_dataset.get_forecast_prediction(
            fighter_names=fighter_ids,
            opponent_names=opponent_ids,
            event_dates=event_dates,
            fighter_odds=fighter_odds,
            opponent_odds=opponent_odds,
            model=mock_model,
            parse_ids=True,
        )

        # Verify the predictions content
        np.testing.assert_almost_equal(
            predictions_1.reshape(-1), [4.0, 5.0, 5.0, 4.0, 1.5, 2.0, 1.8, 1.2]
        )
        np.testing.assert_almost_equal(
            predictions_2.reshape(-1),
            [-4.0, -3.0, -3.0, -4.0, -0.79999995, -0.20000005, -0.5, -1.0],
        )  # As per model's output logic

    def test_get_forecast_prediction_from_name(self):
        # Mock data
        mock_data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f3", "f1", "f2", "f3"],
                "fight_id": [
                    "fight1",
                    "fight1",
                    "fight2",
                    "fight2",
                    "fight3",
                    "fight3",
                ],
                "knockdowns_per_minute": [1, 2, 3, 4, 5, 6],
                "winner": ["f1", "f2", "f1", "f2", "f3", "f1"],
                "opening": [1.5, 2.0, 1.2, 1.8, 2.5, 1.9],
                "fighter_name": ["John", "Doe", "Jane", "John", "Doe", "Jane"],
                "event_date": [
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-03",
                ],
            }
        )

        id_dictionary = {
            name: id_
            for name, id_ in zip(mock_data["fighter_name"], mock_data["fighter_id"])
        }

        mock_processor = MagicMock()
        mock_processor.data_normalized = mock_data
        mock_processor.get_fighter_id = MagicMock(
            side_effect=lambda x: id_dictionary[x]
        )

        forecast_dataset = ForecastDataset(
            data_processor=mock_processor, X_set=self.X_set
        )

        # Prepare mock input data
        fighter_names = ["John", "Doe"]
        opponent_names = ["Doe", "John"]
        event_dates = ["2023-02-01", "2023-03-02"]
        fighter_odds = [1.5, 2.0]
        opponent_odds = [1.8, 1.2]

        # Mock the model's forward method to return fixed predictions
        mock_model = MagicMock()
        mock_model.side_effect = mock_call_return_args

        mock_model.to = MagicMock()
        mock_model.to.return_value = mock_model

        # Call the method under test
        predictions_1, predictions_2 = forecast_dataset.get_forecast_prediction(
            fighter_names=fighter_names,
            opponent_names=opponent_names,
            event_dates=event_dates,
            fighter_odds=fighter_odds,
            opponent_odds=opponent_odds,
            model=mock_model,
        )

        # Verify the predictions content
        np.testing.assert_almost_equal(
            predictions_1.reshape(-1), [4.0, 5.0, 5.0, 4.0, 1.5, 2.0, 1.8, 1.2]
        )
        np.testing.assert_almost_equal(
            predictions_2.reshape(-1),
            [-4.0, -3.0, -3.0, -4.0, -0.79999995, -0.20000005, -0.5, -1.0],
        )  # As per model's output logic

    def test_column_not_found(self):
        # Mock data
        mock_data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f3", "f1", "f2", "f3"],
                "fight_id": [
                    "fight1",
                    "fight1",
                    "fight2",
                    "fight2",
                    "fight3",
                    "fight3",
                ],
                "knockdowns_per_minute": [1, 2, 3, 4, 5, 6],
                "winner": ["f1", "f2", "f1", "f2", "f3", "f1"],
                "opening": [1.5, 2.0, 1.2, 1.8, 2.5, 1.9],
                "fighter_name": ["John", "Doe", "Jane", "John", "Doe", "Jane"],
                "event_date": [
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-03",
                ],
            }
        )

        mock_processor = MagicMock()
        mock_processor.data_normalized = mock_data

        with self.assertRaises(ValueError) as e:
            forecast_dataset = ForecastDataset(
                data_processor=mock_processor,
                X_set=self.X_set
                + [
                    "missing_column",
                ],
            )

        self.assertEqual(
            "Columns not found in normalized data: ['missing_column']", str(e.exception)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
