import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from ufcpredictor.datasets import (
    BasicDataset,
    ForecastDataset,
    DatasetWithTimeEvolution,
)


def mock_call_return_args(*args, **kwargs):
    return torch.Tensor(np.array(([arg for arg in args if arg.numel() != 0])))


class TestBasicDataset(unittest.TestCase):
    fighter_fight_statistics = ["knockdowns_per_minute"]

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
            data_processor=mock_processor,
            fight_ids=fight_ids,
            fighter_fight_statistics=self.fighter_fight_statistics,
        )
        assert len(dataset.data) == 6  # We expect 5 tensors in dataset.data
        assert isinstance(dataset.data[0], torch.FloatTensor)  # Check tensor type
        assert len(dataset) == 2  # 2 fights in fight_ids

        # Check columns not found error
        with self.assertRaises(ValueError):
            BasicDataset(
                data_processor=mock_processor,
                fight_ids=fight_ids,
                fighter_fight_statistics=["invalid_column"],
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
            data_processor=mock_processor,
            fight_ids=fight_ids,
            fighter_fight_statistics=self.fighter_fight_statistics,
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
            data_processor=mock_processor,
            fight_ids=fight_ids,
            fighter_fight_statistics=self.fighter_fight_statistics,
        )

        # Retrieve an item
        (X1, X2, X3), winner, (odds_1, odds_2) = dataset[0]

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

        fight_ids = ["fight1", "fight2", "fight3"]

        dataset = BasicDataset(
            data_processor=mock_processor,
            fight_ids=fight_ids,
            fighter_fight_statistics=self.fighter_fight_statistics,
        )

        # Retrieve an item multiple times to check for swapping
        with patch("numpy.random.random", side_effect=[0.1, 0.8]):
            original = dataset[2]
            swapped = dataset[2]

        assert original[0][0] == swapped[0][1]
        assert not torch.equal(original[1], swapped[1])  # Y should be swapped as well
        assert original[2] == swapped[2][::-1]  # Winner should be swapped

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
            fighter_fight_statistics=self.fighter_fight_statistics,
        )

        # Test retrieving specific fight data
        (X1, X2, X3), Y, (odds1, odds2), fighter_names, opponent_names = (
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
            fighter_fight_statistics=self.fighter_fight_statistics,
        )

        # Test retrieving specific fight data
        (X1, X2, X3), Y, (odds1, odds2), fighter_names, opponent_names = (
            dataset.get_fight_data_from_ids(fight_ids=None)
        )

        assert (fighter_names == ["John", "Jane", "Doe"]).all()  # Correct fighter name


class TestForecastDataset(unittest.TestCase):
    fighter_fight_statistics = ["knockdowns_per_minute"]

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
                "fighter_dob": [
                    "1990-01-01",
                    "1990-01-01",
                    "1990-01-02",
                    "1990-01-02",
                    "1990-01-03",
                    "1990-01-03",
                ],
                "num_fight": [
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                ],
            }
        )
        mock_data["event_date"] = pd.to_datetime(mock_data["event_date"])
        mock_data["fighter_dob"] = pd.to_datetime(mock_data["fighter_dob"])

        mock_processor = MagicMock()
        mock_processor.data_normalized = mock_data

        forecast_dataset = ForecastDataset(
            data_processor=mock_processor,
            fighter_fight_statistics=self.fighter_fight_statistics,
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
                "fighter_dob": [
                    "1990-01-01",
                    "1990-01-01",
                    "1990-01-02",
                    "1990-01-02",
                    "1990-01-03",
                    "1990-01-03",
                ],
                "num_fight": [
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                ],
            }
        )
        mock_data["event_date"] = pd.to_datetime(mock_data["event_date"])
        mock_data["fighter_dob"] = pd.to_datetime(mock_data["fighter_dob"])

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
            data_processor=mock_processor,
            fighter_fight_statistics=self.fighter_fight_statistics,
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
                fighter_fight_statistics=self.fighter_fight_statistics
                + [
                    "missing_column",
                ],
            )

        self.assertEqual(
            "Columns not found in normalized data: ['missing_column']", str(e.exception)
        )


class TestDatasetWithTimeEvolution(unittest.TestCase):
    fighter_fight_statistics = ["knockdowns_per_minute"]

    def test_get_indices_previous_and_next_winner_binary(self):
        # Prepare mock data with winner and fighter_id columns
        mock_data = pd.DataFrame(
            {
                "fight_id": ["fight1", "fight1", "fight2", "fight2"],
                "fighter_id": ["f1", "f2", "f3", "f4"],
                "event_date": pd.to_datetime(
                    ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"]
                ),
                "num_fight": [1, 1, 2, 2],
                "opponent_id": ["f2", "f1", "f4", "f3"],
                "body_strikes_att_per_minute": [1.0, 2.0, 3.0, 4.0],
                "clinch_strikes_att_per_minute": [1.1, 2.1, 3.1, 4.1],
                "knockdowns_per_minute": [0.5, 0.6, 0.7, 0.8],
                "ELO": [1000, 1100, 1200, 1300],
                "opening": [1.5, 2.0, 1.8, 2.2],  # Add this line
                "winner": ["f1", "f2", "f3", "f4"],
            }
        )

        # Patch DataProcessor to provide this DataFrame for both normalized and nonagg
        mock_processor = MagicMock()
        mock_processor.data_normalized_nonagg = mock_data.copy()
        mock_processor.data_normalized = mock_data.copy()
        # previous_fights_statistics and previous_fights_parameters must match the columns above

        previous_fights_statistics = [
            "body_strikes_att_per_minute",
            "clinch_strikes_att_per_minute",
            "knockdowns_per_minute",
            "ELO",
            "opening",  # Add this line
        ]
        previous_fights_parameters = ["winner"]

        dataset = DatasetWithTimeEvolution(
            data_processor=mock_processor,
            fighter_fight_statistics=["body_strikes_att_per_minute"],
            fight_parameters=[],
            previous_fights_statistics=previous_fights_statistics,
            previous_fights_parameters=previous_fights_parameters,
            state_size=2,
        )

        # Call get_indices_previous_and_next directly to check the winner column
        previous_and_next_indices = dataset.get_indices_previous_and_next()
        # The winner column should be 1 for all rows, since winner == fighter_id
        assert (previous_and_next_indices["winner"] == 1).all()

        # Now, set winner to something else and check for 0s
        mock_data2 = mock_data.copy()
        mock_data2.loc[0, "winner"] = "not_f1"
        mock_processor.data_normalized_nonagg = mock_data2
        mock_processor.data_normalized = mock_data2
        previous_and_next_indices2 = dataset.get_indices_previous_and_next()
        assert previous_and_next_indices2.loc[0, "winner"] == 0
        assert (previous_and_next_indices2.loc[1:, "winner"] == 1).all()

    def test_get_fight_data_from_ids_with_fight_ids(self):
        # Prepare mock data
        mock_data = pd.DataFrame(
            {
                "fight_id": ["fight1", "fight1", "fight2", "fight2"],
                "fighter_id": ["f1", "f2", "f3", "f4"],
                "event_date": pd.to_datetime(
                    ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02"]
                ),
                "num_fight": [1, 1, 2, 2],
                "opponent_id": ["f2", "f1", "f4", "f3"],
                "body_strikes_att_per_minute": [1.0, 2.0, 3.0, 4.0],
                "clinch_strikes_att_per_minute": [1.1, 2.1, 3.1, 4.1],
                "knockdowns_per_minute": [0.5, 0.6, 0.7, 0.8],
                "ELO": [1000, 1100, 1200, 1300],
                "winner": ["f1", "f2", "f3", "f4"],
                "opening": [1.5, 2.0, 1.8, 2.2],
                "fighter_name": ["A", "B", "C", "D"],
            }
        )

        mock_processor = MagicMock()
        mock_processor.data_normalized_nonagg = mock_data.copy()
        mock_processor.data_normalized = mock_data.copy()
        mock_processor.data_enhancers = []
        mock_processor.normalization_factors = {}

        previous_fights_statistics = [
            "body_strikes_att_per_minute",
            "clinch_strikes_att_per_minute",
            "knockdowns_per_minute",
            "ELO",
        ]
        previous_fights_parameters = ["winner"]

        dataset = DatasetWithTimeEvolution(
            data_processor=mock_processor,
            fighter_fight_statistics=["body_strikes_att_per_minute"],
            fight_parameters=[],
            previous_fights_statistics=previous_fights_statistics,
            previous_fights_parameters=previous_fights_parameters,
            state_size=2,
        )

        # Use only fight1 for test
        fight_ids = ["fight1"]
        result = dataset.get_fight_data_from_ids(fight_ids=fight_ids)
        # Check that the returned data only contains fight1
        # result[6] and result[7] are fighter_names and opponent_names
        fighter_names = result[-2]
        opponent_names = result[-1]
        self.assertTrue(all(f in ["A", "B"] for f in fighter_names))
        self.assertTrue(all(o in ["A", "B"] for o in opponent_names))
        # Check that the shapes match the number of fights (should be 1 row)
        self.assertEqual(result[0][0].shape[0], 1)


if __name__ == "__main__":  # pragma: no cover    unittest.main()
    unittest.main()
