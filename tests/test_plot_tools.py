import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from ufcpredictor.plot_tools import PredictionPlots


class TestPredictionPlots(unittest.TestCase):
    @patch("matplotlib.pyplot.subplots")  # Mock subplots creation
    def test_show_fight_prediction_detail(self, mock_subplots):
        # Create mock axis object
        mock_ax = MagicMock()
        mock_subplots.return_value = (MagicMock(), mock_ax)

        # Mock data
        X1 = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        X2 = torch.tensor([[0.5, 0.6], [0.7, 0.8], [0.4, 0.4]])
        X3 = torch.empty(2, 0)
        Y = torch.tensor([1.0, 0.0, 0.0])  # True outcomes
        odds1 = torch.tensor([2.0, 3.0, 2.0])
        odds2 = torch.tensor([1.5, 4.0, 1.1])
        fighter_names = ["fighter_1", "fighter_2", "fighter_1"]
        opponent_names = ["fighter_2", "fighter_1", "fighter_2"]

        # Mock model that returns constant prediction values
        mock_model = MagicMock()
        mock_model.side_effect = [
            torch.tensor([0.8, 0.7, 0.1]),  # predictions_1
            torch.tensor([0.2, 0.3, 0.9]),  # predictions_2 (1 - model)
        ]

        # We mock to to return the same mock
        mock_model.to = MagicMock(return_value=mock_model)

        # Call the function with mock data
        PredictionPlots.show_fight_prediction_detail(
            model=mock_model,
            data=((X1, X2, X3), Y, (odds1, odds2), fighter_names, opponent_names),
            print_info=False,
            show_plot=True,
            ax=mock_ax,  # Use the mocked axis
        )

        expected_invest = [
            (0.8 - 0.5) * 10 * 2,
            (0.7 - 0.5) * 10 * 2,
            (0.5 - 0.1) * 10 * 2,
        ]

        expected_earnings = [
            expected_invest[0] * odds2[0],
            0,
            expected_invest[2] * odds1[2],
        ]

        # Check the data passed to ax.plot
        plot_args, _ = mock_ax.plot.call_args

        # Cumulative investment and earnings
        invest_progress = plot_args[0]  # np.cumsum(invest_progress)
        earnings_progress = plot_args[
            1
        ]  # (np.cumsum(earning_progress) - np.cumsum(invest_progress)) / np.cumsum(invest_progress) * 100

        # Check if investment and earnings values match expected values
        expected_invest_progress = np.cumsum(
            expected_invest
        )  # bet values are expected to be 10 for each
        expected_earning_progress = (
            (np.cumsum(expected_earnings) - expected_invest_progress)
            / expected_invest_progress
            * 100
        )

        np.testing.assert_array_equal(invest_progress, expected_invest_progress)
        np.testing.assert_almost_equal(earnings_progress, expected_earning_progress)

        # Check that the axhline is called (for baseline at 0)
        mock_ax.axhline.assert_called_with(0, c="k")

    @patch("matplotlib.pyplot.subplots")
    def test_show_fight_prediction_detail_from_dataset(self, mock_subplots):
        # Create mock axis object
        mock_ax = MagicMock()
        mock_subplots.return_value = (MagicMock(), mock_ax)

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.get_fight_data_from_ids.return_value = (
            (
                torch.tensor([[0.1, 0.2], [0.3, 0.4]]),  # X1
                torch.tensor([[0.5, 0.6], [0.7, 0.8]]),  # X2
                torch.empty(2, 0),  # X3
            ),
            torch.tensor([1.0, 0.0]),  # Y
            (  # odds
                torch.tensor([2.0, 3.0]),  # odds1
                torch.tensor([1.5, 4.0]),  # odds2
            ),
            ["fighter_1", "fighter_2"],  # fighter_names
            ["opponent_1", "opponent_2"],  # opponent_names
        )

        # Mock model that returns constant prediction values
        mock_model = MagicMock()
        mock_model.side_effect = [
            torch.tensor([0.8, 0.7]),  # predictions_1
            torch.tensor([0.2, 0.3]),  # predictions_2
        ]

        # We mock to to return the same mock
        mock_model.to = MagicMock(return_value=mock_model)

        # Call the function with the mocked dataset
        PredictionPlots.show_fight_prediction_detail_from_dataset(
            model=mock_model,
            dataset=mock_dataset,
            fight_ids=["fight_1", "fight_2"],
            print_info=False,
            show_plot=True,
            ax=mock_ax,  # Use the mocked axis
        )

        # Verify that get_fight_data_from_ids was called correctly
        mock_dataset.get_fight_data_from_ids.assert_called_once_with(
            ["fight_1", "fight_2"]
        )

        # Verify that show_fight_prediction_detail was called with the expected arguments
        self.assertEqual(mock_model.call_count, 2)  # Model should be called twice


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
