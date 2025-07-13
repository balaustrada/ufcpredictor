from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from ufcpredictor.data_aggregator import WeightedDataAggregator
from ufcpredictor.data_processor import DataProcessor
from ufcpredictor.data_enhancers import (
    OSR,
    WOSR,
    ELO,
    FlexibleELO,
    RankedFields,
    SumFlexibleELO,
)

THIS_DIR = Path(__file__).parent


class BaseTestDataProcessor(object):
    data_processor = DataProcessor
    init_kwargs = dict()

    def setUp(self):
        """Set up a mock data folder and create a DataProcessor instance."""
        # Mock the scrapers and their data
        ufc_scraper = MagicMock()
        bfo_scraper = MagicMock()

        self.processor = self.data_processor(
            data_folder=None,
            ufc_scraper=ufc_scraper,
            bfo_scraper=bfo_scraper,
            **self.init_kwargs,
        )

        # Create mock dataframes for different sources
        self.mock_fight_data = pd.DataFrame(
            {
                "fight_id": [1, 2],
                "fighter_1": ["f1", "f3"],
                "fighter_2": ["f2", "f4"],
                "event_id": [101, 102],
            }
        )

        self.mock_round_data = pd.DataFrame(
            {
                "fight_id": [1, 1, 2, 2],
                "round": [1, 1, 1, 1],
                "fighter_id": ["f1", "f2", "f3", "f4"],
                "ctrl_time": [5, 10, 5, 10],
            }
        )

        self.mock_fighter_data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f3", "f4"],
                "fighter_f_name": ["John", "Jane", "Jake", "Jill"],
                "fighter_l_name": ["Doe", "Doe", "Smith", "Brown"],
                "fighter_dob": [pd.Timestamp("1990-01-01")] * 4,
                "fighter_nickname": ["Johnny", "Jenny", "Jakey", None],
            }
        )

        self.mock_event_data = pd.DataFrame(
            {
                "event_id": [101, 102],
                "event_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01")],
            }
        )

        self.mock_odds_data = pd.DataFrame(
            {
                "fight_id": [1, 1, 2, 2],
                "fighter_id": ["f1", "f2", "f3", "f4"],
                "opening": [-150, 130, 200, -300],
            }
        )

        self.mock_replacement_data = pd.DataFrame(
            {
                "fight_id": [
                    1,
                ],
                "fighter_id": [
                    "f1",
                ],
                "notice_days": [
                    20,
                ],
            }
        )

        # Attach mock data to the scrapers
        self.processor.scraper.fight_scraper.data = self.mock_fight_data
        self.processor.scraper.fight_scraper.rounds_handler.data = self.mock_round_data
        self.processor.scraper.fighter_scraper.data = self.mock_fighter_data
        self.processor.scraper.event_scraper.data = self.mock_event_data
        self.processor.scraper.replacement_scraper.data = self.mock_replacement_data
        self.processor.bfo_scraper.data = self.mock_odds_data

    def test_get_fighter_name_and_id(self):
        self.processor.fighter_names = dict(f1="fighter1", f2="fighter2")
        self.processor.fighter_ids = dict(fighter1="f1", fighter2="f2")

        self.assertEqual(self.processor.get_fighter_name("f1"), "fighter1")
        self.assertEqual(self.processor.get_fighter_id("fighter1"), "f1")
        self.assertEqual(self.processor.get_fighter_id("fighter2a"), "f2")

    def test_raise_error_if_data_folder_is_none(self):
        """Test that raise_error_if_data_folder_is_none raises an error if data_folder is None."""
        with self.assertRaises(ValueError):
            self.data_processor(data_folder=None, ufc_scraper=None, bfo_scraper=None)

    def test_load_data_calls_all_methods(self):
        methods_to_patch = [
            "join_dataframes",
            "fix_date_and_time_fields",
            "convert_odds_to_decimal",
            "fill_weight",
            "add_key_stats",
            "apply_filters",
            "group_round_data",
        ]

        with patch.multiple(
            self.data_processor, **{method: MagicMock() for method in methods_to_patch}
        ):
            self.processor.load_data()

            for method in methods_to_patch:
                getattr(self.processor, method).assert_called_once()

    def test_join_dataframes(self):
        """Test that join_dataframes correctly joins fight, fighter, event, and odds data."""
        result = self.processor.join_dataframes()

        # Verify the columns exist and the join was successful
        self.assertIn("fight_id", result.columns)
        self.assertIn("fighter_id", result.columns)
        self.assertIn("opponent_id", result.columns)
        self.assertIn("opening", result.columns)  # From odds
        self.assertIn("fighter_name", result.columns)  # From fighter data
        self.assertEqual(len(result), 4)  # Should duplicate rows to 4

    def test_fix_date_and_time_fields(self):
        """Test that fix_date_and_time_fields converts fields correctly."""
        data = pd.DataFrame(
            {
                "ctrl_time": ["1:30", "2:00"],
                "ctrl_time_opponent": ["0:30", "3:00"],
                "finish_round": [2, 3],
                "finish_time": ["2:00", "3:00"],
                "event_date": ["2020-01-01", "2020-06-01"],
                "fighter_id": ["f1", "f2"],
                "fighter_dob": ["1990-01-01", "1995-01-01"],
            }
        )

        result = self.processor.fix_date_and_time_fields(data)
        self.assertEqual(result["ctrl_time"].iloc[0], 90)  # 1.5 minutes to seconds
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["event_date"]))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["fighter_dob"]))

    def test_convert_odds_to_decimal(self):
        """Test that convert_odds_to_decimal correctly converts odds."""
        data = pd.DataFrame(
            {
                "opening": [-150, 200],
                "closing_range_min": [-120, 250],
                "closing_range_max": [-110, -105],
            }
        )

        result = self.processor.convert_odds_to_decimal(data)

        self.assertAlmostEqual(result["opening"].iloc[0], 1.6667, places=4)
        self.assertAlmostEqual(result["closing_range_min"].iloc[1], 3.5, places=4)

    def test_fill_weight(self):
        """Test that fill_weight adds weights based on weight class."""
        data = pd.DataFrame(
            {
                "weight_class": ["Lightweight", "NULL", "Catch Weight", "Heavyweight"],
            }
        )

        result = self.processor.fill_weight(data)
        self.assertNotIn("NULL", result["weight_class"].values)
        self.assertNotIn("Catch Weight", result["weight_class"].values)
        self.assertEqual(result["weight"].iloc[0], 155)  # Assuming Lightweight is 155

    def test_add_key_stats(self):
        """Test that add_key_stats correctly adds KO, Submission, and Win stats."""
        data = pd.DataFrame(
            {
                "result": ["KO", "Submission", "Decision"],
                "winner": ["f1", "f2", "f1"],
                "fighter_id": ["f1", "f2", "f3"],
                "event_date": [
                    pd.Timestamp("2020-01-01"),
                    pd.Timestamp("2020-06-01"),
                    pd.Timestamp("2020-01-01"),
                ],
                "fighter_dob": [
                    pd.Timestamp("1990-01-01"),
                    pd.Timestamp(
                        "1995-01-01",
                    ),
                    pd.Timestamp("1990-01-01"),
                ],
            }
        )

        result = self.processor.add_key_stats(data)
        self.assertEqual(result["KO"].sum(), 1)
        self.assertEqual(result["Sub"].sum(), 1)
        self.assertEqual(result["win"].sum(), 2)

        expected_ages = [
            30.019178,
            25.432877,
            30.019178,
        ]  # Expected ages at the time of events
        np.testing.assert_array_almost_equal(result["age"], expected_ages)

    def test_apply_filters(self):
        """Test that apply_filters correctly filters out unwanted data."""
        data = pd.DataFrame(
            {
                "event_date": [pd.Timestamp("2009-01-01"), pd.Timestamp("2007-01-01")],
                "time_format": ["3 Rnd (5-5-5)", "3 Rnd (5-5-5)"],
                "gender": ["M", "F"],
                "winner": ["f1", "f2"],
                "result": ["Decision", "KO/TKO"],
            }
        )

        result = self.processor.apply_filters(data)
        self.assertEqual(len(result), 1)  # Only 1 match should remain after filters

    def test_round_stat_names(self):
        """Test the round_stat_names property."""
        # Mock the data inside scraper's fight_scraper.rounds_handler.dtypes.keys()
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys = MagicMock()
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys.return_value = [
            "fight_id",
            "fighter_id",
            "round",
            "strikes",
            "takedowns",
        ]

        expected_stat_names = [
            "strikes",
            "takedowns",
            "strikes_opponent",
            "takedowns_opponent",
        ]

        result = self.processor.round_stat_names
        self.assertEqual(result, expected_stat_names)

    def test_stat_names(self):
        """Test the stat_names property."""
        # Mock the round_stat_names property result
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys = MagicMock()
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys.return_value = [
            "fight_id",
            "fighter_id",
            "round",
            "strikes",
            "takedowns",
        ]

        expected_stat_names = [
            "strikes",
            "takedowns",
            "strikes_opponent",
            "takedowns_opponent",
            "KO",
            "KO_opponent",
            "Sub",
            "Sub_opponent",
            "win",
            "win_opponent",
        ]

        result = self.processor.stat_names
        self.assertEqual(result, expected_stat_names)

    def test_normalized_fields(self):
        """Test the normalized_fields property."""
        # Mock the aggregated_fields result
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys = MagicMock()
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys.return_value = [
            "fight_id",
            "fighter_id",
            "round",
            "strikes",
            "takedowns",
        ]

        expected_normalized_fields = [
            "age",
            "time_since_last_fight",
            "fighter_height_cm",
            "weight",
            "strikes",
            "strikes_per_minute",
            "strikes_per_fight",
            "takedowns",
            "takedowns_per_minute",
            "takedowns_per_fight",
            "strikes_opponent",
            "strikes_opponent_per_minute",
            "strikes_opponent_per_fight",
            "takedowns_opponent",
            "takedowns_opponent_per_minute",
            "takedowns_opponent_per_fight",
            "KO",
            "KO_per_minute",
            "KO_per_fight",
            "KO_opponent",
            "KO_opponent_per_minute",
            "KO_opponent_per_fight",
            "Sub",
            "Sub_per_minute",
            "Sub_per_fight",
            "Sub_opponent",
            "Sub_opponent_per_minute",
            "Sub_opponent_per_fight",
            "win",
            "win_per_minute",
            "win_per_fight",
            "win_opponent",
            "win_opponent_per_minute",
            "win_opponent_per_fight",
        ]

        result = self.processor.normalized_fields
        self.assertEqual(result, expected_normalized_fields)

    def test_group_round_data(self):
        """Test the group_round_data method."""
        data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f1", "f2", "f2"],
                "fight_id": ["1", "1", "1", "1"],
                "round": [1, 2, 1, 2],
                "strikes": [10, 15, 5, 1],
                "takedowns": [2, 3, 1, 0],
                "event_date": [
                    pd.Timestamp("2020-01-01"),
                ]
                * 4,
            }
        )

        # Mock round_stat_names
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys = MagicMock()
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys.return_value = [
            "fight_id",
            "fighter_id",
            "round",
            "strikes",
            "takedowns",
        ]

        result = self.processor.group_round_data(data)

        expected = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2"],
                "fight_id": ["1", "1"],
                "event_date": [
                    pd.Timestamp("2020-01-01"),
                ]
                * 2,
                "strikes": [25, 6],
                "takedowns": [5, 1],
            }
        ).sort_values(by=["fighter_id"])

        pd.testing.assert_frame_equal(result, expected)


class TestDataProcessor(BaseTestDataProcessor, unittest.TestCase):
    def test_aggregate_data(self):
        """Test the aggregate_data method."""
        data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f1", "f2"],
                "fight_id": ["1", "1", "2", "2"],
                "event_date": pd.to_datetime(
                    [
                        "2020-01-01",
                    ]
                    * 2
                    + [
                        "2020-01-02",
                    ]
                    * 2
                ),
                "total_time": [5, 5, 2, 2],
                "strikes": [10, 5, 5, 10],
                "strikes_opponent": [5, 10, 10, 5],
                "takedowns": [2, 1, 1, 2],
                "takedowns_opponent": [1, 2, 2, 1],
                "KO": [1, 0, 0, 0],
                "KO_opponent": [0, 1, 0, 0],
                "Sub": [0, 0, 0, 1],
                "Sub_opponent": [0, 0, 1, 0],
                "win": [1, 0, 0, 1],
                "win_opponent": [0, 1, 1, 0],
            }
        )

        self.processor.data = data.copy()

        # Mock aggregated_fields
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys = MagicMock()
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys.return_value = [
            "fight_id",
            "fighter_id",
            "round",
            "strikes",
            "takedowns",
        ]

        self.processor.aggregate_data()

        expected = data.copy()
        expected["num_fight"] = [1, 1, 2, 2]
        expected["previous_fight_date"] = pd.to_datetime(
            [pd.NaT, pd.NaT, "2020-01-01", "2020-01-01"]
        )
        expected["time_since_last_fight"] = [np.nan, np.nan, 1, 1]
        expected["strikes"] = [10.0, 5.0, 15.0, 15.0]
        expected["strikes_opponent"] = [5.0, 10.0, 15.0, 15.0]
        expected["takedowns"] = [2.0, 1.0, 3.0, 3.0]
        expected["takedowns_opponent"] = [1.0, 2.0, 3.0, 3.0]
        expected["total_time"] = [5, 5, 7, 7]
        expected["KO"] = [1.0, 0.0, 1.0, 0.0]
        expected["KO_opponent"] = [0.0, 1.0, 0.0, 1.0]
        expected["Sub"] = [0.0, 0.0, 0.0, 1.0]
        expected["Sub_opponent"] = [0.0, 0.0, 1.0, 0.0]
        expected["win"] = [
            1.0,
            0.0,
            1.0,
            1.0,
        ]
        expected["win_opponent"] = [0.0, 1.0, 1.0, 1.0]
        expected["weighted_total_time"] = expected["total_time"]
        expected["weighted_num_fight"] = expected["num_fight"]

        pd.testing.assert_frame_equal(self.processor.data_aggregated, expected)

    def test_add_per_minute_and_fight_stats(self):
        """Test the add_per_minute_and_fight_stats method."""
        data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f1", "f2"],
                "fight_id": ["1", "1", "2", "2"],
                "event_date": pd.to_datetime(
                    [
                        "2020-01-01",
                    ]
                    * 2
                    + [
                        "2020-01-02",
                    ]
                    * 2
                ),
                "total_time": [5, 5, 2, 2],
                "strikes": [10, 5, 5, 10],
                "strikes_opponent": [5, 10, 10, 5],
                "takedowns": [2, 1, 1, 2],
                "takedowns_opponent": [1, 2, 2, 1],
                "KO": [1, 0, 0, 0],
                "KO_opponent": [0, 1, 0, 0],
                "Sub": [0, 0, 0, 1],
                "Sub_opponent": [0, 0, 1, 0],
                "win": [1, 0, 0, 1],
                "win_opponent": [0, 1, 1, 0],
            }
        )

        self.processor.data = data.copy()

        # Mock aggregated_fields
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys = MagicMock()
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys.return_value = [
            "fight_id",
            "fighter_id",
            "round",
            "strikes",
            "takedowns",
        ]

        self.processor.aggregate_data()

        expected = self.processor.data_aggregated.copy()
        self.processor.add_per_minute_and_fight_stats()

        expected["strikes_per_minute"] = [10.0 / 5, 5.0 / 5, 15.0 / 7, 15.0 / 7]
        expected["strikes_per_fight"] = [10.0, 5.0, 15.0 / 2, 15.0 / 2]
        expected["takedowns_per_minute"] = [2.0 / 5, 1.0 / 5, 3.0 / 7, 3.0 / 7]
        expected["takedowns_per_fight"] = [2.0, 1.0, 3.0 / 2, 3.0 / 2]
        expected["strikes_opponent_per_minute"] = [
            5.0 / 5,
            10.0 / 5,
            15.0 / 7,
            15.0 / 7,
        ]
        expected["strikes_opponent_per_fight"] = [5.0, 10.0, 15.0 / 2, 15.0 / 2]
        expected["takedowns_opponent_per_minute"] = [1.0 / 5, 2.0 / 5, 3.0 / 7, 3.0 / 7]
        expected["takedowns_opponent_per_fight"] = [1.0, 2.0, 3.0 / 2, 3.0 / 2]
        expected["KO_per_minute"] = [1.0 / 5, 0.0 / 5, 1.0 / 7, 0.0 / 7]
        expected["KO_per_fight"] = [1.0, 0.0, 1.0 / 2, 0.0 / 2]
        expected["KO_opponent_per_minute"] = [0.0 / 5, 1.0 / 5, 0.0 / 7, 1.0 / 7]
        expected["KO_opponent_per_fight"] = [0.0, 1.0, 0.0 / 2, 1.0 / 2]
        expected["Sub_per_minute"] = [0.0 / 5, 0.0 / 5, 0.0 / 7, 1.0 / 7]
        expected["Sub_per_fight"] = [0.0, 0.0, 0.0 / 2, 1.0 / 2]
        expected["Sub_opponent_per_minute"] = [0.0 / 5, 0.0 / 5, 1.0 / 7, 0.0 / 7]
        expected["Sub_opponent_per_fight"] = [0.0, 0.0, 1.0 / 2, 0.0 / 2]
        expected["win_per_minute"] = [1.0 / 5, 0.0 / 5, 1.0 / 7, 1.0 / 7]
        expected["win_per_fight"] = [1.0, 0.0, 1.0 / 2, 1.0 / 2]
        expected["win_opponent_per_minute"] = [0.0 / 5, 1.0 / 5, 1.0 / 7, 1.0 / 7]
        expected["win_opponent_per_fight"] = [0.0, 1.0, 1.0 / 2, 1.0 / 2]

        pd.testing.assert_frame_equal(self.processor.data_aggregated, expected)

    def test_normalize_data(self):
        """Test the normalize_data method."""
        data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f1", "f2"],
                "fighter_height_cm": [185, 190, 185, 190],
                "fight_id": ["1", "1", "2", "2"],
                "event_date": pd.to_datetime(
                    [
                        "2020-01-01",
                    ]
                    * 2
                    + [
                        "2020-01-02",
                    ]
                    * 2
                ),
                "total_time": [5, 5, 2, 2],
                "strikes": [10, 5, 5, 10],
                "strikes_opponent": [5, 10, 10, 5],
                "takedowns": [2, 1, 1, 2],
                "takedowns_opponent": [1, 2, 2, 1],
                "KO": [1, 0, 0, 0],
                "KO_opponent": [0, 1, 0, 0],
                "Sub": [0, 0, 0, 1],
                "Sub_opponent": [0, 0, 1, 0],
                "win": [1, 0, 0, 1],
                "win_opponent": [0, 1, 1, 0],
                "age": [30, 25, 30, 25],
                "weight": [145, 145, 145, 145],
            }
        )

        self.processor.data = data.copy()

        # Mock aggregated_fields
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys = MagicMock()
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys.return_value = [
            "fight_id",
            "fighter_id",
            "round",
            "strikes",
            "takedowns",
        ]

        self.processor.aggregate_data()

        expected = self.processor.data_aggregated.copy()
        self.processor.add_per_minute_and_fight_stats()
        self.processor.normalize_data()

        output = self.processor.data_normalized

        np.testing.assert_almost_equal(output["KO"].values, [2, 0, 2, 0])
        np.testing.assert_almost_equal(
            output["strikes_per_minute"].values,
            [1.09803922, 0.54901961, 1.17647059, 1.17647059],
        )
        np.testing.assert_almost_equal(
            output["win_per_fight"].values,
            [
                2,
                0,
                1,
                1,
            ],
        )


class TestOSRDataProcessor(BaseTestDataProcessor, unittest.TestCase):
    init_kwargs = {
        "data_enhancers": [
            OSR(),
        ],
    }

    def test_aggregate_data(self):
        """Test the aggregate_data method."""
        data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f1", "f2"],
                "opponent_id": ["f2", "f1", "f2", "f1"],
                "fight_id": ["1", "1", "2", "2"],
                "event_date": pd.to_datetime(
                    [
                        "2020-01-01",
                    ]
                    * 2
                    + [
                        "2020-01-02",
                    ]
                    * 2
                ),
                "total_time": [5, 5, 2, 2],
                "strikes": [10, 5, 5, 10],
                "strikes_opponent": [5, 10, 10, 5],
                "takedowns": [2, 1, 1, 2],
                "takedowns_opponent": [1, 2, 2, 1],
                "KO": [1, 0, 0, 0],
                "KO_opponent": [0, 1, 0, 0],
                "Sub": [0, 0, 0, 1],
                "Sub_opponent": [0, 0, 1, 0],
                "win": [1, 0, 0, 1],
                "win_opponent": [0, 1, 1, 0],
            }
        )

        self.processor.data = data.copy()

        # Mock aggregated_fields
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys = MagicMock()
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys.return_value = [
            "fight_id",
            "fighter_id",
            "round",
            "strikes",
            "takedowns",
        ]

        self.processor.aggregate_data()

        expected = data.copy()
        expected["num_fight"] = [1, 1, 2, 2]
        expected["previous_fight_date"] = pd.to_datetime(
            [pd.NaT, pd.NaT, "2020-01-01", "2020-01-01"]
        )
        expected["time_since_last_fight"] = [np.nan, np.nan, 1, 1]
        expected["strikes"] = [10.0, 5.0, 15.0, 15.0]
        expected["strikes_opponent"] = [5.0, 10.0, 15.0, 15.0]
        expected["takedowns"] = [2.0, 1.0, 3.0, 3.0]
        expected["takedowns_opponent"] = [1.0, 2.0, 3.0, 3.0]
        expected["total_time"] = [5, 5, 7, 7]
        expected["KO"] = [1.0, 0.0, 1.0, 0.0]
        expected["KO_opponent"] = [0.0, 1.0, 0.0, 1.0]
        expected["Sub"] = [0.0, 0.0, 0.0, 1.0]
        expected["Sub_opponent"] = [0.0, 0.0, 1.0, 0.0]
        expected["win"] = [
            1.0,
            0.0,
            1.0,
            1.0,
        ]
        expected["win_opponent"] = [0.0, 1.0, 1.0, 1.0]
        expected["weighted_total_time"] = expected["total_time"]
        expected["weighted_num_fight"] = expected["num_fight"]
        expected["OSR"] = [1.0, 0.0, 0.5, 0.5]

        pd.testing.assert_frame_equal(self.processor.data_aggregated, expected)


class TestWOSRDataProcessor(BaseTestDataProcessor, unittest.TestCase):
    init_kwargs = {
        "data_enhancers": [
            WOSR(weights=[0.1, 0.2, 0.7]),
        ],
    }

    def test_aggregate_data(self):
        """Test the aggregate_data method."""
        data = pd.DataFrame(
            {
                "fighter_id": ["f1", "f2", "f1", "f2"],
                "opponent_id": ["f2", "f1", "f2", "f1"],
                "fight_id": ["1", "1", "2", "2"],
                "event_date": pd.to_datetime(
                    [
                        "2020-01-01",
                    ]
                    * 2
                    + [
                        "2020-01-02",
                    ]
                    * 2
                ),
                "total_time": [5, 5, 2, 2],
                "strikes": [10, 5, 5, 10],
                "strikes_opponent": [5, 10, 10, 5],
                "takedowns": [2, 1, 1, 2],
                "takedowns_opponent": [1, 2, 2, 1],
                "KO": [1, 0, 0, 0],
                "KO_opponent": [0, 1, 0, 0],
                "Sub": [0, 0, 0, 1],
                "Sub_opponent": [0, 0, 1, 0],
                "win": [1, 0, 0, 1],
                "win_opponent": [0, 1, 1, 0],
            }
        )

        self.processor.data = data.copy()

        # Mock aggregated_fields
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys = MagicMock()
        self.processor.scraper.fight_scraper.rounds_handler.dtypes.keys.return_value = [
            "fight_id",
            "fighter_id",
            "round",
            "strikes",
            "takedowns",
        ]

        self.processor.aggregate_data()

        expected = data.copy()
        expected["num_fight"] = [1, 1, 2, 2]
        expected["previous_fight_date"] = pd.to_datetime(
            [pd.NaT, pd.NaT, "2020-01-01", "2020-01-01"]
        )
        expected["time_since_last_fight"] = [np.nan, np.nan, 1, 1]
        expected["strikes"] = [10.0, 5.0, 15.0, 15.0]
        expected["strikes_opponent"] = [5.0, 10.0, 15.0, 15.0]
        expected["takedowns"] = [2.0, 1.0, 3.0, 3.0]
        expected["takedowns_opponent"] = [1.0, 2.0, 3.0, 3.0]
        expected["total_time"] = [5, 5, 7, 7]
        expected["KO"] = [1.0, 0.0, 1.0, 0.0]
        expected["KO_opponent"] = [0.0, 1.0, 0.0, 1.0]
        expected["Sub"] = [0.0, 0.0, 0.0, 1.0]
        expected["Sub_opponent"] = [0.0, 0.0, 1.0, 0.0]
        expected["win"] = [
            1.0,
            0.0,
            1.0,
            1.0,
        ]
        expected["win_opponent"] = [0.0, 1.0, 1.0, 1.0]
        expected["weighted_total_time"] = expected["total_time"]
        expected["weighted_num_fight"] = expected["num_fight"]
        expected["OSR"] = [1.0, 0.0, 0.25, 0.75]

        pd.testing.assert_frame_equal(self.processor.data_aggregated, expected)

    # def test_from_id_to_fight(self):
    #     # Mock the data attribute for the DataProcessor
    #     self.processor.data = pd.DataFrame({
    #         'fight_id': ['fight1', 'fight1', 'fight2', 'fight2'],
    #         'fighter_id': ['f1', 'f2', 'f1','f2'],
    #         'opponent_id': ['f2', 'f1', 'f2', 'f1'],
    #         'event_date': [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-01'), pd.Timestamp('2020-03-01'), pd.Timestamp('2020-03-01')],
    #         'winner': ['f1', 'f1', 'f2', 'f2'],
    #         'UFC_names': ['Fighter One', 'Fighter Two', 'Fighter One', 'Fighter Two'],
    #         'opponent_UFC_names': ['Fighter Two', 'Fighter One', 'Fighter Two', 'Fighter One'],
    #         'some_stat': [1, 2, 3, 6],
    #         'other_stat': [4, 5, 6, 9],
    #     })

    #     # Mock the bfo_scraper data
    #     self.processor.bfo_scraper = MagicMock()
    #     self.processor.bfo_scraper.data = pd.DataFrame({
    #         'fight_id': ['fight1', 'fight1', 'fight2', 'fight2'],
    #         'fighter_id': ['f1', 'f2', 'f1', 'f2'],
    #         'opening': [1.5, 2.0, 1.8, 3.0]
    #     })

    #     # Define the input set of stats to fetch
    #     fighter_fight_statistics = ['some_stat', 'other_stat']
    #     fight_id = 'fight1'

    #     # Call the method
    #     x1, x2, outcome = self.processor.from_id_to_fight(fighter_fight_statistics, fight_id)

    #     # Check the values of the returned tensors
    #     expected_x1 = [1, 4]  # fighter 'f1' stats
    #     expected_x2 = [2, 5]  # fighter 'f3' (opponent 'f2') stats
    #     expected_outcome = [1.0]  # 'f1' won, which is the winner in fight2

    #     # Assert that the returned tensors match the expected values
    #     self.assertTrue(torch.equal(x1, torch.FloatTensor(expected_x1)))
    #     self.assertTrue(torch.equal(x2, torch.FloatTensor(expected_x2)))
    #     self.assertTrue(torch.equal(outcome, torch.FloatTensor(expected_outcome)))

    # def test_from_id_to_fight_with_print_info(self):

    #     # Mock the data attribute for the DataProcessor
    #     self.processor.data = pd.DataFrame({
    #         'fight_id': ['fight1', 'fight2', 'fight3'],
    #         'fighter_id': ['f1', 'f2', 'f1'],
    #         'opponent_id': ['f2', 'f1', 'f3'],
    #         'event_date': [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-02-01'), pd.Timestamp('2020-03-01')],
    #         'winner': ['f1', 'f2', 'f1'],
    #         'UFC_names': ['Fighter One', 'Fighter Two', 'Fighter Three'],
    #         'opponent_UFC_names': ['Opponent One', 'Opponent Two', 'Opponent Three'],
    #         'some_stat': [1, 2, 3],
    #         'other_stat': [4, 5, 6]
    #     })

    #     # Mock the bfo_scraper data
    #     self.processor.bfo_scraper = MagicMock()
    #     self.processor.bfo_scraper.data = pd.DataFrame({
    #         'fight_id': ['fight1', 'fight2', 'fight3'],
    #         'fighter_id': ['f1', 'f2', 'f1'],
    #         'opening': [1.5, 2.0, 1.8]
    #     })

    #     # Define the input set of stats to fetch
    #     fighter_fight_statistics = ['some_stat', 'other_stat']
    #     fight_id = 'fight2'

    #     # Mock print function to check print output
    #     with unittest.mock.patch('builtins.print') as mock_print:
    #         # Call the method with print_info set to True
    #         self.processor.from_id_to_fight(fighter_fight_statistics, fight_id, print_info=True)

    #         # Check if the print function was called with the correct information
    #         mock_print.assert_any_call('Fighter Two', ' vs ', 'Opponent Two')
    #         mock_print.assert_any_call('2.0 vs 1.5')


class TestELODataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define the configuration for initializing ELODataProcessor
        cls.data_processor_kwargs = {
            "data_folder": THIS_DIR / "test_files",
            "data_aggregator": WeightedDataAggregator(),
            "data_enhancers": [
                ELO(),
                RankedFields(
                    fields=["age", "fighter_height_cm"],
                    exponents=[1.2, 4],
                ),
            ],
        }

        csv_files_path = THIS_DIR / "test_files" / "ELODataProcessor"
        # Expected CSV file paths
        cls.expected_data_path = csv_files_path / "expected_data.csv"
        cls.expected_aggregated_data_path = (
            csv_files_path / "expected_data_aggregated.csv"
        )
        cls.expected_normalized_data_path = (
            csv_files_path / "expected_data_normalized.csv"
        )

        # Initialize DataProcessor instance
        cls.data_processor = DataProcessor(**cls.data_processor_kwargs)

        # Process data through the necessary steps
        cls.data_processor.load_data()
        cls.data_processor.aggregate_data()
        cls.data_processor.add_per_minute_and_fight_stats()
        cls.data_processor.normalize_data()

    def compare_dataframes(self, df, expected_csv_path):
        """Helper method to compare a dataframe to an expected CSV file."""
        # Check if we should update the CSV files
        if os.getenv("UPDATE_TEST_FILES") == "True":  # pragma: no cover
            df.to_csv(expected_csv_path, index=False)
            # Load expected data

        expected_df = pd.read_csv(expected_csv_path)

        # Align dtypes and datetime formats for comparison
        for col in df.columns:
            if col in expected_df.columns:
                # Convert datetime columns to string format YYYY-MM-DD
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")
                    expected_df[col] = pd.to_datetime(expected_df[col]).dt.strftime(
                        "%Y-%m-%d"
                    )
                else:
                    # Align types
                    expected_df[col] = expected_df[col].astype(df[col].dtype)

        # Compare dataframes
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True), expected_df.reset_index(drop=True)
        )

    def test_data(self):
        """Test that data matches expected CSV output."""
        self.compare_dataframes(self.data_processor.data, self.expected_data_path)

    def test_data_aggregated(self):
        """Test that aggregated data matches expected CSV output."""
        self.compare_dataframes(
            self.data_processor.data_aggregated, self.expected_aggregated_data_path
        )

    def test_data_normalized(self):
        """Test that normalized data matches expected CSV output."""
        self.compare_dataframes(
            self.data_processor.data_normalized, self.expected_normalized_data_path
        )


class TestFlexibleELODataProcessor(TestELODataProcessor, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define the configuration for initializing DataProcessor
        cls.data_processor_kwargs = {
            "data_folder": THIS_DIR / "test_files",
            "data_aggregator": WeightedDataAggregator(),
            "data_enhancers": [
                FlexibleELO(
                    n_boost_bins=3,
                    boost_values=[1, 1.2, 1.4],
                ),
                RankedFields(
                    fields=["age", "fighter_height_cm"],
                    exponents=[1.2, 4],
                ),
            ],
        }

        csv_files_path = THIS_DIR / "test_files" / "FlexibleELODataProcessor"
        # Expected CSV file paths
        cls.expected_data_path = csv_files_path / "expected_data.csv"
        cls.expected_aggregated_data_path = (
            csv_files_path / "expected_data_aggregated.csv"
        )
        cls.expected_normalized_data_path = (
            csv_files_path / "expected_data_normalized.csv"
        )

        # Initialize DataProcessor instance
        cls.data_processor = DataProcessor(**cls.data_processor_kwargs)

        # Process data through the necessary steps
        cls.data_processor.load_data()
        cls.data_processor.aggregate_data()
        cls.data_processor.add_per_minute_and_fight_stats()
        cls.data_processor.normalize_data()


class TestSumFlexibleELODataProcessor(TestELODataProcessor, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define the configuration for initializing DataProcessor
        cls.data_processor_kwargs = {
            "data_folder": THIS_DIR / "test_files",
            "data_aggregator": WeightedDataAggregator(),
            "data_enhancers": [
                SumFlexibleELO(
                    scaling_factor=0.8,
                ),
                RankedFields(
                    fields=["age", "fighter_height_cm"],
                    exponents=[1.2, 4],
                ),
            ],
        }

        csv_files_path = THIS_DIR / "test_files" / "SumFlexibleELODataProcessor"
        # Expected CSV file paths
        cls.expected_data_path = csv_files_path / "expected_data.csv"
        cls.expected_aggregated_data_path = (
            csv_files_path / "expected_data_aggregated.csv"
        )
        cls.expected_normalized_data_path = (
            csv_files_path / "expected_data_normalized.csv"
        )

        # Initialize ELODataProcessor instance
        cls.data_processor = DataProcessor(**cls.data_processor_kwargs)

        # Process data through the necessary steps
        cls.data_processor.load_data()
        cls.data_processor.aggregate_data()
        cls.data_processor.add_per_minute_and_fight_stats()
        cls.data_processor.normalize_data()
