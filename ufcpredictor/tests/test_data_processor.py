from __future__ import annotations

import unittest
from pathlib import Path
from shutil import rmtree
import torch
from torch import nn
import numpy as np

from ufcpredictor.loss_functions import BettingLoss


import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from pathlib import Path
from ufcpredictor.data_processor import DataProcessor  # Assuming the class is in 'your_module'
import datetime
import numpy as np
from unittest.mock import patch

THIS_DIR = Path(__file__).parent

@patch.object(DataProcessor, "__init__", lambda self, data_folder: None)
class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        """Set up a mock data folder and create a DataProcessor instance."""        
        # Mock the scrapers and their data
        ufc_scraper = MagicMock()
        bfo_scraper = MagicMock()

        self.processor = DataProcessor(
            data_folder = None,
            ufc_scraper = ufc_scraper,
            bfo_scraper = bfo_scraper,
        )


        # Create mock dataframes for different sources
        self.mock_fight_data = pd.DataFrame({
            'fight_id': [1, 2],
            'fighter_1': ['f1', 'f3'],
            'fighter_2': ['f2', 'f4'],
            'event_id': [101, 102],
        })

        self.mock_round_data = pd.DataFrame({
            'fight_id': [1, 1, 2, 2],
            'round': [1, 1, 1, 1],
            'fighter_id': ['f1', 'f2', 'f3', 'f4'],
            'ctrl_time': [5, 10, 5, 10],
        })

        self.mock_fighter_data = pd.DataFrame({
            'fighter_id': ['f1', 'f2', 'f3', 'f4'],
            'fighter_f_name': ['John', 'Jane', 'Jake', 'Jill'],
            'fighter_l_name': ['Doe', 'Doe', 'Smith', 'Brown'],
            'fighter_dob': [pd.Timestamp('1990-01-01')] * 4,
            'fighter_nickname': ['Johnny', 'Jenny', 'Jakey', None],
        })

        self.mock_event_data = pd.DataFrame({
            'event_id': [101, 102],
            'event_date': [pd.Timestamp('2020-01-01'), pd.Timestamp('2020-06-01')],
        })

        self.mock_odds_data = pd.DataFrame({
            'fight_id': [1, 1, 2, 2],
            'fighter_id': ['f1', 'f2', 'f3', 'f4'],
            'opening': [-150, 130, 200, -300],
        })

        # Attach mock data to the scrapers
        self.processor.scraper.fight_scraper.data = self.mock_fight_data
        self.processor.scraper.fight_scraper.rounds_handler.data = self.mock_round_data
        self.processor.scraper.fighter_scraper.data = self.mock_fighter_data
        self.processor.scraper.event_scraper.data = self.mock_event_data
        self.processor.bfo_scraper.data = self.mock_odds_data

    def test_join_dataframes(self):
        """Test that join_dataframes correctly joins fight, fighter, event, and odds data."""
        result = self.processor.join_dataframes()

        # Verify the columns exist and the join was successful
        self.assertIn('fight_id', result.columns)
        self.assertIn('fighter_id', result.columns)
        self.assertIn('opponent_id', result.columns)
        self.assertIn('opening', result.columns)  # From odds
        self.assertIn('fighter_name', result.columns)  # From fighter data
        self.assertEqual(len(result), 4)  # Should duplicate rows to 4

    def test_fix_date_and_time_fields(self):
        """Test that fix_date_and_time_fields converts fields correctly."""
        data = pd.DataFrame({
            'ctrl_time': ["1:30", "2:00"],
            'ctrl_time_opponent': ["0:30", "3:00"],
            'finish_round': [2, 3],
            'finish_time': ["2:00", "3:00"],
            'event_date': ['2020-01-01', '2020-06-01'],
            'fighter_id': ['f1', 'f2'],
            'fighter_dob': ['1990-01-01', '1995-01-01'],
        })

        result = self.processor.fix_date_and_time_fields(data)
        self.assertEqual(result['ctrl_time'].iloc[0], 90)  # 1.5 minutes to seconds
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['event_date']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['fighter_dob']))

    def test_convert_odds_to_european(self):
        """Test that convert_odds_to_european correctly converts odds."""
        data = pd.DataFrame({
            'opening': [-150, 200],
            'closing_range_min': [-120, 250],
            'closing_range_max': [-110, -105],
        })

        result = self.processor.convert_odds_to_european(data)

        self.assertAlmostEqual(result['opening'].iloc[0], 1.6667, places=4)
        self.assertAlmostEqual(result['closing_range_min'].iloc[1], 3.5, places=4)

    def test_fill_weight(self):
        """Test that fill_weight adds weights based on weight class."""
        data = pd.DataFrame({
            'weight_class': ['Lightweight', 'NULL', 'Catch Weight', 'Heavyweight'],
        })

        result = self.processor.fill_weight(data)
        self.assertNotIn('NULL', result['weight_class'].values)
        self.assertNotIn('Catch Weight', result['weight_class'].values)
        self.assertEqual(result['weight'].iloc[0], 155)  # Assuming Lightweight is 155

    def test_add_key_stats(self):
        """Test that add_key_stats correctly adds KO, Submission, and Win stats."""
        data = pd.DataFrame({
            'result': ['KO', 'Submission', 'Decision'],
            'winner': ['f1', 'f2', 'f1'],
            'fighter_id': ['f1', 'f2', 'f3'],
            'event_date': [
                pd.Timestamp('2020-01-01'), 
                pd.Timestamp('2020-06-01'),
                pd.Timestamp('2020-01-01'),
            ],
            'fighter_dob': [
                pd.Timestamp('1990-01-01'),
                pd.Timestamp('1995-01-01',),
                pd.Timestamp('1990-01-01'),
            ],
        })

        result = self.processor.add_key_stats(data)
        self.assertEqual(result['KO'].sum(), 1)
        self.assertEqual(result['Sub'].sum(), 1)
        self.assertEqual(result['win'].sum(), 2)

        expected_ages = [30.019178, 25.432877, 30.019178]  # Expected ages at the time of events
        np.testing.assert_array_almost_equal(result['age'], expected_ages)

    def test_apply_filters(self):
        """Test that apply_filters correctly filters out unwanted data."""
        data = pd.DataFrame({
            'event_date': [pd.Timestamp('2009-01-01'), pd.Timestamp('2007-01-01')],
            'time_format': ['3 Rnd (5-5-5)', '3 Rnd (5-5-5)'],
            'gender': ['M', 'F'],
            'result': ['Decision', 'KO/TKO'],
        })

        result = self.processor.apply_filters(data)
        self.assertEqual(len(result), 1)  # Only 1 match should remain after filters

    def test_round_stat_names(self):
        """Test the round_stat_names property."""
        # Mock the data inside scraper's fight_scraper.rounds_handler.columns
        self.processor.scraper.fight_scraper.rounds_handler.columns = [
            'fight_id', 'fighter_id', 'round', 'strikes', 'takedowns'
        ]

        expected_stat_names = ['strikes', 'takedowns', 'strikes_opponent', 'takedowns_opponent']
        
        result = self.processor.round_stat_names
        self.assertEqual(result, expected_stat_names)

    def test_stat_names(self):
        """Test the stat_names property."""
        # Mock the round_stat_names property result
        self.processor.scraper.fight_scraper.rounds_handler.columns = [
            'fight_id', 'fighter_id', 'round', 'strikes', 'takedowns'
        ]
        
        expected_stat_names = [
            'strikes', 'takedowns', 'strikes_opponent', 'takedowns_opponent',
            'KO', 'KO_opponent', 'Sub', 'Sub_opponent', 'win', 'win_opponent'
        ]
        
        result = self.processor.stat_names
        self.assertEqual(result, expected_stat_names)

    def test_aggregated_fields(self):
        """Test the aggregated_fields property."""
        # Mock the round_stat_names result
        self.processor.scraper.fight_scraper.rounds_handler.columns = [
            'fight_id', 'fighter_id', 'round', 'strikes', 'takedowns'
        ]
        
        expected_aggregated_fields = [
            'strikes', 'takedowns', 'strikes_opponent', 'takedowns_opponent',
            'KO', 'KO_opponent', 'Sub', 'Sub_opponent', 'win', 'win_opponent'
        ]
        
        result = self.processor.aggregated_fields
        self.assertEqual(result, expected_aggregated_fields)

    def test_normalized_fields(self):
        """Test the normalized_fields property."""
        # Mock the aggregated_fields result
        self.processor.scraper.fight_scraper.rounds_handler.columns = [
            'fight_id', 'fighter_id', 'round', 'strikes', 'takedowns'
        ]
        
        expected_normalized_fields = [
            'age', 'time_since_last_fight', 'fighter_height_cm',
            'strikes', 'strikes_per_minute', 'strikes_per_fight',
            'takedowns', 'takedowns_per_minute', 'takedowns_per_fight',
            'strikes_opponent', 'strikes_opponent_per_minute', 'strikes_opponent_per_fight',
            'takedowns_opponent', 'takedowns_opponent_per_minute', 'takedowns_opponent_per_fight',
            'KO', 'KO_per_minute', 'KO_per_fight',
            'KO_opponent', 'KO_opponent_per_minute', 'KO_opponent_per_fight',
            'Sub', 'Sub_per_minute', 'Sub_per_fight',
            'Sub_opponent', 'Sub_opponent_per_minute', 'Sub_opponent_per_fight',
            'win', 'win_per_minute', 'win_per_fight',
            'win_opponent', 'win_opponent_per_minute', 'win_opponent_per_fight'
        ]
        
        result = self.processor.normalized_fields
        self.assertEqual(result, expected_normalized_fields)

    def test_group_round_data(self):
        """Test the group_round_data method."""
        data = pd.DataFrame({
            'fighter_id': ['f1', 'f1', 'f2', 'f2'],
            'fight_id': ['1', '1', '1', '1'],
            'round': [1, 2, 1, 2],
            'strikes': [10, 15, 5, 1],
            'takedowns': [2, 3, 1, 0],
            'event_date': [pd.Timestamp('2020-01-01'),]*4,
        })

        # Mock round_stat_names
        self.processor.scraper.fight_scraper.rounds_handler.columns = [
            'fight_id', 'fighter_id', 'round', 'strikes', 'takedowns'
        ]
        
        result = self.processor.group_round_data(data)
        
        expected = pd.DataFrame({
            'fighter_id': ['f1', 'f2'],
            'fight_id': ['1', '1'],
            'event_date': [pd.Timestamp('2020-01-01'),]*2,
            'strikes': [25, 6],
            'takedowns': [5, 1]
        }).sort_values(by=["fighter_id"])

        pd.testing.assert_frame_equal(result, expected)

    def test_aggregate_data(self):
        """Test the aggregate_data method."""
        data = pd.DataFrame({
            'fighter_id': ['f1', 'f2', 'f1', 'f2'],
            'fight_id': ['1', '1', '2', '2'],
            'event_date': pd.to_datetime(['2020-01-01',]*2+['2020-01-02',]*2),
            'total_time': [5, 5, 2, 2],
            'strikes': [10, 5, 5, 10],
            'strikes_opponent': [5, 10, 10, 5],
            'takedowns': [2, 1, 1, 2],
            'takedowns_opponent': [1, 2, 2, 1],
            'KO': [1, 0, 0, 0],
            'KO_opponent': [0, 1, 0, 0],
            'Sub': [0, 0, 0, 1],
            'Sub_opponent': [0, 0, 1, 0],
            'win': [1, 0, 0, 1],
            'win_opponent': [0, 1, 1, 0],
        })
        
        self.processor.data = data.copy()

        # Mock aggregated_fields
        self.processor.scraper.fight_scraper.rounds_handler.columns = [
            'fight_id', 'fighter_id', 'round', 'strikes', 'takedowns'
        ]

        self.processor.aggregate_data()
        
        expected = data.copy()
        expected['num_fight'] = [1, 1, 2, 2]
        expected['previous_fight_date'] = pd.to_datetime([pd.NaT, pd.NaT, '2020-01-01', '2020-01-01'])
        expected['time_since_last_fight'] = [np.nan, np.nan, 1, 1]
        expected['strikes'] = [10., 5., 15.0, 15.]
        expected['strikes_opponent'] = [5., 10., 15., 15.]
        expected['takedowns'] = [2., 1., 3., 3.]
        expected['takedowns_opponent'] = [1., 2., 3., 3.]
        expected['total_time'] = [5, 5, 7, 7]
        expected['KO'] = [1., 0., 1., 0.]
        expected['KO_opponent'] = [0., 1., 0., 1.]
        expected['Sub'] = [0., 0., 0., 1.]
        expected['Sub_opponent'] = [0., 0., 1., 0.]
        expected['win'] = [1., 0., 1., 1.,]
        expected['win_opponent'] = [0., 1., 1., 1.]

        
        pd.testing.assert_frame_equal(self.processor.data_aggregated, expected)

    def test_add_per_minute_and_fight_stats(self):
        """Test the add_per_minute_and_fight_stats method."""
        data = pd.DataFrame({
            'fighter_id': ['f1', 'f2', 'f1', 'f2'],
            'fight_id': ['1', '1', '2', '2'],
            'event_date': pd.to_datetime(['2020-01-01',]*2+['2020-01-02',]*2),
            'total_time': [5, 5, 2, 2],
            'strikes': [10, 5, 5, 10],
            'strikes_opponent': [5, 10, 10, 5],
            'takedowns': [2, 1, 1, 2],
            'takedowns_opponent': [1, 2, 2, 1],
            'KO': [1, 0, 0, 0],
            'KO_opponent': [0, 1, 0, 0],
            'Sub': [0, 0, 0, 1],
            'Sub_opponent': [0, 0, 1, 0],
            'win': [1, 0, 0, 1],
            'win_opponent': [0, 1, 1, 0],
        })
        
        self.processor.data = data.copy()

        # Mock aggregated_fields
        self.processor.scraper.fight_scraper.rounds_handler.columns = [
            'fight_id', 'fighter_id', 'round', 'strikes', 'takedowns'
        ]

        self.processor.aggregate_data()

        expected = self.processor.data_aggregated.copy()
        self.processor.add_per_minute_and_fight_stats()

        expected['strikes_per_minute'] = [10./5, 5./5, 15./7, 15./7]
        expected['strikes_per_fight'] = [10., 5., 15./2, 15./2]
        expected['takedowns_per_minute'] = [2./5, 1./5, 3./7, 3./7]
        expected['takedowns_per_fight'] = [2., 1., 3./2, 3./2]
        expected['strikes_opponent_per_minute'] = [5./5, 10./5, 15./7, 15./7]
        expected['strikes_opponent_per_fight'] = [5., 10., 15./2, 15./2]
        expected['takedowns_opponent_per_minute'] = [1./5, 2./5, 3./7, 3./7]
        expected['takedowns_opponent_per_fight'] = [1., 2., 3./2, 3./2]
        expected['KO_per_minute'] = [1./5, 0./5, 1./7, 0./7]
        expected['KO_per_fight'] = [1., 0., 1./2, 0./2]
        expected['KO_opponent_per_minute'] = [0./5, 1./5, 0./7, 1./7]
        expected['KO_opponent_per_fight'] = [0., 1., 0./2, 1./2]
        expected['Sub_per_minute'] = [0./5, 0./5, 0./7, 1./7]
        expected['Sub_per_fight'] = [0., 0., 0./2, 1./2]
        expected['Sub_opponent_per_minute'] = [0./5, 0./5, 1./7, 0./7]
        expected['Sub_opponent_per_fight'] = [0., 0., 1./2, 0./2]
        expected['win_per_minute'] = [1./5, 0./5, 1./7, 1./7]
        expected['win_per_fight'] = [1., 0., 1./2, 1./2]
        expected['win_opponent_per_minute'] = [0./5, 1./5, 1./7, 1./7]
        expected['win_opponent_per_fight'] = [0., 1., 1./2, 1./2]

        pd.testing.assert_frame_equal(self.processor.data_aggregated, expected)

    def test_normalize_data(self):
        """Test the normalize_data method."""
        data = pd.DataFrame({
            'fighter_id': ['f1', 'f2', 'f1', 'f2'],
            'fighter_height_cm': [185, 190, 185, 190],
            'fight_id': ['1', '1', '2', '2'],
            'event_date': pd.to_datetime(['2020-01-01',]*2+['2020-01-02',]*2),
            'total_time': [5, 5, 2, 2],
            'strikes': [10, 5, 5, 10],
            'strikes_opponent': [5, 10, 10, 5],
            'takedowns': [2, 1, 1, 2],
            'takedowns_opponent': [1, 2, 2, 1],
            'KO': [1, 0, 0, 0],
            'KO_opponent': [0, 1, 0, 0],
            'Sub': [0, 0, 0, 1],
            'Sub_opponent': [0, 0, 1, 0],
            'win': [1, 0, 0, 1],
            'win_opponent': [0, 1, 1, 0],
            'age': [30, 25, 30, 25],
        })
        
        self.processor.data = data.copy()

        # Mock aggregated_fields
        self.processor.scraper.fight_scraper.rounds_handler.columns = [
            'fight_id', 'fighter_id', 'round', 'strikes', 'takedowns'
        ]

        self.processor.aggregate_data()

        expected = self.processor.data_aggregated.copy()
        self.processor.add_per_minute_and_fight_stats()
        self.processor.normalize_data()

        output = self.processor.data_normalized

        np.testing.assert_almost_equal(
            output['KO'].values,
            [2, 0, 2, 0]
        )
        np.testing.assert_almost_equal(
            output['strikes_per_minute'].values,
            [1.09803922, 0.54901961, 1.17647059, 1.17647059],
        )
        np.testing.assert_almost_equal(
            output['win_per_fight'].values,
            [2, 0, 1, 1,]
        )

if __name__ == '__main__':
    unittest.main()
