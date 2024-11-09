from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ufcpredictor.utils import (
    convert_minutes_to_seconds,
    convert_odds_to_decimal,
    convert_odds_to_moneyline,
)

THIS_DIR = Path(__file__).parent


class TestUtils(unittest.TestCase):
    def test_convert_minutes_to_seconds(self):
        # Test valid input
        self.assertEqual(convert_minutes_to_seconds("2:30"), 150)
        self.assertEqual(convert_minutes_to_seconds("0:45"), 45)

        # Test edge cases
        self.assertEqual(convert_minutes_to_seconds("--"), 0)
        self.assertEqual(convert_minutes_to_seconds(None), None)
        self.assertEqual(convert_minutes_to_seconds("NULL"), None)
        self.assertEqual(convert_minutes_to_seconds(np.nan), None)

    def test_convert_odds_to_decimal(self):
        # Test conversion for positive odds (greater than 0)

        odds = [150, 200, 300]
        expected = [2.5, 3.0, 4.0]
        result = convert_odds_to_decimal(np.asarray(odds, dtype=np.float64))
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        # Test conversion for negative odds (less than 0)
        odds = [-150, -200, -300]
        expected = [1.66667, 1.5, 1.33333]
        result = convert_odds_to_decimal(np.asarray(odds, dtype=np.float64))
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        result = convert_odds_to_decimal(odds)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_convert_odds_to_moneyline(self):
        # Test conversion for decimal odds greater than 2
        odds = [2.5, 3.0, 4.0]
        expected = [150.0, 200.0, 300.0]
        result = convert_odds_to_moneyline(np.asarray(odds, dtype=np.float64))
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        # Test conversion for decimal odds less than or equal to 2
        odds = [5 / 3, 1.5, 12 / 9]
        expected = [-150.0, -200, -300]
        result = convert_odds_to_moneyline(np.asarray(odds, dtype=np.float64))
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

        result = convert_odds_to_moneyline(odds)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
