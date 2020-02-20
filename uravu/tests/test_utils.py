"""
Tests for utils module
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from uravu import utils


class TestUtils(unittest.TestCase):
    """
    Tests for the relationship module and class.
    """

    def test_straight_line(self):
        """
        Test straight line function.
        """
        test_x = np.linspace(0, 9, 10)
        test_y = np.linspace(0, 18, 10)
        result_y = utils.straight_line(test_x, 2, 0)
        assert_almost_equal(result_y, test_y)
