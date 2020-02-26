"""
Tests for utils module
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import unittest
import numpy as np
from uncertainties import ufloat
from numpy.testing import assert_almost_equal, assert_equal
from uravu import utils
from uravu.relationship import Relationship


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

    def test_bayes_factor_a(self):
        """
        Test the bayes factor function.
        """
        model_1 = 2
        model_2 = 4
        expected_result = -4
        actual_result = utils.bayes_factor(model_1, model_2)
        assert_almost_equal(actual_result, expected_result)

    def test_bayes_factor_b(self):
        """
        Test the bayes factor function with uncertainties.
        """
        model_1 = ufloat(2, 1)
        model_2 = ufloat(4, 1)
        expected_result = ufloat(-4, 2.82842712474619032)
        actual_result = utils.bayes_factor(model_1, model_2)
        assert_almost_equal(actual_result.n, expected_result.n)
        assert_almost_equal(actual_result.s, expected_result.s)

    def test_correlation_matrix(self):
        """
        Test correlation_matrix function.
        """
        test_x = np.linspace(0, 99, 10)
        test_y = (
            np.linspace(1, 199, 10)
            + np.linspace(1, 199, 10) * np.random.randn(10) * 0.05
        )
        test_y_e = test_y * 0.2
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e,)
        test_rel.max_likelihood()
        test_rel.mcmc(n_burn=10, n_samples=10)
        actual_matrix = utils.correlation_matrix(test_rel)
        assert_equal(actual_matrix.shape, (2, 2))
        assert_almost_equal(actual_matrix[1, 0], actual_matrix[0, 1])
        assert_almost_equal(actual_matrix[0, 0], 1.0)
        assert_almost_equal(actual_matrix[1, 1], 1.0)
        assert_equal(test_rel.mcmc_done, True)
