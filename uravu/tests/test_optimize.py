"""
Tests for optimize module
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from uravu import optimize, utils, relationship


class TestOptimize(unittest.TestCase):
    """
    Unit tests for optimize module.
    """

    def test_ln_likelihood_a(self):
        """
        Test ln_likelihood function.
        """
        test_x = np.linspace(0, 99, 10)
        test_y = np.ones(10)
        test_y_e = np.ones(10) * 0.1
        test_rel = relationship.Relationship(
            utils.straight_line, test_x, test_y, test_y_e
        )
        expected_lnl = -1724226.97414907
        actual_lnl = optimize.ln_likelihood(
            test_rel.variables, test_rel.function, test_rel.x, test_rel.y
        )
        assert_almost_equal(actual_lnl, expected_lnl)

    def test_ln_likelihood_b(self):
        """
        Test ln_likelihood function with no uncertainty.
        """
        test_x = np.linspace(0, 99, 10)
        test_y = np.ones(10)
        test_rel = relationship.Relationship(
            utils.straight_line, test_x, test_y
        )
        expected_lnl = -45.21054955719477
        actual_lnl = optimize.ln_likelihood(
            test_rel.variables, test_rel.function, test_rel.x, test_rel.y, unaccounted_uncertainty=test_rel.unaccounted_uncertainty
        )
        assert_almost_equal(actual_lnl, expected_lnl)

    def test_negative_lnl_a(self):
        """
        Test negative_lnl function.
        """
        test_x = np.linspace(0, 99, 10)
        test_y = np.ones(10)
        test_y_e = np.ones(10) * 0.1
        test_rel = relationship.Relationship(
            utils.straight_line, test_x, test_y, test_y_e
        )
        expected_negtive_lnl = 1724226.97414907
        actual_negative_lnl = optimize.negative_lnl(
            test_rel.variables, test_rel.function, test_rel.x, test_rel.y
        )
        assert_almost_equal(actual_negative_lnl, expected_negtive_lnl)

    def test_negative_lnl_b(self):
        """
        Test negative_lnl function with additional uncertainty.
        """
        test_x = np.linspace(0, 99, 10)
        test_y = np.ones(10)
        test_y_e = np.ones(10) * 0.1
        test_rel = relationship.Relationship(
            utils.straight_line,
            test_x,
            test_y,
            test_y_e,
            unaccounted_uncertainty=True,
        )
        expected_negtive_lnl = 45.21123241122563
        actual_negative_lnl = optimize.negative_lnl(
            test_rel.variables,
            test_rel.function,
            test_rel.x,
            test_rel.y,
            test_rel.unaccounted_uncertainty,
        )
        assert_almost_equal(actual_negative_lnl, expected_negtive_lnl)

    def test_max_lnlikelihood(self):
        """
        Test negative_lnl function.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(1, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = relationship.Relationship(
            utils.straight_line, test_x, test_y, test_y_e
        )
        actual_best_variables = optimize.max_ln_likelihood(test_rel)
        assert_almost_equal(actual_best_variables, np.array([2, 1]))
