"""
Tests for optimize module
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import scipy.stats
from uravu import optimize, utils, relationship
from uravu.distribution import Distribution

TEST_Y = []
for i in np.arange(1, 9, 1):
    TEST_Y.append(Distribution(scipy.stats.norm.rvs(loc=i, scale=0.5, size=200)))
TEST_X = np.arange(1, 9, 1)

class TestOptimize(unittest.TestCase):
    """
    Unit tests for optimize module.
    """

    def test_ln_likelihood(self):
        test_rel = relationship.Relationship(
            utils.straight_line, TEST_X, TEST_Y 
        )
        expected_lnl = -3
        actual_lnl = optimize.ln_likelihood(
            [1., 0.], test_rel.function, test_rel.x, test_rel.y
        )
        assert_almost_equal(actual_lnl, expected_lnl, decimal=0)

    def test_negative_lnl(self):
        test_rel = relationship.Relationship(
            utils.straight_line, TEST_X, TEST_Y 
        )
        expected_negtive_lnl = 3
        actual_negative_lnl = optimize.negative_lnl(
            [1., 0.], test_rel.function, test_rel.x, test_rel.y
        )
        assert_almost_equal(actual_negative_lnl, expected_negtive_lnl, decimal=0)

    def test_max_lnlikelihood(self):
        test_rel = relationship.Relationship(
            utils.straight_line, TEST_X, TEST_Y 
        )
        actual_best_variables = optimize.max_ln_likelihood(test_rel, 'mini')
        assert_almost_equal(actual_best_variables, np.array([1, 0]), decimal=0)
