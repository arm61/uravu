"""
Tests for sampling module
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import unittest
import numpy as np
from numpy.testing import assert_equal
from scipy.stats import norm
from uravu import sampling, utils
from uravu import relationship
from uravu.distribution import Distribution
import scipy.stats


TEST_Y = []
for i in np.arange(1, 9, 1):
    TEST_Y.append(Distribution(scipy.stats.norm.rvs(loc=i, scale=0.5, size=200)))
TEST_X = np.arange(1, 9, 1)

class TestSampling(unittest.TestCase):
    """
    Unit tests for optimize module.
    """

    def test_mcmc(self):
        test_rel = relationship.Relationship(
            utils.straight_line, TEST_X, TEST_Y 
        )
        test_rel.max_likelihood('mini')
        actual_results = sampling.mcmc(test_rel, n_burn=10, n_samples=10)
        assert_equal(isinstance(actual_results["distributions"][0], Distribution), True)
        assert_equal(isinstance(actual_results["distributions"][1], Distribution), True)
        assert_equal(actual_results["distributions"][0].size, 500)
        assert_equal(actual_results["distributions"][1].size, 500)

    def test_mcmc_with_other_prior(self):
        test_rel = relationship.Relationship(
            utils.straight_line, TEST_X, TEST_Y 
        )
        test_rel.max_likelihood('mini')

        def other_prior():
            """
            Another potential prior.
            """
            priors = []
            for i, variable in enumerate(test_rel.variables):
                loc = variable.n
                scale = 1
                priors.append(norm(loc=loc, scale=scale))
            return priors

        actual_results = sampling.mcmc(
            test_rel, prior_function=other_prior, n_burn=10, n_samples=10
        )
        assert_equal(isinstance(actual_results["distributions"][0], Distribution), True)
        assert_equal(isinstance(actual_results["distributions"][1], Distribution), True)
        assert_equal(actual_results["distributions"][0].size, 500)
        assert_equal(actual_results["distributions"][1].size, 500)

    def test_mcmc_with_variable_median_zero(self):
        test_rel = relationship.Relationship(
            utils.straight_line, TEST_X, TEST_Y 
        )
        test_rel.variables[0] = Distribution(np.zeros((7)))
        actual_results = sampling.mcmc(test_rel, n_burn=10, n_samples=10)
        assert_equal(isinstance(actual_results["distributions"][0], Distribution), True)
        assert_equal(isinstance(actual_results["distributions"][1], Distribution), True)
        assert_equal(actual_results["distributions"][0].size, 500)
        assert_equal(actual_results["distributions"][1].size, 500)

    def test_nested_sampling(self):
        test_rel = relationship.Relationship(
            utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        actual_results = sampling.nested_sampling(test_rel, maxiter=100)
        assert_equal(isinstance(actual_results, dict), True)
        assert_equal(isinstance(actual_results["logz"][-1], float), True)
        assert_equal(isinstance(actual_results["logzerr"][-1], float), True)

    def test_nested_sampling_b_with_other_prior(self):
        test_rel = relationship.Relationship(
            utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        test_rel.max_likelihood('mini')

        def other_prior():
            """
            Another potential prior.
            """
            priors = []
            for i, variable in enumerate(test_rel.variables):
                loc = variable.n
                scale = 1
                priors.append(norm(loc=loc, scale=scale))
            return priors

        actual_results = sampling.nested_sampling(
            test_rel, prior_function=other_prior, maxiter=100
        )
        assert_equal(isinstance(actual_results, dict), True)
        assert_equal(isinstance(actual_results["logz"][-1], float), True)
        assert_equal(isinstance(actual_results["logzerr"][-1], float), True)

    def test_dynamic_nested_sampling(self):
        test_rel = relationship.Relationship(
            utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        actual_results = sampling.nested_sampling(test_rel, dynamic=True, maxiter=100)
        assert_equal(isinstance(actual_results, dict), True)
        assert_equal(isinstance(actual_results["logz"][-1], float), True)
        assert_equal(isinstance(actual_results["logzerr"][-1], float), True)
