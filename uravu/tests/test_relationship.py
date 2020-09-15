"""
Tests for relationship module
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import unittest
import numpy as np
import scipy.stats
from uncertainties import unumpy as unp
from numpy.testing import assert_almost_equal, assert_equal
from uravu import utils
from uravu.relationship import Relationship
from uravu.distribution import Distribution
from uravu.axis import Axis


TEST_Y = []
for i in np.arange(1, 9, 1):
    TEST_Y.append(Distribution(scipy.stats.norm.rvs(loc=i, scale=0.5, size=200)))
TEST_X = np.arange(1, 9, 1)


class TestRelationship(unittest.TestCase):
    """
    Tests for the relationship module and class.
    """
    def test_function_init(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y)
        assert_equal(r.function, utils.straight_line)

    def test_abscissa_init(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y)
        assert_equal(isinstance(r.abscissa, np.ndarray), True)
        assert_equal(r.abscissa, TEST_X)

    def test_ordinate_init(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y)
        assert_equal(isinstance(r.ordinate, Axis), True)
        assert_equal(r.ordinate.values, TEST_Y)

    def test_ordinate_no_distribution(self):
        with self.assertRaises(ValueError):
            Relationship(utils.straight_line, TEST_X, TEST_X)

    def test_ordinate_and_ordinate_error(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_X, ordinate_error=[1]*len(TEST_X))
        assert_equal(r.x, TEST_X)
        assert_almost_equal(r.y.n, TEST_X, decimal=0)
        assert_almost_equal(r.y.s, np.ones((2, len(TEST_X))) * 1.96, decimal=1)

    def test_ordinate_stats(self):
        test_y = []
        for i in np.arange(1, 9, 1):
            test_y.append(scipy.stats.lognorm(i, 1, 1))
        r = Relationship(utils.straight_line, TEST_X, test_y)
        assert_equal(isinstance(r.ordinate, Axis), True)

    def test_ordinate_and_abscissa_different_length(self):
        with self.assertRaises(ValueError):
            Relationship(utils.straight_line, np.arange(1, 8, 1), TEST_Y)

    def test_ordinate_error_with_zero(self):
        with self.assertRaises(ValueError):
            Relationship(utils.straight_line, TEST_X, TEST_X, ordinate_error=np.linspace(0, 10, len(TEST_X)))

    def test_bounds_init(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        assert_equal(r.bounds, ((0, 10), (-1, 1)))

    def test_variables_init(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y)
        assert_almost_equal(r.variables[0].n, 1)
        assert_almost_equal(r.variables[1].n, 1)

    def test_bounds_init_wrong_number_a(self):
        with self.assertRaises(ValueError):
            Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1), (1, 2)))

    def test_bounds_init_wrong_number_b(self):
        with self.assertRaises(ValueError):
            Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10)))

    def test_variables_init_with_bounds(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        assert_equal(np.isclose(r.variables[0].n, 5, atol=0.75), True)
        assert_equal(np.isclose(r.variables[1].n, 0, atol=0.5), True)

    def test_ln_evidence_init(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y)
        assert_equal(r.ln_evidence, None)

    def test_mcmc_results_init(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y)
        assert_equal(r.mcmc_results, None)

    def test_nested_sampling_results_init(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y)
        assert_equal(r.nested_sampling_results, None)

    def test_x(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y)
        assert_equal(isinstance(r.x, np.ndarray), True)
        assert_equal(r.x, TEST_X)

    def test_y(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y)
        assert_equal(isinstance(r.y, Axis), True)
        assert_equal(r.y.values, TEST_Y)

    def test_variable_medians(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        r.max_likelihood('diff_evo')
        assert_equal(np.allclose(r.variable_medians, [1, 0], atol=1.5), True)

    def test_variable_modes(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        r.max_likelihood('diff_evo')
        r.mcmc(n_burn=10, n_samples=10, progress=False, walkers=5)
        assert_equal(np.allclose(r.variable_modes, [1, 0], atol=1.5), True)

    def test_variable_modes_no_sampling(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        r.max_likelihood('diff_evo')
        assert_equal(np.allclose(r.variable_modes, [1, 0], atol=1.5), True)

    def test_mcmc_done(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        r.mcmc(n_burn=10, n_samples=10, progress=False, walkers=5)
        assert_equal(r.mcmc_done, True)

    def test_mcmc_not_done(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        assert_equal(r.mcmc_done, False)

    def test_nested_sampling_done(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        r.nested_sampling(maxiter=100, progress=False)
        assert_equal(r.nested_sampling_done, True)

    def test_nested_sampling_not_done(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        assert_equal(r.nested_sampling_done, False)

    def test_get_sample(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        r.nested_sampling(maxiter=100, progress=False)
        assert_equal(isinstance(r.get_sample(1), list), True)

    def test_len_parameters(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y)
        assert_equal(r.len_parameters, 2)

    def test_bayesian_information_criteria(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        assert_equal(isinstance(r.bayesian_information_criteria(), float), True)

    def test_max_likelihood(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        r.max_likelihood('diff_evo')
        assert_equal(isinstance(r.variables[0], Distribution), True)
        assert_equal(isinstance(r.variables[1], Distribution), True)
        assert_equal(np.isclose(r.variables[0].n, 1, atol=0.75), True)
        assert_equal(np.isclose(r.variables[1].n, 0, atol=0.75), True)

    def test_prior(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y)
        priors = r.prior()
        assert_equal(len(priors), 2)
        assert_equal(isinstance(priors[0], scipy.stats._distn_infrastructure.rv_frozen), True)
        assert_equal(isinstance(priors[1], scipy.stats._distn_infrastructure.rv_frozen), True)
        assert_equal(priors[0].interval(1), [-9, 11])
        assert_equal(priors[1].interval(1), [-9, 11])

    def test_prior_with_bounds(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        priors = r.prior()
        assert_equal(len(priors), 2)
        assert_equal(isinstance(priors[0], scipy.stats._distn_infrastructure.rv_frozen), True)
        assert_equal(isinstance(priors[1], scipy.stats._distn_infrastructure.rv_frozen), True)
        assert_equal(priors[0].interval(1), [0, 10])
        assert_equal(priors[1].interval(1), [-1, 1])

    def test_mcmc(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        r.mcmc(n_burn=10, n_samples=10, progress=False, walkers=5)
        assert_equal(isinstance(r.variables[0], Distribution), True)
        assert_equal(isinstance(r.variables[1], Distribution), True)
        assert_equal(r.variables[0].size, 50)
        assert_equal(r.variables[1].size, 50)
        assert_equal(r.variables[0].min > 0, True)
        assert_equal(r.variables[0].max < 10, True)
        assert_equal(r.variables[1].min > -1, True)
        assert_equal(r.variables[1].max < 1, True)

    def test_nested_sampling(self):
        r = Relationship(utils.straight_line, TEST_X, TEST_Y, bounds=((0, 10), (-1, 1)))
        r.nested_sampling(maxiter=100, progress=False)
        assert_equal(isinstance(r.variables[0], Distribution), True)
        assert_equal(isinstance(r.variables[1], Distribution), True)
        assert_equal(r.variables[0].min > 0, True)
        assert_equal(r.variables[0].max < 10, True)
        assert_equal(r.variables[1].min > -1, True)
        assert_equal(r.variables[1].max < 1, True)
