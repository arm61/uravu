"""
Tests for relationship module
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import unittest
import numpy as np
import uncertainties
from uncertainties import unumpy as unp
from numpy.testing import assert_almost_equal, assert_equal
from uravu import UREG, utils
from uravu.relationship import Relationship
from uravu.distribution import Distribution


class TestRelationship(unittest.TestCase):
    """
    Tests for the relationship module and class.
    """

    def test_init_all_default_one_dimensional(self):
        """
        Test the initialisation of the relationship class with one
        dimensional data and all defaults.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e)
        assert_equal(test_rel.function, utils.straight_line)
        assert_almost_equal(test_rel.abscissa.m, test_x)
        assert_almost_equal(unp.nominal_values(test_rel.ordinate.m), test_y)
        assert_almost_equal(unp.std_devs(test_rel.ordinate.m), test_y_e)
        assert_almost_equal(test_rel.variables, np.ones((2)))

    def test_init_all_default_two_dimensional(self):
        """
        Test the initialisation of the relationship class with two
        dimensional data and all defaults.
        """
        test_x = np.array([np.linspace(0, 99, 100), np.linspace(0, 99, 100)]).T
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e)
        assert_equal(test_rel.function, utils.straight_line)
        assert_almost_equal(test_rel.abscissa.m, test_x)
        assert_almost_equal(unp.nominal_values(test_rel.ordinate.m), test_y)
        assert_almost_equal(unp.std_devs(test_rel.ordinate.m), test_y_e)
        assert_almost_equal(test_rel.variables, np.ones((2)))

    def test_init_additional_uncertainty_one_dimensional(self):
        """
        Test the initialisation of the relationship class with one
        dimensional data and an additional uncertainty.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e, unaccounted_uncertainty=True)
        assert_equal(test_rel.function, utils.straight_line)
        assert_almost_equal(test_rel.abscissa.m, test_x)
        assert_almost_equal(unp.nominal_values(test_rel.ordinate.m), test_y)
        assert_almost_equal(unp.std_devs(test_rel.ordinate.m), test_y_e)
        assert_almost_equal(test_rel.variables, np.ones((3)))

    def test_init_additional_uncertainty_two_dimensional(self):
        """
        Test the initialisation of the relationship class with two
        dimensional data and an additional uncertainty.
        """
        test_x = np.array([np.linspace(0, 99, 100), np.linspace(0, 99, 100)]).T
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e, unaccounted_uncertainty=True)
        assert_equal(test_rel.function, utils.straight_line)
        assert_almost_equal(test_rel.abscissa.m, test_x)
        assert_almost_equal(unp.nominal_values(test_rel.ordinate.m), test_y)
        assert_almost_equal(unp.std_devs(test_rel.ordinate.m), test_y_e)
        assert_almost_equal(test_rel.variables, np.ones((3)))

    def test_init_different_length_x_and_y_one_dimension(self):
        """
        Test initialisation with different array lengths and one dimensional
        data.
        """
        with self.assertRaises(ValueError):
            test_x = np.linspace(0, 99, 100)
            test_y = np.linspace(0, 199, 99)
            test_y_e = test_y * 0.1
            Relationship(utils.straight_line, test_x, test_y, test_y_e)

    def test_init_different_length_x_and_y_two_dimension(self):
        """
        Test initialisation with different array lengths and two dimensional
        data.
        """
        with self.assertRaises(ValueError):
            test_x = np.array(
                [np.linspace(0, 99, 100), np.linspace(0, 99, 100)]
            ).T
            test_y = np.linspace(0, 199, 99)
            test_y_e = test_y * 0.1
            Relationship(utils.straight_line, test_x, test_y, test_y_e)

    def test_init_x_u_one_dimensional(self):
        """
        Test the initialisation of the relationship class with one
        dimensional data and an uncertainty in the abscissa.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_x_e = test_x * 0.1
        test_rel = Relationship(
            utils.straight_line,
            test_x,
            test_y,
            test_y_e,
            abscissa_uncertainty=test_x_e,
        )
        assert_equal(test_rel.function, utils.straight_line)
        assert_almost_equal(unp.nominal_values(test_rel.abscissa.m), test_x)
        assert_almost_equal(unp.std_devs(test_rel.abscissa.m), test_x_e)
        assert_almost_equal(unp.nominal_values(test_rel.ordinate.m), test_y)
        assert_almost_equal(unp.std_devs(test_rel.ordinate.m), test_y_e)
        assert_almost_equal(test_rel.variables, np.ones((2)))

    def test_init_x_unit_one_dimensional(self):
        """
        Test the initialisation of the relationship class with one
        dimensional data and an unit in abscissa.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(
            utils.straight_line,
            test_x,
            test_y,
            test_y_e,
            abscissa_unit=UREG.meter,
        )
        assert_equal(test_rel.function, utils.straight_line)
        assert_almost_equal(test_rel.abscissa.m, test_x)
        assert_equal(test_rel.abscissa.u, UREG.meter)
        assert_almost_equal(unp.nominal_values(test_rel.ordinate.m), test_y)
        assert_almost_equal(unp.std_devs(test_rel.ordinate.m), test_y_e)
        assert_almost_equal(test_rel.variables, np.ones((2)))

    def test_init_y_unit_one_dimensional(self):
        """
        Test the initialisation of the relationship class with one
        dimensional data and an unit in ordinate.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(
            utils.straight_line,
            test_x,
            test_y,
            test_y_e,
            ordinate_unit=UREG.meter,
        )
        assert_equal(test_rel.function, utils.straight_line)
        assert_almost_equal(test_rel.abscissa.m, test_x)
        assert_equal(test_rel.ordinate.u, UREG.meter)
        assert_almost_equal(unp.nominal_values(test_rel.ordinate.m), test_y)
        assert_almost_equal(unp.std_devs(test_rel.ordinate.m), test_y_e)
        assert_almost_equal(test_rel.variables, np.ones((2)))

    def test_x_a(self):
        """
        Test the x property with no uncertainty.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e)
        assert_almost_equal(test_rel.x.m, test_x)
        assert_equal(test_rel.x.u, UREG.dimensionless)

    def test_x_b(self):
        """
        Test the x property with uncertainty.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_x_e = test_x * 0.1
        test_rel = Relationship(
            utils.straight_line,
            test_x,
            test_y,
            test_y_e,
            abscissa_uncertainty=test_x_e,
        )
        assert_almost_equal(unp.nominal_values(test_rel.x.m), test_x)
        assert_almost_equal(unp.std_devs(test_rel.x.m), test_x_e)
        assert_equal(test_rel.x.u, UREG.dimensionless)

    def test_y(self):
        """
        Test the y property.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e)
        assert_almost_equal(unp.nominal_values(test_rel.y.m), test_y)
        assert_almost_equal(unp.std_devs(test_rel.y.m), test_y_e)
        assert_equal(test_rel.y.u, UREG.dimensionless)

    def test_x_m_a(self):
        """
        Test the x_m property with no uncertainty.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e)
        assert_almost_equal(test_rel.x_m, test_x)

    def test_x_m_b(self):
        """
        Test the x_m property with uncertainty.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_x_e = test_x * 0.1
        test_rel = Relationship(
            utils.straight_line,
            test_x,
            test_y,
            test_y_e,
            abscissa_uncertainty=test_x_e,
        )
        assert_almost_equal(unp.nominal_values(test_rel.x_m), test_x)
        assert_almost_equal(unp.std_devs(test_rel.x_m), test_x_e)

    def test_y_m(self):
        """
        Test the y_m property.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e)
        assert_almost_equal(unp.nominal_values(test_rel.y_m), test_y)
        assert_almost_equal(unp.std_devs(test_rel.y_m), test_y_e)

    def test_x_u(self):
        """
        Test the x_u property.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(
            utils.straight_line,
            test_x,
            test_y,
            test_y_e,
            abscissa_unit=UREG.meter,
        )
        assert_equal(test_rel.x_u, UREG.meter)

    def test_y_u(self):
        """
        Test the y_u property.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(
            utils.straight_line,
            test_x,
            test_y,
            test_y_e,
            ordinate_unit=UREG.meter,
        )
        assert_equal(test_rel.y_u, UREG.meter)

    def test_x_n_a(self):
        """
        Test the x_n property with no uncertainty.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e)
        assert_almost_equal(test_rel.x_n, test_x)

    def test_x_n_b(self):
        """
        Test the x_n property with uncertainty.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_x_e = test_x * 0.1
        test_rel = Relationship(
            utils.straight_line,
            test_x,
            test_y,
            test_y_e,
            abscissa_uncertainty=test_x_e,
        )
        assert_almost_equal(test_rel.x_n, test_x)

    def test_x_s_a(self):
        """
        Test the x_n property with no uncertainty.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e)
        assert_equal(test_rel.x_s, None)

    def test_x_s_b(self):
        """
        Test the x_n property with uncertainty.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_x_e = test_x * 0.1
        test_rel = Relationship(
            utils.straight_line,
            test_x,
            test_y,
            test_y_e,
            abscissa_uncertainty=test_x_e,
        )
        assert_almost_equal(test_rel.x_s, test_x_e)

    def test_y_n(self):
        """
        Test the y_n property.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e)
        assert_almost_equal(test_rel.y_n, test_y)

    def test_y_s(self):
        """
        Test the y_s property.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e)
        assert_almost_equal(test_rel.y_s, test_y_e)

    def test_variable_medians_b(self):
        """
        Test variable_medians property when the variables are Distributions.
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
        medians = test_rel.variable_medians
        assert_equal(medians.shape, (2,))

    def test_variable_medians_a(self):
        """
        Test variable_medians property when the variables are floats.
        """
        test_x = np.linspace(0, 99, 10)
        test_y = (
            np.linspace(1, 199, 10)
            + np.linspace(1, 199, 10) * np.random.randn(10) * 0.05
        )
        test_y_e = test_y * 0.2
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e,)
        test_rel.max_likelihood()
        medians = test_rel.variable_medians
        assert_equal(medians.shape, (2,))

    def test_len_parameters(self):
        """
        test the len_parameters function.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(0, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e)
        assert_equal(test_rel.len_parameters(), 2)

    def test_max_likelihood(self):
        """
        Test max_likelihood function.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(1, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e)
        test_rel.max_likelihood()
        assert_almost_equal(test_rel.variables, np.array([2, 1]))

    def test_prior(self):
        """
        Test prior function.
        """
        test_x = np.linspace(0, 99, 100)
        test_y = np.linspace(1, 199, 100)
        test_y_e = test_y * 0.1
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e)
        test_rel.max_likelihood()
        result_priors = test_rel.prior(np.random.random((2, 100)))
        assert_equal(result_priors.shape, (2, 100))
        assert_equal(result_priors[0].min() > -18, True)
        assert_equal(result_priors[0].max() < 22, True)
        assert_equal(result_priors[1].min() > -9, True)
        assert_equal(result_priors[1].max() < 11, True)

    def test_mcmc(self):
        """
        Test mcmc function.
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
        assert_equal(isinstance(test_rel.variables[0], Distribution), True)
        assert_equal(isinstance(test_rel.variables[1], Distribution), True)
        assert_equal(test_rel.variables[0].size, 1000)
        assert_equal(test_rel.variables[1].size, 1000)

    def test_nested_sampling(self):
        """
        Test nested sampling.
        """
        test_y = np.ones(10) * np.random.randn(10)
        test_y_e = np.ones(10) * 0.1
        test_x = np.linspace(1, 10, 10)
        test_rel = Relationship(utils.straight_line, test_x, test_y, test_y_e)
        test_rel.nested_sampling(maxiter=10)
        assert_equal(
            isinstance(test_rel.ln_evidence, uncertainties.core.Variable), True
        )
