"""
The optimize module includes the functionality necessary for maximum likelihood determination. 
Furthermore, the natural log likelihood function used in the :func:`~uravu.sampling.mcmc()` and :func:`~uravu.sampling.nested_sampling()` methods may be found here.
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import numpy as np
from scipy.optimize import minimize, differential_evolution
from uncertainties import unumpy as unp


def max_ln_likelihood(relationship, method, x0=None, **kwargs):
    """
    Determine the variable values which maximize the likelihood for the given relationship. For keyword arguments see the :func:`scipy.optimize.minimize()` documentation.

    Args:
        relationship (:py:class:`uravu.relationship.Relationship`): The relationship for which variables should be found.
        x0 (:py:attr:`array_like`, optional): Initial guesses for the variable values. Default to the current :py:attr:`~uravu.relationship.Relationship.variables` values which are initialized as all :py:attr:`1`.

    Return:
        :py:attr:`array_like`: Optimized variables for the relationship.
    """
    if x0 is None:
        x0 = relationship.variable_medians
    if method == 'diff_evo':
        res = differential_evolution(
            negative_lnl, 
            relationship.bounds, 
            args=(
                relationship.function,
                relationship.abscissa,
                relationship.ordinate,
                relationship.unaccounted_uncertainty,
            ),
            **kwargs,
        )
    elif method == 'mini':
        res = minimize(
            negative_lnl,
            x0,
            args=(
                relationship.function,
                relationship.abscissa,
                relationship.ordinate,
                relationship.unaccounted_uncertainty,
            ),
            bounds=relationship.bounds,
            **kwargs,
        )
    return res.x


def negative_lnl(
    variables, function, abscissa, ordinate, unaccounted_uncertainty=False,
):
    """
    Calculate the negative natural logarithm of the likelihood given a set of variables, when there is no uncertainty in the abscissa.

    Args:
        variables (:py:attr:`array_like`): Variables for the function evaluation.
        function (:py:attr:`callable`): The function to be evaluated.
        abscissa (:py:attr:`array_like`): The abscissa values.
        ordinate (:py:attr:`array_like`): The ordinate values.
        unaccounted_uncertainty (:py:attr:`bool`, optional): Should an unaccounted uncertainty parameter be considered. Default is :py:attr:`False`.

    Returns:
        :py:attr:`float`: Negative natural log-likelihood between model and data.
    """
    return -ln_likelihood(
        variables,
        function,
        abscissa,
        ordinate,
        unaccounted_uncertainty=unaccounted_uncertainty,
    )


def ln_likelihood(
    variables, function, abscissa, ordinate, unaccounted_uncertainty=False
):
    """
    Calculate the natural logarithm of the likelihood given a set of variables, when there is no uncertainty in the abscissa.

    Args:
        variables (:py:attr:`array_like`): Variables for the function evaluation.
        function (:py:attr:`callable`): The function to be evaluated.
        abscissa (:py:attr:`array_like`): The abscissa values.
        ordinate (:py:attr:`array_like`): The ordinate values.
        unaccounted_uncertainty (:py:attr:`bool`, optional): Should an unaccounted uncertainty parameter be considered. Default is :py:attr:`False`.

    Returns:
         :py:attr:`float`: Natural log-likelihood between model and data.
    """
    if unaccounted_uncertainty:
        var = variables[:-1]
        log_f = variables[-1]
    else:
        var = variables
        log_f = -np.inf
    model = function(abscissa.m, *var)
    y_data = unp.nominal_values(ordinate.m)
    dy_data = unp.std_devs(ordinate.m)

    if np.isclose(dy_data.all(), 0.0):
        sigma2 = model ** 2 * np.exp(2 * log_f)
    else:
        sigma2 = dy_data ** 2 + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((model - y_data) ** 2 / sigma2 + np.log(sigma2))
