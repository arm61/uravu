"""
The optimize module includes the functionality necessary for maximum
likelihood determination. Further, the natural log likelihood function used
in the ``mcmc`` and ``nested_sampling`` methods may be found here.
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import numpy as np
import uncertainties
from scipy.optimize import minimize
from uncertainties import unumpy as unp


def max_ln_likelihood(relationship, x0=None, **kwargs):
    """
    Determine the variable values which maximize the likelihood for the
    given relationship. For keyword arguments see the
    `scipy.optimize.minimize()`_ documentation.

    Args:
        relationship (uravu.relationship.Relationship): The relationship for
            which variables should be found.
        x0 (array_like, optional): Initial guesses for the variable values.
            Default to the current ``relationship.variables`` values which
            are initialized as all ones.

    Return:
        (array_like): optimized variables for the relationship.

    .. _scipy.optimize.minimize(): https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    if x0 is None:
        x0 = relationship.variable_medians
    return minimize(
        negative_lnl,
        x0,
        args=(
            relationship.function,
            relationship.abscissa,
            relationship.ordinate,
            relationship.unaccounted_uncertainty,
        ),
        **kwargs,
    ).x


def negative_lnl(
    variables, function, abscissa, ordinate, unaccounted_uncertainty=False
):
    """
    Calculate the negative natural logarithm of the likelihood given a set
    of variables, when there is no uncertainty in the abscissa.

    Args:
        variables (array_like): Variables for the function evaluation.
        function (callable): The function to be evaluated.
        abscissa (array_like): The abscissa values.
        ordinate (array_like): The ordinate values.
        unaccounted_uncertainty (bool, optional): Should an unaccounted
            uncertainty parameter be considered. Default is ``False``.

    Returns:
        (float): Negative ln-likelihood between model and data.
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
    Calculate the natural logarithm of the likelihood given a set of
    variables, when there is no uncertainty in the abscissa.

    Args:
        variables (array_like): Variables for the function evaluation.
        function (callable): The function to be evaluated.
        abscissa (array_like): The abscissa values.
        ordinate (array_like): The ordinate values.
        unaccounted_uncertainty (bool, optional): Should an unaccounted
            uncertainty parameter be considered. Default is ``False``.

    Returns:
        (float): ln-likelihood between model and data.
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
        
