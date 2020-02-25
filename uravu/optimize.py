"""
This module enables the optimization of the maximum likelihood, amoung other
features.
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import numpy as np
from scipy.optimize import minimize
from uncertainties import unumpy as unp


def max_ln_likelihood(relationship, x0=None, **kwargs):
    """
    Determine the maximum natural log likelihood for the relationship object.

    Args:
        relationship (uravu.relationship.Relationship): the relationship to be
            evaluated.

    Return:
        (array_like): optimized variables for the relationship.
    """
    if x0 is None:
        x0 = relationship.variables
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
    Evaluate the negative natural logarithm of the joint likelihood, when
    there is no uncertainty in the abscissa.

    Args:
        model (array_like): Model ordinate data.
        y_data (array_like): Actual ordinate data.

    Returns:
        (float): negative ln-likelihood between model and data.
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
    The natural logarithm of the joint likelihood, when there is no
    uncertainty in the abscissa equation from
    DOI: 10.1107/S1600576718017296.

    Args:
        model (array_like): Model ordinate data.
        y_data (array_like): Experimental ordinate data.
        dy_data (array_like): Experimental ordinate-uncertainty data.

    Returns:
        (float): ln-likelihood between model and data.
    """
    if unaccounted_uncertainty:
        var = variables[:-1]
        log_f = variables[-1]
    else:
        var = variables
        log_f = -np.inf
    y_data = unp.nominal_values(ordinate.m)
    dy_data = unp.std_devs(ordinate.m)
    model = function(abscissa.m, *var)

    sigma2 = dy_data ** 2 + model ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((model - y_data) ** 2 / sigma2 + np.log(sigma2))
