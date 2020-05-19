"""
The optimize module includes the functionality necessary for maximum likelihood determination.
Furthermore, the natural log likelihood function used in the :func:`~uravu.sampling.mcmc()` and :func:`~uravu.sampling.nested_sampling()` methods may be found here.
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import numpy as np
from scipy.optimize import minimize, differential_evolution


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
    args = (relationship.function, relationship.abscissa, relationship.ordinate)
    if method == 'diff_evo':
        res = differential_evolution(negative_lnl, relationship.bounds, args=args, **kwargs)
    elif method == 'mini':
        res = minimize(negative_lnl, x0, args=args, bounds=relationship.bounds, **kwargs)
    return res.x


def negative_lnl(variables, function, abscissa, ordinate):
    """
    Calculate the negative natural logarithm of the likelihood given a set of variables, when there is no uncertainty in the abscissa.

    Args:
        variables (:py:attr:`array_like`): Variables for the function evaluation.
        function (:py:attr:`callable`): The function to be evaluated.
        abscissa (:py:attr:`array_like`): The abscissa values.
        ordinate (:py:attr:`array_like`): The ordinate values.

    Returns:
        :py:attr:`float`: Negative natural log-likelihood between model and data.
    """
    return -ln_likelihood(variables, function, abscissa, ordinate)


def ln_likelihood(variables, function, abscissa, ordinate):
    """
    Calculate the natural logarithm of the likelihood given a set of variables, when there is no uncertainty in the abscissa.

    Args:
        variables (:py:attr:`array_like`): Variables for the function evaluation.
        function (:py:attr:`callable`): The function to be evaluated.
        abscissa (:py:attr:`array_like`): The abscissa values.
        ordinate (:py:attr:`array_like`): The ordinate values.

    Returns:
         :py:attr:`float`: Natural log-likelihood between model and data.
    """
    model = function(abscissa, *variables)
    ln_l = ordinate.logpdf(model)
    return np.sum(ln_l)
