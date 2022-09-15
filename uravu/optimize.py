"""
The optimize module includes the functionality necessary for maximum likelihood determination.
Furthermore, the natural log likelihood function used in the :func:`~uravu.sampling.mcmc()` and :func:`~uravu.sampling.nested_sampling()` methods may be found here.
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

from typing import Callable
import numpy as np
from scipy.optimize import minimize, differential_evolution


def max_ln_likelihood(relationship: 'uravu.relationship.Relationship', method: str, x0: np.ndarray=None, **kwargs) -> np.ndarray:
    """
    Determine the variable values which maximize the likelihood for the given relationship. For keyword arguments see the :func:`scipy.optimize.minimize()` documentation.

    :param relationship: The relationship for which variables should be found.
    :param method: Method for optimisation to be used.
    :param x0: Initial guesses for the variable values. Default to the current :py:attr:`~uravu.relationship.Relationship.variables` values which are initialized as all :py:attr:`1`.

    :return: Optimized variables for the relationship.
    """
    if x0 is None:
        x0 = relationship.variable_medians
    args = (relationship.function, relationship.abscissa, relationship.ordinate)
    if method == 'diff_evo':
        res = differential_evolution(negative_lnl, relationship.bounds, args=args, **kwargs)
    elif method == 'mini':
        res = minimize(negative_lnl, x0, args=args, bounds=relationship.bounds, **kwargs)
    return res.x


def negative_lnl(variables: np.ndarray, function: Callable, abscissa: np.ndarray, ordinate: np.ndarray) -> float:
    """
    Calculate the negative natural logarithm of the likelihood given a set of variables, when there is no uncertainty in the abscissa.

    :param variables: Variables for the function evaluation.
    :param function: The function to be evaluated.
    :param abscissa: The abscissa values.
    :param ordinate: The ordinate values.

    :return: Negative natural log-likelihood between model and data.
    """
    return -ln_likelihood(variables, function, abscissa, ordinate)


def ln_likelihood(variables: np.ndarray, function: Callable, abscissa: np.ndarray, ordinate: np.ndarray) -> float:
    """
    Calculate the natural logarithm of the likelihood given a set of variables, when there is no uncertainty in the abscissa.

    :param variables: Variables for the function evaluation.
    :param function: The function to be evaluated.
    :param abscissa: The abscissa values.
    :param ordinate: The ordinate values.

    :return: Natural log-likelihood between model and data.
    """
    model = function(abscissa, *variables)
    ln_l = ordinate.logpdf(model)
    return np.sum(ln_l)
