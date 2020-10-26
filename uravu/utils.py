"""
A few additional utility functions to improve the usability of :py:mod:`uravu`.
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey


import numpy as np
from scipy.stats import pearsonr


def straight_line(abscissa, gradient, intercept):
    """
    A one dimensional straight line function.

    Args:
        abscissa (:py:attr:`array_like`): The abscissa data.
        gradient (:py:attr:`float`): The slope of the line.
        intercept (:py:attr:`float`): The y-intercept of the line.

    Returns:
        :py:attr:`array_like`: The resulting ordinate.
    """
    return gradient * abscissa + intercept


def bayes_factor(model_1, model_2):
    """
    Use the `Bayes factor`_ to compare two models. Using Table from `Kass and Raftery`_ to compare.

    Args:
        model_1 (:py:class:`uncertainties.core.Variable` or :py:attr:`float`): ln evidence for model 1.
        model_2 (:py:class:`uncertainties.core.Variable` or :py:attr:`float`): ln evidence for model 2.

    Return:
        :py:class:`uncertainties.core.Variable` or :py:attr:`float`: 2ln(B), where B is the Bayes Factor between the two models.

    .. _Bayes factor: https://en.wikipedia.org/wiki/Bayes_factor
    .. _Kass and Raftery: https://www.colorado.edu/amath/sites/default/files/attached-files/kassraftery95.pdf
    """
    return 2 * (model_1 - model_2)


def correlation_matrix(relationship):
    """
    Evaluate the `Pearsons correlation coefficient`_ matrix for the variables in a given relationship.

    Args:
        relationship (:py:class:`uravu.relationship.Relationship`): The relationship to determine the correlation matrix for.

    Returns:
        :py:attr:`array_like`: The correlation matrix for the relationships variables.

    .. _Pearsons correlation coefficient: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    """
    n = len(relationship.variables)
    matrix = np.zeros((n, n))
    subset = np.random.randint(0, relationship.variables[0].size, size=400)
    for i in range(n):
        samples_i = relationship.variables[i].samples[subset]
        for j in range(i, n):
            samples_j = relationship.variables[j].samples[subset]
            matrix[i, j] = pearsonr(samples_i, samples_j)[0]
            matrix[j, i] = pearsonr(samples_j, samples_i)[0]
    return matrix


def latex(distribution):
    """
    Get some LaTeX math-type code that describes the mean and confidence intervals of the distribution.

    Args:
        distribution (:py:class:`uravu.distribution.Distribution`): The distribution to return LaTeX for.

    Returns:
        (:py:attr:`str`): A LaTeX formatted string for the mean and confidence intervals of the distribution.
    """
    mean = distribution.n
    con_int = distribution.con_int
    if distribution.normal:
        return f'${mean:.3e}' + r'\pm' + f'{{{con_int[1] - mean:.3e}}}$'
    else:
        return (f'${mean:.3e}^{{+{con_int[1] - mean:.3e}}}'
                f'_{{-{mean - con_int[0]:.3e}}}$')
