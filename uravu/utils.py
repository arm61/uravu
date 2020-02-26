"""
A few additional utility functions to improve the usability of
``uravu``.
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
        abscissa (array_like): the abscissa data.
        gradient (float): the slope of the line.
        intercept (float): the y-intercept of the line.

    Returns:
        (array_like): the resulting ordinate.
    """
    return gradient * abscissa + intercept


def bayes_factor(model_1, model_2):
    """
    Use the Bayes factor to compare two models.

    Args:
        model_1 (uncertainties.core.Variable or float): ln evidence for
            model 1.
        model_2 (uncertainties.core.Variable or float): ln evidence for
            model 2.

    Return:
        (uncertainties.core.Variable or float): 2ln(B) where B is the Bayes
            Factor between the two models.
    """
    return 2 * (model_1 - model_2)


def correlation_matrix(relationship):
    """
    Evalutate the Pearsons correlation coefficient matrix for the
    variables in a given relationship.

    Args:
        relationship (uravu.relationship.Relationship): The relationship
            to determine the posteriors of.

    Returns:
        (array_like): The correlation matrix for the relationships
            variables.
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
