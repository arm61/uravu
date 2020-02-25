"""
A few additional utility functions to improve the usability of 
``uravu``. 
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey


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
