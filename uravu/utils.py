"""
Some utility functions for the uravu package.
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
