"""
These are plotting functions that take either :py:class:`~uravu.relationship.Relationship` or :py:class:`~uravu.distribution.Distribution` class objects.

The aim is to produce *publication quality* plots. However, we recognise that taste exists, and ours may be different from yours. The colorscheme in this work was chosen to be colorblind friendly.
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import numpy as np

try:
    import matplotlib.pyplot as plt
    from corner import corner
    from uravu import _fig_params
    colors = _fig_params.colors
except ModuleNotFoundError:
    raise ModuleNotFoundError("The matplotlib, corner and seaborn packages are necessary for the use of the plotting module, please install these.")


def plot_relationship(relationship, axes=None, figsize=(10, 6)):  # pragma: no cover
    """
    Plot the relationship. Additional plots will be included on this if posterior sampling has been used to find distributions.

    Args:
        relationship (:py:class:`uravu.relationship.Relationship`): The relationship to be plotted.
        axes (:py:class:`matplotlib.axes.Axes`, optional): Axes to which the plot should be added. If :py:attr:`None` given new axes will be created. Default is :py:attr:`None`.
        fig_size (:py:attr:`tuple`, optional): horizontal and veritcal size for figure (in inches). Default is :py:attr:`(10, 6)`.

    Returns:
        (:py:class:`matplotlib.axes.Axes`): The axes with new plots.
    """
    if axes is None:
        axes = plt.subplots(figsize=figsize)[1]
    variables = relationship.variables
    axes.errorbar(relationship.x, relationship.y.mode, relationship.y.s, c=colors[0], ecolor=colors[0] + '40', marker='.', ls='')
    smooth_x = np.linspace(relationship.x.min(), relationship.x.max(), 1000)
    if relationship.mcmc_done or relationship.nested_sampling_done:
        plot_samples = np.random.randint(0, variables[0].samples.size, size=100)
        for i in plot_samples:
            float_variables = relationship.get_sample(i)
            axes.plot(smooth_x, relationship.function(smooth_x, *float_variables), color=colors[1], alpha=0.05)
    else:
        float_variables = relationship.variable_medians
        axes.plot(smooth_x, relationship.function(smooth_x, *float_variables), color=colors[1])
    return axes


def plot_distribution(distro, axes=None, figsize=(5, 3)):  # pragma: no cover
    """
    Plot the probability density function for a distribution.

    Args:
        distro (:py:class`uravu.distriobution.Distribution`): The distribution to be plotted.
        axes (:py:class:`matplotlib.axes.Axes`, optional): Axes to which the plot should be added. If :py:attr:`None` given new axes will be created. Default is :py:attr:`None`.
        fig_size (:py:class:`tuple`): Horizontal and veritcal size for figure (in inches). Default is :py:attr:`(10, 6)`.

    Returns:
        (:py:class:`matplotlib.axes.Axes`): The axes with new plots.
    """
    if axes is None:
        axes = plt.subplots(figsize=figsize)[1]
    kde = distro.kde
    abscissa = np.linspace(distro.samples.min(), distro.samples.max(), 100)
    ordinate = kde.evaluate(abscissa)
    axes.plot(abscissa, ordinate, color=colors[0])
    axes.hist(distro.samples, bins=25, density=True, color=colors[0], alpha=0.5)
    axes.fill_betweenx(np.linspace(0, ordinate.max() + ordinate.max() * 0.1), distro.con_int[0], distro.con_int[1], alpha=0.2)
    axes.set_ylim((0, ordinate.max() + ordinate.max() * 0.1))
    return axes


def plot_corner(relationship, figsize=(8, 8)):  # pragma: no cover
    """
    Plot the :py:mod:`corner` (named for the Python package) plot between the relationships variables.

    Args:
        relationship (:py:class:`uravu.relationship.Relationship`): The relationship containing the distributions to be plotted.
        fig_size (:py:attr:`tuple`, optional): horizontal and veritcal size for figure (in inches). Default is :py:attr:`(10, 6)`.

    Returns:
        :py:attr:`tuple`: Containing:
            - :py:class:`matplotlib.figure.Figure`: The figure with new plots.
            - :py:class:`matplotlib.axes.Axes`: The axes with new plots.
    """
    n = len(relationship.variables)
    fig, ax = plt.subplots(n, n, figsize=figsize)
    samples = np.zeros((relationship.variables[0].size, len(relationship.variables)))
    for i, v in enumerate(relationship.variables):
        samples[:, i] = v.samples
    corner(samples, color=colors[0], hist_kwargs={"lw": 4, "histtype": "step"}, label_kwargs={"fontsize": _fig_params.rcParams["axes.labelsize"]}, fig=fig)
    return fig, ax
