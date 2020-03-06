"""
These are plotting functions that take either :class:`Relationship`
or :class:`Distribution` class objects.

The aim is to produce *publication quality* plots. However, we
recognise that taste exists, and ours may be different from yours.
The colorscheme in this work was chosen to be colorblind friendly.
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import numpy as np
import uncertainties
import matplotlib.pyplot as plt
from corner import corner
from scipy.stats import gaussian_kde
from uravu import UREG, _fig_params
from uravu.distribution import Distribution


def plot_relationship(
    relationship, axes=None, figsize=(10, 6)
):  # pragma: no cover
    """
    Plot the relationship. Additional plots will be included on this if
    the MCMC sampling has been used to find distributions.

    Args:
        relationship (uravu.relationship.Relationship): The relationship to
            be plotted.
        axes (matplotlib.axes): Axes to which the plot should be added.
            If none given new axes will be created.
        fig_size (tuple, optional): horizontal and veritcal size for
            figure (in inches). Default is `(10, 6)`.

    Returns:
        (matplotlib.axes): The axes with new plots.
    """
    if axes is None:
        axes = plt.subplots(figsize=figsize)[1]
    variables = relationship.variables
    if relationship.unaccounted_uncertainty:
        variables = relationship.variables[:-1]
    axes.plot(
        relationship.x_n, relationship.y_n, c=list(_fig_params.TABLEAU)[0]
    )
    x_label = "{}".format(relationship.abscissa_name)
    if relationship.x_u != UREG.dimensionless:
        x_label += "/${:~L}$".format(relationship.x_u)
    axes.set_xlabel(x_label)
    y_label = "{}".format(relationship.ordinate_name)
    if relationship.y_u != UREG.dimensionless:
        y_label += "/${:~L}$".format(relationship.y_u)
    axes.set_ylabel(y_label)
    if isinstance(
        relationship.ordinate.m.any(), uncertainties.core.AffineScalarFunc
    ):
        axes.fill_between(
            relationship.x_n,
            relationship.y_n - relationship.y_s,
            relationship.y_n + relationship.y_s,
            alpha=0.5,
            color=list(_fig_params.TABLEAU)[0],
            lw=0,
        )
    if not isinstance(variables[0], Distribution):
        axes.plot(
            relationship.x_n,
            relationship.function(relationship.x_n, *variables),
            color=list(_fig_params.TABLEAU)[1],
        )
    else:
        plot_samples = np.random.randint(
            0, variables[0].samples.size, size=100
        )
        for i in plot_samples:
            float_variables = [var.samples[i] for var in variables]
            axes.plot(
                relationship.x_n,
                relationship.function(relationship.x_n, *float_variables),
                color=list(_fig_params.TABLEAU)[1],
                alpha=0.05,
            )
    if relationship.unaccounted_uncertainty:
        var = relationship.variable_medians
        additional_uncertainty = np.abs(
            var[-1] * relationship.function(relationship.x_n, *var[:-1])
        )
        axes.fill_between(
            relationship.x_n,
            relationship.y_n + relationship.y_s,
            relationship.y_n + relationship.y_s + additional_uncertainty,
            alpha=0.5,
            color=list(_fig_params.TABLEAU)[2],
            lw=0,
        )
        axes.fill_between(
            relationship.x_n,
            relationship.y_n - relationship.y_s,
            relationship.y_n - relationship.y_s - additional_uncertainty,
            alpha=0.5,
            color=list(_fig_params.TABLEAU)[2],
            lw=0,
        )
    return axes


def plot_distribution(distro, axes=None, figsize=(10, 6)):  # pragma: no cover
    """
    Plot the probability density function for a distribution.

    Args:
        distro (uravu.distriobution.Distribution): The distribution to be
            plotted.
        fig_size (tuple): Horizontal and veritcal size for figure
            (in inches).

    Returns:
        (matplotlib.axes): The axes with new plots.
    """
    if axes is None:
        axes = plt.subplots(figsize=figsize)[1]
    kde = gaussian_kde(distro.samples)
    abscissa = np.linspace(distro.samples.min(), distro.samples.max(), 100)
    ordinate = kde.evaluate(abscissa)
    axes.plot(abscissa, ordinate, color=list(_fig_params.TABLEAU)[0])
    axes.hist(
        distro.samples,
        bins=25,
        density=True,
        color=list(_fig_params.TABLEAU)[0],
        alpha=0.5,
    )
    axes.fill_betweenx(
        np.linspace(0, ordinate.max() + ordinate.max() * 0.1),
        distro.con_int[0],
        distro.con_int[1],
        alpha=0.2,
    )
    x_label = "{}".format(distro.name)
    if distro.unit != UREG.dimensionless:
        x_label += "/${:~L}$".format(distro.unit)
    axes.set_xlabel(x_label)
    axes.set_ylabel("$p(${}$)$".format(distro.name))
    axes.set_ylim((0, ordinate.max() + ordinate.max() * 0.1))
    return axes


def plot_corner(relationship, figsize=(8, 8)):  # pragma: no cover
    """
    Plot the corner (named for the Python package) plot between the
    relationships variables.

    Args:
        relationship (uravu.relationship.Relationship): The relationship
            containing the distributions to be plotted.
        fig_size (tuple, optional): horizontal and veritcal size for
            figure (in inches). Default is `(10, 6)`.

    Returns:
        (matplotlib.figure): The figure with new plots.
        (matplotlib.axes): The axes with new plots.
    """
    n = len(relationship.variables)
    if not all(
        [isinstance(relationship.variables[i], Distribution) for i in range(n)]
    ):
        raise ValueError(
            "In order to use the corner plot functionality, all relationship "
            "variables must be Distributions. Please run MCMC before "
            "plotting the corner."
        )
    fig, ax = plt.subplots(n, n, figsize=figsize)
    var_labels = []
    for i in range(n):
        if relationship.variable_units[i] == UREG.dimensionless:
            var_labels.append("{}".format(relationship.variable_names[i]))
        else:
            var_labels.append(
                "{}/${:L}$".format(
                    relationship.variable_names[i],
                    relationship.variable_units[i],
                )
            )
    corner(
        relationship.mcmc_results["samples"],
        color=list(_fig_params.TABLEAU)[0],
        hist_kwargs={"lw": 4, "histtype": "stepfilled"},
        label_kwargs={"fontsize": _fig_params.rcParams["axes.labelsize"]},
        fig=fig,
        labels=var_labels,
    )
    for j in range(n):
        ax[n - 1, j].set_xticks(
            [
                i
                for i in np.percentile(
                    relationship.variables[j].samples, [2.5, 50, 97.5]
                )
            ]
        )
        ax[n - 1, j].set_xlim(
            [
                i
                for i in np.percentile(
                    relationship.variables[j].samples, [0.5, 99.5]
                )
            ]
        )
    for j in range(n - 1):
        ax[j + 1, 0].set_yticks(
            [
                i
                for i in np.percentile(
                    relationship.variables[j + 1].samples, [2.5, 50, 97.5]
                )
            ]
        )
        ax[j + 1, 0].set_ylim(
            [
                i
                for i in np.percentile(
                    relationship.variables[j + 1].samples, [0.5, 99.5]
                )
            ]
        )
    return fig, ax
