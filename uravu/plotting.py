"""
Plotting functions
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from uravu import UREG, _fig_params
from uravu.distribution import Distribution


def plot_relationship(relationship, figsize=(10, 6)):  # pragma: no cover
    """
    Plot the relationship. Additional plots will be included on this if
    the MCMC sampling has been used to find the activation_energy
    and prefactor distributions.

    Args:
        fig_size (tuple, optional): horizontal and veritcal size for
            figure (in inches). Default is `(10, 6)`.
    """
    fig, axes = plt.subplots(figsize=figsize)
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
    axes.fill_between(
        relationship.x_n,
        relationship.y_n - relationship.y_s,
        relationship.y_n + relationship.y_s,
        alpha=0.5,
        color=list(_fig_params.TABLEAU)[0],
        lw=0,
    )
    axes.set_xlim((relationship.x_n.min() - 2, relationship.x_n.max() + 2))
    axes.set_ylim(
        (
            (relationship.y_n - relationship.y_s).min() - 2,
            (relationship.y_n + relationship.y_s).max() + 2,
        )
    )
    if not isinstance(relationship.variables.any(), Distribution):
        axes.plot(
            relationship.x_n,
            relationship.function(relationship.x_n, *relationship.variables),
            color=list(_fig_params.TABLEAU)[1],
        )
    # else:
    #    plot_samples = np.random.randint(
    #        0, self.variables[0].samples.size, size=100)
    #    for i in plot_samples:
    #        variables = [var.samples[i] for var in self.variables]
    #        axes.plot(
    #            self.abscissa,
    #            self.equation(self.abscissa, *variables),
    #            color=list(_fig_params.TABLEAU)[1], alpha=0.05)
    return fig, axes


def plot_distribution(distro, figsize=(10, 6)):  # pragma: no cover
    """
    Plot the probability density function for the distribution.

    Args:
        fig_size (tuple): Horizontal and veritcal size for figure
            (in inches).

    Returns:
        (matplotlib.figure.Figure, matplotlib.axes.Axes): Figure and axes
            for the plot.
    """
    fig, axes = plt.subplots(figsize=figsize)
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
    return fig, axes
