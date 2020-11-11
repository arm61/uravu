"""
The storage and manipulation of probability distributions is fundamental to the operation of ``uravu`` and Bayesian inference.
The :py:class:`~uravu.distribution.Distribution` class oversees these operations.
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import numpy as np
from scipy.stats import normaltest, gaussian_kde
from scipy.optimize import minimize


class Distribution:
    """
    In addition to storage of the probability distribution, this class allows for some basic analysis, such as determination of normality.

    Attributes:
        samples (:py:attr:`array_like`): Samples in the distribution.
        name (:py:attr:`str`): Distribution name.
        ci_points (:py:attr:`array_like`): The percentiles at which confidence intervals should be found.
        normal (:py:attr:`bool`): Are the samples normally distributed?
        kde (:py:class:`scipy.stats.kde.gaussian_kde`): Kernel density approximation for the distribution.

    Args:
        samples (:py:attr:`array_like`): Sample for the distribution.
        name (:py:attr:`str`, optional): A name to identify the distribution. Default is :py:attr:`'Distribution'`.
        ci_points (:py:attr:`array_like`, optional): The two percentiles at which confidence intervals should be found. Default is :py:attr:`[2.5, 97.5]` (a 95 % confidence interval).

    .. _FAQ: ./faq.html
    """

    def __init__(self, samples, name="Distribution", ci_points=None):
        """
        Initialisation function for a :py:class:`~uravu.distribution.Distribution` object.
        """
        self.name = name
        self.samples = np.array([])
        if ci_points is None:
            self.ci_points = np.array([2.5, 97.5])
        else:
            if len(ci_points) != 2:
                raise ValueError("The ci_points must be an array of length two.")
            self.ci_points = np.array(ci_points)
        self.normal = False
        self.add_samples(np.array(samples))

    @property
    def size(self):
        """
        Get the number of samples in the distribution.

        Returns:
            :py:attr:`int`: Number of samples.
        """
        return self.samples.size

    def check_normality(self):
        """
        Uses a :func:`scipy.stats.normaltest()` to evaluate if samples are normally distributed and updates the :py:attr:`~uravu.distribution.Distribution.normal` attribute.
        """
        alpha = 0.05
        test_samples = self.samples
        if self.size > 500:
            test_samples = np.random.choice(self.samples, size=500)
        p_value = normaltest(test_samples)[1]
        if p_value > alpha:
            self.normal = True
        else:
            self.normal = False

    def pdf(self, x):
        """
        Get the probability density function for the distribution.

        Args:
            x (:py:attr:`float`): Value to return probability of.

        Return:
            :py:attr:`float`: Probability.
        """
        return self.kde.pdf(x)

    def logpdf(self, x):
        """
        Get the natural log probability density function for the distribution.

        Args:
            x (:py:attr:`float`): Value to return natural log probability of.

        Return:
            :py:attr:`float`: Natural log probability.
        """
        return self.kde.logpdf(x)

    def negative_pdf(self, x):
        """
        Get the negative of the probability density function for the distribution.

        Args:
            x (:py:attr:`float`): Value to return negative probability of.

        Return:
            :py:attr:`float`: Negative probability.
        """
        return -self.kde.pdf(x)

    @property
    def dist_max(self):
        """
        Get the value that maximises the distribution. If no :py:attr:`kde` has been created (for example if the distribution has fewer than 8 values) the median is returned.

        Returns
            :py:attr:`float`: Most likely value.
        """
        try:
            return minimize(self.negative_pdf, x0=[self.n]).x
        except AttributeError:
            return self.n


    @property
    def min(self):
        """
        Get sample minimum.

        Returns:
            :py:attr:`float`: Sample minimum.
        """
        return self.samples.min()

    @property
    def max(self):
        """
        Get sample maximum.

        Returns:
            :py:attr:`float`: Sample maximum.
        """
        return self.samples.max()

    @property
    def n(self):
        """
        Get the median value of the distribution (for a normal distribution this is the same as the mean).

        Returns:
            :py:attr:`float`: Median value.
        """
        return np.percentile(self.samples, [50])[0]

    @property
    def s(self):
        """
        Get the standard deviation of the distribution. For a non-normal distribution, this will return :py:attr:`None`.

        Returns:
            :py:attr:`float` or :py:attr:`None`: Standard deviation of the distribution.
        """
        if self.normal:
            return np.std(self.samples, ddof=1)
        return None

    @property
    def v(self):
        """
        Get the variance of the distribution. For a non-normal distribution, this will return :py:attr:`None`.

        Returns:
            :py:attr:`float` or :py:attr:`None`: Standard deviation of the distribution.
        """
        if self.normal:
            return np.var(self.samples, ddof=1)
        return None

    @property
    def con_int(self):
        """
        Get the extrema of the confidence intervals of the distribution.

        Returns:
            :py:attr:`array_like`: Distribution values at the confidence interval.
        """
        return np.percentile(self.samples, self.ci_points)

    def add_samples(self, samples):
        """
        Add samples to the distribution.

        Args:
            samples (:py:attr:`array_like`): Samples to be added to the distribution.
        """
        self.samples = np.append(self.samples, np.array(samples).flatten())
        if self.size > 8:
            self.check_normality()
            self.kde = gaussian_kde(self.samples)
