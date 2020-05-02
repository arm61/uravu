"""
The storage and manipulation of probability distributions is fundamental to the operation of ``uravu`` and Bayesian inference. 
The :py:class:`~uravu.distribution.Distribution` class oversees these operations. 
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import numpy as np
from scipy.stats import normaltest
from uncertainties import ufloat
from uravu import UREG


class Distribution:
    """
    In addition to storage of the probability distribution, this class allows for some basic analysis, such as determination of normality.

    Attributes:
        samples (:py:attr:`array_like`): Samples in the distribution.
        name (:py:attr:`str`): Distribution name.
        ci_points (:py:attr:`array_like`): The percentiles at which confidence intervals should be found.
        unit (:py:class:`~pint.unit.Unit`): The unit of the values in the Distribution.
        normal (:py:attr:`bool`): Are the samples normally distributed?

    Args:
        samples (:py:attr:`array_like`): Sample for the distribution.
        name (:py:attr:`str`, optional): A name to identify the distribution. Default is :py:attr:`'Distribution'`.
        ci_points (:py:attr:`array_like`, optional): The two percentiles at which confidence intervals should be found. Default is :py:attr:`[2.5, 97.5]` (a 95 % confidence interval).
        unit (:py:class:`~pint.unit.Unit`, optional): The unit for the distribution. For information about unit assignment see the `FAQ`_. Default is :py:attr:`~pint.unit.Unit.dimensionless`.

    .. _FAQ: ./faq.html
    """

    def __init__(
        self, samples, name="Distribution", ci_points=None, unit=UREG.dimensionless,
    ):
        """
        Initialisation function for a :py:class:`~uravu.distribution.Distribution` object.
        """
        self.name = name
        self.unit = unit
        self.samples = np.array([])
        if ci_points is None:
            self.ci_points = [2.5, 97.5]
        else:
            if len(ci_points) != 2:
                raise ValueError(
                    "The ci_points must be an array or tuple of length two."
                )
            self.ci_points = ci_points
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
        if self.size <= 8:
            return False
        test_samples = self.samples
        if self.size > 500:
            test_samples = np.random.choice(self.samples, size=500)
        p_value = normaltest(test_samples)[1]
        if p_value > alpha:
            self.normal = True
        else:
            self.normal = False

    @property
    def n(self):
        """
        Get the median value of the distribution (for a normal distribution this is the same as the mean).

        Returns:
            :py:attr:`float`: Median value.
        """
        return np.percentile(self.samples, [50])[0]

    @property
    def s(self, ddof=1):
        """
        Get the standard deviation of the distribution. For a non-normal distribution, this will return :py:attr:`None`.

        Args:
            ddof (:py:attr:`int`): Degrees of freedom to be included in calculation.

        Returns:
            :py:attr:`float` or :py:attr:`None`: Standard deviation of the distribution.
        """
        if self.normal:
            return np.std(self.samples, ddof=ddof)
        else:
            return None

    @property
    def v(self, ddof=1):
        """
        Get the variance of the distribution. For a non-normal distribution, this will return :py:attr:`None`.

        Args:
            ddof (:py:attr:`int`): Degrees of freedom to be included in calculation.

        Returns:
            :py:attr:`float` or :py:attr:`None`: Standard deviation of the distribution.
        """
        if self.normal:
            return np.var(self.samples, ddof=ddof)
        else:
            return None

    @property
    def con_int(self):
        """
        Get the extrema of the confidence intervals of the distribution.

        Returns:
            :py:attr:`array_like`: Distribution values at the confidence interval.
        """
        return np.percentile(self.samples, self.ci_points)

    def __repr__(self):
        """
        A custom representation, which is the same as the custom string representation.

        Returns:
            :py:attr:`str`: Description of the distribution.
        """
        return self.__str__()

    def __str__(self):
        """
        A custom string.

        Returns:
            :py:attr:`str`: Description of the distribution.
        """
        representation = "Distribution: {}\nSize: {}\n".format(self.name, self.size)
        representation += "Samples: "
        representation += "["
        representation += " ".join(["{:.2e}".format(i) for i in self.samples[:2]])
        representation += " ... "
        representation += " ".join(["{:.2e}".format(i) for i in self.samples[-2:]])
        representation += "]\n"
        representation += "Median: {:.2e}\n".format(self.n)
        if self.normal:
            representation += "Symetrical Error: {:.2e}\n".format(self.s)
        representation += "Confidence intervals: ["
        representation += " ".join(["{:.2e}".format(i) for i in self.con_int])
        representation += "]\n"
        representation += "Confidence interval points: ["
        representation += " ".join(["{}".format(i) for i in self.ci_points])
        representation += "]\n"
        if self.n is not None:
            representation += "Reporting Value: "
            if self.normal:
                representation += "{}\n".format(ufloat(self.n, self.s))
            else:
                representation += "{:.2e}+{:.2e}-{:.2e}\n".format(
                    self.n, self.con_int[1] - self.n, self.n - self.con_int[0]
                )
        representation += "Unit: {}\n".format(self.unit)
        representation += "Normal: {}\n".format(self.normal)
        return representation

    def add_samples(self, samples):
        """
        Add samples to the distribution.

        Args:
            samples (:py:attr:`array_like`): Samples to be added to the distribution.
        """
        self.samples = np.append(self.samples, np.array(samples).flatten())
        self.check_normality()
