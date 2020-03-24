"""
The Distribution class allows the storage and analysis of probability
distributions.
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
    The storage of probability distributions is fundamental to Bayesian
    inference, this class enables this. In addition to storage some basic
    analysis of the distribution is possible.

    Attributes:
        name (str): A name for the distribution.
        unit (pint.UnitRegistry(), optional): The unit of the values in the
            Distribution.
        samples (array_like): Samples in the distribution.
        ci_points (array_like, optional): The percentiles at which
            confidence intervals should be found. Default is
            `[2.5, 97.5]` (a 95 % confidence interval).
    """

    def __init__(
        self,
        samples,
        name="Distribution",
        ci_points=None,
        unit=UREG.dimensionless,
    ):
        """
        Args:
            samples (array_like): Sample for the distribution.
            name (str, optional): A name to identify the distribution.
                Default is `Distribution`.
            ci_points (array_like, optional): The percentiles at which
                confidence intervals should be found. Default is
                `[2.5, 97.5]` (a 95 % confidence interval).
            unit (pint.UnitRegistry(), optional) The unit for the
                distribution. Default is dimensionless.
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
    def mean(self):
        """
        Get the mean value and uncertainty. Will return ``None`` if
        distribution is not normal.

        Returns:
            (uncertainties.core.Variable or None): Mean value with uncertainty.
        """
        if self.normal:
            return ufloat(self.n, self.s)
        return None

    @property
    def size(self):
        """
        Get the number of samples in the distribution.

        Returns:
            (int): Number of samples.
        """
        return self.samples.size

    def check_normality(self):
        """
        Uses a Shapiro-Wilks statistical test to evaluate if samples are
        normally distributed.
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
        Get the median value of the distribution (for a normal distribution
        this is the same as the mean).

        Returns:
            (float): Median value.
        """
        return np.percentile(self.samples, [50])[0]

    @property
    def s(self):
        """
        Get the standard deviation of the distribution. For a non-normal
        distribution, this will return ``None``.

        Returns:
            (float, or None): Standard deviation of the distribution.
        """
        if self.normal:
            return np.std(self.samples)
        else:
            return None

    @property
    def con_int(self):
        """
        Get the extrema of the confidence intervals of the distribution.

        Returns:
            (array_like): The confidence interval values.
        """
        return np.percentile(self.samples, self.ci_points)

    def __repr__(self):
        """
        A custom representation, which is the same as the custom string
        representation.

        Returns:
            (str): String representation.
        """
        return self.__str__()

    def __str__(self):
        """
        A custom string.

        Returns:
            (str): Detailed string representation.
        """
        representation = "Distribution: {}\nSize: {}\n".format(
            self.name, self.size
        )
        representation += "Samples: "
        representation += "["
        representation += " ".join(
            ["{:.2e}".format(i) for i in self.samples[:2]]
        )
        representation += " ... "
        representation += " ".join(
            ["{:.2e}".format(i) for i in self.samples[-2:]]
        )
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
                    self.n, self.con_int[1]-self.n, self.n-self.con_int[0]
                )
        representation += "Unit: {}\n".format(self.unit)
        representation += "Normal: {}\n".format(self.normal)
        return representation

    def add_samples(self, samples):
        """
        Add samples to the distribution and update values such as median and
        uncertainties as appropriate.

        Args:
            samples (array_like): Samples to be added to the distribution.
        """
        self.samples = np.append(self.samples, np.array(samples).flatten())
        self.check_normality()
