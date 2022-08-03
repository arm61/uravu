"""
The storage and manipulation of probability distributions is fundamental to the operation of ``uravu`` and Bayesian inference.
The :py:class:`~uravu.distribution.Distribution` class oversees these operations.
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

from typing import Union, List
import numpy as np
from scipy.stats import normaltest, gaussian_kde
from scipy.optimize import minimize


class Distribution:
    """
    In addition to storage of the probability distribution, this class allows for some basic analysis, such as determination of normality.

    Attributes:
        samples (:py:attr:`array_like`): Samples in the distribution.
        name (:py:attr:`str`): Distribution name.
        normal (:py:attr:`bool`): Are the samples normally distributed?
        kde (:py:class:`scipy.stats.kde.gaussian_kde`): Kernel density approximation for the distribution.

    Args:
        samples (:py:attr:`array_like`): Sample for the distribution.
        name (:py:attr:`str`, optional): A name to identify the distribution. Default is :py:attr:`'Distribution'`.
        ci_points (:py:attr:`array_like`, optional): The two percentiles at which confidence intervals should be found. Default is :py:attr:`[2.5, 97.5]` (a 95 % confidence interval).

    .. _FAQ: ./faq.html
    """

    def __init__(self, samples: Union[List[Union[float, int]], np.ndarray], name: str="Distribution", random_state: np.random._generator.Generator=None) -> 'Distribution':
        """
        Initialisation function for a :py:class:`~uravu.distribution.Distribution` object.
        """
        self.name = name
        self.samples = np.array([])
        self.normal = False
        self._random_state = random_state
        if random_state is None:
            self._random_state = np.random.default_rng(np.random.randint(1))
        self.add_samples(np.array(samples))

    def to_dict(self) -> dict:
        """
        :return: Dictionary description of the object.
        """
        my_dict = {'name': self.name,
                   'samples': self.samples
                   }
        return my_dict
    
    @classmethod
    def from_dict(cls, my_dict: dict) -> 'Distribution':
        """
        :param my_dict: Dictionary description of the distribution.
        
        :return: Distribution object form the dictionary.
        """
        return cls(my_dict['samples'], name=my_dict['name'])

    @property
    def size(self) -> int:
        """
        :return: Number of samples in the distribution.
        """
        return self.samples.size

    def check_normality(self) -> bool:
        """
        Uses a :func:`scipy.stats.normaltest()` to evaluate if samples are normally distributed and updates 
        the :py:attr:`~uravu.distribution.Distribution.normal` attribute.
        """
        alpha = 0.05
        test_samples = self.samples
        if self.size > 500:
            test_samples = self._random_state.choice(self.samples, size=500)
        p_value = normaltest(test_samples)[1]
        if p_value > alpha:
            self.normal = True
        else:
            self.normal = False

    def pdf(self, x: Union[float, List[Union[float, int]], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get the probability density function for the distribution.

        :param x: Value to return probability of.
        :return: Probability.
        """
        return self.kde.pdf(x)

    def logpdf(self, x: Union[float, List[Union[float, int]], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get the natural log probability density function for the distribution.

        :param x: Value to return natural log probability of.
        :return: Natural log probability.
        """
        return self.kde.logpdf(x)

    def negative_pdf(self, x: Union[float, List[Union[float, int]], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Get the negative of the probability density function for the distribution.

        :param x: Value to return negative probability of.
        :return: Negative probability.
        """
        return -self.kde.pdf(x)

    @property
    def dist_max(self) -> float:
        """
        Get the value that maximises the distribution. If no :py:attr:`kde` has been created (for example
        if the distribution has fewer than 8 values) the median is returned.

        :return: Most likely value.
        """
        try:
            return minimize(self.negative_pdf, x0=[self.n]).x[0]
        except AttributeError:
            return self.n

    @property
    def min(self) -> float:
        """
        :return: Sample minimum.
        """
        return self.samples.min()

    @property
    def max(self) -> float:
        """
        :return: Sample maximum.
        """
        return self.samples.max()

    @property
    def n(self) -> float:
        """
        :return: Median value.
        """
        return np.percentile(self.samples, [50])[0]

    @property
    def s(self) -> Union[float, None]:
        """
        :return: Standard deviation of the distribution. For a non-normal distribution, this will return :py:attr:`None`.
        """
        if self.normal:
            return np.std(self.samples, ddof=1)
        return None

    @property
    def v(self) -> Union[float, None]:
        """
        :return: Standard deviation of the distribution. For a non-normal distribution, this will return :py:attr:`None`.
        """
        if self.normal:
            return np.var(self.samples, ddof=1)
        return None

    def con_int(self, ci_points: List[float]=[2.5, 97.5]) -> np.ndarray:
        """
        Get the extrema of the confidence intervals of the distribution.

        :param ci_points: The confidence interval points to return.
        :return: Distribution values at the confidence interval.
        """
        return np.percentile(self.samples, ci_points)

    def add_samples(self, samples: Union[List[Union[float, int]], np.ndarray]):
        """
        Add samples to the distribution.

        Args:
            samples (:py:attr:`array_like`): Samples to be added to the distribution.
        """
        self.samples = np.append(self.samples, np.array(samples).flatten())
        if self.size > 8:
            self.check_normality()
            self.kde = gaussian_kde(self.samples)

    def __repr__(self) -> np.ndarray:
        """
        :return: Representation.
        """
        return self.samples

    def __str__(self) -> np.ndarray:
        """
        :return: String representation.
        """
        return self.samples
