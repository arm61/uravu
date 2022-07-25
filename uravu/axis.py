"""
The :py:class:`~uravu.axis.Axis` class controls the organisation of axes in the uravu code, including the evaluation of an axis-level multidimensional kernal density estimate.
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey


from typing import Union, List, Tuple
import numpy as np
from scipy.stats import gaussian_kde
from uravu.distribution import Distribution


class Axis:
    """
    The Axes class is a flexible storage option for both numerical (:py:class:`numpy.ndarray`) and distribution (:py:class:`uravu.distribution.Distribution`) arrays.

    :param values: Array of values.
    """
    def __init__(self, values: Union[list, Distribution]) -> 'Axis':
        """
        Initialisation function for a :py:class:`~uravu.axis.Axis` object.
        """
        self._kde = None
        if isinstance(values[0], Distribution):
            self._values = values
            self._kde = _get_kde(self.values)
        else:
            self._values = np.array(values)

    def to_dict(self) -> dict:
        """
        :return: Dictionary of Axis.
        """
        if isinstance(self.values[0], Distribution):
            return {'values': [i.to_dict() for i in self.values]}
        else: 
            return {'values': self.values.tolist()}

    @classmethod
    def from_dict(cls, my_dict: dict) -> 'Axis':
        """
        Class method to produce from a dictionary.

        :param my_dict: Input dictionary
        
        :return: Axis from dictionary.
        """
        if isinstance(my_dict['values'][0], dict):
            v = [Distribution.from_dict(i) for i in my_dict['values']]
            return Axis(v)
        else:
            return Axis(my_dict['values'])

    @property
    def kde(self) -> 'scipy.stats._kde.gaussian_kde':
        """
        :return: Multi-dimensional kernel density estimation for the axis.
        """
        return self._kde

    @property
    def values(self) -> Union[List[Distribution], np.ndarray]:
        """
        :return: Array of values.
        """
        return self._values

    @property
    def n(self) -> np.ndarray:
        """
        :return: Medians for axis.
        """
        v = np.zeros(self.shape)
        if isinstance(self.values[0], Distribution):
            for i, o in enumerate(self.values):
                v[i] = o.n
            return v
        return self.values

    @property
    def s(self) -> np.ndarray:
        """
        :return: Uncertainties from confidence intervals for axis.
        """
        if isinstance(self.values[0], Distribution):
            dv = np.zeros((2, self.size))
            for i, o in enumerate(self.values):
                dv[:, i] = np.abs(o.con_int - o.n)
            return dv
        return np.zeros(self.shape)

    @property
    def mode(self) -> np.ndarray:
        """
        :return: Values that maximise the probability for axis.
        """
        v = np.zeros(self.shape)
        if isinstance(self.values[0], Distribution):
            for i, o in enumerate(self.values):
                v[i] = o.dist_max
            return v
        return self.values

    @property
    def size(self) -> int:
        """
        :return: Size of axis.
        """
        if isinstance(self.values[0], Distribution):
            return len(self.values)
        return self.values.size

    @property
    def shape(self) -> Union[int, Tuple[int]]:
        """
        :return: Shape of axis.
        """
        if isinstance(self.values[0], Distribution):
            return len(self.values)
        return self.values.shape

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """"
        Get the probability density function for all of the distributions in the axes.

        :param x: Values to return probability of.

        :return: Probability.
        """
        return self.kde.pdf(x)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """"
        Get the natural log probability density function for all of the distributions in the axes.

        :param x: Values to return natural log probability of.

        :return: Natural log probability.
        """
        return self.kde.logpdf(x)


def _get_kde(values: np.ndarray) -> 'scipy.stats._kde.gaussian_kde':
    """
    Determine the kernel density estimate for a given set of values.

    :param values: Sample for kde.

    :return: Kernel density estimate for samples.
    """
    min_size = values[0].size
    for v in values:
        if v.size < min_size:
            min_size = v.size
    random_list = [np.random.choice(v.samples, size=min_size) for v in values]
    return gaussian_kde(np.vstack(random_list))
