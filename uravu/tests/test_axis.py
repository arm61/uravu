"""
Tests for axis module
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey


import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from uravu.distribution import Distribution
import scipy.stats
from uravu.axis import Axis
from scipy.stats import norm, uniform, gaussian_kde


DISTRO1 = Distribution(norm.rvs(loc=0, scale=1, size=10000, random_state=np.random.RandomState(1)))
DISTRO2 = Distribution(norm.rvs(loc=1, scale=1, size=10000, random_state=np.random.RandomState(2)))
AX = Axis([DISTRO1, DISTRO2])

AX_ARRAY = Axis([0, 1])


class TestDistribution(unittest.TestCase):
    """
    Testing the Axis class.
    """
    def test_init_values(self):
        assert_equal(AX.values[0].samples, DISTRO1.samples)
        assert_equal(AX.values[1].samples, DISTRO2.samples)

    def test_init_kde(self):
        assert_equal(isinstance(AX.kde, gaussian_kde), True)

    def test_init_kde_size_change(self):
        distro2 = Distribution(norm.rvs(loc=1, scale=1, size=1000, random_state=np.random.RandomState(2)))
        AX = Axis([DISTRO1, distro2])
        assert_equal(AX.values[1].samples, distro2.samples)

    def test_n(self):
        assert_almost_equal(AX.n, [0, 1], decimal=1)

    def test_n_array(self):
        assert_equal(AX_ARRAY.n, [0, 1])

    def test_s(self):
        assert_almost_equal(AX.s, [[1.96, 1.96], [1.96, 1.96]], decimal=1)

    def test_s_array(self):
        assert_equal(AX_ARRAY.s, [0, 0])

    def test_mode(self):
        assert_almost_equal(AX.mode, [0, 1], decimal=1)

    def test_mode_array(self):
        assert_equal(AX_ARRAY.mode, [0, 1])

    def test_size(self):
        assert_equal(AX.size, 2)

    def test_size_array(self):
        assert_equal(AX_ARRAY.size, 2)

    def test_shape(self):
        assert_equal(AX.shape, 2)

    def test_shape_array(self):
        ax = Axis(np.ones((3, 3)))
        assert_equal(ax.shape, (3, 3))

    def test_pdf(self):
        assert_almost_equal(AX.pdf([0, 1]), [0.1495], decimal=0)

    def test_logpdf(self):
        assert_almost_equal(AX.logpdf([0, 1]), np.log([0.1495]), decimal=1)
