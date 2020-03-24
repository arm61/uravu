"""
Tests for distribution module

Copyright (c) Andrew R. McCluskey

Distributed under the terms of the MIT License

@author: Andrew R. McCluskey
"""

# pylint: disable=R0201

import unittest
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from uravu.distribution import Distribution


EXPECTED_STRING1 = (
    "Distribution: Distribution\n"
    "Size: 4\n"
    "Samples: [1.62e+00 -6.12e-01 ... -5.28e-01 -1.07e+00]\n"
    "Median: -5.70e-01\n"
    "Confidence intervals: [-1.04e+00 1.46e+00]\n"
    "Confidence interval points: [2.5 97.5]\n"
    "Reporting Value: -5.70e-01+2.03e+00-4.68e-01\n"
    "Unit: dimensionless\n"
    "Normal: False\n"
)

EXPECTED_STRING2 = (
    "Distribution: Distribution\nSize: 1000\nSamples: [1.62e+00 -6.12e-01 "
    "... 3.54e-01 -1.87e-01]\nMedian: 4.13e-02\nSymetrical Error: "
    "9.81e-01\nConfidence intervals: [-1.96e+00 1.91e+00]\nConfidence "
    "interval points: [2.5 97.5]\nReporting Value: 0.0+/-1.0\nUnit: "
    "dimensionless\nNormal: True\n"
)


class TestDistribution(unittest.TestCase):
    """
    Testing the Distribution class.
    """

    def test_init_a(self):
        """
        Test initialisation with defaults.
        """
        distro = Distribution([1])
        assert_equal(distro.samples, np.array([1]))
        assert_equal(distro.n, 1)
        assert_almost_equal(distro.con_int, [1, 1])
        assert_almost_equal(distro.ci_points, [2.5, 97.5])

    def test_init_b(self):
        """
        Test initialisation without defaults.
        """
        distro = Distribution([1], ci_points=[5.0, 95.0])
        assert_equal(distro.samples, np.array([1]))
        assert_equal(distro.n, 1)
        assert_almost_equal(distro.con_int, [1, 1])
        assert_almost_equal(distro.ci_points, [5.0, 95.0])

    def test_init_c(self):
        """
        Test initialisation with bad ci_points.
        """
        with self.assertRaises(ValueError):
            Distribution([1, 2], ci_points=[5.0, 95.0, 102.0])

    def test_check_normality_true_less_than_5000(self):
        """
        Test check_normality with less than 5000 samples.
        """
        np.random.seed(1)
        distro = Distribution(np.random.randn(4999))
        assert_equal(distro.normal, True)

    def test_check_normality_true_more_than_5000(self):
        """
        Test check_normality with more than 5000 samples.
        """
        np.random.seed(1)
        distro = Distribution(np.random.randn(10000))
        assert_equal(distro.normal, True)

    def test_check_normality_false_more_than_5000(self):
        """
        Test check_normality with more than 5000 samples.
        """
        np.random.seed(1)
        distro = Distribution(np.random.rand(10000))
        assert_equal(distro.normal, False)

    def test_check_normality_false_less_than_5000(self):
        """
        Test check_normality with less than 5000 samples.
        """
        np.random.seed(1)
        distro = Distribution(np.random.rand(1000))
        assert_equal(distro.normal, False)

    def test_check_normality_less_than_3(self):
        """
        Test check_normality with more than 5000 samples.
        """
        np.random.seed(1)
        distro = Distribution(np.random.randn(2))
        assert_equal(distro.normal, False)

    def test_mean_a(self):
        """
        Test the mean parameter
        """
        np.random.seed(1)
        distro = Distribution(np.random.randn(2000))
        assert_almost_equal(distro.mean.n, 0.034504725958795925)
        assert_almost_equal(distro.mean.s, 1.0058360180170436)

    def test_mean_b(self):
        """
        Test the mean parameter when not normal
        """
        np.random.seed(1)
        distro = Distribution(np.random.random(2000))
        assert_equal(distro.mean, None)

    def test_add_samples_single(self):
        """
        Test add_samples with a single value.
        """
        distro = Distribution(1)
        assert_equal(distro.size, 1)
        assert_almost_equal(distro.samples, np.array([1]))
        assert_almost_equal(distro.n, 1)
        assert_equal(distro.s, None)
        assert_almost_equal(distro.con_int, [1, 1])
        assert_equal(distro.normal, False)

    def test_string_output_a(self):
        """
        Test the string that is printed.
        """
        np.random.seed(1)
        distro = Distribution(np.random.randn((1000)))
        assert_equal(distro.__str__(), EXPECTED_STRING2)

    def test_string_output_b(self):
        """
        Test the string that is printed.
        """
        np.random.seed(1)
        distro = Distribution(np.random.randn((4)))
        assert_equal(distro.__str__(), EXPECTED_STRING1)

    def test_repr_output(self):
        """
        Test the representation that is printed.
        """
        np.random.seed(1)
        distro = Distribution(np.random.randn((1000)))
        assert_equal(distro.__repr__(), EXPECTED_STRING2)
