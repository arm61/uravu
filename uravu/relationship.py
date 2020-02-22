"""
The ``Relationship`` class is a foundational component of the 
``uravu`` package, and acts as an API for use of the package. 
This class enables the storage of the relationship between 
the model and the data. 

Objects of this class offer easy methods to perform maximum
likelihood evaluation, Markiv chain Monte Carlo (MCMC) for 
posterior probabiltiy determination and Bayesian evidence 
estimation by nested sampling. 

See the `tutorials online`_ for more guidence of how to use this package

.. _tutorials online: https://uravu.rtfd.io
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

from inspect import getfullargspec
import numpy as np
import uncertainties
from scipy.stats import uniform
from uncertainties import unumpy as unp
from uravu import UREG, optimize, sampling
from uravu.distribution import Distribution


class Relationship:
    """
    The ``Relationship`` class is the base of the ``uravu`` package, enabling the use of Bayesian inference for the assessment of a model's ability to describe some data.
    
    Attributes:
        function (callable): The function that is modelled.
        abscissa (array_like with pint.UnitRegistry()): The abscissa data that the modelling should be performed from. This includes some unit from the `pint`_ package and possibly some uncertainty from the `uncertainties`_ package.
        abscissa_name (str): A name for the abscissa data, used in the production of plots.
        ordinate (unumpy.array_like with pint.UnitRegistry()): The ordinate data against with the model should be compared. This will include uncertainty values and some unit. 
        ordinate_name (str): A name for the ordinate data, used in the production of plots.
        unaccounted_uncertainty (bool): This boolean describes if an unaccounted for uncertainty should be considered in the modelling process. 
        ln_evidence (uncertainties.core.Variable): The natural-log of the Bayesian evidence for the model to the given data. 
    
    Args:
        function (callable): The functional relationship to be modelled.
        abscissa (array_like): The abscissa data. If multi-dimensional, the array is expected to have the shape `(N, d)`, where `N` is the number of data points and `d` is the dimensionality.
        ordinate (array_like): The ordinate data. This should have a shape `(N,)`.
        ordinate_uncertainty (array_like): The uncertainty in each of the ordinate data points. This should have a shape `(N,)`.
        abscissa_uncertainty (array_like, optional): The uncertainty in each of the absiccsa data points. This should have a shape `(N, d)`. Default is no uncertainties on absicca.
        abscissa_unit (pint.UnitRegistry()): The unit for the abscissa. If `abscissa` is multi-dimensional, this should be a list with the units for each dimension.
        ordinate_unit (pint.UnitRegistry()): The unit for the ordinate.
        abscissa_name (str, optional): A name for the abscissa. Default is `'x'`.
        ordinate_name (str, optional): A name for the ordinate. Default is `'y'`.
        unaccounted_uncertainty (bool, optional): Describes if an additional variable be included to account for an unknown uncertainty in the data.
        
    .. _pint: https://pint.readthedocs.io/
    .. _uncertainties: https://uncertainties-python-package.readthedocs.io/
    """

    def __init__(
        self,
        function,
        abscissa,
        ordinate,
        ordinate_uncertainty,
        abscissa_uncertainty=None,
        abscissa_unit=UREG.dimensionless,
        ordinate_unit=UREG.dimensionless,
        abscissa_name="x",
        ordinate_name="y",
        unaccounted_uncertainty=False,
    ):
        self.function = function
        self.unaccounted_uncertainty = unaccounted_uncertainty
        abscissa = np.array(abscissa)
        ordinate = np.array(ordinate)
        ordinate_uncertainty = np.array(ordinate_uncertainty)
        if abscissa_uncertainty is not None:
            abscissa_uncertainty = np.array(abscissa_uncertainty)
        if abscissa.shape[0] == ordinate.shape[0]:
            if ordinate.shape[0] == ordinate_uncertainty.shape[0]:
                if (abscissa_uncertainty is not None) and (
                    abscissa.shape[0] == abscissa_uncertainty.shape[0]
                ):
                    self.abscissa = unp.uarray(abscissa, abscissa_uncertainty)
                else:
                    self.abscissa = abscissa
                self.abscissa *= abscissa_unit
                self.ordinate = (
                    unp.uarray(ordinate, ordinate_uncertainty) * ordinate_unit
                )
        else:
            raise ValueError(
                "The number of data points in the abscissa does "
                "not match that for the ordinate."
            )
        if unaccounted_uncertainty:
            self.variables = np.ones((self.len_parameters() + 1))
        else:
            self.variables = np.ones((self.len_parameters()))
        self.abscissa_name = abscissa_name
        self.ordinate_name = ordinate_name
        self.ln_evidence = None

    @property
    def x(self):
        """
        Abscissa values.

        Returns:
            (array_like): Abscissa values.
        """
        return self.abscissa

    @property
    def y(self):
        """
        Ordinate values.

        Returns:
            (array_like): Ordinate values.
        """
        return self.ordinate

    @property
    def x_m(self):
        """
        Abscissa values without units.

        Returns:
            (array_like): Abscissa values without units.
        """
        return self.abscissa.m

    @property
    def y_m(self):
        """
        Ordinate values without units.

        Returns:
            (array_like): Ordinate values without units.
        """
        return self.ordinate.m

    @property
    def x_u(self):
        """
        Abscissa unit.

        Returns:
            (pint.UnitRegistry()): Abscissa values.
        """
        return self.abscissa.u

    @property
    def y_u(self):
        """
        Ordinate unit.

        Returns:
            (pint.UnitRegistry()): Ordinate unit.
        """
        return self.ordinate.u

    @property
    def x_n(self):
        """
        Abscissa nominal values.

        Returns:
            (array_like): Abscissa nominal values.
        """
        return unp.nominal_values(self.abscissa.m)

    @property
    def x_s(self):
        """
        Abscissa uncertainties.

        Returns:
            (array_like): Abscissa uncertainties.
        """
        if isinstance(
            self.abscissa.m.any(), uncertainties.core.AffineScalarFunc
        ):
            return unp.std_devs(self.abscissa.m)
        else:
            return None

    @property
    def y_n(self):
        """
        Ordinate nominal values.

        Returns:
            (array_like): Ordinate nominal values.
        """
        return unp.nominal_values(self.ordinate.m)

    @property
    def y_s(self):
        """
        Ordinate uncertainties.

        Returns:
            (array_like): Ordinate uncertainties.
        """
        return unp.std_devs(self.ordinate.m)

    @property
    def variable_medians(self):
        """
        The mean values for the variables.

        Returns:
            (list of float): The variable medians
        """
        means = np.zeros((len(self.variables)))
        for i, var in enumerate(self.variables):
            if isinstance(var, Distribution):
                means[i] = var.n
            if isinstance(var, float):
                means[i] = var
        return means

    def len_parameters(self):
        """
        Determine the number of variables in the assessment function.

        Returns:
            (int): number of variables.
        """
        # The minus one is to remove the abscissa data which is a
        # argument in the assessment function
        return len(getfullargspec(self.function).args) - 1

    def max_likelihood(self):
        """
        Determine the variables which offer the maximum likelihood and assign
        these to the class variable.
        """
        self.variables = optimize.max_ln_likelihood(self)

    def prior(self, array):
        """
        A broad uniform distribution for the priors, this acts as the default
        when no priors are given to the MCMC or nested sampling.

        Args:
            array (array_like): Aa array of random uniform numbers (0, 1].
                The shape of which is M x N, where M is the number of
                parameters and N is the number of walkers.

        Returns:
            (array_like): an array of random uniform numbers broadly
                distributed in the range [x - x * 10, x + x * 10), where x is
                the current variable value.
        """
        broad = np.copy(array)
        for i, variable in enumerate(self.variable_medians):
            loc = variable - variable * 5
            scale = (variable + variable * 5) - loc
            broad[i] = uniform.ppf(broad[i], loc=loc, scale=scale)
        return broad

    def mcmc(
        self,
        prior_function=None,
        walkers=100,
        n_samples=500,
        n_burn=500,
        progress=True,
    ):
        """
        Perform MCMC to get the probability distributions for the variables
        of the relationship. Note running this method will populate the
        `relationship.variables` attribute with
        `uravu.distribution.Distribution` objects.

        Args:
            relationship (uravu.relationship.Relationship): the relationship to
                determine the posteriors of.
            prior_function (callable, optional): the function to populated some
                prior distributions. Default is the broad uniform priors in
                uravu.relationship.Relationship.
            walkers (int, optional): Number of MCMC walkers. Default is `100`.
            n_samples (int, optional): Number of sample points. Default is
                `500`.
            n_burn (int, optional): Number of burn in samples. Default is
                `500`.
            progress (bool, optional): Show tqdm progress for sampling.
                Default is `True`.
        """
        self.variables = sampling.mcmc(
            self,
            prior_function=prior_function,
            walkers=walkers,
            n_samples=n_samples,
            n_burn=n_burn,
            progress=progress,
        )

    def nested_sampling(self, prior_function=None, progress=True, **kwargs):
        """
        Perform the nested sampling in order to determine the Bayesian
        natural log evidence.

        Args:
            prior_function (callable, optional): the function to populated some
                prior distributions. Default is the broad uniform priors in
                uravu.relationship.Relationship.
            progress (bool, optional): Show tqdm progress for sampling.
                Default is `True`.

        Keyword Args:
            See the `dynesty.run_nested()` documentation.
        """
        self.ln_evidence = sampling.nested_sampling(
            self, prior_function=prior_function, progress=progress, **kwargs
        )
