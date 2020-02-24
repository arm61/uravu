"""
The ``Relationship`` class is a foundational component of the ``uravu``
package, and acts as an API for use of the package.
This class enables the storage of the relationship between the model and
the data.

Objects of this class offer easy methods to perform maximum likelihood
evaluation, Markiv chain Monte Carlo (MCMC) for posterior probabiltiy
determination and Bayesian evidence estimation by nested sampling.

See the `tutorials online`_ for more guidence of how to use this package.

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
    The ``Relationship`` class is the base of the ``uravu`` package, enabling
    the use of Bayesian inference for the assessment of a model's ability to
    describe some data.

    Attributes:
        function (callable): The function that is modelled.
        abscissa (array_like with pint.UnitRegistry()): The abscissa data
            that the modelling should be performed from. This includes some
            unit from the `pint`_ package and possibly some uncertainty from
            the `uncertainties`_ package.
        abscissa_name (str): A name for the abscissa data, used in the
            production of plots.
        ordinate (unumpy.array_like with pint.UnitRegistry()): The ordinate
            data against with the model should be compared. This will include
            uncertainty values and some unit.
        ordinate_name (str): A name for the ordinate data, used in the
            production of plots.
        unaccounted_uncertainty (bool): This boolean describes if an
            unaccounted for uncertainty should be considered in the modelling
            process.
        ln_evidence (uncertainties.core.Variable): The natural-log of the
            Bayesian evidence for the model to the given data.

    Args:
        function (callable): The functional relationship to be modelled.
        abscissa (array_like): The abscissa data. If multi-dimensional, the
            array is expected to have the shape `(N, d)`, where `N` is the
            number of data points and `d` is the dimensionality.
        ordinate (array_like): The ordinate data. This should have a
            shape `(N,)`.
        ordinate_uncertainty (array_like): The uncertainty in each of the
            ordinate data points. This should have a shape `(N,)`.
        abscissa_uncertainty (array_like, optional): The uncertainty in each of
            the absiccsa data points. This should have a shape `(N, d)`.
            Default is no uncertainties on absicca.
        abscissa_unit (pint.UnitRegistry()): The unit for the abscissa.
            If `abscissa` is multi-dimensional, this should be a list with
            the units for each dimension.
        ordinate_unit (pint.UnitRegistry()): The unit for the ordinate.
        abscissa_name (str, optional): A name for the abscissa. Default
            is `'x'`.
        ordinate_name (str, optional): A name for the ordinate. Default
            is `'y'`.
        unaccounted_uncertainty (bool, optional): Describes if an additional
            variable be included to account for an unknown uncertainty in the
            data.

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
                    "The number of data points in the ordinate does not "
                    "match that in the ordinate uncertainty."
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

    def __str__(self):
        """
        A custom string function.

        Returns:
            (str): Custom string description.
        """
        string = "Function Name: {} \n".format(self.function.__name__)
        if self.abscissa.shape[0] < 4:
            string += "Abscissa: [ "
            for i in self.x_n:
                string += "{:.2e} ".format(i)
            string += "] \n"
            if isinstance(
                self.abscissa.m.any(), uncertainties.core.AffineScalarFunc
            ):
                string += "Abscissa uncertainty: [ "
                for i in self.x_s:
                    string += "{:.2e} ".format(i)
                string += "] \n"
            string += "Ordinate: [ "
            for i in self.y_n:
                string += "{:.2e} ".format(i)
            string += "] \n"
            string += "Ordinate uncertainty: [ "
            for i in self.y_s:
                string += "{:.2e} ".format(i)
            string += "] \n"
        else:
            string += (
                "Abscissa: "
                "[ {:.2e} {:.2e} ... {:.2e} {:.2e} ] \n".format(
                    *self.x_n[:2], *self.x_n[-2:]
                )
            )
            if isinstance(
                self.abscissa.m.any(), uncertainties.core.AffineScalarFunc
            ):
                string += (
                    "Abscissa: "
                    "[ {:.2e} {:.2e} ... {:.2e} {:.2e} ] \n".format(
                        *self.x_s[:2], *self.x_s[-2:]
                    )
                )
            string += (
                "Ordinate: "
                "[ {:.2e} {:.2e} ... {:.2e} {:.2e} ] \n".format(
                    *self.y_n[:2], *self.y_n[-2:]
                )
            )
            string += (
                "Ordinate uncertainty: "
                "[ {:.2e} {:.2e} ... {:.2e} {:.2e} ]\n".format(
                    *self.y_s[:2], *self.y_s[-2:]
                )
            )
        string += "Abscissa Name: {} \n".format(self.abscissa_name)
        string += "Ordinate Name: {} \n".format(self.ordinate_name)
        string += "Abscissa Unit: {} \n".format(self.x_u)
        string += "Ordinate Unit: {} \n".format(self.y_u)
        string += "Variables: [ "
        for var in self.variables:
            if isinstance(var, Distribution):
                if var.normal:
                    string += "{:.2e}+/-{:.2e} ".format(
                        var.n, var.n - var.con_int[0]
                    )
                else:
                    string += "{:.2e}+{:.2e}-{:.2e} ".format(
                        var.n, var.con_int[1] + var.n, var.n - var.con_int[0]
                    )
            else:
                string += "{:.2e} ".format(var)
        string += "] \n"
        if self.nested_sampling_done:
            string += "ln(evidence): {:.2e} \n".format(self.ln_evidence)
        string += "Unaccounted uncertainty: {} \n".format(
            self.unaccounted_uncertainty
        )
        string += "MCMC performed: {} \n".format(self.mcmc_done)
        string += "Nested sampling performed: {} \n".format(
            self.nested_sampling_done
        )
        return string

    def __repr__(self):
        """
        A custom representation.

        Returns:
            (str): Custom string description.
        """
        return self.__str__()

    @property
    def x(self):
        """
        Abscissa values with unit and uncertainty (where present).

        Returns:
            (array_like): Abscissa values.
        """
        return self.abscissa

    @property
    def y(self):
        """
        Ordinate values with unit and uncertainty.

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
            (array_like or None): Abscissa uncertainties.
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
        The median values for each of the variables.

        Returns:
            (array_like): The variable medians.
        """
        means = np.zeros((len(self.variables)))
        for i, var in enumerate(self.variables):
            if isinstance(var, Distribution):
                means[i] = var.n
            if isinstance(var, float):
                means[i] = var
        return means

    @property
    def mcmc_done(self):
        """
        Has MCMC been performed, determined based on the type of the variables.

        Returns:
            (bool): Has MCMC been performed.
        """
        for var in self.variables:
            if isinstance(var, Distribution):
                return True
        return False

    @property
    def nested_sampling_done(self):
        """
        Has nested sampling been performed, determined based on if the
        ln_evidence has a value.

        Returns:
            (bool): Has nested sampling been performed.
        """
        if self.ln_evidence is not None:
            return True
        return False

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
        Determine values for the variables which maximise the likelihood
        for the relationship.
        """
        self.variables = optimize.max_ln_likelihood(self)

    def prior(self, array):
        """
        *Standard priors*. Broad, uniform distributions spread evenly
        around the current values for the variables.

        This is used as the default where no priors are given.

        Args:
            array (array_like): An array of random uniform numbers (0, 1].
                The shape of which is M x N, where M is the number of
                parameters and N is the number of walkers.

        Returns:
            (array_like): an array of random uniform numbers broadly
            distributed in the range [x - x * 10, x + x * 10), where x is the
            current variable value.
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
        Perform MCMC to get the posterior probability distributions for
        the variables of the relationship. Note running this method will
        populate the `relationship.variables` attribute with
        `uravu.distribution.Distribution` objects.

        Args:
            prior_function (callable, optional): the function to populated
                some prior distributions. Default is the broad uniform
                priors in uravu.relationship.Relationship.
            walkers (int, optional): Number of MCMC walkers. Default is `100`.
            n_samples (int, optional): Number of sample points. Default
                is `500`.
            n_burn (int, optional): Number of burn in samples. Default
                is `500`.
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
        Perform nested sampling to determine the Bayesian
        natural-log evidence.

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
