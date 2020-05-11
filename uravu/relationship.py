"""
The :py:class:`~uravu.relationship.Relationship` class is a foundational component of the :py:mod:`uravu` package, and acts as the main API for use of the package.
This class enables the storage of the relationship between the model and the data.

Objects of this class offer easy methods to perform maximum likelihood evaluation, Markiv chain Monte Carlo (MCMC) for posterior probabiltiy determination and Bayesian evidence estimation by nested sampling.

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
from uncertainties import ufloat
from uravu import UREG, optimize, sampling, __version__
from uravu.distribution import Distribution


class Relationship:
    """
    The :py:class:`~uravu.relationship.Relationship` class is the base of the :py:mod:`uravu` package, enabling the use of Bayesian inference for the assessment of a model's ability to describe some data.

    Attributes:
        function (:py:attr:`callable`): The function that is modelled.
        abscissa (:py:attr:`array_like` with :py:class:`~pint.unit.Unit`): The abscissa data that the modelling should be performed from. This includes some unit from :py:class:`~pint.unit.Unit`.
        abscissa_name (:py:attr:`str`): A name for the abscissa data, used in the production of plots.
        ordinate (:py:attr:`array_like` with :py:class:`~pint.unit.Unit`): The ordinate data against with the model should be compared. This may include uncertainty values and some unit.
        ordinate_name (:py:attr:`str`): A name for the ordinate data, used in the production of plots.
        variables (:py:attr:`list` of :py:attr:`float` or :py:class:`~uravu.distribution.Distribution`): Variables in the :py:attr:`~uravu.relationship.Relationship.function`.
        unaccounted_uncertainty (:py:attr:`bool`): This boolean describes if an unaccounted for uncertainty should be considered in the modelling process.
        bounds (:py:attr:`tuple`): The minimum and maximum values for each parameters.
        ln_evidence (:py:class:`uncertainties.core.Variable`): The natural-log of the Bayesian evidence for the model to the given data.
        mcmc_results (:py:attr:`dict`): The results from :func:`emcee.EnsembleSampler.run_mcmc()` sampling.
        nested_sampling_results (:py:attr:`dict`): The results from :func:`ddynesty.NestedSampler.run_nested()` nested sampling.

    Args:
        function (:py:attr:`callable`): The functional relationship to be modelled.
        abscissa (:py:attr:`array_like`): The abscissa data. If multi-dimensional, the array is expected to have the shape :py:attr:`(N, d)`, where :py:attr:`N` is the number of data points and :py:attr:`d` is the dimensionality.
        ordinate (:py:attr:`array_like`): The ordinate data. This should have a shape :py:attr:`(N,)`.
        ordinate_uncertainty (:py:attr:`array_like`, optional): The uncertainty in each of the ordinate data points. This should have a shape :py:attr:`(N,)`. Default to no uncertainties on the ordiante. If there is no ordinate uncertainty, an unaccounted uncertainty is automatically added.
        abscissa_unit (:py:class:`~pint.unit.Unit`, optional): The unit for the :py:attr:`abscissa`. If :py:attr:`abscissa` is multi-dimensional, this should be a list with the units for each dimension. Default is :py:attr:`~pint.unit.Unit.dimensionless`.
        ordinate_unit (:py:class:`~pint.unit.Unit`, optional): The unit for the :py:attr:`ordinate`. Default is :py:attr:`~pint.unit.Unit.dimensionless`.
        abscissa_name (:py:attr:`str`, optional): A name for the :py:attr:`abscissa`. Default is :py:attr:`'x'`.
        ordinate_name (:py:attr:`str`, optional): A name for the :py:attr:`ordinate`. Default is :py:attr:`'y'`.
        variable_names (:py:attr:`list` of :py:attr:`str`, optional): Names for each of the variables. Default is the variable name in the :py:attr:`function` definition.
        variable_units (:py:class:`~pint.unit.Unit`, optional): The units for the variables. Default is :py:attr:`~pint.unit.Unit.dimensionless`.
        bounds (:py:attr:`tuple`): The minimum and maximum values for each parameters. Defaults to :py:attr:`None`.
        unaccounted_uncertainty (:py:attr:`bool`, optional): Describes if an additional variable be included to account for an unknown uncertainty in the data.
    """

    def __init__(
        self,
        function,
        abscissa,
        ordinate,
        ordinate_uncertainty=None,
        abscissa_unit=UREG.dimensionless,
        ordinate_unit=UREG.dimensionless,
        abscissa_name="x",
        ordinate_name="y",
        variable_names=None,
        variable_units=None,
        bounds=None,
        unaccounted_uncertainty=False,
    ):
        self.function = function
        self.unaccounted_uncertainty = unaccounted_uncertainty
        abscissa = np.array(abscissa)
        ordinate = np.array(ordinate)

        if ordinate_uncertainty is None:
            self.unaccounted_uncertainty = True
            self.ordinate = ordinate
        else:
            if ordinate_uncertainty.shape != ordinate.shape:
                raise ValueError(
                    "The number of data points in the ordinate does not "
                    "match that in the ordinate uncertainty."
                )
            else:
                ordinate_uncertainty = np.array(ordinate_uncertainty)
                self.ordinate = unp.uarray(ordinate, ordinate_uncertainty)
        self.ordinate *= ordinate_unit

        self.abscissa = abscissa
        self.abscissa *= abscissa_unit

        if abscissa.shape[0] != ordinate.shape[0]:
            raise ValueError(
                "The number of data points in the abscissa does "
                "not match that for the ordinate."
            )

        self.variables = np.ones((self.len_parameters()))
        self.abscissa_name = abscissa_name
        self.ordinate_name = ordinate_name
        if variable_names is None:
            self.variable_names = getfullargspec(self.function).args[1:]
            if self.unaccounted_uncertainty:
                self.variable_names.append("unaccounted uncertainty")
        else:
            if len(variable_names) != self.len_parameters():
                raise ValueError(
                    "The number of variable names does not match the number "
                    "of variables."
                )
            self.variable_names = variable_names
        if variable_units is None:
            self.variable_units = [
                UREG.dimensionless for i in range(self.len_parameters())
            ]
            if self.unaccounted_uncertainty:
                self.variable_units.append(self.ordinate.u)
        else:
            if len(variable_units) != self.len_parameters():
                raise ValueError(
                    "The number of variable units does not match the number "
                    "of variables."
                )
            self.variable_units = variable_units
        self.bounds = bounds
        if (self.unaccounted_uncertainty) and (self.bounds is not None):
            self.bounds = self.bounds + ((-10, 11),)
        self.ln_evidence = None
        self.mcmc_results = None
        self.nested_sampling_results = None

    def __str__(self):
        """
        A custom string function.

        Returns:
            :py:attr:`str`: Custom string description.
        """
        string = "Function Name: {} \n".format(self.function.__name__)
        if self.abscissa.shape[0] < 4:
            string += "Abscissa: [ "
            for i in self.x_n:
                string += "{:.2e} ".format(i)
            string += "] \n"
            string += "Ordinate: [ "
            for i in self.y_n:
                string += "{:.2e} ".format(i)
            string += "] \n"
            if isinstance(self.ordinate.m.any(), uncertainties.core.AffineScalarFunc):
                string += "Ordinate uncertainty: [ "
                for i in self.y_s:
                    string += "{:.2e} ".format(i)
                string += "] \n"
        else:
            string += "Abscissa: " "[ {:.2e} {:.2e} ... {:.2e} {:.2e} ] \n".format(
                *self.x_n[:2], *self.x_n[-2:]
            )
            string += "Ordinate: " "[ {:.2e} {:.2e} ... {:.2e} {:.2e} ] \n".format(
                *self.y_n[:2], *self.y_n[-2:]
            )
            if isinstance(self.ordinate.m.any(), uncertainties.core.AffineScalarFunc):
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
                    string += "{:.2e}+/-{:.2e} ".format(var.n, var.n - var.con_int[0])
                else:
                    string += "{:.2e}+{:.2e}-{:.2e} ".format(
                        var.n, var.con_int[1] - var.n, var.n - var.con_int[0]
                    )
            else:
                string += "{:.2e} ".format(var)
        string += "] \n"
        if self.nested_sampling_done:
            string += "ln(evidence): {:.2e} \n".format(self.ln_evidence)
        string += "Unaccounted uncertainty: {} \n".format(self.unaccounted_uncertainty)
        string += "MCMC performed: {} \n".format(self.mcmc_done)
        string += "Nested sampling performed: {} \n".format(self.nested_sampling_done)
        return string

    def __repr__(self):
        """
        A custom representation.

        Returns:
            :py:attr:`str`: Custom string description.
        """
        return self.__str__()

    @property
    def x(self):
        """
        Abscissa values with unit and uncertainty (where present).

        Returns:
            :py:attr:`array_like`: Abscissa values.
        """
        return self.abscissa

    @property
    def y(self):
        """
        Ordinate values with unit and uncertainty.

        Returns:
            :py:attr:`array_like`: Ordinate values.
        """
        return self.ordinate

    @property
    def x_m(self):
        """
        Abscissa values without units.

        Returns:
            :py:attr:`array_like`: Abscissa values without units.
        """
        return self.abscissa.m

    @property
    def y_m(self):
        """
        Ordinate values without units.

        Returns:
            :py:attr:`array_like`: Ordinate values without units.
        """
        return self.ordinate.m

    @property
    def x_u(self):
        """
        Abscissa unit.

        Returns:
            :py:class:`~pint.unit.Unit`: Abscissa values.
        """
        return self.abscissa.u

    @property
    def y_u(self):
        """
        Ordinate unit.

        Returns:
            :py:class:`~pint.unit.Unit`: Ordinate unit.
        """
        return self.ordinate.u

    @property
    def x_n(self):
        """
        Abscissa nominal values.

        Returns:
            :py:attr:`array_like`: Abscissa nominal values.
        """
        return unp.nominal_values(self.abscissa.m)

    @property
    def y_n(self):
        """
        Ordinate nominal values.

        Returns:
            :py:attr:`array_like`: Ordinate nominal values.
        """
        return unp.nominal_values(self.ordinate.m)

    @property
    def y_s(self):
        """
        Ordinate uncertainties.

        Returns:
            :py:attr:`array_like`: Ordinate uncertainties.
        """
        if isinstance(self.ordinate.m.any(), uncertainties.core.AffineScalarFunc):
            return unp.std_devs(self.ordinate.m)
        else:
            return None

    @property
    def variable_medians(self):
        """
        The median values for each of the variables.

        Returns:
            :py:attr:`array_like`: Variable medians.
        """
        medians = np.zeros((len(self.variables)))
        for i, var in enumerate(self.variables):
            if isinstance(var, Distribution):
                medians[i] = var.n
            if isinstance(var, float):
                medians[i] = var
        return medians

    @property
    def mcmc_done(self):
        """
        Has MCMC been performed? Determined based on the type of the variables.

        Returns:
            :py:attr:`bool`: Has MCMC been performed?
        """
        for var in self.variables:
            if isinstance(var, Distribution):
                return True
        return False

    @property
    def nested_sampling_done(self):
        """
        Has nested sampling been performed? Determined based on if the ln_evidence has a value.

        Returns:
            :py:attr:`bool`: Has nested sampling been performed?
        """
        if self.ln_evidence is not None:
            return True
        return False

    @property
    def citations(self):
        """
        Return the relevant references based on the analysis performed.

        Returns:
            :py:attr:`str`: Relevant references.
        """
        string = "Please consider citing the following:\n"
        string += " - Publication of uravu (to come).\n"
        string += " - Zenodo DOI for uravu version: {}\n".format(__version__())
        if not np.isclose(
            self.variable_medians, np.ones((self.len_parameters()))
        ).all():
            string += (
                "The scipy.optimize.minimize function was used to "
                "maximise the ln likelihood. Please consider citing:\n"
            )
            string += (
                " - P. Virtanen, R. Gommers, T. E. Oliphant, M. "
                "Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. "
                "Peterson, W. Weckesser, J. Bright, S. J. van der "
                "Walt, M. Brett, J. Wilson, K. J. Millman, N. "
                "Mayorov, A. R. J. Nelson, E. Jones, R. Kern, E. "
                "Larson, C. Carey, I. Polat, Y. Feng, E. W. Moore, "
                "J. VanderPlas, D. Laxalde, J. Perktold, R. Cimrman, "
                "I. Henriksen, E. A. Quintero, C. R Harris, A. M. "
                "Archibald, A. H. Ribeiro, F. Pedregosa, P. van "
                "Mulbregt, & SciPy 1.0 Contributors, (2020). Nature "
                "Methods, in press. DOI: 10.1038/s41592-019-0686-2\n"
            )
        if self.mcmc_done:
            string += (
                "The emcee package was used to perform the MCMC "
                "analysis. Please consider citing:\n"
            )
            string += (
                " - D. Foreman-Mackey, W. Farr, M. Sinha, A. "
                "Archibald, D. Hogg, J. Sanders, J. Zuntz, P. "
                "Williams, A. Nelson, M. de Val-Borro, T. Erhardt, "
                "I. Pashchenko, & O. Pla, (2019). Journal of Open "
                "Source Software, 4(43), 1864. DOI: "
                "10.21105/joss.01864\n"
            )
            string += (
                " - J. Goodman & J. Weare, (2010). Communications in "
                "applied mathematics and computational science, 5(1), "
                "65. DOI: 10.2140/camcos.2010.5.65\n"
            )
        if self.nested_sampling_done:
            string += (
                "The dynesty package was used to carry out the nested "
                "sampling. Please consider citing:\n"
            )
            string += (
                " - J. S. Speagle, (2019), Monthly Notices of the "
                "Royal Astronomical Society, staa278. DOI: "
                "10.1093/mnras/staa278.\n"
            )
            string += (
                " - J. Skilling (2004), AIP Conference Proceedings, "
                "735(1), 395. DOI: 10.1063/1.1835238.\n"
            )
            string += (
                " - J. Skilling (2006), Bayesian Analysis, "
                "1(4), 833. DOI: 10.1214/06-BA127.\n"
            )
        return string

    def len_parameters(self):
        """
        Determine the number of variables in the assessment function.

        Returns:
            :py:attr`int`: Number of variables.
        """
        # The minus one is to remove the abscissa data which is a
        # argument in the assessment function
        if self.unaccounted_uncertainty:
            return len(getfullargspec(self.function).args)
        return len(getfullargspec(self.function).args) - 1

    def bayesian_information_criteria(self):
        """
        Calculate the `Bayesian information criteria`_ for the relationship.

        Returns:
            :py:attr:`float`: Bayesian information criteria.

        .. _Bayesian information criteria: https://en.wikipedia.org/wiki/Bayesian_information_criterion
        """
        self.max_likelihood('mini')
        return np.log(
            self.x_n.size
        ) * self.len_parameters() - 2 * optimize.ln_likelihood(
            self.variable_medians,
            self.function,
            self.abscissa,
            self.ordinate,
            self.unaccounted_uncertainty,
        )

    def max_likelihood(self, method, x0=None, **kwargs):
        """
        Determine values for the variables which maximise the likelihood for the :py:class:`~uravu.relationship.Relationship`. For keyword arguments see the :func:`scipy.optimize.minimize()` documentation.
        
        Args:
            x0 (:py:attr:`array_like`): Initial guess values for the parameters.
        """
        self.variables = optimize.max_ln_likelihood(self, method, x0, **kwargs)

    def prior(self):
        """
        *Standard priors* for the relationship. These priors are broad, uninformative, for normal variables running the range :py:attr:`[x - x * 10, x + x * 10)` (where :py:attr:`x` is the variable value). For an unaccounted uncertainty natural log factor the range is :py:attr:`[-10, 1)`.

        Returns:
            :py:attr:`list` of :py:class:`scipy.stats.rv_continuous`: :py:mod:`scipy.stats` functions describing the priors.
        """
        priors = []
        if self.bounds is not None:
            for i, var in enumerate(self.variable_medians):
                loc = self.bounds[i][0]
                scale = self.bounds[i][1] - loc
                priors.append(uniform(loc=loc, scale=scale))
        else:
            for var in self.variable_medians:
                loc = var - np.abs(var) * 10
                scale = (var + np.abs(var) * 10) - loc
                priors.append(uniform(loc=loc, scale=scale))
        if self.unaccounted_uncertainty:
            priors[-1] = uniform(loc=-10, scale=11)
        return priors

    def mcmc(
        self,
        prior_function=None,
        walkers=100,
        n_samples=500,
        n_burn=500,
        progress=True,
        seed=None,
    ):
        """
        Perform MCMC to get the posterior probability distributions for the variables of the relationship. *Note*, running this method will populate the :py:attr:`~uravu.relationship.Relationship.variables` attribute with :py:class:`~uravu.distribution.Distribution` objects.

        Args:
            prior_function (:py:attr:`callable`, optional): The function to populated some prior distributions. Default is the broad uniform priors in :func:`~uravu.relationship.Relationship.prior()`.
            walkers (:py:attr:`int`, optional): Number of MCMC walkers. Default is :py:attr:`100`.
            n_samples (:py:attr:`int`, optional): Number of sample points. Default is :py:attr:500`.
            n_burn (:py:attr:`int`, optional): Number of burn in samples. Default is :py:attr:`500`.
            progress (:py:attr:`bool`, optional): Show :py:mod:`tqdm` progress for sampling. Default is :py:attr:`True`.
        """
        self.mcmc_results = sampling.mcmc(
            self,
            prior_function=prior_function,
            walkers=walkers,
            n_samples=n_samples,
            n_burn=n_burn,
            progress=progress,
        )
        self.variables = self.mcmc_results["distributions"]

    def nested_sampling(self, prior_function=None, progress=True, **kwargs):
        """
        Perform nested sampling to determine the Bayesian natural-log evidence. For keyword arguments see the :func:`dynesty.NestedSampler.run_nested()` documentation.

        Args:
            prior_function (:py:attr:`callable`, optional): The function to populate some prior distributions. Default is the broad uniform priors in :func:`~uravu.relationship.Relationship.prior()`.
            progress (:py:attr:`bool`, optional): Show :py:mod:`tqdm` progress for sampling. Default is :py:attr:`True`.
        """
        self.nested_sampling_results = sampling.nested_sampling(
            self, prior_function=prior_function, progress=progress, **kwargs
        )
        self.ln_evidence = ufloat(
            self.nested_sampling_results["logz"][-1],
            self.nested_sampling_results["logzerr"][-1],
        )
