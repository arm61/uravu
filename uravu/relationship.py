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
from scipy import stats
from uncertainties import ufloat
from uravu import optimize, sampling
from uravu.distribution import Distribution
from uravu.axis import Axis


class Relationship:
    """
    The :py:class:`~uravu.relationship.Relationship` class is the base of the :py:mod:`uravu` package, enabling the use of Bayesian inference for the assessment of a model's ability to describe some data.

    Attributes:
        function (:py:attr:`callable`): The function that is modelled.
        abscissa (:py:attr:`array_like`): The abscissa data that the modelling should be performed on.
        ordinate (:py:attr:`list` or :py:class:`uravu.distribution.Distribution` or :py:attr:`array_like`): The ordinate data against with the model should be compared. This should be an :py:attr:`list` or :py:class:`uravu.distribution.Distribution` unless a :py:attr:`ordinate_error` is given.
        variables (:py:attr:`list` of :py:class:`uravu.distribution.Distribution`): Variables in the :py:attr:`~uravu.relationship.Relationship.function`.
        bounds (:py:attr:`tuple`): The minimum and maximum values for each parameters.
        ln_evidence (:py:class:`uncertainties.core.Variable`): The natural-log of the Bayesian evidence for the model to the given data.
        mcmc_results (:py:attr:`dict`): The results from :func:`emcee.EnsembleSampler.run_mcmc()` sampling.
        nested_sampling_results (:py:attr:`dict`): The results from :func:`dynesty.NestedSampler.run_nested()` nested sampling.

    Args:
        function (:py:attr:`callable`): The functional relationship to be modelled.
        abscissa (:py:attr:`array_like`): The abscissa data. If multi-dimensional, the array is expected to have the shape :py:attr:`(N, d)`, where :py:attr:`N` is the number of data points and :py:attr:`d` is the dimensionality.
        ordinate (:py:attr:`list` or :py:class:`uravu.distribution.Distribution` or :py:attr:`array_like`): The ordinate data. This should have a shape :py:attr:`(N,)`.
        bounds (:py:attr:`tuple`, optional): The minimum and maximum values for each parameters. Defaults to :py:attr:`None`.
        ordinate_error (:py:attr:`array_like`, optional): The uncertainty in the ordinate, this should be the standard error in the measurement. Only used if :py:attr:`ordinate` is not a :py:attr:`list` or :py:class:`uravu.distribution.Distribution`. Defaults to :py:attr:`None`.
    """

    def __init__(self, function, abscissa, ordinate, bounds=None, ordinate_error=None):
        """
        Initialisation function for a :py:class:`~uravu.relationship.Relationship` object.
        """
        self.function = function
        self.abscissa = abscissa

        potential_y = []
        for i, y in enumerate(ordinate):
            if not isinstance(y, Distribution):
                if not isinstance(y, stats._distn_infrastructure.rv_frozen):
                    if ordinate_error is None:
                        raise ValueError("uravu ordinate should be a list of uravu.distribution.Distribution objects or an ordinate_error should be given.")
                    if ordinate_error[i] == 0:
                        raise ValueError("The ordinate_error has a 0 value, this is incompatible with uravu.")
                    potential_y.append(Distribution(stats.norm.rvs(loc=y, scale=ordinate_error[i], size=5000)))
                else:
                    potential_y.append(Distribution(y.rvs(size=5000)))
                self.ordinate = Axis(potential_y)
            else:
                self.ordinate = Axis(ordinate)

        if abscissa.shape[0] != len(ordinate):
            raise ValueError("The number of data points in the abscissa does not match that for the ordinate.")

        self.bounds = bounds
        self.variables = []
        if bounds is not None:
            if len(self.bounds) != self.len_parameters or not isinstance(bounds[0], tuple):
                raise ValueError("The number of bounds does not match the number of parameters")
            for i, b in enumerate(self.bounds):
                self.variables.append(Distribution(stats.uniform.rvs(loc=b[0], scale=b[1] - b[0], size=500)))
        else:
            for i in range(self.len_parameters):
                self.variables.append(Distribution(1))

        self.ln_evidence = None
        self.mcmc_results = None
        self.nested_sampling_results = None

    @property
    def x(self):
        """
        Abscissa values.

        Returns:
            :py:attr:`array_like`: Abscissa values.
        """
        return self.abscissa

    @property
    def y(self):
        """
        Ordinate values.

        Returns:
            :py:attr:`array_like`: Ordinate values.
        """
        return self.ordinate

    @property
    def variable_medians(self):
        """
        The median values for each of the variables.

        Returns:
            :py:attr:`array_like`: Variable medians.
        """
        medians = np.zeros((len(self.variables)))
        for i, var in enumerate(self.variables):
            medians[i] = var.n
        return medians

    @property
    def variable_modes(self):
        """
        The mode values for each of the variables.

        Returns:
            :py:attr:`array_like`: Variable modes.
        """
        modes = np.zeros((len(self.variables)))
        for i, var in enumerate(self.variables):
            modes[i] = var.dist_max
        return modes

    @property
    def mcmc_done(self):
        """
        Has MCMC been performed? Determined based on the type of the variables.

        Returns:
            :py:attr:`bool`: Has MCMC been performed?
        """
        if self.mcmc_results is not None:
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

    def get_sample(self, i):
        """
        Return the variable values for a given sample.

        Args:
            i (:py:attr:`int`): The sample index.

        Returns:
            :py:attr:`list` of :py:attr:`float`: Variable values at given index.
        """
        return [self.variables[j].samples[i] for j in range(self.len_parameters)]

    @property
    def len_parameters(self):
        """
        Determine the number of variables in the assessment function.

        Returns:
            :py:attr`int`: Number of variables.
        """
        # The minus one is to remove the abscissa data which is a
        # argument in the assessment function
        return len(getfullargspec(self.function).args) - 1

    def bayesian_information_criteria(self):
        """
        Calculate the `Bayesian information criteria`_ for the relationship.

        Returns:
            :py:attr:`float`: Bayesian information criteria.

        .. _Bayesian information criteria: https://en.wikipedia.org/wiki/Bayesian_information_criterion
        """
        self.max_likelihood('diff_evo')
        l_hat = optimize.ln_likelihood(self.variable_medians, self.function, self.abscissa, self.ordinate)
        return np.log(self.x.size) * self.len_parameters - 2 * l_hat

    def max_likelihood(self, method, x0=None, **kwargs):
        """
        Determine values for the variables which maximise the likelihood for the :py:class:`~uravu.relationship.Relationship`. For keyword arguments see the :func:`scipy.optimize.minimize()` documentation.

        Args:
            x0 (:py:attr:`array_like`): Initial guess values for the parameters.
        """
        var = optimize.max_ln_likelihood(self, method, x0, **kwargs)
        for i, v in enumerate(var):
            self.variables[i] = Distribution(v)

    def prior(self):
        """
        *Standard priors* for the relationship. These priors are broad, uninformative, for normal variables running the range :py:attr:`[x - x * 10, x + x * 10)` (where :py:attr:`x` is the variable value).

        Returns:
            :py:attr:`list` of :py:class:`scipy.stats.rv_continuous`: :py:mod:`scipy.stats` functions describing the priors.
        """
        priors = []
        if self.bounds is not None:
            for i, var in enumerate(self.variable_medians):
                loc = self.bounds[i][0]
                scale = self.bounds[i][1] - loc
                priors.append(stats.uniform(loc=loc, scale=scale))
        else:
            for var in self.variable_medians:
                loc = var - 10
                scale = (var + 10) - loc
                priors.append(stats.uniform(loc=loc, scale=scale))
        return priors

    def mcmc(self, prior_function=None, walkers=50, n_samples=500, n_burn=500, progress=True):
        """
        Perform MCMC to get the posterior probability distributions for the variables of the relationship. *Note*, running this method will populate the :py:attr:`~uravu.relationship.Relationship.variables` attribute with :py:class:`~uravu.distribution.Distribution` objects. Once run, a result dictionary containing the :py:attr:`distributions`, :py:attr:`chain`, and :py:attr:`samples` from :py:mod:`emcee` is piped into the class variable :py:attr:`mcmc_results`.

        Args:
            prior_function (:py:attr:`callable`, optional): The function to populated some prior distributions. Default is the broad uniform priors in :func:`~uravu.relationship.Relationship.prior()`.
            walkers (:py:attr:`int`, optional): Number of MCMC walkers. Default is :py:attr:`50`.
            n_samples (:py:attr:`int`, optional): Number of sample points. Default is :py:attr:500`.
            n_burn (:py:attr:`int`, optional): Number of burn in samples. Default is :py:attr:`500`.
            progress (:py:attr:`bool`, optional): Show :py:mod:`tqdm` progress for sampling. Default is :py:attr:`True`.
        """
        self.mcmc_results = sampling.mcmc(self, prior_function=prior_function, walkers=walkers, n_samples=n_samples, n_burn=n_burn, progress=progress)
        self.variables = self.mcmc_results["distributions"]

    def nested_sampling(self, prior_function=None, progress=True, dynamic=False, **kwargs):
        """
        Perform nested sampling, or dynamic nested sampling, to determine the Bayesian natural-log evidence. For keyword arguments see the :func:`dynesty.NestedSampler.run_nested()` documentation. Once run, the result dictionary produced by :func:`dynesty.NestedSampler.run_nested()` is piped into the class variable :py:attr:`nested_sampling_results`.

        Args:
            prior_function (:py:attr:`callable`, optional): The function to populate some prior distributions. Default is the broad uniform priors in :func:`~uravu.relationship.Relationship.prior()`.
            progress (:py:attr:`bool`, optional): Show :py:mod:`tqdm` progress for sampling. Default is :py:attr:`True`.
        """
        self.nested_sampling_results = sampling.nested_sampling(self, prior_function=prior_function, progress=progress, dynamic=dynamic, **kwargs)
        self.ln_evidence = ufloat(self.nested_sampling_results["logz"][-1], self.nested_sampling_results["logzerr"][-1])
        self.variables = self.nested_sampling_results["distributions"]
