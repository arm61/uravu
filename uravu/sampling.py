"""
The :mod:`sampling` module implements the use of a generalised MCMC (using :py:mod:`emcee`) and nested sampling (using :py:mod:`dynesty`) for the :py:class:`~uravu.relationship.Relationship` objects.
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import numpy as np
import emcee
import dynesty
from uravu import optimize
from uravu.distribution import Distribution
from dynesty import utils as dyfunc


def mcmc(relationship, prior_function=None, walkers=50, n_samples=500, n_burn=500, progress=True):
    """
    Perform MCMC to get the probability distributions for the variables of the relationship.

    Args:
        relationship (:py:class:`uravu.relationship.Relationship`): The relationship to determine the posteriors of.
        prior_function (:py:attr:`callable`, optional): The function to populated some prior distributions. Default is :func:`uravu.relationship.Relationship.prior()`.
        walkers (:py:attr:`int`, optional): Number of MCMC walkers. Default is :py:attr:`50`.
        n_samples (:py:attr:`int`, optional): Number of sample points. Default is :py:attr:`500`.
        n_burn (:py:attr:`int`, optional): Number of burn in samples. Default is :py:attr:`500`.
        progress (:py:attr:`bool`, optional): Show tqdm progress for sampling. Default is :py:attr:`True`.

    Returns:
        :py:attr:`dict`: Dictionary with the distributions as a list (:py:attr:`'distributions'`), the chain (:py:attr:`'chain'`) and the samples as an :py:attr:`array_like` (:py:attr:`'samples'`).
    """
    if prior_function is None:
        prior_function = relationship.prior

    initial_prior = np.zeros((walkers, len(relationship.variable_medians)))
    called_prior = prior_function()

    ndims = len(relationship.variable_medians)
    for i in range(ndims):
        if relationship.variable_medians[i] != 0:
            initial_prior[:, i] = relationship.variable_medians[i] + 1e-2 * np.random.randn(walkers) * relationship.variable_medians[i]
        else:
            initial_prior[:, i] = 1e-4 * np.random.randn(walkers)

    args = [relationship.function, relationship.abscissa, relationship.ordinate, called_prior]

    sampler = emcee.EnsembleSampler(walkers, ndims, ln_probability, args=args)
    sampler.run_mcmc(initial_prior, n_samples + n_burn, progress=progress)

    post_samples = sampler.get_chain(discard=n_burn).reshape((-1, ndims))

    distributions = []
    for i in range(ndims):
        distributions.append(Distribution(post_samples[:, i]))

    results = {"distributions": distributions, "chain": sampler.get_chain().reshape((-1, ndims)), "samples": post_samples}
    return results


def ln_probability(variables, function, abscissa, ordinate, priors):
    """
    Determine the natural log probability for a given set of variables, by
    summing the prior and likelihood.

    Args:
        variables (:py:attr:`array_like`): Variables for the function evaluation.
        function (:py:attr:`callable`): The function to be evaluated.
        abscissa (:py:attr:`array_like`): The abscissa values.
        ordinate (:py:attr:`array_like`): The ordinate values.
        unaccounted_uncertainty (:py:attr:`bool`): Should an unaccounted uncertainty parameter be considered.
        prior_function (:py:attr:`callable`, optional): The function to populated some prior distributions. Default is :func:`~uravu.relationship.Relationship.prior()`.

    Returns:
        :py:attr:`float`: Negative natural log-probability between model and data, considering priors.
    """
    log_prior = 0
    for i, var in enumerate(variables):
        log_prior += priors[i].logpdf(var)
    lnl = optimize.ln_likelihood(variables, function, abscissa, ordinate)
    return log_prior + lnl


def nested_sampling(relationship, prior_function=None, progress=True, dynamic=False, **kwargs):
    """
    Perform the nested sampling, or dynamic nested sampling, in order to determine the Bayesian natural log evidence. See the :py:func:`dynesty.NestedSampler.run_nested()` documentation.

    Args:
        relationship (:py:class:`~uravu.relationship.Relationship`): The relationship to estimate the evidence for.
        prior_function (:py:attr:`callable`, optional): The function to populated some prior distributions. Default is the broad uniform priors in :func:`~uravu.relationship.Relationship.prior()`.
        progress (:py:attr:`bool`, optional): Show :py:mod:`tqdm` progress for sampling. Default is :py:attr:`True`.
        dynamic (:py:attr:`bool`, optional): Should dynamic nested sampling be used?. Default is :py:attr:`False`.

    Returns:
        :py:attr:`dict`: The results from :py:func:`dynesty.NestedSampler.run_nested()`.
    """
    if prior_function is None:
        prior_function = relationship.prior
    priors = prior_function()
    nested_sampler = dynesty.NestedSampler
    if dynamic:
        nested_sampler = dynesty.DynamicNestedSampler
    logl_args = [relationship.function, relationship.abscissa, relationship.ordinate]
    sampler = nested_sampler(optimize.ln_likelihood, nested_prior, len(relationship.variables), logl_args=logl_args, ptform_args=[priors])

    sampler.run_nested(print_progress=progress, **kwargs)
    results = sampler.results
    samples = results['samples']
    weights = np.exp(results['logwt'] - results['logz'][-1])
    new_samples = dyfunc.resample_equal(samples, weights)
    distributions = []
    for i in range(new_samples.shape[1]):
        distributions.append(Distribution(new_samples[:, i]))
    results['distributions'] = distributions

    return results


def nested_prior(array, priors):
    """
    Convert to dynesty prior style from at used within :py:mod:`uravu`.

    Args:
        array (:py:attr:`array_like`): An array of random uniform numbers (0, 1]. The shape of which is M x N, where M is the number of parameters and N is the number of walkers.
        prior_function (:py:attr:`callable`, optional): The function to populated some prior distributions. Default is :func:`uravu.relationship.Relationship.prior()`.

    Returns:
        :py:attr:`array_like`: An array of random uniform numbers distributed in agreement with the priors.
    """
    broad = np.copy(array)
    for i, prior in enumerate(priors):
        broad[i] = prior.ppf(broad[i])
    return broad
