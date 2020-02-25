"""
This is the Relationship class, which allows for the storage and manipulation
of analytical relationships between and abscissa and ordinate.

This enables the determination of maximum likelihood, the evalulation of
posterior probability distributions by Markov chain Monte-Carlo (MCMC) and
the determination of Bayesian evidence using nested sampling.
"""

# Copyright (c) Andrew R. McCluskey
# Distributed under the terms of the MIT License
# author: Andrew R. McCluskey

import numpy as np
import emcee
import dynesty
from uravu import optimize
from uravu.distribution import Distribution


def mcmc(
    relationship,
    prior_function=None,
    walkers=100,
    n_samples=500,
    n_burn=500,
    progress=True,
):
    """
    Perform MCMC to get the probability distributions for the variables
    of the relationship.

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

    Returns:
        (list of uravu.distribution.Distribution): a
            uravu.distribution.Distribution to describe each of the variables.
    """
    if prior_function is None:
        prior_function = relationship.prior

    initial_prior = relationship.variable_medians + 1e-4 * np.random.randn(
        walkers, len(relationship.variable_medians)
    )
    ndims = initial_prior.shape[1]
    k = prior_function()

    sampler = emcee.EnsembleSampler(
        walkers,
        ndims,
        ln_probability,
        args=[
            relationship.function,
            relationship.abscissa,
            relationship.ordinate,
            relationship.unaccounted_uncertainty,
            k,
        ],
    )

    sampler.run_mcmc(initial_prior, n_samples + n_burn, progress=progress)

    post_samples = sampler.get_chain(discard=n_burn).reshape((-1, ndims))

    distributions = []
    for i in range(ndims):
        distributions.append(Distribution(post_samples[:, i]))

    results = {'distributions': distributions, 'chain': sampler.get_chain().reshape((-1, ndims)), 'samples': post_samples}
    return results


def ln_probability(
    variables, function, abscissa, ordinate, unaccounted_uncertainty, priors
):
    """
    Determine the natural log probability for a given set of variables.
    """
    log_prior = 0
    for i, var in enumerate(variables):
        log_prior += priors[i].logpdf(var)
    return log_prior + optimize.ln_likelihood(
        variables, function, abscissa, ordinate, unaccounted_uncertainty
    )


def nested_sampling(
    relationship, prior_function=None, progress=True, **kwargs
):
    """
    Perform the nested sampling in order to determine the Bayesian natural log
    evidence.

    Args:
        prior_function (callable, optional): the function to populated some
            prior distributions. Default is the broad uniform priors in
            uravu.relationship.Relationship.
        progress (bool, optional): Show tqdm progress for sampling.
            Default is `True`.

    Keyword Args:
        See the `dynesty.run_nested()` documentation.

    Returns:
        (uncertainties.core.Variable): Log-evidence (and uncertainty) as
            estimated by the nested sampling.
    """
    if prior_function is None:
        prior_function = relationship.prior
    priors = prior_function()
    sampler = dynesty.NestedSampler(
        optimize.ln_likelihood,
        nested_prior,
        len(relationship.variables),
        logl_args=[
            relationship.function,
            relationship.abscissa,
            relationship.ordinate,
            relationship.unaccounted_uncertainty,
        ],
        ptform_args=[priors],
    )
    sampler.run_nested(print_progress=progress, **kwargs)
    results = sampler.results
    return results


def nested_prior(array, priors):
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
        distributed in the range [x - x * 5, x + x * 5), where x is the
        current variable value.
    """
    broad = np.copy(array)
    for i, prior in enumerate(priors):
        broad[i] = prior.ppf(broad[i])
    return broad
