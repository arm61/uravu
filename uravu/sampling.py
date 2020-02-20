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
    uniform = np.random.uniform(size=(len(relationship.variables), walkers))
    if prior_function is None:
        initial_prior = relationship.prior(uniform).T
    else:
        initial_prior = prior_function(uniform).T
    ndims = initial_prior.shape[1]

    sampler = emcee.EnsembleSampler(
        walkers,
        ndims,
        optimize.ln_likelihood,
        args=[
            relationship.function,
            relationship.abscissa,
            relationship.ordinate,
        ],
    )

    sampler.run_mcmc(initial_prior, n_samples + n_burn, progress=progress)

    post_samples = sampler.get_chain(discard=n_burn).reshape((-1, ndims))

    distributions = []
    for i in range(ndims):
        distributions.append(Distribution(post_samples[:, i]))

    return distributions
