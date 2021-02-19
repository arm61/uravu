![uravu logo](https://github.com/arm61/uravu/raw/master/docs/source/logo/uravu_logo.png)

**making Bayesian modelling easy(er)**

[![status](https://joss.theoj.org/papers/e9047e48bf024589e0765f955b3e4c76/status.svg)](https://joss.theoj.org/papers/e9047e48bf024589e0765f955b3e4c76)
[![DOI](https://zenodo.org/badge/241184437.svg)](https://zenodo.org/badge/latestdoi/241184437)

[![PyPI version](https://badge.fury.io/py/uravu.svg)](https://badge.fury.io/py/uravu)
[![Documentation Status](https://readthedocs.org/projects/uravu/badge/?version=latest)](https://uravu.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/arm61/uravu/badge.svg?branch=master)](https://coveralls.io/github/arm61/uravu?branch=master)
[![Build Status](https://github.com/arm61/uravu/workflows/python-ci/badge.svg)](https://github.com/arm61/uravu/actions?query=workflow%3Apython-ci)
[![Build status](https://ci.appveyor.com/api/projects/status/eo426m99lmkbh5rx?svg=true)](https://ci.appveyor.com/project/arm61/uravu)

``uravu`` (from the Tamil for relationship) is about the relationship between some data and a function that may be used to describe the data.

The aim of ``uravu`` is to make using the **amazing** Bayesian inference libraries that are available in Python as easy as [`scipy.optimize.curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html).
Therefore enabling many more to make use of these exciting tools and powerful libraries.
Plus, we have some nice plotting functionalities available in the `plotting` module, capable of generating publication quality figures.

![An example of the type of figures that uravu can produce. Showing straight line distribution with increasing uncertainty.](https://github.com/arm61/uravu/raw/master/docs/source/sample_fig.png)

In an effort to make the ``uravu`` API friendly to those new to Bayesian inference, ``uravu`` is *opinionated*, making assumptions about priors among other things.
However, we have endevoured to make it straightforward to ignore these opinions.

In addition to the library and API, we also have some [basic tutorials](https://uravu.readthedocs.io/en/latest/tutorials.html) discussing how Bayesian inference methods can be used in the analysis of data.

## Bayesian inference in Python

There are a couple of fantastic Bayesian inference libraries available in Python that `uravu` makes use of:

- [emcee](https://emcee.readthedocs.io/): enables the use of the [Goodman & Weare’s Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler](https://doi.org/10.2140/camcos.2010.5.65) to evaluate the structure of the model parameter posterior distributions,
- [dynesty](https://dynesty.readthedocs.io/): implements the [nested sampling](https://doi.org/10.1063/1.1835238) algorithm for evidence estimation.

## Problems

If you discover any issues with `uravu` please feel free to submit an issue to our issue tracker on [Github](https://github.com/arm61/uravu).
Alternatively, if you are feeling confident, fix the bug yourself and make a pull request to the main codebase (be sure to check out our [contributing guidelines](https://github.com/arm61/uravu/blob/master/CONTRIBUTING.md) first).

## Installation

`uravu` is available from the [PyPI](https://pypi.org/project/uravu/) repository so can be [installed using `pip`](https://uravu.readthedocs.io/en/latest/installation.html) or alternatively `clone` this repository and install the latest development build with the commands below.

```
pip install -r requirements.txt
python setup.py build
python setup.py install
pytest
```

## [Contributors](https://github.com/arm61/uravu/graphs/contributors)
