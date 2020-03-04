![uravu logo](https://github.com/arm61/uravu/raw/master/docs/source/logo/uravu_logo.png)

**making Bayesian modelling easy(er)**


[![Documentation Status](https://readthedocs.org/projects/uravu/badge/?version=latest)](https://uravu.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/arm61/uravu/badge.svg?branch=master)](https://coveralls.io/github/arm61/uravu?branch=master)
[![Build Status](https://travis-ci.org/arm61/uravu.svg?branch=master)](https://travis-ci.org/arm61/uravu)
[![Build status](https://ci.appveyor.com/api/projects/status/eo426m99lmkbh5rx?svg=true)](https://ci.appveyor.com/project/arm61/uravu)
[![Gitter](https://badges.gitter.im/uravu/community.svg)](https://gitter.im/uravu/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

``uravu`` (from the Tamil for relationship) is about the relationship between some data and a function that may be used to describe the data. 

The aim of ``uravu`` is to make using the **amazing** Bayesian inference libraries that are available in Python as easy as [scipy.optimize.curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html).
Therefore enabling many more to make use of these exciting tools and powerful libraries.
Plus, we have some nice plotting functionalities available in the `plotting` module, capable of generating publication quality figures.

![An example of the type of figures that uravu can produce. Showing straight line distribution with increasing uncertainty.](https://github.com/arm61/uravu/raw/master/docs/source/sample_fig.png)

In an effort to make the ``uravu`` API friendly to those new to Bayesian inference, ``uravu`` is *opinionated*, making assumptions about priors amoung other things. 
However, we have endevoured to make it straightforward to ignore these opinions.

In addition to the library and API, we also have some [basic tutorials](https://uravu.readthedocs.io/en/latest/tutorials.html) discussing how Bayesian inference methods can be used in the analysis of data. 

## Bayesian inference in Python

There are a couple of fantastic Bayesian inference libraries available in Python that `uravu` makes use of:

- [emcee](https://emcee.readthedocs.io/): enables the use of the [Goodman & Weare’s Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler](https://doi.org/10.2140/camcos.2010.5.65) to evaluate the structure of the model parameter posterior distributions,
- [dynesty](https://dynesty.readthedocs.io/): implements the [nested sampling](https://doi.org/10.1063/1.1835238) algorithm for evidence estimation.

## Problems

If you discover any issues with `uravu` please feel free to submit an issue to our issue tracker on [Github](https://github.com/arm61/uravu). 
Alternatively, if you are feeling confident, fix the bug yourself and make a pull request to the main codebase (be sure to check out our [contributing guidelines](https://github.com/arm61/uravu/CONTRIBUTING.md) first). 
Finally, if you are just wanting to ask a question and cannot find the information elsewhere, we have a [gitter chat room](https://gitter.im/uravu/community?utm_source=share-link&utm_medium=link&utm_campaign=share-link) as another way to seek support. 

## Installation

```
pip install -r requirements.txt
python setup.py build
python setup.py install 
pytest
```

## Contributors 

- [Andrew R. McCluskey](armccluskey.com)