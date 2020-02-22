# uravu

உறவு


[![Documentation Status](https://readthedocs.org/projects/uravu/badge/?version=latest)](https://uravu.readthedocs.io/en/latest/?badge=latest)
[![Gitter](https://badges.gitter.im/uravu/community.svg)](https://gitter.im/uravu/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

u-rav-u (from the Tamil for relationship) is a library for fitting, assessing, and optimising the relationship between a model and some data, using Bayesian methods. 

The world of Bayesian modelling can be a confusing place, but uravu makes it easier. 
All you need is a function to describe the analytical relationship and some data that you want to model.
Other fun things like prior probabilities for your model parameters can be included. 

From just a single function and your data, uravu can help you perform:

- Maximum likelihood fitting
- MCMC modelling for posterior estimation (using [`emcee`](https://emcee.readthedocs.io/))
- Evidence evaluation from nested sampling (using [`dynesty`](https://dynesty.readthedocs.io/))
- A plotting library to present the results (using [`matplotlib`](https://matplotlib.org/) and [`corner`](https://corner.readthedocs.io/))

With this library, we hope to lower the barrier to entry to Bayesian modelling methods, while improving understanding and appreciation for the power of these methods. 

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