# uravu

உறவு

u-rav-u (from the Tamil for relationship) is a library for fitting, assessing, and optimising the relationship between a model and some data, using Bayesian methods. 

The world of Bayesian modelling can be a confusing place, but uravu makes it easier. 
All you need is a function to describe the analytical relationship and some data that you want to model.
Other fun things like prior probabilities for your model parameters can be included. 

From just a single function and your data, uravu can help you perform:

- Maximum likelihood fitting
- MCMC modelling for posterior estimation (using `emcee`)
- Evidence evaluation from nested sampling (using `dynesty`)
- A plotting library to present the results (using `matplotlib`)

With this library, we hope to lower the barrier to entry to Bayesian modelling methods, while improving understanding and appreciation for the power of these methods. 

## Installation

```
pip install -r requirements.txt
python setup.py build
python setup.py install 
pytest
```

## Contributors 

- [Andrew R. McCluskey](armccluskey.com)