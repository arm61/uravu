---
title: 'uravu: Making Bayesian modelling easy(er)'
tags:
  - Python
  - Bayesian
  - analysis
  - evidence estimation
  - nested sampling
  - mcmc
authors:
  - name: Andrew R. McCluskey
    orcid: 0000-0003-3381-5911
    affiliation: "1, 2"
  - name: Tim Snow
    orcid: 0000-0001-7146-6885
    affiliation: "1, 3"
affiliations:
 - name: Diamond Light Source, Rutherford Appleton Laboratory, Harwell Science and Innovation Campus, Didcot, OX11 0DE, UK
   index: 1
 - name: Department of Chemistry, University of Bath, Claverton Down, Bath, BA2 7AY, UK
   index: 2
 - name: School of Chemistry, University of Bristol, Bristol, BS8 1TS, UK
   index: 3
date: 04 May 2020
bibliography: paper.bib
---

# Summary

`uravu` offers an easy to use interface for data analysis using Bayesian modelling in the Python programming language; aiming to make Bayesian modelling as easy to use as the `scipy.optimize.curve_fit()` method. 
This software acts to lower the barrier of entry to the use of packages such as:

- `scipy`: for maximum likelihood estimation [@virtanen_scipy_2020]
- `emcee`: for Markov chain Monte Carlo investigation of posterior probabilities [@foremanmackey_emcee_2019]
- `dynesty`: for nested sampling of posterior probabilities and estimation of the Bayesian evidence [@speagle_dynesty_2020].

In addition to standard normal ordinate uncertainties, `uravu` also add the ability to investigate non-normal ordinate value distributions, with the `Distribution` class and its integration into the `Relationship` class. 
Furthermore, `uravu` provides the functionality to create simple, publication-quality plots of relationships, data, and distributions with `matplotlib`[@hunter_matplotlib_2007] and `corner` [@foremanmackey_corner_2019]. 

Alongside API information, the `uravu` documentation offers brief tutorials covering many aspects of the package.
This allows those unfamiliar with Bayesian modelling to get to grips with these important tools for data analysis.
While Bayesian data modelling is possible with packages such as `emcee` [@foremanmackey_emcee_2019], `dynesty` [@speagle_dynesty_2020], and `PyMC3` [salvatier_pymc3_2016], to the authors' knowledge `uravu` is unique in offering a simple to use interface, where it is not necessary to explicitly define functions for as the likelihood or prior probabilities (though custom priors are possible). 
`uravu` is being actively applied to scientific problems, such as data reduction at large scale scientific facilities and the modelling of diffusion in battery materials.

# Statement of Need

The Python language has a large number of powerful packages related to the application of Bayesian modelling. 
However, to apply these methods to their problems, scientific users need a straightforward environment. 
For maximum-likelihood modelling, this is achieved using the `scipy.optimize.curve_fit()` method for many users, but, to the best of the author's knowledge, there is no equivalent method for Bayesian modelling.
`uravu` fills this gap by offering easy access to powerful Python packages to perform Markov chain Monte Carlo and nested sampling, that is capable of handling uncertainties with any distribution. 
Furthermore, the tutorials, available as documentation online, allow users to become more comfortable with the use of Bayesian methods for data modelling.

# Acknowledgements

This work is supported by the Ada Lovelace Centre – a joint initiative between the Science and Technology Facilities Council (as part of UK Research and Innovation), Diamond Light Source, and the UK Atomic Energy Authority.

# References