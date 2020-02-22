.. uravu documentation master file, created by
   sphinx-quickstart on Fri Feb 21 09:20:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``uravu`` 
=========

**making Bayesian modelling easy(er)**

``uravu`` is about the relationship between some data and a function that may be used to describe the data. 

The aim of ``uravu`` is to make using the **amazing** Bayesian inference libraries that are available in Python as easy as `scipy.optimize.curve_fit`_.
Therefore enabling many more to make use of these exciting tools and powerful libraries.
Plus, we have some nice plotting functionalities available in the :mod:`plotting` module, capable of generating *publication quality* figures. 

In an effect to make the ``uravu`` API friendly to those new to Bayesian inference, ``uravu`` is *opinionated*, making assumptions about priors amoung other things. 
However, we have endevoured to make it straightforward to ignore these opinions.

In addition to the library and API, we also have some basic tutorials discussing how Bayesian inference methods can be used in the analysis of *experimental* data. 

.. _scipy.optimize.curve_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html


.. toctree::
   :hidden:
   :maxdepth: 2

   api

Searching
=========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
