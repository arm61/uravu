FAQ
===

- How does one assign units to a :py:class:`~uravu.distribution.Distribution` object?

    :py:mod:`~uravu` allows the use of the :py:mod:`~pint` module for the definition of units to be associated with :py:class:`~uravu.distribution.Distribution` objects. 
    The :func:`uravu.distribution.Distribution.__init__()` function includes a keyword argument :py:attr:`unit` where the :py:class:`~pint.unit.Unit` object should be assigned. 
    However, to `avoid creating multiple unit registries`_, :py:mod:`~uravu` is packaged with a :py:class:`~pint.UnitRegistry`. 
    Therefore, if you wish to create a :py:class:`~uravu.distribution.Distribution` where the :py:attr:`unit` is metres, the following syntax should be used. 

.. code-block:: python

   import numpy as np
   from uravu import UREG
   from uravu.distribution import Distribution

   sample = np.random.randn(100)
   my_distribution = Distribution(sample, name='Random Distribution', units=UREG.meter)

- How do I use the :func:`uravu.utils.bayes_factor()` function to compare different models?

    The :func:`uravu.utils.bayes_factor()` function uses the the second Table on page 777 of `Kass and Raftery's`_ paper on Bayesian model comparison.
    The function will return a value for :math:`2\ln(B_{10})`, where the model 1 is the first argument in the function and model 0 is the second. 
    The table mentioned above is reproduced below.

    +-----------------------+-----------------+--------------------------+
    | :math:`2\ln(B_{10})`  |  :math:`B_{10}` |  Interpretation          |
    +-----------------------+-----------------+--------------------------+
    | 0 to 2                | 1 to 3          | Not worth a bare mention |
    +-----------------------+-----------------+--------------------------+
    | 2 to 6                | 3 to 20         | Positive                 |
    +-----------------------+-----------------+--------------------------+
    | 6 to 10               | 20 to 150       | Strong                   |
    +-----------------------+-----------------+--------------------------+
    | > 10                  | > 150           | Very Strong              |
    +-----------------------+-----------------+--------------------------+

    So if :py:class:`uravu.utils.bayes_factor(model1, model2)` returns :py:attr:`4.3`, there is "Positive" evidence for :py:attr:`model1` over :py:attr:`model2`.


.. _avoid creating multiple unit registries: https://pint.readthedocs.io/en/0.11/tutorial.html#using-pint-in-your-projects
.. _Kass and Raftery's: https://www.colorado.edu/amath/sites/default/files/attached-files/kassraftery95.pdf