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


.. _avoid creating multiple unit registries: https://pint.readthedocs.io/en/0.11/tutorial.html#using-pint-in-your-projects