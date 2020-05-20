FAQ
===

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


.. _Kass and Raftery's: https://www.colorado.edu/amath/sites/default/files/attached-files/kassraftery95.pdf