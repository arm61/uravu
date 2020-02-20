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

from inspect import getfullargspec
import numpy as np
from uncertainties import unumpy as unp
from uravu import UREG


class Relationship:
    """
    The `Relationship` class enables the investigation of variable
    distributions in mathematical models.
    """

    def __init__(
        self,
        function,
        abscissa,
        ordinate,
        ordinate_uncertainty,
        abscissa_uncertainty=None,
        abscissa_unit=UREG.dimensionless,
        ordinate_unit=UREG.dimensionless,
    ):
        """
        Args:
            f (callable): the functional relationship to be considered.
            abscissa (array_like): the abscissa data. If multi-dimensional,
                the array is expected to have the shape `(N, d)`, where `N`
                is the number of data points and `d` is the dimensionality.
            ordinate (array_like): the ordinate data. This should have a
                shape `(N,)`.
            ordinate_uncertainty (array_like): the uncertainty in each of the
                ordinate data points. This should have a shape `(N,)`.
            abscissa_uncertainty (array_like, optional): the uncertainty in
                each of the absiccsa data points. This should have a shape
                `(N, d)`. Default is no uncertainties on absicca.
            abscissa_unit (pint.UnitRegistry()): The unit for the
                abscissa. If `abscissa` is multi-dimensional, this should be
                a list with the units for each dimension.
            ordinate_unit (pint.UnitRegistry()): The unit for the
                ordinate.
        """
        self.function = function
        abscissa = np.array(abscissa)
        ordinate = np.array(ordinate)
        ordinate_uncertainty = np.array(ordinate_uncertainty)
        if abscissa_uncertainty is not None:
            abscissa_uncertainty = np.array(abscissa_uncertainty)
        if abscissa.shape[0] == ordinate.shape[0]:
            if ordinate.shape[0] == ordinate_uncertainty.shape[0]:
                if (abscissa_uncertainty is not None) and (
                    abscissa.shape[0] == abscissa_uncertainty.shape[0]
                ):
                    self.abscissa = unp.uarray(abscissa, abscissa_uncertainty)
                else:
                    self.abscissa = abscissa
                self.abscissa *= abscissa_unit
                self.ordinate = (
                    unp.uarray(ordinate, ordinate_uncertainty) * ordinate_unit
                )
        else:
            raise ValueError(
                "The number of data points in the abscissa does "
                "not match that for the ordinate."
            )
        self.variables = np.ones((self.len_parameters()))

    @property
    def x(self):
        """
        Abscissa values.

        Returns:
            (array_like): Abscissa values.
        """
        return self.abscissa

    @property
    def y(self):
        """
        Ordinate values.

        Returns:
            (array_like): Ordinate values.
        """
        return self.ordinate

    @property
    def x_m(self):
        """
        Abscissa values without units.

        Returns:
            (array_like): Abscissa values without units.
        """
        return self.abscissa.m

    @property
    def y_m(self):
        """
        Ordinate values without units.

        Returns:
            (array_like): Ordinate values without units.
        """
        return self.ordinate.m

    @property
    def x_u(self):
        """
        Abscissa unit.

        Returns:
            (pint.UnitRegistry()): Abscissa values.
        """
        return self.abscissa.u

    @property
    def y_u(self):
        """
        Ordinate unit.

        Returns:
            (pint.UnitRegistry()): Ordinate unit.
        """
        return self.ordinate.u

    @property
    def y_n(self):
        """
        Ordinate nominal values.

        Returns:
            (array_like): Ordinate nominal values.
        """
        return unp.nominal_values(self.ordinate.m)

    @property
    def y_s(self):
        """
        Ordinate uncertainties.

        Returns:
            (array_like): Ordinate uncertainties.
        """
        return unp.std_devs(self.ordinate.m)

    def len_parameters(self):
        """
        Determine the number of variables in the assessment function.

        Returns:
            (int): number of variables.
        """
        # The minus one is to remove the abscissa data which is a
        # argument in the assessment function
        return len(getfullargspec(self.function).args) - 1
