"""
Initialisation of uravu.
"""

import warnings
from pint import UnitRegistry, set_application_registry, Quantity

UREG = UnitRegistry()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Quantity([])
Q_ = UREG.Quantity
set_application_registry(UREG)


def __version__():
    """
    Return the current version of uravu.

    Returns:
        (str): Version number.
    """
    major = 0
    minor = 0
    micro = 1
    return "{}.{}.{}".format(major, minor, micro)
