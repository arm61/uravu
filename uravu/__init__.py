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
