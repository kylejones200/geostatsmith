"""
High-level kriging API

This module provides user-friendly functions for kriging interpolation.
"""

from .algorithms.cokriging import (
    Cokriging as _Cokriging,
)
from .algorithms.cokriging import (
    CollocatedCokriging as _CollocatedCokriging,
)
from .algorithms.indicator_kriging import (
    IndicatorKriging as _IndicatorKriging,
)
from .algorithms.indicator_kriging import (
    MultiThresholdIndicatorKriging as _MultiThresholdIndicatorKriging,
)
from .algorithms.ordinary_kriging import OrdinaryKriging as _OrdinaryKriging
from .algorithms.simple_kriging import SimpleKriging as _SimpleKriging
from .algorithms.universal_kriging import UniversalKriging as _UniversalKriging

__all__ = [
    "SimpleKriging",
    "OrdinaryKriging",
    "UniversalKriging",
    "IndicatorKriging",
    "MultiThresholdIndicatorKriging",
    "Cokriging",
    "CollocatedCokriging",
]

# Re-export kriging classes
SimpleKriging = _SimpleKriging
OrdinaryKriging = _OrdinaryKriging
UniversalKriging = _UniversalKriging
IndicatorKriging = _IndicatorKriging
MultiThresholdIndicatorKriging = _MultiThresholdIndicatorKriging
Cokriging = _Cokriging
CollocatedCokriging = _CollocatedCokriging
