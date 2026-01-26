"""
High-level kriging API

This module provides user-friendly functions for kriging interpolation.
"""

from typing import Optional, Tuple, Dict
import numpy as np
import numpy.typing as npt

from .algorithms.simple_kriging import SimpleKriging as _SimpleKriging
from .algorithms.ordinary_kriging import OrdinaryKriging as _OrdinaryKriging
from .algorithms.universal_kriging import UniversalKriging as _UniversalKriging
from .algorithms.indicator_kriging import (
    IndicatorKriging as _IndicatorKriging,
    MultiThresholdIndicatorKriging as _MultiThresholdIndicatorKriging,
)
from .algorithms.cokriging import (
    Cokriging as _Cokriging,
    CollocatedCokriging as _CollocatedCokriging,
)

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
