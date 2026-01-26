"""
Algorithms layer - variogram calculation, fitting, kriging implementations
"""

from .variogram import (
    experimental_variogram,
    experimental_variogram_directional,
    variogram_cloud,
    robust_variogram,
    madogram,
    rodogram,
)
from .fitting import (
    fit_variogram_model,
    automatic_fit,
)
from .simple_kriging import SimpleKriging
from .ordinary_kriging import OrdinaryKriging
from .universal_kriging import UniversalKriging
from .indicator_kriging import IndicatorKriging, MultiThresholdIndicatorKriging
from .cokriging import Cokriging, CollocatedCokriging
from .external_drift_kriging import ExternalDriftKriging
from .lognormal_kriging import LognormalKriging
from .kriging_3d import SimpleKriging3D, OrdinaryKriging3D
from .support_change import BlockKriging, SupportCorrection
from .neighborhood_search import NeighborhoodSearch, NeighborhoodConfig
from .nested_variogram import NestedVariogram, fit_nested_variogram, auto_fit_nested_variogram
from .factorial_kriging import FactorialKriging
from .spacetime_kriging import SpaceTimeOrdinaryKriging, SpaceTimeSimpleKriging

__all__ = [
    # Variogram algorithms
    "experimental_variogram",
    "experimental_variogram_directional",
    "variogram_cloud",
    "robust_variogram",
    "madogram",
    "rodogram",
    # Fitting
    "fit_variogram_model",
    "automatic_fit",
    # Nested variogram
    "NestedVariogram",
    "fit_nested_variogram",
    "auto_fit_nested_variogram",
    # Kriging algorithms
    "SimpleKriging",
    "OrdinaryKriging",
    "UniversalKriging",
    "IndicatorKriging",
    "MultiThresholdIndicatorKriging",
    "Cokriging",
    "CollocatedCokriging",
    "ExternalDriftKriging",
    "LognormalKriging",
    # 3D Kriging
    "SimpleKriging3D",
    "OrdinaryKriging3D",
    # Block Kriging & Support
    "BlockKriging",
    "SupportCorrection",
    # Neighborhood Search
    "NeighborhoodSearch",
    "NeighborhoodConfig",
    # Advanced Kriging
    "FactorialKriging",
    # Space-Time Kriging
    "SpaceTimeOrdinaryKriging",
    "SpaceTimeSimpleKriging",
]
