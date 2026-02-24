"""
Algorithms layer - variogram calculation, fitting, kriging implementations
"""

from .cokriging import Cokriging, CollocatedCokriging
from .disjunctive_kriging import DisjunctiveKriging
from .external_drift_kriging import ExternalDriftKriging
from .factorial_kriging import FactorialKriging
from .fitting import (
    automatic_fit,
    fit_variogram_model,
)
from .indicator_kriging import IndicatorKriging, MultiThresholdIndicatorKriging
from .kriging_3d import OrdinaryKriging3D, SimpleKriging3D
from .lognormal_kriging import LognormalKriging
from .neighborhood_search import NeighborhoodConfig, NeighborhoodSearch
from .nested_variogram import (
    NestedVariogram,
    auto_fit_nested_variogram,
    fit_nested_variogram,
)
from .ordinary_kriging import OrdinaryKriging
from .simple_kriging import SimpleKriging
from .spacetime_kriging import SpaceTimeOrdinaryKriging, SpaceTimeSimpleKriging
from .support_change import BlockKriging, SupportCorrection
from .universal_kriging import UniversalKriging
from .variogram import (
    experimental_variogram,
    experimental_variogram_directional,
    madogram,
    robust_variogram,
    rodogram,
    variogram_cloud,
)

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
    "DisjunctiveKriging",
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
