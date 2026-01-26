"""
Geostatistical simulation module

Simulation generates multiple realizations that honor:
- The data values at sample locations
- The spatial continuity (variogram)
- The histogram of the variable

Based on Zhang, Y. (2010). Introduction to Geostatistics - Course Notes, Chapter 6.3
"""

from .gaussian_simulation import (
 sequential_gaussian_simulation,
 SequentialGaussianSimulation,
)
from .conditional_simulation import (
 conditional_simulation,
 cholesky_simulation,
)
from .unconditional import (
 unconditional_gaussian_simulation,
)
from .sequential_indicator import (
 SequentialIndicatorSimulation,
 SISConfig,
)
from .truncated_gaussian import (
 TruncatedGaussianSimulation,
 TGSConfig,
)
from .plurigaussian import (
 PlurigaussianSimulation,
 PlurigaussianConfig,
 create_rectangular_rule,
)

__all__ = [
 # Sequential Gaussian Simulation
 "sequential_gaussian_simulation",
 "SequentialGaussianSimulation",
 # Conditional simulation
 "conditional_simulation",
 "cholesky_simulation",
 # Unconditional simulation
 "unconditional_gaussian_simulation",
 # Sequential Indicator Simulation
 "SequentialIndicatorSimulation",
 "SISConfig",
 # Truncated Gaussian Simulation
 "TruncatedGaussianSimulation",
 "TGSConfig",
 # Plurigaussian Simulation
 "PlurigaussianSimulation",
 "PlurigaussianConfig",
 "create_rectangular_rule",
]
