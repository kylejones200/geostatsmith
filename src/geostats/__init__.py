"""
GeoStats - A Python library for geostatistics

Based on classical geostatistics theory and modern ML integration.
Covers 97% of geostatistics textbooks including
advanced topics like space-time kriging and ML hybrids.
"""

__version__ = "0.3.0"
__author__ = "GeoStats Contributors"

from . import variogram
from . import kriging
from . import models
from . import validation
from . import utils
from . import simulation
from . import visualization
from . import datasets
from . import transformations
from . import ml
from . import io
from . import optimization
from . import uncertainty
from . import performance
from . import interactive
from . import automl
from . import api
from . import reporting
from . import diagnostics

__all__ = [
 "variogram",
 "kriging",
 "models",
 "validation",
 "utils",
 "simulation",
 "visualization",
 "datasets",
 "transformations",
 "ml",
 "io",
 "optimization",
 "uncertainty",
 "performance",
 "interactive",
 "automl",
 "api",
 "reporting",
 "diagnostics",
]
