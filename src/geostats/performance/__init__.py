"""
Performance Optimization Module
================================

Tools for accelerating geostatistical computations.

Key Features:
- Parallel kriging (multi-core)
- Chunked processing for large datasets
- Result caching
- Approximate methods for speed

Examples
--------
>>> from geostats.performance import parallel_kriging, ChunkedKriging
>>>
>>> # Parallel kriging (uses all CPU cores)
>>> z_pred, var = parallel_kriging(
... x, y, z,
... x_pred, y_pred,
... variogram_model=model,
... n_jobs=-1 # Use all cores
... )
>>>
>>> # Chunked processing for large grids
>>> chunked = ChunkedKriging(x, y, z, variogram_model)
>>> z_pred = chunked.predict_large_grid(x_grid, y_grid, chunk_size=10000)
"""

from .parallel import (
 parallel_kriging,
 parallel_cross_validation,
 parallel_variogram_fit,
)

from .chunked import (
 ChunkedKriging,
 chunked_predict,
)

from .caching import (
 CachedKriging,
 clear_cache,
)

from .approximate import (
 approximate_kriging,
 coarse_to_fine,
)

__all__ = [
 # Parallel processing
 'parallel_kriging',
 'parallel_cross_validation',
 'parallel_variogram_fit',
 # Chunked processing
 'ChunkedKriging',
 'chunked_predict',
 # Caching
 'CachedKriging',
 'clear_cache',
 # Approximate methods
 'approximate_kriging',
 'coarse_to_fine',
]
