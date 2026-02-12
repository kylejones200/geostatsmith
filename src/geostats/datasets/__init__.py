"""
Sample datasets for testing and demonstrations.

Provides access to:
- Walker Lake dataset: Classic geostatistics dataset
- Synthetic data generators: Create custom test datasets
- Elevation samples: DEM-like data for interpolation examples

Reference: Python Recipes for Earth Sciences (Trauth 2024)
"""

from .walker_lake import load_walker_lake
from .synthetic import (
    generate_random_field,
    generate_clustered_samples,
    generate_elevation_like_data,
    generate_anisotropic_field,
    generate_sparse_dense_mix,
)
from .elevation_samples import (
    load_synthetic_dem_sample,
    load_volcano_sample,
    load_valley_sample,
)

__all__ = [
    # Classic dataset
    "load_walker_lake",
    # Synthetic generators
    "generate_random_field",
    "generate_clustered_samples",
    "generate_elevation_like_data",
    "generate_anisotropic_field",
    "generate_sparse_dense_mix",
    # Elevation samples
    "load_synthetic_dem_sample",
    "load_volcano_sample",
    "load_valley_sample",
]
