"""
Walker Lake Dataset

Classic geostatistics dataset from Nevada.
Used in Isaaks & Srivastava (1989) and Zhang (2010) course notes.

Two variables measured on a 10x10 grid:
- V: Arsenious contaminant concentration (ppm)
- U: PCE concentration (ppm)

Reference:
- Zhang, Y. (2010). Course Notes, Chapter 2.7, Figures 2.5 and 2.6
- Isaaks, E.H., & Srivastava, R.M. (1989). An Introduction to Applied Geostatistics
"""

from typing import Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_walker_lake() -> Dict:
    Load the Walker Lake dataset

    Returns
    -------
    dict
    Dictionary with keys:
    - 'x': X coordinates (m)
    - 'y': Y coordinates (m)
    - 'V': Arsenious contaminant concentration (ppm)
    - 'U': PCE concentration (ppm)
    - 'description': Dataset description

    Examples
    --------
    >>> from geostats.datasets import load_walker_lake
    >>> data = load_walker_lake()
    >>> logger.info(f"Number of samples: {len(data['x'])}")
    >>> logger.info(f"V range: [{np.min(data['V'])}, {np.max(data['V'])}] ppm")
    """

    # V values from Figure 2.5 (Zhang 2010, p. 28)
    # 10x10 grid, values in ppm (rounded to integers)
    V = np.array(
        [
            [81, 77, 103, 112, 123, 19, 40, 111, 114, 120],
            [82, 61, 110, 121, 119, 77, 52, 111, 117, 124],
            [82, 74, 97, 105, 112, 91, 73, 115, 118, 129],
            [88, 70, 103, 111, 122, 64, 84, 105, 113, 123],
            [89, 88, 94, 110, 116, 108, 73, 107, 118, 127],
            [77, 82, 86, 101, 109, 113, 79, 102, 120, 121],
            [74, 80, 85, 90, 97, 101, 96, 72, 128, 130],
            [75, 80, 83, 87, 94, 99, 95, 48, 139, 145],
            [77, 84, 74, 108, 121, 143, 91, 52, 136, 144],
            [87, 100, 47, 111, 124, 109, 0, 98, 134, 144],
        ]
    ).flatten()

    # U values from Figure 2.6 (Zhang 2010, p. 29)
    # 10x10 grid, values in ppm (rounded to integers)
    U = np.array(
        [
            [15, 12, 24, 27, 30, 0, 2, 18, 18, 18],
            [16, 7, 34, 36, 29, 7, 4, 18, 18, 20],
            [16, 9, 22, 24, 25, 10, 7, 19, 19, 22],
            [21, 8, 27, 27, 32, 4, 10, 15, 17, 19],
            [21, 18, 20, 27, 29, 19, 7, 16, 19, 22],
            [15, 16, 16, 23, 24, 25, 7, 15, 21, 20],
            [14, 15, 15, 16, 17, 18, 14, 6, 28, 25],
            [14, 15, 15, 15, 16, 17, 13, 2, 40, 38],
            [16, 17, 11, 29, 37, 55, 11, 3, 34, 35],
            [22, 28, 4, 32, 38, 20, 0, 14, 31, 34],
        ]
    ).flatten()

    # Create 10x10 grid coordinates (10m spacing)
    x_coords = np.arange(10) * 10.0
    y_coords = np.arange(10) * 10.0
    X, Y = np.meshgrid(x_coords, y_coords)
    x = X.flatten()
    y = Y.flatten()

    return {
        "x": x,
        "y": y,
        "V": V,
        "U": U,
        "description": (
            "Walker Lake dataset from Nevada. "
            "Measurements on a 10x10 grid (10m spacing). "
            "V: Arsenious contaminant (ppm), U: PCE concentration (ppm). "
            "Source: Isaaks & Srivastava (1989), Zhang (2010) Course Notes."
        ),
        "grid_size": (10, 10),
        "spacing": 10.0,
        "unit": "meters",
        "variables": ["V (ppm)", "U (ppm)"],
    }


def get_walker_lake_subset(n_samples: int = 50, seed: int = 42) -> Dict:
    Get a random subset of Walker Lake data

    Useful for testing algorithms on smaller datasets.

    Parameters
    ----------
    n_samples : int
    Number of samples to extract
    seed : int
    Random seed for reproducibility

    Returns
    -------
    dict
    Subset of Walker Lake data
    """
    np.random.seed(seed)

    data = load_walker_lake()
    n_total = len(data["x"])

    # Random sample
    indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

    return {
        "x": data["x"][indices],
        "y": data["y"][indices],
        "V": data["V"][indices],
        "U": data["U"][indices],
        "description": f"Random subset ({n_samples} samples) of Walker Lake dataset",
    }
