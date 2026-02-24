"""
Constants Configuration Schema

Allows YAML configs to override default constants for config-driven workloads.
Python constants serve as defaults, YAML can override them.
"""

from pydantic import BaseModel, Field


class ConstantsConfig(BaseModel):
    """
    Configuration for overriding default constants.

    All fields are optional - missing values use Python defaults from core.constants.
    This allows partial overrides while maintaining backward compatibility.
    """

    # Numerical stability
    epsilon: float | None = Field(
        None, description="Numerical epsilon (default: 1e-10)"
    )
    small_number: float | None = Field(
        None, description="Small number threshold (default: 1e-6)"
    )
    regularization_factor: float | None = Field(
        None, description="Matrix regularization (default: 1e-8)"
    )

    # Kriging defaults
    default_max_neighbors: int | None = Field(
        None, description="Max neighbors for kriging (default: 25)"
    )
    default_min_neighbors: int | None = Field(
        None, description="Min neighbors for kriging (default: 3)"
    )
    default_search_radius_multiplier: float | None = Field(
        None, description="Search radius multiplier (default: 3.0)"
    )

    # Variogram calculation
    default_n_lags: int | None = Field(
        None, description="Default number of lag bins (default: 15)"
    )
    default_max_distance: float | None = Field(
        None, description="Default max distance (default: 100.0)"
    )

    # Sequential simulation
    default_n_realizations: int | None = Field(
        None, description="Default realizations (default: 100)"
    )
    default_n_thresholds: int | None = Field(
        None, description="Default thresholds (default: 5)"
    )

    # Outlier detection
    z_score_threshold: float | None = Field(
        None, description="Z-score threshold (default: 3.0)"
    )
    iqr_multiplier: float | None = Field(
        None, description="IQR multiplier (default: 1.5)"
    )
    spatial_threshold_factor: float | None = Field(
        None, description="Spatial threshold factor (default: 3.0)"
    )

    # Visualization
    scatter_size_multiplier: float | None = Field(
        None, description="Scatter size multiplier (default: 100)"
    )
    scatter_alpha_default: float | None = Field(
        None, description="Default scatter alpha (default: 0.6)"
    )

    # Optimization
    max_iterations_global: int | None = Field(
        None, description="Max global iterations (default: 500)"
    )
    max_iterations_local: int | None = Field(
        None, description="Max local iterations (default: 1000)"
    )
    convergence_tolerance: float | None = Field(
        None, description="Convergence tolerance (default: 1e-6)"
    )

    class Config:
        extra = "forbid"  # Don't allow extra fields
