"""
Constants Configuration Schema

Allows YAML configs to override default constants for config-driven workloads.
Python constants serve as defaults, YAML can override them.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ConstantsConfig(BaseModel):
    """
    Configuration for overriding default constants.
    
    All fields are optional - missing values use Python defaults from core.constants.
    This allows partial overrides while maintaining backward compatibility.
    """
    
    # Numerical stability
    epsilon: Optional[float] = Field(None, description="Numerical epsilon (default: 1e-10)")
    small_number: Optional[float] = Field(None, description="Small number threshold (default: 1e-6)")
    regularization_factor: Optional[float] = Field(None, description="Matrix regularization (default: 1e-8)")
    
    # Kriging defaults
    default_max_neighbors: Optional[int] = Field(None, description="Max neighbors for kriging (default: 25)")
    default_min_neighbors: Optional[int] = Field(None, description="Min neighbors for kriging (default: 3)")
    default_search_radius_multiplier: Optional[float] = Field(None, description="Search radius multiplier (default: 3.0)")
    
    # Variogram calculation
    default_n_lags: Optional[int] = Field(None, description="Default number of lag bins (default: 15)")
    default_max_distance: Optional[float] = Field(None, description="Default max distance (default: 100.0)")
    
    # Sequential simulation
    default_n_realizations: Optional[int] = Field(None, description="Default realizations (default: 100)")
    default_n_thresholds: Optional[int] = Field(None, description="Default thresholds (default: 5)")
    
    # Outlier detection
    z_score_threshold: Optional[float] = Field(None, description="Z-score threshold (default: 3.0)")
    iqr_multiplier: Optional[float] = Field(None, description="IQR multiplier (default: 1.5)")
    spatial_threshold_factor: Optional[float] = Field(None, description="Spatial threshold factor (default: 3.0)")
    
    # Visualization
    scatter_size_multiplier: Optional[float] = Field(None, description="Scatter size multiplier (default: 100)")
    scatter_alpha_default: Optional[float] = Field(None, description="Default scatter alpha (default: 0.6)")
    
    # Optimization
    max_iterations_global: Optional[int] = Field(None, description="Max global iterations (default: 500)")
    max_iterations_local: Optional[int] = Field(None, description="Max local iterations (default: 1000)")
    convergence_tolerance: Optional[float] = Field(None, description="Convergence tolerance (default: 1e-6)")
    
    class Config:
        extra = "forbid"  # Don't allow extra fields
