"""
Configuration management for geostats

Provides schemas and parsers for config-driven analysis.
"""

from .schemas import (
    AnalysisConfig,
    ProjectConfig,
    DataConfig,
    PreprocessingConfig,
    VariogramConfig,
    KrigingConfig,
    ValidationConfig,
    VisualizationConfig,
    OutputConfig,
    NeighborhoodConfig,
    GridConfig,
    PlotConfig,
)
from .constants_config import ConstantsConfig

from .parser import (
    load_config,
    validate_config,
    load_config_dict,
    merge_configs,
    ConfigError,
)

__all__ = [
    # Schemas
    "AnalysisConfig",
    "ProjectConfig",
    "DataConfig",
    "PreprocessingConfig",
    "VariogramConfig",
    "KrigingConfig",
    "ValidationConfig",
    "VisualizationConfig",
    "OutputConfig",
    "NeighborhoodConfig",
    "GridConfig",
    "PlotConfig",
    "ConstantsConfig",
    # Parser
    "load_config",
    "validate_config",
    "load_config_dict",
    "merge_configs",
    "ConfigError",
]
