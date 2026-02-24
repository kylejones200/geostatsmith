"""
Constants Loader

Loads constants with optional YAML overrides for config-driven workloads.
Python constants serve as defaults, YAML configs can override them.
"""

from pathlib import Path
from typing import Any

import yaml


class ConstantsLoader:
    """
    Load constants with optional YAML overrides.

    Usage:
        # Use Python defaults
        from geostats.core.constants_loader import get_constants
        EPSILON = get_constants()['EPSILON']

        # Override with YAML
        EPSILON = get_constants('config/constants.yaml')['EPSILON']
    """

    _cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def load(cls, config_path: str | None = None) -> dict[str, Any]:
        """
        Load constants, optionally overriding with YAML config.

        Parameters
        ----------
        config_path : str, optional
            Path to YAML constants config file. If None, returns Python defaults.

        Returns
        -------
        dict
            Dictionary of constants (uppercase keys matching Python constants)
        """
        # Check cache
        cache_key = config_path or "defaults"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        # Start with Python defaults
        constants = cls._get_python_defaults()

        # Override with YAML if provided
        if config_path:
            yaml_overrides = cls._load_yaml_overrides(config_path)
            constants.update(yaml_overrides)

        # Cache result
        cls._cache[cache_key] = constants
        return constants

    @classmethod
    def _get_python_defaults(cls) -> dict[str, Any]:
        """Get all Python constants as a dictionary."""
        # Import the default constants dict directly to avoid circular import
        from .constants import _DEFAULT_CONSTANTS

        return _DEFAULT_CONSTANTS.copy()

    @classmethod
    def _load_yaml_overrides(cls, config_path: str) -> dict[str, Any]:
        """
        Load YAML config and convert to uppercase constant names.

        YAML uses lowercase_with_underscores, converts to UPPERCASE.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Constants config not found: {config_path}")

        with open(config_path) as f:
            yaml_data = yaml.safe_load(f)

        if not yaml_data:
            return {}

        # Convert lowercase YAML keys to uppercase constant names
        overrides = {}
        for yaml_key, value in yaml_data.items():
            # Convert 'default_max_neighbors' -> 'DEFAULT_MAX_NEIGHBORS'
            const_key = yaml_key.upper()
            overrides[const_key] = value

        return overrides

    @classmethod
    def clear_cache(cls):
        """Clear the constants cache."""
        cls._cache.clear()


def get_constants(config_path: str | None = None) -> dict[str, Any]:
    """
    Convenience function to get constants.

    Parameters
    ----------
    config_path : str, optional
        Path to YAML constants config file

    Returns
    -------
    dict
        Dictionary of constants
    """
    return ConstantsLoader.load(config_path)


def get_constant(name: str, config_path: str | None = None) -> Any:
    """
    Get a single constant by name.

    Parameters
    ----------
    name : str
        Constant name (case-insensitive, but returns uppercase key)
    config_path : str, optional
        Path to YAML constants config file

    Returns
    -------
    Any
        Constant value
    """
    constants = get_constants(config_path)
    name_upper = name.upper()
    if name_upper not in constants:
        raise KeyError(
            f"Constant '{name}' not found. Available: {list(constants.keys())}"
        )
    return constants[name_upper]
