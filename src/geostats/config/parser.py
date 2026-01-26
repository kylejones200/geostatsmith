"""
Configuration file parser and loader

Supports YAML and JSON formats with validation.
"""

import yaml
import json
from pathlib import Path
from typing import Union
from pydantic import ValidationError

from .schemas import AnalysisConfig
from ..core.exceptions import GeoStatsError
import logging

logger = logging.getLogger(__name__)

class ConfigError(GeoStatsError):
 """Configuration error"""
 pass

def load_config(config_path: Union[str, Path]) -> AnalysisConfig:
 """
 Load and validate configuration file

 Parameters
 ----------
 config_path : str or Path
 Path to configuration file (.yaml, .yml, or .json)

 Returns
 -------
 AnalysisConfig
 Validated configuration object

 Raises
 ------
 ConfigError
 If file not found, invalid format, or validation fails

 Examples
 --------
 >>> config = load_config('analysis.yaml')
 >>> logger.info(config.project.name)
 'My Analysis'
 """
 config_path = Path(config_path)

 # Check file exists
 if not config_path.exists():
 raise ConfigError(f"Configuration file not found: {config_path}")

 # Load based on extension
 suffix = config_path.suffix.lower()

 try:
 with open(config_path, 'r') as f:
 if suffix in ['.yaml', '.yml']:
 config_dict = yaml.safe_load(f)
 elif suffix == '.json':
 config_dict = json.load(f)
 else:
 raise ConfigError(
 f"Unsupported config format: {suffix}. "
 f"Use .yaml, .yml, or .json"
 )
 except yaml.YAMLError as e:
 raise ConfigError(f"Invalid YAML syntax: {e}")
 except json.JSONDecodeError as e:
 raise ConfigError(f"Invalid JSON syntax: {e}")
 except Exception as e:
 raise ConfigError(f"Error reading config file: {e}")

 # Validate with Pydantic
 try:
 config = AnalysisConfig(**config_dict)
 return config
 except ValidationError as e:
 # Format validation errors nicely
 error_msg = "Configuration validation failed:\n"
 for error in e.errors():
 field = " -> ".join(str(loc) for loc in error['loc'])
 msg = error['msg']
 error_msg += f" • {field}: {msg}\n"
 raise ConfigError(error_msg)

def validate_config(config_path: Union[str, Path]) -> tuple[bool, str]:
 """
 Validate configuration file without loading

 Parameters
 ----------
 config_path : str or Path
 Path to configuration file

 Returns
 -------
 is_valid : bool
 True if valid, False otherwise
 message : str
 Success message or error details

 Examples
 --------
 >>> valid, msg = validate_config('analysis.yaml')
 >>> if valid:
 ... logger.info("Config is valid!")
 ... else:
 ... logger.error("Errors: {msg}")
 """
 try:
 config = load_config(config_path)
 return True, f" Configuration is valid ({config_path})"
 except ConfigError as e:
 return False, str(e)

def load_config_dict(config_dict: dict) -> AnalysisConfig:
 """
 Load configuration from dictionary

 Useful for programmatic config creation or testing.

 Parameters
 ----------
 config_dict : dict
 Configuration dictionary

 Returns
 -------
 AnalysisConfig
 Validated configuration object
 """
 try:
 return AnalysisConfig(**config_dict)
 except ValidationError as e:
 error_msg = "Configuration validation failed:\n"
 for error in e.errors():
 field = " -> ".join(str(loc) for loc in error['loc'])
 msg = error['msg']
 error_msg += f" • {field}: {msg}\n"
 raise ConfigError(error_msg)

def merge_configs(base_config: AnalysisConfig, override_dict: dict) -> AnalysisConfig:
 """
 Merge configuration with overrides

 Useful for command-line overrides or parameter sweeps.

 Parameters
 ----------
 base_config : AnalysisConfig
 Base configuration
 override_dict : dict
 Dictionary with values to override

 Returns
 -------
 AnalysisConfig
 Merged configuration

 Examples
 --------
 >>> base = load_config('base.yaml')
 >>> overrides = {'project': {'name': 'Modified Analysis'}}
 >>> config = merge_configs(base, overrides)
 """
 # Convert base config to dict
 config_dict = base_config.model_dump()

 # Deep merge overrides
 def deep_merge(d1, d2):
 """Recursively merge d2 into d1"""
 for key, value in d2.items():
 if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
 deep_merge(d1[key], value)
 else:
 d1[key] = value
 return d1

 merged_dict = deep_merge(config_dict, override_dict)

 # Validate and return
 return load_config_dict(merged_dict)
