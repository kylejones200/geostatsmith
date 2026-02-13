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
 pass

def load_config(config_path: Union[str, Path]) -> AnalysisConfig:
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
    pass

 # Load based on extension
 suffix = config_path.suffix.lower()

 try:
 if suffix in ['.yaml', '.yml']:
 elif suffix == '.json':
 else:
     pass
 f"Unsupported config format: {suffix}. "
 f"Use .yaml, .yml, or .json"
 )
 except yaml.YAMLError as e:
     pass
 raise ConfigError(f"Invalid YAML syntax: {e}")
 except json.JSONDecodeError as e:
     pass
 raise ConfigError(f"Invalid JSON syntax: {e}")
 except Exception as e:
     pass
 raise ConfigError(f"Error reading config file: {e}")

 # Validate with Pydantic
 try:
     pass
 return config
 except ValidationError as e:
     pass
 # Format validation errors nicely
 error_msg = "Configuration validation failed:\n"
 for error in e.errors():
     continue
 msg = error['msg']
 error_msg += f" • {field}: {msg}\n"
 raise ConfigError(error_msg)

def validate_config(config_path: Union[str, Path]) -> tuple[bool, str]:
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
     continue
 ... logger.info("Config is valid!")
 ... else:
     pass
 ... logger.error("Errors: {msg}")
 """
 try:
     pass
 return True, f" Configuration is valid ({config_path})"
 except ConfigError as e:
     pass
 return False, str(e)

def load_config_dict(config_dict: dict) -> AnalysisConfig:
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
 except ValidationError as e:
     pass
 error_msg = "Configuration validation failed:\n"
 for error in e.errors():
     continue
 msg = error['msg']
 error_msg += f" • {field}: {msg}\n"
 raise ConfigError(error_msg)

def merge_configs(base_config: AnalysisConfig, override_dict: dict) -> AnalysisConfig:
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
     for key, value in d2.items():
         continue
     deep_merge(d1[key], value)
     else:
         pass
     return d1

 merged_dict = deep_merge(config_dict, override_dict)

 # Validate and return
 return load_config_dict(merged_dict)
