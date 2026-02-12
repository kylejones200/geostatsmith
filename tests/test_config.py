"""
Test config system functionality

Tests the config-driven architecture without requiring full pipeline execution.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from geostats.config import (
 AnalysisConfig,
 load_config,
 validate_config,
 load_config_dict,
 merge_configs,
 ConfigError,
)

def test_minimal_config():
 config_dict = {
 'project': {
 'name': 'Test',
 'output_dir': './results'
 },
 'data': {
 'input_file': __file__, # Use this file as it exists
 'x_column': 'X',
 'y_column': 'Y',
 'z_column': 'Z'
 }
 }

 config = load_config_dict(config_dict)
 assert config.project.name == 'Test'
 assert config.data.x_column == 'X'
 assert config.kriging.method == 'ordinary' # Default

def test_config_validation():
    # Invalid: missing required fields
    with pytest.raises(ConfigError):
        # Missing data section
        load_config({
            'project': {'name': 'test'}
        })

    # Invalid: wrong type
    with pytest.raises(ConfigError):
        load_config({
            'data': {
                'input_file': __file__,
                'x_column': 'X',
                'y_column': 'Y',
                'z_column': 'Z'
            },
            'variogram': {
                'n_lags': 'invalid'  # Should be int
            }
        })

 # Invalid: value out of range
 with pytest.raises(ConfigError):
     'data': {
 'input_file': __file__,
 'x_column': 'X',
 'y_column': 'Y',
 'z_column': 'Z'
 },
 'variogram': {
 'n_lags': 2 # Must be >= 5
 }
 })

def test_yaml_loading():
 config_dict = {
 'project': {'name': 'YAML Test', 'output_dir': './results'},
 'data': {
 'input_file': __file__,
 'x_column': 'X',
 'y_column': 'Y',
 'z_column': 'Z'
 }
 }

 # Write to temp file
 with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:

     try:
 config = load_config(temp_path)
 assert config.project.name == 'YAML Test'

 # Validate
 valid, msg = validate_config(temp_path)
 assert valid
 assert 'valid' in msg.lower()
 finally:
 Path(temp_path).unlink()

def test_config_merging():
     base_dict = {
 'project': {'name': 'Base', 'output_dir': './results'},
 'data': {
 'input_file': __file__,
 'x_column': 'X',
 'y_column': 'Y',
 'z_column': 'Z'
 },
 'kriging': {
 'method': 'ordinary',
 'grid': {'resolution': 1.0}
 }
 }

 base_config = load_config_dict(base_dict)

 # Override some values
 overrides = {
 'project': {'name': 'Modified'},
 'kriging': {
 'method': 'simple',
 'grid': {'resolution': 0.5}
 }
 }

 merged = merge_configs(base_config, overrides)

 assert merged.project.name == 'Modified'
 assert merged.kriging.method == 'simple'
 assert merged.kriging.grid.resolution == 0.5
 assert merged.data.x_column == 'X' # Preserved from base

def test_default_values():
 config_dict = {
 'project': {'name': 'Test', 'output_dir': './results'},
 'data': {
 'input_file': __file__,
 'x_column': 'X',
 'y_column': 'Y',
 'z_column': 'Z'
 }
 }

 config = load_config_dict(config_dict)

 # Check defaults
 assert config.preprocessing.remove_outliers == False
 assert config.preprocessing.transform is None
 assert config.variogram.n_lags == 15
 assert config.variogram.estimator == 'matheron'
 assert config.variogram.auto_fit == True
 assert config.kriging.method == 'ordinary'
 assert config.kriging.neighborhood.max_neighbors == 25
 assert config.validation.cross_validation == True
 assert config.visualization.style == 'minimalist'
 assert config.output.save_predictions == True

def test_cross_field_validation():
 # Cokriging without secondary variable
 with pytest.raises(ConfigError) as excinfo:
     'data': {
 'input_file': __file__,
 'x_column': 'X',
 'y_column': 'Y',
 'z_column': 'Z'
 # Missing z_secondary
 },
 'kriging': {
 'method': 'cokriging' # Requires z_secondary
 }
 })
 assert 'cokriging' in str(excinfo.value).lower()

 # Indicator kriging without thresholds
 with pytest.raises(ConfigError) as excinfo:
     'data': {
 'input_file': __file__,
 'x_column': 'X',
 'y_column': 'Y',
 'z_column': 'Z'
 },
 'kriging': {
 'method': 'indicator'
 # Missing thresholds
 }
 })
 assert 'indicator' in str(excinfo.value).lower()

def test_neighborhood_validation():
 # max_neighbors < min_neighbors
 with pytest.raises(ConfigError):
     'data': {
 'input_file': __file__,
 'x_column': 'X',
 'y_column': 'Y',
 'z_column': 'Z'
 },
 'kriging': {
 'neighborhood': {
 'max_neighbors': 5,
 'min_neighbors': 10 # Invalid: max < min
 }
 }
 })

def test_file_not_found_validation():
 with pytest.raises(ConfigError) as excinfo:
     'data': {
 'input_file': '/nonexistent/file.csv',
 'x_column': 'X',
 'y_column': 'Y',
 'z_column': 'Z'
 }
 })
 assert 'not found' in str(excinfo.value).lower()

if __name__ == '__main__':
