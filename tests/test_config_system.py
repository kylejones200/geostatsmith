"""
Tests for config system: schemas, parser, constants config
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from geostats.config import (
    load_config,
    validate_config,
    load_config_dict,
    merge_configs,
    ConfigError,
    AnalysisConfig,
    ConstantsConfig,
)


class TestConfigSchemas:
    """Test configuration schemas"""
    
    def test_project_config(self):
        """Test ProjectConfig"""
        from geostats.config.schemas import ProjectConfig
        config = ProjectConfig(name="Test Project", output_dir="./output")
        assert config.name == "Test Project"
        assert config.output_dir == Path("./output")
    
    def test_data_config(self):
        """Test DataConfig"""
        from geostats.config.schemas import DataConfig
        config = DataConfig(
            input_file="data.csv",
            x_column="x",
            y_column="y",
            z_column="z"
        )
        assert config.input_file == "data.csv"
        assert config.x_column == "x"
    
    def test_constants_config(self):
        """Test ConstantsConfig"""
        config = ConstantsConfig(
            epsilon=1e-12,
            default_max_neighbors=30
        )
        assert config.epsilon == 1e-12
        assert config.default_max_neighbors == 30
    
    def test_analysis_config_full(self):
        """Test complete AnalysisConfig"""
        config_dict = {
            'project': {
                'name': 'Full Test',
                'output_dir': './output',
            },
            'data': {
                'input_file': 'data.csv',
                'x_column': 'x',
                'y_column': 'y',
                'z_column': 'z',
            },
            'constants': {
                'epsilon': 1e-12,
            },
        }
        config = AnalysisConfig(**config_dict)
        assert config.project.name == 'Full Test'
        assert config.constants.epsilon == 1e-12


class TestConfigParser:
    """Test configuration parser"""
    
    def test_load_config_yaml(self):
        """Test loading YAML config"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'project': {
                    'name': 'YAML Config',
                    'output_dir': './output',
                },
                'data': {
                    'input_file': 'data.csv',
                    'x_column': 'x',
                    'y_column': 'y',
                    'z_column': 'z',
                },
            }, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert config.project.name == 'YAML Config'
        finally:
            Path(config_path).unlink()
    
    def test_load_config_json(self):
        """Test loading JSON config"""
        import json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'project': {
                    'name': 'JSON Config',
                    'output_dir': './output',
                },
                'data': {
                    'input_file': 'data.csv',
                    'x_column': 'x',
                    'y_column': 'y',
                    'z_column': 'z',
                },
            }, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert config.project.name == 'JSON Config'
        finally:
            Path(config_path).unlink()
    
    def test_validate_config(self):
        """Test config validation"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'project': {
                    'name': 'Valid Config',
                    'output_dir': './output',
                },
                'data': {
                    'input_file': 'data.csv',
                    'x_column': 'x',
                    'y_column': 'y',
                    'z_column': 'z',
                },
            }, f)
            config_path = f.name
        
        try:
            valid, msg = validate_config(config_path)
            assert valid is True
        finally:
            Path(config_path).unlink()
    
    def test_load_config_dict(self):
        """Test loading from dictionary"""
        config_dict = {
            'project': {
                'name': 'Dict Config',
                'output_dir': './output',
            },
            'data': {
                'input_file': 'data.csv',
                'x_column': 'x',
                'y_column': 'y',
                'z_column': 'z',
            },
        }
        config = load_config_dict(config_dict)
        assert config.project.name == 'Dict Config'
    
    def test_merge_configs(self):
        """Test config merging"""
        base_dict = {
            'project': {
                'name': 'Base',
                'output_dir': './output',
            },
            'data': {
                'input_file': 'data.csv',
                'x_column': 'x',
                'y_column': 'y',
                'z_column': 'z',
            },
        }
        base = load_config_dict(base_dict)
        
        override = {
            'project': {
                'name': 'Merged',
            },
        }
        
        merged = merge_configs(base, override)
        assert merged.project.name == 'Merged'
        assert merged.data.input_file == 'data.csv'  # Should be preserved
    
    def test_config_error_missing_file(self):
        """Test ConfigError for missing file"""
        with pytest.raises(ConfigError):
            load_config('nonexistent.yaml')
    
    def test_config_error_invalid_format(self):
        """Test ConfigError for invalid format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("not a config file")
            config_path = f.name
        
        try:
            with pytest.raises(ConfigError):
                load_config(config_path)
        finally:
            Path(config_path).unlink()
