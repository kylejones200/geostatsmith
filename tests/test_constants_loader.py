"""
Tests for constants loader and config-driven constants
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from geostats.core.constants_loader import (
    ConstantsLoader,
    get_constants,
    get_constant,
)
from geostats.core.constants import set_constants_config


class TestConstantsLoader:
    """Test constants loader functionality"""
    
    def test_load_defaults(self):
        """Test loading default constants"""
        constants = ConstantsLoader.load()
        assert isinstance(constants, dict)
        assert 'EPSILON' in constants
        assert 'DEFAULT_MAX_NEIGHBORS' in constants
        assert constants['EPSILON'] == 1e-10
    
    def test_load_with_yaml(self):
        """Test loading constants from YAML"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'epsilon': 1e-12,
                'default_max_neighbors': 30,
            }, f)
            config_path = f.name
        
        try:
            constants = ConstantsLoader.load(config_path)
            assert constants['EPSILON'] == 1e-12
            assert constants['DEFAULT_MAX_NEIGHBORS'] == 30
            # Other constants should remain as defaults
            assert 'DEFAULT_N_REALIZATIONS' in constants
        finally:
            Path(config_path).unlink()
    
    def test_cache(self):
        """Test caching mechanism"""
        constants1 = ConstantsLoader.load()
        constants2 = ConstantsLoader.load()
        # Should be same object (cached)
        assert constants1 is constants2
        
        # Clear cache
        ConstantsLoader.clear_cache()
        constants3 = ConstantsLoader.load()
        # Should be new object
        assert constants3 is not constants1
    
    def test_get_constants_function(self):
        """Test convenience function"""
        constants = get_constants()
        assert isinstance(constants, dict)
        assert 'EPSILON' in constants
    
    def test_get_constant_function(self):
        """Test getting single constant"""
        epsilon = get_constant('epsilon')
        assert epsilon == 1e-10
        
        # Case insensitive
        epsilon2 = get_constant('EPSILON')
        assert epsilon2 == epsilon
        
        # Non-existent constant
        with pytest.raises(KeyError):
            get_constant('NON_EXISTENT')
    
    def test_yaml_key_conversion(self):
        """Test lowercase to uppercase conversion"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'default_n_realizations': 200,
                'z_score_threshold': 2.5,
            }, f)
            config_path = f.name
        
        try:
            constants = get_constants(config_path)
            assert constants['DEFAULT_N_REALIZATIONS'] == 200
            assert constants['Z_SCORE_THRESHOLD'] == 2.5
        finally:
            Path(config_path).unlink()
    
    def test_missing_config_file(self):
        """Test handling of missing config file"""
        with pytest.raises(FileNotFoundError):
            ConstantsLoader.load('nonexistent.yaml')
    
    def test_invalid_yaml(self):
        """Test handling of invalid YAML"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            # Should raise an error
            with pytest.raises(Exception):  # Could be yaml.YAMLError or FileNotFoundError
                ConstantsLoader.load(config_path)
        finally:
            Path(config_path).unlink()
