"""
Tests for core functionality: constants, validators, exceptions, logging
"""

import pytest
import numpy as np
from geostats.core.constants import (
    EPSILON,
    DEFAULT_MAX_NEIGHBORS,
    DEFAULT_N_REALIZATIONS,
    get_constants,
    set_constants_config,
    reload_constants,
)
from geostats.core.validators import (
    validate_coordinates,
    validate_values,
    validate_positive,
    validate_in_range,
    validate_array_shapes_match,
)
from geostats.core.exceptions import GeoStatsError, ValidationError, KrigingError
from geostats.core.logging_config import get_logger, setup_logging
import logging
from pathlib import Path
import tempfile
import yaml


class TestConstants:
    """Test constants module"""
    
    def test_constants_import(self):
        """Test that constants can be imported"""
        assert EPSILON > 0
        assert DEFAULT_MAX_NEIGHBORS > 0
        assert DEFAULT_N_REALIZATIONS > 0
    
    def test_get_constants(self):
        """Test getting all constants"""
        constants = get_constants()
        assert isinstance(constants, dict)
        assert 'EPSILON' in constants
        assert 'DEFAULT_MAX_NEIGHBORS' in constants
    
    def test_constants_config_override(self):
        """Test YAML config override"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'epsilon': 1e-12, 'default_max_neighbors': 30}, f)
            config_path = f.name
        
        try:
            set_constants_config(config_path)
            constants = get_constants(config_path)
            assert constants['EPSILON'] == 1e-12
            assert constants['DEFAULT_MAX_NEIGHBORS'] == 30
        finally:
            Path(config_path).unlink()
            set_constants_config(None)  # Reset
    
    def test_reload_constants(self):
        """Test reloading constants"""
        reload_constants()
        constants = get_constants()
        assert 'EPSILON' in constants


class TestValidators:
    """Test validation utilities"""
    
    def test_validate_coordinates_1d(self):
        """Test 1D coordinate validation"""
        x = np.array([1.0, 2.0, 3.0])
        result = validate_coordinates(x)
        assert len(result) == 1
        assert np.array_equal(result[0], x)
    
    def test_validate_coordinates_2d(self):
        """Test 2D coordinate validation"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        x_val, y_val = validate_coordinates(x, y)
        assert np.array_equal(x_val, x)
        assert np.array_equal(y_val, y)
    
    def test_validate_coordinates_mismatch(self):
        """Test coordinate shape mismatch"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0])
        with pytest.raises(ValidationError):
            validate_coordinates(x, y)
    
    def test_validate_values(self):
        """Test value validation"""
        values = np.array([1.0, 2.0, 3.0])
        result = validate_values(values, n_expected=3)
        assert np.array_equal(result, values)
    
    def test_validate_values_wrong_length(self):
        """Test value length validation"""
        values = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValidationError):
            validate_values(values, n_expected=5)
    
    def test_validate_values_nan(self):
        """Test NaN detection"""
        values = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValidationError):
            validate_values(values, allow_nan=False)
    
    def test_validate_positive(self):
        """Test positive value validation"""
        assert validate_positive(5.0) == 5.0
        with pytest.raises(ValidationError):
            validate_positive(-1.0)
        with pytest.raises(ValidationError):
            validate_positive(0.0)
    
    def test_validate_in_range(self):
        """Test range validation"""
        assert validate_in_range(5.0, min_val=0.0, max_val=10.0) == 5.0
        with pytest.raises(ValidationError):
            validate_in_range(15.0, min_val=0.0, max_val=10.0)
        with pytest.raises(ValidationError):
            validate_in_range(-5.0, min_val=0.0, max_val=10.0)
    
    def test_validate_array_shapes_match(self):
        """Test array shape matching"""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        validate_array_shapes_match(arr1, arr2)  # Should not raise
        
        arr3 = np.array([1, 2])
        with pytest.raises(ValidationError):
            validate_array_shapes_match(arr1, arr3)


class TestExceptions:
    """Test exception hierarchy"""
    
    def test_geostats_error(self):
        """Test base exception"""
        error = GeoStatsError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)
    
    def test_validation_error(self):
        """Test validation error"""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, GeoStatsError)
    
    def test_kriging_error(self):
        """Test kriging error"""
        error = KrigingError("Kriging failed")
        assert str(error) == "Kriging failed"
        assert isinstance(error, GeoStatsError)


class TestLogging:
    """Test logging configuration"""
    
    def test_get_logger(self):
        """Test logger creation"""
        logger = get_logger(__name__)
        assert isinstance(logger, logging.Logger)
        assert logger.name == __name__
    
    def test_setup_logging(self):
        """Test logging setup"""
        setup_logging()
        logger = get_logger("test")
        assert logger.level <= logging.INFO
