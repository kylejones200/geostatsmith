"""
Tests for diagnostics and validation suite
"""

import pytest
import numpy as np
from geostats.diagnostics.validation_suite import comprehensive_validation
from geostats.diagnostics.outlier_detection import outlier_analysis


class TestValidationSuite:
    """Test comprehensive validation suite"""
    
    def test_comprehensive_validation(self, sample_data_2d, variogram_model):
        """Test comprehensive validation"""
        x, y, z = sample_data_2d
        
        result = comprehensive_validation(x, y, z, variogram_model)
        
        assert isinstance(result, dict)
        assert 'diagnostics' in result
        assert 'metrics' in result
    
    def test_spatial_validation(self, sample_data_2d):
        """Test spatial validation"""
        from geostats.diagnostics.validation_suite import spatial_validation
        
        x, y, z = sample_data_2d
        
        result = spatial_validation(x, y, z)
        
        assert isinstance(result, dict)
        assert 'spatial_autocorrelation' in result or 'morans_i' in result


class TestOutlierDetection:
    """Test outlier detection"""
    
    def test_outlier_analysis(self, sample_data_2d):
        """Test outlier analysis"""
        x, y, z = sample_data_2d
        
        result = outlier_analysis(x, y, z)
        
        assert isinstance(result, dict)
        assert 'outliers' in result or 'outlier_indices' in result
    
    def test_robust_validation(self, sample_data_2d):
        """Test robust validation"""
        from geostats.diagnostics.outlier_detection import robust_validation
        
        x, y, z = sample_data_2d
        
        result = robust_validation(x, y, z)
        
        assert isinstance(result, dict)
        assert 'metrics' in result or 'validation' in result
