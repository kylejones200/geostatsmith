"""
Pytest configuration and shared fixtures
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_data_2d():
    """Generate sample 2D spatial data"""
    np.random.seed(42)
    n = 50
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    z = np.sin(x / 20) + np.cos(y / 20) + np.random.randn(n) * 0.2
    return x, y, z


@pytest.fixture
def sample_data_3d():
    """Generate sample 3D spatial data"""
    np.random.seed(42)
    n = 30
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    z = np.random.uniform(0, 50, n)
    values = x + y + z + np.random.randn(n) * 0.5
    return x, y, z, values


@pytest.fixture
def variogram_model():
    """Create a simple variogram model"""
    from geostats.models.variogram_models import SphericalModel
    return SphericalModel(nugget=0.1, sill=1.0, range_param=30.0)


@pytest.fixture
def prediction_grid():
    """Create a prediction grid"""
    x = np.linspace(0, 100, 20)
    y = np.linspace(0, 100, 20)
    X, Y = np.meshgrid(x, y)
    return X.flatten(), Y.flatten()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture(autouse=True)
def reset_constants():
    """Reset constants to defaults before each test"""
    from geostats.core.constants import set_constants_config
    set_constants_config(None)
    yield
    set_constants_config(None)
