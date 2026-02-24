"""
Tests for base classes: BaseModel, BaseKriging
"""

import numpy as np
import pytest

from geostats.core.base import BaseKriging, BaseModel
from geostats.models.variogram_models import SphericalModel


class TestBaseModel:
    """Test BaseModel abstract class"""

    def test_base_model_initialization(self):
        """Test BaseModel initialization"""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            BaseModel()

    def test_concrete_model(self):
        """Test concrete model implementation"""
        model = SphericalModel(nugget=0.1, sill=1.0, range_param=30.0)
        assert isinstance(model, BaseModel)
        assert model.is_fitted is False

    def test_model_parameters(self):
        """Test model parameters"""
        model = SphericalModel(nugget=0.1, sill=1.0, range_param=30.0)
        params = model.parameters
        assert isinstance(params, dict)
        assert "nugget" in params
        assert "sill" in params
        assert "range" in params

    def test_set_parameters(self):
        """Test setting parameters"""
        model = SphericalModel()
        model.set_parameters(nugget=0.2, sill=1.5, range=25.0)
        assert model.is_fitted is True
        assert model.parameters["nugget"] == 0.2
        assert model.parameters["sill"] == 1.5

    def test_model_call(self):
        """Test model evaluation"""
        model = SphericalModel(nugget=0.1, sill=1.0, range_param=30.0)
        h = np.array([0.0, 10.0, 30.0, 60.0])
        values = model(h)
        assert len(values) == len(h)
        assert values[0] == 0.1  # Nugget at h=0
        assert values[-1] == 1.0  # Sill at large h


class TestBaseKriging:
    """Test BaseKriging abstract class"""

    def test_base_kriging_initialization(self):
        """Test BaseKriging initialization"""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            BaseKriging(
                np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0])
            )

    def test_concrete_kriging(self):
        """Test concrete kriging implementation"""
        from geostats.algorithms.ordinary_kriging import OrdinaryKriging

        x = np.array([0.0, 10.0, 20.0])
        y = np.array([0.0, 10.0, 20.0])
        z = np.array([1.0, 2.0, 1.5])
        model = SphericalModel(nugget=0.1, sill=1.0, range_param=30.0)

        krig = OrdinaryKriging(x, y, z, variogram_model=model)
        assert isinstance(krig, BaseKriging)
        assert krig.n_points == 3

    def test_kriging_repr(self):
        """Test kriging string representation"""
        from geostats.algorithms.ordinary_kriging import OrdinaryKriging

        x = np.array([0.0, 10.0])
        y = np.array([0.0, 10.0])
        z = np.array([1.0, 2.0])
        model = SphericalModel(nugget=0.1, sill=1.0, range_param=30.0)

        krig = OrdinaryKriging(x, y, z, variogram_model=model)
        repr_str = repr(krig)
        assert "OrdinaryKriging" in repr_str
        assert "n_points=2" in repr_str
