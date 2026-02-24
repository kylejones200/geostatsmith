"""
Tests for AutoML functionality
"""

from geostats.automl.auto_method import auto_select_method
from geostats.automl.auto_variogram import auto_fit_variogram
from geostats.automl.hyperparameter_tuning import tune_variogram_hyperparameters


class TestAutoVariogram:
    """Test automatic variogram fitting"""

    def test_auto_fit_variogram(self, sample_data_2d):
        """Test automatic variogram model selection"""
        x, y, z = sample_data_2d

        result = auto_fit_variogram(x, y, z)

        assert "model" in result
        assert "parameters" in result
        assert "criterion" in result
        assert result["model"] is not None

    def test_auto_fit_with_models(self, sample_data_2d):
        """Test auto fit with specific models"""
        x, y, z = sample_data_2d

        from geostats.models.variogram_models import ExponentialModel, SphericalModel

        models = [SphericalModel, ExponentialModel]

        result = auto_fit_variogram(x, y, z, models=models)

        assert result["model"] is not None
        assert result["model"].__class__ in [SphericalModel, ExponentialModel]


class TestAutoMethod:
    """Test automatic method selection"""

    def test_auto_select_method(self, sample_data_2d):
        """Test automatic interpolation method selection"""
        x, y, z = sample_data_2d

        result = auto_select_method(x, y, z)

        assert "method" in result
        assert "score" in result
        assert result["method"] in ["ordinary_kriging", "simple_kriging", "idw"]


class TestHyperparameterTuning:
    """Test hyperparameter tuning"""

    def test_tune_variogram_hyperparameters(self, sample_data_2d):
        """Test variogram hyperparameter tuning"""
        x, y, z = sample_data_2d

        from geostats.models.variogram_models import SphericalModel

        result = tune_variogram_hyperparameters(
            x,
            y,
            z,
            model_class=SphericalModel,
            n_trials=5,  # Small number for testing
        )

        assert "best_params" in result
        assert "best_score" in result
        assert "nugget" in result["best_params"]
        assert "sill" in result["best_params"]
        assert "range" in result["best_params"]
