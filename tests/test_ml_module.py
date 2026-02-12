"""
Tests for ML module (regression kriging, gaussian process, ensemble)
"""

import pytest
import numpy as np

# Check if sklearn is available
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Check if xgboost is available
try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from geostats.ml.regression_kriging import RegressionKriging, RandomForestKriging
from geostats.ml.gaussian_process import GaussianProcessGeostat
from geostats.ml.ensemble import EnsembleKriging
from geostats.algorithms.ordinary_kriging import OrdinaryKriging
from geostats.models.variogram_models import SphericalModel, ExponentialModel

# Skip all tests if sklearn not available
pytestmark = pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")

class TestRegressionKriging:

    def test_initialization(self):
        ml_model = LinearRegression()
        rk = RegressionKriging(ml_model=ml_model, variogram_model='spherical')
        assert rk is not None
        assert rk.variogram_model_type == 'spherical'

    def test_fit_predict(self):
        # Generate synthetic data
        np.random.seed(42)
        x = np.random.uniform(0, 10, 30)
        y = np.random.uniform(0, 10, 30)
        z = 2 * x + 3 * y + np.random.normal(0, 0.5, 30)

        # Create covariates
        covariates = np.column_stack([x**2, y**2])

        ml_model = LinearRegression()
        rk = RegressionKriging(ml_model=ml_model, variogram_model='spherical')
        rk.fit(x, y, z, covariates=covariates)

        # Predict at new locations
        x_new = np.array([5.0, 7.0])
        y_new = np.array([5.0, 7.0])
        cov_new = np.column_stack([x_new**2, y_new**2])

        predictions, variance = rk.predict(x_new, y_new, covariates_new=cov_new)

        assert predictions.shape == (2,)
        assert variance.shape == (2,)
        assert np.all(variance >= 0)

    def test_without_covariates(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 20)
        y = np.random.uniform(0, 10, 20)
        z = np.sin(x) + np.cos(y) + np.random.normal(0, 0.1, 20)

        ml_model = LinearRegression()
        rk = RegressionKriging(ml_model=ml_model, variogram_model='exponential')
        rk.fit(x, y, z)

        predictions, variance = rk.predict(np.array([5.0]), np.array([5.0]))

        assert predictions.shape == (1,)
        assert variance.shape == (1,)

class TestRandomForestKriging:

    def test_initialization(self):
        rfk = RandomForestKriging(n_estimators=10, variogram_model='spherical')
        assert rfk is not None
        assert rfk.n_estimators == 10

    def test_fit_predict_with_covariates(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 40)
        y = np.random.uniform(0, 10, 40)

        # Non-linear relationship
        covariates = np.column_stack([x, y, x**2, y**2, x*y])
        z = np.sin(x) * np.cos(y) + 0.5 * x + np.random.normal(0, 0.2, 40)

        rfk = RandomForestKriging(n_estimators=10, variogram_model='exponential')
        rfk.fit(x, y, z, covariates=covariates)

        x_new = np.array([5.0, 6.0])
        y_new = np.array([5.0, 6.0])
        cov_new = np.column_stack([x_new, y_new, x_new**2, y_new**2, x_new*y_new])

        predictions, variance = rfk.predict(x_new, y_new, covariates_new=cov_new)

        assert predictions.shape == (2,)
        assert variance.shape == (2,)


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="xgboost not installed")
class TestXGBoostKriging:

    def test_initialization(self):
        from geostats.ml.regression_kriging import XGBoostKriging
        xgbk = XGBoostKriging(n_estimators=10, variogram_model='gaussian')
        assert xgbk is not None

    def test_fit_predict(self):
        from geostats.ml.regression_kriging import XGBoostKriging
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 10, 50)

        covariates = np.column_stack([x, y, np.sqrt(x), np.sqrt(y)])
        z = x**1.5 + y**1.5 + np.random.normal(0, 0.3, 50)

        xgbk = XGBoostKriging(n_estimators=10, variogram_model='spherical')
        xgbk.fit(x, y, z, covariates=covariates)

        x_new = np.array([3.0, 7.0])
        y_new = np.array([4.0, 8.0])
        cov_new = np.column_stack([x_new, y_new, np.sqrt(x_new), np.sqrt(y_new)])

        predictions, variance = xgbk.predict(x_new, y_new, covariates_new=cov_new)

        assert predictions.shape == (2,)
        assert variance.shape == (2,)

class TestGaussianProcessGeostat:

    def test_initialization(self):
        gp = GaussianProcessGeostat(kernel='rbf')
        assert gp is not None

    def test_fit_predict(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 30)
        y = np.random.uniform(0, 10, 30)
        z = np.sin(x) + np.cos(y) + np.random.normal(0, 0.1, 30)

        gp = GaussianProcessGeostat(kernel='rbf')
        gp.fit(x, y, z)

        x_new = np.array([5.0, 6.0, 7.0])
        y_new = np.array([5.0, 6.0, 7.0])

        predictions, std = gp.predict(x_new, y_new, return_std=True)

        assert predictions.shape == (3,)
        assert std.shape == (3,)
        assert np.all(std >= 0)

    def test_different_kernels(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 25)
        y = np.random.uniform(0, 10, 25)
        z = x + y + np.random.normal(0, 0.1, 25)

        for kernel in ['rbf', 'matern', 'rational_quadratic']:
            gp.fit(x, y, z)

        predictions, _ = gp.predict(np.array([5.0]), np.array([5.0]), return_std=True)
        assert predictions.shape == (1,)

    def test_hyperparameter_optimization(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 40)
        y = np.random.uniform(0, 10, 40)
        z = np.sin(x) * np.cos(y) + np.random.normal(0, 0.2, 40)

        gp = GaussianProcessGeostat(kernel='rbf', optimize=True)
        gp.fit(x, y, z)

        # Check that hyperparameters were optimized
        assert hasattr(gp, 'kernel_params_')

        predictions, _ = gp.predict(np.array([5.0]), np.array([5.0]), return_std=True)
        assert predictions.shape == (1,)

class TestEnsembleKriging:

    def test_initialization(self):
        # Create some simple kriging models
        np.random.seed(42)
        x = np.random.uniform(0, 10, 20)
        y = np.random.uniform(0, 10, 20)
        z = x + y + np.random.normal(0, 0.2, 20)

        model1 = OrdinaryKriging(x, y, z, SphericalModel(sill=1.0, range_param=5.0))
        model2 = OrdinaryKriging(x, y, z, ExponentialModel(sill=1.0, range_param=5.0))

        ek = EnsembleKriging(models=[model1, model2])
        assert ek is not None
        assert len(ek.models) == 2

    def test_fit_predict(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 35)
        y = np.random.uniform(0, 10, 35)
        z = np.sin(x) + np.cos(y) + np.random.normal(0, 0.15, 35)

        model1 = OrdinaryKriging(x, y, z, SphericalModel(sill=1.0, range_param=5.0))
        model2 = OrdinaryKriging(x, y, z, ExponentialModel(sill=1.0, range_param=5.0))

        ek = EnsembleKriging(models=[model1, model2])

        x_new = np.array([5.0, 6.0])
        y_new = np.array([5.0, 6.0])

        predictions, variance = ek.predict(x_new, y_new)

        assert predictions.shape == (2,)
        assert variance.shape == (2,)
        assert np.all(variance >= 0)

    def test_weighted_ensemble(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 30)
        y = np.random.uniform(0, 10, 30)
        z = x + 2*y + np.random.normal(0, 0.2, 30)

        model1 = OrdinaryKriging(x, y, z, SphericalModel(sill=1.0, range_param=5.0))
        model2 = OrdinaryKriging(x, y, z, ExponentialModel(sill=1.0, range_param=5.0))

        ek = EnsembleKriging(
        models=[model1, model2],
        weighting='equal'
        )

        predictions, variance = ek.predict(np.array([5.0]), np.array([5.0]))

        assert predictions.shape == (1,)
        assert variance.shape == (1,)

    def test_inverse_variance_weighting(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 40)
        y = np.random.uniform(0, 10, 40)
        z = np.sin(x) * np.cos(y) + np.random.normal(0, 0.2, 40)

        model1 = OrdinaryKriging(x, y, z, SphericalModel(sill=1.0, range_param=5.0))
        model2 = OrdinaryKriging(x, y, z, ExponentialModel(sill=1.0, range_param=5.0))

        ek = EnsembleKriging(
        models=[model1, model2],
        weighting='inverse_variance'
        )

        predictions, variance = ek.predict(np.array([5.0, 6.0]), np.array([5.0, 6.0]))

        assert predictions.shape == (2,)
        assert variance.shape == (2,)

class TestMLIntegration:

    def test_compare_methods(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 10, 50)
        z = 2*x + 3*y + np.random.normal(0, 0.5, 50)

        covariates = np.column_stack([x, y])

        # Test multiple methods
        ml_model1 = LinearRegression()
        ml_model2 = RandomForestRegressor(n_estimators=10, random_state=42)

        methods = [
        RegressionKriging(ml_model=ml_model1, variogram_model='spherical'),
        RandomForestKriging(n_estimators=10, variogram_model='spherical'),
        GaussianProcessGeostat(kernel='spherical'),
        ]

        x_new = np.array([5.0])
        y_new = np.array([5.0])
        cov_new = np.column_stack([x_new, y_new])

        predictions = []
        for method in methods:
            X = np.column_stack([x, y])
        X_new = np.column_stack([x_new, y_new])
        method.fit(X, z)
        pred, _ = method.predict(X_new, return_std=True)
        else:
            pred, _ = method.predict(x_new, y_new, covariates_new=cov_new)

        predictions.append(pred[0])

        # All methods should give reasonable predictions
        assert len(predictions) == 3
        # For linear data, predictions should be similar
        assert np.std(predictions) < 10.0

if __name__ == "__main__":
    pytest.main([__file__])
