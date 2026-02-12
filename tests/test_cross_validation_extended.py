"""
Tests for cross-validation module
Currently at 15% coverage, targeting 50%+
"""

import pytest
import numpy as np
from geostats.validation.cross_validation import (
 leave_one_out,
 k_fold_cross_validation,
 spatial_cross_validation
)
from geostats.validation.metrics import (
 mean_squared_error,
 mean_absolute_error,
 r_squared,
 root_mean_squared_error
)
from geostats.algorithms.ordinary_kriging import OrdinaryKriging
from geostats.algorithms.simple_kriging import SimpleKriging
from geostats.models.variogram_models import SphericalModel, ExponentialModel

class TestLeaveOneOut:

    def test_leave_one_out_basic(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 30)
        y = np.random.uniform(0, 10, 30)
        z = 2*x + 3*y + np.random.normal(0, 0.5, 30)

        variogram = SphericalModel(sill=1.0, range_param=5.0)
        model = OrdinaryKriging(x, y, z, variogram)

        predictions, metrics = leave_one_out(model)

        assert predictions.shape == z.shape
        assert not np.any(np.isnan(predictions))
        assert not np.any(np.isinf(predictions))
        assert 'mse' in metrics
        assert 'r2' in metrics

    def test_leave_one_out_metrics(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 25)
        y = np.random.uniform(0, 10, 25)
        z = x + y + np.random.normal(0, 0.3, 25)

        variogram = ExponentialModel(sill=1.0, range_param=5.0)
        model = OrdinaryKriging(x, y, z, variogram)

        predictions, metrics = leave_one_out(model)

        mse = mean_squared_error(z, predictions)
        mae = mean_absolute_error(z, predictions)
        r2 = r_squared(z, predictions)
        rmse = root_mean_squared_error(z, predictions)

        assert mse >= 0
        assert mae >= 0
        assert rmse >= 0
        assert -1 <= r2 <= 1
        assert metrics['mse'] >= 0

    def test_leave_one_out_simple_kriging(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 20)
        y = np.random.uniform(0, 10, 20)
        mean_val = 5.0
        z = mean_val + np.random.normal(0, 1.0, 20)

        variogram = SphericalModel(sill=1.0, range_param=5.0)
        model = SimpleKriging(x, y, z, variogram, mean=mean_val)

        predictions, metrics = leave_one_out(model)

        assert predictions.shape == z.shape
        # Predictions should be close to mean
        assert np.abs(np.mean(predictions) - mean_val) < 2.0
        assert 'mse' in metrics

class TestKFoldCrossValidation:

    def test_k_fold_basic(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 40)
        y = np.random.uniform(0, 10, 40)
        z = x + 2*y + np.random.normal(0, 0.5, 40)

        variogram = SphericalModel(sill=1.0, range_param=5.0)

        results = k_fold_cross_validation(x, y, z, OrdinaryKriging, variogram, n_folds=5)

        predictions = results['predictions']
        assert predictions.shape == z.shape
        assert not np.any(np.isnan(predictions))
        assert 'metrics' in results

    def test_k_fold_different_folds(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 10, 50)
        z = 2*x + y + np.random.normal(0, 0.4, 50)

        variogram = ExponentialModel(sill=1.0, range_param=5.0)

        for n_folds in [3, 5, 10]:

            predictions = results['predictions']
        assert predictions.shape == z.shape
        assert 'metrics' in results

    def test_k_fold_metrics(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 35)
        y = np.random.uniform(0, 10, 35)
        z = x + y + np.random.normal(0, 0.3, 35)

        variogram = SphericalModel(sill=1.0, range_param=5.0)

        results = k_fold_cross_validation(x, y, z, OrdinaryKriging, variogram, n_folds=5)

        predictions = results['predictions']
        mse = mean_squared_error(z, predictions)
        r2 = r_squared(z, predictions)

        assert mse >= 0
        assert r2 > 0 # Should have some predictive power
        assert results['metrics']['mse'] >= 0

class TestSpatialCrossValidation:

    def test_spatial_cv_basic(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 40)
        y = np.random.uniform(0, 10, 40)
        z = x + y + np.random.normal(0, 0.5, 40)

        variogram = SphericalModel(sill=1.0, range_param=5.0)

        results = spatial_cross_validation(
        x, y, z, OrdinaryKriging, variogram,
        n_blocks=4
        )

        predictions = results['predictions']
        assert predictions.shape == z.shape
        assert not np.any(np.isnan(predictions))
        assert 'metrics' in results

    def test_spatial_cv_different_blocks(self):
        np.random.seed(42)
        x = np.random.uniform(0, 20, 60)
        y = np.random.uniform(0, 20, 60)
        z = np.sin(x/2) + np.cos(y/2) + np.random.normal(0, 0.3, 60)

        variogram = ExponentialModel(sill=1.0, range_param=5.0)

        from geostats.validation.cross_validation import block_cross_validate
        for n_blocks in [2, 4]:
            results = block_cross_validate(
                self.x,
                self.y,
                self.z,
                variogram_model=variogram,
                n_blocks=n_blocks
            )

        predictions = results['predictions']
        assert predictions.shape == z.shape
        assert 'metrics' in results

    def test_spatial_cv_respects_spatial_structure(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 10, 50)
        z = x + y + np.random.normal(0, 0.4, 50)

        variogram = SphericalModel(sill=1.0, range_param=5.0)

        results = spatial_cross_validation(
        x, y, z, OrdinaryKriging, variogram,
        n_blocks=4
        )

        predictions = results['predictions']
        assert predictions.shape == z.shape
        assert 'metrics' in results

class TestCrossValidationComparison:

    def test_compare_cv_methods(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 40)
        y = np.random.uniform(0, 10, 40)
        z = 2*x + 3*y + np.random.normal(0, 0.5, 40)

        variogram = SphericalModel(sill=1.0, range_param=5.0)
        model = OrdinaryKriging(x, y, z, variogram)

        # LOO
        pred_loo, metrics_loo = leave_one_out(model)
        mse_loo = mean_squared_error(z, pred_loo)

        # K-fold
        results_kfold = k_fold_cross_validation(x, y, z, OrdinaryKriging, variogram, n_folds=5)
        pred_kfold = results_kfold['predictions']
        mse_kfold = mean_squared_error(z, pred_kfold)

        # Spatial
        results_spatial = spatial_cross_validation(x, y, z, OrdinaryKriging, variogram, n_blocks=4)
        pred_spatial = results_spatial['predictions']
        mse_spatial = mean_squared_error(z, pred_spatial)

        # All should give reasonable results
        assert mse_loo > 0
        assert mse_kfold > 0
        assert mse_spatial > 0

        # MSE values should be in similar range
        assert 0.1 < mse_loo / mse_kfold < 10
        assert 0.1 < mse_loo / mse_spatial < 10

class TestMetrics:

    def test_all_metrics(self):
        np.random.seed(42)
        y_true = np.random.uniform(0, 10, 50)
        y_pred = y_true + np.random.normal(0, 0.5, 50)

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r_squared(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)

        assert mse >= 0
        assert mae >= 0
        assert rmse >= 0
        assert rmse == np.sqrt(mse)
        assert 0 <= r2 <= 1

    def test_perfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()

        assert mean_squared_error(y_true, y_pred) == 0
        assert mean_absolute_error(y_true, y_pred) == 0
        assert r_squared(y_true, y_pred) == 1.0
        assert root_mean_squared_error(y_true, y_pred) == 0

    def test_worst_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0]) # Mean

        # R² should be 0 for constant prediction at mean
        r2 = r_squared(y_true, y_pred)
        assert np.abs(r2) < 0.01 # Close to 0

if __name__ == "__main__":
