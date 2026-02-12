"""
Integration tests for complete geostatistical workflows

Tests full end-to-end workflows:
- Data loading → Variogram → Kriging → Validation
- Transform → Variogram → Kriging → Back-transform
- Simulation workflows
- Cross-validation workflows
- Multiple algorithms comparison
"""

import pytest
import numpy as np
from geostats import variogram, kriging
from geostats.algorithms.ordinary_kriging import OrdinaryKriging
from geostats.algorithms.simple_kriging import SimpleKriging
from geostats.algorithms.universal_kriging import UniversalKriging
from geostats.transformations.normal_score import NormalScoreTransform
from geostats.transformations.log_transform import LogTransform
from geostats.validation.cross_validation import leave_one_out, k_fold_cross_validation
from geostats.validation import metrics
from geostats.models.variogram_models import SphericalModel, ExponentialModel
from geostats.utils.data_utils import split_train_test
from geostats.simulation.gaussian_simulation import sequential_gaussian_simulation

class TestBasicKrigingWorkflow:

    def test_full_ok_workflow(self):
        np.random.seed(42)

        # Step 1: Generate/load data
        n = 100
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = np.sin(x / 20) + np.cos(y / 20) + np.random.randn(n) * 0.2

        # Step 2: Compute experimental variogram
        lags, gamma, n_pairs = variogram.experimental_variogram(
        x, y, z, n_lags=15
        )

        assert len(lags) > 0
        assert len(gamma) == len(lags)

        # Step 3: Fit variogram model
        model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

        assert model is not None
        assert model.sill > 0

        # Step 4: Perform kriging
        ok = OrdinaryKriging(x, y, z, variogram_model=model)

        # Step 5: Predict on grid
        x_pred = np.linspace(0, 100, 20)
        y_pred = np.linspace(0, 100, 20)
        X, Y = np.meshgrid(x_pred, y_pred)

        pred, var = ok.predict(X.flatten(), Y.flatten())

        # Step 6: Validate results
        assert len(pred) == 400
        assert all(np.isfinite(pred))
        assert all(var >= 0)

        # Predictions should be in reasonable range
        assert np.mean(pred) > -2
        assert np.mean(pred) < 2

    def test_sk_with_known_mean(self):
        np.random.seed(42)

        # Data with known mean
        n = 80
        true_mean = 10.0
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = true_mean + np.random.randn(n)

        # Compute variogram
        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=12)
        model = variogram.fit_model('exponential', lags, gamma, weights=n_pairs)

        # Simple kriging with known mean
        sk = SimpleKriging(x, y, z, variogram_model=model, mean=true_mean)

        # Predict
        pred, var = sk.predict(np.array([50.0]), np.array([50.0]))

        # Should be close to mean
        assert abs(pred[0] - true_mean) < 2.0
        assert var[0] > 0

    def test_uk_with_trend(self):
        np.random.seed(42)

        # Data with linear trend
        n = 100
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = 0.1 * x + 0.05 * y + 5 + np.random.randn(n) * 0.5

        # Compute variogram on residuals
        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=12)
        model = variogram.fit_model('gaussian', lags, gamma, weights=n_pairs)

        # Universal kriging
        uk = UniversalKriging(
        x, y, z,
        variogram_model=model,
        drift_terms='linear'
        )

        # Predict
        x_test = np.array([50, 75])
        y_test = np.array([50, 75])
        pred, var = uk.predict(x_test, y_test)

        # Expected values based on trend
        expected = 0.1 * x_test + 0.05 * y_test + 5

        # Should be reasonably close
        assert all(np.abs(pred - expected) < 3.0)

class TestTransformationWorkflow:

    def test_normal_score_transform_workflow(self):
        np.random.seed(42)

        # Skewed data
        n = 80
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = np.random.exponential(scale=2.0, size=n)

        # Transform to normal
        ns_transform = NormalScoreTransform()
        z_normal = ns_transform.fit_transform(z)

        # Variogram on transformed data
        lags, gamma, n_pairs = variogram.experimental_variogram(
        x, y, z_normal, n_lags=12
        )
        model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

        # Kriging in transformed space
        ok = OrdinaryKriging(x, y, z_normal, variogram_model=model)
        pred_normal, var_normal = ok.predict(
        np.array([50.0]),
        np.array([50.0])
        )

        # Back-transform
        pred_original = ns_transform.inverse_transform(pred_normal)

        # Should be positive (like original data)
        assert pred_original[0] > 0

    def test_log_transform_workflow(self):
        np.random.seed(42)

        # Lognormal data
        n = 70
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = np.random.lognormal(mean=1, sigma=0.5, size=n)

        # Log transform
        log_transform = LogTransform()
        log_transform.fit(z)
        z_log = log_transform.transform(z)

        # Variogram on log-transformed data
        lags, gamma, n_pairs = variogram.experimental_variogram(
        x, y, z_log, n_lags=10
        )
        model = variogram.fit_model('exponential', lags, gamma, weights=n_pairs)

        # Kriging
        ok = OrdinaryKriging(x, y, z_log, variogram_model=model)
        pred_log, _ = ok.predict(np.array([50.0]), np.array([50.0]))

        # Back-transform
        pred_original = log_transform.inverse_transform(pred_log)

        assert pred_original[0] > 0

class TestCrossValidationWorkflow:

    def test_leave_one_out_cross_validation(self):
        np.random.seed(42)

        # Generate data
        n = 40 # Smaller for LOO CV
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = np.sin(x / 20) + np.random.randn(n) * 0.3

        # Fit variogram
        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=10)
        model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

        # Perform LOO CV
        ok = OrdinaryKriging(x, y, z, variogram_model=model)
        predictions, metrics_dict = leave_one_out(ok)

        results = {
        'predictions': predictions,
        'observations': z,
        'residuals': z - predictions
        }

        # Check results
        assert 'predictions' in results
        assert 'observations' in results
        assert 'residuals' in results

        assert len(results['predictions']) == n
        assert len(results['observations']) == n

        # Calculate metrics
        rmse = np.sqrt(np.mean(results['residuals']**2))
        assert rmse < 1.0 # Should have reasonable accuracy

    def test_k_fold_cross_validation(self):
        np.random.seed(42)

        # Generate data
        n = 60
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = np.random.randn(n) + 10

        # Fit variogram
        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=12)
        model = variogram.fit_model('exponential', lags, gamma, weights=n_pairs)

        # Perform 5-fold CV
        ok = OrdinaryKriging(x, y, z, variogram_model=model)
        predictions, metrics_dict = k_fold_cross_validation(
        ok,
        n_folds=5
        )

        results = {'predictions': predictions}

        # Check results
        assert 'predictions' in results
        assert len(results['predictions']) == n

        # All predictions should be finite
        assert all(np.isfinite(results['predictions']))

class TestSimulationWorkflow:

    def test_sgs_workflow(self):
        np.random.seed(42)

        # Conditioning data
        n = 40
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = np.random.randn(n) * 2 + 10

        # Fit variogram
        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=10)
        model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

        # Simulation grid
        x_sim = np.linspace(0, 100, 15)
        y_sim = np.linspace(0, 100, 15)
        X, Y = np.meshgrid(x_sim, y_sim)

        # Run SGS
        realizations = sequential_gaussian_simulation(
        x_data=x,
        y_data=y,
        z_data=z,
        x_grid=X.flatten(),
        y_grid=Y.flatten(),
        variogram_model=model,
        n_realizations=5,
        seed=42
        )

        assert realizations.shape == (5, 225)
        assert all(np.isfinite(realizations.flatten()))

        # E-type (mean of realizations)
        etype = np.mean(realizations, axis=0)

        # E-type should be smoother than individual realizations
        assert np.std(etype) < np.mean([np.std(r) for r in realizations])

class TestTrainTestWorkflow:

    def test_train_test_split_workflow(self):
        np.random.seed(42)

        # Generate data
        n = 100
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = np.sin(x / 20) + np.cos(y / 20) + np.random.randn(n) * 0.2

        # Split data
        x_train, y_train, z_train, x_test, y_test, z_test = split_train_test(
        x, y, z,
        test_fraction=0.2,
        random_state=42
        )

        assert len(x_train) == 80
        assert len(x_test) == 20

        # Fit variogram on training data
        lags, gamma, n_pairs = variogram.experimental_variogram(
        x_train, y_train, z_train, n_lags=12
        )
        model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

        # Kriging with training data
        ok = OrdinaryKriging(x_train, y_train, z_train, variogram_model=model)

        # Predict on test data
        z_pred, var_pred = ok.predict(x_test, y_test)

        # Evaluate on test set
        rmse = np.sqrt(np.mean((z_test - z_pred)**2))
        mae = np.mean(np.abs(z_test - z_pred))

        assert rmse < 1.0 # Reasonable prediction error
        assert mae < 0.8

class TestMultipleAlgorithmsComparison:

    def test_compare_kriging_methods(self):
        np.random.seed(42)

        # Data with slight trend
        n = 80
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = 0.02 * x + 10 + np.random.randn(n) * 0.5

        # Fit variogram
        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=12)
        model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

        # Prediction points
        x_pred = np.array([25, 50, 75])
        y_pred = np.array([50, 50, 50])

        # Simple Kriging
        sk = SimpleKriging(x, y, z, variogram_model=model, mean=np.mean(z))
        pred_sk, var_sk = sk.predict(x_pred, y_pred)

        # Ordinary Kriging
        ok = OrdinaryKriging(x, y, z, variogram_model=model)
        pred_ok, var_ok = ok.predict(x_pred, y_pred)

        # Universal Kriging
        uk = UniversalKriging(x, y, z, variogram_model=model, drift_terms='linear')
        pred_uk, var_uk = uk.predict(x_pred, y_pred)

        # All should give finite predictions
        assert all(np.isfinite(pred_sk))
        assert all(np.isfinite(pred_ok))
        assert all(np.isfinite(pred_uk))

        # Predictions should be similar but not identical
        assert not np.allclose(pred_sk, pred_ok)
        assert not np.allclose(pred_ok, pred_uk)

    def test_compare_variogram_models(self):
        np.random.seed(42)

        n = 80
        x = np.random.uniform(0, 100, n)
        y = np.random.uniform(0, 100, n)
        z = np.random.randn(n)

        # Fit multiple models
        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=12)

        model_spherical = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)
        model_exponential = variogram.fit_model('exponential', lags, gamma, weights=n_pairs)
        model_gaussian = variogram.fit_model('gaussian', lags, gamma, weights=n_pairs)

        # All should be valid
        assert model_spherical.sill > 0
        assert model_exponential.sill > 0
        assert model_gaussian.sill > 0

        # Compare predictions
        ok_sph = OrdinaryKriging(x, y, z, variogram_model=model_spherical)
        ok_exp = OrdinaryKriging(x, y, z, variogram_model=model_exponential)
        ok_gau = OrdinaryKriging(x, y, z, variogram_model=model_gaussian)

        x_test = np.array([50.0])
        y_test = np.array([50.0])

        pred_sph, _ = ok_sph.predict(x_test, y_test)
        pred_exp, _ = ok_exp.predict(x_test, y_test)
        pred_gau, _ = ok_gau.predict(x_test, y_test)

        # Predictions should be similar but not identical
        assert abs(pred_sph[0] - pred_exp[0]) < 2.0
        assert abs(pred_exp[0] - pred_gau[0]) < 2.0

class TestRobustnessAndEdgeCases:

    def test_workflow_with_few_points(self):
        x = np.array([0, 50, 100])
        y = np.array([0, 50, 100])
        z = np.array([1, 2, 3])

        # Variogram with few points
        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=2)

        # Should still work
        assert len(lags) > 0

        # Fit model
        model = SphericalModel(nugget=0.1, sill=1.0, range_param=50.0)

        # Kriging
        ok = OrdinaryKriging(x, y, z, variogram_model=model)
        pred, var = ok.predict(np.array([25.0]), np.array([25.0]))

        assert np.isfinite(pred[0])

    def test_workflow_with_constant_values(self):
        np.random.seed(42)
        x = np.random.uniform(0, 100, 30)
        y = np.random.uniform(0, 100, 30)
        z = np.ones(30) * 10.0 # All same value

        # Variogram should be flat
        lags, gamma, n_pairs = variogram.experimental_variogram(x, y, z, n_lags=10)

        # Gamma should be near zero
        assert np.mean(gamma) < 0.5

        # Kriging should return constant
        model = SphericalModel(nugget=0.0, sill=0.01, range_param=30.0)
        ok = OrdinaryKriging(x, y, z, variogram_model=model)
        pred, var = ok.predict(np.array([50.0]), np.array([50.0]))

        # Should be close to constant value
        assert abs(pred[0] - 10.0) < 1.0

if __name__ == "__main__":
