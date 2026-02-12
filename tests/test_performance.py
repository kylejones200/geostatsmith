"""
Tests for Performance Optimization Module

Tests:
- Parallel kriging
- Caching
- Chunked processing
- Approximate methods
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from geostats.performance.parallel import (
    parallel_kriging,
    parallel_cross_validation,
)
from geostats.performance.caching import (
    CachedKriging,
    clear_cache,
)
from geostats.performance.chunked import (
    ChunkedKriging,
    chunked_predict,
)
from geostats.performance.approximate import (
    approximate_kriging,
    coarse_to_fine,
)
from geostats.algorithms.variogram import experimental_variogram
from geostats.algorithms.fitting import fit_variogram_model
from geostats.models.variogram_models import SphericalModel
from geostats.algorithms.ordinary_kriging import OrdinaryKriging


class TestParallelKriging:
    """Tests for parallel kriging"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 50
        self.x = np.random.uniform(0, 100, self.n_samples)
        self.y = np.random.uniform(0, 100, self.n_samples)
        self.z = (
            50 + 0.3 * self.x + 0.2 * self.y + np.random.normal(0, 3, self.n_samples)
        )

        # Fit variogram
        lags, gamma, n_pairs = experimental_variogram(self.x, self.y, self.z, n_lags=10)
        self.model = fit_variogram_model(SphericalModel(), lags, gamma, weights=n_pairs)

        # Prediction points
        self.n_pred = 200
        self.x_pred = np.random.uniform(0, 100, self.n_pred)
        self.y_pred = np.random.uniform(0, 100, self.n_pred)

    def test_parallel_kriging_basic(self):
        """Test basic parallel kriging"""
        pred, var = parallel_kriging(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            n_jobs=2,
            return_variance=True,
        )

        assert len(pred) == self.n_pred
        assert len(var) == self.n_pred
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(var))
        assert np.all(var >= 0)

    def test_parallel_kriging_single_job(self):
        """Test parallel kriging with single job"""
        pred, var = parallel_kriging(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            n_jobs=1,
            return_variance=True,
        )

        assert len(pred) == self.n_pred
        assert len(var) == self.n_pred

    def test_parallel_kriging_all_cores(self):
        """Test parallel kriging with all cores"""
        pred, var = parallel_kriging(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            n_jobs=-1,
            return_variance=True,
        )

        assert len(pred) == self.n_pred
        assert len(var) == self.n_pred

    def test_parallel_kriging_without_variance(self):
        """Test parallel kriging without variance"""
        pred, var = parallel_kriging(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            return_variance=False,
        )

        assert len(pred) == self.n_pred
        assert var is None

    def test_parallel_kriging_batch_size(self):
        """Test parallel kriging with different batch sizes"""
        pred1, var1 = parallel_kriging(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            batch_size=50,
            n_jobs=2,
        )

        pred2, var2 = parallel_kriging(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            batch_size=100,
            n_jobs=2,
        )

        # Results should be similar (within numerical precision)
        np.testing.assert_allclose(pred1, pred2, rtol=1e-10)
        np.testing.assert_allclose(var1, var2, rtol=1e-10)

    def test_parallel_kriging_consistency(self):
        """Test that parallel kriging matches sequential"""
        # Sequential (single job)
        pred_seq, var_seq = parallel_kriging(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            n_jobs=1,
        )

        # Parallel
        pred_par, var_par = parallel_kriging(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            n_jobs=2,
        )

        # Should be identical
        np.testing.assert_allclose(pred_seq, pred_par, rtol=1e-10)
        np.testing.assert_allclose(var_seq, var_par, rtol=1e-10)

    def test_parallel_cross_validation_loo(self):
        """Test parallel leave-one-out cross-validation"""
        results = parallel_cross_validation(
            self.x,
            self.y,
            self.z,
            variogram_model=self.model,
            method="leave_one_out",
            n_jobs=2,
        )

        assert "predictions" in results
        assert "variances" in results
        assert "errors" in results
        assert "rmse" in results
        assert "mae" in results
        assert "r2" in results
        assert "observed" in results

        assert len(results["predictions"]) == self.n_samples
        assert len(results["variances"]) == self.n_samples
        assert np.isfinite(results["rmse"])
        assert np.isfinite(results["r2"])

    def test_parallel_cross_validation_kfold(self):
        """Test parallel k-fold cross-validation"""
        results = parallel_cross_validation(
            self.x,
            self.y,
            self.z,
            variogram_model=self.model,
            method="k_fold",
            n_folds=5,
            n_jobs=2,
        )

        assert "predictions" in results
        assert "rmse" in results
        assert len(results["predictions"]) == self.n_samples
        assert np.isfinite(results["rmse"])

    def test_parallel_cross_validation_invalid_method(self):
        """Test that invalid method raises error"""
        with pytest.raises(ValueError, match="Unknown method"):
        with pytest.raises(ValueError, match="Unknown method"):
                self.x,
                self.y,
                self.z,
                variogram_model=self.model,
                method="invalid_method",
            )


class TestCaching:
    """Tests for caching functionality"""

    def setup_method(self):
        """Set up test data and temporary cache directory"""
        np.random.seed(42)
        self.n_samples = 40
        self.x = np.random.uniform(0, 100, self.n_samples)
        self.y = np.random.uniform(0, 100, self.n_samples)
        self.z = 50 + 0.3 * self.x + np.random.normal(0, 3, self.n_samples)

        # Fit variogram
        lags, gamma, n_pairs = experimental_variogram(self.x, self.y, self.z, n_lags=10)
        self.model = fit_variogram_model(SphericalModel(), lags, gamma, weights=n_pairs)

        # Create temporary cache directory
        self.temp_cache = Path(tempfile.mkdtemp())

        # Prediction points
        self.x_pred = np.array([50.0, 60.0, 70.0])
        self.y_pred = np.array([50.0, 60.0, 70.0])

    def teardown_method(self):
        """Clean up temporary cache directory"""
        if self.temp_cache.exists():
        if self.temp_cache.exists():

    def test_cached_kriging_initialization(self):
        """Test CachedKriging initialization"""
        cached = CachedKriging(
            self.x,
            self.y,
            self.z,
            variogram_model=self.model,
            cache_dir=self.temp_cache,
        )

        assert cached.krig is not None
        assert cached.cache_dir == self.temp_cache
        assert hasattr(cached, "data_hash")

    def test_cached_kriging_first_call(self):
        """Test that first call computes and caches"""
        cached = CachedKriging(
            self.x,
            self.y,
            self.z,
            variogram_model=self.model,
            cache_dir=self.temp_cache,
        )

        # First call should compute
        pred1, var1 = cached.predict(self.x_pred, self.y_pred, use_cache=True)

        assert len(pred1) == len(self.x_pred)
        assert len(var1) == len(self.x_pred)
        assert np.all(np.isfinite(pred1))

        # Cache file should exist
        pred_hash = cached._compute_pred_hash(self.x_pred, self.y_pred)
        cache_path = cached._get_cache_path(pred_hash)
        assert cache_path.exists()

    def test_cached_kriging_second_call(self):
        """Test that second call uses cache"""
        cached = CachedKriging(
            self.x,
            self.y,
            self.z,
            variogram_model=self.model,
            cache_dir=self.temp_cache,
        )

        # First call
        pred1, var1 = cached.predict(self.x_pred, self.y_pred, use_cache=True)

        # Second call should use cache
        pred2, var2 = cached.predict(self.x_pred, self.y_pred, use_cache=True)

        # Should be identical
        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(var1, var2)

    def test_cached_kriging_without_cache(self):
        """Test that use_cache=False bypasses cache"""
        cached = CachedKriging(
            self.x,
            self.y,
            self.z,
            variogram_model=self.model,
            cache_dir=self.temp_cache,
        )

        # First call with cache
        pred1, var1 = cached.predict(self.x_pred, self.y_pred, use_cache=True)

        # Second call without cache
        pred2, var2 = cached.predict(self.x_pred, self.y_pred, use_cache=False)

        # Should still be identical (same computation)
        np.testing.assert_allclose(pred1, pred2, rtol=1e-10)

    def test_cached_kriging_different_locations(self):
        """Test that different locations create different cache entries"""
        cached = CachedKriging(
            self.x,
            self.y,
            self.z,
            variogram_model=self.model,
            cache_dir=self.temp_cache,
        )

        x_pred1 = np.array([50.0, 60.0])
        y_pred1 = np.array([50.0, 60.0])
        x_pred2 = np.array([70.0, 80.0])
        y_pred2 = np.array([70.0, 80.0])

        pred1, _ = cached.predict(x_pred1, y_pred1, use_cache=True)
        pred2, _ = cached.predict(x_pred2, y_pred2, use_cache=True)

        # Different locations should give different predictions
        assert not np.array_equal(pred1, pred2)

        # Both cache files should exist
        hash1 = cached._compute_pred_hash(x_pred1, y_pred1)
        hash2 = cached._compute_pred_hash(x_pred2, y_pred2)
        assert cached._get_cache_path(hash1).exists()
        assert cached._get_cache_path(hash2).exists()

    def test_clear_cache(self):
        """Test clearing cache"""
        cached = CachedKriging(
            self.x,
            self.y,
            self.z,
            variogram_model=self.model,
            cache_dir=self.temp_cache,
        )

        # Create some cache entries
        cached.predict(self.x_pred, self.y_pred, use_cache=True)
        cached.predict(self.x_pred[:2], self.y_pred[:2], use_cache=True)

        # Clear cache
        n_deleted = clear_cache(cache_dir=self.temp_cache)

        assert n_deleted >= 2
        assert len(list(self.temp_cache.glob("*.pkl"))) == 0

    def test_clear_cache_empty(self):
        """Test clearing empty cache"""
        n_deleted = clear_cache(cache_dir=self.temp_cache)
        assert n_deleted == 0


class TestChunkedProcessing:
    """Tests for chunked processing"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 50
        self.x = np.random.uniform(0, 100, self.n_samples)
        self.y = np.random.uniform(0, 100, self.n_samples)
        self.z = 50 + 0.3 * self.x + np.random.normal(0, 3, self.n_samples)

        # Fit variogram
        lags, gamma, n_pairs = experimental_variogram(self.x, self.y, self.z, n_lags=10)
        self.model = fit_variogram_model(SphericalModel(), lags, gamma, weights=n_pairs)

        # Large prediction grid
        self.x_grid = np.linspace(0, 100, 100)
        self.y_grid = np.linspace(0, 100, 100)

    def test_chunked_kriging_initialization(self):
        """Test ChunkedKriging initialization"""
        chunked = ChunkedKriging(self.x, self.y, self.z, variogram_model=self.model)

        assert chunked.krig is not None
        assert chunked.variogram_model is not None

    def test_predict_chunked(self):
        """Test chunked prediction"""
        chunked = ChunkedKriging(self.x, self.y, self.z, variogram_model=self.model)

        x_2d, y_2d = np.meshgrid(self.x_grid, self.y_grid)
        x_flat = x_2d.ravel()
        y_flat = y_2d.ravel()

        pred, var = chunked.predict_chunked(
            x_flat, y_flat, chunk_size=1000, return_variance=True, verbose=False
        )

        assert len(pred) == len(x_flat)
        assert len(var) == len(x_flat)
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(var))

    def test_predict_chunked_without_variance(self):
        """Test chunked prediction without variance"""
        chunked = ChunkedKriging(self.x, self.y, self.z, variogram_model=self.model)

        x_2d, y_2d = np.meshgrid(self.x_grid, self.y_grid)
        x_flat = x_2d.ravel()
        y_flat = y_2d.ravel()

        pred, var = chunked.predict_chunked(
            x_flat, y_flat, chunk_size=1000, return_variance=False, verbose=False
        )

        assert len(pred) == len(x_flat)
        assert var is None

    def test_predict_large_grid(self):
        """Test prediction on large grid"""
        chunked = ChunkedKriging(self.x, self.y, self.z, variogram_model=self.model)

        z_grid, var_grid = chunked.predict_large_grid(
            self.x_grid,
            self.y_grid,
            chunk_size=2000,
            return_variance=True,
            verbose=False,
        )

        assert z_grid.shape == (len(self.y_grid), len(self.x_grid))
        assert var_grid.shape == (len(self.y_grid), len(self.x_grid))
        assert np.all(np.isfinite(z_grid))
        assert np.all(np.isfinite(var_grid))

    def test_predict_large_grid_without_variance(self):
        """Test large grid prediction without variance"""
        chunked = ChunkedKriging(self.x, self.y, self.z, variogram_model=self.model)

        z_grid, var_grid = chunked.predict_large_grid(
            self.x_grid, self.y_grid, return_variance=False, verbose=False
        )

        assert z_grid.shape == (len(self.y_grid), len(self.x_grid))
        assert var_grid is None

    def test_chunked_predict_function(self):
        """Test convenience function for chunked prediction"""
        x_2d, y_2d = np.meshgrid(self.x_grid, self.y_grid)
        x_flat = x_2d.ravel()[:500]  # Smaller for speed
        y_flat = y_2d.ravel()[:500]

        pred, var = chunked_predict(
            self.x,
            self.y,
            self.z,
            x_flat,
            y_flat,
            variogram_model=self.model,
            chunk_size=100,
            return_variance=True,
        )

        assert len(pred) == len(x_flat)
        assert len(var) == len(x_flat)
        assert np.all(np.isfinite(pred))

    def test_chunked_consistency(self):
        """Test that chunked results match non-chunked"""
        chunked = ChunkedKriging(self.x, self.y, self.z, variogram_model=self.model)

        # Small grid for comparison
        x_small = np.linspace(0, 100, 20)
        y_small = np.linspace(0, 100, 20)
        x_2d, y_2d = np.meshgrid(x_small, y_small)
        x_flat = x_2d.ravel()
        y_flat = y_2d.ravel()

        # Chunked
        pred_chunked, var_chunked = chunked.predict_chunked(
            x_flat, y_flat, chunk_size=50, return_variance=True, verbose=False
        )

        # Non-chunked (direct)
        pred_direct, var_direct = chunked.krig.predict(
            x_flat, y_flat, return_variance=True
        )

        # Should be identical
        np.testing.assert_allclose(pred_chunked, pred_direct, rtol=1e-10)
        np.testing.assert_allclose(var_chunked, var_direct, rtol=1e-10)

    def test_different_chunk_sizes(self):
        """Test that different chunk sizes give same results"""
        chunked = ChunkedKriging(self.x, self.y, self.z, variogram_model=self.model)

        x_small = np.linspace(0, 100, 30)
        y_small = np.linspace(0, 100, 30)
        x_2d, y_2d = np.meshgrid(x_small, y_small)
        x_flat = x_2d.ravel()
        y_flat = y_2d.ravel()

        pred1, var1 = chunked.predict_chunked(
            x_flat, y_flat, chunk_size=100, verbose=False
        )

        pred2, var2 = chunked.predict_chunked(
            x_flat, y_flat, chunk_size=200, verbose=False
        )

        # Should be identical
        np.testing.assert_allclose(pred1, pred2, rtol=1e-10)
        np.testing.assert_allclose(var1, var2, rtol=1e-10)


class TestApproximateKriging:
    """Tests for approximate kriging methods"""

    def setup_method(self):
        """Set up test data"""
        np.random.seed(42)
        self.n_samples = 100
        self.x = np.random.uniform(0, 100, self.n_samples)
        self.y = np.random.uniform(0, 100, self.n_samples)
        self.z = (
            50 + 0.3 * self.x + 0.2 * self.y + np.random.normal(0, 3, self.n_samples)
        )

        # Fit variogram
        lags, gamma, n_pairs = experimental_variogram(self.x, self.y, self.z, n_lags=10)
        self.model = fit_variogram_model(SphericalModel(), lags, gamma, weights=n_pairs)

        # Prediction points
        self.n_pred = 50
        self.x_pred = np.random.uniform(0, 100, self.n_pred)
        self.y_pred = np.random.uniform(0, 100, self.n_pred)

    def test_approximate_kriging_basic(self):
        """Test basic approximate kriging"""
        pred, var = approximate_kriging(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            max_neighbors=30,
        )

        assert len(pred) == self.n_pred
        assert len(var) == self.n_pred
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(var))
        assert np.all(var >= 0)

    def test_approximate_kriging_different_neighbors(self):
        """Test with different numbers of neighbors"""
        pred1, var1 = approximate_kriging(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            max_neighbors=20,
        )

        pred2, var2 = approximate_kriging(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            max_neighbors=50,
        )

        # Both should be valid
        assert np.all(np.isfinite(pred1))
        assert np.all(np.isfinite(pred2))
        # More neighbors should generally give better results
        # (but not always, so just check they're different)
        assert not np.array_equal(pred1, pred2)

    def test_approximate_kriging_with_radius(self):
        """Test approximate kriging with search radius"""
        pred, var = approximate_kriging(
            self.x,
            self.y,
            self.z,
            self.x_pred,
            self.y_pred,
            variogram_model=self.model,
            max_neighbors=30,
            search_radius=20.0,
        )

        assert len(pred) == self.n_pred
        assert len(var) == self.n_pred
        assert np.all(np.isfinite(pred))

    def test_approximate_vs_exact(self):
        """Test that approximate kriging is close to exact"""
        # Exact kriging
        ok = OrdinaryKriging(self.x, self.y, self.z, variogram_model=self.model)
        pred_exact, var_exact = ok.predict(self.x_pred[:10], self.y_pred[:10])

        # Approximate kriging (with many neighbors, should be close)
        pred_approx, var_approx = approximate_kriging(
            self.x,
            self.y,
            self.z,
            self.x_pred[:10],
            self.y_pred[:10],
            variogram_model=self.model,
            max_neighbors=80,  # Use most neighbors
        )

        # Should be reasonably close (within 5% for most points)
        diff = np.abs(pred_exact - pred_approx)
        rel_diff = diff / (np.abs(pred_exact) + 1e-10)
        assert np.mean(rel_diff) < 0.1  # Mean relative difference < 10%

    def test_approximate_kriging_no_neighbors(self):
        """Test handling when no neighbors found"""
        # Predict far from all samples
        x_far = np.array([1000.0, 2000.0])
        y_far = np.array([1000.0, 2000.0])

        pred, var = approximate_kriging(
            self.x,
            self.y,
            self.z,
            x_far,
            y_far,
            variogram_model=self.model,
            max_neighbors=10,
            search_radius=5.0,  # Very small radius
        )

        # Should handle gracefully (may have NaN or inf)
        assert len(pred) == 2
        assert len(var) == 2

    def test_coarse_to_fine(self):
        """Test coarse-to-fine kriging"""
        x_grid = np.linspace(0, 100, 50)
        y_grid = np.linspace(0, 100, 50)

        pred, var = coarse_to_fine(
            self.x,
            self.y,
            self.z,
            x_grid,
            y_grid,
            variogram_model=self.model,
            coarse_factor=4,
        )

        assert pred.shape == (len(y_grid), len(x_grid))
        assert var.shape == (len(y_grid), len(x_grid))
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(var))

    def test_coarse_to_fine_different_factors(self):
        """Test coarse-to-fine with different factors"""
        x_grid = np.linspace(0, 100, 40)
        y_grid = np.linspace(0, 100, 40)

        pred1, var1 = coarse_to_fine(
            self.x,
            self.y,
            self.z,
            x_grid,
            y_grid,
            variogram_model=self.model,
            coarse_factor=2,
        )

        pred2, var2 = coarse_to_fine(
            self.x,
            self.y,
            self.z,
            x_grid,
            y_grid,
            variogram_model=self.model,
            coarse_factor=4,
        )

        # Both should be valid
        assert np.all(np.isfinite(pred1))
        assert np.all(np.isfinite(pred2))
        assert pred1.shape == pred2.shape
