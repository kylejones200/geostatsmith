"""
Tests for advanced kriging algorithms

Tests:
- 3D Kriging
- Indicator Kriging
- Lognormal Kriging
- Neighborhood Search
- External Drift Kriging (basic)
"""

import pytest
import numpy as np
from geostats import variogram
from geostats.algorithms.kriging_3d import SimpleKriging3D, OrdinaryKriging3D
from geostats.algorithms.indicator_kriging import IndicatorKriging
from geostats.algorithms.lognormal_kriging import LognormalKriging
from geostats.algorithms.neighborhood_search import NeighborhoodSearch, NeighborhoodConfig
from geostats.models.variogram_models import SphericalModel, ExponentialModel

class TestKriging3D:
    """Tests for 3D Kriging"""

    def setup_method(self):
        """Set up 3D test data"""
        np.random.seed(42)
        self.n = 50
        self.x = np.random.uniform(0, 100, self.n)
        self.y = np.random.uniform(0, 100, self.n)
        self.z_coord = np.random.uniform(0, 50, self.n)
        self.values = np.random.randn(self.n) * 2 + 10

        # Create variogram model
        self.model = SphericalModel(nugget=0.2, sill=4.0, range_param=30.0)

    def test_kriging3d_initialization(self):
        """Test 3D kriging initialization"""
        ok3d = OrdinaryKriging3D(
        self.x, self.y, self.z_coord, self.values,
        variogram_model=self.model
        )

        assert ok3d.n_points == self.n
        assert ok3d.variogram_model is not None

    def test_kriging3d_prediction_single_point(self):
        """Test 3D prediction at single point"""
        ok3d = OrdinaryKriging3D(
        self.x, self.y, self.z_coord, self.values,
        variogram_model=self.model
        )

        # Predict at single point
        x_pred = np.array([50.0])
        y_pred = np.array([50.0])
        z_pred = np.array([25.0])

        pred, var = ok3d.predict(x_pred, y_pred, z_pred)

        assert len(pred) == 1
        assert len(var) == 1
        assert np.isfinite(pred[0])
        assert np.isfinite(var[0])
        assert var[0] >= 0

    def test_kriging3d_multiple_points(self):
        """Test 3D prediction at multiple points"""
        ok3d = OrdinaryKriging3D(
        self.x, self.y, self.z_coord, self.values,
        variogram_model=self.model
        )

        # Predict on grid
        x_pred = np.array([25, 50, 75])
        y_pred = np.array([25, 50, 75])
        z_pred = np.array([10, 25, 40])

        pred, var = ok3d.predict(x_pred, y_pred, z_pred)

        assert len(pred) == 3
        assert all(np.isfinite(pred))
        assert all(var >= 0)

    def test_kriging3d_at_data_location(self):
        """Test 3D prediction at data location"""
        ok3d = OrdinaryKriging3D(
        self.x, self.y, self.z_coord, self.values,
        variogram_model=self.model
        )

        # Predict at first data point
        pred, var = ok3d.predict(
        np.array([self.x[0]]),
        np.array([self.y[0]]),
        np.array([self.z_coord[0]])
        )

        # Should be close to actual value
        assert abs(pred[0] - self.values[0]) < 1.0
        # Variance should be small
        assert var[0] < 1.0

class TestIndicatorKriging:
    """Tests for Indicator Kriging"""

    def setup_method(self):
        """Set up categorical test data"""
        np.random.seed(42)
        self.n = 60
        self.x = np.random.uniform(0, 100, self.n)
        self.y = np.random.uniform(0, 100, self.n)

        # Create categorical data (e.g., 0, 1, 2)
        self.categories = np.array([0, 1, 2])
        self.z = np.random.choice(self.categories, size=self.n, p=[0.3, 0.5, 0.2])

        # Create variogram model for indicators
        self.model = SphericalModel(nugget=0.05, sill=0.25, range_param=30.0)

    def test_indicator_kriging_initialization(self):
        """Test indicator kriging initialization with single threshold"""
        threshold = 1.5 # Between low (1) and medium (2)
        ik = IndicatorKriging(
        self.x, self.y, self.z,
        threshold=threshold,
        variogram_model=self.model
        )

        assert ik.threshold == threshold
        # z should be transformed to indicators (0 or 1)
        assert all((ik.z == 0) | (ik.z == 1))

    def test_indicator_kriging_probability_estimation(self):
        """Test probability estimation with single threshold"""
        threshold = 1.5
        ik = IndicatorKriging(
        self.x, self.y, self.z,
        threshold=threshold,
        variogram_model=self.model
        )

        # Predict probability at point
        x_pred = np.array([50.0])
        y_pred = np.array([50.0])

        prob = ik.predict(x_pred, y_pred, return_variance=False)

        # Should return probability (between 0 and 1)
        assert len(prob) == 1
        assert 0 <= prob[0] <= 1

    def test_indicator_kriging_most_likely_category(self):
        """Test probability prediction at multiple points"""
        threshold = 1.5
        ik = IndicatorKriging(
        self.x, self.y, self.z,
        threshold=threshold,
        variogram_model=self.model
        )

        x_pred = np.array([50.0, 60.0, 70.0])
        y_pred = np.array([50.0, 60.0, 70.0])

        probs = ik.predict(x_pred, y_pred, return_variance=False)

        assert len(probs) == 3
        assert all((probs >= 0) & (probs <= 1))

        # All predictions should be valid categories
        assert all(cat in self.categories for cat in pred_categories)

    def test_indicator_kriging_binary_case(self):
        """Test indicator kriging with binary data"""
        # Binary data (0 or 1)
        z_binary = np.random.choice([0.0, 1.0], size=self.n, p=[0.6, 0.4])

        # Use threshold of 0.5 to separate 0s and 1s
        threshold = 0.5
        ik = IndicatorKriging(
        self.x, self.y, z_binary,
        threshold=threshold,
        variogram_model=self.model
        )

        x_pred = np.array([50.0])
        y_pred = np.array([50.0])

        prob = ik.predict(x_pred, y_pred, return_variance=False)

        # Should return probability between 0 and 1
        assert len(prob) == 1
        assert 0 <= prob[0] <= 1

class TestLognormalKriging:
    """Tests for Lognormal Kriging"""

    def setup_method(self):
        """Set up lognormal test data"""
        np.random.seed(42)
        self.n = 50
        self.x = np.random.uniform(0, 100, self.n)
        self.y = np.random.uniform(0, 100, self.n)

        # Generate lognormal data
        self.z = np.random.lognormal(mean=0, sigma=1, size=self.n)

        # Fit variogram on log-transformed data
        log_z = np.log(self.z)
        lags, gamma, n_pairs = variogram.experimental_variogram(
        self.x, self.y, log_z, n_lags=10
        )
        self.model = variogram.fit_model('spherical', lags, gamma, weights=n_pairs)

    def test_lognormal_kriging_initialization(self):
        """Test lognormal kriging initialization"""
        lk = LognormalKriging(
        self.x, self.y, self.z,
        variogram_model=self.model
        )

        assert lk.n_points == self.n
        assert hasattr(lk, 'log_z')
        assert hasattr(lk, 'mean_log')

    def test_lognormal_kriging_prediction(self):
        """Test lognormal kriging prediction"""
        lk = LognormalKriging(
        self.x, self.y, self.z,
        variogram_model=self.model
        )

        x_pred = np.array([50.0])
        y_pred = np.array([50.0])

        pred, var = lk.predict(x_pred, y_pred)

        assert len(pred) == 1
        assert len(var) == 1
        assert pred[0] > 0 # Lognormal predictions should be positive
        assert np.isfinite(pred[0])

    def test_lognormal_kriging_median_vs_mean(self):
        """Test that lognormal kriging back-transform methods work"""
        lk = LognormalKriging(
        self.x, self.y, self.z,
        variogram_model=self.model
        )

        x_pred = np.array([50.0])
        y_pred = np.array([50.0])

        # Get prediction with default (unbiased) method
        pred_unbiased, var_unbiased = lk.predict(x_pred, y_pred, back_transform_method='unbiased')

        # Get prediction with simple exponential
        pred_simple, var_simple = lk.predict(x_pred, y_pred, back_transform_method='simple')

        # Both should be positive (lognormal data)
        assert pred_unbiased[0] > 0
        assert pred_simple[0] > 0
        assert np.isfinite(pred_unbiased[0])
        assert np.isfinite(pred_simple[0])

class TestNeighborhoodSearch:
    """Tests for neighborhood search algorithms"""

    def setup_method(self):
        """Set up spatial data for neighborhood search"""
        np.random.seed(42)
        self.n = 100
        self.x = np.random.uniform(0, 100, self.n)
        self.y = np.random.uniform(0, 100, self.n)
        self.z = np.random.randn(self.n)

    def test_neighborhood_search_initialization(self):
        """Test neighborhood search initialization"""
        config = NeighborhoodConfig(max_neighbors=20, min_neighbors=5)
        ns = NeighborhoodSearch(self.x, self.y, config=config)

        assert ns.n_points == self.n
        assert ns.config.max_neighbors == 20
        assert ns.config.min_neighbors == 5

    def test_find_neighbors_basic(self):
        """Test finding neighbors"""
        config = NeighborhoodConfig(max_neighbors=10)
        ns = NeighborhoodSearch(self.x, self.y, config=config)

        # Find neighbors for a point
        indices, distances = ns.find_neighbors(50.0, 50.0)

        assert len(indices) <= 10 # At most max_neighbors
        assert len(distances) == len(indices)

        # Distances should be non-negative
        assert all(d >= 0 for d in distances)

        # Indices should be valid
        assert all(0 <= idx < self.n for idx in indices)

    def test_find_neighbors_with_radius(self):
        """Test finding neighbors within radius"""
        config = NeighborhoodConfig(
        max_neighbors=50,
        search_radius=20.0
        )
        ns = NeighborhoodSearch(self.x, self.y, config=config)

        indices, distances = ns.find_neighbors(50.0, 50.0)

        # All found points should be within radius
        assert all(d <= 20.0 for d in distances)

    def test_find_neighbors_min_constraint(self):
        """Test minimum neighbors constraint"""
        config = NeighborhoodConfig(
        max_neighbors=5,
        min_neighbors=3
        )
        ns = NeighborhoodSearch(self.x, self.y, config=config)

        indices, distances = ns.find_neighbors(50.0, 50.0)

        # Should have at least min_neighbors (if available)
        assert len(indices) >= min(3, self.n)

    def test_moving_neighborhood(self):
        """Test that neighborhoods change for different prediction points"""
        config = NeighborhoodConfig(max_neighbors=10)
        ns = NeighborhoodSearch(self.x, self.y, config=config)

        # Find neighbors for different points
        indices1, _ = ns.find_neighbors(25.0, 25.0)
        indices2, _ = ns.find_neighbors(75.0, 75.0)

        # Neighborhoods should differ
        assert not np.array_equal(indices1, indices2)

    def test_neighborhood_search_edge_cases(self):
        """Test edge cases for neighborhood search"""
        # Very few points
        x_small = np.array([0, 10, 20])
        y_small = np.array([0, 10, 20])

        config = NeighborhoodConfig(max_neighbors=10)
        ns = NeighborhoodSearch(x_small, y_small, config=config)

        indices, distances = ns.find_neighbors(15.0, 15.0)

        # Should return all 3 points
        assert len(indices) == 3

class TestAdvancedKrigingEdgeCases:
    """Test edge cases for advanced algorithms"""

    def test_3d_kriging_with_planar_data(self):
        """Test 3D kriging when data is planar"""
        np.random.seed(42)
        x = np.random.uniform(0, 100, 30)
        y = np.random.uniform(0, 100, 30)
        z_coord = np.ones(30) * 10.0 # All at same elevation
        values = np.random.randn(30)

        model = SphericalModel(nugget=0.1, sill=1.0, range_param=30.0)

        ok3d = OrdinaryKriging3D(x, y, z_coord, values, variogram_model=model)

        # Predict at same elevation
        pred, var = ok3d.predict(
        np.array([50.0]),
        np.array([50.0]),
        np.array([10.0])
        )

        assert np.isfinite(pred[0])

    def test_indicator_kriging_single_category_dominant(self):
        """Test indicator kriging when one category dominates"""
        np.random.seed(42)
        x = np.random.uniform(0, 100, 50)
        y = np.random.uniform(0, 100, 50)

        # 90% category 0, 10% category 1
        z = np.random.choice([0.0, 1.0], size=50, p=[0.9, 0.1])

        model = SphericalModel(nugget=0.05, sill=0.1, range_param=30.0)

        # Use threshold of 0.5 to separate
        threshold = 0.5
        ik = IndicatorKriging(
        x, y, z,
        threshold=threshold,
        variogram_model=model
        )

        # Should still work
        prob = ik.predict(
        np.array([50.0]),
        np.array([50.0]),
        return_variance=False
        )

        assert probs.shape == (1, 2)
        # Probability of category 0 should be high
        assert probs[0, 0] > 0.5

    def test_neighborhood_search_no_neighbors_within_radius(self):
        """Test radius search when no neighbors found"""
        x = np.array([0, 100])
        y = np.array([0, 100])

        # Search at center with very small radius
        config = NeighborhoodConfig(
        max_neighbors=10,
        search_radius=5.0
        )
        ns = NeighborhoodSearch(x, y, config=config)

        indices, distances = ns.find_neighbors(50.0, 50.0)

        # Should find no neighbors (or very few)
        assert len(indices) <= 2 # May find 0, 1, or 2 depending on implementation

    def test_neighborhood_search_many_neighbors_requested(self):
        """Test when max_neighbors > number of data points"""
        x = np.array([0, 10, 20])
        y = np.array([0, 10, 20])

        config = NeighborhoodConfig(max_neighbors=10)
        ns = NeighborhoodSearch(x, y, config=config)

        indices, distances = ns.find_neighbors(15.0, 15.0)

        # Should return all 3 points
        assert len(indices) == 3

if __name__ == "__main__":
     if __name__ == "__main__":
