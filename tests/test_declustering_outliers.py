"""
Tests for declustering and outlier detection
- Declustering: Currently at 10% coverage, targeting 40%+
- Outlier detection: Currently at 17% coverage, targeting 40%+
"""

import pytest
import numpy as np
from geostats.transformations.declustering import (
 cell_declustering,
 polygonal_declustering,
 detect_clustering
)
from geostats.utils.outliers import (
 detect_outliers_iqr,
 detect_outliers_zscore,
 detect_outliers_modified_zscore,
 detect_spatial_outliers,
 detect_outliers_ensemble
)

class TestCellDeclustering:

    def test_cell_declustering_basic(self):
        np.random.seed(42)
        # Create clustered data
        cluster1_x = np.random.normal(2, 0.5, 30)
        cluster1_y = np.random.normal(2, 0.5, 30)
        cluster2_x = np.random.normal(8, 0.5, 10)
        cluster2_y = np.random.normal(8, 0.5, 10)

        x = np.concatenate([cluster1_x, cluster2_x])
        y = np.concatenate([cluster1_y, cluster2_y])
        z = np.concatenate([np.ones(30) * 5, np.ones(10) * 10])

        weights = cell_declustering(x, y, z, cell_size=1.0)

        assert weights.shape == z.shape
        assert np.all(weights > 0)
        assert np.all(weights <= 1.0)
        # Clustered points should have lower weights
        assert np.mean(weights[:30]) < np.mean(weights[30:])

    def test_cell_declustering_different_cell_sizes(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 10, 50)
        z = x + y + np.random.normal(0, 0.5, 50)

        for cell_size in [0.5, 1.0, 2.0]:
            pass

        for cell_size in [0.5, 1.0, 2.0]:
            pass
        assert np.all(weights > 0)
        assert np.all(weights <= 1.0)

    def test_cell_declustering_uniform_data(self):
        np.random.seed(42)
        # Uniformly distributed data
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 10, 50)
        z = np.random.uniform(0, 10, 50)

        weights = cell_declustering(x, y, z, cell_size=1.0)

        # Weights should be relatively uniform
        assert np.std(weights) < 0.2

    def test_cell_declustering_reduces_bias(self):
        np.random.seed(42)
        # Create heavily clustered low values
        cluster_x = np.random.normal(2, 0.3, 80)
        cluster_y = np.random.normal(2, 0.3, 80)
        cluster_z = np.random.normal(2, 0.5, 80)

        # Few high values
        sparse_x = np.random.uniform(5, 10, 20)
        sparse_y = np.random.uniform(5, 10, 20)
        sparse_z = np.random.normal(8, 0.5, 20)

        x = np.concatenate([cluster_x, sparse_x])
        y = np.concatenate([cluster_y, sparse_y])
        z = np.concatenate([cluster_z, sparse_z])

        # Naive mean is biased toward clustered values
        naive_mean = np.mean(z)

        # Declustered mean should be higher
        weights = cell_declustering(x, y, z, cell_size=1.0)
        declustered_mean = np.average(z, weights=weights)

        assert declustered_mean > naive_mean

class TestPolygonalDeclustering:

    def test_polygonal_declustering_basic(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 40)
        y = np.random.uniform(0, 10, 40)
        z = x + y + np.random.normal(0, 0.5, 40)

        weights = polygonal_declustering(x, y, z)

        assert weights.shape == z.shape
        assert np.all(weights > 0)
        assert np.sum(weights) > 0

class TestDetectClustering:

    def test_detect_clustering_clustered_data(self):
        np.random.seed(42)
        # Create clustered data
        cluster1 = np.random.normal(2, 0.3, 30)
        cluster2 = np.random.normal(8, 0.3, 30)
        x = np.concatenate([cluster1, cluster2])
        y = np.concatenate([cluster1, cluster2])

        is_clustered = detect_clustering(x, y, threshold=0.5)

        assert isinstance(is_clustered, (bool, np.bool_))
        assert is_clustered # Should detect clustering

    def test_detect_clustering_uniform_data(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 10, 50)

        is_clustered = detect_clustering(x, y, threshold=0.5)

        assert isinstance(is_clustered, (bool, np.bool_))
        # Uniform data should not be clustered
        assert not is_clustered

class TestIQROutlierDetection:

    def test_iqr_outliers_basic(self):
        np.random.seed(42)
        # Normal data with outliers
        data = np.random.normal(5, 1, 100)
        data = np.append(data, [15, 20, -5, -10]) # Add outliers

        outliers = detect_outliers_iqr(data)

        assert outliers.shape == data.shape
        assert outliers.dtype == bool
        assert np.sum(outliers) > 0 # Should detect some outliers
        assert np.sum(outliers) < len(data) # Not all points are outliers

    def test_iqr_outliers_no_outliers(self):
        np.random.seed(42)
        data = np.random.normal(5, 1, 50)

        outliers = detect_outliers_iqr(data, factor=1.5)

        # Should detect very few or no outliers
        assert np.sum(outliers) < 5

    def test_iqr_outliers_different_factors(self):
        np.random.seed(42)
        data = np.random.normal(5, 1, 100)
        data = np.append(data, [15, 20])

        outliers_strict = detect_outliers_iqr(data, factor=1.5)
        outliers_lenient = detect_outliers_iqr(data, factor=3.0)

        # Strict should detect more outliers
        assert np.sum(outliers_strict) >= np.sum(outliers_lenient)

class TestZScoreOutlierDetection:

    def test_zscore_outliers_basic(self):
        np.random.seed(42)
        data = np.random.normal(10, 2, 100)
        data = np.append(data, [25, 30, -5]) # Add outliers

        outliers = detect_outliers_zscore(data, threshold=3.0)

        assert outliers.shape == data.shape
        assert outliers.dtype == bool
        assert np.sum(outliers) > 0

    def test_zscore_outliers_different_thresholds(self):
        np.random.seed(42)
        data = np.random.normal(5, 1, 100)
        data = np.append(data, [10, 12, 15])

        outliers_strict = detect_outliers_zscore(data, threshold=2.0)
        outliers_lenient = detect_outliers_zscore(data, threshold=4.0)

        assert np.sum(outliers_strict) >= np.sum(outliers_lenient)

    def test_zscore_vs_modified_zscore(self):
        np.random.seed(42)
        data = np.random.normal(5, 1, 100)
        data = np.append(data, [15, 20])

        outliers_zscore = detect_outliers_zscore(data, threshold=3.0)
        outliers_modified = detect_outliers_modified_zscore(data, threshold=3.5)

        # Both should detect outliers
        assert np.sum(outliers_zscore) > 0
        assert np.sum(outliers_modified) > 0

class TestSpatialOutliers:

    def test_spatial_outliers_basic(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 10, 50)
        z = x + y + np.random.normal(0, 0.5, 50)

        # Add spatial outliers (values very different from neighbors)
        z = np.append(z, [100])
        x = np.append(x, [5.0])
        y = np.append(y, [5.0])

        outliers = detect_spatial_outliers(x, y, z, n_neighbors=5)

        assert outliers.shape == z.shape
        assert outliers.dtype == bool
        assert np.sum(outliers) > 0 # Should detect the outlier

class TestEnsembleOutlierDetection:

    def test_ensemble_outliers_basic(self):
        np.random.seed(42)
        data = np.random.normal(5, 1, 100)
        data = np.append(data, [15, 20, -5])

        outliers = detect_outliers_ensemble(data, methods=['iqr', 'zscore'])

        assert outliers.shape == data.shape
        assert outliers.dtype == bool
        assert np.sum(outliers) > 0

class TestIntegration:

    def test_decluster_then_detect_outliers(self):
        np.random.seed(42)
        # Clustered data with outliers
        cluster_x = np.random.normal(2, 0.5, 40)
        cluster_y = np.random.normal(2, 0.5, 40)
        cluster_z = np.random.normal(5, 1, 40)

        sparse_x = np.random.uniform(5, 10, 10)
        sparse_y = np.random.uniform(5, 10, 10)
        sparse_z = np.random.normal(8, 1, 10)

        x = np.concatenate([cluster_x, sparse_x])
        y = np.concatenate([cluster_y, sparse_y])
        z = np.concatenate([cluster_z, sparse_z])

        # Add outliers
        z = np.append(z, [50, -20])
        x = np.append(x, [5, 6])
        y = np.append(y, [5, 6])

        # Step 1: Detect outliers
        outliers = detect_outliers_iqr(z)

        # Step 2: Decluster non-outliers
        x_clean = x[~outliers]
        y_clean = y[~outliers]
        z_clean = z[~outliers]

        weights = cell_declustering(x_clean, y_clean, z_clean, cell_size=1.0)

        # Calculate declustered mean
        declustered_mean = np.average(z_clean, weights=weights)

        assert len(x_clean) < len(x) # Outliers removed
        assert weights.shape == z_clean.shape
        assert declustered_mean > np.mean(cluster_z) # Corrected for clustering

if __name__ == "__main__":
    pass
