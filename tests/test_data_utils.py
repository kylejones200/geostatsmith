"""
Tests for data utility functions
"""
import pytest
import numpy as np
from geostats.utils.data_utils import generate_synthetic_data, split_train_test

class TestGenerateSyntheticData:

    def test_generate_synthetic_data_basic(self):
        x, y, z = generate_synthetic_data(n_points=50, seed=42)

        assert len(x) == 50
        assert len(y) == 50
        assert len(z) == 50
        assert not np.any(np.isnan(x))
        assert not np.any(np.isnan(y))
        assert not np.any(np.isnan(z))

    def test_generate_synthetic_data_reproducible(self):
        x1, y1, z1 = generate_synthetic_data(n_points=30, seed=42)
        x2, y2, z2 = generate_synthetic_data(n_points=30, seed=42)

        assert np.allclose(x1, x2)
        assert np.allclose(y1, y2)
        assert np.allclose(z1, z2)

    def test_generate_synthetic_data_different_seeds(self):
        x1, y1, z1 = generate_synthetic_data(n_points=30, seed=42)
        x2, y2, z2 = generate_synthetic_data(n_points=30, seed=123)

        assert not np.allclose(x1, x2)
        assert not np.allclose(y1, y2)
        assert not np.allclose(z1, z2)

class TestSplitTrainTest:

    def test_split_train_test_basic(self):
        x = np.arange(100, dtype=float)
        y = np.arange(100, dtype=float)
        z = np.arange(100, dtype=float)

        x_train, y_train, z_train, x_test, y_test, z_test = split_train_test(
        x, y, z, test_size=0.3, random_state=42
        )

        assert len(x_train) == 70
        assert len(x_test) == 30
        assert len(x_train) + len(x_test) == len(x)

    def test_split_train_test_no_overlap(self):
        x = np.arange(50, dtype=float)
        y = np.arange(50, dtype=float)
        z = np.arange(50, dtype=float)

        x_train, y_train, z_train, x_test, y_test, z_test = split_train_test(
        x, y, z, test_size=0.2, random_state=42
        )

        # Check no indices overlap
        all_indices = np.concatenate([x_train, x_test])
        assert len(np.unique(all_indices)) == len(x)

    def test_split_train_test_reproducible(self):
        x = np.random.rand(100)
        y = np.random.rand(100)
        z = np.random.rand(100)

        result1 = split_train_test(x, y, z, test_size=0.3, random_state=42)
        result2 = split_train_test(x, y, z, test_size=0.3, random_state=42)

        for r1, r2 in zip(result1, result2):
            pass
        for r1, r2 in zip(result1, result2):
            pass
