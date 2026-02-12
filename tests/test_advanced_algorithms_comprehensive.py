"""
Tests for advanced kriging algorithms
Tests for algorithms with low coverage (<30%)
    """

import pytest
import numpy as np
from geostats.algorithms.cokriging import Cokriging, CollocatedCokriging
from geostats.algorithms.external_drift_kriging import ExternalDriftKriging
from geostats.algorithms.factorial_kriging import FactorialKriging
from geostats.algorithms.spacetime_kriging import SpaceTimeOrdinaryKriging, SpaceTimeSimpleKriging
from geostats.algorithms.support_change import BlockKriging
from geostats.models.variogram_models import SphericalModel, ExponentialModel

class TestCokriging:

    def test_initialization(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 20)
        y = np.random.uniform(0, 10, 20)
        z1 = x + y + np.random.normal(0, 0.2, 20)
        z2 = 1.5*x + 1.5*y + np.random.normal(0, 0.3, 20)

        variogram1 = SphericalModel(sill=1.0, range_param=5.0)
        variogram2 = SphericalModel(sill=1.5, range_param=5.0)

        ck = Cokriging(
        x, y, z1,
        x, y, z2,
        variogram_primary=variogram1,
        variogram_secondary=variogram2
        )
        assert ck is not None

    def test_predict(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 30)
        y = np.random.uniform(0, 10, 30)
        z1 = 2*x + 3*y + np.random.normal(0, 0.5, 30)
        z2 = 1.5*x + 2*y + np.random.normal(0, 0.3, 30)

        variogram1 = SphericalModel(sill=1.0, range_param=5.0)
        variogram2 = SphericalModel(sill=0.8, range_param=5.0)

        ck = Cokriging(
        x, y, z1,
        x, y, z2,
        variogram_primary=variogram1,
        variogram_secondary=variogram2
        )

        x_new = np.array([5.0])
        y_new = np.array([5.0])

        predictions, variance = ck.predict(x_new, y_new)

        assert predictions.shape == (1,)
        assert variance.shape == (1,)
        assert variance[0] >= 0

class TestCollocatedCokriging:

    def test_initialization(self):
        np.random.seed(42)
        x_primary = np.random.uniform(0, 10, 15)
        y_primary = np.random.uniform(0, 10, 15)
        z_primary = x_primary + y_primary + np.random.normal(0, 0.2, 15)

        x_secondary = np.random.uniform(0, 10, 30)
        y_secondary = np.random.uniform(0, 10, 30)
        z_secondary = 1.5*x_secondary + 1.5*y_secondary + np.random.normal(0, 0.3, 30)

        variogram = SphericalModel(sill=1.0, range_param=5.0)

        cck = CollocatedCokriging(
        x_primary, y_primary, z_primary,
        x_secondary, y_secondary, z_secondary,
        variogram_model=variogram
        )
        assert cck is not None

class TestExternalDriftKriging:

    def test_initialization(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 30)
        y = np.random.uniform(0, 10, 30)
        drift = x**2 + y**2
        z = 0.5 * drift + np.random.normal(0, 0.5, 30)

        variogram = ExponentialModel(sill=1.0, range_param=5.0)

        edk = ExternalDriftKriging(x, y, z, drift, variogram_model=variogram)
        assert edk is not None

    def test_predict_single_drift(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 40)
        y = np.random.uniform(0, 10, 40)
        drift = x + y
        z = 2 * drift + np.random.normal(0, 0.5, 40)

        variogram = SphericalModel(sill=1.0, range_param=5.0)
        edk = ExternalDriftKriging(x, y, z, drift, variogram_model=variogram)

        x_new = np.array([5.0, 6.0])
        y_new = np.array([5.0, 6.0])
        drift_new = x_new + y_new

        predictions, variance = edk.predict(x_new, y_new, drift_new)

        assert predictions.shape == (2,)
        assert variance.shape == (2,)
        assert np.all(variance >= 0)

    def test_predict_multiple_drifts(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 10, 50)

        drift1 = x + y
        drift2 = x**2
        drifts = np.column_stack([drift1, drift2])

        z = 2*drift1 + 0.5*drift2 + np.random.normal(0, 0.5, 50)

        variogram = SphericalModel(sill=1.0, range_param=5.0)
        edk = ExternalDriftKriging(x, y, z, drifts, variogram_model=variogram)

        x_new = np.array([5.0])
        y_new = np.array([5.0])
        drifts_new = np.column_stack([x_new + y_new, x_new**2])

        predictions, variance = edk.predict(x_new, y_new, drifts_new)

        assert predictions.shape == (1,)
        assert variance.shape == (1,)

class TestFactorialKriging:

    def test_initialization(self):
        np.random.seed(42)
        x = np.random.uniform(0, 20, 50)
        y = np.random.uniform(0, 20, 50)
        z = np.sin(x/2) + np.sin(x/10) + np.random.normal(0, 0.2, 50)

        fk = FactorialKriging(x, y, z, n_structures=2)
        assert fk is not None
        assert fk.n_structures == 2

    def test_predict_two_structures(self):
        np.random.seed(42)
        x = np.random.uniform(0, 20, 60)
        y = np.random.uniform(0, 20, 60)

        # Short-range + long-range variation
        z = np.sin(x) + 0.1 * x + np.random.normal(0, 0.2, 60)

        fk = FactorialKriging(x, y, z, n_structures=2)

        x_new = np.array([10.0])
        y_new = np.array([10.0])

        predictions, variance = fk.predict(x_new, y_new)

        assert predictions.shape == (1,)
        assert variance.shape == (1,)
        assert variance[0] >= 0

class TestSpaceTimeKriging:

    def test_ordinary_kriging_initialization(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 30)
        y = np.random.uniform(0, 10, 30)
        t = np.random.uniform(0, 5, 30)
        z = x + y + 0.5*t + np.random.normal(0, 0.2, 30)

        spatial_variogram = SphericalModel(sill=1.0, range_param=5.0)
        temporal_variogram = ExponentialModel(sill=0.5, range_param=2.0)

        stok = SpaceTimeOrdinaryKriging(
        x, y, t, z,
        spatial_variogram=spatial_variogram,
        temporal_variogram=temporal_variogram
        )
        assert stok is not None

    def test_ordinary_kriging_predict(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 40)
        y = np.random.uniform(0, 10, 40)
        t = np.random.uniform(0, 5, 40)
        z = 2*x + 3*y + 0.5*t + np.random.normal(0, 0.3, 40)

        spatial_variogram = SphericalModel(sill=1.0, range_param=5.0)
        temporal_variogram = ExponentialModel(sill=0.5, range_param=2.0)

        stok = SpaceTimeOrdinaryKriging(
        x, y, t, z,
        spatial_variogram=spatial_variogram,
        temporal_variogram=temporal_variogram
        )

        x_new = np.array([5.0, 6.0])
        y_new = np.array([5.0, 6.0])
        t_new = np.array([2.5, 3.0])

        predictions, variance = stok.predict(x_new, y_new, t_new)

        assert predictions.shape == (2,)
        assert variance.shape == (2,)
        assert np.all(variance >= 0)

    def test_simple_kriging_initialization(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 30)
        y = np.random.uniform(0, 10, 30)
        t = np.random.uniform(0, 5, 30)
        z = x + y + t + np.random.normal(0, 0.2, 30)

        spatial_variogram = SphericalModel(sill=1.0, range_param=5.0)
        temporal_variogram = ExponentialModel(sill=0.5, range_param=2.0)

        stsk = SpaceTimeSimpleKriging(
        x, y, t, z,
        spatial_variogram=spatial_variogram,
        temporal_variogram=temporal_variogram,
        mean=np.mean(z)
        )
        assert stsk is not None

class TestBlockKriging:

    def test_initialization(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 30)
        y = np.random.uniform(0, 10, 30)
        z = x + y + np.random.normal(0, 0.3, 30)

        variogram = SphericalModel(sill=1.0, range_param=5.0)

        bk = BlockKriging(
        x, y, z,
        variogram_model=variogram,
        block_size=(2.0, 2.0),
        n_disc=5
        )
        assert bk is not None

    def test_predict_block(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 40)
        y = np.random.uniform(0, 10, 40)
        z = 2*x + 3*y + np.random.normal(0, 0.5, 40)

        variogram = SphericalModel(sill=1.0, range_param=5.0)

        bk = BlockKriging(
        x, y, z,
        variogram_model=variogram,
        block_size=(2.0, 2.0),
        n_disc=5
        )

        x_new = np.array([5.0])
        y_new = np.array([5.0])

        predictions, variance = bk.predict(x_new, y_new)

        assert predictions.shape == (1,)
        assert variance.shape == (1,)
        # Block variance should be lower than point variance
        assert variance[0] >= 0

    def test_different_block_sizes(self):
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = np.random.uniform(0, 10, 50)
        z = x + y + np.random.normal(0, 0.3, 50)

        variogram = SphericalModel(sill=1.0, range_param=5.0)

        # Small block
        bk_small = BlockKriging(
        x, y, z,
        variogram_model=variogram,
        block_size=(1.0, 1.0),
        n_disc=5
        )

        # Large block
        bk_large = BlockKriging(
        x, y, z,
        variogram_model=variogram,
        block_size=(4.0, 4.0),
        n_disc=5
        )

        x_new = np.array([5.0])
        y_new = np.array([5.0])

        _, var_small = bk_small.predict(x_new, y_new)
        _, var_large = bk_large.predict(x_new, y_new)

        # Larger blocks should have lower variance
        assert var_large[0] < var_small[0]

if __name__ == "__main__":
    pass
