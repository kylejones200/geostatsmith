"""
    Support Change and Block Kriging

Support refers to the volume (1D, 2D, or 3D) over which a measurement is made:
    pass
- Point support: measurement at a specific location (core sample)
- Block support: average over a volume (mining block, pixel)

Support change addresses:
    pass
1. Point-to-block kriging: estimate block average from point data
2. Block-to-point: disaggregation (rarely done)
3. Block-to-block: change of support corrections
4. Variance relationships between supports

Key concepts from geokniga §2097-2240, §6058-6078:
    pass
"Block kriging estimates the average over a volume V:"
 Z(V) = 1/|V| ∫_V Z(u) du

The estimation variance is:
 σ^2(V) = -γ(V,V) - ΣΣ λi λj γ(ui - uj) + 2Σ λi γ(ui, V)

where γ(V,V) is the internal block variance (within-block variogram).

As block size increases, estimation variance decreases."

References:
    pass
- geokniga-introductiontogeostatistics.txt §4.2.2 (Block Kriging)
- Deutsch & Journel (1998) GSLIB, Chapter V.3
- Journel & Huijbregts (1978) Mining Geostatistics, Chapter VI
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from ..core.base import BaseKriging
from ..core.constants import REGULARIZATION_FACTOR
from ..core.logging_config import get_logger
from ..core.validators import validate_coordinates, validate_values
from ..math.distance import euclidean_distance
from ..math.matrices import regularize_matrix

logger = get_logger(__name__)


class BlockKriging(BaseKriging):
    """
    Block Kriging - Estimate block averages from point data

    Kriging for block support (volume V) requires:
    1. Point-to-point variogram gamma(ui - uj)
    2. Point-to-block variogram gamma(ui, V) = avg_v gamma(ui - v)
    3. Block-to-block variogram gamma(V, V) = avg_v1 avg_v2 gamma(v1 - v2)

    The block average has lower variance than point estimates:
    "As block size increases, estimation variance decreases."
    """

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        variogram_model: object | None = None,
        block_size: tuple[float, float] = (10.0, 10.0),
        n_disc: int = 5,
    ):
        """
           Initialize Block Kriging

           Parameters
           ----------
           x, y : np.ndarray
        Coordinates of sample points (point support)
        z : np.ndarray
        Values at sample points
        variogram_model : VariogramModelBase
        Fitted point variogram model
        block_size : tuple
        Block dimensions (width, height) in same units as coordinates
        n_disc : int
        Number of discretization points per dimension for block integration
        (n_disc^2 points used to approximate block average)
        """
        super().__init__(x, y, z, variogram_model)

        self.x, self.y = validate_coordinates(x, y)
        self.z = validate_values(z, n_expected=len(self.x))

        self.block_size = block_size
        self.n_disc = n_disc

        # Discretization points for block (relative to block center)
        disc_x = np.linspace(-block_size[0] / 2, block_size[0] / 2, n_disc)
        disc_y = np.linspace(-block_size[1] / 2, block_size[1] / 2, n_disc)
        self.disc_xx, self.disc_yy = np.meshgrid(disc_x, disc_y)
        self.disc_xx = self.disc_xx.flatten()
        self.disc_yy = self.disc_yy.flatten()
        self.n_disc_points = len(self.disc_xx)

        if self.variogram_model is not None:
            self._build_kriging_matrix()

    def _precompute_block_variance(self):
        """
        Precompute gamma(V,V) - internal block variance (vectorized)

        gamma(V,V) = 1/|V|^2 integral integral gamma(u-v) du dv

        Approximated using discretization points.
        """
        # Vectorized distance matrix for all discretization point pairs

        dist_matrix = euclidean_distance(
            self.disc_xx, self.disc_yy, self.disc_xx, self.disc_yy
        )

        # Vectorized variogram evaluation
        gamma_matrix = self.variogram_model(dist_matrix)

        # Average over all pairs (including self-pairs)
        self.gamma_VV = np.mean(gamma_matrix)
        logger.debug(
            f"Precomputed block variance gamma(V,V) = {self.gamma_VV:.6f} (vectorized)"
        )

    def _point_to_block_variogram(
        self, x_point: float, y_point: float, x_block: float, y_block: float
    ) -> float:
        """
        Calculate gamma(point, block) - average variogram from point to block (vectorized)

        gamma(u_i, V) = 1/|V| integral_V gamma(u_i - v) dv

        Approximated by averaging over discretization points in block.

        Parameters
        ----------
        x_point, y_point : float
            Point coordinates
        x_block, y_block : float
            Block center coordinates

        Returns
        -------
        float
            Point-to-block variogram
        """
        # Discretization points in block (vectorized)
        x_disc = x_block + self.disc_xx
        y_disc = y_block + self.disc_yy

        # Vectorized distance calculation
        dx = x_point - x_disc
        dy = y_point - y_disc
        h = np.sqrt(dx * dx + dy * dy)

        # Vectorized variogram evaluation
        gamma_vals = self.variogram_model(h)

        return np.mean(gamma_vals)

    def predict(
        self,
        x_new: npt.NDArray[np.float64],
        y_new: npt.NDArray[np.float64],
        return_variance: bool = True,
    ) -> (
        npt.NDArray[np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ):
        """
        Predict block averages at new locations

        Parameters
        ----------
        x_new, y_new : np.ndarray
            Coordinates of block centers
        return_variance : bool
            If True, return both predictions and block kriging variance

        Returns
        -------
        predictions : np.ndarray
            Predicted block averages
        variance : np.ndarray, optional
            Block kriging variance (lower than point variance)
        """
        if self.variogram_model is None:
            raise ValueError("Variogram model must be fitted before prediction")

        from ..utils.validation import validate_coordinates

        x_new, y_new = validate_coordinates(x_new, y_new)
        n_pred = len(x_new)
        n_data = len(self.x)

        # Build kriging matrix (vectorized point-to-point distances)
        K = np.zeros((n_data + 1, n_data + 1), dtype=np.float64)

        # Vectorized distance matrix

        dist_matrix = euclidean_distance(self.x, self.y, self.x, self.y)

        # Vectorized variogram evaluation
        K[:n_data, :n_data] = self.variogram_model(dist_matrix)

        # Unbiasedness constraint
        K[:n_data, n_data] = 1.0
        K[n_data, :n_data] = 1.0
        K[n_data, n_data] = 0.0

        K = regularize_matrix(K, epsilon=REGULARIZATION_FACTOR)
        logger.debug(
            f"Block Kriging matrix built (vectorized): {n_data + 1}x{n_data + 1}"
        )

        predictions = np.zeros(n_pred)
        variances = np.zeros(n_pred) if return_variance else None

        for i in range(n_pred):
            rhs = np.zeros(n_data + 1, dtype=np.float64)

            # Vectorized point-to-block calculation
            for j in range(n_data):
                rhs[j] = self._point_to_block_variogram(
                    self.x[j], self.y[j], x_new[i], y_new[i]
                )

            from ..core.constants import UNBIASEDNESS_CONSTRAINT

            rhs[n_data] = UNBIASEDNESS_CONSTRAINT  # Unbiasedness

            # Solve
            try:
                weights = np.linalg.solve(K, rhs)
            except np.linalg.LinAlgError as e:
                from ..exceptions import KrigingError

                logger.error(f"Failed to solve block kriging system at point {i}: {e}")
                raise KrigingError(f"Failed to solve kriging system: {e}")

            lambdas = weights[:n_data]

            # Prediction
            predictions[i] = np.dot(lambdas, self.z)

            # Block variance (from geokniga §6058-6070)
            # sigma^2(V) = -gamma(V,V) - sum sum lambda_i lambda_j gamma(u_i-u_j) + 2 sum lambda_i gamma(u_i,V)
            if return_variance:
                # First term: -gamma(V,V)
                variance_term1 = -self.gamma_VV

                # Second term: -sum sum lambda_i lambda_j gamma(u_i-u_j)
                variance_term2 = -np.dot(lambdas, np.dot(K[:n_data, :n_data], lambdas))

                # Third term: 2 sum lambda_i gamma(u_i,V)
                variance_term3 = 2 * np.dot(lambdas, rhs[:n_data])

                variances[i] = variance_term1 + variance_term2 + variance_term3

        logger.info(
            f"Block Kriging completed for {n_pred} blocks (vectorized discretization)"
        )

        if return_variance:
            return predictions, variances
        return predictions

    def cross_validate(self) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
        """
        Perform leave-one-out cross-validation

        Returns
        -------
        predictions : np.ndarray
            Cross-validated predictions
        metrics : dict
            Dictionary of performance metrics
        """
        from ..validation.cross_validation import leave_one_out
        from ..validation.metrics import mean_squared_error, r_squared

        predictions = leave_one_out(self, self.x, self.y, self.z)

        metrics = {
            "mse": mean_squared_error(self.z, predictions),
            "r2": r_squared(self.z, predictions),
        }

        return predictions, metrics


class SupportCorrection:
    """
    Support Correction Tools

    Provides methods for:
    1. Regularization: point variogram -> block variogram
    2. Variance relationships between supports
    3. Dispersion variance calculations
    """

    @staticmethod
    def regularize_variogram(
        variogram_point: Callable, block_size: tuple[float, float], n_disc: int = 10
    ) -> Callable[[float], float]:
        """
        Regularize point variogram to block variogram

        Block variogram gamma_V(h) relates to point variogram gamma(h) by:
        gamma_V(h) = gamma_bar(V, V+h) - gamma_bar(V, V)

        where:
        gamma_bar(V, V+h) = avg over V1 and V2 of gamma(u1 - u2)
        for u1 in V1, u2 in V2 separated by h

        Parameters
        ----------
        variogram_point : callable
            Point variogram function gamma(h)
        block_size : tuple
            Block dimensions (width, height)
        n_disc : int
            Discretization resolution

        Returns
        -------
        callable
            Block variogram function gamma_V(h)
        """
        # Create discretization grid for block
        disc_x = np.linspace(-block_size[0] / 2, block_size[0] / 2, n_disc)
        disc_y = np.linspace(-block_size[1] / 2, block_size[1] / 2, n_disc)
        disc_xx, disc_yy = np.meshgrid(disc_x, disc_y)
        disc_xx = disc_xx.flatten()
        disc_yy = disc_yy.flatten()
        len(disc_xx)

        # Compute gamma_bar(V, V) - internal block variance (vectorized)

        dist_matrix_VV = euclidean_distance(disc_xx, disc_yy, disc_xx, disc_yy)
        gamma_matrix_VV = variogram_point(dist_matrix_VV)
        gamma_VV = np.mean(gamma_matrix_VV)

        def block_variogram(h: float | npt.NDArray) -> float | npt.NDArray:
            h = np.asarray(h)
            scalar_input = h.ndim == 0
            h = np.atleast_1d(h)

            gamma_block = np.zeros_like(h)

            for k, h_val in enumerate(h):
                # Shift first block by h_val in x direction
                disc_xx_shifted = disc_xx + h_val

                # Distance matrix between shifted and original blocks
                dist_matrix_VVh = euclidean_distance(
                    disc_xx_shifted, disc_yy, disc_xx, disc_yy
                )
                gamma_matrix_VVh = variogram_point(dist_matrix_VVh)
                gamma_VVh = np.mean(gamma_matrix_VVh)

                # Block variogram
                gamma_block[k] = gamma_VVh - gamma_VV

            return gamma_block[0] if scalar_input else gamma_block

        return block_variogram

    @staticmethod
    def dispersion_variance(
        variogram: Callable,
        domain_size: tuple[float, float],
        block_size: tuple[float, float],
        n_disc: int = 10,
    ) -> float:
        """
        Calculate dispersion variance D^2(v/V)

        Variance of block values within a larger domain:
        D^2(v/V) = gamma_bar(V,V) - gamma_bar(v,v)

        Important for:
        - Resource estimation
        - Grade control
        - Selectivity studies

        Parameters
        ----------
        variogram : callable
            Point variogram function
        domain_size : tuple
            Size of large domain (V)
        block_size : tuple
            Size of small blocks (v)
        n_disc : int
            Discretization resolution

        Returns
        -------
        float
            Dispersion variance D^2(v/V)
        """
        # Discretize domain
        disc_x_domain = np.linspace(0, domain_size[0], n_disc)
        disc_y_domain = np.linspace(0, domain_size[1], n_disc)
        xx_domain, yy_domain = np.meshgrid(disc_x_domain, disc_y_domain)
        xx_domain = xx_domain.flatten()
        yy_domain = yy_domain.flatten()
        len(xx_domain)

        # Discretize block
        disc_x_block = np.linspace(-block_size[0] / 2, block_size[0] / 2, n_disc)
        disc_y_block = np.linspace(-block_size[1] / 2, block_size[1] / 2, n_disc)
        xx_block, yy_block = np.meshgrid(disc_x_block, disc_y_block)
        xx_block = xx_block.flatten()
        yy_block = yy_block.flatten()
        len(xx_block)

        # gamma_bar(V,V) - domain internal variance (vectorized)

        dist_matrix_domain = euclidean_distance(
            xx_domain, yy_domain, xx_domain, yy_domain
        )
        gamma_matrix_domain = variogram(dist_matrix_domain)
        gamma_VV = np.mean(gamma_matrix_domain)

        # gamma_bar(v,v) - block internal variance (vectorized)
        dist_matrix_block = euclidean_distance(xx_block, yy_block, xx_block, yy_block)
        gamma_matrix_block = variogram(dist_matrix_block)
        gamma_vv = np.mean(gamma_matrix_block)

        return gamma_VV - gamma_vv
