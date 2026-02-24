"""
3D Kriging

Extension of kriging methods to three-dimensional space.
Common applications:
    pass
- Mining (ore grade in x, y, z coordinates)
- Hydrogeology (groundwater properties in 3D aquifers)
- Atmospheric science (temperature, pollutants at different altitudes)
- Oceanography (salinity, temperature at depth)

The mathematics are identical to 2D kriging, but distances are computed
in 3D space. Anisotropy becomes more complex with different ranges in
vertical vs horizontal directions.

References:
    pass
- Deutsch & Journel (1998) - GSLIB
- Wackernagel (2003) - Multivariate Geostatistics (Chapter 10)
- Chilès & Delfiner (2012) - Geostatistics (Chapter 3)
"""

import numpy as np
import numpy.typing as npt

from ..core.base import BaseKriging
from ..core.constants import REGULARIZATION_FACTOR
from ..core.logging_config import get_logger
from ..math.distance import (
    euclidean_distance,
)
from ..math.matrices import regularize_matrix

logger = get_logger(__name__)


def validate_coordinates_3d(
    y: npt.NDArray[np.float64], z: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate 3D coordinates"""
    x = np.asarray(x, dtype=np.float64).flatten()
    y = np.asarray(y, dtype=np.float64).flatten()
    z = np.asarray(z, dtype=np.float64).flatten()

    if not (len(x) == len(y) == len(z)):
        raise ValueError("x, y, z must have the same length")

    return x, y, z


class SimpleKriging3D(BaseKriging):
    """
    Simple Kriging in 3D space

    Assumes known constant mean. Kriging equations are identical to 2D,
    but distances are computed in 3D Euclidean space.
    """

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        variogram_model: object | None = None,
        known_mean: float | None = None,
    ):
        """
        Initialize 3D Simple Kriging

        Parameters
        ----------
        x, y, z : np.ndarray
            3D coordinates of sample points
        values : np.ndarray
            Values at sample points
        variogram_model : VariogramModelBase, optional
            Fitted variogram model (3D variogram)
        known_mean : float, optional
            Known mean of the random field
        """
        # Note: BaseKriging expects x, y, z as data, not coordinates
        # We'll override to handle 3D properly
        self.x, self.y, self.z = validate_coordinates_3d(x, y, z)
        from ..utils.validation import validate_values

        self.values = validate_values(values, n_expected=len(self.x))
        self.variogram_model = variogram_model

        # Estimate mean if not provided
        if known_mean is not None:
            self.mean = known_mean
        else:
            self.mean = np.mean(self.values)

        # Center the data
        self.residuals = self.values - self.mean

        if self.variogram_model is not None:
            self._build_kriging_matrix()

    def _build_kriging_matrix(self):
        n = len(self.x)

        # Build covariance matrix (vectorized distance calculation)
        # For simple kriging: C(h) = sill - gamma(h)
        from ..core.constants import DEFAULT_SILL_VALUE

        sill = (
            self.variogram_model.sill
            if hasattr(self.variogram_model, "sill")
            else DEFAULT_SILL_VALUE
        )

        # Vectorized 3D distance matrix

        coords = np.column_stack([self.x, self.y, self.z])
        dist_matrix = euclidean_distance(coords, coords, coords, coords)

        # Vectorized variogram evaluation
        gamma_matrix = self.variogram_model(dist_matrix)

        # Covariance = sill - variogram
        K = sill - gamma_matrix

        # Regularize if needed

        K = regularize_matrix(K, epsilon=REGULARIZATION_FACTOR)

        self.kriging_matrix = K
        logger.debug(f"3D Simple Kriging matrix built (vectorized): {n}x{n}")

    def predict(
        self,
        x_new: npt.NDArray[np.float64],
        y_new: npt.NDArray[np.float64],
        z_new: npt.NDArray[np.float64],
        return_variance: bool = True,
    ) -> (
        npt.NDArray[np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ):
        """
        Predict at new 3D locations

        Parameters
        ----------
        x_new, y_new, z_new : np.ndarray
            3D coordinates of prediction points
        return_variance : bool
            If True, return both predictions and kriging variance

        Returns
        -------
        predictions : np.ndarray
            Predicted values
        variance : np.ndarray, optional
            Kriging variance at each point
        """
        if self.variogram_model is None:
            raise ValueError("Variogram model must be fitted before prediction")

        x_new, y_new, z_new = validate_coordinates_3d(x_new, y_new, z_new)
        n_pred = len(x_new)
        len(self.x)

        from ..core.constants import DEFAULT_SILL_VALUE

        sill = (
            self.variogram_model.sill
            if hasattr(self.variogram_model, "sill")
            else DEFAULT_SILL_VALUE
        )

        # Vectorized distance calculation from data to prediction points

        coords_data = np.column_stack([self.x, self.y, self.z])
        coords_pred = np.column_stack([x_new, y_new, z_new])
        from scipy.spatial.distance import cdist

        dist_to_pred = cdist(coords_pred, coords_data)

        # Vectorized variogram evaluation
        gamma_to_pred = self.variogram_model(dist_to_pred)

        # Covariance vectors (vectorized)
        cov_to_pred = sill - gamma_to_pred  # shape: (n_data, n_pred)

        predictions = np.zeros(n_pred, dtype=np.float64)
        variances = np.zeros(n_pred, dtype=np.float64) if return_variance else None

        # Still need loop over prediction points for solving (inherent to kriging)
        for i in range(n_pred):
            # Solve for weights
            try:
                lambdas = np.linalg.solve(self.kriging_matrix, cov_to_pred[:, i])
            except np.linalg.LinAlgError as e:
                from ..exceptions import KrigingError

                logger.error(f"Failed to solve 3D kriging system at point {i}: {e}")
                raise KrigingError(f"Failed to solve kriging system: {e}")

            # Prediction: m + sum lambda_i * (z_i - m)
            predictions[i] = self.mean + np.dot(lambdas, self.residuals)

            # Variance: sigma^2(x_0) = C(0) - sum lambda_i * C(x_i - x_0)
            if return_variance:
                variances[i] = sill - np.dot(lambdas, cov_to_pred[:, i])

        logger.info(
            f"3D Simple Kriging completed for {n_pred} prediction points (vectorized)"
        )
        if return_variance:
            return predictions, variances
        return predictions


class OrdinaryKriging3D(BaseKriging):
    """
    Ordinary Kriging in 3D space

    Accounts for unknown mean through Lagrange multiplier.
    Most common kriging variant for 3D applications.
    """

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        variogram_model: object | None = None,
    ):
        """
        Initialize 3D Ordinary Kriging

        Parameters
        ----------
        x, y, z : np.ndarray
            3D coordinates of sample points
        values : np.ndarray
            Values at sample points
        variogram_model : VariogramModelBase, optional
            Fitted variogram model
        """
        self.x, self.y, self.z = validate_coordinates_3d(x, y, z)
        from ..utils.validation import validate_values

        self.values = validate_values(values, n_expected=len(self.x))
        self.variogram_model = variogram_model

        if self.variogram_model is not None:
            self._build_kriging_matrix()

    def _build_kriging_matrix(self):
        n = len(self.x)

        # Build variogram matrix with Lagrange constraint
        K = np.zeros((n + 1, n + 1), dtype=np.float64)

        # Vectorized 3D distance matrix

        coords = np.column_stack([self.x, self.y, self.z])
        from scipy.spatial.distance import cdist

        dist_matrix = cdist(coords, coords)

        # Vectorized variogram evaluation
        K[:n, :n] = self.variogram_model(dist_matrix)

        # Unbiasedness constraint
        from ..core.constants import UNBIASEDNESS_CONSTRAINT, ZERO_VALUE

        K[:n, n] = UNBIASEDNESS_CONSTRAINT
        K[n, :n] = UNBIASEDNESS_CONSTRAINT
        K[n, n] = ZERO_VALUE

        # Regularize

        K = regularize_matrix(K, epsilon=REGULARIZATION_FACTOR)

        self.kriging_matrix = K
        logger.debug(f"3D Ordinary Kriging matrix built (vectorized): {n + 1}x{n + 1}")

    def predict(
        self,
        x_new: npt.NDArray[np.float64],
        y_new: npt.NDArray[np.float64],
        z_new: npt.NDArray[np.float64],
        return_variance: bool = True,
    ) -> (
        npt.NDArray[np.float64]
        | tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ):
        """
        Predict at new 3D locations using ordinary kriging

        Parameters
        ----------
        x_new, y_new, z_new : np.ndarray
            3D coordinates of prediction points
        return_variance : bool
            If True, return both predictions and kriging variance

        Returns
        -------
        predictions : np.ndarray
            Predicted values
        variance : np.ndarray, optional
            Kriging variance
        """
        if self.variogram_model is None:
            raise ValueError("Variogram model must be fitted before prediction")

        x_new, y_new, z_new = validate_coordinates_3d(x_new, y_new, z_new)
        n_pred = len(x_new)
        n_data = len(self.x)

        # Vectorized distance calculation from data to prediction points

        coords_data = np.column_stack([self.x, self.y, self.z])
        coords_pred = np.column_stack([x_new, y_new, z_new])
        from scipy.spatial.distance import cdist

        dist_to_pred = cdist(coords_pred, coords_data)

        # Vectorized variogram evaluation
        gamma_to_pred = self.variogram_model(dist_to_pred)  # shape: (n_data, n_pred)

        predictions = np.zeros(n_pred, dtype=np.float64)
        variances = np.zeros(n_pred, dtype=np.float64) if return_variance else None

        # Still need loop over prediction points for solving (inherent to kriging)
        for i in range(n_pred):
            rhs = np.zeros(n_data + 1, dtype=np.float64)
            rhs[:n_data] = gamma_to_pred[:, i]
            from ..core.constants import UNBIASEDNESS_CONSTRAINT

            rhs[n_data] = UNBIASEDNESS_CONSTRAINT  # Unbiasedness constraint

            # Solve
            try:
                weights = np.linalg.solve(self.kriging_matrix, rhs)
            except np.linalg.LinAlgError as e:
                from ..exceptions import KrigingError

                logger.error(f"Failed to solve 3D OK system at point {i}: {e}")
                raise KrigingError(f"Failed to solve kriging system: {e}")

            # Extract lambda weights
            lambdas = weights[:n_data]
            weights[n_data]  # Lagrange multiplier

            # Prediction
            predictions[i] = np.dot(lambdas, self.values)

            # Variance
            if return_variance:
                variances[i] = np.dot(weights, rhs)

        logger.info(
            f"3D Ordinary Kriging completed for {n_pred} prediction points (vectorized)"
        )
        if return_variance:
            return predictions, variances
        return predictions

    def cross_validate(self) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
        n = len(self.x)
        predictions = np.zeros(n)

        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False

            # Temporary kriging without point i
            ok3d_temp = OrdinaryKriging3D(
                self.x[mask],
                self.y[mask],
                self.z[mask],
                self.values[mask],
                self.variogram_model,
            )

            # Predict at left-out point
            pred = ok3d_temp.predict(
                np.array([self.x[i]]),
                np.array([self.y[i]]),
                np.array([self.z[i]]),
                return_variance=False,
            )
            predictions[i] = pred[0]

        # Calculate errors
        errors = self.values - predictions

        metrics = {
            "MSE": np.mean(errors**2),
            "RMSE": np.sqrt(np.mean(errors**2)),
            "MAE": np.mean(np.abs(errors)),
            "R2": 1
            - np.sum(errors**2) / np.sum((self.values - np.mean(self.values)) ** 2),
            "bias": np.mean(errors),
            "predictions": predictions,
        }

        return errors, metrics
