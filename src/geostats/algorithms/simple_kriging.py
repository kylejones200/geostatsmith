"""
    Simple Kriging implementation

Simple Kriging assumes a known constant mean μ.

Prediction equation:
 ẑ(x₀) = μ + Σλᵢ[z(xᵢ) - μ]

where λᵢ are weights obtained by solving:
 C * λ = c₀

C is the covariance matrix between sample points,
c₀ is the covariance vector between sample points and prediction point.
"""

import numpy as np
import numpy.typing as npt

from ..core.base import BaseKriging
from ..core.exceptions import KrigingError
from ..core.validators import validate_coordinates, validate_values
from ..math.distance import euclidean_distance
from ..math.matrices import regularize_matrix, solve_kriging_system
from ..math.numerical import cross_validation_score


class SimpleKriging(BaseKriging):
    """
    Simple Kriging interpolation

    Assumes a known stationary mean. Best when the mean is well-estimated
    from the data or known from theory.
    """

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        variogram_model: object | None = None,
        mean: float | None = None,
    ):
        """
            Initialize Simple Kriging

        Parameters
        ----------
        x, y : np.ndarray
        Coordinates of sample points
        z : np.ndarray
        Values at sample points
           variogram_model : VariogramModelBase, optional
           Fitted variogram model
           mean : float, optional
           Known mean value. If None, estimated from data.
        """
        super().__init__(x, y, z, variogram_model)

        # Validate inputs
        self.x, self.y = validate_coordinates(x, y)
        self.z = validate_values(z, n_expected=len(self.x))

        # Set or estimate mean
        if mean is None:
            self.mean = np.mean(self.z)
        else:
            self.mean = mean

        # Center the data
        self.z_centered = self.z - self.mean

        # Build covariance matrix
        if self.variogram_model is not None:
            self._build_kriging_matrix()

    def _build_kriging_matrix(self) -> None:
        # Calculate pairwise distances
        dist_matrix = euclidean_distance(self.x, self.y, self.x, self.y)

        # Get sill from variogram
        from ..core.constants import DEFAULT_SILL_VALUE

        sill = self.variogram_model.parameters.get("sill", DEFAULT_SILL_VALUE)
        nugget = self.variogram_model.parameters.get("nugget", 0.0)

        # Convert variogram to covariance: C(h) = sill - gamma(h)
        gamma_matrix = self.variogram_model(dist_matrix)
        self.cov_matrix = sill - gamma_matrix

        # Add nugget effect to diagonal (measurement error)
        np.fill_diagonal(self.cov_matrix, sill)

        # Regularize for numerical stability
        from ..core.constants import EPSILON

        self.cov_matrix = regularize_matrix(self.cov_matrix, epsilon=EPSILON)

    def predict(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        return_variance: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
        """
            Perform Simple Kriging prediction

        Parameters
        ----------
        x, y : np.ndarray
        Coordinates for prediction
        return_variance : bool
        Whether to return kriging variance

        Returns
        -------
        predictions : np.ndarray
        Predicted values
        variance : np.ndarray or None
        Kriging variance (if return_variance=True)
        """
        if self.variogram_model is None:
            raise KrigingError("Variogram model must be provided for prediction")

        x_pred, y_pred = validate_coordinates(x, y)
        n_pred = len(x_pred)

        predictions = np.zeros(n_pred)
        variances = np.zeros(n_pred) if return_variance else None

        # Get sill for variance calculation
        from ..core.constants import DEFAULT_SILL_VALUE

        sill = self.variogram_model.parameters.get("sill", DEFAULT_SILL_VALUE)

        # Predict at each location
        for i in range(n_pred):
            dist_to_samples = euclidean_distance(
                np.array([x_pred[i]]),
                np.array([y_pred[i]]),
                self.x,
                self.y,
            ).flatten()

            # Covariance vector: c0 = sill - gamma(h)
            gamma_vec = self.variogram_model(dist_to_samples)
            cov_vec = sill - gamma_vec

            # Solve for weights: C * lambda = c0
            try:
                weights = solve_kriging_system(self.cov_matrix, cov_vec)
            except KrigingError:
                # Fallback: use nearest neighbor
                nearest_idx = np.argmin(dist_to_samples)
                predictions[i] = self.z[nearest_idx]
                if return_variance:
                    variances[i] = sill
                continue

            # Simple kriging prediction: z_hat(x0) = mu + sum(lambda_i * [z(xi) - mu])
            predictions[i] = self.mean + np.dot(weights, self.z_centered)

            # Kriging variance: sigma^2(x0) = C(0) - sum(lambda_i * c0_i)
            if return_variance:
                variances[i] = sill - np.dot(weights, cov_vec)
                # Variance should be non-negative; negative indicates numerical issues
                from ..core.constants import ZERO_VALUE

                if variances[i] < ZERO_VALUE:
                    import warnings

                    warnings.warn(
                        f"Negative kriging variance {variances[i]:.6e} at prediction point {i}. "
                        "This may indicate numerical instability. Variance will be clamped to 0.",
                        RuntimeWarning,
                    )
                    variances[i] = ZERO_VALUE

        if return_variance:
            return predictions, variances
        else:
            return predictions, None

    def cross_validate(self) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
        """
        Perform leave-one-out cross-validation

        Returns
        -------
        predictions : np.ndarray
        Cross-validated predictions at sample points
        metrics : dict
        Dictionary of validation metrics
        """
        if self.variogram_model is None:
            raise KrigingError("Variogram model must be provided for cross-validation")

        predictions = np.zeros(self.n_points)

        # Leave-one-out cross-validation
        for i in range(self.n_points):
            mask = np.ones(self.n_points, dtype=bool)
            mask[i] = False

            x_train = self.x[mask]
            y_train = self.y[mask]
            z_train = self.z[mask]

            # Create temporary kriging object
            sk_temp = SimpleKriging(
                x_train,
                y_train,
                z_train,
                variogram_model=self.variogram_model,
                mean=self.mean,
            )

            # Predict at left-out point
            pred, _ = sk_temp.predict(
                np.array([self.x[i]]),
                np.array([self.y[i]]),
                return_variance=False,
            )
            predictions[i] = pred[0]

        # Calculate metrics
        metrics = cross_validation_score(self.z, predictions)

        return predictions, metrics
