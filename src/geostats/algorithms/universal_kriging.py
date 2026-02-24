"""
    Universal Kriging implementation

Universal Kriging (UK) accounts for a trend (drift) in the data.
The mean is modeled as a function of coordinates:
    pass

 m(x) = Σβf(x)

where f are basis functions (e.g., polynomials).

Common trends:
    pass
- Linear: m(x,y) = β₀ + β₁x + β₂y
- Quadratic: m(x,y) = β₀ + β₁x + β₂y + β₃x^2 + β₄xy + β₅y^2

The kriging system includes additional constraints for unbiasedness.
"""

import numpy as np
import numpy.typing as npt

from ..core.base import BaseKriging
from ..core.validators import validate_coordinates, validate_values
from ..math.distance import euclidean_distance
from ..math.matrices import regularize_matrix


class UniversalKriging(BaseKriging):
    """
    Universal Kriging interpolation

    Accounts for large-scale trends (drift) in the data by modeling
    the mean as a function of coordinates.
    """

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        variogram_model: object | None = None,
        drift_terms: str = "linear",
    ):
        """
           Initialize Universal Kriging

           Parameters
           ----------
           x, y : np.ndarray
           Coordinates of sample points
           z : np.ndarray
        Values at sample points
        variogram_model : VariogramModelBase, optional
        Fitted variogram model (should be fitted to residuals)
        drift_terms : str
        Type of drift/trend:
            pass
        - 'linear': β₀ + β₁x + β₂y
        - 'quadratic': β₀ + β₁x + β₂y + β₃x^2 + β₄xy + β₅y^2
        """
        super().__init__(x, y, z, variogram_model)

        # Validate inputs
        self.x, self.y = validate_coordinates(x, y)
        self.z = validate_values(z, n_expected=len(self.x))

        self.drift_terms = drift_terms

        # Build drift matrix
        self.drift_matrix = self._build_drift_matrix(self.x, self.y)
        self.n_drift = self.drift_matrix.shape[1]

        # Build kriging matrix
        if self.variogram_model is not None:
            self._build_kriging_matrix()

    def _build_drift_matrix(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Build matrix of drift basis functions

        Parameters
        ----------
        x, y : np.ndarray
            Coordinates

        Returns
        -------
        np.ndarray
            Drift matrix F, shape (n_points, n_drift_terms)
        """
        n = len(x)

        if self.drift_terms == "linear":
            F = np.column_stack(
                [
                    np.ones(n),
                    x,
                    y,
                ]
            )

        elif self.drift_terms == "quadratic":
            F = np.column_stack(
                [
                    np.ones(n),
                    x,
                    y,
                    x**2,
                    x * y,
                    y**2,
                ]
            )

        else:
            raise ValueError(f"Unknown drift_terms: {self.drift_terms}")

        return F

    def _build_kriging_matrix(self) -> None:
        # Calculate pairwise distances

        dist_matrix = euclidean_distance(self.x, self.y, self.x, self.y)

        # Get variogram values
        gamma_matrix = self.variogram_model(dist_matrix)

        # Build augmented matrix for Universal Kriging
        # | Gamma F |
        # | F^T 0 |
        n = len(self.x)
        p = self.n_drift

        self.kriging_matrix = np.zeros((n + p, n + p))
        self.kriging_matrix[:n, :n] = gamma_matrix
        self.kriging_matrix[:n, n:] = self.drift_matrix
        self.kriging_matrix[n:, :n] = self.drift_matrix.T
        self.kriging_matrix[n:, n:] = 0.0

        # Regularize for numerical stability

        self.kriging_matrix[:n, :n] = regularize_matrix(
            self.kriging_matrix[:n, :n], epsilon=1e-10
        )

    def predict(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        return_variance: bool = True,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
        """
        Perform Universal Kriging prediction

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
            raise ValueError("Variogram model must be fitted before prediction")

        from ..utils.validation import validate_coordinates

        x_pred, y_pred = validate_coordinates(x, y)
        n_pred = len(x_pred)

        predictions = np.zeros(n_pred)
        variances = np.zeros(n_pred) if return_variance else None

        # Predict at each location
        for i in range(n_pred):
            dist_to_samples = euclidean_distance(
                np.array([x_pred[i]]),
                np.array([y_pred[i]]),
                self.x,
                self.y,
            ).flatten()

            # Variogram vector
            gamma_vec = self.variogram_model(dist_to_samples)

            # Drift basis functions at prediction point
            drift_vec = self._build_drift_matrix(
                np.array([x_pred[i]]),
                np.array([y_pred[i]]),
            ).flatten()

            # Augmented right-hand side: [gamma(h), f(x_0)]^T
            n_points = len(self.x)
            rhs = np.zeros(n_points + self.n_drift)
            rhs[:n_points] = gamma_vec
            rhs[n_points:] = drift_vec

            # Solve for weights and Lagrange multipliers
            try:
                solution = np.linalg.solve(self.kriging_matrix, rhs)
            except Exception:
                # Fallback: use nearest neighbor
                nearest_idx = np.argmin(dist_to_samples)
                predictions[i] = self.z[nearest_idx]
                if return_variance:
                    variances[i] = 0.0
                continue

            weights = solution[:n_points]
            lagrange = solution[n_points:]

            # Universal kriging prediction: z_hat(x_0) = sum lambda_i * z(x_i)
            predictions[i] = np.dot(weights, self.z)

            # Kriging variance
            if return_variance:
                variances[i] = np.dot(weights, gamma_vec) + np.dot(lagrange, drift_vec)
                # Check for negative variance (indicates numerical issues)
                from ..core.constants import ZERO_VALUE

                if variances[i] < ZERO_VALUE:
                    import warnings

                    warnings.warn(
                        f"Negative kriging variance {variances[i]:.6e} at prediction point {i}. "
                        "This may indicate numerical instability or trend overfitting. "
                        "Variance will be clamped to 0.",
                        RuntimeWarning, stacklevel=2,
                    )
                    variances[i] = ZERO_VALUE

        if return_variance:
            return predictions, variances
        return predictions

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
            raise ValueError("Variogram model must be fitted before cross-validation")

        n_points = len(self.x)
        predictions = np.zeros(n_points)

        # Leave-one-out cross-validation
        for i in range(n_points):
            mask = np.ones(n_points, dtype=bool)
            mask[i] = False

            x_train = self.x[mask]
            y_train = self.y[mask]
            z_train = self.z[mask]

            # Create temporary kriging object
            uk_temp = UniversalKriging(
                x_train,
                y_train,
                z_train,
                variogram_model=self.variogram_model,
                drift_terms=self.drift_terms,
            )

            # Predict at left-out point
            pred, _ = uk_temp.predict(
                np.array([self.x[i]]),
                np.array([self.y[i]]),
                return_variance=False,
            )
            predictions[i] = pred[0]

        # Calculate metrics
        from ..validation.metrics import cross_validation_score

        metrics = cross_validation_score(self.z, predictions)

        return predictions, metrics

    def estimate_trend(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Estimate trend coefficients and residuals

        Returns
        -------
        coefficients : np.ndarray
            Estimated drift coefficients beta
        residuals : np.ndarray
            Residuals after removing trend
        """
        # Solve F'F * beta = F'z for trend coefficients
        FtF = self.drift_matrix.T @ self.drift_matrix
        Ftz = self.drift_matrix.T @ self.z

        try:
            coefficients = np.linalg.solve(FtF, Ftz)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            coefficients = np.linalg.lstsq(self.drift_matrix, self.z, rcond=None)[0]

        # Calculate residuals
        trend = self.drift_matrix @ coefficients
        residuals = self.z - trend

        return coefficients, residuals
