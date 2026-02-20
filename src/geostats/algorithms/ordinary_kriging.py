"""
    Ordinary Kriging implementation

Ordinary Kriging (OK) does not assume a known mean. Instead, it estimates
the local mean by adding a Lagrange multiplier constraint.

Prediction equation:
 ẑ(x₀) = Σλᵢz(xᵢ)

Subject to the unbiasedness constraint:
 Σλᵢ = 1

The kriging system becomes:
 | C 1 | |λ| |c₀|
 | 1ᵀ 0 | |μ| = |1 |

where μ is the Lagrange multiplier.
"""

from typing import Optional, Tuple, Dict
import numpy as np
import numpy.typing as npt

from ..core.base import BaseKriging
from ..core.exceptions import KrigingError
from ..core.validators import validate_coordinates, validate_values
from ..math.distance import euclidean_distance
from ..math.matrices import solve_kriging_system, regularize_matrix
from ..math.numerical import cross_validation_score

class OrdinaryKriging(BaseKriging):
    """
    Ordinary Kriging interpolation.

    Most commonly used kriging variant. Does not assume a known mean,
    making it more robust than Simple Kriging.
    """

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        variogram_model: Optional[object] = None,
    ):
        """
        Initialize Ordinary Kriging.

        Parameters
        ----------
        x, y : np.ndarray
            Coordinates of sample points
        z : np.ndarray
            Values at sample points
        variogram_model : VariogramModelBase, optional
            Fitted variogram model
        """
        super().__init__(x, y, z, variogram_model)

        # Validate inputs
        self.x, self.y = validate_coordinates(x, y)
        self.z = validate_values(z, n_expected=len(self.x))

        # Build kriging matrix
        if self.variogram_model is not None:
            self._build_kriging_matrix()

    def _build_kriging_matrix(self) -> None:
        """Build the kriging system matrix for Ordinary Kriging"""
        # Calculate pairwise distances
        dist_matrix = euclidean_distance(self.x, self.y, self.x, self.y)

        # Get variogram values
        gamma_matrix = self.variogram_model(dist_matrix)

        # Build augmented matrix for Ordinary Kriging
        # | γ(h) 1 |
        # | 1ᵀ 0 |
        n = self.n_points
        self.kriging_matrix = np.zeros((n + 1, n + 1))
        self.kriging_matrix[:n, :n] = gamma_matrix
        self.kriging_matrix[:n, n] = 1.0
        self.kriging_matrix[n, :n] = 1.0
        self.kriging_matrix[n, n] = 0.0

        # Regularize for numerical stability
        self.kriging_matrix[:n, :n] = regularize_matrix(
            self.kriging_matrix[:n, :n],
            epsilon=1e-10
        )

    def predict(
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        return_variance: bool = True,
        ) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.float64]]]:
        """
        Perform Ordinary Kriging prediction.

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

            # Augmented right-hand side: [γ(h), 1]ᵀ
            rhs = np.zeros(self.n_points + 1)
            rhs[:self.n_points] = gamma_vec
            rhs[self.n_points] = 1.0

            # Solve for weights and Lagrange multiplier
            try:
                solution = solve_kriging_system(self.kriging_matrix, rhs)
            except np.linalg.LinAlgError:
                # Fallback: use nearest neighbor
                nearest_idx = np.argmin(dist_to_samples)
                predictions[i] = self.z[nearest_idx]
                if return_variance:
                    variances[i] = np.var(self.z)
                continue

            weights = solution[:self.n_points]
            lagrange = solution[self.n_points]

            # Ordinary kriging prediction: ẑ(x₀) = Σλᵢz(xᵢ)
            predictions[i] = np.dot(weights, self.z)

            # Kriging variance: σ^2(x₀) = Σλᵢγ(xᵢ, x₀) + μ
            if return_variance:
                variances[i] = np.dot(weights, gamma_vec) + lagrange
                # Kriging variance should theoretically be non-negative
                # Negative values indicate numerical issues or invalid variogram
                if variances[i] < 0.0:
                    import warnings
                    warnings.warn(
                        f"Negative kriging variance {variances[i]:.6e} at prediction point {i}. "
                        "This may indicate numerical instability or an invalid variogram model. "
                        "Variance will be clamped to 0.",
                        RuntimeWarning
                    )
                    variances[i] = 0.0

        if return_variance:
            return predictions, variances
        else:
            return predictions

    def cross_validate(self) -> Tuple[npt.NDArray[np.float64], Dict[str, float]]:
        """
        Perform leave-one-out cross-validation

        Returns
        -------
        predictions : np.ndarray
            Cross-validated predictions at sample points
        metrics : dict
            Dictionary of validation metrics (MSE, RMSE, MAE, R-squared, bias)
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
            ok_temp = OrdinaryKriging(
                x_train,
                y_train,
                z_train,
                variogram_model=self.variogram_model,
            )

            # Predict at left-out point
            pred, _ = ok_temp.predict(
                np.array([self.x[i]]),
                np.array([self.y[i]]),
                return_variance=False,
            )
            predictions[i] = pred[0]

        # Calculate metrics
        metrics = cross_validation_score(self.z, predictions)

        return predictions, metrics

    def predict_block(
        self,
        x_block: Tuple[float, float],
        y_block: Tuple[float, float],
        discretization: int = 10,
    ) -> Tuple[float, float]:
        """
        Block kriging: predict average value over a block

        Parameters
        ----------
        x_block : tuple
            (x_min, x_max) of block
        y_block : tuple
            (y_min, y_max) of block
        discretization : int
            Number of points per dimension for discretization

        Returns
        -------
        prediction : float
            Predicted block average
        variance : float
            Block kriging variance
        """
        # Create grid of points within block
        x_grid = np.linspace(x_block[0], x_block[1], discretization)
        y_grid = np.linspace(y_block[0], y_block[1], discretization)
        X, Y = np.meshgrid(x_grid, y_grid)
        x_points = X.flatten()
        y_points = Y.flatten()

        # Predict at all discretization points
        predictions, variances = self.predict(x_points, y_points, return_variance=True)

        # Block average
        block_prediction = np.mean(predictions)
        block_variance = np.mean(variances)

        return block_prediction, block_variance
