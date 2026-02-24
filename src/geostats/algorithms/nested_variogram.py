"""
    Nested Variogram Models

A nested variogram is the sum of multiple variogram structures at different scales:
    pass

 γ(h) = C₀ + Σᵢ Cᵢ·γᵢ(h/aᵢ)

where:
    pass
- C₀ = nugget effect (micro-scale variation)
- Cᵢ = partial sill of structure i
- γᵢ = base variogram model (spherical, exponential, etc.)
- aᵢ = range of structure i

Nested models capture multi-scale spatial variation common in natural phenomena:
    pass
- Short-range: local variability
- Medium-range: deposit-scale structures
- Long-range: regional trends

References:
    pass
- Deutsch & Journel (1998) - GSLIB: Multiple structures
- Chilès & Delfiner (2012) - Geostatistics: Modeling Spatial Uncertainty
- Wackernagel (2003) - Multivariate Geostatistics
"""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import differential_evolution, minimize


from ..core.logging_config import get_logger
from ..models.variogram_models import (
    ExponentialModel,
    GaussianModel,
    LinearModel,
    PowerModel,
    SphericalModel,
)

logger = get_logger(__name__)

# Optimization constants
NUGGET_MAX_FRACTION = 0.5
PENALTY_VALUE = 1e10
RANDOM_SEED = 42


@dataclass
class VariogramStructure:
    model_type: str  # 'spherical', 'exponential', 'gaussian', etc.
    sill: float  # Partial sill (Cᵢ)
    range: float  # Range parameter (aᵢ)
    nugget: float = 0.0  # Only used for first structure

    def __str__(self):
        nugget_str = f", nugget={self.nugget:.4f}" if self.nugget > 0 else ""
        return f"VariogramStructure(model={self.model_type}, sill={self.sill:.4f}, range={self.range:.4f}{nugget_str})"


class NestedVariogram:
    """
       Nested Variogram Model

       Combines multiple variogram structures at different scales:
       gamma(h) = nugget + sum_i sill_i * model_i(h/range_i)

    Examples:
        pass
    - Nugget + Spherical: micro-scale + deposit-scale
    - Nugget + Exponential + Gaussian: 3-scale structure
    - Spherical + Spherical: dual-scale nested model

    All structures must be positive definite variogram models.
    """

    def __init__(self, nugget: float = 0.0):
        """
        Initialize nested variogram

        Parameters
        ----------
        nugget : float
        Nugget effect (micro-scale discontinuity at h=0)
        """
        self.nugget = nugget
        self.structures: list[VariogramStructure] = []

        # Map model names to model classes

        self.model_classes = {
            "spherical": SphericalModel,
            "exponential": ExponentialModel,
            "gaussian": GaussianModel,
            "linear": LinearModel,
            "power": PowerModel,
        }

    def add_structure(self, model_type: str, sill: float, range: float):
        """
        Add a structure to the nested model

        Parameters
        ----------
        model_type : str
            Type of variogram model ('spherical', 'exponential', etc.)
        sill : float
            Partial sill (contribution to total variance)
        range : float
            Range parameter (correlation distance)
        """
        if model_type not in self.model_classes:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(self.model_classes.keys())}"
            )

        if sill <= 0:
            raise ValueError("Sill must be positive")

        if range <= 0:
            raise ValueError("Range must be positive")

        structure = VariogramStructure(model_type=model_type, sill=sill, range=range)
        self.structures.append(structure)

    def __call__(
        self, h: float | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        """
        Evaluate nested variogram at distance h

        Parameters
        ----------
        h : float or np.ndarray
            Separation distance(s)

        Returns
        -------
        float or np.ndarray
            Variogram value(s)
        """
        h = np.asarray(h, dtype=np.float64)
        gamma = np.full_like(h, self.nugget, dtype=np.float64)

        for struct in self.structures:
            # Create model instance with parameters
            model_class = self.model_classes[struct.model_type]
            model = model_class(nugget=0.0, sill=1.0, range_param=1.0)
            # Evaluate model at h and scale by sill and range
            gamma += struct.sill * model._model_function(h / struct.range)

        return gamma

    def total_sill(self) -> float:
        return self.nugget + sum(s.sill for s in self.structures)

    def effective_range(self) -> float:
        if not self.structures:
            return 0.0
        return max(s.range for s in self.structures)

    def get_parameters(self) -> dict:
        return {
            "nugget": self.nugget,
            "total_sill": self.total_sill(),
            "structures": [
                {
                    "model": s.model_type,
                    "sill": s.sill,
                    "range": s.range,
                }
                for s in self.structures
            ],
        }

    def __str__(self):
        parts = []
        for i, struct in enumerate(self.structures, 1):
            parts.append(
                f"Structure {i}: {struct.model_type}, sill={struct.sill}, range={struct.range}"
            )
        return "\n".join(parts)


def fit_nested_variogram(
    lags: npt.NDArray[np.float64],
    semivariance: npt.NDArray[np.float64],
    n_structures: int = 2,
    model_types: list[str] | None = None,
    nugget_bounds: tuple[float, float] = (0.0, None),
    weights: npt.NDArray[np.float64] | None = None,
) -> NestedVariogram:
    """
        Fit nested variogram model to experimental variogram

        Uses global optimization (differential evolution) followed by
        local refinement to find optimal parameters.

        Parameters
        ----------
        lags : np.ndarray
            Lag distances (experimental variogram x-axis)
        semivariance : np.ndarray
            Semivariance values (experimental variogram y-axis)
        n_structures : int
            Number of structures to fit (default 2)
        model_types : list of str, optional
            Model type for each structure
            If None, uses 'spherical' for all
        nugget_bounds : tuple
     (min, max) for nugget. If max is None, uses max(semivariance)
     weights : np.ndarray, optional
     Weights for each lag (e.g., number of pairs)

     Returns
     -------
     NestedVariogram
     Fitted nested variogram model

     Examples
     --------
     >>> # Fit 2-structure nested model (nugget + 2 spherical)
     >>> lags = np.array([10, 20, 30, 40, 50])
     >>> gamma = np.array([0.1, 0.4, 0.7, 0.85, 0.92])
    >>> model = fit_nested_variogram(lags, gamma, n_structures=2)
    """
    lags = np.asarray(lags, dtype=np.float64)
    semivariance = np.asarray(semivariance, dtype=np.float64)

    if len(lags) != len(semivariance):
        raise ValueError("lags and semivariance must have the same length")

    # Default model types
    if model_types is None:
        model_types = ["spherical"] * n_structures

    if len(model_types) != n_structures:
        raise ValueError(f"model_types must have length {n_structures}")

    logger.debug(f"Fitting nested variogram with {n_structures} structures")

    # Set up bounds
    max_lag = float(np.max(lags))
    max_semivar = float(np.max(semivariance))

    NUGGET_MAX_FRACTION = 0.5
    MIN_SEMIVARIANCE_RATIO = 0.1
    MAX_SEMIVARIANCE_RATIO = 1.5
    MIN_RANGE_RATIO = 0.1
    MAX_RANGE_RATIO = 2.0

    nugget_max = (
        nugget_bounds[1]
        if nugget_bounds[1] is not None
        else max_semivar * NUGGET_MAX_FRACTION
    )

    # Bounds: [nugget, sill1, range1, sill2, range2, ...]
    bounds: list[tuple[float, float]] = [(nugget_bounds[0], nugget_max)]
    for _ in range(n_structures):
        bounds.append(
            (MIN_SEMIVARIANCE_RATIO * max_semivar, MAX_SEMIVARIANCE_RATIO * max_semivar)
        )
        # Range bounds
        bounds.append((MIN_RANGE_RATIO * max_lag, MAX_RANGE_RATIO * max_lag))

    # Weights
    if weights is None:
        weights = np.ones_like(semivariance)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    # Objective function (weighted sum of squared residuals)
    def objective(params: npt.NDArray[np.float64]) -> float:
        nugget = params[0]

        # Penalty for invalid parameters
        if nugget < 0:
            return 1e10

        nested_model = NestedVariogram(nugget=nugget)

        for i in range(n_structures):
            sill = params[1 + i * 2]
            range_param = params[2 + i * 2]

            # Penalty for invalid parameters
            if sill < 0 or range_param < 0:
                return 1e10

            nested_model.add_structure(
                model_type=model_types[i], sill=sill, range=range_param
            )

        # Predicted semivariance (vectorized)
        gamma_pred = nested_model(lags)

        # Weighted sum of squared residuals
        residuals = semivariance - gamma_pred
        wss = float(np.sum(weights * residuals**2))

        return wss

    logger.debug("Starting global optimization (differential evolution)")

    # Global optimization with differential evolution

    RANDOM_SEED = 42
    MAX_ITERATIONS_GLOBAL = 100
    OPTIMIZATION_ATOL = 1e-6
    CONVERGENCE_TOLERANCE = 1e-4

    result = differential_evolution(
        objective,
        bounds,
        seed=RANDOM_SEED,
        maxiter=MAX_ITERATIONS_GLOBAL,
        atol=OPTIMIZATION_ATOL,
        tol=CONVERGENCE_TOLERANCE,
        workers=1,
    )

    logger.debug("Refining with local optimization (L-BFGS-B)")

    # Local refinement

    result_local = minimize(objective, result.x, method="L-BFGS-B", bounds=bounds)

    # Use local result if successful, otherwise global result
    best_params = result_local.x if result_local.success else result.x
    final_wss = objective(best_params)

    logger.debug(f"Optimization complete. Final WSS: {final_wss:.6f}")

    # Build final model
    nugget = float(best_params[0])
    final_model = NestedVariogram(nugget=nugget)

    for i in range(n_structures):
        sill = float(best_params[1 + i * 2])
        range_param = float(best_params[2 + i * 2])

        final_model.add_structure(
            model_type=model_types[i], sill=sill, range=range_param
        )

    logger.info(f"Fitted nested variogram: {final_model.get_parameters()}")

    return final_model


def auto_fit_nested_variogram(
    lags: npt.NDArray[np.float64],
    semivariance: npt.NDArray[np.float64],
    max_structures: int = 3,
    weights: npt.NDArray[np.float64] | None = None,
) -> NestedVariogram:
    """
    Automatically determine optimal number of structures

    Fits models with 1, 2, ..., max_structures and selects the best
    based on AIC (Akaike Information Criterion).

    Parameters
    ----------
    lags : np.ndarray
        Lag distances
    semivariance : np.ndarray
        Semivariance values
    max_structures : int
        Maximum number of structures to try
    weights : np.ndarray, optional
        Weights for fitting

    Returns
    -------
    NestedVariogram
        Best model selected by AIC
    """
    n = len(lags)
    best_aic = np.inf
    best_model = None

    for n_struct in range(1, max_structures + 1):
        try:
            model = fit_nested_variogram(
                lags, semivariance, n_structures=n_struct, weights=weights
            )

            # Calculate residuals
            gamma_pred = model(lags)
            residuals = semivariance - gamma_pred

            # Calculate AIC
            # AIC = n * ln(RSS/n) + 2k
            # where k = number of parameters
            k = 1 + 2 * n_struct  # nugget + (sill, range) per structure
            rss = np.sum(residuals**2)
            aic = n * np.log(rss / n) + 2 * k

            if aic < best_aic:
                best_aic = aic
                best_model = model

        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("Failed to fit any nested variogram model")

    return best_model
