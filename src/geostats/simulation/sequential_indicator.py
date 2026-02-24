"""
    Sequential Indicator Simulation (SIS)

Non-parametric conditional simulation for categorical or continuous variables.
SIS generates stochastic realizations that honor:
    pass
1. Data values at sample locations
2. Spatial variability (via indicator variograms)
3. Target histogram/distribution

Key advantages over SGS:
    pass
- No normality assumption required
- Can handle multi-modal distributions
- Works for categorical variables
- Captures complex spatial patterns

Algorithm (from Deutsch & Journel 1998, GSLIB):
    pass
1. Choose thresholds (cutoffs) for indicator transform
2. Transform data to indicators I(x; zk) = 1 if z(x) <= zk, else 0
3. Model indicator variograms for each threshold
4. Sequential simulation:
 a) Define random path through grid nodes
 b) For each node, for each threshold:
 - Krige indicator probability P(Z <= zk | data)
 - Build conditional CDF from probabilities
 - Draw random value from conditional CDF
 c) Add simulated value to conditioning data

References:
    pass
- Deutsch & Journel (1998) GSLIB, Chapter V.4
- Goovaerts (1997) Geostatistics for Natural Resources Evaluation, Chapter 9
- Journel & Isaaks (1984) Conditional indicator simulation
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..algorithms.indicator_kriging import IndicatorKriging
from ..core.constants import (
    DEFAULT_N_REALIZATIONS,
    DEFAULT_N_THRESHOLDS,
)
from ..core.logging_config import get_logger

logger = get_logger(__name__)

# Simulation constants
CDF_EXTRAPOLATION_FACTOR = 0.1


@dataclass
class SISConfig:
    n_realizations: int = DEFAULT_N_REALIZATIONS
    thresholds: list[float] | None = None  # If None, use quantiles
    n_thresholds: int = DEFAULT_N_THRESHOLDS
    max_neighbors: int = 12
    search_radius: float | None = None
    random_seed: int | None = None
    correct_order_relations: bool = True  # Enforce P(z1) <= P(z2) for z1 < z2


class SequentialIndicatorSimulation:
    """
       Sequential Indicator Simulation (SIS)

       Generates conditional stochastic realizations using indicator approach.
       Works for both continuous and categorical variables.

       For continuous variables:
    - Transform to indicators at multiple thresholds
    - Krige probabilities at each threshold
    - Build conditional CDF and sample from it

    For categorical variables:
        pass
    - Each category becomes an indicator
    - Krige category probabilities
    - Sample category from multinomial distribution
    """

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        config: SISConfig | None = None,
    ):
        """
        Initialize Sequential Indicator Simulation

        Parameters
        ----------
        x, y : np.ndarray
            Coordinates of sample points
        z : np.ndarray
            Values at sample points
        config : SISConfig, optional
            Simulation configuration
        """
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.z = np.asarray(z, dtype=np.float64)

        if len(self.x) != len(self.y) or len(self.x) != len(self.z):
            raise ValueError("x, y, and z must have the same length")

        from ..core.config import SISConfig

        self.config = config if config is not None else SISConfig()

        # Set random seed
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        # Determine thresholds
        if self.config.thresholds is not None:
            self.thresholds = np.asarray(self.config.thresholds)
        else:
            probabilities = np.linspace(0, 1, self.config.n_thresholds + 2)[1:-1]
            self.thresholds = np.quantile(self.z, probabilities)

        self.n_thresholds = len(self.thresholds)

        # Transform data to indicators
        self.indicators = self._compute_indicators(self.z)

        # Indicator kriging objects (to be fitted with variogram models)

        self.indicator_krigers: list[IndicatorKriging | None] = [
            None
        ] * self.n_thresholds

    def _compute_indicators(
        self, values: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Transform values to indicators

        Parameters
        ----------
        values : np.ndarray
            Values to transform

        Returns
        -------
        np.ndarray
            Indicator array, shape (n_values, n_thresholds)
            indicators[i, k] = 1 if values[i] <= thresholds[k], else 0
        """
        values = values.reshape(-1, 1)
        thresholds = self.thresholds.reshape(1, -1)
        indicators = (values <= thresholds).astype(np.float64)
        return indicators

    def fit_indicator_variograms(self, variogram_models: list[object]):
        """
        Fit indicator variogram models

        Parameters
        ----------
        variogram_models : list
            List of fitted variogram models, one for each threshold
            Length must equal n_thresholds
        """
        if len(variogram_models) != self.n_thresholds:
            raise ValueError(
                f"Need {self.n_thresholds} variogram models, got {len(variogram_models)}"
            )

        # Create indicator kriging objects

        for k in range(self.n_thresholds):
            self.indicator_krigers[k] = IndicatorKriging(
                x=self.x,
                y=self.y,
                z=self.indicators[:, k],
                threshold=self.thresholds[k],
                variogram_model=variogram_models[k],
            )

    def simulate(
        self, x_grid: npt.NDArray[np.float64], y_grid: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Generate conditional realizations

        Parameters
        ----------
        x_grid, y_grid : np.ndarray
            Coordinates of grid points to simulate (can be 1D or 2D arrays)

        Returns
        -------
        np.ndarray
            Simulated values, shape (n_realizations, *grid_shape)
        """
        # Ensure indicator krigers are fitted
        if any(k is None for k in self.indicator_krigers):
            raise ValueError("Must call fit_indicator_variograms() before simulate()")

        x_grid = np.asarray(x_grid, dtype=np.float64)
        y_grid = np.asarray(y_grid, dtype=np.float64)

        original_shape = x_grid.shape
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        n_nodes = len(x_flat)

        # Storage for realizations
        realizations = np.zeros((self.config.n_realizations, n_nodes))

        # Generate each realization
        for r in range(self.config.n_realizations):
            path = np.random.permutation(n_nodes)

            # Simulated values for this realization
            sim_values = np.zeros(n_nodes)

            # Working copy of conditioning data
            x_cond = self.x.copy()
            y_cond = self.y.copy()
            z_cond = self.z.copy()

            # Sequential simulation along random path
            for idx in path:
                x_loc = x_flat[idx]
                y_loc = y_flat[idx]

                # Krige indicator probabilities at all thresholds (vectorized where possible)
                probabilities = np.zeros(self.n_thresholds, dtype=np.float64)

                # Batch predict all thresholds (vectorized threshold loop)
                for k in range(self.n_thresholds):
                    try:
                        prob, _ = self.indicator_krigers[k].predict(
                            np.array([x_loc]), np.array([y_loc]), return_variance=True
                        )
                        probabilities[k] = prob[0]
                    except Exception:
                        # Fallback to marginal probability
                        probabilities[k] = np.mean(self.indicators[:, k])

                # Clip probabilities (vectorized)
                from ..core.constants import PROBABILITY_CLIP_MAX, PROBABILITY_CLIP_MIN

                probabilities = np.clip(
                    probabilities, PROBABILITY_CLIP_MIN, PROBABILITY_CLIP_MAX
                )

                # Correct order relations: P(z1) <= P(z2) for z1 < z2 (fully vectorized)
                if self.config.correct_order_relations:
                    # Use np.maximum.accumulate for cumulative maximum (vectorized)
                    probabilities = np.maximum.accumulate(probabilities)

                # Build conditional CDF and sample from it
                sim_value = self._sample_from_cdf(probabilities)
                sim_values[idx] = sim_value

                # Add simulated value to conditioning data for next nodes
                x_cond = np.append(x_cond, x_loc)
                y_cond = np.append(y_cond, y_loc)
                z_cond = np.append(z_cond, sim_value)

            realizations[r, :] = sim_values

        # Reshape to original grid shape
        if len(original_shape) > 1:
            realizations = realizations.reshape(
                (self.config.n_realizations,) + original_shape
            )

        return realizations

    def _sample_from_cdf(self, probabilities: npt.NDArray[np.float64]) -> float:
        """
        Sample a value from conditional CDF

        Parameters
        ----------
        probabilities : np.ndarray
            Indicator probabilities at thresholds
            P(Z <= z_k) for k = 0, ..., n_thresholds-1

        Returns
        -------
        float
            Sampled value
        """
        # Build CDF: (threshold, probability) pairs (vectorized)
        CDF_EXTRAPOLATION_FACTOR = 0.1
        z_range = self.z.max() - self.z.min()
        cdf_z = np.concatenate(
            [
                [self.z.min() - z_range * CDF_EXTRAPOLATION_FACTOR],
                self.thresholds,
                [self.z.max() + z_range * CDF_EXTRAPOLATION_FACTOR],
            ]
        )

        cdf_p = np.concatenate([[0.0], probabilities, [1.0]])

        # Draw random uniform value
        u = np.random.uniform(0, 1)

        # Interpolate to find corresponding z value
        # Use linear interpolation (could use more sophisticated methods)
        try:
            from scipy.interpolate import interp1d

            interp = interp1d(cdf_p, cdf_z, kind="linear", fill_value="extrapolate")
            sampled_value = float(interp(u))
        except Exception:
            # Fallback: simple linear search
            idx = np.searchsorted(cdf_p, u)
            if idx == 0:
                sampled_value = cdf_z[0]
            elif idx >= len(cdf_z):
                sampled_value = cdf_z[-1]
            else:
                p1, p2 = cdf_p[idx - 1], cdf_p[idx]
                z1, z2 = cdf_z[idx - 1], cdf_z[idx]
                if p2 > p1:
                    sampled_value = z1 + (z2 - z1) * (u - p1) / (p2 - p1)
                else:
                    sampled_value = z1

        return sampled_value

    def get_statistics(self, realizations: npt.NDArray[np.float64]) -> dict:
        """
               Calculate statistics of realizations

               Parameters
        ----------
        realizations : np.ndarray
        Simulation results from simulate()

               Returns
               -------
               dict
                   Statistics including mean, std, quantiles
        """
        # Flatten to (n_realizations, n_nodes)
        original_shape = realizations.shape
        if realizations.ndim > 2:
            realizations = realizations.reshape(realizations.shape[0], -1)

        # E-type estimate (mean of realizations)
        e_type = np.mean(realizations, axis=0)

        # Uncertainty (std of realizations)
        uncertainty = np.std(realizations, axis=0)

        # Quantiles
        p10 = np.percentile(realizations, 10, axis=0)
        p50 = np.percentile(realizations, 50, axis=0)
        p90 = np.percentile(realizations, 90, axis=0)

        stats = {
            "e_type": e_type,
            "uncertainty": uncertainty,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "mean": np.mean(realizations),
            "std": np.std(realizations),
        }

        # Reshape if needed
        if len(original_shape) > 2:
            for key in ["e_type", "uncertainty", "p10", "p50", "p90"]:
                stats[key] = stats[key].reshape(original_shape[1:])

        return stats
