"""
    Pipeline runner for config-driven geostatistical analysis

Orchestrates the complete workflow from data loading to output generation.
"""

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..algorithms.cokriging import Cokriging
from ..algorithms.indicator_kriging import IndicatorKriging
from ..algorithms.ordinary_kriging import OrdinaryKriging
from ..algorithms.simple_kriging import SimpleKriging
from ..algorithms.universal_kriging import UniversalKriging
from ..config.schemas import AnalysisConfig
from ..core.exceptions import GeoStatsError
from ..diagnostics.outlier_detection import outlier_analysis
from ..visualization.minimal_style import set_minimalist_rcparams


class PipelineError(GeoStatsError):
    pass


class AnalysisPipeline:
    """
       Config-driven geostatistical analysis pipeline

       Executes complete workflow based on configuration:
       1. Data loading and validation
    2. Preprocessing (outliers, transforms, declustering)
    3. Variogram modeling
    4. Kriging prediction
    5. Validation
    6. Visualization
    7. Output generation

    Parameters
    ----------
    config : AnalysisConfig
    Complete analysis configuration

    Examples
    --------
    >>> from geostats.config import load_config
    >>> config = load_config('analysis.yaml')
    >>> pipeline = AnalysisPipeline(config)
    >>> pipeline.run()
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.output_dir = Path(config.project.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline state
        self.data = None
        self.x = None
        self.y = None
        self.z = None
        self.z_secondary = None
        self.weights = None
        self.transform = None
        self.variogram_model = None
        self.kriging_model = None
        self.predictions = None
        self.variance = None
        self.cv_results = None
        self.start_time: datetime | None = None

        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("geostats.pipeline")
        logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)

        # File handler
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(logging.DEBUG)
            detailed_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)

        return logger

    def run(self):
        self.logger.info(f"=== Starting analysis: {self.config.project.name} ===")
        self.start_time = datetime.now()
        start_time = self.start_time

        try:
            if self.config.random_seed is not None:
                np.random.seed(self.config.random_seed)
                self.logger.info(f"Random seed set to: {self.config.random_seed}")

            # Set visualization style
            if self.config.visualization.style == "minimalist":
                set_minimalist_rcparams()

            # Execute pipeline steps
            self.load_data()
            self.preprocess_data()
            self.model_variogram()
            self.perform_kriging()

            if self.config.validation.cross_validation:
                self.cross_validate()

            self.visualize()
            self.save_outputs()

            # Summary
            elapsed = datetime.now() - start_time
            self.logger.info(f"=== Analysis complete in {elapsed} ===")

            if self.config.output.save_report:
                self.generate_report()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise PipelineError(f"Pipeline execution failed: {e}")

    def load_data(self):
        self.logger.info("Loading data...")

        # Read data - handle both CSV and TXT files
        try:
            file_path = self.config.data.input_file
            if file_path.endswith(".csv"):
                self.data = pd.read_csv(file_path, low_memory=False)
            else:
                for encoding in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
                    try:
                        self.data = pd.read_csv(
                            file_path, sep="\t", low_memory=False, encoding=encoding
                        )
                        self.logger.info(
                            f"Successfully loaded with {encoding} encoding"
                        )
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise PipelineError(
                        f"Could not decode file {file_path} with any encoding"
                    )

            self.logger.info(f"Loaded {len(self.data)} records from {file_path}")
        except Exception as e:
            raise PipelineError(f"Failed to load data: {e}")

        # Apply filters
        if self.config.data.filter_column and self.config.data.filter_value is not None:
            mask = (
                self.data[self.config.data.filter_column]
                == self.config.data.filter_value
            )
            self.data = self.data[mask].copy()
            self.logger.info(
                f"Filtered to {len(self.data)} records where {self.config.data.filter_column}={self.config.data.filter_value}"
            )

        # Extract coordinates and values
        try:
            for col_name, col_config in [
                ("X", self.config.data.x_column),
                ("Y", self.config.data.y_column),
                ("Value", self.config.data.z_column),
            ]:
                if col_config not in self.data.columns:
                    raise KeyError(
                        f"{col_name} column '{col_config}' not found. Available columns: {list(self.data.columns[:20])}"
                    )

            self.x = pd.to_numeric(
                self.data[self.config.data.x_column], errors="coerce"
            ).values
            self.y = pd.to_numeric(
                self.data[self.config.data.y_column], errors="coerce"
            ).values
            self.z = pd.to_numeric(
                self.data[self.config.data.z_column], errors="coerce"
            ).values

            if self.config.data.z_secondary:
                self.z_secondary = pd.to_numeric(
                    self.data[self.config.data.z_secondary], errors="coerce"
                ).values
            else:
                self.z_secondary = None

        except KeyError as e:
            raise PipelineError(f"Column not found in data: {e}")

        # Remove NaNs
        mask = ~(np.isnan(self.x) | np.isnan(self.y) | np.isnan(self.z))
        if self.z_secondary is not None:
            mask = mask & ~np.isnan(self.z_secondary)

        n_removed = (~mask).sum()
        if n_removed > 0:
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.z = self.z[mask]
            if self.z_secondary is not None:
                self.z_secondary = self.z_secondary[mask]
            self.data = self.data[mask].reset_index(drop=True)

        if len(self.z) == 0:
            raise PipelineError("No valid data points after removing NaNs")

        self.logger.info(f"Final dataset: {len(self.z)} valid points")
        self.logger.info(f"Value range: [{self.z.min():.4f}, {self.z.max():.4f}]")
        self.logger.info(f"Value mean: {self.z.mean():.4f}, std: {self.z.std():.4f}")

    def preprocess_data(self):
        self.logger.info("Preprocessing data...")

        # Outlier removal
        if self.config.preprocessing.remove_outliers:
            results = outlier_analysis(
                self.x,
                self.y,
                self.z,
                method=self.config.preprocessing.outlier_method,
                threshold=self.config.preprocessing.outlier_threshold,
            )
            outlier_indices = results["outlier_indices"]
            n_outliers = len(outlier_indices)

            if n_outliers > 0:
                mask = np.ones(len(self.z), dtype=bool)
                mask[outlier_indices] = False
                self.x = self.x[mask]
                self.y = self.y[mask]
                self.z = self.z[mask]
                if self.z_secondary is not None:
                    self.z_secondary = self.z_secondary[mask]
                self.logger.info(f"Removed {n_outliers} outliers")

        # Handle negative values if needed
        if self.config.preprocessing.transform in ["log", "boxcox"]:
            self.logger.warning(f"Found {(self.z <= 0).sum()} non-positive values")
            if self.config.preprocessing.handle_negatives == "shift":
                shift = abs(self.z.min()) + 1e-6
                self.z = self.z + shift
                self.logger.info(f"Shifted data by {shift:.6f}")
            elif self.config.preprocessing.handle_negatives == "remove":
                mask = self.z > 0
                self.x = self.x[mask]
                self.y = self.y[mask]
                self.z = self.z[mask]
                self.logger.info(f"Removed {(~mask).sum()} non-positive values")
            elif self.config.preprocessing.handle_negatives == "absolute":
                self.z = np.abs(self.z)
                self.logger.info("Took absolute values")

        # Data transformation
        if self.config.preprocessing.transform:
            if self.config.preprocessing.transform == "log":
                from sklearn.preprocessing import FunctionTransformer

                self.transform = FunctionTransformer(np.log, np.exp)
                self.z = self.transform.fit_transform(self.z.reshape(-1, 1)).flatten()

            elif self.config.preprocessing.transform == "boxcox":
                from scipy.stats import boxcox

                self.z, self.transform = boxcox(self.z + 1e-6)
                self.logger.info(f"Box-Cox lambda: {self.transform:.4f}")

            elif self.config.preprocessing.transform == "normal_score":
                from ..simulation.gaussian_simulation import normal_score_transform

                self.z, self.transform = normal_score_transform(self.z)

            elif self.config.preprocessing.transform == "sqrt":
                self.z = np.sqrt(self.z)
                self.transform = (
                    None  # Mark that we did transform but no inverse needed
                )

        # Declustering
        if self.config.preprocessing.declustering:
            from ..algorithms.declustering import decluster

            if self.config.preprocessing.declustering_method == "cell":
                self.weights, info = decluster(
                    self.x,
                    self.y,
                    self.z,
                    method="cell",
                    cell_size=self.config.preprocessing.declustering_cell_size,
                )
                self.logger.info(
                    f"Optimal cell size: {info.get('optimal_cell_size', 'N/A')}"
                )
            else:
                self.weights, info = decluster(
                    self.x,
                    self.y,
                    self.z,
                    method=self.config.preprocessing.declustering_method,
                )
            self.logger.info(
                f"Weight range: [{self.weights.min():.4f}, {self.weights.max():.4f}]"
            )
        else:
            self.weights = np.ones(len(self.z))

    def model_variogram(self):
        self.logger.info("Modeling variogram...")

        if self.x is None or self.y is None or self.z is None:
            raise PipelineError("Data not loaded for variogram modeling")

        # Convert to numpy arrays
        x_arr = np.asarray(self.x, dtype=np.float64)
        y_arr = np.asarray(self.y, dtype=np.float64)
        z_arr = np.asarray(self.z, dtype=np.float64)

        # Compute experimental variogram
        max_lag = self.config.variogram.max_lag
        if max_lag is None:
            dx = float(x_arr.max() - x_arr.min())
            dy = float(y_arr.max() - y_arr.min())
            max_lag = np.sqrt(dx**2 + dy**2) / 3
            self.logger.info(f"Auto max_lag: {max_lag:.2f}")

        # Import functions at module level
        from ..algorithms.variogram import experimental_variogram, madogram

        # Use specified estimator
        estimator = self.config.variogram.estimator
        if estimator == "matheron":
            lags, gamma, n_pairs = experimental_variogram(
                x_arr,
                y_arr,
                z_arr,
                n_lags=self.config.variogram.n_lags,
                maxlag=max_lag,
            )
        elif estimator == "cressie":
            lags, gamma, n_pairs = experimental_variogram(
                x_arr,
                y_arr,
                z_arr,
                n_lags=self.config.variogram.n_lags,
                maxlag=max_lag,
                estimator="cressie",
            )
        elif estimator == "dowd":
            lags, gamma, n_pairs = experimental_variogram(
                x_arr,
                y_arr,
                z_arr,
                n_lags=self.config.variogram.n_lags,
                maxlag=max_lag,
                estimator="dowd",
            )
        elif estimator == "madogram":
            lags, gamma, n_pairs = madogram(
                x_arr,
                y_arr,
                z_arr,
                n_lags=self.config.variogram.n_lags,
                maxlag=max_lag,
            )
        else:
            lags, gamma, n_pairs = experimental_variogram(
                x_arr,
                y_arr,
                z_arr,
                n_lags=self.config.variogram.n_lags,
                maxlag=max_lag,
            )

        self.logger.info(f"Experimental variogram: {len(lags)} lags")

        # Fit model
        if self.config.variogram.auto_fit:
            best_model = None
            best_score = (
                float("inf")
                if self.config.variogram.fit_criterion != "r2"
                else float("-inf")
            )

            # Model factory mapping (Pythonic dictionary-based dispatch)
            from ..models.variogram_models import (
                CubicModel,
                ExponentialModel,
                GaussianModel,
                LinearModel,
                MaternModel,
                SphericalModel,
                StableModel,
            )

            model_factory = {
                "spherical": SphericalModel,
                "exponential": ExponentialModel,
                "gaussian": GaussianModel,
                "matern": MaternModel,
                "cubic": CubicModel,
                "stable": StableModel,
                "linear": LinearModel,
            }

            # Score computation mapping
            def compute_rmse(gamma, gamma_pred):
                return np.sqrt(np.mean((gamma - gamma_pred) ** 2))

            def compute_r2(gamma, gamma_pred):
                ss_res = np.sum((gamma - gamma_pred) ** 2)
                ss_tot = np.sum((gamma - gamma.mean()) ** 2)
                return 1 - ss_res / ss_tot if ss_tot > 0 else 0

            score_computers = {
                "rmse": lambda g, gp: compute_rmse(g, gp),
                "r2": lambda g, gp: compute_r2(g, gp),
            }

            is_maximizing = self.config.variogram.fit_criterion == "r2"
            score_func = score_computers.get(
                self.config.variogram.fit_criterion, lambda g, gp: compute_rmse(g, gp)
            )

            for model_type in self.config.variogram.models:
                if model_type not in model_factory:
                    continue

                try:
                    # Create model instance using factory
                    model = model_factory[model_type]()

                    # Fit the model
                    from ..algorithms.fitting import fit_variogram_model

                    fitted_model = fit_variogram_model(model, lags, gamma)

                    # Compute fit quality (vectorized)
                    gamma_pred = fitted_model(lags)
                    score = score_func(gamma, gamma_pred)
                    is_better = (
                        (score > best_score) if is_maximizing else (score < best_score)
                    )

                    self.logger.info(
                        f"  {model_type}: {self.config.variogram.fit_criterion}={score:.4f}"
                    )

                    if is_better:
                        best_score = score
                        best_model = fitted_model

                except Exception as e:
                    self.logger.warning(f"  {model_type}: fitting failed ({e})")

            if best_model is None:
                raise PipelineError("No variogram model could be fitted")

            self.variogram_model = best_model
            model_name = best_model.__class__.__name__.replace("Model", "").lower()
            self.logger.info(f"Selected model: {model_name}")
            params = best_model._parameters  # type: ignore
            self.logger.info(f"  Nugget: {params.get('nugget', 0):.4f}")
            self.logger.info(f"  Sill: {params.get('sill', 0):.4f}")
            self.logger.info(f"  Range: {params.get('range', 0):.4f}")

        else:
            if not self.config.variogram.manual_model:
                raise PipelineError(
                    "manual_model must be specified when auto_fit=False"
                )

            if (
                self.config.variogram.manual_nugget is None
                or self.config.variogram.manual_sill is None
                or self.config.variogram.manual_range is None
            ):
                raise PipelineError(
                    "manual_nugget, manual_sill, and manual_range must be specified when auto_fit=False"
                )

            # Create model with manual parameters using factory pattern
            from ..models.variogram_models import (
                CubicModel,
                ExponentialModel,
                GaussianModel,
                LinearModel,
                MaternModel,
                SphericalModel,
                StableModel,
            )

            model_factory = {
                "spherical": SphericalModel,
                "exponential": ExponentialModel,
                "gaussian": GaussianModel,
                "matern": MaternModel,
                "cubic": CubicModel,
                "stable": StableModel,
                "linear": LinearModel,
            }

            model_type = self.config.variogram.manual_model.lower()
            if model_type not in model_factory:
                raise PipelineError(
                    f"Unknown model type: {model_type}. Available: {list(model_factory.keys())}"
                )

            ModelClass = model_factory[model_type]
            self.variogram_model = ModelClass(
                nugget=self.config.variogram.manual_nugget,
                sill=self.config.variogram.manual_sill,
                range_param=self.config.variogram.manual_range,
            )

            self.logger.info(f"Using manual model: {model_type}")
            params = self.variogram_model._parameters  # type: ignore
            self.logger.info(f"  Nugget: {params.get('nugget', 0):.4f}")
            self.logger.info(f"  Sill: {params.get('sill', 0):.4f}")
            self.logger.info(f"  Range: {params.get('range', 0):.4f}")

    def perform_kriging(self):
        self.logger.info(f"Performing {self.config.kriging.method} kriging...")

        # Create prediction grid
        grid_cfg = self.config.kriging.grid

        if self.x is None or self.y is None:
            raise PipelineError("Data not loaded for grid creation")

        # Convert to numpy arrays
        x_arr = np.asarray(self.x, dtype=np.float64)
        y_arr = np.asarray(self.y, dtype=np.float64)

        x_min = (
            grid_cfg.x_min
            if grid_cfg.x_min is not None
            else float(x_arr.min() - grid_cfg.buffer)
        )
        x_max = (
            grid_cfg.x_max
            if grid_cfg.x_max is not None
            else float(x_arr.max() + grid_cfg.buffer)
        )
        y_min = (
            grid_cfg.y_min
            if grid_cfg.y_min is not None
            else float(y_arr.min() - grid_cfg.buffer)
        )
        y_max = (
            grid_cfg.y_max
            if grid_cfg.y_max is not None
            else float(y_arr.max() + grid_cfg.buffer)
        )

        if grid_cfg.nx and grid_cfg.ny:
            nx = grid_cfg.nx
            ny = grid_cfg.ny
        else:
            nx = int((x_max - x_min) / grid_cfg.resolution)
            ny = int((y_max - y_min) / grid_cfg.resolution)

        grid_x = np.linspace(x_min, x_max, nx)
        grid_y = np.linspace(y_min, y_max, ny)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

        self.logger.info(f"Prediction grid: {nx} x {ny} = {nx * ny} points")

        # Initialize kriging model using factory pattern (Pythonic dispatch)

        def create_ordinary_kriging():
            if self.x is None or self.y is None or self.z is None:
                raise PipelineError("Data not loaded")
            x_arr = np.asarray(self.x, dtype=np.float64)
            y_arr = np.asarray(self.y, dtype=np.float64)
            z_arr = np.asarray(self.z, dtype=np.float64)
            return OrdinaryKriging(
                x=x_arr, y=y_arr, z=z_arr, variogram_model=self.variogram_model
            )

        def create_simple_kriging():
            if self.x is None or self.y is None or self.z is None:
                raise PipelineError("Data not loaded")
            x_arr = np.asarray(self.x, dtype=np.float64)
            y_arr = np.asarray(self.y, dtype=np.float64)
            z_arr = np.asarray(self.z, dtype=np.float64)
            return SimpleKriging(
                x=x_arr,
                y=y_arr,
                z=z_arr,
                variogram_model=self.variogram_model,
                mean=float(z_arr.mean()),
            )

        def create_universal_kriging():
            if self.x is None or self.y is None or self.z is None:
                raise PipelineError("Data not loaded")
            x_arr = np.asarray(self.x, dtype=np.float64)
            y_arr = np.asarray(self.y, dtype=np.float64)
            z_arr = np.asarray(self.z, dtype=np.float64)
            return UniversalKriging(
                x=x_arr,
                y=y_arr,
                z=z_arr,
                variogram_model=self.variogram_model,
                drift_terms=self.config.kriging.drift_terms,
            )

        def create_indicator_kriging():
            if self.x is None or self.y is None or self.z is None:
                raise PipelineError("Data not loaded")
            x_arr = np.asarray(self.x, dtype=np.float64)
            y_arr = np.asarray(self.y, dtype=np.float64)
            z_arr = np.asarray(self.z, dtype=np.float64)
            thresholds = self.config.kriging.thresholds
            if not thresholds or len(thresholds) == 0:
                raise PipelineError("Indicator kriging requires at least one threshold")
            # Use first threshold for single-threshold indicator kriging
            return IndicatorKriging(
                x=x_arr,
                y=y_arr,
                z=z_arr,
                variogram_model=self.variogram_model,
                threshold=float(thresholds[0]),
            )

        def create_cokriging():
            if self.x is None or self.y is None or self.z is None:
                raise PipelineError("Data not loaded")
            if self.z_secondary is None:
                raise PipelineError(
                    "Cokriging requires secondary variable (z_secondary)"
                )
            x_arr = np.asarray(self.x, dtype=np.float64)
            y_arr = np.asarray(self.y, dtype=np.float64)
            z_primary_arr = np.asarray(self.z, dtype=np.float64)
            z_secondary_arr = np.asarray(self.z_secondary, dtype=np.float64)
            return Cokriging(
                x_primary=x_arr,
                y_primary=y_arr,
                z_primary=z_primary_arr,
                x_secondary=x_arr,  # Assuming same locations
                y_secondary=y_arr,
                z_secondary=z_secondary_arr,
                variogram_primary=self.variogram_model,
                variogram_secondary=self.variogram_model,
            )

        kriging_factory = {
            "ordinary": create_ordinary_kriging,
            "simple": create_simple_kriging,
            "universal": create_universal_kriging,
            "indicator": create_indicator_kriging,
            "cokriging": create_cokriging,
        }

        method = self.config.kriging.method
        if method not in kriging_factory:
            raise PipelineError(
                f"Unknown kriging method: {method}. Available: {list(kriging_factory.keys())}"
            )

        self.kriging_model = kriging_factory[method]()

        # Predict
        predictions, variance = self.kriging_model.predict(
            grid_xx.ravel(), grid_yy.ravel(), return_variance=True
        )

        # Reshape to grid
        if predictions is not None:
            self.predictions = predictions.reshape(grid_yy.shape)
        if variance is not None:
            self.variance = variance.reshape(grid_yy.shape)

        # Back-transform if needed
        if self.transform is not None:
            if hasattr(self.transform, "inverse_transform"):
                self.predictions = self.transform.inverse_transform(
                    self.predictions.ravel()
                ).reshape(grid_yy.shape)

        self.logger.info(
            f"Prediction range: [{np.nanmin(self.predictions):.4f}, {np.nanmax(self.predictions):.4f}]"
        )
        self.logger.info(
            f"Variance range: [{np.nanmin(self.variance):.4f}, {np.nanmax(self.variance):.4f}]"
        )

    def cross_validate(self):
        self.logger.info("Performing cross-validation...")

        try:
            if self.kriging_model is None:
                raise PipelineError("Kriging model not fitted")

            cv_predictions, cv_metrics = self.kriging_model.cross_validate()

            self.cv_results = {
                "predicted": cv_predictions,
                "observed": self.z,
                "metrics": cv_metrics,
            }

            # Log metrics
            for metric_name, metric_val in cv_metrics.items():
                self.logger.info(f"  {metric_name}: {metric_val:.4f}")

        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {e}")
            self.cv_results = None

    def visualize(self):
        self.logger.info("Visualization generation...")

        # For now, just log what would be created
        self.logger.info(f"Plots to create: {self.config.visualization.plots}")
        self.logger.info(
            "Note: Visualization integration pending - plots not yet generated"
        )

        # PLACEHOLDER:P-001: Visualization integration
        # Reason: Visualization module integration pending
        # Exit condition: Visualization functions integrated with pipeline
        # See PLACEHOLDERS.md for details
        # This would require:
        # - Saving predictions to appropriate format
        # - Calling plot functions with correct signatures
        # - Handling plot saving

    def save_outputs(self):
        self.logger.info("Saving outputs...")

        # Save predictions
        if self.config.output.save_predictions and self.predictions is not None:
            pred_path = self.output_dir / "predictions.npy"
            np.save(pred_path, self.predictions)
            self.logger.info(f"Saved predictions to {pred_path}")

            if "csv" in self.config.output.formats:
                pred_csv = self.output_dir / "predictions.csv"
                np.savetxt(pred_csv, self.predictions.ravel(), delimiter=",")
                self.logger.info(f"Saved predictions to {pred_csv}")

        # Save variance
        if self.config.output.save_variance and self.variance is not None:
            var_path = self.output_dir / "variance.npy"
            np.save(var_path, self.variance)
            self.logger.info(f"Saved variance to {var_path}")

        # Save cross-validation results
        if self.cv_results is not None and self.config.validation.save_predictions:
            cv_path = self.output_dir / "cross_validation.csv"
            cv_df = pd.DataFrame(
                {
                    "observed": self.cv_results["observed"],
                    "predicted": self.cv_results["predicted"],
                }
            )
            cv_df.to_csv(cv_path, index=False)
            self.logger.info(f"Saved CV results to {cv_path}")

    def generate_report(self):
        report_path = self.output_dir / "analysis_report.txt"
        if self.start_time is not None:
            elapsed = datetime.now() - self.start_time
        else:
            elapsed = "N/A"

        with open(report_path, "w") as f:
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Elapsed Time: {elapsed}\n\n")

            f.write("Data Summary\n")
            f.write(f"{'-' * 70}\n")
            f.write(f"Input: {self.config.data.input_file}\n")
            if self.z is not None:
                f.write(f"Records: {len(self.z)}\n")
                f.write(f"Variable: {self.config.data.z_column}\n")
                z_arr = np.asarray(self.z)
                f.write(f"Range: [{z_arr.min():.4f}, {z_arr.max():.4f}]\n")
                f.write(f"Mean: {z_arr.mean():.4f}\n")
                f.write(f"Std: {z_arr.std():.4f}\n\n")

            if self.variogram_model:
                f.write("Variogram Model\n")
                f.write(f"{'-' * 70}\n")
                model_name = self.variogram_model.__class__.__name__.replace(
                    "Model", ""
                )
                f.write(f"Type: {model_name}\n")
                params = self.variogram_model._parameters  # type: ignore[attr-defined]
                f.write(f"Nugget: {params.get('nugget', 0):.4f}\n")
                f.write(f"Sill: {params.get('sill', 0):.4f}\n")
                f.write(f"Range: {params.get('range', 0):.4f}\n\n")

            if self.cv_results:
                f.write("Cross-Validation Results\n")
                f.write(f"{'-' * 70}\n")
                for metric in self.config.validation.metrics:
                    if metric == "rmse":
                        val = np.sqrt(
                            np.mean(
                                (
                                    self.cv_results["observed"]
                                    - self.cv_results["predicted"]
                                )
                                ** 2
                            )
                        )
                    elif metric == "mae":
                        val = np.mean(
                            np.abs(
                                self.cv_results["observed"]
                                - self.cv_results["predicted"]
                            )
                        )
                    elif metric == "r2":
                        ss_res = np.sum(
                            (self.cv_results["observed"] - self.cv_results["predicted"])
                            ** 2
                        )
                        ss_tot = np.sum(
                            (
                                self.cv_results["observed"]
                                - self.cv_results["observed"].mean()
                            )
                            ** 2
                        )
                        val = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    elif metric == "bias":
                        val = np.mean(
                            self.cv_results["predicted"] - self.cv_results["observed"]
                        )
                    else:
                        continue
                    f.write(f"{metric.upper()}: {val:.4f}\n")

        self.logger.info(f"Report saved to {report_path}")
