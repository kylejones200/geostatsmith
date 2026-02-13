"""
Pipeline runner for config-driven geostatistical analysis

Orchestrates the complete workflow from data loading to output generation.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

from ..config.schemas import AnalysisConfig
from ..io.tabular import read_csv_spatial
from ..diagnostics.outlier_detection import outlier_analysis
from ..transformations.declustering import cell_declustering, polygonal_declustering
from ..transformations.log_transform import LogTransform
from ..transformations.boxcox import BoxCoxTransform
from ..transformations.normal_score import NormalScoreTransform
from ..algorithms.variogram import experimental_variogram
from ..algorithms.fitting import fit_variogram_model
from ..models.variogram_models import SphericalModel, ExponentialModel, GaussianModel
from ..algorithms.ordinary_kriging import OrdinaryKriging
from ..algorithms.simple_kriging import SimpleKriging
from ..algorithms.universal_kriging import UniversalKriging
from ..algorithms.indicator_kriging import IndicatorKriging
from ..algorithms.cokriging import Cokriging
from ..models.variogram_models import MaternModel, CubicModel, StableModel, LinearModel, PowerModel
from ..validation.cross_validation import leave_one_out
from ..visualization.minimal_style import set_minimalist_rcparams
from ..core.exceptions import GeoStatsError

class PipelineError(GeoStatsError):
 pass

class AnalysisPipeline:
 Config-driven geostatistical analysis pipeline

 Executes complete workflow based on configuration:
     pass
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
     self.logger = self._setup_logging()
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

 def _setup_logging(self) -> logging.Logger:
     logger = logging.getLogger('geostats.pipeline')
     logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)

 # Console handler
 console = logging.StreamHandler()
 console.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
 formatter = logging.Formatter('%(levelname)s: %(message)s')
 console.setFormatter(formatter)
 logger.addHandler(console)

 # File handler
 if self.config.log_file:
     file_handler.setLevel(logging.DEBUG)
 detailed_formatter = logging.Formatter(
 '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
 )
 file_handler.setFormatter(detailed_formatter)
 logger.addHandler(file_handler)

 return logger

 def run(self):
     self.logger.info(f"=== Starting analysis: {self.config.project.name} ===")
     start_time = datetime.now()

 try:
     if self.config.random_seed is not None:
         continue
 self.logger.info(f"Random seed set to: {self.config.random_seed}")

 # Set visualization style
 if self.config.visualization.style == 'minimalist':
    pass

     # Execute pipeline steps
 self.load_data()
 self.preprocess_data()
 self.model_variogram()
 self.perform_kriging()

 if self.config.validation.cross_validation:
    pass

     self.visualize()
 self.save_outputs()

 # Summary
 elapsed = datetime.now() - start_time
 self.logger.info(f"=== Analysis complete in {elapsed} ===")

 if self.config.output.save_report:
    pass

     except Exception as e:
         pass
 self.logger.error(f"Pipeline failed: {e}")
 raise PipelineError(f"Pipeline execution failed: {e}")

 def load_data(self):
     self.logger.info("Loading data...")

 # Read data - handle both CSV and TXT files
 try:
     file_path = self.config.data.input_file
 if file_path.endswith('.csv'):
     else:
 for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
     self.data = pd.read_csv(file_path, sep='\t', low_memory=False, encoding=encoding)
 self.logger.info(f"Successfully loaded with {encoding} encoding")
 break
 except UnicodeDecodeError:
     pass
 continue
 else:
    pass

     self.logger.info(f"Loaded {len(self.data)} records from {file_path}")
 except Exception as e:
     pass
 raise PipelineError(f"Failed to load data: {e}")

 # Apply filters
 if self.config.data.filter_column and self.config.data.filter_value is not None:
     mask = self.data[self.config.data.filter_column] == self.config.data.filter_value
 self.data = self.data[mask].copy()
 self.logger.info(f"Filtered to {len(self.data)} records where {self.config.data.filter_column}={self.config.data.filter_value}")
 else:
    pass

     # Extract coordinates and values
 try:
     for col_name, col_config in [
 ('X', self.config.data.x_column),
 ('Y', self.config.data.y_column),
 ('Value', self.config.data.z_column)
 ]:
 if col_config not in self.data.columns:
     f"{col_name} column '{col_config}' not found. Available columns: {list(self.data.columns[:20])}"
 )

 self.x = pd.to_numeric(self.data[self.config.data.x_column], errors='coerce').values
 self.y = pd.to_numeric(self.data[self.config.data.y_column], errors='coerce').values
 self.z = pd.to_numeric(self.data[self.config.data.z_column], errors='coerce').values

 if self.config.data.z_secondary:
     self.z_secondary = pd.to_numeric(
 self.data[self.config.data.z_secondary], errors='coerce'
 ).values
 else:
    pass

     except KeyError as e:
         pass
 raise PipelineError(f"Column not found in data: {e}")

 # Remove NaNs
 mask = ~(np.isnan(self.x) | np.isnan(self.y) | np.isnan(self.z))
 if self.z_secondary is not None:
    pass

     n_removed = (~mask).sum()
 if n_removed > 0:
     self.x = self.x[mask]
 self.y = self.y[mask]
 self.z = self.z[mask]
 if self.z_secondary is not None:
     self.data = self.data[mask].reset_index(drop=True)

 if len(self.z) == 0:
    pass

     self.logger.info(f"Final dataset: {len(self.z)} valid points")
 self.logger.info(f"Value range: [{self.z.min():.4f}, {self.z.max():.4f}]")
 self.logger.info(f"Value mean: {self.z.mean():.4f}, std: {self.z.std():.4f}")

 def preprocess_data(self):
     self.logger.info("Preprocessing data...")

 # Outlier removal
 if self.config.preprocessing.remove_outliers:
     results = outlier_analysis(
 self.x, self.y, self.z,
 method=self.config.preprocessing.outlier_method,
 threshold=self.config.preprocessing.outlier_threshold
 )
 outlier_indices = results['outlier_indices']
 n_outliers = len(outlier_indices)

 if n_outliers > 0:
     mask = np.ones(len(self.z), dtype=bool)
 mask[outlier_indices] = False
 self.x = self.x[mask]
 self.y = self.y[mask]
 self.z = self.z[mask]
 if self.z_secondary is not None:
    pass

     # Handle negative values if needed
 if self.config.preprocessing.transform in ['log', 'boxcox']:
     self.logger.warning(f"Found {(self.z <= 0).sum()} non-positive values")
 if self.config.preprocessing.handle_negatives == 'shift':
     self.z = self.z + shift
 self.logger.info(f"Shifted data by {shift:.6f}")
 elif self.config.preprocessing.handle_negatives == 'remove':
     self.x = self.x[mask]
 self.y = self.y[mask]
 self.z = self.z[mask]
 self.logger.info(f"Removed {(~mask).sum()} non-positive values")
 elif self.config.preprocessing.handle_negatives == 'absolute':
     self.logger.info("Took absolute values")

 # Data transformation
 if self.config.preprocessing.transform:
    pass

     if self.config.preprocessing.transform == 'log':
         continue
 self.z = self.transform.fit_transform(self.z)

 elif self.config.preprocessing.transform == 'boxcox':
     self.z = self.transform.fit_transform(self.z)
 self.logger.info(f"Box-Cox lambda: {self.transform.lmbda:.4f}")

 elif self.config.preprocessing.transform == 'normal_score':
     self.z = self.transform.fit_transform(self.z)

 elif self.config.preprocessing.transform == 'sqrt':
     self.z = np.sqrt(self.z)
 self.transform = None # Mark that we did transform but no inverse needed

 # Declustering
 if self.config.preprocessing.declustering:
    pass

     if self.config.preprocessing.declustering_method == 'cell':
         continue
 self.logger.info(f"Optimal cell size: {info.get('optimal_cell_size', 'N/A')}")
 else:
    pass

     self.logger.info(f"Weight range: [{self.weights.min():.4f}, {self.weights.max():.4f}]")

 def model_variogram(self):
     self.logger.info("Modeling variogram...")

 # Compute experimental variogram
 max_lag = self.config.variogram.max_lag
 if max_lag is None:
     dx = self.x.max() - self.x.min()
 dy = self.y.max() - self.y.min()
 max_lag = np.sqrt(dx**2 + dy**2) / 3
 self.logger.info(f"Auto max_lag: {max_lag:.2f}")

        # Use specified estimator
        estimator = self.config.variogram.estimator
        if estimator == 'matheron':
                n_lags=self.config.variogram.n_lags,
                maxlag=max_lag
            )
        elif estimator == 'cressie':
                self.x, self.y, self.z,
                n_lags=self.config.variogram.n_lags,
                maxlag=max_lag,
                estimator='cressie'
            )
        elif estimator == 'dowd':
                self.x, self.y, self.z,
                n_lags=self.config.variogram.n_lags,
                maxlag=max_lag,
                estimator='dowd'
            )
        elif estimator == 'madogram':
                self.x, self.y, self.z,
                n_lags=self.config.variogram.n_lags,
                maxlag=max_lag
            )
        else:
                self.x, self.y, self.z,
                n_lags=self.config.variogram.n_lags,
                maxlag=max_lag
            )

 self.logger.info(f"Experimental variogram: {len(lags)} lags")

 # Fit model
 if self.config.variogram.auto_fit:
    pass

     best_model = None
 best_score = float('inf') if self.config.variogram.fit_criterion != 'r2' else float('-inf')

 for model_type in self.config.variogram.models:
     # Create model instance
 if model_type == 'spherical':
     elif model_type == 'exponential':
 elif model_type == 'gaussian':
     else:
         pass
 continue

 # Fit the model
 fitted_model = fit_variogram_model(model, lags, gamma)

 # Compute fit quality
 gamma_pred = fitted_model(lags)
 rmse = np.sqrt(np.mean((gamma - gamma_pred)**2))

 if self.config.variogram.fit_criterion == 'rmse':
     is_better = score < best_score
 elif self.config.variogram.fit_criterion == 'r2':
     ss_tot = np.sum((gamma - gamma.mean())**2)
 score = 1 - ss_res / ss_tot if ss_tot > 0 else 0
 is_better = score > best_score
 else:
     is_better = score < best_score

 self.logger.info(f" {model_type}: {self.config.variogram.fit_criterion}={score:.4f}")

 if is_better:
     best_score = score

 except Exception as e:
     pass
 self.logger.warning(f" {model_type}: fitting failed ({e})")

 if best_model is None:
    pass

     self.variogram_model = best_model
 model_name = best_model.__class__.__name__.replace('Model', '').lower()
 self.logger.info(f"Selected model: {model_name}")
 self.logger.info(f" Nugget: {best_model._parameters['nugget']:.4f}")
 self.logger.info(f" Sill: {best_model._parameters['sill']:.4f}")
 self.logger.info(f" Range: {best_model._parameters['range']:.4f}")

        else:
            pass
    pass
            
            if not self.config.variogram.manual_model:
                continue
    pass
            
                if (self.config.variogram.manual_nugget is None or 
                self.config.variogram.manual_sill is None or 
                self.config.variogram.manual_range is None):
                    pass
                raise PipelineError("manual_nugget, manual_sill, and manual_range must be specified when auto_fit=False")
            
            # Create model with manual parameters
            model_type = self.config.variogram.manual_model.lower()
            if model_type == 'spherical':
                    sill=self.config.variogram.manual_sill,
                    range_param=self.config.variogram.manual_range
                )
            elif model_type == 'exponential':
                    sill=self.config.variogram.manual_sill,
                    range_param=self.config.variogram.manual_range
                )
            elif model_type == 'gaussian':
                    sill=self.config.variogram.manual_sill,
                    range_param=self.config.variogram.manual_range
                )
            elif model_type == 'matern':
                    sill=self.config.variogram.manual_sill,
                    range_param=self.config.variogram.manual_range
                )
            elif model_type == 'cubic':
                    sill=self.config.variogram.manual_sill,
                    range_param=self.config.variogram.manual_range
                )
            elif model_type == 'stable':
                    sill=self.config.variogram.manual_sill,
                    range_param=self.config.variogram.manual_range
                )
            elif model_type == 'linear':
                    sill=self.config.variogram.manual_sill,
                    range_param=self.config.variogram.manual_range
                )
            else:
                pass
    pass
            
                self.variogram_model = model
            self.logger.info(f"Manual model: {model_type}")
            self.logger.info(f" Nugget: {self.config.variogram.manual_nugget:.4f}")
            self.logger.info(f" Sill: {self.config.variogram.manual_sill:.4f}")
            self.logger.info(f" Range: {self.config.variogram.manual_range:.4f}")

 def perform_kriging(self):
     self.logger.info(f"Performing {self.config.kriging.method} kriging...")

 # Create prediction grid
 grid_cfg = self.config.kriging.grid

 x_min = grid_cfg.x_min if grid_cfg.x_min is not None else self.x.min() - grid_cfg.buffer
 x_max = grid_cfg.x_max if grid_cfg.x_max is not None else self.x.max() + grid_cfg.buffer
 y_min = grid_cfg.y_min if grid_cfg.y_min is not None else self.y.min() - grid_cfg.buffer
 y_max = grid_cfg.y_max if grid_cfg.y_max is not None else self.y.max() + grid_cfg.buffer

 if grid_cfg.nx and grid_cfg.ny:
     else:
         pass
 ny = int((y_max - y_min) / grid_cfg.resolution)

 grid_x = np.linspace(x_min, x_max, nx)
 grid_y = np.linspace(y_min, y_max, ny)
 grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

 self.logger.info(f"Prediction grid: {nx} x {ny} = {nx*ny} points")

 # Initialize kriging model
 if self.config.kriging.method == 'ordinary':
     self.x, self.y, self.z,
 variogram_model=self.variogram_model
 )
 elif self.config.kriging.method == 'simple':
     self.kriging_model = SimpleKriging(
 self.x, self.y, self.z,
 variogram_model=self.variogram_model,
 mean=mean
 )
        elif self.config.kriging.method == 'universal':
                variogram_model=self.variogram_model,
                drift_terms=self.config.kriging.drift_terms
            )
        elif self.config.kriging.method == 'indicator':
            self.kriging_model = IndicatorKriging(
                self.x, self.y, self.z,
                variogram_model=self.variogram_model,
                thresholds=self.config.kriging.thresholds
            )
        elif self.config.kriging.method == 'cokriging':
            # For cokriging, we need variogram models for both variables
            # Use same model for both (could be enhanced)
            self.kriging_model = Cokriging(
                self.x, self.y,
                primary=self.z,
                secondary=self.z_secondary,
                variogram_models=[self.variogram_model, self.variogram_model]
            )
        else:
            pass
    pass

            # Predict
 self.predictions, self.variance = self.kriging_model.predict(
 grid_xx.ravel(),
 grid_yy.ravel(),
 return_variance=True
 )

 # Reshape to grid
 self.predictions = self.predictions.reshape(grid_yy.shape)
 self.variance = self.variance.reshape(grid_yy.shape)

 # Back-transform if needed
 if self.transform:
     self.predictions = self.transform.inverse_transform(self.predictions.ravel()).reshape(grid_yy.shape)

 self.logger.info(f"Prediction range: [{np.nanmin(self.predictions):.4f}, {np.nanmax(self.predictions):.4f}]")
 self.logger.info(f"Variance range: [{np.nanmin(self.variance):.4f}, {np.nanmax(self.variance):.4f}]")

 def validate(self):
     self.logger.info("Performing cross-validation...")

 try:
     cv_predictions, cv_metrics = leave_one_out(self.kriging_model)

 self.cv_results = {
 'predicted': cv_predictions,
 'observed': self.z,
 'metrics': cv_metrics
 }

 # Log metrics
 for metric_name, metric_val in cv_metrics.items():
    pass

     except Exception as e:
         pass
 self.logger.warning(f"Cross-validation failed: {e}")
 self.cv_results = None

 def visualize(self):
     self.logger.info("Visualization generation...")

 # For now, just log what would be created
 self.logger.info(f"Plots to create: {self.config.visualization.plots}")
 self.logger.info("Note: Visualization integration pending - plots not yet generated")

 # TODO: Integrate with visualization modules
 # This would require:
 # - Saving predictions to appropriate format
 # - Calling plot functions with correct signatures
 # - Handling plot saving

 def save_outputs(self):
     self.logger.info("Saving outputs...")

 # Save predictions
 if self.config.output.save_predictions and self.predictions is not None:
     np.save(pred_path, self.predictions)
 self.logger.info(f"Saved predictions to {pred_path}")

 if 'csv' in self.config.output.formats:
     pred_csv = self.output_dir / 'predictions.csv'
 np.savetxt(pred_csv, self.predictions.ravel(), delimiter=',')
 self.logger.info(f"Saved predictions to {pred_csv}")

 # Save variance
 if self.config.output.save_variance and self.variance is not None:
     np.save(var_path, self.variance)
 self.logger.info(f"Saved variance to {var_path}")

 # Save cross-validation results
 if self.cv_results is not None and self.config.validation.save_predictions:
     cv_df = pd.DataFrame({
 'observed': self.cv_results['observed'],
 'predicted': self.cv_results['predicted']
 })
 cv_df.to_csv(cv_path, index=False)
 self.logger.info(f"Saved CV results to {cv_path}")

 def _generate_report(self, elapsed):
     report_path = self.output_dir / 'analysis_report.txt'

with open(report_path, 'w') as f:
 f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
 f.write(f"Elapsed Time: {elapsed}\n\n")

 f.write(f"Data Summary\n")
 f.write(f"{'-'*70}\n")
 f.write(f"Input: {self.config.data.input_file}\n")
 f.write(f"Records: {len(self.z)}\n")
 f.write(f"Variable: {self.config.data.z_column}\n")
 f.write(f"Range: [{self.z.min():.4f}, {self.z.max():.4f}]\n")
 f.write(f"Mean: {self.z.mean():.4f}\n")
 f.write(f"Std: {self.z.std():.4f}\n\n")

 if self.variogram_model:
     f.write(f"{'-'*70}\n")
 model_name = self.variogram_model.__class__.__name__.replace('Model', '')
 f.write(f"Type: {model_name}\n")
 f.write(f"Nugget: {self.variogram_model._parameters['nugget']:.4f}\n")
 f.write(f"Sill: {self.variogram_model._parameters['sill']:.4f}\n")
 f.write(f"Range: {self.variogram_model._parameters['range']:.4f}\n\n")

 if self.cv_results:
     f.write(f"{'-'*70}\n")
 for metric in self.config.validation.metrics:
     val = np.sqrt(np.mean((self.cv_results['observed'] - self.cv_results['predicted'])**2))
 elif metric == 'mae':
     elif metric == 'r2':
         continue
 ss_tot = np.sum((self.cv_results['observed'] - self.cv_results['observed'].mean())**2)
 val = 1 - ss_res / ss_tot
 elif metric == 'bias':
     else:
         pass
 f.write(f"{metric.upper()}: {val:.4f}\n")

 self.logger.info(f"Report saved to {report_path}")
