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
from ..validation.cross_validation import leave_one_out
from ..visualization.minimal_style import set_minimalist_rcparams
from ..core.exceptions import GeoStatsError


class PipelineError(GeoStatsError):
    """Pipeline execution error"""
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
        """Setup logging"""
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
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(logging.DEBUG)
            detailed_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def run(self):
        """Execute complete pipeline"""
        self.logger.info(f"=== Starting analysis: {self.config.project.name} ===")
        start_time = datetime.now()
        
        try:
            # Set random seed
            if self.config.random_seed is not None:
                np.random.seed(self.config.random_seed)
                self.logger.info(f"Random seed set to: {self.config.random_seed}")
            
            # Set visualization style
            if self.config.visualization.style == 'minimalist':
                set_minimalist_rcparams()
            
            # Execute pipeline steps
            self.load_data()
            self.preprocess_data()
            self.model_variogram()
            self.perform_kriging()
            
            if self.config.validation.cross_validation:
                self.validate()
            
            self.visualize()
            self.save_outputs()
            
            # Summary
            elapsed = datetime.now() - start_time
            self.logger.info(f"=== Analysis complete in {elapsed} ===")
            
            if self.config.output.save_report:
                self._generate_report(elapsed)
                
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise PipelineError(f"Pipeline execution failed: {e}")
    
    def load_data(self):
        """Load and filter data"""
        self.logger.info("Loading data...")
        
        # Read data - handle both CSV and TXT files
        try:
            # Read the file - Alaska AGDB4 is tab-delimited txt
            file_path = self.config.data.input_file
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            else:
                # Try tab-delimited for .txt files with various encodings
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        self.data = pd.read_csv(file_path, sep='\t', low_memory=False, encoding=encoding)
                        self.logger.info(f"Successfully loaded with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise Exception("Could not decode file with any standard encoding")
            
            self.logger.info(f"Loaded {len(self.data)} records from {file_path}")
        except Exception as e:
            raise PipelineError(f"Failed to load data: {e}")
        
        # Apply filters
        if self.config.data.filter_column and self.config.data.filter_value is not None:
            if self.config.data.filter_column in self.data.columns:
                mask = self.data[self.config.data.filter_column] == self.config.data.filter_value
                self.data = self.data[mask].copy()
                self.logger.info(f"Filtered to {len(self.data)} records where {self.config.data.filter_column}={self.config.data.filter_value}")
            else:
                self.logger.warning(f"Filter column '{self.config.data.filter_column}' not found in data")
        
        # Extract coordinates and values
        try:
            # Check if columns exist
            for col_name, col_config in [
                ('X', self.config.data.x_column),
                ('Y', self.config.data.y_column),
                ('Value', self.config.data.z_column)
            ]:
                if col_config not in self.data.columns:
                    raise PipelineError(
                        f"{col_name} column '{col_config}' not found. Available columns: {list(self.data.columns[:20])}"
                    )
            
            self.x = pd.to_numeric(self.data[self.config.data.x_column], errors='coerce').values
            self.y = pd.to_numeric(self.data[self.config.data.y_column], errors='coerce').values
            self.z = pd.to_numeric(self.data[self.config.data.z_column], errors='coerce').values
            
            if self.config.data.z_secondary:
                if self.config.data.z_secondary in self.data.columns:
                    self.z_secondary = pd.to_numeric(
                        self.data[self.config.data.z_secondary], errors='coerce'
                    ).values
                else:
                    self.logger.warning(f"Secondary variable '{self.config.data.z_secondary}' not found")
                
        except KeyError as e:
            raise PipelineError(f"Column not found in data: {e}")
        
        # Remove NaNs
        mask = ~(np.isnan(self.x) | np.isnan(self.y) | np.isnan(self.z))
        if self.z_secondary is not None:
            mask &= ~np.isnan(self.z_secondary)
        
        n_removed = (~mask).sum()
        if n_removed > 0:
            self.logger.warning(f"Removed {n_removed} records with NaN values")
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.z = self.z[mask]
            if self.z_secondary is not None:
                self.z_secondary = self.z_secondary[mask]
            self.data = self.data[mask].reset_index(drop=True)
        
        if len(self.z) == 0:
            raise PipelineError("No valid data points after filtering")
        
        self.logger.info(f"Final dataset: {len(self.z)} valid points")
        self.logger.info(f"Value range: [{self.z.min():.4f}, {self.z.max():.4f}]")
        self.logger.info(f"Value mean: {self.z.mean():.4f}, std: {self.z.std():.4f}")
    
    def preprocess_data(self):
        """Preprocess data (outliers, transforms, declustering)"""
        self.logger.info("Preprocessing data...")
        
        # Outlier removal
        if self.config.preprocessing.remove_outliers:
            self.logger.info(f"Detecting outliers using {self.config.preprocessing.outlier_method}...")
            results = outlier_analysis(
                self.x, self.y, self.z,
                method=self.config.preprocessing.outlier_method,
                threshold=self.config.preprocessing.outlier_threshold
            )
            outlier_indices = results['outlier_indices']
            n_outliers = len(outlier_indices)
            
            if n_outliers > 0:
                self.logger.info(f"Removing {n_outliers} outliers ({100*n_outliers/len(self.z):.1f}%)")
                mask = np.ones(len(self.z), dtype=bool)
                mask[outlier_indices] = False
                self.x = self.x[mask]
                self.y = self.y[mask]
                self.z = self.z[mask]
                if self.z_secondary is not None:
                    self.z_secondary = self.z_secondary[mask]
        
        # Handle negative values if needed
        if self.config.preprocessing.transform in ['log', 'boxcox']:
            if (self.z <= 0).any():
                self.logger.warning(f"Found {(self.z <= 0).sum()} non-positive values")
                if self.config.preprocessing.handle_negatives == 'shift':
                    shift = abs(self.z.min()) + 1e-6
                    self.z = self.z + shift
                    self.logger.info(f"Shifted data by {shift:.6f}")
                elif self.config.preprocessing.handle_negatives == 'remove':
                    mask = self.z > 0
                    self.x = self.x[mask]
                    self.y = self.y[mask]
                    self.z = self.z[mask]
                    self.logger.info(f"Removed {(~mask).sum()} non-positive values")
                elif self.config.preprocessing.handle_negatives == 'absolute':
                    self.z = np.abs(self.z)
                    self.logger.info("Took absolute values")
        
        # Data transformation
        if self.config.preprocessing.transform:
            self.logger.info(f"Applying {self.config.preprocessing.transform} transform...")
            
            if self.config.preprocessing.transform == 'log':
                self.transform = LogTransform()
                self.z = self.transform.fit_transform(self.z)
            
            elif self.config.preprocessing.transform == 'boxcox':
                self.transform = BoxCoxTransform()
                self.z = self.transform.fit_transform(self.z)
                self.logger.info(f"Box-Cox lambda: {self.transform.lmbda:.4f}")
            
            elif self.config.preprocessing.transform == 'normal_score':
                self.transform = NormalScoreTransform()
                self.z = self.transform.fit_transform(self.z)
            
            elif self.config.preprocessing.transform == 'sqrt':
                # Simple sqrt doesn't need a class
                self.z = np.sqrt(self.z)
                self.transform = None  # Mark that we did transform but no inverse needed
        
        # Declustering
        if self.config.preprocessing.declustering:
            self.logger.info(f"Computing {self.config.preprocessing.declustering_method} declustering weights...")
            
            if self.config.preprocessing.declustering_method == 'cell':
                self.weights, info = cell_declustering(self.x, self.y, self.z)
                self.logger.info(f"Optimal cell size: {info.get('optimal_cell_size', 'N/A')}")
            else:
                self.weights = polygonal_declustering(self.x, self.y)
            
            self.logger.info(f"Weight range: [{self.weights.min():.4f}, {self.weights.max():.4f}]")
    
    def model_variogram(self):
        """Model variogram"""
        self.logger.info("Modeling variogram...")
        
        # Compute experimental variogram
        max_lag = self.config.variogram.max_lag
        if max_lag is None:
            # Auto: 1/3 of max distance
            dx = self.x.max() - self.x.min()
            dy = self.y.max() - self.y.min()
            max_lag = np.sqrt(dx**2 + dy**2) / 3
            self.logger.info(f"Auto max_lag: {max_lag:.2f}")
        
        lags, gamma, n_pairs = experimental_variogram(
            self.x, self.y, self.z,
            n_lags=self.config.variogram.n_lags,
            maxlag=max_lag
        )
        
        # Note: Different estimators (cressie, dowd, madogram) would need separate functions
        if self.config.variogram.estimator != 'matheron':
            self.logger.warning(f"Estimator '{self.config.variogram.estimator}' not yet implemented in pipeline, using Matheron")
        
        self.logger.info(f"Experimental variogram: {len(lags)} lags")
        
        # Fit model
        if self.config.variogram.auto_fit:
            self.logger.info(f"Auto-fitting variogram models: {self.config.variogram.models}")
            
            best_model = None
            best_score = float('inf') if self.config.variogram.fit_criterion != 'r2' else float('-inf')
            
            for model_type in self.config.variogram.models:
                try:
                    # Create model instance
                    if model_type == 'spherical':
                        model = SphericalModel()
                    elif model_type == 'exponential':
                        model = ExponentialModel()
                    elif model_type == 'gaussian':
                        model = GaussianModel()
                    else:
                        self.logger.warning(f"Model type '{model_type}' not recognized, skipping")
                        continue
                    
                    # Fit the model
                    fitted_model = fit_variogram_model(model, lags, gamma)
                    
                    # Compute fit quality
                    gamma_pred = fitted_model(lags)
                    rmse = np.sqrt(np.mean((gamma - gamma_pred)**2))
                    
                    if self.config.variogram.fit_criterion == 'rmse':
                        score = rmse
                        is_better = score < best_score
                    elif self.config.variogram.fit_criterion == 'r2':
                        ss_res = np.sum((gamma - gamma_pred)**2)
                        ss_tot = np.sum((gamma - gamma.mean())**2)
                        score = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                        is_better = score > best_score
                    else:
                        score = rmse  # Fallback
                        is_better = score < best_score
                    
                    self.logger.info(f"  {model_type}: {self.config.variogram.fit_criterion}={score:.4f}")
                    
                    if is_better:
                        best_model = fitted_model
                        best_score = score
                
                except Exception as e:
                    self.logger.warning(f"  {model_type}: fitting failed ({e})")
            
            if best_model is None:
                raise PipelineError("No variogram model could be fitted")
            
            self.variogram_model = best_model
            model_name = best_model.__class__.__name__.replace('Model', '').lower()
            self.logger.info(f"Selected model: {model_name}")
            self.logger.info(f"  Nugget: {best_model._parameters['nugget']:.4f}")
            self.logger.info(f"  Sill: {best_model._parameters['sill']:.4f}")
            self.logger.info(f"  Range: {best_model._parameters['range']:.4f}")
        
        else:
            # Manual model
            self.logger.info("Using manual variogram parameters")
            raise NotImplementedError("Manual variogram modeling not yet implemented in pipeline")
    
    def perform_kriging(self):
        """Perform kriging prediction"""
        self.logger.info(f"Performing {self.config.kriging.method} kriging...")
        
        # Create prediction grid
        grid_cfg = self.config.kriging.grid
        
        x_min = grid_cfg.x_min if grid_cfg.x_min is not None else self.x.min() - grid_cfg.buffer
        x_max = grid_cfg.x_max if grid_cfg.x_max is not None else self.x.max() + grid_cfg.buffer
        y_min = grid_cfg.y_min if grid_cfg.y_min is not None else self.y.min() - grid_cfg.buffer
        y_max = grid_cfg.y_max if grid_cfg.y_max is not None else self.y.max() + grid_cfg.buffer
        
        if grid_cfg.nx and grid_cfg.ny:
            nx, ny = grid_cfg.nx, grid_cfg.ny
        else:
            nx = int((x_max - x_min) / grid_cfg.resolution)
            ny = int((y_max - y_min) / grid_cfg.resolution)
        
        grid_x = np.linspace(x_min, x_max, nx)
        grid_y = np.linspace(y_min, y_max, ny)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
        
        self.logger.info(f"Prediction grid: {nx} x {ny} = {nx*ny} points")
        
        # Initialize kriging model
        if self.config.kriging.method == 'ordinary':
            self.kriging_model = OrdinaryKriging(
                self.x, self.y, self.z,
                variogram_model=self.variogram_model
            )
        elif self.config.kriging.method == 'simple':
            mean = self.config.kriging.mean if self.config.kriging.mean else self.z.mean()
            self.kriging_model = SimpleKriging(
                self.x, self.y, self.z,
                variogram_model=self.variogram_model,
                mean=mean
            )
        elif self.config.kriging.method == 'universal':
            self.kriging_model = UniversalKriging(
                self.x, self.y, self.z,
                variogram_model=self.variogram_model,
                drift_terms=self.config.kriging.drift_terms
            )
        else:
            raise NotImplementedError(f"Kriging method '{self.config.kriging.method}' not yet implemented")
        
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
            self.logger.info("Back-transforming predictions...")
            self.predictions = self.transform.inverse_transform(self.predictions.ravel()).reshape(grid_yy.shape)
        
        self.logger.info(f"Prediction range: [{np.nanmin(self.predictions):.4f}, {np.nanmax(self.predictions):.4f}]")
        self.logger.info(f"Variance range: [{np.nanmin(self.variance):.4f}, {np.nanmax(self.variance):.4f}]")
    
    def validate(self):
        """Perform cross-validation"""
        self.logger.info("Performing cross-validation...")
        
        try:
            # Use the kriging object's built-in cross-validation
            cv_predictions, cv_metrics = leave_one_out(self.kriging_model)
            
            self.cv_results = {
                'predicted': cv_predictions,
                'observed': self.z,
                'metrics': cv_metrics
            }
            
            # Log metrics
            for metric_name, metric_val in cv_metrics.items():
                self.logger.info(f"  {metric_name.upper()}: {metric_val:.4f}")
                
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {e}")
            self.cv_results = None
    
    def visualize(self):
        """Create visualizations"""
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
        """Save outputs"""
        self.logger.info("Saving outputs...")
        
        # Save predictions
        if self.config.output.save_predictions and self.predictions is not None:
            pred_path = self.output_dir / 'predictions.npy'
            np.save(pred_path, self.predictions)
            self.logger.info(f"Saved predictions to {pred_path}")
            
            if 'csv' in self.config.output.formats:
                # Flatten and save as CSV
                pred_csv = self.output_dir / 'predictions.csv'
                np.savetxt(pred_csv, self.predictions.ravel(), delimiter=',')
                self.logger.info(f"Saved predictions to {pred_csv}")
        
        # Save variance
        if self.config.output.save_variance and self.variance is not None:
            var_path = self.output_dir / 'variance.npy'
            np.save(var_path, self.variance)
            self.logger.info(f"Saved variance to {var_path}")
        
        # Save cross-validation results
        if self.cv_results is not None and self.config.validation.save_predictions:
            cv_path = self.output_dir / 'cv_results.csv'
            cv_df = pd.DataFrame({
                'observed': self.cv_results['observed'],
                'predicted': self.cv_results['predicted']
            })
            cv_df.to_csv(cv_path, index=False)
            self.logger.info(f"Saved CV results to {cv_path}")
    
    def _generate_report(self, elapsed):
        """Generate text report"""
        report_path = self.output_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write(f"Geostatistical Analysis Report\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Project: {self.config.project.name}\n")
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
                f.write(f"Variogram Model\n")
                f.write(f"{'-'*70}\n")
                model_name = self.variogram_model.__class__.__name__.replace('Model', '')
                f.write(f"Type: {model_name}\n")
                f.write(f"Nugget: {self.variogram_model._parameters['nugget']:.4f}\n")
                f.write(f"Sill: {self.variogram_model._parameters['sill']:.4f}\n")
                f.write(f"Range: {self.variogram_model._parameters['range']:.4f}\n\n")
            
            if self.cv_results:
                f.write(f"Cross-Validation\n")
                f.write(f"{'-'*70}\n")
                for metric in self.config.validation.metrics:
                    if metric == 'rmse':
                        val = np.sqrt(np.mean((self.cv_results['observed'] - self.cv_results['predicted'])**2))
                    elif metric == 'mae':
                        val = np.mean(np.abs(self.cv_results['observed'] - self.cv_results['predicted']))
                    elif metric == 'r2':
                        ss_res = np.sum((self.cv_results['observed'] - self.cv_results['predicted'])**2)
                        ss_tot = np.sum((self.cv_results['observed'] - self.cv_results['observed'].mean())**2)
                        val = 1 - ss_res / ss_tot
                    elif metric == 'bias':
                        val = np.mean(self.cv_results['predicted'] - self.cv_results['observed'])
                    else:
                        continue
                    f.write(f"{metric.upper()}: {val:.4f}\n")
        
        self.logger.info(f"Report saved to {report_path}")
