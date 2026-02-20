"""
    Configuration schemas for config-driven geostatistical analysis

Uses Pydantic for validation and type checking.
"""

from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np

class ProjectConfig(BaseModel):
 name: str = Field(..., description="Project name")
 output_dir: str = Field("./results", description="Output directory path")
 description: Optional[str] = Field(None, description="Project description")
 author: Optional[str] = Field(None, description="Analysis author")

 @field_validator('output_dir')
 @classmethod
 def validate_output_dir(cls, v):
     Path(v).mkdir(parents=True, exist_ok=True)
     return v

class DataConfig(BaseModel):
 input_file: str = Field(..., description="Path to input data file")
 x_column: str = Field(..., description="X coordinate column name")
 y_column: str = Field(..., description="Y coordinate column name")
 z_column: str = Field(..., description="Value column name")
 z_secondary: Optional[str] = Field(None, description="Secondary variable for cokriging")
 filter_column: Optional[str] = Field(None, description="Column to filter by")
 filter_value: Optional[Any] = Field(None, description="Value to filter for")

 @field_validator('input_file')
 @classmethod
 def validate_file_exists(cls, v):
     if not Path(v).exists():
     return v

class PreprocessingConfig(BaseModel):
 remove_outliers: bool = Field(False, description="Remove outliers")
 outlier_method: Literal['iqr', 'zscore', 'isolation_forest'] = Field('iqr', description="Outlier detection method")
 outlier_threshold: float = Field(3.0, description="Threshold for outlier detection")

 transform: Optional[Literal['log', 'boxcox', 'normal_score', 'sqrt']] = Field(None, description="Data transformation")
 transform_params: Optional[Dict[str, Any]] = Field(None, description="Transform parameters")

 declustering: bool = Field(False, description="Apply declustering weights")
 declustering_method: Literal['cell', 'polygonal'] = Field('cell', description="Declustering method")

 handle_negatives: Literal['shift', 'remove', 'absolute'] = Field('shift', description="How to handle negative values for log/boxcox")

class VariogramConfig(BaseModel):
 n_lags: int = Field(15, ge=5, le=50, description="Number of lag bins")
 max_lag: Optional[float] = Field(None, description="Maximum lag distance (auto if None)")
 lag_tolerance: Optional[float] = Field(None, description="Lag tolerance")

 estimator: Literal['matheron', 'cressie', 'dowd', 'madogram'] = Field('matheron', description="Variogram estimator")

 models: List[Literal['spherical', 'exponential', 'gaussian', 'matern', 'cubic', 'stable', 'linear']] = Field(
 ['spherical', 'exponential', 'gaussian'],
 description="Models to try"
 )

 auto_fit: bool = Field(True, description="Automatically select best model")
 fit_method: Literal['ols', 'wls'] = Field('wls', description="Fitting method")
 fit_criterion: Literal['rmse', 'mae', 'r2', 'aic'] = Field('rmse', description="Model selection criterion")

 # Manual parameters (used if auto_fit=False)
 manual_model: Optional[str] = Field(None, description="Manual model selection")
 manual_nugget: Optional[float] = Field(None, description="Manual nugget")
 manual_sill: Optional[float] = Field(None, description="Manual sill")
 manual_range: Optional[float] = Field(None, description="Manual range")

 # Directional variography
 check_anisotropy: bool = Field(False, description="Check for anisotropy")
 anisotropy_angles: List[float] = Field([0, 45, 90, 135], description="Angles for directional variograms")
 anisotropy_tolerance: float = Field(22.5, description="Angular tolerance")

class NeighborhoodConfig(BaseModel):
 max_neighbors: int = Field(25, ge=1, description="Maximum neighbors to use")
 min_neighbors: int = Field(3, ge=1, description="Minimum neighbors required")
 search_radius: Optional[float] = Field(None, description="Search radius (auto if None)")
 use_octant_search: bool = Field(False, description="Use octant search")

 @model_validator(mode='after')
 def validate_neighbors(self):
     if self.max_neighbors < self.min_neighbors:
     return self

class GridConfig(BaseModel):
 x_min: Optional[float] = Field(None, description="Grid X minimum (auto from data if None)")
 x_max: Optional[float] = Field(None, description="Grid X maximum")
 y_min: Optional[float] = Field(None, description="Grid Y minimum")
 y_max: Optional[float] = Field(None, description="Grid Y maximum")
 resolution: float = Field(1.0, gt=0, description="Grid cell size")
 nx: Optional[int] = Field(None, description="Number of X cells (overrides resolution)")
 ny: Optional[int] = Field(None, description="Number of Y cells")

 buffer: float = Field(0.0, ge=0, description="Buffer around data extent (in data units)")

class KrigingConfig(BaseModel):
 method: Literal['ordinary', 'simple', 'universal', 'indicator', 'cokriging'] = Field()
 'ordinary',
 description="Kriging method"
 )

 # Universal kriging specific
 drift_terms: Literal['linear', 'quadratic'] = Field('linear', description="Drift terms for universal kriging")

 # Simple kriging specific
 mean: Optional[float] = Field(None, description="Known mean for simple kriging (auto if None)")

 # Indicator kriging specific
 thresholds: Optional[List[float]] = Field(None, description="Thresholds for indicator kriging")

 neighborhood: NeighborhoodConfig = Field(default_factory=NeighborhoodConfig, description="Neighborhood search")
 grid: GridConfig = Field(default_factory=GridConfig, description="Prediction grid")

 return_variance: bool = Field(True, description="Return kriging variance")
 parallel: bool = Field(False, description="Use parallel processing")
 n_jobs: int = Field(-1, description="Number of parallel jobs (-1 = all cores)")

class ValidationConfig(BaseModel):
 cross_validation: bool = Field(True, description="Perform cross-validation")
 cv_method: Literal['loo', 'kfold', 'spatial'] = Field('loo', description="Cross-validation method")
 n_folds: int = Field(5, ge=2, description="Number of folds for k-fold CV")

 metrics: List[Literal['rmse', 'mae', 'mse', 'r2', 'bias', 'mape']] = Field(
 ['rmse', 'mae', 'r2'],
 description="Metrics to compute"
 )

 save_predictions: bool = Field(True, description="Save CV predictions")

class PlotConfig(BaseModel):
 enabled: bool = Field(True, description="Create this plot")
 dpi: int = Field(300, description="Plot DPI")
 figsize: Optional[tuple] = Field(None, description="Figure size (width, height)")
 colormap: str = Field('viridis', description="Colormap name")
 title: Optional[str] = Field(None, description="Custom plot title")

class VisualizationConfig(BaseModel):
 style: Literal['minimalist', 'default', 'seaborn'] = Field('minimalist', description="Plot style")

 # What to plot
 plots: List[Literal[
 'variogram', 'variogram_cloud', 'directional_variograms',
 'data_locations', 'kriging_map', 'variance_map',
 'cross_validation', 'histogram', 'qq_plot',
 'residuals', 'h_scatterplot'
 ]] = Field(
 ['variogram', 'kriging_map', 'cross_validation'],
 description="Plots to create"
 )

 # Global plot settings
 dpi: int = Field(300, ge=72, description="Default DPI for all plots")
 colormap: str = Field('viridis', description="Default colormap")
 save_format: List[Literal['png', 'pdf', 'svg', 'jpg']] = Field(['png'], description="Output formats")

 # Per-plot configurations
 variogram: PlotConfig = Field(default_factory=PlotConfig, description="Variogram plot config")
 kriging_map: PlotConfig = Field(default_factory=PlotConfig, description="Kriging map config")
 variance_map: PlotConfig = Field(default_factory=PlotConfig, description="Variance map config")
 cross_validation: PlotConfig = Field(default_factory=PlotConfig, description="CV plot config")

class OutputConfig(BaseModel):
 save_predictions: bool = Field(True, description="Save prediction grid")
 save_variance: bool = Field(True, description="Save variance grid")
 save_weights: bool = Field(False, description="Save declustering weights")
 save_model: bool = Field(True, description="Save fitted variogram model")
 save_report: bool = Field(True, description="Generate text report")

 formats: List[Literal['npy', 'csv', 'geotiff', 'netcdf', 'shapefile']] = Field(
 ['npy', 'csv'],
 description="Output formats"
 )

 compression: bool = Field(True, description="Compress output files")
 precision: Literal['float32', 'float64'] = Field('float32', description="Numerical precision")

class AnalysisConfig(BaseModel):
    pass

 project: ProjectConfig = Field(..., description="Project metadata")
 data: DataConfig = Field(..., description="Data configuration")
 preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig, description="Preprocessing")
 variogram: VariogramConfig = Field(default_factory=VariogramConfig, description="Variogram modeling")
 kriging: KrigingConfig = Field(default_factory=KrigingConfig, description="Kriging configuration")
 validation: ValidationConfig = Field(default_factory=ValidationConfig, description="Validation")
 visualization: VisualizationConfig = Field(default_factory=VisualizationConfig, description="Visualization")
 output: OutputConfig = Field(default_factory=OutputConfig, description="Output configuration")

 # Advanced options
 random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
 verbose: bool = Field(True, description="Verbose output")
 log_file: Optional[str] = Field(None, description="Log file path")

 class Config:
     extra = 'forbid' # Don'
     validate_assignment = True

 @model_validator(mode='after')
 def validate_config(self):
     # Check cokriging requirements
     if self.kriging.method == 'cokriging' and self.data.z_secondary is None:
    pass

 # Check indicator kriging requirements
 if self.kriging.method == 'indicator' and self.kriging.thresholds is None:
    pass

     # Check simple kriging mean
 if self.kriging.method == 'simple' and self.kriging.mean is None:
     pass

 return self

 def model_dump_yaml(self) -> str:
     import yaml
     return yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False)

 def save_yaml(self, path: str):
     with open(path, 'w') as f:
