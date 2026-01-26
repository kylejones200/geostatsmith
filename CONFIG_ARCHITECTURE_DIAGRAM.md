# Config-Driven Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERACTION LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Command Line Interface (CLI)              Python API            │
│  ┌──────────────────────────┐              ┌─────────────────┐  │
│  │ geostats-init            │              │ from geostats   │  │
│  │ geostats-validate        │              │   .config import│  │
│  │ geostats-run             │              │   load_config   │  │
│  │ geostats-templates       │              │                 │  │
│  └──────────────────────────┘              └─────────────────┘  │
│                                                                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CONFIGURATION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  YAML/JSON Files                Pydantic Schemas                 │
│  ┌──────────────┐               ┌──────────────────────┐        │
│  │ basic.yaml   │──────────────▶│ AnalysisConfig       │        │
│  │ advanced.yaml│   Parser      │  ├─ ProjectConfig    │        │
│  │ custom.yaml  │               │  ├─ DataConfig       │        │
│  └──────────────┘               │  ├─ PreprocessingCfg │        │
│                                  │  ├─ VariogramConfig  │        │
│  Templates                       │  ├─ KrigingConfig    │        │
│  ┌──────────────┐               │  ├─ ValidationConfig │        │
│  │ basic        │               │  ├─ VisualizationCfg │        │
│  │ advanced     │               │  └─ OutputConfig     │        │
│  │ exploration  │               └──────────────────────┘        │
│  └──────────────┘                                                │
│                                                                   │
│  Validation: Type checking, constraints, cross-field rules       │
│                                                                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WORKFLOW ORCHESTRATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  AnalysisPipeline                                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                                                            │ │
│  │  1. Load Data         ───▶  Read CSV, filter, validate   │ │
│  │         │                                                  │ │
│  │  2. Preprocess        ───▶  Outliers → Transform → Weights│ │
│  │         │                                                  │ │
│  │  3. Variogram         ───▶  Experimental → Fit → Select   │ │
│  │         │                                                  │ │
│  │  4. Kriging           ───▶  Grid → Predict → Variance     │ │
│  │         │                                                  │ │
│  │  5. Validation        ───▶  Cross-validate → Metrics      │ │
│  │         │                                                  │ │
│  │  6. Visualization     ───▶  Generate plots                │ │
│  │         │                                                  │ │
│  │  7. Output            ───▶  Save results + report         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Features: Logging, error handling, state management, reporting  │
│                                                                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ALGORITHM LAYER (Existing)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Data I/O              Preprocessing          Algorithms          │
│  ┌──────────┐         ┌──────────┐           ┌─────────────┐    │
│  │ CSV      │         │ Outliers │           │ Variogram   │    │
│  │ GeoTIFF  │         │ Transform│           │ Kriging     │    │
│  │ NetCDF   │         │ Decluster│           │ Simulation  │    │
│  └──────────┘         └──────────┘           └─────────────┘    │
│                                                                   │
│  Visualization         Validation            Math/Stats          │
│  ┌──────────┐         ┌──────────┐           ┌─────────────┐    │
│  │ Plots    │         │ CV       │           │ Distance    │    │
│  │ Maps     │         │ Metrics  │           │ Matrices    │    │
│  │ Reports  │         │ Diagnost.│           │ Numerical   │    │
│  └──────────┘         └──────────┘           └─────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Flow

```
User Action                Config Processing              Execution
───────────               ─────────────────              ──────────

[Write YAML]                                            
     │                                                  
     ├──────────▶ [Parse YAML]                         
                      │                                 
                      ├──────────▶ [Validate Schema]   
                                       │                
                                       ├──────▶ [Check Types]
                                       ├──────▶ [Check Ranges]
                                       ├──────▶ [Check Files]
                                       ├──────▶ [Cross-validate]
                                       │                
                                       ▼                
                                  [AnalysisConfig]      
                                       │                
                                       ├──────────▶ [Load Data]
                                       ├──────────▶ [Preprocess]
                                       ├──────────▶ [Variogram]
                                       ├──────────▶ [Kriging]
                                       ├──────────▶ [Validate]
                                       ├──────────▶ [Visualize]
                                       ├──────────▶ [Output]
                                       │                
                                       ▼                
                                  [Results/Report]      
```

## Data Flow Through Pipeline

```
Input Data (CSV)
      │
      ▼
┌──────────────┐
│ Load & Filter│ ◀─── data.input_file, filter_column
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Preprocess   │ ◀─── preprocessing.remove_outliers
│              │      preprocessing.transform
│              │      preprocessing.declustering
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Variogram    │ ◀─── variogram.n_lags, estimator
│ Modeling     │      variogram.models, auto_fit
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Kriging      │ ◀─── kriging.method
│ Prediction   │      kriging.neighborhood
│              │      kriging.grid
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Cross-       │ ◀─── validation.cv_method
│ Validation   │      validation.metrics
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Visualization│ ◀─── visualization.style
│              │      visualization.plots
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Output       │ ◀─── output.formats
│ Generation   │      output.save_predictions
└──────┬───────┘
       │
       ▼
Results + Report
```

## CLI Command Flow

```
Terminal Command                    Internal Processing
────────────────                    ───────────────────

$ geostats-init my_project
       │
       ├──────────▶ Select template
       ├──────────▶ Create YAML file
       └──────────▶ Populate defaults
                         │
                         ▼
                   my_project.yaml created ✓


$ geostats-validate my_project.yaml
       │
       ├──────────▶ Load YAML
       ├──────────▶ Parse config
       ├──────────▶ Validate schema
       │                │
       │                ├─ Valid ────────▶ ✓ Config valid
       │                └─ Invalid ──────▶ ✗ Error details


$ geostats-run my_project.yaml
       │
       ├──────────▶ Load & validate config
       ├──────────▶ Initialize pipeline
       ├──────────▶ Execute workflow
       │                │
       │                ├─ Load data
       │                ├─ Preprocess
       │                ├─ Model variogram
       │                ├─ Perform kriging
       │                ├─ Validate
       │                ├─ Visualize
       │                └─ Save outputs
       │                      │
       │                      ▼
       └──────────────── ✓ Analysis complete
                         Results in output_dir/
```

## Component Relationships

```
┌─────────────────────────────────────────────────┐
│              geostats Package                    │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │ config/                                    │ │
│  │  ├─ schemas.py    (Pydantic models)       │ │
│  │  └─ parser.py     (YAML/JSON loading)     │ │
│  └────────────┬───────────────────────────────┘ │
│               │                                  │
│  ┌────────────▼───────────────────────────────┐ │
│  │ workflows/                                 │ │
│  │  └─ pipeline.py   (Orchestration)         │ │
│  └────────────┬───────────────────────────────┘ │
│               │                                  │
│  ┌────────────▼───────────────────────────────┐ │
│  │ algorithms/      (Existing modules)        │ │
│  │  ├─ variogram.py                           │ │
│  │  ├─ ordinary_kriging.py                    │ │
│  │  ├─ simple_kriging.py                      │ │
│  │  └─ ...                                    │ │
│  └────────────┬───────────────────────────────┘ │
│               │                                  │
│  ┌────────────▼───────────────────────────────┐ │
│  │ preprocessing/, transformations/           │ │
│  │ visualization/, validation/                │ │
│  │ io/, utils/, math/                         │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │ cli.py          (Command-line interface)  │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
└──────────────────────────────────────────────────┘

External Dependencies:
  pydantic  ────▶  Config validation
  pyyaml    ────▶  YAML parsing
  click     ────▶  CLI framework
```

## Config Schema Hierarchy

```
AnalysisConfig
├── ProjectConfig
│   ├── name: str
│   ├── output_dir: str
│   ├── description: Optional[str]
│   └── author: Optional[str]
│
├── DataConfig
│   ├── input_file: str
│   ├── x_column: str
│   ├── y_column: str
│   ├── z_column: str
│   ├── z_secondary: Optional[str]
│   ├── filter_column: Optional[str]
│   └── filter_value: Optional[Any]
│
├── PreprocessingConfig
│   ├── remove_outliers: bool
│   ├── outlier_method: Literal[...]
│   ├── transform: Optional[Literal[...]]
│   └── declustering: bool
│
├── VariogramConfig
│   ├── n_lags: int
│   ├── estimator: Literal[...]
│   ├── models: List[str]
│   ├── auto_fit: bool
│   └── check_anisotropy: bool
│
├── KrigingConfig
│   ├── method: Literal[...]
│   ├── neighborhood: NeighborhoodConfig
│   │   ├── max_neighbors: int
│   │   ├── min_neighbors: int
│   │   └── search_radius: Optional[float]
│   └── grid: GridConfig
│       ├── resolution: float
│       ├── nx: Optional[int]
│       └── ny: Optional[int]
│
├── ValidationConfig
│   ├── cross_validation: bool
│   ├── cv_method: Literal[...]
│   └── metrics: List[str]
│
├── VisualizationConfig
│   ├── style: Literal[...]
│   ├── plots: List[str]
│   └── dpi: int
│
└── OutputConfig
    ├── save_predictions: bool
    ├── formats: List[str]
    └── compression: bool
```

## Template Hierarchy

```
examples/configs/
│
├── basic_template.yaml
│   └── Minimal config (quick analysis)
│       ├── Ordinary kriging
│       ├── Auto variogram
│       ├── Default preprocessing
│       └── Basic plots
│
├── advanced_template.yaml
│   └── Complete workflow (all features)
│       ├── Outlier removal
│       ├── Box-Cox transform
│       ├── Declustering
│       ├── Anisotropy check
│       ├── Multiple models
│       └── All plots
│
├── gold_exploration_template.yaml
│   └── Mineral exploration (specialized)
│       ├── Log transform
│       ├── Cressie estimator (robust)
│       ├── Octant search
│       ├── Anisotropy analysis
│       └── Grade-appropriate plots
│
└── alaska_gold_example.yaml
    └── Real-world example (AGDB4 data)
        ├── Pre-configured paths
        ├── Alaska-specific settings
        └── Ready to run
```

## Validation Chain

```
User Input (YAML)
      │
      ▼
┌─────────────────┐
│ Syntax Check    │ ◀─── Valid YAML?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Schema Match    │ ◀─── All required fields?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Type Check      │ ◀─── Correct types?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Value Range     │ ◀─── Within constraints?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cross-Field     │ ◀─── Dependencies met?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ File Existence  │ ◀─── Input files exist?
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ✓ Valid Config  │
└─────────────────┘
```

## Integration Points

```
Config System    ←──────▶    Existing Algorithms
─────────────              ───────────────────────

preprocessing    ──────▶    transformations/
                               boxcox.py
                               log_transform.py
                               normal_score.py
                               declustering.py

variogram        ──────▶    algorithms/
                               variogram.py
                               fitting.py

kriging          ──────▶    algorithms/
                               ordinary_kriging.py
                               simple_kriging.py
                               universal_kriging.py
                               cokriging.py

validation       ──────▶    validation/
                               cross_validation.py
                               metrics.py

visualization    ──────▶    visualization/
                               variogram_plots.py
                               spatial_plots.py
                               diagnostic_plots.py
                               minimal_style.py

output           ──────▶    io/
                               formats.py
                               raster.py
                               tabular.py
```

---

*This diagram shows how all components work together in the config-driven architecture.*
