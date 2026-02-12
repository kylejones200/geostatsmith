Mining and Exploration Applications
====================================

This guide covers geostatistical workflows specific to mining and mineral exploration.

Typical Workflow
----------------

1. **Data Preparation**
   - Load drill hole data
   - Handle missing values
   - Check for clustering (declustering)
   - Transform if needed (log transform for grades)

2. **Variogram Analysis**
   - Calculate directional variograms
   - Check for anisotropy (mineralization direction)
   - Fit nested models if needed

3. **Resource Estimation**
   - Block kriging for mining blocks
   - Multiple realizations for uncertainty
   - Grade-tonnage curves

4. **Validation**
   - Cross-validation
   - Comparison with production data
   - Sensitivity analysis

Data Characteristics
--------------------

**Lognormal Distribution**
- Metal grades are typically lognormal
- Use lognormal kriging or log transformation

**Clustered Sampling**
- Drill holes often clustered
- Apply declustering before estimation

**Anisotropy**
- Mineralization often directional
- Use directional variograms

**Multiple Elements**
- Often analyze multiple elements
- Use cokriging if correlated

Example: Gold Grade Estimation
--------------------------------

.. code-block:: python

   import numpy as np
   from geostats.transformations import LogTransform
   from geostats.algorithms import (
       experimental_variogram,
       fit_variogram_model,
       BlockKriging
   )
   from geostats.simulation import SequentialGaussianSimulation

   # 1. Load drill hole data
   # Assume data loaded: x, y, z (east, north, gold grade)

   # 2. Log transform (gold grades are lognormal)
   log_transform = LogTransform()
   z_log = log_transform.transform(z)

   # 3. Calculate variogram
   lags, gamma, n_pairs = experimental_variogram(
       x, y, z_log,
       n_lags=15
   )

   # 4. Check for anisotropy
   # Calculate directional variograms
   lags_0, gamma_0, _ = experimental_variogram(
       x, y, z_log,
       angle=0,      # East-West
       tolerance=22.5
   )
   lags_90, gamma_90, _ = experimental_variogram(
       x, y, z_log,
       angle=90,     # North-South
       tolerance=22.5
   )

   # If different, use anisotropic model
   model = fit_variogram_model(
       lags, gamma,
       model_type='spherical',
       anisotropy_ratio=2.0,  # e.g., 2:1 anisotropy
       anisotropy_angle=45     # Direction of maximum continuity
   )

   # 5. Block kriging for mining blocks
   block_size = 10.0  # 10m x 10m blocks
   bk = BlockKriging(
       x, y, z_log,
       variogram_model=model,
       block_size=block_size
   )

   # Block centers
   x_blocks = np.arange(0, 1000, block_size)
   y_blocks = np.arange(0, 1000, block_size)
   X_blocks, Y_blocks = np.meshgrid(x_blocks, y_blocks)

   block_predictions_log, block_variance = bk.predict(
       X_blocks.flatten(),
       Y_blocks.flatten()
   )

   # 6. Back-transform to original scale
   block_predictions = log_transform.back_transform(block_predictions_log)

   # 7. Generate realizations for uncertainty
   sgs = SequentialGaussianSimulation(
       x, y, z_log,
       variogram_model=model
   )

   realizations_log = sgs.simulate(
       X_blocks.flatten(),
       Y_blocks.flatten(),
       n_realizations=100
   )

   # Back-transform
   realizations = log_transform.back_transform(realizations_log)

   # 8. Calculate grade-tonnage curves
   cutoff = 1.0  # g/t cutoff
   tonnage = []
   grade = []

   for realization in realizations:
       above_cutoff = realization > cutoff
       tonnage.append(np.sum(above_cutoff) * block_size**2)
       grade.append(np.mean(realization[above_cutoff]))

   # Uncertainty in grade-tonnage
   tonnage_mean = np.mean(tonnage)
   tonnage_std = np.std(tonnage)

Declustering
------------

Drill holes are often clustered in high-grade areas. Declustering corrects for this bias.

.. code-block:: python

   from geostats.transformations import cell_declustering

   # Apply declustering
   weights, declustered_mean = cell_declustering(
       x, y, z,
       cell_size=50.0  # Cell size for declustering
   )

   # Use weights in variogram calculation
   lags, gamma, n_pairs = experimental_variogram(
       x, y, z,
       weights=weights
   )

Multiple Elements
-----------------

When analyzing multiple elements, use cokriging if they're correlated.

.. code-block:: python

   from geostats.algorithms import Cokriging

   # Primary: Gold (sparse, expensive)
   # Secondary: Copper (dense, cheaper)

   ck = Cokriging(
       x, y,
       primary=gold_grade,
       secondary=copper_grade,
       variogram_models=[model_gold, model_copper, model_cross]
   )

   predictions = ck.predict(x_pred, y_pred)

Best Practices
--------------

1. **Always log-transform** metal grades
2. **Check for clustering** and apply declustering
3. **Analyze anisotropy** - mineralization is often directional
4. **Use block kriging** for mining blocks
5. **Generate multiple realizations** for uncertainty
6. **Validate with production data** when available
7. **Document assumptions** and methodology

Common Pitfalls
---------------

- **Forgetting to back-transform** after log kriging
- **Ignoring anisotropy** when present
- **Not declustering** clustered data
- **Using point kriging** when block estimates needed
- **Not quantifying uncertainty** with simulations
