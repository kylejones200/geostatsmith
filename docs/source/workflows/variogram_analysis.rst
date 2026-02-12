Variogram Analysis
==================

The variogram is the foundation of geostatistics. This guide covers how to calculate, model, and interpret variograms in GeoStats.

Calculating Experimental Variograms
------------------------------------

The experimental variogram measures spatial correlation by calculating the average squared difference between pairs of points at different lag distances.

Basic Calculation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from geostats.algorithms import experimental_variogram
   import numpy as np

   # Sample data
   x = np.array([...])  # X coordinates
   y = np.array([...])  # Y coordinates
   z = np.array([...])  # Values

   # Calculate variogram
   lags, gamma, n_pairs = experimental_variogram(
       x, y, z,
       n_lags=15,        # Number of lag bins
       maxlag=None       # Auto-determine from data
   )

The result gives you:
- ``lags``: Lag distances (bin centers)
- ``gamma``: Semivariance values
- ``n_pairs``: Number of pairs in each bin

Choosing Parameters
~~~~~~~~~~~~~~~~~~~~

**Number of Lags (n_lags)**
- Too few: Lose detail in variogram structure
- Too many: Each bin has few pairs, noisy variogram
- Rule of thumb: 10-20 lags, ensure at least 30 pairs per bin

**Maximum Lag (maxlag)**
- Should be less than half the maximum distance
- Too large: Include pairs with no correlation
- Too small: Miss important structure
- Default: Half the maximum distance

**Lag Tolerance (lag_tol)**
- Controls binning width
- Larger: More pairs per bin, smoother variogram
- Smaller: More precise, but noisier
- Default: Half the lag width

Variogram Estimators
--------------------

Different estimators handle outliers and non-Gaussian data differently:

**Matheron's Estimator** (default)
- Classical estimator
- Sensitive to outliers
- Use for: Clean, Gaussian data

.. code-block:: python

   lags, gamma, n_pairs = experimental_variogram(
       x, y, z,
       estimator='matheron'
   )

**Cressie-Hawkins Estimator**
- Robust to outliers
- Use for: Data with outliers, skewed distributions

.. code-block:: python

   lags, gamma, n_pairs = experimental_variogram(
       x, y, z,
       estimator='cressie_hawkins'
   )

**Dowd Estimator**
- Very robust
- Use for: Highly contaminated data

Fitting Theoretical Models
---------------------------

The experimental variogram is noisy. Fitting a theoretical model provides:
- Smooth function for kriging
- Parameters (nugget, sill, range)
- Model validation

Available Models
~~~~~~~~~~~~~~~~

**Spherical Model**
- Most common in practice
- Linear behavior near origin
- Reaches sill at range
- Use for: Most applications, especially mining

.. code-block:: python

   from geostats.models import SphericalModel
   from geostats.algorithms import fit_variogram_model

   model = fit_variogram_model(
       lags, gamma,
       model_type='spherical',
       weights=n_pairs  # Weight by number of pairs
   )

**Exponential Model**
- Smooth near origin
- Approaches sill asymptotically
- Effective range = 3 × range parameter
- Use for: Smooth, continuous processes

**Gaussian Model**
- Very smooth near origin
- Approaches sill asymptotically
- Effective range = √3 × range parameter
- Use for: Very smooth processes (e.g., topography)

**Matérn Model**
- Flexible, includes smoothness parameter
- Can model various smoothness levels
- Use for: When you need flexibility

**Power Model**
- No sill (unbounded)
- Use for: Non-stationary processes
- Note: Cannot use with simple/ordinary kriging

Model Selection
~~~~~~~~~~~~~~

**Visual Inspection**
- Plot experimental and fitted variogram
- Check for good fit, especially near origin
- Origin behavior is critical for kriging

**Cross-Validation**
- Leave-one-out validation
- Compare different models
- Choose model with best CV statistics

**Automatic Selection**
- Use ``auto_fit_variogram()``
- Tests multiple models
- Selects based on cross-validation

.. code-block:: python

   from geostats.algorithms import auto_fit_variogram

   model = auto_fit_variogram(
       lags, gamma,
       weights=n_pairs
   )

Interpreting Variogram Parameters
----------------------------------

**Nugget Effect**
- Variance at zero distance
- Sources: Measurement error, micro-scale variation
- High nugget: Noisy data or micro-scale variability

**Sill**
- Maximum variance (total variance)
- Should match sample variance (approximately)
- Much higher: Non-stationarity, outliers
- Much lower: Clustering, bias

**Range**
- Distance of spatial correlation
- Beyond range: No correlation
- Practical range: Distance where variogram reaches 95% of sill

**Anisotropy**
- Direction-dependent correlation
- Common in geology (e.g., mineralization direction)
- Requires directional variograms

Common Issues and Solutions
----------------------------

**No Clear Structure**
- Problem: Variogram is flat or noisy
- Causes: No spatial correlation, too few samples
- Solution: Check data, increase lag tolerance

**Nugget Too High**
- Problem: High variance at origin
- Causes: Measurement error, micro-scale variation
- Solution: Check data quality, consider robust estimators

**Sill Too High**
- Problem: Sill much higher than sample variance
- Causes: Non-stationarity, outliers
- Solution: Check for trends, remove outliers

**Poor Fit Near Origin**
- Problem: Model doesn't match experimental near origin
- Causes: Wrong model type, insufficient data
- Solution: Try different models, increase data density

**Negative Weights in Kriging**
- Problem: Variogram model issues
- Causes: Poor model fit, negative correlations
- Solution: Re-fit variogram, check for errors

Best Practices
--------------

1. **Always plot the experimental variogram** before fitting
2. **Check number of pairs** in each bin (should be >30)
3. **Validate model** with cross-validation
4. **Consider anisotropy** if data shows directional patterns
5. **Use robust estimators** for contaminated data
6. **Document your choices** and rationale

Example: Complete Workflow
---------------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from geostats.algorithms import (
       experimental_variogram,
       fit_variogram_model,
       cross_validate_variogram
   )

   # 1. Calculate experimental variogram
   lags, gamma, n_pairs = experimental_variogram(
       x, y, z,
       n_lags=15,
       estimator='matheron'
   )

   # 2. Check data quality
   print(f"Pairs per bin: {n_pairs.min()} to {n_pairs.max()}")
   if n_pairs.min() < 30:
       print("Warning: Some bins have few pairs")

   # 3. Fit model
   model = fit_variogram_model(
       lags, gamma,
       model_type='spherical',
       weights=n_pairs
   )

   # 4. Validate
   cv_results = cross_validate_variogram(
       x, y, z, model
   )
   print(f"Mean error: {cv_results['mean_error']:.4f}")
   print(f"RMSE: {cv_results['rmse']:.4f}")

   # 5. Plot
   plt.figure(figsize=(10, 6))
   plt.scatter(lags, gamma, s=n_pairs, alpha=0.6, label='Experimental')
   lag_fit = np.linspace(0, lags.max(), 100)
   gamma_fit = model.predict(lag_fit)
   plt.plot(lag_fit, gamma_fit, 'r-', label='Fitted Model')
   plt.xlabel('Lag Distance')
   plt.ylabel('Semivariance')
   plt.legend()
   plt.title('Variogram')
   plt.show()
