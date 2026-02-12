Geostatistical Simulation
==========================

Simulation generates multiple realizations of a spatial field, providing full uncertainty quantification.

When to Use Simulation
-----------------------

- **Uncertainty Quantification**: Need full uncertainty distribution
- **Non-linear Operations**: Operations on predictions (e.g., grade-tonnage)
- **Risk Assessment**: Multiple scenarios
- **Optimization**: Under uncertainty

Simulation vs. Kriging
----------------------

**Kriging**
- Single "best" estimate
- Provides variance (uncertainty measure)
- Smooth predictions
- Use for: Mapping, interpolation

**Simulation**
- Multiple realizations
- Full uncertainty distribution
- Reproduces spatial variability
- Use for: Uncertainty propagation, risk assessment

Sequential Gaussian Simulation (SGS)
-------------------------------------

Most common simulation method. Assumes Gaussian distribution.

Workflow
~~~~~~~~

1. Transform to Gaussian (if needed)
2. Calculate variogram in Gaussian space
3. Simulate in Gaussian space
4. Back-transform to original space

Example
~~~~~~~

.. code-block:: python

   from geostats.simulation import SequentialGaussianSimulation
   from geostats.transformations import NormalScoreTransform

   # 1. Transform to Gaussian
   transform = NormalScoreTransform()
   z_gaussian = transform.transform(z)

   # 2. Calculate variogram in Gaussian space
   lags, gamma, n_pairs = experimental_variogram(
       x, y, z_gaussian
   )
   model = fit_variogram_model(lags, gamma)

   # 3. Simulate
   sgs = SequentialGaussianSimulation(
       x, y, z_gaussian,
       variogram_model=model
   )

   # Generate realizations
   realizations_gaussian = sgs.simulate(
       x_grid, y_grid,
       n_realizations=100
   )

   # 4. Back-transform
   realizations = transform.back_transform(realizations_gaussian)

   # 5. Analyze uncertainty
   mean_map = np.mean(realizations, axis=0)
   std_map = np.std(realizations, axis=0)
   p10_map = np.percentile(realizations, 10, axis=0)
   p90_map = np.percentile(realizations, 90, axis=0)

Conditional vs. Unconditional
------------------------------

**Conditional Simulation**
- Honors data values
- Use for: Realistic scenarios, uncertainty quantification

**Unconditional Simulation**
- Doesn't honor data
- Use for: Exploring variability, testing methods

.. code-block:: python

   # Conditional (default)
   sgs = SequentialGaussianSimulation(
       x, y, z,
       variogram_model=model,
       conditional=True  # Honors data
   )

   # Unconditional
   sgs = SequentialGaussianSimulation(
       x, y, z,
       variogram_model=model,
       conditional=False  # Doesn't honor data
   )

Sequential Indicator Simulation (SIS)
---------------------------------------

For non-Gaussian or categorical data.

.. code-block:: python

   from geostats.simulation import SequentialIndicatorSimulation

   # Define categories/thresholds
   thresholds = [10, 20, 30]

   sis = SequentialIndicatorSimulation(
       x, y, z,
       thresholds=thresholds,
       variogram_models=[model1, model2, model3]  # One per threshold
   )

   realizations = sis.simulate(x_grid, y_grid, n_realizations=100)

Analyzing Results
-----------------

**Mean and Standard Deviation**
- Mean: Best estimate (similar to kriging)
- Std: Uncertainty measure

**Percentiles**
- P10, P50, P90: Uncertainty bounds
- P10-P90: 80% confidence interval

**Probability Maps**
- Probability of exceeding threshold
- Risk assessment

.. code-block:: python

   # Probability of exceeding threshold
   threshold = 50.0
   probability = np.mean(realizations > threshold, axis=0)

   # Percentiles
   p10 = np.percentile(realizations, 10, axis=0)
   p50 = np.percentile(realizations, 50, axis=0)
   p90 = np.percentile(realizations, 90, axis=0)

Best Practices
--------------

1. **Use enough realizations** (typically 100-1000)
2. **Check convergence** - mean should stabilize
3. **Validate** - check statistics match data
4. **Transform appropriately** - ensure Gaussian for SGS
5. **Document assumptions** and methodology
