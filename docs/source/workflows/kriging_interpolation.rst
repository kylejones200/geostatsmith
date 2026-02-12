Kriging Interpolation
=====================

Kriging is the core interpolation method in geostatistics. This guide covers how to choose and use the right kriging method for your application.

Choosing a Kriging Method
--------------------------

Decision Tree
~~~~~~~~~~~~~

1. **Do you have a secondary variable?**
   - Yes → Cokriging or Collocated Cokriging
   - No → Continue

2. **Is the mean known and constant?**
   - Yes → Simple Kriging
   - No → Continue

3. **Are there trends in the data?**
   - Yes → Universal Kriging
   - No → Ordinary Kriging

4. **Do you need probabilities?**
   - Yes → Indicator Kriging
   - No → Continue

5. **Is data lognormal?**
   - Yes → Lognormal Kriging
   - No → Continue

6. **Default**: Ordinary Kriging

Ordinary Kriging
----------------

The most commonly used method. Assumes unknown but constant mean.

When to Use
~~~~~~~~~~~

- Default choice for most applications
- Mean is unknown
- No strong trends
- Standard interpolation

Example
~~~~~~~

.. code-block:: python

   from geostats.algorithms import OrdinaryKriging
   from geostats.models import SphericalModel

   # Create variogram model
   variogram_model = SphericalModel(
       nugget=0.1,
       sill=1.0,
       range=100.0
   )

   # Create kriging object
   ok = OrdinaryKriging(
       x, y, z,
       variogram_model=variogram_model
   )

   # Predict at new locations
   x_pred = np.linspace(0, 1000, 100)
   y_pred = np.linspace(0, 1000, 100)
   X_pred, Y_pred = np.meshgrid(x_pred, y_pred)

   predictions, variance = ok.predict(
       X_pred.flatten(),
       Y_pred.flatten()
   )

   # Reshape for plotting
   Z_pred = predictions.reshape(X_pred.shape)
   Z_var = variance.reshape(X_pred.shape)

Neighborhood Search
~~~~~~~~~~~~~~~~~~

For large datasets, use neighborhood search to limit the number of samples used:

.. code-block:: python

   from geostats.algorithms import NeighborhoodSearch, NeighborhoodConfig

   config = NeighborhoodConfig(
       max_neighbors=20,
       min_neighbors=8,
       search_radius=200.0
   )

   ok = OrdinaryKriging(
       x, y, z,
       variogram_model=variogram_model,
       neighborhood_config=config
   )

Universal Kriging
-----------------

Handles trends (drift) in the data.

When to Use
~~~~~~~~~~~

- Clear trends in the data
- Trend cannot be removed
- Need to model drift explicitly

Example
~~~~~~~

.. code-block:: python

   from geostats.algorithms import UniversalKriging

   # Define drift functions (e.g., linear trend)
   def drift_x(x, y):
       return x

   def drift_y(x, y):
       return y

   uk = UniversalKriging(
       x, y, z,
       variogram_model=variogram_model,
       drift_functions=[drift_x, drift_y]
   )

   predictions, variance = uk.predict(x_pred, y_pred)

Indicator Kriging
-----------------

Estimates probability of exceeding a threshold.

When to Use
~~~~~~~~~~~

- Need probability maps
- Risk assessment
- Non-Gaussian data
- Threshold-based decisions

Example
~~~~~~~

.. code-block:: python

   from geostats.algorithms import IndicatorKriging

   threshold = 50.0  # e.g., contamination threshold

   ik = IndicatorKriging(
       x, y, z,
       threshold=threshold,
       variogram_model=variogram_model
   )

   # Get probability of exceeding threshold
   probability, variance = ik.predict(x_pred, y_pred)

   # Probability map shows risk areas
   prob_map = probability.reshape(X_pred.shape)

Cokriging
---------

Uses a secondary variable to improve predictions.

When to Use
~~~~~~~~~~~

- Secondary variable available
- Secondary variable more densely sampled
- Correlation between variables
- Cost-effective data collection

Example
~~~~~~~

.. code-block:: python

   from geostats.algorithms import Cokriging

   # Primary variable (sparse, expensive)
   z1 = np.array([...])  # e.g., gold grade

   # Secondary variable (dense, cheap)
   z2 = np.array([...])  # e.g., geophysical survey

   # Variogram models
   model1 = SphericalModel(...)  # Primary
   model2 = SphericalModel(...)  # Secondary
   model_cross = SphericalModel(...)  # Cross-variogram

   ck = Cokriging(
       x, y,
       primary=z1,
       secondary=z2,
       variogram_models=[model1, model2, model_cross]
   )

   predictions = ck.predict(x_pred, y_pred)

Block Kriging
-------------

Averages predictions over a support (block size).

When to Use
~~~~~~~~~~~

- Need block estimates (e.g., mining blocks)
- Point-to-block conversion
- Resource estimation

Example
~~~~~~~

.. code-block:: python

   from geostats.algorithms import BlockKriging

   block_size = 10.0  # 10x10 block

   bk = BlockKriging(
       x, y, z,
       variogram_model=variogram_model,
       block_size=block_size
   )

   # Predict block averages
   block_predictions, block_variance = bk.predict(
       x_block_centers,
       y_block_centers
   )

Validation
----------

Always validate your kriging results.

Cross-Validation
~~~~~~~~~~~~~~~

.. code-block:: python

   cv_results = ok.cross_validate()

   print(f"Mean error: {cv_results['mean_error']:.4f}")
   print(f"RMSE: {cv_results['rmse']:.4f}")
   print(f"Mean standardized error: {cv_results['mean_std_error']:.4f}")
   print(f"Variance of standardized errors: {cv_results['var_std_error']:.4f}")

   # Good validation:
   # - Mean error ≈ 0 (unbiased)
   # - Mean standardized error ≈ 0
   # - Variance of standardized errors ≈ 1

Common Issues
-------------

**High Prediction Variance**
- Causes: Too few samples, poor variogram fit
- Solutions: Add samples, re-fit variogram, use larger neighborhood

**Negative Kriging Weights**
- Causes: Poor variogram model, negative correlations
- Solutions: Re-fit variogram, check for errors

**Smooth Predictions**
- Causes: High nugget, large range
- Solutions: Check variogram parameters, consider different model

**Edge Effects**
- Causes: Predictions near data boundaries
- Solutions: Use larger search radius, be cautious near edges

Best Practices
--------------

1. **Always validate** with cross-validation
2. **Check kriging variance** for uncertainty
3. **Use appropriate neighborhood** for large datasets
4. **Document your choices** and assumptions
5. **Consider multiple methods** and compare results
