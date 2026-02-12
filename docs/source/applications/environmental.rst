Environmental Applications
===========================

This guide covers geostatistical workflows for environmental monitoring and assessment.

Common Applications
-------------------

- Soil contamination mapping
- Groundwater quality assessment
- Air pollution monitoring
- Remediation planning
- Risk assessment

Typical Workflow
----------------

1. **Data Quality Assessment**
   - Check for outliers
   - Assess detection limits
   - Handle censored data

2. **Spatial Analysis**
   - Calculate variogram
   - Check for trends
   - Identify hotspots

3. **Interpolation**
   - Kriging for mapping
   - Indicator kriging for risk
   - Uncertainty quantification

4. **Risk Assessment**
   - Probability of exceeding thresholds
   - Contamination extent
   - Remediation planning

Handling Censored Data
-----------------------

Environmental data often has detection limits (non-detects).

.. code-block:: python

   # Replace non-detects with detection limit / 2
   detection_limit = 0.5
   z_censored = np.where(z < detection_limit, detection_limit / 2, z)

   # Or use indicator kriging
   from geostats.algorithms import IndicatorKriging

   ik = IndicatorKriging(
       x, y, z,
       threshold=detection_limit,
       variogram_model=model
   )

   # Probability of exceeding detection limit
   probability, _ = ik.predict(x_pred, y_pred)

Outlier Detection
-----------------

Environmental data often has outliers from contamination events.

.. code-block:: python

   from geostats.validation import detect_outliers

   outliers = detect_outliers(x, y, z, method='iqr')
   
   # Remove or flag outliers
   z_clean = z[~outliers]

   # Or use robust variogram estimator
   lags, gamma, n_pairs = experimental_variogram(
       x, y, z,
       estimator='cressie_hawkins'  # Robust to outliers
   )

Risk Assessment
--------------

Use indicator kriging for probability maps.

.. code-block:: python

   # Regulatory threshold
   threshold = 10.0  # e.g., mg/kg

   ik = IndicatorKriging(
       x, y, z,
       threshold=threshold,
       variogram_model=model
   )

   # Probability map
   probability, variance = ik.predict(x_pred, y_pred)

   # Areas with high probability need attention
   risk_areas = probability > 0.5

Example: Soil Contamination
----------------------------

.. code-block:: python

   import numpy as np
   from geostats.algorithms import (
       experimental_variogram,
       fit_variogram_model,
       IndicatorKriging,
       OrdinaryKriging
   )

   # 1. Load soil sample data
   # x, y, z (coordinates and contaminant concentration)

   # 2. Check for outliers
   outliers = detect_outliers(x, y, z)
   print(f"Found {np.sum(outliers)} outliers")

   # 3. Calculate variogram (use robust estimator)
   lags, gamma, n_pairs = experimental_variogram(
       x, y, z,
       estimator='cressie_hawkins',
       n_lags=15
   )

   # 4. Fit model
   model = fit_variogram_model(
       lags, gamma,
       model_type='spherical',
       weights=n_pairs
   )

   # 5. Create contamination map
   ok = OrdinaryKriging(
       x, y, z,
       variogram_model=model
   )

   predictions, variance = ok.predict(x_pred, y_pred)

   # 6. Risk assessment (regulatory threshold)
   threshold = 10.0
   ik = IndicatorKriging(
       x, y, z,
       threshold=threshold,
       variogram_model=model
   )

   probability, _ = ik.predict(x_pred, y_pred)

   # 7. Identify areas needing remediation
   remediation_areas = probability > 0.7

Best Practices
--------------

1. **Handle detection limits** appropriately
2. **Use robust methods** for contaminated data
3. **Check for outliers** before analysis
4. **Use indicator kriging** for risk assessment
5. **Quantify uncertainty** with variance or simulation
6. **Document assumptions** and methodology
