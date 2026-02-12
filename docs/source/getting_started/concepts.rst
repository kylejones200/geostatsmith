Geostatistical Concepts
========================

This section introduces the fundamental concepts of geostatistics that you'll need to understand to use GeoStats effectively.

What is Geostatistics?
----------------------

Geostatistics is a branch of statistics that deals with spatial or spatiotemporal datasets. Unlike classical statistics, which assumes independence between observations, geostatistics explicitly models spatial correlation through the variogram.

Key Concepts
------------

Spatial Correlation
~~~~~~~~~~~~~~~~~~~

Spatial correlation means that nearby samples are more similar than distant ones. This is captured by the variogram, which measures how dissimilarity increases with distance.

The Variogram
~~~~~~~~~~~~~

The variogram (or semivariogram) is the fundamental tool of geostatistics:

.. math::

   \gamma(h) = \frac{1}{2N(h)} \sum_{i=1}^{N(h)} [z(x_i) - z(x_i + h)]^2

where:
- :math:`h` is the lag distance
- :math:`N(h)` is the number of pairs at lag :math:`h`
- :math:`z(x_i)` is the value at location :math:`x_i`

The variogram typically shows:
- **Nugget**: Variance at zero distance (measurement error, micro-scale variation)
- **Sill**: Maximum variance (total variance of the process)
- **Range**: Distance at which correlation becomes negligible

Kriging
~~~~~~~

Kriging is a best linear unbiased predictor (BLUP) that uses the variogram to make predictions at unsampled locations. It provides:
- **Predictions**: Best estimate of the value
- **Variance**: Uncertainty quantification

Different kriging methods:
- **Simple Kriging**: Assumes known mean
- **Ordinary Kriging**: Estimates local mean
- **Universal Kriging**: Handles trends (drift)

Stationarity
~~~~~~~~~~~~

Most geostatistical methods assume some form of stationarity:
- **Second-order stationarity**: Mean and covariance are constant
- **Intrinsic stationarity**: Mean is constant, variogram exists

In practice, we often work with:
- **Quasi-stationarity**: Stationarity within local neighborhoods
- **Trend removal**: Detrending before analysis

The Geostatistical Workflow
----------------------------

1. **Exploratory Data Analysis**
   - Check for outliers
   - Assess distribution
   - Identify trends
   - Check for clustering

2. **Variogram Analysis**
   - Calculate experimental variogram
   - Fit theoretical model
   - Check for anisotropy
   - Validate model

3. **Interpolation/Simulation**
   - Choose appropriate method
   - Perform kriging or simulation
   - Validate results

4. **Uncertainty Quantification**
   - Assess prediction variance
   - Generate multiple realizations
   - Calculate probability maps

When to Use What?
-----------------

**Ordinary Kriging**: Default choice for most applications. Use when:
- Mean is unknown but assumed constant
- No strong trends present
- Standard interpolation needed

**Universal Kriging**: Use when:
- Clear trends in the data
- Need to model drift explicitly
- Trend removal not appropriate

**Indicator Kriging**: Use when:
- Need probability of exceeding threshold
- Non-Gaussian data
- Risk assessment

**Simulation**: Use when:
- Need multiple realizations
- Uncertainty propagation
- Non-linear operations on predictions

**Cokriging**: Use when:
- Secondary variable available
- Secondary variable more densely sampled
- Correlation between variables

Further Reading
---------------

- Matheron, G. (1963). Principles of geostatistics.
- Journel, A. G., & Huijbregts, C. J. (1978). Mining geostatistics.
- Goovaerts, P. (1997). Geostatistics for natural resources evaluation.
