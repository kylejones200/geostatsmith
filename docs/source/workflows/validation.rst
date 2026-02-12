Validation and Cross-Validation
================================

Validation is essential to ensure your geostatistical model is appropriate and reliable.

Cross-Validation
----------------

Leave-one-out cross-validation tests model performance by predicting each data point using all other points.

What to Check
~~~~~~~~~~~~~

1. **Mean Error (ME)**: Should be ≈ 0 (unbiased)
2. **Root Mean Squared Error (RMSE)**: Measure of accuracy
3. **Mean Standardized Error (MSE)**: Should be ≈ 0
4. **Variance of Standardized Errors**: Should be ≈ 1

Example
~~~~~~~

.. code-block:: python

   from geostats.algorithms import OrdinaryKriging
   from geostats.validation import cross_validate

   ok = OrdinaryKriging(x, y, z, variogram_model=model)

   cv_results = ok.cross_validate()

   print(f"Mean Error: {cv_results['mean_error']:.4f}")
   print(f"RMSE: {cv_results['rmse']:.4f}")
   print(f"Mean Standardized Error: {cv_results['mean_std_error']:.4f}")
   print(f"Variance of Standardized Errors: {cv_results['var_std_error']:.4f}")

   # Good validation:
   # - |ME| < 0.1 * std(data)
   # - MSE ≈ 0
   # - Var(SE) ≈ 1

Interpreting Results
--------------------

**Mean Error**
- Close to 0: Unbiased predictions
- Large positive: Systematic overestimation
- Large negative: Systematic underestimation

**RMSE**
- Lower is better
- Compare to data standard deviation
- RMSE < std(data): Model is useful

**Mean Standardized Error**
- Should be ≈ 0
- Systematic bias if not

**Variance of Standardized Errors**
- Should be ≈ 1
- < 1: Variance overestimated
- > 1: Variance underestimated

Diagnostic Plots
----------------

**Q-Q Plot**
- Check if standardized errors are normal
- Should follow 1:1 line

**Scatter Plot**
- Observed vs. Predicted
- Should follow 1:1 line
- Check for bias

**Residual Map**
- Spatial pattern in errors
- Should be random
- Clustering indicates problems

.. code-block:: python

   import matplotlib.pyplot as plt
   from scipy import stats

   # Q-Q plot
   standardized_errors = cv_results['standardized_errors']
   stats.probplot(standardized_errors, dist="norm", plot=plt)
   plt.show()

   # Scatter plot
   plt.scatter(cv_results['observed'], cv_results['predicted'])
   plt.plot([z.min(), z.max()], [z.min(), z.max()], 'r--')
   plt.xlabel('Observed')
   plt.ylabel('Predicted')
   plt.show()

   # Residual map
   residuals = cv_results['observed'] - cv_results['predicted']
   plt.scatter(x, y, c=residuals, cmap='RdBu')
   plt.colorbar(label='Residual')
   plt.show()

Common Issues
-------------

**High Mean Error**
- Problem: Systematic bias
- Causes: Poor variogram, non-stationarity
- Solutions: Re-fit variogram, check for trends

**High RMSE**
- Problem: Poor accuracy
- Causes: Wrong model, insufficient data
- Solutions: Try different models, add samples

**Variance of SE ≠ 1**
- Problem: Variance misestimated
- Causes: Poor variogram fit
- Solutions: Re-fit variogram, check nugget

**Spatial Pattern in Residuals**
- Problem: Model doesn't capture structure
- Causes: Missing anisotropy, wrong model
- Solutions: Check anisotropy, try different models

Best Practices
--------------

1. **Always validate** your model
2. **Check multiple statistics** (not just RMSE)
3. **Examine diagnostic plots**
4. **Compare different models**
5. **Document validation results**
