Method Selection Guide
======================

Choosing the right geostatistical method for your problem is critical. This guide helps you make informed decisions.

Decision Framework
------------------

Data Characteristics
~~~~~~~~~~~~~~~~~~~~

**Distribution**
- Gaussian → Standard kriging
- Lognormal → Lognormal kriging or log transform
- Skewed → Transform or robust methods
- Bimodal → Indicator kriging

**Spatial Structure**
- Stationary → Ordinary kriging
- Trend present → Universal kriging
- Anisotropic → Directional variograms
- Non-stationary → Local methods

**Sample Density**
- Dense → Standard methods
- Sparse → Cokriging if secondary variable available
- Clustered → Declustering required

**Secondary Variables**
- Available and correlated → Cokriging
- Dense secondary → Collocated cokriging
- Not available → Standard kriging

Problem Type
~~~~~~~~~~~~

**Interpolation**
- Single value needed → Kriging
- Uncertainty needed → Kriging with variance
- Multiple scenarios → Simulation

**Risk Assessment**
- Probability maps → Indicator kriging
- Exceedance probability → Indicator kriging
- Threshold-based → Indicator kriging

**Resource Estimation**
- Block estimates → Block kriging
- Uncertainty → Simulation
- Grade-tonnage → Multiple realizations

**Optimization**
- Sampling design → Optimization tools
- Cost-benefit → Optimization tools

Method Comparison
----------------

+------------------+------------------+------------------+------------------+
| Method           | When to Use      | Advantages       | Limitations      |
+==================+==================+==================+==================+
| Ordinary Kriging | Default choice,  | Simple, robust,  | Assumes constant |
|                  | unknown mean     | unbiased         | mean             |
+------------------+------------------+------------------+------------------+
| Universal        | Trends present   | Handles trends   | More complex,    |
| Kriging          |                  |                  | more parameters  |
+------------------+------------------+------------------+------------------+
| Indicator        | Probabilities,   | Non-parametric,  | Less efficient   |
| Kriging          | risk assessment  | robust           | than OK          |
+------------------+------------------+------------------+------------------+
| Cokriging        | Secondary        | Uses more info,  | Requires cross- |
|                  | variable         | better accuracy  | variogram        |
+------------------+------------------+------------------+------------------+
| Block Kriging    | Block estimates | Correct support,  | More computation |
|                  |                  | mining blocks    |                  |
+------------------+------------------+------------------+------------------+
| Simulation       | Uncertainty,     | Full uncertainty,| Computationally |
|                  | multiple        | non-linear ops   | intensive        |
|                  | realizations     |                  |                  |
+------------------+------------------+------------------+------------------+

Quick Reference
---------------

**Standard Interpolation**
→ Ordinary Kriging

**Trends in Data**
→ Universal Kriging

**Need Probabilities**
→ Indicator Kriging

**Lognormal Data**
→ Lognormal Kriging or Log Transform + OK

**Secondary Variable**
→ Cokriging

**Block Estimates**
→ Block Kriging

**Uncertainty Quantification**
→ Simulation

**Clustered Data**
→ Declustering + Standard Method

**Anisotropic**
→ Directional Variograms + Standard Method

**Non-Gaussian**
→ Transform or Indicator Kriging

**Outliers Present**
→ Robust Estimators or Indicator Kriging
