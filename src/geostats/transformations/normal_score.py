"""
Normal Score Transform

Implementation based on Olea (2009) §2134-2177:
"The transformation is performed by assigning to the cumulative probability of
every observation the value of the standard normal distribution for the same
cumulative probability."

Reference:
- ofr20091103.txt (USGS Practical Primer)
- Page 102-103: Normal Scores transformation
- Used for Sequential Gaussian Simulation (SGS)
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.interpolate import interp1d

from ..core.constants import RANK_OFFSET
from ..core.logging_config import get_logger
import logging

logger = logging.getLogger(__name__)

logger = get_logger(__name__)

class NormalScoreTransform:
class NormalScoreTransform:
 Normal Score Transform for geostatistical data

 Transforms data to follow a standard normal distribution while
 preserving the rank order. Essential preprocessing for:
 - Sequential Gaussian Simulation (SGS)
 - Simple Kriging of normal scores
 - Methods assuming multivariate normality

 The transformation maps empirical CDF values to standard normal
 quantiles, ensuring the transformed data has mean=0 and variance=1.
 """

 def __init__(self):
 def __init__(self):
     self.original_values: Optional[npt.NDArray[np.float64]] = None
     self.normal_scores: Optional[npt.NDArray[np.float64]] = None
     self.forward_interp: Optional[interp1d] = None
     self.backward_interp: Optional[interp1d] = None
     self.is_fitted: bool = False

 def fit(self, data: npt.NDArray[np.float64]) -> 'NormalScoreTransform':
 def fit(self, data: npt.NDArray[np.float64]) -> 'NormalScoreTransform':
     Fit the normal score transform to data

 Parameters
 ----------
 data : np.ndarray
 Original data values (1D array)

 Returns
 -------
 self : NormalScoreTransform
 Fitted transformer
 """
 data = np.asarray(data, dtype=np.float64).flatten()

 # Remove NaN values
 valid_data = data[~np.isnan(data)]
 n_valid = len(valid_data)

 if n_valid == 0:

     logger.debug(f"Fitting normal score transform on {n_valid} valid points")

 # Sort data and get ranks (fully vectorized)
 sorted_data = np.sort(valid_data)

 # Calculate empirical cumulative probabilities
 # Using (i - RANK_OFFSET) / n to avoid 0 and 1 at boundaries
 ranks = np.arange(1, n_valid + 1, dtype=np.float64)
 empirical_cdf = (ranks - RANK_OFFSET) / n_valid

 # Map to standard normal quantiles
 # norm.ppf is the inverse CDF (quantile function)
 normal_scores = stats.norm.ppf(empirical_cdf)

 # Store for back-transformation
 self.original_values = sorted_data.copy()
 self.normal_scores = normal_scores.copy()

 # Create interpolation functions for forward and backward transforms
 # Handle duplicate values by taking unique values
 unique_orig, unique_idx = np.unique(sorted_data, return_index=True)
 unique_scores = normal_scores[unique_idx]

 # Forward: original -> normal score
 self.forward_interp = interp1d(
 unique_orig,
 unique_scores,
 kind='linear',
 bounds_error=False,
 fill_value=(unique_scores[0], unique_scores[-1]) # Extrapolate at boundaries
 )

 # Backward: normal score -> original
 self.backward_interp = interp1d(
 unique_scores,
 unique_orig,
 kind='linear',
 bounds_error=False,
 fill_value=(unique_orig[0], unique_orig[-1]) # Extrapolate at boundaries
 )

 self.is_fitted = True
 return self

 def transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
 def transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     Transform data to normal scores

 Parameters
 ----------
 data : np.ndarray
 Data to transform

 Returns
 -------
 np.ndarray
 Normal scores (mean≈0, std≈1)
 """
 if not self.is_fitted:

     data = np.asarray(data, dtype=np.float64)
 original_shape = data.shape
 data_flat = data.flatten()

 # Transform using interpolation
 transformed = self.forward_interp(data_flat)

 return transformed.reshape(original_shape)

 def inverse_transform(self, normal_scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
 def inverse_transform(self, normal_scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     Back-transform normal scores to original scale

 Parameters
 ----------
 normal_scores : np.ndarray
 Normal scores to back-transform

 Returns
 -------
 np.ndarray
 Values in original data space

 Notes
 -----
 From Olea (2009): "Backtransformation of estimated normal scores outside
 the interval of variation for the normal scores of the data is highly uncertain."

 Values outside the training range are extrapolated using the boundary values.
 """
 if not self.is_fitted:

     normal_scores = np.asarray(normal_scores, dtype=np.float64)
 original_shape = normal_scores.shape
 scores_flat = normal_scores.flatten()

 # Back-transform using interpolation
 back_transformed = self.backward_interp(scores_flat)

 return back_transformed.reshape(original_shape)

 def fit_transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
 def fit_transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
     Fit and transform data in one step

 Parameters
 ----------
 data : np.ndarray
 Data to transform

 Returns
 -------
 np.ndarray
 Normal scores
 """
 self.fit(data)
 return self.transform(data)

 def get_statistics(self) -> Dict[str, float]:
 def get_statistics(self) -> Dict[str, float]:
     Get statistics of the fitted transform

 Returns
 -------
 Dict[str, float]
 Statistics including original and transformed statistics
 """
 if not self.is_fitted:

     assert self.original_values is not None
 assert self.normal_scores is not None

 return {
 'n_samples': float(len(self.original_values)),
 'original_mean': float(np.mean(self.original_values)),
 'original_std': float(np.std(self.original_values)),
 'original_min': float(np.min(self.original_values)),
 'original_max': float(np.max(self.original_values)),
 'scores_mean': float(np.mean(self.normal_scores)),
 'scores_std': float(np.std(self.normal_scores)),
 'scores_min': float(np.min(self.normal_scores)),
 'scores_max': float(np.max(self.normal_scores)),
 }

def normal_score_transform(
def normal_score_transform(
    ) -> Tuple[npt.NDArray[np.float64], NormalScoreTransform]:
 """
 Convenience function for normal score transform

 Parameters
 ----------
 data : np.ndarray
 Data to transform

 Returns
 -------
 transformed_data : np.ndarray
 Normal scores
 transformer : NormalScoreTransform
 Fitted transformer (for back-transformation)

 Examples
 --------
 >>> import numpy as np
 >>> data = np.array([1.2, 5.6, 2.3, 8.9, 3.4])
 >>> normal_scores, transformer = normal_score_transform(data)
 >>> logger.info(normal_scores) # Should have mean≈0, std≈1
 >>> original = transformer.inverse_transform(normal_scores)
 >>> np.allclose(original, data) # Should be True
 """
 transformer = NormalScoreTransform()
 transformed = transformer.fit_transform(data)
 return transformed, transformer

def back_transform(
def back_transform(
 transformer: NormalScoreTransform
    ) -> npt.NDArray[np.float64]:
 """
 Back-transform normal scores to original scale

 Parameters
 ----------
 normal_scores : np.ndarray
 Normal scores from SGS or kriging
 transformer : NormalScoreTransform
 Fitted transformer from forward transform

 Returns
 -------
 np.ndarray
 Values in original data space
 """
 return transformer.inverse_transform(normal_scores)
