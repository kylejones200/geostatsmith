"""
    Logarithmic Transform

Implementation based on Olea (2009) §2162-2176:
    pass
"
is simply the logarithm of the original observation."

"
to lognormal, which explains the great attention paid to this distribution since
the early days of geostatistics."

Reference:
    pass
- ofr20091103.txt (USGS Practical Primer)
- Lognormal transformation for kriging (§2805-2810)
"""

from typing import Tuple, Optional
import numpy as np
import numpy.typing as npt
import warnings
import logging

logger = logging.getLogger(__name__)

class LogTransform:
    """
    Logarithmic Transform for geostatistical data

    Useful for data that follows a lognormal distribution, common in:
    - Mineral concentrations
    - Chemical element distributions
    - Permeability values
    - Many environmental concentrations

    Handles zero and negative values appropriately.
    """

    def __init__(self, base: str = 'natural', epsilon: Optional[float] = None):
        """
        Initialize Log Transform.

        Parameters
        ----------
        base : str, optional
            'natural' for ln, '10' for log10, '2' for log2
        epsilon : float, optional
            Small value to add to zeros before logging.
            If None, automatically set to 1% of minimum positive value
        """
        self.base = base
        self.epsilon_user = epsilon
        self.epsilon_fitted: Optional[float] = None
        self.min_original: Optional[float] = None
        self.max_original: Optional[float] = None
        self.has_zeros: bool = False
        self.has_negatives: bool = False
        self.is_fitted: bool = False

        # Set log/exp functions using dispatch
        log_functions = {
        'natural': (np.log, np.exp),
        '10': (np.log10, lambda x: np.power(10, x)),
        '2': (np.log2, lambda x: np.power(2, x)),
        }

        if base not in log_functions:
            raise ValueError(
                f"base must be one of {valid_bases}, got '{base}'"
        )

        self.log_func, self.exp_func = log_functions[base]

    def fit(self, data: npt.NDArray[np.float64]) -> 'LogTransform':
        """
        Fit the log transform to data.

        Parameters
        ----------
        data : np.ndarray
            Original data values

        Returns
        -------
        self : LogTransform
            Fitted transformer
        """
        data = np.asarray(data, dtype=np.float64).flatten()
        valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            raise ValueError("No valid (positive) data found for log transform")

        self.min_original = np.min(valid_data)
        self.max_original = np.max(valid_data)

        # Check for zeros and negatives
        self.has_zeros = np.any(valid_data == 0)
        self.has_negatives = np.any(valid_data < 0)

        if self.has_negatives:
            raise ValueError(
                "Log transform cannot handle negative values. "
                "Consider adding a constant or using a different transform."
            )

        # Determine epsilon for zeros
        if self.has_zeros:
            self.epsilon_fitted = self.epsilon_user
        else:
            positive_values = valid_data[valid_data > 0]
            if len(positive_values) > 0:
                self.epsilon_fitted = np.min(positive_values) * 0.01
            else:
                self.epsilon_fitted = 1e-10

        if self.has_zeros:
            import warnings
            warnings.warn(
                f"Data contains zeros. Adding epsilon={self.epsilon_fitted:.2e} "
                "before log transform."
            )

        self.is_fitted = True
        return self

    def transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Transform data using logarithm.

        Parameters
        ----------
        data : np.ndarray
            Data to transform

        Returns
        -------
        np.ndarray
            Log-transformed data
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")

        data = np.asarray(data, dtype=np.float64)
        original_shape = data.shape
        data_flat = data.flatten()

        # Add epsilon to handle zeros
        if self.epsilon_fitted > 0:
            data_flat = data_flat + self.epsilon_fitted

        # Transform
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed = self.log_func(data_flat)

        return transformed.reshape(original_shape)

    def inverse_transform(self, log_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Back-transform log data to original scale.

        Parameters
        ----------
        log_data : np.ndarray
            Log-transformed data

        Returns
        -------
        np.ndarray
            Values in original data space
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before inverse_transform")

        log_data = np.asarray(log_data, dtype=np.float64)
        original_shape = log_data.shape
        log_flat = log_data.flatten()

        # Exponentiate
        back_transformed = self.exp_func(log_flat)

        # Subtract epsilon if it was added
        if self.epsilon_fitted > 0:
            # Ensure non-negative
            back_transformed = np.maximum(back_transformed - self.epsilon_fitted, 0)

        return back_transformed.reshape(original_shape)

    def fit_transform(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Fit and transform data in one step.
     
        Parameters
        ----------
        data : np.ndarray
        Data to transform
 
        Returns
        -------
        np.ndarray
            Log-transformed data
        """
        self.fit(data)
        return self.transform(data)


def log_transform(
    data: npt.NDArray[np.float64],
    base: str = 'natural'
) -> Tuple[npt.NDArray[np.float64], LogTransform]:
    """
    Convenience function for log transform.
 
        Parameters
        ----------
        data : np.ndarray
        Data to transform
        base : str
        'natural', '10', or '2'

        Returns
        -------
        transformed_data : np.ndarray
        Log-transformed data
        transformer : LogTransform
        Fitted transformer
 
        Examples
        --------
        >>> import numpy as np
        >>> data = np.array([1.0, 10.0, 100.0, 1000.0])
        >>> log_data, transformer = log_transform(data, base='10')
        >>> logger.info(log_data) # [0, 1, 2, 3]
        >>> original = transformer.inverse_transform(log_data)
        >>> np.allclose(original, data) # True
        transformer = LogTransform(base=base)
        transformed = transformer.fit_transform(data)
        return transformed, transformer

        def log_back_transform(
        log_data: npt.NDArray[np.float64],
        transformer: LogTransform
        ) -> npt.NDArray[np.float64]:
        """
        Back-transform log data to original scale

        Parameters
        ----------
        log_data : np.ndarray
        Log-transformed data
        transformer : LogTransform
        Fitted transformer

        Returns
        -------
        np.ndarray
        Original scale data
        """
        return transformer.inverse_transform(log_data)
