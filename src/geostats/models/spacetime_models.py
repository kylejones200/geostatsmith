"""
Space-Time Variogram Models

Space-time geostatistics extends classical geostatistics to handle
data that varies in both space AND time. This is crucial for:
- Environmental monitoring (air quality over time)
- Climate data (temperature, precipitation time series)
- Epidemiology (disease spread)
- Oceanography (ocean currents, temperature)
- Hydrology (groundwater levels)

Mathematical Framework:
For a space-time random field Z(s, t) where s is location and t is time,
the space-time variogram is:
    γ(h, u) = ½ E[(Z(s, t) - Z(s+h, t+u))²]

where:
- h is spatial lag (distance)
- u is temporal lag (time difference)

Key concepts:
1. Separability: γ(h, u) = γ_s(h) · γ_t(u)
   - Simplest case: spatial and temporal structures independent
   
2. Non-separability: More complex interaction
   - Product-sum: γ(h,u) = (C_s·γ_t(u) + C_t·γ_s(h) + γ_s(h)·γ_t(u))
   - Cressie-Huang: More general non-separable models

3. Full symmetry: γ(h, u) = γ(h, -u) (time-reversible)
4. Stationarity: Same structure across space-time

References:
- Cressie, N. & Huang, H-C. (1999). "Classes of nonseparable, spatio-temporal 
  stationary covariance functions". JASA, 94:1330-1340.
- Gneiting, T. (2002). "Nonseparable, stationary covariance functions for 
  space-time data". JASA, 97:590-600.
- Kyriakidis, P.C. & Journel, A.G. (1999). "Geostatistical space-time models"
- De Iaco, S. et al. (2001). "Space-time variograms and a functional form 
  for total air pollution measurements"
"""

from typing import Optional, Tuple, Callable, Dict
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

from ..core.logging_config import get_logger
from ..core.constants import EPSILON

logger = get_logger(__name__)


class SpaceTimeVariogramModel(ABC):
    """
    Base class for space-time variogram models
    
    All space-time models must implement:
    - __call__(h, u): evaluate variogram at spatial lag h and temporal lag u
    - is_separable(): whether model is separable
    """
    
    @abstractmethod
    def __call__(
        self,
        h: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Evaluate space-time variogram"""
        pass
    
    @abstractmethod
    def is_separable(self) -> bool:
        """Check if model is separable"""
        pass


class SeparableModel(SpaceTimeVariogramModel):
    """
    Separable space-time variogram model
    
    γ(h, u) = C_s · C_t · [γ_s(h)/C_s + γ_t(u)/C_t - γ_s(h)·γ_t(u)/(C_s·C_t)]
    
    Simplified form when normalized:
    γ(h, u) = γ_s(h) + γ_t(u) - γ_s(h)·γ_t(u)
    
    where γ_s and γ_t are spatial and temporal variograms.
    
    Properties:
    - Simplest space-time model
    - Assumes spatial and temporal structures are independent
    - Computationally efficient
    
    Examples
    --------
    >>> from geostats.models.variogram_models import SphericalModel
    >>> 
    >>> # Spatial variogram
    >>> spatial_model = SphericalModel(nugget=0.1, sill=1.0, range=100)
    >>> 
    >>> # Temporal variogram
    >>> temporal_model = SphericalModel(nugget=0.05, sill=0.8, range=10)
    >>> 
    >>> # Separable space-time model
    >>> st_model = SeparableModel(spatial_model, temporal_model)
    >>> 
    >>> # Evaluate
    >>> gamma = st_model(h=50, u=5)  # h: spatial, u: temporal
    """
    
    def __init__(
        self,
        spatial_model: Callable,
        temporal_model: Callable
    ):
        """
        Initialize separable space-time model
        
        Parameters
        ----------
        spatial_model : callable
            Spatial variogram model γ_s(h)
        temporal_model : callable
            Temporal variogram model γ_t(u)
        """
        self.spatial_model = spatial_model
        self.temporal_model = temporal_model
        
        logger.info("Initialized separable space-time variogram model")
    
    def __call__(
        self,
        h: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Evaluate separable space-time variogram
        
        Parameters
        ----------
        h : np.ndarray
            Spatial lag distances
        u : np.ndarray
            Temporal lag distances
            
        Returns
        -------
        gamma : np.ndarray
            Space-time semivariance values
        """
        h = np.asarray(h, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        
        # Get spatial and temporal components
        gamma_s = self.spatial_model(h)
        gamma_t = self.temporal_model(u)
        
        # Separable model: γ(h,u) = γ_s(h) + γ_t(u) - γ_s(h)·γ_t(u)
        # This ensures γ(0,0) = 0
        gamma = gamma_s + gamma_t - gamma_s * gamma_t
        
        return gamma
    
    def is_separable(self) -> bool:
        return True


class ProductSumModel(SpaceTimeVariogramModel):
    """
    Product-Sum space-time variogram model (Cressie & Huang, 1999)
    
    γ(h, u) = C_s·γ_t(u) + C_t·γ_s(h) + k·γ_s(h)·γ_t(u)
    
    where:
    - C_s: spatial sill
    - C_t: temporal sill
    - k: interaction parameter (controls space-time interaction strength)
    - γ_s, γ_t: normalized spatial and temporal variograms
    
    Properties:
    - Non-separable (unless k=0)
    - Allows space-time interaction
    - More flexible than separable model
    - k > 0: positive interaction (common)
    - k < 0: negative interaction (rare)
    
    Examples
    --------
    >>> spatial_model = SphericalModel(nugget=0, sill=1.0, range=100)
    >>> temporal_model = SphericalModel(nugget=0, sill=1.0, range=10)
    >>> 
    >>> # Non-separable with interaction
    >>> st_model = ProductSumModel(
    ...     spatial_model, temporal_model,
    ...     C_s=1.0, C_t=0.8, k=0.5
    ... )
    """
    
    def __init__(
        self,
        spatial_model: Callable,
        temporal_model: Callable,
        C_s: float = 1.0,
        C_t: float = 1.0,
        k: float = 0.5
    ):
        """
        Initialize product-sum space-time model
        
        Parameters
        ----------
        spatial_model : callable
            Spatial variogram model
        temporal_model : callable
            Temporal variogram model
        C_s : float
            Spatial sill parameter
        C_t : float
            Temporal sill parameter
        k : float
            Interaction parameter
        """
        self.spatial_model = spatial_model
        self.temporal_model = temporal_model
        self.C_s = C_s
        self.C_t = C_t
        self.k = k
        
        logger.info(
            f"Initialized product-sum space-time model "
            f"(C_s={C_s:.3f}, C_t={C_t:.3f}, k={k:.3f})"
        )
    
    def __call__(
        self,
        h: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Evaluate product-sum space-time variogram
        
        Parameters
        ----------
        h : np.ndarray
            Spatial lag distances
        u : np.ndarray
            Temporal lag distances
            
        Returns
        -------
        gamma : np.ndarray
            Space-time semivariance values
        """
        h = np.asarray(h, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        
        gamma_s = self.spatial_model(h)
        gamma_t = self.temporal_model(u)
        
        # Product-sum: γ(h,u) = C_s·γ_t + C_t·γ_s + k·γ_s·γ_t
        gamma = self.C_s * gamma_t + self.C_t * gamma_s + self.k * gamma_s * gamma_t
        
        return gamma
    
    def is_separable(self) -> bool:
        return np.abs(self.k) < EPSILON


class GneitingModel(SpaceTimeVariogramModel):
    """
    Gneiting space-time covariance model (Gneiting, 2002)
    
    Fully symmetric, non-separable space-time model.
    
    Covariance form:
    C(h, u) = σ² / (a(u)^d) · ϕ(h² / a(u)²) · ψ(u)
    
    where:
    - a(u) = (a₀ + u^α)^(1/α) : temporal scaling function
    - ϕ: spatial correlation function (e.g., exponential)
    - ψ: temporal correlation function
    - σ²: variance
    - d: spatial dimension
    
    Variogram: γ(h, u) = σ² - C(h, u)
    
    Properties:
    - Fully symmetric (time-reversible)
    - Non-separable
    - Flexible space-time interaction
    - Commonly used in atmospheric sciences
    
    References
    ----------
    Gneiting, T. (2002). "Nonseparable, stationary covariance functions 
    for space-time data". JASA, 97:590-600.
    """
    
    def __init__(
        self,
        sigma2: float = 1.0,
        a0: float = 1.0,
        alpha: float = 1.0,
        spatial_range: float = 100.0,
        temporal_range: float = 10.0,
        spatial_smoothness: float = 1.0
    ):
        """
        Initialize Gneiting space-time model
        
        Parameters
        ----------
        sigma2 : float
            Variance parameter
        a0 : float
            Base temporal scaling parameter
        alpha : float
            Temporal scaling exponent (0 < alpha ≤ 2)
        spatial_range : float
            Spatial correlation range
        temporal_range : float
            Temporal correlation range
        spatial_smoothness : float
            Spatial smoothness parameter (κ in Matérn)
        """
        self.sigma2 = sigma2
        self.a0 = a0
        self.alpha = alpha
        self.spatial_range = spatial_range
        self.temporal_range = temporal_range
        self.spatial_smoothness = spatial_smoothness
        
        if not 0 < alpha <= 2:
            logger.warning(f"alpha={alpha} outside typical range (0, 2]")
        
        logger.info(
            f"Initialized Gneiting space-time model "
            f"(spatial_range={spatial_range:.1f}, temporal_range={temporal_range:.1f})"
        )
    
    def _temporal_scaling(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute temporal scaling function a(u)"""
        return (self.a0 + np.abs(u) ** self.alpha) ** (1.0 / self.alpha)
    
    def _spatial_correlation(
        self,
        h: npt.NDArray[np.float64],
        a_u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute spatial correlation ϕ(h²/a(u)²)"""
        # Using exponential correlation as default
        # Can be extended to other forms (Gaussian, Matérn, etc.)
        scaled_h = h / (a_u * self.spatial_range)
        return np.exp(-scaled_h)
    
    def _temporal_correlation(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute temporal correlation ψ(u)"""
        scaled_u = np.abs(u) / self.temporal_range
        return np.exp(-scaled_u)
    
    def __call__(
        self,
        h: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Evaluate Gneiting space-time variogram
        
        Parameters
        ----------
        h : np.ndarray
            Spatial lag distances
        u : np.ndarray
            Temporal lag distances
            
        Returns
        -------
        gamma : np.ndarray
            Space-time semivariance values
        """
        h = np.asarray(h, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        
        # Temporal scaling
        a_u = self._temporal_scaling(u)
        
        # Spatial correlation (dimension d=2 for 2D space)
        d = 2.0
        spatial_term = self._spatial_correlation(h, a_u) / (a_u ** d)
        
        # Temporal correlation
        temporal_term = self._temporal_correlation(u)
        
        # Covariance
        cov = self.sigma2 * spatial_term * temporal_term
        
        # Variogram: γ = σ² - C
        gamma = self.sigma2 - cov
        
        return gamma
    
    def is_separable(self) -> bool:
        return False


def create_spacetime_model(
    model_type: str,
    spatial_model: Callable,
    temporal_model: Optional[Callable] = None,
    **kwargs
) -> SpaceTimeVariogramModel:
    """
    Factory function to create space-time variogram models
    
    Parameters
    ----------
    model_type : str
        Type of space-time model:
        - 'separable': Simple separable model
        - 'product_sum': Product-sum model (Cressie-Huang)
        - 'gneiting': Gneiting's non-separable model
    spatial_model : callable
        Spatial variogram model
    temporal_model : callable, optional
        Temporal variogram model (required for separable and product_sum)
    **kwargs
        Additional parameters for specific models
        
    Returns
    -------
    model : SpaceTimeVariogramModel
        Initialized space-time variogram model
        
    Examples
    --------
    >>> from geostats.models.variogram_models import SphericalModel
    >>> 
    >>> # Create models
    >>> spatial = SphericalModel(nugget=0.1, sill=1.0, range=100)
    >>> temporal = SphericalModel(nugget=0.05, sill=0.8, range=10)
    >>> 
    >>> # Separable model
    >>> model1 = create_spacetime_model('separable', spatial, temporal)
    >>> 
    >>> # Product-sum model
    >>> model2 = create_spacetime_model(
    ...     'product_sum', spatial, temporal,
    ...     C_s=1.0, C_t=0.8, k=0.5
    ... )
    >>> 
    >>> # Gneiting model
    >>> model3 = create_spacetime_model(
    ...     'gneiting', spatial,
    ...     sigma2=1.0, spatial_range=100, temporal_range=10
    ... )
    """
    models = {
        'separable': SeparableModel,
        'product_sum': ProductSumModel,
        'gneiting': GneitingModel,
    }
    
    if model_type not in models:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available: {list(models.keys())}"
        )
    
    if model_type in ['separable', 'product_sum']:
        if temporal_model is None:
            raise ValueError(f"{model_type} model requires temporal_model")
        return models[model_type](spatial_model, temporal_model, **kwargs)
    else:
        return models[model_type](**kwargs)
