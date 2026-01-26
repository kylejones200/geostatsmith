"""
Example Workflow: Performance Optimization
===========================================

Demonstrates speed optimizations for large datasets.

Shows:
1. Parallel kriging (multi-core)
2. Chunked processing
3. Result caching
4. Approximate methods

Author: geostats development team
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import time

try:
    from geostats.performance import (
        parallel_kriging,
        ChunkedKriging,
        CachedKriging,
        approximate_kriging,
    )
    from geostats.algorithms.ordinary_kriging import OrdinaryKriging
    from geostats.models.variogram_models import SphericalModel
    from geostats.algorithms.variogram import experimental_variogram
    from geostats.algorithms.fitting import fit_variogram
except ImportError:
    print("Please install geostats: pip install -e .")
    exit(1)


def example_1_parallel_kriging():
    """Example 1: Parallel kriging for speed."""
    print("\n" + "="*60)
    print("Example 1: Parallel Kriging (Multi-Core)")
    print("="*60)
    
    # Create dataset
    np.random.seed(42)
    n_samples = 200
    x = np.random.uniform(0, 100, n_samples)
    y = np.random.uniform(0, 100, n_samples)
    z = 50 + 0.3*x + 0.2*y + 10*np.sin(x/20) + np.random.normal(0, 3, n_samples)
    
    # Fit variogram
    lags, gamma = experimental_variogram(x, y, z)
    model = fit_variogram(lags, gamma, model_type='spherical')
    
    # Create large prediction grid
    nx, ny = 200, 200
    x_grid = np.linspace(0, 100, nx)
    y_grid = np.linspace(0, 100, ny)
    x_2d, y_2d = np.meshgrid(x_grid, y_grid)
    x_pred = x_2d.ravel()
    y_pred = y_2d.ravel()
    
    print(f"\nDataset: {n_samples} samples, {len(x_pred):,} predictions")
    
    # Sequential kriging
    print("\n1. Sequential kriging...")
    start = time.time()
    krig = OrdinaryKriging(x, y, z, model)
    z_pred_seq, _ = krig.predict(x_pred, y_pred, return_variance=True)
    time_seq = time.time() - start
    print(f"   Time: {time_seq:.2f}s")
    
    # Parallel kriging
    print("\n2. Parallel kriging (all cores)...")
    start = time.time()
    z_pred_par, _ = parallel_kriging(
        x, y, z, x_pred, y_pred,
        variogram_model=model,
        n_jobs=-1,
        batch_size=1000
    )
    time_par = time.time() - start
    print(f"   Time: {time_par:.2f}s")
    print(f"   Speedup: {time_seq/time_par:.1f}x")
    
    # Check agreement
    diff = np.abs(z_pred_seq - z_pred_par).max()
    print(f"   Max difference: {diff:.6f} (should be ~0)")


def example_2_chunked_processing():
    """Example 2: Chunked processing for memory efficiency."""
    print("\n" + "="*60)
    print("Example 2: Chunked Processing")
    print("="*60)
    
    # Large dataset
    np.random.seed(42)
    n = 100
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    z = 50 + 0.3*x + np.random.normal(0, 3, n)
    
    lags, gamma = experimental_variogram(x, y, z)
    model = fit_variogram(lags, gamma, model_type='spherical')
    
    # Very large grid (would use lots of memory)
    print("\nProcessing 100,000 prediction points in chunks...")
    nx, ny = 316, 316  # ~100k points
    x_grid = np.linspace(0, 100, nx)
    y_grid = np.linspace(0, 100, ny)
    
    chunked = ChunkedKriging(x, y, z, model)
    z_grid, _ = chunked.predict_large_grid(
        x_grid, y_grid,
        chunk_size=5000,
        return_variance=False,
        verbose=True
    )
    
    print(f"\n✓ Completed! Grid shape: {z_grid.shape}")


def example_3_caching():
    """Example 3: Result caching for repeated predictions."""
    print("\n" + "="*60)
    print("Example 3: Result Caching")
    print("="*60)
    
    # Dataset
    np.random.seed(42)
    x = np.random.uniform(0, 100, 50)
    y = np.random.uniform(0, 100, 50)
    z = 50 + 0.3*x + np.random.normal(0, 3, 50)
    
    lags, gamma = experimental_variogram(x, y, z)
    model = fit_variogram(lags, gamma, model_type='spherical')
    
    # Prediction locations
    x_pred = np.linspace(0, 100, 100)
    y_pred = np.linspace(0, 100, 100)
    
    # First call - computes and caches
    print("\n1. First call (computes and caches)...")
    cached_krig = CachedKriging(x, y, z, model)
    start = time.time()
    z_pred1, _ = cached_krig.predict(x_pred, y_pred)
    time1 = time.time() - start
    print(f"   Time: {time1:.3f}s")
    
    # Second call - uses cache
    print("\n2. Second call (uses cache)...")
    start = time.time()
    z_pred2, _ = cached_krig.predict(x_pred, y_pred)
    time2 = time.time() - start
    print(f"   Time: {time2:.3f}s")
    print(f"   Speedup: {time1/time2:.0f}x (instant!)")
    
    # Clear cache
    from geostats.performance import clear_cache
    n_cleared = clear_cache()
    print(f"\n✓ Cleared {n_cleared} cache files")


def example_4_approximate_methods():
    """Example 4: Fast approximate kriging."""
    print("\n" + "="*60)
    print("Example 4: Approximate Kriging")
    print("="*60)
    
    # Large dataset
    np.random.seed(42)
    n = 500
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)
    z = 50 + 0.3*x + 0.2*y + np.random.normal(0, 3, n)
    
    lags, gamma = experimental_variogram(x, y, z)
    model = fit_variogram(lags, gamma, model_type='spherical')
    
    # Prediction points
    n_pred = 1000
    x_pred = np.random.uniform(0, 100, n_pred)
    y_pred = np.random.uniform(0, 100, n_pred)
    
    print(f"\nDataset: {n} samples, {n_pred} predictions")
    
    # Exact kriging
    print("\n1. Exact kriging...")
    start = time.time()
    krig = OrdinaryKriging(x, y, z, model)
    z_exact, _ = krig.predict(x_pred, y_pred, return_variance=True)
    time_exact = time.time() - start
    print(f"   Time: {time_exact:.2f}s")
    
    # Approximate kriging (30 nearest neighbors)
    print("\n2. Approximate kriging (30 neighbors)...")
    start = time.time()
    z_approx, _ = approximate_kriging(
        x, y, z, x_pred, y_pred,
        variogram_model=model,
        max_neighbors=30
    )
    time_approx = time.time() - start
    print(f"   Time: {time_approx:.2f}s")
    print(f"   Speedup: {time_exact/time_approx:.1f}x")
    
    # Accuracy
    rmse = np.sqrt(np.mean((z_exact - z_approx)**2))
    rel_rmse = rmse / np.std(z)
    print(f"   RMSE: {rmse:.3f} ({rel_rmse*100:.1f}% of std)")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("GEOSTATS PERFORMANCE OPTIMIZATION EXAMPLES")
    print("="*70)
    
    example_1_parallel_kriging()
    example_2_chunked_processing()
    example_3_caching()
    example_4_approximate_methods()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • Parallel kriging: 2-8x speedup on multi-core systems")
    print("  • Chunked processing: Handle millions of points")
    print("  • Caching: Instant repeated predictions")
    print("  • Approximate methods: 10-100x speedup with <5% error")
    print("\n")


if __name__ == '__main__':
    main()
