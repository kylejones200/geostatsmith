"""
Example Workflow: Enterprise Deployment & Advanced Features
============================================================

Demonstrates Phase 3 enterprise features:
1. Web API deployment
2. CLI tools
3. Professional reporting
4. Advanced diagnostics

Author: geostats development team
Date: January 2026
"""

import numpy as np


def example_1_cli_tools():
    """Example 1: Command-line tools."""
    print("\n" + "="*60)
    print("Example 1: CLI Tools")
    print("="*60)
    
    print("\nAvailable CLI commands:")
    print("  geostats predict input.csv output.csv")
    print("  geostats variogram data.csv --plot --auto")
    print("  geostats validate data.csv --method leave-one-out")
    print("  geostats serve --port 8000")
    
    print("\nExample usage:")
    print("  $ geostats predict samples.csv predictions.csv --method kriging")
    print("  $ geostats serve  # Start API server at http://localhost:8000")


def example_2_professional_reporting():
    """Example 2: Generate professional reports."""
    print("\n" + "="*60)
    print("Example 2: Professional Reporting")
    print("="*60)
    
    try:
        from geostats.reporting import generate_report
        
        # Create sample data
        np.random.seed(42)
        x = np.random.uniform(0, 100, 50)
        y = np.random.uniform(0, 100, 50)
        z = 50 + 0.3*x + np.random.normal(0, 3, 50)
        
        print("\nGenerating professional HTML report...")
        output = generate_report(
            x, y, z,
            output='professional_report.html',
            title='Soil Contamination Analysis',
            author='Environmental Team',
            include_cv=True,
            include_uncertainty=True
        )
        
        print(f"✓ Report generated: {output}")
        print("  Open in browser to view!")
    
    except Exception as e:
        print(f"⚠ Error: {e}")


def example_3_advanced_diagnostics():
    """Example 3: Comprehensive validation."""
    print("\n" + "="*60)
    print("Example 3: Advanced Diagnostics")
    print("="*60)
    
    try:
        from geostats.diagnostics import comprehensive_validation, outlier_analysis
        from geostats.automl import auto_variogram
        
        # Sample data
        np.random.seed(42)
        x = np.random.uniform(0, 100, 60)
        y = np.random.uniform(0, 100, 60)
        z = 50 + 0.3*x + np.random.normal(0, 3, 60)
        
        # Add some outliers
        z[0] = 100  # Obvious outlier
        z[1] = 0
        
        print("\n1. Outlier Detection:")
        outliers = outlier_analysis(x, y, z, method='zscore', threshold=3.0)
        print(f"   Found {outliers['n_outliers']} potential outliers")
        print(f"   Outlier indices: {outliers['outlier_indices']}")
        
        print("\n2. Comprehensive Validation:")
        model = auto_variogram(x, y, z, verbose=False)
        results = comprehensive_validation(x, y, z, model)
        print(results['diagnostics'])
    
    except Exception as e:
        print(f"⚠ Error: {e}")


def example_4_web_api():
    """Example 4: Web API usage."""
    print("\n" + "="*60)
    print("Example 4: Web API Deployment")
    print("="*60)
    
    print("\nTo start the API server:")
    print("  $ geostats serve --port 8000")
    print("  or")
    print("  $ uvicorn geostats.api:app --reload")
    
    print("\nAPI Endpoints:")
    print("  POST /predict - Make kriging predictions")
    print("  POST /variogram - Fit variogram model")
    print("  POST /auto-interpolate - Automatic everything")
    print("  GET /health - Health check")
    print("  GET /docs - Interactive API documentation")
    
    print("\nExample API call (Python):")
    print("""
    import requests
    
    # Prepare data
    data = {
        "x_samples": [0, 50, 100],
        "y_samples": [0, 50, 100],
        "z_samples": [10, 15, 20],
        "x_pred": [25, 75],
        "y_pred": [25, 75],
        "variogram_type": "spherical"
    }
    
    # Make prediction
    response = requests.post(
        "http://localhost:8000/predict",
        json=data
    )
    
    predictions = response.json()["predictions"]
    """)
    
    print("\nExample API call (curl):")
    print("""
    curl -X POST "http://localhost:8000/predict" \\
      -H "Content-Type: application/json" \\
      -d '{
        "x_samples": [0, 50, 100],
        "y_samples": [0, 50, 100],
        "z_samples": [10, 15, 20],
        "x_pred": [25, 75],
        "y_pred": [25, 75]
      }'
    """)


def example_5_complete_workflow():
    """Example 5: Complete enterprise workflow."""
    print("\n" + "="*60)
    print("Example 5: Complete Enterprise Workflow")
    print("="*60)
    
    try:
        from geostats.automl import auto_interpolate
        from geostats.diagnostics import comprehensive_validation
        from geostats.reporting import generate_report
        
        # 1. Load and analyze data
        print("\n1. Loading and analyzing data...")
        np.random.seed(42)
        x = np.random.uniform(0, 100, 80)
        y = np.random.uniform(0, 100, 80)
        z = 50 + 0.3*x + 0.2*y + np.random.normal(0, 3, 80)
        print(f"   Loaded {len(x)} samples")
        
        # 2. Automatic interpolation
        print("\n2. Automatic interpolation...")
        x_pred = np.linspace(0, 100, 100)
        y_pred = np.linspace(50, 50, 100)
        results = auto_interpolate(x, y, z, x_pred, y_pred, verbose=False)
        print(f"   Best method: {results['best_method']}")
        print(f"   CV RMSE: {results['cv_rmse']:.3f}")
        
        # 3. Diagnostics
        print("\n3. Running diagnostics...")
        diagnostics = comprehensive_validation(x, y, z, results['model'])
        print(f"   Overall score: {diagnostics['overall_score']}/100")
        
        # 4. Generate report
        print("\n4. Generating professional report...")
        report_path = generate_report(
            x, y, z,
            output='complete_analysis.html',
            title='Complete Enterprise Analysis'
        )
        
        print(f"\n✓ Complete workflow finished!")
        print(f"✓ Report: {report_path}")
    
    except Exception as e:
        print(f"⚠ Error: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("GEOSTATS PHASE 3: ENTERPRISE DEPLOYMENT")
    print("="*70)
    
    example_1_cli_tools()
    example_2_professional_reporting()
    example_3_advanced_diagnostics()
    example_4_web_api()
    example_5_complete_workflow()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE!")
    print("="*70)
    print("\nPhase 3 Capabilities:")
    print("  ✅ CLI tools for command-line usage")
    print("  ✅ Web API for remote/cloud deployment")
    print("  ✅ Professional HTML/PDF reports")
    print("  ✅ Advanced diagnostics & validation")
    print("  ✅ Outlier detection")
    print("  ✅ Enterprise-ready deployment")
    print("\n")


if __name__ == '__main__':
    main()
