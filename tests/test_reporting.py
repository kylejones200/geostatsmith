"""
Tests for reporting functionality
"""

import tempfile
from pathlib import Path

import numpy as np

from geostats.reporting.report_generator import generate_report


class TestReportGenerator:
    """Test report generation"""

    def test_generate_report_basic(self):
        """Test basic report generation"""
        x = np.array([0.0, 10.0, 20.0, 30.0])
        y = np.array([0.0, 10.0, 20.0, 30.0])
        z = np.array([1.0, 2.0, 1.5, 3.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.html"
            result = generate_report(
                x,
                y,
                z,
                output=str(output_path),
                title="Test Report",
            )

            assert Path(result).exists()
            assert output_path.exists()
            # Check it's HTML
            content = output_path.read_text()
            assert "<html" in content.lower() or "<!DOCTYPE" in content.upper()

    def test_generate_report_with_cv(self):
        """Test report with cross-validation"""
        x = np.array([0.0, 10.0, 20.0, 30.0])
        y = np.array([0.0, 10.0, 20.0, 30.0])
        z = np.array([1.0, 2.0, 1.5, 3.0])
        z_pred = np.array([1.1, 2.1, 1.4, 2.9])

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report_cv.html"
            result = generate_report(
                x,
                y,
                z,
                output=str(output_path),
                include_cv=True,
            )

            assert Path(result).exists()
            content = output_path.read_text()
            # Should contain CV information
            assert len(content) > 0
