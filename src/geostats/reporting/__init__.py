"""
Professional Reporting Module
==============================

Generate professional PDF and HTML reports with analysis results.

Features:
- Automatic report generation
- Publication-quality figures
- Statistical summaries
- Professional formatting

Examples
--------
>>> from geostats.reporting import generate_report
>>>
>>> # Generate report
>>> generate_report(
... x, y, z,
... output='analysis_report.pdf',
... title='Soil Contamination Analysis',
... author='Your Name'
... )
"""

from .report_generator import (
    create_kriging_report,
    create_validation_report,
    generate_report,
)
from .templates import (
    KrigingTemplate,
    ReportTemplate,
    ValidationTemplate,
)

__all__ = [
    "generate_report",
    "create_kriging_report",
    "create_validation_report",
    "ReportTemplate",
    "KrigingTemplate",
    "ValidationTemplate",
]
