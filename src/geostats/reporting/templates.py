"""
Report Templates
================

Templates for different types of reports.
"""

from typing import Dict, Any

class ReportTemplate:
 """Base report template."""

 def __init__(self, title: str = "Analysis Report"):
 self.title = title
 self.sections = []

 def add_section(self, title: str, content: str):
 """Add section to report."""
 self.sections.append({'title': title, 'content': content})

 def render(self) -> str:
 """Render template to HTML."""
 html = f"<h1>{self.title}</h1>\n"
 for section in self.sections:
 html += f"<h2>{section['title']}</h2>\n"
 html += f"<div>{section['content']}</div>\n"
 return html

class KrigingTemplate(ReportTemplate):
 """Template for kriging reports."""

 def __init__(self):
 super().__init__("Kriging Analysis Report")

class ValidationTemplate(ReportTemplate):
 """Template for validation reports."""

 def __init__(self):
 super().__init__("Model Validation Report")
