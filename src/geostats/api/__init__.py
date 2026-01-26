"""
Web API Module
==============

FastAPI-based REST API for remote geostatistics services.

Enables:
- Remote kriging predictions
- Cloud deployment
- Web services
- Multi-user access

Examples
--------
>>> from geostats.api import create_app
>>>
>>> # Create FastAPI application
>>> app = create_app()
>>>
>>> # Run server
>>> # uvicorn geostats.api:app --reload
>>>
>>> # Access API:
>>> # POST /predict - Make predictions
>>> # POST /variogram - Fit variogram
>>> # GET /health - Health check
"""

from .app import create_app, app
from .endpoints import router

__all__ = [
 'create_app',
 'app',
 'router',
]
