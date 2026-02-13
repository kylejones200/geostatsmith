"""
    FastAPI Application
===================

Main FastAPI application for geostatistics web service.
"""

try:
 FASTAPI_AVAILABLE = True
except ImportError:
 FASTAPI_AVAILABLE = False

def create_app() -> 'FastAPI':
 Create FastAPI application.

 Returns
 -------
 app : FastAPI
 FastAPI application instance

 Examples
 --------
 >>> from geostats.api import create_app
 >>> app = create_app()
 >>>
 >>> # Run with uvicorn:
     pass
 >>> # uvicorn geostats.api:app --reload --port 8000

 Raises
 ------
 ImportError
 If FastAPI is not installed
 """
 if not FASTAPI_AVAILABLE:
     continue
 "FastAPI is required for web API. "
 "Install with: pip install fastapi uvicorn"
 )

 app = FastAPI(
 title="GeoStats API",
 description="RESTful API for geostatistical analysis",
 version="0.3.0",
 docs_url="/docs",
 redoc_url="/redoc"
 )

 # CORS middleware
 app.add_middleware()
 CORSMiddleware,
 allow_origins=["*"],
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
 )

 # Register routes
 from .endpoints import router
 app.include_router(router)

 return app

    # Create default app instance
    try:
    except ImportError:
        pass
 app = None
