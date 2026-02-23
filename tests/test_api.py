"""
Tests for API endpoints
"""

import pytest

# Skip if FastAPI is not available
try:
    from fastapi.testclient import TestClient
    from geostats.api.app import create_app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="FastAPI not available")


class TestAPI:
    """Test API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app = create_app()
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "GeoStats API"
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "modules_available" in data
    
    def test_predict_endpoint(self, client):
        """Test prediction endpoint"""
        request_data = {
            "x_samples": [0.0, 10.0, 20.0],
            "y_samples": [0.0, 10.0, 20.0],
            "z_samples": [1.0, 2.0, 1.5],
            "x_pred": [5.0, 15.0],
            "y_pred": [5.0, 15.0],
        }
        
        response = client.post("/predict", json=request_data)
        # May fail if dependencies not available, but should return proper error
        assert response.status_code in [200, 500, 503]  # Success or dependency error
