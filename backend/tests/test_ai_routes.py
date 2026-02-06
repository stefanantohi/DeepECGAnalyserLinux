"""
Tests for AI Engine routes.

These tests mock the AI Engine responses to test the backend integration.
"""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import io

from app.main import app
from app.ai_engine_client import HealthCheckResult, AnalysisResult


client = TestClient(app)


class TestAIHealthEndpoint:
    """Tests for GET /api/ai/health endpoint."""

    @patch('app.ai_routes.check_ai_engine_health')
    def test_health_check_healthy(self, mock_health):
        """Test health check when AI Engine is healthy."""
        mock_health.return_value = HealthCheckResult(
            status="healthy",
            latency_ms=45.2,
            engine_url="http://localhost:8001",
            details={"version": "1.0.0", "gpu": "available"}
        )

        response = client.get("/api/ai/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["latency_ms"] == 45.2
        assert data["engine_url"] == "http://localhost:8001"
        assert data["details"]["version"] == "1.0.0"
        assert "request_id" in data

    @patch('app.ai_routes.check_ai_engine_health')
    def test_health_check_unreachable(self, mock_health):
        """Test health check when AI Engine is unreachable."""
        mock_health.return_value = HealthCheckResult(
            status="unreachable",
            latency_ms=5000.0,
            engine_url="http://localhost:8001",
            error="Connection refused"
        )

        response = client.get("/api/ai/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unreachable"
        assert data["error"] == "Connection refused"

    @patch('app.ai_routes.check_ai_engine_health')
    def test_health_check_unhealthy(self, mock_health):
        """Test health check when AI Engine returns error status."""
        mock_health.return_value = HealthCheckResult(
            status="unhealthy",
            latency_ms=120.5,
            engine_url="http://localhost:8001",
            error="HTTP 503: Service Unavailable"
        )

        response = client.get("/api/ai/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "503" in data["error"]


class TestAIAnalyzeEndpoint:
    """Tests for POST /api/ai/analyze endpoint."""

    @patch('app.ai_routes.analyze_file')
    def test_analyze_csv_success(self, mock_analyze):
        """Test successful CSV file analysis."""
        mock_analyze.return_value = AnalysisResult(
            success=True,
            job_id="job-12345",
            outputs={
                "predictions": [
                    {"label": "Normal", "score": 0.95},
                    {"label": "Arrhythmia", "score": 0.05}
                ],
                "processing_time": 1.5
            },
            processing_time_ms=1500.0
        )

        # Create a mock CSV file
        csv_content = b"time,lead_I,lead_II\n0.0,0.1,0.2\n0.01,0.15,0.25"
        files = {"file": ("test_ecg.csv", io.BytesIO(csv_content), "text/csv")}

        response = client.post("/api/ai/analyze", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["job_id"] == "job-12345"
        assert data["filename"] == "test_ecg.csv"
        assert "predictions" in data["outputs"]

    @patch('app.ai_routes.analyze_file')
    def test_analyze_parquet_success(self, mock_analyze):
        """Test successful Parquet file analysis."""
        mock_analyze.return_value = AnalysisResult(
            success=True,
            job_id="job-67890",
            outputs={"result": "analysis complete"},
            processing_time_ms=2000.0
        )

        # Create a mock parquet file (just bytes for testing)
        parquet_content = b"PAR1..."  # Mock parquet header
        files = {"file": ("test_ecg.parquet", io.BytesIO(parquet_content), "application/octet-stream")}

        response = client.post("/api/ai/analyze", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "test_ecg.parquet"

    def test_analyze_invalid_extension(self):
        """Test rejection of invalid file extensions."""
        # Try to upload a PDF file
        pdf_content = b"%PDF-1.4..."
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}

        response = client.post("/api/ai/analyze", files=files)

        assert response.status_code == 415
        data = response.json()
        assert "error" in data
        assert "Unsupported file type" in str(data)

    def test_analyze_invalid_extension_txt(self):
        """Test rejection of .txt files."""
        txt_content = b"Hello World"
        files = {"file": ("test.txt", io.BytesIO(txt_content), "text/plain")}

        response = client.post("/api/ai/analyze", files=files)

        assert response.status_code == 415

    @patch('app.ai_routes.analyze_file')
    def test_analyze_ai_engine_error(self, mock_analyze):
        """Test handling of AI Engine errors."""
        mock_analyze.return_value = AnalysisResult(
            success=False,
            error="AI Engine unreachable after 3 attempts",
            processing_time_ms=30000.0
        )

        csv_content = b"time,value\n0,1"
        files = {"file": ("test.csv", io.BytesIO(csv_content), "text/csv")}

        response = client.post("/api/ai/analyze", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "unreachable" in data["error"]


class TestAIInfoEndpoint:
    """Tests for GET /api/ai/info endpoint."""

    def test_get_engine_info(self):
        """Test getting AI Engine configuration info."""
        response = client.get("/api/ai/info")

        assert response.status_code == 200
        data = response.json()
        assert "url" in data
        assert "timeout" in data
        assert "allowed_extensions" in data
        assert ".csv" in data["allowed_extensions"]
        assert ".parquet" in data["allowed_extensions"]


class TestRequestIdMiddleware:
    """Tests for request ID middleware."""

    @patch('app.ai_routes.check_ai_engine_health')
    def test_request_id_in_response(self, mock_health):
        """Test that request ID is included in response headers."""
        mock_health.return_value = HealthCheckResult(
            status="healthy",
            latency_ms=50.0,
            engine_url="http://localhost:8001"
        )

        response = client.get("/api/ai/health")

        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 36  # UUID format

    @patch('app.ai_routes.check_ai_engine_health')
    def test_request_id_in_body(self, mock_health):
        """Test that request ID is included in response body."""
        mock_health.return_value = HealthCheckResult(
            status="healthy",
            latency_ms=50.0,
            engine_url="http://localhost:8001"
        )

        response = client.get("/api/ai/health")
        data = response.json()

        assert "request_id" in data
        assert data["request_id"] == response.headers["X-Request-ID"]
