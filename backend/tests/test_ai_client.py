"""Unit tests for ai_client module."""
import pytest
from unittest.mock import patch, AsyncMock
import httpx

from app.ai_client import predict_from_xml, check_health, AIEngineError


class TestPredictFromXML:
    """Test AI Engine prediction function."""
    
    @patch('app.ai_client.httpx.AsyncClient')
    async def test_predict_from_xml_success(self, mock_client_class):
        """Test successful prediction from AI Engine."""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "predictions": [
                {"label": "Normal", "score": 0.95, "description": "Normal rhythm"}
            ],
            "scores": {"normal": 0.95, "abnormal": 0.05},
            "metadata": {"model_version": "1.0.0"}
        }
        mock_response.raise_for_status = AsyncMock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Execute
        result = await predict_from_xml("<Document>Test</Document>")
        
        # Assert
        assert result is not None
        assert "predictions" in result
        assert len(result["predictions"]) == 1
        assert result["predictions"][0]["label"] == "Normal"
        assert result["predictions"][0]["score"] == 0.95
        assert result["scores"]["normal"] == 0.95
        mock_client.post.assert_called_once()
    
    @patch('app.ai_client.httpx.AsyncClient')
    async def test_predict_from_xml_timeout(self, mock_client_class):
        """Test timeout when calling AI Engine."""
        # Setup mock to raise timeout
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Execute and assert
        with pytest.raises(AIEngineError, match="AI Engine request timeout"):
            await predict_from_xml("<Document>Test</Document>")
    
    @patch('app.ai_client.httpx.AsyncClient')
    async def test_predict_from_xml_http_error(self, mock_client_class):
        """Test HTTP error from AI Engine."""
        # Setup mock to raise HTTP error
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server error", request=None, response=mock_response
        )
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Execute and assert
        with pytest.raises(AIEngineError, match="AI Engine returned error"):
            await predict_from_xml("<Document>Test</Document>")
    
    @patch('app.ai_client.httpx.AsyncClient')
    async def test_predict_from_xml_connection_error(self, mock_client_class):
        """Test connection error to AI Engine."""
        # Setup mock to raise connection error
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Execute and assert
        with pytest.raises(AIEngineError, match="Failed to connect to AI Engine"):
            await predict_from_xml("<Document>Test</Document>")
    
    @patch('app.ai_client.httpx.AsyncClient')
    async def test_predict_from_xml_invalid_response(self, mock_client_class):
        """Test invalid JSON response from AI Engine."""
        # Setup mock with invalid JSON
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = AsyncMock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Execute and assert
        with pytest.raises(AIEngineError, match="Invalid response from AI Engine"):
            await predict_from_xml("<Document>Test</Document>")


class TestCheckHealth:
    """Test AI Engine health check function."""
    
    @patch('app.ai_client.httpx.AsyncClient')
    async def test_check_health_success(self, mock_client_class):
        """Test successful health check."""
        # Setup mock
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy", "version": "1.0.0"}
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Execute
        is_healthy = await check_health()
        
        # Assert
        assert is_healthy is True
        mock_client.get.assert_called_once()
    
    @patch('app.ai_client.httpx.AsyncClient')
    async def test_check_health_failure(self, mock_client_class):
        """Test health check failure."""
        # Setup mock to raise error
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Execute
        is_healthy = await check_health()
        
        # Assert
        assert is_healthy is False
    
    @patch('app.ai_client.httpx.AsyncClient')
    async def test_check_health_unhealthy_status(self, mock_client_class):
        """Test health check with unhealthy status code."""
        # Setup mock with 503 status
        mock_response = AsyncMock()
        mock_response.status_code = 503
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Service unavailable", request=None, response=mock_response
        )
        
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Execute
        is_healthy = await check_health()
        
        # Assert
        assert is_healthy is False