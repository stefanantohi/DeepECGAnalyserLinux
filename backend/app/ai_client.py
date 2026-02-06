"""AI Engine client for making predictions."""
import httpx
import logging
from typing import Dict, Any, Optional
import subprocess
import json
import os

from .settings import settings
from .circuit_breaker import CircuitBreaker, CircuitState
from .exceptions import AIEngineError

logger = logging.getLogger(__name__)

# Circuit breaker instance with configurable thresholds
_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=60,
    half_open_max_calls=3,
    name="ai_engine"
)


async def predict_from_xml(xml_str: str, timeout: Optional[int] = None) -> Dict[str, Any]:
    """
    Send XML to AI Engine and get predictions.
    
    This function includes circuit breaker pattern for resilience.
    If the AI Engine is experiencing issues, requests will be
    temporarily blocked to prevent cascading failures.
    
    Args:
        xml_str: XML string to send to AI Engine
        timeout: Optional timeout override (seconds). Uses settings.AI_ENGINE_TIMEOUT if None.
    
    Returns:
        Dictionary with predictions, scores, and metadata
    
    Raises:
        AIEngineError: If prediction fails or circuit is open
        httpx.TimeoutException: If request times out
        httpx.HTTPStatusError: If HTTP error occurs
    """
    # Check if request should be allowed through circuit breaker
    if not _circuit_breaker.can_request():
        state = _circuit_breaker.get_state()
        if state == CircuitState.OPEN:
            raise AIEngineError(
                "AI Engine en mode dégradé - trop d'échecs récents. "
                "Veuillez réessayer dans quelques minutes.",
                degraded_mode=True
            )
        else:
            raise AIEngineError(
                "AI Engine temporairement indisponible. "
                "Tentative de récupération en cours...",
                degraded_mode=True
            )
    
    timeout_seconds = timeout or settings.AI_ENGINE_TIMEOUT
    
    if settings.AI_ENGINE_MODE == "cli":
        logger.info("Using CLI mode for AI Engine")
        return await _predict_via_cli(xml_str, timeout_seconds)
    else:
        logger.info(f"Using REST mode for AI Engine: {settings.AI_ENGINE_URL}")
        return await _predict_via_rest(xml_str, timeout_seconds)


async def _predict_via_rest(xml_str: str, timeout: int) -> Dict[str, Any]:
    """
    Call AI Engine via REST API.
    
    Args:
        xml_str: XML string to send
        timeout: Timeout in seconds
    
    Returns:
        Prediction dictionary
    
    Raises:
        AIEngineError: If prediction fails
        httpx.TimeoutException: If request times out
        httpx.HTTPStatusError: If HTTP error occurs
    """
    url = f"{settings.AI_ENGINE_URL}/predict"
    logger.info(f"Sending prediction request to: {url}")
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                url,
                content=xml_str,
                headers={
                    "Content-Type": "application/xml",
                    "Accept": "application/json"
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Record success in circuit breaker
            _circuit_breaker.record_success()
            
            logger.info(f"Successfully received prediction from AI Engine (circuit: {_circuit_breaker.get_state().value})")
            return result
            
        except httpx.TimeoutException as e:
            logger.error(f"AI Engine request timed out after {timeout}s: {e}")
            _circuit_breaker.record_failure()
            raise AIEngineError(f"AI Engine timeout: {str(e)}")
            
        except httpx.HTTPStatusError as e:
            logger.error(f"AI Engine returned HTTP error {e.response.status_code}: {e.response.text}")
            _circuit_breaker.record_failure()
            raise AIEngineError(f"AI Engine HTTP error: {e.response.status_code}")
            
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to AI Engine: {e}")
            _circuit_breaker.record_failure()
            raise AIEngineError(f"Failed to connect to AI Engine: {str(e)}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI Engine response as JSON: {e}")
            _circuit_breaker.record_failure()
            raise AIEngineError(f"Invalid JSON response from AI Engine")


async def _predict_via_cli(xml_str: str, timeout: int) -> Dict[str, Any]:
    """
    Call AI Engine via CLI (docker exec).
    
    Args:
        xml_str: XML string to send
        timeout: Timeout in seconds
    
    Returns:
        Prediction dictionary
    
    Raises:
        AIEngineError: If prediction fails
    """
    # Create temporary XML file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as f:
        f.write(xml_str)
        xml_path = f.name
    
    try:
        # Execute prediction script inside container
        container_name = settings.AI_ENGINE_URL.split("://")[1].split(":")[0]
        command = [
            "docker", "exec",
            container_name,
            "python", "/app/predict.py",
            "--xml", "/data/input.xml"
        ]
        
        # Copy XML file to container
        copy_command = [
            "docker", "cp",
            xml_path,
            f"{container_name}:/data/input.xml"
        ]
        
        try:
            # Copy file to container
            logger.info(f"Copying XML to container {container_name}")
            subprocess.run(copy_command, check=True, timeout=10)
            
            # Run prediction
            logger.info(f"Running prediction in container")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
            
            # Parse JSON output
            prediction = json.loads(result.stdout)
            
            # Record success in circuit breaker
            _circuit_breaker.record_success()
            
            logger.info("Successfully received prediction via CLI")
            return prediction
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"CLI prediction timed out after {timeout}s")
            _circuit_breaker.record_failure()
            raise AIEngineError(f"CLI prediction timeout: {str(e)}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"CLI prediction failed with return code {e.returncode}")
            logger.error(f"stderr: {e.stderr}")
            _circuit_breaker.record_failure()
            raise AIEngineError(f"CLI prediction failed: {e.stderr}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse CLI output as JSON: {e}")
            _circuit_breaker.record_failure()
            raise AIEngineError(f"Invalid JSON response from CLI")
            
    finally:
        # Clean up temporary file
        try:
            os.unlink(xml_path)
        except Exception:
            pass


async def check_health() -> bool:
    """
    Check if AI Engine is accessible.
    
    Also returns circuit breaker state for monitoring.
    
    Returns:
        True if AI Engine is accessible, False otherwise
    """
    if settings.AI_ENGINE_MODE == "cli":
        try:
            container_name = settings.AI_ENGINE_URL.split("://")[1].split(":")[0]
            result = subprocess.run(
                ["docker", "inspect", "--format={{.State.Running}}", container_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            is_running = result.returncode == 0 and "true" in result.stdout.lower()
            logger.debug(f"AI Engine health check via CLI: {is_running} (circuit: {_circuit_breaker.get_state().value})")
            return is_running
        except Exception as e:
            logger.warning(f"Failed to check AI Engine health via CLI: {e}")
            return False
    else:
        try:
            url = f"{settings.AI_ENGINE_URL}/health"
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(url)
                is_healthy = response.status_code == 200
                logger.debug(f"AI Engine health check via REST: {is_healthy} (circuit: {_circuit_breaker.get_state().value})")
                return is_healthy
        except Exception as e:
            logger.warning(f"Failed to check AI Engine health via REST: {e}")
            return False


def get_circuit_breaker_state() -> str:
    """
    Get current circuit breaker state for monitoring.
    
    Returns:
        Current circuit breaker state as string
    """
    return _circuit_breaker.get_state().value


def reset_circuit_breaker() -> None:
    """
    Manually reset circuit breaker.
    
    This should only be used for maintenance or testing purposes.
    """
    _circuit_breaker.reset()
    logger.warning("Circuit breaker manually reset")