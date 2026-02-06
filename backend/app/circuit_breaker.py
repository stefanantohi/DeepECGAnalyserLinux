"""Circuit breaker pattern for AI Engine resilience."""
import time
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation, requests flow through
    OPEN = "open"  # Failing, reject requests immediately
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreaker:
    """
    Circuit breaker implementation for AI Engine resilience.
    
    Prevents cascading failures by temporarily blocking requests
    to a service that is experiencing issues.
    """
    
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
        name: str = "default"
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_max_calls: Number of calls to test in half-open state
            name: Circuit breaker name for logging
        """
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.half_open_call_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self.name = name
        
        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"threshold={failure_threshold}, timeout={recovery_timeout}s"
        )
    
    def record_success(self) -> None:
        """Record a successful request."""
        if self.state in (CircuitState.OPEN, CircuitState.HALF_OPEN):
            logger.info(f"Circuit breaker '{self.name}' recovered successfully")
        
        self.failure_count = 0
        self.half_open_call_count = 0
        self.state = CircuitState.CLOSED
        logger.debug(f"Circuit breaker '{self.name}': Success recorded, state={self.state.value}")
    
    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            old_state = self.state
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker '{self.name}' opened after "
                f"{self.failure_count} consecutive failures"
            )
            if old_state != CircuitState.OPEN:
                # Log alert when first opening
                logger.warning(
                    f"⚠️ AI Engine service degraded - "
                    f"Circuit breaker '{self.name}' is OPEN"
                )
    
    def can_request(self) -> bool:
        """
        Check if request should be allowed.
        
        Returns:
            True if request can proceed, False otherwise
        """
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if self.last_failure_time is not None:
                elapsed = time.time() - self.last_failure_time
                if elapsed > self.recovery_timeout:
                    logger.info(
                        f"Circuit breaker '{self.name}' recovery timeout elapsed, "
                        f"transitioning to HALF_OPEN"
                    )
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_call_count = 0
                    return True
            
            logger.debug(f"Circuit breaker '{self.name}': Request blocked (OPEN)")
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            # Allow limited number of calls to test recovery
            if self.half_open_call_count < self.half_open_max_calls:
                self.half_open_call_count += 1
                logger.debug(
                    f"Circuit breaker '{self.name}': Testing recovery "
                    f"({self.half_open_call_count}/{self.half_open_max_calls})"
                )
                return True
            else:
                logger.warning(f"Circuit breaker '{self.name}': Half-open limit reached, opening")
                self.state = CircuitState.OPEN
                return False
        
        return False
    
    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.state
    
    def reset(self) -> None:
        """Manually reset circuit breaker (for testing or admin override)."""
        self.failure_count = 0
        self.half_open_call_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        logger.info(f"Circuit breaker '{self.name}' manually reset")