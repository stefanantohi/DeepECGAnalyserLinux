"""Custom exceptions for the backend application."""
from typing import Optional


class ClinicalError(Exception):
    """
    Error related to clinical interpretation or medical analysis.
    
    These errors indicate issues with the medical analysis results,
    not technical infrastructure problems.
    """
    def __init__(
        self,
        message: str,
        clinical_code: str,
        severity: str = "warning"
    ):
        self.message = message
        self.clinical_code = clinical_code
        self.severity = severity  # "info", "warning", "critical"
        super().__init__(message)


class TechnicalError(Exception):
    """
    Error related to infrastructure, system, or technical components.
    
    These errors indicate problems with the underlying systems,
    not the medical analysis itself.
    """
    pass


class AIEngineError(Exception):
    """Error communicating with or processing in the AI Engine."""
    def __init__(self, message: str, degraded_mode: bool = False):
        self.message = message
        self.degraded_mode = degraded_mode
        super().__init__(message)