"""
API routes for AI Engine integration.

Provides endpoints for:
- Health checks of the AI Engine
- File analysis (CSV/parquet upload)
"""
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Request
from pydantic import BaseModel, Field

from .ai_engine_client import (
    check_ai_engine_health,
    analyze_file,
    get_engine_info,
    HealthCheckResult,
    AnalysisResult
)
from .settings import settings
from .utils import generate_request_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["ai-engine"])


# ============================================
# Response Models
# ============================================

class AIHealthResponse(BaseModel):
    """Response model for AI Engine health check."""
    status: str = Field(..., description="Health status: healthy, unhealthy, or unreachable")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    engine_url: str = Field(..., description="AI Engine URL")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")
    error: Optional[str] = Field(None, description="Error message if unhealthy")
    request_id: str = Field(..., description="Correlation ID for this request")


class AIAnalysisResponse(BaseModel):
    """Response model for AI analysis."""
    success: bool = Field(..., description="Whether analysis succeeded")
    job_id: Optional[str] = Field(None, description="Job ID from AI Engine")
    outputs: Optional[Dict[str, Any]] = Field(None, description="Analysis outputs and results")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: float = Field(..., description="Total processing time in ms")
    request_id: str = Field(..., description="Correlation ID for this request")
    filename: str = Field(..., description="Uploaded filename")


class AIEngineInfoResponse(BaseModel):
    """Response model for AI Engine configuration info."""
    url: str
    timeout: int
    health_timeout: int
    max_retries: int
    allowed_extensions: List[str]


# ============================================
# Endpoints
# ============================================

@router.get("/health", response_model=AIHealthResponse)
async def ai_health_check(request: Request) -> AIHealthResponse:
    """
    Check AI Engine health status.

    Tests connectivity to the DeepECG AI Engine and returns status,
    latency, and any error details.

    Returns:
        AIHealthResponse with health status and metrics
    """
    request_id = getattr(request.state, "request_id", generate_request_id())

    logger.info(f"[{request_id}] AI health check requested")

    result: HealthCheckResult = await check_ai_engine_health(request_id)

    return AIHealthResponse(
        status=result.status,
        latency_ms=result.latency_ms,
        engine_url=result.engine_url,
        details=result.details,
        error=result.error,
        request_id=request_id
    )


@router.post("/analyze", response_model=AIAnalysisResponse)
async def ai_analyze_file(
    request: Request,
    file: UploadFile = File(..., description="CSV or Parquet file to analyze")
) -> AIAnalysisResponse:
    """
    Upload a file to the AI Engine for analysis.

    Accepts CSV (.csv) or Parquet (.parquet) files containing ECG data.
    The file is forwarded to the DeepECG AI Engine for processing.

    Args:
        file: Uploaded file (CSV or Parquet)

    Returns:
        AIAnalysisResponse with job_id and analysis results

    Raises:
        HTTPException 415: If file type is not supported
        HTTPException 413: If file is too large
    """
    request_id = getattr(request.state, "request_id", generate_request_id())
    filename = file.filename or "unknown"

    logger.info(f"[{request_id}] AI analysis requested for file: {filename}")

    # Validate file extension
    file_ext = Path(filename).suffix.lower()
    if file_ext not in settings.ALLOWED_DATA_EXTENSIONS:
        logger.warning(
            f"[{request_id}] Invalid file extension: {file_ext}"
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "error": f"Unsupported file type: {file_ext}",
                "allowed_extensions": settings.ALLOWED_DATA_EXTENSIONS,
                "request_id": request_id
            }
        )

    # Read file content
    try:
        file_content = await file.read()
    except Exception as e:
        logger.error(f"[{request_id}] Failed to read uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Failed to read uploaded file",
                "request_id": request_id
            }
        )

    # Validate file size
    file_size_mb = len(file_content) / (1024 * 1024)
    if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
        logger.warning(
            f"[{request_id}] File too large: {file_size_mb:.1f}MB > {settings.MAX_UPLOAD_SIZE_MB}MB"
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": f"File too large: {file_size_mb:.1f}MB",
                "max_size_mb": settings.MAX_UPLOAD_SIZE_MB,
                "request_id": request_id
            }
        )

    # Send to AI Engine
    result: AnalysisResult = await analyze_file(
        file_content=file_content,
        filename=filename,
        request_id=request_id
    )

    if not result.success:
        logger.error(f"[{request_id}] AI analysis failed: {result.error}")

    return AIAnalysisResponse(
        success=result.success,
        job_id=result.job_id,
        outputs=result.outputs,
        error=result.error,
        processing_time_ms=result.processing_time_ms,
        request_id=request_id,
        filename=filename
    )


@router.get("/info", response_model=AIEngineInfoResponse)
async def ai_engine_info() -> AIEngineInfoResponse:
    """
    Get AI Engine configuration information.

    Returns current configuration for the AI Engine connection,
    including URL, timeouts, and allowed file types.

    Returns:
        AIEngineInfoResponse with configuration details
    """
    info = get_engine_info()
    return AIEngineInfoResponse(**info)
