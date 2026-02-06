"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Literal, Tuple
from datetime import datetime, timezone
from enum import Enum


class AnalysisStatus(str, Enum):
    """Analysis status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Prediction schemas
class Prediction(BaseModel):
    """Basic prediction result from AI engine."""
    label: str = Field(..., description="Prediction label")
    score: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    description: Optional[str] = Field(None, description="Optional description")


class PredictionWithConfidence(BaseModel):
    """
    Enhanced prediction with confidence intervals and clinical categorization.
    
    This schema provides additional context for clinical interpretation,
    including confidence intervals and categorization levels.
    """
    label: str = Field(..., description="Predicted label or class")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    confidence_interval: Optional[Tuple[float, float]] = Field(
        None,
        description="95% confidence interval (min, max) if available"
    )
    confidence_level: Optional[str] = Field(
        None,
        description="Confidence category: 'high' (>0.9), 'medium' (0.7-0.9), 'low' (<0.7)"
    )
    description: Optional[str] = Field(None, description="Human-readable description")
    category: Optional[str] = Field(
        None,
        description="Clinical category: 'rhythm', 'morphology', 'diagnosis', etc."
    )


class PredictionResult(BaseModel):
    """Complete prediction result from AI engine."""
    predictions: List[Prediction] = Field(default_factory=list, description="List of predictions")
    scores: Dict[str, float] = Field(default_factory=dict, description="Raw scores dictionary")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


# Request schemas
class AnalysisRequest(BaseModel):
    """Request body for analysis."""
    request_id: str = Field(..., description="Unique request identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")


# Response schemas
class AnalysisResponse(BaseModel):
    """Response for analysis request."""
    request_id: str = Field(..., description="Unique request identifier")
    status: AnalysisStatus = Field(..., description="Analysis status")
    filename: str = Field(..., description="Original filename")
    result: Optional[PredictionResult] = Field(None, description="Prediction results if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of request")


class AnalysisMetadata(BaseModel):
    """
    Complete lifecycle metadata for an analysis request.
    
    This schema provides full traceability for clinical compliance,
    including model version, processing details, and timing information.
    """
    request_id: str = Field(..., description="Unique request identifier")
    timestamp_start: datetime = Field(..., description="Analysis start time (UTC)")
    timestamp_end: Optional[datetime] = Field(None, description="Analysis end time (UTC)")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    
    # AI Engine specifics
    ai_engine_url: str = Field(..., description="AI Engine endpoint URL")
    ai_engine_mode: str = Field(..., description="Communication mode: 'rest' or 'cli'")
    ai_engine_version: Optional[str] = Field(None, description="Model version from AI Engine")
    ai_engine_hash: Optional[str] = Field(None, description="Model hash for reproducibility")
    
    # Document specifics
    filename: str = Field(..., description="Original filename")
    file_size_bytes: int = Field(..., description="File size in bytes")
    pdf_page_count: int = Field(..., description="Number of pages in PDF")
    pdf_text_length: int = Field(..., description="Total extracted text characters")
    xml_size_bytes: int = Field(..., description="Size of generated XML in bytes")


class ErrorResponse(BaseModel):
    """
    Structured error response distinguishing technical from clinical errors.
    
    This schema helps users understand whether they're facing a system issue
    or a clinical interpretation problem.
    """
    request_id: str = Field(..., description="Associated request ID")
    error_type: Literal["technical", "clinical"] = Field(..., description="Type of error")
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    actionable: bool = Field(..., description="Whether user can resolve the issue")
    suggested_action: Optional[str] = Field(None, description="Suggested resolution steps")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    ai_engine_connected: bool = Field(..., description="AI engine connectivity status")
    ai_engine_status: Optional[str] = Field(None, description="AI engine detailed status")
    circuit_breaker_state: Optional[str] = Field(None, description="Circuit breaker state")
    temp_dir_exists: bool = Field(..., description="Temporary directory status")


# ECG Analysis schemas
class ECGModelInfo(BaseModel):
    """Information about an available ECG model."""
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable model name")
    architecture: str = Field(..., description="Model architecture: efficientnet or wcr")
    type: str = Field(..., description="Model type: multi_label or binary")
    description: Optional[str] = Field(None, description="Model description")


class ECGModelsResponse(BaseModel):
    """Response with available ECG models."""
    models: List[ECGModelInfo] = Field(default_factory=list, description="Available models")
    default_selection: List[str] = Field(default_factory=list, description="Default selected models")


class ECGDiagnosisResult(BaseModel):
    """Result for a single ECG diagnosis."""
    name: str = Field(..., description="Diagnosis name")
    probability: float = Field(..., ge=0, le=100, description="Probability percentage (0-100)")
    threshold: float = Field(..., ge=0, le=100, description="Detection threshold percentage")
    status: Literal["normal", "borderline", "abnormal"] = Field(..., description="Risk status")
    category: str = Field(..., description="Diagnosis category")


class ECGBinaryResult(BaseModel):
    """Result for a binary ECG model (LVEF, AF)."""
    model_id: str = Field(..., description="Model identifier")
    model_name: str = Field(..., description="Model name")
    probability: float = Field(..., ge=0, le=100, description="Positive probability percentage")
    threshold: float = Field(..., ge=0, le=100, description="Detection threshold")
    status: Literal["normal", "borderline", "abnormal"] = Field(..., description="Risk status")
    positive: bool = Field(..., description="Whether the result is above threshold")


class ECGModelResult(BaseModel):
    """Results from a single ECG model."""
    model_id: str = Field(..., description="Model identifier")
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="multi_label or binary")
    architecture: str = Field(..., description="efficientnet or wcr")
    success: bool = Field(..., description="Whether model ran successfully")
    error: Optional[str] = Field(None, description="Error message if failed")
    diagnoses: List[ECGDiagnosisResult] = Field(default_factory=list, description="Diagnosis results")
    binary_result: Optional[ECGBinaryResult] = Field(None, description="Binary result for LVEF/AF models")
    by_category: Dict[str, List[ECGDiagnosisResult]] = Field(
        default_factory=dict, description="Diagnoses grouped by category"
    )


class ECGCriticalFinding(BaseModel):
    """A critical finding that requires attention."""
    diagnosis: str = Field(..., description="Diagnosis name")
    probability: float = Field(..., description="Probability percentage")
    model: str = Field(..., description="Model that detected this finding")
    category: Optional[str] = Field(None, description="Diagnosis category")


class ECGAnalysisSummary(BaseModel):
    """Summary of the ECG analysis."""
    overall_status: Literal["normal", "borderline", "abnormal"] = Field(
        ..., description="Overall status based on all models"
    )
    total_abnormal: int = Field(0, description="Total abnormal findings across all models")
    total_borderline: int = Field(0, description="Total borderline findings")
    critical_findings: List[ECGCriticalFinding] = Field(
        default_factory=list, description="List of critical findings"
    )


class FullECGAnalysisRequest(BaseModel):
    """Request for full ECG analysis."""
    models: List[str] = Field(
        default=["all"],
        description="List of model IDs to run, or ['all'] for all models"
    )
    use_gpu: bool = Field(default=True, description="Whether to use GPU")
    patient_id: Optional[str] = Field(None, description="Patient identifier")


class ECGFileFormatInfo(BaseModel):
    """Information about ECG file format detection and conversion."""
    original_format: str = Field(
        default="unknown",
        description="Detected original format: 'mhi', 'ge_muse', 'philips_pagewriter', 'unknown'"
    )
    original_encoding: str = Field(
        default="utf-8",
        description="Original file encoding: 'utf-8', 'utf-16-le', 'utf-16-be', etc."
    )
    conversions_applied: List[str] = Field(
        default_factory=list,
        description="List of conversions applied: 'utf16_to_utf8', 'philips_to_mhi', etc."
    )
    conversion_notes: Optional[str] = Field(
        None,
        description="Additional notes about the conversion process"
    )


class FullECGAnalysisResponse(BaseModel):
    """Complete response from full ECG analysis."""
    success: bool = Field(..., description="Whether analysis completed successfully")
    patient_id: str = Field(..., description="Patient identifier")
    ecg_filename: str = Field(..., description="Original ECG filename")
    file_format_info: Optional[ECGFileFormatInfo] = Field(
        None, description="Information about file format detection and conversion"
    )
    models_executed: List[str] = Field(default_factory=list, description="Models that were run")
    results: Dict[str, ECGModelResult] = Field(
        default_factory=dict, description="Results by model ID"
    )
    summary: ECGAnalysisSummary = Field(..., description="Analysis summary")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchECGResult(BaseModel):
    """Result for a single ECG in a batch."""
    index: int = Field(..., description="Index in the batch (0-based)")
    filename: str = Field(..., description="Original filename")
    patient_id: str = Field(..., description="Patient identifier")
    success: bool = Field(..., description="Whether analysis succeeded")
    result: Optional[FullECGAnalysisResponse] = Field(None, description="Analysis result if successful")
    error: Optional[str] = Field(None, description="Error message if failed")


class ConfigResponse(BaseModel):
    """Response for workspace configuration."""
    workspace_path: str = Field(..., description="Current workspace path")
    workspace_exists: bool = Field(..., description="Whether workspace directory exists")
    subdirectories: Dict[str, bool] = Field(
        default_factory=dict, description="Status of each subdirectory"
    )


class ConfigUpdateRequest(BaseModel):
    """Request to update workspace configuration."""
    workspace_path: str = Field(..., description="New workspace path")


class BatchECGAnalysisResponse(BaseModel):
    """Response for batch ECG analysis."""
    success: bool = Field(..., description="Whether batch completed")
    total_files: int = Field(..., description="Total number of files processed")
    successful: int = Field(..., description="Number of successful analyses")
    failed: int = Field(..., description="Number of failed analyses")
    results: List[BatchECGResult] = Field(default_factory=list, description="Results for each ECG")
    total_processing_time_ms: int = Field(..., description="Total processing time")
    models_used: List[str] = Field(default_factory=list, description="Models used for all analyses")