"""FastAPI application endpoints."""
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from .schemas import (
    AnalysisResponse,
    HealthResponse,
    PredictionResult,
    Prediction,
    AnalysisStatus,
    ECGModelInfo,
    ECGModelsResponse,
    ECGDiagnosisResult,
    ECGModelResult,
    ECGAnalysisSummary,
    ECGCriticalFinding,
    FullECGAnalysisResponse,
    BatchECGResult,
    BatchECGAnalysisResponse,
    ECGFileFormatInfo,
    ConfigResponse,
    ConfigUpdateRequest,
)
from .pdf_to_xml import pdf_to_xml, validate_xml
from .ai_client import predict_from_xml, check_health, AIEngineError
from .settings import settings, ensure_workspace, WORKSPACE_SUBDIRS
from .ecg_analysis import detect_and_convert_encoding
from .philips_converter import auto_convert_if_philips
from .ecg_normalizer import normalize_to_muse
from .utils import (
    generate_request_id,
    ensure_temp_directory,
    safe_remove_file,
    validate_mime_type,
    Timer
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["api"])


@router.get("")
@router.get("/")
async def api_root():
    """Root /api endpoint — returns available sub-routes."""
    return {
        "name": "DeepECG API",
        "endpoints": {
            "health": "/api/health",
            "analyze": "/api/analyze",
            "config": "/api/config",
            "ecg_full_analysis": "/api/ecg/full-analysis",
            "ecg_models": "/api/ecg/available-models",
            "ecg_parse_existing": "/api/ecg/parse-existing",
            "ecg_signal_data": "/api/ecg/signal-data",
            "ai_health": "/api/ai/health",
            "docker_status": "/api/docker/status",
            "docker_start": "/api/docker/start",
            "docker_stop": "/api/docker/stop",
        }
    }


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_pdf(file: UploadFile = File(...)) -> AnalysisResponse:
    """
    Analyze uploaded PDF file.
    
    Args:
        file: Uploaded PDF file
        
    Returns:
        AnalysisResponse with predictions or error details
    """
    request_id = generate_request_id()
    filename = file.filename or "unknown.pdf"
    file_size = 0
    
    logger.info(f"Received analysis request {request_id} for file: {filename}")
    
    # Ensure temp directory exists
    temp_dir = ensure_temp_directory(settings.TEMP_DIR)
    
    # Validate file size
    content = await file.read()
    file_size = len(content)
    max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    
    if file_size > max_size:
        logger.warning(f"File {filename} exceeds size limit: {file_size} > {max_size}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB"
        )
    
    # Validate MIME type
    if not validate_mime_type(file.content_type, settings.ALLOWED_MIME_TYPES):
        logger.warning(f"Invalid file type {file.content_type} for file: {filename}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF files are allowed (got: {file.content_type})"
        )
    
    # Save uploaded file temporarily
    pdf_path = temp_dir / f"{request_id}.pdf"
    try:
        pdf_path.write_bytes(content)
        logger.info(f"Saved uploaded file to: {pdf_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process uploaded file"
        )
    
    # Process the file
    timer = Timer(f"Analysis request {request_id}")
    
    with timer:
        try:
            # Step 1: Convert PDF to XML
            logger.info(f"Converting PDF to XML for request {request_id}")
            xml_string = pdf_to_xml(pdf_path)
            
            # Validate XML
            if not validate_xml(xml_string):
                raise ValueError("Generated XML is not well-formed")
            
            # Step 2: Call AI Engine
            logger.info(f"Calling AI Engine for request {request_id}")
            ai_result = await predict_from_xml(xml_string)
            
            # Step 3: Parse AI Engine response
            logger.info(f"Parsing AI Engine response for request {request_id}")
            prediction_result = PredictionResult(
                predictions=[
                    Prediction(**pred) for pred in ai_result.get("predictions", [])
                ],
                scores=ai_result.get("scores", {}),
                metadata=ai_result.get("metadata", {})
            )
            
            logger.info(f"Successfully processed request {request_id}")
            
            return AnalysisResponse(
                request_id=request_id,
                status=AnalysisStatus.COMPLETED,
                filename=filename,
                result=prediction_result,
                processing_time_ms=timer.elapsed_ms()
            )
            
        except AIEngineError as e:
            logger.error(f"AI Engine error for request {request_id}: {e}")
            return AnalysisResponse(
                request_id=request_id,
                status=AnalysisStatus.FAILED,
                filename=filename,
                error=f"AI Engine error: {str(e)}",
                processing_time_ms=timer.elapsed_ms()
            )
            
        except ValueError as e:
            logger.error(f"Validation error for request {request_id}: {e}")
            return AnalysisResponse(
                request_id=request_id,
                status=AnalysisStatus.FAILED,
                filename=filename,
                error=f"Validation error: {str(e)}",
                processing_time_ms=timer.elapsed_ms()
            )
            
        except Exception as e:
            logger.error(f"Unexpected error processing request {request_id}: {e}", exc_info=True)
            return AnalysisResponse(
                request_id=request_id,
                status=AnalysisStatus.FAILED,
                filename=filename,
                error=f"Internal server error: {str(e)}",
                processing_time_ms=timer.elapsed_ms()
            )
            
        finally:
            # Clean up temporary files
            safe_remove_file(pdf_path)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse with service status
    """
    # Check AI Engine connectivity
    ai_connected = await check_health()

    # Check temp directory
    temp_dir = Path(settings.TEMP_DIR)
    temp_exists = temp_dir.exists() and temp_dir.is_dir()

    return HealthResponse(
        status="healthy" if (ai_connected and temp_exists) else "degraded",
        version=settings.APP_VERSION,
        ai_engine_connected=ai_connected,
        temp_dir_exists=temp_exists
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """
    Get current workspace configuration and directory status.
    """
    workspace_path = settings.WORKSPACE_PATH
    ws = Path(workspace_path)
    workspace_exists = ws.exists()

    subdirs = {}
    for subdir in WORKSPACE_SUBDIRS:
        subdirs[subdir] = (ws / subdir).exists()

    return ConfigResponse(
        workspace_path=workspace_path,
        workspace_exists=workspace_exists,
        subdirectories=subdirs,
    )


@router.put("/config", response_model=ConfigResponse)
async def update_config(request: ConfigUpdateRequest) -> ConfigResponse:
    """
    Update workspace path, persist it, and create directories if needed.
    """
    new_path = request.workspace_path.strip()
    if not new_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workspace path cannot be empty",
        )

    # Validate that parent directory exists
    parent = Path(new_path).parent
    if not parent.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Parent directory does not exist: {parent}",
        )

    # Update settings + persist + create directories
    subdirs = settings.update_workspace_path(new_path)
    logger.info(f"Workspace path updated to: {new_path}")

    return ConfigResponse(
        workspace_path=new_path,
        workspace_exists=Path(new_path).exists(),
        subdirectories=subdirs,
    )


# ===================== ECG Analysis Endpoints =====================

@router.get("/ecg/available-models", response_model=ECGModelsResponse)
async def get_available_models() -> ECGModelsResponse:
    """
    Get list of available ECG analysis models.

    Returns:
        ECGModelsResponse with available models and default selection
    """
    from .ecg_analysis import AVAILABLE_MODELS

    models = []
    for model_id, config in AVAILABLE_MODELS.items():
        models.append(ECGModelInfo(
            id=model_id,
            name=config["name"],
            architecture=config["architecture"],
            type=config["type"],
            description=f"{config['name']} - {config['architecture'].upper()} architecture"
        ))

    # Default selection: all EfficientNet models
    default_selection = [m for m in AVAILABLE_MODELS.keys() if "efficientnet" in m]

    return ECGModelsResponse(
        models=models,
        default_selection=default_selection
    )


@router.get("/ecg/supported-formats")
async def get_supported_formats():
    """Get list of supported ECG file formats with conversion details."""
    from .ecg_normalizer import SUPPORTED_FORMATS
    return {"formats": SUPPORTED_FORMATS}


@router.post("/ecg/full-analysis", response_model=FullECGAnalysisResponse)
async def run_full_ecg_analysis(
    file: UploadFile = File(...),
    models: str = Form(default="all"),
    use_gpu: bool = Form(default=False),
    patient_id: Optional[str] = Form(default=None)
) -> FullECGAnalysisResponse:
    """
    Run complete ECG analysis with selected models.

    Args:
        file: ECG file (XML format)
        models: Comma-separated model IDs or "all"
        use_gpu: Whether to use GPU acceleration
        patient_id: Optional patient identifier

    Returns:
        FullECGAnalysisResponse with comprehensive results
    """
    from .ecg_analysis import run_full_ecg_analysis as do_analysis, AVAILABLE_MODELS

    # Parse models list
    if models == "all":
        selected_models = ["all"]
    else:
        selected_models = [m.strip() for m in models.split(",") if m.strip()]

    # Validate models
    invalid_models = [m for m in selected_models if m != "all" and m not in AVAILABLE_MODELS]
    if invalid_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model IDs: {invalid_models}"
        )

    # Generate patient ID if not provided
    if not patient_id:
        patient_id = f"PAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Validate file extension
    filename = file.filename or "ecg.xml"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ['.xml', '.npy']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Expected XML or NPY, got: {ext}"
        )

    # Save file to workspace
    workspace_path = settings.WORKSPACE_PATH
    ecg_signals_dir = os.path.join(workspace_path, "ecg_signals")
    os.makedirs(ecg_signals_dir, exist_ok=True)

    # Save uploaded file (using original filename)
    ecg_file_path = os.path.join(ecg_signals_dir, filename)
    content = await file.read()
    with open(ecg_file_path, 'wb') as f:
        f.write(content)

    # Also save a backup copy for ECG visualization (won't be deleted by preprocessing)
    ecg_backup_path = os.path.join(ecg_signals_dir, f"original_{filename}")
    with open(ecg_backup_path, 'wb') as f:
        f.write(content)

    logger.info(f"Saved ECG file to: {ecg_file_path}")

    # Track file format info for response
    file_format_info = {
        "original_format": "unknown",
        "original_encoding": "utf-8",
        "conversions_applied": [],
        "conversion_notes": None
    }

    # Auto-detect and convert encoding (UTF-16 -> UTF-8) for XML files
    if ext == '.xml':
        was_converted, original_encoding, conv_msg = detect_and_convert_encoding(ecg_file_path)
        file_format_info["original_encoding"] = original_encoding or "utf-8"
        if was_converted:
            logger.info(f"Encoding conversion: {conv_msg}")
            file_format_info["conversions_applied"].append(f"Encodage: {original_encoding} → UTF-8")
            # Also convert the backup copy
            detect_and_convert_encoding(ecg_backup_path)

        # Auto-detect and convert Philips PageWriter format to GE MUSE
        philips_converted, format_detected, philips_msg = auto_convert_if_philips(ecg_file_path)
        if format_detected != "other":
            file_format_info["original_format"] = format_detected
        if philips_converted:
            logger.info(f"Philips conversion: {philips_msg}")
            file_format_info["conversions_applied"].append("Format: Philips PageWriter → GE MUSE")
            file_format_info["conversion_notes"] = "Battements représentatifs extraits et étendus à 10s"
            # Note: We do NOT convert the backup copy - keep original for debugging
            # The ECG viewer will use the original format from ecg_backup_path
        elif format_detected == "philips_pagewriter":
            logger.warning(f"Philips file detected but conversion failed: {philips_msg}")
            file_format_info["conversion_notes"] = f"Conversion échouée: {philips_msg}"
        else:
            # Not Philips - try universal normalizer (CardiologyXML, single-waveform MUSE, etc.)
            normalized, norm_format, norm_msg = normalize_to_muse(ecg_file_path)
            if normalized:
                logger.info(f"ECG normalization: {norm_msg}")
                file_format_info["original_format"] = norm_format
                file_format_info["conversions_applied"].append(f"Format: {norm_format} → GE MUSE")
                file_format_info["conversion_notes"] = norm_msg
            else:
                file_format_info["original_format"] = norm_format if norm_format != "unknown" else "ge_muse"
    elif ext == '.npy':
        file_format_info["original_format"] = "numpy"

    logger.info(f"Backup copy saved to: {ecg_backup_path}")
    logger.info(f"Running analysis with models: {selected_models}")

    try:
        # Run analysis
        result = await do_analysis(
            ecg_file_path=ecg_file_path,
            patient_id=patient_id,
            workspace_path=workspace_path,
            selected_models=selected_models,
            use_gpu=use_gpu
        )

        # Convert to response model
        model_results = {}
        for model_id, model_data in result.get("results", {}).items():
            diagnoses = []
            by_category = {}

            if model_data.get("success") and "results" in model_data:
                for diag in model_data["results"].get("diagnoses", []):
                    diag_result = ECGDiagnosisResult(
                        name=diag["name"],
                        probability=diag["probability"],
                        threshold=diag["threshold"],
                        status=diag["status"],
                        category=diag["category"]
                    )
                    diagnoses.append(diag_result)

                # Group by category
                for cat, cat_diags in model_data["results"].get("by_category", {}).items():
                    by_category[cat] = [
                        ECGDiagnosisResult(
                            name=d["name"],
                            probability=d["probability"],
                            threshold=d["threshold"],
                            status=d["status"],
                            category=d["category"]
                        ) for d in cat_diags
                    ]

            model_results[model_id] = ECGModelResult(
                model_id=model_id,
                model_name=model_data.get("model_name", model_id),
                model_type=model_data.get("model_type", "unknown"),
                architecture=model_data.get("architecture", "unknown"),
                success=model_data.get("success", False),
                error=model_data.get("error"),
                diagnoses=diagnoses,
                by_category=by_category
            )

        # Build critical findings
        critical_findings = [
            ECGCriticalFinding(
                diagnosis=cf["diagnosis"],
                probability=cf["probability"],
                model=cf["model"]
            ) for cf in result.get("summary", {}).get("critical_findings", [])
        ]

        summary = ECGAnalysisSummary(
            overall_status=result.get("summary", {}).get("overall_status", "normal"),
            total_abnormal=result.get("summary", {}).get("total_abnormal", 0),
            total_borderline=result.get("summary", {}).get("total_borderline", 0),
            critical_findings=critical_findings
        )

        return FullECGAnalysisResponse(
            success=result.get("success", False),
            patient_id=result.get("patient_id", patient_id),
            ecg_filename=result.get("ecg_filename", filename),
            file_format_info=ECGFileFormatInfo(
                original_format=file_format_info["original_format"],
                original_encoding=file_format_info["original_encoding"],
                conversions_applied=file_format_info["conversions_applied"],
                conversion_notes=file_format_info["conversion_notes"]
            ),
            models_executed=result.get("models_executed", []),
            results=model_results,
            summary=summary,
            warnings=result.get("warnings", []),
            processing_time_ms=result.get("processing_time_ms", 0),
            error=result.get("error")
        )

    except Exception as e:
        logger.error(f"ECG analysis failed: {e}", exc_info=True)
        return FullECGAnalysisResponse(
            success=False,
            patient_id=patient_id,
            ecg_filename=filename,
            file_format_info=ECGFileFormatInfo(
                original_format=file_format_info.get("original_format", "unknown"),
                original_encoding=file_format_info.get("original_encoding", "utf-8"),
                conversions_applied=file_format_info.get("conversions_applied", []),
                conversion_notes=file_format_info.get("conversion_notes")
            ),
            models_executed=[],
            results={},
            summary=ECGAnalysisSummary(
                overall_status="normal",
                total_abnormal=0,
                total_borderline=0,
                critical_findings=[]
            ),
            warnings=[],
            processing_time_ms=0,
            error=str(e)
        )


@router.post("/ecg/batch-analysis", response_model=BatchECGAnalysisResponse)
async def run_batch_ecg_analysis(
    files: List[UploadFile] = File(...),
    models: str = Form(default="all"),
    use_gpu: bool = Form(default=False)
) -> BatchECGAnalysisResponse:
    """
    Run ECG analysis on multiple files (batch processing).

    Args:
        files: List of ECG files (XML format)
        models: Comma-separated model IDs or "all"
        use_gpu: Whether to use GPU acceleration

    Returns:
        BatchECGAnalysisResponse with results for each file
    """
    import time
    from .ecg_analysis import run_full_ecg_analysis as do_analysis, AVAILABLE_MODELS

    start_time = time.time()

    # Parse models list
    if models == "all":
        selected_models = ["all"]
    else:
        selected_models = [m.strip() for m in models.split(",") if m.strip()]

    # Validate models
    invalid_models = [m for m in selected_models if m != "all" and m not in AVAILABLE_MODELS]
    if invalid_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model IDs: {invalid_models}"
        )

    workspace_path = settings.WORKSPACE_PATH
    ecg_signals_dir = os.path.join(workspace_path, "ecg_signals")
    os.makedirs(ecg_signals_dir, exist_ok=True)

    results = []
    successful = 0
    failed = 0

    for idx, file in enumerate(files):
        filename = file.filename or f"ecg_{idx}.xml"
        patient_id = f"PAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx:03d}"

        # Validate file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ['.xml', '.npy']:
            results.append(BatchECGResult(
                index=idx,
                filename=filename,
                patient_id=patient_id,
                success=False,
                error=f"Invalid file type: {ext}. Expected XML or NPY."
            ))
            failed += 1
            continue

        try:
            # Save file to workspace
            ecg_file_path = os.path.join(ecg_signals_dir, filename)
            content = await file.read()
            with open(ecg_file_path, 'wb') as f:
                f.write(content)

            # Also save backup
            backup_path = os.path.join(ecg_signals_dir, f"original_{filename}")
            with open(backup_path, 'wb') as f:
                f.write(content)

            # Track file format info
            batch_format_info = {
                "original_format": "unknown",
                "original_encoding": "utf-8",
                "conversions_applied": [],
                "conversion_notes": None
            }

            # Auto-detect and convert encoding (UTF-16 -> UTF-8) for XML files
            if ext == '.xml':
                was_converted, original_enc, conv_msg = detect_and_convert_encoding(ecg_file_path)
                batch_format_info["original_encoding"] = original_enc or "utf-8"
                if was_converted:
                    logger.info(f"Encoding conversion for {filename}: {conv_msg}")
                    batch_format_info["conversions_applied"].append(f"Encodage: {original_enc} → UTF-8")
                    detect_and_convert_encoding(backup_path)

                # Auto-detect and convert Philips PageWriter format to GE MUSE
                philips_converted, format_detected, philips_msg = auto_convert_if_philips(ecg_file_path)
                if format_detected != "other":
                    batch_format_info["original_format"] = format_detected
                if philips_converted:
                    logger.info(f"Philips conversion for {filename}: {philips_msg}")
                    batch_format_info["conversions_applied"].append("Format: Philips PageWriter → GE MUSE")
                    batch_format_info["conversion_notes"] = "Battements représentatifs extraits et étendus à 10s"
                    # Note: Do NOT convert backup - keep original for debugging
                elif format_detected == "philips_pagewriter":
                    logger.warning(f"Philips file {filename} detected but conversion failed: {philips_msg}")
                    batch_format_info["conversion_notes"] = f"Conversion échouée: {philips_msg}"
                else:
                    # Not Philips - try universal normalizer
                    normalized, norm_format, norm_msg = normalize_to_muse(ecg_file_path)
                    if normalized:
                        logger.info(f"ECG normalization for {filename}: {norm_msg}")
                        batch_format_info["original_format"] = norm_format
                        batch_format_info["conversions_applied"].append(f"Format: {norm_format} → GE MUSE")
                        batch_format_info["conversion_notes"] = norm_msg
                    else:
                        batch_format_info["original_format"] = norm_format if norm_format != "unknown" else "ge_muse"
            elif ext == '.npy':
                batch_format_info["original_format"] = "numpy"

            logger.info(f"Processing batch file {idx+1}/{len(files)}: {filename}")

            # Run analysis
            result = await do_analysis(
                ecg_file_path=ecg_file_path,
                patient_id=patient_id,
                workspace_path=workspace_path,
                selected_models=selected_models,
                use_gpu=use_gpu
            )

            # Convert to response model
            model_results = {}
            for model_id, model_data in result.get("results", {}).items():
                diagnoses = []
                by_category = {}

                if model_data.get("success") and "results" in model_data:
                    for diag in model_data["results"].get("diagnoses", []):
                        diag_result = ECGDiagnosisResult(
                            name=diag["name"],
                            probability=diag["probability"],
                            threshold=diag["threshold"],
                            status=diag["status"],
                            category=diag["category"]
                        )
                        diagnoses.append(diag_result)

                    for cat, cat_diags in model_data["results"].get("by_category", {}).items():
                        by_category[cat] = [
                            ECGDiagnosisResult(
                                name=d["name"],
                                probability=d["probability"],
                                threshold=d["threshold"],
                                status=d["status"],
                                category=d["category"]
                            ) for d in cat_diags
                        ]

                model_results[model_id] = ECGModelResult(
                    model_id=model_id,
                    model_name=model_data.get("model_name", model_id),
                    model_type=model_data.get("model_type", "unknown"),
                    architecture=model_data.get("architecture", "unknown"),
                    success=model_data.get("success", False),
                    error=model_data.get("error"),
                    diagnoses=diagnoses,
                    by_category=by_category
                )

            critical_findings = [
                ECGCriticalFinding(
                    diagnosis=cf["diagnosis"],
                    probability=cf["probability"],
                    model=cf["model"]
                ) for cf in result.get("summary", {}).get("critical_findings", [])
            ]

            summary = ECGAnalysisSummary(
                overall_status=result.get("summary", {}).get("overall_status", "normal"),
                total_abnormal=result.get("summary", {}).get("total_abnormal", 0),
                total_borderline=result.get("summary", {}).get("total_borderline", 0),
                critical_findings=critical_findings
            )

            full_response = FullECGAnalysisResponse(
                success=result.get("success", False),
                patient_id=result.get("patient_id", patient_id),
                ecg_filename=result.get("ecg_filename", filename),
                file_format_info=ECGFileFormatInfo(
                    original_format=batch_format_info["original_format"],
                    original_encoding=batch_format_info["original_encoding"],
                    conversions_applied=batch_format_info["conversions_applied"],
                    conversion_notes=batch_format_info["conversion_notes"]
                ),
                models_executed=result.get("models_executed", []),
                results=model_results,
                summary=summary,
                warnings=result.get("warnings", []),
                processing_time_ms=result.get("processing_time_ms", 0),
                error=result.get("error")
            )

            results.append(BatchECGResult(
                index=idx,
                filename=filename,
                patient_id=patient_id,
                success=result.get("success", False),
                result=full_response,
                error=result.get("error")
            ))

            if result.get("success"):
                successful += 1
            else:
                failed += 1

        except Exception as e:
            logger.error(f"Batch processing failed for {filename}: {e}", exc_info=True)
            results.append(BatchECGResult(
                index=idx,
                filename=filename,
                patient_id=patient_id,
                success=False,
                error=str(e)
            ))
            failed += 1

    total_time = int((time.time() - start_time) * 1000)

    return BatchECGAnalysisResponse(
        success=failed == 0,
        total_files=len(files),
        successful=successful,
        failed=failed,
        results=results,
        total_processing_time_ms=total_time,
        models_used=selected_models if "all" not in selected_models else list(AVAILABLE_MODELS.keys())
    )


@router.get("/ecg/parse-existing")
async def parse_existing_results():
    """
    Debug endpoint: Parse existing output files without running Docker.
    """
    from .ecg_analysis import parse_probabilities_csv, AVAILABLE_MODELS
    import glob
    import time

    start_time = time.time()
    workspace_path = settings.WORKSPACE_PATH
    output_dir = os.path.join(workspace_path, "outputs")

    logger.info(f"Parsing existing files from: {output_dir}")

    # Find all probability files
    prob_files = glob.glob(os.path.join(output_dir, "*probabilities*.csv"))
    logger.info(f"Found probability files: {prob_files}")

    if not prob_files:
        return {"error": "No probability files found", "output_dir": output_dir}

    # Parse the most recent file
    prob_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    prob_path = prob_files[0]

    logger.info(f"Parsing: {prob_path}")
    parsed = parse_probabilities_csv(prob_path)

    # Build response
    diagnoses = []
    for diag in parsed.get("diagnoses", []):
        diagnoses.append(ECGDiagnosisResult(
            name=diag["name"],
            probability=diag["probability"],
            threshold=diag["threshold"],
            status=diag["status"],
            category=diag["category"]
        ))

    by_category = {}
    for cat, cat_diags in parsed.get("by_category", {}).items():
        by_category[cat] = [
            ECGDiagnosisResult(
                name=d["name"],
                probability=d["probability"],
                threshold=d["threshold"],
                status=d["status"],
                category=d["category"]
            ) for d in cat_diags
        ]

    model_results = {
        "efficientnet_77": ECGModelResult(
            model_id="efficientnet_77",
            model_name="77 Classes ECG (EfficientNet)",
            model_type="multi_label",
            architecture="efficientnet",
            success=True,
            diagnoses=diagnoses,
            by_category=by_category
        )
    }

    # Calculate summary
    abnormal_count = parsed["summary"]["abnormal"]
    borderline_count = parsed["summary"]["borderline"]

    critical_findings = [
        ECGCriticalFinding(
            diagnosis=d["name"],
            probability=d["probability"],
            model="efficientnet_77"
        ) for d in parsed["diagnoses"] if d["status"] == "abnormal"
    ]

    overall_status = "abnormal" if abnormal_count > 0 else ("borderline" if borderline_count > 0 else "normal")

    return FullECGAnalysisResponse(
        success=True,
        patient_id="TEST_EXISTING",
        ecg_filename=os.path.basename(prob_path),
        models_executed=["efficientnet_77"],
        results=model_results,
        summary=ECGAnalysisSummary(
            overall_status=overall_status,
            total_abnormal=abnormal_count,
            total_borderline=borderline_count,
            critical_findings=critical_findings
        ),
        warnings=[],
        processing_time_ms=int((time.time() - start_time) * 1000)
    )


async def _get_signal_from_base64(base64_path: str) -> dict:
    """
    Read ECG signal from preprocessed base64 file.
    The base64 file contains 12 leads × 2500 samples = 30000 float32 values.
    """
    import base64
    import numpy as np

    logger.info(f"Reading preprocessed ECG from: {base64_path}")

    try:
        with open(base64_path, 'r') as f:
            b64_data = f.read()

        # Decode base64 to numpy array
        data = np.frombuffer(base64.b64decode(b64_data), dtype=np.float32)

        # Standard 12-lead names
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        # Reshape to 12 leads
        if len(data) == 30000:
            # 12 leads × 2500 samples
            samples_per_lead = 2500
            data = data.reshape(12, samples_per_lead)
        elif len(data) % 12 == 0:
            samples_per_lead = len(data) // 12
            data = data.reshape(12, samples_per_lead)
        else:
            return {"error": f"Invalid data size: {len(data)} (not divisible by 12)", "success": False}

        # The preprocessed data is already normalized, but we need to scale for display
        # Typical ECG range is -2mV to +2mV, the normalized data is roughly in [-1, 1]
        leads_data = {}
        for i, name in enumerate(lead_names):
            # Scale to approximate mV range for display
            leads_data[name] = (data[i] * 1.5).tolist()  # Scale factor for reasonable display

        # Assume 500Hz sample rate (standard for preprocessing)
        sample_rate = 500
        duration_seconds = samples_per_lead / sample_rate

        logger.info(f"Loaded {len(lead_names)} leads, {samples_per_lead} samples at {sample_rate}Hz ({duration_seconds:.2f}s)")

        return {
            "success": True,
            "filename": os.path.basename(base64_path),
            "leads": lead_names,
            "samples_per_lead": samples_per_lead,
            "original_samples": samples_per_lead,
            "sample_rate": sample_rate,
            "duration_seconds": duration_seconds,
            "data": leads_data,
            "source": "preprocessed"
        }

    except Exception as e:
        logger.error(f"Error reading base64 ECG: {e}", exc_info=True)
        return {"error": str(e), "success": False}


def _calculate_heart_rate(leads_data: dict, sample_rate: float) -> int | None:
    """Calculate heart rate from R-peaks in lead II (or I as fallback)."""
    hr_lead = 'II' if 'II' in leads_data else ('I' if 'I' in leads_data else None)
    if not hr_lead or sample_rate <= 0 or len(leads_data.get(hr_lead, [])) <= sample_rate:
        return None
    try:
        import numpy as np
        signal = np.array(leads_data[hr_lead])
        min_distance = int(sample_rate * 0.4)  # Min 0.4s between beats
        threshold = np.mean(signal) + 0.5 * np.std(signal)
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        if len(peaks) >= 2:
            rr_intervals = np.diff(peaks) / sample_rate
            mean_rr = np.mean(rr_intervals)
            hr = round(60.0 / mean_rr)
            logger.info(f"Heart rate from {hr_lead}: {hr} bpm ({len(peaks)} R-peaks)")
            return hr
    except Exception as e:
        logger.warning(f"Heart rate calculation failed: {e}")
    return None


async def _parse_xml_waveform(xml_path: str, filename: str) -> dict:
    """
    Parse ECG waveform data from XML file.
    """
    import base64
    import xml.etree.ElementTree as ET
    import struct

    logger.info(f"Parsing ECG waveform from XML: {xml_path}")

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Standard 12-lead names in order
        standard_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        leads_data = {}
        sample_rate = 500  # Default
        samples_per_lead = 0

        # Detect if this is CardiologyXML format
        root_tag = root.tag.split('}')[-1] if '}' in root.tag else root.tag
        is_cardiology_xml = root_tag.lower() == 'cardiologyxml'
        is_hl7_aecg = root_tag.lower() == 'annotatedecg' or 'urn:hl7-org:v3' in root.tag
        is_philips = root_tag.lower() == 'restingecgdata' or 'medical.philips.com' in root.tag

        # Try to get sample rate from XML
        for sr_elem in root.iter():
            tag = sr_elem.tag.split('}')[-1] if '}' in sr_elem.tag else sr_elem.tag
            if tag in ('SampleRate', 'SamplingRate'):
                try:
                    sample_rate = int(sr_elem.text)
                except:
                    pass
                break

        if is_cardiology_xml:
            # CardiologyXML: <LeadData lead="I"><WaveformData>...</WaveformData></LeadData>
            for lead_elem in root.iter('LeadData'):
                lead_name = lead_elem.get('lead', '')
                if not lead_name:
                    continue

                wf_elem = lead_elem.find('WaveformData')
                if wf_elem is None or not wf_elem.text:
                    continue

                try:
                    raw_bytes = base64.b64decode(''.join(wf_elem.text.split()))
                    num_samples = len(raw_bytes) // 2
                    samples = struct.unpack(f'<{num_samples}h', raw_bytes)
                    # CardiologyXML uses raw ADC values, convert to mV
                    # Assume 4.88 uV/bit (standard GE MUSE resolution)
                    mv_data = [s * 4.88 / 1000.0 for s in samples]

                    if lead_name in standard_leads:
                        leads_data[lead_name] = mv_data
                        samples_per_lead = max(samples_per_lead, len(mv_data))
                    else:
                        leads_data[lead_name] = mv_data
                except Exception as e:
                    logger.warning(f"Failed to parse CardiologyXML lead {lead_name}: {e}")

            if leads_data:
                available_leads = [l for l in standard_leads if l in leads_data]
                duration_seconds = samples_per_lead / sample_rate if sample_rate > 0 else 0
                logger.info(f"Parsed CardiologyXML: {len(leads_data)} leads, {samples_per_lead} samples at {sample_rate}Hz")
                result = {
                    "success": True,
                    "filename": filename,
                    "leads": available_leads,
                    "samples_per_lead": samples_per_lead,
                    "original_samples": samples_per_lead,
                    "sample_rate": sample_rate,
                    "duration_seconds": duration_seconds,
                    "data": leads_data,
                    "source": "xml"
                }
                heart_rate = _calculate_heart_rate(leads_data, sample_rate)
                if heart_rate is not None:
                    result["heart_rate"] = heart_rate
                return result
            return {"error": "No waveform data found in CardiologyXML", "success": False}

        if is_hl7_aecg:
            # HL7 aECG: <sequence> with <code code="MDC_ECG_LEAD_X"/> and <digits>
            ns = ''
            if '}' in root.tag:
                ns = root.tag.split('}')[0] + '}'

            for sequence in root.iter(f'{ns}sequence'):
                code_elem = sequence.find(f'{ns}code')
                if code_elem is None:
                    continue
                code_val = code_elem.get('code', '')

                # Extract sample rate from TIME_ABSOLUTE
                if code_val == 'TIME_ABSOLUTE':
                    value_elem = sequence.find(f'{ns}value')
                    if value_elem is not None:
                        inc_elem = value_elem.find(f'{ns}increment')
                        if inc_elem is not None:
                            try:
                                inc_val = float(inc_elem.get('value', '0.002'))
                                inc_unit = inc_elem.get('unit', 's').lower()
                                if inc_unit == 'ms':
                                    inc_val = inc_val / 1000.0
                                elif inc_unit == 'us':
                                    inc_val = inc_val / 1_000_000.0
                                if inc_val > 0:
                                    sample_rate = int(round(1.0 / inc_val))
                            except (ValueError, ZeroDivisionError):
                                pass
                    continue

                if not code_val.startswith('MDC_ECG_LEAD_'):
                    continue

                lead_name = code_val.replace('MDC_ECG_LEAD_', '')
                value_elem = sequence.find(f'{ns}value')
                if value_elem is None:
                    continue

                scale_val = 1.0
                scale_elem = value_elem.find(f'{ns}scale')
                if scale_elem is not None:
                    try:
                        scale_val = float(scale_elem.get('value', '1.0'))
                    except ValueError:
                        pass

                origin_val = 0.0
                origin_elem = value_elem.find(f'{ns}origin')
                if origin_elem is not None:
                    try:
                        origin_val = float(origin_elem.get('value', '0'))
                    except ValueError:
                        pass

                digits_elem = value_elem.find(f'{ns}digits')
                if digits_elem is None or not digits_elem.text:
                    continue

                try:
                    raw_digits = [int(d) for d in digits_elem.text.strip().split()]
                    # Convert to mV: (digit * scale + origin) uV -> mV
                    mv_data = [(d * scale_val + origin_val) / 1000.0 for d in raw_digits]
                    if lead_name in standard_leads:
                        leads_data[lead_name] = mv_data
                        samples_per_lead = max(samples_per_lead, len(mv_data))
                    else:
                        leads_data[lead_name] = mv_data
                except Exception as e:
                    logger.warning(f"Failed to parse HL7 aECG lead {lead_name}: {e}")

            if leads_data:
                available_leads = [l for l in standard_leads if l in leads_data]
                duration_seconds = samples_per_lead / sample_rate if sample_rate > 0 else 0
                logger.info(f"Parsed HL7 aECG: {len(leads_data)} leads, {samples_per_lead} samples at {sample_rate}Hz")
                result = {
                    "success": True,
                    "filename": filename,
                    "leads": available_leads,
                    "samples_per_lead": samples_per_lead,
                    "original_samples": samples_per_lead,
                    "sample_rate": sample_rate,
                    "duration_seconds": duration_seconds,
                    "data": leads_data,
                    "source": "xml"
                }
                heart_rate = _calculate_heart_rate(leads_data, sample_rate)
                if heart_rate is not None:
                    result["heart_rate"] = heart_rate
                return result
            return {"error": "No waveform data found in HL7 aECG", "success": False}

        if is_philips:
            # Philips PageWriter: <repbeats><repbeat leadname="I"><waveform duration="1200">base64</waveform></repbeat>
            ns = ''
            if '}' in root.tag:
                ns = root.tag.split('}')[0] + '}'

            # Get sample rate and resolution from repbeats attributes
            repbeats_elem = root.find(f'.//{ns}repbeats')
            philips_sr = 1000
            philips_res = 1.0
            if repbeats_elem is not None:
                try:
                    philips_sr = int(repbeats_elem.get('samplespersec', '1000'))
                except ValueError:
                    pass
                try:
                    philips_res = float(repbeats_elem.get('resolution', '1.0'))
                except ValueError:
                    pass

            for repbeat in root.iter(f'{ns}repbeat'):
                lead_name = repbeat.get('leadname', '')
                if not lead_name:
                    continue
                wf_elem = repbeat.find(f'{ns}waveform')
                if wf_elem is None or not wf_elem.text:
                    continue
                try:
                    raw_bytes = base64.b64decode(''.join(wf_elem.text.split()))
                    num_samples = len(raw_bytes) // 2
                    samples = struct.unpack(f'<{num_samples}h', raw_bytes)
                    # Convert to mV: sample * resolution (uV) / 1000
                    mv_data = [s * philips_res / 1000.0 for s in samples]
                    if lead_name in standard_leads:
                        leads_data[lead_name] = mv_data
                        samples_per_lead = max(samples_per_lead, len(mv_data))
                    else:
                        leads_data[lead_name] = mv_data
                except Exception as e:
                    logger.warning(f"Failed to parse Philips repbeat lead {lead_name}: {e}")

            if leads_data:
                sample_rate = philips_sr
                available_leads = [l for l in standard_leads if l in leads_data]
                duration_seconds = samples_per_lead / sample_rate if sample_rate > 0 else 0
                logger.info(f"Parsed Philips repbeats: {len(leads_data)} leads, {samples_per_lead} samples at {sample_rate}Hz")
                result = {
                    "success": True,
                    "filename": filename,
                    "leads": available_leads,
                    "samples_per_lead": samples_per_lead,
                    "original_samples": samples_per_lead,
                    "sample_rate": sample_rate,
                    "duration_seconds": duration_seconds,
                    "data": leads_data,
                    "source": "xml"
                }
                heart_rate = _calculate_heart_rate(leads_data, sample_rate)
                if heart_rate is not None:
                    result["heart_rate"] = heart_rate
                return result
            return {"error": "No waveform data found in Philips XML", "success": False}

        # GE MUSE format: Find Waveform elements - prefer Rhythm (longest) over Median
        waveform_candidates = []
        for elem in root.iter():
            if 'Waveform' in elem.tag:
                has_leads = False
                for ld in elem.iter():
                    if 'LeadData' in ld.tag or ld.tag.endswith('LeadData'):
                        has_leads = True
                        break
                if has_leads:
                    # Check WaveformType
                    wf_type = None
                    for child in elem:
                        tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                        if tag_name == 'WaveformType':
                            wf_type = child.text
                            break
                    waveform_candidates.append((elem, wf_type))

        waveform_elem = None
        if waveform_candidates:
            # Prefer Rhythm section; fallback to last (longest) Waveform
            for wf, wf_type in waveform_candidates:
                if wf_type and 'Rhythm' in wf_type:
                    waveform_elem = wf
                    break
            if waveform_elem is None:
                waveform_elem = waveform_candidates[-1][0]

        if waveform_elem is None:
            return {"error": "No Waveform element found in XML", "success": False}

        # Parse each lead
        for lead_data in waveform_elem.iter():
            if not ('LeadData' in lead_data.tag or lead_data.tag.endswith('LeadData')):
                continue

            lead_id = None
            waveform_data = None
            amplitude_units = 1.0

            for child in lead_data:
                tag_name = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if tag_name == 'LeadID':
                    lead_id = child.text
                elif tag_name == 'WaveFormData':
                    waveform_data = child.text
                elif tag_name == 'LeadAmplitudeUnitsPerBit':
                    try:
                        amplitude_units = float(child.text)
                    except:
                        pass

            if lead_id and waveform_data:
                try:
                    raw_bytes = base64.b64decode(waveform_data)
                    num_samples = len(raw_bytes) // 2
                    samples = struct.unpack(f'<{num_samples}h', raw_bytes)
                    mv_data = [s * amplitude_units / 1000.0 for s in samples]

                    normalized_lead = lead_id.replace(' ', '').upper()
                    if normalized_lead in standard_leads:
                        leads_data[normalized_lead] = mv_data
                        samples_per_lead = max(samples_per_lead, len(mv_data))
                    elif normalized_lead == 'AVR':
                        leads_data['aVR'] = mv_data
                    elif normalized_lead == 'AVL':
                        leads_data['aVL'] = mv_data
                    elif normalized_lead == 'AVF':
                        leads_data['aVF'] = mv_data
                    else:
                        leads_data[lead_id] = mv_data

                except Exception as e:
                    logger.warning(f"Failed to parse lead {lead_id}: {e}")

        if not leads_data:
            return {"error": "No waveform data found in XML", "success": False}

        # Calculate derived leads if missing (III, aVR, aVL, aVF from I and II)
        if 'I' in leads_data and 'II' in leads_data:
            lead_i = leads_data['I']
            lead_ii = leads_data['II']
            n_samples = min(len(lead_i), len(lead_ii))

            if 'III' not in leads_data:
                leads_data['III'] = [lead_ii[i] - lead_i[i] for i in range(n_samples)]
                logger.info("Calculated derived lead: III")

            if 'aVR' not in leads_data:
                leads_data['aVR'] = [-(lead_i[i] + lead_ii[i]) / 2 for i in range(n_samples)]
                logger.info("Calculated derived lead: aVR")

            if 'aVL' not in leads_data:
                leads_data['aVL'] = [lead_i[i] - lead_ii[i] / 2 for i in range(n_samples)]
                logger.info("Calculated derived lead: aVL")

            if 'aVF' not in leads_data:
                leads_data['aVF'] = [lead_ii[i] - lead_i[i] / 2 for i in range(n_samples)]
                logger.info("Calculated derived lead: aVF")

        available_leads = [l for l in standard_leads if l in leads_data]
        duration_seconds = samples_per_lead / sample_rate if sample_rate > 0 else 0

        logger.info(f"Parsed {len(leads_data)} leads, {samples_per_lead} samples at {sample_rate}Hz ({duration_seconds:.2f}s)")

        result = {
            "success": True,
            "filename": filename,
            "leads": available_leads,
            "samples_per_lead": samples_per_lead,
            "original_samples": samples_per_lead,
            "sample_rate": sample_rate,
            "duration_seconds": duration_seconds,
            "data": leads_data,
            "source": "xml"
        }
        heart_rate = _calculate_heart_rate(leads_data, sample_rate)
        if heart_rate is not None:
            result["heart_rate"] = heart_rate
        return result

    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
        return {"error": f"Invalid XML format: {e}", "success": False}
    except Exception as e:
        logger.error(f"Error reading ECG XML: {e}", exc_info=True)
        return {"error": str(e), "success": False}


@router.get("/ecg/signal-data")
async def get_ecg_signal_data(filename: str = "ecg.xml"):
    """
    Get ECG signal data for visualization.
    First tries to read from XML file (original backup), then falls back to preprocessed base64.
    Returns 12-lead ECG data as JSON.
    """
    import glob

    workspace_path = settings.WORKSPACE_PATH
    ecg_signals_dir = os.path.join(workspace_path, "ecg_signals")
    preprocessing_dir = os.path.join(workspace_path, "preprocessing")

    logger.info(f"Looking for ECG signal data for: {filename}")
    logger.info(f"ECG signals dir: {ecg_signals_dir} (exists: {os.path.isdir(ecg_signals_dir)})")

    # Try to find the SPECIFIC original backup file matching requested filename
    specific_backup = os.path.join(ecg_signals_dir, f"original_{filename}")
    logger.info(f"Looking for specific backup: {specific_backup}")
    if os.path.exists(specific_backup):
        result = await _parse_xml_waveform(specific_backup, filename)
        if result.get("success"):
            return result
        logger.warning(f"Backup parse failed ({result.get('error', 'unknown')}), trying converted file")

    # Try specified filename (exact match) - may be the converted GE MUSE version
    xml_path = os.path.join(ecg_signals_dir, filename)
    logger.info(f"Checking specified file: {xml_path} (exists: {os.path.exists(xml_path)})")
    if os.path.exists(xml_path):
        result = await _parse_xml_waveform(xml_path, filename)
        if result.get("success"):
            return result

    # Fallback: find the most recent original backup file
    original_xml_files = sorted(
        glob.glob(os.path.join(ecg_signals_dir, "original_*.xml")),
        key=os.path.getmtime,
        reverse=True  # Most recent first
    )
    logger.info(f"Original backup files found: {original_xml_files}")
    if original_xml_files:
        xml_path = original_xml_files[0]
        return await _parse_xml_waveform(xml_path, os.path.basename(xml_path))

    # Try to find any XML file (excluding _12lead files to get original)
    all_xml_files = glob.glob(os.path.join(ecg_signals_dir, "*.xml"))
    xml_files = [f for f in all_xml_files if not f.endswith("_12lead.xml")]
    logger.info(f"All XML files found: {all_xml_files}")
    if xml_files:
        xml_path = sorted(xml_files, key=os.path.getmtime, reverse=True)[0]
        return await _parse_xml_waveform(xml_path, os.path.basename(xml_path))

    # Try _12lead files as fallback
    if all_xml_files:
        xml_path = sorted(all_xml_files, key=os.path.getmtime, reverse=True)[0]
        return await _parse_xml_waveform(xml_path, os.path.basename(xml_path))

    # Fallback to preprocessed base64 file
    base64_files = glob.glob(os.path.join(preprocessing_dir, "*.base64"))
    logger.info(f"Base64 files found: {base64_files}")
    if base64_files:
        return await _get_signal_from_base64(base64_files[0])

    return {"error": f"File not found: {filename}", "success": False, "searched_dir": ecg_signals_dir}