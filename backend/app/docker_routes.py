"""
API routes for Docker control and system diagnostics.

Provides endpoints for:
- Starting/stopping AI Engine container
- Checking Docker status
- Running system diagnostics
"""
import logging
import os
from typing import Optional, Dict, Any, List
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field
import shutil
import uuid
from datetime import datetime

from .docker_control import (
    get_docker_status,
    start_ai_engine,
    stop_ai_engine,
    get_container_logs,
    run_diagnostics,
    run_preprocessing,
    run_inference,
    run_full_pipeline,
    DockerStatus,
    CONTAINER_NAME,
    DEFAULT_IMAGE,
    AI_ENGINE_PORT
)
from .settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/docker", tags=["docker-control"])


# ============================================
# Response Models
# ============================================

class DockerStatusResponse(BaseModel):
    """Docker status response."""
    docker_installed: bool
    docker_running: bool
    container_exists: bool
    container_running: bool
    container_id: Optional[str] = None
    gpu_available: bool
    error: Optional[str] = None


class StartEngineRequest(BaseModel):
    """Request to start AI Engine."""
    image: str = Field(default=DEFAULT_IMAGE, description="Docker image to use")
    workspace_path: Optional[str] = Field(default=None, description="Path to mount as /workspace")


class StartEngineResponse(BaseModel):
    """Response from starting AI Engine."""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    container_id: Optional[str] = None
    port: Optional[int] = None
    gpu_enabled: Optional[bool] = None
    health: Optional[Dict[str, Any]] = None
    warning: Optional[str] = None


class StopEngineResponse(BaseModel):
    """Response from stopping AI Engine."""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None


class ContainerLogsResponse(BaseModel):
    """Container logs response."""
    success: bool
    logs: Optional[str] = None
    error: Optional[str] = None


class DiagnosticTestResult(BaseModel):
    """Single diagnostic test result."""
    name: str
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: float


class DiagnosticsResponse(BaseModel):
    """Full diagnostics response."""
    overall_status: str
    tests: List[DiagnosticTestResult]
    timestamp: str
    platform_info: Dict[str, str]


# ============================================
# Endpoints
# ============================================

@router.get("/status", response_model=DockerStatusResponse)
async def docker_status() -> DockerStatusResponse:
    """
    Get Docker and AI Engine container status.

    Returns current state of Docker daemon and the AI Engine container.
    """
    try:
        logger.info("Checking Docker status")
        status = await get_docker_status()

        return DockerStatusResponse(
            docker_installed=status.docker_installed,
            docker_running=status.docker_running,
            container_exists=status.container_exists,
            container_running=status.container_running,
            container_id=status.container_id,
            gpu_available=status.gpu_available,
            error=status.error
        )
    except Exception as e:
        logger.error(f"Error checking Docker status: {e}", exc_info=True)
        return DockerStatusResponse(
            docker_installed=False,
            docker_running=False,
            container_exists=False,
            container_running=False,
            gpu_available=False,
            error=f"Error checking status: {str(e)}"
        )


@router.post("/start", response_model=StartEngineResponse)
async def start_engine(request: StartEngineRequest = None) -> StartEngineResponse:
    """
    Start the AI Engine Docker container.

    Starts the DeepECG AI Engine container with GPU support if available.
    Uses the command: docker run --gpus all -p 8001:8001 -v {workspace}:/workspace deepecg
    """
    try:
        # Handle None or empty request
        image = DEFAULT_IMAGE
        workspace = None

        if request:
            if request.image:
                image = request.image
            if request.workspace_path:
                workspace = request.workspace_path

        logger.info(f"Starting AI Engine with image: {image}")

        # Note: workspace is optional - the AI Engine has all files built into the image
        # Only pass workspace if user explicitly provides it for data access

        result = await start_ai_engine(
            image=image,
            workspace_path=workspace
        )

        return StartEngineResponse(**result)

    except Exception as e:
        logger.error(f"Error starting AI Engine: {e}", exc_info=True)
        return StartEngineResponse(
            success=False,
            error=f"Internal error: {str(e)}"
        )


@router.post("/stop", response_model=StopEngineResponse)
async def stop_engine() -> StopEngineResponse:
    """
    Stop the AI Engine Docker container.

    Stops and removes the running AI Engine container.
    """
    logger.info("Stopping AI Engine")
    result = await stop_ai_engine()

    return StopEngineResponse(**result)


@router.get("/logs", response_model=ContainerLogsResponse)
async def container_logs(
    lines: int = Query(default=100, ge=1, le=1000, description="Number of log lines to retrieve")
) -> ContainerLogsResponse:
    """
    Get recent AI Engine container logs.

    Returns the last N lines of container logs.
    """
    result = await get_container_logs(lines=lines)

    return ContainerLogsResponse(**result)


@router.get("/diagnostics", response_model=DiagnosticsResponse)
async def system_diagnostics() -> DiagnosticsResponse:
    """
    Run comprehensive system diagnostics.

    Tests Docker, GPU, AI Engine, and other components.
    Returns detailed results for each test.
    """
    logger.info("Running system diagnostics")
    diagnostics = await run_diagnostics()

    return DiagnosticsResponse(
        overall_status=diagnostics.overall_status,
        tests=[
            DiagnosticTestResult(
                name=t.name,
                status=t.status,
                message=t.message,
                details=t.details,
                duration_ms=t.duration_ms
            )
            for t in diagnostics.tests
        ],
        timestamp=diagnostics.timestamp,
        platform_info=diagnostics.platform_info
    )


@router.get("/info")
async def docker_info() -> Dict[str, Any]:
    """
    Get Docker control configuration info.

    Returns container name, image, port, and other settings.
    """
    return {
        "container_name": CONTAINER_NAME,
        "default_image": DEFAULT_IMAGE,
        "port": AI_ENGINE_PORT,
        "health_endpoint": f"http://localhost:{AI_ENGINE_PORT}/health"
    }


# ============================================
# Preprocessing & Inference Models
# ============================================

class PreprocessingRequest(BaseModel):
    """Request to run ECG preprocessing."""
    data_path: str = Field(..., description="Path to input CSV file (inside /data/)")
    output_folder: str = Field(default="/data/outputs", description="Output folder path")
    ecg_signals_path: str = Field(default="/data/ecg_signals", description="ECG signals folder")
    preprocessing_folder: str = Field(default="/data/preprocessing", description="Preprocessing folder")
    batch_size: int = Field(default=1, ge=1, description="Batch size")
    n_workers: int = Field(default=1, ge=1, description="Number of workers")
    device: str = Field(default="cpu", description="Device (cpu or cuda)")


class PreprocessingResponse(BaseModel):
    """Response from preprocessing."""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    output: Optional[str] = None


class InferenceRequest(BaseModel):
    """Request to run ECG inference."""
    data_path: str = Field(..., description="Path to input CSV file (inside /data/)")
    output_folder: str = Field(default="/data/outputs", description="Output folder path")
    ecg_signals_path: str = Field(default="/data/ecg_signals", description="ECG signals folder")
    preprocessing_folder: str = Field(default="/data/preprocessing", description="Preprocessing folder")
    batch_size: int = Field(default=1, ge=1, description="Batch size")
    device: str = Field(default="cuda", description="Device (cpu or cuda)")


class InferenceResponse(BaseModel):
    """Response from inference."""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    output: Optional[str] = None


# ============================================
# Preprocessing & Inference Endpoints
# ============================================

@router.post("/preprocessing", response_model=PreprocessingResponse)
async def preprocessing(request: PreprocessingRequest) -> PreprocessingResponse:
    """
    Run ECG preprocessing via docker exec.

    Executes the preprocessing pipeline on the specified input file.
    The container must be running with /data volume mounted.
    """
    logger.info(f"Running preprocessing: {request.data_path}")

    result = await run_preprocessing(
        data_path=request.data_path,
        output_folder=request.output_folder,
        ecg_signals_path=request.ecg_signals_path,
        preprocessing_folder=request.preprocessing_folder,
        batch_size=request.batch_size,
        n_workers=request.n_workers,
        device=request.device
    )

    return PreprocessingResponse(**result)


@router.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest) -> InferenceResponse:
    """
    Run ECG inference via docker exec.

    Executes the inference pipeline on the specified input file.
    The container must be running with /data volume mounted.
    """
    logger.info(f"Running inference: {request.data_path}")

    result = await run_inference(
        data_path=request.data_path,
        output_folder=request.output_folder,
        ecg_signals_path=request.ecg_signals_path,
        preprocessing_folder=request.preprocessing_folder,
        batch_size=request.batch_size,
        device=request.device
    )

    return InferenceResponse(**result)


# ============================================
# ECG File Upload & Preparation
# ============================================

class UploadECGResponse(BaseModel):
    """Response from ECG file upload."""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    ecg_filename: Optional[str] = None
    csv_filename: Optional[str] = None
    patient_id: Optional[str] = None


@router.post("/upload-ecg", response_model=UploadECGResponse)
async def upload_ecg(
    file: UploadFile = File(...),
    workspace_path: str = Form(...),
    patient_id: Optional[str] = Form(None)
) -> UploadECGResponse:
    """
    Upload an ECG file (XML) and prepare it for preprocessing.

    1. Saves the file to {workspace_path}/ecg_signals/
    2. Generates a CSV file in {workspace_path}/inputs/

    The CSV will be ready for preprocessing.
    """
    try:
        # Validate file extension
        if not file.filename:
            return UploadECGResponse(success=False, error="No filename provided")

        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.xml', '.npy']:
            return UploadECGResponse(
                success=False,
                error=f"Invalid file type: {file_ext}. Supported: .xml, .npy"
            )

        # Generate patient ID if not provided
        if not patient_id:
            patient_id = "TEST001"

        # Create directories if they don't exist
        ecg_signals_dir = os.path.join(workspace_path, "ecg_signals")
        inputs_dir = os.path.join(workspace_path, "inputs")
        os.makedirs(ecg_signals_dir, exist_ok=True)
        os.makedirs(inputs_dir, exist_ok=True)

        # Save ECG file - keep exact filename
        ecg_filename = file.filename
        ecg_filepath = os.path.join(ecg_signals_dir, ecg_filename)

        with open(ecg_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"ECG file saved: {ecg_filepath}")

        # Generate CSV file - use simple name
        csv_filename = "ecg_test.csv"
        csv_filepath = os.path.join(inputs_dir, csv_filename)

        # Also copy the file to create a second entry (workaround for single-file bug)
        ecg_filename_base = os.path.splitext(ecg_filename)[0]
        ecg_filename_ext = os.path.splitext(ecg_filename)[1]
        ecg_filename_copy = f"{ecg_filename_base}_copy{ecg_filename_ext}"
        ecg_filepath_copy = os.path.join(ecg_signals_dir, ecg_filename_copy)
        shutil.copy2(ecg_filepath, ecg_filepath_copy)

        # CSV content with 2 entries (workaround for preprocessing bug with single file)
        csv_content = "patient_id,ecg_id,efficientnet_ecg_file_name,wcr_ecg_file_name,bert_ecg_file_name\n"
        csv_content += f"{patient_id},ECG001,{ecg_filename},{ecg_filename},{ecg_filename}\n"
        csv_content += f"{patient_id}_2,ECG002,{ecg_filename_copy},{ecg_filename_copy},{ecg_filename_copy}\n"

        with open(csv_filepath, "w") as f:
            f.write(csv_content)

        logger.info(f"CSV file generated: {csv_filepath}")

        return UploadECGResponse(
            success=True,
            message=f"ECG file uploaded and CSV generated",
            ecg_filename=ecg_filename,
            csv_filename=csv_filename,
            patient_id=patient_id
        )

    except Exception as e:
        logger.error(f"Error uploading ECG file: {e}", exc_info=True)
        return UploadECGResponse(
            success=False,
            error=f"Upload failed: {str(e)}"
        )


class WorkspaceFilesResponse(BaseModel):
    """Response listing workspace files."""
    success: bool
    ecg_files: List[str] = []
    csv_files: List[str] = []
    preprocessing_files: List[str] = []
    output_files: List[str] = []
    error: Optional[str] = None


@router.get("/workspace-files", response_model=WorkspaceFilesResponse)
async def list_workspace_files(
    workspace_path: str = Query(..., description="Path to workspace folder")
) -> WorkspaceFilesResponse:
    """
    List files in the workspace directories.
    """
    try:
        ecg_files = []
        csv_files = []
        preprocessing_files = []
        output_files = []

        # List ECG signals
        ecg_dir = os.path.join(workspace_path, "ecg_signals")
        if os.path.exists(ecg_dir):
            ecg_files = [f for f in os.listdir(ecg_dir) if os.path.isfile(os.path.join(ecg_dir, f))]

        # List input CSVs
        inputs_dir = os.path.join(workspace_path, "inputs")
        if os.path.exists(inputs_dir):
            csv_files = [f for f in os.listdir(inputs_dir) if f.endswith('.csv')]

        # List preprocessing files
        preproc_dir = os.path.join(workspace_path, "preprocessing")
        if os.path.exists(preproc_dir):
            preprocessing_files = [f for f in os.listdir(preproc_dir) if os.path.isfile(os.path.join(preproc_dir, f))]

        # List output files
        output_dir = os.path.join(workspace_path, "outputs")
        if os.path.exists(output_dir):
            output_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]

        return WorkspaceFilesResponse(
            success=True,
            ecg_files=ecg_files,
            csv_files=csv_files,
            preprocessing_files=preprocessing_files,
            output_files=output_files
        )

    except Exception as e:
        logger.error(f"Error listing workspace files: {e}", exc_info=True)
        return WorkspaceFilesResponse(
            success=False,
            error=str(e)
        )


# ============================================
# Full Pipeline (Preprocessing + Analysis)
# ============================================

class FullPipelineResponse(BaseModel):
    """Response from full pipeline execution."""
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None
    step: Optional[str] = None
    preprocessing_output: Optional[str] = None
    analysis_output: Optional[str] = None
    result_files: List[str] = []
    output_folder: Optional[str] = None


@router.post("/full-pipeline", response_model=FullPipelineResponse)
async def full_pipeline(
    file: UploadFile = File(...),
    use_gpu: bool = Form(default=False),
    use_wcr: bool = Form(default=False),
    use_efficientnet: bool = Form(default=True),
    batch_size: int = Form(default=1),
    patient_id: Optional[str] = Form(default=None)
) -> FullPipelineResponse:
    """
    Run the full ECG analysis pipeline.

    1. Saves uploaded CSV to workspace/inputs/
    2. Runs preprocessing (XML -> base64)
    3. Runs analysis
    4. Returns results

    The CSV should reference ECG XML files that exist in workspace/ecg_signals/.
    """
    try:
        workspace_path = settings.WORKSPACE_PATH

        # Validate file
        if not file.filename:
            return FullPipelineResponse(success=False, error="No filename provided")

        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.csv']:
            return FullPipelineResponse(
                success=False,
                error=f"Invalid file type: {file_ext}. Expected: .csv"
            )

        # Create directories if needed
        inputs_dir = os.path.join(workspace_path, "inputs")
        outputs_dir = os.path.join(workspace_path, "outputs")
        preprocessing_dir = os.path.join(workspace_path, "preprocessing")
        os.makedirs(inputs_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(preprocessing_dir, exist_ok=True)

        # Save CSV file
        csv_filename = file.filename
        csv_filepath = os.path.join(inputs_dir, csv_filename)

        with open(csv_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"CSV file saved: {csv_filepath}")

        # Run full pipeline
        result = await run_full_pipeline(
            csv_filename=csv_filename,
            workspace_path=workspace_path,
            use_gpu=use_gpu,
            use_wcr=use_wcr,
            use_efficientnet=use_efficientnet,
            batch_size=batch_size,
            n_workers=1
        )

        return FullPipelineResponse(**result)

    except Exception as e:
        logger.error(f"Error running full pipeline: {e}", exc_info=True)
        return FullPipelineResponse(
            success=False,
            error=f"Pipeline failed: {str(e)}"
        )


@router.post("/analyze-ecg", response_model=FullPipelineResponse)
async def analyze_ecg_file(
    file: UploadFile = File(...),
    use_gpu: bool = Form(default=False),
    use_wcr: bool = Form(default=False),
    use_efficientnet: bool = Form(default=True),
    patient_id: Optional[str] = Form(default=None)
) -> FullPipelineResponse:
    """
    Upload a single ECG XML file and run full analysis.

    1. Saves XML to workspace/ecg_signals/
    2. Generates CSV in workspace/inputs/
    3. Runs preprocessing
    4. Runs analysis
    5. Returns results

    This is the simplest endpoint - just upload an XML and get results.
    """
    try:
        workspace_path = settings.WORKSPACE_PATH

        # Validate file
        if not file.filename:
            return FullPipelineResponse(success=False, error="No filename provided")

        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.xml', '.npy']:
            return FullPipelineResponse(
                success=False,
                error=f"Invalid file type: {file_ext}. Expected: .xml or .npy"
            )

        # Create directories
        ecg_signals_dir = os.path.join(workspace_path, "ecg_signals")
        inputs_dir = os.path.join(workspace_path, "inputs")
        outputs_dir = os.path.join(workspace_path, "outputs")
        preprocessing_dir = os.path.join(workspace_path, "preprocessing")
        os.makedirs(ecg_signals_dir, exist_ok=True)
        os.makedirs(inputs_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(preprocessing_dir, exist_ok=True)

        # Generate patient ID if not provided
        if not patient_id:
            patient_id = f"PAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save ECG file
        ecg_filename = file.filename
        ecg_filepath = os.path.join(ecg_signals_dir, ecg_filename)

        with open(ecg_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"ECG file saved: {ecg_filepath}")

        # Create a copy (workaround for single-file bug in HeartWise)
        ecg_filename_base = os.path.splitext(ecg_filename)[0]
        ecg_filename_ext = os.path.splitext(ecg_filename)[1]
        ecg_filename_copy = f"{ecg_filename_base}_copy{ecg_filename_ext}"
        ecg_filepath_copy = os.path.join(ecg_signals_dir, ecg_filename_copy)
        shutil.copy2(ecg_filepath, ecg_filepath_copy)

        # Generate CSV with HeartWise-compatible format
        csv_filename = f"ecg_{patient_id}.csv"
        csv_filepath = os.path.join(inputs_dir, csv_filename)

        # Use the format that works with your HeartWise setup
        csv_content = "patient_id,ecg_id,efficientnet_ecg_file_name,wcr_ecg_file_name,bert_ecg_file_name\n"
        csv_content += f"{patient_id},ECG001,{ecg_filename},{ecg_filename},{ecg_filename}\n"
        csv_content += f"{patient_id}_2,ECG002,{ecg_filename_copy},{ecg_filename_copy},{ecg_filename_copy}\n"

        with open(csv_filepath, "w") as f:
            f.write(csv_content)

        logger.info(f"CSV file generated: {csv_filepath}")

        # Run full pipeline
        result = await run_full_pipeline(
            csv_filename=csv_filename,
            workspace_path=workspace_path,
            use_gpu=use_gpu,
            use_wcr=use_wcr,
            use_efficientnet=use_efficientnet,
            batch_size=1,
            n_workers=1
        )

        return FullPipelineResponse(**result)

    except Exception as e:
        logger.error(f"Error analyzing ECG file: {e}", exc_info=True)
        return FullPipelineResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        )
