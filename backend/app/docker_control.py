"""
Docker control and system diagnostics module.

Provides functionality to:
- Start/stop AI Engine Docker container
- Check Docker status
- Run system diagnostics
"""
import asyncio
import logging
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .settings import settings

logger = logging.getLogger(__name__)

# Docker container configuration
CONTAINER_NAME = "deepecg-ai-engine"
DEFAULT_IMAGE = "deepecg-docker"  # Match user's actual image name
AI_ENGINE_PORT = 8001


@dataclass
class DockerStatus:
    """Docker daemon and container status."""
    docker_installed: bool = False
    docker_running: bool = False
    container_exists: bool = False
    container_running: bool = False
    container_id: Optional[str] = None
    gpu_available: bool = False
    error: Optional[str] = None


@dataclass
class DiagnosticResult:
    """System diagnostic result."""
    name: str
    status: str  # "pass", "fail", "warning", "skip"
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: float = 0


@dataclass
class SystemDiagnostics:
    """Complete system diagnostics."""
    overall_status: str  # "healthy", "degraded", "critical"
    tests: List[DiagnosticResult] = field(default_factory=list)
    timestamp: str = ""
    platform_info: Dict[str, str] = field(default_factory=dict)


def _run_command(cmd: List[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        # Build kwargs for subprocess.run
        kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
            "encoding": "utf-8",  # Force UTF-8 encoding for Docker output
            "errors": "replace",  # Replace invalid chars instead of crashing
        }

        # Add Windows-specific flag to hide console window
        if platform.system() == "Windows":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

        logger.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, **kwargs)
        logger.debug(f"Command result: code={result.returncode}, stdout={result.stdout[:100] if result.stdout else ''}")

        return result.returncode, result.stdout.strip() if result.stdout else "", result.stderr.strip() if result.stderr else ""
    except subprocess.TimeoutExpired:
        logger.warning(f"Command timed out: {' '.join(cmd)}")
        return -1, "", "Command timed out"
    except FileNotFoundError:
        logger.warning(f"Command not found: {cmd[0]}")
        return -1, "", f"Command not found: {cmd[0]}"
    except Exception as e:
        logger.error(f"Command error: {e}")
        return -1, "", str(e)


async def get_docker_status() -> DockerStatus:
    """Get current Docker and container status."""
    status = DockerStatus()

    try:
        # Check if Docker is installed
        code, _, _ = _run_command(["docker", "--version"], timeout=5)
        status.docker_installed = code == 0

        if not status.docker_installed:
            status.error = "Docker is not installed"
            return status

        # Check if Docker daemon is running
        code, _, stderr = _run_command(["docker", "info"], timeout=10)
        status.docker_running = code == 0

        if not status.docker_running:
            status.error = "Docker daemon is not running"
            return status

        # Check if container exists
        code, stdout, _ = _run_command(
            ["docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.ID}}"],
            timeout=10
        )
        if code == 0 and stdout:
            status.container_exists = True
            status.container_id = stdout.split('\n')[0]

        # Check if container is running
        code, stdout, _ = _run_command(
            ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Status}}"],
            timeout=10
        )
        if code == 0 and stdout and "Up" in stdout:
            status.container_running = True

        # Check GPU availability (quick check with nvidia-smi on host)
        code, _, _ = _run_command(["nvidia-smi", "-L"], timeout=5)
        status.gpu_available = code == 0

    except Exception as e:
        logger.error(f"Error checking Docker status: {e}", exc_info=True)
        status.error = f"Error: {str(e)}"

    return status


async def start_ai_engine(
    image: str = DEFAULT_IMAGE,
    workspace_path: Optional[str] = None,
    detached: bool = True
) -> Dict[str, Any]:
    """
    Start the AI Engine Docker container.

    Uses the command:
    docker run -d --gpus all -p 8001:8001 deepecg

    Args:
        image: Docker image to use
        workspace_path: Path to mount as /workspace
        detached: Run in detached mode (background)

    Returns:
        Dict with success status and message
    """
    logger.info(f"Starting AI Engine container with image: {image}")

    # Check Docker status first
    status = await get_docker_status()

    if not status.docker_installed:
        return {"success": False, "error": "Docker is not installed"}

    if not status.docker_running:
        return {"success": False, "error": "Docker daemon is not running. Please start Docker Desktop."}

    # Stop existing container if running
    if status.container_running:
        logger.info("Stopping existing container...")
        _run_command(["docker", "stop", CONTAINER_NAME], timeout=30)
        await asyncio.sleep(2)

    # Remove existing container if exists
    if status.container_exists:
        logger.info("Removing existing container...")
        _run_command(["docker", "rm", "-f", CONTAINER_NAME], timeout=10)

    # Build docker run command:
    # docker run --gpus all -p 8001:8001 deepecg
    # NOTE: Do NOT mount volume to /workspace as it would overwrite the app files!
    cmd = ["docker", "run", "-d", "--name", CONTAINER_NAME]

    # Always try with GPU (--gpus all)
    cmd.extend(["--gpus", "all"])

    # Port mapping (not used for batch processing, but keep for compatibility)
    # cmd.extend(["-p", f"{AI_ENGINE_PORT}:{AI_ENGINE_PORT}"])

    # Mount data directory to /data if workspace provided (not /workspace!)
    if workspace_path:
        cmd.extend(["-v", f"{workspace_path}:/data"])

        # Mount patched ecg_signal_processor.py if it exists
        # This fixes bugs with np.squeeze and hardcoded 12 leads
        import os
        patched_file = os.path.join(workspace_path, "ecg_signal_processor.py")
        if os.path.exists(patched_file):
            cmd.extend(["-v", f"{patched_file}:/app/utils/ecg_signal_processor.py"])
            logger.info(f"Mounting patched ecg_signal_processor.py from: {patched_file}")

    # Add image and keep-alive command (no server, just sleep for batch processing)
    cmd.append(image)
    cmd.extend(["sh", "-c", "while true; do sleep 3600; done"])

    logger.info(f"Docker command: {' '.join(cmd)}")

    # Run container
    code, stdout, stderr = _run_command(cmd, timeout=60)

    if code != 0:
        # If GPU fails, try without GPU
        if "gpu" in stderr.lower() or "nvidia" in stderr.lower():
            logger.warning("GPU failed, retrying without GPU...")
            cmd = ["docker", "run", "-d", "--name", CONTAINER_NAME]
            if workspace_path:
                cmd.extend(["-v", f"{workspace_path}:/data"])
                # Also mount patched file for CPU fallback
                import os
                patched_file = os.path.join(workspace_path, "ecg_signal_processor.py")
                if os.path.exists(patched_file):
                    cmd.extend(["-v", f"{patched_file}:/app/utils/ecg_signal_processor.py"])
            cmd.append(image)
            cmd.extend(["sh", "-c", "while true; do sleep 3600; done"])
            code, stdout, stderr = _run_command(cmd, timeout=60)

    if code == 0:
        container_id = stdout[:12] if stdout else "unknown"
        logger.info(f"Container started: {container_id}")

        # Wait for container to be ready
        await asyncio.sleep(5)

        # Check if container is still running
        check_code, check_out, _ = _run_command(
            ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Status}}"],
            timeout=5
        )

        if check_code == 0 and check_out and "Up" in check_out:
            # Container is running, check health
            for attempt in range(10):
                try:
                    import httpx
                    async with httpx.AsyncClient(timeout=5) as client:
                        response = await client.get(f"http://localhost:{AI_ENGINE_PORT}/health")
                        if response.status_code == 200:
                            logger.info("AI Engine API is responding")
                            return {
                                "success": True,
                                "message": "AI Engine started and healthy",
                                "container_id": container_id,
                                "port": AI_ENGINE_PORT,
                                "gpu_enabled": True
                            }
                except Exception as e:
                    logger.debug(f"Health check attempt {attempt + 1}/10: {e}")
                    await asyncio.sleep(3)

            return {
                "success": True,
                "message": "AI Engine started (initializing...)",
                "container_id": container_id,
                "port": AI_ENGINE_PORT,
                "warning": "API may need more time to load models"
            }
        else:
            # Container exited - get logs
            _, logs, _ = _run_command(["docker", "logs", "--tail", "20", CONTAINER_NAME], timeout=5)
            return {
                "success": False,
                "error": f"Container exited immediately. Logs:\n{logs}"
            }
    else:
        error_msg = stderr or "Unknown error"
        logger.error(f"Failed to start container: {error_msg}")
        return {
            "success": False,
            "error": f"Failed to start: {error_msg}"
        }


async def stop_ai_engine() -> Dict[str, Any]:
    """Stop the AI Engine Docker container."""
    logger.info("Stopping AI Engine container...")

    status = await get_docker_status()

    if not status.container_exists:
        return {"success": True, "message": "Container does not exist"}

    if not status.container_running:
        # Remove stopped container
        _run_command(["docker", "rm", CONTAINER_NAME], timeout=10)
        return {"success": True, "message": "Container was not running, removed"}

    # Stop container
    code, _, stderr = _run_command(["docker", "stop", CONTAINER_NAME], timeout=30)

    if code == 0:
        # Remove container
        _run_command(["docker", "rm", CONTAINER_NAME], timeout=10)
        logger.info("Container stopped and removed successfully")
        return {"success": True, "message": "AI Engine stopped successfully"}
    else:
        error_msg = stderr or "Unknown error"
        logger.error(f"Failed to stop container: {error_msg}")
        return {"success": False, "error": f"Failed to stop container: {error_msg}"}


async def get_container_logs(lines: int = 100) -> Dict[str, Any]:
    """Get recent container logs."""
    code, stdout, stderr = _run_command(
        ["docker", "logs", "--tail", str(lines), CONTAINER_NAME],
        timeout=10
    )

    if code == 0:
        return {"success": True, "logs": stdout or stderr}
    else:
        return {"success": False, "error": stderr or "Failed to get logs"}


async def run_preprocessing(
    data_path: str,
    output_folder: str = "/data/outputs",
    ecg_signals_path: str = "/data/ecg_signals",
    preprocessing_folder: str = "/data/preprocessing",
    batch_size: int = 1,
    n_workers: int = 1,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run ECG preprocessing via docker exec.

    Command:
    docker exec deepecg-ai-engine python /app/main.py \
        --mode preprocessing \
        --data_path /data/inputs/file.csv \
        --output_folder /data/outputs \
        --ecg_signals_path /data/ecg_signals \
        --preprocessing_folder /data/preprocessing \
        --batch_size 1 \
        --preprocessing_n_workers 1 \
        --diagnosis_classifier_device cpu \
        --signal_processing_device cpu \
        --hugging_face_api_key_path /data/api_key.json \
        --use_wcr False \
        --use_efficientnet False
    """
    logger.info(f"Running preprocessing on: {data_path}")

    # Check if container is running
    status = await get_docker_status()
    if not status.container_running:
        return {"success": False, "error": "AI Engine container is not running. Please start it first."}

    # Build docker exec command
    cmd = [
        "docker", "exec", CONTAINER_NAME,
        "python", "/app/main.py",
        "--mode", "preprocessing",
        "--data_path", data_path,
        "--output_folder", output_folder,
        "--ecg_signals_path", ecg_signals_path,
        "--preprocessing_folder", preprocessing_folder,
        "--batch_size", str(batch_size),
        "--preprocessing_n_workers", str(n_workers),
        "--diagnosis_classifier_device", device,
        "--signal_processing_device", device,
        "--hugging_face_api_key_path", "/data/api_key.json",
        "--use_wcr", "False",
        "--use_efficientnet", "False"
    ]

    logger.info(f"Preprocessing command: {' '.join(cmd)}")

    # Run with longer timeout for preprocessing
    code, stdout, stderr = _run_command(cmd, timeout=600)  # 10 min timeout

    if code == 0:
        logger.info("Preprocessing completed successfully")
        return {
            "success": True,
            "message": "Preprocessing completed",
            "output": stdout
        }
    else:
        error_msg = stderr or stdout or "Unknown error"
        logger.error(f"Preprocessing failed: {error_msg}")
        return {
            "success": False,
            "error": error_msg
        }


async def run_inference(
    data_path: str,
    output_folder: str = "/data/outputs",
    ecg_signals_path: str = "/data/ecg_signals",
    preprocessing_folder: str = "/data/preprocessing",
    batch_size: int = 1,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Run ECG inference via docker exec.
    """
    logger.info(f"Running inference on: {data_path}")

    status = await get_docker_status()
    if not status.container_running:
        return {"success": False, "error": "AI Engine container is not running."}

    # Build docker exec command for inference
    cmd = [
        "docker", "exec", CONTAINER_NAME,
        "python", "/app/main.py",
        "--mode", "inference",
        "--data_path", data_path,
        "--output_folder", output_folder,
        "--ecg_signals_path", ecg_signals_path,
        "--preprocessing_folder", preprocessing_folder,
        "--batch_size", str(batch_size),
        "--diagnosis_classifier_device", device,
        "--signal_processing_device", device,
        "--hugging_face_api_key_path", "/data/api_key.json",
        "--use_wcr", "False",
        "--use_efficientnet", "False"
    ]

    logger.info(f"Inference command: {' '.join(cmd)}")

    code, stdout, stderr = _run_command(cmd, timeout=600)

    if code == 0:
        return {
            "success": True,
            "message": "Inference completed",
            "output": stdout
        }
    else:
        return {
            "success": False,
            "error": stderr or stdout or "Unknown error"
        }


async def run_full_pipeline(
    csv_filename: str,
    workspace_path: str,
    use_gpu: bool = False,
    use_wcr: bool = False,
    use_efficientnet: bool = True,
    batch_size: int = 1,
    n_workers: int = 1
) -> Dict[str, Any]:
    """
    Run the full ECG analysis pipeline: preprocessing + analysis.

    Args:
        csv_filename: Name of the CSV file in workspace/inputs/
        workspace_path: Local path to workspace (mounted as /data in container)
        use_gpu: Use GPU for processing
        use_wcr: Use WCR model
        use_efficientnet: Use EfficientNet model
        batch_size: Batch size for processing
        n_workers: Number of preprocessing workers

    Returns:
        Dict with success status, preprocessing output, analysis output, and result files
    """
    import os
    import glob

    logger.info(f"Running full pipeline for: {csv_filename}")

    # Check if container is running
    status = await get_docker_status()
    if not status.container_running:
        return {"success": False, "error": "AI Engine container is not running. Please start it first."}

    device = "cuda" if use_gpu else "cpu"
    data_path = f"/data/inputs/{csv_filename}"

    # Step 1: Run preprocessing
    logger.info("Step 1: Running preprocessing...")
    preprocess_cmd = [
        "docker", "exec", CONTAINER_NAME,
        "python", "/app/main.py",
        "--mode", "preprocessing",
        "--data_path", data_path,
        "--output_folder", "/data/outputs",
        "--ecg_signals_path", "/data/ecg_signals",
        "--preprocessing_folder", "/data/preprocessing",
        "--batch_size", str(batch_size),
        "--preprocessing_n_workers", str(n_workers),
        "--diagnosis_classifier_device", device,
        "--signal_processing_device", device,
        "--hugging_face_api_key_path", "/data/api_key.json",
        "--use_wcr", str(use_wcr),
        "--use_efficientnet", str(use_efficientnet)
    ]

    logger.info(f"Preprocessing command: {' '.join(preprocess_cmd)}")
    code, stdout, stderr = _run_command(preprocess_cmd, timeout=600)

    if code != 0:
        error_msg = stderr or stdout or "Preprocessing failed"
        logger.error(f"Preprocessing failed: {error_msg}")
        return {
            "success": False,
            "error": f"Preprocessing failed: {error_msg}",
            "step": "preprocessing"
        }

    preprocessing_output = stdout
    logger.info("Preprocessing completed successfully")

    # Step 2: Run analysis
    logger.info("Step 2: Running analysis...")
    analysis_cmd = [
        "docker", "exec", CONTAINER_NAME,
        "python", "/app/main.py",
        "--mode", "analysis",
        "--data_path", data_path,
        "--output_folder", "/data/outputs",
        "--ecg_signals_path", "/data/ecg_signals",
        "--preprocessing_folder", "/data/preprocessing",
        "--batch_size", str(batch_size),
        "--preprocessing_n_workers", str(n_workers),
        "--diagnosis_classifier_device", device,
        "--signal_processing_device", device,
        "--hugging_face_api_key_path", "/data/api_key.json",
        "--use_wcr", str(use_wcr),
        "--use_efficientnet", str(use_efficientnet)
    ]

    logger.info(f"Analysis command: {' '.join(analysis_cmd)}")
    code, stdout, stderr = _run_command(analysis_cmd, timeout=600)

    if code != 0:
        error_msg = stderr or stdout or "Analysis failed"
        logger.error(f"Analysis failed: {error_msg}")
        return {
            "success": False,
            "error": f"Analysis failed: {error_msg}",
            "step": "analysis",
            "preprocessing_output": preprocessing_output
        }

    analysis_output = stdout
    logger.info("Analysis completed successfully")

    # Step 3: List output files
    output_dir = os.path.join(workspace_path, "outputs")
    result_files = []
    if os.path.exists(output_dir):
        result_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]

    return {
        "success": True,
        "message": "Full pipeline completed successfully",
        "preprocessing_output": preprocessing_output,
        "analysis_output": analysis_output,
        "result_files": result_files,
        "output_folder": output_dir
    }


async def run_diagnostics() -> SystemDiagnostics:
    """Run comprehensive system diagnostics."""
    import httpx
    from datetime import datetime

    diagnostics = SystemDiagnostics(
        overall_status="pass",
        timestamp=datetime.utcnow().isoformat(),
        platform_info={
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
            "machine": platform.machine()
        }
    )

    tests = []

    # Test 1: Docker Installation
    start = time.time()
    code, stdout, _ = _run_command(["docker", "--version"], timeout=5)
    duration = (time.time() - start) * 1000

    if code == 0:
        tests.append(DiagnosticResult(
            name="Docker Installation",
            status="pass",
            message=f"Docker installed: {stdout}",
            duration_ms=duration
        ))
    else:
        tests.append(DiagnosticResult(
            name="Docker Installation",
            status="fail",
            message="Docker is not installed",
            duration_ms=duration
        ))
        diagnostics.overall_status = "fail"

    # Test 2: Docker Daemon
    start = time.time()
    code, _, _ = _run_command(["docker", "info"], timeout=10)
    duration = (time.time() - start) * 1000

    if code == 0:
        tests.append(DiagnosticResult(
            name="Docker Daemon",
            status="pass",
            message="Docker daemon is running",
            duration_ms=duration
        ))
    else:
        tests.append(DiagnosticResult(
            name="Docker Daemon",
            status="fail",
            message="Docker daemon is not running",
            duration_ms=duration
        ))
        diagnostics.overall_status = "fail"

    # Test 3: GPU Support (check nvidia-smi on host)
    start = time.time()
    code, stdout, _ = _run_command(["nvidia-smi", "-L"], timeout=5)
    duration = (time.time() - start) * 1000

    if code == 0:
        gpu_info = stdout.split('\n')[0] if stdout else "GPU detected"
        tests.append(DiagnosticResult(
            name="GPU Support",
            status="pass",
            message=gpu_info,
            duration_ms=duration
        ))
    else:
        tests.append(DiagnosticResult(
            name="GPU Support",
            status="warning",
            message="GPU not available (will use CPU)",
            duration_ms=duration
        ))
        if diagnostics.overall_status == "pass":
            diagnostics.overall_status = "warning"

    # Test 4: AI Engine Container
    start = time.time()
    status = await get_docker_status()
    duration = (time.time() - start) * 1000

    if status.container_running:
        tests.append(DiagnosticResult(
            name="AI Engine Container",
            status="pass",
            message=f"Container running (ID: {status.container_id[:12] if status.container_id else 'unknown'})",
            duration_ms=duration
        ))
    elif status.container_exists:
        tests.append(DiagnosticResult(
            name="AI Engine Container",
            status="warning",
            message="Container exists but not running",
            duration_ms=duration
        ))
    else:
        tests.append(DiagnosticResult(
            name="AI Engine Container",
            status="warning",
            message="Container not created",
            duration_ms=duration
        ))

    # Test 5: AI Engine API Health
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"http://localhost:{AI_ENGINE_PORT}/health")
            duration = (time.time() - start) * 1000

            if response.status_code == 200:
                tests.append(DiagnosticResult(
                    name="AI Engine API",
                    status="pass",
                    message="API responding",
                    details=response.json() if response.headers.get("content-type", "").startswith("application/json") else None,
                    duration_ms=duration
                ))
            else:
                tests.append(DiagnosticResult(
                    name="AI Engine API",
                    status="fail",
                    message=f"API returned status {response.status_code}",
                    duration_ms=duration
                ))
    except Exception as e:
        duration = (time.time() - start) * 1000
        tests.append(DiagnosticResult(
            name="AI Engine API",
            status="fail",
            message=f"Cannot connect: {str(e)[:100]}",
            duration_ms=duration
        ))

    # Test 6: Backend Temp Directory
    start = time.time()
    temp_dir_exists = shutil.os.path.exists(settings.TEMP_DIR)
    duration = (time.time() - start) * 1000

    if temp_dir_exists:
        tests.append(DiagnosticResult(
            name="Temp Directory",
            status="pass",
            message=f"Exists: {settings.TEMP_DIR}",
            duration_ms=duration
        ))
    else:
        tests.append(DiagnosticResult(
            name="Temp Directory",
            status="warning",
            message=f"Not found: {settings.TEMP_DIR}",
            duration_ms=duration
        ))

    # Test 7: Network Connectivity (localhost)
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"http://localhost:{settings.PORT}/")
            duration = (time.time() - start) * 1000
            tests.append(DiagnosticResult(
                name="Backend API",
                status="pass",
                message="Backend responding",
                duration_ms=duration
            ))
    except Exception:
        duration = (time.time() - start) * 1000
        tests.append(DiagnosticResult(
            name="Backend API",
            status="skip",
            message="Self-check skipped",
            duration_ms=duration
        ))

    diagnostics.tests = tests

    # Determine overall status based on tests
    fail_count = sum(1 for t in tests if t.status == "fail")
    warn_count = sum(1 for t in tests if t.status == "warning")

    if fail_count > 0:
        diagnostics.overall_status = "fail"
    elif warn_count > 0:
        diagnostics.overall_status = "warning"
    else:
        diagnostics.overall_status = "pass"

    return diagnostics
