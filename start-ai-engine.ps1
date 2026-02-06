# ============================================
# DeepECG AI Engine - Docker Startup Script
# ============================================
# This script starts the DeepECG AI Engine in Docker with GPU support
#
# Prerequisites:
# - Docker Desktop with NVIDIA GPU support
# - NVIDIA Container Toolkit installed
# - DeepECG Docker image built (deepecg:latest)

param(
    [string]$ImageName = "deepecg:latest",
    [int]$Port = 8001,
    [switch]$Interactive,
    [switch]$Detached
)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " DeepECG AI Engine - Docker Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceDir = $ScriptDir

Write-Host "[INFO] Workspace directory: $WorkspaceDir" -ForegroundColor Yellow

# Create required directories if they don't exist
$Directories = @(
    "inputs",
    "outputs",
    "ecg_signals",
    "preprocessing",
    "weights",
    ".hf"
)

foreach ($dir in $Directories) {
    $fullPath = Join-Path $WorkspaceDir $dir
    if (!(Test-Path $fullPath)) {
        Write-Host "[INFO] Creating directory: $dir" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    }
}

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "[OK] Docker is running" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check for NVIDIA GPU support
try {
    $gpuCheck = docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] NVIDIA GPU support detected" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] NVIDIA GPU support may not be available" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[WARNING] Could not verify GPU support" -ForegroundColor Yellow
}

# Check if image exists
$imageExists = docker images -q $ImageName 2>$null
if (!$imageExists) {
    Write-Host "[ERROR] Docker image '$ImageName' not found." -ForegroundColor Red
    Write-Host "        Please build the image first with: docker build -t $ImageName ." -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] Docker image found: $ImageName" -ForegroundColor Green

# Stop any existing container
$containerName = "deepecg-ai-engine"
$existingContainer = docker ps -aq -f name=$containerName 2>$null
if ($existingContainer) {
    Write-Host "[INFO] Stopping existing container..." -ForegroundColor Yellow
    docker stop $containerName 2>$null | Out-Null
    docker rm $containerName 2>$null | Out-Null
}

Write-Host ""
Write-Host "[INFO] Starting DeepECG AI Engine..." -ForegroundColor Cyan
Write-Host "       Image: $ImageName" -ForegroundColor White
Write-Host "       Port: $Port" -ForegroundColor White
Write-Host "       Workspace: $WorkspaceDir" -ForegroundColor White
Write-Host ""

# Build Docker run command
$dockerArgs = @(
    "run"
    "--name", $containerName
    "--gpus", "all"
    "-p", "${Port}:${Port}"
    "-v", "${WorkspaceDir}:/workspace"
    "-w", "/workspace"
    "-e", "HF_HOME=/workspace/.hf"
    "-e", "HUGGINGFACE_HUB_CACHE=/workspace/.hf/hub"
    "-e", "HF_HUB_DISABLE_SYMLINKS=1"
    "-e", "CUDA_VISIBLE_DEVICES=0"
)

if ($Detached) {
    $dockerArgs += "-d"
    $dockerArgs += "--restart", "unless-stopped"
} elseif ($Interactive) {
    $dockerArgs += "-it"
} else {
    $dockerArgs += "--rm"
}

$dockerArgs += $ImageName

if (!$Interactive) {
    # Start the API server
    $dockerArgs += "bash", "-c", "cd /workspace && uvicorn engine_api:app --host 0.0.0.0 --port $Port"
}

# Run Docker container
if ($Detached) {
    Write-Host "[INFO] Starting in detached mode..." -ForegroundColor Yellow
    $containerId = & docker @dockerArgs

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host " DeepECG AI Engine Started!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "   API Endpoint: http://localhost:$Port" -ForegroundColor White
        Write-Host "   Health Check: http://localhost:$Port/health" -ForegroundColor White
        Write-Host "   Container ID: $containerId" -ForegroundColor Gray
        Write-Host ""
        Write-Host "   To view logs:  docker logs -f $containerName" -ForegroundColor Yellow
        Write-Host "   To stop:       docker stop $containerName" -ForegroundColor Yellow
        Write-Host ""
    } else {
        Write-Host "[ERROR] Failed to start container" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[INFO] Starting container (press Ctrl+C to stop)..." -ForegroundColor Yellow
    Write-Host ""
    & docker @dockerArgs
}
