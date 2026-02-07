# DeepECGAnalyser
<img width="941" height="668" alt="image" src="https://github.com/user-attachments/assets/40427f59-070d-4b27-a2a7-f45f91265e58" />


**AI-Powered ECG Analysis Platform using HeartWise Foundation Models**

Local-first medical AI application for 12-lead ECG interpretation. Leverages the [HeartWise DeepECG Docker](https://github.com/HeartWise-AI/DeepECG_Docker) engine with EfficientNet and WCR (self-supervised) architectures across 77 diagnostic classes. All processing happens locally on your machine with GPU acceleration - no cloud, no external APIs, complete data privacy.

<img width="713" height="519" alt="image" src="https://github.com/user-attachments/assets/f9a59634-a283-40c1-82b3-27657dc82a51" />

<img width="674" height="577" alt="image" src="https://github.com/user-attachments/assets/a5d796b0-aed8-43c6-bbf8-f9d394b63589" />


---

## Features

- **Multi-Model Analysis**: 5 AI models (EfficientNet + WCR) for comprehensive ECG interpretation
- **77 Diagnostic Classes**: Arrhythmias, conduction disorders, hypertrophy, ischemia, and more
- **Binary Screening Models**: LVEF <= 40%, LVEF < 50%, Atrial Fibrillation risk at 5 years
- **Model Comparison**: Side-by-side EfficientNet vs WCR results with probability differences
- **Multi-Format Support**: GE MUSE XML, Philips PageWriter XML (auto-converted), NPY
- **Batch Processing**: Analyze multiple ECG files at once
- **ECG Visualization**: Interactive 12-lead ECG waveform viewer
- **CSV Export**: Detailed results export with multi-model columns and diff
- **GPU Accelerated**: NVIDIA CUDA-powered inference via Docker
- **100% Local**: Zero data leaves your machine
- **Configurable Workspace**: Directories auto-created on first startup

## Architecture

```
+-------------------+     +-------------------+     +-------------------------+
|    Frontend       |     |    Backend         |     |   HeartWise AI Engine   |
|  React + Vite     |---->|  FastAPI + Python  |---->|   Docker + GPU (CUDA)   |
|  TypeScript       |     |  Port 8000         |     |   deepecg-ai-engine     |
|  Tailwind CSS     |     |                    |     |                         |
|  Port 5173        |     |  - ECG Analysis    |     |  - EfficientNet models  |
+-------------------+     |  - File Conversion |     |  - WCR models           |
                          |  - Docker Control  |     |  - 77-class inference   |
                          |  - Config API      |     |  - Binary classifiers   |
                          +-------------------+     +-------------------------+
```

## Available AI Models

| Model ID | Name | Architecture | Type | Description |
|----------|------|-------------|------|-------------|
| `efficientnet_77` | ECG 77 Classes | EfficientNet | Multi-label (77) | Full ECG interpretation - supervised |
| `wcr_77` | ECG 77 Classes (WCR) | WCR | Multi-label (77) | Full ECG interpretation - self-supervised |
| `efficientnet_lvef40` | LVEF <= 40% | EfficientNet | Binary | Left ventricular ejection fraction screening |
| `efficientnet_lvef50` | LVEF < 50% | EfficientNet | Binary | Left ventricular ejection fraction screening |
| `efficientnet_af5y` | AF Risk 5 years | EfficientNet | Binary | Atrial fibrillation risk prediction |

---

## Prerequisites

### Required

- **Docker Desktop** (v20.10+) with WSL2 backend (Windows)
- **NVIDIA GPU** (RTX series recommended, 8GB+ VRAM)
- **NVIDIA Drivers** (v525+)
- **NVIDIA Container Toolkit** (for Docker GPU passthrough)
- **Python** 3.11+
- **Node.js** 18+
- **HuggingFace account** with access to HeartWise models

### Windows GPU Setup

1. Install Docker Desktop with WSL2 backend
2. Install NVIDIA drivers for Windows
3. In Docker Desktop Settings > Resources > WSL Integration, enable your WSL distro
4. Verify GPU:
   ```powershell
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
   ```

---

## Installation

### Step 1: Install the HeartWise AI Engine (Docker)

Follow the official instructions at **[HeartWise-AI/DeepECG_Docker](https://github.com/HeartWise-AI/DeepECG_Docker)**:

```bash
# 1. Clone the HeartWise Docker repository
git clone https://github.com/HeartWise-AI/DeepECG_Docker.git
cd DeepECG_Docker

# 2. Create your HuggingFace API key file
#    - Create a HuggingFace account at https://huggingface.co
#    - Request access to the "heartwise-ai/DeepECG" model collection
#    - Generate a read-access API token (Settings > API Keys)
echo '{"huggingface_api_key": "hf_YOUR_TOKEN_HERE"}' > api_key.json

# 3. Build the Docker image
docker build -t deepecg-docker .
```

> **Important**: The image name must be `deepecg-docker` (used by DeepECGAnalyser to start the container).

### Step 2: Clone DeepECGAnalyser

```bash
git clone https://github.com/YOUR_USER/DeepECGAnalyser.git
cd DeepECGAnalyser
```

### Step 3: Install Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 4: Install Frontend

```powershell
cd frontend
npm install
```

---

## Quick Start

### Option 1: start.bat (Recommended for Windows)

```powershell
# Double-click start.bat or run:
.\start.bat

# Choose option 6: Dev complet (AI Engine + Backend + Frontend)
```

This will:
1. Start the HeartWise Docker container with GPU
2. Start the FastAPI backend on port 8000
3. Start the Vite frontend on port 5173

### Option 2: Manual Startup

**Terminal 1 - Backend:**

```powershell
cd backend
.\venv\Scripts\Activate.ps1
$env:DEBUG = "true"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**

```powershell
cd frontend
npm run dev
```

The AI Engine container can be started/stopped directly from the application UI (sidebar panel).

### Access the Application

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Documentation (Swagger) | http://localhost:8000/docs |

---

## Usage

### Single File Analysis

1. Start the AI Engine from the sidebar (click "Start Engine")
2. Select the AI models to use (EfficientNet, WCR, or both 77-class models)
3. Drag & drop an ECG file (XML or NPY) or click to browse
4. View results: summary, per-category breakdown, all diagnoses, and model comparison
5. Export results to CSV

### Batch Analysis

1. Toggle "Mode Lot" (batch mode)
2. Select multiple ECG files
3. Navigate results per file with previous/next controls
4. Export all results to CSV

### Model Comparison

When both 77-class models (EfficientNet + WCR) are selected:
- **Comparison tab**: Side-by-side probability table with diff column
- **All Results tab**: Merged view with per-model columns
- **CSV Export**: Separate columns per architecture + diff column

### Workspace Configuration

The workspace directory stores ECG files and analysis outputs. It is mounted as `/data` in the Docker container. Configure it in the sidebar panel:

```
WORKSPACE_PATH/
  ecg_signals/      # Uploaded ECG files
  inputs/            # Input data for the AI engine
  outputs/           # Analysis results (probabilities CSVs)
  preprocessing/     # Preprocessed data (base64 tensors)
```

Directories are auto-created on first startup. The path is persisted in `backend/app/config.json` and can be changed from the UI when the container is stopped.

---

## API Endpoints

### Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/config` | Get current workspace configuration |
| `PUT` | `/api/config` | Update workspace path (auto-creates directories) |

### ECG Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/ecg/available-models` | List available AI models |
| `POST` | `/api/ecg/full-analysis` | Run analysis on single ECG file |
| `POST` | `/api/ecg/batch-analysis` | Run analysis on multiple files |
| `GET` | `/api/ecg/signal-data` | Get ECG waveform data for visualization |

### Docker Control

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/docker/status` | Docker & container status |
| `POST` | `/api/docker/start` | Start the AI Engine container |
| `POST` | `/api/docker/stop` | Stop the AI Engine container |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Backend health check |

---

## Project Structure

```
DeepECGAnalyser/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI entry point, lifespan
│   │   ├── api.py                  # Core API endpoints (analysis, config)
│   │   ├── docker_routes.py        # Docker control endpoints
│   │   ├── ecg_analysis.py         # ECG analysis logic, model orchestration
│   │   ├── ai_client.py            # AI Engine Docker exec client
│   │   ├── docker_control.py       # Docker operations (start/stop/status)
│   │   ├── schemas.py              # Pydantic response models
│   │   ├── settings.py             # Configuration (env + config.json)
│   │   ├── philips_converter.py    # Philips PageWriter -> GE MUSE conversion
│   │   ├── security.py             # File validation & sanitization
│   │   ├── circuit_breaker.py      # Resilience pattern
│   │   └── utils.py                # Utility functions
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx                 # Root component, layout
│   │   ├── api.ts                  # Axios API client
│   │   └── components/
│   │       ├── ECGAnalysisPanel.tsx    # Main analysis interface
│   │       ├── ECGVisualResults.tsx    # Results display (tabs, comparison)
│   │       ├── ECGViewer.tsx           # 12-lead ECG waveform viewer
│   │       ├── ModelSelector.tsx       # AI model selection UI
│   │       ├── SystemStatusPanel.tsx   # Docker status + workspace config
│   │       └── ConfidenceBadge.tsx     # Probability confidence indicator
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── Dockerfile
├── docker-compose.yml              # Full stack orchestration
├── start.bat                       # Windows launcher (menu-based)
├── start-ai-engine.ps1             # PowerShell AI Engine launcher
├── stop.bat                        # Stop all services
├── .env.example                    # Environment variables template
├── .gitignore
└── README.md
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode and verbose logging |
| `HOST` | `0.0.0.0` | Backend bind address |
| `PORT` | `8000` | Backend port |
| `AI_ENGINE_URL` | `docker://deepecg-ai-engine` | AI Engine connection (CLI mode) |
| `AI_ENGINE_MODE` | `cli` | Engine mode: `cli` (docker exec) or `rest` (HTTP) |
| `AI_ENGINE_TIMEOUT` | `120` | AI inference timeout in seconds |
| `MAX_UPLOAD_SIZE_MB` | `100` | Maximum upload file size |
| `WORKSPACE_PATH` | *(configurable via UI)* | Data workspace directory |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | `json` | Log format: `json` or `text` |
| `VITE_BACKEND_URL` | `http://localhost:8000` | Frontend -> Backend URL |

---

## Troubleshooting

### AI Engine / Docker

```powershell
# Check if container is running
docker ps | findstr deepecg

# View container logs
docker logs -f deepecg-ai-engine

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Restart the container
docker restart deepecg-ai-engine
```

### Models Not Loading

- Ensure the backend is running (`http://localhost:8000/docs`)
- Check the backend console for errors
- Verify the AI Engine container is running in the sidebar panel

### Workspace Issues

- Check workspace path in the sidebar panel (expand AI Engine section)
- Click "Creer les repertoires manquants" if subdirectories are missing
- Ensure the parent directory exists before setting a new workspace path

### File Format Issues

- **GE MUSE XML**: Supported natively
- **Philips PageWriter XML**: Auto-converted to GE MUSE format
- **NPY**: NumPy arrays (12 leads x samples)
- **UTF-16 XML**: Auto-converted to UTF-8

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| Backend | FastAPI, Python 3.11, Pydantic, Uvicorn |
| AI Engine | [HeartWise DeepECG Docker](https://github.com/HeartWise-AI/DeepECG_Docker), PyTorch, CUDA |
| Infrastructure | Docker, Docker Compose, NVIDIA Container Toolkit |

## Citation

This application uses the HeartWise AI engine. If you use it, please cite:

```bibtex
@article{Nolin-Lapalme2025,
  author  = {Nolin-Lapalme, Alexis and Sowa, Achille and Delfrate, Jacques and
             Tastet, Olivier and Corbin, Denis and Kulbay, Merve and Ozdemir, Derman and
             No{\"e}l, Marie-Jeanne and Marois-Blanchet, Fran{\c{c}}ois-Christophe and
             Harvey, Fran{\c{c}}ois and Sharma, Surbhi and Ansari, Minhaj and Chiu, I-Min and
             Dsouza, Valentina and Friedman, Sam F. and Chass{\'e}, Micha{\"e}l and
             Potter, Brian J. and Afilalo, Jonathan and Elias, Pierre Adil and
             Jabbour, Gilbert and Bahani, Mourad and Dub{\'e}, Marie-Pierre and
             Boyle, Patrick M. and Chatterjee, Neal A. and Barrios, Joshua and
             Tison, Geoffrey H. and Ouyang, David and Maddah, Mahnaz and Khurshid, Shaan and
             Cadrin-Tourigny, Julia and Tadros, Rafik and Hussin, Julie and Avram, Robert},
  title   = {Foundation models for generalizable electrocardiogram interpretation:
             comparison of supervised and self-supervised electrocardiogram foundation models},
  journal = {medRxiv},
  year    = {2025},
  doi     = {10.1101/2025.03.02.25322575},
  url     = {https://www.medrxiv.org/content/early/2025/03/05/2025.03.02.25322575}
}
```

## Author

**Benoit LEQUEUX** - Cercle IA, Societe Francaise de Cardiologie (SFC)

## License

This project is proprietary and confidential. All rights reserved.

---

*DeepECGAnalyser - 100% Local AI-Powered ECG Analysis*
