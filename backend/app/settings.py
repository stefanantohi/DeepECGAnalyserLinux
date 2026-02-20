import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Union, Dict, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Path to persistent config file (next to settings.py)
CONFIG_FILE = Path(__file__).parent / "config.json"

# Workspace subdirectories that should always exist
WORKSPACE_SUBDIRS = ["ecg_signals", "inputs", "outputs", "preprocessing"]


def load_config() -> dict:
    """Load persistent configuration from config.json."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")
    return {}


def save_config(config: dict) -> None:
    """Save persistent configuration to config.json."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save config.json: {e}")


def ensure_workspace(workspace_path: str) -> dict:
    """Create workspace directory and subdirectories if they don't exist.
    Returns a dict with the status of each subdirectory."""
    subdirs_status = {}
    ws = Path(workspace_path)

    # Create workspace root
    if not ws.exists():
        ws.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created workspace directory: {ws}")

    # Create subdirectories
    for subdir in WORKSPACE_SUBDIRS:
        subdir_path = ws / subdir
        existed = subdir_path.exists()
        if not existed:
            subdir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created subdirectory: {subdir_path}")
        subdirs_status[subdir] = subdir_path.exists()

    return subdirs_status


# Available ECG analysis models
ECG_MODELS: Dict[str, Dict[str, Any]] = {
    "ecg_77labels": {
        "name": "ECG Interpretation (77 labels)",
        "description": "Classification ECG globale - 77 diagnostics possibles",
        "category": "interpretation",
        "priority": 1,  # Run first in cascade
    },
    "lvef_40": {
        "name": "LVEF ≤40%",
        "description": "Détection de fraction d'éjection ventriculaire gauche ≤40%",
        "category": "cardiac_function",
        "priority": 2,
        "triggers": ["ecg_77labels"],  # Run if certain conditions from ecg_77labels
    },
    "lvef_50": {
        "name": "LVEF <50%",
        "description": "Détection de fraction d'éjection ventriculaire gauche <50%",
        "category": "cardiac_function",
        "priority": 2,
    },
    "af_5y": {
        "name": "AF Risk 5 years",
        "description": "Risque de fibrillation auriculaire à 5 ans",
        "category": "arrhythmia",
        "priority": 2,
    },
    "lqts_detect": {
        "name": "LQTS Detection",
        "description": "Détection du syndrome du QT long",
        "category": "channelopathy",
        "priority": 3,
        "triggers": ["ecg_77labels"],
    },
    "lqts_genotype": {
        "name": "LQTS Genotype",
        "description": "Prédiction du génotype LQTS (LQT1, LQT2, LQT3)",
        "category": "channelopathy",
        "priority": 4,
        "triggers": ["lqts_detect"],
    },
    "hcm_detect": {
        "name": "HCM Detection",
        "description": "Détection de cardiomyopathie hypertrophique",
        "category": "cardiomyopathy",
        "priority": 2,
    },
    "mortality_risk": {
        "name": "Mortality Risk",
        "description": "Estimation du risque de mortalité",
        "category": "prognosis",
        "priority": 3,
    },
}

# Default models for screening
DEFAULT_SCREENING_MODELS = ["ecg_77labels", "lvef_40", "af_5y", "lqts_detect"]


def get_default_temp_dir() -> str:
    """Get platform-appropriate temp directory."""
    return os.path.join(tempfile.gettempdir(), "deepecg")


class Settings(BaseSettings):
    """Application configuration using environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

    # Server settings
    APP_NAME: str = "DeepECG Backend API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # AI Engine settings
    AI_ENGINE_URL: str = "docker://deepecg-ai-engine"  # Container name for CLI mode
    AI_ENGINE_MODE: str = "cli"  # Options: "rest" or "cli" - use cli for batch processing
    AI_ENGINE_TIMEOUT: int = 120  # seconds (longer for AI inference)
    AI_ENGINE_HEALTH_TIMEOUT: int = 10  # seconds for health checks
    AI_ENGINE_MAX_RETRIES: int = 3
    AI_ENGINE_RETRY_DELAY: float = 1.0  # seconds between retries

    # File upload settings
    MAX_UPLOAD_SIZE_MB: int = 100  # Increased for CSV/parquet files
    ALLOWED_MIME_TYPES: List[str] = ["application/pdf"]
    ALLOWED_DATA_EXTENSIONS: List[str] = [".csv", ".parquet"]
    TEMP_DIR: str = get_default_temp_dir()

    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # Options: "json" or "text"

    # Workspace settings (for Docker volume mounts)
    WORKSPACE_PATH: str = os.path.join(os.getcwd(), "workspace_data")

    def update_workspace_path(self, new_path: str) -> dict:
        """Update workspace path, persist to config.json, and ensure directories exist."""
        object.__setattr__(self, 'WORKSPACE_PATH', new_path)
        save_config({"workspace_path": new_path})
        return ensure_workspace(new_path)

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v: Union[str, bool]) -> bool:
        """Parse DEBUG value, handling strings with whitespace."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            v = v.strip().lower()
            return v in ("true", "1", "yes", "on")
        return bool(v)

    @field_validator("ALLOWED_MIME_TYPES", mode="before")
    @classmethod
    def parse_mime_types(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse MIME types from comma-separated string or list."""
        if isinstance(v, str):
            return [mime.strip() for mime in v.split(",") if mime.strip()]
        return v


# Create settings instance, then override with config.json if present
settings = Settings()

# Load persisted configuration
_persisted = load_config()
if "workspace_path" in _persisted:
    object.__setattr__(settings, 'WORKSPACE_PATH', _persisted["workspace_path"])