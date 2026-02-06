"""Utility functions for the backend application."""
import uuid
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_request_id() -> str:
    """
    Generate a unique request ID.
    
    Returns:
        Unique request identifier string
    """
    return str(uuid.uuid4())


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "2.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logger.info(f"{self.description} completed in {elapsed:.3f}s")
    
    def elapsed_ms(self) -> int:
        """
        Get elapsed time in milliseconds.
        
        Returns:
            Elapsed time in milliseconds
        """
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return int((end - self.start_time) * 1000)


def ensure_temp_directory(temp_dir: str) -> Path:
    """
    Ensure temporary directory exists and return Path object.
    
    Args:
        temp_dir: Path to temporary directory
        
    Returns:
        Path object for the temporary directory
    """
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)
    return temp_path


def safe_remove_file(file_path: Path | str) -> bool:
    """
    Safely remove a file if it exists.
    
    Args:
        file_path: Path to file to remove
        
    Returns:
        True if file was removed, False otherwise
    """
    try:
        file_path = Path(file_path)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Removed file: {file_path}")
            return True
        return False
    except Exception as e:
        logger.warning(f"Failed to remove file {file_path}: {e}")
        return False


def validate_mime_type(content_type: str, allowed_types: list[str]) -> bool:
    """
    Validate MIME type against allowed types.
    
    Args:
        content_type: Content-Type header value
        allowed_types: List of allowed MIME types
        
    Returns:
        True if content type is allowed, False otherwise
    """
    if not content_type:
        return False
    return any(
        content_type.lower().startswith(allowed.lower())
        for allowed in allowed_types
    )