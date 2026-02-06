"""Security utilities for file validation and sanitization."""
import os
from pathlib import Path
from typing import Optional, Tuple
import struct
import logging

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for file system operations
    """
    # Remove path traversal attempts
    filename = os.path.basename(filename)
    
    # Remove dangerous characters
    dangerous_chars = ['..', '/', '\\', '\0', '|', ';', '&', '$', '`', '>', '<']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Limit filename length
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext
    
    logger.debug(f"Sanitized filename: {filename}")
    return filename


def validate_file_path(base_dir: Path, file_path: Path) -> bool:
    """
    Validate that a file is within the base directory (prevent path traversal).
    
    Args:
        base_dir: Base directory for file operations
        file_path: File path to validate
        
    Returns:
        True if file is within base directory, False otherwise
    """
    try:
        file_path.resolve().relative_to(base_dir.resolve())
        return True
    except ValueError:
        logger.warning(f"Path traversal attempt detected: {file_path}")
        return False


def validate_pdf_structure(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Perform deep structural validation of PDF file.
    
    This checks more than just MIME type - it validates the actual
    PDF structure to prevent processing corrupted or malformed files.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if file exists
        if not file_path.exists():
            return False, "Le fichier n'existe pas"
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "Le fichier PDF est vide"
        
        if file_size < 100:
            return False, "Le fichier PDF est trop petit pour être valide"
        
        # Read PDF header
        with open(file_path, 'rb') as f:
            header = f.read(4)
        
        # Check PDF signature
        if header != b'%PDF':
            logger.warning(f"Invalid PDF header in file: {file_path.name}")
            return False, "Le fichier n'est pas un PDF valide (signature incorrecte)"
        
        # Check for EOF marker (read last 1KB)
        with open(file_path, 'rb') as f:
            f.seek(-1024, 2)  # Seek to last 1KB
            tail = f.read()
        
        if b'%%EOF' not in tail and b'%EOF' not in tail:
            logger.warning(f"Missing EOF marker in PDF: {file_path.name}")
            return False, "Le fichier PDF est incomplet ou corrompu (EOF manquant)"
        
        # Check for common PDF structure elements
        with open(file_path, 'rb') as f:
            content = f.read(8192)  # Read first 8KB
            
            # Look for PDF structural elements
            has_obj = b'obj' in content
            has_endobj = b'endobj' in content
            has_xref = b'xref' in content or b'/XRef' in content
            
            if not (has_obj and has_endobj):
                logger.warning(f"PDF missing basic structure: {file_path.name}")
                return False, "Le fichier PDF a une structure invalide"
        
        logger.info(f"PDF validation passed: {file_path.name} ({file_size} bytes)")
        return True, None
        
    except Exception as e:
        error_msg = f"Erreur lors de la validation PDF: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def validate_mime_type(content_type: str, allowed_types: list[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate MIME type against allowed list.
    
    Args:
        content_type: MIME type from file upload
        allowed_types: List of allowed MIME types
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not content_type:
        return False, "Type de contenu non fourni"
    
    if content_type not in allowed_types:
        logger.warning(f"Invalid MIME type: {content_type}")
        return False, f"Type de fichier non autorisé. Types acceptés: {', '.join(allowed_types)}"
    
    return True, None


def validate_file_size(file_size: int, max_size_mb: int) -> Tuple[bool, Optional[str]]:
    """
    Validate file size against maximum allowed size.
    
    Args:
        file_size: File size in bytes
        max_size_mb: Maximum size in megabytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if file_size > max_size_bytes:
        logger.warning(f"File too large: {file_size} bytes (max: {max_size_bytes})")
        return False, f"Le fichier dépasse la taille maximale de {max_size_mb} MB"
    
    if file_size == 0:
        return False, "Le fichier est vide"
    
    return True, None