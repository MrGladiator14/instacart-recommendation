import os
from typing import Dict, Any

import pandas as pd


def validate_file_path(file_path: str) -> bool:
    """
    Validate if a file path exists and is accessible.
    
    Args:
        file_path: Path to validate
        
    Returns:
        True if valid, raises FileNotFoundError if invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"File not readable: {file_path}")
    return True


def create_output_directory(output_path: str) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_path: Path to output file
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
