import os
import shutil
from pathlib import Path
from typing import List, Union, Dict, Any
import json
import yaml

def create_directories(paths: List[Union[str, Path]]) -> None:
    """
    Create multiple directories if they don't exist.
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def remove_directory(path: Union[str, Path]) -> None:
    """
    Remove directory and all its contents.
    
    Args:
        path: Directory path to remove
    """
    if Path(path).exists():
        shutil.rmtree(path)

def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Copy file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def get_file_size(path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        path: File path
    
    Returns:
        File size in bytes
    """
    return Path(path).stat().st_size

def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        path: JSON file path
    
    Returns:
        Parsed JSON data
    """
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        path: JSON file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def validate_file_exists(path: Union[str, Path]) -> bool:
    """
    Check if file exists.
    
    Args:
        path: File path to check
    
    Returns:
        True if file exists, False otherwise
    """
    return Path(path).exists()

def get_file_extension(path: Union[str, Path]) -> str:
    """
    Get file extension.
    
    Args:
        path: File path
    
    Returns:
        File extension (e.g., '.jpg', '.png')
    """
    return Path(path).suffix.lower()

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: File size in bytes
    
    Returns:
        Formatted file size (e.g., '1.2 MB')
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"