import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Resolve environment variables
    config = _resolve_env_variables(config)
    
    # Validate required fields
    _validate_config(config)
    
    return config

def _resolve_env_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve environment variables in configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configuration with resolved environment variables
    """
    def resolve_value(value):
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item) for item in value]
        return value
    
    return resolve_value(config)

def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
    
    Raises:
        ValueError: If required fields are missing
    """
    required_fields = ['paths', 'dataset']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' missing from configuration")
    
    # Validate paths
    required_paths = ['raw_data', 'processed_data', 'metadata']
    for path_key in required_paths:
        if path_key not in config['paths']:
            raise ValueError(f"Required path '{path_key}' missing from configuration")
    
    # Validate dataset config
    required_dataset_fields = ['classes', 'image_size', 'train_split', 'val_split', 'test_split']
    for field in required_dataset_fields:
        if field not in config['dataset']:
            raise ValueError(f"Required dataset field '{field}' missing from configuration")

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, indent=2)

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'dataset.image_size')
        default: Default value if key not found
    
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

def update_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Update configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value