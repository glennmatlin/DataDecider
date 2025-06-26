"""Configuration utilities for OLMo training."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif config_path.suffix == ".json":
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.

    Args:
        base_config: Base configuration
        override_config: Configuration with override values

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def validate_config(config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> bool:
    """
    Validate configuration against a schema.

    Args:
        config: Configuration to validate
        schema: Optional schema to validate against

    Returns:
        True if valid, raises ValueError if invalid
    """
    # Basic validation - ensure required fields exist
    required_fields = {
        "training": ["model_size", "learning_rate", "batch_size"],
    }

    for section, fields in required_fields.items():
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

        for field in fields:
            if field not in config[section]:
                raise ValueError(f"Missing required field: {section}.{field}")

    return True


def get_model_config(model_size: str, config_dir: Union[str, Path] = "configs/model_configs") -> Dict[str, Any]:
    """
    Load model configuration for a specific size.

    Args:
        model_size: Model size identifier (e.g., "150M", "1B")
        config_dir: Directory containing model configs

    Returns:
        Model configuration dictionary
    """
    config_path = Path(config_dir) / f"olmo_{model_size.lower()}.yaml"

    if not config_path.exists():
        raise ValueError(f"No configuration found for model size: {model_size}")

    return load_config(config_path)
