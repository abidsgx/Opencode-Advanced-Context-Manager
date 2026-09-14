"""Server configuration management."""

import os
from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).parent.parent
    if (current / "data" / "config.yaml").exists():
        return current
    if (current.parent / "data" / "config.yaml").exists():
        return current.parent
    return current


def get_config_path() -> Path:
    """Get the path to config.yaml."""
    return get_project_root() / "data" / "config.yaml"


def load_config() -> dict:
    """Load the full configuration from config.yaml."""
    config_path = get_config_path()
    if not config_path.exists():
        return _default_config()
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    merged = _default_config()
    _deep_merge(merged, config)
    return merged


def save_config(config: dict) -> None:
    """Save the full configuration to config.yaml."""
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_setting(section: str, key: str, default=None):
    """Get a specific configuration setting."""
    config = load_config()
    section_data = config.get(section, {})
    if isinstance(section_data, dict):
        return section_data.get(key, default)
    return default


def update_setting(section: str, key: str, value: str) -> dict:
    """Update a specific configuration setting."""
    config = load_config()
    if section not in config:
        config[section] = {}

    parsed_value = yaml.safe_load(value)
    config[section][key] = parsed_value

    save_config(config)
    return {"status": "ok", "section": section, "key": key, "value": parsed_value}


def _default_config() -> dict:
    return {
        "version": "1.0",
        "agent": {"goals": []},
        "interconnectedness": {
            "files": {},
            "functions": {},
            "auto_detect": {
                "enabled": True,
                "methods": ["imports", "calls", "co_edits"],
                "min_co_edit_count": 2,
            },
        },
        "importance": {
            "ml": {
                "enabled": True,
                "model_path": "data/models/importance_model.pkl",
                "scaler_path": "data/models/feature_scaler.pkl",
                "confidence_threshold": 0.6,
                "online_learning": {
                    "enabled": True,
                    "decay_rate": 0.023,
                    "min_samples_for_training": 10,
                },
            },
            "heuristic_weights": {
                "edit_frequency": 0.25,
                "dependency_centrality": 0.20,
                "goal_alignment": 0.30,
                "error_association": 0.10,
                "recency": 0.15,
            },
        },
        "pruning": {
            "enabled": True,
            "importance_threshold": 3.0,
            "protected_importance": 7.0,
            "blast_radius_expansion": True,
            "max_compression_ratio": 0.3,
        },
        "session": {
            "extended_recording": True,
            "capture_edit_metadata": True,
            "capture_tradeoffs": True,
            "storage_path": "data/sessions",
        },
        "languages": {
            "python": {"extensions": [".py"], "enabled": True},
            "javascript": {"extensions": [".js", ".jsx", ".mjs"], "enabled": False},
            "typescript": {"extensions": [".ts", ".tsx"], "enabled": False},
            "rust": {"extensions": [".rs"], "enabled": False},
            "go": {"extensions": [".go"], "enabled": False},
            "java": {"extensions": [".java"], "enabled": False},
            "ruby": {"extensions": [".rb"], "enabled": False},
            "c": {"extensions": [".c", ".h"], "enabled": False},
            "cpp": {"extensions": [".cpp", ".cc", ".cxx", ".hpp"], "enabled": False},
            "csharp": {"extensions": [".cs"], "enabled": False},
            "php": {"extensions": [".php"], "enabled": False},
            "elixir": {"extensions": [".ex", ".exs"], "enabled": False},
        },
        "mcp": {
            "command": "python",
            "args": ["-m", "mcp_server"],
            "cwd": "./server",
        },
    }


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
