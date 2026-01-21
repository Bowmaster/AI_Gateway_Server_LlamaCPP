"""
config.py - Configuration loader for Enhanced AI Client

Configuration priority:
1. AI_SERVER_URL environment variable (highest priority)
2. config.yaml file in current directory
3. Default: http://localhost:8080
"""

import os
from pathlib import Path
from typing import Optional

# Try to import yaml, but make it optional
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


DEFAULT_SERVER_URL = "http://localhost:8080"


def load_config() -> dict:
    """
    Load configuration from environment variable or config file.

    Returns:
        dict with configuration values
    """
    config = {
        "server_url": DEFAULT_SERVER_URL,
        "history_file": str(Path.home() / ".ai_client_history"),
        "streaming_enabled": True,
    }

    # 1. Check environment variable (highest priority)
    if env_url := os.environ.get("AI_SERVER_URL"):
        config["server_url"] = env_url.rstrip("/")
        return config

    # 2. Look for config.yaml in current directory
    config_path = Path("config.yaml")
    if config_path.exists() and YAML_AVAILABLE:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
                if file_config and isinstance(file_config, dict):
                    if "server_url" in file_config:
                        config["server_url"] = file_config["server_url"].rstrip("/")
                    if "history_file" in file_config:
                        config["history_file"] = file_config["history_file"]
                    if "streaming_enabled" in file_config:
                        config["streaming_enabled"] = file_config["streaming_enabled"]
        except Exception:
            pass  # Fall back to defaults if config file is invalid

    # 3. Also check for config.yaml in script directory
    script_dir = Path(__file__).parent
    alt_config_path = script_dir / "config.yaml"
    if alt_config_path.exists() and alt_config_path != config_path and YAML_AVAILABLE:
        try:
            with open(alt_config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
                if file_config and isinstance(file_config, dict):
                    if "server_url" in file_config:
                        config["server_url"] = file_config["server_url"].rstrip("/")
                    if "history_file" in file_config:
                        config["history_file"] = file_config["history_file"]
                    if "streaming_enabled" in file_config:
                        config["streaming_enabled"] = file_config["streaming_enabled"]
        except Exception:
            pass

    return config


def get_server_url() -> str:
    """Convenience function to get just the server URL."""
    return load_config()["server_url"]


def get_history_file() -> str:
    """Convenience function to get the history file path."""
    return load_config()["history_file"]
