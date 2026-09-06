"""Configuration handling: .env loading, defaults, load/save of the JSON config file."""

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# === Configuration Paths ===
# Resolved from .env / environment variables (see .env.example).
CONFIG_FILE = Path(__file__).resolve().parent.parent / "llama_server_config.json"


def load_dotenv(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Parse a .env file (KEY=VALUE per line) and merge into os.environ.

    Lines starting with '#' are treated as comments and skipped.
    Values may be optionally quoted with single or double quotes.
    Returns the dict of loaded key-value pairs.
    If *env_path* is None, defaults to .env in the project root (parent of launcher/).
    """
    if env_path is None:
        env_path = CONFIG_FILE.parent / ".env"

    if not env_path.is_file():
        return {}

    loaded: Dict[str, str] = {}
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)
            loaded[key] = value

    return loaded


# Load .env before any configuration is read (side-effect: populates os.environ).
load_dotenv()

# Resolve paths from environment variables (set via .env above; None if not configured).
MODELS_DIR = os.environ.get("MODELS_DIR")
LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR")


def resolve_api_keys(config: Dict[str, Any]) -> None:
    """Populate config['api_keys'] from environment variables if not already set.

    Looks for API_KEY_<slug> in os.environ and fills any missing entries so that
    users can keep real secrets out of version-controlled files entirely."""
    api_keys = config.setdefault("api_keys", {})
    for slug, env_var in (("rp", "API_KEY_rp"), ("coder", "API_KEY_coder")):
        if not api_keys.get(slug) and (env_value := os.environ.get(env_var)):
            api_keys[slug] = env_value


# === Default Configuration ===
DEFAULT_CONFIG: Dict[str, Any] = {
    "host": "192.168.1.177",
    "port": 5056,
    "api_keys": {},   # Populated from .env (API_KEY_rp / API_KEY_coder) or saved config
    "defaults": {
        "context": 32768,
        "gpu_offload": 99,
        "cpu_moe": None,
        "threads": 8,
        "batch_threads": 8,
        "batch_size": 2048,
        "parallel": 1,
        "mmap": False,
        "flash_attention": True,
        "k_quant": "q4_0",
        "v_quant": "q4_0",
        "mtp": None,
        "draft_n_max": None,
        "temp": 1,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "presence_penalty": 1.5,
        "thinking": None,
        "p_thinking": None,
        "jinja": None,
        "vision": None,
        "image_min_tokens": None,
        "tensor_split": None,
        "mmproj_offload": None
    }
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge *override* into a deep-copy of *base*. Nested dicts are merged recursively."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_file: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from JSON file, or create with defaults.

    After loading, API keys are resolved from environment variables (set via .env)."""
    path = config_file or CONFIG_FILE
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
            merged = deep_merge(DEFAULT_CONFIG, config)
            resolve_api_keys(merged)
            return merged
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] Could not load config file: {e}")
            print("Using default configuration.")
    config = copy.deepcopy(DEFAULT_CONFIG)
    resolve_api_keys(config)
    return config


def save_config(config: Dict[str, Any], config_file: Optional[Path] = None) -> None:
    """Save configuration to JSON file."""
    path = config_file or CONFIG_FILE
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Configuration saved to {path}")
    except IOError as e:
        print(f"[ERROR] Could not save config file: {e}")
