#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama.cpp Server Launcher Script
Scans for GGUF models, manages configurations, and launches llama server.
"""

import os
import json
import copy
import subprocess
import threading
import msvcrt
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

# === Configuration Paths ===
MODELS_DIR = r"C:\Users\WHITEsharky\.lmstudio\models"
LLAMA_CPP_DIR = r"C:\Users\WHITEsharky\.lmstudio\extensions\backends\llama-cpp-mtp-turboquant"
CONFIG_FILE = Path(__file__).parent / "llama_server_config.json"


def _load_dotenv(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Parse a .env file (KEY=VALUE per line) and merge into os.environ.

    Lines starting with '#' are treated as comments and skipped.
    Values may be optionally quoted with single or double quotes.
    Returns the dict of loaded key-value pairs.
    If *env_path* is None, defaults to .env in the script's directory.
    """
    if env_path is None:
        env_path = Path(__file__).parent / ".env"

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
            os.environ.setdefault(key, value)  # do not override existing env vars
            loaded[key] = value

    return loaded


# Load .env before any configuration is read (side-effect: populates os.environ).
_load_dotenv()


def _resolve_api_keys(config: Dict[str, Any]) -> None:
    """Populate config['api_keys'] from environment variables if not already set.

    Looks for API_KEY_<slug> in os.environ and fills any missing entries so that
    users can keep real secrets out of version-controlled files entirely."""
    api_keys = config.setdefault("api_keys", {})
    for slug, env_var in (("rp", "API_KEY_rp"), ("coder", "API_KEY_coder")):
        if not api_keys.get(slug) and (env_value := os.environ.get(env_var)):
            api_keys[slug] = env_value


# === Default Configuration ===
DEFAULT_CONFIG = {
    "host": "192.168.1.177",
    "port": 5056,
    "api_keys": {},   # Populated from .env (API_KEY_rp / API_KEY_coder) or saved config
    "defaults": {
        "context": 32768,
        "gpu_offload": 99,
        "cpu_moe": None,
        "threads": 8,
        "batch_size": 1024,
        "parallel": 1,
        "mmap": False,
        "flash_attention": True,
        "k_quant": "turbo3",
        "v_quant": "turbo3",
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
        "tensor_split": None
    }
}


# === KV Cache Quantization Options ===
KV_QUANT_OPTIONS = ["turbo4", "turbo3", "turbo2", "q8_0", "q4_0", "none"]

# === Preset Definitions ===
PRESETS = {
    "rp": {"display": "RP"},
    "coder": {"display": "Coder"},
    "commit": {"display": "Commit"}
}

# Maps each preset slug to its corresponding API key field name.
_PRESET_API_KEY_MAP = {
    "rp": "rp",
    "coder": "coder",
    "commit": "coder",  # commit shares the coder key
}


def preset_api_key(config: Dict[str, Any], preset_slug: str) -> str:
    """Return the API key for a given preset. RP uses 'rp' key; Coder and Commit use 'coder' key."""
    api_keys = config.get("api_keys", {})
    key_field = _PRESET_API_KEY_MAP.get(preset_slug, "coder")
    return api_keys.get(key_field, "")


def apply_preset_overrides(preset_slug: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Apply preset-specific overrides to settings.

    Commit forces thinking=False and p_thinking=False regardless of saved values.
    Returns a new dict with overrides applied (does not mutate the original).
    """
    result = copy.deepcopy(settings)
    if preset_slug == "commit":
        result["thinking"] = False
        result["p_thinking"] = False
    return result


# === Settings Menu Definition (shared between display_settings_menu and edit_settings) ===
# Each entry: (display_number, config_key, human_readable_name)
SETTINGS_INFO = [
    ("1", "context", "Context"),
    ("2", "gpu_offload", "GPU Offload"),
    ("3", "cpu_moe", "CPU MOE"),
    ("4", "threads", "Threads"),
    ("5", "batch_size", "Batch Size"),
    ("6", "parallel", "Parallel"),
    ("7", "mmap", "mmap"),
    ("8", "flash_attention", "Flash Attention"),
    ("9", "k_quant", "K Quant"),
    ("10", "v_quant", "V Quant"),
    ("11", "mtp", "MTP"),
    ("12", "draft_n_max", "Draft N Max"),
    ("13", "temp", "Temp"),
    ("14", "top_k", "Top K"),
    ("15", "top_p", "Top P"),
    ("16", "min_p", "Min P"),
    ("17", "repeat_penalty", "Repeat Penalty"),
    ("18", "presence_penalty", "Presence Penalty"),
    ("19", "thinking", "Thinking"),
    ("20", "p_thinking", "Preserve Think"),
    ("21", "jinja", "Jinja"),
    ("22", "vision", "Vision"),
    ("23", "tensor_split", "GPU Tensor Split"),
]

# Build a lookup: numeric choice → (config_key, display_name) for O(1) selection
_SETTINGS_LOOKUP: Dict[str, Tuple[str, str]] = {num: (key, name) for num, key, name in SETTINGS_INFO}

# Settings that use the on/off/none toggle editor
_TOGGLE_KEYS: frozenset = frozenset({"mmap", "flash_attention", "thinking", "p_thinking", "jinja", "vision", "mtp"})

# Settings that use the KV-quant dropdown editor
_KV_QUANT_KEYS: frozenset = frozenset({"k_quant", "v_quant"})


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge *override* into a deep-copy of *base*. Nested dicts are merged recursively."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file, or create with defaults.

    After loading, API keys are resolved from environment variables (set via .env)."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
            merged = _deep_merge(DEFAULT_CONFIG, config)
            _resolve_api_keys(merged)
            return merged
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] Could not load config file: {e}")
            print("Using default configuration.")
    config = copy.deepcopy(DEFAULT_CONFIG)
    _resolve_api_keys(config)
    return config


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to JSON file."""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Configuration saved to {CONFIG_FILE}")
    except IOError as e:
        print(f"[ERROR] Could not save config file: {e}")


def scan_models() -> List[Dict[str, str]]:
    """Scan models directory for GGUF files, up to 4 levels deep (author/model_folder/[subdir]/model.gguf).
    Returns list of dicts with 'display' (author/model_name) and 'path' (full relative path)."""
    gguf_files = []
    models_path = Path(MODELS_DIR)
    if not models_path.exists():
        print(f"[ERROR] Models directory not found: {MODELS_DIR}")
        return gguf_files

    # Scan recursively for .gguf files using pathlib (proper resource management)
    for gguf_file in models_path.rglob("*.gguf"):
        if not gguf_file.is_file():
            continue
        # Skip mmproj files (vision projection files, not models)
        if gguf_file.name.lower().startswith("mmproj"):
            continue
        # Remove .gguf extension for internal representation
        model_name = gguf_file.stem
        # Build relative path using forward slashes for consistency
        rel_parts = gguf_file.relative_to(models_path).parts
        rel_path = "/".join(rel_parts[:-1]) + "/" + model_name
        display_name = f"{rel_parts[0]}/{model_name}"  # author/model_name
        gguf_files.append({"display": display_name, "path": rel_path})

    gguf_files.sort(key=lambda x: x["display"])
    return gguf_files


def get_model_path(rel_path: str) -> str:
    """Convert relative model path to absolute path."""
    # rel_path is like author/model_folder/model_name (no .gguf extension)
    # Replace forward slashes with OS-specific separators
    rel_path = rel_path.replace("/", os.sep)
    # Add .gguf extension back
    full_path = rel_path + ".gguf"
    return os.path.join(MODELS_DIR, full_path)


def get_llama_server_path() -> Optional[str]:
    """Find llama-server executable (result is cached after first call)."""
    # Use a mutable default stored on the function itself to avoid global state
    if not hasattr(get_llama_server_path, "_cache"):
        server_names = ("llama-server.exe", "llama_server.exe")
        # Try top-level directory first (common naming)
        for name in server_names:
            candidate = os.path.join(LLAMA_CPP_DIR, name)
            if os.path.exists(candidate):
                get_llama_server_path._cache = candidate  # type: ignore[attr-defined]
                return candidate

        # Search in subdirectories
        for root, _dirs, files in os.walk(LLAMA_CPP_DIR):
            for file in files:
                if file.lower() in server_names:
                    path = os.path.join(root, file)
                    get_llama_server_path._cache = path  # type: ignore[attr-defined]
                    return path

        get_llama_server_path._cache = None  # type: ignore[attr-defined]

    return getattr(get_llama_server_path, "_cache", None)  # type: ignore[return-value]


def get_model_config(config: Dict[str, Any], model_key: str, preset_slug: str) -> Dict[str, Any]:
    """Get model-specific config for a given preset, falling back to defaults.

    Commit falls back to Coder settings if no commit-specific config exists.
    """
    base_defaults = config.get("defaults", copy.deepcopy(DEFAULT_CONFIG["defaults"]))
    models = config.get("models", {})
    model_cfg: Dict[str, Any] = {} if not isinstance(models.get(model_key), dict) else models[model_key]

    # Try direct preset match first
    for candidate_slug in (preset_slug, "coder" if preset_slug == "commit" else None):
        if candidate_slug and candidate_slug in model_cfg:
            merged = base_defaults.copy()
            merged.update(model_cfg[candidate_slug])
            return merged

    return base_defaults.copy()


def has_preset_config(config: Dict[str, Any], model_key: str, preset_slug: str) -> bool:
    """Check whether a model+preset combo has explicit saved config.

    For 'commit', also returns True if 'coder' settings exist (since commit falls back to coder).
    """
    models = config.get("models", {})
    model_cfg = models.get(model_key)
    if not isinstance(model_cfg, dict):
        return False
    effective_slugs = {preset_slug} | ({ "coder" } if preset_slug == "commit" else set())
    return any(slug in model_cfg for slug in effective_slugs)


def best_preset_for_model(config: Dict[str, Any], model_key: str, preferred: str) -> str:
    """Return the most appropriate preset slug for a given model.

    Priority order:
      1. The explicitly saved 'last_preset' (preferred), if it has config for this model.
      2. Any preset that has explicit config for this model (rp > coder).
      3. Fallback to the preferred default ('rp').
    """
    # If preferred preset already has config, use it directly
    if has_preset_config(config, model_key, preferred):
        return preferred

    # Otherwise pick the first preset (in priority order) with saved settings
    for slug in ("rp", "coder"):  # commit excluded since it mirrors coder
        if has_preset_config(config, model_key, slug):
            return slug

    return preferred


def save_model_config(
    config: Dict[str, Any], model_key: str, preset_slug: str, model_settings: Dict[str, Any]
) -> Dict[str, Any]:
    """Save model-specific configuration for a given preset.

    For the 'commit' preset, settings are saved under the 'coder' key instead,
    with thinking and p_thinking preserved from existing coder config (not overwritten).
    """
    target_slug = "coder" if preset_slug == "commit" else preset_slug
    config.setdefault("models", {}).setdefault(model_key, {})

    settings_to_save = copy.deepcopy(model_settings)

    # When saving for commit, preserve thinking/p_thinking from existing coder config
    if preset_slug == "commit":
        settings_to_save.pop("thinking", None)
        settings_to_save.pop("p_thinking", None)
        existing_coder = dict(config["models"][model_key].get("coder", {}))
        existing_coder.update(settings_to_save)
        config["models"][model_key][target_slug] = existing_coder
    else:
        config["models"][model_key][target_slug] = settings_to_save

    save_config(config)
    return config


def display_settings(settings: Dict[str, Any]) -> None:
    """Display current model settings in a readable format."""
    print("\n" + "=" * 50)
    print("Model Settings:")
    print("=" * 50)
    print("\n--- Model Parameters ---")
    print(f"  Context:             {settings.get('context', 'N/A')}")
    print(f"  GPU Offload:         {settings.get('gpu_offload', 'N/A')}")
    print(f"  CPU MOE:             {settings.get('cpu_moe', 'N/A')}")
    print(f"  Threads:             {settings.get('threads', 'N/A')}")
    print(f"  Batch Size:          {settings.get('batch_size', 'N/A')}")
    print(f"  Parallel:            {settings.get('parallel', 'N/A')}")
    print(f"  mmap:                {settings.get('mmap', 'N/A')}")
    print(f"  Flash Attention:     {settings.get('flash_attention', 'N/A')}")
    print(f"  K Quant:             {settings.get('k_quant', 'N/A')}")
    print(f"  V Quant:             {settings.get('v_quant', 'N/A')}")
    mtp_val = settings.get("mtp")
    draft_val = settings.get("draft_n_max")
    mtp_display = "on" if mtp_val is True else ("off" if mtp_val is False else "none")
    print(f"  MTP:                 {mtp_display}")
    print(f"  Draft N Max:         {draft_val if mtp_val is True else '(ignored)'}")
    print("\n--- Generation Settings ---")
    print(f"  Temp:                {settings.get('temp', 'N/A')}")
    print(f"  Top K:               {settings.get('top_k', 'N/A')}")
    print(f"  Top P:               {settings.get('top_p', 'N/A')}")
    print(f"  Min P:               {settings.get('min_p', 'N/A')}")
    print(f"  Repeat Penalty:      {settings.get('repeat_penalty', 'N/A')}")
    print(f"  Presence Penalty:    {settings.get('presence_penalty', 'N/A')}")
    print(f"  Thinking:            {settings.get('thinking', 'N/A')}")
    print(f"  Preserve Think:      {settings.get('p_thinking', 'N/A')}")
    print(f"  Jinja:               {settings.get('jinja', 'N/A')}")
    print(f"  Vision:              {settings.get('vision', 'N/A')}")
    print("\n--- Multi-GPU Settings ---")
    ts = settings.get("tensor_split")
    if ts is not None and isinstance(ts, list):
        print(f"  GPU Tensor Split:    {','.join(_format_number(v) for v in ts)}%")
    else:
        print(f"  GPU Tensor Split:    {'disabled'}")
    print("=" * 50 + "\n")


def display_settings_menu(settings: Dict[str, Any]) -> None:
    """Display the settings menu."""
    print("\nAvailable settings to modify:")
    print("=" * 60)

    for num, key, name in SETTINGS_INFO:
        current = settings.get(key, "(not set)")
        print(f"  {num}. {name:<18} [{current}]")

    print("=" * 60)


def edit_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Interactive settings editor."""
    display_settings_menu(settings)
    print("Enter the number of the setting to modify (0 when done):")

    while True:
        choice = input("> ").strip()
        if choice == "0":
            break

        # Look up the selected setting by number (O(1))
        selected = _SETTINGS_LOOKUP.get(choice)
        if selected is None:
            print("Invalid choice. Try again.")
            continue

        key, name = selected
        current = settings.get(key, "(not set)")
        print(f"\nCurrent value for {name}: {current}")

        if key in _TOGGLE_KEYS:
            # Show selection options for toggle-like settings
            print(f"Select value for {name}:")
            print("  1. on")
            print("  2. off")
            print("  3. none")

            while True:
                choice = input("> ").strip()
                if choice in ("", "3"):
                    settings[key] = None
                    break
                elif choice == "1":
                    settings[key] = True   # on
                    break
                elif choice == "2":
                    settings[key] = False  # off
                    break
                else:
                    print("Invalid choice. Enter 1 (on), 2 (off), or 3/empty (none)")

        elif key in _KV_QUANT_KEYS:
            # Show selection options for KV quant settings
            print(f"Select value for {name}:")
            for i, opt in enumerate(KV_QUANT_OPTIONS, 1):
                print(f"  {i}. {opt}")

            while True:
                choice = input("> ").strip()
                if choice == "" or choice == str(len(KV_QUANT_OPTIONS)):
                    settings[key] = None
                    break
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(KV_QUANT_OPTIONS):
                        val = KV_QUANT_OPTIONS[idx]
                        settings[key] = None if val == "none" else val
                        break
                    else:
                        print(f"Invalid choice. Enter 1-{len(KV_QUANT_OPTIONS)} or leave empty for 'none'")
                except ValueError:
                    print("Invalid input.")

        elif key == "tensor_split":
            # GPU Tensor Split: single percentage for first GPU, second is 100-x.
            # Precision up to tenths (one decimal place). Empty input disables the flag.
            print("Enter GPU load percentage for the first GPU (the second gets the remainder).")
            print("Example: '60' means GPU1=60%, GPU2=40%. Precision to 0.1.")
            print("Leave empty to disable tensor-split.")
            while True:
                new_value = input("> ").strip()
                if new_value == "" or new_value.lower() in ("none", "null"):
                    settings[key] = None
                    break
                try:
                    val = float(new_value)
                    if val < 0 or val > 100:
                        print("Percentage must be between 0 and 100.")
                        continue
                    gpu1 = round(val, 1)
                    gpu2 = round(100.0 - gpu1, 1)
                    settings[key] = [gpu1, gpu2]
                    break
                except ValueError:
                    print("Invalid input. Enter a number (e.g. '60' or '75.5').")

        elif key == "draft_n_max":
            # Draft N Max: positive integer, ignored when MTP is not enabled
            mtp_val = settings.get("mtp")
            if mtp_val is not True:
                print("[INFO] MTP is not enabled (on). Draft N Max will be ignored at launch.")
                print("Enable MTP first if you want this setting to take effect.")
            print("Enter a positive integer value for max draft tokens (or 'none' to clear):")
            while True:
                new_value = input("> ").strip().lower()
                if new_value == "" or new_value in ("none", "null"):
                    settings[key] = None
                    break
                try:
                    val = int(new_value)
                    if val <= 0:
                        print("Value must be a positive integer (> 0).")
                        continue
                    settings[key] = val
                    break
                except ValueError:
                    print("Invalid input. Enter a positive integer.")
        else:
            # Free text input for numeric/text settings
            print("Enter new value (or 'none' to skip this parameter):")
            new_value = input("> ").strip().lower()
            if new_value in ("", "none", "null"):
                settings[key] = None
            else:
                # Try to convert to appropriate numeric type
                try:
                    settings[key] = float(new_value) if "." in new_value else int(new_value)
                except ValueError:
                    settings[key] = new_value

        print(f"[OK] {name} set to: {settings[key]}")
        display_settings_menu(settings)
        print("Enter the number of the setting to modify (0 when done):")

    return settings


# === Simple parameter mapping: setting_key → CLI flag ===
SIMPLE_PARAM_MAP = [
    ("context", "-c"),
    ("gpu_offload", "-ngl"),
    ("cpu_moe", "--n-cpu-moe"),
    ("threads", "-t"),
    ("batch_size", "-b"),
    ("parallel", "-np"),
    ("temp", "--temp"),
    ("k_quant", "-ctk"),
    ("v_quant", "-ctv"),
    ("top_k", "--top-k"),
    ("top_p", "--top-p"),
    ("min_p", "--min-p"),
    ("repeat_penalty", "--repeat-penalty"),
    ("presence_penalty", "--presence-penalty"),
]


def _format_number(v: float) -> str:
    """Format a number: whole numbers as int string, otherwise keep decimals."""
    return str(int(v)) if v == int(v) else str(v)


def _display_effective_settings(preset_slug: str, base_settings: Dict[str, Any]) -> None:
    """Apply preset overrides and display the effective settings."""
    effective = apply_preset_overrides(preset_slug, base_settings)
    display_settings(effective)


def _add_simple_param(cmd: List[str], flag: str, value: Any) -> None:
    """Add a simple parameter to the command when value is \"truthy\" (not None/False/0)."""
    if value not in (None, False, 0):
        cmd += [flag, str(value)]


def build_command(model_path: str, settings: Dict[str, Any], host: str, port: int, api_key: str) -> List[str]:
    """Build the llama-server command with all parameters."""
    server_exe = get_llama_server_path()
    if server_exe is None:
        print("[ERROR] llama-server executable not found!")
        return []

    cmd = [server_exe, "--model", model_path]

    # --- Simple parameters (flag + value) ---
    for key, flag in SIMPLE_PARAM_MAP:
        _add_simple_param(cmd, flag, settings.get(key))

    # mmap (--no-mmap): False explicitly disables memory-mapped file loading
    if settings.get("mmap") is False:
        cmd.append("--no-mmap")

    # Flash Attention: True enables flash attention optimization
    if settings.get("flash_attention") is True:
        cmd += ["--flash-attn", "on"]

    # Chat Template Kwargs: Merge thinking + preserve_thinking into a single argument
    chat_kwargs = {}
    if (thinking_val := settings.get("thinking")) is not None and isinstance(thinking_val, bool):
        chat_kwargs["enable_thinking"] = thinking_val
    if (p_thinking_val := settings.get("p_thinking")) is not None and isinstance(p_thinking_val, bool):
        chat_kwargs["preserve_thinking"] = p_thinking_val
    if chat_kwargs:
        cmd += ["--chat-template-kwargs", json.dumps(chat_kwargs)]

    # Jinja (--jinja): True enables jinja template parsing
    if settings.get("jinja") is True:
        cmd.append("--jinja")

    # Vision / mmproj: When enabled, add --mmproj pointing to first mmproj*.gguf in model directory
    if settings.get("vision") is True:
        model_dir = os.path.dirname(model_path)
        mmproj_files = sorted(
            f for f in os.listdir(model_dir)
            if f.lower().startswith("mmproj") and f.endswith(".gguf")
        )
        if not mmproj_files:
            print(f"[WARNING] Vision enabled but no mmproj file found in {model_dir}")
        else:
            cmd += ["--mmproj", os.path.join(model_dir, mmproj_files[0])]

    # --- Server Settings ---
    cmd += ["--host", host, "--port", str(port)]
    cmd.append("--no-webui")
    cmd += ["--n-predict", "-1"]  # -1 = unlimited prediction tokens
    cmd += ["-mg", "0"]           # memory guard: 0 = no safety margin

    # MTP (Multi-Token Prediction): --spec-type draft-mtp when enabled
    if settings.get("mtp") is True:
        cmd += ["--spec-type", "draft-mtp"]
        if isinstance(draft_n_max := settings.get("draft_n_max"), int) and draft_n_max > 0:
            cmd += ["--spec-draft-n-max", str(draft_n_max)]

    # Tensor Split: multi-GPU layer distribution (--tensor-split)
    tensor_split = settings.get("tensor_split")
    if isinstance(tensor_split, list) and len(tensor_split) >= 2:
        split_str = ",".join(_format_number(v) for v in tensor_split)
        cmd += ["--tensor-split", split_str]

    cmd.append("--no-slots")
    cmd.append("--swa-full")

    if api_key:
        cmd += ["--api-key", api_key]

    return cmd


# --- Keyboard listener for graceful server shutdown ---

_STOP_KEY = "q"


def _listen_for_stop_key(stop_flag: List[bool], stop_event: threading.Event) -> None:
    """Background thread that listens for 'q' key press and sets the stop flag."""
    while not stop_event.is_set():
        if msvcrt.kbhit():
            char = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if char == 'q':
                stop_flag[0] = True
                return
        time.sleep(0.05)


def _terminate_process(process: subprocess.Popen) -> None:
    """Gracefully terminate a process, killing it if it doesn't exit in time."""
    try:
        process.terminate()
        process.wait(timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        process.kill()
        process.wait()


def launch_server(cmd: List[str]) -> bool:
    """Launch the llama server process.

    Returns True if user pressed 'q' to stop, False otherwise (e.g., Ctrl+C or natural exit).
    """
    if not cmd:
        print("[ERROR] No command to launch!")
        return False

    print("\n" + "=" * 60)
    print("Launching Llama Server...")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    print("\nPress 'q' to stop the server and return to menu.")
    print("Press Ctrl+C to forcefully terminate.\n")

    process = None
    user_stopped_via_q = False

    try:
        # Mutable list used as shared flag between threads
        stop_flag = [False]
        stop_event = threading.Event()
        kb_thread = threading.Thread(
            target=_listen_for_stop_key, args=(stop_flag, stop_event), daemon=True
        )
        kb_thread.start()

        process = subprocess.Popen(cmd)

        # Wait for either the process to exit or user to press 'q'
        while True:
            if stop_flag[0]:
                print("\n\n[INFO] User requested stop (pressed 'q'). Stopping server...")
                _terminate_process(process)
                user_stopped_via_q = True
                break

            if process.poll() is not None:  # Process exited on its own
                break

            time.sleep(0.05)

        stop_event.set()
        kb_thread.join(timeout=1)

    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user (Ctrl+C).")
        if process is not None:
            _terminate_process(process)
    except FileNotFoundError:
        print(f"\n[ERROR] Executable not found: {cmd[0]}")
    except Exception as e:
        print(f"\n[ERROR] Failed to launch server: {e}")

    return user_stopped_via_q


def select_model(models: List[Dict[str, str]], allow_cancel: bool = False, last_model: Optional[Dict[str, str]] = None) -> Optional[Dict[str, str]]:
    """Display model list and prompt user to select one.

    If last_model is provided and the user enters an empty string, returns last_model.
    Returns the selected model dict, or None if cancelled (when allow_cancel=True).
    """
    print("\n--- Select a model ---\n")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model['display']}")

    parts = []
    if allow_cancel:
        parts.append("0 to cancel")
    if last_model is not None:
        parts.append(f"Enter for {last_model['display']}")

    if parts:
        prompt = f"Select model by number (or {', '.join(parts)}): "
    else:
        prompt = "Select model by number: "

    while True:
        choice = input(prompt).strip()

        # Handle empty input -> use last model
        if choice == "" and last_model is not None:
            return last_model

        if allow_cancel and choice == "0":
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            else:
                print(f"Invalid choice. Please enter 1-{len(models)}")
        except ValueError:
            # Empty input with no last model is handled above; otherwise invalid
            if choice != "":
                print("Invalid input. Please enter a number.")


def apply_model_selection(selected_model: Dict[str, str], config: Dict[str, Any], preset_slug: str) -> Tuple[str, Dict[str, Any]]:
    """Apply model selection for a given preset: load config, display settings.

    Returns (model_key, base_settings) tuple WITHOUT overrides applied.
    Overrides are only used at launch time and for display in the main loop.
    This ensures that when saving under 'commit', we work with raw coder values
    rather than commit-forced ones.
    """
    print(f"\n[OK] Selected: {selected_model['display']}")

    # Get model key for config storage (use full path as unique key)
    model_key = selected_model["path"]

    # Check if model has existing configuration for this preset (including Coder fallback for Commit)
    base_settings = get_model_config(config, model_key, preset_slug)
    has_existing = has_preset_config(config, model_key, preset_slug)

    _display_effective_settings(preset_slug, base_settings)

    if has_existing:
        print("This model has existing configuration.")
    else:
        print("Using default configuration.")

    return model_key, base_settings


def switch_preset(config: Dict[str, Any], model_key: str, current_preset: str) -> Tuple[str, Dict[str, Any]]:
    """Display preset selection menu and return (new_preset_slug, base_settings)."""
    print("\n--- Select a Preset ---\n")

    available = list(PRESETS.items())  # [("rp", {"display": "RP"}), ...]
    for i, (slug, info) in enumerate(available, 1):
        marker = " [current]" if slug == current_preset else ""
        print(f"  {i}. {info['display']}{marker}")

    prompt = f"\nSelect preset ({', '.join(s for s, _ in available)}): "
    while True:
        choice = input(prompt).strip()
        selected_slug = None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                selected_slug = available[idx][0]
            else:
                print(f"Invalid choice. Please enter 1-{len(available)}")
                continue
        except ValueError:
            # Allow typing preset name directly (case-insensitive)
            lower_choice = choice.lower()
            for slug, info in available:
                if slug == lower_choice or info["display"].lower() == lower_choice:
                    selected_slug = slug
                    break
            if selected_slug is None:
                print("Invalid input. Please enter a number or preset name.")
                continue

        return selected_slug, get_model_config(config, model_key, selected_slug)


# --- Terminal utilities ---

def clear_screen() -> None:
    """Clear the terminal screen (cross-platform)."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    """Main entry point for the launcher."""
    clear_screen()
    print_header("Llama.cpp Server Launcher")

    # Load configuration
    config = load_config()
    host = config.get("host", DEFAULT_CONFIG["host"])
    port = config.get("port", DEFAULT_CONFIG["port"])

    # Ensure api_keys structure exists (migrate legacy single key to new format)
    if "api_key" in config and "api_keys" not in config:
        legacy = config.pop("api_key")
        config["api_keys"] = {"rp": legacy, "coder": legacy}
        print("[INFO] Migrated API key — please verify Coder preset key.")

    # Resolve any remaining empty keys from environment variables (.env)
    _resolve_api_keys(config)

    # Scan for models
    print("\n[INFO] Scanning for GGUF models...")
    models = scan_models()

    if not models:
        print(f"[ERROR] No GGUF models found in {MODELS_DIR}")
        print("Please place your models in the following structure:")
        print(f"  {MODELS_DIR}/<author>/<model_name>.gguf")
        input("\nPress Enter to exit...")
        return

    print(f"[OK] Found {len(models)} model(s).")

    # Restore last-used model from config
    _last_key = config.get("last_model")
    last_model = next((m for m in models if m["path"] == _last_key), None) if _last_key else None

    # Get last used preset (default to "rp")
    current_preset = config.get("last_preset", "rp")
    if current_preset not in PRESETS:
        current_preset = "rp"

    # Select model (empty input uses last model, if available)
    selected_model = select_model(models, allow_cancel=False, last_model=last_model)

    # Determine the best preset for this model: prefer saved 'last_preset' if it has config;
    # otherwise pick any preset that has explicit settings for this model.
    current_preset = best_preset_for_model(config, selected_model["path"], current_preset)
    print(f"\n[INFO] Using preset: {PRESETS[current_preset]['display']}")

    model_key, model_settings = apply_model_selection(selected_model, config, current_preset)

    while True:
        preset_display = PRESETS[current_preset]["display"]
        print(f"\n--- Preset: {preset_display} ---")
        print("What would you like to do?")
        print("  1. Use current settings and launch")
        print("  2. Modify settings")
        print("  3. Switch preset (RP / Coder / Commit)")
        print("  4. Change model")
        print("  5. Exit")

        action = input("\nSelect (1/2/3/4/5): ").strip()

        if action == "1":
            # Apply preset overrides before launching
            effective_settings = apply_preset_overrides(current_preset, model_settings)
            api_key = preset_api_key(config, current_preset)

            # Show configuration and launch immediately
            print("\n" + "=" * 60)
            print("Server Configuration:")
            print(f"  Preset:    {preset_display}")
            print(f"  Host:      {host}")
            print(f"  Port:      {port}")
            print(f"  Model:     {selected_model['display']}")
            print("=" * 60)

            # Save last used model and preset to config before launching
            config["last_model"] = model_key
            config["last_preset"] = current_preset
            save_config(config)

            # Build and launch command
            cmd = build_command(
                model_path=get_model_path(selected_model["path"]),
                settings=effective_settings,
                host=host,
                port=port,
                api_key=api_key
            )

            if cmd:
                launch_server(cmd)

            # After server stops (via 'q', Ctrl+C, or natural exit), return to menu
            print("\n[OK] Server stopped.")
        elif action == "2":
            # Modify settings
            model_settings = edit_settings(model_settings)
            # Save updated settings for current preset
            config = save_model_config(config, model_key, current_preset, model_settings)

            _display_effective_settings(current_preset, model_settings)

        elif action == "3":
            # Switch preset — use base (raw) settings so edits save correctly.
            # Overrides are only applied at launch and display time.
            new_preset, base_settings = switch_preset(config, model_key, current_preset)
            if new_preset != current_preset:
                print(f"\n[OK] Preset switched to {PRESETS[new_preset]['display']}")

            _display_effective_settings(new_preset, base_settings)

            current_preset = new_preset
            model_settings = base_settings

            # Persist preset switch to config so next launch restores it
            config["last_model"] = model_key
            config["last_preset"] = current_preset
            save_config(config)

        elif action == "4":
            # Return to model selection (empty input reuses current model)
            new_model = select_model(models, allow_cancel=True, last_model=selected_model)
            if new_model is None:
                print("Model change cancelled.")
                continue

            selected_model = new_model
            new_model_key = selected_model["path"]
            # Determine the best preset for this model
            current_preset = best_preset_for_model(config, new_model_key, current_preset)
            print(f"\n[INFO] Using preset: {PRESETS[current_preset]['display']}")

            model_key, model_settings = apply_model_selection(selected_model, config, current_preset)

            # Persist last used model and preset to config after any state change
            config["last_model"] = model_key
            config["last_preset"] = current_preset
            save_config(config)

        elif action == "5":
            # Save final state before exiting so next launch restores correctly
            config["last_model"] = model_key
            config["last_preset"] = current_preset
            save_config(config)

            print("\nExiting...")
            return
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
