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
LLAMA_CPP_DIR = r"C:\Users\WHITEsharky\.lmstudio\extensions\backends\turboquant-plus-tqp-v0.1.1-windows-x64-cuda12.4"
CONFIG_FILE = Path(__file__).parent / "llama_server_config.json"

# === Default Configuration ===
DEFAULT_CONFIG = {
    "host": "192.168.1.177",
    "port": 5056,
    "api_keys": {
        "rp": "ek-cvpxgU0aMOLzFhbwOs3XcVgH8jyWpHqdX7cRRIlbtzwKtF7LvR",
        "coder": "ek-cvpxgU0aMOLzFhbwOs3XcVgH8jyWpHqdX7cRRIlbtzwKtF7LvV"
    },
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


def preset_api_key(config: Dict[str, Any], preset_slug: str) -> str:
    """Return the API key for a given preset. RP uses 'rp' key; Coder and Commit use 'coder' key."""
    api_keys = config.get("api_keys", {})
    if preset_slug == "rp":
        return api_keys.get("rp", "")
    else:
        # coder, commit both use the coder key
        return api_keys.get("coder", "")


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
SETTINGS_INFO = [
    ("1", "Context", "context"),
    ("2", "GPU Offload", "gpu_offload"),
    ("3", "CPU MOE", "cpu_moe"),
    ("4", "Threads", "threads"),
    ("5", "Batch Size", "batch_size"),
    ("6", "Parallel", "parallel"),
    ("7", "mmap", "mmap"),
    ("8", "Flash Attention", "flash_attention"),
    ("9", "K Quant", "k_quant"),
    ("10", "V Quant", "v_quant"),
    ("11", "MTP", "mtp"),
    ("12", "Draft N Max", "draft_n_max"),
    ("13", "Temp", "temp"),
    ("14", "Top K", "top_k"),
    ("15", "Top P", "top_p"),
    ("16", "Min P", "min_p"),
    ("17", "Repeat Penalty", "repeat_penalty"),
    ("18", "Presence Penalty", "presence_penalty"),
    ("19", "Thinking", "thinking"),
    ("20", "Preserve Think", "p_thinking"),
    ("21", "Jinja", "jinja"),
    ("22", "Vision", "vision"),
    ("23", "GPU Tensor Split", "tensor_split"),
]


def load_config() -> Dict[str, Any]:
    """Load configuration from JSON file, or create with defaults."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
            # Merge with defaults to ensure all keys exist (deep copy to avoid mutating module-level constants)
            merged = copy.deepcopy(DEFAULT_CONFIG)
            merged.update(config)
            if "defaults" in config:
                default_defaults = copy.deepcopy(DEFAULT_CONFIG["defaults"])
                default_defaults.update(config["defaults"])
                merged["defaults"] = default_defaults
            return merged
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARNING] Could not load config file: {e}")
            print("Using default configuration.")
    return copy.deepcopy(DEFAULT_CONFIG)


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
    """Find llama-server executable."""
    # Try llama-server.exe first (common naming)
    server_path = os.path.join(LLAMA_CPP_DIR, "llama-server.exe")
    if os.path.exists(server_path):
        return server_path

    # Try llama_server.exe
    server_path = os.path.join(LLAMA_CPP_DIR, "llama_server.exe")
    if os.path.exists(server_path):
        return server_path

    # Search in subdirectories
    for root, dirs, files in os.walk(LLAMA_CPP_DIR):
        for file in files:
            if file.lower() in ["llama-server.exe", "llama_server.exe"]:
                return os.path.join(root, file)

    return None


def get_model_config(config: Dict[str, Any], model_key: str, preset_slug: str) -> Dict[str, Any]:
    """Get model-specific config for a given preset, falling back to defaults.

    Commit falls back to Coder settings if no commit-specific config exists.
    """
    models = config.get("models", {})
    default_defaults = config.get("defaults", copy.deepcopy(DEFAULT_CONFIG["defaults"]))
    if model_key in models and isinstance(models[model_key], dict):
        preset_settings = models[model_key].get(preset_slug)
        if preset_settings is not None:
            merged = default_defaults.copy()
            merged.update(preset_settings)
            return merged
        # Commit falls back to Coder settings when no commit-specific config exists
        if preset_slug == "commit":
            coder_settings = models[model_key].get("coder")
            if coder_settings is not None:
                merged = default_defaults.copy()
                merged.update(coder_settings)
                return merged
    return default_defaults.copy()


def has_preset_config(config: Dict[str, Any], model_key: str, preset_slug: str) -> bool:
    """Check whether a model+preset combo has explicit saved config.

    For 'commit', also returns True if 'coder' settings exist (since commit falls back to coder).
    """
    models = config.get("models", {})
    if not isinstance(models.get(model_key), dict):
        return False
    direct = preset_slug in models[model_key]
    fallback = preset_slug == "commit" and "coder" in models[model_key]
    return direct or fallback


def best_preset_for_model(config: Dict[str, Any], model_key: str, preferred: str) -> str:
    """Return the most appropriate preset slug for a given model.

    Priority order:
      1. The explicitly saved 'last_preset' (preferred), if it has config for this model.
      2. Any preset that has explicit config for this model (rp > coder > commit).
      3. Fallback to the preferred default ('rp').
    """
    models = config.get("models", {})
    stored_presets = models[model_key] if isinstance(models.get(model_key), dict) else {}

    # If preferred preset has explicit config, use it
    if has_preset_config(config, model_key, preferred):
        return preferred

    # Otherwise pick the first preset (in priority order) that has saved settings for this model
    fallback_order = ["rp", "coder"]  # commit is excluded here since it mirrors coder
    for slug in fallback_order:
        if slug in stored_presets or has_preset_config(config, model_key, slug):
            return slug

    return preferred


def save_model_config(config: Dict[str, Any], model_key: str, preset_slug: str, model_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Save model-specific configuration for a given preset."""
    if "models" not in config:
        config["models"] = {}
    if model_key not in config["models"]:
        config["models"][model_key] = {}
    config["models"][model_key][preset_slug] = model_settings
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

    for num, name, key in SETTINGS_INFO:
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

        # Find the selected setting
        selected = None
        for num, name, key in SETTINGS_INFO:
            if choice == num:
                selected = (key, name)
                break

        if selected is None:
            print("Invalid choice. Try again.")
            continue

        key, name = selected
        current = settings.get(key, "(not set)")
        print(f"\nCurrent value for {name}: {current}")

        # Determine if this setting should use selection menu or free input
        toggle_settings = ["mmap", "flash_attention", "thinking", "p_thinking", "jinja", "vision", "mtp"]
        kv_quant_settings = ["k_quant", "v_quant"]

        if key in toggle_settings:
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

        elif key in kv_quant_settings:
            # Show selection options for KV quant settings
            print(f"Select value for {name}:")
            for i, opt in enumerate(KV_QUANT_OPTIONS, 1):
                print(f"  {i}. {opt}")

            while True:
                choice = input("> ").strip()
                if choice == "" or choice == "6":
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
            if new_value == "none" or new_value == "" or new_value == "null":
                settings[key] = None
            else:
                # Try to convert to appropriate numeric type
                try:
                    if "." in new_value:
                        settings[key] = float(new_value)
                    else:
                        settings[key] = int(new_value)
                except ValueError:
                    settings[key] = new_value

        print(f"[OK] {name} set to: {settings[key]}")
        # Re-display the menu after each change
        display_settings_menu(settings)
        print("Enter the number of the setting to modify (0 when done):")

    return settings


# === Simple parameter mapping: setting_key -> CLI flag ===
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
    """Format float as int string when whole, otherwise keep decimals."""
    return str(int(v)) if v == int(v) else str(v)


def _add_simple_param(cmd: List[str], flag: str, value: Any) -> None:
    """Add a simple parameter to the command if value is not None, False, or 0."""
    if value is not None and value is not False and value != 0:
        cmd.extend([flag, str(value)])


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

    # --- Special parameters (non-standard logic) ---
    # mmap (--no-mmap): False means --no-mmap, True means nothing special
    mmap = settings.get("mmap")
    if mmap is False:
        cmd.append("--no-mmap")

    # Flash Attention (--flash-attn): True means enable
    fa = settings.get("flash_attention")
    if fa is True:
        cmd.extend(["--flash-attn", "on"])

    # Chat Template Kwargs: Merge thinking + preserve_thinking into a single --chat-template-kwargs argument
    chat_kwargs = {}
    thinking_val = settings.get("thinking")
    if thinking_val is not None and isinstance(thinking_val, bool):
        chat_kwargs["enable_thinking"] = thinking_val
    p_thinking_val = settings.get("p_thinking")
    if p_thinking_val is not None and isinstance(p_thinking_val, bool):
        chat_kwargs["preserve_thinking"] = p_thinking_val
    if chat_kwargs:
        cmd.extend(["--chat-template-kwargs", json.dumps(chat_kwargs)])

    # Jinja (--jinja): True means enable
    jinja = settings.get("jinja")
    if jinja is True:
        cmd.append("--jinja")

    # Vision / mmproj: When enabled, add --mmproj flag pointing to first mmproj*.gguf found in the same directory as the model
    vision = settings.get("vision")
    if vision is True:
        model_dir = os.path.dirname(model_path)
        # Scan for any mmproj file matching mmproj*.gguf pattern
        mmproj_files = [f for f in os.listdir(model_dir)
                        if f.lower().startswith("mmproj") and f.endswith(".gguf")]
        if not mmproj_files:
            print(f"[WARNING] Vision enabled but no mmproj file found in {model_dir}")
        else:
            mmproj_path = os.path.join(model_dir, sorted(mmproj_files)[0])
            cmd.extend(["--mmproj", mmproj_path])

    # --- Server Settings ---
    cmd.extend(["--host", host])
    cmd.extend(["--port", str(port)])

    cmd.extend(["--no-webui"])
    cmd.extend(["--n-predict", str(-1)])
    cmd.extend(["-mg", str(0)])

    # MTP (Multi-Token Prediction): --spec-type draft-mtp when enabled
    mtp = settings.get("mtp")
    if mtp is True:
        cmd.extend(["--spec-type", "draft-mtp"])
        draft_n_max = settings.get("draft_n_max")
        if draft_n_max is not None and isinstance(draft_n_max, int) and draft_n_max > 0:
            cmd.extend(["--spec-draft-n-max", str(draft_n_max)])

    # Tensor Split: multi-GPU layer distribution (--tensor-split)
    tensor_split = settings.get("tensor_split")
    if tensor_split is not None and isinstance(tensor_split, list) and len(tensor_split) >= 2:
        # Format each value: use int representation when .0 (e.g. 60 instead of 60.0), otherwise keep decimal
        split_str = ",".join(_format_number(v) for v in tensor_split)
        cmd.extend(["--tensor-split", split_str])

    #cmd.extend(["--no-slots"])
    cmd.extend(["--swa-full"])

    if api_key:
        cmd.extend(["--api-key", api_key])

    return cmd


def _listen_for_stop_key(stop_flag: list, stop_event: threading.Event) -> None:
    """Background thread that listens for 'q' key press and sets the stop flag."""
    while not stop_event.is_set():
        if msvcrt.kbhit():
            char = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if char == 'q':
                stop_flag[0] = True
                return
        time.sleep(0.05)


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
        # Use a mutable list as shared flag between threads
        stop_flag = [False]  # index is set by keyboard listener thread

        # Start keyboard listener daemon thread
        stop_event = threading.Event()
        kb_thread = threading.Thread(target=_listen_for_stop_key, args=(stop_flag, stop_event), daemon=True)
        kb_thread.start()

        # Start the server process
        process = subprocess.Popen(cmd)

        # Wait for either the process to exit or user to press 'q'
        while True:
            if stop_flag[0]:
                print("\n\n[INFO] User requested stop (pressed 'q'). Stopping server...")
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
                user_stopped_via_q = True
                break

            # Check if process is still running
            retcode = process.poll()
            if retcode is not None:
                break

            time.sleep(0.05)  # Avoid busy-waiting too much

        stop_event.set()
        kb_thread.join(timeout=1)

    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user (Ctrl+C).")
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
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

    Returns (model_key, model_settings) tuple with preset overrides applied.
    """
    print(f"\n[OK] Selected: {selected_model['display']}")

    # Get model key for config storage (use full path as unique key)
    model_key = selected_model["path"]

    # Check if model has existing configuration for this preset (including Coder fallback for Commit)
    base_settings = get_model_config(config, model_key, preset_slug)
    has_existing = has_preset_config(config, model_key, preset_slug)

    # Apply preset-specific overrides (e.g., Commit forces thinking off)
    model_settings = apply_preset_overrides(preset_slug, base_settings)

    # Display current settings for the new model
    display_settings(model_settings)

    if has_existing:
        print("This model has existing configuration.")
    else:
        print("Using default configuration.")

    return model_key, model_settings


def switch_preset(config: Dict[str, Any], models: List[Dict[str, str]],
                  selected_model: Dict[str, str], model_key: str, current_preset: str) -> Tuple[str, Dict[str, Any]]:
    """Display preset selection menu and return (new_preset_slug, new_settings)."""
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

        # If user selects the same preset, confirm and keep it
        return selected_slug, get_model_config(config, model_key, selected_slug)


def clear_screen():
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title: str):
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

    # Get last used model from config (if any)
    last_model_key = config.get("last_model", None)
    last_model = None
    if last_model_key:
        for m in models:
            if m["path"] == last_model_key:
                last_model = m
                break

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

            # Re-apply overrides and display
            effective_settings = apply_preset_overrides(current_preset, model_settings)
            display_settings(effective_settings)
        elif action == "3":
            # Switch preset
            new_preset, base_settings = switch_preset(
                config, models, selected_model, model_key, current_preset
            )
            if new_preset != current_preset:
                print(f"\n[OK] Preset switched to {PRESETS[new_preset]['display']}")

            # Apply preset overrides and update state
            effective_settings = apply_preset_overrides(new_preset, base_settings)
            display_settings(effective_settings)

            # Update current preset reference (don't reassign model_settings directly;
            # the base settings may differ from what's displayed due to overrides).
            # We store both: the raw saved settings and compute effective on-the-fly.
            current_preset = new_preset
            model_settings = apply_model_selection(selected_model, config, current_preset)[1]

        elif action == "4":
            # Return to model selection (empty input reuses current model)
            new_model = select_model(models, allow_cancel=True, last_model=selected_model)
            if new_model is None:
                print("Model change cancelled.")
                continue

            selected_model = new_model
            _mk = selected_model["path"]
            # Determine the best preset for this model
            current_preset = best_preset_for_model(config, _mk, current_preset)
            if PRESETS[current_preset]["display"]:
                print(f"\n[INFO] Using preset: {PRESETS[current_preset]['display']}")

            model_key, model_settings = apply_model_selection(selected_model, config, current_preset)
        elif action == "5":
            print("\nExiting...")
            return
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
