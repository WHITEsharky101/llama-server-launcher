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
from pathlib import Path
from typing import Optional, Dict, Any, List

# === Configuration Paths ===
MODELS_DIR = r"C:\Users\WHITEsharky\.lmstudio\models"
LLAMA_CPP_DIR = r"C:\Users\WHITEsharky\.lmstudio\extensions\backends\turboquant-plus-tqp-v0.1.1-windows-x64-cuda12.4"
CONFIG_FILE = Path(__file__).parent / "llama_server_config.json"

# === Default Configuration ===
DEFAULT_CONFIG = {
    "host": "192.168.1.177",
    "port": 5056,
    "api_key": "ek-cvpxgU0aMOLzFhbwOs3XcVgH8jyWpHqdX7cRRIlbtzwKtF7LvV",
    "defaults": {
        "context": 32768,
        "gpu_offload": 99,
        "cpu_moe": None,
        "threads": 8,
        "batch_size": 512,
        "mmap": False,
        "flash_attention": True,
        "k_quant": "turbo3",
        "v_quant": "turbo3",
        "temp": 1,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "presence_penalty": None,
        "thinking": None,
        "jinja": None,
        "vision": None
    }
}

# === KV Cache Quantization Options ===
KV_QUANT_OPTIONS = ["turbo4", "turbo3", "turbo2", "q8_0", "q4_0", "none"]

# === Model Parameter Settings (for info display) ===
MODEL_PARAM_SETTINGS = ["context", "gpu_offload", "cpu_moe", "threads", "batch_size", "mmap", "flash_attention", "k_quant", "v_quant"]

# === Generation Settings ===
GENERATION_SETTINGS = ["temp", "top_k", "top_p", "min_p", "repeat_penalty", "presence_penalty", "thinking", "jinja"]

# === Settings Menu Definition (shared between display_settings_menu and edit_settings) ===
SETTINGS_INFO = [
    ("1", "Context", "context"),
    ("2", "GPU Offload", "gpu_offload"),
    ("3", "CPU MOE", "cpu_moe"),
    ("4", "Threads", "threads"),
    ("5", "Batch Size", "batch_size"),
    ("6", "mmap", "mmap"),
    ("7", "Flash Attention", "flash_attention"),
    ("8", "K Quant", "k_quant"),
    ("9", "V Quant", "v_quant"),
    ("10", "Temp", "temp"),
    ("11", "Top K", "top_k"),
    ("12", "Top P", "top_p"),
    ("13", "Min P", "min_p"),
    ("14", "Repeat Penalty", "repeat_penalty"),
    ("15", "Presence Penalty", "presence_penalty"),
    ("16", "Thinking", "thinking"),
    ("17", "Jinja", "jinja"),
    ("18", "Vision", "vision"),
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


def get_model_config(config: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    """Get model-specific config, falling back to defaults."""
    models = config.get("models", {})
    if model_key in models:
        model_defaults = config.get("defaults", DEFAULT_CONFIG["defaults"]).copy()
        model_defaults.update(models[model_key])
        return model_defaults
    return config.get("defaults", DEFAULT_CONFIG["defaults"]).copy()


def save_model_config(config: Dict[str, Any], model_key: str, model_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Save model-specific configuration."""
    if "models" not in config:
        config["models"] = {}
    config["models"][model_key] = model_settings
    save_config(config)
    return config


def display_settings(settings: Dict[str, Any]) -> None:
    """Display current model settings in a readable format."""
    print("\n" + "=" * 50)
    print("Model Settings:")
    print("=" * 50)
    print("\n--- Model Parameters ---")
    print(f"  Context:              {settings.get('context', 'N/A')}")
    print(f"  GPU Offload:          {settings.get('gpu_offload', 'N/A')}")
    print(f"  CPU MOE:             {settings.get('cpu_moe', 'N/A')}")
    print(f"  Threads:             {settings.get('threads', 'N/A')}")
    print(f"  Batch Size:          {settings.get('batch_size', 'N/A')}")
    print(f"  mmap:                {settings.get('mmap', 'N/A')}")
    print(f"  Flash Attention:     {settings.get('flash_attention', 'N/A')}")
    print(f"  K Quant:             {settings.get('k_quant', 'N/A')}")
    print(f"  V Quant:             {settings.get('v_quant', 'N/A')}")
    print("\n--- Generation Settings ---")
    print(f"  Temp:                {settings.get('temp', 'N/A')}")
    print(f"  Top K:               {settings.get('top_k', 'N/A')}")
    print(f"  Top P:               {settings.get('top_p', 'N/A')}")
    print(f"  Min P:               {settings.get('min_p', 'N/A')}")
    print(f"  Repeat Penalty:      {settings.get('repeat_penalty', 'N/A')}")
    print(f"  Presence Penalty:    {settings.get('presence_penalty', 'N/A')}")
    print(f"  Thinking:            {settings.get('thinking', 'N/A')}")
    print(f"  Jinja:               {settings.get('jinja', 'N/A')}")
    print(f"  Vision:              {settings.get('vision', 'N/A')}")
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
        toggle_settings = ["mmap", "flash_attention", "thinking", "jinja", "vision"]
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
    ("temp", "--temp"),
    ("k_quant", "-ctk"),
    ("v_quant", "-ctv"),
    ("top_k", "--top-k"),
    ("top_p", "--top-p"),
    ("min_p", "--min-p"),
    ("repeat_penalty", "--repeat-penalty"),
    ("presence_penalty", "--presence-penalty"),
]


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

    # Thinking (--chat-template-kwargs): True/False for enable_thinking
    thinking = settings.get("thinking")
    if thinking is True:
        cmd.extend(["--chat-template-kwargs", '{"enable_thinking":true}'])
    elif thinking is False:
        cmd.extend(["--chat-template-kwargs", '{"enable_thinking":false}'])

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

    if api_key:
        cmd.extend(["--api-key", api_key])

    return cmd


def launch_server(cmd: List[str]) -> None:
    """Launch the llama server process."""
    if not cmd:
        print("[ERROR] No command to launch!")
        return

    print("\n" + "=" * 60)
    print("Launching Llama Server...")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server.\n")

    process = None
    try:
        process = subprocess.Popen(cmd)
        process.wait()  # Block until process exits
        if process.returncode is not None and process.returncode != 0:
            print(f"\n[INFO] Server process ended with code: {process.returncode}")
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user.")
        # Gracefully terminate the server process
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                process.kill()
    except FileNotFoundError:
        print(f"\n[ERROR] Executable not found: {cmd[0]}")
    except Exception as e:
        print(f"\n[ERROR] Failed to launch server: {e}")


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


def apply_model_selection(selected_model: Dict[str, str], config: Dict[str, Any]) -> tuple:
    """Apply model selection: load config, display settings.

    Returns (model_key, model_settings) tuple.
    """
    print(f"\n[OK] Selected: {selected_model['display']}")

    # Get model key for config storage (use full path as unique key)
    model_key = selected_model["path"]

    # Check if model has existing configuration
    model_settings = get_model_config(config, model_key)
    has_existing = model_key in config.get("models", {})

    # Display current settings for the new model
    display_settings(model_settings)

    if has_existing:
        print("This model has existing configuration.")
    else:
        print("Using default configuration.")

    return model_key, model_settings


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
    api_key = config.get("api_key", DEFAULT_CONFIG["api_key"])

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

    # Select model (empty input uses last model, if available)
    selected_model = select_model(models, allow_cancel=False, last_model=last_model)
    model_key, model_settings = apply_model_selection(selected_model, config)

    # Inner loop for settings; outer loop allows returning to model selection
    launching = False
    while not launching:
        print("\nWhat would you like to do?")
        print("  1. Use current settings and launch")
        print("  2. Modify settings")
        print("  3. Change model")
        print("  4. Exit")

        action = input("\nSelect (1/2/3/4): ").strip()

        if action == "1":
            # Show configuration and launch immediately
            print("\n" + "=" * 60)
            print("Server Configuration:")
            print(f"  Host:    {host}")
            print(f"  Port:    {port}")
            print(f"  Model:   {selected_model['display']}")
            print("=" * 60)
            launching = True
        elif action == "2":
            # Modify settings
            model_settings = edit_settings(model_settings)
            # Save updated settings
            config = save_model_config(config, model_key, model_settings)
            display_settings(model_settings)
        elif action == "3":
            # Return to model selection (empty input reuses current model)
            new_model = select_model(models, allow_cancel=True, last_model=selected_model)
            if new_model is None:
                print("Model change cancelled.")
                continue
            
            selected_model = new_model
            model_key, model_settings = apply_model_selection(selected_model, config)
        elif action == "4":
            print("\nExiting...")
            return
        else:
            print("Invalid choice. Try again.")

    # Save last used model to config before launching
    config["last_model"] = model_key
    save_config(config)

    # Build and launch command
    cmd = build_command(
        model_path=get_model_path(selected_model["path"]),
        settings=model_settings,
        host=host,
        port=port,
        api_key=api_key
    )

    if cmd:
        launch_server(cmd)


if __name__ == "__main__":
    main()
