"""Terminal UI helpers: screen/header utilities and interactive selection prompts."""

import os
from typing import Any, Dict, List, Optional, Tuple

from launcher.models import Model
from launcher.presets import PRESETS, apply_preset_overrides, get_model_config, has_preset_config
from launcher.settings import display_settings


# --- Terminal utilities ---

def clear_screen() -> None:
    """Clear the terminal screen (cross-platform)."""
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# --- Selection prompts ---

def select_model(
    models: List[Model], allow_cancel: bool = False, last_model: Optional[Model] = None
) -> Optional[Model]:
    """Display model list and prompt user to select one.

    If last_model is provided and the user enters an empty string, returns last_model.
    Returns the selected Model, or None if cancelled (when allow_cancel=True).
    """
    print("\n--- Select a model ---\n")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model.display}")

    parts = []
    if allow_cancel:
        parts.append("0 to cancel")
    if last_model is not None:
        parts.append(f"Enter for {last_model.display}")

    if parts:
        prompt = f"Select model by number (or {', '.join(parts)}): "
    else:
        prompt = "Select model by number: "

    while True:
        choice = input(prompt).strip()

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
            if choice != "":
                print("Invalid input. Please enter a number.")


def switch_preset(config: Dict[str, Any], model_key: str, current_preset: str) -> Tuple[str, Dict[str, Any]]:
    """Display preset selection menu and return (new_preset_slug, base_settings)."""
    print("\n--- Select a Preset ---\n")

    available = list(PRESETS.items())  # [("rp", Preset), ...]
    for i, (slug, preset) in enumerate(available, 1):
        marker = " [current]" if slug == current_preset else ""
        print(f"  {i}. {preset.display}{marker}")

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
            for slug, preset in available:
                if slug == lower_choice or preset.display.lower() == lower_choice:
                    selected_slug = slug
                    break
            if selected_slug is None:
                print("Invalid input. Please enter a number or preset name.")
                continue

        return selected_slug, get_model_config(config, model_key, selected_slug)


def apply_model_selection(
    selected_model: Model, config: Dict[str, Any], preset_slug: str
) -> Tuple[str, Dict[str, Any]]:
    """Apply model selection for a given preset: load config, display settings.

    Returns (model_key, base_settings) tuple WITHOUT overrides applied.
    Overrides are only used at launch time and for display in the main loop.
    This ensures that when saving under 'commit', we work with raw coder values
    rather than commit-forced ones.
    """
    print(f"\n[OK] Selected: {selected_model.display}")

    # Use full relative path as the unique config key
    model_key = selected_model.path

    base_settings = get_model_config(config, model_key, preset_slug)
    has_existing = has_preset_config(config, model_key, preset_slug)

    display_settings(apply_preset_overrides(preset_slug, base_settings))

    if has_existing:
        print("This model has existing configuration.")
    else:
        print("Using default configuration.")

    return model_key, base_settings


def print_server_config(preset_display: str, host: str, port: int, model_display: str) -> None:
    """Print the 'Server Configuration' banner shown right before launch."""
    print("\n" + "=" * 60)
    print("Server Configuration:")
    print(f"  Preset:    {preset_display}")
    print(f"  Host:      {host}")
    print(f"  Port:      {port}")
    print(f"  Model:     {model_display}")
    print("=" * 60)
