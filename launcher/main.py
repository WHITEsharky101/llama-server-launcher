"""Main entry point: model/preset selection loop and server launching."""

from typing import Any, Dict

from launcher import config
from launcher import command as cmd_builder
from launcher import models as models_mod
from launcher import presets
from launcher import server as server_mod
from launcher import settings as settings_mod
from launcher import ui
from launcher.models import Model


def _persist_state(cfg: Dict[str, Any], model_key: str, preset_slug: str) -> None:
    """Save last used model and preset to config so the next launch restores them."""
    cfg["last_model"] = model_key
    cfg["last_preset"] = preset_slug
    config.save_config(cfg)


def _handle_launch(
    cfg: Dict[str, Any], model_key: str, selected_model: Model,
    model_settings: Dict[str, Any], current_preset: str, host: str, port: int,
) -> None:
    """Build and launch the server for the current model/preset, then return to menu."""
    preset_display = presets.PRESETS[current_preset].display

    effective_settings = presets.apply_preset_overrides(current_preset, model_settings)
    api_key = presets.preset_api_key(cfg, current_preset)

    ui.print_server_config(preset_display, host, port, selected_model.display)

    _persist_state(cfg, model_key, current_preset)

    cmd = cmd_builder.build_command(
        model_path=models_mod.get_model_path(selected_model.path),
        settings=effective_settings,
        host=host,
        port=port,
        api_key=api_key,
    )

    if cmd:
        server_mod.launch_server(cmd)

    print("\n[OK] Server stopped.")


def _handle_edit_settings(
    cfg: Dict[str, Any], model_key: str, model_settings: Dict[str, Any], current_preset: str
) -> Dict[str, Any]:
    """Interactive settings edit + save for the current preset. Returns updated settings."""
    model_settings = settings_mod.edit_settings(model_settings)
    cfg = presets.save_model_config(cfg, model_key, current_preset, model_settings)
    settings_mod.display_settings(
        presets.apply_preset_overrides(current_preset, model_settings)
    )
    return model_settings


def _handle_switch_preset(
    cfg: Dict[str, Any], model_key: str, current_preset: str
):
    """Preset switch prompt; returns (new_preset_slug, base_settings)."""
    new_preset, base_settings = ui.switch_preset(cfg, model_key, current_preset)
    if new_preset != current_preset:
        print(f"\n[OK] Preset switched to {presets.PRESETS[new_preset].display}")

    settings_mod.display_settings(presets.apply_preset_overrides(new_preset, base_settings))

    _persist_state(cfg, model_key, new_preset)

    return new_preset, base_settings


def _handle_change_model(
    cfg: Dict[str, Any], models: list, selected_model: Model, current_preset: str
):
    """Model re-selection; returns (new_selected_model, model_key, model_settings, preset)
    or None if the user cancelled."""
    new_model = ui.select_model(models, allow_cancel=True, last_model=selected_model)
    if new_model is None:
        print("Model change cancelled.")
        return None

    new_model_key = new_model.path
    new_preset = presets.best_preset_for_model(cfg, new_model_key, current_preset)
    print(f"\n[INFO] Using preset: {presets.PRESETS[new_preset].display}")

    model_key, model_settings = ui.apply_model_selection(new_model, cfg, new_preset)

    _persist_state(cfg, model_key, new_preset)

    return new_model, model_key, model_settings, new_preset


def main():
    """Main entry point for the launcher."""
    ui.clear_screen()
    ui.print_header("Llama.cpp Server Launcher")

    cfg = config.load_config()
    host = cfg.get("host", config.DEFAULT_CONFIG["host"])
    port = cfg.get("port", config.DEFAULT_CONFIG["port"])

    # Ensure api_keys structure exists (migrate legacy single key to new format)
    if "api_key" in cfg and "api_keys" not in cfg:
        legacy = cfg.pop("api_key")
        cfg["api_keys"] = {"rp": legacy, "coder": legacy}
        print("[INFO] Migrated API key — please verify Coder preset key.")

    print("\n[INFO] Scanning for GGUF models...")
    models = models_mod.scan_models()

    if not models:
        if not config.MODELS_DIR:
            print("[ERROR] MODELS_DIR is not set. Copy .env.example to .env and set MODELS_DIR.")
        else:
            print(f"[ERROR] No GGUF models found in {config.MODELS_DIR}")
            print("Please place your models in the following structure:")
            print(f"  {config.MODELS_DIR}/<author>/<model_name>.gguf")
        input("\nPress Enter to exit...")
        return

    print(f"[OK] Found {len(models)} model(s).")

    _last_key = cfg.get("last_model")
    last_model = next((m for m in models if m.path == _last_key), None) if _last_key else None

    current_preset = cfg.get("last_preset", presets.DEFAULT_PRESET)
    if current_preset not in presets.PRESETS:
        current_preset = presets.DEFAULT_PRESET

    selected_model = ui.select_model(models, allow_cancel=False, last_model=last_model)

    current_preset = presets.best_preset_for_model(cfg, selected_model.path, current_preset)
    print(f"\n[INFO] Using preset: {presets.PRESETS[current_preset].display}")

    model_key, model_settings = ui.apply_model_selection(selected_model, cfg, current_preset)

    while True:
        preset_display = presets.PRESETS[current_preset].display
        print(f"\n--- Preset: {preset_display} ---")
        print("What would you like to do?")
        print("  1. Use current settings and launch")
        print("  2. Modify settings")
        print("  3. Switch preset (RP / Coder / Commit)")
        print("  4. Change model")
        print("  5. Exit")

        action = input("\nSelect (1/2/3/4/5): ").strip()

        if action == "1":
            _handle_launch(cfg, model_key, selected_model, model_settings, current_preset, host, port)

        elif action == "2":
            model_settings = _handle_edit_settings(cfg, model_key, model_settings, current_preset)

        elif action == "3":
            # Switch preset — use base (raw) settings so edits save correctly.
            # Overrides are only applied at launch and display time.
            current_preset, model_settings = _handle_switch_preset(cfg, model_key, current_preset)

        elif action == "4":
            result = _handle_change_model(cfg, models, selected_model, current_preset)
            if result is None:
                continue
            selected_model, model_key, model_settings, current_preset = result

        elif action == "5":
            _persist_state(cfg, model_key, current_preset)

            print("\nExiting...")
            return
        else:
            print("Invalid choice. Try again.")
