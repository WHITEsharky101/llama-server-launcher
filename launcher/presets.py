"""Preset (RP / Coder / Commit) definitions and preset-specific config logic.

Preset behavior is data-driven: each Preset declares which api_keys field it uses,
which settings it forces at launch, and which other preset it falls back to.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from launcher import config


@dataclass(frozen=True)
class Preset:
    slug: str
    display: str
    api_key_field: str
    # Settings forced to these values at launch time (e.g. Commit disables thinking).
    forced_overrides: Tuple[Tuple[str, Any], ...] = ()
    # Preset to fall back to when this one has no saved config (e.g. Commit -> Coder).
    fallback_slug: Optional[str] = None


# Insertion order defines menu order in the UI.
PRESETS: Dict[str, Preset] = {
    "rp": Preset(slug="rp", display="RP", api_key_field="rp"),
    "coder": Preset(slug="coder", display="Coder", api_key_field="coder"),
    "commit": Preset(
        slug="commit",
        display="Commit",
        api_key_field="coder",  # commit shares the coder key
        forced_overrides=(("thinking", False), ("p_thinking", False)),
        fallback_slug="coder",  # commit falls back to coder settings
    ),
}

# Default slug when none is configured / recognized.
DEFAULT_PRESET = "rp"


def get_preset(slug: str) -> Preset:
    """Return the Preset for *slug*, falling back to DEFAULT_PRESET for unknown slugs."""
    return PRESETS.get(slug, PRESETS[DEFAULT_PRESET])


def preset_api_key(cfg: Dict[str, Any], preset_slug: str) -> str:
    """Return the API key for a given preset.

    Unknown slugs conservatively fall back to the 'coder' key field (same as the
    original implementation's default)."""
    api_keys = cfg.get("api_keys", {})
    preset = PRESETS.get(preset_slug)
    key_field = preset.api_key_field if preset is not None else "coder"
    return api_keys.get(key_field, "")


def apply_preset_overrides(preset_slug: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Apply preset-specific overrides to settings.

    Returns a new dict with overrides applied (does not mutate the original)."""
    result = copy.deepcopy(settings)
    for key, value in get_preset(preset_slug).forced_overrides:
        result[key] = value
    return result


def get_model_config(cfg: Dict[str, Any], model_key: str, preset_slug: str) -> Dict[str, Any]:
    """Get model-specific config for a given preset, falling back to defaults.

    Falls back to the preset's fallback_slug settings if no direct config exists
    (e.g. Commit uses Coder settings when no commit-specific config is saved).
    """
    base_defaults = cfg.get("defaults", copy.deepcopy(config.DEFAULT_CONFIG["defaults"]))
    models = cfg.get("models", {})
    model_cfg: Dict[str, Any] = {} if not isinstance(models.get(model_key), dict) else models[model_key]

    preset = get_preset(preset_slug)
    for candidate_slug in (preset_slug, preset.fallback_slug):
        if candidate_slug and candidate_slug in model_cfg:
            merged = base_defaults.copy()
            merged.update(model_cfg[candidate_slug])
            return merged

    return base_defaults.copy()


def has_preset_config(cfg: Dict[str, Any], model_key: str, preset_slug: str) -> bool:
    """Check whether a model+preset combo has explicit saved config.

    Also returns True if the preset's fallback preset has saved settings
    (e.g. Commit counts Coder settings since it falls back to them).
    """
    models = cfg.get("models", {})
    model_cfg = models.get(model_key)
    if not isinstance(model_cfg, dict):
        return False
    preset = get_preset(preset_slug)
    effective_slugs = {preset_slug} | ({preset.fallback_slug} if preset.fallback_slug else set())
    return any(slug in model_cfg for slug in effective_slugs)


def best_preset_for_model(cfg: Dict[str, Any], model_key: str, preferred: str) -> str:
    """Return the most appropriate preset slug for a given model.

    Priority order:
      1. The explicitly saved 'last_preset' (preferred), if it has config for this model.
      2. Any independent preset that has explicit config for this model (in menu order).
      3. Fallback to the preferred default.
    """
    # If preferred preset already has config, use it directly
    if has_preset_config(cfg, model_key, preferred):
        return preferred

    # Otherwise pick the first independent preset (in menu order) with saved settings.
    # Presets with a fallback (e.g. commit) are excluded since they mirror their fallback.
    for slug, preset in PRESETS.items():
        if preset.fallback_slug:
            continue
        if has_preset_config(cfg, model_key, slug):
            return slug

    return preferred


def save_model_config(
    cfg: Dict[str, Any], model_key: str, preset_slug: str, model_settings: Dict[str, Any]
) -> Dict[str, Any]:
    """Save model-specific configuration for a given preset.

    For presets with a fallback_slug (commit), settings are saved under the fallback
    preset's key ('coder'), with forced-override settings (thinking/p_thinking) preserved
    from existing saved config (not overwritten).
    """
    preset = get_preset(preset_slug)
    target_slug = preset.fallback_slug or preset_slug
    cfg.setdefault("models", {}).setdefault(model_key, {})

    settings_to_save = copy.deepcopy(model_settings)

    if preset.fallback_slug:
        # Preserve forced-override values from existing saved config
        for key, _value in preset.forced_overrides:
            settings_to_save.pop(key, None)
        existing = dict(cfg["models"][model_key].get(target_slug, {}))
        existing.update(settings_to_save)
        cfg["models"][model_key][target_slug] = existing
    else:
        cfg["models"][model_key][target_slug] = settings_to_save

    config.save_config(cfg)
    return cfg
