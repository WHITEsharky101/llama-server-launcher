"""Tests for launcher/presets.py: preset metadata, overrides, fallbacks, config lookup/save."""

from launcher import presets


def make_config(models=None):
    return {"defaults": {"context": 32768, "temp": 1}, "models": models or {}}


# --- preset metadata ---

def test_preset_definitions():
    assert presets.PRESETS["rp"].display == "RP"
    assert presets.PRESETS["rp"].api_key_field == "rp"
    assert presets.PRESETS["rp"].forced_overrides == ()
    assert presets.PRESETS["rp"].fallback_slug is None

    assert presets.PRESETS["coder"].api_key_field == "coder"
    assert presets.PRESETS["coder"].fallback_slug is None

    commit = presets.PRESETS["commit"]
    assert commit.api_key_field == "coder"
    assert commit.fallback_slug == "coder"
    assert dict(commit.forced_overrides) == {"thinking": False, "p_thinking": False}


def test_get_preset_unknown_falls_back_to_default():
    assert presets.get_preset("nope").slug == presets.DEFAULT_PRESET


# --- preset_api_key ---

def test_preset_api_key_mapping():
    cfg = {"api_keys": {"rp": "k-rp", "coder": "k-coder"}}
    assert presets.preset_api_key(cfg, "rp") == "k-rp"
    assert presets.preset_api_key(cfg, "coder") == "k-coder"
    assert presets.preset_api_key(cfg, "commit") == "k-coder"  # shares coder key
    # Unknown slug falls back to the 'coder' key field (original behavior)
    assert presets.preset_api_key(cfg, "missing") == "k-coder"


def test_preset_api_key_missing_field():
    cfg = {"api_keys": {"rp": "k-rp"}}
    assert presets.preset_api_key(cfg, "rp") == "k-rp"
    # 'coder' field absent -> empty string
    assert presets.preset_api_key(cfg, "coder") == ""


# --- apply_preset_overrides ---

def test_commit_forces_thinking_off_without_mutation():
    settings = {"thinking": True, "p_thinking": True, "temp": 0.7}
    effective = presets.apply_preset_overrides("commit", settings)
    assert effective["thinking"] is False
    assert effective["p_thinking"] is False
    assert effective["temp"] == 0.7
    # Original untouched
    assert settings["thinking"] is True
    assert settings["p_thinking"] is True


def test_rp_has_no_overrides():
    settings = {"thinking": True}
    assert presets.apply_preset_overrides("rp", settings) == settings


# --- get_model_config ---

def test_get_model_config_uses_defaults_when_no_saved():
    cfg = make_config()
    assert presets.get_model_config(cfg, "a/b", "rp") == cfg["defaults"]


def test_get_model_config_merges_saved_over_defaults():
    cfg = make_config({"a/b": {"rp": {"context": 16384}}})
    merged = presets.get_model_config(cfg, "a/b", "rp")
    assert merged["context"] == 16384
    assert merged["temp"] == 1  # default preserved


def test_commit_falls_back_to_coder_settings():
    cfg = make_config({"a/b": {"coder": {"context": 8192}}})
    merged = presets.get_model_config(cfg, "a/b", "commit")
    assert merged["context"] == 8192
    assert merged["temp"] == 1


def test_commit_direct_config_wins_over_coder_fallback():
    cfg = make_config({"a/b": {"coder": {"context": 8192}, "commit": {"context": 4096}}})
    merged = presets.get_model_config(cfg, "a/b", "commit")
    assert merged["context"] == 4096


def test_get_model_config_ignores_non_dict_model_entry():
    cfg = make_config({"a/b": "garbage"})
    assert presets.get_model_config(cfg, "a/b", "rp") == cfg["defaults"]


# --- has_preset_config ---

def test_has_preset_config_basic():
    cfg = make_config({"a/b": {"rp": {"context": 1}}})
    assert presets.has_preset_config(cfg, "a/b", "rp") is True
    assert presets.has_preset_config(cfg, "a/b", "coder") is False
    assert presets.has_preset_config(cfg, "unknown", "rp") is False


def test_has_preset_config_commit_counts_coder():
    cfg = make_config({"a/b": {"coder": {"context": 1}}})
    assert presets.has_preset_config(cfg, "a/b", "commit") is True
    assert presets.has_preset_config(cfg, "a/b", "rp") is False


# --- best_preset_for_model ---

def test_best_preset_prefers_preferred_when_configured():
    cfg = make_config({"a/b": {"coder": {"context": 1}}})
    # preferred=coder has config -> kept
    assert presets.best_preset_for_model(cfg, "a/b", "coder") == "coder"


def test_best_preset_falls_back_to_any_configured():
    cfg = make_config({"a/b": {"coder": {"context": 1}}})
    # preferred=rp has no config -> coder wins
    assert presets.best_preset_for_model(cfg, "a/b", "rp") == "coder"


def test_best_preset_returns_preferred_when_nothing_saved():
    cfg = make_config()
    assert presets.best_preset_for_model(cfg, "a/b", "rp") == "rp"


def test_best_preset_excludes_fallback_presets():
    # Only commit config exists (mirrors coder) -> should NOT be picked, falls back to preferred
    cfg = make_config({"a/b": {"commit": {"context": 1}}})
    assert presets.best_preset_for_model(cfg, "a/b", "rp") == "rp"


# --- save_model_config ---

def test_save_model_config_stores_under_preset(monkeypatch):
    saved = {}
    monkeypatch.setattr(presets.config, "save_config", lambda cfg, config_file=None: saved.update(cfg))
    cfg = make_config()
    result = presets.save_model_config(cfg, "a/b", "rp", {"context": 123})
    assert result["models"]["a/b"]["rp"] == {"context": 123}
    assert saved is not None


def test_save_model_config_commit_stores_under_coder(monkeypatch):
    saved = {}
    monkeypatch.setattr(presets.config, "save_config", lambda cfg, config_file=None: saved.update(cfg))
    cfg = make_config({"a/b": {"coder": {"thinking": True, "context": 999}}})
    result = presets.save_model_config(cfg, "a/b", "commit", {"context": 123})
    # Saved under 'coder', forced-override keys preserved from existing coder config
    assert "commit" not in result["models"]["a/b"]
    assert result["models"]["a/b"]["coder"]["thinking"] is True
    assert result["models"]["a/b"]["coder"]["context"] == 123
