"""Smoke tests for launcher/main.py: launch path and state persistence."""

from launcher import main
from launcher import models as models_mod
from launcher import presets


def test_preset_display_lookup_is_attribute_not_index():
    """Regression: PRESETS values are Preset dataclasses (use .display, not ['display'])."""
    for slug, preset in presets.PRESETS.items():
        assert isinstance(preset.display, str) and preset.display


def test_handle_launch_builds_command_and_persists(monkeypatch):
    """Action '1' path: overrides applied, command built, state persisted, server launched."""
    captured = {}

    def fake_build(**kw):
        captured["kwargs"] = kw
        return ["fake", "exe"]

    monkeypatch.setattr(main.cmd_builder, "build_command", fake_build)
    monkeypatch.setattr(main.server_mod, "launch_server", lambda cmd: captured.setdefault("cmd", cmd))
    monkeypatch.setattr(main.config, "save_config", lambda cfg: captured.setdefault("saved", cfg))

    cfg = {"api_keys": {"rp": "k"}}
    model = models_mod.Model(display="author/model", path="author/model")
    settings = {"context": 4096, "thinking": True}

    main._handle_launch(cfg, "author/model", model, settings, "commit", "127.0.0.1", 5056)

    # Commit forces thinking off in the effective settings passed to build_command
    assert captured["kwargs"]["settings"]["thinking"] is False
    # API key resolved via the preset's key field (commit -> coder key)
    assert captured["kwargs"]["api_key"] == ""
    assert captured["cmd"] == ["fake", "exe"]
    assert captured["saved"]["last_model"] == "author/model"
    assert captured["saved"]["last_preset"] == "commit"


def test_handle_launch_with_coder_key(monkeypatch):
    captured = {}

    def fake_build(**kw):
        captured["kwargs"] = kw
        return ["fake", "exe"]

    monkeypatch.setattr(main.cmd_builder, "build_command", fake_build)
    monkeypatch.setattr(main.server_mod, "launch_server", lambda cmd: captured.setdefault("cmd", cmd))
    monkeypatch.setattr(main.config, "save_config", lambda cfg: None)

    cfg = {"api_keys": {"coder": "coder-key"}}
    model = models_mod.Model(display="a/m", path="a/m")
    main._handle_launch(cfg, "a/m", model, {}, "coder", "h", 1)
    assert captured["kwargs"]["api_key"] == "coder-key"


def test_persist_state_writes_last_model_and_preset(monkeypatch):
    captured = {}
    monkeypatch.setattr(main.config, "save_config", lambda cfg: captured.setdefault("cfg", cfg))
    cfg = {}
    main._persist_state(cfg, "a/b", "rp")
    assert captured["cfg"]["last_model"] == "a/b"
    assert captured["cfg"]["last_preset"] == "rp"
