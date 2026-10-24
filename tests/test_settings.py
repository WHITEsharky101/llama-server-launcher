"""Tests for launcher/settings.py: dependent-setting visibility and display."""

import io
from contextlib import redirect_stdout

from launcher import settings


def _capture(func, *args, **kwargs):
    """Run *func* capturing printed output; returns (result, output_text)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = func(*args, **kwargs)
    return result, buf.getvalue()


BASE_SETTINGS = {
    "context": 8192,
    "gpu_offload": 33,
    "thinking": True,
    "mtp": True,
    "vision": True,
    "p_thinking": True,
    "draft_n_max": 8,
    "image_min_tokens": 256,
}


# --- _is_visible: draft_n_max (MTP-dependent) ---

def test_draft_n_max_visible_when_mtp_on():
    assert settings._is_visible(dict(BASE_SETTINGS), "draft_n_max") is True


def test_draft_n_max_hidden_when_mtp_off():
    s = dict(BASE_SETTINGS)
    s["mtp"] = False
    assert settings._is_visible(s, "draft_n_max") is False


def test_draft_n_max_hidden_when_mtp_none():
    s = dict(BASE_SETTINGS)
    s["mtp"] = None
    assert settings._is_visible(s, "draft_n_max") is False


# --- _is_visible: p_thinking (Thinking-dependent) ---

def test_p_thinking_visible_when_thinking_on():
    assert settings._is_visible(dict(BASE_SETTINGS), "p_thinking") is True


def test_p_thinking_hidden_when_thinking_off():
    s = dict(BASE_SETTINGS)
    s["thinking"] = False
    assert settings._is_visible(s, "p_thinking") is False


def test_p_thinking_hidden_when_thinking_none():
    s = dict(BASE_SETTINGS)
    s["thinking"] = None
    assert settings._is_visible(s, "p_thinking") is False


# --- _is_visible: vision-dependent keys still work ---

def test_image_min_tokens_hidden_when_vision_off():
    s = dict(BASE_SETTINGS)
    s["vision"] = False
    assert settings._is_visible(s, "image_min_tokens") is False


def test_mmproj_offload_hidden_when_vision_none():
    s = dict(BASE_SETTINGS)
    s["vision"] = None
    assert settings._is_visible(s, "mmproj_offload") is False


# --- display_settings_menu ---

def test_menu_hides_draft_n_max_when_mtp_off():
    s = dict(BASE_SETTINGS)
    s["mtp"] = False
    _result, out = _capture(settings.display_settings_menu, s)
    assert "Draft N Max" not in out
    assert "MTP" in out


def test_menu_hides_p_thinking_when_thinking_off():
    s = dict(BASE_SETTINGS)
    s["thinking"] = False
    _result, out = _capture(settings.display_settings_menu, s)
    assert "Preserve Think" not in out


def test_menu_no_longer_prints_vision_notice():
    s = dict(BASE_SETTINGS)
    s["vision"] = False
    _result, out = _capture(settings.display_settings_menu, s)
    assert "Vision-dependent settings are hidden" not in out


def test_menu_shows_all_dependent_entries_when_prerequisites_on():
    _result, out = _capture(settings.display_settings_menu, dict(BASE_SETTINGS))
    assert "Draft N Max" in out
    assert "Preserve Think" in out
    assert "Image Min Tokens" in out
    assert "MMProj Offload" in out


# --- display_settings ---

def test_display_hides_draft_n_max_when_mtp_off():
    s = dict(BASE_SETTINGS)
    s["mtp"] = False
    _result, out = _capture(settings.display_settings, s)
    assert "Draft N Max" not in out


def test_display_hides_p_thinking_when_thinking_off():
    s = dict(BASE_SETTINGS)
    s["thinking"] = False
    _result, out = _capture(settings.display_settings, s)
    assert "Preserve Think" not in out


# --- formatting ---

def test_fmt_draft_n_max_returns_plain_value_when_visible():
    s = dict(BASE_SETTINGS)
    assert settings._fmt_draft_n_max(s) == "8"


def test_fmt_draft_n_max_none_shows_none():
    s = dict(BASE_SETTINGS)
    s["draft_n_max"] = None
    assert settings._fmt_draft_n_max(s) == "None"


# --- edit_settings: hidden entries are blocked ---

def test_edit_blocks_hidden_draft_n_max_when_mtp_off(monkeypatch):
    s = dict(BASE_SETTINGS)
    s["mtp"] = False
    original = s["draft_n_max"]

    # Simulate user typing "13" (Draft N Max) then "0" (done)
    inputs = iter(["13", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    _result, out = _capture(settings.edit_settings, s)
    assert "available only when MTP is on" in out
    assert s["draft_n_max"] == original  # value untouched


def test_edit_blocks_hidden_p_thinking_when_thinking_off(monkeypatch):
    s = dict(BASE_SETTINGS)
    s["thinking"] = False
    original = s["p_thinking"]
    inputs = iter(["21", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    _result, out = _capture(settings.edit_settings, s)
    assert "available only when Thinking is on" in out
    assert s["p_thinking"] == original  # value untouched


# --- custom_flags ---

def test_custom_flags_in_settings_info():
    assert ("27", "custom_flags", "Custom Flags") in settings.SETTINGS_INFO


def test_custom_flags_in_display_sections():
    all_keys = [k for _title, keys in settings._DISPLAY_SECTIONS for k in keys]
    assert "custom_flags" in all_keys


def test_fmt_custom_flags_shows_value():
    assert settings._fmt_custom_flags({"custom_flags": "--swa-full --cache-type-k q8_0"}) == "--swa-full --cache-type-k q8_0"


def test_fmt_custom_flags_none_shows_disabled():
    for value in (None, "", "   "):
        assert settings._fmt_custom_flags({"custom_flags": value}) == "disabled"


def test_menu_shows_custom_flags_entry():
    _result, out = _capture(settings.display_settings_menu, dict(BASE_SETTINGS))
    assert "Custom Flags" in out


def test_menu_shows_custom_flags_current_value():
    s = dict(BASE_SETTINGS)
    s["custom_flags"] = "--swa-full"
    _result, out = _capture(settings.display_settings_menu, s)
    assert "Custom Flags" in out
    assert "--swa-full" in out


def test_display_shows_custom_flags_section_entry():
    s = dict(BASE_SETTINGS)
    s["custom_flags"] = "--swa-full"
    _result, out = _capture(settings.display_settings, s)
    assert "Custom Flags:" in out
    assert "--swa-full" in out


def test_display_shows_custom_flags_disabled_when_unset():
    s = dict(BASE_SETTINGS)
    s["custom_flags"] = None
    _result, out = _capture(settings.display_settings, s)
    assert "Custom Flags:" in out
    assert "disabled" in out


def test_edit_custom_flags_stores_case_preserved(monkeypatch):
    s = dict(BASE_SETTINGS)
    s["custom_flags"] = None
    # Simulate user typing "27" (Custom Flags), the flag string, then "0" (done)
    inputs = iter(["27", "--cache-type-k Q8_0 --swa-full", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    result, out = _capture(settings.edit_settings, s)
    # Case preserved (unlike the generic free-text editor which lowercases)
    assert result["custom_flags"] == "--cache-type-k Q8_0 --swa-full"
    assert "[OK] Custom Flags set to: --cache-type-k Q8_0 --swa-full" in out


def test_edit_custom_flags_empty_clears(monkeypatch):
    s = dict(BASE_SETTINGS)
    s["custom_flags"] = "--swa-full"
    inputs = iter(["27", "", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    result, _out = _capture(settings.edit_settings, s)
    assert result["custom_flags"] is None


def test_edit_custom_flags_collapses_whitespace(monkeypatch):
    s = dict(BASE_SETTINGS)
    inputs = iter(["27", "  --swa-full    --cache-type-k  q8_0  ", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    result, _out = _capture(settings.edit_settings, s)
    assert result["custom_flags"] == "--swa-full --cache-type-k q8_0"


# --- model_alias ---

def test_model_alias_in_settings_info():
    assert ("28", "model_alias", "Model Alias") in settings.SETTINGS_INFO


def test_model_alias_in_display_sections():
    all_keys = [k for _title, keys in settings._DISPLAY_SECTIONS for k in keys]
    assert "model_alias" in all_keys


def test_fmt_model_alias_shows_value():
    assert settings._fmt_model_alias({"model_alias": "qwen3-coder"}) == "qwen3-coder"


def test_fmt_model_alias_unset_shows_auto():
    for value in (None, "", "   "):
        assert settings._fmt_model_alias({"model_alias": value}) == "auto (model name)"


def test_menu_shows_model_alias_entry():
    _result, out = _capture(settings.display_settings_menu, dict(BASE_SETTINGS))
    assert "Model Alias" in out


def test_display_shows_model_alias_auto_when_unset():
    s = dict(BASE_SETTINGS)
    s["model_alias"] = None
    _result, out = _capture(settings.display_settings, s)
    assert "Model Alias:" in out
    assert "auto (model name)" in out


def test_display_shows_model_alias_value():
    s = dict(BASE_SETTINGS)
    s["model_alias"] = "qwen3-coder"
    _result, out = _capture(settings.display_settings, s)
    assert "Model Alias:" in out
    assert "qwen3-coder" in out


def test_edit_model_alias_stores_case_preserved(monkeypatch):
    s = dict(BASE_SETTINGS)
    s["model_alias"] = None
    # Simulate user typing "28" (Model Alias), the alias, then "0" (done)
    inputs = iter(["28", "Qwen3.8-27B-Coder", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    result, out = _capture(settings.edit_settings, s)
    # Case preserved (unlike the generic free-text editor which lowercases)
    assert result["model_alias"] == "Qwen3.8-27B-Coder"
    assert "[OK] Model Alias set to: Qwen3.8-27B-Coder" in out


def test_edit_model_alias_empty_clears(monkeypatch):
    s = dict(BASE_SETTINGS)
    s["model_alias"] = "old-alias"
    inputs = iter(["28", "", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    result, _out = _capture(settings.edit_settings, s)
    assert result["model_alias"] is None


def test_edit_model_alias_trims_whitespace(monkeypatch):
    s = dict(BASE_SETTINGS)
    inputs = iter(["28", "  my-alias  ", "0"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    result, _out = _capture(settings.edit_settings, s)
    assert result["model_alias"] == "my-alias"
