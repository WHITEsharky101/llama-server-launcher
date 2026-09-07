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
