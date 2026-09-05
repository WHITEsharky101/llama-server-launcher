"""Tests for launcher/config.py: merge, dotenv parsing, API key resolution, load/save."""

import json
import os

from launcher import config


# --- deep_merge ---

def test_deep_merge_nested_dicts():
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "keep": "base"}
    override = {"a": 99, "nested": {"y": 3, "z": 4}, "add": "new"}
    merged = config.deep_merge(base, override)
    assert merged["a"] == 99
    assert merged["keep"] == "base"
    assert merged["nested"] == {"x": 1, "y": 3, "z": 4}
    assert merged["add"] == "new"


def test_deep_merge_does_not_mutate_base():
    base = {"nested": {"x": 1}}
    config.deep_merge(base, {"nested": {"x": 2}, "new": 3})
    assert base == {"nested": {"x": 1}}


# --- resolve_api_keys ---

def test_resolve_api_keys_fills_missing_from_env(monkeypatch):
    monkeypatch.setenv("API_KEY_rp", "rp-secret")
    monkeypatch.setenv("API_KEY_coder", "coder-secret")
    cfg = {"api_keys": {}}
    config.resolve_api_keys(cfg)
    assert cfg["api_keys"] == {"rp": "rp-secret", "coder": "coder-secret"}


def test_resolve_api_keys_does_not_override_existing(monkeypatch):
    monkeypatch.setenv("API_KEY_rp", "env-value")
    cfg = {"api_keys": {"rp": "saved-value"}}
    config.resolve_api_keys(cfg)
    assert cfg["api_keys"]["rp"] == "saved-value"


# --- load_dotenv ---

def test_load_dotenv_parses_comments_quotes_and_blanks(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment line\n\n"
        "PLAIN=hello\n"
        'QUOTED="world"\n'
        "SINGLE='ok'\n"
        "SPACES=  padded  \n",
        encoding="utf-8",
    )
    for key in ("PLAIN", "QUOTED", "SINGLE", "SPACES"):
        monkeypatch.delenv(key, raising=False)

    loaded = config.load_dotenv(env_file)
    assert loaded["PLAIN"] == "hello"
    assert loaded["QUOTED"] == "world"
    assert loaded["SINGLE"] == "ok"
    assert loaded["SPACES"] == "padded"
    assert os.environ["PLAIN"] == "hello"


def test_load_dotenv_missing_file_returns_empty(tmp_path):
    assert config.load_dotenv(tmp_path / "nope.env") == {}


# --- load / save config ---

def test_load_config_missing_file_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("API_KEY_rp", raising=False)
    monkeypatch.delenv("API_KEY_coder", raising=False)
    cfg = config.load_config(config_file=tmp_path / "missing.json")
    assert cfg["host"] == config.DEFAULT_CONFIG["host"]
    assert cfg["port"] == config.DEFAULT_CONFIG["port"]
    assert cfg["defaults"] == config.DEFAULT_CONFIG["defaults"]
    assert cfg["api_keys"] == {}


def test_load_config_merges_saved_over_defaults(tmp_path, monkeypatch):
    # API keys are resolved from env during load, so set env BEFORE load_config
    monkeypatch.setenv("API_KEY_rp", "from-env")
    saved = {"port": 9999, "defaults": {"context": 8192}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(saved), encoding="utf-8")

    cfg = config.load_config(config_file=path)
    assert cfg["port"] == 9999
    assert cfg["defaults"]["context"] == 8192
    assert cfg["defaults"]["gpu_offload"] == config.DEFAULT_CONFIG["defaults"]["gpu_offload"]
    assert cfg["api_keys"].get("rp") == "from-env"


def test_load_config_invalid_json_falls_back_to_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not valid json", encoding="utf-8")
    cfg = config.load_config(config_file=path)
    assert cfg["host"] == config.DEFAULT_CONFIG["host"]


def test_save_config_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    cfg = {"host": "127.0.0.1", "port": 1234, "models": {"a/b": {"rp": {"context": 1}}}}
    config.save_config(cfg, config_file=path)
    assert json.loads(path.read_text(encoding="utf-8")) == cfg


def test_default_config_keys_match_settings_menu():
    """Every key in the settings menu must exist in the defaults (and vice versa)."""
    from launcher.settings import SETTINGS_INFO
    menu_keys = {key for _num, key, _name in SETTINGS_INFO}
    assert menu_keys == set(config.DEFAULT_CONFIG["defaults"].keys())
