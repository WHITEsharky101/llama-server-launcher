"""Tests for launcher/command.py: command builder flag generation and ordering."""

import json

import pytest

from launcher import command
from launcher.config import DEFAULT_CONFIG


def default_settings():
    return dict(DEFAULT_CONFIG["defaults"])


def has_seq(cmd, seq):
    """Check that *seq* appears in *cmd* as a contiguous subsequence."""
    for i in range(len(cmd) - len(seq) + 1):
        if cmd[i:i + len(seq)] == seq:
            return True
    return False


@pytest.fixture
def fake_exe(monkeypatch):
    """Stub the server executable lookup so no filesystem is needed."""
    monkeypatch.setattr(command, "get_llama_server_path", lambda: r"C:\fake\llama-server.exe")
    return r"C:\fake\llama-server.exe"


# --- basic structure ---

def test_build_command_returns_empty_when_exe_missing(monkeypatch):
    monkeypatch.setattr(command, "get_llama_server_path", lambda: None)
    assert command.build_command("model.gguf", {}, "127.0.0.1", 5056, "") == []


def test_build_command_head_and_fixed_flags(fake_exe):
    cmd = command.build_command(r"C:\m\a.gguf", default_settings(), "10.0.0.5", 5056, "")
    assert cmd[0] == fake_exe
    assert cmd[1:3] == ["--model", r"C:\m\a.gguf"]
    # Fixed server settings block
    assert has_seq(cmd, ["--host", "10.0.0.5", "--port", "5056"])
    assert "--no-webui" in cmd
    assert has_seq(cmd, ["-mg", "0"])
    assert "--n-predict" not in cmd
    # Trailing fixed flags, in order
    assert "--no-slots" in cmd
    assert has_seq(cmd, ["--timeout", "30000"])
    assert has_seq(cmd, ["-n", "-1"])
    assert cmd.index("--no-slots") < cmd.index("--timeout") < cmd.index("-n")


def test_simple_params_present_with_defaults(fake_exe):
    """Every simple param with a truthy value in the defaults is emitted as flag+value.

    Derives expectations from SIMPLE_PARAM_MAP so it stays valid if defaults change."""
    s = default_settings()
    cmd = command.build_command("m.gguf", s, "h", 1, "")
    for key, flag in command.SIMPLE_PARAM_MAP:
        value = s.get(key)
        if value in (None, False, 0):
            continue
        assert has_seq(cmd, [flag, str(value)]), f"missing {flag} {value}"


def test_none_or_zero_params_are_omitted(fake_exe):
    s = default_settings()
    s["cpu_moe"] = None
    s["k_quant"] = None
    s["parallel"] = 0
    cmd = command.build_command("m.gguf", s, "h", 1, "")
    assert "--n-cpu-moe" not in cmd
    assert "-ctk" not in cmd
    assert "-np" not in cmd


# --- boolean flags ---

def test_mmap_false_adds_no_mmap(fake_exe):
    s = default_settings()
    s["mmap"] = False
    cmd = command.build_command("m.gguf", s, "h", 1, "")
    assert "--no-mmap" in cmd

    s["mmap"] = True
    cmd = command.build_command("m.gguf", s, "h", 1, "")
    assert "--no-mmap" not in cmd


def test_flash_attn_and_jinja_flags(fake_exe):
    s = default_settings()
    s["flash_attention"] = True
    s["jinja"] = True
    cmd = command.build_command("m.gguf", s, "h", 1, "")
    assert has_seq(cmd, ["-fa", "on"])
    assert "--jinja" in cmd


def test_chat_template_kwargs(fake_exe):
    s = default_settings()
    s["thinking"] = True
    s["p_thinking"] = False
    cmd = command.build_command("m.gguf", s, "h", 1, "")
    kwargs_idx = cmd.index("--chat-template-kwargs")
    kwargs = json.loads(cmd[kwargs_idx + 1])
    assert kwargs == {"enable_thinking": True, "preserve_thinking": False}


def test_chat_template_kwargs_omitted_when_none(fake_exe):
    s = default_settings()
    s["thinking"] = None
    s["p_thinking"] = None
    cmd = command.build_command("m.gguf", s, "h", 1, "")
    assert "--chat-template-kwargs" not in cmd


# --- vision ---

def test_vision_adds_mmproj(fake_exe, tmp_path):
    (tmp_path / "mmproj-blob.gguf").write_bytes(b"x")
    model_path = str(tmp_path / "model.gguf")
    s = default_settings()
    s["vision"] = True
    cmd = command.build_command(model_path, s, "h", 1, "")
    assert has_seq(cmd, ["--mmproj", str(tmp_path / "mmproj-blob.gguf")])


def test_vision_without_mmproj_warns(fake_exe, tmp_path, capsys):
    model_path = str(tmp_path / "model.gguf")
    s = default_settings()
    s["vision"] = True
    cmd = command.build_command(model_path, s, "h", 1, "")
    assert "--mmproj" not in cmd
    assert "[WARNING]" in capsys.readouterr().out


def test_image_min_tokens(fake_exe):
    s = default_settings()
    s["image_min_tokens"] = 256
    cmd = command.build_command("m.gguf", s, "h", 1, "")
    assert has_seq(cmd, ["--image-min-tokens", "256"])

    s["image_min_tokens"] = None
    cmd = command.build_command("m.gguf", s, "h", 1, "")
    assert "--image-min-tokens" not in cmd


# --- MTP / tensor split ---

def test_mtp_flags(fake_exe):
    s = default_settings()
    s["mtp"] = True
    s["draft_n_max"] = 16
    cmd = command.build_command("m.gguf", s, "h", 1, "")
    assert has_seq(cmd, ["--spec-type", "draft-mtp"])
    assert has_seq(cmd, ["--spec-draft-n-max", "16"])


def test_mtp_without_draft_n_max(fake_exe):
    s = default_settings()
    s["mtp"] = True
    s["draft_n_max"] = None
    cmd = command.build_command("m.gguf", s, "h", 1, "")
    assert has_seq(cmd, ["--spec-type", "draft-mtp"])
    assert "--spec-draft-n-max" not in cmd


def test_tensor_split_formatting(fake_exe):
    s = default_settings()
    s["tensor_split"] = [60.0, 40.0]
    cmd = command.build_command("m.gguf", s, "h", 1, "")
    assert has_seq(cmd, ["--tensor-split", "60,40"])

    s["tensor_split"] = [62.5, 37.5]
    cmd = command.build_command("m.gguf", s, "h", 1, "")
    assert has_seq(cmd, ["--tensor-split", "62.5,37.5"])

    s["tensor_split"] = None
    cmd = command.build_command("m.gguf", s, "h", 1, "")
    assert "--tensor-split" not in cmd


# --- api key ---

def test_api_key_appended(fake_exe):
    cmd = command.build_command("m.gguf", {}, "h", 1, "secret")
    assert cmd[-2:] == ["--api-key", "secret"]

    cmd = command.build_command("m.gguf", {}, "h", 1, "")
    assert "--api-key" not in cmd


# --- format_number ---

def test_format_number():
    assert command.format_number(60.0) == "60"
    assert command.format_number(62.5) == "62.5"
    assert command.format_number(1) == "1"


# --- find_llama_server ---

def test_find_llama_server_top_level(tmp_path):
    exe = tmp_path / "llama-server.exe"
    exe.write_text("")
    assert command.find_llama_server(str(tmp_path)) == str(exe)


def test_find_llama_server_in_subdir(tmp_path):
    sub = tmp_path / "build" / "bin"
    sub.mkdir(parents=True)
    exe = sub / "llama_server.exe"
    exe.write_text("")
    found = command.find_llama_server(str(tmp_path))
    assert found == str(exe)


def test_find_llama_server_not_found(tmp_path):
    assert command.find_llama_server(str(tmp_path)) is None
