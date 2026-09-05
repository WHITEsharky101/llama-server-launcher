"""Tests for launcher/models.py: GGUF scanning and path resolution."""

import os

import pytest

from launcher import models


def make_tree(base, names):
    """Create gguf files at relative paths (e.g. 'author/folder/model.gguf')."""
    for rel in names:
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00")


def test_scan_models_finds_and_sorts(tmp_path):
    make_tree(tmp_path, [
        "zeta/zm/zm.gguf",
        "alpha/am/am.gguf",
        "alpha/deep/nested/dn.gguf",
    ])
    found = models.scan_models(str(tmp_path))
    displays = [m.display for m in found]
    assert displays == ["alpha/am", "alpha/dn", "zeta/zm"]
    # Paths: forward-slash relative, no extension
    by_display = {m.display: m for m in found}
    assert by_display["alpha/dn"].path == "alpha/deep/nested/dn"


def test_scan_models_skips_mmproj(tmp_path):
    make_tree(tmp_path, [
        "author/model.gguf",
        "author/mmproj-00001-of-00002.gguf",
    ])
    found = models.scan_models(str(tmp_path))
    assert [m.display for m in found] == ["author/model"]


def test_scan_models_missing_dir_returns_empty(tmp_path):
    assert models.scan_models(str(tmp_path / "does-not-exist")) == []


def test_get_model_path(tmp_path):
    expected = models.get_model_path("author/sub/model", str(tmp_path))
    assert expected == os.path.join(str(tmp_path), "author", "sub", "model.gguf")


def test_find_mmproj_picks_first_sorted(tmp_path):
    (tmp_path / "mmproj-b.gguf").write_bytes(b"x")
    (tmp_path / "mmproj-a.gguf").write_bytes(b"x")
    (tmp_path / "model.gguf").write_bytes(b"x")
    found = models.find_mmproj(str(tmp_path / "model.gguf"))
    assert found == os.path.join(str(tmp_path), "mmproj-a.gguf")


def test_find_mmproj_none_when_absent(tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"x")
    assert models.find_mmproj(str(tmp_path / "model.gguf")) is None


def test_model_dataclass_is_frozen():
    m = models.Model(display="a/b", path="a/b")
    assert m.display == "a/b"
    with pytest.raises(AttributeError):
        m.display = "other"
