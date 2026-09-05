"""GGUF model discovery and path resolution."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from launcher import config


@dataclass(frozen=True)
class Model:
    """A discovered GGUF model.

    display: human-readable "author/model_name" label
    path:    relative path "author/model_folder/model_name" (no .gguf extension, forward slashes)
    """
    display: str
    path: str


def scan_models(models_dir: Optional[str] = None) -> List[Model]:
    """Scan models directory for GGUF files (any depth, e.g. author/model_folder/[subdir]/model.gguf).

    Returns a sorted list of Model dataclasses. mmproj files are skipped (vision projections, not models)."""
    models: List[Model] = []
    resolved_dir = models_dir or config.MODELS_DIR
    if not resolved_dir:
        print("[ERROR] MODELS_DIR is not set (configure it in .env).")
        return models
    models_path = Path(resolved_dir)
    if not models_path.exists():
        print(f"[ERROR] Models directory not found: {models_path}")
        return models

    for gguf_file in models_path.rglob("*.gguf"):
        if not gguf_file.is_file():
            continue
        if gguf_file.name.lower().startswith("mmproj"):
            continue

        model_name = gguf_file.stem
        rel_parts = gguf_file.relative_to(models_path).parts
        rel_path = "/".join(rel_parts[:-1]) + "/" + model_name
        display_name = f"{rel_parts[0]}/{model_name}"
        models.append(Model(display=display_name, path=rel_path))

    models.sort(key=lambda m: m.display)
    return models


def get_model_path(rel_path: str, models_dir: Optional[str] = None) -> str:
    """Convert a relative model path (from Model.path) to an absolute .gguf file path."""
    base = models_dir or config.MODELS_DIR
    full_path = rel_path.replace("/", os.sep) + ".gguf"
    return os.path.join(base, full_path)


def find_mmproj(model_path: str) -> Optional[str]:
    """Find the first mmproj*.gguf file in the directory containing *model_path*.

    Returns the absolute path, or None if not found."""
    model_dir = os.path.dirname(model_path)
    try:
        candidates = sorted(
            f for f in os.listdir(model_dir)
            if f.lower().startswith("mmproj") and f.endswith(".gguf")
        )
    except OSError:
        return None
    if not candidates:
        return None
    return os.path.join(model_dir, candidates[0])
