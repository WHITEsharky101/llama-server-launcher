"""llama-server command-line builder.

build_command() assembles the full command in the exact same flag order as the
original single-file script; each feature is an independent _append_* helper.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from launcher import config, models


# === Simple parameter mapping: setting_key → CLI flag ===
SIMPLE_PARAM_MAP = [
    ("gpu_offload", "-ngl"),
    ("cpu_moe", "--n-cpu-moe"),
    ("threads", "-t"),
    ("batch_threads", "-tb"),
    ("batch_size", "-b"),
    ("parallel", "-np"),
    ("temp", "--temp"),
    ("k_quant", "-ctk"),
    ("v_quant", "-ctv"),
    ("top_k", "--top-k"),
    ("top_p", "--top-p"),
    ("min_p", "--min-p"),
    ("repeat_penalty", "--repeat-penalty"),
    ("presence_penalty", "--presence-penalty"),
]


def format_number(v: float) -> str:
    """Format a number: whole numbers as int string, otherwise keep decimals."""
    return str(int(v)) if v == int(v) else str(v)


@lru_cache(maxsize=None)
def find_llama_server(llama_cpp_dir: Optional[str] = None) -> Optional[str]:
    """Find the llama-server executable under *llama_cpp_dir* (result is cached)."""
    base = llama_cpp_dir or config.LLAMA_CPP_DIR
    if not base:
        print("[ERROR] LLAMA_CPP_DIR is not set (configure it in .env).")
        return None
    server_names = ("llama-server.exe", "llama_server.exe")

    for name in server_names:
        candidate = os.path.join(base, name)
        if os.path.exists(candidate):
            return candidate

    for root, _dirs, files in os.walk(base):
        for file in files:
            if file.lower() in server_names:
                return os.path.join(root, file)

    return None


def get_llama_server_path() -> Optional[str]:
    """Find llama-server executable in the configured llama.cpp directory."""
    return find_llama_server(config.LLAMA_CPP_DIR)


# --- Command append helpers ---

def _add_simple_param(cmd: List[str], flag: str, value: Any) -> None:
    """Add a simple parameter to the command when value is "truthy" (not None/False/0)."""
    if value not in (None, False, 0):
        cmd += [flag, str(value)]


def _append_simple_params(cmd: List[str], settings: Dict[str, Any]) -> None:
    """Append flag+value pairs for all simple settings."""
    for key, flag in SIMPLE_PARAM_MAP:
        _add_simple_param(cmd, flag, settings.get(key))


# --- Context resolution (vision-aware) ---
#
# 'context' can be either:
#   int  — single context length used in all modes (legacy format)
#   [base, gpu_vision] — base context for vision off / mmproj offloaded to RAM,
#                        gpu_vision context for vision enabled on GPU.
# A missing/None gpu_vision element falls back to the base value.

def is_vision_on_gpu(settings: Dict[str, Any]) -> bool:
    """True when vision is enabled and the mmproj stays on the GPU.

    Matches the --no-mmproj-offload condition in _append_vision(): the mmproj is
    offloaded to RAM only when mmproj_offload is explicitly False."""
    return settings.get("vision") is True and settings.get("mmproj_offload") is not False


def _normalize_context(raw: Any) -> Tuple[Optional[int], Optional[int]]:
    """Normalize a context value (scalar or [base, gpu_vision]) to (base, gpu_vision)."""
    if isinstance(raw, bool):
        return None, None
    if isinstance(raw, int):
        return raw, raw
    if isinstance(raw, (list, tuple)):
        base = raw[0] if len(raw) > 0 else None
        gpu = raw[1] if len(raw) > 1 else None
        return base, gpu
    return None, None


def resolve_context(settings: Dict[str, Any]) -> Optional[int]:
    """Return the effective context length for the current vision state.

    Vision on GPU  -> [base, gpu_vision][1] (falls back to base when unset)
    Otherwise      -> base value"""
    base, gpu_vision = _normalize_context(settings.get("context"))
    if is_vision_on_gpu(settings):
        return gpu_vision if isinstance(gpu_vision, int) and gpu_vision > 0 else base
    return base


def _append_context(cmd: List[str], settings: Dict[str, Any]) -> None:
    """-c with the vision-resolved context length."""
    _add_simple_param(cmd, "-c", resolve_context(settings))


def _append_mmap(cmd: List[str], settings: Dict[str, Any]) -> None:
    """mmap: False explicitly disables memory-mapped loading (--load-mode none).

    True/None keep the llama.cpp default (auto) and add no flag."""
    if settings.get("mmap") is False:
        cmd += ["--load-mode", "none"]


def _append_flash_attn(cmd: List[str], settings: Dict[str, Any]) -> None:
    """Flash Attention: True enables flash attention optimization."""
    if settings.get("flash_attention") is True:
        cmd += ["-fa", "on"]


def _append_reasoning(cmd: List[str], settings: Dict[str, Any]) -> None:
    """Thinking: --reasoning on/off (replaces the deprecated enable_thinking kwarg)."""
    thinking_val = settings.get("thinking")
    if thinking_val is True:
        cmd += ["--reasoning", "on"]
    elif thinking_val is False:
        cmd += ["--reasoning", "off"]


def _append_reasoning_preserve(cmd: List[str], settings: Dict[str, Any]) -> None:
    """Preserve thinking: --reasoning-preserve / --no-reasoning-preserve.

    Replaces the deprecated preserve_thinking chat-template kwarg. When the
    setting is off or unset the flag is forced to --no-reasoning-preserve, which
    also suppresses the "enabled by default (may use more tokens)" notice."""
    if settings.get("p_thinking") is True:
        cmd.append("--reasoning-preserve")
    else:
        cmd.append("--no-reasoning-preserve")


def _append_jinja(cmd: List[str], settings: Dict[str, Any]) -> None:
    """Jinja (--jinja): True enables jinja template parsing."""
    if settings.get("jinja") is True:
        cmd.append("--jinja")


def _append_vision(cmd: List[str], settings: Dict[str, Any], model_path: str) -> None:
    """Vision / mmproj: when enabled, add --mmproj pointing to first mmproj*.gguf in model directory."""
    if settings.get("vision") is True:
        mmproj = models.find_mmproj(model_path)
        if mmproj is None:
            print(f"[WARNING] Vision enabled but no mmproj file found in {os.path.dirname(model_path)}")
        else:
            cmd += ["--mmproj", mmproj]
            if settings.get("mmproj_offload") is False:
                cmd.append("--no-mmproj-offload")


def _append_image_min_tokens(cmd: List[str], settings: Dict[str, Any]) -> None:
    """Image Min Tokens: --image-min-tokens N when set (only when vision is on)."""
    if settings.get("vision") is not True:
        return
    if isinstance(image_min_tokens := settings.get("image_min_tokens"), int) and image_min_tokens > 0:
        cmd += ["--image-min-tokens", str(image_min_tokens)]


def _append_server_settings(cmd: List[str], host: str, port: int) -> None:
    """Host/port plus fixed server flags."""
    cmd += ["--host", host, "--port", str(port)]
    cmd.append("--no-ui")
    cmd += ["-mg", "0"]


def _append_mtp(cmd: List[str], settings: Dict[str, Any]) -> None:
    """MTP (Multi-Token Prediction): --spec-type draft-mtp when enabled."""
    if settings.get("mtp") is True:
        cmd += ["--spec-type", "draft-mtp"]
        if isinstance(draft_n_max := settings.get("draft_n_max"), int) and draft_n_max > 0:
            cmd += ["--spec-draft-n-max", str(draft_n_max)]


def _append_tensor_split(cmd: List[str], settings: Dict[str, Any]) -> None:
    """Tensor Split: multi-GPU layer distribution (--tensor-split)."""
    tensor_split = settings.get("tensor_split")
    if isinstance(tensor_split, list) and len(tensor_split) >= 2:
        split_str = ",".join(format_number(v) for v in tensor_split)
        cmd += ["--tensor-split", split_str]


def _append_fixed_flags(cmd: List[str]) -> None:
    """Flags that are always appended at the end."""
    cmd.append("--no-slots")
    #cmd.append("--swa-full")
    cmd += ["--fit", "off"]  # avoid "failed to fit params" warning (n_gpu_layers is set)
    cmd += ["--timeout", "30000"]
    cmd += ["-n", "-1"]


def build_command(
    model_path: str, settings: Dict[str, Any], host: str, port: int, api_key: str
) -> List[str]:
    """Build the llama-server command with all parameters."""
    server_exe = get_llama_server_path()
    if server_exe is None:
        print("[ERROR] llama-server executable not found!")
        return []

    cmd: List[str] = [server_exe, "--model", model_path]
    _append_context(cmd, settings)
    _append_simple_params(cmd, settings)
    _append_mmap(cmd, settings)
    _append_flash_attn(cmd, settings)
    _append_reasoning(cmd, settings)
    _append_reasoning_preserve(cmd, settings)
    _append_jinja(cmd, settings)
    _append_vision(cmd, settings, model_path)
    _append_image_min_tokens(cmd, settings)
    _append_server_settings(cmd, host, port)
    _append_mtp(cmd, settings)
    _append_tensor_split(cmd, settings)
    _append_fixed_flags(cmd)

    if api_key:
        cmd += ["--api-key", api_key]

    return cmd
