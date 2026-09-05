"""llama-server command-line builder.

build_command() assembles the full command in the exact same flag order as the
original single-file script; each feature is an independent _append_* helper.
"""

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from launcher import config, models


# === Simple parameter mapping: setting_key → CLI flag ===
SIMPLE_PARAM_MAP = [
    ("context", "-c"),
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


def _append_mmap(cmd: List[str], settings: Dict[str, Any]) -> None:
    """mmap (--no-mmap): False explicitly disables memory-mapped file loading."""
    if settings.get("mmap") is False:
        cmd.append("--no-mmap")


def _append_flash_attn(cmd: List[str], settings: Dict[str, Any]) -> None:
    """Flash Attention: True enables flash attention optimization."""
    if settings.get("flash_attention") is True:
        cmd += ["-fa", "on"]


def _append_chat_template_kwargs(cmd: List[str], settings: Dict[str, Any]) -> None:
    """Chat Template Kwargs: merge thinking + preserve_thinking into a single argument."""
    chat_kwargs: dict = {}
    if (thinking_val := settings.get("thinking")) is not None and isinstance(thinking_val, bool):
        chat_kwargs["enable_thinking"] = thinking_val
    if (p_thinking_val := settings.get("p_thinking")) is not None and isinstance(p_thinking_val, bool):
        chat_kwargs["preserve_thinking"] = p_thinking_val
    if chat_kwargs:
        cmd += ["--chat-template-kwargs", json.dumps(chat_kwargs)]


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
            #cmd.append("--no-mmproj-offload")


def _append_image_min_tokens(cmd: List[str], settings: Dict[str, Any]) -> None:
    """Image Min Tokens: --image-min-tokens N when set (vision-related)."""
    if isinstance(image_min_tokens := settings.get("image_min_tokens"), int) and image_min_tokens > 0:
        cmd += ["--image-min-tokens", str(image_min_tokens)]


def _append_server_settings(cmd: List[str], host: str, port: int) -> None:
    """Host/port plus fixed server flags."""
    cmd += ["--host", host, "--port", str(port)]
    cmd.append("--no-webui")
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
    #cmd += ["--reasoning", "off"]
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
    _append_simple_params(cmd, settings)
    _append_mmap(cmd, settings)
    _append_flash_attn(cmd, settings)
    _append_chat_template_kwargs(cmd, settings)
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
