"""Interactive settings editor and display.

The settings menu is driven by the SETTINGS_INFO table; per-key editors are
registered in _SPECIAL_EDITORS / key groups, so adding a setting only requires
a table entry (plus an editor if it needs non-free-text input).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from launcher.command import _normalize_context, format_number, is_vision_on_gpu, resolve_context

# === Settings Menu Definition (shared between display_settings_menu and edit_settings) ===
# Each entry: (display_number, config_key, human_readable_name)
SETTINGS_INFO: List[Tuple[str, str, str]] = [
    ("1", "context", "Context"),
    ("2", "gpu_offload", "GPU Offload"),
    ("3", "cpu_moe", "CPU MOE"),
    ("4", "threads", "Threads"),
    ("5", "batch_threads", "Batch Threads"),
    ("6", "batch_size", "Batch Size"),
    ("7", "parallel", "Parallel"),
    ("8", "mmap", "mmap"),
    ("9", "flash_attention", "Flash Attention"),
    ("10", "k_quant", "K Quant"),
    ("11", "v_quant", "V Quant"),
    ("12", "mtp", "MTP"),
    ("13", "draft_n_max", "Draft N Max"),
    ("14", "temp", "Temp"),
    ("15", "top_k", "Top K"),
    ("16", "top_p", "Top P"),
    ("17", "min_p", "Min P"),
    ("18", "repeat_penalty", "Repeat Penalty"),
    ("19", "presence_penalty", "Presence Penalty"),
    ("20", "thinking", "Thinking"),
    ("21", "p_thinking", "Preserve Think"),
    ("22", "jinja", "Jinja"),
    ("23", "vision", "Vision"),
    ("24", "image_min_tokens", "Image Min Tokens"),
    ("25", "mmproj_offload", "MMProj Offload"),
    ("26", "tensor_split", "GPU Tensor Split"),
]

# Build a lookup: numeric choice → (config_key, display_name) for O(1) selection
_SETTINGS_LOOKUP: Dict[str, Tuple[str, str]] = {num: (key, name) for num, key, name in SETTINGS_INFO}

# === KV Cache Quantization Options ===
KV_QUANT_OPTIONS = ["turbo4", "turbo3", "turbo2", "q8_0", "q4_0", "iq4_nl", "none"]

# Settings that use the on/off/none toggle editor
_TOGGLE_KEYS: frozenset = frozenset({"mmap", "flash_attention", "thinking", "p_thinking", "jinja", "vision", "mtp", "mmproj_offload"})

# Settings that use the KV-quant dropdown editor
_KV_QUANT_KEYS: frozenset = frozenset({"k_quant", "v_quant"})

# Settings that are only relevant (and visible) when vision is enabled (on).
# Hidden from the menu and display when vision is off/none.
_VISION_DEPENDENT: frozenset = frozenset({"image_min_tokens", "mmproj_offload"})


def _is_visible(settings: Dict[str, Any], key: str) -> bool:
    """Return True if *key* should be shown; vision-dependent keys are hidden when vision is not on."""
    if key in _VISION_DEPENDENT and settings.get("vision") is not True:
        return False
    return True


# --- Value editors (each mutates settings[key]) ---

def _edit_toggle(settings: Dict[str, Any], key: str, name: str) -> None:
    """on/off/none toggle editor."""
    print(f"Select value for {name}:")
    print("  1. on")
    print("  2. off")
    print("  3. none")

    while True:
        choice = input("> ").strip()
        if choice in ("", "3"):
            settings[key] = None
            break
        elif choice == "1":
            settings[key] = True
            break
        elif choice == "2":
            settings[key] = False
            break
        else:
            print("Invalid choice. Enter 1 (on), 2 (off), or 3/empty (none)")


def _edit_kv_quant(settings: Dict[str, Any], key: str, name: str) -> None:
    """Dropdown editor for KV-cache quantization options."""
    print(f"Select value for {name}:")
    for i, opt in enumerate(KV_QUANT_OPTIONS, 1):
        print(f"  {i}. {opt}")

    while True:
        choice = input("> ").strip()
        if choice == "" or choice == str(len(KV_QUANT_OPTIONS)):
            settings[key] = None
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(KV_QUANT_OPTIONS):
                val = KV_QUANT_OPTIONS[idx]
                settings[key] = None if val == "none" else val
                break
            else:
                print(f"Invalid choice. Enter 1-{len(KV_QUANT_OPTIONS)} or leave empty for 'none'")
        except ValueError:
            print("Invalid input.")


def _edit_tensor_split(settings: Dict[str, Any], key: str, name: str) -> None:
    """GPU Tensor Split: single percentage for first GPU, second is 100-x.

    Precision up to tenths (one decimal place). Empty input disables the flag."""
    print("Enter GPU load percentage for the first GPU (the second gets the remainder).")
    print("Example: '60' means GPU1=60%, GPU2=40%. Precision to 0.1.")
    print("Leave empty to disable tensor-split.")
    while True:
        new_value = input("> ").strip()
        if new_value == "" or new_value.lower() in ("none", "null"):
            settings[key] = None
            break
        try:
            val = float(new_value)
            if val < 0 or val > 100:
                print("Percentage must be between 0 and 100.")
                continue
            gpu1 = round(val, 1)
            gpu2 = round(100.0 - gpu1, 1)
            settings[key] = [gpu1, gpu2]
            break
        except ValueError:
            print("Invalid input. Enter a number (e.g. '60' or '75.5').")


def _read_context_value(prompt: str) -> Optional[int]:
    """Read a positive integer context value; empty/'none'/'null' means unset (fallback)."""
    print(prompt)
    while True:
        raw = input("> ").strip().lower()
        if raw == "" or raw in ("none", "null"):
            return None
        try:
            val = int(raw)
        except ValueError:
            print("Invalid input. Enter a positive integer (or 'none').")
            continue
        if val <= 0:
            print("Value must be a positive integer (> 0).")
            continue
        return val


def _edit_context(settings: Dict[str, Any], key: str, name: str) -> None:
    """Context editor: shows and edits only the value active for the current vision state.

    Settings store [base, gpu]; only the slot that will actually be used at launch
    (given the current vision / MMProj Offload settings) is displayed and editable.
    Legacy scalars are treated as the base value (no separate gpu value was ever set)."""
    raw = settings.get(key)
    base, gpu = _normalize_context(raw)
    if isinstance(raw, int) and not isinstance(raw, bool):
        gpu = None  # Legacy scalar: no separate gpu value was ever set

    on_gpu = is_vision_on_gpu(settings)
    mode = "vision on GPU" if on_gpu else "vision off / mmproj in RAM"
    current = resolve_context(settings)

    print(f"Mode: {mode}")
    print(f"Context for this mode (currently: {current if current is not None else 'unset'}):")
    value = _read_context_value("New value (or 'none' to clear):")

    if value is None:
        if on_gpu:
            gpu = None
        else:
            base = None
    else:
        if on_gpu:
            gpu = value
        else:
            base = value

    if base is None and gpu is None:
        settings[key] = None
    elif base is None:
        settings[key] = [None, gpu]
    elif gpu is None or gpu == base:
        settings[key] = base  # single value = same for both modes (legacy-compatible)
    else:
        settings[key] = [base, gpu]


def _read_positive_int(prompt: str) -> Optional[int]:
    """Read a positive integer; returns None for empty/'none'/'null' input."""
    print(prompt)
    while True:
        new_value = input("> ").strip().lower()
        if new_value == "" or new_value in ("none", "null"):
            return None
        try:
            val = int(new_value)
        except ValueError:
            print("Invalid input. Enter a positive integer.")
            continue
        if val <= 0:
            print("Value must be a positive integer (> 0).")
            continue
        return val


def _edit_draft_n_max(settings: Dict[str, Any], key: str, name: str) -> None:
    """Draft N Max: positive integer, ignored when MTP is not enabled."""
    if settings.get("mtp") is not True:
        print("[INFO] MTP is not enabled (on). Draft N Max will be ignored at launch.")
        print("Enable MTP first if you want this setting to take effect.")
    settings[key] = _read_positive_int("Enter a positive integer value for max draft tokens (or 'none' to clear):")


def _edit_image_min_tokens(settings: Dict[str, Any], key: str, name: str) -> None:
    """Image Min Tokens: positive integer for --image-min-tokens N."""
    if settings.get("vision") is not True:
        print("[INFO] Vision is not enabled (on). Image Min Tokens will be ignored at launch.")
        print("Enable Vision first if you want this setting to take effect.")
    settings[key] = _read_positive_int("Enter a positive integer value for image min tokens (or 'none' to clear):")


def _edit_free_text(settings: Dict[str, Any], key: str, name: str) -> None:
    """Free text input for numeric/text settings."""
    print("Enter new value (or 'none' to skip this parameter):")
    new_value = input("> ").strip().lower()
    if new_value in ("", "none", "null"):
        settings[key] = None
    else:
        # Try to convert to appropriate numeric type
        try:
            settings[key] = float(new_value) if "." in new_value else int(new_value)
        except ValueError:
            settings[key] = new_value


# Registry of keys that need a dedicated (non-free-text) editor
_SPECIAL_EDITORS: Dict[str, Callable[[Dict[str, Any], str, str], None]] = {
    "context": _edit_context,
    "tensor_split": _edit_tensor_split,
    "draft_n_max": _edit_draft_n_max,
    "image_min_tokens": _edit_image_min_tokens,
}


def _editor_for(key: str) -> Callable[[Dict[str, Any], str, str], None]:
    """Return the value editor function for a settings key."""
    if key in _TOGGLE_KEYS:
        return _edit_toggle
    if key in _KV_QUANT_KEYS:
        return _edit_kv_quant
    return _SPECIAL_EDITORS.get(key, _edit_free_text)


# --- Display ---

def _fmt_context(settings: Dict[str, Any]) -> str:
    """Context display: only the value active for the current vision state."""
    effective = resolve_context(settings)
    return str(effective) if effective is not None else "N/A"


def _fmt_mtp(settings: Dict[str, Any]) -> str:
    mtp_val = settings.get("mtp")
    return "on" if mtp_val is True else ("off" if mtp_val is False else "none")


def _fmt_draft_n_max(settings: Dict[str, Any]) -> str:
    draft_val = settings.get("draft_n_max")
    return str(draft_val) if settings.get("mtp") is True else "(ignored)"


def _fmt_image_min_tokens(settings: Dict[str, Any]) -> str:
    image_min_tokens = settings.get("image_min_tokens")
    return str(image_min_tokens) if isinstance(image_min_tokens, int) and image_min_tokens > 0 else "disabled"


def _fmt_mmproj_offload(settings: Dict[str, Any]) -> str:
    mmproj_val = settings.get("mmproj_offload")
    return "on" if mmproj_val is True else ("off" if mmproj_val is False else "none")


def _fmt_tensor_split(settings: Dict[str, Any]) -> str:
    ts = settings.get("tensor_split")
    if ts is not None and isinstance(ts, list):
        return ",".join(format_number(v) for v in ts) + "%"
    return "disabled"


# Keys that need non-trivial display formatting
_FORMATTERS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "context": _fmt_context,
    "mtp": _fmt_mtp,
    "draft_n_max": _fmt_draft_n_max,
    "image_min_tokens": _fmt_image_min_tokens,
    "mmproj_offload": _fmt_mmproj_offload,
    "tensor_split": _fmt_tensor_split,
}

# Display sections in the order shown by the original UI
_DISPLAY_SECTIONS: List[Tuple[str, List[str]]] = [
    ("Model Parameters", [
        "context", "gpu_offload", "cpu_moe", "threads", "batch_threads", "batch_size", "parallel",
        "mmap", "flash_attention", "k_quant", "v_quant", "mtp", "draft_n_max",
    ]),
    ("Generation Settings", [
        "temp", "top_k", "top_p", "min_p", "repeat_penalty", "presence_penalty",
        "thinking", "p_thinking", "jinja", "vision", "image_min_tokens", "mmproj_offload",
    ]),
    ("Multi-GPU Settings", ["tensor_split"]),
]


def _format_value(settings: Dict[str, Any], key: str) -> str:
    formatter = _FORMATTERS.get(key)
    if formatter is not None:
        return formatter(settings)
    return str(settings.get(key, "N/A"))


def _display_section(title: str, keys: List[str], settings: Dict[str, Any]) -> None:
    name_by_key = {key: name for _num, key, name in SETTINGS_INFO}
    print(f"\n--- {title} ---")
    for key in keys:
        if not _is_visible(settings, key):
            continue
        print(f"  {name_by_key[key] + ':':<20} {_format_value(settings, key)}")


def display_settings(settings: Dict[str, Any]) -> None:
    """Display current model settings in a readable format."""
    print("\n" + "=" * 50)
    print("Model Settings:")
    print("=" * 50)
    for title, keys in _DISPLAY_SECTIONS:
        _display_section(title, keys, settings)
    print("=" * 50 + "\n")


def display_settings_menu(settings: Dict[str, Any]) -> None:
    """Display the settings menu (vision-dependent entries are hidden while vision is off/none)."""
    print("\nAvailable settings to modify:")
    print("=" * 60)

    for num, key, name in SETTINGS_INFO:
        if not _is_visible(settings, key):
            continue
        current = _format_value(settings, key) if key in settings else "(not set)"
        print(f"  {num}. {name:<18} [{current}]")

    if not all(_is_visible(settings, key) for _num, key, _name in SETTINGS_INFO):
        print("  (Vision-dependent settings are hidden while Vision is not on)")
    print("=" * 60)


# --- Interactive editor ---

def edit_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Interactive settings editor."""
    display_settings_menu(settings)
    print("Enter the number of the setting to modify (0 when done):")

    while True:
        choice = input("> ").strip()
        if choice == "0":
            break

        selected = _SETTINGS_LOOKUP.get(choice)
        if selected is None:
            print("Invalid choice. Try again.")
            continue

        key, name = selected
        if not _is_visible(settings, key):
            print(f"[INFO] {name} is available only when Vision is on. Enable Vision first.")
            continue

        current = _format_value(settings, key) if key in settings else "(not set)"
        print(f"\nCurrent value for {name}: {current}")

        editor = _editor_for(key)
        editor(settings, key, name)

        print(f"[OK] {name} set to: {_format_value(settings, key)}")
        display_settings_menu(settings)
        print("Enter the number of the setting to modify (0 when done):")

    return settings
