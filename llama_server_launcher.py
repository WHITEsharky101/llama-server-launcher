#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Llama.cpp Server Launcher — entry point.

The implementation lives in the `launcher/` package:
    launcher/config.py    - configuration file handling, environment loading
    launcher/models.py    - GGUF model scanning and path resolution
    launcher/presets.py   - preset (RP / Coder / Commit) definitions and logic
    launcher/settings.py  - interactive settings editor and display
    launcher/command.py   - llama-server command-line builder
    launcher/server.py    - server process lifecycle (launch / stop)
    launcher/ui.py        - terminal UI helpers (headers, selection prompts)
    launcher/main.py      - main entry point and menu loop
"""

from launcher.main import main

if __name__ == "__main__":
    main()
