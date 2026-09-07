"""Server process lifecycle: launch, keyboard stop listening, and graceful termination."""

import msvcrt
import subprocess
import threading
import time
from typing import List

_STOP_KEY = "q"


def listen_for_stop_key(stop_flag: List[bool], stop_event: threading.Event) -> None:
    """Background thread that listens for 'q' key press and sets the stop flag."""
    while not stop_event.is_set():
        if msvcrt.kbhit():
            char = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            if char == _STOP_KEY:
                stop_flag[0] = True
                return
        time.sleep(0.05)


def terminate_process(process: subprocess.Popen) -> None:
    """Gracefully terminate a process, killing it if it doesn't exit in time."""
    try:
        process.terminate()
        process.wait(timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        process.kill()
        process.wait()


def launch_server(cmd: List[str]) -> bool:
    """Launch the llama server process.

    Returns True if user pressed 'q' to stop, False otherwise (e.g., Ctrl+C or natural exit).
    """
    if not cmd:
        print("[ERROR] No command to launch!")
        return False

    print("\n" + "=" * 60)
    print("Launching Llama Server...")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    print("\nPress 'q' to stop the server and return to menu.")
    print("Press Ctrl+C to forcefully terminate.\n")

    process = None
    user_stopped_via_q = False

    try:
        stop_flag = [False]
        stop_event = threading.Event()
        kb_thread = threading.Thread(
            target=listen_for_stop_key, args=(stop_flag, stop_event), daemon=True
        )
        kb_thread.start()

        process = subprocess.Popen(cmd)

        # Wait for either the process to exit or user to press 'q'
        while True:
            if stop_flag[0]:
                print("\n\n[INFO] User requested stop (pressed 'q'). Stopping server...")
                terminate_process(process)
                user_stopped_via_q = True
                break

            if process.poll() is not None:
                break

            time.sleep(0.05)

        stop_event.set()
        kb_thread.join(timeout=1)

    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user (Ctrl+C).")
        if process is not None:
            terminate_process(process)
    except FileNotFoundError:
        print(f"\n[ERROR] Executable not found: {cmd[0]}")
    except Exception as e:
        print(f"\n[ERROR] Failed to launch server: {e}")

    return user_stopped_via_q
