"""
PID Lock utility for preventing duplicate trader instances.

Usage:
    from pid_lock import acquire_pid_lock, release_pid_lock

    pid_file = acquire_pid_lock("ta_paper")  # Creates ta_paper.pid
    try:
        # ... run trader ...
    finally:
        release_pid_lock("ta_paper")
"""

import os
import sys
import subprocess
from pathlib import Path
from functools import partial

print = partial(print, flush=True)

PID_DIR = Path(__file__).parent


def _is_pid_alive(pid: int) -> bool:
    """Check if a process is alive (Windows-compatible)."""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
            capture_output=True, text=True, timeout=5
        )
        return str(pid) in result.stdout
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def acquire_pid_lock(name: str) -> Path:
    """Acquire PID lock. Exits if another instance is running.

    Args:
        name: Lock name (e.g., "ta_paper", "weather_paper", "whale_watcher")

    Returns:
        Path to the PID file
    """
    pid_file = PID_DIR / f"{name}.pid"

    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text().strip())
            if old_pid == os.getpid():
                pass  # Re-acquiring our own lock
            elif _is_pid_alive(old_pid):
                print(f"[FATAL] Another {name} is already running (PID {old_pid})")
                print(f"[FATAL] Kill it first: taskkill /F /PID {old_pid}")
                print(f"[FATAL] Or delete {pid_file} if the process is dead")
                sys.exit(1)
            else:
                print(f"[PID] Stale lock (PID {old_pid} dead) - taking over")
        except (ValueError, IOError):
            pass  # Corrupt file, overwrite

    pid_file.write_text(str(os.getpid()))
    print(f"[PID] Lock acquired: {name} (PID {os.getpid()})")
    return pid_file


def release_pid_lock(name: str):
    """Release PID lock on exit."""
    pid_file = PID_DIR / f"{name}.pid"
    try:
        if pid_file.exists():
            stored_pid = int(pid_file.read_text().strip())
            if stored_pid == os.getpid():
                pid_file.unlink()
                print(f"[PID] Lock released: {name}")
    except Exception:
        pass
