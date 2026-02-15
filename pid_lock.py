"""
PID Lock utility for preventing duplicate trader instances.

Uses OS-level file locking (msvcrt on Windows, fcntl on Unix) so the lock
is held for the entire process lifetime and auto-released on crash/death.
NO race conditions — the OS guarantees exclusivity.

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
import atexit
from pathlib import Path
from functools import partial

print = partial(print, flush=True)

PID_DIR = Path(__file__).parent

# Global registry: name -> open file descriptor (kept open = lock held)
_held_locks: dict = {}


def acquire_pid_lock(name: str) -> Path:
    """Acquire an OS-level exclusive file lock. Exits if another instance holds it.

    The lock is held by keeping the file descriptor open. When this process
    exits (even via crash/kill), the OS auto-releases the lock.

    Args:
        name: Lock name (e.g., "maker", "ta_paper")

    Returns:
        Path to the PID file
    """
    pid_file = PID_DIR / f"{name}.pid"
    lock_file = PID_DIR / f"{name}.lock"

    # Try to acquire OS-level exclusive lock
    try:
        # Open lock file (create if needed)
        fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR)

        if sys.platform == "win32":
            import msvcrt
            try:
                # Non-blocking exclusive lock on first byte
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except (IOError, OSError):
                # Lock held by another process — read PID for error message
                os.close(fd)
                old_pid = "?"
                try:
                    old_pid = pid_file.read_text().strip()
                except Exception:
                    pass
                print(f"[FATAL] Another {name} is already running (PID {old_pid})")
                print(f"[FATAL] Kill it first: taskkill /F /PID {old_pid}")
                sys.exit(1)
        else:
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (IOError, OSError):
                os.close(fd)
                old_pid = "?"
                try:
                    old_pid = pid_file.read_text().strip()
                except Exception:
                    pass
                print(f"[FATAL] Another {name} is already running (PID {old_pid})")
                print(f"[FATAL] Kill it first or delete {pid_file}")
                sys.exit(1)

        # Lock acquired — store fd so it stays open (lock held)
        _held_locks[name] = fd

        # Write PID to .pid file for human debugging
        pid_file.write_text(str(os.getpid()))
        print(f"[PID] Lock acquired: {name} (PID {os.getpid()})")

        # Auto-release on exit
        atexit.register(release_pid_lock, name)

        return pid_file

    except SystemExit:
        raise  # Re-raise sys.exit
    except Exception as e:
        print(f"[PID] Lock error: {e} — proceeding without lock")
        pid_file.write_text(str(os.getpid()))
        return pid_file


def release_pid_lock(name: str):
    """Release PID lock and close the file descriptor."""
    # Close the lock file descriptor (releases OS lock)
    fd = _held_locks.pop(name, None)
    if fd is not None:
        try:
            if sys.platform == "win32":
                import msvcrt
                try:
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            os.close(fd)
        except Exception:
            pass

    # Clean up PID file
    pid_file = PID_DIR / f"{name}.pid"
    try:
        if pid_file.exists():
            stored_pid = int(pid_file.read_text().strip())
            if stored_pid == os.getpid():
                pid_file.unlink()
    except Exception:
        pass

    # Clean up lock file
    lock_file = PID_DIR / f"{name}.lock"
    try:
        if lock_file.exists():
            lock_file.unlink()
    except Exception:
        pass  # Another process may hold it

    print(f"[PID] Lock released: {name}")
