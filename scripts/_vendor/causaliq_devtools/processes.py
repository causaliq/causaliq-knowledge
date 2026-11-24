# generated from causaliq@49a27b66b20d src/causaliq/devtools/processes.py - do not edit
"""Platform helpers for launching and terminating command trees.

The behaviour differs enough between Windows and POSIX that keeping it in one
place makes the run/wait protocol in :mod:`run_checked` easier to follow.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import IO, Sequence

IS_WINDOWS = os.name == "nt"


def spawn(
    command: Sequence[str],
    handle: IO[bytes],
    breakaway: bool = False,
) -> subprocess.Popen[bytes]:
    """Start a detached process with its output redirected to ``handle``.

    Only a supervisor asks to break away from the host's job object; its
    children stay in whatever job the supervisor ended up in. GitHub's
    Windows runner forbids breakaway, so the request is best-effort.
    """
    if IS_WINDOWS:
        return _windows_process(command, handle, breakaway)
    return subprocess.Popen(
        list(command),
        stdout=handle,
        stderr=subprocess.STDOUT,
        close_fds=True,
        start_new_session=True,
    )


def windows_flags(breakaway: bool = True) -> int:
    """Return flags that detach a Windows process from its console and job."""
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    if breakaway:
        flags |= getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    return int(flags)


def kill_tree(pid: int) -> None:
    """Terminate a process and its children."""
    if pid <= 0:
        return
    if IS_WINDOWS:
        _kill_tree_windows(pid)
        return
    _kill_tree_posix(pid)


def is_alive(pid: int) -> bool:
    """Return True when a process with ``pid`` is still running."""
    if pid <= 0:
        return False
    if IS_WINDOWS:
        return _is_alive_windows(pid)
    return _is_alive_posix(pid)


def _kill_tree_windows(pid: int) -> None:
    """Terminate a Windows process tree with taskkill."""
    subprocess.call(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _kill_tree_posix(pid: int) -> None:
    """Signal the process group that contains ``pid``."""
    try:
        group = getattr(os, "getpgid")(pid)
        getattr(os, "killpg")(group, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return


def _is_alive_posix(pid: int) -> bool:
    """Return True when a POSIX process accepts a null signal."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_alive_windows(pid: int) -> bool:
    """Return True when a Windows process is still running."""
    import ctypes

    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x00100000, False, pid)
    if not handle:
        return False
    try:
        return bool(kernel32.WaitForSingleObject(handle, 0) == 0x00000102)
    finally:
        kernel32.CloseHandle(handle)


def _windows_process(
    command: Sequence[str], handle: IO[bytes], breakaway: bool
) -> subprocess.Popen[bytes]:
    """Start a Windows process, dropping breakaway when the job forbids it."""
    try:
        return _popen(command, handle, windows_flags(breakaway))
    except OSError:
        if not breakaway:
            raise
        return _popen(command, handle, windows_flags(False))


def _popen(
    command: Sequence[str], handle: IO[bytes], flags: int
) -> subprocess.Popen[bytes]:
    """Start a Windows process with the given creation flags."""
    return subprocess.Popen(
        list(command),
        stdout=handle,
        stderr=subprocess.STDOUT,
        close_fds=True,
        creationflags=flags,
    )
