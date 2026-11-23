# generated from causaliq@27a2f560f378 src/causaliq/devtools/run_checked.py - do not edit
"""Launch a long-running command and wait for an exit-code artefact.

Completion is defined by ``<log>.exit``, never by console output: ``run``
starts a detached supervisor which records the pid and writes
``EXIT_CODE=<n>`` when the child ends. ``wait`` blocks once, prints the log
tail and mirrors the recorded exit code; a timeout kills the process tree and
reports exit 124.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from . import processes

TIMEOUT_EXIT = 124
POLL_SECONDS = 0.25
STARTUP_SECONDS = 10.0
DEFAULT_TAIL = 40


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch to the supervisor, the launcher or the waiter."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "--supervise":
        return _supervise(arguments[1:])
    options = _build_parser().parse_args(arguments)
    if options.wait:
        return _wait(options)
    return _launch(options)


def _build_parser() -> argparse.ArgumentParser:
    """Return the parser shared by the launcher and the waiter."""
    parser = argparse.ArgumentParser(prog="run_checked")
    parser.add_argument("--log", required=True)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--tail", type=int, default=DEFAULT_TAIL)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def _launch(options: argparse.Namespace) -> int:
    """Start the supervisor and return immediately with its pid."""
    command = _clean(options.command)
    if not command:
        print("run_checked: no command given")
        return 2
    log = Path(options.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    supervisor = _supervisor_command(log, options.timeout, command)
    child = _spawn_detached(supervisor, log)
    print(f"launched pid {child.pid}, log {log}")
    return 0


def _spawn_detached(
    command: Sequence[str], log: Path
) -> subprocess.Popen[bytes]:
    """Start a detached process with its diagnostics in a sidecar file.

    A detached process has no console, so its standard handles must be
    redirected explicitly; otherwise startup output is lost and the process
    can fail before it records anything.
    """
    sidecar = Path(f"{log}.supervisor")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    handle = sidecar.open("wb")
    try:
        return processes.spawn(command, handle, breakaway=True)
    finally:
        handle.close()


def _supervisor_command(
    log: Path, timeout: Optional[float], command: Sequence[str]
) -> List[str]:
    """Return the argv for the detached supervisor process.

    The supervisor is started through ``-c`` rather than by path: the module
    file is not the entry point, and running it directly would define the
    functions without ever calling ``main``.
    """
    supervisor = [
        sys.executable,
        "-c",
        _bootstrap_code(),
        "--supervise",
        "--log",
        str(log),
    ]
    if timeout is not None:
        supervisor.extend(["--timeout", str(timeout)])
    supervisor.append("--")
    supervisor.extend(command)
    return supervisor


def _bootstrap_code() -> str:
    """Return the ``-c`` program that imports this module and calls main."""
    package = __package__ or ""
    depth = len(package.split(".")) + 1 if package else 1
    root = Path(__file__).resolve()
    for _ in range(depth):
        root = root.parent
    target = f"{package}.run_checked" if package else "run_checked"
    return (
        "import sys; "
        f"sys.path.insert(0, {str(root)!r}); "
        f"from {target} import main; "
        "sys.exit(main())"
    )


def _supervise(argv: List[str]) -> int:
    """Run the child, record its exit code and return."""
    parser = argparse.ArgumentParser(prog="run_checked --supervise")
    parser.add_argument("--log", required=True)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    log = Path(options.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    exit_path = _exit_path(log)
    exit_path.unlink(missing_ok=True)
    _pid_path(log).write_text(str(os.getpid()), encoding="utf-8")
    code = _run_child(_clean(options.command), log, options.timeout)
    exit_path.write_text(f"EXIT_CODE={code}\n", encoding="utf-8")
    return 0


def _run_child(
    command: Sequence[str], log: Path, timeout: Optional[float]
) -> int:
    """Run the command with its output in ``log`` and return its code."""
    handle = log.open("wb")
    try:
        return _wait_child(processes.spawn(command, handle), timeout)
    finally:
        handle.close()


def _wait_child(
    child: subprocess.Popen[bytes], timeout: Optional[float]
) -> int:
    """Wait for the child, killing the tree if the timeout expires."""
    try:
        return int(child.wait(timeout=timeout))
    except subprocess.TimeoutExpired:
        processes.kill_tree(child.pid)
        return TIMEOUT_EXIT


def _wait(options: argparse.Namespace) -> int:
    """Block until the exit artefact appears, then report the result."""
    log = Path(options.log)
    exit_path = _exit_path(log)
    deadline = _deadline(options.timeout)
    pid = _read_pid(log) or _await_pid(log)
    while not exit_path.exists():
        if pid and not processes.is_alive(pid):
            break
        if _has_timed_out(deadline):
            return _report_timeout(log, pid, options.tail)
        time.sleep(POLL_SECONDS)
    _print_tail(log, options.tail)
    code = _read_exit(exit_path)
    print(f"EXIT_CODE={code}")
    return code


def _has_timed_out(deadline: Optional[float]) -> bool:
    """Return True when the wait deadline has passed."""
    if deadline is None:
        return False
    return time.monotonic() >= deadline


def _report_timeout(log: Path, pid: int, tail: int) -> int:
    """Kill the process tree, print the log tail and return exit 124."""
    processes.kill_tree(pid)
    _print_tail(log, tail)
    print(f"EXIT_CODE={TIMEOUT_EXIT} (timed out)")
    return TIMEOUT_EXIT


def _await_pid(log: Path) -> int:
    """Return the supervisor pid, allowing for a slow process start."""
    deadline = time.monotonic() + STARTUP_SECONDS
    while time.monotonic() < deadline:
        pid = _read_pid(log)
        if pid:
            return pid
        if _exit_path(log).exists():
            return 0
        time.sleep(POLL_SECONDS)
    return 0


def _deadline(timeout: Optional[float]) -> Optional[float]:
    """Return the monotonic deadline for a timeout, if one was given."""
    if timeout is None:
        return None
    return time.monotonic() + timeout


def _clean(command: Sequence[str]) -> List[str]:
    """Drop the ``--`` separator that argparse keeps in the remainder."""
    return [item for item in command if item != "--"]


def _exit_path(log: Path) -> Path:
    """Return the exit-code artefact path for a log path."""
    return Path(f"{log}.exit")


def _pid_path(log: Path) -> Path:
    """Return the pid artefact path for a log path."""
    return Path(f"{log}.pid")


def _print_tail(log: Path, count: int) -> None:
    """Print the last ``count`` lines of the log, if it exists."""
    if not log.exists():
        return
    text = log.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if lines:
        start = max(len(lines) - max(count, 1), 0)
        print("\n".join(lines[start:]))


def _read_pid(log: Path) -> int:
    """Return the recorded supervisor pid, or 0 when unavailable."""
    try:
        return int(_pid_path(log).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def _read_exit(path: Path) -> int:
    """Return the exit code recorded in ``path``, or 1 when unreadable."""
    try:
        text = path.read_text(encoding="utf-8").strip()
        return int(text.split("=", 1)[1])
    except (OSError, IndexError, ValueError):
        return 1
