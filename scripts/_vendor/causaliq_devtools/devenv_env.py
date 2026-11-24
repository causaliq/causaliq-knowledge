# generated from causaliq@49a27b66b20d src/causaliq/devtools/devenv_env.py - do not edit
"""Environment locations, interpreter lookup and command construction.

A location is derived from the umbrella checkout and the requested Python
version, never from an environment variable: ``venv/py312`` and ``venv/py314``
under the checkout that owns this module. Keeping the command builders pure
lets the tests assert them without creating anything.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .devenv_repos import Repo
from .processes import IS_WINDOWS

ENV_DIR = "venv"
POSIX_FALLBACK = "python3"


class DevenvError(Exception):
    """Raised when the request cannot be carried out."""


@dataclass(frozen=True)
class Environment:
    """One shared virtual environment."""

    version: str
    target: Path

    @property
    def python(self) -> Path:
        """Return the interpreter inside the environment."""
        folder = "Scripts" if IS_WINDOWS else "bin"
        name = "python.exe" if IS_WINDOWS else "python"
        return self.target / folder / name


def environment(root: Path, version: str) -> Environment:
    """Return the shared environment for ``version`` below ``root``."""
    return Environment(version, root / ENV_DIR / f"py{_key(version)}")


def _key(version: str) -> str:
    """Return the directory suffix used for a Python version."""
    return version.replace(".", "")


def launcher(
    version: str, interpreter: Optional[str] = None
) -> Tuple[str, ...]:
    """Return the command prefix that runs the requested interpreter."""
    if interpreter:
        return (interpreter,)
    if IS_WINDOWS:
        return ("py", f"-{version}")
    return (_posix_interpreter(version),)


def _posix_interpreter(version: str) -> str:
    """Return the first POSIX interpreter name found on PATH."""
    for name in (f"python{version}", POSIX_FALLBACK):
        if shutil.which(name):
            return name
    raise DevenvError(f"Python {version} is not available; pass --interpreter")


def create_command(
    env: Environment,
    interpreter: Optional[str],
    clear: bool,
) -> Tuple[str, ...]:
    """Return the command that creates or clears one environment."""
    arguments = ["-m", "venv"]
    if clear:
        arguments.append("--clear")
    arguments.append(str(env.target))
    return (*launcher(env.version, interpreter), *arguments)


def install_command(
    env: Environment, repos: Sequence[Repo]
) -> Tuple[str, ...]:
    """Return the single pip resolve that installs every package editable."""
    return (str(env.python), "-m", "pip", "install", *_editable(repos))


def _editable(repos: Sequence[Repo]) -> Tuple[str, ...]:
    """Return one ``-e <requirement>`` pair per repository."""
    return tuple(part for repo in repos for part in ("-e", repo.requirement))


def smoke_command(env: Environment, repo: Repo) -> Tuple[str, ...]:
    """Return the command that imports a package and prints its location."""
    probe = f"import {repo.package} as m; print(m.__file__)"
    return (str(env.python), "-c", probe)


def run(command: Sequence[str], dry_run: bool) -> int:
    """Run one command, or print it and stop in dry-run mode."""
    print(f"devenv: {shlex.join(command)}")
    if dry_run:
        return 0
    try:
        return subprocess.call(list(command))
    except OSError as error:
        print(f"devenv: {error}")
        return 1
