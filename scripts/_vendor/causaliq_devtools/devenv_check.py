# generated from causaliq@49a27b66b20d src/causaliq/devtools/devenv_check.py - do not edit
"""Verify that the shared development environments are complete.

``--check`` is the "confirm the strategy" step: it reports each environment,
its interpreter and the importability of every discovered package, and exits
non-zero unless every package imports. With ``--strict`` a package must also
resolve into its own sibling checkout, which is what separates a real shared
environment from a stale one that still has published wheels installed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence

from .devenv_env import Environment, environment, smoke_command
from .devenv_repos import Repo

VERSION_PROBE = "import sys; print('%d.%d' % sys.version_info[:2])"


def check(
    root: Path,
    versions: Sequence[str],
    repos: Sequence[Repo],
    strict: bool = False,
) -> int:
    """Report each environment and return a process exit code."""
    failures = 0
    for version in versions:
        failures += _report(environment(root, version), repos, strict)
    return 0 if failures == 0 else 1


def _report(env: Environment, repos: Sequence[Repo], strict: bool) -> int:
    """Report one environment and return how many problems it has."""
    if not env.python.is_file():
        print(f"devenv: {env.target.as_posix()} is missing")
        return 1
    print(f"devenv: {env.target.as_posix()} -> {env.python}")
    failures = _report_version(env)
    for repo in repos:
        failures += _report_import(env, repo, strict)
    return failures


def _report_version(env: Environment) -> int:
    """Report the interpreter version and return a problem count."""
    reported = _reported_version(env)
    if reported == env.version:
        print(f"devenv:   interpreter Python {reported}")
        return 0
    print(
        f"devenv:   interpreter {reported or 'unknown'}, wants {env.version}"
    )
    return 1


def _reported_version(env: Environment) -> Optional[str]:
    """Return the ``X.Y`` version reported by the environment interpreter."""
    completed = _probe((str(env.python), "-c", VERSION_PROBE))
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _report_import(env: Environment, repo: Repo, strict: bool) -> int:
    """Report one package import and return a problem count."""
    problem = _import_problem(env, repo, strict)
    print(f"devenv:   {repo.distribution}: {problem or 'ok'}")
    return 1 if problem else 0


def _import_problem(
    env: Environment, repo: Repo, strict: bool
) -> Optional[str]:
    """Return a message when ``repo`` does not import from its own tree."""
    completed = _probe(smoke_command(env, repo))
    if completed.returncode != 0:
        return _last_line(completed.stderr)
    if strict and not _resolves_into(completed.stdout, repo):
        return f"resolved outside {repo.path.as_posix()}"
    return None


def _resolves_into(output: str, repo: Repo) -> bool:
    """Return True when an import resolved inside the sibling checkout."""
    location = output.strip()
    if not location:
        return False
    return Path(location).resolve().is_relative_to(repo.path.resolve())


def _probe(command: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    """Run a probe command and capture its output."""
    return subprocess.run(list(command), capture_output=True, text=True)


def _last_line(text: str) -> str:
    """Return the last non-empty line of captured standard error."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "import failed"
