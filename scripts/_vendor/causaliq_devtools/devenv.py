# generated from causaliq@49a27b66b20d src/causaliq/devtools/devenv.py - do not edit
"""Create the shared CausalIQ development environments.

One environment per Python version holds every sibling CausalIQ package
installed editable in a single pip resolve, so a local edit is visible from
every repository without building five environments per repository. Locations
come from this checkout and from ``[tool.causaliq]``: no environment variable
is read or set.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .config import ConfigError, load_shared
from .devenv_check import check
from .devenv_env import (
    DevenvError,
    Environment,
    create_command,
    environment,
    install_command,
    run,
    smoke_command,
)
from .devenv_repos import Repo, discover, umbrella_root

LAUNCHER_HINT = "py -0p"


@dataclass(frozen=True)
class Options:
    """Resolved command-line switches."""

    python: Tuple[str, ...] = ()
    extras: Tuple[str, ...] = ()
    repos: Tuple[Tuple[str, Path], ...] = ()
    install_only: bool = False
    recreate: bool = False
    check: bool = False
    dry_run: bool = False
    strict: bool = False
    interpreter: Optional[str] = None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the shared-environment command and return an exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        return _run(_parse(arguments))
    except (DevenvError, ConfigError) as error:
        print(f"devenv: {error}")
        return 2


def _run(options: Options) -> int:
    """Build or verify the environments selected by ``options``."""
    root = umbrella_root()
    shared = load_shared(root)
    versions = options.python or shared.python_versions
    extras = options.extras or shared.extras
    repos = discover(root.parent, extras, dict(options.repos))
    _require_work(options, versions, repos)
    if options.check:
        return check(root, versions, repos, options.strict)
    return build(root, versions, repos, options)


def _require_work(
    options: Options,
    versions: Sequence[str],
    repos: Sequence[Repo],
) -> None:
    """Raise when the request cannot produce a usable environment set."""
    if not versions:
        raise DevenvError("no Python version selected")
    if not repos:
        raise DevenvError("no CausalIQ repository found beside the umbrella")
    if options.interpreter and len(versions) != 1:
        raise DevenvError("--interpreter requires a single --python version")


def build(
    root: Path,
    versions: Sequence[str],
    repos: Sequence[Repo],
    options: Options,
) -> int:
    """Create or refresh each environment and return an exit code."""
    failures = 0
    for version in versions:
        if _prepare(environment(root, version), repos, options):
            continue
        failures += 1
        if options.strict:
            break
    return 0 if failures == 0 else 1


def _prepare(
    env: Environment,
    repos: Sequence[Repo],
    options: Options,
) -> bool:
    """Create the environment when needed, install, then smoke-test."""
    if options.install_only and not env.target.is_dir():
        print(f"devenv: {env.target.as_posix()} is missing")
        print("devenv: run without --install-only to create it")
        return False
    if _should_create(env, options) and not _create(env, options):
        return False
    if run(install_command(env, repos), options.dry_run) != 0:
        print(f"devenv: pip install failed in {env.target.as_posix()}")
        return False
    return _smoke(env, repos, options)


def _should_create(env: Environment, options: Options) -> bool:
    """Return True when the environment must be created or cleared."""
    return options.recreate or not env.target.is_dir()


def _create(env: Environment, options: Options) -> bool:
    """Create the environment and report a missing interpreter clearly."""
    command = create_command(env, options.interpreter, options.recreate)
    if run(command, options.dry_run) == 0:
        return True
    print(
        f"devenv: Python {env.version} is not available; see {LAUNCHER_HINT}"
    )
    return False


def _smoke(
    env: Environment,
    repos: Sequence[Repo],
    options: Options,
) -> bool:
    """Import every package and report the ones that fail."""
    failures = 0
    for repo in repos:
        if run(smoke_command(env, repo), options.dry_run) != 0:
            print(f"devenv: {repo.package} does not import")
            failures += 1
    return failures == 0


def _parse(argv: Sequence[str]) -> Options:
    """Return the options selected by ``argv``."""
    return _to_options(_parser().parse_args(list(argv)))


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(prog="setup_dev_env")
    parser.add_argument("--python", action="append", metavar="VERSION")
    parser.add_argument("--extras", metavar="LIST")
    parser.add_argument("--repo", action="append", metavar="NAME=PATH")
    parser.add_argument("--install-only", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--interpreter", metavar="PATH")
    return parser


def _to_options(parsed: argparse.Namespace) -> Options:
    """Return typed options from the parsed namespace."""
    repos = tuple(_parse_override(item) for item in parsed.repo or ())
    return Options(
        python=tuple(parsed.python or ()),
        extras=_split_extras(parsed.extras),
        repos=repos,
        install_only=parsed.install_only,
        recreate=parsed.recreate,
        check=parsed.check,
        dry_run=parsed.dry_run,
        strict=parsed.strict,
        interpreter=parsed.interpreter,
    )


def _parse_override(item: str) -> Tuple[str, Path]:
    """Return the ``NAME=PATH`` pair supplied to ``--repo``."""
    name, separator, path = item.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise DevenvError(f"--repo expects NAME=PATH, got {item!r}")
    return name.strip(), Path(path.strip())


def _split_extras(value: Optional[str]) -> Tuple[str, ...]:
    """Return the comma-separated extras, dropping empty items."""
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())
