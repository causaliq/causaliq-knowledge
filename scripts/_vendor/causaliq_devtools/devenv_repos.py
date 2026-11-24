# generated from causaliq@49a27b66b20d src/causaliq/devtools/devenv_repos.py - do not edit
"""Discover the sibling repositories that share development environments.

``setup_dev_env`` installs every CausalIQ package editable into one environment
per Python version. The list of packages is not a separate manifest: a sibling
directory qualifies when its ``pyproject.toml`` declares the same
``[tool.causaliq]`` section the gate reads, so discovery and the gate always
agree about which repositories exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .config import SECTION, CausaliqConfig, build_config, read_section

MANIFEST = "pyproject.toml"
SKIP_SUFFIX = "-template"
UMBRELLA_DEPTH = 3


@dataclass(frozen=True)
class Repo:
    """One sibling repository installed into the shared environments."""

    distribution: str
    package: str
    path: Path
    extras: Tuple[str, ...]

    @property
    def requirement(self) -> str:
        """Return the pip requirement used for the editable install."""
        if not self.extras:
            return self.path.as_posix()
        return f"{self.path.as_posix()}[{','.join(self.extras)}]"


def discover(
    root: Path,
    extras: Sequence[str],
    overrides: Optional[Mapping[str, Path]] = None,
) -> List[Repo]:
    """Return the repositories below ``root``, with overrides applied."""
    found = [_repo_at(path, extras) for path in sorted(root.iterdir())]
    repos = [repo for repo in found if repo is not None]
    return apply_overrides(repos, overrides or {}, extras)


def umbrella_root() -> Path:
    """Return the umbrella checkout that owns this module."""
    return Path(__file__).resolve().parents[UMBRELLA_DEPTH]


def apply_overrides(
    repos: Sequence[Repo],
    overrides: Mapping[str, Path],
    extras: Sequence[str],
) -> List[Repo]:
    """Return ``repos`` with each ``--repo NAME=PATH`` entry applied."""
    merged: Dict[str, Repo] = {repo.distribution: repo for repo in repos}
    for name, path in overrides.items():
        merged[name] = _repo(name, path, _read_config(path), extras)
    return [merged[name] for name in sorted(merged)]


def select_extras(
    config: CausaliqConfig, wanted: Sequence[str]
) -> Tuple[str, ...]:
    """Return the requested extras the package actually declares."""
    return tuple(extra for extra in wanted if extra in config.dev_extras)


def _repo_at(path: Path, extras: Sequence[str]) -> Optional[Repo]:
    """Return the repository at ``path``, or None when it is not one."""
    if not path.is_dir() or path.name.endswith(SKIP_SUFFIX):
        return None
    manifest = path / MANIFEST
    if not manifest.is_file() or not _declares_section(manifest):
        return None
    config = _read_config(path)
    return _repo(config.distribution, path, config, extras)


def _repo(
    name: str,
    path: Path,
    config: CausaliqConfig,
    extras: Sequence[str],
) -> Repo:
    """Return the repository at ``path`` under the given distribution name."""
    resolved = path.resolve()
    return Repo(name, config.package, resolved, select_extras(config, extras))


def _read_config(path: Path) -> CausaliqConfig:
    """Return the configuration of the repository at ``path``."""
    return build_config(read_section(path / MANIFEST))


def _declares_section(manifest: Path) -> bool:
    """Return True when ``manifest`` contains the configuration header."""
    text = manifest.read_text(encoding="utf-8")
    lines = (line.split("#", 1)[0].strip() for line in text.splitlines())
    return any(line == SECTION for line in lines)
