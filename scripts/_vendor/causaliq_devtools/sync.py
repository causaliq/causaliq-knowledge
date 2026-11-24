# generated from causaliq@49a27b66b20d src/causaliq/devtools/sync.py - do not edit
"""Verify or refresh the generated development scripts in a repository.

Packages never import the umbrella: ``update`` copies the devtools modules and
renders the script templates into ``scripts/``, and ``check`` compares the
committed files with the hashes recorded in ``scripts/generated.json``. The
check is offline, so a clone, fork or CI job can verify provenance without
network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import templates
from .config import CausaliqConfig, load_config

SCRIPTS = Path("scripts")
VENDOR = SCRIPTS / "_vendor" / "causaliq_devtools"
MANIFEST = SCRIPTS / "generated.json"
DEVTOOLS = Path("src") / "causaliq" / "devtools"
# ``devenv*`` is umbrella-only tooling that no package imports, so vendoring
# it would place unused copies in eleven repositories.
UMBRELLA_ONLY = "devenv"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Verify the generated files or refresh them from a source checkout."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    options = _parse(arguments)
    target = Path(options.target).resolve()
    if not options.update:
        return check(target)
    if not options.source:
        print("sync: --update requires --source")
        return 2
    return update(
        target,
        Path(options.source).resolve(),
        options.ref,
        options.bootstrap,
    )


def update(
    target: Path,
    source: Path,
    ref: Optional[str] = None,
    bootstrap: bool = False,
) -> int:
    """Write the generated files and manifest for ``target``.

    ``bootstrap`` renders the scripts the umbrella needs - the environment and
    wrapper scripts plus the shared-environment entry point - and omits the
    entry points and vendored copy that only a package uses.
    """
    config = load_config(target)
    resolved = ref or source_ref(source)
    files = render(config, source, resolved, bootstrap)
    for relative, text in files.items():
        _write(target / relative, text)
    _write(target / MANIFEST, _manifest_text(resolved, files))
    print(f"sync: wrote {len(files)} files for {config.distribution}")
    print(f"sync: source {resolved}")
    return 0


def check(target: Path) -> int:
    """Compare the committed files with the manifest and report drift."""
    manifest = _read_manifest(target)
    if manifest is None:
        print(f"sync: no {MANIFEST.as_posix()}; run sync --update")
        return 1
    files = manifest.get("files", {})
    problems = _compare(target, files)
    if problems:
        for problem in problems:
            print(f"sync: {problem}")
        return 1
    source = manifest.get("source", "unknown")
    print(f"sync: {len(files)} generated files match {source}")
    return 0


def render(
    config: CausaliqConfig,
    source: Path,
    ref: str,
    bootstrap: bool = False,
) -> Dict[str, str]:
    """Return the generated file contents keyed by relative path."""
    if bootstrap:
        files = _bootstrap_files(config)
        files["scripts/setup_dev_env.py"] = (
            templates.setup_dev_env_entry_point()
        )
    else:
        files = _package_files(config, source)
    return {
        relative: _with_header(relative, text, ref)
        for relative, text in files.items()
    }


def _bootstrap_files(config: CausaliqConfig) -> Dict[str, str]:
    """Return only the environment and wrapper scripts."""
    return {
        "scripts/check_ci.ps1": templates.check_ci_wrapper(),
        "scripts/activate.ps1": templates.activate_ps1(config),
        "scripts/setup-env.ps1": templates.setup_env_ps1(config),
    }


def _package_files(config: CausaliqConfig, source: Path) -> Dict[str, str]:
    """Return the full generated set for a package."""
    files = {
        "scripts/check_ci.py": templates.entry_point(
            "gate", "main", "Run the CausalIQ quality gates."
        ),
        "scripts/run_checked.py": templates.entry_point(
            "run_checked", "main", "Run a command and record its exit code."
        ),
        "scripts/sync.py": templates.entry_point(
            "sync", "main", "Verify or refresh generated dev scripts."
        ),
        "scripts/build-and-publish.ps1": templates.build_and_publish_ps1(
            config
        ),
    }
    files.update(_bootstrap_files(config))
    files.update(_vendor_files(source))
    return files


def source_ref(source: Path) -> str:
    """Return a ``causaliq@<sha>`` reference for the source checkout."""
    try:
        sha = _git(source, "rev-parse", "HEAD")
        dirty = _git(source, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return "causaliq@unknown"
    if dirty:
        print("sync: warning: the source checkout has uncommitted changes")
    suffix = "-dirty" if dirty else ""
    return f"causaliq@{sha[:12]}{suffix}"


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    """Return the parsed command line."""
    parser = argparse.ArgumentParser(prog="sync")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--source", default=None)
    parser.add_argument("--target", default=".")
    parser.add_argument("--ref", default=None)
    parser.add_argument("--bootstrap", action="store_true")
    options = parser.parse_args(argv)
    if options.check and options.update:
        parser.error("choose either --check or --update")
    return options


def _vendor_files(source: Path) -> Dict[str, str]:
    """Return the vendored devtools modules keyed by relative path."""
    root = source / DEVTOOLS
    if not root.is_dir():
        raise SystemExit(f"sync: no devtools package under {root}")
    files: Dict[str, str] = {}
    for path in sorted(root.glob("*.py")):
        if path.stem.startswith(UMBRELLA_ONLY):
            continue
        relative = (VENDOR / path.name).as_posix()
        files[relative] = path.read_text(encoding="utf-8")
    return files


def _with_header(relative: str, text: str, ref: str) -> str:
    """Prepend the provenance header to a generated file."""
    name = Path(relative).name
    origin = f"src/causaliq/devtools/{name}"
    if not relative.startswith(VENDOR.as_posix()):
        origin = "src/causaliq/devtools/templates.py"
    return f"# generated from {ref} {origin} - do not edit\n{text}"


def _write(path: Path, text: str) -> None:
    """Write text with LF line endings, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _hash(text: str) -> str:
    """Return the SHA-256 of the LF-normalised, UTF-8 encoded text."""
    normalised = text.replace("\r\n", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _manifest_text(ref: str, files: Dict[str, str]) -> str:
    """Return the JSON manifest for a set of generated files."""
    payload = {
        "source": ref,
        "files": {name: _hash(text) for name, text in sorted(files.items())},
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _read_manifest(target: Path) -> Optional[Dict[str, Dict[str, str]]]:
    """Return the parsed manifest, or None when it is absent or invalid."""
    path = target / MANIFEST
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    files = payload.get("files")
    if not isinstance(files, dict):
        return None
    return payload


def _compare(target: Path, files: Dict[str, str]) -> List[str]:
    """Return one message per missing or modified generated file."""
    problems: List[str] = []
    for name, expected in sorted(files.items()):
        path = target / name
        if not path.is_file():
            problems.append(f"missing {name}")
            continue
        if _hash(path.read_text(encoding="utf-8")) != expected:
            problems.append(f"modified {name}")
    return problems


def _git(source: Path, *args: str) -> str:
    """Return the stripped output of a git command in ``source``."""
    completed = subprocess.check_output(
        ["git", "-C", str(source), *args], text=True
    )
    return completed.strip()
