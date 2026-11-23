# generated from causaliq@27a2f560f378 src/causaliq/devtools/gate.py - do not edit
"""Run the CausalIQ quality gates for the current package."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from .config import CausaliqConfig, ConfigError, find_pyproject, load_config
from .options import GateOptions, OptionError, parse_args
from .steps import StepResult, build_steps, run_all

CI_DIR = Path("build") / "ci"


def run_gate(argv: Optional[Sequence[str]] = None) -> int:
    """Run the gate and return a process exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        options = parse_args(arguments)
        root = find_pyproject(Path.cwd()).parent
        config = load_config(root)
    except (ConfigError, OptionError) as error:
        print(f"gate: {error}")
        return 2
    if not _is_repository_environment(root, options):
        return 2
    options = _with_ci_outputs(options, root)
    _print_plan(root, config, options)
    results = run_all(build_steps(config, options), root)
    return _report(results)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Alias used by generated entry points."""
    return run_gate(argv)


def _is_repository_environment(root: Path, options: GateOptions) -> bool:
    """Return True when the running interpreter belongs to this repository."""
    if Path(sys.prefix).resolve().is_relative_to(root):
        return True
    message = "gate: the running interpreter is not this repository's venv"
    active = os.environ.get("VIRTUAL_ENV")
    if active:
        message = f"{message} (VIRTUAL_ENV={active})"
    if options.ci:
        print(f"{message} (continuing in CI mode)")
        return True
    print(f"{message}. Run .\\scripts\\activate.ps1 first.")
    return False


def _with_ci_outputs(options: GateOptions, root: Path) -> GateOptions:
    """Return options with CI report paths filled in and created."""
    if not options.ci:
        return options
    junit = options.junit or str(root / CI_DIR / "junit.xml")
    coverage = options.coverage_xml or str(root / CI_DIR / "coverage.xml")
    Path(junit).parent.mkdir(parents=True, exist_ok=True)
    Path(coverage).parent.mkdir(parents=True, exist_ok=True)
    return replace(options, junit=junit, coverage_xml=coverage)


def _print_plan(
    root: Path, config: CausaliqConfig, options: GateOptions
) -> None:
    """Print the interpreter, configuration and selected test paths."""
    print(f"gate: {root}")
    print(f"gate: {sys.executable}")
    print(f"gate: package {config.package}, extras {config.extras_spec}")
    selection = "fast" if options.fast else "full"
    paths = ", ".join(config.testpaths)
    print(f"gate: {selection} test selection from {paths}")
    if options.fix:
        print("gate: fix mode writes formatting changes")
    if options.include_slow:
        print(f"gate: slow tests ({config.slow_marker}) are included")


def _report(results: Sequence[StepResult]) -> int:
    """Print the summary and return the gate exit code."""
    print("\n====================")
    print("Gate results:")
    for result in results:
        status = "PASSED" if result.passed else "FAILED"
        print(f"  {result.name}: {status} {result.detail}".rstrip())
    passed = all(result.passed for result in results)
    print("All checks passed." if passed else "Some checks FAILED.")
    return 0 if passed else 1
