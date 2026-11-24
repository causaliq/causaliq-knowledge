# generated from causaliq@49a27b66b20d src/causaliq/devtools/steps.py - do not edit
"""Build and execute the individual quality-gate steps."""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import code_rules
from .config import CausaliqConfig
from .options import GateOptions


@dataclass(frozen=True)
class Step:
    """One gate step; ``command`` is None for the in-process rules check."""

    name: str
    command: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class StepResult:
    """The outcome of one gate step."""

    name: str
    passed: bool
    detail: str = ""


def build_steps(config: CausaliqConfig, options: GateOptions) -> List[Step]:
    """Return the ordered steps selected by the options."""
    if options.rules_only:
        return [Step("code rules")]
    steps = _quality_steps(options)
    steps.extend(_extra_steps(config))
    if not options.no_tests:
        steps.append(Step("tests", _pytest_command(config, options)))
    return steps


def run_step(step: Step, root: Path) -> StepResult:
    """Execute one step and return its outcome."""
    if step.command is None:
        code = code_rules.main(["check_code_rules", str(root / "src")])
        return StepResult(step.name, code == 0)
    code = subprocess.call(list(step.command))
    return StepResult(step.name, code == 0, f"exit {code}")


def run_all(steps: Sequence[Step], root: Path) -> List[StepResult]:
    """Execute every step, printing a heading before each one."""
    results: List[StepResult] = []
    for step in steps:
        print(f"\n== {step.name} ==")
        result = run_step(step, root)
        status = "passed" if result.passed else "FAILED"
        print(f"-- {step.name}: {status}")
        results.append(result)
    return results


def _quality_steps(options: GateOptions) -> List[Step]:
    """Return formatting, linting, rules and typing steps."""
    steps = [
        Step("black", _black_command(options.fix)),
        Step("isort", _isort_command(options.fix)),
        Step("flake8", _module_command("flake8", "src/", "tests/")),
        Step("code rules"),
    ]
    if not options.fast:
        steps.append(Step("mypy", _module_command("mypy", "src/")))
    return steps


def _module_command(module: str, *paths: str) -> Tuple[str, ...]:
    """Return a ``python -m <module>`` command for the current interpreter."""
    return (sys.executable, "-m", module, *paths)


def _black_command(fix: bool) -> Tuple[str, ...]:
    """Return the black command, writing changes only in fix mode."""
    check = () if fix else ("--check",)
    return _module_command("black", *check, "src/", "tests/")


def _isort_command(fix: bool) -> Tuple[str, ...]:
    """Return the isort command, writing changes only in fix mode."""
    check = () if fix else ("--check-only",)
    return _module_command("isort", *check, "src/", "tests/")


def _pytest_command(
    config: CausaliqConfig, options: GateOptions
) -> Tuple[str, ...]:
    """Return the pytest command for the selected test selection."""
    paths = config.fast_testpaths if options.fast else config.testpaths
    args = [sys.executable, "-m", "pytest", "-v", "-rfE", "--tb=short"]
    args.extend(paths)
    if not options.fast:
        args.extend([f"--cov={config.package}", "--cov-report=term-missing"])
    if options.coverage_xml:
        args.append(f"--cov-report=xml:{options.coverage_xml}")
    if options.junit:
        args.append(f"--junitxml={options.junit}")
    args.extend(_slow_expression(config, options))
    return tuple(args)


def _slow_expression(
    config: CausaliqConfig, options: GateOptions
) -> Tuple[str, ...]:
    """Return the marker expression for the selected slow-test mode.

    ``--include-slow`` passes a matching expression rather than omitting the
    option: ``addopts`` may still carry ``-m 'not slow'`` for the benefit of
    bare ``pytest`` runs and the interim CI workflow, and a later ``-m``
    replaces it.
    """
    if not options.include_slow:
        return config.slow_filter
    marker = config.slow_marker
    return ("-m", f"{marker} or not {marker}")


def _extra_steps(config: CausaliqConfig) -> List[Step]:
    """Return steps declared by ``extra_steps`` in the configuration."""
    steps: List[Step] = []
    for command in config.extra_steps:
        parts = shlex.split(command)
        if parts:
            steps.append(Step(parts[0], tuple(parts)))
    return steps
