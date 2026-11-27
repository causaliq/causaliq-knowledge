# generated from causaliq@c7917d9e7fa8466b07548514cdf37d494b861321 src/causaliq/devtools/options.py - do not edit
"""Command-line options for the CausalIQ gate."""

from dataclasses import dataclass
from typing import Sequence


class OptionError(Exception):
    """Raised when the gate is given an unsupported option."""


@dataclass(frozen=True)
class GateOptions:
    """Resolved switches for one gate run."""

    fix: bool = False
    fast: bool = False
    include_slow: bool = False
    ci: bool = False
    rules_only: bool = False
    no_tests: bool = False
    junit: str | None = None
    coverage_xml: str | None = None


FLAGS = {
    "--fix": "fix",
    "--fast": "fast",
    "--include-slow": "include_slow",
    "--ci": "ci",
    "--rules": "rules_only",
    "--no-tests": "no_tests",
}
VALUE_FLAGS = {"--junit": "junit", "--coverage-xml": "coverage_xml"}
LEGACY_FLAGS = {
    "-fix": "--fix",
    "-fast": "--fast",
    "-includeslow": "--include-slow",
}


def parse_args(argv: Sequence[str]) -> GateOptions:
    """Return the options selected by ``argv``."""
    arguments = _normalise(argv)
    names = set(_flag_names(arguments))
    unknown = sorted(names - set(FLAGS) - set(VALUE_FLAGS))
    if unknown:
        raise OptionError(f"unsupported option(s): {', '.join(unknown)}")
    values: dict[str, object] = {
        attribute: flag in names for flag, attribute in FLAGS.items()
    }
    for flag, attribute in VALUE_FLAGS.items():
        values[attribute] = _value_of(arguments, flag)
    return GateOptions(**values)  # type: ignore[arg-type]


def _normalise(argv: Sequence[str]) -> list[str]:
    """Accept the PowerShell switch spellings used before the Python gate."""
    return [LEGACY_FLAGS.get(item.lower(), item) for item in argv]


def _flag_names(argv: Sequence[str]) -> list[str]:
    """Return option names, skipping the values attached to value flags."""
    names: list[str] = []
    expect_value = False
    for item in argv:
        if expect_value:
            expect_value = False
            continue
        if item in VALUE_FLAGS:
            expect_value = True
            continue
        names.append(item.split("=", 1)[0])
    return names


def _value_of(argv: Sequence[str], flag: str) -> str | None:
    """Return the value supplied for a value flag, if any."""
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None
