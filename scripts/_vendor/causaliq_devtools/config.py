# generated from causaliq@49a27b66b20d src/causaliq/devtools/config.py - do not edit
"""Read the ``[tool.causaliq]`` block from a package ``pyproject.toml``.

``devtools`` must stay stdlib-only on Python 3.10, where ``tomllib`` does not
exist, so the block is read with a deliberately small parser. It understands
strings, booleans, integers and arrays of those values; anything else is
reported as a configuration error rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

SECTION = "[tool.causaliq]"
DEFAULT_PYTHONS = ("3.10",)
DEFAULT_EXTRAS = ("dev",)
DEFAULT_TESTPATHS = ("tests",)
DEFAULT_FAST_TESTPATHS = ("tests/unit", "tests/functional")
DEFAULT_SHARED_PYTHONS = ("3.12", "3.14")
DEFAULT_SHARED_EXTRAS = ("dev", "test")


class ConfigError(Exception):
    """Raised when the configuration block is missing or invalid."""


@dataclass(frozen=True)
class CausaliqConfig:
    """Resolved ``[tool.causaliq]`` settings for one package."""

    distribution: str
    package: str
    python_versions: Tuple[str, ...]
    default_python: str
    dev_extras: Tuple[str, ...]
    testpaths: Tuple[str, ...]
    fast_testpaths: Tuple[str, ...]
    slow_marker: str = "slow"
    extra_steps: Tuple[str, ...] = ()

    @property
    def slow_filter(self) -> Tuple[str, ...]:
        """Return the pytest arguments that exclude marked slow tests."""
        return ("-m", f"not {self.slow_marker}")

    @property
    def extras_spec(self) -> str:
        """Return the ``pip`` extras specification for the package."""
        return ",".join(self.dev_extras)


@dataclass(frozen=True)
class SharedConfig:
    """Settings that describe the umbrella's shared environments.

    These keys are umbrella-only: no package uses them, so they are read
    through their own accessor rather than widening :class:`CausaliqConfig`.
    """

    python_versions: Tuple[str, ...]
    extras: Tuple[str, ...]


def load_config(root: Optional[Path] = None) -> CausaliqConfig:
    """Return the configuration declared by the nearest pyproject.toml."""
    path = find_pyproject(root or Path.cwd())
    return build_config(read_section(path))


def load_shared(root: Optional[Path] = None) -> SharedConfig:
    """Return the shared-environment settings of the nearest pyproject."""
    path = find_pyproject(root or Path.cwd())
    return build_shared(read_section(path))


def build_shared(values: Mapping[str, object]) -> SharedConfig:
    """Return the shared-environment settings declared by the umbrella."""
    return SharedConfig(
        python_versions=_str_tuple(
            values, "shared_python_versions", DEFAULT_SHARED_PYTHONS
        ),
        extras=_str_tuple(values, "shared_extras", DEFAULT_SHARED_EXTRAS),
    )


def find_pyproject(start: Path) -> Path:
    """Return the first pyproject.toml at or above ``start``."""
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            return candidate
    raise ConfigError(f"no pyproject.toml found at or above {start}")


def read_section(path: Path) -> Dict[str, object]:
    """Return the key/value pairs declared inside ``[tool.causaliq]``."""
    pairs = _parse_pairs(_section_lines(path), path)
    if not pairs:
        raise ConfigError(f"{path} has an empty {SECTION} section")
    return pairs


def build_config(values: Mapping[str, object]) -> CausaliqConfig:
    """Validate raw pairs and return a typed configuration."""
    package = _require_str(values, "package")
    versions = _str_tuple(values, "python_versions", DEFAULT_PYTHONS)
    if not versions:
        raise ConfigError("python_versions must not be empty")
    return CausaliqConfig(
        distribution=_optional_str(values, "distribution", package),
        package=package,
        python_versions=versions,
        default_python=_optional_str(values, "default_python", versions[-1]),
        dev_extras=_str_tuple(values, "dev_extras", DEFAULT_EXTRAS),
        testpaths=_str_tuple(values, "testpaths", DEFAULT_TESTPATHS),
        fast_testpaths=_str_tuple(
            values, "fast_testpaths", DEFAULT_FAST_TESTPATHS
        ),
        slow_marker=_optional_str(values, "slow_marker", "slow"),
        extra_steps=_str_tuple(values, "extra_steps", ()),
    )


def _section_lines(path: Path) -> List[str]:
    """Return the non-comment lines that belong to the section."""
    text = path.read_text(encoding="utf-8")
    collected: List[str] = []
    inside = False
    found = False
    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if _is_header(line):
            inside = line == SECTION
            found = found or inside
            continue
        if inside and line:
            collected.append(line)
    if not found:
        raise ConfigError(f"{path} has no {SECTION} section")
    return collected


def _is_header(line: str) -> bool:
    """Return True when a stripped line is a TOML table header."""
    return line.startswith("[") and line.endswith("]")


def _strip_comment(line: str) -> str:
    """Remove a TOML comment that does not sit inside a string."""
    kept: List[str] = []
    quoted = False
    for char in line:
        if char == '"':
            quoted = not quoted
        elif char == "#" and not quoted:
            break
        kept.append(char)
    return "".join(kept).rstrip()


def _parse_pairs(lines: List[str], path: Path) -> Dict[str, object]:
    """Parse ``key = value`` lines, joining values that span lines."""
    pairs: Dict[str, object] = {}
    pending = ""
    for line in lines:
        pending = f"{pending} {line}".strip()
        if not _is_complete(pending):
            continue
        key, raw = _split_pair(pending, path)
        pairs[key] = _parse_value(raw, path)
        pending = ""
    if pending:
        raise ConfigError(f"{path}: unterminated value in {SECTION}")
    return pairs


def _is_complete(text: str) -> bool:
    """Return True when arrays and quotes in ``text`` are balanced."""
    depth = 0
    quote = ""
    for char in text:
        if char == '"':
            quote = "" if quote else '"'
            continue
        if not quote:
            depth += _bracket_delta(char)
    return depth == 0 and not quote


def _bracket_delta(char: str) -> int:
    """Return the nesting change caused by one bracket character."""
    if char == "[":
        return 1
    if char == "]":
        return -1
    return 0


def _split_pair(text: str, path: Path) -> Tuple[str, str]:
    """Split ``key = value`` into its two parts."""
    key, separator, raw = text.partition("=")
    if not separator or not key.strip():
        raise ConfigError(
            f"{path}: expected 'key = value' in {SECTION}: {text}"
        )
    return key.strip(), raw.strip()


def _parse_value(raw: str, path: Path) -> object:
    """Parse a scalar or an array value."""
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        return [_parse_scalar(item, path) for item in _split_items(text)]
    return _parse_scalar(text, path)


def _split_items(text: str) -> List[str]:
    """Return the non-empty items of an array literal."""
    body = text[1:-1].strip()
    if not body:
        return []
    return [item.strip() for item in body.split(",") if item.strip()]


def _parse_scalar(text: str, path: Path) -> object:
    """Parse one string, boolean or integer value."""
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    if text in ("true", "false"):
        return text == "true"
    try:
        return int(text)
    except ValueError as error:
        raise ConfigError(f"{path}: unsupported value {text!r}") from error


def _require_str(values: Mapping[str, object], key: str) -> str:
    """Return a required non-empty string setting."""
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{SECTION} must define a non-empty string {key!r}")
    return value


def _optional_str(values: Mapping[str, object], key: str, default: str) -> str:
    """Return an optional non-empty string setting."""
    value = values.get(key, default)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{SECTION}.{key} must be a non-empty string")
    return value


def _str_tuple(
    values: Mapping[str, object],
    key: str,
    default: Tuple[str, ...],
) -> Tuple[str, ...]:
    """Return an optional array of non-empty strings as a tuple."""
    value = values.get(key, default)
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{SECTION}.{key} must be an array of strings")
    items = tuple(value)
    if not all(isinstance(item, str) and item for item in items):
        raise ConfigError(f"{SECTION}.{key} must contain non-empty strings")
    return items
