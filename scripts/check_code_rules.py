"""Report per-function nesting depth and statement counts for source modules.

The limits and their counting rules are defined in the umbrella development
policy (`01-code.md`): statements include nested statements, while docstrings
and `# pragma: no cover` blocks do not count; nesting counts blocks below the
function body, with guard clauses exempt; a module targets ~200 statements.

Advisory by default — breaches are reported and the exit code stays 0, which
suits legacy code that is still quarantined. With `--strict` the script exits
1 when any scanned path breaches a limit, so it can gate new or re-structured
code.

Usage:

    python scripts/check_code_rules.py
    python scripts/check_code_rules.py --strict \\
        src/causaliq_analysis/workflow_action
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# Limits from the development policy
MAX_DEPTH = 2
TARGET_STATEMENTS = 15
MAX_STATEMENTS = 30
MAX_MODULE_STATEMENTS = 200

PRAGMA = "# pragma: no cover"
SKIP_PARTS = ("venv", ".venv", "__pycache__", ".mypy_cache", "site-packages")

# Nodes that introduce an indented block below the function body
BLOCK_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.ExceptHandler,
)
GUARD_NODES = (ast.Continue, ast.Break, ast.Return, ast.Raise)


def _is_docstring(node: ast.stmt) -> bool:
    """Return True when a statement is a bare string expression."""
    if not isinstance(node, ast.Expr):
        return False
    return isinstance(node.value, ast.Constant) and isinstance(
        node.value.value, str
    )


def _is_guard(node: ast.If) -> bool:
    """Return True when an `if` body only exits the current block."""
    if len(node.body) != 1:
        return False
    return isinstance(node.body[0], GUARD_NODES)


def _is_excluded(node: ast.AST, lines: List[str]) -> bool:
    """Return True when a statement carries a coverage pragma."""
    if not hasattr(node, "lineno") or node.lineno > len(lines):
        return False
    return PRAGMA in lines[node.lineno - 1]


def _count_statements(node: ast.AST, lines: List[str]) -> int:
    """Count executable statements below a node, including nested ones."""
    total = 0
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, (ast.stmt, ast.ExceptHandler)):
            continue
        if isinstance(child, ast.stmt) and _is_docstring(child):
            continue
        if _is_excluded(child, lines):
            continue
        total += 1 + _count_statements(child, lines)
    return total


def _is_block(node: ast.AST) -> bool:
    """Return True when a node indents a block that counts for depth."""
    if not isinstance(node, BLOCK_NODES):
        return False
    if isinstance(node, ast.If):
        return not _is_guard(node)
    return True


def _deepest_level(node: ast.AST, level: int = 0) -> int:
    """Return the deepest block level below a statement or handler."""
    inner = level + 1 if _is_block(node) else level
    best = inner
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.stmt, ast.ExceptHandler)):
            best = max(best, _deepest_level(child, inner))
    return best


def _measure_function(
    func: ast.AST,
    lines: List[str],
) -> Tuple[int, int]:
    """Return the nesting depth and statement count of a function."""
    body = getattr(func, "body", [])
    depth = max((_deepest_level(stmt) for stmt in body), default=0)
    return depth, _count_statements(func, lines)


def _report_function(path: Path, func: ast.AST, lines: List[str]) -> int:
    """Report a function whose depth or statement count breaches."""
    depth, stmts = _measure_function(func, lines)
    if depth <= MAX_DEPTH and stmts <= TARGET_STATEMENTS:
        return 0
    note = " (over hard maximum)" if stmts > MAX_STATEMENTS else ""
    name = getattr(func, "name", "?")
    line = getattr(func, "lineno", 0)
    print(f"{path}:{line} {name} depth={depth} stmts={stmts}{note}")
    return 1


def _report_functions(path: Path, tree: ast.AST, lines: List[str]) -> int:
    """Report every breaching function in a module."""
    breaches = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        breaches += _report_function(path, node, lines)
    return breaches


def _report_module_size(path: Path, tree: ast.AST, lines: List[str]) -> int:
    """Report a module that exceeds the module statement target."""
    total = _count_statements(tree, lines)
    if total <= MAX_MODULE_STATEMENTS:
        return 0
    print(f"{path}: module has {total} statements")
    return 1


def _report_module(path: Path) -> int:
    """Print the breaches of one module and return how many were found."""
    lines = _load_lines(path)
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError as error:
        print(f"{path}: could not parse ({error})")
        return 1
    breaches = _report_module_size(path, tree, lines)
    breaches += _report_functions(path, tree, lines)
    return breaches


def _load_lines(path: Path) -> List[str]:
    """Return the source lines of a module."""
    return path.read_text(encoding="utf-8").splitlines()


def _is_source(path: Path) -> bool:
    """Return True when a path is outside an environment or cache."""
    return not any(part in SKIP_PARTS for part in path.parts)


def _python_files(targets: Iterable[str]) -> List[Path]:
    """Expand path arguments into a sorted list of source modules."""
    paths: List[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_file():
            paths.append(path)
            continue
        found = sorted(path.rglob("*.py"))
        paths.extend(item for item in found if _is_source(item))
    return paths


def _print_summary(checked: int, breaches: int, strict: bool) -> None:
    """Print the summary line and the advisory note."""
    mode = "strict" if strict else "advisory"
    print(f"rules ({mode}): {breaches} breaches across all {checked} modules")
    if not strict and breaches:
        print("Advisory mode: legacy breaches are quarantined.")


def main(argv: List[str]) -> int:
    """Report breaches for the requested paths and return an exit code."""
    strict = "--strict" in argv
    targets = [item for item in argv[1:] if not item.startswith("--")]
    paths = _python_files(targets or ["src"])
    breaches = 0
    for path in paths:
        breaches += _report_module(path)
    _print_summary(len(paths), breaches, strict)
    return 1 if strict and breaches else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
