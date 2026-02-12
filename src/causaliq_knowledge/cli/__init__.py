"""Command-line interface for causaliq-knowledge.

This package provides the CLI implementation split into logical modules:

- main: Core CLI entry point
- cache: Cache management commands (stats, export, import)
- generate: Graph generation commands
- models: Model listing command
"""

from __future__ import annotations

from causaliq_knowledge.cli.main import cli, main

__all__ = ["cli", "main"]
