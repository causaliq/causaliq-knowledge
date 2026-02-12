"""Main CLI entry point and core commands.

This module provides the main CLI group and commands for LLM-based
causal graph generation.
"""

from __future__ import annotations

import click

from causaliq_knowledge import __version__


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """CausalIQ Knowledge - LLM knowledge for causal discovery.

    Use 'cqknow generate_graph' to generate causal graphs.
    Use 'cqknow list_models' to list available LLM models.

    LLM Cache Management (for caching LLM API responses):

    Use 'cqknow cache_stats' to view LLM cache statistics.
    Use 'cqknow export_cache' to export LLM cache entries.
    Use 'cqknow import_cache' to import LLM cache entries.
    """
    pass


# Import and register commands
from causaliq_knowledge.cli.cache import (  # noqa: E402
    cache_stats,
    export_cache,
    import_cache,
)
from causaliq_knowledge.cli.generate import generate_graph  # noqa: E402
from causaliq_knowledge.cli.models import list_models  # noqa: E402

cli.add_command(cache_stats)
cli.add_command(export_cache)
cli.add_command(import_cache)
cli.add_command(generate_graph)
cli.add_command(list_models)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":  # pragma: no cover
    main()
