"""Cache management CLI commands.

This module provides commands for managing the LLM response cache:
- cache_stats: Show cache statistics
- export_cache: Export cache entries to files
- import_cache: Import cache entries from files

The reporting and transfer helpers live in
`causaliq_knowledge.cli.cache_helpers`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from causaliq_knowledge.cli.cache_helpers import (
    collect_model_stats,
    compute_savings,
    count_cache_entries,
    export_entries_to_target,
    finalise_model_stats,
    format_stats_lines,
    import_entries,
)
from causaliq_knowledge.cli.cache_helpers import (  # noqa: F401
    is_graph_entry as _is_graph_entry,
)
from causaliq_knowledge.cli.cache_helpers import (  # noqa: F401
    is_llm_entry as _is_llm_entry,
)
from causaliq_knowledge.cli.cache_helpers import (
    prepare_import_dir,
    remove_temp_dir,
)


def _collect_stats(
    cache: Any,
    entry_count: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Collect model statistics, skipping the work for an empty cache."""
    if entry_count <= 0:
        return {}, {"cost": 0.0, "input_tokens": 0, "output_tokens": 0}
    model_stats, totals = collect_model_stats(cache)
    finalise_model_stats(model_stats)
    return model_stats, totals


@click.command("cache-stats")
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the LLM cache database.",
)
def cache_stats(input_path: str) -> None:
    """Show LLM cache statistics.

    Shows entry counts, token dictionary size, cache hit statistics,
    and breakdown by model with token usage and costs.

    The LLM cache stores responses from LLM API calls to avoid
    redundant queries and reduce costs.

    Example:

        cqknow cache-stats -i ./llm_cache.db
    """
    from causaliq_core.cache import TokenCache

    try:
        with TokenCache(input_path) as cache:
            entry_count = cache.entry_count()
            token_count = cache.token_count()
            total_hits = cache.total_hits()

            model_stats, totals = _collect_stats(cache, entry_count)
            savings = compute_savings(
                model_stats, totals["cost"], entry_count, total_hits
            )
            lines = format_stats_lines(
                input_path,
                entry_count,
                token_count,
                total_hits,
                model_stats,
                totals,
                savings,
            )
            click.echo("\n".join(lines))
    except Exception as e:
        click.echo(f"Error opening cache: {e}", err=True)
        sys.exit(1)


@click.command("export-cache")
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the LLM cache database to export from.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(),
    help="Output directory or .zip file path for exported entries.",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output result as JSON.",
)
def export_cache(input_path: str, output_path: str, output_json: bool) -> None:
    """Export LLM cache entries to human-readable files.

    Exports cached LLM API responses to JSON files for backup,
    sharing, or inspection. The LLM cache stores responses to
    avoid redundant API calls.

    If output path ends with .zip, entries are exported to a zip archive.
    Otherwise, entries are exported to a directory.

    Examples:

        cqknow export-cache -i ./llm_cache.db -o ./export_dir

        cqknow export-cache -i ./llm_cache.db -o ./export.zip

        cqknow export-cache -i ./llm_cache.db -o ./export_dir --json
    """
    try:
        _run_export(input_path, output_path, output_json)
    except Exception as e:
        click.echo(f"Error exporting cache: {e}", err=True)
        sys.exit(1)


def _run_export(input_path: str, output_path: str, output_json: bool) -> None:
    """Export cache entries, emitting the command output."""
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCompressor

    dest_path = Path(output_path)
    is_zip = dest_path.suffix.lower() == ".zip"

    with TokenCache(input_path) as cache:
        # Use LLMCompressor for decompression
        compressor = LLMCompressor()
        cache.set_compressor(compressor)

        # Count entries
        if count_cache_entries(cache) == 0:
            _emit_empty_export(output_json)
            return

        # Export entries
        exported = export_entries_to_target(
            cache, compressor, dest_path, is_zip
        )

        # Output results
        _emit_export_result(
            input_path, output_path, is_zip, exported, output_json
        )


def _emit_empty_export(output_json: bool) -> None:
    """Report that an empty cache was exported."""
    if output_json:
        click.echo(json.dumps({"exported": 0, "error": None}))
    else:
        click.echo("No entries to export.")


def _emit_export_result(
    input_path: str,
    output_path: str,
    is_zip: bool,
    exported: int,
    output_json: bool,
) -> None:
    """Report the outcome of a cache export."""
    if output_json:
        result = {
            "input_path": input_path,
            "output_path": output_path,
            "format": "zip" if is_zip else "directory",
            "exported": exported,
        }
        click.echo(json.dumps(result, indent=2))
    else:
        fmt = "zip archive" if is_zip else "directory"
        click.echo(f"\nExported {exported} entries to {fmt}: {output_path}")
        click.echo()


@click.command("import-cache")
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Directory or .zip file containing JSON files to import.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(),
    help="Path to cache database (created if needed).",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output result as JSON.",
)
def import_cache(input_path: str, output_path: str, output_json: bool) -> None:
    """Import LLM cache entries from files.

    Imports previously exported LLM API responses back into a cache.
    Useful for restoring backups or sharing cached responses.

    Only LLM entries (with cache_key.model, cache_key.messages, response)
    are imported. Other entry types are skipped.

    Examples:

        cqknow import-cache -i ./import_dir -o ./llm_cache.db

        cqknow import-cache -i ./export.zip -o ./llm_cache.db

        cqknow import-cache -i ./import_dir -o ./llm_cache.db --json
    """
    try:
        _run_import(input_path, output_path, output_json)
    except Exception as e:
        click.echo(f"Error importing cache: {e}", err=True)
        sys.exit(1)


def _run_import(input_path: str, output_path: str, output_json: bool) -> None:
    """Import cache entries, emitting the command output."""
    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCompressor

    input_file = Path(input_path)
    is_zip = input_file.suffix.lower() == ".zip"

    with TokenCache(output_path) as cache:
        # Use LLMCompressor for compression
        compressor = LLMCompressor()
        cache.set_compressor(compressor)

        # Import entries
        import_dir, temp_dir = prepare_import_dir(input_file, is_zip)
        imported, skipped = import_entries(cache, import_dir)

        # Clean up temp directory
        remove_temp_dir(temp_dir)

        # Output results
        _emit_import_result(
            input_path,
            output_path,
            is_zip,
            imported,
            skipped,
            output_json,
        )


def _emit_import_result(
    input_path: str,
    output_path: str,
    is_zip: bool,
    imported: int,
    skipped: int,
    output_json: bool,
) -> None:
    """Report the outcome of a cache import."""
    if output_json:
        result = {
            "input_path": input_path,
            "output_path": output_path,
            "format": "zip" if is_zip else "directory",
            "imported": imported,
            "skipped": skipped,
        }
        click.echo(json.dumps(result, indent=2))
    else:
        fmt = "zip archive" if is_zip else "directory"
        click.echo(f"\nImported {imported} entries from {fmt}: {input_path}")
        if skipped:
            click.echo(f"  Skipped: {skipped}")
        click.echo()
