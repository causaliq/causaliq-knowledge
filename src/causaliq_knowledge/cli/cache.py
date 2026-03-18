"""Cache management CLI commands.

This module provides commands for managing the LLM response cache:
- cache_stats: Show cache statistics
- export_cache: Export cache entries to files
- import_cache: Import cache entries from files
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click


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

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMCompressor

    try:
        with TokenCache(input_path) as cache:
            entry_count = cache.entry_count()
            token_count = cache.token_count()
            total_hits = cache.total_hits()

            # Aggregate LLM-specific stats by model
            model_stats: dict[str, dict[str, Any]] = {}
            total_cost = 0.0
            total_input_tokens = 0
            total_output_tokens = 0

            if entry_count > 0:
                # Use LLMCompressor for decompression
                compressor = LLMCompressor()
                cache.set_compressor(compressor)

                # Query all entries
                cursor = cache.conn.execute(
                    "SELECT data, hit_count FROM cache_entries"
                )

                for row in cursor.fetchall():
                    try:
                        data = compressor.decompress(row[0], cache)
                        entry = LLMCacheEntry.from_dict(data)
                        hit_count = row[1] or 0

                        model = entry.model
                        if model not in model_stats:
                            model_stats[model] = {
                                "provider": entry.metadata.provider,
                                "entries": 0,
                                "hits": 0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "cost_usd": 0.0,
                                "avg_latency_ms": 0,
                                "total_latency_ms": 0,
                            }

                        stats = model_stats[model]
                        stats["entries"] += 1
                        stats["hits"] += hit_count
                        stats["input_tokens"] += entry.metadata.tokens.input
                        stats["output_tokens"] += entry.metadata.tokens.output
                        stats["cost_usd"] += entry.metadata.cost_usd
                        stats["total_latency_ms"] += entry.metadata.latency_ms

                        total_cost += entry.metadata.cost_usd
                        total_input_tokens += entry.metadata.tokens.input
                        total_output_tokens += entry.metadata.tokens.output
                    except Exception:
                        # Skip entries that can't be decompressed
                        pass

                # Calculate averages
                for stats in model_stats.values():
                    if stats["entries"] > 0:
                        stats["avg_latency_ms"] = (
                            stats["total_latency_ms"] // stats["entries"]
                        )
                    del stats["total_latency_ms"]

            # Calculate savings (cost avoided by cache hits)
            total_requests = sum(
                s["entries"] + s["hits"] for s in model_stats.values()
            )
            if total_requests > 0 and entry_count > 0:
                avg_cost_per_request = total_cost / entry_count
                savings = total_hits * avg_cost_per_request
            else:
                savings = 0.0

            lines: list[str] = []
            lines.append(f"\nCache: {input_path}")
            lines.append("=" * 60)
            lines.append(f"Entries:          {entry_count:,}")
            lines.append(f"Token dictionary: {token_count:,}")
            lines.append(f"Total cache hits: {total_hits:,}")

            if model_stats:
                lines.append(f"\nTotal cost:       ${total_cost:.4f}")
                lines.append(f"Est. savings:     ${savings:.4f}")
                lines.append(
                    f"Total tokens:     {total_input_tokens:,} in / "
                    f"{total_output_tokens:,} out"
                )

                # Table header
                lines.append("")
                lines.append(
                    f"{'Model':<32}  {'Entries':>8}  {'Hits':>8}  "
                    f"{'Hit Rate':>8}  {'Tokens In':>12}  "
                    f"{'Tokens Out':>12}  "
                    f"{'Cost':>10}  {'Latency':>10}"
                )
                lines.append("-" * 114)

                for model, stats in sorted(model_stats.items()):
                    hit_rate = (
                        stats["hits"]
                        / (stats["entries"] + stats["hits"])
                        * 100
                        if (stats["entries"] + stats["hits"]) > 0
                        else 0
                    )
                    # Truncate model name if too long
                    model_display = (
                        model[:29] + "..." if len(model) > 32 else model
                    )
                    cost_str = f"${stats['cost_usd']:.4f}"
                    latency_str = f"{stats['avg_latency_ms']:,} ms"
                    lines.append(
                        f"{model_display:<32}  {stats['entries']:>8,}  "
                        f"{stats['hits']:>8,}  {hit_rate:>7.1f}%  "
                        f"{stats['input_tokens']:>12,}  "
                        f"{stats['output_tokens']:>12,}  "
                        f"{cost_str:>10}  {latency_str:>10}"
                    )

            lines.append("")
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
    import tempfile
    import zipfile
    from pathlib import Path

    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCacheEntry, LLMCompressor

    dest_path = Path(output_path)
    is_zip = dest_path.suffix.lower() == ".zip"

    try:
        with TokenCache(input_path) as cache:
            # Use LLMCompressor for decompression
            compressor = LLMCompressor()
            cache.set_compressor(compressor)

            # Count entries
            cursor = cache.conn.execute("SELECT COUNT(*) FROM cache_entries")
            entry_count = cursor.fetchone()[0]

            if entry_count == 0:
                if output_json:
                    click.echo(json.dumps({"exported": 0, "error": None}))
                else:
                    click.echo("No entries to export.")
                return

            # Determine export directory (temp if zipping)
            if is_zip:
                temp_dir = tempfile.mkdtemp()
                export_dir = Path(temp_dir)
            else:
                export_dir = dest_path
                export_dir.mkdir(parents=True, exist_ok=True)

            # Export entries
            exported = 0
            cursor = cache.conn.execute("SELECT hash, data FROM cache_entries")
            for cache_key, blob in cursor:
                try:
                    data = compressor.decompress(blob, cache)
                    entry = LLMCacheEntry.from_dict(data)
                    filename = compressor.generate_export_filename(
                        entry, cache_key
                    )
                    file_path = export_dir / filename
                    compressor.export_entry(entry, file_path)
                    exported += 1
                except Exception:
                    # Skip entries that can't be decompressed as LLM entries
                    continue

            # Create zip archive if requested
            if is_zip:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(
                    dest_path, "w", zipfile.ZIP_DEFLATED
                ) as zf:
                    for file_path in export_dir.iterdir():
                        if file_path.is_file():
                            zf.write(file_path, file_path.name)
                # Clean up temp directory
                import shutil

                shutil.rmtree(temp_dir)

            # Output results
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
                click.echo(
                    f"\nExported {exported} entries to {fmt}: {output_path}"
                )
                click.echo()

    except Exception as e:
        click.echo(f"Error exporting cache: {e}", err=True)
        sys.exit(1)


def _is_llm_entry(data: Any) -> bool:
    """Check if JSON data represents an LLM cache entry.

    LLM entries have a specific structure with cache_key containing
    model and messages, plus a response object.
    """
    if not isinstance(data, dict):
        return False
    cache_key = data.get("cache_key", {})
    return (
        isinstance(cache_key, dict)
        and "model" in cache_key
        and "messages" in cache_key
        and "response" in data
    )


def _is_graph_entry(data: Any) -> bool:
    """Check if JSON data represents a graph cache entry.

    Graph entries have edges list, variables list, and optionally metadata.
    """
    if not isinstance(data, dict):
        return False
    return (
        "edges" in data
        and isinstance(data["edges"], list)
        and "variables" in data
        and isinstance(data["variables"], list)
    )


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
    import hashlib
    import tempfile
    import zipfile
    from pathlib import Path

    from causaliq_core.cache import TokenCache

    from causaliq_knowledge.llm.cache import LLMCompressor

    input_file = Path(input_path)
    is_zip = input_file.suffix.lower() == ".zip"

    try:
        with TokenCache(output_path) as cache:
            # Use LLMCompressor for compression
            compressor = LLMCompressor()
            cache.set_compressor(compressor)

            # Determine input directory
            if is_zip:
                temp_dir = tempfile.mkdtemp()
                import_dir = Path(temp_dir)
                with zipfile.ZipFile(input_file, "r") as zf:
                    zf.extractall(import_dir)
            else:
                import_dir = input_file
                temp_dir = None

            # Import entries
            imported = 0
            skipped = 0

            for file_path in import_dir.iterdir():
                if (
                    not file_path.is_file()
                    or file_path.suffix.lower() != ".json"
                ):
                    continue

                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    skipped += 1
                    continue

                # Only import LLM entries
                if _is_llm_entry(data):
                    # LLM entry - generate hash from cache_key contents
                    cache_key_data = data.get("cache_key", {})
                    key_str = json.dumps(cache_key_data, sort_keys=True)
                    cache_key = hashlib.sha256(key_str.encode()).hexdigest()[
                        :16
                    ]
                    cache.put_data(cache_key, data)
                    imported += 1
                else:
                    skipped += 1

            # Clean up temp directory
            if temp_dir:
                import shutil

                shutil.rmtree(temp_dir)

            # Output results
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
                click.echo(
                    f"\nImported {imported} entries from {fmt}: {input_path}"
                )
                if skipped:
                    click.echo(f"  Skipped: {skipped}")
                click.echo()

    except Exception as e:
        click.echo(f"Error importing cache: {e}", err=True)
        sys.exit(1)
